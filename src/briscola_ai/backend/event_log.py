"""
Event log “da laboratorio” (SQLite) per Briscola AI.

Obiettivo didattico
-------------------
Quando iniziamo a fare ML (dataset, self-play, valutazione), diventa fondamentale poter:
- riprodurre una partita (seed + sequenza di azioni);
- capire *cosa* è successo e *quando* (ordering, `server_version`);
- esportare i dati in un formato adatto al training (es. JSONL).

Questo modulo implementa un event log append-only su SQLite.
È volutamente semplice:
- usa solo la stdlib (`sqlite3`, `json`);
- non impone uno schema “finale” dei payload: i dettagli vivono in `payload_json`.

Configurazione
-------------
Il percorso del DB è configurabile (env/CLI) dal livello applicativo.
Se non viene fornito alcun path, la feature può restare disabilitata.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class EventLogConfig:
    """
    Configurazione del logger.

    - `path`: percorso del file SQLite (es. `./data/briscola_events.sqlite3`).
      Se `:memory:` usa un database in memoria (utile nei test).
    """

    path: str


class EventLog:
    """
    Logger append-only su SQLite.

    Note implementative
    -------------------
    - Usare SQLite in un server async è OK per carichi bassi e per un progetto didattico.
      Qui rendiamo le scritture thread-safe con un lock.
    - Abilitiamo WAL per migliorare la concorrenza (letture mentre scriviamo).
    """

    def __init__(self, config: EventLogConfig):
        self._config = config
        self._lock = threading.Lock()
        self._conn = self._connect(config.path)
        self._init_schema()

    @property
    def path(self) -> str:
        """Percorso del DB (utile per debug)."""
        return self._config.path

    def close(self) -> None:
        """Chiude la connessione SQLite."""
        with self._lock:
            self._conn.close()

    def _connect(self, path: str) -> sqlite3.Connection:
        """
        Apre una connessione SQLite.

        Se il path è un file su disco, crea la directory padre se manca.
        """
        if path != ":memory:":
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)

        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_schema(self) -> None:
        """Crea tabelle e indici se non esistono."""
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS games (
                    game_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    num_players INTEGER NOT NULL,
                    seed INTEGER,
                    code_version TEXT,
                    rules_version TEXT
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    server_version INTEGER,
                    player_index INTEGER,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY (game_id) REFERENCES games(game_id)
                );
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_game_id ON events(game_id);")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);")
            # Compatibilità DB già esistenti (created prima di aggiungere colonne).
            self._ensure_column("games", "code_version", "TEXT")
            self._ensure_column("games", "rules_version", "TEXT")
            self._conn.commit()

    def _ensure_column(self, table: str, column: str, col_type: str) -> None:
        """
        Migrazione minimale: aggiunge una colonna se manca.

        SQLite supporta `ALTER TABLE ... ADD COLUMN` per aggiunte semplici.
        Questo è sufficiente per un progetto didattico e mantiene compatibilità con DB già creati.
        """
        existing = {row[1] for row in self._conn.execute(f"PRAGMA table_info({table});").fetchall()}
        if column in existing:
            return
        self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type};")

    def ensure_game(
        self,
        game_id: str,
        *,
        num_players: int,
        seed: Optional[int] = None,
        code_version: Optional[str] = None,
        rules_version: Optional[str] = None,
    ) -> None:
        """
        Inserisce la riga della partita (idempotente).

        La tabella `games` serve principalmente come metadato e come “anchor” per le FK.
        """
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO games(game_id, created_at, num_players, seed, code_version, rules_version)
                VALUES(?, ?, ?, ?, ?, ?);
                """,
                (game_id, now, num_players, seed, code_version, rules_version),
            )
            self._conn.commit()

    def log_event(
        self,
        game_id: str,
        event_type: str,
        payload: dict[str, Any],
        *,
        server_version: Optional[int] = None,
        player_index: Optional[int] = None,
        created_at: Optional[float] = None,
    ) -> None:
        """
        Appende un evento alla tabella `events`.

        Parametri
        ---------
        - `event_type`: stringa breve e stabile (es. `game_created`, `action_play_card`, `observation_sent`).
        - `payload`: dict JSON-serializzabile (DTO o informazioni minimali).
        - `server_version`: versione monotona (se nota) per ordering/debug.
        - `player_index`: destinatario o autore (se applicabile).
        """
        ts = created_at if created_at is not None else time.time()
        payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO events(game_id, created_at, server_version, player_index, event_type, payload_json)
                VALUES(?, ?, ?, ?, ?, ?);
                """,
                (game_id, ts, server_version, player_index, event_type, payload_json),
            )
            self._conn.commit()


def parse_event_db_path(raw: Optional[str]) -> Optional[str]:
    """
    Normalizza un path di configurazione (env/CLI).

    Ritorna `None` se la feature deve essere disabilitata.
    """
    if raw is None:
        return None
    cleaned = raw.strip()
    if cleaned == "":
        return None
    return cleaned
