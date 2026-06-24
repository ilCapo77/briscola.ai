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

import contextlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class EventLogConfig:
    """
    Configurazione del logger.

    - `path`: percorso del file SQLite (es. `./data/briscola_events.sqlite3`).
      Se `:memory:` usa un database in memoria (utile nei test).
    """

    path: str


@runtime_checkable
class EventLogProtocol(Protocol):
    """
    Interfaccia comune dei backend event log (SQLite locale, Postgres in cloud).

    Permette al resto dell'app di non dipendere dall'implementazione concreta:
    `main.py` sceglie il backend (factory) e `server.py` usa solo questi metodi.
    """

    @property
    def path(self) -> str: ...

    def close(self) -> None: ...

    def ensure_game(
        self,
        game_id: str,
        *,
        num_players: int,
        seed: Optional[int] = None,
        code_version: Optional[str] = None,
        rules_version: Optional[str] = None,
    ) -> None: ...

    def set_client_id(self, game_id: str, *, client_id: str) -> None: ...

    def try_mark_game_finished(self, game_id: str, *, finished_at: Optional[float] = None) -> bool: ...

    def try_mark_game_aborted(
        self, game_id: str, *, aborted_reason: str, aborted_at: Optional[float] = None
    ) -> bool: ...

    def log_event(
        self,
        game_id: str,
        event_type: str,
        payload: dict[str, Any],
        *,
        server_version: Optional[int] = None,
        player_index: Optional[int] = None,
        created_at: Optional[float] = None,
    ) -> None: ...


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
                    rules_version TEXT,
                    client_id TEXT,
                    finished_at REAL,
                    aborted_at REAL,
                    aborted_reason TEXT
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
            self._ensure_column("games", "client_id", "TEXT")
            self._ensure_column("games", "finished_at", "REAL")
            self._ensure_column("games", "aborted_at", "REAL")
            self._ensure_column("games", "aborted_reason", "TEXT")
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

    def set_client_id(self, game_id: str, *, client_id: str) -> None:
        """
        Salva un identificatore pseudonimo del client (best-effort).

        Nota privacy:
        questo campo serve a poter fare split train/val "per giocatore" senza salvare PII.
        È responsabilità del frontend generare un UUID (localStorage) o un identificatore
        equivalente non riconducibile alla persona.
        """
        cleaned = str(client_id).strip()
        if not cleaned:
            return
        with self._lock:
            self._conn.execute(
                """
                UPDATE games
                SET client_id = COALESCE(client_id, ?)
                WHERE game_id = ?;
                """,
                (cleaned, game_id),
            )
            self._conn.commit()

    def try_mark_game_finished(self, game_id: str, *, finished_at: Optional[float] = None) -> bool:
        """
        Marca una partita come conclusa (`game_over=true`) in modo idempotente.

        Ritorna True se lo stato è stato aggiornato (prima volta), False se era già marcata.
        """
        ts = time.time() if finished_at is None else float(finished_at)
        with self._lock:
            cur = self._conn.execute(
                """
                UPDATE games
                SET finished_at = ?
                WHERE game_id = ? AND finished_at IS NULL;
                """,
                (ts, game_id),
            )
            self._conn.commit()
            return bool(cur.rowcount == 1)

    def try_mark_game_aborted(
        self,
        game_id: str,
        *,
        aborted_reason: str,
        aborted_at: Optional[float] = None,
    ) -> bool:
        """
        Marca una partita come abortita (timeout/inactivity) in modo idempotente.

        Nota:
        non abortiamo una partita già finita (`finished_at` non null).
        """
        ts = time.time() if aborted_at is None else float(aborted_at)
        reason = str(aborted_reason).strip()[:200]
        if not reason:
            reason = "unknown"

        with self._lock:
            row = self._conn.execute(
                "SELECT finished_at, aborted_at FROM games WHERE game_id = ?;",
                (game_id,),
            ).fetchone()
            if row is None:
                return False
            finished_at, existing_aborted_at = row[0], row[1]
            if finished_at is not None:
                return False
            if existing_aborted_at is not None:
                return False

            cur = self._conn.execute(
                """
                UPDATE games
                SET aborted_at = ?, aborted_reason = ?
                WHERE game_id = ? AND finished_at IS NULL AND aborted_at IS NULL;
                """,
                (ts, reason, game_id),
            )
            self._conn.commit()
            return bool(cur.rowcount == 1)

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


class PostgresEventLog:
    """
    Event log append-only su **Postgres** (deploy multi-replica, es. Neon).

    Stessa interfaccia di `EventLog` (vedi `EventLogProtocol`), ma persistente e condiviso tra
    repliche (a differenza dell'SQLite locale, che in cloud è per-replica ed effimero).

    Note implementative
    -------------------
    - `psycopg` (v3) è importato lazy: la dipendenza è installata, ma il modulo si carica solo se
      si usa davvero Postgres (cioè se è impostata `DATABASE_URL`).
    - Connessione in `autocommit` + `threading.Lock`: scritture best-effort, serializzate (come
      l'SQLite locale). Per il traffico hobby di un event log append-only è adeguato.
    - Le operazioni "mark finished/aborted" sono UPDATE atomici con guardia in `WHERE` e usano
      `rowcount` per l'idempotenza (nessun SELECT-then-UPDATE).
    - Il client può essere iniettato (`conn=`) per i test senza un Postgres reale.
    """

    def __init__(self, dsn: Optional[str] = None, *, conn: Any = None) -> None:
        self._lock = threading.Lock()
        self._dsn = dsn
        if conn is not None:
            self._conn = conn
        elif dsn is not None:
            import psycopg  # import lazy: solo se si usa Postgres

            self._conn = psycopg.connect(dsn, autocommit=True)
        else:
            raise ValueError("PostgresEventLog richiede `dsn` oppure `conn`.")
        self._init_schema()

    @property
    def path(self) -> str:
        """Identità del backend (per il confronto di ricreazione nel lifespan)."""
        return self._dsn or "postgres"

    def close(self) -> None:
        with self._lock, contextlib.suppress(Exception):
            self._conn.close()

    def _execute(self, sql: str, params: tuple = ()) -> int:
        """Esegue una statement e ritorna `rowcount` (per l'idempotenza)."""
        with self._lock, self._conn.cursor() as cur:
            cur.execute(sql, params)
            return int(cur.rowcount)

    def _init_schema(self) -> None:
        """Crea tabelle e indici se non esistono (schema completo: niente migrazioni)."""
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS games (
                    game_id TEXT PRIMARY KEY,
                    created_at DOUBLE PRECISION NOT NULL,
                    num_players INTEGER NOT NULL,
                    seed BIGINT,
                    code_version TEXT,
                    rules_version TEXT,
                    client_id TEXT,
                    finished_at DOUBLE PRECISION,
                    aborted_at DOUBLE PRECISION,
                    aborted_reason TEXT
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id BIGSERIAL PRIMARY KEY,
                    game_id TEXT NOT NULL REFERENCES games(game_id),
                    created_at DOUBLE PRECISION NOT NULL,
                    server_version INTEGER,
                    player_index INTEGER,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_game_id ON events(game_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);")

    def ensure_game(
        self,
        game_id: str,
        *,
        num_players: int,
        seed: Optional[int] = None,
        code_version: Optional[str] = None,
        rules_version: Optional[str] = None,
    ) -> None:
        self._execute(
            """
            INSERT INTO games(game_id, created_at, num_players, seed, code_version, rules_version)
            VALUES(%s, %s, %s, %s, %s, %s)
            ON CONFLICT (game_id) DO NOTHING;
            """,
            (game_id, time.time(), num_players, seed, code_version, rules_version),
        )

    def set_client_id(self, game_id: str, *, client_id: str) -> None:
        cleaned = str(client_id).strip()
        if not cleaned:
            return
        self._execute(
            "UPDATE games SET client_id = COALESCE(client_id, %s) WHERE game_id = %s;",
            (cleaned, game_id),
        )

    def try_mark_game_finished(self, game_id: str, *, finished_at: Optional[float] = None) -> bool:
        ts = time.time() if finished_at is None else float(finished_at)
        rc = self._execute(
            "UPDATE games SET finished_at = %s WHERE game_id = %s AND finished_at IS NULL;",
            (ts, game_id),
        )
        return rc == 1

    def try_mark_game_aborted(self, game_id: str, *, aborted_reason: str, aborted_at: Optional[float] = None) -> bool:
        ts = time.time() if aborted_at is None else float(aborted_at)
        reason = (str(aborted_reason).strip() or "unknown")[:200]
        rc = self._execute(
            """
            UPDATE games SET aborted_at = %s, aborted_reason = %s
            WHERE game_id = %s AND finished_at IS NULL AND aborted_at IS NULL;
            """,
            (ts, reason, game_id),
        )
        return rc == 1

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
        ts = created_at if created_at is not None else time.time()
        payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self._execute(
            """
            INSERT INTO events(game_id, created_at, server_version, player_index, event_type, payload_json)
            VALUES(%s, %s, %s, %s, %s, %s);
            """,
            (game_id, ts, server_version, player_index, event_type, payload_json),
        )


def resolve_database_url() -> Optional[str]:
    """URL Postgres dalle env candidate (override esplicito prima), o None."""
    for name in ("BRISCOLA_DATABASE_URL", "DATABASE_URL"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def build_event_log(*, sqlite_path: Optional[str], database_url: Optional[str]) -> Optional[EventLogProtocol]:
    """
    Crea il backend event log: Postgres se `database_url` è presente, altrimenti SQLite se è dato un
    path, altrimenti `None` (feature disabilitata). In cloud multi-replica usare sempre Postgres:
    l'SQLite locale è per-replica ed effimero.
    """
    if database_url:
        return PostgresEventLog(database_url)
    if sqlite_path:
        return EventLog(EventLogConfig(path=sqlite_path))
    return None


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
