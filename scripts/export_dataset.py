#!/usr/bin/env python3
"""
Export dataset da SQLite (event log) a JSONL.

Perché JSONL?
------------
JSON Lines (JSONL) è un formato “streaming friendly”: ogni riga è un JSON valido.
È comodo per:
- append e processing riga-per-riga;
- import in tool di data processing (anche senza caricare tutto in memoria);
- training pipelines dove ogni record è una transizione o un esempio (observation → action).

Schema (versione 1)
-------------------
Ogni record è una singola azione `action_play_card` associata alla migliore observation
disponibile *prima* dell'azione e alla observation *subito dopo* (se presente).

Campi principali:
- `schema_version`: versione dello schema record (int)
- `game_id`: id della partita
- `event_id`: id dell'evento `action_play_card` nel DB
- `server_version`: versione monotona server-side (ordering/debug)
- `player_index`: indice del player che ha giocato
- `is_ai`: se l'azione è stata fatta dall'IA
- `observation`: snapshot (DTO observation) prima dell'azione (può essere `null` se mancante)
- `action`: `{ "card_index": int }`
- `reward`: reward “sparse” (punti della mano, signed)
- `next_observation`: observation dopo l'azione (se presente)
- `done`: True se `next_observation.game_over` è True, altrimenti False (o `null` se mancante)

Nota didattica
--------------
Questo export è una base “minima ma utile” per:
- imitation learning (supervised): basta `observation` + `action`;
- RL: si può usare anche `reward` + `next_observation` + `done`.

Uso
---
  python scripts/export_dataset.py --db ./data/briscola_events.sqlite3 --out ./data/dataset.jsonl

Per esportare solo azioni “umane” del player 0 (default UI):
  python scripts/export_dataset.py --player-index 0 --exclude-ai
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class ExportConfig:
    """Configurazione dell'export."""

    db_path: Path
    out_path: Path
    player_index: Optional[int]
    include_ai: bool
    include_next_state: bool
    schema_version: int = 1


def _compute_trick_points(trick_cards: Any) -> int:
    """
    Calcola i punti della mano a partire da `trick_cards` (DTO).

    `trick_cards` è atteso come lista di oggetti:
      { "card": { ..., "points": int }, "player_index": int }

    Se il formato non è quello atteso, ritorniamo 0 (best-effort).
    """
    if not isinstance(trick_cards, list):
        return 0

    total = 0
    for item in trick_cards:
        if not isinstance(item, dict):
            continue
        card = item.get("card")
        if not isinstance(card, dict):
            continue
        points = card.get("points")
        if isinstance(points, int):
            total += points
    return total


def _find_best_observation(observations: list[dict[str, Any]], *, card_index: int) -> Optional[dict[str, Any]]:
    """
    Trova la migliore observation “prima dell'azione”.

    Strategia:
    - scorriamo all'indietro (più recente → più vecchia)
    - scegliamo la prima observation che:
      - `my_turn == true` (è effettivamente il turno del player)
      - `valid_actions` contiene `card_index` (l'azione è coerente con lo snapshot)

    Se non troviamo una observation coerente, ritorniamo l'ultima disponibile (fallback),
    oppure `None` se non c'è nulla.
    """
    for obs in reversed(observations):
        if not isinstance(obs, dict):
            continue
        if (
            obs.get("my_turn") is True
            and isinstance(obs.get("valid_actions"), list)
            and card_index in obs["valid_actions"]
        ):
            return obs

    return observations[-1] if observations else None


def export_dataset(config: ExportConfig) -> dict[str, int]:
    """
    Esegue l'export SQLite → JSONL.

    Ritorna un riepilogo (contatori) utile per debug:
    - `rows_total`: eventi letti
    - `records_written`: righe JSONL scritte
    - `records_skipped_ai`: azioni IA scartate per filtro
    - `records_skipped_player`: azioni scartate per filtro player_index
    - `records_missing_observation`: record scritti senza observation coerente
    - `records_missing_next_observation`: record flushati senza next_observation
    """
    config.out_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(config.db_path))
    conn.row_factory = sqlite3.Row

    games = {
        row["game_id"]: {"num_players": row["num_players"], "seed": row["seed"]}
        for row in conn.execute("SELECT game_id, num_players, seed FROM games;")
    }

    counters = {
        "rows_total": 0,
        "records_written": 0,
        "records_skipped_ai": 0,
        "records_skipped_player": 0,
        "records_missing_observation": 0,
        "records_missing_next_observation": 0,
    }

    current_game_id: Optional[str] = None
    observations_by_player: dict[int, list[dict[str, Any]]] = {}
    pending_by_player: dict[int, dict[str, Any]] = {}

    def flush_game_pending() -> None:
        """
        Flush best-effort delle transizioni rimaste senza `next_observation`.

        In un log “completo” non dovrebbe succedere spesso (dopo ogni azione inviamo observation),
        ma questo rende l'export robusto a disconnessioni o log incompleti.
        """
        nonlocal pending_by_player
        if not pending_by_player:
            return

        with config.out_path.open("a", encoding="utf-8") as f:
            for _, record in list(pending_by_player.items()):
                record["next_observation"] = None
                record["done"] = None
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                counters["records_written"] += 1
                counters["records_missing_next_observation"] += 1
            pending_by_player = {}

    # Puliamo il file di output se esiste: export deterministico e ripetibile.
    config.out_path.write_text("", encoding="utf-8")

    try:
        rows = conn.execute(
            """
            SELECT id, game_id, server_version, player_index, event_type, payload_json
            FROM events
            ORDER BY game_id, id;
            """
        )

        for row in rows:
            counters["rows_total"] += 1
            game_id = row["game_id"]

            if current_game_id is None:
                current_game_id = game_id
            elif current_game_id != game_id:
                flush_game_pending()
                current_game_id = game_id
                observations_by_player = {}
                pending_by_player = {}

            event_type = row["event_type"]
            server_version = row["server_version"]
            db_player_index = row["player_index"]

            try:
                payload = json.loads(row["payload_json"])
            except json.JSONDecodeError:
                continue

            if event_type == "observation_sent":
                # Observation inviata al player: ci serve per costruire (s, a, r, s', done).
                if isinstance(db_player_index, int):
                    player_idx = db_player_index
                else:
                    # Fallback: alcuni payload includono `my_index`.
                    player_idx_raw = payload.get("my_index")
                    player_idx = int(player_idx_raw) if isinstance(player_idx_raw, int) else -1

                if player_idx < 0:
                    continue

                observations_by_player.setdefault(player_idx, []).append(payload)

                if config.include_next_state and player_idx in pending_by_player:
                    record = pending_by_player.pop(player_idx)
                    record["next_observation"] = payload
                    record["done"] = bool(payload.get("game_over") is True)
                    with config.out_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counters["records_written"] += 1

                continue

            if event_type != "action_play_card":
                continue

            is_ai = bool(payload.get("is_ai") is True)
            player_idx_raw = payload.get("player_index")
            card_index_raw = payload.get("card_index")
            result = payload.get("result")

            if (
                not isinstance(player_idx_raw, int)
                or not isinstance(card_index_raw, int)
                or not isinstance(result, dict)
            ):
                continue

            player_idx = player_idx_raw
            card_index = card_index_raw

            if not config.include_ai and is_ai:
                counters["records_skipped_ai"] += 1
                continue
            if config.player_index is not None and player_idx != config.player_index:
                counters["records_skipped_player"] += 1
                continue

            obs_history = observations_by_player.get(player_idx, [])
            observation = _find_best_observation(obs_history, card_index=card_index)

            if observation is None:
                counters["records_missing_observation"] += 1

            # Reward “sparse”: quando una mano si completa, assegniamo i punti della mano.
            # Usiamo un reward signed:
            # - +points se vince il player che ha agito
            # - -points se perde (stesso valore assoluto)
            # - 0 se la mano non si completa in questa azione
            reward = 0
            if result.get("trick_completed") is True:
                trick_points = _compute_trick_points(result.get("trick_cards"))
                winner = result.get("trick_winner")
                if isinstance(winner, int):
                    reward = trick_points if winner == player_idx else -trick_points

            record: dict[str, Any] = {
                "schema_version": config.schema_version,
                "game_id": game_id,
                "event_id": row["id"],
                "server_version": server_version,
                "player_index": player_idx,
                "is_ai": is_ai,
                "metadata": games.get(game_id, {}),
                "observation": observation,
                "action": {"card_index": card_index},
                "reward": reward,
                "result": result,
                "next_observation": None,
                "done": None,
            }

            if config.include_next_state:
                # Aspettiamo la prossima observation per completare la transizione.
                pending_by_player[player_idx] = record
            else:
                # Export "supervised-only": scriviamo subito (observation → action).
                record["done"] = bool((observation or {}).get("game_over") is True)
                with config.out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                counters["records_written"] += 1

        flush_game_pending()
        return counters
    finally:
        conn.close()


def main() -> int:
    """Entry point CLI."""
    parser = argparse.ArgumentParser(description="Export dataset da SQLite (event log) a JSONL")
    parser.add_argument("--db", default="./data/briscola_events.sqlite3", help="Path DB SQLite (event log)")
    parser.add_argument("--out", default="./data/dataset.jsonl", help="Path output JSONL")
    parser.add_argument("--player-index", type=int, default=0, help="Player da esportare (default: 0)")
    parser.add_argument("--all-players", action="store_true", help="Esporta azioni di tutti i player")
    parser.add_argument("--include-ai", action="store_true", help="Includi anche le azioni dell'IA")
    parser.add_argument(
        "--exclude-ai",
        action="store_true",
        help="Escludi le azioni dell'IA (override di --include-ai). Default: True per dataset umano.",
    )
    parser.add_argument(
        "--no-next-state",
        action="store_true",
        help="Esporta senza `next_observation` (solo observation → action).",
    )
    args = parser.parse_args()

    player_index = None if args.all_players else args.player_index
    include_ai = bool(args.include_ai)
    if args.exclude_ai:
        include_ai = False

    config = ExportConfig(
        db_path=Path(args.db),
        out_path=Path(args.out),
        player_index=player_index,
        include_ai=include_ai,
        include_next_state=not args.no_next_state,
    )

    if not config.db_path.exists():
        print(f"DB non trovato: {config.db_path}")
        return 2

    counters = export_dataset(config)
    print("Export completato.")
    for k, v in counters.items():
        print(f"- {k}: {v}")
    print(f"- output: {config.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
