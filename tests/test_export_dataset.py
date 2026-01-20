"""
Test per lo script di export dataset (SQLite → JSONL).

Scopo didattico:
- verificare che l'export produca un record coerente (observation → action → next_observation)
- bloccare regressioni sul formato minimo atteso (schema_version, reward, done)

Nota:
Qui non testiamo il gameplay completo; usiamo un DB SQLite minimale con pochi eventi
“finti ma realistici” per mantenere il test veloce e ripetibile.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path


def _init_minimal_event_db(db_path: Path) -> None:
    """Crea lo schema minimo (games/events) come prodotto dal logger reale."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE games (
                game_id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                num_players INTEGER NOT NULL,
                seed INTEGER
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                server_version INTEGER,
                player_index INTEGER,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_export_dataset_script_writes_jsonl(tmp_path: Path) -> None:
    """
    Verifica che lo script:
    - legga un DB valido
    - scriva un file JSONL
    - produca reward/done coerenti quando una mano si completa
    """
    db_path = tmp_path / "events.sqlite3"
    out_path = tmp_path / "dataset.jsonl"
    _init_minimal_event_db(db_path)

    game_id = "g1"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO games(game_id, created_at, num_players, seed) VALUES(?, ?, ?, ?);",
            (game_id, 0.0, 2, 123),
        )

        insert_event_sql = (
            "INSERT INTO events(game_id, created_at, server_version, player_index, event_type, payload_json) "
            "VALUES(?, ?, ?, ?, ?, ?);"
        )

        # Observation pre-azione: è il turno del player 0 e l'azione 1 è valida.
        obs_before = {
            "type": "observation",
            "server_version": 0,
            "my_index": 0,
            "my_turn": True,
            "my_hand": [
                {"suit": "cups", "rank": "ACE", "number": 1, "points": 11},
                {"suit": "coins", "rank": "TWO", "number": 2, "points": 0},
            ],
            "valid_actions": [0, 1],
            "table_cards": [],
            "trump_suit": "clubs",
            "cards_remaining_in_deck": 10,
            "game_over": False,
            "num_players": 2,
            "is_team_game": False,
            "players": [
                {"index": 0, "name": "P0", "points": 0, "hand_size": 2},
                {"index": 1, "name": "P1", "points": 0, "hand_size": 2},
            ],
        }
        conn.execute(
            insert_event_sql,
            (game_id, 0.0, 0, 0, "observation_sent", json.dumps(obs_before)),
        )

        # Azione player 0: completa la mano e vince (reward atteso +11).
        action_payload = {
            "is_ai": False,
            "player_index": 0,
            "card_index": 0,
            "result": {
                "server_version": 1,
                "played_card": {"suit": "cups", "rank": "ACE", "number": 1, "points": 11},
                "player": 0,
                "trick_completed": True,
                "trick_winner": 0,
                "trick_size": 2,
                "cards_dealt": False,
                "trick_cards": [
                    {"card": {"suit": "cups", "rank": "ACE", "number": 1, "points": 11}, "player_index": 0},
                    {"card": {"suit": "coins", "rank": "TWO", "number": 2, "points": 0}, "player_index": 1},
                ],
                "captured_cards": [],
            },
        }
        conn.execute(
            insert_event_sql,
            (game_id, 0.1, 1, 0, "action_play_card", json.dumps(action_payload)),
        )

        # Observation post-azione: il gioco non è finito.
        obs_after = {
            "type": "observation",
            "server_version": 1,
            "my_index": 0,
            "my_turn": False,
            "my_hand": [{"suit": "coins", "rank": "TWO", "number": 2, "points": 0}],
            "valid_actions": [],
            "table_cards": [],
            "trump_suit": "clubs",
            "cards_remaining_in_deck": 9,
            "game_over": False,
            "num_players": 2,
            "is_team_game": False,
            "players": [
                {"index": 0, "name": "P0", "points": 11, "hand_size": 1},
                {"index": 1, "name": "P1", "points": 0, "hand_size": 2},
            ],
        }
        conn.execute(
            insert_event_sql,
            (game_id, 0.2, 1, 0, "observation_sent", json.dumps(obs_after)),
        )

        conn.commit()
    finally:
        conn.close()

    script_path = Path(__file__).resolve().parent.parent / "scripts" / "export_dataset.py"
    proc = subprocess.run(
        [sys.executable, str(script_path), "--db", str(db_path), "--out", str(out_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Export completato." in proc.stdout

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])

    assert record["schema_version"] == 1
    assert record["game_id"] == game_id
    assert record["player_index"] == 0
    assert record["is_ai"] is False
    assert record["action"]["card_index"] == 0
    assert record["reward"] == 11
    assert record["observation"] is not None
    assert record["next_observation"] is not None
    assert record["done"] is False
