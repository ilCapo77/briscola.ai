"""
Test di integrazione minimale per l'event log SQLite.

Perché esiste questo test?
--------------------------
L'event log è una feature "da laboratorio": serve a rendere riproducibili e
osservabili le partite quando inizieremo a costruire dataset per ML.

Qui non testiamo la UI né il training:
verifichiamo solo che, quando il DB è configurato via env, il backend scriva
almeno alcuni eventi base (creazione partita e azione giocata).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from briscola_ai.backend import server


def test_event_log_writes_basic_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Abilita l'event log via env e verifica che il DB contenga eventi base.

    Nota:
    - Usiamo `TestClient` in un context manager per garantire startup/shutdown (lifespan).
    - Il path è sotto tempdir per evitare side effects nel repository.
    """
    # `tmp_path` è un path unico per test, già isolato: non serve creare sottocartelle.
    db_path = tmp_path / "events.sqlite3"
    monkeypatch.setenv("BRISCOLA_EVENT_DB_PATH", str(db_path))

    # Puliamo lo stato globale (come in `tests/test_api_integration.py`) per evitare
    # interferenze con altri test che usano `briscola_ai.backend.server`.
    server.active_games.clear()
    server.game_timestamps.clear()
    server.game_data.clear()
    server.game_locks.clear()
    server.game_versions.clear()
    server.connected_clients.clear()

    with TestClient(server.app) as client:
        create = client.post("/games", json={"num_players": 2, "player_names": ["A", "B"]})
        assert create.status_code == 200
        game_id = create.json()["game_id"]

        obs = client.get(f"/games/{game_id}", params={"player_index": 0}).json()
        action = client.post(
            f"/games/{game_id}/actions",
            json={"game_id": game_id, "player_index": 0, "card_index": obs["valid_actions"][0]},
        )
        assert action.status_code == 200

    # Verifica contenuto DB dopo lo shutdown (connessione chiusa e flush su disco).
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT event_type FROM events WHERE game_id = ? ORDER BY id ASC;",
            (game_id,),
        ).fetchall()
    finally:
        conn.close()

    event_types = [r[0] for r in rows]
    assert "game_created" in event_types
    assert "action_play_card" in event_types
