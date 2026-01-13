"""
Test di integrazione per l'API HTTP/WebSocket.

Questi test usano `fastapi.testclient.TestClient` e, di conseguenza, lavorano
contro lo stato globale mantenuto in `briscola_ai.backend.server`.

Nota: puliamo sempre lo stato globale con una fixture `autouse` per evitare
interferenze tra casi di test.
"""

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from briscola_ai.backend import server
from briscola_ai.main import app as main_app


@pytest.fixture(autouse=True)
def _clean_server_state() -> None:
    """
    I test d'integrazione usano stato globale in `briscola_ai.backend.server`.

    Per evitare interferenze tra test, puliamo le strutture in memoria
    prima/dopo ogni test.
    """
    server.active_games.clear()
    server.game_timestamps.clear()
    server.game_data.clear()
    server.game_locks.clear()
    server.game_versions.clear()
    server.connected_clients.clear()
    yield
    server.active_games.clear()
    server.game_timestamps.clear()
    server.game_data.clear()
    server.game_locks.clear()
    server.game_versions.clear()
    server.connected_clients.clear()


def test_backend_root_healthcheck() -> None:
    """Smoke test: l'endpoint root del backend risponde e contiene un messaggio."""
    client = TestClient(server.app)
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["message"]


def test_create_game_get_state_and_play_action_happy_path() -> None:
    """Happy path: crea partita, legge observation e gioca una carta valida."""
    client = TestClient(server.app)

    create = client.post(
        "/games",
        json={"num_players": 2, "player_names": ["Alice", "Bob"]},
    )
    assert create.status_code == 200
    payload = create.json()
    game_id = payload["game_id"]
    assert payload["status"] == "created"
    assert payload["num_players"] == 2
    assert payload["player_names"] == ["Alice", "Bob"]

    state_p0 = client.get(f"/games/{game_id}", params={"player_index": 0})
    assert state_p0.status_code == 200
    obs = state_p0.json()
    assert obs["my_index"] == 0
    assert obs["my_turn"] is True
    assert obs["valid_actions"]
    initial_version = obs.get("server_version", 0)

    action = client.post(
        f"/games/{game_id}/actions",
        json={"game_id": game_id, "player_index": 0, "card_index": obs["valid_actions"][0]},
    )
    assert action.status_code == 200
    result = action.json()
    assert "played_card" in result or "error" in result
    assert "error" not in result
    played_card = result.get("played_card")

    state_p0_after = client.get(f"/games/{game_id}", params={"player_index": 0})
    assert state_p0_after.status_code == 200
    obs_after = state_p0_after.json()

    # Con modello server-driven l'IA può giocare "subito" (in un task asincrono) e quindi,
    # tra la POST e questa GET, lo stato potrebbe essere già avanzato oltre il semplice
    # "dopo la carta umana" (es. mano completa + nuova carta IA come prima di mano).
    #
    # Invece di assumere un `my_turn` specifico, verifichiamo invarianti più robuste:
    # - la `server_version` è avanzata almeno di 1 (abbiamo giocato un'azione umana)
    # - la carta giocata non è più nella mano del giocatore 0
    assert obs_after.get("server_version", 0) >= initial_version + 1

    if played_card:
        played_suit = played_card.get("suit")
        played_number = played_card.get("number")
        assert played_suit is not None
        assert played_number is not None

        assert not any(
            (card.get("suit") == played_suit and card.get("number") == played_number) for card in obs_after["my_hand"]
        )


def test_play_action_rejects_wrong_turn() -> None:
    """Regola di turnazione: un giocatore non può giocare quando non è il suo turno."""
    client = TestClient(server.app)

    create = client.post("/games", json={"num_players": 2, "player_names": ["A", "B"]})
    game_id = create.json()["game_id"]

    # A inizia (player_index=0). B prova a giocare: deve fallire.
    r = client.post(
        f"/games/{game_id}/actions",
        json={"game_id": game_id, "player_index": 1, "card_index": 0},
    )
    assert r.status_code == 400
    assert r.json()["detail"] == "Non è il tuo turno"


def test_main_app_serves_ui_and_mounts_api() -> None:
    """La FastAPI principale deve servire UI statica e montare `/api/`."""
    client = TestClient(main_app)

    root = client.get("/")
    assert root.status_code == 200
    assert "text/html" in root.headers.get("content-type", "")

    card_asset = client.get("/static/assets/cards/clubs_1.png")
    assert card_asset.status_code == 200
    assert "image" in card_asset.headers.get("content-type", "")

    api_root = client.get("/api/")
    assert api_root.status_code == 200
    assert api_root.json()["message"]


def test_get_game_state_returns_404_for_unknown_game() -> None:
    """Errore corretto: stato partita inesistente => 404."""
    client = TestClient(server.app)
    r = client.get("/games/not-a-real-game-id")
    assert r.status_code == 404
    assert r.json()["detail"] == "Partita non trovata"


def test_get_game_state_rejects_invalid_player_index() -> None:
    """Errore corretto: player_index fuori range => 400."""
    client = TestClient(server.app)
    create = client.post("/games", json={"num_players": 2, "player_names": ["A", "B"]})
    game_id = create.json()["game_id"]

    r = client.get(f"/games/{game_id}", params={"player_index": 999})
    assert r.status_code == 400
    assert "indice giocatore" in r.json()["detail"].lower()


def test_get_game_result_returns_404_for_unknown_game() -> None:
    """Errore corretto: result di partita inesistente => 404."""
    client = TestClient(server.app)
    r = client.get("/games/not-a-real-game-id/result")
    assert r.status_code == 404
    assert r.json()["detail"] == "Partita non trovata"


def test_get_game_result_returns_in_progress_when_game_not_finished() -> None:
    """Se la partita non è terminata, `/result` deve indicare che è in progress."""
    client = TestClient(server.app)
    create = client.post("/games", json={"num_players": 2, "player_names": ["A", "B"]})
    game_id = create.json()["game_id"]

    r = client.get(f"/games/{game_id}/result")
    assert r.status_code == 200
    assert r.json() == {"game_in_progress": True}


def test_websocket_rejects_unknown_game() -> None:
    """WebSocket su partita inesistente: il server chiude subito la connessione."""
    client = TestClient(server.app)
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/ws/not-a-real-game-id/0"):
            pass
    assert excinfo.value.code == 1000


def test_websocket_ping_pong_and_receives_update_after_action() -> None:
    """WS: ping/pong funziona e, dopo una giocata HTTP, arriva uno snapshot aggiornato."""
    client = TestClient(server.app)

    create = client.post("/games", json={"num_players": 2, "player_names": ["Alice", "Bob"]})
    game_id = create.json()["game_id"]

    with client.websocket_connect(f"/ws/{game_id}/0") as ws:
        initial = ws.receive_json()
        assert initial["type"] == "observation"
        assert initial["my_index"] == 0
        assert initial["my_turn"] is True
        assert initial["valid_actions"]

        ws.send_json({"type": "ping"})
        pong = ws.receive_json()
        assert pong == {"type": "pong"}

        play = client.post(
            f"/games/{game_id}/actions",
            json={"game_id": game_id, "player_index": 0, "card_index": initial["valid_actions"][0]},
        )
        assert play.status_code == 200

        updated = ws.receive_json()
        assert updated["type"] == "observation"
        assert updated["my_index"] == 0
        assert updated["my_turn"] is False
        assert updated["valid_actions"] == []


def test_http_observation_matches_ws_observation_format() -> None:
    """
    Contratto: `GET /games/{id}?player_index=X` deve restituire lo stesso formato
    di uno snapshot WS (`type: "observation"`).

    Nota:
    - non confrontiamo l'intero payload per uguaglianza byte-per-byte (ordine chiavi),
      ma fissiamo campi e shape principali: `type`, `players`, `table_cards`, `my_hand`.
    """
    client = TestClient(server.app)

    create = client.post("/games", json={"num_players": 2, "player_names": ["Alice", "Bob"]})
    assert create.status_code == 200
    game_id = create.json()["game_id"]

    with client.websocket_connect(f"/ws/{game_id}/0") as ws:
        ws_obs = ws.receive_json()

    http_obs = client.get(f"/games/{game_id}", params={"player_index": 0}).json()

    assert ws_obs["type"] == "observation"
    assert http_obs["type"] == "observation"

    # Shape principale: players e table_cards sono strutture "esplicite" (DTO), non tuple/chiavi dinamiche.
    assert isinstance(ws_obs.get("players"), list)
    assert isinstance(http_obs.get("players"), list)
    assert isinstance(ws_obs.get("table_cards"), list)
    assert isinstance(http_obs.get("table_cards"), list)
    assert isinstance(ws_obs.get("my_hand"), list)
    assert isinstance(http_obs.get("my_hand"), list)

    # Controllo minimo su un elemento card: deve avere i campi DTO attesi.
    if http_obs["my_hand"]:
        card = http_obs["my_hand"][0]
        assert set(card.keys()) >= {"suit", "rank", "number", "points"}


def test_server_lifespan_cancels_cleanup_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allo shutdown dell'app, il task di cleanup periodico deve essere cancellato."""
    cancelled = {"value": False}

    async def fake_cleanup_inactive_games():  # type: ignore[no-untyped-def]
        try:
            while True:
                await server.asyncio.sleep(3600)
        except server.asyncio.CancelledError:
            cancelled["value"] = True
            raise

    monkeypatch.setattr(server, "cleanup_inactive_games", fake_cleanup_inactive_games)

    with TestClient(server.app):
        pass

    assert cancelled["value"] is True
