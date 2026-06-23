"""
Test di integrazione per l'API HTTP/WebSocket.

Questi test usano `fastapi.testclient.TestClient` e, di conseguenza, lavorano
contro lo stato globale mantenuto in `briscola_ai.backend.server`.

Nota: puliamo sempre lo stato globale con una fixture `autouse` per evitare
interferenze tra casi di test.
"""

import asyncio
import json
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from briscola_ai.backend import server
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.state import GameState, PlayerState
from briscola_ai.main import app as main_app


@pytest.fixture(autouse=True)
def _clean_server_state() -> Generator[None, None, None]:
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
    server.game_ai_agents.clear()
    server.game_action_rngs.clear()
    server.connected_clients.clear()
    yield
    server.active_games.clear()
    server.game_timestamps.clear()
    server.game_data.clear()
    server.game_locks.clear()
    server.game_versions.clear()
    server.game_ai_agents.clear()
    server.game_action_rngs.clear()
    server.connected_clients.clear()


def _write_dummy_bc_model_npz(path: Path) -> None:
    """
    Crea un file `.npz` minimo compatibile con `BCModelAgent`.

    Nota:
    Usiamo `D=248` che è la dimensione feature attuale di `encode_player_observation_2p` (v1).
    """
    d = 248
    h = 8
    rng = np.random.default_rng(0)
    w1 = rng.normal(size=(d, h)).astype(np.float32)
    b1 = np.zeros((h,), dtype=np.float32)
    w2 = rng.normal(size=(h, 40)).astype(np.float32)
    b2 = np.zeros((40,), dtype=np.float32)
    metadata = {
        "format": "mlp_bc_v1",
        "feature_dim": d,
        "hidden_dim": h,
        "action_dim": 40,
        "train": {"algorithm": "bc", "num_games": 1234},
        "description_it": "Modello dummy per test (non usare in produzione).",
    }
    np.savez(path, w1=w1, b1=b1, w2=w2, b2=b2, metadata_json=json.dumps(metadata, ensure_ascii=False))


def _write_dummy_bc_model_npz_with_feature_dim(
    path: Path,
    *,
    feature_dim: int,
    metrics: list[dict[str, float]] | None = None,
) -> None:
    """Come `_write_dummy_bc_model_npz`, ma con feature_dim configurabile (per test compatibilità)."""
    d = int(feature_dim)
    h = 8
    rng = np.random.default_rng(0)
    w1 = rng.normal(size=(d, h)).astype(np.float32)
    b1 = np.zeros((h,), dtype=np.float32)
    w2 = rng.normal(size=(h, 40)).astype(np.float32)
    b2 = np.zeros((40,), dtype=np.float32)
    metadata = {
        "format": "mlp_bc_v1",
        "feature_dim": d,
        "hidden_dim": h,
        "action_dim": 40,
        "train": {"algorithm": "bc", "num_games": 1},
        "label": "Dummy",
        "description_it": "Modello dummy per test compatibilità.",
    }
    if metrics is not None:
        metadata["metrics"] = metrics
    np.savez(path, w1=w1, b1=b1, w2=w2, b2=b2, metadata_json=json.dumps(metadata, ensure_ascii=False))


def test_backend_root_healthcheck() -> None:
    """Smoke test: l'endpoint root del backend risponde e contiene un messaggio."""
    client = TestClient(server.app)
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["message"]


def test_meta_exposes_event_log_mode_and_consent(monkeypatch: pytest.MonkeyPatch) -> None:
    """`GET /meta` deve indicare la modalità logging e se serve il consenso (dataset)."""
    monkeypatch.setenv("BRISCOLA_EVENT_LOG_MODE", "dataset")
    client = TestClient(server.app)
    r = client.get("/meta")
    assert r.status_code == 200
    payload = r.json()
    assert payload["event_log_mode"] == "dataset"
    assert payload["dataset_requires_consent"] is True

    monkeypatch.setenv("BRISCOLA_EVENT_LOG_MODE", "debug")
    r2 = client.get("/meta")
    assert r2.status_code == 200
    payload2 = r2.json()
    assert payload2["event_log_mode"] == "debug"
    assert payload2["dataset_requires_consent"] is False


def test_create_game_requires_consent_in_dataset_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """In `event_log_mode=dataset` il backend deve richiedere consenso esplicito."""
    monkeypatch.setenv("BRISCOLA_EVENT_LOG_MODE", "dataset")
    client = TestClient(server.app)

    missing = client.post(
        "/games",
        json={"num_players": 2, "player_names": ["Alice", "Bob"]},
    )
    assert missing.status_code == 400
    assert "Consenso" in missing.json()["detail"]

    ok = client.post(
        "/games",
        json={"num_players": 2, "player_names": ["Alice", "Bob"], "consent_to_data_collection": True},
    )
    assert ok.status_code == 200


def test_list_ai_agents_exposes_metadata_in_italian() -> None:
    """`GET /ai/agents` deve esporre nomi e descrizioni (in italiano) per la UI."""
    client = TestClient(server.app)
    r = client.get("/ai/agents")
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload, dict)
    assert isinstance(payload.get("common_note_it"), str)
    assert payload["common_note_it"]

    agents = payload.get("agents")
    assert isinstance(agents, list)
    assert agents

    by_name = {a["name"]: a for a in agents}
    assert "random" in by_name
    assert "greedy_points" in by_name
    assert "heuristic_v1" in by_name
    assert "heuristic_v2" in by_name
    assert "bc_model" in by_name

    assert isinstance(by_name["heuristic_v1"].get("description_it"), str)
    assert by_name["heuristic_v1"]["description_it"]
    assert isinstance(by_name["heuristic_v2"].get("description_it"), str)
    assert by_name["heuristic_v2"]["description_it"]


def test_list_ai_models_returns_model_catalog(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`GET /ai/models` deve elencare i modelli `.npz` disponibili (senza path assoluti)."""
    monkeypatch.setenv("BRISCOLA_MODELS_DIR", str(tmp_path))

    _write_dummy_bc_model_npz_with_feature_dim(
        tmp_path / "compatible_v1.npz",
        feature_dim=248,
        metrics=[{"episode": 1.0, "avg_score_diff": 2.0}, {"episode": 2.0, "avg_score_diff": 3.0}],
    )
    _write_dummy_bc_model_npz_with_feature_dim(tmp_path / "compatible_v2.npz", feature_dim=288)
    _write_dummy_bc_model_npz_with_feature_dim(tmp_path / "incompatible.npz", feature_dim=10)

    client = TestClient(server.app)
    r = client.get("/ai/models")
    assert r.status_code == 200

    payload = r.json()
    models = payload.get("models")
    assert isinstance(models, list)
    assert models

    assert "models_dir" not in payload  # non vogliamo esporre path server-side
    by_id = {m["id"]: m for m in models}
    assert "compatible_v1.npz" in by_id
    assert "compatible_v2.npz" in by_id
    assert "incompatible.npz" in by_id

    ok_v1 = by_id["compatible_v1.npz"]
    assert ok_v1["is_compatible"] is True
    assert ok_v1.get("compatibility_reason_it") is None
    assert ok_v1["metadata"]["metrics_count"] == 2
    assert "metrics" not in ok_v1["metadata"]

    ok_v2 = by_id["compatible_v2.npz"]
    assert ok_v2["is_compatible"] is True
    assert ok_v2.get("compatibility_reason_it") is None

    bad = by_id["incompatible.npz"]
    assert bad["is_compatible"] is False
    assert isinstance(bad.get("compatibility_reason_it"), str)
    assert bad["compatibility_reason_it"]


def test_create_game_supports_bc_model_with_ai_model_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`POST /games` deve supportare `ai_agent=bc_model` + `ai_model_id` (whitelisted)."""
    monkeypatch.setenv("BRISCOLA_MODELS_DIR", str(tmp_path))
    _write_dummy_bc_model_npz_with_feature_dim(tmp_path / "dummy_model.npz", feature_dim=248)
    _write_dummy_bc_model_npz_with_feature_dim(tmp_path / "dummy_model_v2.npz", feature_dim=288)
    _write_dummy_bc_model_npz_with_feature_dim(tmp_path / "bad_model.npz", feature_dim=10)

    client = TestClient(server.app)

    missing = client.post(
        "/games",
        json={"num_players": 2, "player_names": ["A", "B"], "ai_agent": "bc_model"},
    )
    assert missing.status_code == 400
    assert "ai_model_id" in missing.json()["detail"]

    traversal = client.post(
        "/games",
        json={
            "num_players": 2,
            "player_names": ["A", "B"],
            "ai_agent": "bc_model",
            "ai_model_id": "../dummy_model.npz",
        },
    )
    assert traversal.status_code == 400
    assert "path traversal" in traversal.json()["detail"].lower()

    ok = client.post(
        "/games",
        json={
            "num_players": 2,
            "player_names": ["A", "B"],
            "ai_agent": "bc_model",
            "ai_model_id": "dummy_model.npz",
        },
    )
    assert ok.status_code == 200
    payload = ok.json()
    assert payload["ai_agent"] == "bc_model"
    assert payload["ai_model_id"] == "dummy_model.npz"

    ok_v2 = client.post(
        "/games",
        json={
            "num_players": 2,
            "player_names": ["A", "B"],
            "ai_agent": "bc_model",
            "ai_model_id": "dummy_model_v2.npz",
        },
    )
    assert ok_v2.status_code == 200
    payload_v2 = ok_v2.json()
    assert payload_v2["ai_agent"] == "bc_model"
    assert payload_v2["ai_model_id"] == "dummy_model_v2.npz"

    bad = client.post(
        "/games",
        json={
            "num_players": 2,
            "player_names": ["A", "B"],
            "ai_agent": "bc_model",
            "ai_model_id": "bad_model.npz",
        },
    )
    assert bad.status_code == 400
    assert "feature_dim" in bad.json()["detail"]


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


def test_play_action_rejects_ai_controlled_player(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un client HTTP non deve poter pilotare manualmente il player controllato dall'IA."""

    async def _no_ai(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(server, "_maybe_ai_turn", _no_ai)

    client = TestClient(server.app)
    create = client.post("/games", json={"num_players": 2, "player_names": ["A", "IA"]})
    game_id = create.json()["game_id"]

    obs = client.get(f"/games/{game_id}", params={"player_index": 0}).json()
    first = client.post(
        f"/games/{game_id}/actions",
        json={"game_id": game_id, "player_index": 0, "card_index": obs["valid_actions"][0]},
    )
    assert first.status_code == 200

    blocked = client.post(
        f"/games/{game_id}/actions",
        json={"game_id": game_id, "player_index": 1, "card_index": 0},
    )
    assert blocked.status_code == 400
    assert "controllato dall'IA" in blocked.json()["detail"]


def test_main_app_serves_ui_and_mounts_api() -> None:
    """La FastAPI principale deve servire UI statica e montare `/api/`."""
    client = TestClient(main_app)

    root = client.get("/")
    assert root.status_code == 200
    assert "text/html" in root.headers.get("content-type", "")
    assert root.headers.get("cache-control") == "no-cache"
    assert "__BRISCOLA_ASSET_VERSION__" not in root.text
    assert "/static/css/style.css?v=" in root.text
    assert "/static/js/game.js?v=" in root.text

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


def test_get_game_state_without_player_index_returns_game_state_dto() -> None:
    """Contratto: `GET /games/{id}` (senza player_index) ritorna `type: \"game_state\"`."""
    client = TestClient(server.app)
    create = client.post("/games", json={"num_players": 2, "player_names": ["A", "B"]})
    game_id = create.json()["game_id"]

    r = client.get(f"/games/{game_id}")
    assert r.status_code == 200
    payload = r.json()

    assert payload["type"] == "game_state"
    assert payload["num_players"] == 2
    assert payload["is_team_game"] is False
    assert isinstance(payload.get("players"), list)
    assert len(payload["players"]) == 2

    # Deve includere mani complete (debug/spectator), quindi `hand` è presente.
    assert "hand" in payload["players"][0]
    assert isinstance(payload["players"][0]["hand"], list)


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
    payload = r.json()

    # Contratto: anche quando la partita è in corso, il risultato ha shape stabile (DTO).
    assert payload["type"] == "game_result"
    assert payload["game_in_progress"] is True
    assert payload["game_over"] is False
    assert payload["is_team_game"] is False
    assert payload["points"] == {}


def test_get_game_result_2p_finished_returns_stable_dto() -> None:
    """
    `/result` deve avere shape stabile anche a partita terminata (2-player).

    Nota didattica:
    qui usiamo lo stato in memoria del server per creare un end-game "deterministico"
    (evitando di dover giocare 40 mosse via HTTP in un test d'integrazione).
    """
    client = TestClient(server.app)
    create = client.post("/games", json={"num_players": 2, "player_names": ["A", "B"]})
    game_id = create.json()["game_id"]

    # Forziamo uno stato finale coerente.
    server.active_games[game_id] = GameState(
        num_players=2,
        is_team_game=False,
        teams=None,
        players=(
            PlayerState(name="A", hand=tuple(), captured_cards=tuple(), points=70),
            PlayerState(name="B", hand=tuple(), captured_cards=tuple(), points=50),
        ),
        deck=tuple(),
        trump_card=Card(Suit.CUPS, Rank.TWO),
        table_cards=tuple(),
        current_turn=0,
        first_player=0,
        game_over=True,
        winner_index=0,
        winning_team=None,
    )
    server.game_versions[game_id] = 123

    r = client.get(f"/games/{game_id}/result")
    assert r.status_code == 200
    payload = r.json()

    assert payload["type"] == "game_result"
    assert payload["server_version"] == 123
    assert payload["game_in_progress"] is False
    assert payload["game_over"] is True
    assert payload["is_team_game"] is False
    assert payload["winner"] == "A"
    assert payload["winner_index"] == 0
    assert payload["points"] == {"A": 70, "B": 50}
    assert payload["point_difference"] == 20


def test_get_game_result_2p_tie_omits_winner_index() -> None:
    """In pareggio 2-player, `winner_index` deve essere None (e quindi non presente nel JSON)."""
    client = TestClient(server.app)
    create = client.post("/games", json={"num_players": 2, "player_names": ["A", "B"]})
    game_id = create.json()["game_id"]

    server.active_games[game_id] = GameState(
        num_players=2,
        is_team_game=False,
        teams=None,
        players=(
            PlayerState(name="A", hand=tuple(), captured_cards=tuple(), points=60),
            PlayerState(name="B", hand=tuple(), captured_cards=tuple(), points=60),
        ),
        deck=tuple(),
        trump_card=Card(Suit.CUPS, Rank.TWO),
        table_cards=tuple(),
        current_turn=0,
        first_player=0,
        game_over=True,
        winner_index=None,
        winning_team=None,
    )

    r = client.get(f"/games/{game_id}/result")
    assert r.status_code == 200
    payload = r.json()

    assert payload["winner"] == "Pareggio"
    assert "winner_index" not in payload


def test_get_game_result_4p_finished_returns_team_fields() -> None:
    """`/result` deve esporre `team_points` e `winning_team` quando la partita è a squadre."""
    client = TestClient(server.app)
    create = client.post("/games", json={"num_players": 4, "player_names": ["A", "B", "C", "D"]})
    game_id = create.json()["game_id"]

    server.active_games[game_id] = GameState(
        num_players=4,
        is_team_game=True,
        teams=((0, 2), (1, 3)),
        players=(
            PlayerState(name="A", hand=tuple(), captured_cards=tuple(), points=40),
            PlayerState(name="B", hand=tuple(), captured_cards=tuple(), points=20),
            PlayerState(name="C", hand=tuple(), captured_cards=tuple(), points=30),
            PlayerState(name="D", hand=tuple(), captured_cards=tuple(), points=30),
        ),
        deck=tuple(),
        trump_card=Card(Suit.CUPS, Rank.TWO),
        table_cards=tuple(),
        current_turn=0,
        first_player=0,
        game_over=True,
        winner_index=None,
        winning_team=0,
    )

    r = client.get(f"/games/{game_id}/result")
    assert r.status_code == 200
    payload = r.json()

    assert payload["is_team_game"] is True
    assert payload["winner"] == "Squadra 0 (A e C)"
    assert payload["winning_team"] == 0
    assert payload["team_points"] == {"Team 0": 70, "Team 1": 50}
    assert payload["points"] == {"A": 40, "B": 20, "C": 30, "D": 30}
    assert payload["point_difference"] == 20


def test_get_game_result_4p_tie_omits_winning_team() -> None:
    """In pareggio 4-player, `winning_team` deve essere None (e quindi non presente nel JSON)."""
    client = TestClient(server.app)
    create = client.post("/games", json={"num_players": 4, "player_names": ["A", "B", "C", "D"]})
    game_id = create.json()["game_id"]

    server.active_games[game_id] = GameState(
        num_players=4,
        is_team_game=True,
        teams=((0, 2), (1, 3)),
        players=(
            PlayerState(name="A", hand=tuple(), captured_cards=tuple(), points=30),
            PlayerState(name="B", hand=tuple(), captured_cards=tuple(), points=30),
            PlayerState(name="C", hand=tuple(), captured_cards=tuple(), points=30),
            PlayerState(name="D", hand=tuple(), captured_cards=tuple(), points=30),
        ),
        deck=tuple(),
        trump_card=Card(Suit.CUPS, Rank.TWO),
        table_cards=tuple(),
        current_turn=0,
        first_player=0,
        game_over=True,
        winner_index=None,
        winning_team=None,
    )

    r = client.get(f"/games/{game_id}/result")
    assert r.status_code == 200
    payload = r.json()

    assert payload["winner"] == "Pareggio"
    assert "winning_team" not in payload


def test_server_version_is_monotone_on_actions_when_ai_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifica che `server_version` sia monotona sugli endpoint HTTP, senza rumore da task IA.

    Nota:
    - disabilitiamo il task IA per rendere il test deterministico.
    - giochiamo 3 azioni seguendo `current_turn` del GameStateDTO (debug).
    """

    async def _no_ai(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(server, "_maybe_ai_turn", _no_ai)

    client = TestClient(server.app)
    create = client.post("/games", json={"num_players": 2, "player_names": ["A", "B"]})
    game_id = create.json()["game_id"]
    # Questo test esercita volutamente il loop HTTP manuale per entrambi i player.
    server.game_ai_agents.pop(game_id, None)

    # Versione iniziale: 0
    obs0 = client.get(f"/games/{game_id}", params={"player_index": 0}).json()
    assert obs0["server_version"] == 0

    versions = []
    for _ in range(3):
        full = client.get(f"/games/{game_id}").json()
        current_turn = full["current_turn"]
        card_index = full["valid_actions"][0]

        out = client.post(
            f"/games/{game_id}/actions",
            json={"game_id": game_id, "player_index": current_turn, "card_index": card_index},
        ).json()
        versions.append(out["server_version"])

    assert versions == sorted(versions)
    assert versions == [1, 2, 3]


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
    assert isinstance(ws_obs.get("seen_cards_onehot"), list)
    assert isinstance(http_obs.get("seen_cards_onehot"), list)
    assert len(ws_obs["seen_cards_onehot"]) == 40
    assert len(http_obs["seen_cards_onehot"]) == 40

    # Controllo minimo su un elemento card: deve avere i campi DTO attesi.
    if http_obs["my_hand"]:
        card = http_obs["my_hand"][0]
        assert set(card.keys()) >= {"suit", "rank", "number", "points"}


def test_server_lifespan_cancels_cleanup_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allo shutdown dell'app, il task di cleanup periodico deve essere cancellato."""
    cancelled = {"value": False}

    async def fake_cleanup_inactive_games() -> None:
        # Implementazione fake: serve solo a verificare che il lifespan cancelli il task.
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


def test_create_game_allows_selecting_ai_agent_and_ai_turn_uses_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Regressione anti-cheat:
    - il client può scegliere l'agente IA all'avvio (`ai_agent`)
    - quando l'IA gioca, la policy riceve una `PlayerObservation` (non `GameState`)
    """

    async def _no_ai(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(server, "_maybe_ai_turn", _no_ai)

    class CheatDetectingAgent:
        name = "cheat_detector"

        def choose_card_index(self, observation, *, rng):  # noqa: ANN001
            assert observation.player_index == 1
            assert not hasattr(observation, "deck")
            assert not hasattr(observation, "players")
            assert observation.deck_size >= 0
            assert len(observation.hand) > 0
            return 0

    client = TestClient(server.app)
    create = client.post(
        "/games",
        json={"num_players": 2, "player_names": ["Alice", "IA"], "ai_agent": "heuristic_v1"},
    )
    assert create.status_code == 200
    game_id = create.json()["game_id"]

    # Verifica che la selezione sia stata applicata (per la UI: player 1 è l'IA).
    assert server.game_ai_agents[game_id][1].name == "heuristic_v1"

    # Sostituiamo l'agente con uno che fallisce se vede informazione nascosta.
    server.game_ai_agents[game_id][1] = CheatDetectingAgent()

    # Giochiamo una mossa umana per passare il turno all'IA.
    obs0 = client.get(f"/games/{game_id}", params={"player_index": 0}).json()
    assert obs0["my_turn"] is True
    play = client.post(
        f"/games/{game_id}/actions",
        json={"game_id": game_id, "player_index": 0, "card_index": obs0["valid_actions"][0]},
    )
    assert play.status_code == 200

    # Eseguiamo una mossa IA (sincrona per test) sotto lock.
    async def _run_ai_once() -> None:
        async with server.game_locks[game_id]:
            await server._execute_ai_turn_locked(game_id, human_player_index=0)

    asyncio.run(_run_ai_once())
