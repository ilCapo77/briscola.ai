"""
API backend (FastAPI) per Briscola AI.

Questo modulo espone:
- endpoint HTTP per creare una partita, ottenere lo stato e giocare una carta
- endpoint WebSocket per inviare aggiornamenti in tempo reale ai client

Scelte implementative (didattiche):
- Lo stato delle partite è tenuto in memoria (`active_games`). È semplice ma non persistente:
  riavviando il server si perdono le partite in corso.
- Le azioni vengono registrate in `game_data` come base per una pipeline ML futura.

Nel refactor profondo previsto in `PLAN.md` sposteremo verso:
- dominio “puro” (motore) separato dall'API
- persistenza su SQLite (event log) e export dataset.
"""

import asyncio
import json
import os
import random
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..ai.agents import AI_AGENTS_COMMON_NOTE_IT, Agent, build_agent, list_agent_specs
from ..ai.model_catalog import (
    get_models_dir_from_env,
    list_local_models,
    resolve_model_path,
    validate_model_compatible_for_ui,
)
from ..domain.engine import PlayCardAction, step
from ..domain.observation import make_player_observation
from ..domain.state import GameState as DomainGameState
from ..domain.state import new_game_state
from ..versioning import get_code_version, get_rules_version
from .dto import (
    AiCardRevealDTO,
    CardDTO,
    GameResultDTO,
    PlayActionResultDTO,
    TableCardDTO,
    TrickResultDTO,
)
from .event_log import EventLog, EventLogConfig, parse_event_db_path
from .observation_builder import build_game_state_dto, build_observation_dto


# Modelli per richieste e risposte API
class GameConfig(BaseModel):
    """Payload per creare una partita."""

    num_players: int
    player_names: Optional[List[str]] = None
    ai_agent: Optional[str] = None
    ai_model_id: Optional[str] = None
    client_id: Optional[str] = None
    consent_to_data_collection: Optional[bool] = None


class GameAction(BaseModel):
    """Payload per giocare una carta."""

    game_id: str
    player_index: int
    card_index: int
    # Metadati client-side (opzionali): utili per analisi qualità dati umani.
    client_observed_server_version: Optional[int] = None
    client_decision_time_ms: Optional[int] = None


class GameState(BaseModel):
    """Payload per richiedere lo stato di una partita (opzionale: vista per giocatore)."""

    game_id: str
    player_index: Optional[int] = None


def _get_event_log() -> Optional[EventLog]:
    """
    Helper per accedere al logger dalla app FastAPI.

    Per semplicità, il riferimento vive in `app.state.event_log` e viene inizializzato
    nel lifespan. Se la feature non è configurata, ritorniamo `None`.
    """
    return getattr(app.state, "event_log", None)


def _get_event_log_mode() -> str:
    """
    Modalità di logging eventi.

    - `debug` (default): log completo, utile per debug (include `observation_sent` e lifecycle WS).
    - `dataset`: log minimale, orientato a dataset umano (riduce molto la dimensione del DB).
    - `off`: non loggare nulla (senza cambiare DB path).
    """
    raw = os.getenv("BRISCOLA_EVENT_LOG_MODE", "debug").strip().lower()
    if raw in {"debug", "dataset", "off"}:
        return raw
    return "debug"


def _safe_log_event(
    game_id: str,
    event_type: str,
    payload: dict,
    *,
    server_version: Optional[int] = None,
    player_index: Optional[int] = None,
) -> None:
    """
    Wrapper “best-effort” per loggare eventi.

    Il logging è un optional feature: se il DB non è configurato o se una scrittura fallisce
    non vogliamo interrompere la partita.
    """
    log = _get_event_log()
    if log is None:
        return

    mode = _get_event_log_mode()
    if mode == "off":
        return
    if mode == "dataset":
        # In modalità dataset riduciamo il DB evitando payload ridondanti e molto grandi
        # (soprattutto `observation_sent` per tutti i player dopo ogni azione).
        allowed = {"game_created", "human_action", "game_finished", "game_aborted"}
        if event_type not in allowed:
            return

    try:
        # Garantiamo che la partita esista nella tabella `games` (idempotente).
        state = active_games.get(game_id)
        if state is not None:
            # Compatibilità: se il dominio non espone `seed`, proviamo a prenderlo dal payload.
            seed = getattr(state, "seed", None)
            if seed is None:
                seed_from_payload = payload.get("seed")
                seed = seed_from_payload if isinstance(seed_from_payload, int) else None

            log.ensure_game(
                game_id,
                num_players=state.num_players,
                seed=seed,
                code_version=get_code_version(),
                rules_version=get_rules_version(),
            )
        log.log_event(
            game_id,
            event_type,
            payload,
            server_version=server_version,
            player_index=player_index,
        )
    except Exception:
        # Best-effort: non propaghiamo eccezioni lato API/WS.
        print(f"Event log SQLite: errore scrittura evento {event_type!r} (game_id={game_id}).")


def _safe_set_client_id(game_id: str, client_id: Optional[str]) -> None:
    """Best-effort: salva `client_id` nella tabella `games` (se event log abilitato)."""
    if not client_id:
        return
    log = _get_event_log()
    if log is None or _get_event_log_mode() == "off":
        return
    try:
        log.set_client_id(game_id, client_id=str(client_id))
    except Exception:
        pass


def _maybe_log_game_finished(game_id: str, *, state: DomainGameState) -> None:
    """
    Se la partita è finita (`game_over=true`), logga un evento `game_finished` (best-effort).

    Nota didattica:
    questo evento serve soprattutto per filtrare dataset: esportiamo solo partite complete.
    """
    if not state.game_over:
        return

    log = _get_event_log()
    if log is None or _get_event_log_mode() == "off":
        return

    try:
        # Garantiamo l'anchor in tabella `games` (idempotente).
        seed = getattr(state, "seed", None)
        log.ensure_game(
            game_id,
            num_players=state.num_players,
            seed=seed if isinstance(seed, int) else None,
            code_version=get_code_version(),
            rules_version=get_rules_version(),
        )
        updated = log.try_mark_game_finished(game_id)
    except Exception:
        updated = False

    if not updated:
        return

    final_points = [p.points for p in state.players]
    winning_index: int | None = None
    if not state.is_team_game:
        best = max(final_points) if final_points else 0
        winners = [i for i, pts in enumerate(final_points) if pts == best]
        winning_index = winners[0] if len(winners) == 1 else None

    _safe_log_event(
        game_id,
        "game_finished",
        {
            "game_over": True,
            "num_players": state.num_players,
            "is_team_game": state.is_team_game,
            "final_points_by_player_index": final_points,
            "winning_player_index": winning_index,
        },
        server_version=game_versions.get(game_id, 0),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestisce startup/shutdown dell'app FastAPI.

    Usiamo un task in background per fare periodicamente cleanup delle partite inattive.
    """
    # Inizializzazione event log (Phase 4).
    #
    # Nota importante: questo backend può essere eseguito in due modi:
    # - direttamente (`TestClient(server.app)` / uvicorn su `briscola_ai.backend.server:app`)
    # - montato dentro l'app principale (`briscola_ai.main:app`)
    #
    # In alcuni setup i mounted sub-app non ricevono eventi lifespan; per questo motivo,
    # l'app principale può inizializzare `app.state.event_log` in anticipo.
    #
    # Qui adottiamo quindi una regola semplice:
    # - se `app.state.event_log` esiste già, non lo tocchiamo (né lo chiudiamo).
    # - altrimenti, proviamo a crearlo da env.
    event_log_created_here = False
    raw_path = parse_event_db_path(os.getenv("BRISCOLA_EVENT_DB_PATH"))

    existing_event_log = getattr(app.state, "event_log", None)
    event_log: Optional[EventLog] = existing_event_log

    # Se il path cambia tra due startup (tipico nei test), ricreiamo la connessione.
    # Se il path è disabilitato, chiudiamo e azzeriamo.
    if event_log is not None:
        if raw_path is None:
            try:
                event_log.close()
            except Exception:
                pass
            event_log = None
            app.state.event_log = None
        elif event_log.path != raw_path:
            try:
                event_log.close()
            except Exception:
                pass
            event_log = None
            app.state.event_log = None

    if event_log is None and raw_path is not None:
        try:
            event_log = EventLog(EventLogConfig(path=raw_path))
            event_log_created_here = True
        except Exception:
            # Il logger è un "optional feature": se fallisce non vogliamo bloccare il server.
            print("Event log SQLite: inizializzazione fallita, feature disabilitata.")
            event_log = None
        app.state.event_log = event_log

    cleanup_task = asyncio.create_task(cleanup_inactive_games())
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        if event_log is not None and event_log_created_here:
            event_log.close()
            app.state.event_log = None


# Crea l'app FastAPI
#
# Nota:
# usiamo `get_code_version()` per tenere allineata la versione OpenAPI con la versione del pacchetto
# (o con l'override via env `BRISCOLA_CODE_VERSION`).
app = FastAPI(title="Briscola AI API", version=get_code_version(), lifespan=lifespan)


def _parse_cors_allow_origins() -> list[str]:
    """
    Parsea `BRISCOLA_CORS_ALLOW_ORIGINS` (CSV) per limitare le origin ammesse.

    Esempi:
    - `BRISCOLA_CORS_ALLOW_ORIGINS=https://example.com`
    - `BRISCOLA_CORS_ALLOW_ORIGINS=https://a.com,https://b.com`

    Default:
    - se la variabile non è impostata, usiamo `*` (comportamento “dev-friendly”).
    """
    raw = os.getenv("BRISCOLA_CORS_ALLOW_ORIGINS", "").strip()
    if not raw:
        return ["*"]
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or ["*"]


# Aggiunge middleware CORS per consentire richieste cross-origin.
#
# Nota:
# In produzione è consigliato impostare `BRISCOLA_CORS_ALLOW_ORIGINS` con il tuo dominio.
cors_allow_origins = _parse_cors_allow_origins()
cors_allow_credentials = "*" not in cors_allow_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Conserva le partite attive in memoria (in un'app reale, usare un database)
active_games: Dict[str, DomainGameState] = {}
game_timestamps: Dict[str, datetime] = {}
game_data: Dict[str, List[Dict]] = {}  # Memorizza le azioni per il training ML

# Sincronizzazione e versioning server-side (per evitare race/doppi trigger IA):
# - `game_locks`: garantisce che le mutazioni di stato di una partita siano serializzate.
# - `game_versions`: contatore monotono incrementato ad ogni `play_action` (umano o IA).
#   Lo includiamo negli snapshot/messaggi WS come metadato debug-friendly (ordering/reconnect).
game_locks: Dict[str, asyncio.Lock] = {}
game_versions: Dict[str, int] = {}

# Config IA per partita (2-player UI: umano=0, IA=1).
#
# Nota:
# - gli agenti sono stateless: li istanziamo una volta per partita e li riusiamo.
# - l'RNG delle scelte IA è separato dallo shuffle del mazzo (seed partita), per riproducibilità.
game_ai_agents: Dict[str, dict[int, Agent]] = {}
game_action_rngs: Dict[str, random.Random] = {}

# Connessioni WebSocket
connected_clients: Dict[str, Dict[int, WebSocket]] = {}

_DEFAULT_AI_AGENT_NAME = "random"


def _is_ai_controlled_player(game_id: str, player_index: int) -> bool:
    """
    Ritorna True se `player_index` è controllato dal backend per questa partita.

    Nota di sicurezza:
    la UI attuale espone solo l'umano come player 0 e l'IA come player 1. L'endpoint HTTP
    resta però chiamabile manualmente: questa guardia evita che un client giochi le mosse
    del player controllato dall'IA.
    """
    return player_index in game_ai_agents.get(game_id, {})


def _remove_websocket_if_current(game_id: str, player_index: int, websocket: WebSocket) -> None:
    """
    Rimuove una connessione WS solo se è ancora quella registrata per il player.

    Questo evita che una connessione vecchia, chiudendosi dopo un reconnect, cancelli
    la connessione nuova appena registrata per lo stesso `player_index`.
    """
    if connected_clients.get(game_id, {}).get(player_index) is websocket:
        del connected_clients[game_id][player_index]


@app.get("/ai/agents", response_model=Dict)
async def list_ai_agents():
    """Elenca gli agenti IA disponibili (metadati per UI)."""
    return {
        "common_note_it": AI_AGENTS_COMMON_NOTE_IT,
        "agents": [
            {"name": spec.name, "label": spec.label, "description_it": spec.description_it}
            for spec in list_agent_specs()
        ],
    }


@app.get("/ai/models", response_model=Dict)
async def list_ai_models():
    """
    Elenca i modelli `.npz` disponibili sul server (per l'agente `bc_model`).

    Nota sicurezza:
    la UI riceve solo `model_id` (path relativo dentro una directory whitelisted).
    Il backend risolverà poi l'id in un path reale con controlli anti-path-traversal.
    """
    models_dir = get_models_dir_from_env()
    models = list_local_models(models_dir, recursive=False)
    return {
        "models": [
            {
                "id": m.id,
                "filename": m.filename,
                "label": m.label,
                "description_it": m.description_it,
                "metadata": m.metadata,
                "last_modified_utc": m.last_modified_utc,
                "is_compatible": m.is_compatible,
                "compatibility_reason_it": m.compatibility_reason_it,
            }
            for m in models
        ]
    }


@app.get("/")
async def root():
    """Health-check minimale."""
    return {"message": "Benvenuto nelle API di Briscola AI"}


@app.get("/meta", response_model=Dict)
async def meta() -> Dict:
    """
    Metadati “di runtime” per UI/deploy.

    Scopi:
    - mostrare/abilitare UX legata alla raccolta dati (consenso) quando `event_log_mode=dataset`.
    - debug rapido (versioni).
    """
    mode = _get_event_log_mode()
    return {
        "code_version": get_code_version(),
        "rules_version": get_rules_version(),
        "event_log_mode": mode,
        "dataset_requires_consent": mode == "dataset",
        "cors_allow_origins": cors_allow_origins,
    }


@app.post("/games", response_model=Dict)
async def create_game(config: GameConfig):
    """Crea una nuova partita di Briscola"""
    try:
        if _get_event_log_mode() == "dataset" and config.consent_to_data_collection is not True:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Consenso mancante: per raccogliere dati umani (event_log_mode=dataset) "
                    "devi accettare la registrazione anonima delle mosse."
                ),
            )

        # Seed per rendere riproducibile lo shuffle in fase di debugging/dataset.
        # In produzione useremmo RNG più robusto o un seed esplicito del client.
        seed = random.randrange(0, 2**32)
        state = new_game_state(config.num_players, config.player_names, seed=seed)

        # Genera un ID univoco per la partita
        game_id = str(uuid.uuid4())
        active_games[game_id] = state
        game_timestamps[game_id] = datetime.now()
        ai_agent_name = config.ai_agent or _DEFAULT_AI_AGENT_NAME

        # Config IA (solo 2-player, come la UI attuale).
        if config.num_players == 2:
            if ai_agent_name == "bc_model":
                models_dir = get_models_dir_from_env()
                model_path = resolve_model_path(models_dir, config.ai_model_id or "")
                validate_model_compatible_for_ui(model_path)
                game_ai_agents[game_id] = {1: build_agent(ai_agent_name, model_path=model_path)}
            else:
                game_ai_agents[game_id] = {1: build_agent(ai_agent_name)}
            game_action_rngs[game_id] = random.Random(seed ^ 0x9E3779B9)

        game_data[game_id] = [
            {
                "timestamp": datetime.now().isoformat(),
                "event": "game_created",
                "seed": seed,
                "ai_agent": ai_agent_name if config.num_players == 2 else None,
                "ai_model_id": config.ai_model_id
                if (config.num_players == 2 and ai_agent_name == "bc_model")
                else None,
            }
        ]
        game_locks[game_id] = asyncio.Lock()
        game_versions[game_id] = 0

        # Inizializza il dizionario delle connessioni WebSocket per questa partita
        connected_clients[game_id] = {}

        # Event log (opzionale): metadati partita.
        _safe_log_event(
            game_id,
            "game_created",
            {
                "seed": seed,
                "code_version": get_code_version(),
                "rules_version": get_rules_version(),
                "num_players": config.num_players,
                "is_team_game": state.is_team_game,
                "ai_agent": ai_agent_name if config.num_players == 2 else None,
                "ai_model_id": config.ai_model_id
                if (config.num_players == 2 and ai_agent_name == "bc_model")
                else None,
                "client_id": config.client_id,
                "consent_to_data_collection": bool(config.consent_to_data_collection is True),
            },
            server_version=0,
        )
        _safe_set_client_id(game_id, config.client_id)

        return {
            "game_id": game_id,
            "status": "created",
            "num_players": config.num_players,
            "is_team_game": state.is_team_game,
            "player_names": [p.name for p in state.players],
            "ai_agent": ai_agent_name if config.num_players == 2 else None,
            "ai_model_id": config.ai_model_id if (config.num_players == 2 and ai_agent_name == "bc_model") else None,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Modello .npz non trovato (ai_model_id)")


@app.get("/games/{game_id}", response_model=Dict)
async def get_game_state(game_id: str, player_index: Optional[int] = None):
    """Ottiene lo stato corrente di una partita"""
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Partita non trovata")

    game = active_games[game_id]

    # Aggiorna il timestamp per mantenere la partita attiva
    game_timestamps[game_id] = datetime.now()

    if player_index is not None:
        # Restituisce una vista specifica per il giocatore (stesso formato dei messaggi WS)
        try:
            observation_dto = build_observation_dto(game, player_index, game_versions.get(game_id, 0))
            return observation_dto.model_dump()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        # Restituisce lo stato completo (per spettatori o debugging) come DTO Pydantic.
        game_state_dto = build_game_state_dto(game, game_versions.get(game_id, 0))
        return game_state_dto.model_dump()


@app.post("/games/{game_id}/actions", response_model=PlayActionResultDTO, response_model_exclude_none=True)
async def play_action(game_id: str, action: GameAction) -> PlayActionResultDTO:
    """Gioca una carta nella partita"""
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Partita non trovata")

    should_schedule_ai = False

    async with game_locks[game_id]:
        state = active_games[game_id]
        server_version_before = game_versions.get(game_id, 0)

        # Verifica che sia il turno del giocatore
        if state.current_turn != action.player_index:
            raise HTTPException(status_code=400, detail="Non è il tuo turno")

        if _is_ai_controlled_player(game_id, action.player_index):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Il giocatore {action.player_index} è controllato dall'IA: "
                    "le sue mosse vengono eseguite automaticamente dal server."
                ),
            )

        observation_before: dict | None = None
        if _get_event_log_mode() == "dataset":
            # Per dataset umano usiamo sempre ObservationDTO (vista parziale anti-cheat).
            try:
                observation_before = build_observation_dto(
                    state, action.player_index, server_version_before
                ).model_dump()
            except Exception:
                observation_before = None

        # Esegue l'azione
        new_state, step_result = step(
            state,
            PlayCardAction(player_index=action.player_index, card_index=action.card_index),
        )
        if step_result.error:
            # Usiamo un errore HTTP standard invece di un payload con `error`.
            # Questo rende l'API più prevedibile per i client e coerente con gli altri endpoint.
            raise HTTPException(status_code=400, detail=step_result.error)

        active_games[game_id] = new_state
        game_versions[game_id] = game_versions.get(game_id, 0) + 1
        server_version = game_versions.get(game_id, 0)

        if step_result.played_card is None or step_result.player is None:
            # Invariante: su successo, `step()` deve restituire sempre played_card e player.
            raise HTTPException(status_code=500, detail="Risposta dominio incompleta (played_card/player)")

        trick_cards_dto: list[TableCardDTO] | None = None
        captured_cards_dto: list[CardDTO] = []
        if step_result.trick_completed:
            trick_cards_dto = [TableCardDTO.from_domain(card, idx) for card, idx in step_result.trick_cards]
            captured_cards_dto = [CardDTO.from_domain(card) for card, _ in step_result.trick_cards]

        action_result_dto = PlayActionResultDTO(
            server_version=server_version,
            played_card=CardDTO.from_domain(step_result.played_card),
            player=step_result.player,
            trick_completed=step_result.trick_completed,
            trick_winner=step_result.trick_winner,
            trick_size=len(step_result.trick_cards),
            cards_dealt=step_result.cards_dealt,
            trick_cards=trick_cards_dto,
            captured_cards=captured_cards_dto,
        )

        # Aggiorna timestamp
        game_timestamps[game_id] = datetime.now()

        # Registra l'azione per il training ML
        game_data[game_id].append(
            {
                "timestamp": datetime.now().isoformat(),
                "player_index": action.player_index,
                "card_index": action.card_index,
                # Salviamo il DTO (JSON-friendly) invece di oggetti di dominio.
                "result": action_result_dto.model_dump(exclude_none=True),
            }
        )

        # Event log (opzionale): azione umana + risultato.
        if _get_event_log_mode() == "dataset":
            reward = 0
            if step_result.trick_completed:
                trick_points = sum(card.rank.points for card, _ in step_result.trick_cards)
                winner = step_result.trick_winner
                if isinstance(winner, int):
                    reward = trick_points if winner == action.player_index else -trick_points

            next_observation: dict | None = None
            try:
                next_observation = build_observation_dto(new_state, action.player_index, server_version).model_dump()
            except Exception:
                next_observation = None

            _safe_log_event(
                game_id,
                "human_action",
                {
                    "player_index": action.player_index,
                    "card_index": action.card_index,
                    "observation": observation_before,
                    "reward": reward,
                    "done": bool(new_state.game_over is True),
                    "next_observation": next_observation,
                    "client_observed_server_version": action.client_observed_server_version,
                    "client_decision_time_ms": action.client_decision_time_ms,
                },
                server_version=server_version,
                player_index=action.player_index,
            )
        else:
            _safe_log_event(
                game_id,
                "action_play_card",
                {
                    "is_ai": False,
                    "player_index": action.player_index,
                    "card_index": action.card_index,
                    "result": action_result_dto.model_dump(exclude_none=True),
                },
                server_version=server_version,
                player_index=action.player_index,
            )

        # Se la mano è stata completata dall'umano, invia notifica con carte e vincitore.
        #
        # Nota architetturale:
        # evitiamo `asyncio.sleep()` nel backend per ritardi "di presentazione". Il tempo
        # di visualizzazione del risultato è gestito dal frontend (che può “trattenere”
        # lo snapshot successivo finché l'utente ha letto il risultato della mano).
        if step_result.trick_completed:
            trick_cards = list(step_result.trick_cards)
            winner_index = step_result.trick_winner if step_result.trick_winner is not None else 0
            points = sum(card.rank.points for card, _ in step_result.trick_cards)
            await notify_trick_result(game_id, trick_cards, winner_index, points)
            # Subito dopo inviamo anche lo stato aggiornato (tavolo vuoto, nuove carte pescate).
            # Il frontend decide se applicarlo subito o dopo un delay.
            if game_id in connected_clients:
                await notify_clients(game_id)
        else:
            # Mano non completata: notifica normale
            if game_id in connected_clients:
                await notify_clients(game_id)

        # Calcoliamo qui se dobbiamo far giocare l'IA (fuori dal lock scheduliamo solo il task).
        if not new_state.game_over and new_state.num_players == 2 and new_state.current_turn != action.player_index:
            should_schedule_ai = True
        elif new_state.game_over:
            _maybe_log_game_finished(game_id, state=new_state)

    # Modello "standard": se dopo la mossa umana tocca all'IA, il backend gioca automaticamente.
    # Nota UX: non inseriamo `asyncio.sleep()` per animazioni; il frontend gestisce i timing
    # trattenendo gli update (reveal/risultato mano) quando li riceve.
    #
    # Nota architetturale (task IA fuori lock):
    # Schediliamo il task IA *dopo* aver rilasciato il lock per evitare deadlock e permettere
    # alla risposta HTTP di tornare subito al client. Il check `game_after.game_over` qui è
    # solo un'ottimizzazione: la vera guardia è dentro `_maybe_ai_turn`, che riacquisisce il
    # lock e verifica nuovamente lo stato prima di giocare. Questo pattern è safe perché:
    # 1. Il task può trovare la partita già terminata/rimossa → ritorna subito.
    # 2. Eventuali azioni concorrenti (es. reconnect) sono serializzate dal lock interno.
    if should_schedule_ai:
        asyncio.create_task(_maybe_ai_turn(game_id=game_id, human_player_index=action.player_index))

    return action_result_dto


async def _maybe_ai_turn(game_id: str, human_player_index: int) -> None:
    """
    Esegue automaticamente le mosse dell'IA quando è il suo turno (2-player).

    Nota architetturale:
    - modello standard: il backend avanza la partita senza richiedere un trigger dal client.
    - il frontend controlla solo la *presentazione* (hold/animazioni) senza influenzare il dominio.
    """
    if game_id not in active_games or game_id not in game_locks:
        return

    # In 2-player ci aspettiamo al massimo una mossa IA per volta, ma gestiamo anche
    # eventuali casi futuri dove l'IA potrebbe avere turni consecutivi (safety loop).
    safety = 10
    while safety > 0:
        safety -= 1
        async with game_locks[game_id]:
            if game_id not in active_games:
                return
            state = active_games[game_id]
            if state.game_over:
                return
            if state.num_players != 2:
                return
            if state.current_turn == human_player_index:
                return

            await _execute_ai_turn_locked(game_id, human_player_index)


async def _execute_ai_turn_locked(game_id: str, human_player_index: int) -> None:
    """
    Esegue UNA singola mossa IA.

    Precondizione:
    - il chiamante ha acquisito `game_locks[game_id]`.
    """
    if game_id not in active_games:
        return

    state = active_games[game_id]

    # Se la partita è finita o tocca al giocatore umano, non fare nulla
    if state.game_over or state.current_turn == human_player_index:
        return

    # AI gioca una carta usando l'agente configurato.
    #
    # Nota anti-cheat:
    # la policy riceve `PlayerObservation` (vista parziale lecita), non `GameState` completo.
    ai_player_index = state.current_turn
    valid_actions = list(range(len(state.players[ai_player_index].hand))) if not state.game_over else []

    if not valid_actions:
        return

    rng = game_action_rngs.get(game_id, random.Random())
    agent = game_ai_agents.get(game_id, {}).get(ai_player_index)

    if agent is None:
        card_index = rng.randrange(len(valid_actions))
    else:
        observation = make_player_observation(state, ai_player_index)
        card_index = agent.choose_card_index(observation, rng=rng)
        if card_index not in valid_actions:
            # Fallback di sicurezza: se un agente ritorna un indice invalido, non blocchiamo la partita.
            card_index = rng.randrange(len(valid_actions))

    selected_card = state.players[ai_player_index].hand[card_index]

    # Invia messaggio per rivelare la carta nella mano IA (usando DTO)
    if game_id in connected_clients:
        reveal_dto = AiCardRevealDTO(
            card_index=card_index,
            card=CardDTO.from_domain(selected_card),
        )
        _safe_log_event(
            game_id,
            "ai_card_reveal",
            reveal_dto.model_dump(),
            server_version=game_versions.get(game_id, 0),
            player_index=ai_player_index,
        )
        reveal_json = reveal_dto.model_dump_json()
        for player_idx, client in list(connected_clients[game_id].items()):
            try:
                await client.send_text(reveal_json)
            except Exception:
                _remove_websocket_if_current(game_id, player_idx, client)

    new_state, step_result = step(state, PlayCardAction(player_index=ai_player_index, card_index=card_index))
    if step_result.error:
        return

    active_games[game_id] = new_state
    game_versions[game_id] = game_versions.get(game_id, 0) + 1
    server_version = game_versions.get(game_id, 0)

    # Event log + game_data: usiamo un DTO JSON-friendly anche per le mosse IA.
    trick_cards_dto: list[TableCardDTO] | None = None
    captured_cards_dto: list[CardDTO] = []
    if step_result.trick_completed:
        trick_cards_dto = [TableCardDTO.from_domain(card, idx) for card, idx in step_result.trick_cards]
        captured_cards_dto = [CardDTO.from_domain(card) for card, _ in step_result.trick_cards]

    if step_result.played_card is None or step_result.player is None:
        return

    ai_action_result_dto = PlayActionResultDTO(
        server_version=server_version,
        played_card=CardDTO.from_domain(step_result.played_card),
        player=step_result.player,
        trick_completed=step_result.trick_completed,
        trick_winner=step_result.trick_winner,
        trick_size=len(step_result.trick_cards),
        cards_dealt=step_result.cards_dealt,
        trick_cards=trick_cards_dto,
        captured_cards=captured_cards_dto,
    )

    # Se la mano è stata completata, invia notifica speciale
    if step_result.trick_completed:
        trick_cards = list(step_result.trick_cards)
        winner_index = step_result.trick_winner if step_result.trick_winner is not None else 0
        points = sum(card.rank.points for card, _ in step_result.trick_cards)
        await notify_trick_result(game_id, trick_cards, winner_index, points)
        if game_id in connected_clients:
            await notify_clients(game_id)
    else:
        if game_id in connected_clients:
            await notify_clients(game_id)

    # Registra l'azione AI
    game_timestamps[game_id] = datetime.now()
    game_data[game_id].append(
        {
            "timestamp": datetime.now().isoformat(),
            "player_index": ai_player_index,
            "card_index": card_index,
            "result": ai_action_result_dto.model_dump(exclude_none=True),
            "is_ai": True,
        }
    )

    _safe_log_event(
        game_id,
        "action_play_card",
        {
            "is_ai": True,
            "player_index": ai_player_index,
            "card_index": card_index,
            "result": ai_action_result_dto.model_dump(exclude_none=True),
        },
        server_version=server_version,
        player_index=ai_player_index,
    )
    if new_state.game_over:
        _maybe_log_game_finished(game_id, state=new_state)
    return


@app.get("/games/{game_id}/result", response_model=GameResultDTO, response_model_exclude_none=True)
async def get_game_result(game_id: str) -> GameResultDTO:
    """Ottiene il risultato finale di una partita"""
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Partita non trovata")

    state = active_games[game_id]
    server_version = game_versions.get(game_id, 0)
    if not state.game_over:
        return GameResultDTO(
            server_version=server_version,
            game_in_progress=True,
            game_over=False,
            is_team_game=state.is_team_game,
            points={},
        )

    points_by_player = {p.name: p.points for p in state.players}
    if state.is_team_game and state.teams is not None:
        team_0_points = sum(state.players[i].points for i in state.teams[0])
        team_1_points = sum(state.players[i].points for i in state.teams[1])

        if team_0_points > team_1_points:
            winning_team = 0
        elif team_1_points > team_0_points:
            winning_team = 1
        else:
            winning_team = None

        if winning_team is None:
            winner_str = "Pareggio"
        else:
            team_players = state.teams[winning_team]
            p0_name = state.players[team_players[0]].name
            p1_name = state.players[team_players[1]].name
            winner_str = f"Squadra {winning_team} ({p0_name} e {p1_name})"

        return GameResultDTO(
            server_version=server_version,
            game_in_progress=False,
            game_over=True,
            is_team_game=True,
            winner=winner_str,
            winning_team=winning_team,
            team_points={"Team 0": team_0_points, "Team 1": team_1_points},
            points=points_by_player,
            point_difference=abs(team_0_points - team_1_points),
        )

    p0 = state.players[0].points
    p1 = state.players[1].points
    if p0 > p1:
        winner_index = 0
    elif p1 > p0:
        winner_index = 1
    else:
        winner_index = None

    return GameResultDTO(
        server_version=server_version,
        game_in_progress=False,
        game_over=True,
        is_team_game=False,
        winner=state.players[winner_index].name if winner_index is not None else "Pareggio",
        winner_index=winner_index,
        points=points_by_player,
        point_difference=abs(p0 - p1),
    )


@app.websocket("/ws/{game_id}/{player_index}")
async def websocket_endpoint(websocket: WebSocket, game_id: str, player_index: int):
    """Endpoint WebSocket per aggiornamenti della partita in tempo reale"""
    if game_id not in active_games:
        await websocket.close(code=1000, reason="Partita non trovata")
        return

    state = active_games[game_id]

    if player_index < 0 or player_index >= state.num_players:
        await websocket.close(code=1000, reason="Indice giocatore non valido")
        return

    await websocket.accept()

    # Salva la connessione
    if game_id not in connected_clients:
        connected_clients[game_id] = {}
    previous = connected_clients[game_id].get(player_index)
    connected_clients[game_id][player_index] = websocket
    if previous is not None and previous is not websocket:
        try:
            await previous.close(code=1000, reason="Sostituita da una nuova connessione")
        except Exception:
            pass

    try:
        # Invia lo stato iniziale della partita (usando DTO)
        dto = build_observation_dto(state, player_index, game_versions.get(game_id, 0))
        _safe_log_event(
            game_id,
            "ws_connected",
            {"player_index": player_index},
            server_version=game_versions.get(game_id, 0),
            player_index=player_index,
        )
        _safe_log_event(
            game_id,
            "observation_sent",
            dto.model_dump(),
            server_version=game_versions.get(game_id, 0),
            player_index=player_index,
        )
        await websocket.send_text(dto.model_dump_json())

        # Mantiene la connessione aperta e gestisce i messaggi
        while True:
            # Attende messaggi (le azioni verranno inviate via HTTP)
            data = await websocket.receive_text()

            # Elabora eventuali comandi inviati via WebSocket
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        # Rimuove la connessione quando il client si disconnette
        _remove_websocket_if_current(game_id, player_index, websocket)
        _safe_log_event(
            game_id,
            "ws_disconnected",
            {"player_index": player_index},
            server_version=game_versions.get(game_id, 0),
            player_index=player_index,
        )


async def notify_clients(game_id: str):
    """Notifica tutti i client connessi sugli aggiornamenti della partita"""
    if game_id not in connected_clients:
        return

    state = active_games[game_id]

    # Invia lo stato aggiornato a ogni client connesso (usando DTO)
    server_version = game_versions.get(game_id, 0)
    for player_idx, websocket in list(connected_clients[game_id].items()):
        try:
            dto = build_observation_dto(state, player_idx, server_version)
            _safe_log_event(
                game_id,
                "observation_sent",
                dto.model_dump(),
                server_version=server_version,
                player_index=player_idx,
            )
            await websocket.send_text(dto.model_dump_json())
        except Exception:
            # Gestisce client disconnessi
            _remove_websocket_if_current(game_id, player_idx, websocket)


async def notify_trick_result(game_id: str, trick_cards: list, winner_index: int, points: int):
    """
    Notifica i client del risultato della mano con le carte visibili.

    Questo messaggio speciale permette al frontend di mostrare entrambe le carte
    e indicare chiaramente chi ha vinto la mano.
    """
    if game_id not in connected_clients:
        return

    state = active_games[game_id]
    if winner_index < len(state.players):
        winner_name = state.players[winner_index].name
    else:
        winner_name = f"Giocatore {winner_index + 1}"

    # Costruisci DTO per il risultato della mano
    trick_cards_dto = [TableCardDTO.from_domain(card, idx) for card, idx in trick_cards]
    trick_result_dto = TrickResultDTO(
        trick_cards=trick_cards_dto,
        winner_index=winner_index,
        winner_name=winner_name,
        points=points,
        server_version=game_versions.get(game_id, 0),
    )
    _safe_log_event(
        game_id,
        "trick_result",
        trick_result_dto.model_dump(),
        server_version=game_versions.get(game_id, 0),
    )
    trick_result_json = trick_result_dto.model_dump_json()

    for player_idx, websocket in list(connected_clients[game_id].items()):
        try:
            await websocket.send_text(trick_result_json)
        except Exception:
            _remove_websocket_if_current(game_id, player_idx, websocket)


async def cleanup_inactive_games():
    """Rimuove le partite inattive da più di 1 ora"""
    while True:
        await asyncio.sleep(3600)  # Check every hour
        now = datetime.now()

        # Trova le partite da rimuovere
        games_to_remove = []
        for game_id, timestamp in game_timestamps.items():
            if (now - timestamp).total_seconds() > 3600:  # 1 hour
                games_to_remove.append(game_id)

        # Rimuove le partite inattive
        for game_id in games_to_remove:
            # Event log: partita rimossa per inattività (non completa).
            # Logghiamo prima di eliminare lo stato in memoria, così possiamo salvare anche `server_version`.
            log = _get_event_log()
            if log is not None and _get_event_log_mode() != "off":
                try:
                    state = active_games.get(game_id)
                    if state is not None:
                        seed = getattr(state, "seed", None)
                        log.ensure_game(
                            game_id,
                            num_players=state.num_players,
                            seed=seed if isinstance(seed, int) else None,
                            code_version=get_code_version(),
                            rules_version=get_rules_version(),
                        )
                    updated = log.try_mark_game_aborted(game_id, aborted_reason="inactive_timeout")
                except Exception:
                    updated = False
                if updated:
                    _safe_log_event(
                        game_id,
                        "game_aborted",
                        {"reason": "inactive_timeout"},
                        server_version=game_versions.get(game_id, 0),
                    )

            if game_id in active_games:
                del active_games[game_id]
            if game_id in game_timestamps:
                del game_timestamps[game_id]
            if game_id in game_versions:
                del game_versions[game_id]
            if game_id in game_locks:
                del game_locks[game_id]
            if game_id in game_ai_agents:
                del game_ai_agents[game_id]
            if game_id in game_action_rngs:
                del game_action_rngs[game_id]
            if game_id in connected_clients:
                # Chiude tutte le connessioni WebSocket
                for websocket in connected_clients[game_id].values():
                    try:
                        await websocket.close(code=1000, reason="Partita scaduta")
                    except Exception:
                        pass
                del connected_clients[game_id]

            # Salva i dati della partita prima di rimuoverla
            if game_id in game_data:
                # In un'app reale, salva su database
                # Per ora, logga soltanto che li salveremmo
                print(f"Salverei i dati della partita {game_id} ({len(game_data[game_id])} azioni)")
                del game_data[game_id]
