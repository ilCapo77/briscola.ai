"""
API backend (FastAPI) per Briscola AI.

Questo modulo espone:
- endpoint HTTP per creare una partita, ottenere lo stato e giocare una carta
- endpoint WebSocket per inviare aggiornamenti in tempo reale ai client

Scelte implementative:
- Lo stato delle partite vive in un `GameSessionStore` (vedi `game_store.py`): in-memory in dev,
  Redis in cloud (multi-replica). Questo evita "partita non trovata" quando azioni/WS finiscono su
  repliche diverse. `game_data`/`game_timestamps` restano per-replica (best-effort: buffer ML, cleanup).
- Gli eventi realtime (reveal carta IA, risultato mano, refresh snapshot) viaggiano TUTTI sul
  pub/sub dello store (`publish`/`subscribe`): così raggiungono i client su QUALSIASI replica
  (Redis in prod, fan-out asyncio in dev). Ogni connessione WebSocket avvia un task subscriber
  che inoltra gli eventi al proprio socket; i "refresh" vengono tradotti nell'osservazione
  per-giocatore (anti-cheat: mai lo stato completo).
- L'agente IA non è serializzato: la sessione salva la sua config (nome + model_id) e l'agente viene
  ricostruito per mossa (con cache modello).
"""

import asyncio
import json
import os
import random
import uuid
from contextlib import aclosing, asynccontextmanager, suppress
from datetime import datetime
from typing import Any, Dict, List, Optional

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
from ..domain.serialization import game_state_from_dict, game_state_to_dict
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
from .event_log import EventLogProtocol, build_event_log, parse_event_db_path, resolve_database_url
from .game_store import (
    AiSeatConfig,
    GameSession,
    InMemoryGameSessionStore,
    build_game_session_store,
)
from .observation_builder import build_game_state_dto, build_observation_dto

# `InMemoryGameSessionStore` è ri-esportato qui per comodità dei test, che resettano lo store
# con `server.game_store = InMemoryGameSessionStore()`.
__all__ = ["app", "game_store", "InMemoryGameSessionStore"]


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


def _get_event_log() -> Optional[EventLogProtocol]:
    """
    Helper per accedere al logger dalla app FastAPI.

    Per semplicità, il riferimento vive in `app.state.event_log` e viene inizializzato
    nel lifespan. Se la feature non è configurata, ritorniamo `None`.
    """
    return getattr(app.state, "event_log", None)


def _get_event_log_mode() -> str:
    """
    Modalità di logging eventi.

    - `debug` (default): log completo, utile per debug (azioni, reveal/trick IA e lifecycle WS).
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
    state: Optional[DomainGameState] = None,
) -> None:
    """
    Wrapper “best-effort” per loggare eventi.

    Il logging è un optional feature: se il DB non è configurato o se una scrittura fallisce
    non vogliamo interrompere la partita.

    `state` (opzionale) consente al chiamante, che ha già la sessione in scope, di fornire lo
    stato di dominio per popolare la tabella `games` (num_players/seed) senza una lettura extra
    dallo store.
    """
    log = _get_event_log()
    if log is None:
        return

    mode = _get_event_log_mode()
    if mode == "off":
        return
    if mode == "dataset":
        # In modalità dataset riduciamo il DB tenendo solo gli eventi utili al dataset umano
        # (escludiamo lifecycle WS e gli eventi di gioco non necessari).
        allowed = {"game_created", "human_action", "game_finished", "game_aborted"}
        if event_type not in allowed:
            return

    try:
        # Garantiamo che la partita esista nella tabella `games` (idempotente).
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


def _maybe_log_game_finished(game_id: str, *, state: DomainGameState, server_version: int) -> None:
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
        server_version=server_version,
        state=state,
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
    # Backend event log: Postgres se `DATABASE_URL` è impostata (cloud multi-replica), altrimenti
    # SQLite locale se è dato un path, altrimenti disabilitato. `desired` = identità del backend
    # voluto (per ricreare la connessione se la config cambia tra due startup, tipico nei test).
    database_url = resolve_database_url()
    sqlite_path = parse_event_db_path(os.getenv("BRISCOLA_EVENT_DB_PATH"))
    desired = database_url or sqlite_path

    existing_event_log = getattr(app.state, "event_log", None)
    event_log: Optional[EventLogProtocol] = existing_event_log

    if event_log is not None and (desired is None or event_log.path != desired):
        with suppress(Exception):
            event_log.close()
        event_log = None
        app.state.event_log = None

    if event_log is None and desired is not None:
        try:
            event_log = build_event_log(sqlite_path=sqlite_path, database_url=database_url)
            event_log_created_here = event_log is not None
        except Exception as exc:
            # Il logger è un "optional feature": se fallisce non vogliamo bloccare il server.
            print(f"Event log: inizializzazione fallita, feature disabilitata ({exc!r}).")
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

# Store condiviso delle sessioni partita (stato + versioning + config IA + seed azioni).
#
# In cloud (Redis configurato) lo store è accessibile da tutte le repliche: azioni e WebSocket
# che finiscono su repliche diverse non danno più "partita non trovata". In dev/test è in-memory.
game_store = build_game_session_store()

# Stato best-effort, per-replica (non critico per la correttezza cross-replica):
# - `game_timestamps`: per il cleanup periodico delle partite inattive su questa replica.
# - `game_data`: buffer in memoria delle azioni (base per pipeline ML futura).
#
# Le connessioni WebSocket NON sono più tracciate qui: ogni connessione si iscrive al pub/sub
# dello store (`game_store.subscribe`) e inoltra gli eventi al proprio socket. Questo rende la
# consegna funzionante anche cross-replica (Redis in prod).
game_timestamps: Dict[str, datetime] = {}
game_data: Dict[str, List[Dict]] = {}  # Memorizza le azioni per il training ML

_DEFAULT_AI_AGENT_NAME = "random"
_AI_PLAYER_DISPLAY_NAME = "Giocatore AI"


def _agent_for_seat(cfg: AiSeatConfig) -> Agent:
    """
    Ricostruisce l'agente IA da `AiSeatConfig` (l'oggetto Agent non è serializzato nello store).

    Nota: il caricamento dei modelli è cached, quindi ricostruire l'agente a ogni mossa è cheap.
    """
    if cfg.agent_name == "bc_model":
        path = resolve_model_path(get_models_dir_from_env(), cfg.model_id or "")
        return build_agent("bc_model", model_path=path)
    return build_agent(cfg.agent_name)


def _is_ai_controlled_player(session: GameSession, player_index: int) -> bool:
    """
    Ritorna True se `player_index` è controllato dal backend per questa partita.

    Nota di sicurezza:
    la UI attuale espone solo l'umano come player 0 e l'IA come player 1. L'endpoint HTTP
    resta però chiamabile manualmente: questa guardia evita che un client giochi le mosse
    del player controllato dall'IA.
    """
    return player_index in session.ai_seats


def _display_name_for_player(session: GameSession, player_index: int) -> str:
    """
    Nome pubblico breve per messaggi UI.

    I nomi dei player sono parte dello stato di dominio e possono contenere dettagli lunghi
    (es. label del modello selezionato in versioni vecchie della UI). Nei messaggi di partita
    mostriamo invece un'etichetta stabile e leggibile per i seat controllati dall'IA.
    """
    state = session.state
    if _is_ai_controlled_player(session, player_index):
        return _AI_PLAYER_DISPLAY_NAME
    if 0 <= player_index < len(state.players):
        return state.players[player_index].name
    return f"Giocatore {player_index + 1}"


def _metadata_for_model_catalog_ui(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Riduce i metadati del modello alla parte utile per il browser.

    I file `.npz` possono conservare serie lunghe di metriche di training. Quelle sono corrette
    come artefatto locale, ma non devono viaggiare nell'endpoint della UI a ogni caricamento:
    basta un conteggio sintetico e manteniamo invece i campi descrittivi/inferenziali.
    """
    out = dict(metadata)
    metrics = out.pop("metrics", None)
    if isinstance(metrics, list):
        out["metrics_count"] = len(metrics)
    return out


@app.get("/ai/agents", response_model=Dict)
async def list_ai_agents():
    """
    Elenca gli agenti IA disponibili (metadati per UI), con un flag `available`.

    `available` dice se l'agente è realmente giocabile nel deploy corrente:
    - agenti che richiedono un modello "bundled" (`spec.requires_model_id`, es. `best_a2c.npz`) sono
      disponibili solo se quel file è presente nella directory modelli;
    - `bc_model` è disponibile se esiste almeno un modello `.npz` compatibile nel catalogo;
    - gli altri (random/greedy/euristiche/hybrid_endgame) sono sempre disponibili.
    La UI usa il flag per disabilitare le opzioni rotte (evita "manca il modello") e per scegliere
    un default sensato.
    """
    models_dir = get_models_dir_from_env()
    has_compatible_model = any(m.is_compatible for m in list_local_models(models_dir, recursive=False))

    agents = []
    for spec in list_agent_specs():
        if spec.name == "bc_model":
            available = has_compatible_model
        elif spec.requires_model_id is not None:
            available = (models_dir / spec.requires_model_id).exists()
        else:
            available = True
        agents.append(
            {
                "name": spec.name,
                "label": spec.label,
                "description_it": spec.description_it,
                "requires_model_id": spec.requires_model_id,
                "available": available,
            }
        )

    return {"common_note_it": AI_AGENTS_COMMON_NOTE_IT, "agents": agents}


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
                "metadata": _metadata_for_model_catalog_ui(m.metadata),
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
        ai_agent_name = config.ai_agent or _DEFAULT_AI_AGENT_NAME

        # Config IA (solo 2-player, come la UI attuale).
        ai_seats: dict[int, AiSeatConfig] = {}
        if config.num_players == 2:
            if ai_agent_name == "bc_model":
                models_dir = get_models_dir_from_env()
                # Validiamo il path del modello PRIMA di salvare la sessione (come prima).
                model_path = resolve_model_path(models_dir, config.ai_model_id or "")
                validate_model_compatible_for_ui(model_path)
                ai_seats = {1: AiSeatConfig(agent_name=ai_agent_name, model_id=config.ai_model_id)}
            else:
                # Validiamo l'agente PRIMA di salvare la sessione (come prima): un nome non valido
                # o un alias non disponibile (es. best_a2c senza file) deve dare 400 alla creazione,
                # non un crash più tardi nel task IA. L'oggetto costruito viene scartato (la sessione
                # salva solo la config; l'agente è ricostruito per mossa, con cache).
                build_agent(ai_agent_name)
                ai_seats = {1: AiSeatConfig(agent_name=ai_agent_name, model_id=None)}

        now_iso = datetime.now().isoformat()
        session = GameSession(
            game_id=game_id,
            state=state,
            version=0,
            ai_seats=ai_seats,
            action_seed=seed ^ 0x9E3779B9,
            created_at=now_iso,
            updated_at=now_iso,
        )
        await game_store.set(session)

        game_timestamps[game_id] = datetime.now()
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
            state=state,
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
    session = await game_store.get(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Partita non trovata")

    game = session.state

    # Aggiorna il timestamp per mantenere la partita attiva
    game_timestamps[game_id] = datetime.now()

    if player_index is not None:
        # Restituisce una vista specifica per il giocatore (stesso formato dei messaggi WS)
        try:
            observation_dto = build_observation_dto(game, player_index, session.version)
            return observation_dto.model_dump()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        # Restituisce lo stato completo (per spettatori o debugging) come DTO Pydantic.
        game_state_dto = build_game_state_dto(game, session.version)
        return game_state_dto.model_dump()


@app.post("/games/{game_id}/actions", response_model=PlayActionResultDTO, response_model_exclude_none=True)
async def play_action(game_id: str, action: GameAction) -> PlayActionResultDTO:
    """Gioca una carta nella partita"""
    if await game_store.get(game_id) is None:
        raise HTTPException(status_code=404, detail="Partita non trovata")

    should_schedule_ai = False

    async with game_store.lock(game_id):
        session = await game_store.get(game_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Partita non trovata")
        state = session.state
        server_version_before = session.version

        # Verifica che sia il turno del giocatore
        if state.current_turn != action.player_index:
            raise HTTPException(status_code=400, detail="Non è il tuo turno")

        if _is_ai_controlled_player(session, action.player_index):
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

        session.state = new_state
        session.version += 1
        session.updated_at = datetime.now().isoformat()
        await game_store.set(session)
        server_version = session.version

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

        # Registra l'azione per il training ML.
        # setdefault: game_data e' per-replica; se l'azione arriva su una replica diversa da
        # quella che ha creato la partita, la lista non esiste ancora (stato vive nello store).
        game_data.setdefault(game_id, []).append(
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
                state=new_state,
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
                state=new_state,
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
            await notify_trick_result(session, trick_cards, winner_index, points)
            # Subito dopo inviamo anche lo stato aggiornato (tavolo vuoto, nuove carte pescate).
            # Il frontend decide se applicarlo subito o dopo un delay.
            await notify_clients(game_id, new_state, server_version)
        else:
            # Mano non completata: notifica normale
            await notify_clients(game_id, new_state, server_version)

        # Calcoliamo qui se dobbiamo far giocare l'IA (fuori dal lock scheduliamo solo il task).
        if not new_state.game_over and new_state.num_players == 2 and new_state.current_turn != action.player_index:
            should_schedule_ai = True
        elif new_state.game_over:
            _maybe_log_game_finished(game_id, state=new_state, server_version=server_version)

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
    # In 2-player ci aspettiamo al massimo una mossa IA per volta, ma gestiamo anche
    # eventuali casi futuri dove l'IA potrebbe avere turni consecutivi (safety loop).
    safety = 10
    while safety > 0:
        safety -= 1
        async with game_store.lock(game_id):
            session = await game_store.get(game_id)
            if session is None:
                return
            state = session.state
            if state.game_over:
                return
            if state.num_players != 2:
                return
            if state.current_turn == human_player_index:
                return

            await _execute_ai_turn_locked(session, human_player_index)


async def _execute_ai_turn_locked(session: GameSession, human_player_index: int) -> None:
    """
    Esegue UNA singola mossa IA.

    Precondizione:
    - il chiamante ha acquisito `game_store.lock(game_id)` e passa la sessione già caricata.
    """
    game_id = session.game_id
    state = session.state

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

    # RNG deterministico per mossa: dipende dal seed della partita e dalla versione corrente.
    rng = random.Random(session.action_seed ^ session.version)
    seat_cfg = session.ai_seats.get(ai_player_index)
    agent = _agent_for_seat(seat_cfg) if seat_cfg is not None else None

    if agent is None:
        card_index = rng.randrange(len(valid_actions))
    else:
        observation = make_player_observation(state, ai_player_index)
        card_index = agent.choose_card_index(observation, rng=rng)
        if card_index not in valid_actions:
            # Fallback di sicurezza: se un agente ritorna un indice invalido, non blocchiamo la partita.
            card_index = rng.randrange(len(valid_actions))

    selected_card = state.players[ai_player_index].hand[card_index]

    # Pubblica il messaggio per rivelare la carta nella mano IA (usando DTO).
    # Il fan-out verso i socket avviene tramite i task subscriber (pub/sub dello store).
    reveal_dto = AiCardRevealDTO(
        card_index=card_index,
        card=CardDTO.from_domain(selected_card),
    )
    _safe_log_event(
        game_id,
        "ai_card_reveal",
        reveal_dto.model_dump(),
        server_version=session.version,
        player_index=ai_player_index,
        state=state,
    )
    await game_store.publish(game_id, reveal_dto.model_dump_json())

    new_state, step_result = step(state, PlayCardAction(player_index=ai_player_index, card_index=card_index))
    if step_result.error:
        return

    session.state = new_state
    session.version += 1
    session.updated_at = datetime.now().isoformat()
    await game_store.set(session)
    server_version = session.version

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
        await notify_trick_result(session, trick_cards, winner_index, points)
        await notify_clients(game_id, new_state, server_version)
    else:
        await notify_clients(game_id, new_state, server_version)

    # Registra l'azione AI (setdefault: game_data e' per-replica, vedi nota in play_action).
    game_timestamps[game_id] = datetime.now()
    game_data.setdefault(game_id, []).append(
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
        state=new_state,
    )
    if new_state.game_over:
        _maybe_log_game_finished(game_id, state=new_state, server_version=server_version)
    return


@app.get("/games/{game_id}/result", response_model=GameResultDTO, response_model_exclude_none=True)
async def get_game_result(game_id: str) -> GameResultDTO:
    """Ottiene il risultato finale di una partita"""
    session = await game_store.get(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Partita non trovata")

    state = session.state
    server_version = session.version
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


async def _ws_subscriber(websocket: WebSocket, game_id: str, player_index: int) -> None:
    """
    Task di consegna realtime per UNA connessione WebSocket.

    Si iscrive al pub/sub dello store per la partita e inoltra ogni evento al socket:
    - messaggi `reveal`/`trick_result`: inoltrati verbatim (il `type` è già nel JSON);
    - messaggi `refresh`: NON contengono un'osservazione (sarebbe per-giocatore); il subscriber
      rilegge lo stato dallo store e costruisce l'osservazione PER QUESTO `player_index`
      (anti-cheat: ogni client riceve solo la propria vista parziale).

    Robustezza: un singolo messaggio malformato non deve uccidere il loop; in caso di errore di
    `send` consideriamo la connessione persa e usciamo (il `finally` dell'endpoint farà cleanup).
    """
    # `aclosing` garantisce la chiusura deterministica del generator (unsubscribe/cleanup della
    # coda o del pubsub) anche quando usciamo con `break` o per cancellazione del task.
    async with aclosing(game_store.subscribe(game_id)) as events:
        async for raw in events:
            try:
                msg = json.loads(raw)
                if msg.get("type") == "refresh":
                    # Stato point-in-time incluso nel messaggio (vedi `notify_clients`): lo usiamo
                    # per costruire l'osservazione di QUESTO player, senza rileggere il "latest"
                    # (che potrebbe essere già avanzato dalla mossa IA → ordine eventi sbagliato).
                    state_dict = msg.get("state")
                    if state_dict is None:
                        continue
                    state = game_state_from_dict(state_dict)
                    obs = build_observation_dto(state, player_index, int(msg.get("server_version", 0)))
                    await websocket.send_text(obs.model_dump_json())
                else:
                    # reveal/trick: inoltro verbatim del JSON pubblicato.
                    await websocket.send_text(raw)
            except WebSocketDisconnect, RuntimeError:
                # Il socket è andato (disconnect o "websocket closed"): chiudiamo il loop.
                break
            except Exception:
                # Messaggio malformato o errore non fatale di parsing: ignora e prosegui.
                continue


@app.websocket("/ws/{game_id}/{player_index}")
async def websocket_endpoint(websocket: WebSocket, game_id: str, player_index: int):
    """Endpoint WebSocket per aggiornamenti della partita in tempo reale"""
    session = await game_store.get(game_id)
    if session is None:
        await websocket.close(code=1000, reason="Partita non trovata")
        return

    state = session.state

    if player_index < 0 or player_index >= state.num_players:
        await websocket.close(code=1000, reason="Indice giocatore non valido")
        return

    await websocket.accept()

    # Avvia il task subscriber che inoltra a questo socket gli eventi pubblicati sullo store.
    sub_task = asyncio.create_task(_ws_subscriber(websocket, game_id, player_index))

    try:
        # Invia lo stato iniziale della partita (usando DTO)
        dto = build_observation_dto(state, player_index, session.version)
        _safe_log_event(
            game_id,
            "ws_connected",
            {"player_index": player_index},
            server_version=session.version,
            player_index=player_index,
            state=state,
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
        # Best-effort: rileggiamo la sessione per loggare una server_version aggiornata (se esiste).
        current = await game_store.get(game_id)
        _safe_log_event(
            game_id,
            "ws_disconnected",
            {"player_index": player_index},
            server_version=current.version if current is not None else 0,
            player_index=player_index,
            state=current.state if current is not None else None,
        )
    finally:
        # Ferma il task subscriber: la connessione non esiste più.
        sub_task.cancel()
        try:
            await sub_task
        except asyncio.CancelledError:
            pass


async def notify_clients(game_id: str, state: DomainGameState, server_version: int) -> None:
    """
    Notifica i client connessi che lo snapshot della partita è cambiato.

    Pubblichiamo un messaggio "refresh" sul pub/sub dello store, includendo lo stato
    **point-in-time** (`game_state_to_dict(state)`): ogni task subscriber ne ricava l'osservazione
    per il proprio `player_index` (anti-cheat). È importante includere lo stato di QUESTO preciso
    momento e non rileggere il "latest" dallo store: altrimenti, se l'IA ha già mosso, il client
    riceverebbe lo stato post-IA prima del relativo `ai_card_reveal`, perdendo lo stato intermedio.
    """
    await game_store.publish(
        game_id,
        json.dumps({"type": "refresh", "server_version": server_version, "state": game_state_to_dict(state)}),
    )


async def notify_trick_result(session: GameSession, trick_cards: list, winner_index: int, points: int):
    """
    Notifica i client del risultato della mano con le carte visibili.

    Questo messaggio speciale permette al frontend di mostrare entrambe le carte
    e indicare chiaramente chi ha vinto la mano. È pubblicato sul pub/sub dello store, così
    raggiunge i client su qualsiasi replica.
    """
    game_id = session.game_id
    winner_name = _display_name_for_player(session, winner_index)

    # Costruisci DTO per il risultato della mano
    trick_cards_dto = [TableCardDTO.from_domain(card, idx) for card, idx in trick_cards]
    trick_result_dto = TrickResultDTO(
        trick_cards=trick_cards_dto,
        winner_index=winner_index,
        winner_name=winner_name,
        points=points,
        server_version=session.version,
    )
    _safe_log_event(
        game_id,
        "trick_result",
        trick_result_dto.model_dump(),
        server_version=session.version,
        state=session.state,
    )
    await game_store.publish(game_id, trick_result_dto.model_dump_json())


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
            session = await game_store.get(game_id)

            # Staleness AUTORITATIVA: decidere dal solo timestamp locale è sbagliato su store
            # condiviso (questa replica potrebbe aver creato la partita ma non servire più le
            # azioni, mentre un'altra replica la sta ancora giocando). Usiamo `session.updated_at`.
            truly_stale = session is None
            if session is not None:
                try:
                    updated = datetime.fromisoformat(session.updated_at)
                    truly_stale = (now - updated).total_seconds() > 3600
                except Exception:
                    truly_stale = False

            if not truly_stale:
                # Attiva altrove: NON toccare lo store condiviso; smetti solo di tracciarla
                # localmente (i buffer per-replica). Le connessioni WS locali restano valide.
                game_timestamps.pop(game_id, None)
                game_data.pop(game_id, None)
                continue

            # Event log: partita rimossa per inattività (non completa).
            # Logghiamo prima di eliminare lo stato, così possiamo salvare anche `server_version`.
            log = _get_event_log()
            if log is not None and _get_event_log_mode() != "off":
                try:
                    state = session.state if session is not None else None
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
                        server_version=session.version if session is not None else 0,
                        state=session.state if session is not None else None,
                    )

            await game_store.delete(game_id)
            if game_id in game_timestamps:
                del game_timestamps[game_id]

            # Salva i dati della partita prima di rimuoverla
            if game_id in game_data:
                # In un'app reale, salva su database
                # Per ora, logga soltanto che li salveremmo
                print(f"Salverei i dati della partita {game_id} ({len(game_data[game_id])} azioni)")
                del game_data[game_id]
