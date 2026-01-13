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
import random
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..game.game import BriscolaGame
from ..game.models import Card, Rank, Suit
from .dto import (
    AiCardRevealDTO,
    CardDTO,
    GameStateDTO,
    ObservationDTO,
    PlayerInfoDTO,
    PlayerStateDTO,
    TableCardDTO,
    TrickResultDTO,
)


# Encoder JSON personalizzato per oggetti Card, Suit e Rank
class GameJSONEncoder(json.JSONEncoder):
    """
    Encoder JSON per serializzare oggetti del dominio (`Card`, `Suit`, `Rank`).

    Nota: è una soluzione rapida. In una migrazione a Pydantic v2 preferiremo
    definire esplicitamente gli schemi (DTO) e una serializzazione controllata.
    """

    def default(self, obj):
        """
        Converte oggetti non serializzabili (Card/Suit/Rank) in dizionari/valori JSON.

        Argomenti:
            obj: oggetto Python da serializzare

        Ritorna:
            Una struttura composta solo da tipi JSON-safe (dict, list, str, int, ...).
        """
        if isinstance(obj, Card):
            return {"suit": obj.suit.value, "rank": obj.rank.name, "points": obj.rank.points, "number": obj.rank.number}
        elif isinstance(obj, Suit):
            return obj.value
        elif isinstance(obj, Rank):
            return {"name": obj.name, "number": obj.number, "points": obj.points}
        return super().default(obj)


def _json_safe(payload: object) -> object:
    """
    Converte un oggetto Python (che può contenere `Card`/`Suit`/`Rank`) in una struttura JSON-safe.

    Nota: usiamo questa funzione sugli endpoint HTTP per mantenere lo stesso formato delle carte
    che inviamo via WebSocket (dove usiamo già `GameJSONEncoder`).
    """
    return json.loads(json.dumps(payload, cls=GameJSONEncoder))


def _build_observation_dto(game: BriscolaGame, player_index: int, server_version: int) -> ObservationDTO:
    """
    Costruisce un ObservationDTO dal dominio.

    Questa funzione centralizza la conversione da stato di gioco a payload WS/HTTP,
    garantendo che il formato sia sempre coerente con il contratto DTO.
    Supporta sia modalità 2-player che 4-player.
    """
    obs = game.get_observation_for_player(player_index)

    # Converti carte in mano
    my_hand = [CardDTO.from_domain(card) for card in obs["my_hand"]]

    # Converti carta briscola
    trump_card = CardDTO.from_domain(obs["trump_card"]) if obs["trump_card"] else None
    trump_suit = obs["trump_card"].suit.value if obs["trump_card"] else None

    # Converti carte sul tavolo
    table_cards = [TableCardDTO.from_domain(card, idx) for card, idx in obs["table_cards"]]

    # Costruisci lista players (sostituisce player_{n}_* dinamici)
    players = []
    for i, player in enumerate(game.players):
        players.append(
            PlayerInfoDTO(
                index=i,
                name=player.name,
                points=player.points,
                hand_size=len(player.hand),
            )
        )

    # Campi 4-player (None se 2-player)
    my_team = obs.get("my_team")
    teammate_index = obs.get("teammate_index")
    teammate_points = obs.get("teammate_points")
    my_team_points = obs.get("my_team_points")
    opponent_team_points = obs.get("opponent_team_points")

    return ObservationDTO(
        server_version=server_version,
        my_index=player_index,
        my_hand=my_hand,
        my_points=obs["my_points"],
        my_turn=obs["my_turn"],
        trump_card=trump_card,
        trump_suit=trump_suit,
        table_cards=table_cards,
        cards_remaining_in_deck=obs["cards_remaining_in_deck"],
        valid_actions=obs["valid_actions"],
        game_over=obs["game_over"],
        num_players=obs["num_players"],
        is_team_game=obs["is_team_game"],
        players=players,
        my_team=my_team,
        teammate_index=teammate_index,
        teammate_points=teammate_points,
        my_team_points=my_team_points,
        opponent_team_points=opponent_team_points,
    )


def _build_game_state_dto(game: BriscolaGame, server_version: int) -> GameStateDTO:
    """
    Costruisce un GameStateDTO (stato completo) dal dominio.

    Uso previsto:
    - endpoint HTTP `GET /games/{id}` senza `player_index` (debug/spectator)

    Nota sicurezza/fair-play:
    Questo payload contiene tutte le mani e quindi NON deve essere usato da un client
    che rappresenta un singolo giocatore umano.
    """
    state = game.get_game_state()

    trump_card = CardDTO.from_domain(state["trump_card"]) if state["trump_card"] else None
    trump_suit = state["trump_card"].suit.value if state["trump_card"] else None
    table_cards = [TableCardDTO.from_domain(card, idx) for card, idx in state["table_cards"]]

    players: list[PlayerStateDTO] = []
    for i, player in enumerate(game.players):
        players.append(
            PlayerStateDTO(
                index=i,
                name=player.name,
                points=player.points,
                hand=[CardDTO.from_domain(card) for card in player.hand],
                hand_size=len(player.hand),
                captured_cards=[CardDTO.from_domain(card) for card in player.captured_cards],
            )
        )

    teams = state.get("teams")
    team_0_points = state.get("team_0_points")
    team_1_points = state.get("team_1_points")

    return GameStateDTO(
        server_version=server_version,
        num_players=state["num_players"],
        is_team_game=state["is_team_game"],
        trump_card=trump_card,
        trump_suit=trump_suit,
        table_cards=table_cards,
        current_turn=state["current_turn"],
        first_player=state["first_player"],
        cards_remaining_in_deck=state["cards_remaining_in_deck"],
        valid_actions=state["valid_actions"],
        game_over=state["game_over"],
        trick_in_progress=state["trick_in_progress"],
        trick_size=state["trick_size"],
        expected_trick_size=state["expected_trick_size"],
        players=players,
        teams=teams,
        team_0_points=team_0_points,
        team_1_points=team_1_points,
    )


# Modelli per richieste e risposte API
class GameConfig(BaseModel):
    """Payload per creare una partita."""

    num_players: int
    player_names: Optional[List[str]] = None


class GameAction(BaseModel):
    """Payload per giocare una carta."""

    game_id: str
    player_index: int
    card_index: int


class GameState(BaseModel):
    """Payload per richiedere lo stato di una partita (opzionale: vista per giocatore)."""

    game_id: str
    player_index: Optional[int] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestisce startup/shutdown dell'app FastAPI.

    Usiamo un task in background per fare periodicamente cleanup delle partite inattive.
    """
    cleanup_task = asyncio.create_task(cleanup_inactive_games())
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass


# Crea l'app FastAPI
app = FastAPI(title="Briscola AI API", version="0.1.0", lifespan=lifespan)

# Aggiunge middleware CORS per consentire richieste cross-origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In sviluppo, consente tutte le origini
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Conserva le partite attive in memoria (in un'app reale, usare un database)
active_games: Dict[str, BriscolaGame] = {}
game_timestamps: Dict[str, datetime] = {}
game_data: Dict[str, List[Dict]] = {}  # Memorizza le azioni per il training ML

# Sincronizzazione e versioning server-side (per evitare race/doppi trigger IA):
# - `game_locks`: garantisce che le mutazioni di stato di una partita siano serializzate.
# - `game_versions`: contatore monotono incrementato ad ogni `play_action` (umano o IA).
#   Lo includiamo negli snapshot/messaggi WS come metadato debug-friendly (ordering/reconnect).
game_locks: Dict[str, asyncio.Lock] = {}
game_versions: Dict[str, int] = {}

# Connessioni WebSocket
connected_clients: Dict[str, Dict[int, WebSocket]] = {}


@app.get("/")
async def root():
    """Health-check minimale."""
    return {"message": "Benvenuto nelle API di Briscola AI"}


@app.post("/games", response_model=Dict)
async def create_game(config: GameConfig):
    """Crea una nuova partita di Briscola"""
    try:
        game = BriscolaGame(config.num_players, config.player_names)
        game.start_game()

        # Genera un ID univoco per la partita
        game_id = str(uuid.uuid4())
        active_games[game_id] = game
        game_timestamps[game_id] = datetime.now()
        game_data[game_id] = []
        game_locks[game_id] = asyncio.Lock()
        game_versions[game_id] = 0

        # Inizializza il dizionario delle connessioni WebSocket per questa partita
        connected_clients[game_id] = {}

        return {
            "game_id": game_id,
            "status": "created",
            "num_players": config.num_players,
            "is_team_game": game.is_team_game,
            "player_names": [player.name for player in game.players],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


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
            dto = _build_observation_dto(game, player_index, game_versions.get(game_id, 0))
            return dto.model_dump()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        # Restituisce lo stato completo (per spettatori o debugging) come DTO Pydantic.
        dto = _build_game_state_dto(game, game_versions.get(game_id, 0))
        return dto.model_dump()


@app.post("/games/{game_id}/actions", response_model=Dict)
async def play_action(game_id: str, action: GameAction):
    """Gioca una carta nella partita"""
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Partita non trovata")

    async with game_locks[game_id]:
        game = active_games[game_id]

        # Verifica che sia il turno del giocatore
        if game.current_turn != action.player_index:
            raise HTTPException(status_code=400, detail="Non è il tuo turno")

        # Esegue l'azione
        result = game.play_action(action.card_index)
        game_versions[game_id] = game_versions.get(game_id, 0) + 1

        # Aggiorna timestamp
        game_timestamps[game_id] = datetime.now()

        # Registra l'azione per il training ML
        game_data[game_id].append(
            {
                "timestamp": datetime.now().isoformat(),
                "player_index": action.player_index,
                "card_index": action.card_index,
                "result": result,
            }
        )

        # Se la mano è stata completata dall'umano, invia notifica con carte e vincitore.
        #
        # Nota architetturale:
        # evitiamo `asyncio.sleep()` nel backend per ritardi "di presentazione". Il tempo
        # di visualizzazione del risultato è gestito dal frontend (che può “trattenere”
        # lo snapshot successivo finché l'utente ha letto il risultato della mano).
        if result.get("trick_completed"):
            trick_cards = result.get("trick_cards", [])
            winner_index = result.get("trick_winner", 0)
            points = sum(card.rank.points if hasattr(card, "rank") else 0 for card, _ in trick_cards)
            await notify_trick_result(game_id, trick_cards, winner_index, points)
            # Subito dopo inviamo anche lo stato aggiornato (tavolo vuoto, nuove carte pescate).
            # Il frontend decide se applicarlo subito o dopo un delay.
            if game_id in connected_clients:
                await notify_clients(game_id)
        else:
            # Mano non completata: notifica normale
            if game_id in connected_clients:
                await notify_clients(game_id)

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
    if game_id in active_games:
        game_after = active_games[game_id]
        if not game_after.game_over and game_after.num_players == 2 and game_after.current_turn != action.player_index:
            asyncio.create_task(_maybe_ai_turn(game_id=game_id, human_player_index=action.player_index))

    payload = dict(result)
    payload["server_version"] = game_versions.get(game_id, 0)
    return _json_safe(payload)


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
            game = active_games[game_id]
            if game.game_over:
                return
            if game.num_players != 2:
                return
            if game.current_turn == human_player_index:
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

    game = active_games[game_id]

    # Se la partita è finita o tocca al giocatore umano, non fare nulla
    if game.game_over or game.current_turn == human_player_index:
        return

    # AI gioca una carta casuale tra le valide
    ai_player_index = game.current_turn
    observation = game.get_observation_for_player(ai_player_index)
    valid_actions = observation.get("valid_actions", [])

    if not valid_actions:
        return

    card_index = random.choice(valid_actions)
    selected_card = game.players[ai_player_index].hand[card_index]

    # Invia messaggio per rivelare la carta nella mano IA (usando DTO)
    if game_id in connected_clients:
        reveal_dto = AiCardRevealDTO(
            card_index=card_index,
            card=CardDTO.from_domain(selected_card),
        )
        reveal_json = reveal_dto.model_dump_json()
        for client in connected_clients[game_id].values():
            try:
                await client.send_text(reveal_json)
            except Exception:
                pass

    result = game.play_action(card_index)
    game_versions[game_id] = game_versions.get(game_id, 0) + 1

    # Se la mano è stata completata, invia notifica speciale
    if result.get("trick_completed"):
        trick_cards = result.get("trick_cards", [])
        winner_index = result.get("trick_winner", 0)
        points = sum(card.rank.points if hasattr(card, "rank") else 0 for card, _ in trick_cards)
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
            "result": result,
            "is_ai": True,
        }
    )
    return


@app.get("/games/{game_id}/result", response_model=Dict)
async def get_game_result(game_id: str):
    """Ottiene il risultato finale di una partita"""
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Partita non trovata")

    game = active_games[game_id]
    return game.get_game_result()


@app.websocket("/ws/{game_id}/{player_index}")
async def websocket_endpoint(websocket: WebSocket, game_id: str, player_index: int):
    """Endpoint WebSocket per aggiornamenti della partita in tempo reale"""
    if game_id not in active_games:
        await websocket.close(code=1000, reason="Partita non trovata")
        return

    game = active_games[game_id]

    if player_index < 0 or player_index >= game.num_players:
        await websocket.close(code=1000, reason="Indice giocatore non valido")
        return

    await websocket.accept()

    # Salva la connessione
    if game_id not in connected_clients:
        connected_clients[game_id] = {}
    connected_clients[game_id][player_index] = websocket

    try:
        # Invia lo stato iniziale della partita (usando DTO)
        dto = _build_observation_dto(game, player_index, game_versions.get(game_id, 0))
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
        if game_id in connected_clients and player_index in connected_clients[game_id]:
            del connected_clients[game_id][player_index]


async def notify_clients(game_id: str):
    """Notifica tutti i client connessi sugli aggiornamenti della partita"""
    if game_id not in connected_clients:
        return

    game = active_games[game_id]

    # Invia lo stato aggiornato a ogni client connesso (usando DTO)
    server_version = game_versions.get(game_id, 0)
    for player_idx, websocket in connected_clients[game_id].items():
        try:
            dto = _build_observation_dto(game, player_idx, server_version)
            await websocket.send_text(dto.model_dump_json())
        except Exception:
            # Gestisce client disconnessi
            pass


async def notify_trick_result(game_id: str, trick_cards: list, winner_index: int, points: int):
    """
    Notifica i client del risultato della mano con le carte visibili.

    Questo messaggio speciale permette al frontend di mostrare entrambe le carte
    e indicare chiaramente chi ha vinto la mano.
    """
    if game_id not in connected_clients:
        return

    game = active_games[game_id]
    if winner_index < len(game.players):
        winner_name = game.players[winner_index].name
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
    trick_result_json = trick_result_dto.model_dump_json()

    for _, websocket in connected_clients[game_id].items():
        try:
            await websocket.send_text(trick_result_json)
        except Exception:
            pass


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
            if game_id in active_games:
                del active_games[game_id]
            if game_id in game_timestamps:
                del game_timestamps[game_id]
            if game_id in game_versions:
                del game_versions[game_id]
            if game_id in game_locks:
                del game_locks[game_id]
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
