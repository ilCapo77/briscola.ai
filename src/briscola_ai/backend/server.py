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

from ..domain.engine import PlayCardAction, step
from ..domain.state import GameState as DomainGameState
from ..domain.state import new_game_state
from .dto import (
    AiCardRevealDTO,
    CardDTO,
    GameStateDTO,
    ObservationDTO,
    PlayActionResultDTO,
    PlayerInfoDTO,
    PlayerStateDTO,
    TableCardDTO,
    TrickResultDTO,
)


def _build_observation_dto(state: DomainGameState, player_index: int, server_version: int) -> ObservationDTO:
    """
    Costruisce un ObservationDTO dal dominio (stato puro).

    Questa funzione centralizza la conversione da stato di gioco a payload WS/HTTP (Phase 2B),
    garantendo che il formato sia sempre coerente con il contratto DTO.
    Supporta sia modalità 2-player che 4-player.
    """
    if player_index < 0 or player_index >= state.num_players:
        raise ValueError(f"L'indice giocatore deve essere compreso tra 0 e {state.num_players - 1}")

    me = state.players[player_index]
    my_turn = state.current_turn == player_index

    # Converti carte in mano
    my_hand = [CardDTO.from_domain(card) for card in me.hand]

    # Converti carte in mano
    trump_card = CardDTO.from_domain(state.trump_card) if state.trump_card else None
    trump_suit = state.trump_card.suit.value if state.trump_card else None

    # Converti carte sul tavolo
    table_cards = [TableCardDTO.from_domain(card, idx) for card, idx in state.table_cards]

    # Costruisci lista players (sostituisce player_{n}_* dinamici)
    players: list[PlayerInfoDTO] = []
    for i, player in enumerate(state.players):
        players.append(
            PlayerInfoDTO(
                index=i,
                name=player.name,
                points=player.points,
                hand_size=len(player.hand),
            )
        )

    # Campi 4-player (None se 2-player)
    my_team = None
    teammate_index = None
    teammate_points = None
    my_team_points = None
    opponent_team_points = None
    if state.is_team_game and state.teams is not None:
        if player_index in state.teams[0]:
            my_team = 0
            teammate_index = state.teams[0][0] if state.teams[0][1] == player_index else state.teams[0][1]
        else:
            my_team = 1
            teammate_index = state.teams[1][0] if state.teams[1][1] == player_index else state.teams[1][1]

        teammate_points = state.players[teammate_index].points if teammate_index is not None else 0
        my_team_points = sum(state.players[i].points for i in state.teams[my_team]) if my_team is not None else 0
        opponent_team_points = (
            sum(state.players[i].points for i in state.teams[1 - my_team]) if my_team is not None else 0
        )

    return ObservationDTO(
        server_version=server_version,
        my_index=player_index,
        my_hand=my_hand,
        my_points=me.points,
        my_turn=my_turn,
        trump_card=trump_card,
        trump_suit=trump_suit,
        table_cards=table_cards,
        cards_remaining_in_deck=len(state.deck),
        valid_actions=list(range(len(me.hand))) if my_turn and not state.game_over else [],
        game_over=state.game_over,
        num_players=state.num_players,
        is_team_game=state.is_team_game,
        players=players,
        my_team=my_team,
        teammate_index=teammate_index,
        teammate_points=teammate_points,
        my_team_points=my_team_points,
        opponent_team_points=opponent_team_points,
    )


def _build_game_state_dto(state: DomainGameState, server_version: int) -> GameStateDTO:
    """
    Costruisce un GameStateDTO (stato completo) dal dominio.

    Uso previsto:
    - endpoint HTTP `GET /games/{id}` senza `player_index` (debug/spectator)

    Nota sicurezza/fair-play:
    Questo payload contiene tutte le mani e quindi NON deve essere usato da un client
    che rappresenta un singolo giocatore umano.
    """
    trump_card = CardDTO.from_domain(state.trump_card) if state.trump_card else None
    trump_suit = state.trump_card.suit.value if state.trump_card else None
    table_cards = [TableCardDTO.from_domain(card, idx) for card, idx in state.table_cards]

    players: list[PlayerStateDTO] = []
    for i, player in enumerate(state.players):
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

    teams = list(state.teams) if state.teams is not None else None
    team_0_points = sum(state.players[i].points for i in state.teams[0]) if state.teams is not None else None
    team_1_points = sum(state.players[i].points for i in state.teams[1]) if state.teams is not None else None

    return GameStateDTO(
        server_version=server_version,
        num_players=state.num_players,
        is_team_game=state.is_team_game,
        trump_card=trump_card,
        trump_suit=trump_suit,
        table_cards=table_cards,
        current_turn=state.current_turn,
        first_player=state.first_player,
        cards_remaining_in_deck=len(state.deck),
        valid_actions=list(range(len(state.players[state.current_turn].hand))) if not state.game_over else [],
        game_over=state.game_over,
        trick_in_progress=len(state.table_cards) > 0,
        trick_size=len(state.table_cards),
        expected_trick_size=state.num_players,
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
active_games: Dict[str, DomainGameState] = {}
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
        # Seed per rendere riproducibile lo shuffle in fase di debugging/dataset.
        # In produzione useremmo RNG più robusto o un seed esplicito del client.
        seed = random.randrange(0, 2**32)
        state = new_game_state(config.num_players, config.player_names, seed=seed)

        # Genera un ID univoco per la partita
        game_id = str(uuid.uuid4())
        active_games[game_id] = state
        game_timestamps[game_id] = datetime.now()
        game_data[game_id] = [{"timestamp": datetime.now().isoformat(), "event": "game_created", "seed": seed}]
        game_locks[game_id] = asyncio.Lock()
        game_versions[game_id] = 0

        # Inizializza il dizionario delle connessioni WebSocket per questa partita
        connected_clients[game_id] = {}

        return {
            "game_id": game_id,
            "status": "created",
            "num_players": config.num_players,
            "is_team_game": state.is_team_game,
            "player_names": [p.name for p in state.players],
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
            observation_dto = _build_observation_dto(game, player_index, game_versions.get(game_id, 0))
            return observation_dto.model_dump()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        # Restituisce lo stato completo (per spettatori o debugging) come DTO Pydantic.
        game_state_dto = _build_game_state_dto(game, game_versions.get(game_id, 0))
        return game_state_dto.model_dump()


@app.post("/games/{game_id}/actions", response_model=PlayActionResultDTO, response_model_exclude_none=True)
async def play_action(game_id: str, action: GameAction) -> PlayActionResultDTO:
    """Gioca una carta nella partita"""
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Partita non trovata")

    should_schedule_ai = False

    async with game_locks[game_id]:
        state = active_games[game_id]

        # Verifica che sia il turno del giocatore
        if state.current_turn != action.player_index:
            raise HTTPException(status_code=400, detail="Non è il tuo turno")

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

    # AI gioca una carta casuale tra le valide
    ai_player_index = state.current_turn
    valid_actions = list(range(len(state.players[ai_player_index].hand))) if not state.game_over else []

    if not valid_actions:
        return

    card_index = random.choice(valid_actions)
    selected_card = state.players[ai_player_index].hand[card_index]

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

    new_state, step_result = step(state, PlayCardAction(player_index=ai_player_index, card_index=card_index))
    if step_result.error:
        return

    active_games[game_id] = new_state
    game_versions[game_id] = game_versions.get(game_id, 0) + 1

    result: Dict[str, object] = {
        "played_card": step_result.played_card,
        "player": step_result.player,
        "trick_completed": step_result.trick_completed,
        "trick_winner": step_result.trick_winner,
        "captured_cards": [],
        "cards_dealt": step_result.cards_dealt,
        "trick_size": len(step_result.trick_cards),
        "is_ai": True,
    }
    if step_result.trick_completed:
        result["trick_cards"] = list(step_result.trick_cards)
        result["captured_cards"] = [card for card, _ in step_result.trick_cards]

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

    state = active_games[game_id]
    if not state.game_over:
        return {"game_in_progress": True}

    result: Dict[str, object] = {"game_over": True, "is_team_game": state.is_team_game}
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

        result.update(
            {
                "winner": winner_str,
                "winning_team": winning_team,
                "team_points": {"Team 0": team_0_points, "Team 1": team_1_points},
                "individual_points": {p.name: p.points for p in state.players},
                "point_difference": abs(team_0_points - team_1_points),
            }
        )
        return result

    p0 = state.players[0].points
    p1 = state.players[1].points
    if p0 > p1:
        winner_index = 0
    elif p1 > p0:
        winner_index = 1
    else:
        winner_index = None

    result.update(
        {
            "winner": state.players[winner_index].name if winner_index is not None else "Pareggio",
            "winner_index": winner_index,
            "points": {p.name: p.points for p in state.players},
            "point_difference": abs(p0 - p1),
        }
    )
    return result


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
    connected_clients[game_id][player_index] = websocket

    try:
        # Invia lo stato iniziale della partita (usando DTO)
        dto = _build_observation_dto(state, player_index, game_versions.get(game_id, 0))
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

    state = active_games[game_id]

    # Invia lo stato aggiornato a ogni client connesso (usando DTO)
    server_version = game_versions.get(game_id, 0)
    for player_idx, websocket in connected_clients[game_id].items():
        try:
            dto = _build_observation_dto(state, player_idx, server_version)
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
