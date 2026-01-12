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

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..game.game import BriscolaGame
from ..game.models import Card, Rank, Suit


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


class AiTurnRequest(BaseModel):
    """
    Payload per triggerare la mossa IA in modo idempotente.

    Campi:
        expected_version:
            Se presente, il backend esegue la mossa IA solo se coincide con la `server_version`
            corrente della partita (inclusa negli snapshot WS/HTTP). Questo evita doppi trigger
            dovuti a reconnect o richieste duplicate.
    """

    expected_version: Optional[int] = None


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
#   Il frontend può inviare `expected_version` per rendere il trigger IA idempotente.
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
        # Restituisce una vista specifica per il giocatore
        try:
            payload = game.get_observation_for_player(player_index)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        # Restituisce lo stato completo (per spettatori o debugging)
        payload = game.get_game_state()

    # Aggiungiamo metadati server-side utili a rendere la UI idempotente e debug-friendly.
    payload["server_version"] = game_versions.get(game_id, 0)
    return _json_safe(payload)


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

        # Se la presa è stata completata dall'umano, invia notifica con carte e vincitore.
        #
        # Nota architetturale (trigger model):
        # evitiamo `asyncio.sleep()` nel backend per ritardi "di presentazione". Il tempo
        # di visualizzazione del risultato è gestito dal frontend (che può “trattenere”
        # lo snapshot successivo finché l'utente ha letto la presa).
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
            # Presa non completata: notifica normale
            if game_id in connected_clients:
                await notify_clients(game_id)

    # Il frontend triggerà la mossa IA quando sarà pronto (dopo le animazioni).
    # Non scheduliamo più automaticamente la mossa IA qui.

    payload = dict(result)
    payload["server_version"] = game_versions.get(game_id, 0)
    return _json_safe(payload)


@app.post("/games/{game_id}/ai-turn", response_model=Dict)
async def trigger_ai_turn(game_id: str, request: AiTurnRequest = Body(default=AiTurnRequest())):
    """
    Endpoint per triggerare la mossa dell'IA.
    
    Chiamato dal frontend quando le animazioni sono complete e il giocatore
    è pronto a vedere la mossa dell'IA. Questo separa la logica di presentazione
    (frontend) dalla logica di gioco (backend).
    """
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Partita non trovata")

    expected_version = request.expected_version

    async with game_locks[game_id]:
        game = active_games[game_id]

        if game.game_over:
            return {"status": "no_action", "reason": "game_over", "server_version": game_versions.get(game_id, 0)}

        current_version = game_versions.get(game_id, 0)
        if expected_version is not None and expected_version != current_version:
            # Idempotenza: se il client ha già triggerato (o è in ritardo), non facciamo nulla.
            return {"status": "no_action", "reason": "version_mismatch", "server_version": current_version}

        # Verifica che sia effettivamente il turno dell'IA (non del giocatore umano, index 0).
        # Nota: in 2-player l'umano è sempre player 0; questa assunzione va generalizzata per 4-player.
        human_player_index = 0
        if game.current_turn == human_player_index:
            return {"status": "no_action", "reason": "not_ai_turn", "server_version": current_version}

        result = await _execute_ai_turn(game_id, human_player_index)

        if not result:
            return {"status": "no_action", "reason": "no_action", "server_version": game_versions.get(game_id, 0)}

        out = dict(result)
        out["server_version"] = game_versions.get(game_id, 0)
        return _json_safe(out)


async def _execute_ai_turn(game_id: str, human_player_index: int):
    """
    Esegue la mossa dell'IA.
    
    Nota: non c'è più delay iniziale. Il timing delle animazioni è gestito
    dal frontend che chiama questo endpoint quando è pronto.
    """
    if game_id not in active_games:
        return None

    game = active_games[game_id]

    # Se la partita è finita o tocca al giocatore umano, non fare nulla
    if game.game_over or game.current_turn == human_player_index:
        return None

    # AI gioca una carta casuale tra le valide
    ai_player_index = game.current_turn
    observation = game.get_observation_for_player(ai_player_index)
    valid_actions = observation.get("valid_actions", [])

    if not valid_actions:
        return None

    card_index = random.choice(valid_actions)
    selected_card = game.players[ai_player_index].hand[card_index]

    # Invia messaggio per rivelare la carta nella mano IA
    if game_id in connected_clients:
        reveal_message = {
            "type": "ai_card_reveal",
            "card_index": card_index,
            "card": {
                "suit": selected_card.suit.value,
                "rank": selected_card.rank.name,
                "number": selected_card.rank.number,
                "points": selected_card.rank.points,
            },
        }
        for client in connected_clients[game_id].values():
            try:
                await client.send_json(reveal_message)
            except Exception:
                pass

    result = game.play_action(card_index)
    game_versions[game_id] = game_versions.get(game_id, 0) + 1

    # Se la presa è stata completata, invia notifica speciale
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

    return result


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
        # Invia lo stato iniziale della partita
        observation = game.get_observation_for_player(player_index)
        observation["server_version"] = game_versions.get(game_id, 0)
        json_data = json.dumps(observation, cls=GameJSONEncoder)
        await websocket.send_text(json_data)

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

    # Invia lo stato aggiornato a ogni client connesso
    for player_idx, websocket in connected_clients[game_id].items():
        try:
            observation = game.get_observation_for_player(player_idx)
            observation["server_version"] = game_versions.get(game_id, 0)
            json_data = json.dumps(observation, cls=GameJSONEncoder)
            await websocket.send_text(json_data)
        except Exception:
            # Gestisce client disconnessi
            pass


async def notify_trick_result(game_id: str, trick_cards: list, winner_index: int, points: int):
    """
    Notifica i client del risultato della presa con le carte visibili.

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

    trick_result_message = {
        "type": "trick_result",
        "trick_cards": trick_cards,
        "winner_index": winner_index,
        "winner_name": winner_name,
        "points": points,
        "server_version": game_versions.get(game_id, 0),
    }

    for _, websocket in connected_clients[game_id].items():
        try:
            json_data = json.dumps(trick_result_message, cls=GameJSONEncoder)
            await websocket.send_text(json_data)
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
