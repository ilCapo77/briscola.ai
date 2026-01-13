"""
DTO (Data Transfer Objects) per i messaggi WebSocket.

Questo modulo definisce i modelli Pydantic v2 per tutti i messaggi scambiati
tra backend e frontend via WebSocket. I DTO garantiscono:
- Validazione automatica dei payload
- Documentazione implicita del contratto API
- Serializzazione coerente (no encoder custom)

Note architetturali:
- **Focus 2-player**: i DTO sono progettati per la modalità 2 giocatori.
  I campi 4-player (my_team, teammate_index, ecc.) non sono inclusi.
  Se in futuro si vuole supportare il 4-player via WS, estendere ObservationDTO.
- **HTTP vs WS**: gli endpoint HTTP (/api/games/{id}) usano ancora il formato
  "vecchio" con `GameJSONEncoder`. I DTO sono usati solo per i messaggi WS.
  Questo è intenzionale per minimizzare il rischio di breaking change sugli
  endpoint REST già in uso. In futuro si potrebbe allineare anche l'HTTP.

Nota: i DTO sono separati dai modelli di dominio (`game/models.py`) per mantenere
la separazione tra logica di gioco e serializzazione API.
"""

from typing import Literal

from pydantic import BaseModel

from ..game.models import Card


class CardDTO(BaseModel):
    """
    Rappresentazione JSON di una carta.

    Campi:
    - suit: seme (clubs, cups, coins, swords)
    - rank: nome del rango (ACE, TWO, THREE, ...)
    - number: valore numerico stampato (1-10)
    - points: punti Briscola (11, 10, 4, 3, 2, 0)
    """

    suit: str
    rank: str
    number: int
    points: int

    @classmethod
    def from_domain(cls, card: Card) -> "CardDTO":
        """Converte una Card di dominio in CardDTO."""
        return cls(
            suit=card.suit.value,
            rank=card.rank.name,
            number=card.rank.number,
            points=card.rank.points,
        )


class TableCardDTO(BaseModel):
    """
    Una carta sul tavolo con l'indice del giocatore che l'ha giocata.

    Sostituisce il formato tuple `[card, player_index]` con un oggetto esplicito.
    """

    card: CardDTO
    player_index: int

    @classmethod
    def from_domain(cls, card: Card, player_index: int) -> "TableCardDTO":
        """Converte una tupla (Card, player_index) in TableCardDTO."""
        return cls(card=CardDTO.from_domain(card), player_index=player_index)


class PlayerInfoDTO(BaseModel):
    """
    Informazioni su un giocatore (visibili a tutti).

    Sostituisce le chiavi dinamiche `player_{n}_points`, `player_{n}_hand_size`.
    """

    index: int
    name: str
    points: int
    hand_size: int


class ObservationDTO(BaseModel):
    """
    Snapshot dello stato di gioco dal punto di vista di un giocatore.

    Questo è il messaggio principale inviato via WebSocket dopo ogni azione.
    """

    type: Literal["observation"] = "observation"
    server_version: int

    # Identità del giocatore
    my_index: int
    my_hand: list[CardDTO]
    my_points: int
    my_turn: bool

    # Stato del tavolo
    trump_card: CardDTO | None
    trump_suit: str | None
    table_cards: list[TableCardDTO]
    cards_remaining_in_deck: int

    # Azioni e stato partita
    valid_actions: list[int]
    game_over: bool
    num_players: int
    is_team_game: bool

    # Info sugli altri giocatori (sostituisce player_{n}_*)
    players: list[PlayerInfoDTO]


class AiCardRevealDTO(BaseModel):
    """
    Messaggio inviato quando l'IA sceglie una carta (prima di giocarla).

    Permette al frontend di mostrare la carta in mano prima di spostarla sul tavolo.
    """

    type: Literal["ai_card_reveal"] = "ai_card_reveal"
    card_index: int
    card: CardDTO


class TrickResultDTO(BaseModel):
    """
    Messaggio inviato quando una mano si completa.

    Contiene le carte giocate, il vincitore e i punti della mano.
    """

    type: Literal["trick_result"] = "trick_result"
    trick_cards: list[TableCardDTO]
    winner_index: int
    winner_name: str
    points: int
    server_version: int
