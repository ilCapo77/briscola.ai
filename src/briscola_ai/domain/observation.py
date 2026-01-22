"""
Osservazioni (information set) per agenti/ML.

Problema
--------
Nel dominio, `GameState` contiene informazione *completa* (mazzo e mani di tutti).
Se passiamo `GameState` direttamente a un agente, una IA potrebbe "barare" leggendo:
- l'ordine del mazzo (`state.deck`)
- le carte in mano agli avversari (`state.players[i].hand`)

Soluzione
---------
Questo modulo definisce una vista *parziale* e “lecita” dello stato:
`PlayerObservation` contiene solo ciò che un giocatore potrebbe conoscere durante una partita.

Obiettivo didattico:
- rendere esplicito il confine tra “stato completo” (debug/replay) e “osservazione”
  (policy/ML), così da evitare leak informativi accidentali.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import Card
from .state import GameState


@dataclass(frozen=True, slots=True)
class PlayerObservation:
    """
    Osservazione del gioco dal punto di vista di un singolo giocatore.

    Invarianti (anti-cheat):
    - NON contiene `deck` né informazioni su carte specifiche in mano agli avversari.
    - Contiene solo:
      - mano del giocatore osservante
      - briscola scoperta (se presente)
      - carte sul tavolo (pubbliche) e chi le ha giocate
      - dimensione del mazzo (pubblica)
      - punteggi e dimensioni delle mani (informazione pubblica, dato che le prese sono visibili)

    Nota:
    I campi sono volutamente ridondanti/espliciti per rendere semplice costruire feature per ML.
    """

    num_players: int
    is_team_game: bool
    teams: Optional[tuple[tuple[int, int], tuple[int, int]]]

    player_index: int
    player_name: str

    hand: tuple[Card, ...]

    trump_card: Optional[Card]
    deck_size: int

    table_cards: tuple[tuple[Card, int], ...]
    current_turn: int
    first_player: int

    game_over: bool
    winner_index: Optional[int]  # 2-player
    winning_team: Optional[int]  # 4-player

    players_points: tuple[int, ...]
    players_hand_sizes: tuple[int, ...]


def make_player_observation(state: GameState, player_index: int) -> PlayerObservation:
    """
    Costruisce l'osservazione lecita per `player_index`.

    Argomenti:
        state: stato completo del dominio (include informazione nascosta)
        player_index: indice del giocatore osservante (0..num_players-1)

    Ritorna:
        Un `PlayerObservation` che può essere passato a policy/ML senza leak informativi.
    """
    if player_index < 0 or player_index >= state.num_players:
        raise ValueError(f"player_index fuori range: {player_index} (num_players={state.num_players})")

    return PlayerObservation(
        num_players=state.num_players,
        is_team_game=state.is_team_game,
        teams=state.teams,
        player_index=player_index,
        player_name=state.players[player_index].name,
        hand=state.players[player_index].hand,
        trump_card=state.trump_card,
        deck_size=len(state.deck),
        table_cards=state.table_cards,
        current_turn=state.current_turn,
        first_player=state.first_player,
        game_over=state.game_over,
        winner_index=state.winner_index,
        winning_team=state.winning_team,
        players_points=tuple(p.points for p in state.players),
        players_hand_sizes=tuple(len(p.hand) for p in state.players),
    )
