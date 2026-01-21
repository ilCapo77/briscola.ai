"""
Agenti baseline (policy) per Briscola AI.

Scopo didattico
---------------
Prima delle reti neurali, conviene avere:
- baseline molto semplici (random, euristiche minime);
- un'interfaccia comune (dato uno stato, scegliere un'azione valida);
- un modo per confrontare agenti in modo riproducibile.

Questo modulo definisce alcune policy “plug-in” usabili sia in self-play offline,
sia (in futuro) nel backend come IA.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from ..domain.models import Card
from ..domain.state import GameState


class Agent(Protocol):
    """
    Interfaccia minima di un agente.

    Un agente deve scegliere un `card_index` valido (indice nella mano del player).
    In Briscola qualunque carta in mano è giocabile, quindi l'insieme delle azioni valide
    è sempre `range(len(hand))` (finché la partita non è finita).
    """

    name: str

    def choose_card_index(self, state: GameState, player_index: int, *, rng: random.Random) -> int:
        """Sceglie l'indice della carta da giocare per `player_index`."""


@dataclass(frozen=True)
class RandomAgent:
    """
    Baseline: sceglie una carta casuale tra quelle in mano.

    È una baseline “zero-intelligenza” ma molto utile per:
    - validare che il loop di simulazione funzioni;
    - avere un punto di riferimento per valutare euristiche e modelli.
    """

    name: str = "random"

    def choose_card_index(self, state: GameState, player_index: int, *, rng: random.Random) -> int:
        hand_size = len(state.players[player_index].hand)
        if hand_size <= 0:
            raise ValueError("Mano vuota: nessuna azione possibile")
        return rng.randrange(hand_size)


@dataclass(frozen=True)
class GreedyPointsAgent:
    """
    Euristica minimale: gioca la carta con più punti nella mano.

    Nota:
    Non è detto che sia forte in Briscola (spesso conviene conservare carichi),
    ma è una policy deterministica e “spiegabile”, utile come esempio.
    """

    name: str = "greedy_points"

    def choose_card_index(self, state: GameState, player_index: int, *, rng: random.Random) -> int:
        hand = state.players[player_index].hand
        if not hand:
            raise ValueError("Mano vuota: nessuna azione possibile")

        # In caso di pareggio, scegliamo in modo pseudo-casuale per non fissare sempre la stessa carta.
        best_points = max(card.rank.points for card in hand)
        candidates = [i for i, card in enumerate(hand) if card.rank.points == best_points]
        return candidates[rng.randrange(len(candidates))]


def card_to_short_string(card: Card) -> str:
    """
    Rappresentazione breve di una carta (debug).

    Esempio: `cups_ACE` o `clubs_TWO`.
    """

    return f"{card.suit.value}_{card.rank.name}"
