"""
Core Numba sperimentale per simulazioni 2-player.

Questo modulo è il primo passo JIT: tiene tutto il loop partita dentro funzioni `njit`,
usando solo interi e array NumPy. Per ora copre random-vs-random, così possiamo misurare
il limite superiore del motore compilato senza introdurre policy neurali o oggetti dominio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .evaluation import MatchStats

ACTION_DIM = 40
CARD_SUIT_NUMBA = np.asarray([card_id // 10 for card_id in range(ACTION_DIM)], dtype=np.int64)
CARD_NUMBER_NUMBA = np.asarray([(card_id % 10) + 1 for card_id in range(ACTION_DIM)], dtype=np.int64)
_POINTS_BY_NUMBER = np.asarray([0, 11, 0, 10, 0, 0, 0, 0, 2, 3, 4], dtype=np.int64)
_STRENGTH_BY_NUMBER = np.asarray([0, 10, 1, 9, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
CARD_POINTS_NUMBA = np.asarray([_POINTS_BY_NUMBER[number] for number in CARD_NUMBER_NUMBA], dtype=np.int64)
CARD_STRENGTH_NUMBA = np.asarray([_STRENGTH_BY_NUMBER[number] for number in CARD_NUMBER_NUMBA], dtype=np.int64)


@dataclass(frozen=True, slots=True)
class NumbaRandomSummary:
    """Risultato aggregato del benchmark random-vs-random compilato."""

    num_games: int
    wins0: int
    wins1: int
    draws: int
    sum0: int
    sum1: int

    def to_match_stats(self) -> MatchStats:
        """Converte il summary nel DTO statistico usato dal resto della evaluation."""
        return MatchStats(
            num_games=self.num_games,
            agent0_name="random_numba",
            agent1_name="random_numba",
            wins_agent0=self.wins0,
            wins_agent1=self.wins1,
            draws=self.draws,
            avg_points_agent0=self.sum0 / self.num_games if self.num_games else 0.0,
            avg_points_agent1=self.sum1 / self.num_games if self.num_games else 0.0,
            avg_point_diff_agent0_minus_agent1=(self.sum0 - self.sum1) / self.num_games if self.num_games else 0.0,
        )


@njit(cache=True)
def _who_wins_trick_numba(
    first_card: int, first_player: int, second_card: int, second_player: int, trump_card: int
) -> int:
    """Determina il vincitore di una presa 2-player usando solo card id."""
    trump_suit = CARD_SUIT_NUMBA[trump_card]
    first_suit = CARD_SUIT_NUMBA[first_card]
    second_suit = CARD_SUIT_NUMBA[second_card]

    first_is_trump = first_suit == trump_suit
    second_is_trump = second_suit == trump_suit
    if first_is_trump or second_is_trump:
        if first_is_trump and not second_is_trump:
            return first_player
        if second_is_trump and not first_is_trump:
            return second_player
        if CARD_STRENGTH_NUMBA[first_card] >= CARD_STRENGTH_NUMBA[second_card]:
            return first_player
        return second_player

    if second_suit != first_suit:
        return first_player
    if CARD_STRENGTH_NUMBA[first_card] >= CARD_STRENGTH_NUMBA[second_card]:
        return first_player
    return second_player


@njit(cache=True)
def _shuffle_deck_numba(seed: int) -> np.ndarray:
    """Crea e mescola un mazzo numerico con RNG Numba locale alla funzione."""
    np.random.seed(seed)
    deck = np.arange(ACTION_DIM, dtype=np.int64)
    for i in range(ACTION_DIM - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        tmp = deck[i]
        deck[i] = deck[j]
        deck[j] = tmp
    return deck


@njit(cache=True)
def _play_random_game_numba(seed: int) -> tuple[int, int, int]:
    """
    Gioca una partita random-vs-random interamente in Numba.

    Ritorna `(points0, points1, winner)` con `winner=-1` in caso di pareggio.
    """
    shuffled = _shuffle_deck_numba(seed)

    deck = np.empty(34, dtype=np.int64)
    hands = np.empty((2, 3), dtype=np.int64)
    hand_sizes = np.zeros(2, dtype=np.int64)

    deck_size_source = ACTION_DIM
    for _ in range(3):
        for player_index in range(2):
            deck_size_source -= 1
            hands[player_index, hand_sizes[player_index]] = shuffled[deck_size_source]
            hand_sizes[player_index] += 1

    deck_size_source -= 1
    trump_card = shuffled[deck_size_source]
    deck[0] = trump_card
    for i in range(deck_size_source):
        deck[i + 1] = shuffled[i]
    deck_size = deck_size_source + 1

    points = np.zeros(2, dtype=np.int64)
    table_cards = np.empty(2, dtype=np.int64)
    table_players = np.empty(2, dtype=np.int64)
    table_size = 0
    current_turn = 0

    safety = 5000
    while safety > 0:
        safety -= 1
        if hand_sizes[0] == 0 and hand_sizes[1] == 0:
            break

        hand_size = hand_sizes[current_turn]
        card_index = np.random.randint(0, hand_size)
        played_card = hands[current_turn, card_index]

        for i in range(card_index, hand_size - 1):
            hands[current_turn, i] = hands[current_turn, i + 1]
        hand_sizes[current_turn] -= 1

        table_cards[table_size] = played_card
        table_players[table_size] = current_turn
        table_size += 1

        if table_size == 1:
            current_turn = 1 - current_turn
            continue

        first_card = table_cards[0]
        second_card = table_cards[1]
        first_player = table_players[0]
        second_player = table_players[1]
        winner = _who_wins_trick_numba(first_card, first_player, second_card, second_player, trump_card)
        points[winner] += CARD_POINTS_NUMBA[first_card] + CARD_POINTS_NUMBA[second_card]
        table_size = 0

        if deck_size > 0:
            for i in range(2):
                player_to_deal = (winner + i) % 2
                if deck_size <= 0:
                    break
                deck_size -= 1
                hands[player_to_deal, hand_sizes[player_to_deal]] = deck[deck_size]
                hand_sizes[player_to_deal] += 1

        current_turn = winner

    if points[0] > points[1]:
        winner_out = 0
    elif points[1] > points[0]:
        winner_out = 1
    else:
        winner_out = -1
    return int(points[0]), int(points[1]), int(winner_out)


@njit(cache=True)
def _play_random_games_numba(num_games: int, seed: int) -> tuple[int, int, int, int, int]:
    """Gioca `num_games` partite random-vs-random e aggrega i risultati."""
    wins0 = 0
    wins1 = 0
    draws = 0
    sum0 = 0
    sum1 = 0

    for game_index in range(num_games):
        p0, p1, winner = _play_random_game_numba(seed + game_index)
        sum0 += p0
        sum1 += p1
        if winner == 0:
            wins0 += 1
        elif winner == 1:
            wins1 += 1
        else:
            draws += 1

    return wins0, wins1, draws, sum0, sum1


def warm_up_numba() -> None:
    """Compila le funzioni Numba principali con un input minimo."""
    _play_random_games_numba(1, 0)


def play_random_game_numba(seed: int) -> tuple[int, int, int]:
    """Wrapper Python per una partita random-vs-random JIT."""
    return _play_random_game_numba(int(seed))


def evaluate_random_numba_2p(*, num_games: int, seed: int) -> NumbaRandomSummary:
    """Esegue random-vs-random con core Numba e ritorna statistiche aggregate."""
    if int(num_games) < 0:
        raise ValueError("num_games deve essere >= 0")
    wins0, wins1, draws, sum0, sum1 = _play_random_games_numba(int(num_games), int(seed))
    return NumbaRandomSummary(
        num_games=int(num_games),
        wins0=int(wins0),
        wins1=int(wins1),
        draws=int(draws),
        sum0=int(sum0),
        sum1=int(sum1),
    )
