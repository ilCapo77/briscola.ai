"""
Encoder Numba per osservazioni `fast_2p`.

Questo modulo prepara l'integrazione del rollout A2C su stato numerico/JIT:
mantiene lo stesso layout di `encode_fast_observation_2p`, ma lo costruisce da array
compatti (`hands`, `hand_sizes`, `points`, `table_cards`, ...). Il wrapper Python
serve per i test di equivalenza; il target finale è chiamare la funzione JIT da un
rollout Numba senza riconvertire da liste Python a ogni step.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .fast_2p import Fast2PState
from .fast_numba import ACTION_DIM, CARD_POINTS_NUMBA, CARD_STRENGTH_NUMBA, CARD_SUIT_NUMBA
from .training.observation_encoder import (
    FEATURE_DIM_2P_V1,
    FEATURE_DIM_2P_V2,
    EncodedObservation,
    EncoderVersion,
)


@njit(cache=True)
def encode_fast_observation_arrays_numba(
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    points: np.ndarray,
    table_cards: np.ndarray,
    table_size: int,
    deck_size: int,
    current_turn: int,
    trump_card: int,
    player_index: int,
    seen_cards: np.ndarray,
    feature_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encoda una osservazione fast partendo da array numerici.

    Ritorna `(features, action_mask)`:
    - `features`: float32 con layout v1/v2 canonico;
    - `action_mask`: bool[40], True per le carte presenti nella mano del player.
    """
    features = np.zeros(feature_dim, dtype=np.float32)
    action_mask = np.zeros(ACTION_DIM, dtype=np.bool_)

    for i in range(hand_sizes[player_index]):
        card_id = hands[player_index, i]
        action_mask[card_id] = True
        features[card_id] = 1.0
        features[ACTION_DIM + card_id] = float(CARD_POINTS_NUMBA[card_id])
        features[2 * ACTION_DIM + card_id] = float(CARD_STRENGTH_NUMBA[card_id])

    table_offset = 3 * ACTION_DIM
    for i in range(table_size):
        card_id = table_cards[i]
        features[table_offset + card_id] = 1.0
        features[table_offset + ACTION_DIM + card_id] = float(CARD_POINTS_NUMBA[card_id])
        features[table_offset + 2 * ACTION_DIM + card_id] = float(CARD_STRENGTH_NUMBA[card_id])

    scalar_offset = 6 * ACTION_DIM
    trump_suit = CARD_SUIT_NUMBA[trump_card]
    features[scalar_offset + trump_suit] = 1.0

    opp_index = 1 - player_index
    is_second_in_trick = 1.0 if current_turn == player_index and table_size == 1 else 0.0
    features[scalar_offset + 4] = float(deck_size) / 40.0
    features[scalar_offset + 5] = float(points[player_index]) / 120.0
    features[scalar_offset + 6] = float(points[opp_index]) / 120.0
    features[scalar_offset + 7] = is_second_in_trick

    if feature_dim == int(FEATURE_DIM_2P_V2):
        seen_offset = int(FEATURE_DIM_2P_V1)
        for card_id in range(ACTION_DIM):
            features[seen_offset + card_id] = float(seen_cards[card_id])

    return features, action_mask


def _state_to_numba_arrays(state: Fast2PState) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Converte `Fast2PState` in array compatti per il wrapper/test Python."""
    hands = np.full((2, 3), -1, dtype=np.int64)
    hand_sizes = np.zeros(2, dtype=np.int64)
    for player_index in range(2):
        hand_sizes[player_index] = len(state.hands[player_index])
        for i, card_id in enumerate(state.hands[player_index]):
            hands[player_index, i] = int(card_id)

    points = np.asarray(state.points, dtype=np.int64)
    table_cards = np.full(2, -1, dtype=np.int64)
    for i, card_id in enumerate(state.table_cards):
        table_cards[i] = int(card_id)

    return hands, hand_sizes, points, table_cards


def encode_fast_observation_numba_2p(
    state: Fast2PState,
    *,
    player_index: int,
    seen_cards_onehot: tuple[int, ...],
    version: EncoderVersion = "v1",
) -> EncodedObservation:
    """
    Wrapper Python dell'encoder JIT, con lo stesso contratto di `encode_fast_observation_2p`.

    Il wrapper valida input e converte liste Python in array; nel rollout JIT finale useremo
    direttamente `encode_fast_observation_arrays_numba`.
    """
    if player_index not in (0, 1):
        raise ValueError(f"player_index fuori range: {player_index}")
    if len(seen_cards_onehot) != ACTION_DIM:
        raise ValueError(f"seen_cards_onehot len={len(seen_cards_onehot)} (atteso {ACTION_DIM})")
    if version == "v1":
        feature_dim = int(FEATURE_DIM_2P_V1)
    elif version == "v2":
        feature_dim = int(FEATURE_DIM_2P_V2)
    else:
        raise ValueError(f"Encoder version non supportata: {version!r}")

    seen_cards = np.asarray(seen_cards_onehot, dtype=np.int64)
    if not np.all((seen_cards == 0) | (seen_cards == 1)):
        raise ValueError("seen_cards_onehot deve contenere solo 0/1")

    hands, hand_sizes, points, table_cards = _state_to_numba_arrays(state)
    features, action_mask = encode_fast_observation_arrays_numba(
        hands,
        hand_sizes,
        points,
        table_cards,
        len(state.table_cards),
        len(state.deck),
        int(state.current_turn),
        int(state.trump_card),
        int(player_index),
        seen_cards,
        feature_dim,
    )
    return EncodedObservation(features=features.astype(float).tolist(), action_mask=action_mask.astype(bool).tolist())


def warm_up_numba_observation() -> None:
    """Compila l'encoder osservazione Numba con un input minimo."""
    hands = np.full((2, 3), -1, dtype=np.int64)
    hands[0, 0] = 0
    hands[1, 0] = 10
    hand_sizes = np.asarray([1, 1], dtype=np.int64)
    points = np.asarray([0, 0], dtype=np.int64)
    table_cards = np.full(2, -1, dtype=np.int64)
    seen_cards = np.zeros(ACTION_DIM, dtype=np.int64)
    encode_fast_observation_arrays_numba(
        hands,
        hand_sizes,
        points,
        table_cards,
        0,
        34,
        0,
        20,
        0,
        seen_cards,
        int(FEATURE_DIM_2P_V2),
    )
