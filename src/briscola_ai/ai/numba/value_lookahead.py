"""
Kernel Numba per V-lookahead su stati numerici determinizzati.

Il runtime web usa `ValueLookaheadAgent`, che parte da `PlayerObservation` e campiona
determinizzazioni anti-cheat. Nei training futuri, invece, spesso abbiamo gia' uno stato
2-player completo o determinizzato in array NumPy: in quel caso riconvertire a `GameState`,
`PlayerObservation` e oggetti Python a ogni mossa spreca throughput.

Questo modulo espone quindi un entrypoint JIT che:
- lavora direttamente su `hands`, `deck`, `points`, `table_cards`;
- prova le carte in mano al giocatore corrente;
- usa la policy MLP come continuazione quando la carta candidata apre una presa;
- usa il solver endgame Numba quando la foglia e' a mazzo vuoto;
- valuta le altre foglie con il value model MLP.

Il kernel non fa determinizzazione: il chiamante deve passargli uno stato numerico coerente
con il proprio protocollo di training/evaluation.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from ...domain.card_id import card_to_id
from ...domain.state import GameState
from ..endgame.numba_solver import choose_endgame_card_numba_arrays
from .core import ACTION_DIM
from .observation import (
    _action_id_to_hand_index_numba,
    _apply_numba_card_index,
    _apply_overkill_guard_numba,
    _argmax_mlp_policy_action_numba,
    encode_fast_observation_arrays_numba,
)

_MAX_DECK_SIZE_2P = 34
_HAND_CAPACITY_2P = 3
_TABLE_CAPACITY_2P = 2


def arrays_from_game_state_for_value_lookahead(
    state: GameState,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, int, int, int, np.ndarray, np.ndarray
]:
    """
    Converte un `GameState` 2-player in array compatibili con il kernel V-lookahead.

    Il formato del mazzo segue il fast/numba path del progetto: la prossima pescata e'
    `deck[deck_size - 1]`, come `domain.engine.step` pesca con `deck.pop()`.
    """
    if state.num_players != 2 or state.is_team_game:
        raise ValueError("Il V-lookahead Numba supporta solo partite 2-player")
    if state.trump_card is None:
        raise ValueError("Briscola assente")

    hands = np.full((2, _HAND_CAPACITY_2P), -1, dtype=np.int64)
    hand_sizes = np.zeros(2, dtype=np.int64)
    for player_index in range(2):
        hand = state.players[player_index].hand
        if len(hand) > _HAND_CAPACITY_2P:
            raise ValueError(f"Mano troppo grande per 2-player: player={player_index} size={len(hand)}")
        hand_sizes[player_index] = len(hand)
        for card_index, card in enumerate(hand):
            hands[player_index, card_index] = card_to_id(card)

    deck = np.full(_MAX_DECK_SIZE_2P, -1, dtype=np.int64)
    if len(state.deck) > _MAX_DECK_SIZE_2P:
        raise ValueError(f"Mazzo troppo grande per 2-player: {len(state.deck)}")
    for index, card in enumerate(state.deck):
        deck[index] = card_to_id(card)

    points = np.asarray([state.players[0].points, state.players[1].points], dtype=np.int64)
    table_cards = np.full(_TABLE_CAPACITY_2P, -1, dtype=np.int64)
    table_players = np.full(_TABLE_CAPACITY_2P, -1, dtype=np.int64)
    if len(state.table_cards) > _TABLE_CAPACITY_2P:
        raise ValueError(f"Tavolo troppo grande per 2-player: {len(state.table_cards)}")
    for index, (card, player_index) in enumerate(state.table_cards):
        table_cards[index] = card_to_id(card)
        table_players[index] = int(player_index)

    seen_cards = np.zeros(ACTION_DIM, dtype=np.int64)
    seen_cards[card_to_id(state.trump_card)] = 1
    out_of_play_cards = np.zeros(ACTION_DIM, dtype=np.int64)
    for card, _player_index in state.table_cards:
        card_id = card_to_id(card)
        seen_cards[card_id] = 1
        out_of_play_cards[card_id] = 1
    for player in state.players:
        for card in player.captured_cards:
            card_id = card_to_id(card)
            seen_cards[card_id] = 1
            out_of_play_cards[card_id] = 1

    return (
        hands,
        hand_sizes,
        points,
        deck,
        len(state.deck),
        table_cards,
        table_players,
        len(state.table_cards),
        int(state.current_turn),
        card_to_id(state.trump_card),
        seen_cards,
        out_of_play_cards,
    )


@njit(cache=True)
def _copy_state_arrays(
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    points: np.ndarray,
    deck: np.ndarray,
    table_cards: np.ndarray,
    table_players: np.ndarray,
    seen_cards: np.ndarray,
    out_of_play_cards: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Copia lo stato compatto per simulare una carta candidata senza mutare la radice."""
    hands_copy = np.empty((2, _HAND_CAPACITY_2P), dtype=np.int64)
    for player_index in range(2):
        for card_index in range(_HAND_CAPACITY_2P):
            hands_copy[player_index, card_index] = hands[player_index, card_index]

    hand_sizes_copy = np.empty(2, dtype=np.int64)
    points_copy = np.empty(2, dtype=np.int64)
    for player_index in range(2):
        hand_sizes_copy[player_index] = hand_sizes[player_index]
        points_copy[player_index] = points[player_index]

    deck_copy = np.empty(_MAX_DECK_SIZE_2P, dtype=np.int64)
    for index in range(_MAX_DECK_SIZE_2P):
        deck_copy[index] = deck[index]

    table_cards_copy = np.empty(_TABLE_CAPACITY_2P, dtype=np.int64)
    table_players_copy = np.empty(_TABLE_CAPACITY_2P, dtype=np.int64)
    for index in range(_TABLE_CAPACITY_2P):
        table_cards_copy[index] = table_cards[index]
        table_players_copy[index] = table_players[index]

    seen_copy = np.empty(ACTION_DIM, dtype=np.int64)
    out_copy = np.empty(ACTION_DIM, dtype=np.int64)
    for card_id in range(ACTION_DIM):
        seen_copy[card_id] = seen_cards[card_id]
        out_copy[card_id] = out_of_play_cards[card_id]

    return (
        hands_copy,
        hand_sizes_copy,
        points_copy,
        deck_copy,
        table_cards_copy,
        table_players_copy,
        seen_copy,
        out_copy,
    )


@njit(cache=True)
def _terminal_or_endgame_score_for_root(
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    points: np.ndarray,
    table_cards: np.ndarray,
    table_players: np.ndarray,
    table_size: int,
    current_turn: int,
    trump_card: int,
    root_player: int,
) -> tuple[bool, float]:
    """
    Valuta terminali ed endgame con solver.

    Ritorna `(handled, score_root)`. `handled=False` significa che la foglia va valutata da V.
    """
    if hand_sizes[0] == 0 and hand_sizes[1] == 0 and table_size == 0:
        if root_player == 0:
            return True, float(points[0] - points[1])
        return True, float(points[1] - points[0])

    best_card_index, final_delta_p0_p1 = choose_endgame_card_numba_arrays(
        hands,
        hand_sizes,
        points,
        table_cards,
        table_players,
        table_size,
        current_turn,
        trump_card,
    )
    if best_card_index >= 0 and hand_sizes[0] + hand_sizes[1] > 0:
        if root_player == 0:
            return True, float(final_delta_p0_p1)
        return True, float(-final_delta_p0_p1)
    return False, 0.0


@njit(cache=True)
def _predict_value_points_numba(
    value_w1: np.ndarray,
    value_b1: np.ndarray,
    value_w2: np.ndarray,
    value_b2: float,
    value_target_scale: float,
    value_target_is_residual: bool,
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
    out_of_play_cards: np.ndarray,
) -> float:
    """Forward del value model scalare in punti dal punto di vista di `player_index`."""
    feature_dim = value_w1.shape[0]
    hidden_dim = value_w1.shape[1]
    features, _action_mask = encode_fast_observation_arrays_numba(
        hands,
        hand_sizes,
        points,
        table_cards,
        table_size,
        deck_size,
        current_turn,
        trump_card,
        player_index,
        seen_cards,
        out_of_play_cards,
        feature_dim,
    )

    raw = float(value_b2)
    for hidden_index in range(hidden_dim):
        hidden_value = value_b1[hidden_index]
        for feature_index in range(feature_dim):
            hidden_value += features[feature_index] * value_w1[feature_index, hidden_index]
        if hidden_value < 0.0:
            hidden_value = 0.0
        raw += hidden_value * float(value_w2[hidden_index])

    pred = raw * float(value_target_scale)
    if value_target_is_residual:
        opponent = 1 - player_index
        pred += float(points[player_index] - points[opponent])
    return float(pred)


@njit(cache=True)
def choose_value_lookahead_card_numba_arrays(
    policy_w1: np.ndarray,
    policy_b1: np.ndarray,
    policy_w2: np.ndarray,
    policy_b2: np.ndarray,
    value_w1: np.ndarray,
    value_b1: np.ndarray,
    value_w2: np.ndarray,
    value_b2: float,
    value_target_scale: float,
    value_target_is_residual: bool,
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    points: np.ndarray,
    deck: np.ndarray,
    deck_size: int,
    table_cards: np.ndarray,
    table_players: np.ndarray,
    table_size: int,
    current_turn: int,
    trump_card: int,
    seen_cards: np.ndarray,
    out_of_play_cards: np.ndarray,
    overkill_guard_enabled: bool,
) -> tuple[int, float]:
    """
    Sceglie una carta con depth-1 V-lookahead su uno stato determinizzato.

    Ritorna `(card_index, expected_score_for_root)`, dove `card_index` e' un indice nella
    mano del `current_turn` radice.
    """
    root_player = int(current_turn)
    hand_size = int(hand_sizes[root_player])
    if hand_size <= 0:
        return -1, 0.0

    if int(deck_size) == 0:
        best_card_index, final_delta_p0_p1 = choose_endgame_card_numba_arrays(
            hands,
            hand_sizes,
            points,
            table_cards,
            table_players,
            table_size,
            current_turn,
            trump_card,
        )
        if root_player == 0:
            return int(best_card_index), float(final_delta_p0_p1)
        return int(best_card_index), float(-final_delta_p0_p1)

    best_card_index = 0
    best_score = -1.0e30

    for candidate_index in range(hand_size):
        (
            candidate_hands,
            candidate_hand_sizes,
            candidate_points,
            candidate_deck,
            candidate_table_cards,
            candidate_table_players,
            candidate_seen,
            candidate_out_of_play,
        ) = _copy_state_arrays(
            hands, hand_sizes, points, deck, table_cards, table_players, seen_cards, out_of_play_cards
        )

        candidate_deck_size, candidate_table_size, candidate_turn = _apply_numba_card_index(
            candidate_hands,
            candidate_hand_sizes,
            candidate_points,
            candidate_deck,
            int(deck_size),
            candidate_table_cards,
            candidate_table_players,
            int(table_size),
            int(current_turn),
            int(trump_card),
            candidate_index,
            candidate_seen,
            candidate_out_of_play,
        )

        if candidate_table_size == 1:
            if candidate_deck_size == 0:
                response_index, _response_delta = choose_endgame_card_numba_arrays(
                    candidate_hands,
                    candidate_hand_sizes,
                    candidate_points,
                    candidate_table_cards,
                    candidate_table_players,
                    candidate_table_size,
                    candidate_turn,
                    trump_card,
                )
            else:
                response_action = _argmax_mlp_policy_action_numba(
                    policy_w1,
                    policy_b1,
                    policy_w2,
                    policy_b2,
                    candidate_hands,
                    candidate_hand_sizes,
                    candidate_points,
                    candidate_table_cards,
                    candidate_table_size,
                    candidate_deck_size,
                    candidate_turn,
                    trump_card,
                    candidate_turn,
                    candidate_seen,
                    candidate_out_of_play,
                )
                response_action = _apply_overkill_guard_numba(
                    response_action,
                    candidate_hands,
                    candidate_hand_sizes,
                    candidate_turn,
                    candidate_table_cards,
                    candidate_table_players,
                    candidate_table_size,
                    trump_card,
                    overkill_guard_enabled,
                )
                response_index = _action_id_to_hand_index_numba(
                    candidate_hands, candidate_hand_sizes, candidate_turn, response_action
                )

            candidate_deck_size, candidate_table_size, candidate_turn = _apply_numba_card_index(
                candidate_hands,
                candidate_hand_sizes,
                candidate_points,
                candidate_deck,
                candidate_deck_size,
                candidate_table_cards,
                candidate_table_players,
                candidate_table_size,
                candidate_turn,
                trump_card,
                response_index,
                candidate_seen,
                candidate_out_of_play,
            )

        if candidate_deck_size == 0:
            handled, score = _terminal_or_endgame_score_for_root(
                candidate_hands,
                candidate_hand_sizes,
                candidate_points,
                candidate_table_cards,
                candidate_table_players,
                candidate_table_size,
                candidate_turn,
                trump_card,
                root_player,
            )
            if not handled:
                score = 0.0
        elif candidate_hand_sizes[0] == 0 and candidate_hand_sizes[1] == 0 and candidate_table_size == 0:
            if root_player == 0:
                score = float(candidate_points[0] - candidate_points[1])
            else:
                score = float(candidate_points[1] - candidate_points[0])
        else:
            leaf_player = int(candidate_turn)
            leaf_score = _predict_value_points_numba(
                value_w1,
                value_b1,
                value_w2,
                value_b2,
                value_target_scale,
                value_target_is_residual,
                candidate_hands,
                candidate_hand_sizes,
                candidate_points,
                candidate_table_cards,
                candidate_table_size,
                candidate_deck_size,
                candidate_turn,
                trump_card,
                leaf_player,
                candidate_seen,
                candidate_out_of_play,
            )
            score = leaf_score if leaf_player == root_player else -leaf_score

        if score > best_score:
            best_score = score
            best_card_index = candidate_index

    chosen_action = hands[root_player, best_card_index]
    guarded_action = _apply_overkill_guard_numba(
        chosen_action,
        hands,
        hand_sizes,
        root_player,
        table_cards,
        table_players,
        table_size,
        trump_card,
        overkill_guard_enabled,
    )
    if int(guarded_action) != int(chosen_action):
        best_card_index = _action_id_to_hand_index_numba(hands, hand_sizes, root_player, guarded_action)

    return int(best_card_index), float(best_score)


def choose_value_lookahead_card_numba(
    *,
    policy_w1: np.ndarray,
    policy_b1: np.ndarray,
    policy_w2: np.ndarray,
    policy_b2: np.ndarray,
    value_w1: np.ndarray,
    value_b1: np.ndarray,
    value_w2: np.ndarray,
    value_b2: float,
    value_target_scale: float,
    value_target_is_residual: bool,
    state: GameState,
    overkill_guard_enabled: bool = True,
) -> tuple[int, float]:
    """Wrapper Python comodo per test/benchmark da `GameState` canonico 2-player."""
    arrays = arrays_from_game_state_for_value_lookahead(state)
    return choose_value_lookahead_card_numba_arrays(
        policy_w1,
        policy_b1,
        policy_w2,
        policy_b2,
        value_w1,
        value_b1,
        value_w2,
        float(value_b2),
        float(value_target_scale),
        bool(value_target_is_residual),
        *arrays,
        bool(overkill_guard_enabled),
    )


def warm_up_numba_value_lookahead() -> None:
    """Compila il kernel V-lookahead con pesi minimali sintetici."""
    policy_w1 = np.zeros((310, 1), dtype=np.float32)
    policy_b1 = np.zeros(1, dtype=np.float32)
    policy_w2 = np.zeros((1, ACTION_DIM), dtype=np.float32)
    policy_b2 = np.zeros(ACTION_DIM, dtype=np.float32)
    value_w1 = np.zeros((310, 1), dtype=np.float32)
    value_b1 = np.zeros(1, dtype=np.float32)
    value_w2 = np.zeros(1, dtype=np.float32)

    hands = np.asarray([[0, 1, 2], [10, 11, 12]], dtype=np.int64)
    hand_sizes = np.asarray([3, 3], dtype=np.int64)
    points = np.asarray([0, 0], dtype=np.int64)
    deck = np.full(_MAX_DECK_SIZE_2P, -1, dtype=np.int64)
    deck[:4] = np.asarray([20, 21, 22, 23], dtype=np.int64)
    table_cards = np.full(_TABLE_CAPACITY_2P, -1, dtype=np.int64)
    table_players = np.full(_TABLE_CAPACITY_2P, -1, dtype=np.int64)
    seen_cards = np.zeros(ACTION_DIM, dtype=np.int64)
    out_of_play_cards = np.zeros(ACTION_DIM, dtype=np.int64)
    seen_cards[20] = 1

    choose_value_lookahead_card_numba_arrays(
        policy_w1,
        policy_b1,
        policy_w2,
        policy_b2,
        value_w1,
        value_b1,
        value_w2,
        0.0,
        120.0,
        True,
        hands,
        hand_sizes,
        points,
        deck,
        4,
        table_cards,
        table_players,
        0,
        0,
        20,
        seen_cards,
        out_of_play_cards,
        True,
    )
