"""
Encoder Numba per osservazioni `fast_2p`.

Questo modulo prepara l'integrazione del rollout A2C su stato numerico/JIT:
mantiene lo stesso layout di `encode_fast_observation_2p`, ma lo costruisce da array
compatti (`hands`, `hand_sizes`, `points`, `table_cards`, ...). Il wrapper Python
serve per i test di equivalenza; il target finale è chiamare la funzione JIT da un
rollout Numba senza riconvertire da liste Python a ogni step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .evaluation import MatchStats
from .fast_2p import Fast2PState
from .fast_numba import (
    ACTION_DIM,
    CARD_POINTS_NUMBA,
    CARD_STRENGTH_NUMBA,
    CARD_SUIT_NUMBA,
    _choose_policy_card_index_numba,
    _shuffle_deck_numba,
    _who_wins_trick_numba,
    numba_agent_code,
)
from .training.observation_encoder import (
    FEATURE_DIM_2P_V1,
    FEATURE_DIM_2P_V2,
    EncodedObservation,
    EncoderVersion,
)


@dataclass(frozen=True, slots=True)
class NumbaMLPRolloutSummary:
    """Risultato aggregato del rollout full-JIT di una policy MLP."""

    num_games: int
    policy_name: str
    opponent_name: str
    wins_policy: int
    wins_opponent: int
    draws: int
    sum_policy: int
    sum_opponent: int

    def to_match_stats(self) -> MatchStats:
        """Converte il summary nel DTO statistico standard."""
        return MatchStats(
            num_games=self.num_games,
            agent0_name=self.policy_name,
            agent1_name=self.opponent_name,
            wins_agent0=self.wins_policy,
            wins_agent1=self.wins_opponent,
            draws=self.draws,
            avg_points_agent0=self.sum_policy / self.num_games if self.num_games else 0.0,
            avg_points_agent1=self.sum_opponent / self.num_games if self.num_games else 0.0,
            avg_point_diff_agent0_minus_agent1=(
                (self.sum_policy - self.sum_opponent) / self.num_games if self.num_games else 0.0
            ),
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


@njit(cache=True)
def _action_id_to_hand_index_numba(hands: np.ndarray, hand_sizes: np.ndarray, player_index: int, action_id: int) -> int:
    """Converte action_id/card_id in indice nella mano numerica."""
    for i in range(hand_sizes[player_index]):
        if hands[player_index, i] == action_id:
            return i
    raise ValueError("action_id non presente nella mano")


@njit(cache=True)
def _sample_mlp_policy_action_numba(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    points: np.ndarray,
    table_cards: np.ndarray,
    table_size: int,
    deck_size: int,
    current_turn: int,
    trump_card: int,
    policy_seat: int,
    seen_cards: np.ndarray,
) -> int:
    """Esegue encoder + forward MLP + sampling mascherato dentro Numba."""
    feature_dim = w1.shape[0]
    hidden_dim = w1.shape[1]
    features, action_mask = encode_fast_observation_arrays_numba(
        hands,
        hand_sizes,
        points,
        table_cards,
        table_size,
        deck_size,
        current_turn,
        trump_card,
        policy_seat,
        seen_cards,
        feature_dim,
    )

    hidden = np.empty(hidden_dim, dtype=np.float32)
    for h_idx in range(hidden_dim):
        value = b1[h_idx]
        for f_idx in range(feature_dim):
            value += features[f_idx] * w1[f_idx, h_idx]
        hidden[h_idx] = value if value > 0.0 else 0.0

    logits = np.empty(ACTION_DIM, dtype=np.float32)
    max_logit = -1.0e30
    last_valid = 0
    for action_id in range(ACTION_DIM):
        if not action_mask[action_id]:
            logits[action_id] = -1.0e30
            continue
        value = b2[action_id]
        for h_idx in range(hidden_dim):
            value += hidden[h_idx] * w2[h_idx, action_id]
        logits[action_id] = value
        last_valid = action_id
        if value > max_logit:
            max_logit = value

    total = 0.0
    for action_id in range(ACTION_DIM):
        if action_mask[action_id]:
            total += np.exp(float(logits[action_id] - max_logit))

    threshold = np.random.random() * total
    cumulative = 0.0
    for action_id in range(ACTION_DIM):
        if not action_mask[action_id]:
            continue
        cumulative += np.exp(float(logits[action_id] - max_logit))
        if cumulative >= threshold:
            return action_id
    return last_valid


@njit(cache=True)
def _play_mlp_policy_game_numba(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    opponent_code: int,
    seed: int,
    policy_seat: int,
) -> tuple[int, int, int]:
    """
    Gioca una partita full-JIT: MLP policy vs opponent fast-compatible.

    Ritorna `(policy_points, opponent_points, winner)` con `winner=0` per policy,
    `winner=1` per opponent e `winner=-1` in caso di pareggio.
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
    seen_cards = np.zeros(ACTION_DIM, dtype=np.int64)
    seen_cards[trump_card] = 1

    safety = 5000
    while safety > 0:
        safety -= 1
        if hand_sizes[0] == 0 and hand_sizes[1] == 0:
            break

        if current_turn == policy_seat:
            action_id = _sample_mlp_policy_action_numba(
                w1,
                b1,
                w2,
                b2,
                hands,
                hand_sizes,
                points,
                table_cards,
                table_size,
                deck_size,
                current_turn,
                trump_card,
                policy_seat,
                seen_cards,
            )
            card_index = _action_id_to_hand_index_numba(hands, hand_sizes, current_turn, action_id)
        else:
            card_index = _choose_policy_card_index_numba(
                opponent_code,
                hands,
                hand_sizes,
                current_turn,
                table_cards,
                table_players,
                table_size,
                deck_size,
                trump_card,
                seen_cards,
            )

        played_card = hands[current_turn, card_index]
        hand_size = hand_sizes[current_turn]
        for i in range(card_index, hand_size - 1):
            hands[current_turn, i] = hands[current_turn, i + 1]
        hand_sizes[current_turn] -= 1

        table_cards[table_size] = played_card
        table_players[table_size] = current_turn
        table_size += 1
        seen_cards[played_card] = 1

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

    policy_points = points[policy_seat]
    opponent_points = points[1 - policy_seat]
    if policy_points > opponent_points:
        winner_out = 0
    elif opponent_points > policy_points:
        winner_out = 1
    else:
        winner_out = -1
    return int(policy_points), int(opponent_points), int(winner_out)


@njit(cache=True)
def _evaluate_mlp_policy_numba(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    opponent_code: int,
    num_games: int,
    seed: int,
    seat_fair: bool,
) -> tuple[int, int, int, int, int]:
    """Valuta una MLP policy contro un opponent JIT e aggrega i risultati."""
    wins_policy = 0
    wins_opponent = 0
    draws = 0
    sum_policy = 0
    sum_opponent = 0

    for game_index in range(num_games):
        policy_seat = game_index % 2 if seat_fair else 0
        policy_points, opponent_points, winner = _play_mlp_policy_game_numba(
            w1, b1, w2, b2, opponent_code, seed + game_index, policy_seat
        )
        sum_policy += policy_points
        sum_opponent += opponent_points
        if winner == 0:
            wins_policy += 1
        elif winner == 1:
            wins_opponent += 1
        else:
            draws += 1

    return wins_policy, wins_opponent, draws, sum_policy, sum_opponent


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


def _as_float32_matrix(name: str, value: np.ndarray) -> np.ndarray:
    """Normalizza un peso 2D a float32 e fallisce con errore leggibile se la shape è sbagliata."""
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} deve essere una matrice 2D, ottenuto shape={arr.shape}")
    return np.ascontiguousarray(arr)


def _as_float32_vector(name: str, value: np.ndarray) -> np.ndarray:
    """Normalizza un bias 1D a float32 e fallisce con errore leggibile se la shape è sbagliata."""
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"{name} deve essere un vettore 1D, ottenuto shape={arr.shape}")
    return np.ascontiguousarray(arr)


def evaluate_mlp_policy_numba_2p(
    *,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    opponent_name: str,
    num_games: int,
    seed: int,
    seat_fair: bool = False,
    policy_name: str = "mlp_numba",
) -> NumbaMLPRolloutSummary:
    """
    Valuta una policy MLP con rollout full-JIT contro un opponent fast-compatible.

    Questo è ancora inference/evaluation: non raccoglie `StepRecord` e non aggiorna i pesi.
    Serve a validare il percorso completo stato numerico -> encoder -> MLP -> azione -> step.
    """
    num_games = int(num_games)
    if num_games < 0:
        raise ValueError("num_games deve essere >= 0")

    w1_arr = _as_float32_matrix("w1", w1)
    b1_arr = _as_float32_vector("b1", b1)
    w2_arr = _as_float32_matrix("w2", w2)
    b2_arr = _as_float32_vector("b2", b2)
    feature_dim = int(w1_arr.shape[0])
    hidden_dim = int(w1_arr.shape[1])
    if feature_dim not in (int(FEATURE_DIM_2P_V1), int(FEATURE_DIM_2P_V2)):
        raise ValueError(f"w1 feature_dim={feature_dim}; atteso {int(FEATURE_DIM_2P_V1)} o {int(FEATURE_DIM_2P_V2)}")
    if b1_arr.shape != (hidden_dim,):
        raise ValueError(f"b1 shape={b1_arr.shape}; atteso {(hidden_dim,)}")
    if w2_arr.shape != (hidden_dim, ACTION_DIM):
        raise ValueError(f"w2 shape={w2_arr.shape}; atteso {(hidden_dim, ACTION_DIM)}")
    if b2_arr.shape != (ACTION_DIM,):
        raise ValueError(f"b2 shape={b2_arr.shape}; atteso {(ACTION_DIM,)}")

    wins_policy, wins_opponent, draws, sum_policy, sum_opponent = _evaluate_mlp_policy_numba(
        w1_arr,
        b1_arr,
        w2_arr,
        b2_arr,
        numba_agent_code(opponent_name),
        num_games,
        int(seed),
        bool(seat_fair),
    )
    return NumbaMLPRolloutSummary(
        num_games=num_games,
        policy_name=policy_name,
        opponent_name=opponent_name,
        wins_policy=int(wins_policy),
        wins_opponent=int(wins_opponent),
        draws=int(draws),
        sum_policy=int(sum_policy),
        sum_opponent=int(sum_opponent),
    )


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


def warm_up_numba_mlp_rollout() -> None:
    """Compila il rollout MLP full-JIT con un modello minimale."""
    w1 = np.zeros((int(FEATURE_DIM_2P_V1), 4), dtype=np.float32)
    b1 = np.zeros((4,), dtype=np.float32)
    w2 = np.zeros((4, ACTION_DIM), dtype=np.float32)
    b2 = np.zeros((ACTION_DIM,), dtype=np.float32)
    _evaluate_mlp_policy_numba(w1, b1, w2, b2, numba_agent_code("random"), 1, 0, False)
