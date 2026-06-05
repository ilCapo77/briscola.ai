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
from typing import Sequence

import numpy as np
from numba import njit, prange

from .evaluation import MatchStats, SeatFairStats
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

    def to_seat_fair_stats(self) -> SeatFairStats:
        """
        Converte il summary nel DTO seat-fair standard.

        Il core Numba aggrega sempre i risultati dal punto di vista della policy, indipendentemente dal seat
        con cui gioca. Per questo `policy` corrisponde ad Agent A e `opponent` ad Agent B.
        """
        return SeatFairStats(
            num_games=self.num_games,
            agent_a_name=self.policy_name,
            agent_b_name=self.opponent_name,
            wins_agent_a=self.wins_policy,
            wins_agent_b=self.wins_opponent,
            draws=self.draws,
            avg_points_agent_a=self.sum_policy / self.num_games if self.num_games else 0.0,
            avg_points_agent_b=self.sum_opponent / self.num_games if self.num_games else 0.0,
            avg_point_diff_agent_a_minus_agent_b=(
                (self.sum_policy - self.sum_opponent) / self.num_games if self.num_games else 0.0
            ),
        )


@dataclass(frozen=True, slots=True)
class NumbaDecisionQualitySummary:
    """Risultato aggregato Numba per match seat-fair + metriche decision-quality della policy."""

    num_games: int
    policy_name: str
    opponent_name: str
    wins_policy: int
    wins_opponent: int
    draws: int
    sum_policy: int
    sum_opponent: int
    num_second_hand_decisions: int
    num_second_hand_with_winning_reply: int
    num_trump_waste: int
    num_second_hand_trump_wins: int
    num_trump_overkill: int
    num_second_hand_trump_wins_low_lead_points: int
    num_trump_overkill_low_lead_points: int

    def to_seat_fair_stats(self) -> SeatFairStats:
        """Converte match aggregato nel DTO seat-fair standard."""
        return SeatFairStats(
            num_games=self.num_games,
            agent_a_name=self.policy_name,
            agent_b_name=self.opponent_name,
            wins_agent_a=self.wins_policy,
            wins_agent_b=self.wins_opponent,
            draws=self.draws,
            avg_points_agent_a=self.sum_policy / self.num_games if self.num_games else 0.0,
            avg_points_agent_b=self.sum_opponent / self.num_games if self.num_games else 0.0,
            avg_point_diff_agent_a_minus_agent_b=(
                (self.sum_policy - self.sum_opponent) / self.num_games if self.num_games else 0.0
            ),
        )


@dataclass(frozen=True, slots=True)
class NumbaA2CTrajectory:
    """Traiettoria A2C raccolta da una singola partita full-JIT."""

    policy_points: int
    opponent_points: int
    winner: int
    avg_entropy: float
    xs: np.ndarray
    z1s: np.ndarray
    hs: np.ndarray
    action_masks: np.ndarray
    probs: np.ndarray
    action_ids: np.ndarray
    value_preds: np.ndarray
    rewards: np.ndarray


@dataclass(frozen=True, slots=True)
class NumbaA2CBatch:
    """Batch di traiettorie A2C raccolte da Numba senza wrapper Python per partita."""

    policy_points: np.ndarray
    opponent_points: np.ndarray
    winners: np.ndarray
    step_counts: np.ndarray
    avg_entropies: np.ndarray
    xs: np.ndarray
    z1s: np.ndarray
    hs: np.ndarray
    action_masks: np.ndarray
    probs: np.ndarray
    action_ids: np.ndarray
    value_preds: np.ndarray
    rewards: np.ndarray


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
def _trump_overkill_penalty_numba(
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    table_cards: np.ndarray,
    table_players: np.ndarray,
    table_size: int,
    trump_card: int,
    player_index: int,
    chosen_card_index: int,
    beta: float,
    low_lead_points_max: int,
    mode_code: int,
) -> float:
    """Penalità anti-overkill equivalente al reward shaping canonico, ma su card id."""
    if beta <= 0.0:
        return 0.0
    if table_size != 1:
        return 0.0
    if chosen_card_index < 0 or chosen_card_index >= hand_sizes[player_index]:
        return 0.0

    lead_card = table_cards[0]
    lead_player = table_players[0]
    if low_lead_points_max >= 0 and CARD_POINTS_NUMBA[lead_card] > low_lead_points_max:
        return 0.0

    trump_suit = CARD_SUIT_NUMBA[trump_card]
    chosen = hands[player_index, chosen_card_index]
    if CARD_SUIT_NUMBA[chosen] != trump_suit:
        return 0.0
    if _who_wins_trick_numba(lead_card, lead_player, chosen, player_index, trump_card) != player_index:
        return 0.0

    min_points = 999
    min_strength = 999
    winning_trump_exists = False
    for hand_idx in range(hand_sizes[player_index]):
        card = hands[player_index, hand_idx]
        if CARD_SUIT_NUMBA[card] != trump_suit:
            continue
        if _who_wins_trick_numba(lead_card, lead_player, card, player_index, trump_card) != player_index:
            continue
        points = CARD_POINTS_NUMBA[card]
        strength = CARD_STRENGTH_NUMBA[card]
        if not winning_trump_exists or points < min_points or (points == min_points and strength < min_strength):
            winning_trump_exists = True
            min_points = points
            min_strength = strength

    if not winning_trump_exists:
        return 0.0

    chosen_points = CARD_POINTS_NUMBA[chosen]
    chosen_strength = CARD_STRENGTH_NUMBA[chosen]
    is_overkill = chosen_points > min_points or (chosen_points == min_points and chosen_strength > min_strength)
    if not is_overkill:
        return 0.0
    if mode_code == 0:
        return -float(beta)

    points_gap = chosen_points - min_points
    if points_gap < 0:
        points_gap = 0
    strength_gap = chosen_strength - min_strength
    if strength_gap < 0:
        strength_gap = 0
    gap = float(points_gap) / 11.0 + float(strength_gap) / 10.0
    return -float(beta) * gap


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
def _record_mlp_policy_decision_numba(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    wv: np.ndarray,
    bv: float,
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
    xs: np.ndarray,
    z1s: np.ndarray,
    hs: np.ndarray,
    masks: np.ndarray,
    probs_out: np.ndarray,
    step_index: int,
) -> tuple[int, float, float]:
    """
    Registra uno step A2C completo e ritorna `(action_id, value_pred, entropy)`.

    Le matrici di output sono mutate in-place alla riga `step_index`.
    """
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

    for f_idx in range(feature_dim):
        xs[step_index, f_idx] = features[f_idx]
    for action_id in range(ACTION_DIM):
        masks[step_index, action_id] = action_mask[action_id]

    hidden = np.empty(hidden_dim, dtype=np.float32)
    for h_idx in range(hidden_dim):
        z_value = b1[h_idx]
        for f_idx in range(feature_dim):
            z_value += features[f_idx] * w1[f_idx, h_idx]
        z1s[step_index, h_idx] = z_value
        h_value = z_value if z_value > 0.0 else 0.0
        hidden[h_idx] = h_value
        hs[step_index, h_idx] = h_value

    value_pred = float(bv)
    for h_idx in range(hidden_dim):
        value_pred += float(hidden[h_idx] * wv[h_idx])

    logits = np.empty(ACTION_DIM, dtype=np.float32)
    max_logit = -1.0e30
    last_valid = 0
    for action_id in range(ACTION_DIM):
        if not action_mask[action_id]:
            logits[action_id] = -1.0e30
            continue
        logit = b2[action_id]
        for h_idx in range(hidden_dim):
            logit += hidden[h_idx] * w2[h_idx, action_id]
        logits[action_id] = logit
        last_valid = action_id
        if logit > max_logit:
            max_logit = logit

    total = 0.0
    for action_id in range(ACTION_DIM):
        if action_mask[action_id]:
            total += np.exp(float(logits[action_id] - max_logit))

    threshold = np.random.random() * total
    cumulative = 0.0
    selected_action = last_valid
    entropy = 0.0
    for action_id in range(ACTION_DIM):
        if not action_mask[action_id]:
            probs_out[step_index, action_id] = 0.0
            continue
        prob = float(np.exp(float(logits[action_id] - max_logit)) / total)
        probs_out[step_index, action_id] = prob
        entropy -= prob * np.log(prob + 1.0e-12)
        cumulative += np.exp(float(logits[action_id] - max_logit))
        if selected_action == last_valid and cumulative >= threshold:
            selected_action = action_id

    return int(selected_action), float(value_pred), float(entropy)


@njit(cache=True)
def _argmax_mlp_policy_action_numba(
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
    player_index: int,
    seen_cards: np.ndarray,
) -> int:
    """Forward MLP deterministico: ritorna l'action_id valido con logit massimo."""
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
        player_index,
        seen_cards,
        feature_dim,
    )

    hidden = np.empty(hidden_dim, dtype=np.float32)
    for h_idx in range(hidden_dim):
        value = b1[h_idx]
        for f_idx in range(feature_dim):
            value += features[f_idx] * w1[f_idx, h_idx]
        hidden[h_idx] = value if value > 0.0 else 0.0

    best_action = 0
    best_logit = -1.0e30
    for action_id in range(ACTION_DIM):
        if not action_mask[action_id]:
            continue
        logit = b2[action_id]
        for h_idx in range(hidden_dim):
            logit += hidden[h_idx] * w2[h_idx, action_id]
        if logit > best_logit:
            best_logit = logit
            best_action = action_id

    return int(best_action)


@njit(cache=True)
def _apply_overkill_guard_numba(
    action_id: int,
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    player_index: int,
    table_cards: np.ndarray,
    table_players: np.ndarray,
    table_size: int,
    trump_card: int,
    enabled: bool,
) -> int:
    """Post-processing anti-overkill numerico equivalente al guard del `BCModelAgent`."""
    if not enabled:
        return action_id
    if table_size != 1:
        return action_id

    trump_suit = CARD_SUIT_NUMBA[trump_card]
    if CARD_SUIT_NUMBA[action_id] != trump_suit:
        return action_id

    lead_card = table_cards[0]
    lead_player = table_players[0]
    if _who_wins_trick_numba(lead_card, lead_player, action_id, player_index, trump_card) != player_index:
        return action_id

    best_action = action_id
    best_points = CARD_POINTS_NUMBA[action_id]
    best_strength = CARD_STRENGTH_NUMBA[action_id]
    for i in range(hand_sizes[player_index]):
        card = hands[player_index, i]
        if CARD_SUIT_NUMBA[card] != trump_suit:
            continue
        if _who_wins_trick_numba(lead_card, lead_player, card, player_index, trump_card) != player_index:
            continue
        points = CARD_POINTS_NUMBA[card]
        strength = CARD_STRENGTH_NUMBA[card]
        if points < best_points or (points == best_points and strength < best_strength):
            best_action = card
            best_points = points
            best_strength = strength

    return int(best_action)


@njit(cache=True)
def _trump_waste_metric_numba(
    action_id: int,
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    player_index: int,
    table_cards: np.ndarray,
    table_players: np.ndarray,
    table_size: int,
    trump_card: int,
) -> int:
    """
    Metric `trump_waste` numerica.

    Ritorna:
    - `-1` se la metrica non è applicabile (nessuna risposta vincente);
    - `0` se applicabile ma non waste;
    - `1` se la policy usa briscola pur avendo una risposta vincente non-briscola.
    """
    if table_size != 1:
        return -1

    lead_card = table_cards[0]
    lead_player = table_players[0]
    trump_suit = CARD_SUIT_NUMBA[trump_card]
    winning_any_exists = False
    winning_non_trump_exists = False
    for i in range(hand_sizes[player_index]):
        card = hands[player_index, i]
        if _who_wins_trick_numba(lead_card, lead_player, card, player_index, trump_card) != player_index:
            continue
        winning_any_exists = True
        if CARD_SUIT_NUMBA[card] != trump_suit:
            winning_non_trump_exists = True

    if not winning_any_exists:
        return -1

    chosen_is_trump = CARD_SUIT_NUMBA[action_id] == trump_suit
    if chosen_is_trump and winning_non_trump_exists:
        return 1
    return 0


@njit(cache=True)
def _trump_overkill_metric_numba(
    action_id: int,
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    player_index: int,
    table_cards: np.ndarray,
    table_players: np.ndarray,
    table_size: int,
    trump_card: int,
) -> int:
    """
    Metric `trump_overkill` numerica.

    Ritorna:
    - `-1` se la scelta non è una briscola vincente;
    - `0` se è la briscola vincente minima;
    - `1` se esiste una briscola vincente più economica.
    """
    if table_size != 1:
        return -1

    trump_suit = CARD_SUIT_NUMBA[trump_card]
    if CARD_SUIT_NUMBA[action_id] != trump_suit:
        return -1

    lead_card = table_cards[0]
    lead_player = table_players[0]
    if _who_wins_trick_numba(lead_card, lead_player, action_id, player_index, trump_card) != player_index:
        return -1

    best_points = CARD_POINTS_NUMBA[action_id]
    best_strength = CARD_STRENGTH_NUMBA[action_id]
    for i in range(hand_sizes[player_index]):
        card = hands[player_index, i]
        if CARD_SUIT_NUMBA[card] != trump_suit:
            continue
        if _who_wins_trick_numba(lead_card, lead_player, card, player_index, trump_card) != player_index:
            continue
        points = CARD_POINTS_NUMBA[card]
        strength = CARD_STRENGTH_NUMBA[card]
        if points < best_points or (points == best_points and strength < best_strength):
            return 1

    return 0


@njit(cache=True)
def _choose_opponent_card_index_numba(
    opponent_code: int,
    opponent_model_enabled: bool,
    opponent_w1: np.ndarray,
    opponent_b1: np.ndarray,
    opponent_w2: np.ndarray,
    opponent_b2: np.ndarray,
    opponent_overkill_guard: bool,
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    points: np.ndarray,
    table_cards: np.ndarray,
    table_players: np.ndarray,
    table_size: int,
    deck_size: int,
    current_turn: int,
    trump_card: int,
    seen_cards: np.ndarray,
) -> int:
    """Sceglie l'indice carta per opponent rule-based o MLP `.npz`."""
    if opponent_model_enabled:
        action_id = _argmax_mlp_policy_action_numba(
            opponent_w1,
            opponent_b1,
            opponent_w2,
            opponent_b2,
            hands,
            hand_sizes,
            points,
            table_cards,
            table_size,
            deck_size,
            current_turn,
            trump_card,
            current_turn,
            seen_cards,
        )
        action_id = _apply_overkill_guard_numba(
            action_id,
            hands,
            hand_sizes,
            current_turn,
            table_cards,
            table_players,
            table_size,
            trump_card,
            opponent_overkill_guard,
        )
        return _action_id_to_hand_index_numba(hands, hand_sizes, current_turn, action_id)

    return _choose_policy_card_index_numba(
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


@njit(cache=True)
def _play_mlp_policy_game_numba(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    opponent_code: int,
    opponent_model_enabled: bool,
    opponent_w1: np.ndarray,
    opponent_b1: np.ndarray,
    opponent_w2: np.ndarray,
    opponent_b2: np.ndarray,
    opponent_overkill_guard: bool,
    seed: int,
    policy_seat: int,
    policy_argmax: bool,
    policy_overkill_guard: bool,
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
            if policy_argmax:
                action_id = _argmax_mlp_policy_action_numba(
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
            else:
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
            action_id = _apply_overkill_guard_numba(
                action_id,
                hands,
                hand_sizes,
                current_turn,
                table_cards,
                table_players,
                table_size,
                trump_card,
                policy_overkill_guard,
            )
            card_index = _action_id_to_hand_index_numba(hands, hand_sizes, current_turn, action_id)
        else:
            card_index = _choose_opponent_card_index_numba(
                opponent_code,
                opponent_model_enabled,
                opponent_w1,
                opponent_b1,
                opponent_w2,
                opponent_b2,
                opponent_overkill_guard,
                hands,
                hand_sizes,
                points,
                table_cards,
                table_players,
                table_size,
                deck_size,
                current_turn,
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
    opponent_model_enabled: bool,
    opponent_w1: np.ndarray,
    opponent_b1: np.ndarray,
    opponent_w2: np.ndarray,
    opponent_b2: np.ndarray,
    opponent_overkill_guard: bool,
    game_seeds: np.ndarray,
    seat_fair: bool,
    policy_argmax: bool,
    policy_overkill_guard: bool,
) -> tuple[int, int, int, int, int]:
    """Valuta una MLP policy contro un opponent JIT e aggrega i risultati."""
    wins_policy = 0
    wins_opponent = 0
    draws = 0
    sum_policy = 0
    sum_opponent = 0

    if seat_fair:
        for seed_index in range(game_seeds.shape[0]):
            game_seed = int(game_seeds[seed_index])
            for policy_seat in range(2):
                policy_points, opponent_points, winner = _play_mlp_policy_game_numba(
                    w1,
                    b1,
                    w2,
                    b2,
                    opponent_code,
                    opponent_model_enabled,
                    opponent_w1,
                    opponent_b1,
                    opponent_w2,
                    opponent_b2,
                    opponent_overkill_guard,
                    game_seed,
                    policy_seat,
                    policy_argmax,
                    policy_overkill_guard,
                )
                sum_policy += policy_points
                sum_opponent += opponent_points
                if winner == 0:
                    wins_policy += 1
                elif winner == 1:
                    wins_opponent += 1
                else:
                    draws += 1
    else:
        for seed_index in range(game_seeds.shape[0]):
            policy_points, opponent_points, winner = _play_mlp_policy_game_numba(
                w1,
                b1,
                w2,
                b2,
                opponent_code,
                opponent_model_enabled,
                opponent_w1,
                opponent_b1,
                opponent_w2,
                opponent_b2,
                opponent_overkill_guard,
                int(game_seeds[seed_index]),
                0,
                policy_argmax,
                policy_overkill_guard,
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


@njit(cache=True)
def _play_mlp_policy_quality_game_numba(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    opponent_code: int,
    seed: int,
    policy_seat: int,
    policy_overkill_guard: bool,
) -> tuple[int, int, int, int, int, int, int, int, int, int]:
    """
    Gioca una partita MLP-vs-baseline e raccoglie metriche qualità per la policy.

    Ritorna match result + contatori quality, tutti dal punto di vista della policy.
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

    q_second = 0
    q_second_with_win = 0
    q_waste = 0
    q_trump_wins = 0
    q_trump_overkill = 0
    q_trump_wins_low = 0
    q_trump_overkill_low = 0

    safety = 5000
    while safety > 0:
        safety -= 1
        if hand_sizes[0] == 0 and hand_sizes[1] == 0:
            break

        if current_turn == policy_seat:
            action_id = _argmax_mlp_policy_action_numba(
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
            action_id = _apply_overkill_guard_numba(
                action_id,
                hands,
                hand_sizes,
                current_turn,
                table_cards,
                table_players,
                table_size,
                trump_card,
                policy_overkill_guard,
            )

            if table_size == 1:
                q_second += 1
                waste = _trump_waste_metric_numba(
                    action_id,
                    hands,
                    hand_sizes,
                    current_turn,
                    table_cards,
                    table_players,
                    table_size,
                    trump_card,
                )
                if waste >= 0:
                    q_second_with_win += 1
                    if waste == 1:
                        q_waste += 1

                overkill = _trump_overkill_metric_numba(
                    action_id,
                    hands,
                    hand_sizes,
                    current_turn,
                    table_cards,
                    table_players,
                    table_size,
                    trump_card,
                )
                if overkill >= 0:
                    q_trump_wins += 1
                    low_lead = CARD_POINTS_NUMBA[table_cards[0]] <= 2
                    if low_lead:
                        q_trump_wins_low += 1
                    if overkill == 1:
                        q_trump_overkill += 1
                        if low_lead:
                            q_trump_overkill_low += 1

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
    return (
        int(policy_points),
        int(opponent_points),
        int(winner_out),
        q_second,
        q_second_with_win,
        q_waste,
        q_trump_wins,
        q_trump_overkill,
        q_trump_wins_low,
        q_trump_overkill_low,
    )


@njit(cache=True)
def _evaluate_mlp_policy_quality_numba(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    opponent_code: int,
    game_seeds: np.ndarray,
    policy_overkill_guard: bool,
) -> tuple[int, int, int, int, int, int, int, int, int, int, int, int]:
    """Valuta seat-fair MLP-vs-baseline e aggrega match stats + quality stats."""
    wins_policy = 0
    wins_opponent = 0
    draws = 0
    sum_policy = 0
    sum_opponent = 0
    q_second = 0
    q_second_with_win = 0
    q_waste = 0
    q_trump_wins = 0
    q_trump_overkill = 0
    q_trump_wins_low = 0
    q_trump_overkill_low = 0

    for seed_index in range(game_seeds.shape[0]):
        game_seed = int(game_seeds[seed_index])
        for policy_seat in range(2):
            (
                policy_points,
                opponent_points,
                winner,
                second,
                second_with_win,
                waste,
                trump_wins,
                trump_overkill,
                trump_wins_low,
                trump_overkill_low,
            ) = _play_mlp_policy_quality_game_numba(
                w1,
                b1,
                w2,
                b2,
                opponent_code,
                game_seed,
                policy_seat,
                policy_overkill_guard,
            )
            sum_policy += policy_points
            sum_opponent += opponent_points
            if winner == 0:
                wins_policy += 1
            elif winner == 1:
                wins_opponent += 1
            else:
                draws += 1

            q_second += second
            q_second_with_win += second_with_win
            q_waste += waste
            q_trump_wins += trump_wins
            q_trump_overkill += trump_overkill
            q_trump_wins_low += trump_wins_low
            q_trump_overkill_low += trump_overkill_low

    return (
        wins_policy,
        wins_opponent,
        draws,
        sum_policy,
        sum_opponent,
        q_second,
        q_second_with_win,
        q_waste,
        q_trump_wins,
        q_trump_overkill,
        q_trump_wins_low,
        q_trump_overkill_low,
    )


@njit(cache=True)
def _apply_numba_card_index(
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
    card_index: int,
    seen_cards: np.ndarray,
) -> tuple[int, int, int]:
    """Applica una giocata mutando gli array e ritorna `(deck_size, table_size, current_turn)`."""
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
        return deck_size, table_size, 1 - current_turn

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

    return deck_size, table_size, winner


@njit(cache=True)
def _collect_mlp_policy_game_into_numba(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    wv: np.ndarray,
    bv: float,
    opponent_code: int,
    opponent_model_enabled: bool,
    opponent_w1: np.ndarray,
    opponent_b1: np.ndarray,
    opponent_w2: np.ndarray,
    opponent_b2: np.ndarray,
    opponent_overkill_guard: bool,
    overkill_penalty_beta: float,
    overkill_low_lead_points_max: int,
    overkill_penalty_mode_code: int,
    seed: int,
    policy_seat: int,
    xs: np.ndarray,
    z1s: np.ndarray,
    hs: np.ndarray,
    masks: np.ndarray,
    probs: np.ndarray,
    action_ids: np.ndarray,
    value_preds: np.ndarray,
    rewards: np.ndarray,
) -> tuple[int, int, int, int, float]:
    """
    Raccoglie una traiettoria A2C full-JIT per una singola partita.

    Ritorna punti, vincitore, step_count, entropia media e buffer di traiettoria.
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

    step_count = 0
    entropy_sum = 0.0
    safety = 5000
    while safety > 0:
        safety -= 1

        while not (hand_sizes[0] == 0 and hand_sizes[1] == 0) and current_turn != policy_seat:
            opp_card_index = _choose_opponent_card_index_numba(
                opponent_code,
                opponent_model_enabled,
                opponent_w1,
                opponent_b1,
                opponent_w2,
                opponent_b2,
                opponent_overkill_guard,
                hands,
                hand_sizes,
                points,
                table_cards,
                table_players,
                table_size,
                deck_size,
                current_turn,
                trump_card,
                seen_cards,
            )
            deck_size, table_size, current_turn = _apply_numba_card_index(
                hands,
                hand_sizes,
                points,
                deck,
                deck_size,
                table_cards,
                table_players,
                table_size,
                current_turn,
                trump_card,
                opp_card_index,
                seen_cards,
            )

        if hand_sizes[0] == 0 and hand_sizes[1] == 0:
            break

        diff_before = points[policy_seat] - points[1 - policy_seat]
        action_id, value_pred, entropy = _record_mlp_policy_decision_numba(
            w1,
            b1,
            w2,
            b2,
            wv,
            bv,
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
            xs,
            z1s,
            hs,
            masks,
            probs,
            step_count,
        )
        policy_card_index = _action_id_to_hand_index_numba(hands, hand_sizes, current_turn, action_id)
        action_ids[step_count] = action_id
        value_preds[step_count] = value_pred
        entropy_sum += entropy
        extra_penalty = _trump_overkill_penalty_numba(
            hands,
            hand_sizes,
            table_cards,
            table_players,
            table_size,
            trump_card,
            policy_seat,
            policy_card_index,
            overkill_penalty_beta,
            overkill_low_lead_points_max,
            overkill_penalty_mode_code,
        )

        deck_size, table_size, current_turn = _apply_numba_card_index(
            hands,
            hand_sizes,
            points,
            deck,
            deck_size,
            table_cards,
            table_players,
            table_size,
            current_turn,
            trump_card,
            policy_card_index,
            seen_cards,
        )

        while not (hand_sizes[0] == 0 and hand_sizes[1] == 0) and current_turn != policy_seat:
            opp_card_index = _choose_opponent_card_index_numba(
                opponent_code,
                opponent_model_enabled,
                opponent_w1,
                opponent_b1,
                opponent_w2,
                opponent_b2,
                opponent_overkill_guard,
                hands,
                hand_sizes,
                points,
                table_cards,
                table_players,
                table_size,
                deck_size,
                current_turn,
                trump_card,
                seen_cards,
            )
            deck_size, table_size, current_turn = _apply_numba_card_index(
                hands,
                hand_sizes,
                points,
                deck,
                deck_size,
                table_cards,
                table_players,
                table_size,
                current_turn,
                trump_card,
                opp_card_index,
                seen_cards,
            )

        diff_after = points[policy_seat] - points[1 - policy_seat]
        rewards[step_count] = float(diff_after - diff_before) / 120.0 + extra_penalty
        step_count += 1

    policy_points = points[policy_seat]
    opponent_points = points[1 - policy_seat]
    if policy_points > opponent_points:
        winner_out = 0
    elif opponent_points > policy_points:
        winner_out = 1
    else:
        winner_out = -1
    avg_entropy = entropy_sum / float(step_count) if step_count > 0 else 0.0
    return (
        int(policy_points),
        int(opponent_points),
        int(winner_out),
        int(step_count),
        float(avg_entropy),
    )


@njit(cache=True)
def _collect_mlp_policy_game_numba(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    wv: np.ndarray,
    bv: float,
    opponent_code: int,
    opponent_model_enabled: bool,
    opponent_w1: np.ndarray,
    opponent_b1: np.ndarray,
    opponent_w2: np.ndarray,
    opponent_b2: np.ndarray,
    opponent_overkill_guard: bool,
    overkill_penalty_beta: float,
    overkill_low_lead_points_max: int,
    overkill_penalty_mode_code: int,
    seed: int,
    policy_seat: int,
) -> tuple[
    int,
    int,
    int,
    int,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Wrapper JIT compatibile: alloca i buffer e raccoglie una singola partita."""
    max_steps = 20
    feature_dim = w1.shape[0]
    hidden_dim = w1.shape[1]
    xs = np.zeros((max_steps, feature_dim), dtype=np.float32)
    z1s = np.zeros((max_steps, hidden_dim), dtype=np.float32)
    hs = np.zeros((max_steps, hidden_dim), dtype=np.float32)
    masks = np.zeros((max_steps, ACTION_DIM), dtype=np.bool_)
    probs = np.zeros((max_steps, ACTION_DIM), dtype=np.float32)
    action_ids = np.full(max_steps, -1, dtype=np.int64)
    value_preds = np.zeros(max_steps, dtype=np.float32)
    rewards = np.zeros(max_steps, dtype=np.float32)

    policy_points, opponent_points, winner, step_count, avg_entropy = _collect_mlp_policy_game_into_numba(
        w1,
        b1,
        w2,
        b2,
        wv,
        bv,
        opponent_code,
        opponent_model_enabled,
        opponent_w1,
        opponent_b1,
        opponent_w2,
        opponent_b2,
        opponent_overkill_guard,
        overkill_penalty_beta,
        overkill_low_lead_points_max,
        overkill_penalty_mode_code,
        seed,
        policy_seat,
        xs,
        z1s,
        hs,
        masks,
        probs,
        action_ids,
        value_preds,
        rewards,
    )
    return (
        int(policy_points),
        int(opponent_points),
        int(winner),
        int(step_count),
        float(avg_entropy),
        xs,
        z1s,
        hs,
        masks,
        probs,
        action_ids,
        value_preds,
        rewards,
    )


@njit(cache=True, parallel=True)
def _collect_mlp_policy_batch_numba(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    wv: np.ndarray,
    bv: float,
    opponent_code: int,
    opponent_model_enabled: bool,
    opponent_w1: np.ndarray,
    opponent_b1: np.ndarray,
    opponent_w2: np.ndarray,
    opponent_b2: np.ndarray,
    opponent_overkill_guard: bool,
    overkill_penalty_beta: float,
    overkill_low_lead_points_max: int,
    overkill_penalty_mode_code: int,
    opponent_model_enabled_flags: np.ndarray,
    opponent_codes: np.ndarray,
    game_seeds: np.ndarray,
    policy_seats: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Raccoglie un batch di partite A2C dentro una sola chiamata JIT."""
    batch_size = game_seeds.shape[0]
    max_steps = 20
    feature_dim = w1.shape[0]
    hidden_dim = w1.shape[1]

    policy_points = np.zeros(batch_size, dtype=np.int64)
    opponent_points = np.zeros(batch_size, dtype=np.int64)
    winners = np.zeros(batch_size, dtype=np.int64)
    step_counts = np.zeros(batch_size, dtype=np.int64)
    avg_entropies = np.zeros(batch_size, dtype=np.float32)
    xs = np.zeros((batch_size, max_steps, feature_dim), dtype=np.float32)
    z1s = np.zeros((batch_size, max_steps, hidden_dim), dtype=np.float32)
    hs = np.zeros((batch_size, max_steps, hidden_dim), dtype=np.float32)
    masks = np.zeros((batch_size, max_steps, ACTION_DIM), dtype=np.bool_)
    probs = np.zeros((batch_size, max_steps, ACTION_DIM), dtype=np.float32)
    action_ids = np.full((batch_size, max_steps), -1, dtype=np.int64)
    value_preds = np.zeros((batch_size, max_steps), dtype=np.float32)
    rewards = np.zeros((batch_size, max_steps), dtype=np.float32)

    for game_idx in prange(batch_size):
        p_points, o_points, winner, steps, avg_entropy = _collect_mlp_policy_game_into_numba(
            w1,
            b1,
            w2,
            b2,
            wv,
            bv,
            int(opponent_codes[game_idx]),
            bool(opponent_model_enabled_flags[game_idx]),
            opponent_w1,
            opponent_b1,
            opponent_w2,
            opponent_b2,
            opponent_overkill_guard,
            overkill_penalty_beta,
            overkill_low_lead_points_max,
            overkill_penalty_mode_code,
            int(game_seeds[game_idx]),
            int(policy_seats[game_idx]),
            xs[game_idx],
            z1s[game_idx],
            hs[game_idx],
            masks[game_idx],
            probs[game_idx],
            action_ids[game_idx],
            value_preds[game_idx],
            rewards[game_idx],
        )
        policy_points[game_idx] = p_points
        opponent_points[game_idx] = o_points
        winners[game_idx] = winner
        step_counts[game_idx] = steps
        avg_entropies[game_idx] = avg_entropy

    return (
        policy_points,
        opponent_points,
        winners,
        step_counts,
        avg_entropies,
        xs,
        z1s,
        hs,
        masks,
        probs,
        action_ids,
        value_preds,
        rewards,
    )


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


@dataclass(frozen=True, slots=True)
class _PreparedA2CNumbaInputs:
    """Tensori validati per i wrapper A2C Numba."""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    wv: np.ndarray
    opponent_code: int
    opponent_model_enabled: bool
    opponent_w1: np.ndarray
    opponent_b1: np.ndarray
    opponent_w2: np.ndarray
    opponent_b2: np.ndarray


def _prepare_a2c_numba_inputs(
    *,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    wv: np.ndarray,
    opponent_name: str,
    opponent_w1: np.ndarray | None,
    opponent_b1: np.ndarray | None,
    opponent_w2: np.ndarray | None,
    opponent_b2: np.ndarray | None,
) -> _PreparedA2CNumbaInputs:
    """Valida e normalizza i tensori condivisi dai wrapper A2C Numba."""
    w1_arr = _as_float32_matrix("w1", w1)
    b1_arr = _as_float32_vector("b1", b1)
    w2_arr = _as_float32_matrix("w2", w2)
    b2_arr = _as_float32_vector("b2", b2)
    wv_arr = _as_float32_vector("wv", wv)

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
    if wv_arr.shape != (hidden_dim,):
        raise ValueError(f"wv shape={wv_arr.shape}; atteso {(hidden_dim,)}")

    opponent_model_enabled = (
        opponent_w1 is not None or opponent_b1 is not None or opponent_w2 is not None or opponent_b2 is not None
    )
    if opponent_model_enabled:
        if opponent_w1 is None or opponent_b1 is None or opponent_w2 is None or opponent_b2 is None:
            raise ValueError("Opponent model incompleto: servono opponent_w1/b1/w2/b2.")
        opponent_w1_arr = _as_float32_matrix("opponent_w1", opponent_w1)
        opponent_b1_arr = _as_float32_vector("opponent_b1", opponent_b1)
        opponent_w2_arr = _as_float32_matrix("opponent_w2", opponent_w2)
        opponent_b2_arr = _as_float32_vector("opponent_b2", opponent_b2)
        opp_feature_dim = int(opponent_w1_arr.shape[0])
        opp_hidden_dim = int(opponent_w1_arr.shape[1])
        if opp_feature_dim not in (int(FEATURE_DIM_2P_V1), int(FEATURE_DIM_2P_V2)):
            raise ValueError(
                f"opponent_w1 feature_dim={opp_feature_dim}; atteso {int(FEATURE_DIM_2P_V1)} o {int(FEATURE_DIM_2P_V2)}"
            )
        if opponent_b1_arr.shape != (opp_hidden_dim,):
            raise ValueError(f"opponent_b1 shape={opponent_b1_arr.shape}; atteso {(opp_hidden_dim,)}")
        if opponent_w2_arr.shape != (opp_hidden_dim, ACTION_DIM):
            raise ValueError(f"opponent_w2 shape={opponent_w2_arr.shape}; atteso {(opp_hidden_dim, ACTION_DIM)}")
        if opponent_b2_arr.shape != (ACTION_DIM,):
            raise ValueError(f"opponent_b2 shape={opponent_b2_arr.shape}; atteso {(ACTION_DIM,)}")
        opponent_code = 0
    else:
        opponent_w1_arr = np.zeros((int(FEATURE_DIM_2P_V1), 1), dtype=np.float32)
        opponent_b1_arr = np.zeros((1,), dtype=np.float32)
        opponent_w2_arr = np.zeros((1, ACTION_DIM), dtype=np.float32)
        opponent_b2_arr = np.zeros((ACTION_DIM,), dtype=np.float32)
        opponent_code = numba_agent_code(opponent_name)

    return _PreparedA2CNumbaInputs(
        w1=w1_arr,
        b1=b1_arr,
        w2=w2_arr,
        b2=b2_arr,
        wv=wv_arr,
        opponent_code=int(opponent_code),
        opponent_model_enabled=bool(opponent_model_enabled),
        opponent_w1=opponent_w1_arr,
        opponent_b1=opponent_b1_arr,
        opponent_w2=opponent_w2_arr,
        opponent_b2=opponent_b2_arr,
    )


def _overkill_penalty_mode_code(mode: str) -> int:
    """Codifica la modalità reward shaping anti-overkill per il core JIT."""
    normalized = str(mode).strip().lower()
    if normalized == "flat":
        return 0
    if normalized == "gap":
        return 1
    raise ValueError(f"overkill_penalty_mode non supportato: {mode!r}")


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
    game_seeds: Sequence[int] | None = None,
    deterministic: bool = False,
    policy_overkill_guard: bool = False,
    opponent_w1: np.ndarray | None = None,
    opponent_b1: np.ndarray | None = None,
    opponent_w2: np.ndarray | None = None,
    opponent_b2: np.ndarray | None = None,
    opponent_overkill_guard: bool = False,
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
    if bool(seat_fair) and num_games % 2 != 0:
        raise ValueError("Per la valutazione seat-fair `num_games` deve essere pari.")

    needed_seeds = num_games // 2 if bool(seat_fair) else num_games
    if game_seeds is None:
        seeds_arr = np.asarray([((int(seed) + i) & 0xFFFFFFFF) for i in range(needed_seeds)], dtype=np.int64)
    else:
        raw_seeds = [int(s) & 0xFFFFFFFF for s in game_seeds]
        if len(raw_seeds) < needed_seeds:
            raise ValueError(f"game_seeds insufficiente: attesi >= {needed_seeds}, ottenuti {len(raw_seeds)}")
        seeds_arr = np.asarray(raw_seeds[:needed_seeds], dtype=np.int64)

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

    opponent_model_enabled = (
        opponent_w1 is not None or opponent_b1 is not None or opponent_w2 is not None or opponent_b2 is not None
    )
    if opponent_model_enabled:
        if opponent_w1 is None or opponent_b1 is None or opponent_w2 is None or opponent_b2 is None:
            raise ValueError("Opponent model incompleto: servono opponent_w1/b1/w2/b2.")
        opponent_w1_arr = _as_float32_matrix("opponent_w1", opponent_w1)
        opponent_b1_arr = _as_float32_vector("opponent_b1", opponent_b1)
        opponent_w2_arr = _as_float32_matrix("opponent_w2", opponent_w2)
        opponent_b2_arr = _as_float32_vector("opponent_b2", opponent_b2)
        opp_feature_dim = int(opponent_w1_arr.shape[0])
        opp_hidden_dim = int(opponent_w1_arr.shape[1])
        if opp_feature_dim not in (int(FEATURE_DIM_2P_V1), int(FEATURE_DIM_2P_V2)):
            raise ValueError(
                f"opponent_w1 feature_dim={opp_feature_dim}; atteso {int(FEATURE_DIM_2P_V1)} o {int(FEATURE_DIM_2P_V2)}"
            )
        if opponent_b1_arr.shape != (opp_hidden_dim,):
            raise ValueError(f"opponent_b1 shape={opponent_b1_arr.shape}; atteso {(opp_hidden_dim,)}")
        if opponent_w2_arr.shape != (opp_hidden_dim, ACTION_DIM):
            raise ValueError(f"opponent_w2 shape={opponent_w2_arr.shape}; atteso {(opp_hidden_dim, ACTION_DIM)}")
        if opponent_b2_arr.shape != (ACTION_DIM,):
            raise ValueError(f"opponent_b2 shape={opponent_b2_arr.shape}; atteso {(ACTION_DIM,)}")
        opponent_code = 0
    else:
        opponent_w1_arr = np.zeros((int(FEATURE_DIM_2P_V1), 1), dtype=np.float32)
        opponent_b1_arr = np.zeros((1,), dtype=np.float32)
        opponent_w2_arr = np.zeros((1, ACTION_DIM), dtype=np.float32)
        opponent_b2_arr = np.zeros((ACTION_DIM,), dtype=np.float32)
        opponent_code = numba_agent_code(opponent_name)

    wins_policy, wins_opponent, draws, sum_policy, sum_opponent = _evaluate_mlp_policy_numba(
        w1_arr,
        b1_arr,
        w2_arr,
        b2_arr,
        opponent_code,
        bool(opponent_model_enabled),
        opponent_w1_arr,
        opponent_b1_arr,
        opponent_w2_arr,
        opponent_b2_arr,
        bool(opponent_overkill_guard),
        seeds_arr,
        bool(seat_fair),
        bool(deterministic),
        bool(policy_overkill_guard),
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


def evaluate_mlp_policy_quality_numba_2p(
    *,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    opponent_name: str,
    num_games: int,
    seed: int,
    game_seeds: Sequence[int] | None = None,
    policy_overkill_guard: bool = False,
    policy_name: str = "mlp_numba",
) -> NumbaDecisionQualitySummary:
    """
    Valuta seat-fair una policy MLP con Numba e raccoglie metriche decision-quality.

    Questa funzione replica la semantica inference di `BCModelAgent`: argmax deterministico,
    action mask e post-processing anti-overkill opzionale.
    """
    num_games = int(num_games)
    if num_games < 0:
        raise ValueError("num_games deve essere >= 0")
    if num_games % 2 != 0:
        raise ValueError("Per la valutazione seat-fair `num_games` deve essere pari.")

    needed_seeds = num_games // 2
    if game_seeds is None:
        seeds_arr = np.asarray([((int(seed) + i) & 0xFFFFFFFF) for i in range(needed_seeds)], dtype=np.int64)
    else:
        raw_seeds = [int(s) & 0xFFFFFFFF for s in game_seeds]
        if len(raw_seeds) < needed_seeds:
            raise ValueError(f"game_seeds insufficiente: attesi >= {needed_seeds}, ottenuti {len(raw_seeds)}")
        seeds_arr = np.asarray(raw_seeds[:needed_seeds], dtype=np.int64)

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

    (
        wins_policy,
        wins_opponent,
        draws,
        sum_policy,
        sum_opponent,
        q_second,
        q_second_with_win,
        q_waste,
        q_trump_wins,
        q_trump_overkill,
        q_trump_wins_low,
        q_trump_overkill_low,
    ) = _evaluate_mlp_policy_quality_numba(
        w1_arr,
        b1_arr,
        w2_arr,
        b2_arr,
        numba_agent_code(opponent_name),
        seeds_arr,
        bool(policy_overkill_guard),
    )
    return NumbaDecisionQualitySummary(
        num_games=num_games,
        policy_name=policy_name,
        opponent_name=opponent_name,
        wins_policy=int(wins_policy),
        wins_opponent=int(wins_opponent),
        draws=int(draws),
        sum_policy=int(sum_policy),
        sum_opponent=int(sum_opponent),
        num_second_hand_decisions=int(q_second),
        num_second_hand_with_winning_reply=int(q_second_with_win),
        num_trump_waste=int(q_waste),
        num_second_hand_trump_wins=int(q_trump_wins),
        num_trump_overkill=int(q_trump_overkill),
        num_second_hand_trump_wins_low_lead_points=int(q_trump_wins_low),
        num_trump_overkill_low_lead_points=int(q_trump_overkill_low),
    )


def collect_a2c_trajectory_numba_2p(
    *,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    wv: np.ndarray,
    bv: float,
    opponent_name: str,
    game_seed: int,
    policy_seat: int,
    opponent_w1: np.ndarray | None = None,
    opponent_b1: np.ndarray | None = None,
    opponent_w2: np.ndarray | None = None,
    opponent_b2: np.ndarray | None = None,
    opponent_overkill_guard: bool = False,
    overkill_penalty_beta: float = 0.0,
    overkill_low_lead_points_max: int = 2,
    overkill_penalty_mode: str = "flat",
) -> NumbaA2CTrajectory:
    """
    Raccoglie una traiettoria A2C full-JIT per il trainer.

    Il wrapper valida i tensori e restituisce solo le righe effettivamente popolate.
    """
    if policy_seat not in (0, 1):
        raise ValueError(f"policy_seat fuori range: {policy_seat}")
    if float(overkill_penalty_beta) < 0.0:
        raise ValueError("overkill_penalty_beta deve essere >= 0")
    if int(overkill_low_lead_points_max) < 0:
        raise ValueError("overkill_low_lead_points_max deve essere >= 0")
    mode_code = _overkill_penalty_mode_code(overkill_penalty_mode)
    prepared = _prepare_a2c_numba_inputs(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        wv=wv,
        opponent_name=opponent_name,
        opponent_w1=opponent_w1,
        opponent_b1=opponent_b1,
        opponent_w2=opponent_w2,
        opponent_b2=opponent_b2,
    )

    (
        policy_points,
        opponent_points,
        winner,
        step_count,
        avg_entropy,
        xs,
        z1s,
        hs,
        masks,
        probs,
        action_ids,
        value_preds,
        rewards,
    ) = _collect_mlp_policy_game_numba(
        prepared.w1,
        prepared.b1,
        prepared.w2,
        prepared.b2,
        prepared.wv,
        float(bv),
        prepared.opponent_code,
        prepared.opponent_model_enabled,
        prepared.opponent_w1,
        prepared.opponent_b1,
        prepared.opponent_w2,
        prepared.opponent_b2,
        bool(opponent_overkill_guard),
        float(overkill_penalty_beta),
        int(overkill_low_lead_points_max),
        int(mode_code),
        int(game_seed),
        int(policy_seat),
    )
    count = int(step_count)
    return NumbaA2CTrajectory(
        policy_points=int(policy_points),
        opponent_points=int(opponent_points),
        winner=int(winner),
        avg_entropy=float(avg_entropy),
        xs=xs[:count].copy(),
        z1s=z1s[:count].copy(),
        hs=hs[:count].copy(),
        action_masks=masks[:count].copy(),
        probs=probs[:count].copy(),
        action_ids=action_ids[:count].copy(),
        value_preds=value_preds[:count].copy(),
        rewards=rewards[:count].copy(),
    )


def collect_a2c_batch_numba_2p(
    *,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    wv: np.ndarray,
    bv: float,
    opponent_name: str,
    game_seeds: np.ndarray,
    policy_seats: np.ndarray,
    opponent_w1: np.ndarray | None = None,
    opponent_b1: np.ndarray | None = None,
    opponent_w2: np.ndarray | None = None,
    opponent_b2: np.ndarray | None = None,
    opponent_overkill_guard: bool = False,
    opponent_codes: np.ndarray | None = None,
    opponent_model_enabled_flags: np.ndarray | None = None,
    overkill_penalty_beta: float = 0.0,
    overkill_low_lead_points_max: int = 2,
    overkill_penalty_mode: str = "flat",
) -> NumbaA2CBatch:
    """
    Raccoglie un batch di traiettorie A2C full-JIT per il trainer.

    A differenza di `collect_a2c_trajectory_numba_2p`, questo wrapper valida i tensori una
    volta sola e restituisce buffer `(batch, max_steps, ...)`: il trainer usa `step_counts`
    per considerare solo le righe valide.
    """
    seeds_arr = np.asarray(game_seeds, dtype=np.int64)
    seats_arr = np.asarray(policy_seats, dtype=np.int64)
    if seeds_arr.ndim != 1:
        raise ValueError(f"game_seeds deve essere 1D, ottenuto shape={seeds_arr.shape}")
    if seats_arr.ndim != 1:
        raise ValueError(f"policy_seats deve essere 1D, ottenuto shape={seats_arr.shape}")
    if seats_arr.shape != seeds_arr.shape:
        raise ValueError(f"Shape mismatch: game_seeds={seeds_arr.shape} policy_seats={seats_arr.shape}")
    if not np.all((seats_arr == 0) | (seats_arr == 1)):
        raise ValueError("policy_seats deve contenere solo 0/1")
    if float(overkill_penalty_beta) < 0.0:
        raise ValueError("overkill_penalty_beta deve essere >= 0")
    if int(overkill_low_lead_points_max) < 0:
        raise ValueError("overkill_low_lead_points_max deve essere >= 0")
    mode_code = _overkill_penalty_mode_code(overkill_penalty_mode)

    prepared = _prepare_a2c_numba_inputs(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        wv=wv,
        opponent_name=opponent_name,
        opponent_w1=opponent_w1,
        opponent_b1=opponent_b1,
        opponent_w2=opponent_w2,
        opponent_b2=opponent_b2,
    )
    if opponent_codes is None:
        codes_arr = np.full(seeds_arr.shape, prepared.opponent_code, dtype=np.int64)
    else:
        codes_arr = np.asarray(opponent_codes, dtype=np.int64)
        if codes_arr.ndim != 1:
            raise ValueError(f"opponent_codes deve essere 1D, ottenuto shape={codes_arr.shape}")
        if codes_arr.shape != seeds_arr.shape:
            raise ValueError(f"Shape mismatch: opponent_codes={codes_arr.shape} game_seeds={seeds_arr.shape}")
        if not np.all((codes_arr >= 0) & (codes_arr <= 3)):
            raise ValueError("opponent_codes contiene codici Numba non supportati")
    if opponent_model_enabled_flags is None:
        model_flags_arr = np.full(seeds_arr.shape, bool(prepared.opponent_model_enabled), dtype=np.bool_)
    else:
        model_flags_arr = np.asarray(opponent_model_enabled_flags, dtype=np.bool_)
        if model_flags_arr.ndim != 1:
            raise ValueError(f"opponent_model_enabled_flags deve essere 1D, ottenuto shape={model_flags_arr.shape}")
        if model_flags_arr.shape != seeds_arr.shape:
            raise ValueError(
                f"Shape mismatch: opponent_model_enabled_flags={model_flags_arr.shape} game_seeds={seeds_arr.shape}"
            )
        if bool(np.any(model_flags_arr)) and not prepared.opponent_model_enabled:
            raise ValueError("opponent_model_enabled_flags richiede un opponent model `.npz` caricato.")

    (
        policy_points,
        opponent_points,
        winners,
        step_counts,
        avg_entropies,
        xs,
        z1s,
        hs,
        masks,
        probs,
        action_ids,
        value_preds,
        rewards,
    ) = _collect_mlp_policy_batch_numba(
        prepared.w1,
        prepared.b1,
        prepared.w2,
        prepared.b2,
        prepared.wv,
        float(bv),
        prepared.opponent_code,
        prepared.opponent_model_enabled,
        prepared.opponent_w1,
        prepared.opponent_b1,
        prepared.opponent_w2,
        prepared.opponent_b2,
        bool(opponent_overkill_guard),
        float(overkill_penalty_beta),
        int(overkill_low_lead_points_max),
        int(mode_code),
        np.ascontiguousarray(model_flags_arr),
        np.ascontiguousarray(codes_arr),
        np.ascontiguousarray(seeds_arr),
        np.ascontiguousarray(seats_arr),
    )
    return NumbaA2CBatch(
        policy_points=policy_points,
        opponent_points=opponent_points,
        winners=winners,
        step_counts=step_counts,
        avg_entropies=avg_entropies,
        xs=xs,
        z1s=z1s,
        hs=hs,
        action_masks=masks,
        probs=probs,
        action_ids=action_ids,
        value_preds=value_preds,
        rewards=rewards,
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
    opponent_w1 = np.zeros((int(FEATURE_DIM_2P_V1), 1), dtype=np.float32)
    opponent_b1 = np.zeros((1,), dtype=np.float32)
    opponent_w2 = np.zeros((1, ACTION_DIM), dtype=np.float32)
    opponent_b2 = np.zeros((ACTION_DIM,), dtype=np.float32)
    _evaluate_mlp_policy_numba(
        w1,
        b1,
        w2,
        b2,
        numba_agent_code("random"),
        False,
        opponent_w1,
        opponent_b1,
        opponent_w2,
        opponent_b2,
        False,
        np.asarray([0], dtype=np.int64),
        False,
        False,
        False,
    )
    _evaluate_mlp_policy_quality_numba(
        w1,
        b1,
        w2,
        b2,
        numba_agent_code("random"),
        np.asarray([0], dtype=np.int64),
        False,
    )
    wv = np.zeros((4,), dtype=np.float32)
    _collect_mlp_policy_game_numba(
        w1,
        b1,
        w2,
        b2,
        wv,
        0.0,
        numba_agent_code("random"),
        False,
        opponent_w1,
        opponent_b1,
        opponent_w2,
        opponent_b2,
        False,
        0.0,
        2,
        0,
        0,
        0,
    )
    _collect_mlp_policy_batch_numba(
        w1,
        b1,
        w2,
        b2,
        wv,
        0.0,
        numba_agent_code("random"),
        False,
        opponent_w1,
        opponent_b1,
        opponent_w2,
        opponent_b2,
        False,
        0.0,
        2,
        0,
        np.asarray([False], dtype=np.bool_),
        np.asarray([numba_agent_code("random")], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
    )
