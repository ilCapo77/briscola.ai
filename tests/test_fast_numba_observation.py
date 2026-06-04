"""
Test di equivalenza per l'encoder osservazione Numba.

Il layout feature/mask deve restare identico a `encode_fast_observation_2p`: questo
ci permette di usare lo stesso modello `.npz` quando il rollout A2C passerà a stato
numerico/JIT.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from briscola_ai.ai.fast_2p import new_fast_2p_state, step_fast_2p
from briscola_ai.ai.fast_numba_observation import (
    collect_a2c_trajectory_numba_2p,
    encode_fast_observation_numba_2p,
    evaluate_mlp_policy_numba_2p,
    warm_up_numba_mlp_rollout,
    warm_up_numba_observation,
)
from briscola_ai.ai.fast_observation_encoder import encode_fast_observation_2p
from briscola_ai.ai.training.observation_encoder import FEATURE_DIM_2P_V1, FEATURE_DIM_2P_V2, EncoderVersion


def _assert_encoders_match(*, seed: int, steps: int, version: EncoderVersion) -> None:
    """Avanza una partita fast e confronta encoder Python vs Numba per entrambi i player."""
    state = new_fast_2p_state(seed=seed)
    seen = [0] * 40
    seen[state.trump_card] = 1
    rng = random.Random(seed ^ 0xA2C)

    for _ in range(steps):
        if state.game_over:
            break
        current = state.current_turn
        card_index = rng.randrange(len(state.hands[current]))
        result = step_fast_2p(state, player_index=current, card_index=card_index)
        seen[result.played_card] = 1

    for player_index in (0, 1):
        py_encoded = encode_fast_observation_2p(
            state,
            player_index=player_index,
            seen_cards_onehot=tuple(seen),
            version=version,
        )
        nb_encoded = encode_fast_observation_numba_2p(
            state,
            player_index=player_index,
            seen_cards_onehot=tuple(seen),
            version=version,
        )

        assert nb_encoded.action_mask == py_encoded.action_mask
        assert nb_encoded.features == pytest.approx(py_encoded.features)


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("steps", [0, 1, 2, 7, 20])
def test_numba_fast_observation_matches_python_encoder(version: EncoderVersion, steps: int) -> None:
    """L'encoder JIT deve essere semanticamente equivalente al path fast Python."""
    warm_up_numba_observation()

    _assert_encoders_match(seed=42, steps=steps, version=version)


def test_numba_fast_observation_rejects_invalid_seen() -> None:
    """Il wrapper Python mantiene le stesse validazioni base sul vettore seen."""
    state = new_fast_2p_state(seed=0)

    with pytest.raises(ValueError, match="seen_cards_onehot len"):
        encode_fast_observation_numba_2p(state, player_index=0, seen_cards_onehot=(0,), version="v2")

    bad_seen = [0] * 40
    bad_seen[0] = 2
    with pytest.raises(ValueError, match="solo 0/1"):
        encode_fast_observation_numba_2p(state, player_index=0, seen_cards_onehot=tuple(bad_seen), version="v2")


@pytest.mark.parametrize("feature_dim", [int(FEATURE_DIM_2P_V1), int(FEATURE_DIM_2P_V2)])
def test_numba_mlp_rollout_is_deterministic_and_valid(feature_dim: int) -> None:
    """Il rollout full-JIT MLP deve essere deterministico per seed e rispettare gli invarianti."""
    warm_up_numba_mlp_rollout()

    w1 = np.zeros((feature_dim, 8), dtype=np.float32)
    b1 = np.zeros((8,), dtype=np.float32)
    w2 = np.zeros((8, 40), dtype=np.float32)
    b2 = np.zeros((40,), dtype=np.float32)

    first = evaluate_mlp_policy_numba_2p(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        opponent_name="heuristic_v1",
        num_games=40,
        seed=99,
        seat_fair=True,
    )
    second = evaluate_mlp_policy_numba_2p(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        opponent_name="heuristic_v1",
        num_games=40,
        seed=99,
        seat_fair=True,
    )

    assert first == second
    assert first.wins_policy + first.wins_opponent + first.draws == 40
    assert first.sum_policy + first.sum_opponent == 40 * 120
    stats = first.to_match_stats()
    assert stats.avg_points_agent0 + stats.avg_points_agent1 == pytest.approx(120.0)


def test_numba_mlp_rollout_rejects_bad_shapes() -> None:
    """Il wrapper Python deve intercettare modelli MLP non compatibili prima del JIT."""
    w1 = np.zeros((123, 8), dtype=np.float32)
    b1 = np.zeros((8,), dtype=np.float32)
    w2 = np.zeros((8, 40), dtype=np.float32)
    b2 = np.zeros((40,), dtype=np.float32)

    with pytest.raises(ValueError, match="feature_dim"):
        evaluate_mlp_policy_numba_2p(
            w1=w1,
            b1=b1,
            w2=w2,
            b2=b2,
            opponent_name="random",
            num_games=1,
            seed=0,
        )


@pytest.mark.parametrize("feature_dim", [int(FEATURE_DIM_2P_V1), int(FEATURE_DIM_2P_V2)])
def test_numba_a2c_trajectory_shapes_and_rewards_are_valid(feature_dim: int) -> None:
    """Il collector JIT deve produrre buffer A2C coerenti con una partita completa."""
    warm_up_numba_mlp_rollout()

    hidden_dim = 8
    w1 = np.zeros((feature_dim, hidden_dim), dtype=np.float32)
    b1 = np.zeros((hidden_dim,), dtype=np.float32)
    w2 = np.zeros((hidden_dim, 40), dtype=np.float32)
    b2 = np.zeros((40,), dtype=np.float32)
    wv = np.zeros((hidden_dim,), dtype=np.float32)

    traj = collect_a2c_trajectory_numba_2p(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        wv=wv,
        bv=0.0,
        opponent_name="heuristic_v1",
        game_seed=123,
        policy_seat=0,
    )
    again = collect_a2c_trajectory_numba_2p(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        wv=wv,
        bv=0.0,
        opponent_name="heuristic_v1",
        game_seed=123,
        policy_seat=0,
    )

    assert traj.policy_points + traj.opponent_points == 120
    assert traj.winner in (-1, 0, 1)
    assert 1 <= len(traj.rewards) <= 20
    assert traj.xs.shape == (len(traj.rewards), feature_dim)
    assert traj.z1s.shape == (len(traj.rewards), hidden_dim)
    assert traj.hs.shape == (len(traj.rewards), hidden_dim)
    assert traj.action_masks.shape == (len(traj.rewards), 40)
    assert traj.probs.shape == (len(traj.rewards), 40)
    assert traj.action_ids.shape == (len(traj.rewards),)
    assert traj.value_preds.shape == (len(traj.rewards),)
    assert float(np.sum(traj.rewards)) == pytest.approx((traj.policy_points - traj.opponent_points) / 120.0)
    assert np.allclose(traj.xs, again.xs)
    assert np.allclose(traj.probs, again.probs)
    assert np.array_equal(traj.action_ids, again.action_ids)
    assert np.allclose(traj.rewards, again.rewards)


def test_numba_a2c_trajectory_supports_mlp_opponent() -> None:
    """Il collector JIT deve poter usare un opponent MLP `.npz`-style con argmax mascherato."""
    warm_up_numba_mlp_rollout()

    policy_hidden = 8
    opponent_hidden = 6
    w1 = np.zeros((int(FEATURE_DIM_2P_V1), policy_hidden), dtype=np.float32)
    b1 = np.zeros((policy_hidden,), dtype=np.float32)
    w2 = np.zeros((policy_hidden, 40), dtype=np.float32)
    b2 = np.zeros((40,), dtype=np.float32)
    wv = np.zeros((policy_hidden,), dtype=np.float32)
    opponent_w1 = np.zeros((int(FEATURE_DIM_2P_V1), opponent_hidden), dtype=np.float32)
    opponent_b1 = np.zeros((opponent_hidden,), dtype=np.float32)
    opponent_w2 = np.zeros((opponent_hidden, 40), dtype=np.float32)
    opponent_b2 = np.zeros((40,), dtype=np.float32)

    traj = collect_a2c_trajectory_numba_2p(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        wv=wv,
        bv=0.0,
        opponent_name="bc_model",
        opponent_w1=opponent_w1,
        opponent_b1=opponent_b1,
        opponent_w2=opponent_w2,
        opponent_b2=opponent_b2,
        opponent_overkill_guard=True,
        game_seed=456,
        policy_seat=1,
    )

    assert traj.policy_points + traj.opponent_points == 120
    assert 1 <= len(traj.rewards) <= 20
    assert float(np.sum(traj.rewards)) == pytest.approx((traj.policy_points - traj.opponent_points) / 120.0)
