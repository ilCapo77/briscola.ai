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
