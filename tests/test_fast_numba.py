"""
Test per il primo core Numba.

Questi test non confrontano la sequenza RNG con il dominio canonico: il core JIT usa un RNG
interno separato. Verificano però determinismo, invarianti di Briscola e aggregazione.
"""

from __future__ import annotations

import pytest

from briscola_ai.ai.fast_numba import evaluate_random_numba_2p, play_random_game_numba, warm_up_numba


def test_numba_random_game_is_deterministic_and_valid() -> None:
    """Stesso seed -> stesso risultato; i punti totali devono essere 120."""
    warm_up_numba()

    first = play_random_game_numba(123)
    second = play_random_game_numba(123)

    assert first == second
    points0, points1, winner = first
    assert points0 + points1 == 120
    assert winner in (-1, 0, 1)
    if points0 > points1:
        assert winner == 0
    elif points1 > points0:
        assert winner == 1
    else:
        assert winner == -1


def test_numba_random_evaluation_counts_are_consistent() -> None:
    """L'aggregazione Numba deve produrre contatori e medie coerenti."""
    summary = evaluate_random_numba_2p(num_games=50, seed=0)
    stats = summary.to_match_stats()

    assert stats.num_games == 50
    assert stats.wins_agent0 + stats.wins_agent1 + stats.draws == 50
    assert summary.sum0 + summary.sum1 == 50 * 120
    assert stats.avg_points_agent0 + stats.avg_points_agent1 == pytest.approx(120.0)
