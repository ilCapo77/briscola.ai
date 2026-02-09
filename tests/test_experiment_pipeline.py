"""
Test per la pipeline esperimenti (helper core).

Nota:
non testiamo qui l'esecuzione reale di training/eval (sarebbe lenta e flaky),
ma le parti “pure” e riproducibili: naming e estrazione metrica da JSON.
"""

from __future__ import annotations

import pytest

from briscola_ai.ai.experiment_pipeline import build_experiment_name, extract_best_metric_from_matrix_json


def test_build_experiment_name_is_deterministic_and_safe() -> None:
    name1 = build_experiment_name(
        algo="a2c",
        num_games=200_000,
        seed=5,
        opponent=None,
        opponent_mix="heuristic_v1:0.7,random:0.2,greedy_points:0.1",
        tag=None,
    )
    name2 = build_experiment_name(
        algo="a2c",
        num_games=200_000,
        seed=5,
        opponent=None,
        opponent_mix="heuristic_v1:0.7,random:0.2,greedy_points:0.1",
        tag=None,
    )
    assert name1 == name2
    assert " " not in name1
    assert "/" not in name1
    assert name1.startswith("a2c_")
    assert "200k" in name1
    assert "seed5" in name1


def test_extract_best_metric_from_matrix_json_reads_holdout_diff() -> None:
    matrix_json = {
        "rows": [
            {
                "suite": {"name": "standard", "range_start": 0, "range_step": 1, "num_seeds": 2},
                "opponent": "heuristic_v1",
                "stats": {"avg_point_diff_agent_a_minus_agent_b": 1.0},
            },
            {
                "suite": {"name": "holdout", "range_start": 1_000_000, "range_step": 1, "num_seeds": 2},
                "opponent": "heuristic_v1",
                "stats": {"avg_point_diff_agent_a_minus_agent_b": 7.06},
            },
        ]
    }
    metric = extract_best_metric_from_matrix_json(
        matrix_json, benchmark="big", opponent="heuristic_v1", suite="holdout"
    )
    assert metric.benchmark == "big"
    assert metric.avg_diff == pytest.approx(7.06)


def test_extract_best_metric_from_matrix_json_raises_when_missing() -> None:
    with pytest.raises(ValueError):
        extract_best_metric_from_matrix_json({"rows": []}, benchmark="big")
