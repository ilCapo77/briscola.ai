"""
Test per evaluation matrix.

Obiettivo:
- garantire che la generazione seed suite sia stabile e con dimensioni corrette
- sanity check end-to-end su una valutazione minuscola (pochi game) non necessaria:
  qui testiamo solo la parte di costruzione/serializzazione senza costi eccessivi.
"""

from __future__ import annotations

import json
from pathlib import Path

from briscola_ai.ai.evaluation_matrix import EvaluationMatrix, build_suites_for_benchmark, make_range_seed_suite


def test_make_range_seed_suite_length_and_32bit_normalization() -> None:
    seeds = make_range_seed_suite(start=0xFFFFFFFF, step=1, count=3)
    assert len(seeds) == 3
    assert seeds[0] == 0xFFFFFFFF
    # wrap 32-bit
    assert seeds[1] == 0
    assert seeds[2] == 1


def test_build_suites_for_benchmark_returns_two_suites_with_expected_counts() -> None:
    suites = build_suites_for_benchmark(benchmark="medium", standard_start=0, holdout_start=1_000_000, step=1)
    assert [s.name for s in suites] == ["standard", "holdout"]
    # medium seat-fair: 10k games -> 5k seeds
    assert suites[0].num_seeds == 5000
    assert suites[1].num_seeds == 5000


def test_evaluation_matrix_json_serialization_is_stable(tmp_path: Path) -> None:
    # Creiamo un oggetto minimale e verifichiamo che `to_json_text()` sia JSON valido.
    matrix = EvaluationMatrix(model_path="data/model.npz", benchmark="small", num_games=2000, seed=0, rows=[])
    text = matrix.to_json_text()
    parsed = json.loads(text)
    assert parsed["model_path"] == "data/model.npz"
    assert parsed["benchmark"] == "small"
