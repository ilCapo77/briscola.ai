"""Test per lo Stage 0 value-learning dell'ipotesi V-lookahead."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from briscola_ai.ai.agents import HeuristicAgentV2, PIMCAgent
from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V3
from briscola_ai.ai.models import load_value_model_npz
from briscola_ai.ai.numba.value_dataset import (
    COLLECT_WINDOW,
    PHASE_ENDGAME,
    PHASE_PIMC_WINDOW,
    collect_value_dataset_batch_numba,
    warm_up_numba_value_dataset,
)

_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str) -> Any:
    """Carica uno script da `scripts/` come modulo testabile."""
    path = _ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Legge un JSONL piccolo in memoria."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_generate_value_dataset_and_train_value_roundtrip(tmp_path: Path) -> None:
    """Il dataset value deve essere leggibile dal trainer e salvabile come value model `.npz`."""
    generator = _load_script_module("generate_value_dataset")
    train_value = _load_script_module("train_value")

    data_path = tmp_path / "value.jsonl"
    summary = generator.generate_value_dataset(
        generator.ValueDatasetConfig(
            out_path=data_path,
            agent_name="heuristic_v2",
            model_path=None,
            num_games=3,
            seed=123,
            epsilon=0.2,
            label_mode="same-game",
        )
    )

    records = _read_jsonl(data_path)
    assert summary["records_written"] == len(records)
    assert len(records) > 0
    for record in records:
        assert record["dataset_kind"] == "value_observation"
        assert record["observation"]["num_players"] == 2
        assert record["observation"]["my_turn"] is True
        assert record["target_residual_scaled"] == record["residual_score_delta"] / 120.0
        assert record["target_final_scaled"] == record["final_score_delta"] / 120.0
        assert record["phase"] in {"early", "mid", "pimc_window", "endgame"}

    dataset = train_value.load_value_dataset(data_path, encoder_version="v3", target="residual")
    assert dataset.x.shape == (len(records), int(FEATURE_DIM_2P_V3))
    assert dataset.y.shape == (len(records),)

    model_path = tmp_path / "value_model.npz"
    argv = [
        "train_value.py",
        "--data",
        str(data_path),
        "--out",
        str(model_path),
        "--encoder-version",
        "v3",
        "--hidden-dim",
        "8",
        "--epochs",
        "1",
        "--batch-size",
        "8",
        "--seed",
        "123",
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        assert train_value.main() == 0
    finally:
        sys.argv = old_argv

    model = load_value_model_npz(model_path)
    assert model.feature_dim == int(FEATURE_DIM_2P_V3)
    assert model.hidden_dim == 8
    pred = model.predict_points(np.zeros((int(FEATURE_DIM_2P_V3),), dtype=np.float32), current_score_delta=4.0)
    assert isinstance(pred, float)


def test_generate_value_dataset_numba_npz_roundtrip(tmp_path: Path) -> None:
    """Il generatore compatto Numba deve produrre un `.npz` allenabile dal trainer value."""
    generator = _load_script_module("generate_value_dataset_numba")
    train_value = _load_script_module("train_value")

    feature_dim = int(FEATURE_DIM_2P_V3)
    model_path = tmp_path / "policy.npz"
    np.savez(
        model_path,
        w1=np.zeros((feature_dim, 1), dtype=np.float32),
        b1=np.zeros(1, dtype=np.float32),
        w2=np.zeros((1, 40), dtype=np.float32),
        b2=np.zeros(40, dtype=np.float32),
        metadata_json=json.dumps(
            {
                "model": "mlp",
                "feature_dim": feature_dim,
                "encoder_version": "v3",
                "inference_overkill_guard": True,
                "label": "policy-test",
            }
        ),
    )

    data_path = tmp_path / "value_numba.npz"
    summary = generator.generate_value_dataset_numba(
        model_path=model_path,
        out_path=data_path,
        num_games=8,
        seed=456,
        epsilon=0.1,
        batch_games=4,
        collect_mode="window",
        max_unknown_cards=8,
        include_endgame=True,
        max_records=None,
        feature_dtype="float16",
    )

    assert summary["records_written"] > 0
    assert summary["phase_pimc_window"] > 0
    assert data_path.exists()

    dataset = train_value.load_value_dataset(data_path, encoder_version="v3", target="residual")
    assert dataset.x.shape[0] == summary["records_written"]
    assert dataset.x.shape[1] == feature_dim
    assert dataset.y.shape == (dataset.x.shape[0],)
    assert set(dataset.phases.tolist()) <= {"early", "mid", "pimc_window", "endgame"}

    model_out = tmp_path / "value_model_numba.npz"
    old_argv = sys.argv
    try:
        sys.argv = [
            "train_value.py",
            "--data",
            str(data_path),
            "--out",
            str(model_out),
            "--hidden-dim",
            "8",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--seed",
            "456",
        ]
        assert train_value.main() == 0
    finally:
        sys.argv = old_argv
    assert load_value_model_npz(model_out).feature_dim == feature_dim


def test_numba_value_dataset_collector_shapes_are_valid() -> None:
    """Il collector Numba deve salvare solo osservazioni valide e target in range Briscola."""
    warm_up_numba_value_dataset()
    feature_dim = int(FEATURE_DIM_2P_V3)
    w1 = np.zeros((feature_dim, 1), dtype=np.float32)
    b1 = np.zeros(1, dtype=np.float32)
    w2 = np.zeros((1, 40), dtype=np.float32)
    b2 = np.zeros(40, dtype=np.float32)

    batch = collect_value_dataset_batch_numba(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        overkill_guard_enabled=True,
        epsilon=0.2,
        collect_mode=COLLECT_WINDOW,
        max_unknown_cards=8,
        include_endgame=True,
        game_seeds=np.asarray([100, 101, 102], dtype=np.int64),
    )
    valid = batch.valid
    assert batch.games_completed == 3
    assert batch.xs.shape == (3, 40, feature_dim)
    assert int(np.sum(valid)) > 0
    assert np.all(np.isfinite(batch.xs[valid]))
    assert np.all(batch.final_delta[valid] >= -120)
    assert np.all(batch.final_delta[valid] <= 120)
    assert set(batch.phase[valid].tolist()) <= {PHASE_PIMC_WINDOW, PHASE_ENDGAME}


def test_value_dataset_v6_continuation_labels_each_state(tmp_path: Path) -> None:
    """`label_mode=v6-continuation` deve produrre target terminali senza mutare la partita sorgente."""
    generator = _load_script_module("generate_value_dataset")

    data_path = tmp_path / "value_continuation.jsonl"
    summary = generator.generate_value_dataset(
        generator.ValueDatasetConfig(
            out_path=data_path,
            agent_name="heuristic_v2",
            model_path=None,
            num_games=1,
            seed=77,
            epsilon=0.5,
            label_mode="v6-continuation",
        )
    )
    records = _read_jsonl(data_path)
    assert summary["records_written"] == len(records)
    assert len(records) == 40
    assert {record["generation"]["label_mode"] for record in records} == {"v6-continuation"}
    assert all(-120 <= int(record["final_score_delta"]) <= 120 for record in records)


def test_evaluate_value_ranking_smoke(tmp_path: Path) -> None:
    """Il gate ranking-vs-PIMC deve girare su un dataset diagnostico piccolo."""
    generator = _load_script_module("generate_value_dataset")
    train_value = _load_script_module("train_value")
    pimc_generator = _load_script_module("generate_pimc_teacher_dataset")
    ranking = _load_script_module("evaluate_value_ranking")

    value_data = tmp_path / "value.jsonl"
    generator.generate_value_dataset(
        generator.ValueDatasetConfig(
            out_path=value_data,
            agent_name="heuristic_v2",
            model_path=None,
            num_games=3,
            seed=10,
            epsilon=0.1,
            label_mode="same-game",
        )
    )

    value_model_path = tmp_path / "value_model.npz"
    old_argv = sys.argv
    try:
        sys.argv = [
            "train_value.py",
            "--data",
            str(value_data),
            "--out",
            str(value_model_path),
            "--hidden-dim",
            "8",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--seed",
            "10",
        ]
        assert train_value.main() == 0
    finally:
        sys.argv = old_argv

    teacher_data = tmp_path / "pimc_teacher.jsonl"
    base_agent = HeuristicAgentV2()
    teacher = PIMCAgent(
        rollout_agent=base_agent,
        fallback=base_agent,
        num_determinizations=2,
        max_unknown_cards=8,
    )
    pimc_generator.generate_pimc_teacher_dataset(
        pimc_generator.PIMCTeacherDatasetConfig(
            out_path=teacher_data,
            num_examples=4,
            max_games=30,
            seed=20,
            max_unknown_cards=8,
            include_fallback_examples=False,
        ),
        teacher=teacher,
        play_agent=base_agent,
    )

    summary = ranking.evaluate_value_ranking(
        ranking.RankingConfig(
            data_path=teacher_data,
            value_model_path=value_model_path,
            continuation_agent="heuristic_v2",
            continuation_model_path=None,
            determinizations=1,
            max_records=2,
            seed=99,
            min_pair_margin=0.0,
            strong_margin_min=2.0,
            reliable_margin_ci_low_min=0.0,
        )
    )

    assert summary["counts"]["records_search"] >= 1
    assert summary["counts"]["records_failed"] == 0
    assert "pairwise_accuracy" in summary["metrics"]
