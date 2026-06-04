"""
Test per la pipeline esperimenti (helper core).

Nota:
non testiamo qui l'esecuzione reale di training/eval (sarebbe lenta e flaky),
ma le parti “pure” e riproducibili: naming, estrazione metrica da JSON e costruzione
dei comandi pipeline senza lanciare training reali.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from briscola_ai.ai.experiment_pipeline import build_experiment_name, extract_best_metric_from_matrix_json


def _run_experiment_script() -> Path:
    """Restituisce il path assoluto dello script pipeline usato dai test CLI."""
    return Path(__file__).resolve().parent.parent / "scripts" / "run_experiment.py"


def _load_run_experiment_module():
    """Carica `scripts/run_experiment.py` come modulo per poter monkeypatchare `_run`."""
    spec = importlib.util.spec_from_file_location("_run_experiment_under_test", _run_experiment_script())
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def test_run_experiment_help_exposes_fast_rollout_flags() -> None:
    """La pipeline deve pubblicare i flag Numba senza obbligare a usare `--train-extra`."""
    proc = subprocess.run(
        [sys.executable, str(_run_experiment_script()), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--rollout-engine" in proc.stdout
    assert "--fast-encoder" in proc.stdout
    assert "--fast-rollout" in proc.stdout


def test_run_experiment_rejects_fast_rollout_for_pg() -> None:
    """I flag fast/Numba sono specifici di A2C: PG non ha quel collector."""
    proc = subprocess.run(
        [
            sys.executable,
            str(_run_experiment_script()),
            "--algo",
            "pg",
            "--rollout-engine",
            "fast",
            "--num-games",
            "1",
            "--benchmarks",
            "small",
            "--no-update-best",
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "supportati solo con --algo a2c" in proc.stderr


def test_run_experiment_forwards_fast_rollout_flags_and_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Verifica la parte importante senza allenare davvero:
    il comando A2C generato contiene il path Numba e il manifest lo registra.
    """
    module = _load_run_experiment_module()
    commands: list[list[str]] = []

    def fake_run(cmd: list[str], *, log_path: Path, env_overrides: dict[str, str] | None = None) -> None:
        commands.append(cmd)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("$ " + " ".join(cmd) + "\n", encoding="utf-8")

        if any(part.endswith("train_a2c.py") for part in cmd):
            out_path = Path(cmd[cmd.index("--out") + 1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"fake-npz-for-pipeline-test")
            return

        if any(part.endswith("evaluate_matrix.py") for part in cmd):
            out_json = Path(cmd[cmd.index("--out-json") + 1])
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "suite": {"name": "holdout", "range_start": 1_000_000, "range_step": 1},
                                "opponent": "heuristic_v1",
                                "stats": {"avg_point_diff_agent_a_minus_agent_b": 1.23},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            return

        raise AssertionError(f"Comando inatteso: {cmd}")

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "--algo",
            "a2c",
            "--name",
            "numba_pipeline_test",
            "--models-dir",
            str(tmp_path / "models"),
            "--experiments-dir",
            str(tmp_path / "experiments"),
            "--opponent-mix",
            "heuristic_v1:1.0",
            "--num-games",
            "10",
            "--train-seed",
            "3",
            "--benchmarks",
            "small",
            "--no-update-best",
            "--seat-fair",
            "--rollout-engine",
            "fast",
            "--fast-encoder",
            "numba",
            "--fast-rollout",
            "numba",
        ],
    )

    assert module.main() == 0
    train_cmd = commands[0]
    assert train_cmd[train_cmd.index("--rollout-engine") + 1] == "fast"
    assert train_cmd[train_cmd.index("--fast-encoder") + 1] == "numba"
    assert train_cmd[train_cmd.index("--fast-rollout") + 1] == "numba"

    manifest_path = tmp_path / "experiments" / "numba_pipeline_test" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["train"]["rollout"] == {
        "engine": "fast",
        "fast_encoder": "numba",
        "fast_rollout": "numba",
    }
    assert manifest["train"]["cmd"] == train_cmd
