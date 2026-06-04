"""
Smoke test per il rollout A2C fast.

Il test non valuta la qualità del modello: verifica solo che il trainer possa usare
`--rollout-engine fast`, salvare un modello e dichiarare il metadato corretto.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def _write_dummy_mlp_model(path: Path, *, feature_dim: int = 248, hidden_dim: int = 4) -> None:
    """Scrive un piccolo modello MLP compatibile con `BCModelAgent`."""
    metadata = {
        "format": "mlp_bc_v1",
        "feature_dim": feature_dim,
        "encoder_version": "v1" if feature_dim == 248 else "v2",
        "inference_overkill_guard": True,
    }
    np.savez(
        path,
        w1=np.zeros((feature_dim, hidden_dim), dtype=np.float32),
        b1=np.zeros((hidden_dim,), dtype=np.float32),
        w2=np.zeros((hidden_dim, 40), dtype=np.float32),
        b2=np.zeros((40,), dtype=np.float32),
        metadata_json=json.dumps(metadata),
    )


@pytest.mark.parametrize(
    ("fast_rollout", "fast_encoder"),
    [
        ("python", "python"),
        ("python", "numba"),
        ("numba", "python"),
    ],
)
def test_train_a2c_fast_rollout_smoke(tmp_path: Path, fast_rollout: str, fast_encoder: str) -> None:
    """Esegue pochissime partite A2C con rollout fast e verifica il modello salvato."""
    out_path = tmp_path / f"a2c_fast_{fast_rollout}_{fast_encoder}_smoke.npz"
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "train_a2c.py"

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--out",
            str(out_path),
            "--rollout-engine",
            "fast",
            "--fast-rollout",
            fast_rollout,
            "--fast-encoder",
            fast_encoder,
            "--opponent",
            "random",
            "--num-games",
            "4",
            "--seed",
            "123",
            "--hidden-dim",
            "8",
            "--update-every",
            "2",
            "--log-every",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert out_path.exists()
    with np.load(out_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))

    assert metadata["rollout_engine"] == "fast"
    assert metadata["fast_rollout"] == fast_rollout
    assert metadata["fast_encoder"] == fast_encoder
    assert metadata["train"]["num_games"] == 4


def test_train_a2c_fast_numba_rollout_supports_bc_model_opponent(tmp_path: Path) -> None:
    """Il rollout Numba deve poter usare un opponent `.npz` senza tornare al dominio canonico."""
    out_path = tmp_path / "a2c_fast_numba_bc_opponent_smoke.npz"
    opponent_path = tmp_path / "opponent.npz"
    _write_dummy_mlp_model(opponent_path)
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "train_a2c.py"

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--out",
            str(out_path),
            "--rollout-engine",
            "fast",
            "--fast-rollout",
            "numba",
            "--opponent",
            "bc_model",
            "--opponent-model",
            str(opponent_path),
            "--num-games",
            "4",
            "--seed",
            "123",
            "--hidden-dim",
            "8",
            "--update-every",
            "2",
            "--log-every",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert out_path.exists()
    with np.load(out_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))

    assert metadata["rollout_engine"] == "fast"
    assert metadata["fast_rollout"] == "numba"
    assert metadata["opponent"] == "bc_model"
    assert metadata["opponent_model"] == str(opponent_path)
