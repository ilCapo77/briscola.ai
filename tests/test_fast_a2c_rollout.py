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


@pytest.mark.parametrize("fast_encoder", ["python", "numba"])
def test_train_a2c_fast_rollout_smoke(tmp_path: Path, fast_encoder: str) -> None:
    """Esegue pochissime partite A2C con rollout fast e verifica il modello salvato."""
    out_path = tmp_path / f"a2c_fast_{fast_encoder}_smoke.npz"
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "train_a2c.py"

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--out",
            str(out_path),
            "--rollout-engine",
            "fast",
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
    assert metadata["fast_encoder"] == fast_encoder
    assert metadata["train"]["num_games"] == 4
