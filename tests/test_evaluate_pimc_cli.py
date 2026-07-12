"""Test del confronto PIMC simmetrico fra due budget di determinizzazione."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_evaluate_pimc_can_apply_the_same_belief_to_both_agents(tmp_path: Path) -> None:
    """Il JSON deve provare che entrambi i lati usano lo stesso asset belief, non uno uniforme."""
    model = ROOT / "data/models/best_a2c_v14.npz"
    belief = ROOT / "data/models/belief_v0_h128_50k_seed20260702.npz"
    output = tmp_path / "pimc.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluate_pimc.py"),
            "--model",
            str(model),
            "--num-games",
            "2",
            "--seed",
            "17",
            "--determinizations",
            "1",
            "--max-unknown-cards",
            "8",
            "--opponent",
            "pimc",
            "--opponent-determinizations",
            "1",
            "--opponent-max-unknown-cards",
            "8",
            "--belief-model",
            str(belief),
            "--opponent-belief-model",
            str(belief),
            "--out-json",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    belief_sha256 = hashlib.sha256(belief.read_bytes()).hexdigest()
    assert payload["schema"] == "briscola.pimc_evaluation.v1"
    assert payload["belief_model"] == str(belief)
    assert payload["opponent_belief_model"] == str(belief)
    assert payload["artifacts"]["belief_model"]["sha256"] == belief_sha256
    assert payload["artifacts"]["opponent_belief_model"]["sha256"] == belief_sha256
    assert "belief_v0_h128_50k_seed20260702.npz" in payload["stats"]["agent_a_name"]
    assert "belief_v0_h128_50k_seed20260702.npz" in payload["stats"]["agent_b_name"]
    assert payload["pimc_metrics"]["successful_determinizations"] > 0
    assert payload["opponent_metrics"]["successful_determinizations"] > 0
