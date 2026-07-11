"""Test del wiring A2C per l'augmentation paired dei semi."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V4
from briscola_ai.ai.evaluation.suit_symmetry import IDENTITY_SUIT_PERMUTATION

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _ROOT / "scripts" / "train_a2c.py"
_SPEC = importlib.util.spec_from_file_location("train_a2c_suit_augmentation_tests", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Impossibile caricare {_SCRIPT_PATH}")
_TRAIN_A2C = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TRAIN_A2C
_SPEC.loader.exec_module(_TRAIN_A2C)

A2CPolicy = _TRAIN_A2C.A2CPolicy
_accumulate_numba_trajectory_grads = _TRAIN_A2C._accumulate_numba_trajectory_grads
_accumulate_paired_suit_trajectory_grads = _TRAIN_A2C._accumulate_paired_suit_trajectory_grads
_forward_policy_batch = _TRAIN_A2C._forward_policy_batch


def _policy_and_trajectory() -> tuple[object, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Crea policy e traiettoria numerica deterministiche con tre azioni legali per step."""
    rng = np.random.default_rng(77)
    steps = 5
    feature_dim = int(FEATURE_DIM_2P_V4)
    hidden_dim = 7
    policy = A2CPolicy(
        w1=rng.normal(0.0, 0.03, size=(feature_dim, hidden_dim)).astype(np.float32),
        b1=rng.normal(0.0, 0.03, size=hidden_dim).astype(np.float32),
        w2=rng.normal(0.0, 0.03, size=(hidden_dim, 40)).astype(np.float32),
        b2=rng.normal(0.0, 0.03, size=40).astype(np.float32),
        wv=rng.normal(0.0, 0.03, size=hidden_dim).astype(np.float32),
        bv=0.01,
    )
    xs = rng.normal(0.0, 0.5, size=(steps, feature_dim)).astype(np.float32)
    masks = np.zeros((steps, 40), dtype=bool)
    ids = np.empty(steps, dtype=np.int64)
    for step_index in range(steps):
        legal = np.asarray([step_index, 10 + step_index, 30 + step_index], dtype=np.int64)
        masks[step_index, legal] = True
        ids[step_index] = legal[step_index % len(legal)]
    returns = rng.normal(0.0, 0.2, size=steps).astype(np.float32)
    return policy, xs, masks, ids, returns


def _zero_grads(policy) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Alloca i cinque tensori gradiente mutati dagli helper vettoriali."""
    return (
        np.zeros_like(policy.w1),
        np.zeros_like(policy.b1),
        np.zeros_like(policy.w2),
        np.zeros_like(policy.b2),
        np.zeros_like(policy.wv),
    )


def test_identity_paired_copy_preserves_mean_gradient_and_doubles_step_count() -> None:
    """Originale+copia identica, mediati su 2N, devono uguagliare il gradiente originale su N."""
    policy, xs, masks, ids, returns = _policy_and_trajectory()
    z1s, hs, probs, values = _forward_policy_batch(policy, xs=xs, action_masks=masks)

    original_grads = _zero_grads(policy)
    original_stats = _accumulate_numba_trajectory_grads(
        policy=policy,
        xs=xs,
        z1s=z1s,
        hs=hs,
        action_masks=masks,
        probs=probs,
        action_ids=ids,
        value_preds=values,
        returns_to_go=returns,
        entropy_beta=0.0005,
        value_coef=0.5,
        bc_anchor=None,
        bc_anchor_beta=0.0,
        gw1=original_grads[0],
        gb1=original_grads[1],
        gw2=original_grads[2],
        gb2=original_grads[3],
        gwv=original_grads[4],
    )

    paired_grads = _zero_grads(policy)
    first_stats = _accumulate_numba_trajectory_grads(
        policy=policy,
        xs=xs,
        z1s=z1s,
        hs=hs,
        action_masks=masks,
        probs=probs,
        action_ids=ids,
        value_preds=values,
        returns_to_go=returns,
        entropy_beta=0.0005,
        value_coef=0.5,
        bc_anchor=None,
        bc_anchor_beta=0.0,
        gw1=paired_grads[0],
        gb1=paired_grads[1],
        gw2=paired_grads[2],
        gb2=paired_grads[3],
        gwv=paired_grads[4],
    )
    copy_stats = _accumulate_paired_suit_trajectory_grads(
        policy=policy,
        xs=xs,
        action_masks=masks,
        action_ids=ids,
        returns_to_go=returns,
        encoder_version="v4",
        permutation=IDENTITY_SUIT_PERMUTATION,
        entropy_beta=0.0005,
        value_coef=0.5,
        bc_anchor=None,
        bc_anchor_beta=0.0,
        gw1=paired_grads[0],
        gb1=paired_grads[1],
        gw2=paired_grads[2],
        gb2=paired_grads[3],
        gwv=paired_grads[4],
    )

    assert first_stats.steps + copy_stats.steps == 2 * original_stats.steps
    for original, paired in zip(original_grads, paired_grads, strict=True):
        np.testing.assert_allclose(
            original / np.float32(original_stats.steps),
            paired / np.float32(first_stats.steps + copy_stats.steps),
            rtol=1e-6,
            atol=1e-7,
        )
    assert (first_stats.gbv + copy_stats.gbv) / (first_stats.steps + copy_stats.steps) == pytest.approx(
        original_stats.gbv / original_stats.steps
    )


def _write_synthetic_v4_model(path: Path) -> None:
    """Modello v4 piccolo ma caricabile come init e BC-anchor dello smoke."""
    rng = np.random.default_rng(11)
    feature_dim = int(FEATURE_DIM_2P_V4)
    hidden_dim = 8
    np.savez(
        path,
        w1=rng.normal(0.0, 0.02, size=(feature_dim, hidden_dim)).astype(np.float32),
        b1=np.zeros(hidden_dim, dtype=np.float32),
        w2=rng.normal(0.0, 0.02, size=(hidden_dim, 40)).astype(np.float32),
        b2=np.zeros(40, dtype=np.float32),
        metadata_json=json.dumps(
            {
                "format": "mlp_bc_v1",
                "feature_dim": feature_dim,
                "encoder_version": "v4",
                "inference_overkill_guard": False,
            }
        ),
    )


def _run_cli_smoke(
    *,
    init_path: Path,
    output_path: Path,
    suit_augmentation: str | None,
) -> subprocess.CompletedProcess[str]:
    """Esegue un training minimo; ``None`` lascia al parser il default storico."""
    command = [
        sys.executable,
        str(_SCRIPT_PATH),
        "--init",
        str(init_path),
        "--encoder-version",
        "v4",
        "--rollout-engine",
        "fast",
        "--fast-rollout",
        "numba",
        "--opponent",
        "heuristic_v1",
        "--num-games",
        "20",
        "--update-every",
        "10",
        "--seed",
        "123",
        "--metrics-mode",
        "summary",
        "--out",
        str(output_path),
    ]
    if suit_augmentation is not None:
        command.extend(["--suit-augmentation", suit_augmentation])
    return subprocess.run(
        command,
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.slow
def test_train_a2c_explicit_off_is_identical_to_historical_default(tmp_path: Path) -> None:
    """Il nuovo flag, se spento, non deve cambiare pesi, bias o metadata del trainer."""
    init_path = tmp_path / "init_v4.npz"
    default_path = tmp_path / "default_v4.npz"
    explicit_off_path = tmp_path / "explicit_off_v4.npz"
    _write_synthetic_v4_model(init_path)

    default_result = _run_cli_smoke(
        init_path=init_path,
        output_path=default_path,
        suit_augmentation=None,
    )
    off_result = _run_cli_smoke(
        init_path=init_path,
        output_path=explicit_off_path,
        suit_augmentation="off",
    )

    assert default_result.returncode == 0, default_result.stderr
    assert off_result.returncode == 0, off_result.stderr
    with (
        np.load(default_path, allow_pickle=False) as default_data,
        np.load(explicit_off_path, allow_pickle=False) as off_data,
    ):
        assert set(default_data.files) == set(off_data.files)
        for key in default_data.files:
            np.testing.assert_array_equal(default_data[key], off_data[key])
        metadata = json.loads(str(default_data["metadata_json"].item()))
    assert "suit_augmentation" not in metadata


@pytest.mark.slow
def test_train_a2c_paired_cli_smoke_and_metadata(tmp_path: Path) -> None:
    """Il path Numba batch deve allenare e dichiarare il contratto paired nell'artefatto."""
    init_path = tmp_path / "init_v4.npz"
    output_path = tmp_path / "paired_v4.npz"
    _write_synthetic_v4_model(init_path)

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--init",
            str(init_path),
            "--bc-anchor",
            str(init_path),
            "--bc-anchor-beta",
            "0.01",
            "--encoder-version",
            "v4",
            "--rollout-engine",
            "fast",
            "--fast-rollout",
            "numba",
            "--opponent",
            "heuristic_v1",
            "--num-games",
            "20",
            "--update-every",
            "10",
            "--suit-augmentation",
            "paired",
            "--seed",
            "123",
            "--metrics-mode",
            "summary",
            "--out",
            str(output_path),
        ],
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"trainer fallito:\n{result.stdout}\n{result.stderr}"
    with np.load(output_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
    assert metadata["suit_augmentation"] == {
        "mode": "paired",
        "copies_per_trajectory": 1,
        "permutation_scope": "whole_trajectory",
        "permutation_distribution": "uniform_nonidentity_23",
        "loss_normalization": "mean_over_original_and_copy",
        "rng_seed": 123 ^ 0x51A17A9E,
    }
