"""
Test per la cross-entropy pesata in `train_bc.py` e per il filtro subset PIMC.

Coprono due cose:

1. `train_bc.py` legge `sample_weight` dal JSONL e lo applica alla CE in modo coerente:
   un esempio con peso 2 contribuisce al gradiente esattamente come due copie a peso 1.
2. `filter_pimc_teacher_subset.py` tiene solo le correzioni di search affidabili e calcola
   i pesi attesi.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from briscola_ai.backend.observation_builder import build_observation_dto
from briscola_ai.domain.state import new_game_state

_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(filename: str, alias: str):
    """Carica uno script di `scripts/` come modulo per testarne gli helper non esportati."""
    path = _ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------
# train_bc.py: lettura e applicazione di sample_weight
# --------------------------------------------------------------------------------------


def _write_weighted_jsonl(path: Path, weights: list[float | None]) -> None:
    """Scrive un JSONL v3 minimo dove ogni riga ha (o no) il campo sample_weight."""
    rows: list[str] = []
    for seed, weight in enumerate(weights):
        state = new_game_state(num_players=2, seed=seed)
        obs = build_observation_dto(state, player_index=0, server_version=seed).model_dump(mode="json")
        record: dict[str, Any] = {
            "schema_version": 1,
            "game_id": f"w_{seed}",
            "event_id": seed,
            "player_index": 0,
            "is_ai": True,
            "observation": obs,
            "action": {"card_index": 0},
        }
        if weight is not None:
            record["sample_weight"] = weight
        rows.append(json.dumps(record, ensure_ascii=False))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_build_training_examples_reads_sample_weight(tmp_path: Path) -> None:
    """I pesi del JSONL devono finire nell'array W; assenti -> 1.0."""
    train_bc = _load_script_module("train_bc.py", "train_bc_for_weight_test")
    data_path = tmp_path / "weighted.jsonl"
    _write_weighted_jsonl(data_path, [3.0, None, 0.5])

    _, _, _, weights, target_probs = train_bc._build_training_examples(data_path, encoder_version="v3")
    assert weights.tolist() == [3.0, 1.0, 0.5]
    assert target_probs is None

    # --ignore-sample-weights forza il training uniforme.
    _, _, _, weights_uniform, _ = train_bc._build_training_examples(
        data_path, encoder_version="v3", ignore_sample_weights=True
    )
    assert weights_uniform.tolist() == [1.0, 1.0, 1.0]


def test_invalid_sample_weight_is_rejected(tmp_path: Path) -> None:
    """Un peso negativo è un bug del dataset: deve sollevare, non passare silenziosamente."""
    train_bc = _load_script_module("train_bc.py", "train_bc_for_weight_reject")
    data_path = tmp_path / "bad.jsonl"
    _write_weighted_jsonl(data_path, [-1.0])
    try:
        train_bc._build_training_examples(data_path, encoder_version="v3")
    except ValueError:
        return
    raise AssertionError("Un sample_weight negativo deve sollevare ValueError")


def test_build_training_examples_builds_soft_pimc_targets(tmp_path: Path) -> None:
    """`--soft-labels` deve trasformare i mean_score PIMC in una distribuzione masked a 40 azioni."""
    train_bc = _load_script_module("train_bc.py", "train_bc_for_soft_target_test")
    data_path = tmp_path / "soft.jsonl"
    state = new_game_state(num_players=2, seed=123)
    obs = build_observation_dto(state, player_index=0, server_version=1).model_dump(mode="json")
    action_values = [
        {"card_index": 0, "mean_score": 12.0, "rollout_count": 64},
        {"card_index": 1, "mean_score": 10.0, "rollout_count": 64},
        {"card_index": 2, "mean_score": 0.0, "rollout_count": 64},
    ]
    record = {
        "schema_version": 1,
        "game_id": "soft_0",
        "event_id": 0,
        "player_index": 0,
        "is_ai": True,
        "observation": obs,
        "action": {"card_index": 0},
        "teacher": {"decision_type": "search", "search_diagnostics": {"action_values": action_values}},
    }
    data_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    _, mask, y, _, target_probs = train_bc._build_training_examples(
        data_path,
        encoder_version="v3",
        soft_labels=True,
        soft_temperature=2.0,
    )

    assert target_probs is not None
    assert target_probs.shape == (1, 40)
    assert abs(float(target_probs[0].sum()) - 1.0) < 1e-6
    assert np.all(target_probs[0][~mask[0]] == 0.0)

    action0 = train_bc.card_dto_to_action_id(obs["my_hand"][0])
    action1 = train_bc.card_dto_to_action_id(obs["my_hand"][1])
    action2 = train_bc.card_dto_to_action_id(obs["my_hand"][2])
    assert int(y[0]) == int(action0)
    assert target_probs[0, action0] > target_probs[0, action1] > target_probs[0, action2]

    _, _, _, _, colder = train_bc._build_training_examples(
        data_path,
        encoder_version="v3",
        soft_labels=True,
        soft_temperature=1.0,
    )
    assert colder is not None
    assert colder[0, action0] > target_probs[0, action0]


def _toy_classification_batch(rng: np.random.Generator, *, n: int, d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Costruisce un batch giocattolo con mask piena e target validi."""
    x = rng.normal(size=(n, d)).astype(np.float32)
    mask = np.zeros((n, 40), dtype=bool)
    mask[:, :5] = True  # prime 5 azioni sempre legali
    y = rng.integers(0, 5, size=(n,)).astype(np.int64)
    return x, mask, y


def test_weighted_ce_gradient_matches_duplication(tmp_path: Path) -> None:
    """
    Pesare la riga 0 con 2 (resto 1) deve aggiungere al gradiente esattamente il contributo
    della riga 0 una volta in più: grad(w=[2,1,..]) - grad(w=None) == x0^T (probs0 - onehot0)/B.
    """
    train_bc = _load_script_module("train_bc.py", "train_bc_for_grad_test")
    rng = np.random.default_rng(0)
    n, d = 4, 6
    x, mask, y = _toy_classification_batch(rng, n=n, d=d)
    w = rng.normal(scale=0.1, size=(d, 40)).astype(np.float32)
    b = np.zeros((40,), dtype=np.float32)

    # Gradiente non pesato.
    _, grad_w_unw, grad_b_unw, _ = train_bc._loss_and_grad_linear(x, mask, y, w, b)

    # Gradiente con la riga 0 a peso 2.
    sample_weight = np.ones((n,), dtype=np.float32)
    sample_weight[0] = 2.0
    _, grad_w_wei, grad_b_wei, _ = train_bc._loss_and_grad_linear(x, mask, y, w, b, sample_weight=sample_weight)

    # Contributo extra atteso = quello della riga 0 (probs0 - onehot0)/B, mascherato.
    logits0 = x @ w + b
    masked0 = train_bc._masked_logits(logits0, mask)
    probs0 = train_bc._softmax(masked0)
    dextra = np.zeros_like(probs0)
    dextra[0] = probs0[0]
    dextra[0, y[0]] -= 1.0
    dextra /= float(n)
    dextra[~mask] = 0.0
    expected_grad_w = x.T @ dextra
    expected_grad_b = np.sum(dextra, axis=0)

    np.testing.assert_allclose(grad_w_wei - grad_w_unw, expected_grad_w, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(grad_b_wei - grad_b_unw, expected_grad_b, rtol=1e-5, atol=1e-6)


def test_uniform_weights_equal_unweighted() -> None:
    """sample_weight tutto a 1.0 deve dare loss/grad identici al caso non pesato (mlp)."""
    train_bc = _load_script_module("train_bc.py", "train_bc_for_uniform_test")
    rng = np.random.default_rng(1)
    n, d, h = 5, 6, 8
    x, mask, y = _toy_classification_batch(rng, n=n, d=d)
    w1 = rng.normal(scale=0.1, size=(d, h)).astype(np.float32)
    b1 = np.zeros((h,), dtype=np.float32)
    w2 = rng.normal(scale=0.1, size=(h, 40)).astype(np.float32)
    b2 = np.zeros((40,), dtype=np.float32)

    loss_a, gw1_a, _, gw2_a, _, _ = train_bc._loss_and_grad_mlp(x, mask, y, w1, b1, w2, b2, weight_decay=0.0)
    ones = np.ones((n,), dtype=np.float32)
    loss_b, gw1_b, _, gw2_b, _, _ = train_bc._loss_and_grad_mlp(
        x, mask, y, w1, b1, w2, b2, weight_decay=0.0, sample_weight=ones
    )
    assert abs(loss_a - loss_b) < 1e-6
    np.testing.assert_allclose(gw1_a, gw1_b, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(gw2_a, gw2_b, rtol=1e-6, atol=1e-7)


# --------------------------------------------------------------------------------------
# filter_pimc_teacher_subset.py
# --------------------------------------------------------------------------------------


def _teacher_record(
    *,
    decision_type: str,
    disagrees: bool,
    margin: float,
    ci_low: float,
    margin_z: float | None = 5.0,
) -> dict[str, Any]:
    """Record sintetico col solo necessario per il filtro (nessuna observation reale serve)."""
    return {
        "teacher": {
            "decision_type": decision_type,
            "search_diagnostics": {
                "margin": margin,
                "margin_ci95_low": ci_low,
                "margin_z": margin_z,
            },
        },
        "reference": {"disagrees_with_teacher": disagrees, "card_index": 1},
        "action": {"card_index": 0},
    }


def test_zero_variance_correction_is_kept(tmp_path: Path) -> None:
    """
    Regressione: una correzione a varianza nulla (SE=0 -> margin_z=None) ha margine e CI validi
    e deve essere TENUTA, non scartata come 'missing_diag'. Sono gli esempi più affidabili.
    """
    flt = _load_script_module("filter_pimc_teacher_subset.py", "filter_subset_zerovar_test")
    rec = _teacher_record(decision_type="search", disagrees=True, margin=20.0, ci_low=20.0, margin_z=None)
    config = flt.SubsetFilterConfig(min_margin=2.0, min_ci_low=0.0, weight_mode="margin_clip", clip_max=10.0)
    status, weight = flt.evaluate_record(rec, config)
    assert status == "kept"
    assert weight == 10.0

    # In modalità margin_z, varianza nulla = massima confidenza -> tetto clip_max, non 0.
    config_z = flt.SubsetFilterConfig(min_margin=2.0, min_ci_low=0.0, weight_mode="margin_z", clip_max=12.0)
    assert flt.compute_sample_weight(margin=20.0, margin_z=None, config=config_z) == 12.0


def test_filter_keeps_only_strong_reliable_search_disagreements(tmp_path: Path) -> None:
    """Solo search + disaccordo + margine forte + CI affidabile devono sopravvivere al filtro."""
    flt = _load_script_module("filter_pimc_teacher_subset.py", "filter_subset_test")
    records = [
        _teacher_record(decision_type="search", disagrees=True, margin=24.0, ci_low=22.0),  # kept
        _teacher_record(decision_type="search", disagrees=False, margin=24.0, ci_low=22.0),  # agree
        _teacher_record(decision_type="search", disagrees=True, margin=1.0, ci_low=0.5),  # low margin
        _teacher_record(decision_type="search", disagrees=True, margin=5.0, ci_low=-0.2),  # ci_low < 0
        _teacher_record(decision_type="fallback", disagrees=True, margin=30.0, ci_low=25.0),  # not search
        _teacher_record(decision_type="endgame_solver", disagrees=True, margin=30.0, ci_low=25.0),  # not search
    ]
    in_path = tmp_path / "diag.jsonl"
    out_path = tmp_path / "subset.jsonl"
    in_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")

    config = flt.SubsetFilterConfig(min_margin=2.0, min_ci_low=0.0, weight_mode="margin_clip", clip_max=10.0)
    summary = flt.filter_dataset(in_path, out_path, config)

    assert summary["counters"]["records_kept"] == 1
    assert summary["counters"]["drop_agree"] == 1
    assert summary["counters"]["drop_low_margin"] == 1
    assert summary["counters"]["drop_low_ci_low"] == 1
    assert summary["counters"]["drop_decision_type"] == 2

    kept = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(kept) == 1
    # margin 24 clampato a clip_max 10.
    assert kept[0]["sample_weight"] == 10.0
    assert kept[0]["subset_filter"]["weight_mode"] == "margin_clip"


def test_compute_sample_weight_modes() -> None:
    """Ogni modalità di pesatura deve comportarsi come documentato."""
    flt = _load_script_module("filter_pimc_teacher_subset.py", "filter_subset_weights_test")

    def w(mode: str, *, margin: float, margin_z: float = 4.0, clip_max: float = 10.0) -> float:
        cfg = flt.SubsetFilterConfig(min_margin=2.0, weight_mode=mode, clip_max=clip_max)
        return flt.compute_sample_weight(margin=margin, margin_z=margin_z, config=cfg)

    assert w("uniform", margin=24.0) == 1.0
    assert w("margin", margin=24.0) == 24.0
    assert w("margin_clip", margin=24.0) == 10.0  # clampato in alto
    assert w("margin_clip", margin=1.0) == 2.0  # clampato in basso a min_margin
    assert w("margin_z", margin=24.0, margin_z=18.0) == 18.0
    assert abs(w("log_margin", margin=np.e - 1.0) - 1.0) < 1e-6
