#!/usr/bin/env python3
"""
Allena la belief network (Fase 2): MLP 369 -> H -> 40 sigmoid, BCE mascherata sulle ignote.

Gate offline (stampati a ogni epoca e nel summary finale):
- `val_bce`: BCE mascherata sulla validation (split PER-PARTITA: i record della stessa
  partita sono correlati, non devono attraversare lo split);
- `uniform_bce`: baseline "tiro a caso" — per ogni record, p = opp_hand_size / num_ignote
  su tutte le ignote. La belief DEVE batterla nettamente, altrimenti kill (vedi piano §5.1);
- `topk_recall`: frazione delle carte VERE dell'avversario tra le prime k=opp_hand_size
  predette (baseline attesa dell'uniforme: k / num_ignote).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from briscola_ai.versioning import get_code_version


def _bce_masked(probs: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """BCE media sulle sole posizioni mascherate (carte ignote)."""
    eps = 1e-7
    p = np.clip(probs, eps, 1.0 - eps)
    losses = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)) * mask
    denom = float(mask.sum())
    return float(losses.sum() / denom) if denom > 0 else 0.0


def _uniform_probs(mask: np.ndarray, opp_hand_size: np.ndarray) -> np.ndarray:
    """Baseline uniforme: per ogni record, p = k / n_ignote sulle posizioni ignote."""
    n_unknown = mask.sum(axis=1, keepdims=True).astype(np.float64)
    p = np.divide(
        opp_hand_size[:, None].astype(np.float64),
        n_unknown,
        out=np.zeros_like(n_unknown),
        where=n_unknown > 0,
    )
    return p * mask


def _topk_recall(probs: np.ndarray, y: np.ndarray, mask: np.ndarray, opp_hand_size: np.ndarray) -> float:
    """Frazione media delle carte vere dell'avversario tra le prime k predette (k = |mano|)."""
    scores = np.where(mask > 0, probs, -1.0)
    order = np.argsort(-scores, axis=1)
    hits = 0.0
    total = 0.0
    for i in range(probs.shape[0]):
        k = int(opp_hand_size[i])
        if k <= 0:
            continue
        topk = order[i, :k]
        hits += float(y[i, topk].sum())
        total += k
    return hits / total if total > 0 else 0.0


def _forward(x: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray):
    """Forward MLP: ritorna (probs, hidden) per il backward."""
    h = np.maximum(x @ w1 + b1, 0.0)
    logits = h @ w2 + b2
    probs = 1.0 / (1.0 + np.exp(-logits))
    return probs, h


def main() -> int:
    parser = argparse.ArgumentParser(description="Allena la belief network su dataset belief .npz")
    parser.add_argument("--data", required=True, help="Dataset .npz da generate_belief_dataset.py")
    parser.add_argument("--out", required=True, help="Path del belief model .npz da salvare")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Neuroni hidden layer (default 128)")
    parser.add_argument("--epochs", type=int, default=20, help="Epoche di training (default 20)")
    parser.add_argument("--batch-size", type=int, default=512, help="Dimensione minibatch (default 512)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate Adam (default 1e-3)")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Frazione partite in validation (default 0.1)")
    parser.add_argument("--seed", type=int, default=0, help="Seed RNG per init/shuffle (riproducibilita')")
    args = parser.parse_args()

    with np.load(args.data) as data:
        x = np.asarray(data["x"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.float32)
        mask = np.asarray(data["unknown"], dtype=np.float32)
        opp_hand_size = np.asarray(data["opp_hand_size"], dtype=np.int64)
        game_index = np.asarray(data["game_index"], dtype=np.int64)
        dataset_meta = json.loads(str(data["metadata_json"])) if "metadata_json" in data else {}

    if x.shape[0] == 0:
        raise SystemExit("Dataset vuoto")
    feature_dim = int(x.shape[1])

    # Split PER-PARTITA: ogni partita finisce interamente in train o in val.
    modulo = max(2, round(1.0 / max(args.val_frac, 1e-6)))
    is_val = (game_index % modulo) == 0
    x_tr, y_tr, m_tr = x[~is_val], y[~is_val], mask[~is_val]
    x_va, y_va, m_va, k_va = x[is_val], y[is_val], mask[is_val], opp_hand_size[is_val]
    print(f"record: train={x_tr.shape[0]} val={x_va.shape[0]} | feature_dim={feature_dim}")

    rng = np.random.default_rng(args.seed)
    hidden = int(args.hidden_dim)
    w1 = rng.normal(0.0, np.sqrt(2.0 / feature_dim), size=(feature_dim, hidden)).astype(np.float32)
    b1 = np.zeros(hidden, dtype=np.float32)
    w2 = np.zeros((hidden, 40), dtype=np.float32)
    b2 = np.zeros(40, dtype=np.float32)

    # Adam manuale (stessa famiglia degli altri trainer del progetto).
    params = [w1, b1, w2, b2]
    m_state = [np.zeros_like(p) for p in params]
    v_state = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    adam_t = 0

    uniform_va = _uniform_probs(m_va, k_va)
    uniform_bce = _bce_masked(uniform_va, y_va, m_va)
    uniform_recall = _topk_recall(uniform_va, y_va, m_va, k_va)
    print(f"baseline uniforme | val_bce={uniform_bce:.4f} | topk_recall={uniform_recall:.4f}")

    best = {"epoch": -1, "val_bce": float("inf"), "weights": None, "topk_recall": 0.0}
    started = time.perf_counter()
    for epoch in range(1, int(args.epochs) + 1):
        order = rng.permutation(x_tr.shape[0])
        for start in range(0, len(order), int(args.batch_size)):
            idx = order[start : start + int(args.batch_size)]
            xb, yb, mb = x_tr[idx], y_tr[idx], m_tr[idx]
            probs, h = _forward(xb, w1, b1, w2, b2)

            denom = max(float(mb.sum()), 1.0)
            dlogits = (probs - yb) * mb / denom  # gradiente BCE mascherata
            dw2 = h.T @ dlogits
            db2 = dlogits.sum(axis=0)
            dh = dlogits @ w2.T
            dh[h <= 0.0] = 0.0
            dw1 = xb.T @ dh
            db1 = dh.sum(axis=0)

            adam_t += 1
            for p, g, ms, vs in zip(params, [dw1, db1, dw2, db2], m_state, v_state, strict=True):
                ms[:] = beta1 * ms + (1 - beta1) * g
                vs[:] = beta2 * vs + (1 - beta2) * (g * g)
                m_hat = ms / (1 - beta1**adam_t)
                v_hat = vs / (1 - beta2**adam_t)
                p -= args.lr * m_hat / (np.sqrt(v_hat) + eps)

        probs_va, _ = _forward(x_va, w1, b1, w2, b2)
        val_bce = _bce_masked(probs_va, y_va, m_va)
        recall = _topk_recall(probs_va, y_va, m_va, k_va)
        marker = ""
        if val_bce < best["val_bce"]:
            best = {
                "epoch": epoch,
                "val_bce": val_bce,
                "topk_recall": recall,
                "weights": [p.copy() for p in params],
            }
            marker = " *best"
        print(f"epoch {epoch:02d} | val_bce={val_bce:.4f} | topk_recall={recall:.4f}{marker}")

    assert best["weights"] is not None
    w1, b1, w2, b2 = best["weights"]
    elapsed = time.perf_counter() - started

    metadata = {
        "format": "belief_mlp_v1",
        "encoder_version": str(dataset_meta.get("encoder_version", "v4")),
        "feature_dim": feature_dim,
        "hidden_dim": hidden,
        "target": "opponent_hand",
        "train": {
            "optimizer": "adam",
            "lr": float(args.lr),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "val_frac": float(args.val_frac),
            "seed": int(args.seed),
            "best_epoch": int(best["epoch"]),
            "val_bce": float(best["val_bce"]),
            "val_topk_recall": float(best["topk_recall"]),
            "uniform_bce": float(uniform_bce),
            "uniform_topk_recall": float(uniform_recall),
        },
        "dataset": dataset_meta,
        "code_version": get_code_version(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, w1=w1, b1=b1, w2=w2, b2=b2, metadata_json=json.dumps(metadata))
    print(
        f"Salvato {out_path} | best_epoch={best['epoch']} | val_bce={best['val_bce']:.4f} "
        f"(uniforme {uniform_bce:.4f}) | topk_recall={best['topk_recall']:.4f} "
        f"(uniforme {uniform_recall:.4f}) | {elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
