#!/usr/bin/env python3
"""
Training didattico: Behavior Cloning (supervised) con spazio azioni 40 carte + action mask.

Obiettivo
---------
Allenare un primo modello semplice che imita un "teacher" (es. `heuristic_v1`)
usando esempi (observation -> action).

Input dati
----------
Questo script legge un JSONL esportato da `scripts/export_dataset.py`.
Ogni record contiene:
- `observation` (ObservationDTO) prima dell'azione
- `action.card_index` (indice nella mano) giocato dal player

Da questi due campi ricaviamo:
- target `y`: action_id in [0, 39] (carta canonica) corrispondente alla carta giocata
- mask `m`: action mask in [0, 39] che abilita solo le carte in mano

Modello
-------
Baseline estremamente semplice:
- regressione logistica multinomiale (softmax) con SGD
- logits mascherati: il modello può scegliere solo tra carte in mano

Nota didattica:
Questo NON è un modello "forte". Serve come primo passo:
encoder -> dataset -> training -> salvataggio -> (in futuro) integrazione in `evaluate_agents.py`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from briscola_ai.ai.training.card_action_space import card_dto_to_action_id
from briscola_ai.ai.training.observation_encoder import encode_observation_2p


@dataclass(frozen=True)
class Batch:
    """Batch di training: feature, mask, target."""

    x: np.ndarray  # (B, D)
    mask: np.ndarray  # (B, 40) boolean
    y: np.ndarray  # (B,) int in [0, 39]


def _iter_jsonl(path: Path):
    """Itera record JSON per riga (streaming)."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _build_training_examples(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Carica esempi dal JSONL export.

    Ritorna:
    - X: (N, D) float32
    - M: (N, 40) bool
    - y: (N,) int64
    """
    features_list: list[list[float]] = []
    masks_list: list[list[bool]] = []
    targets: list[int] = []

    for rec in _iter_jsonl(path):
        obs = rec.get("observation")
        action = rec.get("action")
        if obs is None or action is None:
            continue

        # Filtriamo solo 2-player per il primo modello didattico.
        if obs.get("num_players") != 2:
            continue

        card_index = action.get("card_index")
        if not isinstance(card_index, int):
            continue

        my_hand = obs.get("my_hand") or []
        if not isinstance(my_hand, list):
            continue
        if card_index < 0 or card_index >= len(my_hand):
            continue

        played_card = my_hand[card_index]
        y = card_dto_to_action_id(played_card)

        encoded = encode_observation_2p(obs)
        if not encoded.action_mask[y]:
            # Sanity: il target deve essere sempre una carta in mano.
            continue

        features_list.append(encoded.features)
        masks_list.append(encoded.action_mask)
        targets.append(y)

    if not features_list:
        raise ValueError("Nessun esempio valido trovato: controlla input JSONL e filtri (2-player).")

    x = np.asarray(features_list, dtype=np.float32)
    m = np.asarray(masks_list, dtype=bool)
    y_arr = np.asarray(targets, dtype=np.int64)
    return x, m, y_arr


def _masked_logits(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Applica la mask ai logits.

    Azioni non valide -> logits molto negativi (≈ -inf), così la softmax assegna prob ~0.
    """
    very_negative = -1e9
    out = logits.copy()
    out[~mask] = very_negative
    return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax numericamente stabile su ultima dimensione."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _accuracy(masked_logits: np.ndarray, y: np.ndarray) -> float:
    pred = np.argmax(masked_logits, axis=1)
    return float(np.mean(pred == y))


def _iter_minibatches(x: np.ndarray, m: np.ndarray, y: np.ndarray, *, batch_size: int, rng: np.random.Generator):
    idx = np.arange(x.shape[0])
    rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start : start + batch_size]
        yield Batch(x=x[batch_idx], mask=m[batch_idx], y=y[batch_idx])


@dataclass
class TrainMetrics:
    """Metriche per monitorare training/val."""

    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


def _loss_and_grad(x: np.ndarray, mask: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray):
    """
    Cross-entropy mascherata + gradienti per softmax linear.

    - logits = xW + b
    - logits mascherati: solo azioni in mano
    - loss = -log p(y)
    """
    logits = x @ w + b  # (B, 40)
    masked = _masked_logits(logits, mask)
    probs = _softmax(masked)

    # Loss
    bsz = x.shape[0]
    loss = -np.log(probs[np.arange(bsz), y] + 1e-12).mean()

    # Gradienti
    dlogits = probs
    dlogits[np.arange(bsz), y] -= 1.0
    dlogits /= float(bsz)

    grad_w = x.T @ dlogits  # (D, 40)
    grad_b = np.sum(dlogits, axis=0)  # (40,)
    return loss, grad_w, grad_b, masked


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Behavior Cloning (40 carte + action mask)")
    parser.add_argument("--data", required=True, help="Path JSONL (output di scripts/export_dataset.py)")
    parser.add_argument("--out", required=True, help="Path output modello (.npz)")
    parser.add_argument("--epochs", type=int, default=10, help="Numero epoche")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning rate (SGD)")
    parser.add_argument("--seed", type=int, default=0, help="Seed RNG")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Frazione validation (0..1)")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)

    x_all, mask_all, y_all = _build_training_examples(data_path)

    rng = np.random.default_rng(args.seed)
    n = x_all.shape[0]
    d = x_all.shape[1]

    # Split train/val (semplice e didattico).
    idx = np.arange(n)
    rng.shuffle(idx)
    val_size = int(round(n * float(args.val_frac)))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]

    x_train, mask_train, y_train = x_all[train_idx], mask_all[train_idx], y_all[train_idx]
    x_val, mask_val, y_val = x_all[val_idx], mask_all[val_idx], y_all[val_idx]

    # Inizializzazione pesi (piccola).
    w = rng.normal(loc=0.0, scale=0.01, size=(d, 40)).astype(np.float32)
    b = np.zeros((40,), dtype=np.float32)

    metrics: list[TrainMetrics] = []

    for epoch in range(1, args.epochs + 1):
        train_losses = []
        train_accs = []
        for batch in _iter_minibatches(x_train, mask_train, y_train, batch_size=args.batch_size, rng=rng):
            loss, grad_w, grad_b, masked = _loss_and_grad(batch.x, batch.mask, batch.y, w, b)
            w -= float(args.lr) * grad_w
            b -= float(args.lr) * grad_b

            train_losses.append(loss)
            train_accs.append(_accuracy(masked, batch.y))

        # Val (full batch: semplice).
        val_loss, _, _, val_masked = _loss_and_grad(x_val, mask_val, y_val, w, b)
        val_acc = _accuracy(val_masked, y_val)

        row = TrainMetrics(
            epoch=epoch,
            train_loss=float(np.mean(train_losses)),
            train_acc=float(np.mean(train_accs)),
            val_loss=float(val_loss),
            val_acc=float(val_acc),
        )
        metrics.append(row)
        print(
            f"epoch {epoch:02d} | "
            f"train loss {row.train_loss:.4f} acc {row.train_acc:.3f} | "
            f"val loss {row.val_loss:.4f} acc {row.val_acc:.3f}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "linear_softmax_bc_v1",
        "feature_dim": int(d),
        "action_dim": 40,
        "seed": int(args.seed),
        "data_path": str(data_path),
        "encoder": "encode_observation_2p:v1",
        "metrics": [asdict(metric) for metric in metrics],
    }
    np.savez(out_path, w=w, b=b, metadata_json=json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved model: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
