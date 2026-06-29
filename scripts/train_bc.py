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
Supportiamo due varianti (stesso encoder, stessa action mask):

1) `linear` (default)
   - regressione logistica multinomiale (softmax) con SGD
   - logits mascherati: il modello può scegliere solo tra carte in mano

2) `mlp`
   - MLP minimale con 1 hidden layer + ReLU + testa lineare su 40 azioni
   - ottimizzazione con Adam (più stabile del SGD puro su reti non-lineari)

Per distillazione/fine-tuning è possibile partire da un MLP esistente (`--init`)
e aggiungere un anchor congelato (`--bc-anchor`, `--bc-anchor-beta`) per limitare
la deriva rispetto al modello base. `--filter-disagree-with-model` permette di
allenare solo sugli esempi in cui il teacher corregge davvero quel modello base.

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

from briscola_ai.ai.encoding.card_action_space import card_dto_to_action_id
from briscola_ai.ai.encoding.observation_encoder import EncoderVersion, encode_observation_2p_with_version
from briscola_ai.ai.models import BCModelAgent, LoadedBCModel, MLPBCModel, load_bc_model_npz
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.observation import PlayerObservation


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


def _card_dto_to_domain(card: dict) -> Card:
    """Converte un CardDTO JSON in `domain.Card`."""
    if not isinstance(card, dict):
        raise TypeError(f"CardDTO atteso come dict, ottenuto: {type(card)}")
    suit = card.get("suit")
    number = card.get("number")
    if not isinstance(suit, str) or not isinstance(number, int):
        raise ValueError(f"CardDTO invalido: suit={suit!r} number={number!r}")
    try:
        rank = next(rank for rank in Rank if int(rank.number) == int(number))
    except StopIteration as exc:
        raise ValueError(f"Numero carta fuori range: {number}") from exc
    return Card(suit=Suit(suit), rank=rank)


def _observation_dto_to_player_observation(obs: dict) -> PlayerObservation:
    """
    Ricostruisce una `PlayerObservation` da ObservationDTO per confrontare un modello anchor.

    Serve solo per filtri dataset, quindi supporta il caso atteso: record action 2-player con
    `my_turn=true`. Non usa né ricostruisce informazione nascosta.
    """
    if obs.get("num_players") != 2:
        raise ValueError("Il filtro BC supporta solo observation 2-player")
    player_index = int(obs.get("my_index", 0))
    players = obs.get("players") or []
    if not isinstance(players, list) or len(players) != 2:
        raise ValueError("ObservationDTO senza lista players 2-player valida")

    table_cards = []
    for item in obs.get("table_cards") or []:
        if not isinstance(item, dict):
            continue
        table_cards.append((_card_dto_to_domain(item["card"]), int(item["player_index"])))

    current_turn = player_index if bool(obs.get("my_turn")) else -1
    return PlayerObservation(
        num_players=2,
        is_team_game=False,
        teams=None,
        player_index=player_index,
        player_name=str(players[player_index].get("name", f"player_{player_index}")),
        hand=tuple(_card_dto_to_domain(card) for card in (obs.get("my_hand") or [])),
        trump_card=_card_dto_to_domain(obs["trump_card"]) if isinstance(obs.get("trump_card"), dict) else None,
        deck_size=int(obs.get("cards_remaining_in_deck", 0)),
        table_cards=tuple(table_cards),
        current_turn=current_turn,
        first_player=table_cards[0][1] if table_cards else current_turn,
        game_over=bool(obs.get("game_over", False)),
        winner_index=None,
        winning_team=None,
        players_points=tuple(int(player.get("points", 0)) for player in players),
        players_hand_sizes=tuple(int(player.get("hand_size", 0)) for player in players),
        seen_cards_onehot=tuple(int(v) for v in (obs.get("seen_cards_onehot") or [0] * 40)),
        out_of_play_cards_onehot=tuple(int(v) for v in (obs.get("out_of_play_cards_onehot") or [0] * 40)),
    )


def _agent_action_id_from_observation(agent, obs: dict, *, rng: np.random.Generator) -> int:
    """Calcola l'action_id scelto da un agente su una ObservationDTO."""
    import random

    observation = _observation_dto_to_player_observation(obs)
    py_rng = random.Random(int(rng.integers(0, 2**32)))
    card_index = int(agent.choose_card_index(observation, rng=py_rng))
    hand = obs.get("my_hand") or []
    if not 0 <= card_index < len(hand):
        raise ValueError(f"Agente filtro ha prodotto card_index invalido: {card_index}")
    return card_dto_to_action_id(hand[card_index])


def _build_training_examples(
    path: Path,
    *,
    encoder_version: EncoderVersion,
    disagreement_agent=None,
    disagreement_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    disagreement_rng = np.random.default_rng(disagreement_seed)

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

        encoded = encode_observation_2p_with_version(obs, version=encoder_version)
        if not encoded.action_mask[y]:
            # Sanity: il target deve essere sempre una carta in mano.
            continue
        if disagreement_agent is not None:
            try:
                baseline_y = _agent_action_id_from_observation(disagreement_agent, obs, rng=disagreement_rng)
            except ValueError:
                continue
            if int(baseline_y) == int(y):
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


def _anchor_probs_batch(x: np.ndarray, mask: np.ndarray, anchor: LoadedBCModel) -> np.ndarray:
    """
    Distribuzione target dell'anchor su un batch.

    L'anchor è un modello `.npz` congelato. Lo valutiamo riga per riga perché può essere
    lineare o MLP; il costo è accettabile per il fine-tuning BC didattico.
    """
    if x.shape[0] == 0:
        return np.zeros((0, 40), dtype=np.float32)
    logits = np.asarray([anchor.logits(row) for row in x], dtype=np.float32)
    return _softmax(_masked_logits(logits, mask))


def _add_anchor_regularization(
    *,
    x: np.ndarray,
    mask: np.ndarray,
    probs: np.ndarray,
    dlogits: np.ndarray,
    anchor: LoadedBCModel | None,
    anchor_beta: float,
) -> float:
    """
    Aggiunge in-place il gradiente della CE verso un anchor congelato.

    Loss additiva:
        beta * CE(anchor_probs, policy_probs)

    Ritorna la CE non pesata, utile per includere il termine nella loss riportata.
    """
    if anchor is None or float(anchor_beta) <= 0.0 or x.shape[0] == 0:
        return 0.0
    anchor_probs = _anchor_probs_batch(x, mask, anchor)
    anchor_ce = -np.sum(anchor_probs.astype(np.float64) * np.log(probs.astype(np.float64) + 1e-12), axis=1).mean()
    dlogits += (float(anchor_beta) / float(x.shape[0])) * (probs - anchor_probs)
    dlogits[~mask] = 0.0
    return float(anchor_ce)


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


def _loss_and_grad_linear(
    x: np.ndarray,
    mask: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    *,
    anchor: LoadedBCModel | None = None,
    anchor_beta: float = 0.0,
):
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
    dlogits = probs.copy()
    dlogits[np.arange(bsz), y] -= 1.0
    dlogits /= float(bsz)
    # Azioni non valide: gradient ≈ 0 per costruzione (prob ~0), ma rendiamolo esplicito.
    dlogits[~mask] = 0.0
    anchor_ce = _add_anchor_regularization(
        x=x,
        mask=mask,
        probs=probs,
        dlogits=dlogits,
        anchor=anchor,
        anchor_beta=float(anchor_beta),
    )
    loss += float(anchor_beta) * anchor_ce

    grad_w = x.T @ dlogits  # (D, 40)
    grad_b = np.sum(dlogits, axis=0)  # (40,)
    return loss, grad_w, grad_b, masked


def _relu(x: np.ndarray) -> np.ndarray:
    """ReLU: max(0, x)."""
    return np.maximum(x, 0.0)


def _loss_and_grad_mlp(
    x: np.ndarray,
    mask: np.ndarray,
    y: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    *,
    weight_decay: float,
    anchor: LoadedBCModel | None = None,
    anchor_beta: float = 0.0,
):
    """
    Cross-entropy mascherata + gradienti per MLP 1-hidden-layer.

    Architettura:
    - z1 = x W1 + b1
    - h  = relu(z1)
    - logits = h W2 + b2
    - masked_logits = mask(logits)

    Nota:
    questa rete è volutamente minimale e serve solo ad introdurre non-linearità
    rispetto al modello lineare.
    """
    z1 = x @ w1 + b1  # (B, H)
    h = _relu(z1)  # (B, H)
    logits = h @ w2 + b2  # (B, 40)
    masked = _masked_logits(logits, mask)
    probs = _softmax(masked)

    bsz = x.shape[0]
    loss = -np.log(probs[np.arange(bsz), y] + 1e-12).mean()

    # L2 weight decay (solo sui pesi, non sui bias).
    if weight_decay > 0.0:
        loss += 0.5 * float(weight_decay) * (float(np.sum(w1 * w1)) + float(np.sum(w2 * w2)))

    dlogits = probs.copy()
    dlogits[np.arange(bsz), y] -= 1.0
    dlogits /= float(bsz)
    dlogits[~mask] = 0.0
    anchor_ce = _add_anchor_regularization(
        x=x,
        mask=mask,
        probs=probs,
        dlogits=dlogits,
        anchor=anchor,
        anchor_beta=float(anchor_beta),
    )
    loss += float(anchor_beta) * anchor_ce

    grad_w2 = h.T @ dlogits  # (H, 40)
    grad_b2 = np.sum(dlogits, axis=0)  # (40,)

    dh = dlogits @ w2.T  # (B, H)
    dz1 = dh * (z1 > 0.0)  # ReLU grad
    grad_w1 = x.T @ dz1  # (D, H)
    grad_b1 = np.sum(dz1, axis=0)  # (H,)

    if weight_decay > 0.0:
        grad_w1 += float(weight_decay) * w1
        grad_w2 += float(weight_decay) * w2

    return loss, grad_w1, grad_b1, grad_w2, grad_b2, masked


@dataclass
class AdamState:
    """Stato Adam per un singolo tensore."""

    m: np.ndarray
    v: np.ndarray


def _adam_init(param: np.ndarray) -> AdamState:
    """Inizializza `m` e `v` a zero, stessa shape del parametro."""
    return AdamState(m=np.zeros_like(param), v=np.zeros_like(param))


def _adam_update(
    param: np.ndarray,
    grad: np.ndarray,
    *,
    state: AdamState,
    lr: float,
    t: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    """
    Step Adam in-place.

    Nota:
    Implementazione didattica minimale (niente fancy features).
    """
    state.m = beta1 * state.m + (1.0 - beta1) * grad
    state.v = beta2 * state.v + (1.0 - beta2) * (grad * grad)
    m_hat = state.m / (1.0 - beta1**t)
    v_hat = state.v / (1.0 - beta2**t)
    param -= float(lr) * m_hat / (np.sqrt(v_hat) + eps)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Behavior Cloning (40 carte + action mask)")
    parser.add_argument("--data", required=True, help="Path JSONL (output di scripts/export_dataset.py)")
    parser.add_argument("--out", required=True, help="Path output modello (.npz)")
    parser.add_argument(
        "--encoder-version",
        choices=["v1", "v2", "v3"],
        default="v1",
        help=(
            "Versione encoder per observation 2-player. "
            "v1=istantaneo (248 dim), v2=v1 + seen_cards_onehot[40] (288 dim, storia pubblica), "
            "v3=v2 + feature strategiche aggregate (310 dim, solo engine domain)."
        ),
    )
    parser.add_argument(
        "--inference-overkill-guard",
        action="store_true",
        help=(
            "Salva nei metadati del modello un flag per abilitare, a inference-time, "
            "un post-processing anti-overkill: se stiamo per vincere con una briscola da secondi di mano, "
            "giochiamo automaticamente la briscola vincente minima disponibile."
        ),
    )
    parser.add_argument(
        "--model",
        choices=["linear", "mlp"],
        default="linear",
        help="Tipo modello: linear (softmax) oppure mlp (1 hidden layer + ReLU).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Dimensione hidden layer per `--model mlp`.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Numero epoche")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate. Default: 0.5 per linear (SGD), 1e-3 per mlp (Adam).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="L2 weight decay (solo per `--model mlp`, default: 0).",
    )
    parser.add_argument(
        "--init",
        default="",
        help="Warm-start da un modello `.npz` MLP compatibile (stesse feature); utile per fine-tuning da v6.",
    )
    parser.add_argument(
        "--bc-anchor",
        default="",
        help="Modello `.npz` congelato usato come anchor CE per preservare il comportamento base.",
    )
    parser.add_argument(
        "--bc-anchor-beta",
        type=float,
        default=0.0,
        help="Peso della regolarizzazione CE verso `--bc-anchor`. Default: 0.",
    )
    parser.add_argument(
        "--filter-disagree-with-model",
        default="",
        help=(
            "Tiene solo esempi in cui il target del dataset differisce dalla mossa scelta da questo modello `.npz`. "
            "Utile per fine-tuning da v6 su sole correzioni teacher."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed RNG")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Frazione validation (0..1)")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)
    encoder_version: EncoderVersion = str(args.encoder_version)
    init_path = str(args.init).strip()
    bc_anchor_path = str(args.bc_anchor).strip()
    filter_disagree_path = str(args.filter_disagree_with_model).strip()
    if float(args.bc_anchor_beta) < 0.0:
        raise ValueError("--bc-anchor-beta deve essere >= 0")
    if float(args.bc_anchor_beta) > 0.0 and not bc_anchor_path:
        raise ValueError("Se `--bc-anchor-beta > 0` devi impostare anche `--bc-anchor <path.npz>`.")

    # Metadati UI (opzionali ma utili per la selezione del modello in frontend).
    #
    # Se presenti nel `metadata_json`, la UI può mostrare:
    # - un label sintetico (per il dropdown)
    # - una descrizione breve in italiano (per la help text)
    #
    # Nota: sono best-effort e non influenzano il funzionamento del modello.
    def _make_ui_metadata(*, model: str) -> tuple[str, str]:
        dataset = data_path.name
        epochs = int(args.epochs)
        lr = float(args.lr) if args.lr is not None else (0.5 if model == "linear" else 1e-3)
        enc_hint = " (encoder v2, storia pubblica)" if encoder_version == "v2" else ""

        if model == "linear":
            label = f"BC lineare{enc_hint} (epoche {epochs})"
            description_it = (
                "Behavior Cloning (supervised): modello lineare softmax su 40 carte + action mask. "
                f"Addestrato su dataset `{dataset}` (epoche={epochs}, lr={lr:g})."
            )
            return label, description_it

        hidden_dim = int(args.hidden_dim)
        label = f"BC MLP{enc_hint} (epoche {epochs})"
        description_it = (
            "Behavior Cloning (supervised): MLP (1 hidden layer + ReLU) su 40 carte + action mask. "
            f"Addestrato su dataset `{dataset}` (hidden={hidden_dim}, epoche={epochs}, lr={lr:g})."
        )
        return label, description_it

    disagreement_agent = BCModelAgent.from_npz(filter_disagree_path) if filter_disagree_path else None
    x_all, mask_all, y_all = _build_training_examples(
        data_path,
        encoder_version=encoder_version,
        disagreement_agent=disagreement_agent,
        disagreement_seed=int(args.seed),
    )

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

    lr = float(args.lr) if args.lr is not None else (0.5 if args.model == "linear" else 1e-3)

    metrics: list[TrainMetrics] = []
    ui_label, ui_description_it = _make_ui_metadata(model=str(args.model))
    init_model: LoadedBCModel | None = None
    if init_path:
        init_model = load_bc_model_npz(Path(init_path))
        if int(init_model.feature_dim) != int(d):
            raise ValueError(
                "Warm-start non compatibile con l'encoder corrente: "
                f"init.feature_dim={int(init_model.feature_dim)} dataset.feature_dim={int(d)}."
            )

    bc_anchor: LoadedBCModel | None = None
    if bc_anchor_path:
        bc_anchor = load_bc_model_npz(Path(bc_anchor_path))
        if int(bc_anchor.feature_dim) != int(d):
            raise ValueError(
                "BC-anchor non compatibile con l'encoder corrente: "
                f"anchor.feature_dim={int(bc_anchor.feature_dim)} dataset.feature_dim={int(d)}."
            )

    if args.model == "linear":
        if init_model is not None:
            raise ValueError("`--init` per `train_bc.py` è supportato solo con `--model mlp`.")
        # Inizializzazione pesi (piccola).
        w = rng.normal(loc=0.0, scale=0.01, size=(d, 40)).astype(np.float32)
        b = np.zeros((40,), dtype=np.float32)

        for epoch in range(1, args.epochs + 1):
            train_losses = []
            train_accs = []
            for batch in _iter_minibatches(x_train, mask_train, y_train, batch_size=args.batch_size, rng=rng):
                loss, grad_w, grad_b, masked = _loss_and_grad_linear(
                    batch.x,
                    batch.mask,
                    batch.y,
                    w,
                    b,
                    anchor=bc_anchor,
                    anchor_beta=float(args.bc_anchor_beta),
                )
                w -= float(lr) * grad_w
                b -= float(lr) * grad_b

                train_losses.append(loss)
                train_accs.append(_accuracy(masked, batch.y))

            # Val (full batch: semplice).
            val_loss, _, _, val_masked = _loss_and_grad_linear(
                x_val,
                mask_val,
                y_val,
                w,
                b,
                anchor=bc_anchor,
                anchor_beta=float(args.bc_anchor_beta),
            )
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
            "label": ui_label,
            "description_it": ui_description_it,
            "feature_dim": int(d),
            "action_dim": 40,
            "seed": int(args.seed),
            "data_path": str(data_path),
            "encoder": f"encode_observation_2p:{encoder_version}",
            "encoder_version": encoder_version,
            "inference_overkill_guard": bool(args.inference_overkill_guard),
            "init": init_path or None,
            "bc_anchor_path": bc_anchor_path or None,
            "bc_anchor_beta": float(args.bc_anchor_beta),
            "filter_disagree_with_model": filter_disagree_path or None,
            "train": {"model": "linear", "optimizer": "sgd", "lr": float(lr), "epochs": int(args.epochs)},
            "metrics": [asdict(metric) for metric in metrics],
        }
        np.savez(out_path, w=w, b=b, metadata_json=json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"Saved model: {out_path}")
        return 0

    if init_model is None and args.hidden_dim <= 0:
        raise ValueError("--hidden-dim deve essere > 0")

    if init_model is not None:
        if not isinstance(init_model, MLPBCModel):
            raise ValueError("`--init` deve puntare a un modello MLP (chiavi w1/b1/w2/b2).")
        w1 = init_model.w1.copy()
        b1 = init_model.b1.copy()
        w2 = init_model.w2.copy()
        b2 = init_model.b2.copy()
        hdim = int(w1.shape[1])
    else:
        # MLP: inizializzazione piccola (stile Xavier/He molto semplificata).
        hdim = int(args.hidden_dim)
        w1 = (rng.normal(loc=0.0, scale=0.02, size=(d, hdim))).astype(np.float32)
        b1 = np.zeros((hdim,), dtype=np.float32)
        w2 = (rng.normal(loc=0.0, scale=0.02, size=(hdim, 40))).astype(np.float32)
        b2 = np.zeros((40,), dtype=np.float32)

    # Adam state.
    st_w1 = _adam_init(w1)
    st_b1 = _adam_init(b1)
    st_w2 = _adam_init(w2)
    st_b2 = _adam_init(b2)
    t = 0

    for epoch in range(1, args.epochs + 1):
        train_losses = []
        train_accs = []
        for batch in _iter_minibatches(x_train, mask_train, y_train, batch_size=args.batch_size, rng=rng):
            t += 1
            loss, gw1, gb1, gw2, gb2, masked = _loss_and_grad_mlp(
                batch.x,
                batch.mask,
                batch.y,
                w1,
                b1,
                w2,
                b2,
                weight_decay=float(args.weight_decay),
                anchor=bc_anchor,
                anchor_beta=float(args.bc_anchor_beta),
            )
            _adam_update(w1, gw1, state=st_w1, lr=lr, t=t)
            _adam_update(b1, gb1, state=st_b1, lr=lr, t=t)
            _adam_update(w2, gw2, state=st_w2, lr=lr, t=t)
            _adam_update(b2, gb2, state=st_b2, lr=lr, t=t)

            train_losses.append(loss)
            train_accs.append(_accuracy(masked, batch.y))

        # Val (full batch: semplice).
        val_loss, _, _, _, _, val_masked = _loss_and_grad_mlp(
            x_val,
            mask_val,
            y_val,
            w1,
            b1,
            w2,
            b2,
            weight_decay=float(args.weight_decay),
            anchor=bc_anchor,
            anchor_beta=float(args.bc_anchor_beta),
        )
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
        "format": "mlp_bc_v1",
        "label": ui_label,
        "description_it": ui_description_it,
        "feature_dim": int(d),
        "hidden_dim": int(hdim),
        "action_dim": 40,
        "seed": int(args.seed),
        "data_path": str(data_path),
        "encoder": f"encode_observation_2p:{encoder_version}",
        "encoder_version": encoder_version,
        "inference_overkill_guard": bool(args.inference_overkill_guard),
        "init": init_path or None,
        "bc_anchor_path": bc_anchor_path or None,
        "bc_anchor_beta": float(args.bc_anchor_beta),
        "filter_disagree_with_model": filter_disagree_path or None,
        "train": {
            "model": "mlp",
            "optimizer": "adam",
            "lr": float(lr),
            "epochs": int(args.epochs),
            "weight_decay": float(args.weight_decay),
        },
        "metrics": [asdict(metric) for metric in metrics],
    }
    np.savez(
        out_path,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        metadata_json=json.dumps(payload, ensure_ascii=False, indent=2),
    )
    print(f"Saved model: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
