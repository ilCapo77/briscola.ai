#!/usr/bin/env python3
"""
Training RL didattico: Actor-Critic (A2C minimale) + reward shaping "trick delta".

Perché A2C
----------
REINFORCE (policy gradient puro) aggiorna la policy usando un return spesso molto rumoroso.
Un modo semplice per ridurre la varianza è aggiungere un *critic* che stima `V(s)` e
usare l'**advantage**:

  A(s,a) = G_t - V(s)

dove `G_t` è il return-to-go (somma dei reward futuri).

Reward shaping: "trick delta"
-----------------------------
In Briscola i punti cambiano solo quando si chiude una mano (trick). Se usiamo solo il reward finale,
il segnale arriva tardi. Qui rendiamo il reward più denso senza barare:

- definiamo un "time-step" come: **una scelta della policy** (turno della policy)
- reward dello step = delta di `(punti_policy - punti_opp)` accumulato fino al prossimo turno della policy
  (include quindi l'azione dell'avversario che chiude la mano, se necessario).

Anti-cheat
----------
La policy vede solo `PlayerObservation` (osservazione parziale lecita).

Warm-start consigliato
----------------------
Come per REINFORCE, conviene partire da un BC MLP teacher-only:

  python scripts/train_a2c.py \\
    --init ./data/bc_model_teacher_mlp.npz \\
    --out ./data/a2c_shaped.npz \\
    --opponent-mix heuristic_v1:0.7,random:0.2,greedy_points:0.1 \\
    --num-games 200000 --seat-fair --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from briscola_ai.ai.agents import Agent, build_agent
from briscola_ai.ai.encoding.card_action_space import action_id_from_suit_number
from briscola_ai.ai.encoding.observation_encoder import (
    FEATURE_DIM_2P_V1,
    FEATURE_DIM_2P_V2,
    EncoderVersion,
    encode_player_observation_2p,
    feature_dim_for_encoder_version,
)
from briscola_ai.ai.fast.evaluation import FAST_EVALUATION_AGENT_NAMES, choose_fast_card_index
from briscola_ai.ai.fast.observation_encoder import encode_fast_observation_2p
from briscola_ai.ai.fast.state_2p import Fast2PState, new_fast_2p_state, step_fast_2p
from briscola_ai.ai.models import BCModelAgent, LoadedBCModel, MLPBCModel, load_bc_model_npz
from briscola_ai.ai.numba.core import numba_agent_code
from briscola_ai.ai.numba.observation import (
    NumbaA2CBatch,
    NumbaA2CTrajectory,
    collect_a2c_batch_numba_2p,
    collect_a2c_trajectory_numba_2p,
    encode_fast_observation_numba_2p,
)
from briscola_ai.ai.training.opponent_mix import OpponentMixItem, parse_opponent_mix, sample_opponent_name
from briscola_ai.ai.training.policy_regularization import cross_entropy_from_probs, grad_ce_wrt_logits_from_probs
from briscola_ai.ai.training.reward_shaping import trump_overkill_penalty, trump_overkill_penalty_gap
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.observation import make_player_observation
from briscola_ai.domain.state import GameState, new_game_state


def _masked_logits_1d(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Maschera logits 1D: azioni non valide -> numero molto negativo."""
    very_negative = -1e9
    out = logits.copy()
    out[~mask] = very_negative
    return out


def _softmax_1d(logits: np.ndarray) -> np.ndarray:
    """Softmax 1D numericamente stabile."""
    shifted = logits - float(np.max(logits))
    exp = np.exp(shifted)
    return exp / float(np.sum(exp))


def _entropy(probs: np.ndarray) -> float:
    """Entropia (Shannon) per una distribuzione discreta."""
    p = probs + 1e-12
    return float(-np.sum(p * np.log(p)))


@dataclass
class AdamState:
    """Stato Adam per un singolo tensore."""

    m: np.ndarray
    v: np.ndarray


def _adam_init(param: np.ndarray) -> AdamState:
    """Inizializza stato Adam (m,v) con zeri, stessa shape del parametro."""
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
    """Aggiornamento Adam in-place."""
    state.m = beta1 * state.m + (1.0 - beta1) * grad
    state.v = beta2 * state.v + (1.0 - beta2) * (grad * grad)
    m_hat = state.m / (1.0 - beta1**t)
    v_hat = state.v / (1.0 - beta2**t)
    param -= float(lr) * m_hat / (np.sqrt(v_hat) + eps)


@dataclass(frozen=True, slots=True)
class OpponentPool:
    """Pool di avversari campionabili (opponent mix)."""

    items: list[OpponentMixItem]
    agents_by_name: dict[str, Agent]

    def sample(self, *, rng: np.random.Generator) -> Agent:
        """Campiona un avversario secondo la distribuzione."""
        name = sample_opponent_name(self.items, rng=rng)
        return self.agents_by_name[name]

    def to_metadata(self) -> list[dict[str, float | str]]:
        """Rappresentazione serializzabile (ordine stabile) per `metadata_json`."""
        return [{"name": item.name, "prob": float(item.prob)} for item in self.items]


@dataclass(frozen=True, slots=True)
class FastNumbaModelOpponent:
    """Opponent MLP caricato da `.npz` per il rollout A2C Numba."""

    agent: BCModelAgent
    model: MLPBCModel


@dataclass
class A2CPolicy:
    """
    Policy + critic con trunk condiviso (MLP 1 hidden layer + ReLU).

    - trunk: w1/b1
    - actor head: w2/b2 (logits su 40 azioni)
    - critic head: wv/bv (valore scalare)
    """

    w1: np.ndarray  # (D, H)
    b1: np.ndarray  # (H,)
    w2: np.ndarray  # (H, 40)
    b2: np.ndarray  # (40,)
    wv: np.ndarray  # (H,)
    bv: np.ndarray  # ()

    @property
    def feature_dim(self) -> int:
        return int(self.w1.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.w1.shape[1])

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Forward: ritorna (z1, h, logits, value)."""
        z1 = x @ self.w1 + self.b1
        h = np.maximum(z1, 0.0)
        logits = h @ self.w2 + self.b2
        value = float(h @ self.wv + self.bv)
        return z1, h, logits, value


@dataclass
class StepRecord:
    """Dati per un singolo step della policy (una decisione)."""

    x: np.ndarray  # (D,)
    z1: np.ndarray  # (H,)
    h: np.ndarray  # (H,)
    action_mask: np.ndarray  # (40,) bool
    probs: np.ndarray  # (40,)
    anchor_probs: np.ndarray | None
    anchor_ce: float
    action_id: int
    value_pred: float
    reward: float  # shaped reward (delta point diff / 120)


def _action_id_to_card_index(*, action_id: int, hand) -> int:
    """Converte action_id (carta canonica) in indice nella mano corrente."""
    for i, card in enumerate(hand):
        cid = action_id_from_suit_number(suit=card.suit.value, number=card.rank.number)
        if cid == action_id:
            return i
    raise ValueError(f"action_id {action_id} non corrisponde a nessuna carta nella mano (hand_size={len(hand)})")


def _points_diff(state: GameState, *, policy_seat: int) -> int:
    """Ritorna (punti_policy - punti_opp) dallo stato finale/corrente."""
    p0 = state.players[0].points
    p1 = state.players[1].points
    return int(p0 - p1) if policy_seat == 0 else int(p1 - p0)


def _play_one_game_2p_collect(
    *,
    policy: A2CPolicy,
    opponent: Agent,
    rng_opponent: random.Random,
    rng_action: np.random.Generator,
    game_seed: int,
    policy_seat: int,
    entropy_beta: float,
    encoder_version: EncoderVersion,
    overkill_penalty_beta: float,
    overkill_low_lead_points_max: int | None,
    overkill_penalty_mode: str,
    bc_anchor: LoadedBCModel | None,
    bc_anchor_beta: float,
) -> tuple[GameState, list[StepRecord], float]:
    """
    Simula una partita 2-player e colleziona la traiettoria vista come MDP "turno della policy".

    Ritorna:
    - stato finale
    - lista step della policy (uno per azione della policy)
    - entropia media (diagnostica)
    """
    state = new_game_state(num_players=2, seed=game_seed)
    traj: list[StepRecord] = []
    entropies: list[float] = []

    safety = 5000
    while not state.game_over and safety > 0:
        safety -= 1

        # Se non è turno della policy, avanza con l'avversario finché lo diventa.
        while not state.game_over and state.current_turn != policy_seat:
            obs_opp = make_player_observation(state, state.current_turn)
            card_index = opponent.choose_card_index(obs_opp, rng=rng_opponent)
            state, result = step(state, PlayCardAction(player_index=state.current_turn, card_index=card_index))
            if result.error:
                raise RuntimeError(f"Errore dominio durante la simulazione: {result.error}")

        if state.game_over:
            break

        # Ora tocca alla policy: definiamo uno step dell'MDP.
        diff_before = _points_diff(state, policy_seat=policy_seat)
        obs = make_player_observation(state, policy_seat)
        encoded = encode_player_observation_2p(obs, version=encoder_version)

        x = np.asarray(encoded.features, dtype=np.float32)
        mask = np.asarray(encoded.action_mask, dtype=bool)
        if x.shape[0] != policy.feature_dim:
            raise ValueError(f"Feature dim mismatch: got={x.shape[0]} expected={policy.feature_dim}")

        z1, h, logits, value_pred = policy.forward(x)
        masked = _masked_logits_1d(logits, mask)
        probs = _softmax_1d(masked)
        entropies.append(_entropy(probs))
        action_id = int(rng_action.choice(40, p=probs))
        card_index = _action_id_to_card_index(action_id=action_id, hand=obs.hand)

        # BC-anchor: regolarizzazione "stay-close-to-teacher" (senza barare).
        #
        # Idea:
        # - l'anchor è un modello BC fisso (teacher distillato) che non aggiorniamo.
        # - la policy RL viene penalizzata se si allontana troppo dall'anchor (cross-entropy).
        #
        # Questo termine di loss agisce durante training, non a inference-time: quindi
        # se vedi meno overkill nei benchmark senza guard, significa che la policy ha
        # interiorizzato (almeno in parte) la preferenza.
        anchor_probs: np.ndarray | None = None
        anchor_ce: float = 0.0
        if bc_anchor is not None and float(bc_anchor_beta) > 0.0:
            anchor_logits = bc_anchor.logits(x)
            anchor_masked = _masked_logits_1d(anchor_logits, mask)
            anchor_probs = _softmax_1d(anchor_masked)
            anchor_ce = cross_entropy_from_probs(target_probs=anchor_probs, pred_probs=probs)

        # Reward shaping opzionale: penalità "overkill briscola" (soft).
        #
        # Importante:
        # questa penalità è calcolata SOLO da `PlayerObservation` (anti-cheat),
        # quindi non introduce scorciatoie basate su informazione nascosta.
        if overkill_penalty_mode == "flat":
            extra_penalty = trump_overkill_penalty(
                obs,
                chosen_card_index=card_index,
                beta=float(overkill_penalty_beta),
                low_lead_points_max=overkill_low_lead_points_max,
            )
        elif overkill_penalty_mode == "gap":
            extra_penalty = trump_overkill_penalty_gap(
                obs,
                chosen_card_index=card_index,
                beta=float(overkill_penalty_beta),
                low_lead_points_max=overkill_low_lead_points_max,
            )
        else:
            raise ValueError(f"overkill_penalty_mode non supportato: {overkill_penalty_mode!r}")

        # Applica azione policy.
        state, result = step(state, PlayCardAction(player_index=policy_seat, card_index=card_index))
        if result.error:
            raise RuntimeError(f"Errore dominio durante la simulazione: {result.error}")

        # Avanza con l'avversario fino al prossimo turno della policy (o fine partita).
        while not state.game_over and state.current_turn != policy_seat:
            obs_opp = make_player_observation(state, state.current_turn)
            opp_card_index = opponent.choose_card_index(obs_opp, rng=rng_opponent)
            state, result = step(state, PlayCardAction(player_index=state.current_turn, card_index=opp_card_index))
            if result.error:
                raise RuntimeError(f"Errore dominio durante la simulazione: {result.error}")

        diff_after = _points_diff(state, policy_seat=policy_seat)
        reward = float(diff_after - diff_before) / 120.0 + float(extra_penalty)

        traj.append(
            StepRecord(
                x=x,
                z1=z1,
                h=h,
                action_mask=mask,
                probs=probs,
                anchor_probs=anchor_probs,
                anchor_ce=float(anchor_ce),
                action_id=action_id,
                value_pred=float(value_pred),
                reward=reward,
            )
        )

    if safety <= 0:
        raise RuntimeError("Loop di sicurezza: la partita non termina")

    avg_entropy = float(np.mean(entropies)) if entropies else 0.0
    return state, traj, avg_entropy


def _action_id_to_fast_card_index(*, action_id: int, hand: list[int]) -> int:
    """Converte action_id (che nel fast path coincide col card_id) in indice nella mano."""
    for i, card_id in enumerate(hand):
        if int(card_id) == int(action_id):
            return i
    raise ValueError(f"action_id {action_id} non corrisponde a nessuna carta fast nella mano (hand_size={len(hand)})")


def _points_diff_fast(state: Fast2PState, *, policy_seat: int) -> int:
    """Ritorna (punti_policy - punti_opp) dallo stato fast."""
    p0 = int(state.points[0])
    p1 = int(state.points[1])
    return p0 - p1 if policy_seat == 0 else p1 - p0


def _load_fast_numba_model_opponent(*, opponent_name: str, opponent_model_path: str) -> FastNumbaModelOpponent:
    """Carica un opponent `.npz` per `--rollout-engine fast --fast-rollout numba`."""
    if opponent_name == "best_a2c":
        agent = build_agent("best_a2c")
    elif opponent_name == "bc_model":
        if not opponent_model_path.strip():
            raise ValueError("`--opponent bc_model` richiede `--opponent-model <path.npz>`.")
        agent = build_agent("bc_model", model_path=Path(opponent_model_path.strip()))
    else:
        raise ValueError(f"Opponent modello non supportato nel fast rollout Numba: {opponent_name!r}")

    if not isinstance(agent, BCModelAgent):
        raise ValueError(f"Opponent {opponent_name!r} non ha prodotto un BCModelAgent.")
    if not isinstance(agent.model, MLPBCModel):
        raise ValueError("Il fast rollout Numba supporta per ora solo opponent `.npz` MLP (w1/b1/w2/b2).")
    if int(agent.model.feature_dim) not in (int(FEATURE_DIM_2P_V1), int(FEATURE_DIM_2P_V2)):
        raise ValueError(
            "Opponent MLP non compatibile: "
            f"feature_dim={int(agent.model.feature_dim)} atteso {int(FEATURE_DIM_2P_V1)} o {int(FEATURE_DIM_2P_V2)}."
        )
    return FastNumbaModelOpponent(agent=agent, model=agent.model)


def _numba_batch_trajectory_at(batch: NumbaA2CBatch, index: int) -> NumbaA2CTrajectory:
    """Estrae una traiettoria dal batch Numba usando view sulle righe valide."""
    count = int(batch.step_counts[index])
    return NumbaA2CTrajectory(
        policy_points=int(batch.policy_points[index]),
        opponent_points=int(batch.opponent_points[index]),
        winner=int(batch.winners[index]),
        avg_entropy=float(batch.avg_entropies[index]),
        xs=batch.xs[index, :count],
        z1s=batch.z1s[index, :count],
        hs=batch.hs[index, :count],
        action_masks=batch.action_masks[index, :count],
        probs=batch.probs[index, :count],
        action_ids=batch.action_ids[index, :count],
        value_preds=batch.value_preds[index, :count],
        rewards=batch.rewards[index, :count],
    )


def _play_one_fast_game_2p_collect(
    *,
    policy: A2CPolicy,
    opponent_name: str,
    rng_opponent: random.Random,
    rng_action: np.random.Generator,
    game_seed: int,
    policy_seat: int,
    encoder_version: EncoderVersion,
    fast_encoder: str,
    bc_anchor: LoadedBCModel | None,
    bc_anchor_beta: float,
) -> tuple[Fast2PState, list[StepRecord], float]:
    """
    Simula una partita A2C usando `fast_2p`.

    Limitazioni intenzionali:
    - supporta solo avversari tradotti su card id (`random`, `greedy_points`, `heuristic_v1`, `heuristic_v2`);
    - non applica ancora reward shaping anti-overkill, perché quello oggi dipende da `PlayerObservation`.
    """
    if opponent_name not in FAST_EVALUATION_AGENT_NAMES:
        supported = ", ".join(sorted(FAST_EVALUATION_AGENT_NAMES))
        raise ValueError(f"`--rollout-engine fast` supporta solo avversari: {supported}. Ottenuto: {opponent_name!r}")

    state = new_fast_2p_state(seed=game_seed)
    traj: list[StepRecord] = []
    entropies: list[float] = []

    # Storia pubblica per encoder v2: briscola scoperta + ogni carta giocata.
    seen = [0] * 40
    seen[state.trump_card] = 1
    # Carte fuori gioco per encoder v3: SOLO carte giocate (no briscola iniziale).
    out_of_play = [0] * 40

    safety = 5000
    while not state.game_over and safety > 0:
        safety -= 1

        while not state.game_over and state.current_turn != policy_seat:
            current = state.current_turn
            card_index = choose_fast_card_index(
                opponent_name,
                state,
                current,
                rng=rng_opponent,
                seen_cards_onehot=tuple(seen),
            )
            result = step_fast_2p(state, player_index=current, card_index=card_index)
            seen[result.played_card] = 1
            out_of_play[result.played_card] = 1

        if state.game_over:
            break

        diff_before = _points_diff_fast(state, policy_seat=policy_seat)
        if fast_encoder == "numba":
            encoded = encode_fast_observation_numba_2p(
                state,
                player_index=policy_seat,
                seen_cards_onehot=tuple(seen),
                out_of_play_cards_onehot=tuple(out_of_play),
                version=encoder_version,
            )
        elif fast_encoder == "python":
            encoded = encode_fast_observation_2p(
                state,
                player_index=policy_seat,
                seen_cards_onehot=tuple(seen),
                out_of_play_cards_onehot=tuple(out_of_play),
                version=encoder_version,
            )
        else:
            raise ValueError(f"fast_encoder non supportato: {fast_encoder!r}")
        x = np.asarray(encoded.features, dtype=np.float32)
        mask = np.asarray(encoded.action_mask, dtype=bool)
        if x.shape[0] != policy.feature_dim:
            raise ValueError(f"Feature dim mismatch: got={x.shape[0]} expected={policy.feature_dim}")

        z1, h, logits, value_pred = policy.forward(x)
        masked = _masked_logits_1d(logits, mask)
        probs = _softmax_1d(masked)
        entropies.append(_entropy(probs))
        action_id = int(rng_action.choice(40, p=probs))
        card_index = _action_id_to_fast_card_index(action_id=action_id, hand=state.hands[policy_seat])

        anchor_probs: np.ndarray | None = None
        anchor_ce = 0.0
        if bc_anchor is not None and float(bc_anchor_beta) > 0.0:
            anchor_logits = bc_anchor.logits(x)
            anchor_masked = _masked_logits_1d(anchor_logits, mask)
            anchor_probs = _softmax_1d(anchor_masked)
            anchor_ce = cross_entropy_from_probs(target_probs=anchor_probs, pred_probs=probs)

        result = step_fast_2p(state, player_index=policy_seat, card_index=card_index)
        seen[result.played_card] = 1
        out_of_play[result.played_card] = 1

        while not state.game_over and state.current_turn != policy_seat:
            current = state.current_turn
            opp_card_index = choose_fast_card_index(
                opponent_name,
                state,
                current,
                rng=rng_opponent,
                seen_cards_onehot=tuple(seen),
            )
            result = step_fast_2p(state, player_index=current, card_index=opp_card_index)
            seen[result.played_card] = 1
            out_of_play[result.played_card] = 1

        diff_after = _points_diff_fast(state, policy_seat=policy_seat)
        reward = float(diff_after - diff_before) / 120.0

        traj.append(
            StepRecord(
                x=x,
                z1=z1,
                h=h,
                action_mask=mask,
                probs=probs,
                anchor_probs=anchor_probs,
                anchor_ce=float(anchor_ce),
                action_id=action_id,
                value_pred=float(value_pred),
                reward=reward,
            )
        )

    if safety <= 0:
        raise RuntimeError("Loop di sicurezza: la partita fast non termina")

    avg_entropy = float(np.mean(entropies)) if entropies else 0.0
    return state, traj, avg_entropy


def _compute_returns(rewards: list[float], *, gamma: float) -> list[float]:
    """Return-to-go (Monte Carlo) con sconto `gamma` (default tipico: 1.0)."""
    out = [0.0] * len(rewards)
    g = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        g = rewards[i] + gamma * g
        out[i] = g
    return out


def _compute_returns_array(rewards: np.ndarray, *, gamma: float) -> np.ndarray:
    """Return-to-go su array NumPy, usato dal rollout Numba senza passare da `StepRecord`."""
    out = np.zeros_like(rewards, dtype=np.float32)
    g = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        g = float(rewards[i]) + gamma * g
        out[i] = g
    return out


@dataclass(frozen=True, slots=True)
class GradientStats:
    """Contatori prodotti dall'accumulo gradienti per il logging e la normalizzazione."""

    steps: int
    value_loss_sum: float
    anchor_ce_sum: float
    anchor_ce_count: int
    gbv: float


def _accumulate_numba_trajectory_grads(
    *,
    policy: A2CPolicy,
    xs: np.ndarray,
    z1s: np.ndarray,
    hs: np.ndarray,
    action_masks: np.ndarray,
    probs: np.ndarray,
    action_ids: np.ndarray,
    value_preds: np.ndarray,
    returns_to_go: np.ndarray,
    entropy_beta: float,
    value_coef: float,
    bc_anchor: LoadedBCModel | None,
    bc_anchor_beta: float,
    gw1: np.ndarray,
    gb1: np.ndarray,
    gw2: np.ndarray,
    gb2: np.ndarray,
    gwv: np.ndarray,
) -> GradientStats:
    """
    Accumula i gradienti di una traiettoria Numba usando batch matrix multiply.

    Il vecchio path faceva due `np.outer` per ogni decisione della policy. Qui costruiamo
    prima `dlogits` per tutti gli step della partita e poi accumuliamo:
    - `gw2 = H.T @ dlogits`
    - `gw1 = X.T @ dz1`

    La matematica resta la stessa del loop didattico per-step; cambia solo la forma
    computazionale, che riduce overhead Python e allocazioni temporanee nel path caldo.
    """
    steps = int(returns_to_go.shape[0])
    if steps == 0:
        return GradientStats(steps=0, value_loss_sum=0.0, anchor_ce_sum=0.0, anchor_ce_count=0, gbv=0.0)

    adv = returns_to_go.astype(np.float32, copy=False) - value_preds.astype(np.float32, copy=False)
    dlogits = probs.astype(np.float32, copy=True)
    row_idx = np.arange(steps)
    dlogits[row_idx, action_ids.astype(np.int64, copy=False)] -= np.float32(1.0)
    dlogits *= adv[:, None]

    beta = float(entropy_beta)
    if beta > 0.0:
        logp = np.log(probs.astype(np.float32, copy=False) + np.float32(1e-12))
        entropy_center = np.sum(probs * (logp + np.float32(1.0)), axis=1, keepdims=True)
        dent = probs * (logp + np.float32(1.0) - entropy_center)
        dlogits += np.float32(beta) * dent.astype(np.float32, copy=False)

    anchor_ce_sum = 0.0
    anchor_ce_count = 0
    anchor_beta = float(bc_anchor_beta)
    if anchor_beta > 0.0 and bc_anchor is not None:
        for i in range(steps):
            mask = action_masks[i]
            anchor_logits = bc_anchor.logits(xs[i])
            anchor_masked = _masked_logits_1d(anchor_logits, mask)
            anchor_probs = _softmax_1d(anchor_masked)
            anchor_ce_sum += cross_entropy_from_probs(target_probs=anchor_probs, pred_probs=probs[i])
            grad_anchor = grad_ce_wrt_logits_from_probs(
                pred_probs=probs[i],
                target_probs=anchor_probs,
                action_mask=mask,
            )
            dlogits[i] += np.float32(anchor_beta) * grad_anchor.astype(np.float32, copy=False)
            anchor_ce_count += 1

    # Actor head: somma degli outer product h x dlogits in una GEMM.
    gw2 += (hs.T @ dlogits).astype(np.float32, copy=False)
    gb2 += np.sum(dlogits, axis=0, dtype=np.float32)
    dh_policy = dlogits @ policy.w2.T

    # Critic head.
    value_error = value_preds.astype(np.float32, copy=False) - returns_to_go.astype(np.float32, copy=False)
    dv = (np.float32(value_coef) * value_error).astype(np.float32, copy=False)
    value_loss_sum = float(np.sum(0.5 * float(value_coef) * (value_error.astype(np.float64) ** 2)))

    gwv += (hs.T @ dv).astype(np.float32, copy=False)
    gbv = float(np.sum(dv, dtype=np.float64))
    dh_value = dv[:, None] * policy.wv[None, :]

    # Backprop sul trunk condiviso: somma degli outer product x x dz1 in una GEMM.
    dz1 = (dh_policy + dh_value) * (z1s > 0.0)
    gw1 += (xs.T @ dz1).astype(np.float32, copy=False)
    gb1 += np.sum(dz1, axis=0, dtype=np.float32)

    return GradientStats(
        steps=steps,
        value_loss_sum=value_loss_sum,
        anchor_ce_sum=anchor_ce_sum,
        anchor_ce_count=anchor_ce_count,
        gbv=gbv,
    )


def _accumulate_numba_batch_grads(
    *,
    policy: A2CPolicy,
    batch: NumbaA2CBatch,
    gamma: float,
    entropy_beta: float,
    value_coef: float,
    bc_anchor: LoadedBCModel | None,
    bc_anchor_beta: float,
    gw1: np.ndarray,
    gb1: np.ndarray,
    gw2: np.ndarray,
    gb2: np.ndarray,
    gwv: np.ndarray,
) -> GradientStats:
    """
    Appiattisce un batch Numba e accumula i gradienti in una sola chiamata batch.

    `NumbaA2CBatch` conserva un rettangolo `(batch, 20, ...)`; `step_counts` indica
    quante righe sono valide per ogni partita. Qui compattiamo solo quelle righe e
    riusiamo il backprop vettoriale su matrici 2D.
    """
    total_steps = int(np.sum(batch.step_counts, dtype=np.int64))
    if total_steps == 0:
        return GradientStats(steps=0, value_loss_sum=0.0, anchor_ce_sum=0.0, anchor_ce_count=0, gbv=0.0)

    feature_dim = int(batch.xs.shape[2])
    hidden_dim = int(batch.hs.shape[2])
    xs = np.empty((total_steps, feature_dim), dtype=np.float32)
    z1s = np.empty((total_steps, hidden_dim), dtype=np.float32)
    hs = np.empty((total_steps, hidden_dim), dtype=np.float32)
    action_masks = np.empty((total_steps, 40), dtype=bool)
    probs = np.empty((total_steps, 40), dtype=np.float32)
    action_ids = np.empty((total_steps,), dtype=np.int64)
    value_preds = np.empty((total_steps,), dtype=np.float32)
    returns_to_go = np.empty((total_steps,), dtype=np.float32)

    offset = 0
    for game_idx, raw_count in enumerate(batch.step_counts):
        count = int(raw_count)
        if count <= 0:
            continue
        sl = slice(offset, offset + count)
        xs[sl] = batch.xs[game_idx, :count]
        z1s[sl] = batch.z1s[game_idx, :count]
        hs[sl] = batch.hs[game_idx, :count]
        action_masks[sl] = batch.action_masks[game_idx, :count]
        probs[sl] = batch.probs[game_idx, :count]
        action_ids[sl] = batch.action_ids[game_idx, :count]
        value_preds[sl] = batch.value_preds[game_idx, :count]
        returns_to_go[sl] = _compute_returns_array(batch.rewards[game_idx, :count], gamma=gamma)
        offset += count

    return _accumulate_numba_trajectory_grads(
        policy=policy,
        xs=xs,
        z1s=z1s,
        hs=hs,
        action_masks=action_masks,
        probs=probs,
        action_ids=action_ids,
        value_preds=value_preds,
        returns_to_go=returns_to_go,
        entropy_beta=entropy_beta,
        value_coef=value_coef,
        bc_anchor=bc_anchor,
        bc_anchor_beta=bc_anchor_beta,
        gw1=gw1,
        gb1=gb1,
        gw2=gw2,
        gb2=gb2,
        gwv=gwv,
    )


@dataclass
class TrainMetrics:
    """Metriche aggregate (logging)."""

    iter: int
    games: int
    avg_return: float
    win_rate: float
    draw_rate: float
    avg_entropy: float
    value_loss: float
    avg_anchor_ce: float


def main() -> int:
    parser = argparse.ArgumentParser(description="Train RL A2C (MLP, 40 carte + action mask) con reward shaping")
    parser.add_argument("--out", required=True, help="Path output modello (.npz)")
    parser.add_argument("--init", default="", help="Warm-start da un modello `.npz` MLP (es. BC/RL).")
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
        "--upgrade-init-v1-to-v2",
        action="store_true",
        help=(
            "Se usi `--encoder-version v2` e `--init` è un modello v1, "
            "espande `w1` aggiungendo 40 righe a zero (warm-start compatibile)."
        ),
    )
    parser.add_argument(
        "--opponent",
        default="heuristic_v1",
        help=(
            "Nome avversario (se non usi --opponent-mix). "
            "Esempi: heuristic_v1, random, greedy_points, best_a2c "
            "(alias che carica `best_a2c.npz` dalla directory modelli)."
        ),
    )
    parser.add_argument(
        "--opponent-mix",
        default="",
        help=(
            "Miscela avversari: `name:weight,name:weight,...` "
            "(es. `heuristic_v1:0.7,random:0.2,greedy_points:0.1`). "
            "Se presente, sovrascrive `--opponent`."
        ),
    )
    parser.add_argument(
        "--opponent-model",
        default="",
        help=(
            "Path al modello `.npz` quando `--opponent bc_model` (supportato nel rollout domain e fast-rollout numba)."
        ),
    )
    parser.add_argument("--num-games", type=int, default=20000, help="Numero partite di training (2-player).")
    parser.add_argument("--seed", type=int, default=0, help="Seed RNG (riproducibilità).")
    parser.add_argument(
        "--rollout-engine",
        choices=["domain", "fast"],
        default="domain",
        help=(
            "Motore rollout training. `domain` è canonico e supporta tutti gli agenti; `fast` è sperimentale "
            "e supporta solo avversari fast-compatible random/greedy_points/heuristic_v1/heuristic_v2."
        ),
    )
    parser.add_argument(
        "--fast-encoder",
        choices=["python", "numba"],
        default="python",
        help=(
            "Encoder osservazione usato solo con `--rollout-engine fast --fast-rollout python`. "
            "`python` è il path stabile; `numba` usa il wrapper JIT sperimentale equivalente."
        ),
    )
    parser.add_argument(
        "--fast-rollout",
        choices=["python", "numba"],
        default="python",
        help=(
            "Loop rollout usato solo con `--rollout-engine fast`. "
            "`python` usa Fast2PState/list Python; `numba` raccoglie la traiettoria A2C in un core full-JIT."
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dim (se non si usa --init).")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate Adam.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="L2 weight decay (solo pesi).")
    parser.add_argument("--entropy-beta", type=float, default=5e-4, help="Entropia bonus (>=0).")
    parser.add_argument(
        "--bc-anchor",
        default="",
        help=(
            "Path a un modello `.npz` usato come anchor fisso (tipicamente un BC teacher). "
            "Se valorizzato e `--bc-anchor-beta > 0`, aggiunge una regolarizzazione (cross-entropy) "
            "che mantiene la policy vicina all'anchor (utile per preservare stile anti-overkill senza guard)."
        ),
    )
    parser.add_argument(
        "--bc-anchor-beta",
        type=float,
        default=0.0,
        help=("Peso (>=0) della regolarizzazione verso l'anchor BC. Valori tipici: 0.005..0.05. Se 0, disattivata."),
    )
    parser.add_argument(
        "--overkill-penalty-mode",
        choices=["flat", "gap"],
        default="flat",
        help=(
            "Modalità penalità overkill briscola: "
            "`flat` aggiunge `-beta` quando overkill, `gap` aggiunge `-beta * gap_norm` (più informativa)."
        ),
    )
    parser.add_argument(
        "--overkill-penalty-beta",
        type=float,
        default=0.0,
        help=(
            "Penalità flat (>=0) per scoraggiare 'overkill briscola' da secondi di mano. "
            "Se >0 e la policy vince con una briscola pur avendo una briscola vincente più economica, "
            "aggiungiamo `-beta` al reward (soft shaping)."
        ),
    )
    parser.add_argument(
        "--overkill-low-lead-points-max",
        type=int,
        default=2,
        help=(
            "Applica la penalità overkill solo se la carta avversaria sul tavolo vale "
            "al massimo questo numero di punti. "
            "Default: 2 (scarti o quasi)."
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
    parser.add_argument("--value-coef", type=float, default=0.5, help="Peso loss critic (MSE).")
    parser.add_argument("--gamma", type=float, default=1.0, help="Fattore di sconto per return-to-go (default: 1.0).")
    parser.add_argument("--update-every", type=int, default=20, help="Aggiorna i pesi ogni N partite (batch).")
    parser.add_argument("--log-every", type=int, default=200, help="Stampa metriche ogni N update.")
    parser.add_argument("--seat-fair", action="store_true", help="Alterna la seat della policy (riduce bias player0).")
    args = parser.parse_args()

    if args.num_games <= 0:
        raise ValueError("--num-games deve essere > 0")
    if args.hidden_dim <= 0:
        raise ValueError("--hidden-dim deve essere > 0")
    if args.update_every <= 0:
        raise ValueError("--update-every deve essere > 0")
    if args.log_every <= 0:
        raise ValueError("--log-every deve essere > 0")
    if float(args.gamma) <= 0.0 or float(args.gamma) > 1.0:
        raise ValueError("--gamma deve essere in (0,1]")
    if float(args.overkill_penalty_beta) < 0.0:
        raise ValueError("--overkill-penalty-beta deve essere >= 0")
    if int(args.overkill_low_lead_points_max) < 0:
        raise ValueError("--overkill-low-lead-points-max deve essere >= 0")
    if float(args.bc_anchor_beta) < 0.0:
        raise ValueError("--bc-anchor-beta deve essere >= 0")
    if float(args.bc_anchor_beta) > 0.0 and not str(args.bc_anchor).strip():
        raise ValueError("Se `--bc-anchor-beta > 0` devi impostare anche `--bc-anchor <path.npz>`.")
    rollout_engine = str(args.rollout_engine)
    fast_encoder = str(args.fast_encoder)
    fast_rollout = str(args.fast_rollout)
    if rollout_engine != "fast" and fast_encoder != "python":
        raise ValueError("`--fast-encoder numba` richiede `--rollout-engine fast`.")
    if rollout_engine != "fast" and fast_rollout != "python":
        raise ValueError("`--fast-rollout numba` richiede `--rollout-engine fast`.")
    if rollout_engine == "fast" and fast_rollout != "numba" and float(args.overkill_penalty_beta) > 0.0:
        raise ValueError(
            "`--rollout-engine fast --fast-rollout python` non supporta `--overkill-penalty-beta > 0`; "
            "usa `--fast-rollout numba` oppure `--rollout-engine domain`."
        )

    out_path = Path(args.out)
    encoder_version: EncoderVersion = str(args.encoder_version)
    rng_action = np.random.default_rng(args.seed)
    rng_game = np.random.default_rng(args.seed ^ 0x9E3779B9)
    rng_opponent_select = np.random.default_rng(args.seed ^ 0xA5A5A5A5)
    rng_opponent = random.Random(args.seed ^ 0xC0FFEE)

    opponent_pool: OpponentPool | None = None
    fast_numba_model_opponent: FastNumbaModelOpponent | None = None
    fast_numba_model_mix_name: str | None = None
    opponent_mix_raw = args.opponent_mix.strip()
    if opponent_mix_raw:
        items = parse_opponent_mix(opponent_mix_raw)
        if rollout_engine == "fast":
            model_mix_names = [item.name for item in items if item.name in {"best_a2c", "bc_model"}]
            unsupported = [
                item.name
                for item in items
                if item.name not in FAST_EVALUATION_AGENT_NAMES
                and not (fast_rollout == "numba" and item.name in {"best_a2c", "bc_model"})
            ]
            if unsupported:
                supported = ", ".join(sorted(FAST_EVALUATION_AGENT_NAMES))
                raise ValueError(
                    f"`--rollout-engine fast` supporta opponent mix con: {supported}; "
                    "`best_a2c`/`bc_model` sono supportati solo con `--fast-rollout numba`. "
                    f"Non supportati: {unsupported}"
                )
            if len(set(model_mix_names)) > 1:
                raise ValueError(
                    "`--opponent-mix` fast Numba supporta al massimo un tipo di opponent modello "
                    "(`best_a2c` oppure `bc_model`) per batch."
                )
            if fast_rollout == "numba" and model_mix_names:
                fast_numba_model_mix_name = model_mix_names[0]
                fast_numba_model_opponent = _load_fast_numba_model_opponent(
                    opponent_name=fast_numba_model_mix_name,
                    opponent_model_path=str(args.opponent_model),
                )
        agents_by_name = {}
        for item in items:
            if item.name == "bc_model":
                if fast_numba_model_opponent is None or fast_numba_model_mix_name != "bc_model":
                    raise ValueError("`bc_model` in `--opponent-mix` richiede fast Numba e `--opponent-model`.")
                agents_by_name[item.name] = fast_numba_model_opponent.agent
            else:
                agents_by_name[item.name] = build_agent(item.name)
        opponent_pool = OpponentPool(items=items, agents_by_name=agents_by_name)
        opponent = agents_by_name[items[0].name]
    else:
        opponent_name = str(args.opponent)
        if rollout_engine == "fast" and opponent_name in {"best_a2c", "bc_model"}:
            if fast_rollout != "numba":
                raise ValueError("Opponent `.npz` nel fast path richiede `--fast-rollout numba`.")
            fast_numba_model_opponent = _load_fast_numba_model_opponent(
                opponent_name=opponent_name,
                opponent_model_path=str(args.opponent_model),
            )
            opponent = fast_numba_model_opponent.agent
        elif rollout_engine == "fast" and opponent_name not in FAST_EVALUATION_AGENT_NAMES:
            supported = ", ".join(sorted(FAST_EVALUATION_AGENT_NAMES))
            raise ValueError(
                f"`--rollout-engine fast` supporta avversari fast-compatible ({supported}) "
                "oppure `best_a2c`/`bc_model` con `--fast-rollout numba`. "
                f"Ottenuto: {args.opponent!r}"
            )
        elif opponent_name == "bc_model":
            if not str(args.opponent_model).strip():
                raise ValueError("`--opponent bc_model` richiede `--opponent-model <path.npz>`.")
            opponent = build_agent("bc_model", model_path=Path(str(args.opponent_model).strip()))
        else:
            opponent = build_agent(opponent_name)

    # Inizializzazione policy/critic.
    target_feature_dim = int(feature_dim_for_encoder_version(encoder_version))
    if args.init.strip():
        loaded = load_bc_model_npz(Path(args.init))
        if not isinstance(loaded, MLPBCModel):
            raise ValueError("--init deve puntare a un modello MLP (w1/b1/w2/b2).")
        w1 = loaded.w1.copy()
        b1 = loaded.b1.copy()
        w2 = loaded.w2.copy()
        b2 = loaded.b2.copy()
        hdim = int(w1.shape[1])
        init_dim = int(w1.shape[0])
        if init_dim != target_feature_dim:
            if (
                bool(args.upgrade_init_v1_to_v2)
                and init_dim == int(FEATURE_DIM_2P_V1)
                and target_feature_dim == int(FEATURE_DIM_2P_V2)
            ):
                pad = np.zeros((target_feature_dim - init_dim, hdim), dtype=np.float32)
                w1 = np.vstack([w1, pad])
            else:
                raise ValueError(
                    "Feature dim mismatch tra `--init` e encoder scelto: "
                    f"init={init_dim} target={target_feature_dim} (encoder={encoder_version}). "
                    "Soluzioni: usa `--encoder-version` coerente, oppure abilita `--upgrade-init-v1-to-v2`."
                )
    else:
        hdim = int(args.hidden_dim)
        w1 = rng_action.normal(loc=0.0, scale=0.02, size=(target_feature_dim, hdim)).astype(np.float32)
        b1 = np.zeros((hdim,), dtype=np.float32)
        w2 = rng_action.normal(loc=0.0, scale=0.02, size=(hdim, 40)).astype(np.float32)
        b2 = np.zeros((40,), dtype=np.float32)

    # Critic head: inizializziamo vicino a zero (safe).
    wv = np.zeros((hdim,), dtype=np.float32)
    bv = np.float32(0.0)

    policy = A2CPolicy(w1=w1, b1=b1, w2=w2, b2=b2, wv=wv, bv=float(bv))

    # Anchor BC (teacher) opzionale: deve avere stessa feature_dim dell'encoder corrente.
    bc_anchor: LoadedBCModel | None = None
    bc_anchor_path = str(args.bc_anchor).strip()
    if bc_anchor_path:
        loaded_anchor = load_bc_model_npz(Path(bc_anchor_path))
        if int(loaded_anchor.feature_dim) != int(policy.feature_dim):
            raise ValueError(
                "BC-anchor non compatibile con l'encoder corrente: "
                f"anchor.feature_dim={int(loaded_anchor.feature_dim)} policy.feature_dim={int(policy.feature_dim)}. "
                "Suggerimento: usa `--encoder-version` coerente con l'anchor (v1=248, v2=288)."
            )
        bc_anchor = loaded_anchor

    # Adam state.
    st_w1 = _adam_init(policy.w1)
    st_b1 = _adam_init(policy.b1)
    st_w2 = _adam_init(policy.w2)
    st_b2 = _adam_init(policy.b2)
    st_wv = _adam_init(policy.wv)
    st_bv = _adam_init(np.asarray([policy.bv], dtype=np.float32))
    t = 0

    update_every = int(args.update_every)
    metrics: list[TrainMetrics] = []

    # Accumulo grad (batch).
    gw1 = np.zeros_like(policy.w1)
    gb1 = np.zeros_like(policy.b1)
    gw2 = np.zeros_like(policy.w2)
    gb2 = np.zeros_like(policy.b2)
    gwv = np.zeros_like(policy.wv)
    gbv = 0.0

    # Logging accumulators.
    returns_buf: list[float] = []
    wins = 0
    draws = 0
    entropies: list[float] = []
    grad_step_count = 0
    value_loss_sum = 0.0
    anchor_ce_sum = 0.0
    anchor_ce_count = 0

    num_games = int(args.num_games)
    use_numba_batch_rollout = rollout_engine == "fast" and fast_rollout == "numba"
    numba_batch: NumbaA2CBatch | None = None
    numba_batch_offset = 0
    for game_idx in range(1, num_games + 1):
        policy_seat = (game_idx % 2) if args.seat_fair else 0
        game_seed = 0 if use_numba_batch_rollout else int(rng_game.integers(0, 2**32))
        current_opponent = (
            opponent
            if use_numba_batch_rollout
            else (opponent_pool.sample(rng=rng_opponent_select) if opponent_pool is not None else opponent)
        )
        numba_traj_for_backprop = None

        if rollout_engine == "fast":
            if fast_rollout == "numba":
                if use_numba_batch_rollout:
                    if numba_batch is None or numba_batch_offset >= int(numba_batch.step_counts.shape[0]):
                        games_until_update = update_every - ((game_idx - 1) % update_every)
                        batch_size = min(games_until_update, num_games - game_idx + 1)
                        game_seeds = rng_game.integers(0, 2**32, size=batch_size, dtype=np.int64)
                        policy_seats = np.asarray(
                            [((game_idx + offset) % 2) if args.seat_fair else 0 for offset in range(batch_size)],
                            dtype=np.int64,
                        )
                        opponent_codes = None
                        opponent_model_enabled_flags = None
                        if opponent_pool is not None:
                            sampled_names = [
                                sample_opponent_name(opponent_pool.items, rng=rng_opponent_select)
                                for _ in range(batch_size)
                            ]
                            opponent_codes = np.asarray(
                                [
                                    0 if name == fast_numba_model_mix_name else numba_agent_code(name)
                                    for name in sampled_names
                                ],
                                dtype=np.int64,
                            )
                            opponent_model_enabled_flags = np.asarray(
                                [name == fast_numba_model_mix_name for name in sampled_names],
                                dtype=np.bool_,
                            )
                        numba_batch = collect_a2c_batch_numba_2p(
                            w1=policy.w1,
                            b1=policy.b1,
                            w2=policy.w2,
                            b2=policy.b2,
                            wv=policy.wv,
                            bv=float(policy.bv),
                            opponent_name=current_opponent.name,
                            opponent_w1=(
                                fast_numba_model_opponent.model.w1 if fast_numba_model_opponent is not None else None
                            ),
                            opponent_b1=(
                                fast_numba_model_opponent.model.b1 if fast_numba_model_opponent is not None else None
                            ),
                            opponent_w2=(
                                fast_numba_model_opponent.model.w2 if fast_numba_model_opponent is not None else None
                            ),
                            opponent_b2=(
                                fast_numba_model_opponent.model.b2 if fast_numba_model_opponent is not None else None
                            ),
                            opponent_overkill_guard=(
                                bool(fast_numba_model_opponent.agent.overkill_guard_enabled)
                                if fast_numba_model_opponent is not None
                                else False
                            ),
                            game_seeds=game_seeds,
                            policy_seats=policy_seats,
                            opponent_codes=opponent_codes,
                            opponent_model_enabled_flags=opponent_model_enabled_flags,
                            overkill_penalty_beta=float(args.overkill_penalty_beta),
                            overkill_low_lead_points_max=int(args.overkill_low_lead_points_max),
                            overkill_penalty_mode=str(args.overkill_penalty_mode),
                        )
                        batch_grad_stats = _accumulate_numba_batch_grads(
                            policy=policy,
                            batch=numba_batch,
                            gamma=float(args.gamma),
                            entropy_beta=float(args.entropy_beta),
                            value_coef=float(args.value_coef),
                            bc_anchor=bc_anchor,
                            bc_anchor_beta=float(args.bc_anchor_beta),
                            gw1=gw1,
                            gb1=gb1,
                            gw2=gw2,
                            gb2=gb2,
                            gwv=gwv,
                        )
                        gbv += batch_grad_stats.gbv
                        grad_step_count += batch_grad_stats.steps
                        value_loss_sum += batch_grad_stats.value_loss_sum
                        anchor_ce_sum += batch_grad_stats.anchor_ce_sum
                        anchor_ce_count += batch_grad_stats.anchor_ce_count
                        numba_batch_offset = 0
                    assert numba_batch is not None
                    numba_traj = _numba_batch_trajectory_at(numba_batch, numba_batch_offset)
                    numba_batch_offset += 1
                    if numba_batch_offset >= int(numba_batch.step_counts.shape[0]):
                        numba_batch = None
                else:
                    numba_traj = collect_a2c_trajectory_numba_2p(
                        w1=policy.w1,
                        b1=policy.b1,
                        w2=policy.w2,
                        b2=policy.b2,
                        wv=policy.wv,
                        bv=float(policy.bv),
                        opponent_name=current_opponent.name,
                        opponent_w1=(
                            fast_numba_model_opponent.model.w1 if fast_numba_model_opponent is not None else None
                        ),
                        opponent_b1=(
                            fast_numba_model_opponent.model.b1 if fast_numba_model_opponent is not None else None
                        ),
                        opponent_w2=(
                            fast_numba_model_opponent.model.w2 if fast_numba_model_opponent is not None else None
                        ),
                        opponent_b2=(
                            fast_numba_model_opponent.model.b2 if fast_numba_model_opponent is not None else None
                        ),
                        opponent_overkill_guard=(
                            bool(fast_numba_model_opponent.agent.overkill_guard_enabled)
                            if fast_numba_model_opponent is not None
                            else False
                        ),
                        game_seed=game_seed,
                        policy_seat=policy_seat,
                        overkill_penalty_beta=float(args.overkill_penalty_beta),
                        overkill_low_lead_points_max=int(args.overkill_low_lead_points_max),
                        overkill_penalty_mode=str(args.overkill_penalty_mode),
                    )
                numba_traj_for_backprop = None if use_numba_batch_rollout else numba_traj
                traj = []
                avg_entropy = float(numba_traj.avg_entropy)
                policy_points = int(numba_traj.policy_points)
                opp_points = int(numba_traj.opponent_points)
                ep_return = float(policy_points - opp_points) / 120.0
            else:
                final_fast_state, traj, avg_entropy = _play_one_fast_game_2p_collect(
                    policy=policy,
                    opponent_name=current_opponent.name,
                    rng_opponent=rng_opponent,
                    rng_action=rng_action,
                    game_seed=game_seed,
                    policy_seat=policy_seat,
                    encoder_version=encoder_version,
                    fast_encoder=fast_encoder,
                    bc_anchor=bc_anchor,
                    bc_anchor_beta=float(args.bc_anchor_beta),
                )
                ep_return = float(_points_diff_fast(final_fast_state, policy_seat=policy_seat)) / 120.0
                policy_points = int(final_fast_state.points[policy_seat])
                opp_points = int(final_fast_state.points[1 - policy_seat])
        else:
            final_state, traj, avg_entropy = _play_one_game_2p_collect(
                policy=policy,
                opponent=current_opponent,
                rng_opponent=rng_opponent,
                rng_action=rng_action,
                game_seed=game_seed,
                policy_seat=policy_seat,
                entropy_beta=float(args.entropy_beta),
                encoder_version=encoder_version,
                overkill_penalty_beta=float(args.overkill_penalty_beta),
                overkill_low_lead_points_max=int(args.overkill_low_lead_points_max),
                overkill_penalty_mode=str(args.overkill_penalty_mode),
                bc_anchor=bc_anchor,
                bc_anchor_beta=float(args.bc_anchor_beta),
            )
            ep_return = float(_points_diff(final_state, policy_seat=policy_seat)) / 120.0
            p0 = final_state.players[0].points
            p1 = final_state.players[1].points
            policy_points = p0 if policy_seat == 0 else p1
            opp_points = p1 if policy_seat == 0 else p0
        entropies.append(avg_entropy)

        # Episodic return (consistente con shaped reward): diff punti finale / 120.
        returns_buf.append(ep_return)

        # Win/draw tracking (in termini di punti).
        if policy_points > opp_points:
            wins += 1
        elif policy_points == opp_points:
            draws += 1

        if numba_traj_for_backprop is not None:
            returns_to_go_arr = _compute_returns_array(numba_traj_for_backprop.rewards, gamma=float(args.gamma))
            grad_stats = _accumulate_numba_trajectory_grads(
                policy=policy,
                xs=numba_traj_for_backprop.xs,
                z1s=numba_traj_for_backprop.z1s,
                hs=numba_traj_for_backprop.hs,
                action_masks=numba_traj_for_backprop.action_masks,
                probs=numba_traj_for_backprop.probs,
                action_ids=numba_traj_for_backprop.action_ids,
                value_preds=numba_traj_for_backprop.value_preds,
                returns_to_go=returns_to_go_arr,
                entropy_beta=float(args.entropy_beta),
                value_coef=float(args.value_coef),
                bc_anchor=bc_anchor,
                bc_anchor_beta=float(args.bc_anchor_beta),
                gw1=gw1,
                gb1=gb1,
                gw2=gw2,
                gb2=gb2,
                gwv=gwv,
            )
            gbv += grad_stats.gbv
            grad_step_count += grad_stats.steps
            value_loss_sum += grad_stats.value_loss_sum
            anchor_ce_sum += grad_stats.anchor_ce_sum
            anchor_ce_count += grad_stats.anchor_ce_count
        else:
            rewards = [step_rec.reward for step_rec in traj]
            returns_to_go = _compute_returns(rewards, gamma=float(args.gamma))

            # Backprop per ogni step della traiettoria (Monte Carlo A2C).
            for step_rec, g in zip(traj, returns_to_go, strict=True):
                grad_step_count += 1
                v = float(step_rec.value_pred)
                adv = float(g - v)

                # Policy gradient (loss = -adv * log pi(a|s)).
                dlogits = step_rec.probs.copy()
                dlogits[step_rec.action_id] -= 1.0
                dlogits *= float(adv)

                beta = float(args.entropy_beta)
                if beta > 0.0:
                    # Loss include `-beta * H(pi)` per incoraggiare esplorazione.
                    logp = np.log(step_rec.probs + 1e-12)
                    s = float(np.sum(step_rec.probs * (logp + 1.0)))
                    dent = step_rec.probs * (logp + 1.0 - s)
                    dlogits += beta * dent

                # Regularization: stay-close-to-BC anchor (se attivo).
                #
                # Questo termine NON è pesato dall'advantage: è un vincolo "stile" separato dal reward.
                anchor_beta = float(args.bc_anchor_beta)
                if anchor_beta > 0.0 and step_rec.anchor_probs is not None:
                    grad_anchor = grad_ce_wrt_logits_from_probs(
                        pred_probs=step_rec.probs,
                        target_probs=step_rec.anchor_probs,
                        action_mask=step_rec.action_mask,
                    )
                    dlogits += anchor_beta * grad_anchor
                    anchor_ce_sum += float(step_rec.anchor_ce)
                    anchor_ce_count += 1

                # Actor head grads.
                gw2 += np.outer(step_rec.h, dlogits).astype(np.float32)
                gb2 += dlogits.astype(np.float32)
                dh_policy = policy.w2 @ dlogits  # (H,)

                # Critic loss: 0.5 * value_coef * (V - G)^2
                dv = float(args.value_coef) * (v - float(g))
                value_loss_sum += 0.5 * float(args.value_coef) * (v - float(g)) ** 2

                gwv += (step_rec.h * dv).astype(np.float32)
                gbv += dv
                dh_value = policy.wv * dv  # (H,)

                dh = dh_policy + dh_value
                dz1 = dh * (step_rec.z1 > 0.0)
                gw1 += np.outer(step_rec.x, dz1).astype(np.float32)
                gb1 += dz1.astype(np.float32)

        # Update ogni `update_every` partite.
        if game_idx % update_every == 0:
            t += 1

            # Normalizziamo per numero di step policy osservati (più robusto di /update_every).
            # In 2-player i step per game sono ~20, ma può variare per seat-fair/fine partita.
            total_steps = max(1, grad_step_count)
            scale = 1.0 / float(total_steps)
            gw1 *= scale
            gb1 *= scale
            gw2 *= scale
            gb2 *= scale
            gwv *= scale
            gbv *= scale

            wd = float(args.weight_decay)
            if wd > 0.0:
                gw1 += wd * policy.w1
                gw2 += wd * policy.w2
                gwv += wd * policy.wv

            _adam_update(policy.w1, gw1, state=st_w1, lr=float(args.lr), t=t)
            _adam_update(policy.b1, gb1, state=st_b1, lr=float(args.lr), t=t)
            _adam_update(policy.w2, gw2, state=st_w2, lr=float(args.lr), t=t)
            _adam_update(policy.b2, gb2, state=st_b2, lr=float(args.lr), t=t)
            _adam_update(policy.wv, gwv, state=st_wv, lr=float(args.lr), t=t)

            # `bv` lo aggiorniamo come un array 1D di lunghezza 1 per riusare Adam.
            bv_arr = np.asarray([policy.bv], dtype=np.float32)
            _adam_update(bv_arr, np.asarray([gbv], dtype=np.float32), state=st_bv, lr=float(args.lr), t=t)
            policy.bv = float(bv_arr[0])

            gw1.fill(0.0)
            gb1.fill(0.0)
            gw2.fill(0.0)
            gb2.fill(0.0)
            gwv.fill(0.0)
            gbv = 0.0

            avg_ret = float(np.mean(returns_buf)) if returns_buf else 0.0
            win_rate = float(wins) / float(update_every)
            draw_rate = float(draws) / float(update_every)
            avg_ent = float(np.mean(entropies)) if entropies else 0.0
            vloss = float(value_loss_sum) / float(grad_step_count) if grad_step_count > 0 else 0.0
            avg_anchor_ce = float(anchor_ce_sum) / float(anchor_ce_count) if anchor_ce_count > 0 else 0.0

            row = TrainMetrics(
                iter=t,
                games=game_idx,
                avg_return=avg_ret,
                win_rate=win_rate,
                draw_rate=draw_rate,
                avg_entropy=avg_ent,
                value_loss=vloss,
                avg_anchor_ce=avg_anchor_ce,
            )
            metrics.append(row)

            if t % int(args.log_every) == 0 or game_idx == update_every:
                anchor_hint = "" if float(args.bc_anchor_beta) <= 0.0 else f" | anchor_ce {row.avg_anchor_ce:.3f}"
                print(
                    f"iter {t:04d} | games {game_idx:06d} | "
                    f"avg_return {row.avg_return:+.3f} | win {row.win_rate:.3f} draw {row.draw_rate:.3f} | "
                    f"entropy {row.avg_entropy:.3f} | vloss {row.value_loss:.4f}"
                    f"{anchor_hint}"
                )

            returns_buf.clear()
            wins = 0
            draws = 0
            entropies.clear()
            grad_step_count = 0
            value_loss_sum = 0.0
            anchor_ce_sum = 0.0
            anchor_ce_count = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Metadati UI (opzionali ma utili per il dropdown dei modelli in frontend).
    def _format_num_games(n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.0f}k"
        return str(n)

    def _format_opponent_label() -> str:
        if opponent_pool is not None:
            parts = [f"{item.name} {float(item.prob):.2f}" for item in opponent_pool.items]
            return "mix(" + ", ".join(parts) + ")"
        return str(args.opponent).strip()

    ui_label = f"A2C shaped {_format_num_games(int(args.num_games))} game"
    observation_note = (
        "Osservazione anti-cheat: Fast2PState numerico con feature equivalenti a PlayerObservation."
        if rollout_engine == "fast"
        else "Osservazione anti-cheat: PlayerObservation (vista parziale lecita)."
    )
    ui_description_it = (
        "Policy addestrata con A2C (actor-critic) con reward shaping (delta punti per mano), "
        f"contro {_format_opponent_label()}. "
        f"{observation_note}"
    )
    payload = {
        "format": "mlp_a2c_shaped_v1",
        "label": ui_label,
        "description_it": ui_description_it,
        "feature_dim": int(policy.feature_dim),
        "hidden_dim": int(policy.hidden_dim),
        "action_dim": 40,
        "seed": int(args.seed),
        "rollout_engine": rollout_engine,
        "fast_encoder": fast_encoder if rollout_engine == "fast" else None,
        "fast_rollout": fast_rollout if rollout_engine == "fast" else None,
        "opponent": str(args.opponent) if not opponent_mix_raw else None,
        "opponent_model": str(args.opponent_model).strip() or None,
        "opponent_mix": opponent_pool.to_metadata() if opponent_pool is not None else None,
        "init": args.init.strip() or None,
        "encoder": f"encode_observation_2p:{encoder_version}",
        "encoder_version": encoder_version,
        "reward_shaping": "turn_based_trick_delta_points",
        "reward_shaping_overkill_penalty_mode": str(args.overkill_penalty_mode),
        "reward_shaping_overkill_penalty_beta": float(args.overkill_penalty_beta),
        "reward_shaping_overkill_low_lead_points_max": int(args.overkill_low_lead_points_max),
        "bc_anchor_path": bc_anchor_path or None,
        "bc_anchor_beta": float(args.bc_anchor_beta),
        "inference_overkill_guard": bool(args.inference_overkill_guard),
        "train": {
            "algorithm": "a2c",
            "optimizer": "adam",
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "entropy_beta": float(args.entropy_beta),
            "value_coef": float(args.value_coef),
            "gamma": float(args.gamma),
            "update_every": int(args.update_every),
            "seat_fair": bool(args.seat_fair),
            "num_games": int(args.num_games),
        },
        "metrics": [asdict(m) for m in metrics],
    }

    # Nota compatibilità: salviamo `w1/b1/w2/b2` (actor) così `bc_model` può usare il modello direttamente.
    np.savez(
        out_path,
        w1=policy.w1,
        b1=policy.b1,
        w2=policy.w2,
        b2=policy.b2,
        wv=policy.wv,
        bv=np.asarray([policy.bv], dtype=np.float32),
        metadata_json=json.dumps(payload, ensure_ascii=False, indent=2),
    )
    print(f"Saved model: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
