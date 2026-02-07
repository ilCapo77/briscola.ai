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
from briscola_ai.ai.bc_model_agent import MLPBCModel, load_bc_model_npz
from briscola_ai.ai.training.card_action_space import action_id_from_suit_number
from briscola_ai.ai.training.observation_encoder import encode_player_observation_2p
from briscola_ai.ai.training.opponent_mix import OpponentMixItem, parse_opponent_mix, sample_opponent_name
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
    probs: np.ndarray  # (40,)
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
        encoded = encode_player_observation_2p(obs)

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
        reward = float(diff_after - diff_before) / 120.0

        traj.append(
            StepRecord(
                x=x,
                z1=z1,
                h=h,
                probs=probs,
                action_id=action_id,
                value_pred=float(value_pred),
                reward=reward,
            )
        )

    if safety <= 0:
        raise RuntimeError("Loop di sicurezza: la partita non termina")

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Train RL A2C (MLP, 40 carte + action mask) con reward shaping")
    parser.add_argument("--out", required=True, help="Path output modello (.npz)")
    parser.add_argument("--init", default="", help="Warm-start da un modello `.npz` MLP (es. BC/RL).")
    parser.add_argument("--opponent", default="heuristic_v1", help="Nome avversario (se non usi --opponent-mix).")
    parser.add_argument(
        "--opponent-mix",
        default="",
        help=(
            "Miscela avversari: `name:weight,name:weight,...` "
            "(es. `heuristic_v1:0.7,random:0.2,greedy_points:0.1`). "
            "Se presente, sovrascrive `--opponent`."
        ),
    )
    parser.add_argument("--num-games", type=int, default=20000, help="Numero partite di training (2-player).")
    parser.add_argument("--seed", type=int, default=0, help="Seed RNG (riproducibilità).")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dim (se non si usa --init).")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate Adam.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="L2 weight decay (solo pesi).")
    parser.add_argument("--entropy-beta", type=float, default=5e-4, help="Entropia bonus (>=0).")
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

    out_path = Path(args.out)
    rng_action = np.random.default_rng(args.seed)
    rng_game = np.random.default_rng(args.seed ^ 0x9E3779B9)
    rng_opponent_select = np.random.default_rng(args.seed ^ 0xA5A5A5A5)
    rng_opponent = random.Random(args.seed ^ 0xC0FFEE)

    opponent_pool: OpponentPool | None = None
    opponent_mix_raw = args.opponent_mix.strip()
    if opponent_mix_raw:
        items = parse_opponent_mix(opponent_mix_raw)
        agents_by_name = {item.name: build_agent(item.name) for item in items}
        opponent_pool = OpponentPool(items=items, agents_by_name=agents_by_name)
        opponent = build_agent(items[0].name)
    else:
        opponent = build_agent(args.opponent)

    # Inizializzazione policy/critic.
    if args.init.strip():
        loaded = load_bc_model_npz(Path(args.init))
        if not isinstance(loaded, MLPBCModel):
            raise ValueError("--init deve puntare a un modello MLP (w1/b1/w2/b2).")
        w1 = loaded.w1.copy()
        b1 = loaded.b1.copy()
        w2 = loaded.w2.copy()
        b2 = loaded.b2.copy()
        hdim = int(w1.shape[1])
    else:
        # Dimensione feature dal nostro encoder (v1): 40*6 + 4 + 4 = 248.
        feature_dim = 40 * 6 + 4 + 4
        hdim = int(args.hidden_dim)
        w1 = rng_action.normal(loc=0.0, scale=0.02, size=(feature_dim, hdim)).astype(np.float32)
        b1 = np.zeros((hdim,), dtype=np.float32)
        w2 = rng_action.normal(loc=0.0, scale=0.02, size=(hdim, 40)).astype(np.float32)
        b2 = np.zeros((40,), dtype=np.float32)

    # Critic head: inizializziamo vicino a zero (safe).
    wv = np.zeros((hdim,), dtype=np.float32)
    bv = np.float32(0.0)

    policy = A2CPolicy(w1=w1, b1=b1, w2=w2, b2=b2, wv=wv, bv=float(bv))

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
    value_losses: list[float] = []

    num_games = int(args.num_games)
    for game_idx in range(1, num_games + 1):
        game_seed = int(rng_game.integers(0, 2**32))
        policy_seat = (game_idx % 2) if args.seat_fair else 0
        current_opponent = opponent_pool.sample(rng=rng_opponent_select) if opponent_pool is not None else opponent

        final_state, traj, avg_entropy = _play_one_game_2p_collect(
            policy=policy,
            opponent=current_opponent,
            rng_opponent=rng_opponent,
            rng_action=rng_action,
            game_seed=game_seed,
            policy_seat=policy_seat,
            entropy_beta=float(args.entropy_beta),
        )
        entropies.append(avg_entropy)

        # Episodic return (consistente con shaped reward): diff punti finale / 120.
        ep_return = float(_points_diff(final_state, policy_seat=policy_seat)) / 120.0
        returns_buf.append(ep_return)

        # Win/draw tracking (in termini di punti).
        p0 = final_state.players[0].points
        p1 = final_state.players[1].points
        policy_points = p0 if policy_seat == 0 else p1
        opp_points = p1 if policy_seat == 0 else p0
        if policy_points > opp_points:
            wins += 1
        elif policy_points == opp_points:
            draws += 1

        rewards = [step_rec.reward for step_rec in traj]
        returns_to_go = _compute_returns(rewards, gamma=float(args.gamma))

        # Backprop per ogni step della traiettoria (Monte Carlo A2C).
        for step_rec, g in zip(traj, returns_to_go, strict=True):
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

            # Actor head grads.
            gw2 += np.outer(step_rec.h, dlogits).astype(np.float32)
            gb2 += dlogits.astype(np.float32)
            dh_policy = policy.w2 @ dlogits  # (H,)

            # Critic loss: 0.5 * value_coef * (V - G)^2
            dv = float(args.value_coef) * (v - float(g))
            value_losses.append(0.5 * float(args.value_coef) * (v - float(g)) ** 2)

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
            total_steps = max(1, len(value_losses))
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
            vloss = float(np.mean(value_losses)) if value_losses else 0.0

            row = TrainMetrics(
                iter=t,
                games=game_idx,
                avg_return=avg_ret,
                win_rate=win_rate,
                draw_rate=draw_rate,
                avg_entropy=avg_ent,
                value_loss=vloss,
            )
            metrics.append(row)

            if t % int(args.log_every) == 0 or game_idx == update_every:
                print(
                    f"iter {t:04d} | games {game_idx:06d} | "
                    f"avg_return {row.avg_return:+.3f} | win {row.win_rate:.3f} draw {row.draw_rate:.3f} | "
                    f"entropy {row.avg_entropy:.3f} | vloss {row.value_loss:.4f}"
                )

            returns_buf.clear()
            wins = 0
            draws = 0
            entropies.clear()
            value_losses.clear()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "mlp_a2c_shaped_v1",
        "feature_dim": int(policy.feature_dim),
        "hidden_dim": int(policy.hidden_dim),
        "action_dim": 40,
        "seed": int(args.seed),
        "opponent": str(args.opponent) if not opponent_mix_raw else None,
        "opponent_mix": opponent_pool.to_metadata() if opponent_pool is not None else None,
        "init": args.init.strip() or None,
        "encoder": "encode_observation_2p:v1",
        "reward_shaping": "turn_based_trick_delta_points",
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
