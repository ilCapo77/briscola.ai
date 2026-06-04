#!/usr/bin/env python3
"""
Micro-benchmark locale per training/evaluation 2-player.

Obiettivo
---------
Misurare in modo ripetibile il throughput delle simulazioni dominio-only, senza HTTP/WS.
Il valore principale e' `games/sec`: dopo ogni refactor performance possiamo confrontarlo
con lo stesso comando e capire se il cambiamento vale davvero.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from briscola_ai.ai.agents import build_agent, list_agent_specs
from briscola_ai.ai.bc_model_agent import BCModelAgent
from briscola_ai.ai.evaluation import evaluate_seat_fair_match_2p
from briscola_ai.ai.fast_2p import Fast2PState, play_random_fast_2p
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.state import GameState, new_game_state


def _build_agent(*, agent_name: str, model_path: str, flag: str):
    """Costruisce un agente da nome + modello opzionale, coerente con gli script di eval."""
    if agent_name == "bc_model":
        if not model_path.strip():
            raise ValueError(f"`--{flag}-model` e' obbligatorio quando `--{flag} bc_model`.")
        return BCModelAgent.from_npz(Path(model_path.strip()))
    if model_path.strip():
        raise ValueError(f"`--{flag}-model` e' valido solo quando `--{flag} bc_model`.")
    return build_agent(agent_name)


def _play_random_domain_2p(*, seed: int, action_seed: int) -> GameState:
    """
    Gioca una partita random usando il dominio canonico.

    Serve come baseline engine-only: niente observation encoder, niente modello, solo `domain.step`.
    """
    state = new_game_state(num_players=2, seed=seed)
    rng = random.Random(action_seed)
    safety = 5000
    while not state.game_over and safety > 0:
        safety -= 1
        player_index = state.current_turn
        card_index = rng.randrange(len(state.players[player_index].hand))
        state, result = step(state, PlayCardAction(player_index=player_index, card_index=card_index))
        if result.error is not None:
            raise RuntimeError(result.error)
    if safety <= 0:
        raise RuntimeError("Loop di sicurezza: la partita dominio non termina")
    return state


def _benchmark_random_games(*, mode: str, games: int, repeat: int, seed: int) -> None:
    """
    Esegue benchmark engine-only con policy random.

    `domain-random` misura il motore canonico immutabile; `fast-random` misura il nuovo core mutabile.
    """
    elapsed_values: list[float] = []
    for i in range(repeat):
        run_seed = seed + i
        t0 = time.perf_counter()
        point_sum = 0
        for game_index in range(games):
            game_seed = run_seed + game_index
            action_seed = (run_seed * 1_000_003) + game_index
            if mode == "domain-random":
                domain_state = _play_random_domain_2p(seed=game_seed, action_seed=action_seed)
                point_sum += sum(player.points for player in domain_state.players)
            elif mode == "fast-random":
                fast_state: Fast2PState = play_random_fast_2p(seed=game_seed, action_seed=action_seed)
                point_sum += sum(fast_state.points)
            else:
                raise ValueError(f"Modalità non supportata: {mode}")
        elapsed = time.perf_counter() - t0
        elapsed_values.append(elapsed)
        print(
            f"run {i + 1}/{repeat} | mode={mode} | games={games} | elapsed={elapsed:.3f}s | "
            f"games/sec={games / elapsed:.1f} | avg_points={point_sum / games:.1f}"
        )

    best = min(elapsed_values)
    avg = sum(elapsed_values) / len(elapsed_values)
    print(
        "summary | "
        f"mode={mode} | best={best:.3f}s ({games / best:.1f} games/sec) | "
        f"avg={avg:.3f}s ({games / avg:.1f} games/sec)"
    )


def _benchmark_evaluation(
    *,
    games: int,
    repeat: int,
    seed: int,
    agent_a_name: str,
    agent_b_name: str,
    agent_a_model: str,
    agent_b_model: str,
) -> None:
    """Esegue il benchmark storico: evaluation seat-fair tra due agenti."""
    agent_a = _build_agent(agent_name=agent_a_name, model_path=agent_a_model, flag="agent-a")
    agent_b = _build_agent(agent_name=agent_b_name, model_path=agent_b_model, flag="agent-b")

    elapsed_values: list[float] = []
    last_avg_diff = 0.0
    for i in range(repeat):
        run_seed = seed + i
        t0 = time.perf_counter()
        stats = evaluate_seat_fair_match_2p(agent_a, agent_b, num_games=games, seed=run_seed)
        elapsed = time.perf_counter() - t0
        elapsed_values.append(elapsed)
        last_avg_diff = float(stats.avg_point_diff_agent_a_minus_agent_b)
        print(
            f"run {i + 1}/{repeat} | mode=eval | games={games} | elapsed={elapsed:.3f}s | "
            f"games/sec={games / elapsed:.1f} | avg_diff={last_avg_diff:+.2f}"
        )

    best = min(elapsed_values)
    avg = sum(elapsed_values) / len(elapsed_values)
    print(
        "summary | "
        f"mode=eval | agent_a={agent_a.name} | agent_b={agent_b.name} | "
        f"best={best:.3f}s ({games / best:.1f} games/sec) | "
        f"avg={avg:.3f}s ({games / avg:.1f} games/sec) | "
        f"last_avg_diff={last_avg_diff:+.2f}"
    )


def main() -> int:
    """Esegue il benchmark e stampa metriche sintetiche."""
    agent_names = [spec.name for spec in list_agent_specs()] + ["bc_model"]

    parser = argparse.ArgumentParser(description="Benchmark throughput simulazioni 2-player.")
    parser.add_argument(
        "--mode",
        choices=["eval", "domain-random", "fast-random"],
        default="eval",
        help="Tipo di benchmark: evaluation con agenti oppure loop engine-only random.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=2000,
        help="Numero partite. In modalità eval deve essere pari per il seat-fair.",
    )
    parser.add_argument("--repeat", type=int, default=3, help="Numero ripetizioni.")
    parser.add_argument("--seed", type=int, default=0, help="Seed base.")
    parser.add_argument("--agent-a", choices=agent_names, default="bc_model")
    parser.add_argument("--agent-b", choices=agent_names, default="heuristic_v1")
    parser.add_argument("--agent-a-model", default="data/models/best_a2c.npz")
    parser.add_argument("--agent-b-model", default="")
    args = parser.parse_args()

    games = int(args.games)
    repeat = int(args.repeat)
    seed = int(args.seed)

    if games <= 0:
        raise ValueError("--games deve essere > 0")
    if repeat <= 0:
        raise ValueError("--repeat deve essere > 0")
    if args.mode == "eval" and games % 2 != 0:
        raise ValueError("--games deve essere pari in modalità eval seat-fair")

    if args.mode in ("domain-random", "fast-random"):
        _benchmark_random_games(mode=args.mode, games=games, repeat=repeat, seed=seed)
    else:
        _benchmark_evaluation(
            games=games,
            repeat=repeat,
            seed=seed,
            agent_a_name=args.agent_a,
            agent_b_name=args.agent_b,
            agent_a_model=args.agent_a_model,
            agent_b_model=args.agent_b_model,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
