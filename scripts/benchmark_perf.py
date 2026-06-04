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
import time
from pathlib import Path

from briscola_ai.ai.agents import build_agent, list_agent_specs
from briscola_ai.ai.bc_model_agent import BCModelAgent
from briscola_ai.ai.evaluation import evaluate_seat_fair_match_2p


def _build_agent(*, agent_name: str, model_path: str, flag: str):
    """Costruisce un agente da nome + modello opzionale, coerente con gli script di eval."""
    if agent_name == "bc_model":
        if not model_path.strip():
            raise ValueError(f"`--{flag}-model` e' obbligatorio quando `--{flag} bc_model`.")
        return BCModelAgent.from_npz(Path(model_path.strip()))
    if model_path.strip():
        raise ValueError(f"`--{flag}-model` e' valido solo quando `--{flag} bc_model`.")
    return build_agent(agent_name)


def main() -> int:
    """Esegue il benchmark e stampa metriche sintetiche."""
    agent_names = [spec.name for spec in list_agent_specs()] + ["bc_model"]

    parser = argparse.ArgumentParser(description="Benchmark throughput simulazioni 2-player.")
    parser.add_argument("--games", type=int, default=2000, help="Numero partite seat-fair (deve essere pari).")
    parser.add_argument("--repeat", type=int, default=3, help="Numero ripetizioni.")
    parser.add_argument("--seed", type=int, default=0, help="Seed base.")
    parser.add_argument("--agent-a", choices=agent_names, default="bc_model")
    parser.add_argument("--agent-b", choices=agent_names, default="heuristic_v1")
    parser.add_argument("--agent-a-model", default="data/models/best_a2c.npz")
    parser.add_argument("--agent-b-model", default="")
    args = parser.parse_args()

    if int(args.games) <= 0 or int(args.games) % 2 != 0:
        raise ValueError("--games deve essere un intero pari > 0")
    if int(args.repeat) <= 0:
        raise ValueError("--repeat deve essere > 0")

    agent_a = _build_agent(agent_name=args.agent_a, model_path=args.agent_a_model, flag="agent-a")
    agent_b = _build_agent(agent_name=args.agent_b, model_path=args.agent_b_model, flag="agent-b")

    elapsed_values: list[float] = []
    last_avg_diff = 0.0
    for i in range(int(args.repeat)):
        seed = int(args.seed) + i
        t0 = time.perf_counter()
        stats = evaluate_seat_fair_match_2p(agent_a, agent_b, num_games=int(args.games), seed=seed)
        elapsed = time.perf_counter() - t0
        elapsed_values.append(elapsed)
        last_avg_diff = float(stats.avg_point_diff_agent_a_minus_agent_b)
        print(
            f"run {i + 1}/{int(args.repeat)} | games={int(args.games)} | elapsed={elapsed:.3f}s | "
            f"games/sec={int(args.games) / elapsed:.1f} | avg_diff={last_avg_diff:+.2f}"
        )

    best = min(elapsed_values)
    avg = sum(elapsed_values) / len(elapsed_values)
    print(
        "summary | "
        f"agent_a={agent_a.name} | agent_b={agent_b.name} | "
        f"best={best:.3f}s ({int(args.games) / best:.1f} games/sec) | "
        f"avg={avg:.3f}s ({int(args.games) / avg:.1f} games/sec) | "
        f"last_avg_diff={last_avg_diff:+.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
