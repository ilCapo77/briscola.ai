#!/usr/bin/env python3
"""
Valuta agenti offline (dominio-only).

Esempi:
  python scripts/evaluate_agents.py --num-games 1000 --seed 42 --agent0 random --agent1 random
  python scripts/evaluate_agents.py --num-games 1000 --seed 42 --agent0 greedy_points --agent1 random
"""

from __future__ import annotations

import argparse

from briscola_ai.ai.agents import GreedyPointsAgent, RandomAgent
from briscola_ai.ai.evaluation import evaluate_match_2p


def _build_agent(name: str):
    """
    Costruisce un agente a partire da una stringa CLI.

    Nota:
    Manteniamo una mappa esplicita (no import dinamici) per semplicità e sicurezza.
    """
    if name == "random":
        return RandomAgent()
    if name == "greedy_points":
        return GreedyPointsAgent()
    raise ValueError(f"Agente non supportato: {name!r}")


def main() -> int:
    """Entry point CLI."""
    parser = argparse.ArgumentParser(description="Valuta agenti Briscola (dominio-only)")
    parser.add_argument("--num-games", type=int, default=1000, help="Numero partite da simulare")
    parser.add_argument("--seed", type=int, default=0, help="Seed RNG (riproducibilità)")
    parser.add_argument("--agent0", default="random", choices=["random", "greedy_points"], help="Agente per player 0")
    parser.add_argument("--agent1", default="random", choices=["random", "greedy_points"], help="Agente per player 1")
    args = parser.parse_args()

    agent0 = _build_agent(args.agent0)
    agent1 = _build_agent(args.agent1)

    stats = evaluate_match_2p(agent0, agent1, num_games=args.num_games, seed=args.seed)

    print(f"Match 2-player: {stats.agent0_name} (P0) vs {stats.agent1_name} (P1)")
    print(f"- games: {stats.num_games} (seed={args.seed})")
    print(f"- wins P0: {stats.wins_agent0} | wins P1: {stats.wins_agent1} | draws: {stats.draws}")
    print(f"- avg points P0: {stats.avg_points_agent0:.2f} | avg points P1: {stats.avg_points_agent1:.2f}")
    print(f"- avg point diff (P0-P1): {stats.avg_point_diff_agent0_minus_agent1:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
