#!/usr/bin/env python3
"""
Valuta agenti offline (dominio-only).

Esempi:
  python scripts/evaluate_agents.py --num-games 1000 --seed 42 --agent0 random --agent1 random
  python scripts/evaluate_agents.py --num-games 1000 --seed 42 --agent0 greedy_points --agent1 random
"""

from __future__ import annotations

import argparse
from pathlib import Path

from briscola_ai.ai.agents import GreedyPointsAgent, HeuristicAgentV1, RandomAgent
from briscola_ai.ai.evaluation import evaluate_match_2p, evaluate_seat_fair_match_2p


def _load_seed_suite(path: Path) -> list[int]:
    """
    Carica una suite di seed (uno per riga).

    Formato:
    - righe vuote ignorate
    - righe che iniziano con `#` ignorate (commenti)
    - ogni riga valida deve essere un intero (base 10)
    """
    seeds: list[int] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        seeds.append(int(line))
    return seeds


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
    if name == "heuristic_v1":
        return HeuristicAgentV1()
    raise ValueError(f"Agente non supportato: {name!r}")


def main() -> int:
    """Entry point CLI."""
    parser = argparse.ArgumentParser(description="Valuta agenti Briscola (dominio-only)")
    parser.add_argument("--num-games", type=int, default=1000, help="Numero partite da simulare")
    parser.add_argument("--seed", type=int, default=0, help="Seed RNG (riproducibilità)")
    parser.add_argument(
        "--agent0",
        default="random",
        choices=["random", "greedy_points", "heuristic_v1"],
        help="Agente per player 0",
    )
    parser.add_argument(
        "--agent1",
        default="random",
        choices=["random", "greedy_points", "heuristic_v1"],
        help="Agente per player 1",
    )
    parser.add_argument(
        "--seat-fair",
        action="store_true",
        help=(
            "Valutazione seat-fair: per ogni seed si giocano due partite con agenti scambiati "
            "(riduce il bias dovuto a chi inizia = player 0). Richiede num-games pari."
        ),
    )
    parser.add_argument(
        "--seed-suite-file",
        default="",
        help=(
            "Path a una suite di seed per lo shuffle (uno per riga). "
            "Se presente, rende la valutazione confrontabile nel tempo. "
            "In seat-fair serve una seed per coppia (num-games/2)."
        ),
    )
    args = parser.parse_args()

    agent0 = _build_agent(args.agent0)
    agent1 = _build_agent(args.agent1)

    game_seeds = None
    if args.seed_suite_file.strip():
        suite_path = Path(args.seed_suite_file)
        game_seeds = _load_seed_suite(suite_path)

    if args.seat_fair:
        needed = args.num_games // 2
        suite_slice = game_seeds[:needed] if game_seeds is not None else None
        stats = evaluate_seat_fair_match_2p(
            agent0,
            agent1,
            num_games=args.num_games,
            seed=args.seed,
            game_seeds=suite_slice,
        )
        print(f"Match 2-player (seat-fair): {stats.agent_a_name} (A) vs {stats.agent_b_name} (B)")
        print(f"- games: {stats.num_games} (seed={args.seed})")
        print(f"- wins A: {stats.wins_agent_a} | wins B: {stats.wins_agent_b} | draws: {stats.draws}")
        print(f"- avg points A: {stats.avg_points_agent_a:.2f} | avg points B: {stats.avg_points_agent_b:.2f}")
        print(f"- avg point diff (A-B): {stats.avg_point_diff_agent_a_minus_agent_b:.2f}")
        return 0

    suite_slice = game_seeds[: args.num_games] if game_seeds is not None else None
    stats = evaluate_match_2p(agent0, agent1, num_games=args.num_games, seed=args.seed, game_seeds=suite_slice)
    print(f"Match 2-player: {stats.agent0_name} (P0) vs {stats.agent1_name} (P1)")
    print(f"- games: {stats.num_games} (seed={args.seed})")
    print(f"- wins P0: {stats.wins_agent0} | wins P1: {stats.wins_agent1} | draws: {stats.draws}")
    print(f"- avg points P0: {stats.avg_points_agent0:.2f} | avg points P1: {stats.avg_points_agent1:.2f}")
    print(f"- avg point diff (P0-P1): {stats.avg_point_diff_agent0_minus_agent1:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
