#!/usr/bin/env python3
"""
Valuta un prototipo PIMC sopra un modello `.npz`.

Esempio:
  python scripts/evaluate_pimc.py --model data/models/best_a2c_v6.npz --num-games 40 --determinizations 8
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from briscola_ai.ai.agents import PIMCAgent
from briscola_ai.ai.evaluation import evaluate_seat_fair_match_2p
from briscola_ai.ai.evaluation.round_robin import mean_point_diff_interval, wilson_score_interval
from briscola_ai.ai.models import BCModelAgent


def main() -> int:
    parser = argparse.ArgumentParser(description="Valuta PIMC(model) con engine dominio offline.")
    parser.add_argument(
        "--model",
        default="data/models/best_a2c_v6.npz",
        help="Path del modello `.npz` usato come fallback/rollout e opponent puro.",
    )
    parser.add_argument("--num-games", type=int, default=40, help="Numero partite seat-fair. Default: 40.")
    parser.add_argument("--seed", type=int, default=0, help="Seed evaluation. Default: 0.")
    parser.add_argument(
        "--determinizations",
        type=int,
        default=8,
        help="Determinizzazioni per decisione PIMC. Default: 8.",
    )
    parser.add_argument(
        "--max-unknown-cards",
        type=int,
        default=10,
        help="Soglia carte vive ignote oltre cui PIMC delega al modello puro. Default: 10.",
    )
    parser.add_argument(
        "--disable-endgame-solver",
        action="store_true",
        help="Disabilita solver esatto a mazzo vuoto durante PIMC/rollout.",
    )
    parser.add_argument(
        "--opponent",
        choices=["model", "control", "pimc"],
        default="model",
        help=(
            "Opponent: `model` = modello puro; `control` = stesso modello con solver endgame ma senza search PIMC "
            "(`max_unknown=0`); `pimc` = seconda config PIMC. Default: model."
        ),
    )
    parser.add_argument(
        "--opponent-determinizations",
        type=int,
        default=None,
        help="Determinizzazioni dell'opponent `pimc`. Default: stesso valore di --determinizations.",
    )
    parser.add_argument(
        "--opponent-max-unknown-cards",
        type=int,
        default=None,
        help="Soglia carte ignote dell'opponent `pimc`. Default: stesso valore di --max-unknown-cards.",
    )
    parser.add_argument("--out-json", default="", help="Path JSON opzionale per salvare il risultato.")

    args = parser.parse_args()
    model_path = Path(args.model)
    model_agent = BCModelAgent.from_npz(model_path)
    pimc_agent = PIMCAgent(
        rollout_agent=model_agent,
        fallback=model_agent,
        num_determinizations=args.determinizations,
        max_unknown_cards=args.max_unknown_cards,
        use_endgame_solver=not args.disable_endgame_solver,
        name=f"pimc({model_path.name})",
    )
    if args.opponent == "control":
        opponent_determinizations = args.determinizations
        opponent_max_unknown_cards = 0
        opponent_agent = PIMCAgent(
            rollout_agent=model_agent,
            fallback=model_agent,
            num_determinizations=opponent_determinizations,
            max_unknown_cards=opponent_max_unknown_cards,
            use_endgame_solver=not args.disable_endgame_solver,
            name=f"control_solver({model_path.name})",
        )
    elif args.opponent == "pimc":
        opponent_determinizations = (
            args.opponent_determinizations if args.opponent_determinizations is not None else args.determinizations
        )
        opponent_max_unknown_cards = (
            args.opponent_max_unknown_cards if args.opponent_max_unknown_cards is not None else args.max_unknown_cards
        )
        opponent_agent = PIMCAgent(
            rollout_agent=model_agent,
            fallback=model_agent,
            num_determinizations=opponent_determinizations,
            max_unknown_cards=opponent_max_unknown_cards,
            use_endgame_solver=not args.disable_endgame_solver,
            name=f"pimc({model_path.name},d={opponent_determinizations},u={opponent_max_unknown_cards})",
        )
    else:
        opponent_determinizations = None
        opponent_max_unknown_cards = None
        opponent_agent = model_agent

    start = time.perf_counter()
    stats = evaluate_seat_fair_match_2p(
        pimc_agent,
        opponent_agent,
        num_games=args.num_games,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - start

    total = max(1, stats.wins_agent_a + stats.wins_agent_b + stats.draws)
    score_rate = (stats.wins_agent_a + 0.5 * stats.draws) / total
    score_ci = wilson_score_interval(
        wins=stats.wins_agent_a,
        losses=stats.wins_agent_b,
        draws=stats.draws,
        confidence=0.95,
    )
    avg_diff_ci = mean_point_diff_interval(
        mean=stats.avg_point_diff_agent_a_minus_agent_b,
        num_games=stats.num_games,
        sum_sq=stats.sum_sq_point_diff_agent_a_minus_agent_b,
        confidence=0.95,
    )
    avg_diff_ci_text = "n/a" if avg_diff_ci is None else f"{avg_diff_ci.low:+.2f}..{avg_diff_ci.high:+.2f}"
    metrics = pimc_agent.metrics
    print(
        "PIMC evaluation | "
        f"model={model_path.name} | opponent={args.opponent} | games={args.num_games} | "
        f"determinizations={args.determinizations} | max_unknown={args.max_unknown_cards}"
    )
    if opponent_determinizations is not None:
        print(
            f"opponent_config | determinizations={opponent_determinizations} | max_unknown={opponent_max_unknown_cards}"
        )
    print(
        f"score_rate={score_rate:.4f} ci95={score_ci.low:.4f}..{score_ci.high:.4f} | "
        f"avg_diff={stats.avg_point_diff_agent_a_minus_agent_b:+.2f} ci95={avg_diff_ci_text} | "
        f"wins={stats.wins_agent_a} losses={stats.wins_agent_b} draws={stats.draws}"
    )
    print(f"elapsed={elapsed:.2f}s | sec/game={elapsed / max(1, args.num_games):.4f}")
    print(
        "pimc_moves | "
        f"total={metrics.total_decisions} search={metrics.search_decisions} "
        f"fallback={metrics.fallback_decisions} endgame_solver={metrics.endgame_solver_decisions}"
    )
    print(
        "pimc_latency | "
        f"search_elapsed={metrics.search_elapsed_seconds:.4f}s | "
        f"sec/search_move={metrics.seconds_per_search_decision:.4f} | "
        f"determinizations_ok={metrics.successful_determinizations} "
        f"determinizations_failed={metrics.failed_determinizations} | "
        f"rollouts={metrics.completed_rollouts} failed_rollouts={metrics.failed_rollouts} | "
        f"coerced_moves={metrics.coerced_moves}"
    )
    if isinstance(opponent_agent, PIMCAgent):
        opponent_metrics = opponent_agent.metrics
        print(
            "opponent_pimc | "
            f"total={opponent_metrics.total_decisions} search={opponent_metrics.search_decisions} "
            f"fallback={opponent_metrics.fallback_decisions} "
            f"endgame_solver={opponent_metrics.endgame_solver_decisions} "
            f"sec/search_move={opponent_metrics.seconds_per_search_decision:.4f} "
            f"coerced_moves={opponent_metrics.coerced_moves}"
        )
    else:
        opponent_metrics = None

    if args.out_json:
        payload = {
            "model": str(model_path),
            "opponent": args.opponent,
            "num_games": args.num_games,
            "seed": args.seed,
            "determinizations": args.determinizations,
            "max_unknown_cards": args.max_unknown_cards,
            "opponent_determinizations": opponent_determinizations,
            "opponent_max_unknown_cards": opponent_max_unknown_cards,
            "use_endgame_solver": not args.disable_endgame_solver,
            "elapsed_seconds": elapsed,
            "seconds_per_game": elapsed / max(1, args.num_games),
            "score_rate": score_rate,
            "score_rate_ci95": asdict(score_ci),
            "avg_point_diff_ci95": asdict(avg_diff_ci) if avg_diff_ci is not None else None,
            "pimc_metrics": {
                **asdict(metrics),
                "seconds_per_search_decision": metrics.seconds_per_search_decision,
            },
            "opponent_metrics": (
                {
                    **asdict(opponent_metrics),
                    "seconds_per_search_decision": opponent_metrics.seconds_per_search_decision,
                }
                if opponent_metrics is not None
                else None
            ),
            "stats": asdict(stats),
        }
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"JSON salvato in: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
