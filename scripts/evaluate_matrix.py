#!/usr/bin/env python3
"""
Valuta un modello `.npz` su una evaluation matrix (avversari × suite).

Esempio:
  python scripts/evaluate_matrix.py --model ./data/a2c_shaped.npz --benchmark medium --out-json /tmp/matrix.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from briscola_ai.ai.evaluation_matrix import default_opponents, evaluate_model_matrix, format_matrix_table


def main() -> int:
    parser = argparse.ArgumentParser(description="Valuta un modello `.npz` su una evaluation matrix (dominio-only)")
    parser.add_argument("--model", required=True, help="Path al modello `.npz` (BC/RL) da valutare.")
    parser.add_argument(
        "--benchmark",
        choices=["small", "medium", "big"],
        default="medium",
        help="Taglia benchmark (tutte seat-fair). Default: medium.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed RNG (riproducibilità).")
    parser.add_argument(
        "--opponents",
        default="",
        help="Lista avversari CSV (default: heuristic_v1,random,greedy_points).",
    )
    parser.add_argument(
        "--holdout-start",
        type=int,
        default=1_000_000,
        help="Range start per suite holdout (default: 1_000_000).",
    )
    parser.add_argument(
        "--standard-start",
        type=int,
        default=0,
        help="Range start per suite standard (default: 0).",
    )
    parser.add_argument(
        "--range-step",
        type=int,
        default=1,
        help="Step per le suite generate via range (default: 1).",
    )
    parser.add_argument(
        "--out-json",
        default="",
        help="Se valorizzato, salva la matrice in JSON oltre alla stampa a schermo.",
    )
    args = parser.parse_args()

    opponents = default_opponents()
    if args.opponents.strip():
        opponents = [p.strip() for p in args.opponents.split(",") if p.strip()]
        if not opponents:
            raise ValueError("`--opponents` non contiene elementi validi.")

    matrix = evaluate_model_matrix(
        model_path=Path(args.model),
        opponents=opponents,
        benchmark=args.benchmark,
        seed=args.seed,
        standard_start=args.standard_start,
        holdout_start=args.holdout_start,
        range_step=args.range_step,
    )

    print(format_matrix_table(matrix), end="")

    if args.out_json.strip():
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(matrix.to_json_text(), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
