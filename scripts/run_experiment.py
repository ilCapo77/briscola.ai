#!/usr/bin/env python3
"""
Pipeline “training + evaluation” riproducibile per modelli Briscola AI.

Perché esiste
-------------
Allenare modelli RL richiede molte iterazioni. Se lanci comandi “a mano” è facile:
- dimenticare un benchmark (es. holdout);
- perdere la configurazione con cui hai allenato un `.npz`;
- sovrascrivere file o creare nomi poco descrittivi.

Questo script orchestri:
1) training (A2C o PG REINFORCE) -> salva un `.npz` in `data/models/`
2) evaluation matrix (`medium` e/o `big`, include holdout) -> salva JSON in `benchmarks/experiments/<name>/`
3) manifest JSON (config + versioni + percorsi) per riproducibilità
4) (opzionale) aggiorna un “best model” locale: `data/models/best_<algo>.npz`

Nota:
gli artefatti in `data/` e `benchmarks/` sono locali (gitignored).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from briscola_ai.ai.experiment_pipeline import (
    AlgoName,
    build_experiment_name,
    extract_best_metric_from_matrix_json,
    read_json,
    utc_now_iso,
    write_json,
)
from briscola_ai.versioning import get_code_version, get_rules_version


def _run(cmd: list[str], *, log_path: Path) -> None:
    """
    Esegue un comando e fa tee su stdout + file.

    Nota didattica:
    salviamo i log su file per poter ricostruire *esattamente* cosa è successo
    anche dopo giorni (utile quando alleni decine di modelli).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Comando fallito (exit={rc}): {' '.join(cmd)}")


def _copy_best_model(*, model_path: Path, best_path: Path, score: float, meta_path: Path, manifest: dict) -> None:
    """
    Aggiorna il best model se lo score è migliore.

    Persistiamo anche un JSON a fianco, così il best non è “magico”.
    """
    previous_score: float | None = None
    if meta_path.exists():
        try:
            prev = read_json(meta_path)
            raw = prev.get("score")
            if isinstance(raw, (int, float)):
                previous_score = float(raw)
        except Exception:
            previous_score = None

    if previous_score is not None and score <= previous_score:
        print(f"Best model invariato: score={score:+.2f} <= best={previous_score:+.2f}")
        return

    best_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, best_path)
    write_json(
        meta_path,
        {
            "score": float(score),
            "model_path": str(model_path),
            "updated_utc": utc_now_iso(),
            "manifest": manifest,
        },
    )
    print(f"Updated best model: {best_path} (score={score:+.2f})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Esegue training + evaluation matrix + manifest (riproducibile).")
    parser.add_argument("--algo", choices=["a2c", "pg"], default="a2c", help="Algoritmo training (default: a2c).")
    parser.add_argument("--num-games", type=int, default=200000, help="Numero partite training.")
    parser.add_argument("--train-seed", type=int, default=0, help="Seed training.")
    parser.add_argument("--eval-seed", type=int, default=0, help="Seed evaluation (RNG agent).")
    parser.add_argument("--init", default="", help="Warm-start da un `.npz` (opzionale).")
    parser.add_argument("--opponent", default="heuristic_v1", help="Avversario (se non usi --opponent-mix).")
    parser.add_argument(
        "--opponent-mix",
        default="heuristic_v1:0.7,random:0.2,greedy_points:0.1",
        help="Miscela avversari (se valorizzata, sovrascrive --opponent).",
    )
    parser.add_argument("--seat-fair", action="store_true", help="Seat-fair durante training (consigliato).")
    parser.add_argument(
        "--benchmarks",
        default="medium,big",
        help="Benchmark da eseguire in evaluation matrix (CSV: small,medium,big). Default: medium,big.",
    )
    parser.add_argument(
        "--name",
        default="",
        help="Nome esperimento (se vuoto, viene costruito in modo deterministico dai parametri).",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Suffisso opzionale per distinguere run con stessa config (entra nel nome).",
    )
    parser.add_argument(
        "--models-dir",
        default="./data/models",
        help="Directory output modelli (default: ./data/models).",
    )
    parser.add_argument(
        "--experiments-dir",
        default="./benchmarks/experiments",
        help="Directory output esperimenti (default: ./benchmarks/experiments).",
    )
    parser.add_argument(
        "--train-extra",
        nargs=argparse.REMAINDER,
        help=(
            "Argomenti extra passati al trainer (usa `--` per separarli). "
            "Esempio: python scripts/run_experiment.py --algo a2c -- --lr 3e-4 --entropy-beta 1e-3"
        ),
    )
    parser.add_argument(
        "--update-best",
        action="store_true",
        default=True,
        help="Aggiorna `best_<algo>.npz` se lo score migliora (default: true).",
    )
    args = parser.parse_args()

    algo: AlgoName = args.algo
    if int(args.num_games) <= 0:
        raise ValueError("--num-games deve essere > 0")

    opponent_mix = args.opponent_mix.strip()
    opponent = args.opponent.strip()
    if opponent_mix:
        opponent_for_name = None
        mix_for_name = opponent_mix
    else:
        opponent_for_name = opponent
        mix_for_name = None

    name = args.name.strip()
    if not name:
        name = build_experiment_name(
            algo=algo,
            num_games=int(args.num_games),
            seed=int(args.train_seed),
            opponent=opponent_for_name,
            opponent_mix=mix_for_name,
            tag=args.tag.strip() or None,
        )

    models_dir = Path(args.models_dir)
    experiments_dir = Path(args.experiments_dir) / name
    model_path = models_dir / f"{name}.npz"

    train_log = experiments_dir / "train.log"
    manifest_path = experiments_dir / "manifest.json"

    # --- Training ---
    trainer_script = "scripts/train_a2c.py" if algo == "a2c" else "scripts/train_pg.py"
    train_cmd = [sys.executable, trainer_script, "--out", str(model_path)]

    if args.init.strip():
        train_cmd += ["--init", args.init.strip()]

    if opponent_mix:
        train_cmd += ["--opponent-mix", opponent_mix]
    else:
        train_cmd += ["--opponent", opponent]

    train_cmd += ["--num-games", str(int(args.num_games)), "--seed", str(int(args.train_seed))]
    if bool(args.seat_fair):
        train_cmd.append("--seat-fair")

    train_extra = args.train_extra or []
    forbidden = {"--out", "--data"}
    if any(tok in forbidden for tok in train_extra):
        raise ValueError(f"`--train-extra` non può includere {sorted(forbidden)} (gestiti dalla pipeline).")
    if train_extra and train_extra[0] == "--":
        train_extra = train_extra[1:]
    train_cmd += list(train_extra)

    _run(train_cmd, log_path=train_log)
    if not model_path.exists():
        raise RuntimeError(f"Training completato ma il modello non esiste: {model_path}")

    # --- Evaluation matrix ---
    benchmarks = [b.strip() for b in str(args.benchmarks).split(",") if b.strip()]
    allowed = {"small", "medium", "big"}
    if any(b not in allowed for b in benchmarks):
        raise ValueError(f"--benchmarks deve essere subset di {sorted(allowed)} (ottenuto: {benchmarks})")

    matrix_paths: dict[str, Path] = {}
    for b in benchmarks:
        out_json = experiments_dir / f"matrix_{b}.json"
        log_path = experiments_dir / f"matrix_{b}.log"
        eval_cmd = [
            sys.executable,
            "scripts/evaluate_matrix.py",
            "--model",
            str(model_path),
            "--benchmark",
            b,
            "--seed",
            str(int(args.eval_seed)),
            "--out-json",
            str(out_json),
            "--format",
            "csv",
        ]
        _run(eval_cmd, log_path=log_path)
        matrix_paths[b] = out_json

    # --- Manifest + score ---
    manifest: dict = {
        "name": name,
        "created_utc": utc_now_iso(),
        "code_version": get_code_version(),
        "rules_version": get_rules_version(),
        "algo": algo,
        "model_path": str(model_path),
        "train": {"cmd": train_cmd},
        "eval": [
            {
                "benchmark": b,
                "matrix_json": str(p),
            }
            for b, p in matrix_paths.items()
        ],
    }

    score: float | None = None
    best_metric: dict | None = None

    # Preferiamo big holdout vs heuristic_v1 se disponibile.
    preferred = "big" if "big" in matrix_paths else ("medium" if "medium" in matrix_paths else benchmarks[0])
    matrix_json = read_json(matrix_paths[preferred])
    metric = extract_best_metric_from_matrix_json(matrix_json, benchmark=preferred)
    score = float(metric.avg_diff)
    best_metric = {
        "opponent": metric.opponent,
        "suite": metric.suite,
        "benchmark": metric.benchmark,
        "avg_diff": metric.avg_diff,
    }
    manifest["best_metric"] = best_metric

    write_json(manifest_path, manifest)
    print(f"Wrote manifest: {manifest_path}")

    if bool(args.update_best) and score is not None:
        best_path = models_dir / f"best_{algo}.npz"
        best_meta = models_dir / f"best_{algo}.json"
        _copy_best_model(
            model_path=model_path, best_path=best_path, score=score, meta_path=best_meta, manifest=manifest
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
