#!/usr/bin/env python3
"""
Filtra il dataset diagnostico teacher PIMC nel subset di correzioni utili per la distillazione.

Contesto
--------
`scripts/generate_pimc_teacher_dataset.py` produce un JSONL ricco di diagnostica per ogni
mossa (vedi blocco `teacher.search_diagnostics`: margine pareggiato, SE, CI95, ecc.). La maggior
parte di quelle righe NON è segnale utile per un `best_a2c_v7` distillato:

- gli esempi `fallback`/`endgame_solver` coincidono col modello base o col solver runtime, che
  eseguiamo già a inference e non vogliamo comprimere nei pesi;
- molti esempi `search` concordano comunque con v6, oppure hanno un margine piccolo/rumoroso.

Questo script tiene solo le **correzioni di search affidabili**:

1. `teacher.decision_type == "search"` (configurabile);
2. il teacher è in disaccordo col modello base (`reference.disagrees_with_teacher`);
3. il margine medio pareggiato è grande: `margin >= --min-margin`;
4. il margine è statisticamente credibile: `margin_ci95_low >= --min-ci-low`.

A ogni riga tenuta aggiunge un `sample_weight` (più alto per correzioni più forti) così che
`scripts/train_bc.py` possa pesare la cross-entropy. La scala assoluta dei pesi non conta:
`train_bc.py` li normalizza a media 1.0 sul train; conta solo la *spread* relativa, che qui
controlliamo con la modalità di pesatura e il clip.

Esempio
-------
    uv run python scripts/filter_pimc_teacher_subset.py \
        --in data/pimc_teacher_diag_175k_d64_u8_seed20260630.jsonl \
        --out data/pimc_teacher_subset_search_m2_cilo0.jsonl \
        --min-margin 2.0 --min-ci-low 0.0 --weight-mode margin_clip --clip-max 10.0
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

WeightMode = Literal["uniform", "margin", "margin_clip", "margin_z", "log_margin"]
_WEIGHT_MODES: tuple[WeightMode, ...] = ("uniform", "margin", "margin_clip", "margin_z", "log_margin")


@dataclass(frozen=True, slots=True)
class SubsetFilterConfig:
    """Soglie e modalità di pesatura riproducibili per costruire il subset."""

    decision_type: str = "search"
    require_disagree: bool = True
    min_margin: float = 2.0
    min_ci_low: float = 0.0
    weight_mode: WeightMode = "margin_clip"
    clip_max: float = 10.0


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Itera record JSON per riga (streaming: il dataset diagnostico può essere centinaia di MB)."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _search_diagnostics(rec: dict[str, Any]) -> dict[str, Any] | None:
    """Ritorna il blocco diagnostico di search se presente e ben formato, altrimenti None."""
    teacher = rec.get("teacher")
    if not isinstance(teacher, dict):
        return None
    diag = teacher.get("search_diagnostics")
    if not isinstance(diag, dict):
        return None
    return diag


def compute_sample_weight(*, margin: float, margin_z: float | None, config: SubsetFilterConfig) -> float:
    """
    Calcola il peso CE per una correzione tenuta.

    - `uniform`: 1.0 (utile per isolare l'effetto del solo filtro);
    - `margin`: il margine grezzo (effect size: quanti punti medi guadagna la correzione);
    - `margin_clip`: il margine clampato in `[min_margin, clip_max]`, per evitare che poche
      correzioni estreme dominino il training;
    - `margin_z`: il rapporto margine/SE (confidenza statistica della correzione);
    - `log_margin`: `log1p(margin)`, compressione dolce delle code.

    `margin_z` può essere `None`: succede quando la SE pareggiata è 0, cioè *tutte* le
    determinizzazioni concordano sul margine (correzione a varianza nulla, massimamente
    affidabile). In modalità `margin_z` la mappiamo quindi al tetto `clip_max`, non a 0.

    Il peso è sempre >= 0; `train_bc.py` lo normalizza poi a media 1.0 sul train.
    """
    if config.weight_mode == "uniform":
        return 1.0
    if config.weight_mode == "margin":
        return max(margin, 0.0)
    if config.weight_mode == "margin_clip":
        return float(min(max(margin, config.min_margin), config.clip_max))
    if config.weight_mode == "margin_z":
        if margin_z is None:
            return float(config.clip_max)
        return max(margin_z, 0.0)
    if config.weight_mode == "log_margin":
        return float(math.log1p(max(margin, 0.0)))
    raise ValueError(f"weight_mode non supportato: {config.weight_mode!r}")


def evaluate_record(rec: dict[str, Any], config: SubsetFilterConfig) -> tuple[str, float | None]:
    """
    Decide se tenere un record e, in caso, con quale peso.

    Ritorna `(status, weight)`. `status` è `"kept"` oppure una delle ragioni di scarto
    (`drop_*`), usata per i contatori. `weight` è valorizzato solo quando `status == "kept"`.
    """
    teacher = rec.get("teacher")
    decision_type = teacher.get("decision_type") if isinstance(teacher, dict) else None
    if decision_type != config.decision_type:
        return "drop_decision_type", None

    if config.require_disagree:
        reference = rec.get("reference")
        disagrees = reference.get("disagrees_with_teacher") if isinstance(reference, dict) else None
        if disagrees is not True:
            return "drop_agree", None

    diag = _search_diagnostics(rec)
    if diag is None:
        return "drop_missing_diag", None

    # margin e margin_ci95_low decidono l'eleggibilità: se mancano o non sono numerici, il record
    # non è una correzione misurabile e va scartato.
    try:
        margin = float(diag["margin"])
        margin_ci_low = float(diag["margin_ci95_low"])
    except KeyError, TypeError, ValueError:
        return "drop_missing_diag", None

    # margin_z serve solo per la pesatura ed è opzionale: `None` (SE=0, varianza nulla) NON deve
    # squalificare il record, altrimenti scarteremmo proprio le correzioni più affidabili.
    raw_z = diag.get("margin_z")
    try:
        margin_z: float | None = float(raw_z) if raw_z is not None else None
    except TypeError, ValueError:
        margin_z = None
    if margin_z is not None and not math.isfinite(margin_z):
        margin_z = None

    if margin < config.min_margin:
        return "drop_low_margin", None
    if margin_ci_low < config.min_ci_low:
        return "drop_low_ci_low", None

    weight = compute_sample_weight(margin=margin, margin_z=margin_z, config=config)
    return "kept", weight


def _annotated_record(rec: dict[str, Any], *, weight: float, config: SubsetFilterConfig) -> dict[str, Any]:
    """Restituisce il record originale con `sample_weight` e provenienza del filtro."""
    out = dict(rec)
    out["sample_weight"] = float(weight)
    out["subset_filter"] = {
        "source": "filter_pimc_teacher_subset",
        **asdict(config),
    }
    return out


def filter_records(
    records: Iterable[dict[str, Any]],
    config: SubsetFilterConfig,
) -> Iterator[tuple[dict[str, Any] | None, str, float | None]]:
    """Versione testabile/streaming: per ogni record produce `(record_annotato|None, status, weight)`."""
    for rec in records:
        status, weight = evaluate_record(rec, config)
        if status == "kept" and weight is not None:
            yield _annotated_record(rec, weight=weight, config=config), status, weight
        else:
            yield None, status, weight


def filter_dataset(in_path: Path, out_path: Path, config: SubsetFilterConfig) -> dict[str, Any]:
    """Filtra `in_path` verso `out_path` e ritorna contatori + statistiche dei pesi."""
    counters: dict[str, int] = {
        "records_seen": 0,
        "records_kept": 0,
        "drop_decision_type": 0,
        "drop_agree": 0,
        "drop_missing_diag": 0,
        "drop_low_margin": 0,
        "drop_low_ci_low": 0,
    }
    weight_min = math.inf
    weight_max = -math.inf
    weight_sum = 0.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for annotated, status, weight in filter_records(_iter_jsonl(in_path), config):
            counters["records_seen"] += 1
            if annotated is None or weight is None:
                counters[status] += 1
                continue
            counters["records_kept"] += 1
            weight_min = min(weight_min, weight)
            weight_max = max(weight_max, weight)
            weight_sum += weight
            out.write(json.dumps(annotated, ensure_ascii=False, sort_keys=True) + "\n")

    kept = counters["records_kept"]
    summary: dict[str, Any] = {
        "config": asdict(config),
        "counters": counters,
        "weight_stats": {
            "mode": config.weight_mode,
            "min": float(weight_min) if kept else 0.0,
            "max": float(weight_max) if kept else 0.0,
            "mean": float(weight_sum / kept) if kept else 0.0,
        },
    }
    return summary


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filtra il dataset teacher PIMC nelle correzioni di search affidabili."
    )
    parser.add_argument("--in", dest="in_path", required=True, help="JSONL diagnostico in input.")
    parser.add_argument("--out", dest="out_path", required=True, help="JSONL subset in output.")
    parser.add_argument(
        "--decision-type",
        default="search",
        help="Tipo di decisione teacher da tenere. Default: search.",
    )
    parser.add_argument(
        "--keep-agreements",
        action="store_true",
        help="Tieni anche le righe dove il teacher concorda col modello base. Default: solo disaccordi.",
    )
    parser.add_argument("--min-margin", type=float, default=2.0, help="Soglia minima sul margine medio. Default: 2.0.")
    parser.add_argument(
        "--min-ci-low",
        type=float,
        default=0.0,
        help="Soglia minima su margin_ci95_low (affidabilità). Default: 0.0.",
    )
    parser.add_argument(
        "--weight-mode",
        choices=_WEIGHT_MODES,
        default="margin_clip",
        help="Come calcolare sample_weight. Default: margin_clip.",
    )
    parser.add_argument(
        "--clip-max",
        type=float,
        default=10.0,
        help="Tetto del margine per la modalità margin_clip. Default: 10.0.",
    )
    return parser


def main() -> int:
    args = _build_cli_parser().parse_args()
    config = SubsetFilterConfig(
        decision_type=str(args.decision_type),
        require_disagree=not bool(args.keep_agreements),
        min_margin=float(args.min_margin),
        min_ci_low=float(args.min_ci_low),
        weight_mode=str(args.weight_mode),  # type: ignore[arg-type]
        clip_max=float(args.clip_max),
    )
    summary = filter_dataset(Path(args.in_path), Path(args.out_path), config)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
