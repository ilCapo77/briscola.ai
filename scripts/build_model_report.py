#!/usr/bin/env python3
"""
Build the significant-model Excel report.

The report is intentionally curated: it tracks official best models, one teacher
model, and only the rejected candidates that explain an important decision. It
does not try to dump every experiment under `benchmarks/experiments/`.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "reports" / "model_progress.xlsx"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Curated model included in the report."""

    model_id: str
    path: Path
    role: str
    status: str
    order: int
    progress_source: str
    progress_score: float | None
    h2h_source: str
    h2h_score: float | None
    decision: str
    notes: str
    data_quality: str = "exact"


def _rel(path: str) -> Path:
    """Return a repository-relative path as an absolute Path."""
    return ROOT / path


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(
        model_id="bc_v3",
        path=_rel("data/models/bc_v3.npz"),
        role="teacher/anchor",
        status="teacher",
        order=0,
        progress_source="Not plotted: supervised teacher, not a playing baseline.",
        progress_score=None,
        h2h_source="",
        h2h_score=None,
        decision="Use as BC teacher/anchor for v3 A2C runs.",
        notes="Behavior cloning MLP v3; important as init/anchor, not as official best.",
    ),
    ModelSpec(
        model_id="best_a2c",
        path=_rel("data/models/best_a2c.npz"),
        role="official best",
        status="promoted",
        order=1,
        progress_source="benchmarks/experiments/a2c_v2_best_overkill_gap001_1m_seed50_numba/matrix_big.json",
        progress_score=16.77358,
        h2h_source="benchmarks/experiments/a2c_v2_best_overkill_gap001_1m_seed50_numba/head_to_head_best_a2c_v2_big_numba.json",
        h2h_score=0.76442,
        decision="Promoted as v2 best with overkill guard.",
        notes="Strong v2 baseline; remains useful for regression comparisons.",
    ),
    ModelSpec(
        model_id="best_a2c_v3",
        path=_rel("data/models/best_a2c_v3.npz"),
        role="official best",
        status="promoted",
        order=2,
        progress_source="benchmarks/experiments/a2c_v3_league_seed301_1m_numba/baseline_best_a2c_v3_big_vs_heuristic_v1_numba.json",
        progress_score=17.28946,
        h2h_source="benchmarks/experiments/best_a2c_v3_vs_best_a2c_2026-06-28_big_numba.json",
        h2h_score=0.17928,
        decision="Promoted as recommended v3 baseline.",
        notes="Encoder v3, BC/A2C v3 pipeline, guard enabled for runtime/UI.",
    ),
    ModelSpec(
        model_id="best_a2c_v4",
        path=_rel("data/models/best_a2c_v4.npz"),
        role="official best",
        status="promoted",
        order=3,
        progress_source="benchmarks/experiments/a2c_v3_league_seed301_1m_numba/eval_big_vs_heuristic_v1_numba.json",
        progress_score=17.50188,
        h2h_source="benchmarks/experiments/a2c_v3_league_seed301_1m_numba/head_to_head_best_a2c_v3_big_numba.json",
        h2h_score=0.35628,
        decision="Promoted as recommended local/webapp/cloud model.",
        notes="League v3 1M run warm-started from best_a2c_v3 with best_a2c_v3 in the opponent mix.",
    ),
    ModelSpec(
        model_id="best_a2c_v5",
        path=_rel("data/models/best_a2c_v5.npz"),
        role="official best",
        status="promoted",
        order=4,
        progress_source="benchmarks/experiments/a2c_v5_seed401_1m_numba/eval_big_vs_heuristic_v1_numba.json",
        progress_score=17.832,
        h2h_source="benchmarks/experiments/a2c_v5_seed401_1m_numba/head_to_head_best_a2c_v4_big_numba.json",
        h2h_score=0.33972,
        decision="Promoted as recommended model for the v0.11.0 release.",
        notes="League v5 1M run warm-started from best_a2c_v4 with best_a2c_v4 in the opponent mix.",
    ),
    ModelSpec(
        model_id="best_a2c_v6",
        path=_rel("data/models/best_a2c_v6.npz"),
        role="official best",
        status="promoted",
        order=5,
        progress_source="benchmarks/experiments/a2c_v6_scaling_seed501_5m_numba/eval_5m_vs_heuristic_v1_big_numba.json",
        progress_score=18.40148,
        h2h_source="benchmarks/experiments/a2c_v6_scaling_seed501_5m_numba/eval_5m_vs_best_a2c_v5_big_holdout_numba.json",
        h2h_score=0.45866,
        decision="Promoted as recommended model for the v0.12.0 release.",
        notes="Scaling v6 5M run warm-started from best_a2c_v5 with best_a2c_v5 in the opponent mix.",
    ),
]


REJECTED_CANDIDATES: list[dict[str, Any]] = [
    {
        "candidate": "seed301_200k",
        "path": "benchmarks/experiments/a2c_v3_league_seed301_200k_numba/model.npz",
        "training_games": 200000,
        "decision": "not promoted",
        "reason": "Positive big head-to-head, but too small and quality vs heuristic_v1 did not clearly improve.",
        "evidence": "big vs best_a2c_v3 +0.12/+0.13; decision-quality vs heuristic_v1 +16.91.",
    },
    {
        "candidate": "seed302_200k_conservative",
        "path": "benchmarks/experiments/a2c_v3_league_seed302_200k_conservative_numba/model.npz",
        "training_games": 200000,
        "decision": "not promoted",
        "reason": "Did not pass the medium filter against best_a2c_v3.",
        "evidence": "medium vs best_a2c_v3 -0.12/+0.05; vs heuristic_v1 +16.94/+16.95.",
    },
]


MILESTONES: list[dict[str, Any]] = [
    {
        "order": 1,
        "date": "2026-06-08",
        "model_id": "best_a2c",
        "type": "promoted",
        "decision": "Make v2 A2C with overkill guard the official best.",
        "why": "Good big benchmark strength and guard mitigated poor trump overkill behavior.",
        "evidence": "Big holdout vs heuristic_v1 +16.77; promotion H2H source kept in benchmarks.",
        "impact": "Stable v2 baseline for UI and later v3 comparisons.",
        "source": "data/models/best_a2c.npz + a2c_v2_best_overkill_gap001_1m_seed50_numba",
    },
    {
        "order": 2,
        "date": "2026-06-23",
        "model_id": "bc_v3",
        "type": "teacher",
        "decision": "Use encoder v3 BC model as teacher/anchor.",
        "why": "Encoder v3 adds public-history and strategic aggregate features without hidden information.",
        "evidence": "BC v3 metadata: feature_dim 310, 20 epochs.",
        "impact": "Provided a stronger and safer anchor for v3 A2C training.",
        "source": "data/models/bc_v3.npz",
    },
    {
        "order": 3,
        "date": "2026-06-23",
        "model_id": "best_a2c_v3",
        "type": "promoted",
        "decision": "Promote v3 A2C as recommended baseline.",
        "why": "Improved holdout strength and kept overkill under control with runtime guard.",
        "evidence": "Consolidation: big H2H roughly non-regressive vs best_a2c; holdout vs heuristic_v1 improved.",
        "impact": "New default recommended model, v2 best retained for regression.",
        "source": "data/models/best_a2c_v3.npz + PLAN.md",
    },
    {
        "order": 4,
        "date": "2026-06-28",
        "model_id": "seed301_200k",
        "type": "rejected",
        "decision": "Do not promote the 200k league candidate.",
        "why": "The signal was positive but too small, and heuristic_v1 quality was weaker than the v3 best.",
        "evidence": "Big vs best_a2c_v3 +0.12/+0.13; decision-quality +16.91 vs heuristic_v1.",
        "impact": "Confirmed that 200k is only a screening run.",
        "source": "benchmarks/experiments/a2c_v3_league_seed301_200k_numba/",
    },
    {
        "order": 5,
        "date": "2026-06-28",
        "model_id": "seed302_200k_conservative",
        "type": "rejected",
        "decision": "Do not promote the conservative 200k variant.",
        "why": "It did not beat best_a2c_v3 on the medium screen.",
        "evidence": "Medium vs best_a2c_v3 -0.12/+0.05.",
        "impact": "Avoided spending on a big benchmark for a weak candidate.",
        "source": "benchmarks/experiments/a2c_v3_league_seed302_200k_conservative_numba/",
    },
    {
        "order": 6,
        "date": "2026-06-28",
        "model_id": "best_a2c_v4",
        "type": "promoted",
        "decision": "Promote v4 as the recommended local/webapp/cloud model.",
        "why": "It beats best_a2c_v3 head-to-head and does not regress against heuristic_v1.",
        "evidence": "Big vs best_a2c_v3 +0.45/+0.36; big vs heuristic_v1 +17.43/+17.50.",
        "impact": "Frontend/server/cloud default now points to v4 via release asset provisioning.",
        "source": "data/models/best_a2c_v4.npz + a2c_v3_league_seed301_1m_numba",
    },
    {
        "order": 7,
        "date": "2026-06-28",
        "model_id": "best_a2c_v5",
        "type": "promoted",
        "decision": "Promote v5 as the recommended model for v0.11.0.",
        "why": (
            "It beats best_a2c_v4 head-to-head and improves the heuristic_v1 holdout "
            "without material quality regressions."
        ),
        "evidence": (
            "Big vs best_a2c_v4 +0.34; big vs heuristic_v1 +17.83; decision-quality +18.00, overkill 0.0%, waste 0.07%."
        ),
        "impact": "Frontend/server default now points to v5; cloud rollout needs the v0.11.0 asset URL in env.",
        "source": "data/models/best_a2c_v5.npz + a2c_v5_seed401_1m_numba",
    },
    {
        "order": 8,
        "date": "2026-06-28",
        "model_id": "best_a2c_v6",
        "type": "promoted",
        "decision": "Promote v6 as the recommended model for v0.12.0.",
        "why": (
            "The 1M/3M/5M scaling curve improved monotonically, and the 5M checkpoint beat best_a2c_v5 "
            "on both standard and holdout suites without quality regressions."
        ),
        "evidence": (
            "5M big vs best_a2c_v5 +0.46; holdout big vs best_a2c_v5 +0.46; "
            "big vs heuristic_v1 +18.40; decision-quality +18.58, overkill 0.0%, waste 0.07%."
        ),
        "impact": "Frontend/server default now points to v6; cloud rollout needs the v0.12.0 asset URL in env.",
        "source": "data/models/best_a2c_v6.npz + a2c_v6_scaling_seed501_5m_numba",
    },
]


def repo_path(path: Path | str) -> str:
    """Format a path relative to the repository root when possible."""
    p = Path(path)
    if p.is_absolute():
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)
    return str(p)


def load_npz_metadata(path: Path) -> dict[str, Any]:
    """Load `metadata_json` from a model file, returning a small error record if unavailable."""
    if not path.exists():
        return {"_missing": True}
    with np.load(path, allow_pickle=False) as data:
        raw = data.get("metadata_json")
        if raw is None:
            return {"_missing": False, "_metadata_missing": True}
        return json.loads(str(raw))


def load_json(path: str) -> dict[str, Any]:
    """Load a JSON file using a repo-relative path."""
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def short_opponent(name: str) -> str:
    """Normalize verbose bc_model labels to stable report names."""
    match = re.search(r"bc_model\(([^,]+)", name)
    if match:
        return match.group(1)
    return name


def matrix_rows(source: str, *, model_id: str, label: str) -> list[dict[str, Any]]:
    """Flatten an evaluation matrix JSON into report rows."""
    payload = load_json(source)
    out: list[dict[str, Any]] = []
    for row in payload.get("rows", []):
        stats = row["stats"]
        suite = row["suite"]
        out.append(
            {
                "model_id": model_id,
                "label": label,
                "benchmark": payload.get("benchmark"),
                "engine": payload.get("engine"),
                "suite": suite.get("name"),
                "opponent": short_opponent(row.get("opponent") or stats.get("agent_b_name", "")),
                "avg_diff": stats.get("avg_point_diff_agent_a_minus_agent_b"),
                "wins_model": stats.get("wins_agent_a"),
                "wins_opponent": stats.get("wins_agent_b"),
                "draws": stats.get("draws"),
                "eval_games": stats.get("num_games") or payload.get("num_games"),
                "source": source,
                "data_quality": "exact",
            }
        )
    return out


def h2h_rows(source: str, *, model_id: str, label: str, opponent: str) -> list[dict[str, Any]]:
    """Flatten a head-to-head JSON that may be either matrix-shaped or stats-shaped."""
    payload = load_json(source)
    if "rows" in payload:
        return matrix_rows(source, model_id=model_id, label=label)
    stats = payload["stats"]
    return [
        {
            "model_id": model_id,
            "label": label,
            "benchmark": payload.get("benchmark"),
            "engine": payload.get("engine"),
            "suite": payload.get("seed_suite", {}).get("name") or "standard",
            "opponent": opponent,
            "avg_diff": stats.get("avg_point_diff_agent_a_minus_agent_b"),
            "wins_model": stats.get("wins_agent_a"),
            "wins_opponent": stats.get("wins_agent_b"),
            "draws": stats.get("draws"),
            "eval_games": stats.get("num_games") or payload.get("num_games"),
            "source": source,
            "data_quality": "exact",
        }
    ]


def decision_quality_rows() -> list[dict[str, Any]]:
    """Return curated decision-quality rows."""
    sources = [
        (
            "best_a2c_v3",
            "Best A2C v3",
            "benchmarks/experiments/best_a2c_v3_decision_quality_vs_heuristic_v1_2026-06-28_medium_numba.json",
        ),
        (
            "best_a2c_v4",
            "Best A2C v4",
            "benchmarks/experiments/a2c_v3_league_seed301_1m_numba/decision_quality_vs_heuristic_v1_medium_numba.json",
        ),
        (
            "best_a2c_v5",
            "Best A2C v5",
            "benchmarks/experiments/a2c_v5_seed401_1m_numba/decision_quality_vs_heuristic_v1_big_numba.json",
        ),
        (
            "best_a2c_v6",
            "Best A2C v6",
            "benchmarks/experiments/a2c_v6_scaling_seed501_5m_numba/quality_5m_vs_heuristic_v1_big_numba.json",
        ),
    ]
    rows = []
    for model_id, label, source in sources:
        payload = load_json(source)
        quality = payload.get("quality", {})
        match = payload.get("match", {})
        rows.append(
            {
                "model_id": model_id,
                "label": label,
                "benchmark": payload.get("benchmark"),
                "engine": payload.get("engine"),
                "suite": "seat_fair",
                "opponent": "heuristic_v1",
                "avg_diff": match.get("avg_point_diff_agent_a_minus_agent_b"),
                "trump_waste_rate": quality.get("trump_waste_rate"),
                "trump_overkill_rate": quality.get("trump_overkill_rate"),
                "trump_overkill_low_rate": quality.get("trump_overkill_rate_low_lead_points"),
                "eval_games": payload.get("num_games"),
                "source": source,
                "data_quality": "exact",
            }
        )
    return rows


def model_rows() -> list[dict[str, Any]]:
    """Build one summary row per significant model."""
    rows = []
    for spec in MODEL_SPECS:
        meta = load_npz_metadata(spec.path)
        train = meta.get("train") or {}
        rows.append(
            {
                "order": spec.order,
                "model_id": spec.model_id,
                "role": spec.role,
                "status": spec.status,
                "path": repo_path(spec.path),
                "label": meta.get("label", ""),
                "format": meta.get("format", ""),
                "encoder": meta.get("encoder_version", ""),
                "feature_dim": meta.get("feature_dim", ""),
                "training_games": train.get("num_games", ""),
                "seed": meta.get("seed", ""),
                "init": meta.get("init", ""),
                "opponent_mix": json.dumps(meta.get("opponent_mix"), ensure_ascii=False)
                if meta.get("opponent_mix") is not None
                else "",
                "bc_anchor": meta.get("bc_anchor_path", ""),
                "bc_anchor_beta": meta.get("bc_anchor_beta", ""),
                "guard": meta.get("inference_overkill_guard", ""),
                "progress_big_holdout_vs_h1": spec.progress_score,
                "h2h_big_holdout": spec.h2h_score,
                "decision": spec.decision,
                "notes": spec.notes,
                "data_quality": spec.data_quality,
            }
        )
    return rows


def promotion_rows() -> list[dict[str, Any]]:
    """Build the normalized evidence table."""
    rows: list[dict[str, Any]] = []
    rows.extend(
        matrix_rows(
            "benchmarks/experiments/a2c_v2_best_overkill_gap001_1m_seed50_numba/matrix_big.json",
            model_id="best_a2c",
            label="Best A2C v2",
        )
    )
    rows.extend(
        h2h_rows(
            "benchmarks/experiments/a2c_v2_best_overkill_gap001_1m_seed50_numba/head_to_head_best_a2c_v2_big_numba.json",
            model_id="best_a2c",
            label="Best A2C v2",
            opponent="previous_best_v2",
        )
    )
    rows.extend(
        matrix_rows(
            "benchmarks/experiments/a2c_v3_league_seed301_1m_numba/baseline_best_a2c_v3_big_vs_heuristic_v1_numba.json",
            model_id="best_a2c_v3",
            label="Best A2C v3",
        )
    )
    rows.extend(
        matrix_rows(
            "benchmarks/experiments/best_a2c_v3_vs_best_a2c_2026-06-28_big_numba.json",
            model_id="best_a2c_v3",
            label="Best A2C v3",
        )
    )
    rows.extend(
        matrix_rows(
            "benchmarks/experiments/a2c_v3_league_seed301_1m_numba/eval_big_vs_heuristic_v1_numba.json",
            model_id="best_a2c_v4",
            label="Best A2C v4",
        )
    )
    rows.extend(
        matrix_rows(
            "benchmarks/experiments/a2c_v3_league_seed301_1m_numba/head_to_head_best_a2c_v3_big_numba.json",
            model_id="best_a2c_v4",
            label="Best A2C v4",
        )
    )
    rows.extend(
        h2h_rows(
            "benchmarks/experiments/a2c_v5_seed401_1m_numba/eval_big_vs_heuristic_v1_numba.json",
            model_id="best_a2c_v5",
            label="Best A2C v5",
            opponent="heuristic_v1",
        )
    )
    rows.extend(
        h2h_rows(
            "benchmarks/experiments/a2c_v5_seed401_1m_numba/head_to_head_best_a2c_v4_big_numba.json",
            model_id="best_a2c_v5",
            label="Best A2C v5",
            opponent="best_a2c_v4",
        )
    )
    rows.extend(
        h2h_rows(
            "benchmarks/experiments/a2c_v6_scaling_seed501_5m_numba/eval_5m_vs_heuristic_v1_big_numba.json",
            model_id="best_a2c_v6",
            label="Best A2C v6",
            opponent="heuristic_v1",
        )
    )
    rows.extend(
        h2h_rows(
            "benchmarks/experiments/a2c_v6_scaling_seed501_5m_numba/eval_5m_vs_best_a2c_v5_big_numba.json",
            model_id="best_a2c_v6",
            label="Best A2C v6",
            opponent="best_a2c_v5",
        )
    )
    rows.extend(
        h2h_rows(
            "benchmarks/experiments/a2c_v6_scaling_seed501_5m_numba/eval_5m_vs_best_a2c_v5_big_holdout_numba.json",
            model_id="best_a2c_v6",
            label="Best A2C v6",
            opponent="best_a2c_v5_holdout",
        )
    )
    return rows


def sources_rows() -> list[dict[str, Any]]:
    """List sources used by the report."""
    rows = []
    for spec in MODEL_SPECS:
        rows.append(
            {
                "kind": "model",
                "id": spec.model_id,
                "path": repo_path(spec.path),
                "data_quality": spec.data_quality,
                "note": "metadata_json read from .npz",
            }
        )
        if spec.progress_source:
            rows.append(
                {
                    "kind": "progress_metric",
                    "id": spec.model_id,
                    "path": spec.progress_source,
                    "data_quality": "exact" if spec.progress_score is not None else "not_applicable",
                    "note": "source for Dashboard progression score",
                }
            )
        if spec.h2h_source:
            rows.append(
                {
                    "kind": "h2h_metric",
                    "id": spec.model_id,
                    "path": spec.h2h_source,
                    "data_quality": "exact",
                    "note": "source for head-to-head score",
                }
            )
    for row in REJECTED_CANDIDATES:
        rows.append(
            {
                "kind": "rejected_candidate",
                "id": row["candidate"],
                "path": row["path"],
                "data_quality": "manual_summary",
                "note": row["evidence"],
            }
        )
    return rows


def detail_rows(spec: ModelSpec) -> list[list[Any]]:
    """Build a model detail sheet."""
    meta = load_npz_metadata(spec.path)
    train = meta.get("train") or {}
    rows: list[list[Any]] = [
        [f"Detail: {spec.model_id}"],
        [],
        ["Key", "Value"],
        ["Role", spec.role],
        ["Status", spec.status],
        ["Path", repo_path(spec.path)],
        ["Label", meta.get("label", "")],
        ["Format", meta.get("format", "")],
        ["Encoder", meta.get("encoder_version", "")],
        ["Feature dim", meta.get("feature_dim", "")],
        ["Training games", train.get("num_games", "")],
        ["Seed", meta.get("seed", "")],
        ["Init", meta.get("init", "")],
        ["Opponent mix", json.dumps(meta.get("opponent_mix"), ensure_ascii=False) if meta.get("opponent_mix") else ""],
        ["BC anchor", meta.get("bc_anchor_path", "")],
        ["BC anchor beta", meta.get("bc_anchor_beta", "")],
        ["Guard anti-overkill", meta.get("inference_overkill_guard", "")],
        ["Progress score", spec.progress_score if spec.progress_score is not None else ""],
        ["H2H score", spec.h2h_score if spec.h2h_score is not None else ""],
        ["Decision", spec.decision],
        ["Notes", spec.notes],
        [],
        ["Decision milestones"],
        ["Date", "Type", "Decision", "Why", "Evidence"],
    ]
    for milestone in MILESTONES:
        if milestone["model_id"] == spec.model_id:
            rows.append(
                [
                    milestone["date"],
                    milestone["type"],
                    milestone["decision"],
                    milestone["why"],
                    milestone["evidence"],
                ]
            )
    return rows


def sheet_from_dicts(rows: list[dict[str, Any]], columns: list[str]) -> list[list[Any]]:
    """Convert dict rows to a worksheet-like matrix."""
    return [columns] + [[row.get(column, "") for column in columns] for row in rows]


def build_workbook_data() -> dict[str, list[list[Any]]]:
    """Build all report sheets."""
    models = model_rows()
    promotion = promotion_rows()
    quality = decision_quality_rows()
    progress_model_ids = {"best_a2c", "best_a2c_v3", "best_a2c_v4", "best_a2c_v5", "best_a2c_v6"}
    progress_models = [m for m in models if m["model_id"] in progress_model_ids]

    dashboard: list[list[Any]] = [
        ["Briscola AI - Model Progress Report"],
        ["Curated report for significant models only. Generated by scripts/build_model_report.py."],
        [],
        ["Progression curve data"],
        ["Model", "Big holdout vs heuristic_v1", "Training games", "H2H big holdout vs predecessor/current best"],
    ]
    for row in progress_models:
        dashboard.append(
            [
                row["model_id"],
                row["progress_big_holdout_vs_h1"],
                row["training_games"],
                row["h2h_big_holdout"],
            ]
        )
    dashboard.extend(
        [
            [],
            ["Current conclusion"],
            [
                "best_a2c_v6 is the recommended v0.12.0 model: the 5M scaling checkpoint improves "
                "big holdout vs heuristic_v1 and beats best_a2c_v5 head-to-head on standard and holdout suites."
            ],
            [],
            ["Quick comparison"],
            ["Model", "Status", "Encoder", "Training games", "Decision"],
        ]
    )
    for row in models:
        dashboard.append([row["model_id"], row["status"], row["encoder"], row["training_games"], row["decision"]])

    sheets: dict[str, list[list[Any]]] = {
        "Dashboard": dashboard,
        "Milestones": sheet_from_dicts(
            MILESTONES,
            ["order", "date", "model_id", "type", "decision", "why", "evidence", "impact", "source"],
        ),
        "Best Models": sheet_from_dicts(
            models,
            [
                "order",
                "model_id",
                "role",
                "status",
                "path",
                "label",
                "encoder",
                "feature_dim",
                "training_games",
                "seed",
                "init",
                "opponent_mix",
                "bc_anchor",
                "bc_anchor_beta",
                "guard",
                "progress_big_holdout_vs_h1",
                "h2h_big_holdout",
                "decision",
                "notes",
                "data_quality",
            ],
        ),
        "Promotion Evidence": sheet_from_dicts(
            promotion,
            [
                "model_id",
                "label",
                "benchmark",
                "engine",
                "suite",
                "opponent",
                "avg_diff",
                "wins_model",
                "wins_opponent",
                "draws",
                "eval_games",
                "source",
                "data_quality",
            ],
        ),
        "Decision Quality": sheet_from_dicts(
            quality,
            [
                "model_id",
                "label",
                "benchmark",
                "engine",
                "suite",
                "opponent",
                "avg_diff",
                "trump_waste_rate",
                "trump_overkill_rate",
                "trump_overkill_low_rate",
                "eval_games",
                "source",
                "data_quality",
            ],
        ),
        "Rejected Candidates": sheet_from_dicts(
            REJECTED_CANDIDATES,
            ["candidate", "path", "training_games", "decision", "reason", "evidence"],
        ),
        "Sources": sheet_from_dicts(sources_rows(), ["kind", "id", "path", "data_quality", "note"]),
    }
    for spec in MODEL_SPECS:
        sheets[f"Detail {spec.model_id}"[:31]] = detail_rows(spec)
    return sheets


def col_name(index: int) -> str:
    """Return Excel column name for 1-based index."""
    name = ""
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def cell_ref(row: int, col: int, *, absolute: bool = False) -> str:
    """Return an Excel cell reference."""
    c = col_name(col)
    if absolute:
        return f"${c}${row}"
    return f"{c}{row}"


def sheet_xml(rows: list[list[Any]], *, sheet_name: str, drawing_rel: bool = False) -> str:
    """Serialize one worksheet."""
    max_col = max((len(row) for row in rows), default=1)
    max_row = max(len(rows), 1)
    dimension = f"A1:{cell_ref(max_row, max_col)}"
    parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
        f'<dimension ref="{dimension}"/>',
    ]
    if sheet_name != "Dashboard":
        parts.append(
            '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
            'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        )
    else:
        parts.append('<sheetViews><sheetView workbookViewId="0"/></sheetViews>')
    widths = [18, 18, 18, 18, 28, 28, 24, 24, 18, 18, 18, 28, 24, 18, 18, 18, 18, 36, 40]
    parts.append("<cols>")
    for idx in range(1, max_col + 1):
        width = widths[idx - 1] if idx <= len(widths) else 22
        parts.append(f'<col min="{idx}" max="{idx}" width="{width}" customWidth="1"/>')
    parts.append("</cols><sheetData>")
    for r_idx, row in enumerate(rows, start=1):
        parts.append(f'<row r="{r_idx}">')
        for c_idx, value in enumerate(row, start=1):
            if value is None:
                continue
            ref = cell_ref(r_idx, c_idx)
            style = "1" if r_idx == 1 or (r_idx == 5 and sheet_name == "Dashboard") else "0"
            if isinstance(value, bool):
                parts.append(f'<c r="{ref}" s="{style}" t="b"><v>{1 if value else 0}</v></c>')
            elif isinstance(value, int | float) and not isinstance(value, bool):
                number = f"{float(value):.10g}" if isinstance(value, float) else str(value)
                num_style = "2" if isinstance(value, float) else style
                parts.append(f'<c r="{ref}" s="{num_style}"><v>{number}</v></c>')
            else:
                text = escape(str(value))
                wrap_style = "3" if len(str(value)) > 60 else style
                parts.append(f'<c r="{ref}" s="{wrap_style}" t="inlineStr"><is><t>{text}</t></is></c>')
        parts.append("</row>")
    parts.append("</sheetData>")
    if sheet_name != "Dashboard" and max_row > 1 and max_col > 1:
        parts.append(f'<autoFilter ref="A1:{cell_ref(max_row, max_col)}"/>')
    if drawing_rel:
        parts.append('<drawing r:id="rId1"/>')
    parts.append("</worksheet>")
    return "".join(parts)


def workbook_xml(sheet_names: list[str]) -> str:
    """Serialize workbook.xml."""
    sheets = []
    for idx, name in enumerate(sheet_names, start=1):
        sheets.append(f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<workbookPr/>"
        '<bookViews><workbookView xWindow="0" yWindow="0" windowWidth="24000" windowHeight="14000"/></bookViews>'
        f"<sheets>{''.join(sheets)}</sheets>"
        "</workbook>"
    )


def workbook_rels(sheet_names: list[str]) -> str:
    """Serialize workbook relationships."""
    rels = []
    for idx, _ in enumerate(sheet_names, start=1):
        rels.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    rels.append(
        f'<Relationship Id="rId{len(sheet_names) + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{''.join(rels)}</Relationships>"
    )


def content_types_xml(sheet_count: int) -> str:
    """Serialize [Content_Types].xml."""
    overrides = [
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
        '<Override PartName="/xl/drawings/drawing1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.drawing+xml"/>',
        '<Override PartName="/xl/charts/chart1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.drawingml.chart+xml"/>',
    ]
    for idx in range(1, sheet_count + 1):
        overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f"{''.join(overrides)}</Types>"
    )


def root_rels_xml() -> str:
    """Serialize package root relationships."""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )


def styles_xml() -> str:
    """Serialize a compact styles.xml."""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2">'
        '<font><sz val="11"/><color rgb="FF1F2937"/><name val="Aptos"/></font>'
        '<font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Aptos"/></font>'
        "</fonts>"
        '<fills count="3">'
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FF2563EB"/><bgColor indexed="64"/></patternFill></fill>'
        "</fills>"
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="4">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>'
        '<xf numFmtId="4" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1">'
        '<alignment wrapText="1" vertical="top"/></xf>'
        "</cellXfs>"
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        "</styleSheet>"
    )


def sheet_drawing_rels_xml() -> str:
    """Worksheet relationship to the dashboard drawing."""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing" '
        'Target="../drawings/drawing1.xml"/>'
        "</Relationships>"
    )


def drawing_xml() -> str:
    """Drawing anchor for the dashboard chart."""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<xdr:wsDr xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        "<xdr:twoCellAnchor>"
        "<xdr:from><xdr:col>5</xdr:col><xdr:colOff>0</xdr:colOff><xdr:row>3</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:from>"
        "<xdr:to><xdr:col>13</xdr:col><xdr:colOff>0</xdr:colOff><xdr:row>20</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:to>"
        '<xdr:graphicFrame macro="">'
        '<xdr:nvGraphicFramePr><xdr:cNvPr id="2" name="Progression Chart"/>'
        "<xdr:cNvGraphicFramePr/></xdr:nvGraphicFramePr>"
        '<xdr:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/></xdr:xfrm>'
        '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/chart">'
        '<c:chart xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" r:id="rId1"/>'
        "</a:graphicData></a:graphic>"
        "</xdr:graphicFrame>"
        "<xdr:clientData/>"
        "</xdr:twoCellAnchor>"
        "</xdr:wsDr>"
    )


def drawing_rels_xml() -> str:
    """Drawing relationship to the chart."""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart" '
        'Target="../charts/chart1.xml"/>'
        "</Relationships>"
    )


def dashboard_progress_row_count(dashboard_rows: list[list[Any]]) -> int:
    """Count official-best progression rows in the Dashboard sheet.

    The chart range must grow with promoted models. The Dashboard section is deliberately simple:
    title row, header row, then one row per official best until the next blank separator.
    """
    for idx, row in enumerate(dashboard_rows):
        if row[:1] == ["Progression curve data"]:
            data_start = idx + 2
            count = 0
            for data_row in dashboard_rows[data_start:]:
                if not data_row or data_row[0] == "":
                    break
                count += 1
            return count
    return 0


def chart_xml(progress_row_count: int) -> str:
    """Chart XML for the official-best progression curve."""
    first_row = 6
    last_row = max(first_row, first_row + progress_row_count - 1)
    cats_ref = f"Dashboard!$A${first_row}:$A${last_row}"
    vals_ref = f"Dashboard!$B${first_row}:$B${last_row}"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<c:chart>"
        '<c:title><c:tx><c:rich><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="it-IT" sz="1400" b="1"/>'
        "<a:t>Progressione dei best ufficiali</a:t></a:r></a:p></c:rich></c:tx></c:title>"
        "<c:plotArea><c:layout/>"
        '<c:lineChart><c:grouping val="standard"/>'
        '<c:ser><c:idx val="0"/><c:order val="0"/>'
        "<c:tx><c:v>Big holdout vs heuristic_v1</c:v></c:tx>"
        '<c:marker><c:symbol val="circle"/><c:size val="7"/></c:marker>'
        f"<c:cat><c:strRef><c:f>{cats_ref}</c:f></c:strRef></c:cat>"
        f"<c:val><c:numRef><c:f>{vals_ref}</c:f></c:numRef></c:val>"
        "</c:ser>"
        '<c:axId val="100"/><c:axId val="101"/>'
        "</c:lineChart>"
        '<c:catAx><c:axId val="100"/><c:scaling><c:orientation val="minMax"/></c:scaling>'
        '<c:delete val="0"/><c:axPos val="b"/><c:tickLblPos val="nextTo"/><c:crossAx val="101"/>'
        '<c:crosses val="autoZero"/><c:auto val="1"/><c:lblAlgn val="ctr"/><c:lblOffset val="100"/></c:catAx>'
        '<c:valAx><c:axId val="101"/><c:scaling><c:orientation val="minMax"/></c:scaling>'
        '<c:delete val="0"/><c:axPos val="l"/><c:majorGridlines/><c:numFmt formatCode="0.00" sourceLinked="0"/>'
        '<c:tickLblPos val="nextTo"/><c:crossAx val="100"/><c:crosses val="autoZero"/>'
        '<c:crossBetween val="between"/></c:valAx>'
        "</c:plotArea>"
        '<c:legend><c:legendPos val="b"/><c:layout/></c:legend>'
        '<c:plotVisOnly val="1"/>'
        "</c:chart>"
        "</c:chartSpace>"
    )


def write_xlsx(sheets: dict[str, list[list[Any]]], out_path: Path) -> None:
    """Write the report as an .xlsx file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_names = list(sheets)
    progress_row_count = dashboard_progress_row_count(sheets["Dashboard"])
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml(len(sheet_names)))
        zf.writestr("_rels/.rels", root_rels_xml())
        zf.writestr("xl/workbook.xml", workbook_xml(sheet_names))
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels(sheet_names))
        zf.writestr("xl/styles.xml", styles_xml())
        for idx, name in enumerate(sheet_names, start=1):
            zf.writestr(
                f"xl/worksheets/sheet{idx}.xml",
                sheet_xml(sheets[name], sheet_name=name, drawing_rel=(name == "Dashboard")),
            )
            if name == "Dashboard":
                zf.writestr(f"xl/worksheets/_rels/sheet{idx}.xml.rels", sheet_drawing_rels_xml())
        zf.writestr("xl/drawings/drawing1.xml", drawing_xml())
        zf.writestr("xl/drawings/_rels/drawing1.xml.rels", drawing_rels_xml())
        zf.writestr("xl/charts/chart1.xml", chart_xml(progress_row_count))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the significant-model Excel report.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output .xlsx path.")
    args = parser.parse_args()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    sheets = build_workbook_data()
    write_xlsx(sheets, out_path)
    print(f"Wrote {repo_path(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
