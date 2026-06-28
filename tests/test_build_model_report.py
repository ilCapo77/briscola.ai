"""
Regression tests for the curated model-progress Excel report.

The report generator writes XLSX XML directly, so small range mistakes are easy to miss:
the worksheet data can be correct while the embedded chart still points to an older fixed range.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_build_model_report_module():
    """Load the report script as a module without making `scripts/` a package."""
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "build_model_report.py"
    spec = importlib.util.spec_from_file_location("build_model_report", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build_model_report = _load_build_model_report_module()


def test_dashboard_chart_range_tracks_all_progression_rows() -> None:
    """The progression chart must grow when a new official best is added."""
    dashboard_rows = [
        ["Briscola AI - Model Progress Report"],
        ["Curated report for significant models only."],
        [],
        ["Progression curve data"],
        ["Model", "Big holdout vs heuristic_v1", "Training games", "H2H big holdout vs predecessor/current best"],
        ["best_a2c", 16.77, 1_000_000, 0.76],
        ["best_a2c_v3", 17.29, 1_000_000, 0.18],
        ["best_a2c_v4", 17.50, 1_000_000, 0.36],
        ["best_a2c_v5", 17.83, 1_000_000, 0.34],
        [],
        ["Current conclusion"],
    ]

    count = build_model_report.dashboard_progress_row_count(dashboard_rows)
    chart = build_model_report.chart_xml(count)

    assert count == 4
    assert "Dashboard!$A$6:$A$9" in chart
    assert "Dashboard!$B$6:$B$9" in chart
