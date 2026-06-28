"""
Regression tests for the curated model-progress Excel report.

The report generator writes XLSX XML directly, so small range mistakes are easy to miss:
the worksheet data can be correct while the embedded chart still points to an older fixed range.
"""

from __future__ import annotations

import importlib.util
import sys
import xml.etree.ElementTree as ET
import zipfile
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


def test_generated_xlsx_chart_reaches_latest_official_best(tmp_path: Path) -> None:
    """The committed report workflow must keep the Dashboard chart aligned with the latest best model."""
    out_path = tmp_path / "model_progress.xlsx"
    sheets = build_model_report.build_workbook_data()
    build_model_report.write_xlsx(sheets, out_path)

    dashboard_rows = sheets["Dashboard"]
    progress_count = build_model_report.dashboard_progress_row_count(dashboard_rows)
    progress_models = [row[0] for row in dashboard_rows[5 : 5 + progress_count]]
    expected_models = [
        spec.model_id
        for spec in build_model_report.MODEL_SPECS
        if spec.role == "official best" and spec.status == "promoted" and spec.progress_score is not None
    ]
    first_progress_row = 6
    last_progress_row = first_progress_row + len(expected_models) - 1

    assert progress_models == expected_models
    assert progress_models[-1] == "best_a2c_v6"

    with zipfile.ZipFile(out_path) as zf:
        chart_root = ET.fromstring(zf.read("xl/charts/chart1.xml"))
    ns = {"c": "http://schemas.openxmlformats.org/drawingml/2006/chart"}
    chart_ranges = {node.text for node in chart_root.findall(".//c:f", ns)}

    assert f"Dashboard!$A${first_progress_row}:$A${last_progress_row}" in chart_ranges
    assert f"Dashboard!$B${first_progress_row}:$B${last_progress_row}" in chart_ranges
