#!/usr/bin/env python3
"""
Aggiorna il badge di coverage (locale) in `coverage.svg`.

Perché un badge locale?
- Non richiede CI/servizi esterni (CodeCov, ecc.)
- È riproducibile e “versionabile” nel repo (utile per un progetto didattico)

Uso:
  python scripts/update_coverage_badge.py

Prerequisiti:
  - dipendenze dev installate: `uv pip install -e ".[dev]"`
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _color_for_percent(percent: int) -> str:
    if percent >= 90:
        return "#2ecc71"  # bright green
    if percent >= 80:
        return "#27ae60"  # green
    if percent >= 70:
        return "#f1c40f"  # yellow
    if percent >= 60:
        return "#f39c12"  # orange
    return "#e74c3c"  # red


def _render_svg(percent: int) -> str:
    label = "coverage"
    value = f"{percent}%"
    label_width = 70
    value_width = 52
    total_width = label_width + value_width
    color = _color_for_percent(percent)

    aria_label = f"{label}: {value}"
    header = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20" '
        f'role="img" aria-label="{aria_label}">'
    )
    return f"""{header}
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r">
    <rect width="{total_width}" height="20" rx="3" fill="#fff"/>
  </clipPath>
  <g clip-path="url(#r)">
    <rect width="{label_width}" height="20" fill="#555"/>
    <rect x="{label_width}" width="{value_width}" height="20" fill="{color}"/>
    <rect width="{total_width}" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">
    <text x="{label_width / 2}" y="15" fill="#010101" fill-opacity=".3">{label}</text>
    <text x="{label_width / 2}" y="14">{label}</text>
    <text x="{label_width + value_width / 2}" y="15" fill="#010101" fill-opacity=".3">{value}</text>
    <text x="{label_width + value_width / 2}" y="14">{value}</text>
  </g>
</svg>
"""


def main() -> int:
    tmp_json = ROOT / ".coverage.json"
    try:
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--cov=briscola_ai", "--cov-report=term"],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "coverage", "json", "-o", str(tmp_json)],
            cwd=ROOT,
            check=True,
        )
        data = json.loads(tmp_json.read_text(encoding="utf-8"))
        percent = int(round(float(data["totals"]["percent_covered"])))
        (ROOT / "coverage.svg").write_text(_render_svg(percent), encoding="utf-8")
        print(f"coverage.svg aggiornato: {percent}%")
        return 0
    finally:
        try:
            tmp_json.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
