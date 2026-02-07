"""
Valutazione “a matrice” per modelli (seed suite × avversari).

Motivazione
-----------
Quando alleniamo molti modelli (.npz) è facile fare benchmark “a mano” in modo incoerente
o dimenticare un confronto (es. holdout). Una *evaluation matrix* standard:
- riduce errori manuali;
- rende confronti ripetibili nel tempo;
- fornisce una fotografia di robustezza (vs avversari diversi + holdout seed).

Questa implementazione è dominio-only (no HTTP/WS) e riusa `evaluate_seat_fair_match_2p`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .agents import build_agent
from .bc_model_agent import BCModelAgent
from .evaluation import SeatFairStats, evaluate_seat_fair_match_2p

BenchmarkName = Literal["small", "medium", "big"]


@dataclass(frozen=True, slots=True)
class SuiteSpec:
    """Spec di una seed suite generata via range (per big/holdout)."""

    name: str
    range_start: int
    range_step: int
    num_seeds: int


@dataclass(frozen=True, slots=True)
class MatrixRow:
    """Un risultato della matrice: (suite, opponent) -> stats."""

    suite: SuiteSpec
    opponent: str
    stats: SeatFairStats


@dataclass(frozen=True, slots=True)
class EvaluationMatrix:
    """Risultato completo (righe) + metadati di esecuzione."""

    model_path: str
    benchmark: BenchmarkName
    num_games: int
    seed: int
    rows: list[MatrixRow]

    def to_json_dict(self) -> dict:
        """Ritorna una struttura JSON-serializzabile (stabile) per persistere i risultati."""
        return {
            "model_path": self.model_path,
            "benchmark": self.benchmark,
            "num_games": self.num_games,
            "seed": self.seed,
            "rows": [
                {
                    "suite": asdict(r.suite),
                    "opponent": r.opponent,
                    "stats": asdict(r.stats),
                }
                for r in self.rows
            ],
        }

    def to_json_text(self) -> str:
        """JSON pretty-printed."""
        return json.dumps(self.to_json_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def benchmark_num_games(benchmark: BenchmarkName) -> int:
    """Taglie benchmark coerenti con `scripts/evaluate_agents.py` (tutte seat-fair)."""
    if benchmark == "small":
        return 2000
    if benchmark == "medium":
        return 10000
    if benchmark == "big":
        return 100000
    raise ValueError(f"Benchmark non supportato: {benchmark!r}")


def make_range_seed_suite(*, start: int, step: int, count: int) -> list[int]:
    """
    Genera una suite di seed tramite range aritmetico.

    Nota:
    normalizziamo i seed su 32 bit (coerente con uso tipico di RNG).
    """
    if count < 0:
        raise ValueError(f"count deve essere >= 0, ottenuto {count}")
    if step <= 0:
        raise ValueError(f"step deve essere > 0, ottenuto {step}")
    return [((start + i * step) & 0xFFFFFFFF) for i in range(count)]


def build_suites_for_benchmark(
    *,
    benchmark: BenchmarkName,
    standard_start: int = 0,
    holdout_start: int = 1_000_000,
    step: int = 1,
) -> list[SuiteSpec]:
    """
    Ritorna le due suite standard per la matrice:
    - `standard`: start=0
    - `holdout`: start=1_000_000 (default)
    """
    num_games = benchmark_num_games(benchmark)
    needed_seeds = num_games // 2
    return [
        SuiteSpec(name="standard", range_start=standard_start, range_step=step, num_seeds=needed_seeds),
        SuiteSpec(name="holdout", range_start=holdout_start, range_step=step, num_seeds=needed_seeds),
    ]


def evaluate_model_matrix(
    *,
    model_path: str | Path,
    opponents: list[str],
    benchmark: BenchmarkName,
    seed: int,
    standard_start: int = 0,
    holdout_start: int = 1_000_000,
    range_step: int = 1,
) -> EvaluationMatrix:
    """
    Valuta un modello `.npz` contro una lista di avversari su 2 suite (standard + holdout).

    Scelte:
    - sempre seat-fair (riduce bias “chi inizia”)
    - suite generate via range (evitiamo file enormi)
    """
    model = BCModelAgent.from_npz(model_path)
    num_games = benchmark_num_games(benchmark)

    suites = build_suites_for_benchmark(
        benchmark=benchmark,
        standard_start=standard_start,
        holdout_start=holdout_start,
        step=range_step,
    )

    rows: list[MatrixRow] = []
    for suite in suites:
        seeds = make_range_seed_suite(start=suite.range_start, step=suite.range_step, count=suite.num_seeds)
        for opp_name in opponents:
            opponent = build_agent(opp_name)
            stats = evaluate_seat_fair_match_2p(
                model,
                opponent,
                num_games=num_games,
                seed=seed,
                game_seeds=seeds,
            )
            rows.append(MatrixRow(suite=suite, opponent=opp_name, stats=stats))

    return EvaluationMatrix(
        model_path=str(Path(model_path)),
        benchmark=benchmark,
        num_games=num_games,
        seed=int(seed),
        rows=rows,
    )


def default_opponents() -> list[str]:
    """Avversari baseline consigliati per una matrice minima."""
    return ["heuristic_v1", "random", "greedy_points"]


def format_matrix_table(matrix: EvaluationMatrix) -> str:
    """
    Ritorna una tabella testuale compatta.

    Mostriamo:
    - opponent
    - suite
    - avg_point_diff (A-B) dove A = modello, B = avversario
    - win/loss/draw
    """
    lines = []
    lines.append(
        "Evaluation matrix | "
        f"model={Path(matrix.model_path).name} | "
        f"benchmark={matrix.benchmark} games={matrix.num_games}"
    )
    lines.append("opponent,suite,avg_diff,wins_model,wins_opp,draws")
    for r in matrix.rows:
        lines.append(
            f"{r.opponent},{r.suite.name},{r.stats.avg_point_diff_agent_a_minus_agent_b:.2f},"
            f"{r.stats.wins_agent_a},{r.stats.wins_agent_b},{r.stats.draws}"
        )
    return "\n".join(lines) + "\n"
