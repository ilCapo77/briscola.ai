"""Solver e agenti di supporto per il finale a informazione perfetta."""

from .fast_solver import solve_endgame_fast
from .solver import EndgameSolution, solve_endgame

__all__ = ["EndgameSolution", "solve_endgame", "solve_endgame_fast"]
