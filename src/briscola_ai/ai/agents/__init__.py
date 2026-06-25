"""
Agenti giocabili per Briscola AI.

Facciata pubblica del package `ai.agents`: mantiene import comodi come
`from briscola_ai.ai.agents import build_agent`, mentre l'implementazione e'
divisa per responsabilita' nei moduli interni.
"""

from .base import Agent, AgentSpec
from .hybrid_endgame import HybridEndgameAgent, can_solve_endgame_from_observation, reconstruct_endgame_state
from .registry import (
    AI_AGENTS_COMMON_NOTE_IT,
    BC_MODEL_SPEC,
    BEST_A2C_SPEC,
    HYBRID_ENDGAME_BEST_A2C_SPEC,
    build_agent,
    list_agent_specs,
)
from .rule_based import GreedyPointsAgent, HeuristicAgentV1, HeuristicAgentV2, RandomAgent, card_to_short_string

__all__ = [
    "AI_AGENTS_COMMON_NOTE_IT",
    "BC_MODEL_SPEC",
    "BEST_A2C_SPEC",
    "HYBRID_ENDGAME_BEST_A2C_SPEC",
    "Agent",
    "AgentSpec",
    "GreedyPointsAgent",
    "HeuristicAgentV1",
    "HeuristicAgentV2",
    "HybridEndgameAgent",
    "RandomAgent",
    "build_agent",
    "can_solve_endgame_from_observation",
    "card_to_short_string",
    "list_agent_specs",
    "reconstruct_endgame_state",
]
