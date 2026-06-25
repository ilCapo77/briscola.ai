"""
Registry e factory degli agenti.

Qui vive il catalogo stabile usato da CLI, backend e UI. Le implementazioni concrete
stanno nei moduli specializzati; questa factory e' l'unico punto che decide come
costruirle a partire dal nome canonico.
"""

from __future__ import annotations

from pathlib import Path

from ..encoding.observation_encoder import FEATURE_DIM_2P_V1, FEATURE_DIM_2P_V2, FEATURE_DIM_2P_V3
from ..models.bc_model import BCModelAgent
from ..models.catalog import get_models_dir_from_env, resolve_model_path
from .base import Agent, AgentSpec
from .hybrid_endgame import HybridEndgameAgent
from .rule_based import GreedyPointsAgent, HeuristicAgentV1, HeuristicAgentV2, RandomAgent

_AGENT_BUILDERS: dict[str, type[Agent]] = {
    "random": RandomAgent,
    "greedy_points": GreedyPointsAgent,
    "heuristic_v1": HeuristicAgentV1,
    "heuristic_v2": HeuristicAgentV2,
    "hybrid_endgame": HybridEndgameAgent,
}

BC_MODEL_SPEC = AgentSpec(
    name="bc_model",
    label="Modello locale (.npz)",
    description_it=(
        "Usa un modello addestrato e salvato in un file `.npz` (Behavior Cloning / RL). "
        "Il file è scelto dalla UI tra quelli disponibili sul server."
    ),
)

BEST_A2C_SPEC = AgentSpec(
    name="best_a2c",
    label="Best A2C (locale)",
    description_it=(
        "Carica un modello “campione” A2C da un file locale `best_a2c.npz` nella directory modelli. "
        "È pensato per training in stile league (avversario congelato) e per confronti riproducibili."
    ),
    requires_model_id="best_a2c.npz",
)

_BEST_A2C_DEFAULT_MODEL_ID = "best_a2c.npz"

HYBRID_ENDGAME_BEST_A2C_SPEC = AgentSpec(
    name="hybrid_endgame_best_a2c",
    label="Hybrid Endgame (Best A2C)",
    description_it=(
        "Come Hybrid Endgame, ma usa il modello `best_a2c.npz` come policy in mid-game e il solver "
        "esatto a mazzo vuoto. Richiede il file `best_a2c.npz` nella directory modelli (non sempre "
        "presente: in tal caso l'opzione è non disponibile)."
    ),
    requires_model_id="best_a2c.npz",
)

AI_AGENTS_COMMON_NOTE_IT = (
    "Nota anti-cheat: tutte le IA ricevono solo un’osservazione parziale (PlayerObservation). "
    "Non possono leggere l’ordine del mazzo né le carte specifiche in mano all’avversario."
)


def list_agent_specs() -> list[AgentSpec]:
    """Ritorna la lista di agenti disponibili con metadati (ordine stabile)."""
    return [
        RandomAgent.spec,
        GreedyPointsAgent.spec,
        HeuristicAgentV1.spec,
        HeuristicAgentV2.spec,
        HybridEndgameAgent.spec,
        HYBRID_ENDGAME_BEST_A2C_SPEC,
        BC_MODEL_SPEC,
    ]


def _load_best_a2c_agent() -> BCModelAgent:
    """
    Carica il modello campione `best_a2c.npz` dalla directory modelli e ne valida la compatibilità.

    Estratto come helper perché serve sia all'agente `best_a2c` sia a `hybrid_endgame_best_a2c`
    (che lo usa come policy mid-game), così la logica di risoluzione path/validazione resta unica.
    """
    models_dir = get_models_dir_from_env()
    try:
        path = resolve_model_path(models_dir=models_dir, model_id=_BEST_A2C_DEFAULT_MODEL_ID)
    except FileNotFoundError as exc:
        raise ValueError(
            "Modello 'best_a2c' non disponibile: file non trovato. "
            "Convenzione: salva (o copia) un modello `.npz` compatibile in "
            f"{models_dir.resolve()!s}/{_BEST_A2C_DEFAULT_MODEL_ID}. "
            "Puoi cambiare directory impostando `BRISCOLA_MODELS_DIR`."
        ) from exc

    agent = BCModelAgent.from_npz(path)
    supported = {int(FEATURE_DIM_2P_V1), int(FEATURE_DIM_2P_V2), int(FEATURE_DIM_2P_V3)}
    if int(agent.model.feature_dim) not in supported:
        expected = f"{int(FEATURE_DIM_2P_V1)} (v1), {int(FEATURE_DIM_2P_V2)} (v2) or {int(FEATURE_DIM_2P_V3)} (v3)"
        raise ValueError(
            "Modello 'best_a2c' non compatibile: feature_dim non coerente con un encoder 2-player supportato. "
            f"model={int(agent.model.feature_dim)} expected={expected} ({path})."
        )
    return agent


def build_agent(name: str, *, model_path: Path | None = None) -> Agent:
    """
    Costruisce un agente a partire dal nome canonico.

    Nota:
    usiamo una mappa esplicita (no import dinamici) per semplicità e riproducibilità.
    """
    if name == "best_a2c":
        return _load_best_a2c_agent()

    if name == "hybrid_endgame_best_a2c":
        # Variante esplicita di hybrid_endgame con policy mid-game = best_a2c.
        # `hybrid_endgame` resta invariato (fallback heuristic_v2) per stabilità dei benchmark.
        return HybridEndgameAgent(fallback=_load_best_a2c_agent(), name="hybrid_endgame_best_a2c")

    if name == "bc_model":
        if model_path is None:
            raise ValueError("Agente 'bc_model' richiede `model_path` (file .npz)")
        return BCModelAgent.from_npz(model_path)

    try:
        return _AGENT_BUILDERS[name]()
    except KeyError as exc:
        raise ValueError(f"Agente non supportato: {name!r}") from exc
