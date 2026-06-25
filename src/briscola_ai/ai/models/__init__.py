"""Modelli addestrati e catalogo degli artefatti `.npz`."""

from .bc_model import BCModelAgent, LoadedBCModel, MLPBCModel, load_bc_model_npz
from .catalog import (
    LocalModelSpec,
    get_models_dir_from_env,
    list_local_models,
    resolve_model_path,
    validate_model_compatible_for_ui,
)
from .provisioning import ensure_model_available

__all__ = [
    "BCModelAgent",
    "LoadedBCModel",
    "MLPBCModel",
    "LocalModelSpec",
    "ensure_model_available",
    "get_models_dir_from_env",
    "list_local_models",
    "load_bc_model_npz",
    "resolve_model_path",
    "validate_model_compatible_for_ui",
]
