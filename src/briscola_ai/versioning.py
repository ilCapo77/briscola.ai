"""
Versioni “stabili” per dataset e riproducibilità.

Obiettivo
---------
Quando esportiamo dataset (o salviamo eventi su SQLite), vogliamo poter rispondere
a domande come:
- con quale versione del codice è stata generata questa partita?
- con quale versione delle regole (dominio) è stata giocata?

Questo modulo fornisce helper minimali e best-effort:
- `get_code_version()`: versione del pacchetto (SemVer) con override via env
- `get_rules_version()`: versione regole del dominio (stringa)
"""

from __future__ import annotations

import os
from importlib import metadata

from .domain.version import RULES_VERSION


def get_rules_version() -> str:
    """
    Versione delle regole del dominio.

    È una stringa per semplicità (facile da serializzare e confrontare).
    """
    return RULES_VERSION


def get_code_version() -> str:
    """
    Versione del codice (best-effort).

    Ordine di precedenza:
    1) `BRISCOLA_CODE_VERSION` (env) → utile in deploy/CI per iniettare un commit hash o build id.
    2) versione del pacchetto installato (`pyproject.toml` / metadata).
    3) fallback `"unknown"` se non disponibile.
    """
    override = os.getenv("BRISCOLA_CODE_VERSION")
    if override:
        return override.strip()

    try:
        return metadata.version("briscola-ai")
    except metadata.PackageNotFoundError:
        return "unknown"
