"""
Dominio "puro" (Phase 2B): stato + transizioni.

Questa cartella avvia la migrazione verso un motore più adatto a ML e replay:
- uno stato esplicito (`GameState`)
- una transizione pura (`step`) che ritorna un nuovo stato e un risultato

Nota:
Per ora il backend continua a usare `BriscolaGame` (stateful) e questi moduli
vivono in parallelo, con test di parità per migrare in sicurezza.
"""

from .engine import StepResult, step
from .state import GameState, new_game_state

__all__ = ["GameState", "StepResult", "new_game_state", "step"]
