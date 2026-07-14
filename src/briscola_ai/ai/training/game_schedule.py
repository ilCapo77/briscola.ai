"""Schedule riproducibili per il training RL a due giocatori.

La schedule separa il campionamento dell'ambiente dall'esecuzione del rollout. In
particolare, la modalità ``paired`` genera due partite adiacenti con lo stesso seed di
mazzo e lo stesso tipo di avversario, assegnando la policy prima alla seat 0 e poi alla
seat 1. Poiché la coppia resta dentro un singolo optimizer update, entrambe le partite
vedono esattamente gli stessi pesi.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from briscola_ai.ai.training.opponent_mix import OpponentMixItem, sample_opponent_name

TrainingScheduleMode = Literal["serial", "paired"]


@dataclass(frozen=True, slots=True)
class ScheduledTrainingGame:
    """Identità dell'ambiente assegnato a una singola partita di training."""

    ordinal: int
    game_seed: int
    policy_seat: int
    opponent_name: str
    pair_index: int | None


def _sample_opponent(
    *,
    default_opponent_name: str,
    opponent_mix: Sequence[OpponentMixItem] | None,
    rng: np.random.Generator,
) -> str:
    """Campiona dal mix oppure restituisce l'unico opponent configurato."""
    if opponent_mix is None:
        return default_opponent_name
    return sample_opponent_name(list(opponent_mix), rng=rng)


def build_training_game_schedule(
    *,
    num_games: int,
    update_every: int,
    mode: TrainingScheduleMode,
    seat_fair: bool,
    default_opponent_name: str,
    opponent_mix: Sequence[OpponentMixItem] | None,
    rng_game: np.random.Generator,
    rng_opponent: np.random.Generator,
) -> tuple[ScheduledTrainingGame, ...]:
    """Costruisce l'intera schedule prima che la policy inizi ad aggiornarsi.

    ``serial`` conserva il comportamento storico: un seed e un campionamento opponent
    per partita; ``seat_fair`` alterna soltanto la seat. ``paired`` campiona seed e
    opponent una volta per coppia e richiede update completi, così nessuna coppia viene
    divisa tra due versioni dei pesi o lasciata fuori dall'ultimo update.
    """
    if num_games <= 0:
        raise ValueError("num_games deve essere > 0")
    if update_every <= 0:
        raise ValueError("update_every deve essere > 0")
    if not default_opponent_name.strip():
        raise ValueError("default_opponent_name non può essere vuoto")
    if mode not in ("serial", "paired"):
        raise ValueError(f"Training schedule non supportata: {mode!r}")

    schedule: list[ScheduledTrainingGame] = []
    if mode == "serial":
        for offset in range(num_games):
            schedule.append(
                ScheduledTrainingGame(
                    ordinal=offset + 1,
                    game_seed=int(rng_game.integers(0, 2**32)),
                    policy_seat=((offset + 1) % 2) if seat_fair else 0,
                    opponent_name=_sample_opponent(
                        default_opponent_name=default_opponent_name,
                        opponent_mix=opponent_mix,
                        rng=rng_opponent,
                    ),
                    pair_index=None,
                )
            )
        return tuple(schedule)

    if num_games % 2 != 0:
        raise ValueError("La training schedule paired richiede --num-games pari")
    if update_every % 2 != 0:
        raise ValueError("La training schedule paired richiede --update-every pari")
    if num_games % update_every != 0:
        raise ValueError(
            "La training schedule paired richiede --num-games multiplo di --update-every, "
            "per non lasciare una coppia in un update parziale"
        )

    for pair_index in range(num_games // 2):
        game_seed = int(rng_game.integers(0, 2**32))
        opponent_name = _sample_opponent(
            default_opponent_name=default_opponent_name,
            opponent_mix=opponent_mix,
            rng=rng_opponent,
        )
        for policy_seat in (0, 1):
            schedule.append(
                ScheduledTrainingGame(
                    ordinal=len(schedule) + 1,
                    game_seed=game_seed,
                    policy_seat=policy_seat,
                    opponent_name=opponent_name,
                    pair_index=pair_index,
                )
            )
    return tuple(schedule)


def training_schedule_sha256(schedule: Sequence[ScheduledTrainingGame]) -> str:
    """Hash stabile di seed, seat, opponent e appartenenza alla coppia."""
    digest = hashlib.sha256()
    for game in schedule:
        pair_index = -1 if game.pair_index is None else game.pair_index
        digest.update(
            f"{game.ordinal}\t{game.game_seed}\t{game.policy_seat}\t{game.opponent_name}\t{pair_index}\n".encode()
        )
    return digest.hexdigest()
