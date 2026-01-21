"""
Valutazione offline di agenti (dominio-only).

Perché serve?
-------------
Quando modifichiamo un agente (euristica, rete neurale, ecc.), vogliamo un modo
semplice e riproducibile per rispondere a:
- “È meglio di random?”
- “Quanto è stabile su seed diversi?”
- “Quante partite vince e con che margine?”

Questa valutazione non usa HTTP/WS né UI:
lavora direttamente su `GameState + step()` per essere veloce e deterministica.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from ..domain.engine import PlayCardAction, step
from ..domain.state import GameState, new_game_state
from .agents import Agent


@dataclass(frozen=True)
class MatchStats:
    """
    Risultati aggregati di una valutazione.

    Nota:
    in 2-player, ogni partita ha 120 punti totali; quindi è comodo usare anche
    il check `total_points == 120 * num_games` come sanity check.
    """

    num_games: int
    agent0_name: str
    agent1_name: str
    wins_agent0: int
    wins_agent1: int
    draws: int
    avg_points_agent0: float
    avg_points_agent1: float
    avg_point_diff_agent0_minus_agent1: float


def _winner_index_2p(state: GameState) -> Optional[int]:
    """Ritorna 0/1 se c'è un vincitore, altrimenti None (pareggio)."""
    p0 = state.players[0].points
    p1 = state.players[1].points
    if p0 > p1:
        return 0
    if p1 > p0:
        return 1
    return None


def play_one_game_2p(
    agent0: Agent,
    agent1: Agent,
    *,
    rng: random.Random,
    game_seed: int,
) -> GameState:
    """
    Simula una singola partita 2-player.

    - `agent0` controlla player 0
    - `agent1` controlla player 1
    - `rng` controlla la stochasticity dell'agente (scelte) per riproducibilità
    - `game_seed` controlla lo shuffle del mazzo (dominio) per riproducibilità
    """
    state = new_game_state(num_players=2, seed=game_seed)

    safety = 5000
    while not state.game_over and safety > 0:
        safety -= 1
        current = state.current_turn
        agent = agent0 if current == 0 else agent1
        card_index = agent.choose_card_index(state, current, rng=rng)

        state, result = step(state, PlayCardAction(player_index=current, card_index=card_index))
        if result.error:
            raise RuntimeError(f"Errore dominio durante la simulazione: {result.error}")

    if safety <= 0:
        raise RuntimeError("Loop di sicurezza: la partita non termina")

    return state


def evaluate_match_2p(
    agent0: Agent,
    agent1: Agent,
    *,
    num_games: int,
    seed: int,
) -> MatchStats:
    """
    Valuta `agent0` vs `agent1` in 2-player.

    Scelta di design:
    - Usiamo un RNG “master” per generare `game_seed` (shuffle) e per tutte le scelte agenti.
      In questo modo la valutazione è riproducibile dato `seed`.
    """
    rng = random.Random(seed)

    wins0 = 0
    wins1 = 0
    draws = 0
    sum0 = 0
    sum1 = 0
    sum_diff = 0

    for _ in range(num_games):
        game_seed = rng.randrange(0, 2**32)
        final_state = play_one_game_2p(agent0, agent1, rng=rng, game_seed=game_seed)

        p0 = final_state.players[0].points
        p1 = final_state.players[1].points
        sum0 += p0
        sum1 += p1
        sum_diff += p0 - p1

        winner = _winner_index_2p(final_state)
        if winner is None:
            draws += 1
        elif winner == 0:
            wins0 += 1
        else:
            wins1 += 1

    return MatchStats(
        num_games=num_games,
        agent0_name=agent0.name,
        agent1_name=agent1.name,
        wins_agent0=wins0,
        wins_agent1=wins1,
        draws=draws,
        avg_points_agent0=sum0 / num_games if num_games else 0.0,
        avg_points_agent1=sum1 / num_games if num_games else 0.0,
        avg_point_diff_agent0_minus_agent1=sum_diff / num_games if num_games else 0.0,
    )
