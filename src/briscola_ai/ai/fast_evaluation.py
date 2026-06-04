"""
Valutazione sperimentale basata su `fast_2p`.

Questo modulo è volutamente separato da `ai.evaluation`:
- `ai.evaluation` resta il path canonico, anti-cheat e compatibile con tutti gli agenti;
- questo path fast supporta solo agenti semplici traducibili direttamente su card id numerici.

L'obiettivo è misurare il guadagno del motore mutabile 2-player prima di integrarlo in training/evaluation
più complessi. I test devono dimostrare equivalenza aggregata col dominio per gli agenti supportati.
"""

from __future__ import annotations

import random
from typing import Optional, Sequence

from .evaluation import MatchStats, SeatFairStats
from .fast_2p import CARD_POINTS, Fast2PState, new_fast_2p_state, step_fast_2p

FAST_EVALUATION_AGENT_NAMES: frozenset[str] = frozenset({"random", "greedy_points"})


def _validate_fast_agent_name(agent_name: str) -> None:
    """Fallisce presto se un agente non è ancora supportato dal path fast."""
    if agent_name not in FAST_EVALUATION_AGENT_NAMES:
        supported = ", ".join(sorted(FAST_EVALUATION_AGENT_NAMES))
        raise ValueError(f"`--engine fast` supporta solo agenti semplici: {supported}. Ottenuto: {agent_name!r}")


def choose_fast_card_index(
    agent_name: str,
    state: Fast2PState,
    player_index: int,
    *,
    rng: random.Random,
) -> int:
    """
    Sceglie una carta usando solo lo stato numerico fast.

    Gli agenti implementati qui devono consumare RNG nello stesso modo degli agenti canonici equivalenti,
    così i test possono confrontare esiti fast/domain con lo stesso seed.
    """
    hand = state.hands[player_index]
    if not hand:
        raise ValueError("Mano vuota: nessuna azione possibile")

    if agent_name == "random":
        return rng.randrange(len(hand))

    if agent_name == "greedy_points":
        best_points = max(CARD_POINTS[card_id] for card_id in hand)
        candidates = [i for i, card_id in enumerate(hand) if CARD_POINTS[card_id] == best_points]
        return candidates[rng.randrange(len(candidates))]

    _validate_fast_agent_name(agent_name)
    raise AssertionError("Agente validato ma non implementato nel path fast")


def _winner_index_fast_2p(state: Fast2PState) -> Optional[int]:
    """Ritorna 0/1 se c'è un vincitore, altrimenti None (pareggio)."""
    if state.points[0] > state.points[1]:
        return 0
    if state.points[1] > state.points[0]:
        return 1
    return None


def play_one_fast_game_2p(
    agent0_name: str,
    agent1_name: str,
    *,
    rng: random.Random,
    game_seed: int,
) -> Fast2PState:
    """
    Simula una singola partita 2-player con `fast_2p`.

    Supporta solo agenti semplici (`random`, `greedy_points`) perché non costruisce `PlayerObservation`.
    """
    _validate_fast_agent_name(agent0_name)
    _validate_fast_agent_name(agent1_name)

    state = new_fast_2p_state(seed=game_seed)
    safety = 5000
    while not state.game_over and safety > 0:
        safety -= 1
        current = state.current_turn
        agent_name = agent0_name if current == 0 else agent1_name
        card_index = choose_fast_card_index(agent_name, state, current, rng=rng)
        step_fast_2p(state, player_index=current, card_index=card_index)

    if safety <= 0:
        raise RuntimeError("Loop di sicurezza: la partita fast non termina")

    return state


def evaluate_fast_match_2p(
    agent0_name: str,
    agent1_name: str,
    *,
    num_games: int,
    seed: int,
    game_seeds: Optional[Sequence[int]] = None,
) -> MatchStats:
    """
    Valuta due agenti semplici usando `fast_2p`.

    La semantica di seed replica `evaluate_match_2p`: un RNG per gli shuffle e uno per le azioni.
    """
    _validate_fast_agent_name(agent0_name)
    _validate_fast_agent_name(agent1_name)

    rng_game = random.Random(seed)
    rng_action = random.Random(seed ^ 0x9E3779B9)

    seeds = list(game_seeds) if game_seeds is not None else [rng_game.randrange(0, 2**32) for _ in range(num_games)]
    if len(seeds) < num_games:
        raise ValueError(f"game_seeds insufficiente: attesi >= {num_games}, ottenuti {len(seeds)}")

    wins0 = 0
    wins1 = 0
    draws = 0
    sum0 = 0
    sum1 = 0
    sum_diff = 0

    for i in range(num_games):
        final_state = play_one_fast_game_2p(agent0_name, agent1_name, rng=rng_action, game_seed=seeds[i])
        p0, p1 = final_state.points[0], final_state.points[1]
        sum0 += p0
        sum1 += p1
        sum_diff += p0 - p1

        winner = _winner_index_fast_2p(final_state)
        if winner is None:
            draws += 1
        elif winner == 0:
            wins0 += 1
        else:
            wins1 += 1

    return MatchStats(
        num_games=num_games,
        agent0_name=agent0_name,
        agent1_name=agent1_name,
        wins_agent0=wins0,
        wins_agent1=wins1,
        draws=draws,
        avg_points_agent0=sum0 / num_games if num_games else 0.0,
        avg_points_agent1=sum1 / num_games if num_games else 0.0,
        avg_point_diff_agent0_minus_agent1=sum_diff / num_games if num_games else 0.0,
    )


def evaluate_fast_seat_fair_match_2p(
    agent_a_name: str,
    agent_b_name: str,
    *,
    num_games: int,
    seed: int,
    game_seeds: Optional[Sequence[int]] = None,
) -> SeatFairStats:
    """
    Valuta due agenti semplici in modalità seat-fair usando `fast_2p`.

    `num_games` deve essere pari, come nel path canonico.
    """
    _validate_fast_agent_name(agent_a_name)
    _validate_fast_agent_name(agent_b_name)
    if num_games % 2 != 0:
        raise ValueError("Per la valutazione seat-fair `num_games` deve essere pari (giochiamo a coppie).")

    rng_game = random.Random(seed)
    rng_action = random.Random(seed ^ 0x9E3779B9)
    num_pairs = num_games // 2

    seeds = list(game_seeds) if game_seeds is not None else [rng_game.randrange(0, 2**32) for _ in range(num_pairs)]
    if len(seeds) < num_pairs:
        raise ValueError(f"game_seeds insufficiente: attesi >= {num_pairs}, ottenuti {len(seeds)}")

    wins_a = 0
    wins_b = 0
    draws = 0
    sum_a = 0
    sum_b = 0
    sum_diff = 0

    for i in range(num_pairs):
        game_seed = seeds[i]

        s1 = play_one_fast_game_2p(agent_a_name, agent_b_name, rng=rng_action, game_seed=game_seed)
        p0, p1 = s1.points[0], s1.points[1]
        sum_a += p0
        sum_b += p1
        sum_diff += p0 - p1
        w = _winner_index_fast_2p(s1)
        if w is None:
            draws += 1
        elif w == 0:
            wins_a += 1
        else:
            wins_b += 1

        s2 = play_one_fast_game_2p(agent_b_name, agent_a_name, rng=rng_action, game_seed=game_seed)
        p0, p1 = s2.points[0], s2.points[1]
        sum_a += p1
        sum_b += p0
        sum_diff += p1 - p0
        w = _winner_index_fast_2p(s2)
        if w is None:
            draws += 1
        elif w == 0:
            wins_b += 1
        else:
            wins_a += 1

    return SeatFairStats(
        num_games=num_games,
        agent_a_name=agent_a_name,
        agent_b_name=agent_b_name,
        wins_agent_a=wins_a,
        wins_agent_b=wins_b,
        draws=draws,
        avg_points_agent_a=sum_a / num_games if num_games else 0.0,
        avg_points_agent_b=sum_b / num_games if num_games else 0.0,
        avg_point_diff_agent_a_minus_agent_b=sum_diff / num_games if num_games else 0.0,
    )
