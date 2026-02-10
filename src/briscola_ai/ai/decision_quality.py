"""
Metriche di “qualità decisionale” per agenti Briscola (didattico).

Perché servono
--------------
Le metriche standard (win-rate, avg diff punti) dicono *quanto* un agente è forte,
ma non sempre spiegano *come* gioca.

In pratica, quando osservi una policy forte che però “spreca briscole” (es. usa una
briscola alta per prendere uno scarto), vuoi una metrica che renda quel comportamento
misurabile e quindi ottimizzabile.

Questo modulo definisce una prima metrica semplice (2-player):
- **trump_waste_rate**: quante volte, da secondi di mano, l'agente gioca una briscola
  pur avendo almeno una risposta vincente non-briscola.

Nota anti-cheat
---------------
Le metriche sono calcolate usando lo stato completo del dominio (per simulare),
ma la policy continua a ricevere solo `PlayerObservation`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Sequence

from ..domain.engine import PlayCardAction, step
from ..domain.observation import make_player_observation
from ..domain.rules import who_wins_trick
from ..domain.state import GameState, new_game_state
from .agents import Agent
from .evaluation import SeatFairStats, _winner_index_2p  # noqa: PLC2701 (didattico, reuse interno)


@dataclass(frozen=True, slots=True)
class DecisionQualityStats:
    """
    Metriche aggregate di qualità decisionale (2-player).

    Convenzioni:
    - Consideriamo solo decisioni in cui l'agente è **secondo di mano**
      (table_cards ha lunghezza 1).
    - `trump_waste_*` è definito solo quando esiste almeno una risposta vincente.
    """

    num_second_hand_decisions: int
    num_second_hand_with_winning_reply: int
    num_trump_waste: int

    @property
    def trump_waste_rate(self) -> float:
        """Frazione di sprechi briscola (0..1)."""
        if self.num_second_hand_with_winning_reply <= 0:
            return 0.0
        return float(self.num_trump_waste) / float(self.num_second_hand_with_winning_reply)


def _is_trump_waste_second_hand(*, state: GameState, player_index: int, chosen_card_index: int) -> Optional[bool]:
    """
    Ritorna True/False se la scelta è uno “spreco briscola”, oppure None se non applicabile.

    Applicabile solo in 2-player quando:
    - è il turno di `player_index`
    - sul tavolo c'è già una carta (siamo secondi di mano)
    - esiste almeno una carta che vince la presa
    """
    if state.num_players != 2:
        return None
    if state.game_over:
        return None
    if state.current_turn != player_index:
        return None
    if len(state.table_cards) != 1:
        return None

    hand = state.players[player_index].hand
    if chosen_card_index < 0 or chosen_card_index >= len(hand):
        return None

    trump_suit = state.trump_card.suit if state.trump_card else None
    lead_card, lead_player = state.table_cards[0]

    winning_non_trump_exists = False
    winning_any_exists = False
    for i, card in enumerate(hand):
        trick_cards = ((lead_card, lead_player), (card, player_index))
        winner = who_wins_trick(trick_cards, trump_suit)
        if winner != player_index:
            continue
        winning_any_exists = True
        if trump_suit is None or card.suit != trump_suit:
            winning_non_trump_exists = True

    if not winning_any_exists:
        return None

    chosen_card = hand[chosen_card_index]
    chosen_is_trump = trump_suit is not None and chosen_card.suit == trump_suit
    return bool(chosen_is_trump and winning_non_trump_exists)


def play_one_game_2p_collect_quality(
    agent0: Agent,
    agent1: Agent,
    *,
    tracked_agent_index: int,
    rng: random.Random,
    game_seed: int,
) -> tuple[GameState, DecisionQualityStats]:
    """
    Simula una singola partita 2-player e colleziona `DecisionQualityStats` per un agente.

    `tracked_agent_index` indica quale player (0 o 1) vogliamo misurare in questa partita.
    """
    if tracked_agent_index not in (0, 1):
        raise ValueError("tracked_agent_index deve essere 0 o 1")

    state = new_game_state(num_players=2, seed=game_seed)

    num_second = 0
    num_second_with_win = 0
    num_waste = 0

    safety = 5000
    while not state.game_over and safety > 0:
        safety -= 1
        current = state.current_turn
        agent = agent0 if current == 0 else agent1
        observation = make_player_observation(state, current)
        card_index = agent.choose_card_index(observation, rng=rng)

        if current == tracked_agent_index and len(state.table_cards) == 1:
            num_second += 1
            waste = _is_trump_waste_second_hand(state=state, player_index=current, chosen_card_index=card_index)
            if waste is not None:
                num_second_with_win += 1
                if waste:
                    num_waste += 1

        state, result = step(state, PlayCardAction(player_index=current, card_index=card_index))
        if result.error:
            raise RuntimeError(f"Errore dominio durante la simulazione: {result.error}")

    if safety <= 0:
        raise RuntimeError("Loop di sicurezza: la partita non termina")

    return (
        state,
        DecisionQualityStats(
            num_second_hand_decisions=num_second,
            num_second_hand_with_winning_reply=num_second_with_win,
            num_trump_waste=num_waste,
        ),
    )


@dataclass(frozen=True, slots=True)
class SeatFairStatsWithQuality:
    """Seat-fair stats + metriche qualità per l'agente A."""

    match: SeatFairStats
    quality: DecisionQualityStats


def evaluate_seat_fair_match_2p_with_quality(
    agent_a: Agent,
    agent_b: Agent,
    *,
    num_games: int,
    seed: int,
    game_seeds: Optional[Sequence[int]] = None,
) -> SeatFairStatsWithQuality:
    """
    Valuta A vs B (seat-fair) e colleziona quality metrics per A.

    Nota:
    In seat-fair, A gioca metà partite come P0 e metà come P1.
    """
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

    q_num_second = 0
    q_num_second_with_win = 0
    q_num_waste = 0

    for i in range(num_pairs):
        game_seed = seeds[i]

        # Game 1: A=P0, B=P1
        s1, q1 = play_one_game_2p_collect_quality(
            agent_a, agent_b, tracked_agent_index=0, rng=rng_action, game_seed=game_seed
        )
        p0, p1 = s1.players[0].points, s1.players[1].points
        sum_a += p0
        sum_b += p1
        sum_diff += p0 - p1
        w = _winner_index_2p(s1)
        if w is None:
            draws += 1
        elif w == 0:
            wins_a += 1
        else:
            wins_b += 1

        q_num_second += q1.num_second_hand_decisions
        q_num_second_with_win += q1.num_second_hand_with_winning_reply
        q_num_waste += q1.num_trump_waste

        # Game 2: A=P1, B=P0
        s2, q2 = play_one_game_2p_collect_quality(
            agent_b, agent_a, tracked_agent_index=1, rng=rng_action, game_seed=game_seed
        )
        p0, p1 = s2.players[0].points, s2.players[1].points
        sum_b += p0
        sum_a += p1
        sum_diff += p1 - p0
        w = _winner_index_2p(s2)
        if w is None:
            draws += 1
        elif w == 0:
            wins_b += 1
        else:
            wins_a += 1

        q_num_second += q2.num_second_hand_decisions
        q_num_second_with_win += q2.num_second_hand_with_winning_reply
        q_num_waste += q2.num_trump_waste

    match = SeatFairStats(
        num_games=num_games,
        agent_a_name=agent_a.name,
        agent_b_name=agent_b.name,
        wins_agent_a=wins_a,
        wins_agent_b=wins_b,
        draws=draws,
        avg_points_agent_a=sum_a / num_games if num_games else 0.0,
        avg_points_agent_b=sum_b / num_games if num_games else 0.0,
        avg_point_diff_agent_a_minus_agent_b=sum_diff / num_games if num_games else 0.0,
    )
    quality = DecisionQualityStats(
        num_second_hand_decisions=q_num_second,
        num_second_hand_with_winning_reply=q_num_second_with_win,
        num_trump_waste=q_num_waste,
    )
    return SeatFairStatsWithQuality(match=match, quality=quality)
