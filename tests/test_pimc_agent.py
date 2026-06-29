"""
Test del prototipo PIMC.

Questi test non cercano di dimostrare che PIMC sia forte: bloccano gli invarianti importanti
per un agente a informazione imperfetta, cioè determinizzazione senza leak, fallback fuori scope
e scelta valida quando la search è attiva.
"""

from __future__ import annotations

import random

from briscola_ai.ai.agents import HeuristicAgentV1, HeuristicAgentV2
from briscola_ai.ai.agents.hybrid_endgame import reconstruct_endgame_state
from briscola_ai.ai.agents.pimc import (
    PIMCAgent,
    PIMCSearchStats,
    determinize_observation,
    rollout_to_terminal,
    unknown_live_card_count,
)
from briscola_ai.ai.endgame.solver import solve_endgame
from briscola_ai.domain.card_id import card_to_id
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.observation import make_player_observation
from briscola_ai.domain.state import GameState, new_game_state


class _InvalidIndexAgent:
    """Agente volutamente rotto: serve a testare il fallback difensivo del rollout."""

    name = "invalid_index"

    def choose_card_index(self, observation, *, rng: random.Random) -> int:
        return 999


def _play_with_heuristics_until(*, seed: int, max_deck_size: int) -> GameState:
    """Avanza una partita deterministica finché il mazzo arriva sotto soglia."""
    state = new_game_state(num_players=2, seed=seed)
    agents = (HeuristicAgentV2(), HeuristicAgentV1())
    rng = random.Random(seed ^ 0xC0FFEE)
    safety = 200
    while not state.game_over and len(state.deck) > max_deck_size and safety > 0:
        safety -= 1
        current = state.current_turn
        observation = make_player_observation(state, current)
        card_index = agents[current].choose_card_index(observation, rng=rng)
        state, result = step(state, PlayCardAction(player_index=current, card_index=card_index))
        assert result.error is None
    assert safety > 0
    assert not state.game_over
    return state


def _play_with_heuristics_until_deck_empty(*, seed: int) -> GameState:
    """Avanza fino al primo stato non terminale con mazzo vuoto."""
    state = new_game_state(num_players=2, seed=seed)
    agents = (HeuristicAgentV2(), HeuristicAgentV1())
    rng = random.Random(seed ^ 0xBAD5EED)
    safety = 200
    while not state.game_over and len(state.deck) > 0 and safety > 0:
        safety -= 1
        current = state.current_turn
        observation = make_player_observation(state, current)
        card_index = agents[current].choose_card_index(observation, rng=rng)
        state, result = step(state, PlayCardAction(player_index=current, card_index=card_index))
        assert result.error is None
    assert safety > 0
    assert not state.game_over
    assert len(state.deck) == 0
    return state


def _state_card_ids(state: GameState) -> list[int]:
    """Tutte le carte presenti nello stato completo, incluse prese e tavolo."""
    ids: list[int] = []
    for player in state.players:
        ids.extend(card_to_id(card) for card in player.hand)
        ids.extend(card_to_id(card) for card in player.captured_cards)
    ids.extend(card_to_id(card) for card in state.deck)
    ids.extend(card_to_id(card) for card, _player in state.table_cards)
    return ids


def test_determinize_observation_preserves_public_invariants() -> None:
    """Una determinizzazione deve rispettare mano nota, punteggi pubblici, deck size e unicità carte."""
    state = _play_with_heuristics_until(seed=11, max_deck_size=6)
    observation = make_player_observation(state, state.current_turn)

    determinized = determinize_observation(observation, rng=random.Random(123))

    assert determinized.current_turn == observation.current_turn
    assert determinized.players[observation.player_index].hand == observation.hand
    assert (
        len(determinized.players[1 - observation.player_index].hand)
        == observation.players_hand_sizes[1 - observation.player_index]
    )
    assert len(determinized.deck) == observation.deck_size
    assert determinized.players[0].points == observation.players_points[0]
    assert determinized.players[1].points == observation.players_points[1]

    all_ids = _state_card_ids(determinized)
    assert len(all_ids) == 40
    assert len(set(all_ids)) == 40


def test_pimc_delegates_to_fallback_when_too_many_unknown_cards() -> None:
    """Fuori soglia PIMC deve comportarsi come il fallback."""
    state = new_game_state(num_players=2, seed=7)
    observation = make_player_observation(state, state.current_turn)
    fallback = HeuristicAgentV2()
    agent = PIMCAgent(
        rollout_agent=HeuristicAgentV2(),
        fallback=fallback,
        num_determinizations=2,
        max_unknown_cards=0,
    )

    assert unknown_live_card_count(observation) > 0
    assert agent.choose_card_index(observation, rng=random.Random(5)) == fallback.choose_card_index(
        observation,
        rng=random.Random(5),
    )
    assert agent.metrics.total_decisions == 1
    assert agent.metrics.search_decisions == 0
    assert agent.metrics.fallback_decisions == 1


def test_pimc_returns_valid_card_when_search_is_active() -> None:
    """Nel semi-finale il prototipo deve riuscire a campionare e scegliere una carta valida."""
    state = _play_with_heuristics_until(seed=19, max_deck_size=4)
    observation = make_player_observation(state, state.current_turn)
    agent = PIMCAgent(
        rollout_agent=HeuristicAgentV2(),
        fallback=HeuristicAgentV2(),
        num_determinizations=3,
        max_unknown_cards=10,
    )

    choice = agent.choose_card_index(observation, rng=random.Random(9))

    assert 0 <= choice < len(observation.hand)
    assert agent.metrics.search_decisions == 1
    assert agent.metrics.successful_determinizations > 0
    assert agent.metrics.completed_rollouts > 0
    assert agent.metrics.seconds_per_search_decision > 0.0


def test_pimc_uses_exact_solver_when_deck_is_empty() -> None:
    """A mazzo vuoto PIMC deve coincidere col solver endgame ricostruito dall'osservazione."""
    state = _play_with_heuristics_until_deck_empty(seed=23)
    observation = make_player_observation(state, state.current_turn)
    expected = solve_endgame(reconstruct_endgame_state(observation)).best_card_index
    agent = PIMCAgent(
        rollout_agent=HeuristicAgentV2(),
        fallback=HeuristicAgentV1(),
        num_determinizations=1,
        max_unknown_cards=10,
        use_endgame_solver=True,
    )

    assert agent.choose_card_index(observation, rng=random.Random(4)) == expected
    assert agent.metrics.endgame_solver_decisions == 1


def test_rollout_to_terminal_handles_invalid_rollout_index_defensively() -> None:
    """Un rollout agent difettoso non deve abortire: l'indice viene normalizzato a una mossa valida."""
    state = _play_with_heuristics_until(seed=31, max_deck_size=4)
    metrics = PIMCSearchStats()

    final_state = rollout_to_terminal(
        state,
        rollout_agent=_InvalidIndexAgent(),
        rng=random.Random(7),
        use_endgame_solver=False,
        metrics=metrics,
    )

    assert final_state.game_over is True
    assert final_state.players[0].points + final_state.players[1].points == 120
    assert metrics.coerced_moves > 0
