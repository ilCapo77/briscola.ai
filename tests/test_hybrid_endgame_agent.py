"""
Test dell'agente ibrido endgame (Fase 5G, step 2).

Focus:
- ricostruire la mano avversaria solo da `PlayerObservation`;
- azzerare punti/prese nello stato ricostruito senza cambiare la mossa ottima;
- usare il solver solo quando l'osservazione endgame è coerente, altrimenti fallback.
"""

from __future__ import annotations

import random
from dataclasses import replace

from briscola_ai.ai.agents import (
    HybridEndgameAgent,
    build_agent,
    can_solve_endgame_from_observation,
    list_agent_specs,
    reconstruct_endgame_state,
)
from briscola_ai.ai.endgame_solver import solve_endgame
from briscola_ai.domain.card_id import card_to_id
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.observation import PlayerObservation, make_player_observation
from briscola_ai.domain.rules import trick_points
from briscola_ai.domain.state import GameState, PlayerState, new_game_state

TRUMP_CARD = Card(Suit.COINS, Rank.SEVEN)


class FixedFallbackAgent:
    """Fallback minimale per verificare se l'agente ibrido delega davvero."""

    name = "fixed_fallback"

    def __init__(self, card_index: int) -> None:
        self.card_index = card_index
        self.calls = 0

    def choose_card_index(self, observation: PlayerObservation, *, rng: random.Random) -> int:
        self.calls += 1
        return self.card_index


def _all_cards() -> tuple[Card, ...]:
    """Mazzo completo nell'ordine canonico del dominio."""
    return tuple(Card(suit, rank) for suit in Suit for rank in Rank)


def _partitioned_endgame_state(
    *,
    hand0: tuple[Card, ...],
    hand1: tuple[Card, ...],
    current_turn: int,
    table_cards: tuple[tuple[Card, int], ...] = (),
    trump_card: Card = TRUMP_CARD,
) -> GameState:
    """
    Costruisce uno stato endgame coerente su 40 carte.

    Tutte le carte non residue vengono messe nelle prese. La partizione delle prese non deve
    essere realistica trick-per-trick: per questi test basta che `points == trick_points(captured)`
    e che l'osservazione pubblica possa dedurre le mani residue.
    """
    visible_cards = tuple(hand0) + tuple(hand1) + tuple(card for card, _player_idx in table_cards)
    visible_ids = [card_to_id(card) for card in visible_cards]
    assert len(set(visible_ids)) == len(visible_ids)

    captured_pool = tuple(card for card in _all_cards() if card_to_id(card) not in set(visible_ids))
    # Distribuzione volutamente sbilanciata: crea basi punti alte e diverse, utile per testare
    # che la ricostruzione azzerata preservi la scelta ottima ma non il delta assoluto.
    captured0 = tuple(card for card in captured_pool if card.rank.points > 0)
    captured1 = tuple(card for card in captured_pool if card.rank.points == 0)

    first_player = table_cards[0][1] if table_cards else current_turn
    return GameState(
        num_players=2,
        is_team_game=False,
        teams=None,
        players=(
            PlayerState("P0", hand0, captured0, trick_points(captured0)),
            PlayerState("P1", hand1, captured1, trick_points(captured1)),
        ),
        deck=tuple(),
        trump_card=trump_card,
        table_cards=table_cards,
        current_turn=current_turn,
        first_player=first_player,
        game_over=False,
        winner_index=None,
        winning_team=None,
    )


def _tempo_state() -> GameState:
    """
    Stato con best move nota: P0 deve giocare l'Asso di coppe (indice 1), non la briscola.

    È lo scenario "il tempo conta" del solver, ma con tutte le altre carte nelle prese per rendere
    l'osservazione endgame pienamente ricostruibile.
    """
    return _partitioned_endgame_state(
        hand0=(Card(Suit.COINS, Rank.KNIGHT), Card(Suit.CUPS, Rank.ACE)),
        hand1=(Card(Suit.COINS, Rank.KING), Card(Suit.CLUBS, Rank.TWO)),
        current_turn=0,
    )


def test_reconstruction_zeroes_points_but_preserves_best_move() -> None:
    """
    I punti di base vanno azzerati nello stato ricostruito.

    `domain.step` ricalcola i punti dal contenuto di `captured_cards`; copiare solo `points`
    corromperebbe il delta. La mossa ottima però resta uguale perché la base punti è una costante.
    """
    state = _tempo_state()
    observation = make_player_observation(state, player_index=0)

    reconstructed = reconstruct_endgame_state(observation)
    real_solution = solve_endgame(state)
    reconstructed_solution = solve_endgame(reconstructed)

    base_delta = state.players[0].points - state.players[1].points
    assert base_delta != 0
    assert tuple(p.points for p in reconstructed.players) == (0, 0)
    assert tuple(p.captured_cards for p in reconstructed.players) == (tuple(), tuple())

    assert real_solution.best_card_index == 1
    assert reconstructed_solution.best_card_index == real_solution.best_card_index
    assert real_solution.final_delta_p0_p1 == base_delta + reconstructed_solution.final_delta_p0_p1


def test_reconstruction_adds_trump_when_trump_is_in_opponent_hand() -> None:
    """
    La briscola scoperta è sempre marcata come `seen`: se è nella mano avversaria va riaggiunta.
    """
    state = _partitioned_endgame_state(
        hand0=(Card(Suit.CLUBS, Rank.TWO), Card(Suit.CUPS, Rank.FOUR), Card(Suit.SWORDS, Rank.FIVE)),
        hand1=(TRUMP_CARD, Card(Suit.CLUBS, Rank.ACE), Card(Suit.SWORDS, Rank.TWO)),
        current_turn=0,
    )
    observation = make_player_observation(state, player_index=0)

    reconstructed = reconstruct_endgame_state(observation)

    assert can_solve_endgame_from_observation(observation) is True
    assert TRUMP_CARD in reconstructed.players[1].hand
    assert set(reconstructed.players[1].hand) == set(state.players[1].hand)


def test_reconstruction_preserves_second_to_play_table_state() -> None:
    """Lo stato ricostruito supporta anche il caso secondo di mano (`len(table_cards) == 1`)."""
    state = _partitioned_endgame_state(
        hand0=(Card(Suit.SWORDS, Rank.FOUR),),
        hand1=(Card(Suit.CLUBS, Rank.ACE), Card(Suit.SWORDS, Rank.FIVE)),
        current_turn=1,
        table_cards=((Card(Suit.CLUBS, Rank.TWO), 0),),
    )
    observation = make_player_observation(state, player_index=1)

    reconstructed = reconstruct_endgame_state(observation)
    solution = solve_endgame(reconstructed)

    assert reconstructed.table_cards == state.table_cards
    assert reconstructed.current_turn == 1
    assert reconstructed.players[1].hand == state.players[1].hand
    assert solution.best_card_index == 0


def test_hybrid_agent_uses_solver_in_endgame_without_calling_fallback() -> None:
    """A mazzo vuoto e osservazione coerente, l'agente sceglie la mossa del solver."""
    state = _tempo_state()
    observation = make_player_observation(state, player_index=0)
    fallback = FixedFallbackAgent(card_index=0)
    agent = HybridEndgameAgent(fallback=fallback)

    chosen = agent.choose_card_index(observation, rng=random.Random(7))

    assert chosen == 1
    assert fallback.calls == 0


def test_hybrid_agent_falls_back_before_endgame() -> None:
    """Con mazzo non vuoto il solver non è nello scope: deve scegliere il fallback."""
    state = new_game_state(2, seed=42)
    observation = make_player_observation(state, player_index=0)
    fallback = FixedFallbackAgent(card_index=2)
    agent = HybridEndgameAgent(fallback=fallback)

    chosen = agent.choose_card_index(observation, rng=random.Random(7))

    assert chosen == 2
    assert fallback.calls == 1


def test_hybrid_agent_falls_back_on_incoherent_observation() -> None:
    """Osservazioni vecchie/malformate non devono produrre stati inventati."""
    state = _tempo_state()
    observation = make_player_observation(state, player_index=0)
    malformed = replace(observation, seen_cards_onehot=(0,) * 40)
    fallback = FixedFallbackAgent(card_index=0)
    agent = HybridEndgameAgent(fallback=fallback)

    assert can_solve_endgame_from_observation(malformed) is False
    chosen = agent.choose_card_index(malformed, rng=random.Random(7))

    assert chosen == 0
    assert fallback.calls == 1


def test_reconstruction_matches_real_solver_after_real_game_reaches_endgame() -> None:
    """Su una partita prodotta dal dominio, la mossa ricostruita coincide con il solver reale."""
    state = new_game_state(2, seed=123)
    while len(state.deck) > 0 and not state.game_over:
        state, _ = step(state, PlayCardAction(player_index=state.current_turn, card_index=0))

    observation = make_player_observation(state, player_index=state.current_turn)
    real_solution = solve_endgame(state)
    reconstructed_solution = solve_endgame(reconstruct_endgame_state(observation))

    assert reconstructed_solution.best_card_index == real_solution.best_card_index
    base_delta = state.players[0].points - state.players[1].points
    assert real_solution.final_delta_p0_p1 == base_delta + reconstructed_solution.final_delta_p0_p1


def test_hybrid_endgame_is_registered_in_agent_catalog() -> None:
    """Il nuovo agente deve essere costruibile e visibile nel catalogo server-side."""
    assert "hybrid_endgame" in {spec.name for spec in list_agent_specs()}
    assert isinstance(build_agent("hybrid_endgame"), HybridEndgameAgent)
