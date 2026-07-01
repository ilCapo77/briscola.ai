"""Test del kernel Numba V-lookahead per stati numerici determinizzati."""

from __future__ import annotations

import numpy as np

from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V3
from briscola_ai.ai.endgame.solver import solve_endgame
from briscola_ai.ai.numba.value_lookahead import (
    arrays_from_game_state_for_value_lookahead,
    choose_value_lookahead_card_numba,
    choose_value_lookahead_card_numba_arrays,
    warm_up_numba_value_lookahead,
)
from briscola_ai.domain.card_id import card_to_id
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.observation import make_player_observation
from briscola_ai.domain.rules import trick_points
from briscola_ai.domain.state import GameState, PlayerState


def _state(
    *,
    hand0: tuple[Card, ...],
    hand1: tuple[Card, ...],
    deck: tuple[Card, ...],
    table_cards: tuple[tuple[Card, int], ...],
    current_turn: int,
    trump_card: Card = Card(Suit.CLUBS, Rank.SEVEN),
    points0: int = 0,
    points1: int = 0,
) -> GameState:
    """Costruisce uno stato 2-player minimale per testare il kernel."""
    first_player = table_cards[0][1] if table_cards else current_turn
    return GameState(
        num_players=2,
        is_team_game=False,
        teams=None,
        players=(
            PlayerState("P0", hand0, tuple(), points0),
            PlayerState("P1", hand1, tuple(), points1),
        ),
        deck=deck,
        trump_card=trump_card,
        table_cards=table_cards,
        current_turn=current_turn,
        first_player=first_player,
        game_over=False,
        winner_index=None,
        winning_team=None,
    )


def _endgame_state(
    hand0: tuple[Card, ...],
    hand1: tuple[Card, ...],
    *,
    current_turn: int,
    table_cards: tuple[tuple[Card, int], ...] = (),
    trump_card: Card = Card(Suit.COINS, Rank.SEVEN),
) -> GameState:
    """Costruisce uno stato endgame con punti coerenti."""
    return _state(
        hand0=hand0,
        hand1=hand1,
        deck=tuple(),
        table_cards=table_cards,
        current_turn=current_turn,
        trump_card=trump_card,
    )


def _weights() -> tuple[np.ndarray, ...]:
    """Pesi sintetici: policy/value nulli, dimensioni uguali ai modelli v3 reali."""
    feature_dim = int(FEATURE_DIM_2P_V3)
    policy_w1 = np.zeros((feature_dim, 1), dtype=np.float32)
    policy_b1 = np.zeros(1, dtype=np.float32)
    policy_w2 = np.zeros((1, 40), dtype=np.float32)
    policy_b2 = np.zeros(40, dtype=np.float32)
    value_w1 = np.zeros((feature_dim, 1), dtype=np.float32)
    value_b1 = np.zeros(1, dtype=np.float32)
    value_w2 = np.zeros(1, dtype=np.float32)
    return policy_w1, policy_b1, policy_w2, policy_b2, value_w1, value_b1, value_w2


def _choose_from_state(state: GameState) -> int:
    """Wrapper test: chiama il kernel con value model residuale nullo."""
    policy_w1, policy_b1, policy_w2, policy_b2, value_w1, value_b1, value_w2 = _weights()
    card_index, _score = choose_value_lookahead_card_numba(
        policy_w1=policy_w1,
        policy_b1=policy_b1,
        policy_w2=policy_w2,
        policy_b2=policy_b2,
        value_w1=value_w1,
        value_b1=value_b1,
        value_w2=value_w2,
        value_b2=0.0,
        value_target_scale=120.0,
        value_target_is_residual=True,
        state=state,
        overkill_guard_enabled=True,
    )
    return int(card_index)


def test_warm_up_numba_value_lookahead_compiles_kernel() -> None:
    """Il warm-up deve compilare il kernel prima di loop lunghi di training."""
    warm_up_numba_value_lookahead()


def test_value_lookahead_arrays_from_game_state_matches_public_observation_bits() -> None:
    """La conversione conserva le maschere pubbliche usate dall'encoder v3."""
    state = _state(
        hand0=(Card(Suit.CUPS, Rank.TWO), Card(Suit.CLUBS, Rank.TWO), Card(Suit.SWORDS, Rank.TWO)),
        hand1=(Card(Suit.COINS, Rank.TWO), Card(Suit.COINS, Rank.FOUR)),
        deck=(Card(Suit.SWORDS, Rank.THREE), Card(Suit.COINS, Rank.SEVEN)),
        table_cards=((Card(Suit.CUPS, Rank.ACE), 1),),
        current_turn=0,
        trump_card=Card(Suit.CLUBS, Rank.SEVEN),
        points1=trick_points((Card(Suit.SWORDS, Rank.KING),)),
    )

    *_, seen_cards, out_of_play_cards = arrays_from_game_state_for_value_lookahead(state)
    observation = make_player_observation(state, 0)

    assert seen_cards.tolist() == list(observation.seen_cards_onehot)
    assert out_of_play_cards.tolist() == list(observation.out_of_play_cards_onehot)


def test_value_lookahead_uses_numba_endgame_solver_when_deck_is_empty() -> None:
    """A mazzo vuoto il kernel deve scegliere come il solver canonico."""
    state = _endgame_state(
        hand0=(Card(Suit.COINS, Rank.KNIGHT), Card(Suit.CUPS, Rank.ACE)),
        hand1=(Card(Suit.COINS, Rank.KING), Card(Suit.CLUBS, Rank.TWO)),
        current_turn=0,
    )

    assert _choose_from_state(state) == solve_endgame(state).best_card_index


def test_value_lookahead_depth1_prefers_immediate_point_gain() -> None:
    """Su presa gia' aperta, il value residuale nullo deve preferire la carta che vince punti."""
    trump_winning_card = Card(Suit.CLUBS, Rank.TWO)
    state = _state(
        hand0=(Card(Suit.CUPS, Rank.TWO), trump_winning_card, Card(Suit.SWORDS, Rank.TWO)),
        hand1=(Card(Suit.COINS, Rank.TWO), Card(Suit.COINS, Rank.FOUR)),
        deck=(Card(Suit.SWORDS, Rank.THREE), Card(Suit.COINS, Rank.SEVEN)),
        table_cards=((Card(Suit.CUPS, Rank.ACE), 1),),
        current_turn=0,
        trump_card=Card(Suit.CLUBS, Rank.SEVEN),
    )

    assert state.players[0].hand[_choose_from_state(state)] == trump_winning_card


def test_value_lookahead_array_entrypoint_returns_valid_card_index() -> None:
    """L'entrypoint da array e' quello destinato ai futuri loop training full-JIT."""
    state = _state(
        hand0=(Card(Suit.CUPS, Rank.TWO), Card(Suit.CLUBS, Rank.TWO), Card(Suit.SWORDS, Rank.TWO)),
        hand1=(Card(Suit.COINS, Rank.TWO), Card(Suit.COINS, Rank.FOUR)),
        deck=(Card(Suit.SWORDS, Rank.THREE), Card(Suit.COINS, Rank.SEVEN)),
        table_cards=((Card(Suit.CUPS, Rank.ACE), 1),),
        current_turn=0,
        trump_card=Card(Suit.CLUBS, Rank.SEVEN),
    )
    policy_w1, policy_b1, policy_w2, policy_b2, value_w1, value_b1, value_w2 = _weights()

    card_index, score = choose_value_lookahead_card_numba_arrays(
        policy_w1,
        policy_b1,
        policy_w2,
        policy_b2,
        value_w1,
        value_b1,
        value_w2,
        0.0,
        120.0,
        True,
        *arrays_from_game_state_for_value_lookahead(state),
        True,
    )

    assert 0 <= card_index < len(state.players[state.current_turn].hand)
    assert np.isfinite(score)
    assert state.players[state.current_turn].hand[int(card_index)] == Card(Suit.CLUBS, Rank.TWO)
    assert card_to_id(state.players[state.current_turn].hand[int(card_index)]) == 1
