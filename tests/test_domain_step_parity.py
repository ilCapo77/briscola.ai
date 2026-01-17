"""
Test di parità: motore legacy (`BriscolaGame`) vs motore "puro" (Phase 2B).

Obiettivo didattico:
- introdurre `GameState + step()` in parallelo al motore esistente
- garantire che, a parità di seed e azioni, l'evoluzione dello stato sia identica

Questi test sono la rete di sicurezza per migrare il backend al nuovo motore.
"""

from __future__ import annotations

import random

from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.state import GameState, new_game_state
from briscola_ai.game.game import BriscolaGame


def _assert_state_parity(game: BriscolaGame, state: GameState) -> None:
    """Verifica che gli aspetti rilevanti di stato siano equivalenti."""
    assert game.num_players == state.num_players
    assert game.is_team_game == state.is_team_game

    assert game.current_turn == state.current_turn
    assert game.first_player == state.first_player
    assert game.game_over == state.game_over

    assert game.trump_card == state.trump_card
    assert list(game.deck) == list(state.deck)
    assert list(game.table_cards) == list(state.table_cards)

    for i in range(game.num_players):
        assert game.players[i].name == state.players[i].name
        assert list(game.players[i].hand) == list(state.players[i].hand)
        assert list(game.players[i].captured_cards) == list(state.players[i].captured_cards)
        assert game.players[i].points == state.players[i].points


def test_step_parity_2p_random_game() -> None:
    """
    Parità 2-player: partenza con seed noto e sequenza di azioni random deterministica.

    Nota:
    - `BriscolaGame` usa il RNG globale (`random.shuffle`), quindi settiamo `random.seed`
      prima di `start_game()` per allinearci a `new_game_state(seed=...)`.
    """
    seed = 42
    names = ["A", "B"]

    random.seed(seed)
    game = BriscolaGame(num_players=2, player_names=names)
    game.start_game()

    state = new_game_state(2, names, seed=seed)
    _assert_state_parity(game, state)

    chooser = random.Random(123)
    safety = 500
    while not game.game_over and safety > 0:
        safety -= 1

        current = game.current_turn
        hand_size = len(game.players[current].hand)
        assert hand_size > 0
        card_index = chooser.randrange(hand_size)

        legacy_result = game.play_action(card_index)
        state, pure_result = step(state, PlayCardAction(player_index=current, card_index=card_index))

        assert "error" not in legacy_result
        assert pure_result.error is None

        assert pure_result.played_card == legacy_result["played_card"]
        assert pure_result.trick_completed == legacy_result["trick_completed"]
        if legacy_result["trick_completed"]:
            assert pure_result.trick_winner == legacy_result["trick_winner"]

        _assert_state_parity(game, state)

    assert safety > 0
    assert game.game_over is True
    assert state.game_over is True


def test_step_parity_4p_random_game() -> None:
    """
    Parità 4-player: come il 2-player, ma verifica anche il flusso a squadre.
    """
    seed = 7
    names = ["A", "B", "C", "D"]

    random.seed(seed)
    game = BriscolaGame(num_players=4, player_names=names)
    game.start_game()

    state = new_game_state(4, names, seed=seed)
    _assert_state_parity(game, state)

    chooser = random.Random(999)
    safety = 2000
    while not game.game_over and safety > 0:
        safety -= 1
        current = game.current_turn
        hand_size = len(game.players[current].hand)
        assert hand_size > 0
        card_index = chooser.randrange(hand_size)

        legacy_result = game.play_action(card_index)
        state, pure_result = step(state, PlayCardAction(player_index=current, card_index=card_index))

        assert "error" not in legacy_result
        assert pure_result.error is None

        assert pure_result.played_card == legacy_result["played_card"]
        assert pure_result.trick_completed == legacy_result["trick_completed"]
        if legacy_result["trick_completed"]:
            assert pure_result.trick_winner == legacy_result["trick_winner"]

        _assert_state_parity(game, state)

    assert safety > 0
    assert game.game_over is True
    assert state.game_over is True
