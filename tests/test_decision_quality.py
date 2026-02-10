"""
Test per metriche di qualità decisionale.

Obiettivo:
- verificare che `trump_waste` venga rilevato correttamente in un caso semplice.
"""

from __future__ import annotations

from briscola_ai.ai.decision_quality import _is_trump_waste_second_hand
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.state import GameState, PlayerState


def test_trump_waste_detected_when_non_trump_wins() -> None:
    """
    Caso:
    - briscola = cups
    - avversario gioca clubs TWO (scarto)
    - io (player 1) ho in mano clubs ACE (vince senza briscola) e cups ACE (briscola)
    - se gioco cups ACE -> spreco briscola (waste=True)
    """
    state = GameState(
        num_players=2,
        is_team_game=False,
        teams=None,
        players=(
            PlayerState(name="P0", hand=tuple(), captured_cards=tuple(), points=0),
            PlayerState(
                name="P1",
                hand=(Card(Suit.CLUBS, Rank.ACE), Card(Suit.CUPS, Rank.ACE)),
                captured_cards=tuple(),
                points=0,
            ),
        ),
        deck=tuple(),
        trump_card=Card(Suit.CUPS, Rank.THREE),
        table_cards=((Card(Suit.CLUBS, Rank.TWO), 0),),
        current_turn=1,
        first_player=0,
        game_over=False,
        winner_index=None,
        winning_team=None,
    )

    waste = _is_trump_waste_second_hand(state=state, player_index=1, chosen_card_index=1)
    assert waste is True

    no_waste = _is_trump_waste_second_hand(state=state, player_index=1, chosen_card_index=0)
    assert no_waste is False
