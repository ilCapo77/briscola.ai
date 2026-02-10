"""
Test per `ai.training.reward_shaping` (didattico).

Scopo:
- garantire che le penalità siano basate solo su info lecita (PlayerObservation)
- verificare la logica "overkill briscola" in un caso minimale e riproducibile
"""

from __future__ import annotations

from briscola_ai.ai.training.reward_shaping import analyze_trump_overkill_second_hand, trump_overkill_penalty
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.observation import PlayerObservation


def _obs_second_hand(*, hand: tuple[Card, ...], lead: Card, trump: Card) -> PlayerObservation:
    """Costruisce una `PlayerObservation` minimale in cui siamo secondi di mano."""
    return PlayerObservation(
        num_players=2,
        is_team_game=False,
        teams=None,
        player_index=1,
        player_name="P1",
        hand=hand,
        trump_card=trump,
        deck_size=10,
        table_cards=((lead, 0),),
        current_turn=1,
        first_player=0,
        game_over=False,
        winner_index=None,
        winning_team=None,
        players_points=(0, 0),
        players_hand_sizes=(3, len(hand)),
        seen_cards_onehot=tuple([0] * 40),
    )


def test_analyze_trump_overkill_detects_cheaper_winning_trump() -> None:
    """
    Se ho due briscole vincenti, scegliere quella più "costosa" è overkill.
    """
    obs = _obs_second_hand(
        hand=(Card(Suit.CUPS, Rank.TWO), Card(Suit.CUPS, Rank.ACE)),
        lead=Card(Suit.SWORDS, Rank.TWO),
        trump=Card(Suit.CUPS, Rank.THREE),
    )

    info = analyze_trump_overkill_second_hand(obs, chosen_card_index=1, low_lead_points_max=2)
    assert info.applicable is True
    assert info.chosen_is_trump is True
    assert info.chosen_wins is True
    assert info.winning_trump_exists is True
    assert info.is_overkill is True

    info2 = analyze_trump_overkill_second_hand(obs, chosen_card_index=0, low_lead_points_max=2)
    assert info2.applicable is True
    assert info2.is_overkill is False


def test_trump_overkill_penalty_is_flat_negative_when_overkill() -> None:
    obs = _obs_second_hand(
        hand=(Card(Suit.CUPS, Rank.TWO), Card(Suit.CUPS, Rank.ACE)),
        lead=Card(Suit.SWORDS, Rank.TWO),
        trump=Card(Suit.CUPS, Rank.THREE),
    )

    p = trump_overkill_penalty(obs, chosen_card_index=1, beta=0.005, low_lead_points_max=2)
    assert p == -0.005

    p2 = trump_overkill_penalty(obs, chosen_card_index=0, beta=0.005, low_lead_points_max=2)
    assert p2 == 0.0

    p3 = trump_overkill_penalty(obs, chosen_card_index=1, beta=0.0, low_lead_points_max=2)
    assert p3 == 0.0
