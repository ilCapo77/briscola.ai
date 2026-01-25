"""
Test per encoder ML (Behavior Cloning): 40 carte + action mask.

Obiettivo:
- garantire che la mappatura carta -> action_id sia stabile e bijettiva
- garantire che l'encoder produca dimensioni e mask coerenti
"""

from __future__ import annotations

from briscola_ai.ai.training.card_action_space import (
    action_id_from_suit_number,
    action_mask_from_hand,
    suit_number_from_action_id,
)
from briscola_ai.ai.training.observation_encoder import encode_observation_2p


def test_action_id_mapping_is_bijective() -> None:
    """(suit, number) <-> action_id deve essere invertibile per tutte le 40 carte."""
    seen = set()
    for suit in ("clubs", "cups", "coins", "swords"):
        for number in range(1, 11):
            action_id = action_id_from_suit_number(suit=suit, number=number)
            assert 0 <= action_id < 40
            assert action_id not in seen
            seen.add(action_id)

            suit2, number2 = suit_number_from_action_id(action_id)
            assert suit2 == suit
            assert number2 == number

    assert len(seen) == 40


def test_action_mask_from_hand_marks_only_cards_in_hand() -> None:
    """La mask deve attivare esattamente le carte presenti in mano."""
    hand = [
        {"suit": "cups", "number": 1, "rank": "ACE", "points": 11},
        {"suit": "coins", "number": 3, "rank": "THREE", "points": 10},
    ]
    mask = action_mask_from_hand(hand)
    assert len(mask) == 40
    assert sum(mask) == 2
    assert mask[action_id_from_suit_number(suit="cups", number=1)] is True
    assert mask[action_id_from_suit_number(suit="coins", number=3)] is True


def test_encode_observation_2p_shapes_and_mask() -> None:
    """Encoder: feature dimension fissa e mask coerente con my_hand."""
    obs = {
        "type": "observation",
        "server_version": 0,
        "my_index": 0,
        "my_hand": [
            {"suit": "cups", "rank": "ACE", "number": 1, "points": 11},
            {"suit": "clubs", "rank": "TWO", "number": 2, "points": 0},
        ],
        "my_points": 0,
        "my_turn": True,
        "trump_card": {"suit": "coins", "rank": "KING", "number": 10, "points": 4},
        "trump_suit": "coins",
        "table_cards": [{"card": {"suit": "swords", "rank": "THREE", "number": 3, "points": 10}, "player_index": 1}],
        "cards_remaining_in_deck": 20,
        "valid_actions": [0, 1],
        "game_over": False,
        "num_players": 2,
        "is_team_game": False,
        "players": [
            {"index": 0, "name": "A", "points": 0, "hand_size": 2},
            {"index": 1, "name": "B", "points": 0, "hand_size": 2},
        ],
    }

    encoded = encode_observation_2p(obs)
    assert len(encoded.action_mask) == 40
    assert sum(encoded.action_mask) == 2
    assert len(encoded.features) == (40 * 6 + 4 + 4)
