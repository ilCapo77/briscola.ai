"""
Test di regole della mano (trick) per la Briscola.

Qui testiamo solo la funzione di determinazione del vincitore della mano, isolandola
dal resto del flusso (pesca, fine partita, ecc.).
"""

from typing import Optional

import pytest

from briscola_ai.game.game import BriscolaGame
from briscola_ai.game.models import Card, Rank, Suit


def _g(trump: Optional[Card] = None) -> BriscolaGame:
    """Helper: crea un game 2p con briscola impostabile manualmente (senza start_game)."""
    game = BriscolaGame(num_players=2, player_names=["A", "B"])
    game.trump_card = trump
    return game


@pytest.mark.parametrize(
    "higher,lower",
    [
        (Rank.ACE, Rank.THREE),
        (Rank.THREE, Rank.KING),
        (Rank.KING, Rank.KNIGHT),
        (Rank.KNIGHT, Rank.JACK),
        (Rank.JACK, Rank.SEVEN),
        (Rank.SEVEN, Rank.SIX),
        (Rank.SIX, Rank.FIVE),
        (Rank.FIVE, Rank.FOUR),
        (Rank.FOUR, Rank.TWO),
    ],
)
def test_trick_rank_order_within_same_suit(higher: Rank, lower: Rank) -> None:
    """A parità di seme, vince sempre il rank più alto secondo l'ordine Briscola."""
    game = _g(trump=Card(Suit.CUPS, Rank.TWO))
    cards = [
        (Card(Suit.CLUBS, lower), 0),
        (Card(Suit.CLUBS, higher), 1),
    ]
    assert game.who_wins_trick(cards) == 1


def test_trump_beats_non_trump_even_if_low() -> None:
    """Una briscola, anche bassa, batte una carta non-briscola di altro seme."""
    game = _g(trump=Card(Suit.CUPS, Rank.TWO))
    cards = [
        (Card(Suit.SWORDS, Rank.ACE), 0),  # seme di uscita
        (Card(Suit.CUPS, Rank.TWO), 1),  # briscola minima
    ]
    assert game.who_wins_trick(cards) == 1


def test_leading_suit_wins_if_no_trump_played() -> None:
    """Se nessuno gioca briscola, vale il seme di uscita (leading suit)."""
    game = _g(trump=Card(Suit.CUPS, Rank.TWO))
    cards = [
        (Card(Suit.SWORDS, Rank.FOUR), 0),
        (Card(Suit.COINS, Rank.ACE), 1),  # seme diverso, non briscola
    ]
    assert game.who_wins_trick(cards) == 0
