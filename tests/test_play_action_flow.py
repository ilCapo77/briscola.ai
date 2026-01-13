"""
Test di flusso per `BriscolaGame.play_action`.

Obiettivo:
- validare la transizione di stato quando una mano si completa
- verificare la logica di pesca post-mano in modalità 2 giocatori
- assicurare che azioni invalide non corrompano lo stato
"""

from briscola_ai.game.game import BriscolaGame
from briscola_ai.game.models import Card, Rank, Suit


def test_trick_completion_updates_turn_and_points() -> None:
    """
    Esegue una mano completa in modalità 2 giocatori con mani impostate a mano.

    Verifica:
    - `table_cards` si svuota dopo la mano
    - `current_turn` e `first_player` diventano il vincitore della mano
    - le carte vengono aggiunte a `captured_cards` del vincitore e i punti aggiornati
    """
    game = BriscolaGame(num_players=2, player_names=["A", "B"])
    game.trump_card = Card(Suit.CUPS, Rank.TWO)
    game.deck = []
    game.table_cards.clear()
    game.game_over = False
    game.current_turn = 0
    game.first_player = 0

    # A gioca 4 di spade (seme di uscita), B gioca asso di spade: B deve vincere
    game.players[0].hand = [Card(Suit.SWORDS, Rank.FOUR)]
    game.players[1].hand = [Card(Suit.SWORDS, Rank.ACE)]

    r1 = game.play_action(0)
    assert r1["trick_completed"] is False
    assert len(game.table_cards) == 1
    assert game.current_turn == 1

    r2 = game.play_action(0)
    assert r2["trick_completed"] is True
    assert r2["trick_winner"] == 1
    assert game.table_cards == []
    assert game.current_turn == 1
    assert game.first_player == 1

    assert len(game.players[1].captured_cards) == 2
    assert game.players[1].points == Rank.ACE.points + Rank.FOUR.points


def test_after_trick_in_2p_cards_are_dealt_starting_from_trick_winner() -> None:
    """
    In 2 giocatori, dopo una mano completa si pescano due carte (una per giocatore)
    partendo dal vincitore della mano.
    """
    game = BriscolaGame(num_players=2, player_names=["A", "B"])
    game.trump_card = Card(Suit.CUPS, Rank.TWO)
    game.table_cards.clear()
    game.game_over = False
    game.current_turn = 0
    game.first_player = 0

    # Deck: `pop()` pesca dalla fine. Prepariamo due carte note.
    drawn_by_winner = Card(Suit.COINS, Rank.KING)
    drawn_by_other = Card(Suit.SWORDS, Rank.TWO)
    game.deck = [game.trump_card, drawn_by_other, drawn_by_winner]

    # A perde: A gioca 4 spade, B gioca 7 coppe (briscola) => B vince.
    game.players[0].hand = [Card(Suit.SWORDS, Rank.FOUR)]
    game.players[1].hand = [Card(Suit.CUPS, Rank.SEVEN)]

    game.play_action(0)
    result = game.play_action(0)

    assert result["trick_completed"] is True
    assert result["trick_winner"] == 1
    assert result["cards_dealt"] is True

    # Dopo la mano: B pesca per primo (winner), quindi prende `drawn_by_winner`.
    assert game.players[1].hand == [drawn_by_winner]
    assert game.players[0].hand == [drawn_by_other]
    assert game.deck == [game.trump_card]


def test_invalid_action_does_not_modify_state() -> None:
    """Un'azione invalida (indice carta fuori range) deve restituire errore senza side-effect."""
    game = BriscolaGame(num_players=2, player_names=["A", "B"])
    game.trump_card = Card(Suit.CUPS, Rank.TWO)
    game.deck = []
    game.table_cards.clear()
    game.current_turn = 0
    game.players[0].hand = [Card(Suit.SWORDS, Rank.FOUR)]

    before = {
        "hand": list(game.players[0].hand),
        "table": list(game.table_cards),
        "turn": game.current_turn,
    }

    out = game.play_action(999)
    assert "error" in out

    assert game.players[0].hand == before["hand"]
    assert game.table_cards == before["table"]
    assert game.current_turn == before["turn"]
