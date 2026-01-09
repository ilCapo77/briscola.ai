"""
Test di base per `BriscolaGame`.

Questi test verificano invarianti "semplici" ma fondamentali:
- il mazzo iniziale ha 40 carte uniche
- il setup (distribuzione e briscola) è coerente per 2 e 4 giocatori
- una partita 2p completa termina e totalizza 120 punti complessivi
"""

import random

from briscola_ai.game.game import BriscolaGame


def test_deck_has_40_unique_cards_before_start() -> None:
    """Verifica l'invariante del mazzo: 40 carte e tutte distinte, prima dell'avvio."""
    game = BriscolaGame(num_players=2)
    assert len(game.deck) == 40
    assert len(set(game.deck)) == 40


def test_start_game_deals_correctly_for_2_players() -> None:
    """In 2 giocatori: 3 carte in mano ciascuno, briscola impostata e deck coerente."""
    random.seed(123)
    game = BriscolaGame(num_players=2, player_names=["A", "B"])
    game.start_game()

    assert game.is_team_game is False
    assert game.trump_card is not None
    assert len(game.deck) == 34  # 40 - 6 dealt; trump reinserted on bottom (index 0)
    assert game.deck[0] == game.trump_card
    assert len(game.players) == 2
    assert [len(p.hand) for p in game.players] == [3, 3]


def test_start_game_deals_correctly_for_4_players() -> None:
    """In 4 giocatori: partita a squadre e 10 carte in mano a ciascun giocatore."""
    random.seed(456)
    game = BriscolaGame(num_players=4)
    game.start_game()

    assert game.is_team_game is True
    assert game.trump_card is not None
    assert len(game.deck) == 0
    assert [len(p.hand) for p in game.players] == [10, 10, 10, 10]


def test_complete_random_2p_game_ends_and_points_sum_to_120() -> None:
    """Esegue azioni valide fino al game over e verifica il totale punti (120)."""
    random.seed(999)
    game = BriscolaGame(num_players=2, player_names=["A", "B"])
    game.start_game()

    safety = 1000
    while not game.game_over and safety > 0:
        safety -= 1
        valid = game.get_valid_actions()
        assert valid, "Se la partita non è finita deve esserci almeno un'azione valida"
        game.play_action(valid[0])

    assert safety > 0, "Loop di sicurezza: la partita dovrebbe terminare"
    assert game.game_over is True
    assert sum(p.points for p in game.players) == 120
