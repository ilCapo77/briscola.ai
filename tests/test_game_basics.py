import random

from briscola_ai.game.game import BriscolaGame


def test_deck_has_40_unique_cards_before_start() -> None:
    game = BriscolaGame(num_players=2)
    assert len(game.deck) == 40
    assert len(set(game.deck)) == 40


def test_start_game_deals_correctly_for_2_players() -> None:
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
    random.seed(456)
    game = BriscolaGame(num_players=4)
    game.start_game()

    assert game.is_team_game is True
    assert game.trump_card is not None
    assert len(game.deck) == 0
    assert [len(p.hand) for p in game.players] == [10, 10, 10, 10]


def test_complete_random_2p_game_ends_and_points_sum_to_120() -> None:
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
