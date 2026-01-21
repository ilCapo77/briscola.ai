"""
Test per la valutazione offline degli agenti.

Scopo:
- garantire riproducibilità: stessa seed → stessi risultati aggregati
- garantire coerenza: numero partite = wins + draws, punti medi in range plausibile

Non testiamo che un agente sia "forte": testiamo l'infrastruttura.
"""

from __future__ import annotations

import pytest

from briscola_ai.ai.agents import HeuristicAgentV1, RandomAgent
from briscola_ai.ai.evaluation import evaluate_match_2p, evaluate_seat_fair_match_2p


def test_evaluate_match_is_deterministic_for_fixed_seed() -> None:
    """
    Con la stessa seed, l'aggregato deve essere identico.

    Questo è importante perché useremo queste valutazioni come regressioni:
    se cambiamo un agente, vogliamo attribuire le differenze all'agente, non al rumore.
    """
    a0 = RandomAgent()
    a1 = RandomAgent()

    stats1 = evaluate_match_2p(a0, a1, num_games=50, seed=123)
    stats2 = evaluate_match_2p(a0, a1, num_games=50, seed=123)

    assert stats1 == stats2


def test_evaluate_match_counts_are_consistent() -> None:
    """Verifica che i contatori base tornino."""
    a0 = RandomAgent()
    a1 = RandomAgent()

    stats = evaluate_match_2p(a0, a1, num_games=25, seed=0)
    assert stats.wins_agent0 + stats.wins_agent1 + stats.draws == stats.num_games

    # In 2-player il totale punti per partita è 120, quindi la media per giocatore deve stare in [0, 120].
    assert 0.0 <= stats.avg_points_agent0 <= 120.0
    assert 0.0 <= stats.avg_points_agent1 <= 120.0


def test_seat_fair_evaluate_is_deterministic_for_fixed_seed() -> None:
    """Stesso input → stesso aggregato (anche in modalità seat-fair)."""
    a0 = RandomAgent()
    a1 = RandomAgent()

    stats1 = evaluate_seat_fair_match_2p(a0, a1, num_games=100, seed=123)
    stats2 = evaluate_seat_fair_match_2p(a0, a1, num_games=100, seed=123)
    assert stats1 == stats2


def test_evaluate_match_with_explicit_game_seeds_is_stable_for_deterministic_agents() -> None:
    """
    Se forniamo una suite di shuffle esplicita, la parte “game RNG” è fissata.

    Usiamo agenti deterministici (HeuristicAgentV1) così che cambiare `seed`
    (che controlla l'RNG delle scelte agente) non cambi l'esito aggregato.
    """
    a0 = HeuristicAgentV1()
    a1 = HeuristicAgentV1()

    game_seeds = list(range(100))
    stats1 = evaluate_match_2p(a0, a1, num_games=50, seed=1, game_seeds=game_seeds)
    stats2 = evaluate_match_2p(a0, a1, num_games=50, seed=999, game_seeds=game_seeds)

    assert stats1 == stats2


def test_evaluate_raises_if_game_seeds_is_insufficient() -> None:
    """
    Se la suite di seed è più corta del necessario, deve fallire esplicitamente.

    Questo evita regressioni “silenziose” dove un benchmark usa meno partite del previsto.
    """
    a0 = RandomAgent()
    a1 = RandomAgent()

    with pytest.raises(ValueError, match="game_seeds insufficiente"):
        evaluate_match_2p(a0, a1, num_games=10, seed=0, game_seeds=[1, 2, 3])

    # In seat-fair serve una seed per coppia: num_pairs = num_games // 2.
    with pytest.raises(ValueError, match="game_seeds insufficiente"):
        evaluate_seat_fair_match_2p(a0, a1, num_games=10, seed=0, game_seeds=[1, 2])
