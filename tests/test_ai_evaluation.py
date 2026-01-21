"""
Test per la valutazione offline degli agenti.

Scopo:
- garantire riproducibilità: stessa seed → stessi risultati aggregati
- garantire coerenza: numero partite = wins + draws, punti medi in range plausibile

Non testiamo che un agente sia "forte": testiamo l'infrastruttura.
"""

from __future__ import annotations

from briscola_ai.ai.agents import RandomAgent
from briscola_ai.ai.evaluation import evaluate_match_2p


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
