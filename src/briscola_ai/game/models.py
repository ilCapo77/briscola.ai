"""
Modelli legacy per la Briscola.

Questo modulo è mantenuto principalmente per retro-compatibilità.

Nel refactor Phase 2B+ i modelli canonici del gioco (carte e semi) sono stati
spostati in `briscola_ai.domain.models` per rendere il dominio autosufficiente.
Qui re-esportiamo:
- `Suit`, `Rank`, `Card`

In questo modulo resta `Player`, che è un oggetto *stateful* usato dal motore
legacy `BriscolaGame` (in `game/game.py`).

Nota didattica:
in Briscola esistono due “ordini” diversi che spesso si confondono:
- `points`: i punti che una carta vale quando viene raccolta (Asso=11, Tre=10, Re=4, Cavallo=3, Fante=2)
- `trick_strength`: quanto una carta è “forte” per vincere una mano (Asso > Tre > Re > ...)
"""

from typing import List

from ..domain.models import Card, Rank, Suit

# Re-export esplicito: mantiene compatibilità per chi importa da `briscola_ai.game.models`.
# Il codice nuovo dovrebbe importare questi simboli da `briscola_ai.domain.models`.
#
# Nota tecnica:
# `Rank` e `Suit` non sono referenziati direttamente in questo modulo (il `Player` usa solo `Card`),
# ma vogliamo comunque re-esportarli per non rompere gli import legacy. Usiamo una tupla “ancora”
# per evitare che i tool di lint li considerino unused.
_KEEP_LEGACY_EXPORTS = (Rank, Suit)


class Player:
    """
    Rappresenta un giocatore.

    Campi principali:
    - `hand`: carte in mano (giocabili)
    - `captured_cards`: carte vinte nelle prese
    - `points`: punteggio totale (derivato da `captured_cards`)
    """

    def __init__(self, name: str):
        """
        Crea un giocatore.

        Argomenti:
            name: nome visualizzato (UI/log)
        """
        self.name = name
        self.hand: List[Card] = []
        self.captured_cards: List[Card] = []
        self.points = 0

    def add_card(self, card: Card) -> None:
        """Aggiunge una carta alla mano del giocatore"""
        self.hand.append(card)

    def play_card(self, index: int) -> Card:
        """Gioca una carta dalla mano in base all'indice"""
        if 0 <= index < len(self.hand):
            return self.hand.pop(index)
        raise ValueError(f"Indice carta non valido: {index}")

    def take_cards(self, cards: List[Card]) -> None:
        """Prende le carte catturate in una mano vinta e aggiorna i punti"""
        self.captured_cards.extend(cards)
        self.points = sum(card.rank.points for card in self.captured_cards)

    def reset(self) -> None:
        """Reimposta lo stato del giocatore per una nuova partita"""
        self.hand.clear()
        self.captured_cards.clear()
        self.points = 0
