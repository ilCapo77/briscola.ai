"""
Modelli di dominio per la Briscola.

Questo modulo definisce le entità minime con cui rappresentiamo una partita:
- `Suit`: i semi del mazzo italiano (bastoni, coppe, denari, spade)
- `Rank`: i ranghi/valori delle carte con punteggio e ordine di forza (per vincere una mano)
- `Card`: una carta (seme + rango)
- `Player`: un giocatore con mano, carte prese e punteggio

Nota didattica:
in Briscola esistono due “ordini” diversi che spesso si confondono:
- `points`: i punti che una carta vale quando viene raccolta (Asso=11, Tre=10, Re=4, Cavallo=3, Fante=2)
- `trick_strength`: quanto una carta è “forte” per vincere una mano (Asso > Tre > Re > ...)
"""

from dataclasses import dataclass
from enum import Enum
from typing import List


class Suit(Enum):
    """
    Semi delle carte italiane.

    Internamente usiamo stringhe in inglese (`clubs`, `cups`, ...) perché sono comode
    lato frontend/API; i commenti indicano il nome italiano del seme.
    """

    CLUBS = "clubs"  # bastoni
    CUPS = "cups"  # coppe
    COINS = "coins"  # denari
    SWORDS = "swords"  # spade


class Rank(Enum):
    """
    Ranghi delle carte con:
    - `number`: numero “stampato” (1..10)
    - `points`: punti della Briscola

    Importante: l'ordine di forza in una mano non è l'ordine numerico.
    Per determinare il vincitore di una mano usare `trick_strength`.
    """

    ACE = (1, 11)  # (number, points)
    TWO = (2, 0)
    THREE = (3, 10)
    FOUR = (4, 0)
    FIVE = (5, 0)
    SIX = (6, 0)
    SEVEN = (7, 0)
    JACK = (8, 2)  # fante
    KNIGHT = (9, 3)  # cavallo
    KING = (10, 4)  # re

    def __init__(self, number: int, points: int):
        """
        Inizializza i campi associati al membro Enum.

        Argomenti:
            number: valore numerico (1..10)
            points: punti assegnati quando la carta viene raccolta
        """
        self.number = number
        self.points = points

    @property
    def trick_strength(self) -> int:
        """
        Forza relativa della carta per determinare il vincitore di una mano.

        Nota: in Briscola l'ordine di forza NON coincide con il valore numerico della carta.
        Ordine (dal più forte al più debole): Asso, Tre, Re, Cavallo, Fante, 7, 6, 5, 4, 2.

        Ritorna:
            Un intero dove un valore più alto indica una carta più forte nella mano.
        """
        strength_by_name = {
            "ACE": 10,
            "THREE": 9,
            "KING": 8,
            "KNIGHT": 7,
            "JACK": 6,
            "SEVEN": 5,
            "SIX": 4,
            "FIVE": 3,
            "FOUR": 2,
            "TWO": 1,
        }
        return strength_by_name[self.name]


@dataclass
class Card:
    """
    Rappresenta una singola carta da gioco.

    Una carta è definita da:
    - `suit`: seme
    - `rank`: rango/valore
    """

    suit: Suit
    rank: Rank

    def __str__(self) -> str:
        """Rappresentazione breve, utile per debug/log."""
        return f"{self.rank.name} of {self.suit.value}"

    def __repr__(self) -> str:
        """Rappresentazione non ambigua (ricostruibile) per debug."""
        return f"Card({self.suit.name}, {self.rank.name})"

    def __eq__(self, other: object) -> bool:
        """Due carte sono uguali se hanno stesso seme e stesso rango."""
        return isinstance(other, Card) and self.suit == other.suit and self.rank == other.rank

    def __hash__(self) -> int:
        """Permette di usare `Card` in set/dict (utile per test di unicità del mazzo)."""
        return hash((self.suit, self.rank))


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
