"""
Agenti baseline (policy) per Briscola AI.

Scopo didattico
---------------
Prima delle reti neurali, conviene avere:
- baseline molto semplici (random, euristiche minime);
- un'interfaccia comune (dato uno stato, scegliere un'azione valida);
- un modo per confrontare agenti in modo riproducibile.

Questo modulo definisce alcune policy “plug-in” usabili sia in self-play offline,
sia (in futuro) nel backend come IA.

Anti-cheat (importante)
-----------------------
Gli agenti NON ricevono `GameState` completo: ricevono una `PlayerObservation`,
cioè la vista parziale lecita dal punto di vista di un giocatore.
Questo evita che una IA possa barare leggendo l'ordine del mazzo o la mano avversaria.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol

from ..domain.models import Card, Suit
from ..domain.observation import PlayerObservation
from ..domain.rules import trick_points, who_wins_trick
from .bc_model_agent import BCModelAgent


class Agent(Protocol):
    """
    Interfaccia minima di un agente.

    Un agente deve scegliere un `card_index` valido (indice nella mano del player).
    In Briscola qualunque carta in mano è giocabile, quindi l'insieme delle azioni valide
    è sempre `range(len(hand))` (finché la partita non è finita).
    """

    # Nota mypy:
    # definiamo `name` come proprietà in sola lettura. Se lo mettessimo come campo
    # (`name: str`), mypy lo tratterebbe come attributo "settable" e andrebbe in
    # conflitto con agenti implementati come `@dataclass(frozen=True)` (attributi read-only).
    @property
    def name(self) -> str:
        """Nome leggibile dell'agente (usato in CLI/log/metriche)."""

    def choose_card_index(self, observation: PlayerObservation, *, rng: random.Random) -> int:
        """Sceglie l'indice della carta da giocare per il giocatore osservante."""


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """
    Metadati “didattici” di un agente.

    Scopo:
    - avere una sorgente unica di verità per nome/descrizione dell'agente
    - poter esporre questi metadati a UI/CLI senza duplicare stringhe nel frontend

    Nota di design:
    Manteniamo `AgentSpec` *vicino* all'agente: ogni classe agente espone un attributo di classe `spec`.
    In questo modo implementazione e metadati rimangono coerenti e non possono “driftare” facilmente.
    """

    name: str
    label: str
    description_it: str


@dataclass(frozen=True)
class RandomAgent:
    """
    Baseline: sceglie una carta casuale tra quelle in mano.

    È una baseline “zero-intelligenza” ma molto utile per:
    - validare che il loop di simulazione funzioni;
    - avere un punto di riferimento per valutare euristiche e modelli.
    """

    spec: ClassVar[AgentSpec] = AgentSpec(
        name="random",
        label="Random",
        description_it=(
            "Sceglie una carta a caso tra quelle in mano. "
            "È una baseline semplice per verificare che tutto funzioni e misurare i miglioramenti."
        ),
    )
    name: str = "random"

    def choose_card_index(self, observation: PlayerObservation, *, rng: random.Random) -> int:
        hand_size = len(observation.hand)
        if hand_size <= 0:
            raise ValueError("Mano vuota: nessuna azione possibile")
        return rng.randrange(hand_size)


@dataclass(frozen=True)
class GreedyPointsAgent:
    """
    Euristica minimale: gioca la carta con più punti nella mano.

    Nota:
    Non è detto che sia forte in Briscola (spesso conviene conservare carichi),
    ma è una policy deterministica e “spiegabile”, utile come esempio.
    """

    spec: ClassVar[AgentSpec] = AgentSpec(
        name="greedy_points",
        label="Greedy (punti)",
        description_it=(
            "Gioca sempre la carta con più punti tra quelle in mano. "
            "È deterministico e spiegabile, ma spesso sub-ottimale (tende a sprecare carichi)."
        ),
    )
    name: str = "greedy_points"

    def choose_card_index(self, observation: PlayerObservation, *, rng: random.Random) -> int:
        hand = observation.hand
        if not hand:
            raise ValueError("Mano vuota: nessuna azione possibile")

        # In caso di pareggio, scegliamo in modo pseudo-casuale per non fissare sempre la stessa carta.
        best_points = max(card.rank.points for card in hand)
        candidates = [i for i, card in enumerate(hand) if card.rank.points == best_points]
        return candidates[rng.randrange(len(candidates))]


def card_to_short_string(card: Card) -> str:
    """
    Rappresentazione breve di una carta (debug).

    Esempio: `cups_ACE` o `clubs_TWO`.
    """

    return f"{card.suit.value}_{card.rank.name}"


@dataclass(frozen=True)
class HeuristicAgentV1:
    """
    Euristica “v1”: regole semplici e spiegabili (2-player).

    Idea generale:
    - Se siamo secondi di mano e possiamo prendere con una carta “economica”, prendiamo.
    - Se la mano contiene punti alti (es. Asso/Tre) proviamo a prenderla, ma evitando
      di sprecare briscole alte quando i punti in palio sono pochi.
    - Se non conviene prendere, scartiamo una carta “economica” (bassi punti, non briscola).

    Nota didattica:
    Questa policy è volutamente semplice e “spiegabile”.
    Come tutti gli agenti del progetto, vede solo una `PlayerObservation` (vista parziale lecita).
    """

    spec: ClassVar[AgentSpec] = AgentSpec(
        name="heuristic_v1",
        label="Euristica v1",
        description_it=(
            "Euristica 2-player: prova a prendere quando conviene (a basso costo) e a scartare in modo economico "
            "quando non conviene."
        ),
    )
    name: str = "heuristic_v1"

    def choose_card_index(self, observation: PlayerObservation, *, rng: random.Random) -> int:
        hand = observation.hand
        if not hand:
            raise ValueError("Mano vuota: nessuna azione possibile")

        trump_suit: Suit | None = observation.trump_card.suit if observation.trump_card else None

        # In Briscola 2-player la mano sul tavolo è lunga 0 (si apre) o 1 (si risponde).
        # Non implementiamo qui un comportamento “team-play” (4-player).
        if observation.num_players != 2:
            return rng.randrange(len(hand))

        # Se siamo i primi a giocare nella mano (tavolo vuoto).
        if not observation.table_cards:
            return self._choose_lead_card_index(
                hand, trump_suit=trump_suit, cards_remaining_in_deck=observation.deck_size
            )

        # Se siamo secondi di mano: vediamo la carta avversaria sul tavolo.
        lead_card, lead_player = observation.table_cards[0]
        return self._choose_response_card_index(
            hand,
            player_index=observation.player_index,
            lead_card=lead_card,
            lead_player=lead_player,
            trump_suit=trump_suit,
            cards_remaining_in_deck=observation.deck_size,
            rng=rng,
        )

    def _choose_lead_card_index(
        self, hand: tuple[Card, ...], *, trump_suit: Suit | None, cards_remaining_in_deck: int
    ) -> int:
        """
        Scelta quando siamo primi di mano.

        Strategia minimale:
        - inizio partita: preferisci scartare carte con 0 punti e non briscole;
        - endgame (mazzo vuoto): tende a giocare più “forte” (perché non ci sono più pescate).
        """
        if cards_remaining_in_deck <= 0:
            # Endgame: senza pescate successive, dare valore a prendere la mano.
            # Giochiamo la carta più forte (trick_strength alta); a parità, quella con più punti.
            best = max(
                range(len(hand)),
                key=lambda i: (hand[i].rank.trick_strength, hand[i].rank.points),
            )
            return best

        # Early/mid game: scarta “economico” (0 punti, non briscola, debole).
        def lead_key(i: int) -> tuple[int, int, int]:
            card = hand[i]
            is_trump = 1 if trump_suit is not None and card.suit == trump_suit else 0
            return (card.rank.points, is_trump, card.rank.trick_strength)

        return min(range(len(hand)), key=lead_key)

    def _choose_response_card_index(
        self,
        hand: tuple[Card, ...],
        *,
        player_index: int,
        lead_card: Card,
        lead_player: int,
        trump_suit: Suit | None,
        cards_remaining_in_deck: int,
        rng: random.Random,
    ) -> int:
        """
        Scelta quando rispondiamo (secondi di mano).

        Obiettivo:
        - se conviene, prendi la mano usando la carta più “economica” che vince;
        - altrimenti scarta la carta più economica (evitando briscole).
        """
        winning_candidates: list[int] = []
        for i, card in enumerate(hand):
            trick_cards = [(lead_card, lead_player), (card, player_index)]
            winner = who_wins_trick(trick_cards, trump_suit)
            if winner == player_index:
                winning_candidates.append(i)

        if winning_candidates:
            # Scegli la carta che vince con costo minimo:
            # - preferisci non briscola
            # - preferisci pochi punti
            # - preferisci forza bassa (non sprecare carte alte)
            def win_cost(idx: int) -> tuple[int, int, int]:
                c = hand[idx]
                is_trump = 1 if trump_suit is not None and c.suit == trump_suit else 0
                return (is_trump, c.rank.points, c.rank.trick_strength)

            best_win = min(winning_candidates, key=win_cost)
            best_win_card = hand[best_win]

            total_trick_points = trick_points([lead_card, best_win_card])

            # Regola di convenienza (molto semplice):
            # - se ci sono punti “alti” sul tavolo (>=10) proviamo a prenderli;
            # - se è endgame (mazzo vuoto), tendiamo a prendere più spesso;
            # - se la carta per prendere è “gratis” (0 punti e non briscola), prendiamo comunque.
            best_is_trump = trump_suit is not None and best_win_card.suit == trump_suit
            best_is_free = (best_win_card.rank.points == 0) and (not best_is_trump)

            if total_trick_points >= 10 or cards_remaining_in_deck <= 0 or best_is_free:
                return best_win

        # Altrimenti: scarta economico.
        # Se l'avversario ha giocato briscola e non possiamo prenderla, è ancora più importante
        # evitare di buttare una briscola nostra (meglio scartare un non-trump a 0 punti).
        def discard_key(idx: int) -> tuple[int, int, int]:
            c = hand[idx]
            is_trump = 1 if trump_suit is not None and c.suit == trump_suit else 0
            return c.rank.points, is_trump, c.rank.trick_strength

        cheapest = min(range(len(hand)), key=discard_key)
        return cheapest


_AGENT_BUILDERS: dict[str, type[Agent]] = {
    "random": RandomAgent,
    "greedy_points": GreedyPointsAgent,
    "heuristic_v1": HeuristicAgentV1,
}

BC_MODEL_SPEC = AgentSpec(
    name="bc_model",
    label="Modello locale (.npz)",
    description_it=(
        "Usa un modello addestrato e salvato in un file `.npz` (Behavior Cloning / RL). "
        "Il file è scelto dalla UI tra quelli disponibili sul server."
    ),
)

AI_AGENTS_COMMON_NOTE_IT = (
    "Nota anti-cheat: tutte le IA ricevono solo un’osservazione parziale (PlayerObservation). "
    "Non possono leggere l’ordine del mazzo né le carte specifiche in mano all’avversario."
)


def list_agent_specs() -> list[AgentSpec]:
    """Ritorna la lista di agenti disponibili con metadati (ordine stabile)."""
    return [RandomAgent.spec, GreedyPointsAgent.spec, HeuristicAgentV1.spec, BC_MODEL_SPEC]


def build_agent(name: str, *, model_path: Path | None = None) -> Agent:
    """
    Costruisce un agente a partire dal nome canonico.

    Nota:
    usiamo una mappa esplicita (no import dinamici) per semplicità e riproducibilità.
    """
    if name == "bc_model":
        if model_path is None:
            raise ValueError("Agente 'bc_model' richiede `model_path` (file .npz)")
        return BCModelAgent.from_npz(model_path)

    try:
        return _AGENT_BUILDERS[name]()
    except KeyError as exc:
        raise ValueError(f"Agente non supportato: {name!r}") from exc
