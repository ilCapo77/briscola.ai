"""
Agenti rule-based semplici e spiegabili.

Sono baseline intenzionalmente leggibili: servono a validare il motore, confrontare
i modelli neurali e mostrare come costruire una policy senza informazione nascosta.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import ClassVar

from ...domain.card_id import card_to_id, id_to_card
from ...domain.models import Card, Suit
from ...domain.observation import PlayerObservation
from ...domain.rules import trick_points, who_wins_trick
from .base import AgentSpec


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


@dataclass(frozen=True)
class HeuristicAgentV2:
    """
    Euristica “v2”: come v1, ma con segnali di “corso della partita” (2-player).

    Motivazione (didattica)
    -----------------------
    Una policy può sembrare “miopica” (es. spreca briscole) per due ragioni diverse:
    1) non ha abbastanza informazione pubblica aggregata (es. quante briscole sono già uscite);
    2) ha informazione, ma non ha una regola/obiettivo che valorizzi il futuro.

    In questo progetto abbiamo introdotto `PlayerObservation.seen_cards_onehot[40]` proprio per
    abilitare un card counting *lecito* (anti-cheat): tavolo + prese + briscola scoperta.

    Questa euristica usa quel segnale in modo semplice:
    - early game: evita di “spendere briscola” per prendere pochi punti;
    - late game (o quando le briscole rimaste sono poche): è più disposta a usare una briscola economica
      per ottenere il controllo (essere primi di mano) anche su mani a basso valore.

    Nota:
    - È comunque una policy rule-based, non “perfetta”. Serve soprattutto come:
      - baseline alternativa in valutazione, e/o
      - teacher per Behavior Cloning (BC) quando vogliamo uno stile più “umano”.
    """

    spec: ClassVar[AgentSpec] = AgentSpec(
        name="heuristic_v2",
        label="Euristica v2",
        description_it=(
            "Euristica 2-player con un minimo di ‘card counting’ lecito (carte già viste) e gestione briscole: "
            "evita sprechi in early game e gioca più aggressivo in late game quando le briscole rimaste sono poche."
        ),
    )
    name: str = "heuristic_v2"

    def choose_card_index(self, observation: PlayerObservation, *, rng: random.Random) -> int:
        hand = observation.hand
        if not hand:
            raise ValueError("Mano vuota: nessuna azione possibile")

        trump_suit: Suit | None = observation.trump_card.suit if observation.trump_card else None

        if observation.num_players != 2:
            return rng.randrange(len(hand))

        if not observation.table_cards:
            return self._choose_lead_card_index(
                hand, trump_suit=trump_suit, cards_remaining_in_deck=observation.deck_size
            )

        lead_card, lead_player = observation.table_cards[0]
        return self._choose_response_card_index(
            hand,
            player_index=observation.player_index,
            lead_card=lead_card,
            lead_player=lead_player,
            trump_suit=trump_suit,
            cards_remaining_in_deck=observation.deck_size,
            seen_cards_onehot=observation.seen_cards_onehot,
            rng=rng,
        )

    def _choose_lead_card_index(
        self, hand: tuple[Card, ...], *, trump_suit: Suit | None, cards_remaining_in_deck: int
    ) -> int:
        """
        Scelta quando siamo primi di mano.

        Per ora manteniamo una strategia simile alla v1 (semplice e spiegabile):
        - endgame: gioca forte;
        - altrimenti: scarta economico (pochi punti, evita briscole).
        """
        if cards_remaining_in_deck <= 0:
            best = max(
                range(len(hand)),
                key=lambda i: (hand[i].rank.trick_strength, hand[i].rank.points),
            )
            return best

        def lead_key(i: int) -> tuple[int, int, int]:
            card = hand[i]
            is_trump = 1 if trump_suit is not None and card.suit == trump_suit else 0
            return (card.rank.points, is_trump, card.rank.trick_strength)

        return min(range(len(hand)), key=lead_key)

    @staticmethod
    def _count_remaining_trumps_public(seen_cards_onehot: tuple[int, ...], trump_suit: Suit) -> int:
        """
        Conta quante briscole non risultano ancora “viste” pubblicamente.

        Interpretazione:
        - se molte briscole sono già viste, la partita è più avanti e il “controllo” ha valore diverso.
        - usiamo solo `seen_cards_onehot` (anti-cheat), senza mai ispezionare `GameState`.
        """
        remaining = 0
        for card_id, seen in enumerate(seen_cards_onehot):
            if not seen and id_to_card(card_id).suit == trump_suit:
                remaining += 1
        return remaining

    @staticmethod
    def _count_unknown_high_trumps(
        *,
        hand: tuple[Card, ...],
        seen_cards_onehot: tuple[int, ...],
        trump_suit: Suit,
        strength_threshold: int = 9,
    ) -> int:
        """
        Stima quante briscole *molto forti* potrebbero ancora essere in giro (deck o mano avversaria).

        "Unknown" = non viste pubblicamente e non in mano a noi.
        È un proxy semplice per dire: "se non abbiamo ancora visto Asso/Tre di briscola,
        forse conviene conservare le nostre briscole alte per più avanti".
        """
        my_hand_ids = {card_to_id(c) for c in hand}
        unknown = 0
        for card_id, seen in enumerate(seen_cards_onehot):
            if seen:
                continue
            if card_id in my_hand_ids:
                continue
            c = id_to_card(card_id)
            if c.suit == trump_suit and c.rank.trick_strength >= strength_threshold:
                unknown += 1
        return unknown

    @staticmethod
    def _is_high_trump(card: Card) -> bool:
        """Heuristica: considera “alta” una briscola con forza >= Re (strength>=8)."""
        return card.rank.trick_strength >= 8

    def _should_take_with_trump(
        self,
        *,
        hand: tuple[Card, ...],
        lead_card: Card,
        response_card: Card,
        trump_suit: Suit,
        cards_remaining_in_deck: int,
        seen_cards_onehot: tuple[int, ...],
    ) -> bool:
        """
        Decide se vale la pena prendere usando una briscola (anche se potremmo vincere).

        Regole (spiegabili, non ottimali):
        - endgame: prendere più spesso (non ci sono più pescate).
        - mani “ricche” (>=10 punti sul tavolo dopo la risposta): prendere.
        - late game / poche briscole residue: prendere anche mani povere se la briscola è economica
          (per avere il controllo e guidare la mano successiva).
        - early game: evitare di spendere briscole alte quando non necessario.
        """
        total_points = trick_points([lead_card, response_card])
        if cards_remaining_in_deck <= 0:
            return True
        if total_points >= 10:
            return True

        remaining_trumps_public = self._count_remaining_trumps_public(seen_cards_onehot, trump_suit)
        unknown_high_trumps = self._count_unknown_high_trumps(
            hand=hand, seen_cards_onehot=seen_cards_onehot, trump_suit=trump_suit, strength_threshold=9
        )

        is_high_trump = self._is_high_trump(response_card)

        # Late-ish: poche carte nel mazzo o poche briscole rimaste visibili.
        is_late = cards_remaining_in_deck <= 4 or remaining_trumps_public <= 2

        if is_late:
            # In late game il controllo vale di più: se la briscola è economica (0 punti) possiamo
            # permetterci di prendere anche mani povere.
            if response_card.rank.points == 0 and not is_high_trump:
                return True

            # Se non abbiamo più “incognite” (briscole fortissime non ancora viste),
            # siamo più tranquilli a spendere una briscola alta anche su valore medio.
            if unknown_high_trumps == 0 and total_points >= 3:
                return True

        # Default: in early/mid game, non prendere mani povere con briscola (conservazione risorse).
        return False

    def _choose_response_card_index(
        self,
        hand: tuple[Card, ...],
        *,
        player_index: int,
        lead_card: Card,
        lead_player: int,
        trump_suit: Suit | None,
        cards_remaining_in_deck: int,
        seen_cards_onehot: tuple[int, ...],
        rng: random.Random,
    ) -> int:
        """
        Scelta quando rispondiamo (secondi di mano).

        Obiettivo:
        - se possiamo prendere senza briscola, prendiamo “economico”;
        - se possiamo prendere solo con briscola, decidiamo se vale la pena (fase della partita + valore mano),
          e comunque scegliamo la briscola vincente minima (anti-overkill).
        - altrimenti scartiamo economico (evitando briscole).
        """
        if trump_suit is None:
            # Caso raro: senza briscola non esiste concetto di trump; torniamo a comportamento semplice.
            return rng.randrange(len(hand))

        winning_non_trumps: list[int] = []
        winning_trumps: list[int] = []
        for i, card in enumerate(hand):
            trick_cards = [(lead_card, lead_player), (card, player_index)]
            winner = who_wins_trick(trick_cards, trump_suit)
            if winner != player_index:
                continue
            if card.suit == trump_suit:
                winning_trumps.append(i)
            else:
                winning_non_trumps.append(i)

        if winning_non_trumps:
            # Vincere senza briscola è quasi sempre “buono”: non consumiamo trump.
            def win_cost_non_trump(idx: int) -> tuple[int, int]:
                c = hand[idx]
                return (c.rank.points, c.rank.trick_strength)

            return min(winning_non_trumps, key=win_cost_non_trump)

        if winning_trumps:
            # Candidato “minimo” tra i trump vincenti (anti-overkill by design).
            def win_cost_trump(idx: int) -> tuple[int, int]:
                c = hand[idx]
                return (c.rank.points, c.rank.trick_strength)

            best_trump_idx = min(winning_trumps, key=win_cost_trump)
            best_trump = hand[best_trump_idx]

            if self._should_take_with_trump(
                hand=hand,
                lead_card=lead_card,
                response_card=best_trump,
                trump_suit=trump_suit,
                cards_remaining_in_deck=cards_remaining_in_deck,
                seen_cards_onehot=seen_cards_onehot,
            ):
                return best_trump_idx

        # Fallback: scarta economico (evita briscole).
        def discard_key(idx: int) -> tuple[int, int, int]:
            c = hand[idx]
            is_trump = 1 if c.suit == trump_suit else 0
            return (c.rank.points, is_trump, c.rank.trick_strength)

        return min(range(len(hand)), key=discard_key)
