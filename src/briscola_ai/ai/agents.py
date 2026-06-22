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
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Protocol

from ..domain.card_id import card_to_id, id_to_card
from ..domain.models import Card, Suit
from ..domain.observation import PlayerObservation
from ..domain.rules import trick_points, who_wins_trick
from ..domain.state import GameState, PlayerState
from .bc_model_agent import BCModelAgent
from .endgame_solver import solve_endgame
from .model_catalog import get_models_dir_from_env, resolve_model_path
from .training.observation_encoder import FEATURE_DIM_2P_V1, FEATURE_DIM_2P_V2


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


_ALL_CARD_IDS = frozenset(range(40))


def _seen_card_ids_from_observation(observation: PlayerObservation) -> set[int]:
    """
    Converte `seen_cards_onehot` in un set di card id, validando la shape pubblica.

    Il campo è parte del confine anti-cheat: se manca o ha una shape inattesa non proviamo a
    "indovinare" l'endgame, ma lasciamo che l'agente ibrido usi il fallback.
    """
    if len(observation.seen_cards_onehot) != 40:
        raise ValueError(f"seen_cards_onehot deve avere lunghezza 40, trovata {len(observation.seen_cards_onehot)}")

    seen_ids: set[int] = set()
    for card_id, seen in enumerate(observation.seen_cards_onehot):
        if seen not in (0, 1):
            raise ValueError(f"seen_cards_onehot contiene un valore non binario in posizione {card_id}: {seen!r}")
        if seen:
            seen_ids.add(card_id)
    return seen_ids


def _validate_endgame_observation_scope(observation: PlayerObservation) -> None:
    """
    Verifica che l'osservazione sia nello scope risolvibile dal solver endgame 2-player.

    L'agente deve giocare solo dal proprio punto di vista: quindi richiediamo esplicitamente che
    `current_turn == player_index`. Se non è vero, l'osservazione non rappresenta una decisione
    dell'agente e il solver non va consultato.
    """
    if observation.num_players != 2 or observation.is_team_game:
        raise ValueError("Il solver ibrido supporta solo osservazioni 2-player non a squadre")
    if observation.player_index not in (0, 1):
        raise ValueError(f"player_index fuori range: {observation.player_index}")
    if observation.current_turn not in (0, 1):
        raise ValueError(f"current_turn fuori range: {observation.current_turn}")
    if observation.current_turn != observation.player_index:
        raise ValueError("L'osservazione non è sul turno del player osservante")
    if observation.first_player not in (0, 1):
        raise ValueError(f"first_player fuori range: {observation.first_player}")
    if observation.game_over:
        raise ValueError("Partita già terminata")
    if observation.deck_size != 0:
        raise ValueError(f"Il solver endgame richiede deck_size=0, trovato {observation.deck_size}")
    if observation.trump_card is None:
        raise ValueError("Briscola assente: impossibile ricostruire il seme di briscola")
    if len(observation.players_points) != 2:
        raise ValueError(f"players_points deve avere lunghezza 2, trovata {len(observation.players_points)}")
    if len(observation.players_hand_sizes) != 2:
        raise ValueError(f"players_hand_sizes deve avere lunghezza 2, trovata {len(observation.players_hand_sizes)}")
    if any(size < 0 or size > 3 for size in observation.players_hand_sizes):
        raise ValueError(f"Dimensioni mani fuori range endgame: {observation.players_hand_sizes!r}")
    if len(observation.hand) != observation.players_hand_sizes[observation.player_index]:
        raise ValueError("La mano osservata non coincide con players_hand_sizes[player_index]")
    if len(observation.table_cards) not in (0, 1):
        raise ValueError(f"Tavolo non supportato: attese 0 o 1 carte, trovate {len(observation.table_cards)}")

    remaining = sum(observation.players_hand_sizes) + len(observation.table_cards)
    if remaining <= 0:
        raise ValueError("Osservazione non terminale senza carte residue")
    if remaining > 6:
        raise ValueError(f"Troppe carte residue per l'endgame: {remaining}")

    if not observation.table_cards:
        if observation.players_hand_sizes[0] != observation.players_hand_sizes[1]:
            raise ValueError(f"Mani sbilanciate a tavolo vuoto: {observation.players_hand_sizes!r}")
        if observation.first_player != observation.current_turn:
            raise ValueError("first_player incoerente a tavolo vuoto")
        return

    leader = observation.table_cards[0][1]
    if leader not in (0, 1):
        raise ValueError(f"Player id sul tavolo fuori range: {leader}")
    if leader == observation.current_turn:
        raise ValueError("Turno incoerente: chi ha aperto la mano non può rigiocare")
    if observation.first_player != leader:
        raise ValueError("first_player incoerente con la carta sul tavolo")
    if observation.players_hand_sizes[observation.current_turn] != observation.players_hand_sizes[leader] + 1:
        raise ValueError("Mani sbilanciate rispetto alla carta sul tavolo")


def _opponent_hand_ids_from_out_of_play(
    observation: PlayerObservation,
    *,
    my_hand_ids: set[int],
    table_ids: set[int],
    opponent_hand_size: int,
) -> set[int] | None:
    """
    Deduce la mano avversaria dal campo `out_of_play_cards_onehot`, se presente e coerente.

    È il path "pulito": le carte fuori gioco sono SOLO prese + tavolo, quindi a mazzo vuoto
    `mano_avversario = tutte − mia_mano − fuori_gioco` (la briscola non richiede trattamenti
    speciali: se è in mano avversaria non è fuori gioco, quindi ricade nel complemento).

    Ritorna `None` (→ fallback su `seen_cards_onehot`) se il campo è assente/default/incoerente:
    - lunghezza diversa da 40 o valori non binari;
    - non contiene tutte le carte sul tavolo (il tavolo è per definizione fuori gioco);
    - si sovrappone alla mano osservante (la mia mano non è fuori gioco);
    - il complemento non ha la dimensione attesa (es. campo a 40 zeri dei dataset vecchi).
    """
    raw = observation.out_of_play_cards_onehot
    if len(raw) != 40:
        return None

    out_of_play_ids: set[int] = set()
    for card_id, value in enumerate(raw):
        if value not in (0, 1):
            return None
        if value:
            out_of_play_ids.add(card_id)

    if not table_ids.issubset(out_of_play_ids):
        return None
    if my_hand_ids & out_of_play_ids:
        return None

    candidate = set(_ALL_CARD_IDS - my_hand_ids - out_of_play_ids)
    if len(candidate) != opponent_hand_size:
        return None
    return candidate


def _opponent_hand_ids_from_seen(
    observation: PlayerObservation,
    *,
    my_hand_ids: set[int],
    table_ids: set[int],
    trump_id: int,
    opponent_hand_size: int,
) -> set[int]:
    """
    Deduce la mano avversaria dalla sola `seen_cards_onehot` (path compatibile, fallback).

    `seen_cards_onehot` include sempre la briscola scoperta, anche quando è stata pescata in mano:
    è l'unico bit ambiguo, risolto contando le mani (se manca esattamente una carta ai candidati,
    quella carta è la briscola). Solleva `ValueError` se l'osservazione è incoerente.
    """
    seen_ids = _seen_card_ids_from_observation(observation)
    if trump_id not in seen_ids:
        raise ValueError("seen_cards_onehot non contiene la briscola pubblica")
    if not table_ids.issubset(seen_ids):
        raise ValueError("seen_cards_onehot non contiene tutte le carte sul tavolo")

    # La briscola scoperta è l'unica sovrapposizione lecita tra mano osservata e carte "viste".
    illegal_seen_hand_overlap = (my_hand_ids & seen_ids) - {trump_id}
    if illegal_seen_hand_overlap:
        raise ValueError("seen_cards_onehot contiene carte non-briscola ancora nella mano osservata")

    remaining_unknown_ids = set(_ALL_CARD_IDS - my_hand_ids - seen_ids)
    if len(remaining_unknown_ids) == opponent_hand_size:
        return remaining_unknown_ids
    if (
        len(remaining_unknown_ids) == opponent_hand_size - 1
        and trump_id not in my_hand_ids
        and trump_id not in table_ids
    ):
        opponent_hand_ids = set(remaining_unknown_ids)
        opponent_hand_ids.add(trump_id)
        return opponent_hand_ids
    raise ValueError(
        "Impossibile dedurre in modo univoco la mano avversaria "
        f"(candidati={len(remaining_unknown_ids)}, attesi={opponent_hand_size})"
    )


def reconstruct_endgame_state(observation: PlayerObservation) -> GameState:
    """
    Ricostruisce uno `GameState` endgame 2-player dalla sola `PlayerObservation`.

    Anti-cheat
    ----------
    La funzione non legge mai la mano avversaria dal dominio: usa solo informazione pubblica
    (`out_of_play_cards_onehot`/`seen_cards_onehot`, tavolo, dimensioni mani) e la mano propria.

    Deduzione mano avversaria
    -------------------------
    Preferisce `out_of_play_cards_onehot` (path pulito: complemento diretto) quando presente e
    coerente; altrimenti usa il fallback storico su `seen_cards_onehot`. Questo evita una
    migrazione "tutto o niente" e mantiene la compatibilità coi dataset/osservazioni vecchie.

    Punti e prese
    -------------
    Lo stato ricostruito azzera `points` e `captured_cards` di entrambi i player. È intenzionale:
    `domain.step` ricalcola i punti del vincitore da `captured_cards`, quindi copiare i punteggi
    reali senza conoscere la partizione delle prese corromperebbe il delta. La base punti è una
    costante rispetto alle mosse future, perciò non cambia la scelta ottima del solver.
    """
    _validate_endgame_observation_scope(observation)

    player_index = observation.player_index
    opponent_index = 1 - player_index
    trump_card = observation.trump_card
    if trump_card is None:
        # Ridondante rispetto alla validate, ma aiuta mypy a restringere il tipo.
        raise ValueError("Briscola assente")

    trump_id = card_to_id(trump_card)

    my_hand_ids = {card_to_id(card) for card in observation.hand}
    if len(my_hand_ids) != len(observation.hand):
        raise ValueError("La mano osservata contiene carte duplicate")

    table_ids = {card_to_id(card) for card, _player_idx in observation.table_cards}
    if len(table_ids) != len(observation.table_cards):
        raise ValueError("Il tavolo contiene carte duplicate")
    if my_hand_ids & table_ids:
        raise ValueError("Una carta non può essere insieme in mano e sul tavolo")

    opponent_hand_size = observation.players_hand_sizes[opponent_index]

    # Path pulito (out_of_play) con fallback compatibile (seen).
    opponent_hand_ids = _opponent_hand_ids_from_out_of_play(
        observation,
        my_hand_ids=my_hand_ids,
        table_ids=table_ids,
        opponent_hand_size=opponent_hand_size,
    )
    if opponent_hand_ids is None:
        opponent_hand_ids = _opponent_hand_ids_from_seen(
            observation,
            my_hand_ids=my_hand_ids,
            table_ids=table_ids,
            trump_id=trump_id,
            opponent_hand_size=opponent_hand_size,
        )

    if len(opponent_hand_ids) != opponent_hand_size:
        raise ValueError("Dimensione mano avversaria ricostruita incoerente")
    if opponent_hand_ids & my_hand_ids or opponent_hand_ids & table_ids:
        raise ValueError("Mano avversaria ricostruita sovrapposta a carte pubbliche o proprie")

    opponent_hand = tuple(id_to_card(card_id) for card_id in sorted(opponent_hand_ids))
    players = [
        PlayerState(name="P0", hand=tuple(), captured_cards=tuple(), points=0),
        PlayerState(name="P1", hand=tuple(), captured_cards=tuple(), points=0),
    ]
    players[player_index] = PlayerState(
        name=observation.player_name,
        hand=observation.hand,
        captured_cards=tuple(),
        points=0,
    )
    players[opponent_index] = PlayerState(
        name=f"P{opponent_index}",
        hand=opponent_hand,
        captured_cards=tuple(),
        points=0,
    )

    return GameState(
        num_players=2,
        is_team_game=False,
        teams=None,
        players=tuple(players),
        deck=tuple(),
        trump_card=trump_card,
        table_cards=observation.table_cards,
        current_turn=observation.current_turn,
        first_player=observation.first_player,
        game_over=False,
        winner_index=None,
        winning_team=None,
    )


def can_solve_endgame_from_observation(observation: PlayerObservation) -> bool:
    """
    Ritorna True se l'osservazione può essere ricostruita e risolta senza informazione nascosta.

    È una funzione di utilità per test/debug. `HybridEndgameAgent` usa lo stesso criterio, ma evita
    di chiamarla per non risolvere due volte lo stesso stato.
    """
    try:
        solve_endgame(reconstruct_endgame_state(observation))
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class HybridEndgameAgent:
    """
    Agente ibrido: fallback normale in mid-game, solver esatto a mazzo vuoto.

    In endgame prova a ricostruire uno stato completo dalla sola osservazione lecita; se la
    ricostruzione non è valida o il solver rifiuta lo stato, delega al fallback. Questo mantiene
    l'invariante anti-cheat: l'agente non legge mai `GameState.players[opponent].hand`.
    """

    spec: ClassVar[AgentSpec] = AgentSpec(
        name="hybrid_endgame",
        label="Hybrid Endgame",
        description_it=(
            "Usa l'euristica v2 durante la partita e, a mazzo vuoto, passa a un solver esatto "
            "ricostruito dalla sola osservazione pubblica."
        ),
    )

    fallback: Agent = field(default_factory=HeuristicAgentV2)
    name: str = "hybrid_endgame"

    def choose_card_index(self, observation: PlayerObservation, *, rng: random.Random) -> int:
        try:
            reconstructed = reconstruct_endgame_state(observation)
            return solve_endgame(reconstructed).best_card_index
        except ValueError:
            return self.fallback.choose_card_index(observation, rng=rng)


_AGENT_BUILDERS: dict[str, type[Agent]] = {
    "random": RandomAgent,
    "greedy_points": GreedyPointsAgent,
    "heuristic_v1": HeuristicAgentV1,
    "heuristic_v2": HeuristicAgentV2,
    "hybrid_endgame": HybridEndgameAgent,
}

BC_MODEL_SPEC = AgentSpec(
    name="bc_model",
    label="Modello locale (.npz)",
    description_it=(
        "Usa un modello addestrato e salvato in un file `.npz` (Behavior Cloning / RL). "
        "Il file è scelto dalla UI tra quelli disponibili sul server."
    ),
)

BEST_A2C_SPEC = AgentSpec(
    name="best_a2c",
    label="Best A2C (locale)",
    description_it=(
        "Carica un modello “campione” A2C da un file locale `best_a2c.npz` nella directory modelli. "
        "È pensato per training in stile league (avversario congelato) e per confronti riproducibili."
    ),
)

_BEST_A2C_DEFAULT_MODEL_ID = "best_a2c.npz"

HYBRID_ENDGAME_BEST_A2C_SPEC = AgentSpec(
    name="hybrid_endgame_best_a2c",
    label="Hybrid Endgame (Best A2C)",
    description_it=(
        "Come Hybrid Endgame, ma usa il modello campione `best_a2c.npz` come policy in mid-game "
        "e il solver esatto a mazzo vuoto. Unisce la forza mid-game di best_a2c al finale ottimo."
    ),
)

AI_AGENTS_COMMON_NOTE_IT = (
    "Nota anti-cheat: tutte le IA ricevono solo un’osservazione parziale (PlayerObservation). "
    "Non possono leggere l’ordine del mazzo né le carte specifiche in mano all’avversario."
)


def list_agent_specs() -> list[AgentSpec]:
    """Ritorna la lista di agenti disponibili con metadati (ordine stabile)."""
    return [
        RandomAgent.spec,
        GreedyPointsAgent.spec,
        HeuristicAgentV1.spec,
        HeuristicAgentV2.spec,
        HybridEndgameAgent.spec,
        HYBRID_ENDGAME_BEST_A2C_SPEC,
        BC_MODEL_SPEC,
    ]


def _load_best_a2c_agent() -> BCModelAgent:
    """
    Carica il modello campione `best_a2c.npz` dalla directory modelli e ne valida la compatibilità.

    Estratto come helper perché serve sia all'agente `best_a2c` sia a `hybrid_endgame_best_a2c`
    (che lo usa come policy mid-game), così la logica di risoluzione path/validazione resta unica.
    """
    models_dir = get_models_dir_from_env()
    try:
        path = resolve_model_path(models_dir=models_dir, model_id=_BEST_A2C_DEFAULT_MODEL_ID)
    except FileNotFoundError as exc:
        raise ValueError(
            "Modello 'best_a2c' non disponibile: file non trovato. "
            "Convenzione: salva (o copia) un modello `.npz` compatibile in "
            f"{models_dir.resolve()!s}/{_BEST_A2C_DEFAULT_MODEL_ID}. "
            "Puoi cambiare directory impostando `BRISCOLA_MODELS_DIR`."
        ) from exc

    agent = BCModelAgent.from_npz(path)
    if int(agent.model.feature_dim) not in {int(FEATURE_DIM_2P_V1), int(FEATURE_DIM_2P_V2)}:
        expected = f"{int(FEATURE_DIM_2P_V1)} (v1) or {int(FEATURE_DIM_2P_V2)} (v2)"
        raise ValueError(
            "Modello 'best_a2c' non compatibile: feature_dim non coerente con un encoder 2-player supportato. "
            f"model={int(agent.model.feature_dim)} expected={expected} ({path})."
        )
    return agent


def build_agent(name: str, *, model_path: Path | None = None) -> Agent:
    """
    Costruisce un agente a partire dal nome canonico.

    Nota:
    usiamo una mappa esplicita (no import dinamici) per semplicità e riproducibilità.
    """
    if name == "best_a2c":
        return _load_best_a2c_agent()

    if name == "hybrid_endgame_best_a2c":
        # Variante esplicita di hybrid_endgame con policy mid-game = best_a2c.
        # `hybrid_endgame` resta invariato (fallback heuristic_v2) per stabilità dei benchmark.
        return HybridEndgameAgent(fallback=_load_best_a2c_agent(), name="hybrid_endgame_best_a2c")

    if name == "bc_model":
        if model_path is None:
            raise ValueError("Agente 'bc_model' richiede `model_path` (file .npz)")
        return BCModelAgent.from_npz(model_path)

    try:
        return _AGENT_BUILDERS[name]()
    except KeyError as exc:
        raise ValueError(f"Agente non supportato: {name!r}") from exc
