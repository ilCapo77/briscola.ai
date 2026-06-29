"""
Agente PIMC (Perfect-Information Monte Carlo) per prototipi offline.

Obiettivo
---------
Questo modulo implementa una prima search a inference sopra una policy esistente (es. v6):

1. parte solo da `PlayerObservation`, quindi non legge mano avversaria o ordine reale del mazzo;
2. campiona stati completi compatibili con informazione pubblica + mano del player;
3. prova ogni mossa legale su ciascuna determinizzazione;
4. completa la partita con una policy di rollout;
5. sceglie la mossa con miglior delta punti medio.

Scope
-----
È un prototipo offline, non un agente UI di default. Serve a verificare se search + v6 supera v6 puro
nel finale/semi-finale prima di investire in integrazione runtime o distillazione teacher.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import ClassVar

from ...domain.card_id import card_to_id, id_to_card
from ...domain.engine import PlayCardAction, step
from ...domain.observation import PlayerObservation, make_player_observation
from ...domain.state import GameState, PlayerState
from ..endgame.solver import solve_endgame
from .base import Agent, AgentSpec
from .hybrid_endgame import reconstruct_endgame_state
from .rule_based import HeuristicAgentV2

_ALL_CARD_IDS = frozenset(range(40))


@dataclass(slots=True)
class PIMCSearchStats:
    """Metriche runtime raccolte da `PIMCAgent` durante una evaluation offline."""

    total_decisions: int = 0
    search_decisions: int = 0
    fallback_decisions: int = 0
    endgame_solver_decisions: int = 0
    successful_determinizations: int = 0
    failed_determinizations: int = 0
    completed_rollouts: int = 0
    failed_rollouts: int = 0
    coerced_moves: int = 0
    search_elapsed_seconds: float = 0.0

    @property
    def seconds_per_search_decision(self) -> float:
        """Tempo medio speso nelle sole decisioni in cui PIMC ha cercato davvero."""
        if self.search_decisions <= 0:
            return 0.0
        return self.search_elapsed_seconds / self.search_decisions


def _onehot_ids(raw: tuple[int, ...], *, name: str) -> set[int]:
    """Converte una one-hot 40 in set di card id con validazione esplicita."""
    if len(raw) != 40:
        raise ValueError(f"{name} deve avere lunghezza 40, trovata {len(raw)}")
    ids: set[int] = set()
    for card_id, value in enumerate(raw):
        if value not in (0, 1):
            raise ValueError(f"{name} contiene un valore non binario in posizione {card_id}: {value!r}")
        if value:
            ids.add(card_id)
    return ids


def _card_points(card_id: int) -> int:
    """Punti Briscola della carta canonica."""
    return int(id_to_card(card_id).rank.points)


@lru_cache(maxsize=2048)
def _subset_with_points(card_ids: tuple[int, ...], target_points: int) -> frozenset[int] | None:
    """
    Ritorna un sottoinsieme di `card_ids` con somma punti esatta.

    Serve per ricostruire `captured_cards` coerenti con `players_points`. Le carte catturate sono
    pubbliche come insieme (`out_of_play - table`), ma l'osservazione non espone la partizione per
    giocatore: una qualsiasi partizione con lo stesso punteggio è sufficiente per simulare correttamente
    punteggio corrente, card counting e futuri incrementi di punti.
    """
    if target_points < 0:
        return None
    if target_points == 0:
        return frozenset()
    if not card_ids:
        return None

    first, *rest = card_ids
    rest_tuple = tuple(rest)

    with_first = _subset_with_points(rest_tuple, target_points - _card_points(first))
    if with_first is not None:
        return frozenset({first, *with_first})

    return _subset_with_points(rest_tuple, target_points)


def _captured_cards_for_scores(
    *,
    captured_ids: set[int],
    player0_points: int,
    player1_points: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Partiziona le carte catturate pubbliche in due insiemi con i punti osservati."""
    total_points = sum(_card_points(card_id) for card_id in captured_ids)
    if total_points != int(player0_points) + int(player1_points):
        raise ValueError(
            "Carte fuori gioco incoerenti con players_points: "
            f"punti_catturati={total_points}, players_points={player0_points + player1_points}"
        )

    ordered = tuple(sorted(captured_ids, key=lambda cid: (_card_points(cid), cid), reverse=True))
    player0_ids = _subset_with_points(ordered, int(player0_points))
    if player0_ids is None:
        raise ValueError(f"Impossibile partizionare le prese per ottenere {player0_points} punti a P0")

    player1_ids = captured_ids - set(player0_ids)
    if sum(_card_points(card_id) for card_id in player1_ids) != int(player1_points):
        raise ValueError(f"Impossibile partizionare le prese per ottenere {player1_points} punti a P1")

    return tuple(sorted(player0_ids)), tuple(sorted(player1_ids))


def _validate_pimc_observation(observation: PlayerObservation) -> None:
    """Verifica lo scope PIMC 2-player e il fatto che l'agente stia decidendo per sé."""
    if observation.num_players != 2 or observation.is_team_game:
        raise ValueError("PIMC supporta solo osservazioni 2-player non a squadre")
    if observation.player_index not in (0, 1):
        raise ValueError(f"player_index fuori range: {observation.player_index}")
    if observation.current_turn != observation.player_index:
        raise ValueError("PIMC può decidere solo quando current_turn == player_index")
    if observation.game_over:
        raise ValueError("Partita già terminata")
    if observation.trump_card is None:
        raise ValueError("Briscola assente: impossibile determinizzare")
    if len(observation.table_cards) not in (0, 1):
        raise ValueError(f"Tavolo non supportato: attese 0 o 1 carte, trovate {len(observation.table_cards)}")
    if len(observation.players_points) != 2:
        raise ValueError(f"players_points deve avere lunghezza 2, trovata {len(observation.players_points)}")
    if len(observation.players_hand_sizes) != 2:
        raise ValueError(f"players_hand_sizes deve avere lunghezza 2, trovata {len(observation.players_hand_sizes)}")
    if len(observation.hand) != observation.players_hand_sizes[observation.player_index]:
        raise ValueError("La mano osservata non coincide con players_hand_sizes[player_index]")


def unknown_live_card_count(observation: PlayerObservation) -> int:
    """Numero di carte vive non note al player: mano avversaria + mazzo."""
    opponent_index = 1 - observation.player_index
    return int(observation.players_hand_sizes[opponent_index]) + int(observation.deck_size)


def _safe_agent_card_index(
    agent: Agent,
    observation: PlayerObservation,
    *,
    rng: random.Random,
    metrics: PIMCSearchStats | None = None,
) -> int:
    """
    Chiede una mossa a un agente e la normalizza a un indice valido.

    In Briscola ogni carta in mano è giocabile nel dominio del progetto. Se un fallback/rollout agent
    restituisce un indice fuori range, scegliamo in modo difensivo la prima carta invece di abortire la
    determinizzazione o l'intera decisione PIMC.
    """
    if not observation.hand:
        raise ValueError("Mano vuota: nessuna azione possibile")
    try:
        card_index = int(agent.choose_card_index(observation, rng=rng))
    except Exception:
        if metrics is not None:
            metrics.coerced_moves += 1
        return 0
    if 0 <= card_index < len(observation.hand):
        return card_index
    if metrics is not None:
        metrics.coerced_moves += 1
    return 0


def determinize_observation(observation: PlayerObservation, *, rng: random.Random) -> GameState:
    """
    Campiona uno `GameState` completo compatibile con una `PlayerObservation`.

    Anti-cheat: usa solo mano osservata, tavolo, dimensioni mani, deck_size e carte fuori gioco
    pubbliche (`out_of_play_cards_onehot`). Non usa mai lo stato reale nascosto.
    """
    _validate_pimc_observation(observation)

    if observation.deck_size == 0:
        return reconstruct_endgame_state(observation)

    player_index = observation.player_index
    opponent_index = 1 - player_index
    trump_card = observation.trump_card
    if trump_card is None:
        raise ValueError("Briscola assente")

    out_of_play_ids = _onehot_ids(observation.out_of_play_cards_onehot, name="out_of_play_cards_onehot")
    table_ids = {card_to_id(card) for card, _player_idx in observation.table_cards}
    my_hand_ids = [card_to_id(card) for card in observation.hand]
    if len(set(my_hand_ids)) != len(my_hand_ids):
        raise ValueError("La mano osservata contiene carte duplicate")
    if len(table_ids) != len(observation.table_cards):
        raise ValueError("Il tavolo contiene carte duplicate")
    if not table_ids.issubset(out_of_play_ids):
        raise ValueError("out_of_play_cards_onehot non contiene tutte le carte sul tavolo")
    if set(my_hand_ids) & out_of_play_ids:
        raise ValueError("out_of_play_cards_onehot si sovrappone alla mano osservata")

    trump_id = card_to_id(trump_card)
    unknown_live_ids = set(_ALL_CARD_IDS - set(my_hand_ids) - out_of_play_ids)
    opponent_hand_size = int(observation.players_hand_sizes[opponent_index])
    deck_size = int(observation.deck_size)
    if len(unknown_live_ids) != opponent_hand_size + deck_size:
        raise ValueError(
            "Conteggio carte vive incoerente: "
            f"unknown_live={len(unknown_live_ids)}, attese={opponent_hand_size + deck_size}"
        )

    deck_forced_ids: list[int] = []
    opponent_pool = set(unknown_live_ids)
    if deck_size > 0 and trump_id in opponent_pool:
        # Nella Briscola 2-player la briscola scoperta resta nel mazzo e viene pescata per ultima.
        deck_forced_ids.append(trump_id)
        opponent_pool.remove(trump_id)
    if len(deck_forced_ids) > deck_size:
        raise ValueError("Deck size incoerente con la briscola pubblica")

    opponent_hand_ids = set(rng.sample(sorted(opponent_pool), opponent_hand_size))
    deck_rest_ids = list(opponent_pool - opponent_hand_ids)
    rng.shuffle(deck_rest_ids)
    if len(deck_rest_ids) != deck_size - len(deck_forced_ids):
        raise ValueError("Determinizzazione incoerente: dimensione deck errata")

    # `domain.step` pesca da `deck.pop()`: mettendo la briscola in testa la rendiamo l'ultima pescata.
    deck_ids = tuple(deck_forced_ids + deck_rest_ids)

    captured_ids = set(out_of_play_ids - table_ids)
    p0_captured_ids, p1_captured_ids = _captured_cards_for_scores(
        captured_ids=captured_ids,
        player0_points=int(observation.players_points[0]),
        player1_points=int(observation.players_points[1]),
    )

    players = [
        PlayerState(
            name="P0",
            hand=tuple(),
            captured_cards=tuple(id_to_card(card_id) for card_id in p0_captured_ids),
            points=int(observation.players_points[0]),
        ),
        PlayerState(
            name="P1",
            hand=tuple(),
            captured_cards=tuple(id_to_card(card_id) for card_id in p1_captured_ids),
            points=int(observation.players_points[1]),
        ),
    ]
    players[player_index] = PlayerState(
        name=observation.player_name,
        hand=observation.hand,
        captured_cards=players[player_index].captured_cards,
        points=players[player_index].points,
    )
    players[opponent_index] = PlayerState(
        name=f"P{opponent_index}",
        hand=tuple(id_to_card(card_id) for card_id in sorted(opponent_hand_ids)),
        captured_cards=players[opponent_index].captured_cards,
        points=players[opponent_index].points,
    )

    return GameState(
        num_players=2,
        is_team_game=False,
        teams=None,
        players=tuple(players),
        deck=tuple(id_to_card(card_id) for card_id in deck_ids),
        trump_card=trump_card,
        table_cards=observation.table_cards,
        current_turn=observation.current_turn,
        first_player=observation.first_player,
        game_over=False,
        winner_index=None,
        winning_team=None,
    )


def rollout_to_terminal(
    state: GameState,
    *,
    rollout_agent: Agent,
    rng: random.Random,
    use_endgame_solver: bool = True,
    metrics: PIMCSearchStats | None = None,
    max_steps: int = 128,
) -> GameState:
    """Completa una partita determinizzata con la policy di rollout."""
    cursor = state
    steps = 0
    while not cursor.game_over:
        if steps >= max_steps:
            raise RuntimeError("Rollout PIMC non terminato entro il limite di sicurezza")
        steps += 1

        observation = make_player_observation(cursor, cursor.current_turn)
        if use_endgame_solver and len(cursor.deck) == 0:
            try:
                card_index = solve_endgame(cursor).best_card_index
                if not 0 <= card_index < len(observation.hand):
                    card_index = _safe_agent_card_index(rollout_agent, observation, rng=rng, metrics=metrics)
            except ValueError:
                card_index = _safe_agent_card_index(rollout_agent, observation, rng=rng, metrics=metrics)
        else:
            card_index = _safe_agent_card_index(rollout_agent, observation, rng=rng, metrics=metrics)

        cursor, result = step(cursor, PlayCardAction(player_index=cursor.current_turn, card_index=card_index))
        if result.error:
            raise RuntimeError(f"Errore durante rollout PIMC: {result.error}")
    return cursor


@dataclass(frozen=True)
class PIMCAgent:
    """
    Agente PIMC con fallback e rollout configurabili.

    Il fallback viene usato quando lo stato è fuori scope, quando ci sono troppe carte ignote o quando
    una determinizzazione fallisce. Il rollout agent rappresenta la policy approssimata per entrambi i lati.
    """

    spec: ClassVar[AgentSpec] = AgentSpec(
        name="pimc",
        label="PIMC",
        description_it=(
            "Prototipo offline: campiona stati compatibili con l'informazione pubblica e usa una policy "
            "di rollout per scegliere nel finale. Non è esposto come agente UI di default."
        ),
    )

    rollout_agent: Agent = field(default_factory=HeuristicAgentV2)
    fallback: Agent = field(default_factory=HeuristicAgentV2)
    num_determinizations: int = 32
    max_unknown_cards: int = 10
    use_endgame_solver: bool = True
    name: str = "pimc"
    metrics: PIMCSearchStats = field(default_factory=PIMCSearchStats, repr=False, compare=False)

    def choose_card_index(self, observation: PlayerObservation, *, rng: random.Random) -> int:
        if not observation.hand:
            raise ValueError("Mano vuota: nessuna azione possibile")
        self.metrics.total_decisions += 1
        try:
            _validate_pimc_observation(observation)
        except ValueError:
            self.metrics.fallback_decisions += 1
            return _safe_agent_card_index(self.fallback, observation, rng=rng, metrics=self.metrics)

        if observation.deck_size == 0 and self.use_endgame_solver:
            try:
                card_index = solve_endgame(reconstruct_endgame_state(observation)).best_card_index
                if 0 <= card_index < len(observation.hand):
                    self.metrics.endgame_solver_decisions += 1
                    return card_index
            except ValueError:
                pass
            self.metrics.fallback_decisions += 1
            return _safe_agent_card_index(self.fallback, observation, rng=rng, metrics=self.metrics)

        if unknown_live_card_count(observation) > int(self.max_unknown_cards):
            self.metrics.fallback_decisions += 1
            return _safe_agent_card_index(self.fallback, observation, rng=rng, metrics=self.metrics)

        self.metrics.search_decisions += 1
        search_started = time.perf_counter()
        legal_indices = list(range(len(observation.hand)))
        scores = [0.0 for _ in legal_indices]
        counts = [0 for _ in legal_indices]

        determinizations = max(1, int(self.num_determinizations))
        try:
            for sample_index in range(determinizations):
                sample_rng = random.Random(rng.randrange(0, 2**32) ^ (sample_index * 0x9E3779B9))
                try:
                    sampled_state = determinize_observation(observation, rng=sample_rng)
                except ValueError:
                    self.metrics.failed_determinizations += 1
                    continue
                self.metrics.successful_determinizations += 1

                for local_pos, card_index in enumerate(legal_indices):
                    next_state, result = step(
                        sampled_state,
                        PlayCardAction(player_index=sampled_state.current_turn, card_index=card_index),
                    )
                    if result.error:
                        continue
                    rollout_rng = random.Random(sample_rng.randrange(0, 2**32) ^ (card_index * 0x85EBCA6B))
                    try:
                        final_state = rollout_to_terminal(
                            next_state,
                            rollout_agent=self.rollout_agent,
                            rng=rollout_rng,
                            use_endgame_solver=self.use_endgame_solver,
                            metrics=self.metrics,
                        )
                    except RuntimeError:
                        self.metrics.failed_rollouts += 1
                        continue
                    self.metrics.completed_rollouts += 1
                    player_points = final_state.players[observation.player_index].points
                    opponent_points = final_state.players[1 - observation.player_index].points
                    scores[local_pos] += float(player_points - opponent_points)
                    counts[local_pos] += 1
        finally:
            self.metrics.search_elapsed_seconds += time.perf_counter() - search_started

        if not any(counts):
            self.metrics.fallback_decisions += 1
            return _safe_agent_card_index(self.fallback, observation, rng=rng, metrics=self.metrics)

        best_pos = max(
            range(len(legal_indices)),
            key=lambda pos: (scores[pos] / counts[pos] if counts[pos] else float("-inf"), -legal_indices[pos]),
        )
        return legal_indices[best_pos]
