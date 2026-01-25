"""
Test per integrazione del modello BC come agente.

Obiettivo didattico:
- verificare che l'agente scelga una carta valida in mano
- verificare che la action mask impedisca selezioni "impossibili"
- garantire che il caricamento `.npz` sia robusto (shape/metadata)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from briscola_ai.ai.bc_model_agent import BCModelAgent, load_bc_model_npz
from briscola_ai.ai.training.card_action_space import action_id_from_suit_number
from briscola_ai.ai.training.observation_encoder import encode_player_observation_2p
from briscola_ai.domain.models import Card, Rank, Suit
from briscola_ai.domain.observation import PlayerObservation


def _make_2p_observation(*, hand: tuple[Card, ...]) -> PlayerObservation:
    """Crea una `PlayerObservation` minimale (2-player) per test."""
    return PlayerObservation(
        num_players=2,
        is_team_game=False,
        teams=None,
        player_index=0,
        player_name="A",
        hand=hand,
        trump_card=Card(Suit.CUPS, Rank.ACE),
        deck_size=20,
        table_cards=tuple(),
        current_turn=0,
        first_player=0,
        game_over=False,
        winner_index=None,
        winning_team=None,
        players_points=(0, 0),
        players_hand_sizes=(len(hand), len(hand)),
    )


def test_bc_model_agent_picks_valid_card_and_respects_mask(tmp_path: Path) -> None:
    """
    L'agente deve scegliere una carta in mano anche se il modello "preferirebbe" una carta non valida.

    Nota:
    Inseriamo un bias enorme su un'azione NON presente in mano per verificare che la mask la azzeri.
    """
    hand = (
        Card(Suit.CUPS, Rank.THREE),  # in mano
        Card(Suit.CLUBS, Rank.TWO),  # in mano
    )
    obs = _make_2p_observation(hand=hand)
    encoded = encode_player_observation_2p(obs)
    d = len(encoded.features)

    # Costruiamo un modello che preferisce la prima carta in mano (CUPS 3).
    action_a = action_id_from_suit_number(suit=hand[0].suit.value, number=hand[0].rank.number)
    action_b = action_id_from_suit_number(suit=hand[1].suit.value, number=hand[1].rank.number)

    w = np.zeros((d, 40), dtype=np.float32)
    b = np.zeros((40,), dtype=np.float32)

    # Le prime 40 feature sono `my_hand_onehot`, quindi:
    # - feature[action_id] = 1 se quella carta è in mano
    w[action_a, action_a] = 2.0
    w[action_b, action_b] = 1.0

    # Bias enorme su una carta non in mano: la mask deve comunque bloccarla.
    invalid_action = action_id_from_suit_number(suit="coins", number=1)
    assert invalid_action not in (action_a, action_b)
    b[invalid_action] = 10_000.0

    model_path = tmp_path / "bc_model.npz"
    np.savez(model_path, w=w, b=b, metadata_json=f'{{"format":"linear_softmax_bc_v1","feature_dim":{d}}}')

    agent = BCModelAgent.from_npz(model_path)
    idx = agent.choose_card_index(obs, rng=random.Random(0))

    assert 0 <= idx < len(hand)
    assert hand[idx] == hand[0]


def test_load_bc_model_npz_validates_metadata_feature_dim(tmp_path: Path) -> None:
    """Se `metadata_json.feature_dim` non coincide con `w.shape[0]`, il loader deve fallire."""
    w = np.zeros((7, 40), dtype=np.float32)
    b = np.zeros((40,), dtype=np.float32)
    model_path = tmp_path / "bad_model.npz"
    np.savez(model_path, w=w, b=b, metadata_json='{"feature_dim":8}')

    with pytest.raises(ValueError):
        load_bc_model_npz(model_path)


def test_bc_model_agent_supports_mlp_format(tmp_path: Path) -> None:
    """L'agente deve supportare anche il formato MLP (w1/b1/w2/b2) e rispettare la mask."""
    hand = (
        Card(Suit.CUPS, Rank.THREE),
        Card(Suit.CLUBS, Rank.TWO),
    )
    obs = _make_2p_observation(hand=hand)
    encoded = encode_player_observation_2p(obs)
    d = len(encoded.features)

    action_a = action_id_from_suit_number(suit=hand[0].suit.value, number=hand[0].rank.number)
    action_b = action_id_from_suit_number(suit=hand[1].suit.value, number=hand[1].rank.number)

    hidden_dim = 2
    w1 = np.zeros((d, hidden_dim), dtype=np.float32)
    b1 = np.zeros((hidden_dim,), dtype=np.float32)
    w2 = np.zeros((hidden_dim, 40), dtype=np.float32)
    b2 = np.zeros((40,), dtype=np.float32)

    # Proiettiamo due feature della mano su due unità hidden (ReLU pass-through perché x>=0).
    w1[action_a, 0] = 1.0
    w1[action_b, 1] = 1.0
    # Poi preferiamo action_a rispetto ad action_b.
    w2[0, action_a] = 2.0
    w2[1, action_b] = 1.0

    invalid_action = action_id_from_suit_number(suit="coins", number=1)
    assert invalid_action not in (action_a, action_b)
    b2[invalid_action] = 10_000.0

    model_path = tmp_path / "bc_model_mlp.npz"
    np.savez(
        model_path,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        metadata_json=f'{{"format":"mlp_bc_v1","feature_dim":{d},"hidden_dim":{hidden_dim}}}',
    )

    agent = BCModelAgent.from_npz(model_path)
    idx = agent.choose_card_index(obs, rng=random.Random(0))
    assert idx == 0
