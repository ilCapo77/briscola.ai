"""
Encoder diretto da `Fast2PState` a feature/mask 2-player.

Il layout delle feature è identico a `encode_player_observation_2p`, ma evita la costruzione
di `Card`, `PlayerObservation` e DTO. È il pezzo che permette a un rollout neurale di usare
`fast_2p` senza tornare al dominio canonico nel loop caldo.
"""

from __future__ import annotations

from .fast_2p import CARD_POINTS, CARD_STRENGTH, CARD_SUIT, Fast2PState
from .training.observation_encoder import ACTION_DIM, EncodedObservation, EncoderVersion


def _seen_cards_onehot_to_floats(raw: tuple[int, ...]) -> list[float]:
    """Normalizza `seen_cards_onehot` in float 0/1, mantenendo il contratto di lunghezza 40."""
    if len(raw) != ACTION_DIM:
        raise ValueError(f"seen_cards_onehot len={len(raw)} (atteso {ACTION_DIM})")
    out: list[float] = []
    for value in raw:
        if value not in (0, 1):
            raise ValueError("seen_cards_onehot deve contenere solo 0/1")
        out.append(float(value))
    return out


def _one_hot_trump_suit(trump_card: int) -> list[float]:
    """One-hot sui 4 semi della briscola scoperta."""
    out = [0.0, 0.0, 0.0, 0.0]
    out[CARD_SUIT[trump_card]] = 1.0
    return out


def encode_fast_observation_2p(
    state: Fast2PState,
    *,
    player_index: int,
    seen_cards_onehot: tuple[int, ...],
    version: EncoderVersion = "v1",
) -> EncodedObservation:
    """
    Encoda lo stato fast dal punto di vista di `player_index`.

    `seen_cards_onehot` è fornito dal rollout perché `Fast2PState` non conserva le prese storiche.
    Nel loop fast lo aggiorniamo marcando la briscola iniziale e ogni carta giocata.
    """
    if player_index not in (0, 1):
        raise ValueError(f"player_index fuori range: {player_index}")

    mask = [False] * ACTION_DIM
    hand_onehot = [0.0] * ACTION_DIM
    for card_id in state.hands[player_index]:
        mask[card_id] = True
        hand_onehot[card_id] = 1.0

    hand_points = [hand_onehot[i] * float(CARD_POINTS[i]) for i in range(ACTION_DIM)]
    hand_strength = [hand_onehot[i] * float(CARD_STRENGTH[i]) for i in range(ACTION_DIM)]

    table_onehot = [0.0] * ACTION_DIM
    for card_id in state.table_cards:
        table_onehot[card_id] = 1.0

    table_points = [table_onehot[i] * float(CARD_POINTS[i]) for i in range(ACTION_DIM)]
    table_strength = [table_onehot[i] * float(CARD_STRENGTH[i]) for i in range(ACTION_DIM)]

    opp_index = 1 - player_index
    is_second_in_trick = (
        1.0 if (state.current_turn == player_index and (not state.game_over) and len(state.table_cards) == 1) else 0.0
    )

    features: list[float] = (
        hand_onehot
        + hand_points
        + hand_strength
        + table_onehot
        + table_points
        + table_strength
        + _one_hot_trump_suit(state.trump_card)
        + [
            float(len(state.deck)) / 40.0,
            float(state.points[player_index]) / 120.0,
            float(state.points[opp_index]) / 120.0,
            is_second_in_trick,
        ]
    )

    if version == "v1":
        return EncodedObservation(features=features, action_mask=mask)
    if version == "v2":
        return EncodedObservation(features=features + _seen_cards_onehot_to_floats(seen_cards_onehot), action_mask=mask)
    if version == "v3":
        # Guard esplicito (domain-first): v3 cambia la semantica delle feature e per ora vive solo
        # sul path domain. Falliamo invece di ripiegare su v2, che produrrebbe un encoding errato.
        raise ValueError("Encoder v3 non supportato sul path fast: usa l'engine domain (parità fast/numba TODO).")
    raise ValueError(f"Encoder version non supportata: {version!r}")
