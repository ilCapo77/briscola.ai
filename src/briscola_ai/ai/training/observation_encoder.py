"""
Encoder didattico: ObservationDTO -> feature vector + action mask (40 carte).

Contesto
--------
Il backend produce `ObservationDTO` (vedi `backend/dto.py`) che è già una vista
parziale e lecita (anti-cheat). Per un primo modello supervisionato (Behavior
Cloning) vogliamo trasformare questa observation in:
- un vettore di feature numeriche `x`
- una action mask `m` lunga 40 che limita le azioni alle carte realmente in mano

Nota:
Questo encoder è volutamente "semplice ma utile" e pensato per 2-player.
In 4-player funziona parzialmente, ma molte feature (es. avversario unico)
vanno ripensate per il team-play.
"""

from __future__ import annotations

from dataclasses import dataclass

from .card_action_space import build_card_features, card_dto_to_action_id


@dataclass(frozen=True, slots=True)
class EncodedObservation:
    """
    Output dell'encoder.

    - `features`: lista di float (dimensione fissa)
    - `action_mask`: lista di bool (dimensione 40)
    """

    features: list[float]
    action_mask: list[bool]


_CARD_FEATURES = build_card_features()


def _one_hot_suit(suit: str | None) -> list[float]:
    """One-hot sui 4 semi (clubs, cups, coins, swords)."""
    out = [0.0, 0.0, 0.0, 0.0]
    if suit is None:
        return out
    idx = {"clubs": 0, "cups": 1, "coins": 2, "swords": 3}.get(suit)
    if idx is None:
        return out
    out[idx] = 1.0
    return out


def encode_observation_2p(observation: dict) -> EncodedObservation:
    """
    Encoda una `ObservationDTO` (come dict JSON) in feature+mask.

    Feature (ordine, didattico):
    - my_hand_onehot[40]
    - my_hand_points[40]     (onehot * punti carta)
    - my_hand_strength[40]   (onehot * forza carta)
    - table_onehot[40]
    - table_points[40]
    - table_strength[40]
    - trump_suit_onehot[4]
    - scalari: deck_size/40, my_points/120, opp_points/120, is_second_in_trick
    """
    if not isinstance(observation, dict):
        raise TypeError("ObservationDTO attesa come dict")

    if observation.get("num_players") != 2:
        raise ValueError("Questo encoder è pensato per 2-player (num_players=2).")

    my_index = observation.get("my_index")
    if not isinstance(my_index, int):
        raise ValueError("ObservationDTO invalida: my_index mancante/non int")

    my_hand = observation.get("my_hand") or []
    if not isinstance(my_hand, list):
        raise ValueError("ObservationDTO invalida: my_hand non list")

    # Action mask e feature della mano (40 carte).
    mask = [False] * 40
    hand_onehot = [0.0] * 40
    for card in my_hand:
        action_id = card_dto_to_action_id(card)
        mask[action_id] = True
        hand_onehot[action_id] = 1.0

    hand_points = [hand_onehot[i] * float(_CARD_FEATURES.points_by_action_id[i]) for i in range(40)]
    hand_strength = [hand_onehot[i] * float(_CARD_FEATURES.strength_by_action_id[i]) for i in range(40)]

    # Tavolo: carte pubbliche (al massimo 1 prima della nostra azione, in 2-player).
    table_cards = observation.get("table_cards") or []
    if not isinstance(table_cards, list):
        raise ValueError("ObservationDTO invalida: table_cards non list")

    table_onehot = [0.0] * 40
    for item in table_cards:
        if not isinstance(item, dict):
            continue
        card = item.get("card")
        if not isinstance(card, dict):
            continue
        action_id = card_dto_to_action_id(card)
        table_onehot[action_id] = 1.0

    table_points = [table_onehot[i] * float(_CARD_FEATURES.points_by_action_id[i]) for i in range(40)]
    table_strength = [table_onehot[i] * float(_CARD_FEATURES.strength_by_action_id[i]) for i in range(40)]

    # Trump suit (pubblico).
    trump_suit = observation.get("trump_suit")
    trump_onehot = _one_hot_suit(trump_suit if isinstance(trump_suit, str) else None)

    # Scalari "stato partita".
    deck_size = observation.get("cards_remaining_in_deck")
    my_points = observation.get("my_points")
    if not isinstance(deck_size, int) or not isinstance(my_points, int):
        raise ValueError("ObservationDTO invalida: cards_remaining_in_deck/my_points mancanti")

    players = observation.get("players") or []
    opp_points = 0
    if isinstance(players, list):
        for p in players:
            if isinstance(p, dict) and p.get("index") != my_index:
                opp_points = int(p.get("points", 0))
                break

    is_second_in_trick = 1.0 if (observation.get("my_turn") is True and len(table_cards) == 1) else 0.0

    features: list[float] = (
        hand_onehot
        + hand_points
        + hand_strength
        + table_onehot
        + table_points
        + table_strength
        + trump_onehot
        + [
            float(deck_size) / 40.0,
            float(my_points) / 120.0,
            float(opp_points) / 120.0,
            is_second_in_trick,
        ]
    )

    return EncodedObservation(features=features, action_mask=mask)
