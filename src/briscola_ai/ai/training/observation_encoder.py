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
from typing import Literal

from ...domain.models import Card, Suit
from ...domain.observation import PlayerObservation
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

# Costanti di “contratto” (didattiche).
#
# L'encoder v1 costruisce un vettore di feature a dimensione fissa:
# - 6 blocchi da 40 (mano onehot/points/strength + tavolo onehot/points/strength) = 240
# - briscola onehot sui 4 semi = 4  -> 244
# - 4 scalari = 4                 -> 248
FEATURE_DIM_2P_V1 = 248
FEATURE_DIM_2P_V2 = FEATURE_DIM_2P_V1 + 40
ACTION_DIM = 40

EncoderVersion = Literal["v1", "v2"]


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


def _card_to_action_id_fast(card: Card) -> int:
    """
    Converte una `Card` dominio in action id senza passare da DTO/dict.

    Questo helper è usato nel path caldo training/evaluation. Mantiene la stessa convenzione
    dello spazio azioni canonico: `suit_index * 10 + (number - 1)`.
    """
    if card.suit is Suit.CLUBS:
        suit_index = 0
    elif card.suit is Suit.CUPS:
        suit_index = 1
    elif card.suit is Suit.COINS:
        suit_index = 2
    elif card.suit is Suit.SWORDS:
        suit_index = 3
    else:
        raise ValueError(f"Seme non supportato: {card.suit!r}")
    return suit_index * 10 + (int(card.rank.number) - 1)


def _one_hot_suit_from_card(card: Card | None) -> list[float]:
    """One-hot sui 4 semi partendo direttamente da una carta dominio."""
    out = [0.0, 0.0, 0.0, 0.0]
    if card is None:
        return out
    if card.suit is Suit.CLUBS:
        out[0] = 1.0
    elif card.suit is Suit.CUPS:
        out[1] = 1.0
    elif card.suit is Suit.COINS:
        out[2] = 1.0
    elif card.suit is Suit.SWORDS:
        out[3] = 1.0
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


def _seen_cards_onehot_to_floats(raw: object) -> list[float]:
    """
    Normalizza `seen_cards_onehot` in una lista di float lunga 40.

    Per compatibilità:
    - se il campo manca/None, ritorniamo 40 zeri (utile per dataset/modelli vecchi).
    - accettiamo `int`/`bool` e convertiamo in 0.0/1.0.
    """
    if raw is None:
        return [0.0] * 40

    if not isinstance(raw, list):
        raise ValueError("ObservationDTO invalida: seen_cards_onehot non list")

    if len(raw) != 40:
        raise ValueError(f"ObservationDTO invalida: seen_cards_onehot len={len(raw)} (atteso 40)")

    out: list[float] = []
    for v in raw:
        if isinstance(v, bool):
            out.append(1.0 if v else 0.0)
        elif isinstance(v, int):
            if v not in (0, 1):
                raise ValueError("ObservationDTO invalida: seen_cards_onehot deve contenere solo 0/1")
            out.append(float(v))
        else:
            raise ValueError("ObservationDTO invalida: seen_cards_onehot deve contenere int/bool")
    return out


def encode_observation_2p_v2(observation: dict) -> EncodedObservation:
    """
    Encoder 2-player v2 = v1 + storia pubblica (card counting lecito).

    Aggiungiamo in coda al vettore v1:
    - `seen_cards_onehot[40]`: 1 se la carta è già stata vista/giocata nella partita.

    Perché è anti-cheat?
    - Sono **solo** carte pubbliche: briscola scoperta + carte sul tavolo + carte già uscite.
    - Non include l'ordine del mazzo né la mano avversaria.

    Compatibilità:
    - Se `seen_cards_onehot` manca, viene trattato come 40 zeri (utile per dataset vecchi).
      Per training v2 “serio” è comunque consigliato che il backend lo popoli sempre.
    """
    base = encode_observation_2p(observation)
    seen = _seen_cards_onehot_to_floats(observation.get("seen_cards_onehot"))
    features = list(base.features) + seen
    return EncodedObservation(features=features, action_mask=base.action_mask)


def encode_observation_2p_with_version(observation: dict, *, version: EncoderVersion) -> EncodedObservation:
    """Selettore esplicito dell'encoder 2-player (v1/v2)."""
    if version == "v1":
        return encode_observation_2p(observation)
    if version == "v2":
        return encode_observation_2p_v2(observation)
    raise ValueError(f"Encoder version non supportata: {version!r}")


def feature_dim_for_encoder_version(version: EncoderVersion) -> int:
    """Ritorna la feature_dim attesa dall'encoder 2-player (v1/v2)."""
    return int(FEATURE_DIM_2P_V1) if version == "v1" else int(FEATURE_DIM_2P_V2)


def encode_player_observation_2p(
    observation: PlayerObservation, *, version: EncoderVersion = "v1"
) -> EncodedObservation:
    """
    Encoda una `PlayerObservation` (dominio) in feature+mask (2-player).

    Per coerenza con il training BC, convertiamo la `PlayerObservation` in un dict
    compatibile con `ObservationDTO` e poi riusiamo `encode_observation_2p`.

    Nota:
    - `PlayerObservation` non contiene informazione nascosta (anti-cheat).
    - Usiamo solo i campi necessari all'encoder (mano, tavolo, briscola, punti, deck_size).
    """
    if observation.num_players != 2:
        raise ValueError("Questo encoder è pensato per 2-player (num_players=2).")

    my_index = int(observation.player_index)
    if my_index < 0 or my_index >= observation.num_players:
        raise ValueError(f"player_index fuori range: {my_index} (num_players={observation.num_players})")

    # Action mask e feature mano. Questo duplica intenzionalmente la costruzione di
    # `encode_observation_2p`, evitando però la conversione PlayerObservation -> dict DTO
    # nel path caldo training/evaluation.
    mask = [False] * 40
    hand_onehot = [0.0] * 40
    for card in observation.hand:
        action_id = _card_to_action_id_fast(card)
        mask[action_id] = True
        hand_onehot[action_id] = 1.0

    hand_points = [hand_onehot[i] * float(_CARD_FEATURES.points_by_action_id[i]) for i in range(40)]
    hand_strength = [hand_onehot[i] * float(_CARD_FEATURES.strength_by_action_id[i]) for i in range(40)]

    table_onehot = [0.0] * 40
    for card, _ in observation.table_cards:
        table_onehot[_card_to_action_id_fast(card)] = 1.0

    table_points = [table_onehot[i] * float(_CARD_FEATURES.points_by_action_id[i]) for i in range(40)]
    table_strength = [table_onehot[i] * float(_CARD_FEATURES.strength_by_action_id[i]) for i in range(40)]

    if len(observation.players_points) != observation.num_players:
        raise ValueError("PlayerObservation invalida: players_points non coerente con num_players")

    opp_index = 1 - my_index
    is_second_in_trick = (
        1.0
        if (observation.current_turn == my_index and (not observation.game_over) and len(observation.table_cards) == 1)
        else 0.0
    )

    features: list[float] = (
        hand_onehot
        + hand_points
        + hand_strength
        + table_onehot
        + table_points
        + table_strength
        + _one_hot_suit_from_card(observation.trump_card)
        + [
            float(observation.deck_size) / 40.0,
            float(observation.players_points[my_index]) / 120.0,
            float(observation.players_points[opp_index]) / 120.0,
            is_second_in_trick,
        ]
    )

    if version == "v1":
        return EncodedObservation(features=features, action_mask=mask)
    if version == "v2":
        seen = _seen_cards_onehot_to_floats(list(observation.seen_cards_onehot))
        return EncodedObservation(features=features + seen, action_mask=mask)
    raise ValueError(f"Encoder version non supportata: {version!r}")
