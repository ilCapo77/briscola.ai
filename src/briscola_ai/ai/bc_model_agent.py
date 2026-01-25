"""
Agente che gioca usando un modello Behavior Cloning salvato in `.npz`.

Contesto didattico
------------------
Nel progetto alleniamo un primo modello supervisionato con `scripts/train_bc.py`,
usando uno spazio azioni fisso "40 carte + action mask".

Questo modulo integra quel modello come un `Agent` del progetto, così possiamo:
- valutarlo con `scripts/evaluate_agents.py` su seed suite riproducibili;
- confrontarlo con baseline (random, heuristic_v1, ...).

Anti-cheat
----------
L'agente riceve solo una `PlayerObservation` (vista parziale lecita) e NON ha
accesso allo stato completo (`GameState`, `deck`, mani avversarie).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..domain.observation import PlayerObservation
from .training.card_action_space import action_id_from_suit_number
from .training.observation_encoder import encode_player_observation_2p


@dataclass(frozen=True, slots=True)
class BCModel:
    """
    Modello BC lineare (softmax) salvato in `.npz`.

    Convenzione (vedi `scripts/train_bc.py`):
    - `w`: (D, 40) float32
    - `b`: (40,) float32
    - `metadata_json`: stringa JSON (opzionale, informativa)
    """

    w: np.ndarray
    b: np.ndarray
    metadata: dict[str, Any]

    @property
    def feature_dim(self) -> int:
        """Dimensione feature attesa dall'encoder."""
        return int(self.w.shape[0])


def load_bc_model_npz(path: Path) -> BCModel:
    """
    Carica un modello BC da `.npz`.

    Argomenti:
        path: path al file `.npz` salvato da `scripts/train_bc.py`.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Formato non supportato: {path} (atteso .npz)")

    with np.load(path) as data:
        if "w" not in data or "b" not in data:
            raise ValueError(f"File modello invalido: chiavi attese w/b, trovate: {list(data.keys())}")

        w = np.asarray(data["w"], dtype=np.float32)
        b = np.asarray(data["b"], dtype=np.float32)
        if w.ndim != 2 or b.ndim != 1:
            raise ValueError(f"Shape invalide: w.ndim={w.ndim} b.ndim={b.ndim} (attesi 2 e 1)")
        if w.shape[1] != 40 or b.shape[0] != 40:
            raise ValueError(f"Action dim invalida: w={w.shape} b={b.shape} (atteso (*,40) e (40,))")

        metadata: dict[str, Any] = {}
        if "metadata_json" in data:
            raw = data["metadata_json"]
            # `np.savez` salva le stringhe come array 0-d: usiamo `.item()` per ottenere la str.
            try:
                text = str(raw.item())
            except Exception:
                text = str(raw)
            try:
                metadata = json.loads(text) if text else {}
            except json.JSONDecodeError:
                metadata = {"raw_metadata_json": text}

        # Sanity: se metadata dichiara una dimensione, facciamo un check esplicito.
        declared_dim = metadata.get("feature_dim")
        if isinstance(declared_dim, int) and declared_dim != int(w.shape[0]):
            raise ValueError(f"Feature dim mismatch: metadata={declared_dim} w.shape[0]={w.shape[0]}")

        return BCModel(w=w, b=b, metadata=metadata)


@dataclass(frozen=True, slots=True)
class BCModelAgent:
    """
    Agente che seleziona la carta con logit massimo tra quelle in mano.

    Implementazione:
    1) `PlayerObservation` -> `EncodedObservation` (feature + action_mask)
    2) logits = xW + b
    3) mask: azioni non valide -> logit molto negativo
    4) argmax -> action_id (0..39)
    5) action_id -> indice nella mano corrente (card_index)
    """

    model: BCModel
    model_path: Path

    @property
    def name(self) -> str:
        """Nome leggibile dell'agente (includiamo solo il basename per evitare path lunghi)."""
        return f"bc_model({self.model_path.name})"

    @classmethod
    def from_npz(cls, path: str | Path) -> BCModelAgent:
        """Costruisce un agente caricando un `.npz` (output di `scripts/train_bc.py`)."""
        model_path = Path(path)
        return cls(model=load_bc_model_npz(model_path), model_path=model_path)

    def choose_card_index(self, observation: PlayerObservation, *, rng: random.Random) -> int:
        hand = observation.hand
        if not hand:
            raise ValueError("Mano vuota: nessuna azione possibile")

        encoded = encode_player_observation_2p(observation)
        if len(encoded.features) != self.model.feature_dim:
            raise ValueError(
                "Feature dim mismatch: "
                f"encoder={len(encoded.features)} model={self.model.feature_dim} ({self.model_path})"
            )

        x = np.asarray(encoded.features, dtype=np.float32)
        logits = x @ self.model.w + self.model.b  # (40,)

        mask = np.asarray(encoded.action_mask, dtype=bool)
        if mask.shape != (40,):
            raise ValueError(f"Mask shape invalida: {mask.shape} (attesa (40,))")
        if not bool(np.any(mask)):
            raise ValueError("Action mask vuota: nessuna azione valida")

        very_negative = -1e9
        masked_logits = logits.copy()
        masked_logits[~mask] = very_negative

        action_id = int(np.argmax(masked_logits))

        # Convertiamo action_id -> card_index nella mano corrente.
        for i, card in enumerate(hand):
            cid = action_id_from_suit_number(suit=card.suit.value, number=card.rank.number)
            if cid == action_id:
                return i

        # Fallback difensivo: se per qualche motivo non troviamo la carta corrispondente,
        # scegliamo una carta valida in mano in modo riproducibile.
        valid: list[int] = []
        for i, card in enumerate(hand):
            cid = action_id_from_suit_number(suit=card.suit.value, number=card.rank.number)
            if bool(mask[cid]):
                valid.append(i)
        if not valid:
            return rng.randrange(len(hand))
        return valid[rng.randrange(len(valid))]
