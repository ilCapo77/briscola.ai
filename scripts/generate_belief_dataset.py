#!/usr/bin/env python3
"""
Genera il dataset per la belief network (Fase 2): osservazione -> mano avversaria vera.

Come funziona
-------------
Self-play col motore di dominio (serve la `trick_history` dell'encoder v4): a ogni decisione
del giocatore di turno, se lo stato è nella finestra utile (mazzo non vuoto e carte ignote
entro soglia: è dove le determinizzazioni PIMC/lookahead campionano), registriamo:

- `x`: feature encoder v4 dell'osservazione LECITA del giocatore di turno (float16);
- `y`: one-hot (40) della mano VERA dell'avversario — usata SOLO come label di training;
- `unknown`: one-hot (40) delle carte ignote all'osservatore (la loss è mascherata qui);
- `opp_hand_size`, `game_index`: per baseline uniforme e split train/val per-partita.

Anti-cheat: il full-state è letto esclusivamente per la label; l'input resta l'osservazione.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from briscola_ai.ai.agents import build_agent
from briscola_ai.ai.agents.pimc import unknown_live_card_count
from briscola_ai.ai.encoding.observation_encoder import FEATURE_DIM_2P_V4, encode_player_observation_2p
from briscola_ai.domain.card_id import card_to_id
from briscola_ai.domain.engine import PlayCardAction, step
from briscola_ai.domain.observation import make_player_observation
from briscola_ai.domain.state import new_game_state
from briscola_ai.versioning import get_code_version


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera dataset belief (osservazione -> mano avversaria)")
    parser.add_argument("--out", required=True, help="Path .npz di output")
    parser.add_argument("--num-games", type=int, default=20000, help="Partite di self-play (default 20000)")
    parser.add_argument("--seed", type=int, default=0, help="Seed RNG (riproducibilita')")
    parser.add_argument(
        "--policy-model",
        default="data/models/best_a2c_v14.npz",
        help="Modello .npz che gioca entrambi i lati (default: best_a2c_v14)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Probabilita' di mossa casuale (diversita' degli stati; default 0.05)",
    )
    parser.add_argument(
        "--max-unknown-cards",
        type=int,
        default=10,
        help="Registra solo stati con carte vive ignote <= soglia (la finestra della search; 0 = nessun filtro)",
    )
    parser.add_argument("--log-every", type=int, default=5000, help="Log di avanzamento ogni N partite")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    agent0 = build_agent("bc_model", model_path=args.policy_model)
    agent1 = build_agent("bc_model", model_path=args.policy_model)
    agents = (agent0, agent1)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    unknowns: list[np.ndarray] = []
    opp_hand_sizes: list[int] = []
    game_indexes: list[int] = []

    started = time.perf_counter()
    for game_index in range(int(args.num_games)):
        state = new_game_state(2, seed=rng.randrange(0, 2**32))
        action_rng = random.Random(rng.randrange(0, 2**32))

        while not state.game_over:
            current = state.current_turn
            observation = make_player_observation(state, current)

            in_window = state.deck and (
                args.max_unknown_cards <= 0 or unknown_live_card_count(observation) <= args.max_unknown_cards
            )
            if in_window:
                encoded = encode_player_observation_2p(observation, version="v4")
                x = np.asarray(encoded.features, dtype=np.float16)

                opponent_hand = state.players[1 - current].hand
                y = np.zeros(40, dtype=np.int8)
                for card in opponent_hand:
                    y[card_to_id(card)] = 1

                unknown = np.ones(40, dtype=np.int8)
                for card in observation.hand:
                    unknown[card_to_id(card)] = 0
                for card_id, flag in enumerate(observation.out_of_play_cards_onehot):
                    if flag:
                        unknown[card_id] = 0

                # Invarianti di lecita' del dataset: la label vive DENTRO le ignote.
                if int((y & (1 - unknown)).sum()) != 0:
                    raise RuntimeError("Label incoerente: carta avversaria non ignota all'osservatore")
                if int(y.sum()) != len(opponent_hand):
                    raise RuntimeError("Label incoerente: cardinalita' mano avversaria")

                xs.append(x)
                ys.append(y)
                unknowns.append(unknown)
                opp_hand_sizes.append(len(opponent_hand))
                game_indexes.append(game_index)

            hand_size = len(state.players[current].hand)
            if action_rng.random() < float(args.epsilon):
                card_index = action_rng.randrange(hand_size)
            else:
                card_index = agents[current].choose_card_index(observation, rng=action_rng)
            state, result = step(state, PlayCardAction(player_index=current, card_index=card_index))
            if result.error:
                raise RuntimeError(f"Errore dominio in self-play: {result.error}")

        if args.log_every > 0 and (game_index + 1) % args.log_every == 0:
            elapsed = time.perf_counter() - started
            print(f"partite {game_index + 1}/{args.num_games} | record {len(xs)} | {elapsed:.1f}s")

    metadata = {
        "format": "belief_dataset_v1",
        "encoder_version": "v4",
        "feature_dim": int(FEATURE_DIM_2P_V4),
        "policy_model": str(args.policy_model),
        "epsilon": float(args.epsilon),
        "max_unknown_cards": int(args.max_unknown_cards),
        "num_games": int(args.num_games),
        "num_records": len(xs),
        "seed": int(args.seed),
        "code_version": get_code_version(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        x=np.stack(xs) if xs else np.zeros((0, FEATURE_DIM_2P_V4), dtype=np.float16),
        y=np.stack(ys) if ys else np.zeros((0, 40), dtype=np.int8),
        unknown=np.stack(unknowns) if unknowns else np.zeros((0, 40), dtype=np.int8),
        opp_hand_size=np.asarray(opp_hand_sizes, dtype=np.int8),
        game_index=np.asarray(game_indexes, dtype=np.int32),
        metadata_json=json.dumps(metadata),
    )
    elapsed = time.perf_counter() - started
    print(f"Salvato {out_path} | record={len(xs)} | partite={args.num_games} | {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
