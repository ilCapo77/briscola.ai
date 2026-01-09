#!/usr/bin/env python3
"""
Simulatore CLI: esegue N partite senza UI (self-play casuale).

Uso:
  python scripts/simulate_games.py --num-games 100 --seed 42
  python scripts/simulate_games.py --num-games 50 --seed 1 --num-players 4
"""

from __future__ import annotations

import argparse
import random

from briscola_ai.game.game import BriscolaGame


def simulate_one_game(num_players: int) -> dict:
    """
    Simula una singola partita (self-play casuale).

    Argomenti:
        num_players: 2 o 4

    Ritorna:
        Un dizionario con:
        - `status`: "ok" oppure "error"
        - `result`: (solo se ok) output di `BriscolaGame.get_game_result()`
        - `error`: (solo se error) descrizione testuale del problema
    """
    game = BriscolaGame(num_players=num_players)
    game.start_game()

    safety = 5000
    while not game.game_over and safety > 0:
        safety -= 1
        valid = game.get_valid_actions()
        if not valid:
            return {"status": "error", "error": "Nessuna azione valida", "num_players": num_players}
        card_index = random.choice(valid)
        game.play_action(card_index)

    if safety <= 0:
        return {"status": "error", "error": "Loop di sicurezza", "num_players": num_players}

    result = game.get_game_result()
    return {"status": "ok", "result": result, "num_players": num_players}


def main() -> int:
    """Entry point CLI: simula N partite e stampa un riepilogo dei vincitori."""
    parser = argparse.ArgumentParser(description="Simula partite di Briscola senza UI (self-play casuale)")
    parser.add_argument("--num-games", type=int, default=10, help="Numero di partite da simulare")
    parser.add_argument("--seed", type=int, default=0, help="Seed RNG (per riproducibilità)")
    parser.add_argument("--num-players", type=int, default=2, choices=[2, 4], help="Numero di giocatori (2 o 4)")
    args = parser.parse_args()

    random.seed(args.seed)

    ok = 0
    errors = 0
    winners = {}

    for _ in range(args.num_games):
        out = simulate_one_game(args.num_players)
        if out["status"] != "ok":
            errors += 1
            continue
        ok += 1
        winner = out["result"].get("winner", "Sconosciuto")
        winners[winner] = winners.get(winner, 0) + 1

    print(f"Simulazioni: {ok} OK, {errors} errori (seed={args.seed}, players={args.num_players})")
    for winner, count in sorted(winners.items(), key=lambda x: x[1], reverse=True):
        print(f"- {winner}: {count}")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
