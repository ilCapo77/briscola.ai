"""
Script di riproduzione: serializzazione JSON di `trick_cards`.

Scopo:
- simulare la struttura inviata dal backend al frontend quando una presa termina
- verificare che `GameJSONEncoder` serializzi correttamente `Card`, `Suit`, `Rank`

Uso:
  python repro_payload.py
"""

import json

from briscola_ai.backend.server import GameJSONEncoder
from briscola_ai.game.models import Card, Rank, Suit


def main() -> None:
    """Esegue la simulazione e stampa il payload in JSON (pretty-printed)."""
    # Simulate the `trick_cards` structure: list of [Card, player_index].
    card1 = Card(Suit.CUPS, Rank.ACE)
    card2 = Card(Suit.SWORDS, Rank.THREE)

    table_cards = [(card1, 0), (card2, 1)]

    payload = {
        "trick_completed": True,
        "trick_cards": table_cards,
    }

    print(json.dumps(payload, cls=GameJSONEncoder, indent=2))


if __name__ == "__main__":
    main()
