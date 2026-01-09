
import json
from src.briscola_ai.game.models import Card, Suit, Rank
from src.briscola_ai.backend.server import GameJSONEncoder

# Simulate the trick_cards structure
card1 = Card(Suit.CUPS, Rank.ACE)
card2 = Card(Suit.SWORDS, Rank.THREE)

table_cards = [(card1, 0), (card2, 1)]

payload = {
    "trick_completed": True,
    "trick_cards": table_cards
}

print(json.dumps(payload, cls=GameJSONEncoder, indent=2))
