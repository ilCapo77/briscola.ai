"""
Motore di gioco della Briscola.

Questo modulo contiene `BriscolaGame`, una classe “stateful” che gestisce:
- creazione e mescolamento del mazzo
- distribuzione delle carte (2 giocatori con pescata dal mazzo; 4 giocatori a mazzo completo)
- turno di gioco, carte sul tavolo (`table_cards`) e prese
- calcolo del vincitore della presa e accumulo dei punti

Nota didattica:
in una fase successiva del refactor potremo trasformare questa implementazione in un
motore più “funzionale” (es. `GameState` + `step(state, action)`), che è spesso più comodo
per il reinforcement learning e per la riproducibilità. Per ora manteniamo la struttura
semplice e leggibile.
"""

import random
from typing import Dict, List, Optional, Tuple, Union

from .models import Card, Player, Rank, Suit


class BriscolaGame:
    """
    Classe principale che implementa il gioco della Briscola.
    Supporta sia la modalità a 2 giocatori sia quella a 4 giocatori (a squadre).
    Progettata per essere compatibile con workflow di addestramento ML.

    Attributi principali (stato):
    - `players`: lista di `Player` (mano, prese, punti)
    - `deck`: mazzo rimanente (in 2 giocatori si pesca; in 4 viene svuotato in distribuzione)
    - `trump_card`: carta di briscola (definisce il seme di briscola)
    - `table_cards`: carte giocate nella presa corrente come lista di `(Card, player_index)`
    - `current_turn`: indice del giocatore che deve giocare ora
    - `first_player`: indice del giocatore che ha aperto la presa corrente
    - `game_over` / `winner`: fine partita e vincitore (singolo o squadra)
    """

    def __init__(self, num_players: int, player_names: Optional[List[str]] = None):
        """
        Inizializza una partita di Briscola.

        Argomenti:
            num_players: Numero di giocatori (2 o 4)
            player_names: Lista opzionale dei nomi dei giocatori
        """
        if num_players not in [2, 4]:
            raise ValueError("La Briscola supporta solo 2 o 4 giocatori")

        self.num_players = num_players
        self.is_team_game = num_players == 4

        # Genera nomi di default se non forniti
        if player_names is None:
            player_names = [f"Giocatore {i + 1}" for i in range(num_players)]
        elif len(player_names) != num_players:
            raise ValueError(f"Attesi {num_players} nomi giocatore, ottenuti {len(player_names)}")

        self.players = [Player(name) for name in player_names]

        # Configurazione squadre per la partita a 4
        self.teams: Optional[List[Tuple[int, int]]] = [(0, 2), (1, 3)] if self.is_team_game else None

        # Stato della partita
        self.deck: List[Card] = []
        self.trump_card: Optional[Card] = None
        self.table_cards: List[Tuple[Card, int]] = []  # (carta, indice_giocatore)
        self.current_turn = 0
        self.first_player = 0  # Chi gioca per primo nella presa
        self.game_over = False
        self.winner: Optional[Union[Player, Tuple[Player, Player]]] = None  # Giocatore singolo o squadra

        self._create_deck()

    def _create_deck(self) -> None:
        """
        Crea un mazzo completo di 40 carte italiane.

        Il mazzo è il prodotto cartesiano di tutti i semi (`Suit`) e tutti i ranghi (`Rank`).
        L'ordine iniziale non è importante perché verrà poi mescolato da `shuffle_deck()`.
        """
        self.deck = []
        for suit in Suit:
            for rank in Rank:
                self.deck.append(Card(suit, rank))

    def shuffle_deck(self) -> None:
        """
        Mischia il mazzo in modo casuale.

        Nota: questa funzione usa il RNG globale di `random`. Per riproducibilità in ambito ML,
        in futuro potrebbe essere utile iniettare un `random.Random(seed)` dedicato.
        """
        random.shuffle(self.deck)

    def deal_cards(self) -> None:
        """
        Distribuisce le carte in base alla modalità di gioco.

        - 2 giocatori: 3 carte a testa + 1 briscola scoperta; poi si pesca dal mazzo dopo ogni presa.
        - 4 giocatori: 10 carte a testa (mazzo completo); la briscola viene fissata dall'ultima carta
          dell'ultimo giocatore (scelta implementativa coerente con la tradizione “ultima carta distribuita”).
        """
        if self.is_team_game:
            # 4 giocatori: distribuisce tutte le 40 carte (10 a testa), senza briscola scoperta
            if len(self.deck) != 40:
                raise ValueError("Il mazzo deve avere 40 carte per la partita a 4 giocatori")

            # Distribuisce 10 carte a ciascun giocatore
            for _ in range(10):
                for player in self.players:
                    player.add_card(self.deck.pop())

            # L'ultima carta dell'ultimo giocatore diventa la briscola
            if self.players[-1].hand:
                self.trump_card = self.players[-1].hand[-1]
            else:
                raise ValueError("Impossibile impostare la briscola")
        else:
            # 2 giocatori: distribuisce 3 carte a testa, briscola scoperta
            if len(self.deck) < 7:  # 3+3+1 for trump
                raise ValueError("Carte insufficienti nel mazzo per distribuire")

            # Distribuisce 3 carte a ciascun giocatore
            for _ in range(3):
                for player in self.players:
                    player.add_card(self.deck.pop())

            # Scopre la briscola e la mette in fondo al mazzo
            self.trump_card = self.deck.pop()
            self.deck.insert(0, self.trump_card)

    def start_game(self) -> None:
        """
        Inizializza e avvia una nuova partita.

        Effetti:
        - reimposta i giocatori (mano, prese, punti)
        - reimposta lo stato del turno e del tavolo
        - ricrea/mescola il mazzo e distribuisce le carte
        """
        # Reimposta tutti i giocatori
        for player in self.players:
            player.reset()

        # Reimposta lo stato della partita
        self.table_cards.clear()
        self.game_over = False
        self.winner = None
        self.current_turn = 0
        self.first_player = 0

        # Imposta una nuova partita
        self._create_deck()
        self.shuffle_deck()
        self.deal_cards()

    def get_team_points(self, team_index: int) -> int:
        """Restituisce i punti totali di una squadra (solo modalità a 4)"""
        if not self.is_team_game:
            raise ValueError("Le squadre esistono solo nella modalità a 4 giocatori")

        if team_index not in [0, 1]:
            raise ValueError("L'indice squadra deve essere 0 o 1")

        if self.teams is None:
            raise ValueError("Squadre non inizializzate")

        team_players = self.teams[team_index]
        return sum(self.players[player_idx].points for player_idx in team_players)

    def get_player_team(self, player_index: int) -> Optional[int]:
        """Restituisce l'indice della squadra di un giocatore (None nella modalità a 2)"""
        if not self.is_team_game or self.teams is None:
            return None

        for team_idx, team_players in enumerate(self.teams):
            if player_index in team_players:
                return team_idx
        return None

    def get_game_state(self) -> Dict:
        """
        Restituisce lo stato completo osservabile per agenti IA.
        Adatta il formato sia alla modalità a 2 sia a quella a 4.
        """
        state: Dict[str, object] = {
            "num_players": self.num_players,
            "is_team_game": self.is_team_game,
            "trump_card": self.trump_card,
            "trump_suit": self.trump_card.suit if self.trump_card else None,
            "table_cards": self.table_cards.copy(),
            "current_turn": self.current_turn,
            "first_player": self.first_player,
            "cards_remaining_in_deck": len(self.deck),
            "game_over": self.game_over,
            "valid_actions": self.get_valid_actions(),
            "trick_in_progress": len(self.table_cards) > 0,
            "trick_size": len(self.table_cards),
            "expected_trick_size": self.num_players,
        }

        # Aggiunge informazioni per ciascun giocatore
        for i, player in enumerate(self.players):
            state[f"player_{i}_hand"] = [card for card in player.hand]
            state[f"player_{i}_points"] = player.points
            state[f"player_{i}_hand_size"] = len(player.hand)

        # Aggiunge informazioni di squadra per le partite a 4
        if self.is_team_game and self.teams is not None:
            state["teams"] = self.teams
            state["team_0_points"] = self.get_team_points(0)
            state["team_1_points"] = self.get_team_points(1)

        return state

    def get_valid_actions(self) -> List[int]:
        """
        Restituisce la lista degli indici giocabili dalla mano del giocatore di turno.

        In Briscola non esiste un vincolo di “rispondere al seme”: qualunque carta in mano è giocabile.
        L'azione è quindi semplicemente l'indice della carta nella lista `Player.hand`.
        """
        if self.game_over:
            return []
        return list(range(len(self.players[self.current_turn].hand)))

    def who_wins_trick(self, cards_and_players: List[Tuple[Card, int]]) -> int:
        """
        Determina il vincitore di una presa con 2 o 4 carte secondo le regole della Briscola.

        Priorità delle regole:
        1. Vince la briscola più alta
        2. Se non ci sono briscole: vince la carta più alta del seme di uscita
        3. Se ci sono semi diversi e nessuna briscola: vince chi ha aperto la presa

        Argomenti:
            cards_and_players: Lista di tuple (carta, indice_giocatore) in ordine di gioco

        Ritorna:
            Indice del giocatore vincitore
        """
        if not cards_and_players:
            raise ValueError("Nessuna carta da valutare")

        if self.trump_card is None:
            raise ValueError("Briscola non impostata")

        leading_suit = cards_and_players[0][0].suit
        trump_suit = self.trump_card.suit

        # Separa briscole e non-briscole
        trump_cards = [
            (card, player_idx, i) for i, (card, player_idx) in enumerate(cards_and_players) if card.suit == trump_suit
        ]

        if trump_cards:
            # Trova la briscola più alta
            highest_trump = max(trump_cards, key=lambda x: x[0].rank.trick_strength)
            return highest_trump[1]  # indice del giocatore
        else:
            # Nessuna briscola: trova la carta più alta del seme di uscita
            leading_suit_cards = [
                (card, player_idx, i)
                for i, (card, player_idx) in enumerate(cards_and_players)
                if card.suit == leading_suit
            ]

            if leading_suit_cards:
                highest_leading = max(leading_suit_cards, key=lambda x: x[0].rank.trick_strength)
                return highest_leading[1]
            else:
                # Nessuna carta del seme di uscita: vince chi ha aperto la presa
                return cards_and_players[0][1]

    def play_action(self, card_index: int) -> Dict:
        """
        Esegue un'azione di gioco (giocare una carta) e restituisce un risultato dettagliato.
        Gestisce la logica sia a 2 giocatori sia a 4 giocatori.

        Argomenti:
            card_index: Indice della carta da giocare dalla mano del giocatore di turno

        Ritorna:
            Dizionario con i dettagli dell'azione
        """
        if self.game_over:
            return {"error": "Partita già terminata"}

        if card_index not in self.get_valid_actions():
            return {"error": f"Azione non valida: {card_index}"}

        # Esegue la giocata
        current_player = self.players[self.current_turn]
        played_card = current_player.play_card(card_index)
        self.table_cards.append((played_card, self.current_turn))

        result: Dict[str, object] = {
            "played_card": played_card,
            "player": self.current_turn,
            "trick_completed": False,
            "trick_winner": None,
            "captured_cards": [],
            "cards_dealt": False,
            "trick_size": len(self.table_cards),
        }

        # Verifica se la presa è completa
        if len(self.table_cards) == self.num_players:
            # Determina il vincitore della presa
            trick_winner = self.who_wins_trick(self.table_cards)

            # Snapshot della presa prima di ripulire il tavolo.
            # Serve al frontend per mostrare in modo consistente le carte giocate quando lo stato WS
            # viene aggiornato dopo la cattura (e quindi `table_cards` risulta già vuoto).
            trick_cards = self.table_cards.copy()

            # Il vincitore prende tutte le carte della presa
            captured_cards = [card for card, _ in trick_cards]
            self.players[trick_winner].take_cards(captured_cards)

            # Aggiorna risultato
            result.update(
                {
                    "trick_completed": True,
                    "trick_winner": trick_winner,
                    "captured_cards": captured_cards,
                    "trick_cards": trick_cards,
                }
            )

            # Pulisce il tavolo e imposta il vincitore come primo di mano per la presa successiva
            self.table_cards.clear()
            self.first_player = trick_winner
            self.current_turn = trick_winner

            # Distribuisce nuove carte (solo modalità a 2)
            if not self.is_team_game and len(self.deck) > 0:
                # Distribuisce in ordine partendo dal vincitore della presa
                for i in range(self.num_players):
                    player_idx = (trick_winner + i) % self.num_players
                    if len(self.deck) > 0:
                        self.players[player_idx].add_card(self.deck.pop())
                result["cards_dealt"] = True

            # Verifica la condizione di fine partita
            if all(len(player.hand) == 0 for player in self.players):
                self._end_game()
        else:
            # Passa al prossimo giocatore nell'ordine di turno
            self.current_turn = (self.current_turn + 1) % self.num_players

        return result

    def _end_game(self) -> None:
        """Gestisce la fine partita e determina il vincitore"""
        self.game_over = True

        if self.is_team_game and self.teams is not None:
            # 4 giocatori: determina la squadra vincente
            team_0_points = self.get_team_points(0)
            team_1_points = self.get_team_points(1)

            if team_0_points > team_1_points:
                self.winner = (self.players[0], self.players[2])  # Squadra 0
            elif team_1_points > team_0_points:
                self.winner = (self.players[1], self.players[3])  # Squadra 1
            else:
                self.winner = None  # Tie
        else:
            # 2 giocatori: determina il vincitore individuale
            if self.players[0].points > self.players[1].points:
                self.winner = self.players[0]
            elif self.players[1].points > self.players[0].points:
                self.winner = self.players[1]
            else:
                self.winner = None  # Tie

    def get_game_result(self) -> Dict:
        """Restituisce il risultato finale nel formato adatto alla modalità di gioco"""
        if not self.game_over:
            return {"game_in_progress": True}

        result: Dict[str, object] = {"game_over": True, "is_team_game": self.is_team_game}

        if self.is_team_game and self.teams is not None:
            # Risultati modalità a squadre
            team_0_points = self.get_team_points(0)
            team_1_points = self.get_team_points(1)

            if self.winner is None:
                winner_str = "Pareggio"
                winning_team = None
            else:
                winning_team = 0 if self.winner == (self.players[0], self.players[2]) else 1
                if isinstance(self.winner, tuple):
                    winner_str = f"Squadra {winning_team} ({self.winner[0].name} e {self.winner[1].name})"
                else:
                    winner_str = "Squadra sconosciuta"

            result.update(
                {
                    "winner": winner_str,
                    "winning_team": winning_team,
                    "team_points": {"Team 0": team_0_points, "Team 1": team_1_points},
                    "individual_points": {player.name: player.points for player in self.players},
                    "point_difference": abs(team_0_points - team_1_points),
                }
            )
        else:
            # Risultati modalità individuale
            result.update(
                {
                    "winner": self.winner.name if isinstance(self.winner, Player) else "Pareggio",
                    "winner_index": (
                        0 if self.winner == self.players[0] else 1 if self.winner == self.players[1] else None
                    ),
                    "points": {player.name: player.points for player in self.players},
                    "point_difference": abs(self.players[0].points - self.players[1].points),
                }
            )

        return result

    def get_observation_for_player(self, player_index: int) -> Dict:
        """
        Restituisce lo stato della partita dal punto di vista di un giocatore specifico.
        Adatta il formato sia alla modalità a 2 sia a quella a 4.

        Argomenti:
            player_index: Indice del giocatore osservatore

        Ritorna:
            Stato osservabile per il giocatore indicato
        """
        if player_index < 0 or player_index >= self.num_players:
            raise ValueError(f"L'indice giocatore deve essere compreso tra 0 e {self.num_players - 1}")

        obs: Dict[str, object] = {
            "my_index": player_index,
            "my_hand": [card for card in self.players[player_index].hand],
            "my_points": self.players[player_index].points,
            "trump_card": self.trump_card,
            "trump_suit": self.trump_card.suit if self.trump_card else None,
            "table_cards": self.table_cards.copy(),
            "my_turn": self.current_turn == player_index,
            "cards_remaining_in_deck": len(self.deck),
            "valid_actions": self.get_valid_actions() if self.current_turn == player_index else [],
            "game_over": self.game_over,
            "num_players": self.num_players,
            "is_team_game": self.is_team_game,
        }

        # Aggiunge informazioni sugli altri giocatori
        for i in range(self.num_players):
            if i != player_index:
                obs[f"player_{i}_points"] = self.players[i].points
                obs[f"player_{i}_hand_size"] = len(self.players[i].hand)

        # Aggiunge informazioni di squadra per le partite a 4
        if self.is_team_game and self.teams is not None:
            my_team = self.get_player_team(player_index)
            teammate_index = None
            if my_team is not None:
                for team_player in self.teams[my_team]:
                    if team_player != player_index:
                        teammate_index = team_player
                        break

            obs.update(
                {
                    "my_team": my_team,
                    "teammate_index": teammate_index,
                    "teammate_points": self.players[teammate_index].points if teammate_index is not None else 0,
                    "my_team_points": self.get_team_points(my_team) if my_team is not None else 0,
                    "opponent_team_points": self.get_team_points(1 - my_team) if my_team is not None else 0,
                }
            )

        return obs

    def print_state(self) -> None:
        """Stampa lo stato corrente per debugging e visualizzazione"""
        mode = "4 giocatori a squadre" if self.is_team_game else "2 giocatori"
        print(f"\n{'=' * 60}")
        print(f"STATO PARTITA BRISCOLA ({mode})")
        print(f"{'=' * 60}")
        print(f"Briscola: {self.trump_card}")
        if not self.is_team_game:
            print(f"Carte nel mazzo: {len(self.deck)}")
        print(f"Turno corrente: {self.players[self.current_turn].name}")

        if self.is_team_game and self.teams is not None:
            print(f"\nSquadra 0 (Punti: {self.get_team_points(0)}):")
            print(f"  {self.players[0].name} ({self.players[0].points} punti)")
            print(f"  {self.players[2].name} ({self.players[2].points} punti)")
            print(f"Squadra 1 (Punti: {self.get_team_points(1)}):")
            print(f"  {self.players[1].name} ({self.players[1].points} punti)")
            print(f"  {self.players[3].name} ({self.players[3].points} punti)")

        print("\nMani dei giocatori:")
        for i, player in enumerate(self.players):
            print(f"  {player.name}: {len(player.hand)} carte - {[str(card) for card in player.hand]}")

        if self.table_cards:
            print("\nCarte sul tavolo:")
            for card, player_idx in self.table_cards:
                print(f"  {card} (giocata da {self.players[player_idx].name})")

        if self.game_over:
            result = self.get_game_result()
            print(f"\n{'=' * 60}")
            print("PARTITA TERMINATA")
            print(f"{'=' * 60}")
            print(f"Vincitore: {result['winner']}")
            if self.is_team_game:
                print(f"Punteggi squadre: {result['team_points']}")
                print(f"Punteggi individuali: {result['individual_points']}")
            else:
                print(f"Punteggi finali: {result['points']}")
            if result.get("point_difference", 0) > 0:
                print(f"Margine: {result['point_difference']} punti")
