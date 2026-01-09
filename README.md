# Briscola AI

![Coverage](https://img.shields.io/badge/coverage-56%25-red)

Un gioco di Briscola con funzionalità di IA. Il progetto implementa la Briscola con un’interfaccia web e un giocatore controllato dall’IA. L’obiettivo è arrivare a una rete neurale che impari a giocare raccogliendo dati dalle partite dei giocatori umani.

## Funzionalità

- Implementazione completa delle regole della Briscola
- Supporto sia per 2 giocatori sia per 4 giocatori (a squadre)
- Interfaccia utente web
- Aggiornamenti in tempo reale via WebSocket
- IA semplice (attualmente strategia casuale) con architettura "trigger":
  - Il frontend controlla le animazioni e triggera la mossa IA quando pronto
  - Il backend esegue la logica di gioco senza delay artificiali
  - Separazione pulita tra presentazione e logica
- Raccolta dati per machine learning

Nota didattica: lo sviluppo “step-by-step” verso l’addestramento ML è attualmente focalizzato sulla modalità **2 giocatori**; il 4‑player rimane come supporto e regressione.

## Struttura del progetto

- `src/briscola_ai/game/` – logica del gioco
  - `models.py` – classi `Card`, `Suit`, `Rank`, `Player`
  - `game.py` – classe `BriscolaGame` con le regole
- `src/briscola_ai/backend/` – server backend
  - `server.py` – server FastAPI con endpoint API e WebSocket
- `src/briscola_ai/frontend/` – interfaccia frontend
  - `static/` – asset statici (HTML, CSS, JavaScript)
- `src/briscola_ai/ai/` – implementazione IA (in espansione)
- `tests/` – test unitari e d’integrazione (pytest)
- `scripts/` – script di utilità
  - `simulate_games.py` – simulazioni headless (self‑play casuale)
- `PLAN.md` – piano di refactoring e roadmap didattica (sempre aggiornato)

## Architettura comunicazione Backend ↔ Frontend

Il sistema usa un'architettura ibrida HTTP + WebSocket:

### Perché ibrida?

**Perché non solo REST (polling)?**
- Il polling richiede chiamate continue al server → inefficiente e ad alta latenza
- Non adatto a un gioco in tempo reale dove lo stato cambia frequentemente

**Perché non solo WebSocket?**
- Le azioni del giocatore (giocare carta, creare partita) sono operazioni puntuali
- REST offre semantica chiara (POST = azione, GET = lettura)
- Più facile da testare e debuggare con strumenti standard (curl, Postman)
- Gestione errori più semplice (status code HTTP)

**Scelta ibrida:**
- **REST** per le *azioni* del client → semantica chiara, stateless, facile debug
- **WebSocket** per gli *aggiornamenti* dal server → tempo reale, push, efficiente


### Endpoint HTTP (REST)

| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| `POST` | `/api/games` | Crea una nuova partita |
| `GET` | `/api/games/{id}` | Ottiene lo stato della partita |
| `POST` | `/api/games/{id}/actions` | Il giocatore gioca una carta |
| `POST` | `/api/games/{id}/ai-turn` | Triggera la mossa dell'IA |
| `GET` | `/api/games/{id}/result` | Ottiene il risultato finale |

### WebSocket (tempo reale)

Connessione: `ws://host/api/ws/{game_id}/{player_index}`

**Messaggi dal server:**

| Tipo | Descrizione |
|------|-------------|
| `observation` | Stato completo della partita (mano, tavolo, punti, turno) |
| `ai_card_reveal` | L'IA mostra quale carta sta per giocare |
| `trick_result` | Risultato della presa (vincitore, punti) |

### Flusso di gioco tipico

```
┌─────────────┐                              ┌─────────────┐
│   Frontend  │                              │   Backend   │
└──────┬──────┘                              └──────┬──────┘
       │  POST /games                               │
       │ ──────────────────────────────────────────>│
       │                              game_id       │
       │ <──────────────────────────────────────────│
       │                                            │
       │  WS connect /ws/{id}/0                     │
       │ ──────────────────────────────────────────>│
       │                          observation (WS)  │
       │ <──────────────────────────────────────────│
       │                                            │
       │  POST /actions (gioca carta)               │
       │ ──────────────────────────────────────────>│
       │                          observation (WS)  │
       │ <──────────────────────────────────────────│
       │                                            │
       │  POST /ai-turn (trigger IA)                │
       │ ──────────────────────────────────────────>│
       │                       ai_card_reveal (WS)  │
       │ <──────────────────────────────────────────│
       │                          observation (WS)  │
       │ <──────────────────────────────────────────│
       │                                            │
```

### Modello "Trigger" per l'IA

A differenza di un'architettura tradizionale dove il server decide quando l'IA gioca, qui il **frontend controlla il timing**:

1. Il giocatore gioca → backend aggiorna stato → frontend riceve update
2. Frontend mostra animazione (1.4s)
3. Frontend chiama `POST /ai-turn` quando pronto
4. Backend esegue mossa IA → invia `ai_card_reveal` → attende → invia update

Questo separa la **logica di presentazione** (frontend) dalla **logica di gioco** (backend).

## Installazione

Questo progetto usa [uv](https://github.com/astral-sh/uv) come package manager, per una gestione dell’ambiente e delle dipendenze più veloce e affidabile rispetto agli strumenti tradizionali.

Requisiti:
- Python **3.14**
- `uv`

1. Clona il repository:
   ```
   git clone https://github.com/yourusername/briscola.ai.git
   cd briscola.ai
   ```

2. Crea un virtual environment con uv:
   ```
   uv venv -p python3.14
   ```

3. Attiva il virtual environment:
   - Su Windows:
     ```
     .venv\Scripts\activate
     ```
   - Su macOS/Linux:
     ```
     source .venv/bin/activate
     ```

4. Installa il pacchetto con uv:
   ```
   uv pip install -e .
   ```

   Per lo sviluppo puoi installare anche le dipendenze dev:
   ```
   uv pip install -e ".[dev]"
   ```

   Se non hai uv installato, segui le istruzioni su [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv).

   Nota: se modifichi `pyproject.toml`, reinstalla il pacchetto per rendere effettive le modifiche:
   ```
   uv pip install -e .
   ```

## Esecuzione dell’applicazione

Dopo l’installazione assicurati che il virtual environment sia attivo (dovresti vedere `(.venv)` nel prompt). Poi puoi avviare il server con lo script installato:

```
briscola-server
```

Oppure direttamente tramite il modulo Python:

```
python -m briscola_ai.main
```

Opzioni da linea di comando:
- `--host` – host su cui esporre il server (default: 0.0.0.0)
- `--port` – porta su cui esporre il server (default: 8000)
- `--reload` – abilita l’auto-reload per lo sviluppo

Quando il server è avviato, apri il browser su:
```
http://localhost:8000
```

## Sviluppo (test, lint, typecheck)

Con il virtual environment attivo e le dipendenze dev installate (`uv pip install -e ".[dev]"`):

- Test: `pytest`
- Coverage: `pytest --cov=briscola_ai --cov-report=term-missing`
- Badge coverage (manuale): aggiorna la percentuale nel link in cima a questo README.
- Lint: `ruff check src tests scripts`
- Format: `ruff format src tests scripts`
- Typecheck: `mypy src`

## Simulazioni (headless)

Per simulare N partite senza UI (utile per debug e, in futuro, generazione dataset):

```
python scripts/simulate_games.py --num-games 100 --seed 42 --num-players 2
```

## Come giocare

1. Nella schermata iniziale seleziona il numero di giocatori (2 o 4)
2. Inserisci il tuo nome e scegli quale giocatore controllare
3. Premi “Avvia partita” per iniziare
4. Clicca su una carta in mano per giocarla
5. L'IA risponderà automaticamente al suo turno

## Sviluppi futuri

- Implementare un’IA basata su rete neurale usando i dati raccolti
- Aggiungere statistiche e analisi più avanzate
- Migliorare l’interfaccia con animazioni ed effetti sonori
- Aggiungere supporto multiplayer contro altri umani
- Implementare diversi livelli di difficoltà dell’IA

## Licenza

Questo progetto è rilasciato con licenza MIT – vedi il file `LICENSE` per i dettagli.
