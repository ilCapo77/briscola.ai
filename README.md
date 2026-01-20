# Briscola AI

![Coverage](https://img.shields.io/badge/coverage-84%25-brightgreen)

Un progetto didattico “end‑to‑end” nato da un’esigenza concreta: **studiare le reti neurali con un progetto reale**, non con esempi astratti.

La Briscola è un ottimo “laboratorio” perché obbliga a mettere insieme tutti i pezzi:
- un **motore di regole** corretto e testabile;
- un **backend** che espone un contratto stabile (API/WS);
- una **UI** che rende leggibile la sequenza delle mani;
- una **pipeline dati** per arrivare a dataset, baseline e training.

Obiettivo finale: arrivare a un’IA (rete neurale) che impari a giocare in modo riproducibile, misurabile e spiegabile.

## Funzionalità

- Implementazione completa delle regole della Briscola
- Motore (`domain/`) con supporto **2 giocatori** e **4 giocatori** (a squadre)
- Interfaccia utente web
- Aggiornamenti in tempo reale via WebSocket
- IA semplice (attualmente strategia casuale) con modello "standard" server‑driven:
  - Il backend avanza automaticamente la partita quando è il turno dell'IA
  - Il backend non introduce delay di presentazione (niente `asyncio.sleep()` per animazioni)
  - Il frontend controlla solo la presentazione (hold/animazioni) degli update ricevuti via WS
- Base per raccolta dati (roadmap ML in `PLAN.md`)

Nota didattica:
- la UI attuale è pensata e testata soprattutto in **modalità 2 giocatori** (è la modalità “principale”);
- il 4‑player è supportato dal motore e usato come supporto/regressione, ma **non è ancora pienamente supportato dal frontend**.

## Quick start

Questo progetto usa [uv](https://github.com/astral-sh/uv).

Requisiti:
- Python **3.14**
- `uv`

Comandi tipici:
- Crea env: `uv venv -p python3.14`
- Installa (editable): `uv pip install -e .`
- Dev deps: `uv pip install -e ".[dev]"`
- Avvia server: `briscola-server --reload`
- Apri UI: `http://localhost:8000`

## Struttura del progetto

- `src/briscola_ai/domain/` – dominio canonico: **regole e stato** (puro, testabile)
  - `models.py` – modelli `Card`, `Suit`, `Rank`
  - `state.py` – stato completo (`GameState`)
  - `engine.py` – transizione `step(state, action)` (deterministica dato seed/stato)
  - `rules.py` – regole isolate (es. vincitore della mano)
- `src/briscola_ai/backend/` – adattatore HTTP/WS (FastAPI)
  - `dto.py` – DTO Pydantic v2 (contratto dati)
  - `server.py` – endpoint REST + WebSocket, gestione partite in memoria
- `src/briscola_ai/frontend/static/` – UI (HTML/CSS/JS)
  - `assets/cards/` – immagini carte (front `{suit}_{rank}.png`, back `card_back.png`)
- `src/briscola_ai/ai/` – baseline AI (in evoluzione)
- `tests/` – test unitari + integrazione API/WS (pytest)
- `scripts/` – utilità (simulazioni headless)
  - `scripts/simulate_games.py` – simulazioni senza UI
- `PLAN.md` – roadmap didattica (fonte di verità su cosa fare dopo)

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
| `GET` | `/api/games/{id}` | Ottiene lo stato completo della partita (DTO `type: "game_state"`, per debug/spettatori) |
| `GET` | `/api/games/{id}?player_index={i}` | Ottiene la vista del giocatore `i` (stesso formato `type: "observation"` del WebSocket) |
| `POST` | `/api/games/{id}/actions` | Il giocatore gioca una carta |
| `GET` | `/api/games/{id}/result` | Ottiene il risultato finale |

### WebSocket (tempo reale)

Connessione: `ws://host/api/ws/{game_id}/{player_index}`

**Messaggi dal server:**

Nota importante: **tutti i messaggi WebSocket includono un campo `type`**.

- Lo **snapshot di gioco** (“observation”) ha `type: "observation"` e contiene campi come `my_hand`, `my_turn`, `table_cards`, ecc.
- I messaggi evento (es. reveal IA, risultato mano, keepalive) hanno anch'essi `type`.

| Messaggio | Formato | Descrizione |
|----------|---------|-------------|
| Snapshot (observation) | `{ "type": "observation", ... }` | Stato completo della partita per il giocatore indicizzato dal WS |
| Reveal IA | `{ "type": "ai_card_reveal", ... }` | L'IA mostra quale carta sta per giocare |
| Risultato mano | `{ "type": "trick_result", ... }` | Risultato della mano (carte, vincitore, punti) |
| Keepalive | `{ "type": "pong" }` | Risposta ai ping del client (non è uno snapshot) |

Nota: gli snapshot includono `server_version` (intero monotono) come metadato debug‑friendly (incrementato ad ogni azione, umana o IA) per capire se lo stato sta avanzando e per diagnosticare problemi di ordering/reconnect.

Regola pratica lato client:
- `payload.type === "observation"` → snapshot
- altrimenti → messaggio evento

### Flusso di gioco tipico

```
┌─────────────┐                              ┌─────────────┐
│   Frontend  │                              │   Backend   │
└──────┬──────┘                              └──────┬──────┘
       │  POST /api/games                           │
       │ ──────────────────────────────────────────>│
       │                              game_id       │
       │ <──────────────────────────────────────────│
       │                                            │
       │  WS connect /api/ws/{id}/0                 │
       │ ──────────────────────────────────────────>│
       │                              snapshot (WS) │
       │ <──────────────────────────────────────────│
       │                                            │
       │  POST /api/games/{id}/actions (gioca carta)│
       │ ──────────────────────────────────────────>│
       │                              snapshot (WS) │
       │ <──────────────────────────────────────────│
       │                                            │
       │                       ai_card_reveal (WS)  │
       │ <──────────────────────────────────────────│
       │                       trick_result (WS)    │
       │ <──────────────────────────────────────────│
       │                              snapshot (WS) │
       │ <──────────────────────────────────────────│
       │                                            │
```

### Modello server‑driven per l'IA (scelta attuale)

Il backend avanza automaticamente la partita quando è il turno dell'IA (pattern standard “server‑authoritative”):

1. Il giocatore gioca → backend aggiorna stato → frontend riceve update
2. Backend vede che tocca all'IA e gioca automaticamente:
   - invia `ai_card_reveal`
   - invia `trick_result` (quando la mano si completa)
   - invia lo snapshot aggiornato
3. Frontend gestisce la presentazione:
   - mette in coda gli eventi ricevuti via WS
   - applica gli snapshot solo dopo gli hold, per mostrare in modo leggibile: carta 1 → carta 2 → risultato

Questa scelta mantiene separata la **logica di presentazione** (frontend) dalla **logica di gioco** (backend), evitando al contempo che la UI debba “pilotare” il dominio con chiamate dedicate.

## Cosa è stato fatto (stato attuale)

- Dominio canonico isolato in `domain/` (motore `GameState + step()`), con regole testate.
- Backend FastAPI + DTO Pydantic v2 (contratto più esplicito e testabile).
- Contratto WS stabilizzato: snapshot `type: "observation"` + eventi (`ai_card_reveal`, `trick_result`, `pong`) e `server_version` monotona.
- UI resa più robusta: gestione reconnect/backoff, modalità polling `?polling=1`, sequenza mano più leggibile.
- Asset carte e proporzioni: UI mantiene aspect ratio immagini (177x285), retro carta `card_back.png`, placeholder stabili per briscola/mazzo a fine mazzo (niente “salti” layout).
- Test: unit (dominio) + integrazione (API/WS); coverage attuale **84%** su `briscola_ai`.

Dettaglio completo e roadmap: vedi `PLAN.md`.

## Cosa rimane da fare (prossimi step consigliati)

La prossima fase “vera” è trasformare il progetto in un ambiente addestrabile (data pipeline + baseline):

- **Persistenza “da laboratorio” (SQLite)**: event log append‑only di partite/azioni/osservazioni (riproducibilità e debug).
- **Export dataset** (SQLite → JSONL/Parquet) con schema versionato.
- **Self‑play batch** (random + euristica semplice) per generare dati e metriche.
- **Valutazione**: win‑rate su seed fissi; opzionale ELO/TrueSkill.

Non urgenti / futuri:
- lint/format JS (decisione di toolchain) e, se utile, un E2E leggero (Playwright).
- estendere la parte ML al 4-player (team‑play, osservazioni parziali, reward).

## Sviluppo (test, lint, typecheck)

Con il virtual environment attivo e le dipendenze dev installate (`uv pip install -e ".[dev]"`):

- Test: `pytest`
- Coverage: `pytest --cov=briscola_ai --cov-report=term-missing`
- Badge coverage (manuale): aggiorna la percentuale nel link in cima a questo README (Shields.io).
- Lint: `ruff check src tests scripts`
- Format: `ruff format src tests scripts`

### Event log (SQLite “da laboratorio”)

Quando avvii il server con lo script `briscola-server`, per default viene scritto un event log su:
- `./data/briscola_events.sqlite3`
  - Nota: `data/` e i file `*.sqlite3*` sono ignorati da git (sono output runtime, non sorgenti).

Per cambiare percorso (o disabilitare) puoi usare:
- CLI: `briscola-server --event-db ./data/mio_log.sqlite3` oppure `briscola-server --event-db ''`
- Env: `BRISCOLA_EVENT_DB_PATH=./data/mio_log.sqlite3 briscola-server`

### Smoke test UI (manuale)

Obiettivo: una verifica rapida (2–3 minuti) per capire subito se una modifica ha rotto il flusso principale della UI.

Setup:
- Avvia server: `briscola-server --reload`
- Apri UI: `http://localhost:8000`
- Apri DevTools → tab **Console** (lascia aperto durante la partita)

Checklist:
1. Avvia una partita **2-player** inserendo un nome giocatore
2. Gioca **3 mani complete** (tu giochi → IA gioca → appare il risultato della mano)
3. Verifica che la sequenza visiva sia sempre: **reveal** → **seconda carta** → **risultato della mano**
4. (Opzionale) Continua fino a fine partita e verifica che appaia **Partita terminata** + risultato finale

Expected:
- Nessun errore in console (ok log informativi; no eccezioni uncaught)
- Nessun “freeze”: la UI resta interattiva e le carte non scompaiono in modo incoerente
- Nessuna duplicazione sul tavolo (es. due carte attribuite allo stesso player nella stessa mano)
- Typecheck: `mypy src`

### Debug UI (quando qualcosa “si blocca”)

Se noti comportamenti strani (carte che spariscono, sequenza eventi incoerente, UI non cliccabile):

- Apri DevTools → tab **Console** e copia eventuali warning/error (in particolare su `observation` e `server_version`).
- Apri DevTools → tab **Network** → filtro **WS** e verifica:
  - che la connessione a `/api/ws/{game_id}/{player_index}` resti attiva
  - che arrivino messaggi con `type: "observation"` e (se presenti) eventi `ai_card_reveal` / `trick_result`
- Controlla lo stato connessione in alto a destra:
  - “Riconnessione…” indica che la UI sta facendo retry con backoff.
- Modalità debug senza WebSocket (polling):
  - apri la UI con `?polling=1` (es. `http://localhost:8000/?polling=1`)
  - utile per capire se un bug dipende dal WS/reconnect o dalla logica UI.

## Simulazioni (headless)

Per simulare N partite senza UI (utile per debug e, in futuro, generazione dataset):

```
python scripts/simulate_games.py --num-games 100 --seed 42 --num-players 2
```

## Come giocare

1. Inserisci il tuo nome e premi “Avvia partita”
2. Clicca su una carta in mano per giocarla
3. L'IA risponderà automaticamente al suo turno

Nota: la UI attuale avvia una partita **2-player**. Per testare flussi 4-player (senza UI) usa gli script headless o le API.

## Sviluppi futuri

- Implementare un’IA basata su rete neurale usando i dati raccolti
- Aggiungere statistiche e analisi più avanzate
- Migliorare l’interfaccia con animazioni ed effetti sonori
- Aggiungere supporto multiplayer contro altri umani
- Implementare diversi livelli di difficoltà dell’IA

## Licenza

Questo progetto è rilasciato con licenza MIT – vedi il file `LICENSE` per i dettagli.
