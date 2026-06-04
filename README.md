# Briscola AI

![Coverage](https://img.shields.io/badge/coverage-74%25-yellowgreen)

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
- IA selezionabile (baseline + modelli locali) con modello "standard" server‑driven:
  - Agenti baseline: `random`, `greedy_points`, `heuristic_v1`, `heuristic_v2` (metadati e descrizioni in italiano esposti dal backend)
  - Modelli locali: `bc_model` carica un file `.npz` scelto dalla UI (catalogo server-side, no path arbitrari dal browser)
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

## Come giocare

1. Inserisci il tuo nome, scegli l’avversario (IA) e premi “Avvia partita”
   - La UI mostra una breve descrizione dell’IA selezionata (dai metadati del backend).
2. Clicca su una carta in mano per giocarla
3. L'IA risponderà automaticamente al suo turno

Nota: la UI attuale avvia una partita **2-player**. Per testare flussi 4-player (senza UI) usa gli script headless o le API.

### Giocare contro un modello locale (`.npz`)

Se hai addestrato un modello (BC / PG / A2C) e lo hai salvato in un file `.npz`, puoi usarlo come avversario nella UI:

1. Metti il file in una directory “whitelist” lato server:
   - consigliato: `./data/models/` (creala se non esiste)
   - alternativa: imposta `BRISCOLA_MODELS_DIR=/path/alla/tua/cartella`
2. (Ri)avvia il server: `briscola-server --reload`
3. Nella UI:
   - scegli **“Modello locale (.npz)”** come avversario;
   - seleziona il modello dal dropdown “Modello (file .npz)”.

Nota UI:
- il dropdown mostra `metadata_json.label` e la descrizione mostra `metadata_json.description_it` (se presenti nel file `.npz`);
- i trainer del progetto (`scripts/train_*.py`) salvano questi campi in automatico.
- se un file `.npz` non è compatibile (chiavi mancanti o `feature_dim` diverso), il backend lo segnala nel catalogo e la UI ne disabilita la selezione.

Nota sicurezza:
- il browser non invia path arbitrari: invia solo un `ai_model_id` (path relativo) scelto tra quelli esposti da `GET /api/ai/models`;
- il backend rifiuta path traversal (`..`) e carica modelli solo dentro `BRISCOLA_MODELS_DIR`.

## Approccio step-by-step (didattico)

L'idea è costruire una pipeline ML “dal basso”, in modo verificabile:

1. **Dominio/testabile**: regole e transizioni pure in `src/briscola_ai/domain/` + test su invarianti e casi limite.
2. **Backend/UI**: FastAPI + WS per far giocare umani e rendere osservabile lo stato.
3. **Raccolta dati**: event log SQLite (append-only) per debug e dataset.
4. **Export dataset**: conversione SQLite → JSONL con schema versionato.
5. **Self-play**: generazione rapida di partite dal dominio per produrre molti dati.
6. **Valutazione**: match offline riproducibili (win-rate/punti medi) per confrontare agenti.
7. **Training**: imitation/RL quando i contratti e la pipeline sono stabili.

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
  - `scripts/self_play_to_db.py` – self-play dal dominio verso SQLite (no HTTP)
  - `scripts/fast_self_play.py` – self-play veloce 2-player summary-only (no DB/event log completo)
  - `scripts/export_dataset.py` – export SQLite → JSONL
  - `scripts/evaluate_agents.py` – valutazione offline agenti (dominio di default, fast path sperimentale)
  - `scripts/benchmark_perf.py` – benchmark locale throughput simulazioni/training
- `PLAN.md` – roadmap didattica (fonte di verità su cosa fare dopo)

## Backend (FastAPI + WebSocket)

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

## Frontend (UI web)

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

## Sviluppo (test, lint, typecheck)

Con il virtual environment attivo e le dipendenze dev installate (`uv pip install -e ".[dev]"`):

- Test: `pytest`
- Coverage: `pytest --cov=briscola_ai --cov-report=term-missing`
- Badge coverage (manuale): aggiorna la percentuale nel link in cima a questo README (Shields.io).
- Lint: `ruff check src tests scripts`
- Format: `ruff format src tests scripts`
- Typecheck: `mypy src`

## AI & ML (pipeline)

### Event log (SQLite “da laboratorio”)

Quando avvii il server con lo script `briscola-server`, per default viene scritto un event log su:
- `./data/briscola_events.sqlite3`
  - Nota: `data/` e i file `*.sqlite3*` sono ignorati da git (sono output runtime, non sorgenti).

Per cambiare percorso (o disabilitare) puoi usare:
- CLI: `briscola-server --event-db ./data/mio_log.sqlite3` oppure `briscola-server --event-db ''`
- Env: `BRISCOLA_EVENT_DB_PATH=./data/mio_log.sqlite3 briscola-server`

Per cambiare modalità di logging (consigliato `dataset` per raccolta umani):
- CLI: `briscola-server --event-log-mode dataset`
- Env: `BRISCOLA_EVENT_LOG_MODE=dataset briscola-server`

Metadati salvati per partita:
- `seed`
- `code_version` (override possibile con `BRISCOLA_CODE_VERSION`)
- `rules_version` (versione semantica del dominio)

#### Modalità logging: `debug` vs `dataset` (raccolta dati umani)

Per default il backend logga in modalità `debug` (completa, utile per troubleshooting).
Per raccogliere partite giocate da umani e tenere il DB *piccolo*, usa:

- `BRISCOLA_EVENT_LOG_MODE=dataset briscola-server`

In modalità `dataset`:
- il backend **non** salva `observation_sent` (che è il payload più grande e ridondante);
- salva invece un evento `human_action` *self-contained* (observation → action → reward/done → next_observation);
- salva un marker `game_finished` quando `game_over=true` (utile per esportare solo partite complete);
- **non** salva `player_names` nel DB (privacy/igiene dati);
- la UI invia un `client_id` pseudonimo (UUID in `localStorage`) per permettere split train/val per giocatore senza PII.
- (opzionale) salva anche `client_decision_time_ms`: tempo stimato (ms) tra inizio turno umano e click (utile per filtri qualità/analisi).
- consenso: la UI mostra una checkbox e il backend rifiuta `POST /api/games` se il consenso non è accettato.

#### Deploy: CORS (minimo hardening)

Se deployi la UI/API su un dominio pubblico, è consigliato restringere le origin CORS:

- `BRISCOLA_CORS_ALLOW_ORIGINS=https://tuodominio.example briscola-server`

Se non imposti nulla, per default è `*` (comodo in sviluppo, non consigliato in produzione).

### Simulazioni (headless)

Per simulare N partite senza UI (utile per debug e generazione dataset):

```
python scripts/simulate_games.py --num-games 100 --seed 42 --num-players 2
```

### Self-play (dominio → SQLite)

Per generare molte partite velocemente (senza server/UI) e salvarle nel DB:

```
python scripts/self_play_to_db.py --db ./data/briscola_events.sqlite3 --num-games 100 --seed 42 --num-players 2
```

Puoi scegliere gli agenti per ciascun player con `--agents` (CSV, uno per player):

```
# 2-player: heuristic vs random
python scripts/self_play_to_db.py --db ./data/briscola_events.sqlite3 --num-games 100 --seed 42 --num-players 2 --agents heuristic_v1,random
```

Nota: se `--agents` è omesso, usa `random` per tutti i player.

### Fast self-play (summary-only)

Per benchmark o roll-out leggeri 2-player puoi evitare SQLite/DTO e usare il motore numerico `fast_2p`:

```
python scripts/fast_self_play.py --num-games 100000 --seed 0 --agents greedy_points,random
```

Output opzionale JSONL minimale, una riga per partita:

```
python scripts/fast_self_play.py --num-games 1000 --seed 0 --agents greedy_points,random --out-jsonl /tmp/fast_self_play.jsonl
```

Nota: questo comando è volutamente “summary-only”: salva seed, agenti, punti finali e vincitore, ma non salva
osservazioni/azioni step-by-step. Per dataset BC completo usa ancora `scripts/self_play_to_db.py` + export.
Per ora supporta `random`, `greedy_points`, `heuristic_v1` e `heuristic_v2`.

### Benchmark performance

Per misurare il throughput locale dei loop engine-only e delle policy fast-compatible:

```
python scripts/benchmark_perf.py --mode fast-random --games 100000 --repeat 3 --seed 0
python scripts/benchmark_perf.py --mode numba-random --games 100000 --repeat 3 --seed 0
python scripts/benchmark_perf.py --mode fast-eval --games 100000 --repeat 3 --seed 0 --agent-a heuristic_v2 --agent-b heuristic_v1
python scripts/benchmark_perf.py --mode numba-eval --games 100000 --repeat 3 --seed 0 --agent-a heuristic_v2 --agent-b heuristic_v1
python scripts/benchmark_perf.py --mode numba-mlp --games 20000 --repeat 3 --seed 0 --agent-b heuristic_v1 --hidden-dim 128
```

Nota: `numba-random` e `numba-eval` misurano il core JIT in `fast_numba`; non misurano ancora il training
A2C completo. `numba-mlp` misura invece un rollout inference full-JIT con policy MLP zero-inizializzata.
Risultati locali indicativi dopo warm-up:
- random-vs-random: `fast-random` ~21.3k games/sec, `numba-random` ~445k games/sec (`~20.9x`);
- `heuristic_v2` vs `heuristic_v1`: `fast-eval` ~13.3k games/sec, `numba-eval` ~300k games/sec (`~22.6x`).
- MLP hidden=128 vs `heuristic_v1`: `numba-mlp` ~2.84k games/sec.

### Valutazione agenti

Per confrontare agenti in modo riproducibile (senza UI/server):

```
python scripts/evaluate_agents.py --num-games 1000 --seed 42 --agent0 random --agent1 random
python scripts/evaluate_agents.py --num-games 1000 --seed 42 --agent0 greedy_points --agent1 random
python scripts/evaluate_agents.py --num-games 1000 --seed 42 --agent0 heuristic_v1 --agent1 random
```

Per default la valutazione usa il dominio canonico (`--engine domain`), che supporta tutti gli agenti
e costruisce `PlayerObservation` anti-cheat. Esiste anche un path sperimentale più veloce:

```
python scripts/evaluate_agents.py --engine fast --seat-fair --num-games 10000 --seed 42 --agent0 greedy_points --agent1 random
```

Nota: `--engine fast` supporta `random`, `greedy_points`, `heuristic_v1` e `heuristic_v2`. Serve per benchmark
del motore `fast_2p`; non supporta ancora modelli `.npz` come agenti.

Agenti disponibili (baseline):
- `random`: sceglie una carta casuale tra quelle in mano (baseline “zero”).
- `greedy_points`: gioca la carta con più punti in mano (euristica minimale e spiegabile).
- `heuristic_v1`: euristica 2-player che prova a prendere “a basso costo” quando conviene e scarta in modo economico quando non conviene.
- `heuristic_v2`: euristica 2-player che usa anche la “storia pubblica” (`seen_cards_onehot`) per una gestione briscole più strategica (teacher utile per BC).

#### Anti-cheat: osservazione parziale (information set)

Perché serve:
- nel dominio `GameState` contiene informazione **completa** (es. ordine del mazzo e mani di tutti);
- se un agente/una rete riceve `GameState`, può “barare” leggendo informazione nascosta (anche involontariamente), rendendo i benchmark non significativi.

Cosa facciamo:
- gli agenti ricevono una `PlayerObservation`, cioè una vista **parziale e lecita** dello stato;
- l’osservazione è costruita con `make_player_observation(state, player_index)` e poi passata a `Agent.choose_card_index(observation, rng=...)`.

Cosa contiene (alta-level):
- la mano del giocatore osservante, carte sul tavolo, briscola scoperta, `deck_size`, punteggi e dimensioni delle mani.

Cosa NON contiene:
- il mazzo come sequenza di carte (`state.deck`) e le carte specifiche in mano agli avversari.

Riferimenti utili:
- implementazione: `src/briscola_ai/domain/observation.py`
- test anti-regressione: `tests/test_domain_observation.py`

Nota importante (bias “chi inizia”):
- nel dominio attuale il player 0 inizia sempre la partita;
- per confronti più corretti **usa** la modalità **seat-fair**, che gioca due partite per seed scambiando i posti:

```
python scripts/evaluate_agents.py --seat-fair --num-games 10000 --seed 42 --agent0 random --agent1 random
```

Nota: `--seat-fair` richiede `--num-games` pari (si gioca a coppie).

Seed suite (regressioni confrontabili nel tempo):
- suite versionate nel repo: `--seed-suite small` (1000 seed) oppure `--seed-suite medium` (5000 seed)
- suite custom: `--seed-suite-file path/to/seeds.txt`
- per benchmark “big” senza file enorme: `--seed-suite-range-start 0` (genera i seed con `range()`)
- preset benchmark: `--benchmark small|medium|big` (imposta `--num-games` + una seed suite coerente, ed è sempre seat-fair)
- export risultati: `--out-json /path/to/out.json`

Taglie consigliate (benchmark):
- `small=2000` (feedback veloce)
- `medium=10000` (numero “standard” per confronti)
- `big=100000` (misura stabile, più lenta)

#### Evaluation matrix (consigliata)

Quando alleni molti modelli (`.npz`) conviene standardizzare i confronti per evitare errori manuali
e misurare robustezza (anche su holdout).

Lo script `scripts/evaluate_matrix.py` valuta un modello contro una lista di avversari su due suite:
- `standard` (seed generate con `range(start=0)`)
- `holdout` (seed generate con `range(start=1_000_000)`)

Esempio (veloce, `medium`):

```
python scripts/evaluate_matrix.py --model ./data/MODEL.npz --benchmark medium --out-json benchmarks/matrix_medium.json
```

Output “pretty” (tabella colorata) se il terminale è interattivo:
- default: `--format auto` (già il default)
- forzare: `--format rich`
- fallback CSV-like (utile per redirect/log): `--format csv`

Esempio (robusto, `big`):

```
python scripts/evaluate_matrix.py --model ./data/MODEL.npz --benchmark big --out-json benchmarks/matrix_big.json
```

#### Metriche “qualità decisionale” (diagnosi stile di gioco)

Win-rate e punti medi misurano la *forza* di un modello, ma non sempre spiegano *perché* fa certe mosse.
Per diagnosticare comportamenti miopi (es. “spreca briscole alte per prendere scarti”) usiamo una metrica semplice:

- `trump_waste_rate` (2-player, solo “secondo di mano”): quante volte l’agente gioca una briscola
  **pur avendo** almeno una risposta vincente **non-briscola**.
- `trump_overkill_rate` (2-player, solo “secondo di mano”): quando l’agente **vince** giocando una briscola,
  quante volte usa una briscola “più costosa del necessario” rispetto alla briscola vincente minima disponibile.
  Utile per misurare il caso “butta briscole alte per prendere scarti”.

Se vuoi eliminare **senza retrain** gran parte dell’“overkill”:
- abilita il post-processing `inference_overkill_guard` (solo `bc_model`, 2-player, secondo di mano):
  se stai per vincere con una briscola, gioca automaticamente la **briscola vincente minima** in mano.
- come attivarlo:
  - (consigliato) salva il flag nei metadati quando alleni: `scripts/train_a2c.py --inference-overkill-guard` (o `train_bc.py`, `train_pg.py`)
  - (A/B veloce) imposta `BRISCOLA_BC_OVERKILL_GUARD=1` quando lanci server/valutazioni
- nota: essendo un post-processing deterministico, può cambiare *leggermente* la forza complessiva.
  Verifica sempre con `scripts/evaluate_matrix.py` e `scripts/evaluate_decision_quality.py`.

Se invece vuoi che il modello **lo impari davvero** (senza guard):
- Usa un *teacher* che non faccia overkill (es. `heuristic_v2`) e fai Behavior Cloning (BC) su encoder v2.
- Valuta sempre con guard OFF (non impostare `BRISCOLA_BC_OVERKILL_GUARD` e non salvare `inference_overkill_guard` nel `.npz`).

Workflow consigliato (BC → RL):

```
# 1) Self-play del teacher (DB temporaneo)
python scripts/self_play_to_db.py --db /tmp/selfplay.sqlite3 --num-games 5000 --seed 42 --num-players 2 --agents heuristic_v2,heuristic_v2

# 2) Export esempi per BC (solo observation->action, più leggero)
python scripts/export_dataset.py --db /tmp/selfplay.sqlite3 --out /tmp/dataset.jsonl --all-players --include-ai --no-next-state

# 3) BC v2 (teacher distillato in una rete)
python scripts/train_bc.py --encoder-version v2 --model mlp --hidden-dim 128 --epochs 10 --lr 0.001 --seed 42 \
  --data /tmp/dataset.jsonl --out ./data/models/bc_teacher_v2.npz

# 4) Verifica stile (guard OFF)
python scripts/evaluate_decision_quality.py --agent-a bc_model --agent-a-model ./data/models/bc_teacher_v2.npz --agent-b heuristic_v1 --benchmark medium

# 5) (Opzionale) Fine-tuning A2C partendo dal BC
python scripts/run_experiment.py --algo a2c --init ./data/models/bc_teacher_v2.npz --benchmarks medium --num-games 200000 --train-seed 6 --seat-fair \
  --opponent-mix heuristic_v2:0.4,heuristic_v1:0.3,random:0.2,greedy_points:0.1 --no-update-best --minimal-data -- \
  --encoder-version v2
```

Nota: il fine-tuning RL può “allontanarsi” dallo stile del teacher (es. reintrodurre un po’ di overkill).
Se vuoi preservare lo stile *senza* usare il guard, puoi attivare una regolarizzazione verso un anchor BC fisso:

```
python scripts/train_a2c.py \
  --init ./data/models/bc_teacher_v2.npz \
  --bc-anchor ./data/models/bc_teacher_v2.npz \
  --bc-anchor-beta 0.02 \
  --encoder-version v2 \
  --opponent-mix heuristic_v2:0.4,heuristic_v1:0.3,random:0.2,greedy_points:0.1 \
  --num-games 200000 --seat-fair --seed 6 \
  --out ./data/models/a2c_from_bc_anchor.npz
```

Se vuoi *ridurre* questo comportamento durante training A2C, puoi provare un shaping soft:
- `scripts/train_a2c.py --overkill-penalty-mode flat --overkill-penalty-beta <beta>` (default: 0, disattivato)
- `--overkill-low-lead-points-max 2` (default) per colpire soprattutto gli “scarti o quasi”.

Oppure una variante più informativa:
- `--overkill-penalty-mode gap` (penalità proporzionale al “gap” tra briscola scelta e briscola vincente minima).

Esempio:

```
python scripts/evaluate_decision_quality.py \
  --agent-a bc_model --agent-a-model ./data/models/best_a2c.npz \
  --agent-b heuristic_v1 \
  --benchmark medium
```

#### Encoder v2: “storia pubblica” (card counting lecito, anti-cheat)

Molti comportamenti “miopi” (es. usare briscole alte per prendere scarti) emergono quando lo stato
che diamo al modello è troppo “istantaneo”: vede mano+tavolo+briscola+punti, ma non *il corso* della partita.

Per abilitare strategia senza barare, introduciamo un encoder **v2** che aggiunge una feature:

- `seen_cards_onehot[40]`: one-hot delle carte **già viste** (briscola scoperta + tavolo + carte già uscite).

Perché è anti-cheat?
- Sono informazioni pubbliche. In una partita reale il giocatore le vede e può ricordarle.
- Non include mai: ordine del mazzo, mano avversaria, o altre informazioni nascoste.

Uso pratico:
- Training BC (supervised) su dataset esportato:

```
python scripts/train_bc.py --data ./data/dataset.jsonl --out ./data/models/bc_v2.npz --encoder-version v2
```

- Training RL (PG/A2C):
  - da zero: `--encoder-version v2`
  - warm-start da un modello v1: `--upgrade-init-v1-to-v2` (aggiunge 40 righe a zero in `w1` e poi il training le “riempie”)

Esempio A2C:

```
python scripts/train_a2c.py \
  --out ./data/models/a2c_v2.npz \
  --init ./data/models/best_a2c.npz \
  --encoder-version v2 \
  --upgrade-init-v1-to-v2 \
  --opponent-mix heuristic_v1:0.7,random:0.2,greedy_points:0.1 \
  --num-games 200000 --seat-fair --seed 0
```

Nota: in partita (`ai_agent=bc_model`) il backend sceglie automaticamente l’encoder corretto
in base ai metadati del modello (`metadata.encoder`) o, in fallback, dalla `feature_dim`
(248=v1, 288=v2).

#### Pipeline esperimenti (training + eval) (consigliata)

Per iterare velocemente sui modelli senza perdere traccia dei risultati, usa:
- `scripts/run_experiment.py` (un comando unico: training → evaluation matrix → manifest → best model locale).

Output (convenzione):
- modello: `./data/models/<name>.npz`
- risultati: `./benchmarks/experiments/<name>/matrix_medium.json`, `matrix_big.json`, `manifest.json`
- best model (locale): `./data/models/best_<algo>.npz` + `best_<algo>.json` con lo score

Note pratiche:
- log “live”: la pipeline forza `PYTHONUNBUFFERED=1` sui trainer, così vedi le metriche mentre l’esperimento gira (utile per capire se sta imparando o diverge).
- coerenza directory modelli: la pipeline imposta `BRISCOLA_MODELS_DIR=<models-dir>` per i processi figli, così alias come `best_a2c` risolvono sempre nella stessa cartella.
- modalità “data minimale”: se vuoi mantenere `data/models/` pulita, usa `--minimal-data`.
  - se `--update-best` è attivo (default), conserva solo `best_*` e copia il modello finale in `benchmarks/experiments/<name>/model.npz`.
  - se fai screening con `--no-update-best`, mantiene comunque `data/models/` minimale (conserva i best se esistono; altrimenti conserva il modello run-specific).

Esempio A2C (da zero, opponent mix, poi eval medium+big):

```
python scripts/run_experiment.py \
  --algo a2c \
  --opponent-mix heuristic_v1:0.7,random:0.2,greedy_points:0.1 \
  --num-games 200000 \
  --train-seed 5 \
  --seat-fair
```

Warm-start (opzionale):
- se hai già un best locale, puoi ripartire da `--init ./data/models/best_a2c.npz` (stesso schema pesi/encoder).

#### Curriculum (easy → standard → hard)

Se vuoi un percorso di apprendimento più stabile, puoi usare un **curriculum a stage**:
lo stesso run viene spezzato in più blocchi consecutivi, passando il modello come `--init` tra gli stage.

Preset supportato:
- `--curriculum easy_standard_hard` (split 20% / 50% / 30%)
  - `easy`: `random:0.7,greedy_points:0.3`
  - `standard`: `heuristic_v1:0.7,random:0.2,greedy_points:0.1`
  - `hard`: `best_a2c:0.6,heuristic_v1:0.3,random:0.1`

Nota:
- se imposti `--curriculum`, la pipeline **ignora** `--opponent` e `--opponent-mix` (usa i preset).
- log per stage: in `benchmarks/experiments/<name>/train_stage_*.log` (utile per capire dove il training migliora o satura).

Esempio (A2C, seed 6, curriculum + eval medium+big, data minimale):

```
python scripts/run_experiment.py \
  --algo a2c \
  --curriculum easy_standard_hard \
  --num-games 200000 \
  --train-seed 6 \
  --seat-fair \
  --minimal-data
```

#### League training (avversario “best” congelato) — `best_a2c`

Quando inizi a ottenere modelli RL forti, allenare “sempre contro lo stesso baseline” può diventare limitante.
Una tecnica pratica è la **league**: continui ad allenare un nuovo modello, ma lo fai giocare spesso contro
un avversario “campione” **congelato** (non cambia durante quel run).

Per supportare questo in modo riproducibile esiste l’alias agente:
- `best_a2c` → carica un file locale `best_a2c.npz` dalla *directory modelli* (`BRISCOLA_MODELS_DIR`, oppure `./data/models`, fallback `./data`).

Da dove arriva `best_a2c.npz`?
- la pipeline `scripts/run_experiment.py` (default `--update-best`) aggiorna `./data/models/best_a2c.npz` quando trova un modello migliore.

Esempio: allenare A2C contro un mix che include il best congelato:

```
python scripts/train_a2c.py \
  --out ./data/models/a2c_vs_best_mix.npz \
  --opponent-mix best_a2c:0.5,heuristic_v1:0.3,random:0.2 \
  --num-games 200000 \
  --seed 6 \
  --seat-fair
```

Nota:
- se `best_a2c.npz` manca (o è incompatibile con l’encoder 2-player v1), il trainer fallisce subito con un errore esplicativo.

Nota metrica “best model”:
- per default usiamo `avg_diff` su suite `holdout` vs `heuristic_v1` (preferibilmente benchmark `big`).
- per run “di prova” puoi disabilitare l’aggiornamento del best: `--no-update-best`.

### Export dataset (JSONL)

Quando hai raccolto partite nel DB SQLite (event log), puoi esportare un dataset in JSONL:

```
python scripts/export_dataset.py --db ./data/briscola_events.sqlite3 --out ./data/dataset.jsonl
```

Default didattico (coerente con la UI attuale):
- la UI attuale avvia e testa principalmente partite **2-player** (tu vs IA);
- esporta solo le azioni del `player_index=0`;
- esclude le azioni dell'IA.
- esporta **solo partite complete** (`game_over=true`) per avere un dataset più pulito.

Opzioni utili:
- tutti i player: `--all-players`
- includi anche IA: `--include-ai`
- export “supervised only” (senza `next_observation`): `--no-next-state`
- includi anche partite incomplete (sconsigliato per dataset principale): `--include-incomplete`

Nota su modalità `dataset`:
- se il backend gira con `BRISCOLA_EVENT_LOG_MODE=dataset`, l'exporter usa preferibilmente l'evento `human_action`
  (record già self-contained) e non dipende da `observation_sent`.

Nota: lo schema export v1 è pensato soprattutto per il 2-player. In 4-player (a squadre) la nozione di reward
e l'interpretazione del “vincitore della mano” vanno adattate a livello di team (vedi `PLAN.md`).

### Primo modello (Behavior Cloning)

Scopo didattico: partire con un modello supervisionato semplice che imita un “teacher”
(es. `heuristic_v1` o `heuristic_v2`) invece di partire subito con RL.

Scelta chiave: spazio azioni fisso “**40 carte + action mask**”.
- Il modello predice una carta tra 40 classi (ogni carta del mazzo).
- Una **mask** abilita solo le carte realmente in mano (evita azioni impossibili).

Workflow minimo:
1. Genera un DB con self-play (es. teacher vs random):
   - `python scripts/self_play_to_db.py --db ./data/briscola_events.sqlite3 --num-games 200 --seed 42 --num-players 2 --agents heuristic_v1,random`
   - variante (teacher più “strategico”): `python scripts/self_play_to_db.py --db ./data/briscola_events.sqlite3 --num-games 200 --seed 42 --num-players 2 --agents heuristic_v2,random`
2. Esporta un JSONL di esempi:
   - `python scripts/export_dataset.py --db ./data/briscola_events.sqlite3 --out ./data/dataset.jsonl --all-players --include-ai`
3. Allena il primo modello (baseline lineare):
   - `python scripts/train_bc.py --data ./data/dataset.jsonl --out ./data/bc_model.npz --epochs 10 --lr 0.5`
   - Variante più espressiva (MLP 1-hidden-layer):
     - `python scripts/train_bc.py --model mlp --hidden-dim 128 --data ./data/dataset.jsonl --out ./data/bc_model_mlp.npz --epochs 20 --lr 0.001`
4. Valuta il modello vs una baseline (seed suite riproducibile):
   - `python scripts/evaluate_agents.py --seat-fair --num-games 2000 --seed-suite small --agent0 bc_model --agent0-model ./data/bc_model.npz --agent1 heuristic_v1`

Nota:
- `bc_model.npz` è un artefatto locale (non va versionato nel repo).
- Puoi usare due modelli diversi con `--agent0-model` e `--agent1-model`.

### Superare `heuristic_v1` (RL)

Il Behavior Cloning (BC) tende a *eguagliare* il teacher, non a superarlo.
Per superare `heuristic_v1` puoi fare fine-tuning con Reinforcement Learning ottimizzando direttamente il return finale.

#### REINFORCE (policy gradient) + opponent mix

Workflow consigliato (warm-start da BC MLP teacher-only):
1. Allena BC MLP su dataset teacher-only (vedi sopra) → `./data/bc_model_teacher_mlp.npz`
2. Fine-tuning RL contro `heuristic_v1`:
   - `python scripts/train_pg.py --init ./data/bc_model_teacher_mlp.npz --out ./data/rl_vs_heuristic_v1.npz --opponent heuristic_v1 --num-games 20000 --seat-fair --seed 0`
   - se stampa troppo, aggiungi `--log-every 50`
   - per robustezza (consigliato): usa un opponent mix
     - `python scripts/train_pg.py --init ./data/bc_model_teacher_mlp.npz --out ./data/rl_mix.npz --opponent-mix heuristic_v1:0.7,random:0.2,greedy_points:0.1 --num-games 20000 --seat-fair --seed 0`

Mini-grid (esempio) per scegliere il mix:
- setup: warm-start da `bc_model_teacher_mlp.npz`, training 100k game, benchmark `medium` (10k) + holdout `medium` via `--seed-suite-range-start 1000000`.
- metrica: **diff punti media** (A−B) in seat-fair.

| setup | vs heuristic_v1 | holdout vs heuristic_v1 | vs random | vs greedy_points |
|---|---:|---:|---:|---:|
| single (solo heuristic_v1) | +3.36 | +3.57 | +31.64 | +30.19 |
| mix 85/10/05 | +3.33 | +3.23 | +32.19 | +30.39 |
| mix 70/20/10 | **+3.66** | **+3.78** | **+33.28** | **+31.52** |
| mix 60/20/20 | +2.15 | +2.79 | +32.70 | +31.18 |

Nota interpretativa:
- aumentare la quota “easy opponents” tende a rendere la policy più robusta (margini più alti vs random/greedy),
  ma può ridurre la performance vs `heuristic_v1` se la quota di `heuristic_v1` scende troppo.
  Per questo conviene scegliere il mix guardando una piccola “evaluation matrix” (almeno vs `heuristic_v1`, `random`, `greedy_points`) e fare anche un holdout di seed.
3. Valuta:
   - `python scripts/evaluate_agents.py --benchmark small --agent0 bc_model --agent0-model ./data/rl_vs_heuristic_v1.npz --agent1 heuristic_v1`

#### A2C (actor-critic) + reward shaping “trick delta” (consigliato)

REINFORCE funziona, ma è più rumoroso. Un passo successivo “ad alto ROI” è A2C:
- aggiungiamo un *critic* `V(s)` (value head) e usiamo l’**advantage** `A = G - V(s)` come baseline appresa;
- usiamo un reward più denso (senza barare): ogni step è un **turno della policy**, e il reward è il delta di
  `(punti_policy - punti_opp)` accumulato fino al turno successivo (include la chiusura della mano).

Script:
- training: `scripts/train_a2c.py`

Esempio (warm-start + opponent mix):
- `python scripts/train_a2c.py --init ./data/bc_model_teacher_mlp.npz --out ./data/a2c_shaped.npz --opponent-mix heuristic_v1:0.7,random:0.2,greedy_points:0.1 --num-games 200000 --seat-fair --seed 0`

Rollout fast sperimentale (avversari fast-compatible):
- `python scripts/train_a2c.py --rollout-engine fast --out ./data/a2c_fast_hv2.npz --opponent heuristic_v2 --num-games 20000 --seat-fair --seed 0`
- `python scripts/train_a2c.py --rollout-engine fast --fast-encoder numba --out ./data/a2c_fast_numba_encoder_hv2.npz --opponent heuristic_v2 --num-games 20000 --seat-fair --seed 0`
- `python scripts/train_a2c.py --rollout-engine fast --fast-rollout numba --out ./data/a2c_fast_numba_rollout_hv2.npz --opponent heuristic_v2 --num-games 20000 --seat-fair --seed 0`

Nota: `--rollout-engine fast` usa `fast_2p` e feature numeriche equivalenti all’encoder canonico, ma per ora supporta
solo `random`/`greedy_points`/`heuristic_v1`/`heuristic_v2` e non supporta `--overkill-penalty-beta > 0`.
`--fast-encoder numba` valida il wrapper JIT dell’encoder osservazione; non è ancora il rollout A2C full-JIT, perché
lo stato viene ancora convertito da liste Python durante il training.
`--fast-rollout numba` usa invece un collector full-JIT per stato, encoder, forward MLP, sampling, opponent e reward;
il backprop/Adam restano NumPy. Benchmark locale 5k game vs `random`: hidden=32 `~5.06s -> ~2.52s` (`~2.0x`),
hidden=128 `~6.83s -> ~4.93s` (`~1.39x`).

Esempio di risultato (indicativo, dipende da seed/iperparametri/dati):
- con 200k game e mix 70/20/10, A2C + shaping ha superato `heuristic_v1` anche su `big` + holdout con un margine ~`+7` punti medi.

Validazione robusta (consigliata):
- benchmark “big” (più stabile, più lento):
  - `python scripts/evaluate_agents.py --benchmark big --agent0 bc_model --agent0-model ./data/MODEL.npz --agent1 heuristic_v1 --out-json benchmarks/model_vs_heuristic_v1_big.json`
- holdout di seed (evita “overfitting” su una sola suite):
  - `python scripts/evaluate_agents.py --benchmark big --seed-suite-range-start 1000000 --agent0 bc_model --agent0-model ./data/MODEL.npz --agent1 heuristic_v1 --out-json benchmarks/model_vs_heuristic_v1_big_holdout_1M.json`

Nota:
- `scripts/train_pg.py` e `scripts/train_a2c.py` salvano un `.npz` con `w1/b1/w2/b2` (MLP). L’agente `bc_model` lo supporta, come per i modelli BC MLP.
- I file in `data/` (DB SQLite, dataset JSONL, modelli `.npz`) sono artefatti locali: non vanno versionati nel repo.

## Stato e prossimi step

Dettaglio completo e roadmap: vedi `PLAN.md`.

Direzioni consigliate (ML):
- Actor‑Critic (A2C minimale) per ridurre varianza rispetto a REINFORCE puro.
- Opponent mix per robustezza (evitare overfitting su un singolo avversario).
- Reward shaping leggero (delta punti per mano) mantenendo osservazioni anti‑cheat.
- Dati umani (opzionale): raccolta con consenso UI + export “human-only”.

## Sviluppi futuri

- Implementare un’IA basata su rete neurale usando i dati raccolti
- Aggiungere statistiche e analisi più avanzate
- Migliorare l’interfaccia con animazioni ed effetti sonori
- Aggiungere supporto multiplayer contro altri umani
- Implementare diversi livelli di difficoltà dell’IA

## Licenza

Questo progetto è rilasciato con licenza MIT – vedi il file `LICENSE` per i dettagli.
