# Piano di Refactoring (Deep) — Briscola AI

## Obiettivo didattico (vision)

Rendere il progetto **attuale, testabile e “insegnabile”**, così da poter imparare passo‑passo come si costruisce e si addestra un’IA per la Briscola:
- **Separare** chiaramente *motore di gioco* (regole) da *API/serving* e da *training*.
- Avere un percorso incrementale: prima affidabilità e strumenti, poi dataset, poi modelli, poi valutazione.

## Decisioni iniziali (da questa discussione)

- Target runtime: **Python 3.14** + migrazione a **Pydantic v2**.
- Scope didattico: partire da **Briscola 2 giocatori**; estendere al 4‑player in una fase successiva.
  - Nota: il codice supporta già 4 giocatori, ma finché non entriamo nella fase “team-play” useremo il 4‑player solo per smoke/regressioni (non come focus di design/training).
- Persistenza dati: partire con **SQLite file-based** (event log + query) e aggiungere un comando di **export** dataset (JSONL/Parquet); Docker/Postgres solo quando serve.

## Stato attuale (sempre aggiornato)

- Runtime: **Python 3.14**, **FastAPI** + **Pydantic v2**.
- Backend: endpoint HTTP + `WebSocket` (stato partita in memoria, cleanup via lifespan).
- Motore: `GameState + step()` (supporta 2 e 4, ma **focus didattico sul 2-player**).
- Frontend: UI statica (`src/briscola_ai/frontend/static/`).
- Tooling: workflow su `ruff` (lint+format; import sorting via regole `I`) + `mypy`.
- Asset carte: immagini carte in `src/briscola_ai/frontend/static/assets/cards/` (servite a `/static/assets/cards/`).
  - Naming:
    - front: `{suit}_{rank}.png` con `suit` in `{clubs,cups,coins,swords}` e `rank` in `1..10` (es. `clubs_1.png`)
    - back: `card_back.png` (retro carta, usato per mano avversario e mazzo)
  - Nota UI: le carte in UI mantengono l'aspect ratio delle immagini (177x285px).
- UI quality: **stabile in 2-player** (nessun bug visibile segnalato); smoke test UI manuale documentato e decisione su linting JS ancora aperta.
  - Punti da sistemare/considerare (IA server-driven + robustezza):
    - Backend: mossa IA eseguita automaticamente quando è il suo turno (pattern standard); serializzazione mutazioni tramite `game_locks`.
    - Frontend: coda eventi WS + hold per mantenere la sequenza didattica (carta 1 → carta 2 → risultato) anche se gli update arrivano “subito”.
    - Contratto WS: gli snapshot includono `type: "observation"` (allineato in README/UI/test).
    - Chiarire vincolo attuale UI “umano = player 0” (focus 2-player; da generalizzare se aggiungiamo scelta giocatore/4-player in UI).
  - Timing animazioni (scelta architetturale):
    - Il backend evita `asyncio.sleep()` per ritardi di presentazione (reveal/risultato mano).
    - Il frontend “trattiene” gli snapshot WS per mostrare reveal e risultato con tempi controllati lato UI.
- Test: presenti in `tests/` (unit + integrazione API base).
- Test attuali: **177** (pytest).
- Coverage: misurata con `pytest-cov` (attuale ~74% su `briscola_ai`; obiettivo: crescita progressiva).
- Badge coverage: manuale via Shields.io nel `README.md` (niente `coverage.svg` versionato / script di generazione).
- AI: agenti baseline selezionabili (random/greedy/euristica) + possibilità di giocare contro un modello locale `.npz` via UI (catalogo server-side, no path arbitrari dal browser).

Comandi di verifica (sempre validi):
- test: `pytest`
- coverage: `pytest --cov=briscola_ai --cov-report=term-missing`

## Principi guida (per un refactor “che insegna”)

1. **Purezza del dominio**: il motore di gioco non deve dipendere da FastAPI, JSON, filesystem, ecc.
2. **Riproducibilità**: seed e determinismo dove possibile (simulazioni, shuffle, self‑play).
3. **Contratti stabili**: schema dati/API versionato e testato.
4. **Refactor a piccoli passi**: ogni step produce un risultato verificabile (test, benchmark, demo).

## Roadmap (proposta a fasi)

### Fase 0 — Baseline & sicurezza del refactor (1–2 sessioni) ✅ (completata)

- [x] Checklist “smoke”: avvio server locale (`briscola-server --reload`)
- [x] Simulazione senza UI: `python scripts/simulate_games.py --num-games 100 --seed 42`
- [x] Test iniziali del motore in `tests/`:
  - [x] mazzo da 40 carte uniche
  - [x] distribuzione corretta (2p e 4p)
  - [x] fine partita 2p + somma punti = 120
- [x] Rafforzamento test (regole core):
  - [x] ordine delle carte nella mano (Asso > Tre > Re > ...)
  - [x] briscola batte non-briscola anche se “bassa”
  - [x] flusso `play_action`: fine mano, aggiornamento turno, pescata in 2p
- [x] Documentazione didattica:
  - [x] docstring dettagliate su moduli/classi/funzioni/metodi (Python)
- Risultato: comportamento “bloccato” da test prima dell’upgrade dipendenze.

### Fase 1 — Modernizzazione toolchain + dipendenze (2–4 sessioni) ✅ (completata)

Obiettivo: aggiornare senza rompere API e comportamento, usando i test come rete di sicurezza.

- [x] Python: portare `requires-python` al target **3.14**.
- Dipendenze runtime:
  - [x] aggiornare `fastapi` e `uvicorn` (versioni compatibili con Pydantic v2)
  - [x] migrare runtime a **`pydantic v2`** (install/lock)
  - [x] rimuovere la dipendenza diretta da `websockets` (il supporto WS resta via `uvicorn[standard]`)
- Dev tooling (consigliato):
  - [x] `ruff` (lint + format)
  - [x] `pytest` + `pytest-cov` + `coverage`
  - [x] `mypy` (typecheck)
- Esempio flusso con uv:
  - `uv pip install -e ".[dev]"`
  - `uv lock --upgrade`

Stato attuale (dopo upgrade):
- venv: Python 3.14
- stack: FastAPI + Pydantic v2
- tests: verdi (105)
- coverage totale: ~74% (focus prossimo: aumentare copertura di `main.py`, `decision_quality.py`, `model_catalog.py` e dei rami non coperti del backend)

Prossimi step (per aumentare coverage, focus 2-player):
- [x] Testare più rami API: `404 partita`, `player_index` invalido, `get_game_result` e fine partita.
- [x] Test WebSocket (solo happy-path): connessione, ricezione stato iniziale, ping/pong.
- [x] Testare cleanup/lifespan (almeno che non lanci eccezioni in startup/shutdown).

### Workstream UI — Stabilizzazione e refactor frontend (parallelamente a Fase 2+)

Obiettivo: rendere la UI affidabile e “debuggabile” (strumento didattico e, in futuro, di raccolta dati).

Step suggeriti (focus 2-player):
- [x] Stabilizzare rendering carte: immagini in `/static/assets/cards/` e normalizzazione payload carte lato UI (WS/HTTP).
- [x] Sequenza mano stabile e leggibile: 1° carta → 2° carta → risultato (con tempi controllati lato frontend; **senza** carte sovrapposte).
- [x] Fix freeze UI: ignorare messaggi WS keepalive (`ping`/`pong`) che non sono snapshot di gioco.
- [x] Esito mano: usare `trick_result.trick_cards` dal backend per evitare race (niente duplicazioni “Tu/IA” sul tavolo).
- [x] Evitare duplicazione briscola: quando il mazzo è vuoto, mostrare solo `trump_suit` (non la carta) per non visualizzare la stessa carta anche in mano.
- [x] Smoke test UI manuale (documentato): passi ripetibili + expected (utile per regressioni).
  - Documentazione: vedi `README.md` → sezione “Smoke test UI (manuale)”.
- [x] Riprodurre e catalogare eventuali bug UI residui (console JS, tab Network, handshake WebSocket).
- [x] Allineare contratto dati UI↔API:
  - [x] definire un DTO stabile per `Card` e `GameObservation` (Pydantic: `CardDTO`, `ObservationDTO`)
  - [x] definire un DTO stabile per `GameResult` (Pydantic: `GameResultDTO`)
  - [x] ridurre accoppiamento a stringhe “magiche” (es. `player_0_hand_size`) introducendo campi espliciti (es. `players[]`)
- [x] Robustezza runtime:
  - [x] gestione errori in UI (banner/stato connessione, retry/backoff WS, messaggi user-friendly)
  - [x] fallback senza WebSocket (polling) per debug (`?polling=1`)
- [x] Test UI:
  - [x] smoke test manuale documentato (passi + expected)
  - [ ] (futuro, opzionale) E2E leggero con Playwright quando introduciamo una toolchain JS

- [x] **Refactor IA → modello server-driven (standard)**:
  - Backend: rimuovere endpoint di trigger e far avanzare automaticamente la partita quando tocca all'IA
  - Frontend: mantenere UX invariata con hold/coda eventi (no dipendenza da un trigger client)
  - Backend/UI: scelta dell'agente IA all'avvio (`ai_agent`) e policy basata su osservazione parziale (anti-cheat)
  - UI: mostrare descrizione dell'agente IA selezionato (metadati dal backend)
  - Obiettivo: UI didattica leggibile senza `asyncio.sleep()` nel backend

Deliverable minimo:
- la UI permette di avviare una partita 2-player, giocare carte e vedere fine partita senza errori in console.

### Fase 2 — Ristrutturazione architetturale (dominio vs adattatori) (3–6 sessioni) ✅ (completata)

Obiettivo: rendere chiaro “cosa è Briscola” vs “come la servo” vs “come la alleno”.

Stato (Phase 2B): ✅ completata
- [x] Introdotto motore “funzionale” in parallelo: `GameState + step()` in `src/briscola_ai/domain/`.
- [x] Migrato il backend (in-memory store) a `GameState` + `step`.
- [x] Migrare gli endpoint HTTP `/actions` a DTO (rimuovere `GameJSONEncoder` e `_json_safe`).
- [x] Spostati `Card/Rank/Suit` nel dominio (`src/briscola_ai/domain/models.py`) e rimossa la cartella `src/briscola_ai/game/`.

Proposta struttura (indicativa):

```
src/briscola_ai/
  domain/            # regole pure, stato, transizioni (no FastAPI)
  api/               # FastAPI app, schema, routing, websocket
  web/               # static assets (o invariato in frontend/static)
  agents/            # bot baseline (random, heuristic)
  training/          # dataset, env RL, training scripts
  cli.py             # entrypoint comandi: serve/simulate/train/eval
tests/
```

Azioni chiave:
- Rendere il motore **stateless o quasi**:
  - introdurre un oggetto `GameState` (immutabile o copiabile) + funzione `step(state, action) -> new_state, reward, done, info`
  - isolare RNG in un componente iniettato (es. `random.Random(seed)`)
- Separare “osservazione per giocatore” da “stato completo”:
  - utile per ML e per evitare leak informativo.
  - Stato attuale:
    - [x] introdotta `PlayerObservation` e usata nelle valutazioni offline (agenti non vedono `GameState` completo)
- API: trasformare oggetti Python in JSON con **Pydantic schema** invece di encoder custom.

### Fase 3 — Test “seri” e qualità (parallela alle fasi 1–2) ✅ (completata)

Obiettivo: coprire il motore e assicurare stabilità in evoluzione.

- Unit test dominio (priorità alta):
  - [x] invarianti: nessuna carta duplicata, 40 carte totali, turni coerenti, tavolo coerente (`tests/test_domain_invariants.py`)
  - [x] regole mano (`who_wins_trick`) con casi noti (`tests/test_trick_rules.py`)
  - [x] punteggi (somma punti carte catturate) come invariante dei test dominio
  - [x] casi limite: ultima carta/briscola in 2p e pareggi (2p/4p) (`tests/test_domain_endgame_cases.py`)
- Test di integrazione API (priorità media):
  - [x] crea partita → gioca azione → stato cambia (incl. `server_version` monotona) (`tests/test_api_integration.py`)
  - [x] endpoint `/result` (in progress + game over 2p/4p + pareggio) (`tests/test_api_integration.py`)
  - [x] WebSocket: connessione + ricezione update (anche test “light”) (`tests/test_api_integration.py`)
- Aggiungere CI (GitHub Actions) per `pytest` e lint (quando il repo è versionato).

### Fase 4 — Data pipeline per ML (didattica) (4–8 sessioni)

Obiettivo: passare da “gioco” a “ambiente addestrabile”.

- Introdurre una persistenza “da laboratorio” (SQLite):
  - tabella partite, step/azioni, osservazioni, metadati (seed, versione regole, versione codice)
  - scrittura append-only (stile event log) per semplificare debug e riproducibilità
  - Stato attuale:
    - [x] event log SQLite (schema + writer append-only) configurabile via env/CLI (`BRISCOLA_EVENT_DB_PATH`, `--event-db`)
    - [x] metadati “stabili” salvati per partita: `code_version` + `rules_version` (tabella `games` + payload `game_created`)
    - [x] raccolta dati umani “dataset mode” (DB più piccolo):
      - env: `BRISCOLA_EVENT_LOG_MODE=dataset`
      - eventi: `human_action` (self-contained) + marker `game_finished` (solo quando `game_over=true`)
      - igiene dati: non salvare `player_names` nel DB; usare `client_id` pseudonimo (UUID UI)
      - qualità dati: salvare `client_decision_time_ms` (tempo decisionale stimato in ms)
      - consenso: in `dataset` la UI mostra una checkbox e il backend rifiuta `POST /games` senza consenso
      - deploy: endpoint `GET /api/meta` per UI e env `BRISCOLA_CORS_ALLOW_ORIGINS` per restringere CORS in produzione
- Definire un comando di export dataset (per training):
  - da SQLite → JSONL/Parquet con schema versionato
  - campi minimi: `state` (osservazione), `valid_actions`, `action`, `reward`, `done`, `metadata`
  - Stato attuale:
    - [x] export SQLite → JSONL (script `scripts/export_dataset.py`)
    - [ ] decidere schema “finale” per training (es. reward shaping, include/exclude info IA)
- Implementare un simulatore “self‑play”:
  - due agenti baseline (random + heuristic)
  - generazione di partite in batch con seed
  - scrittura su SQLite + export dataset + metriche
  - Stato attuale:
    - [x] self-play → SQLite con agenti configurabili (script `scripts/self_play_to_db.py`)
- Introdurre una baseline di valutazione:
  - win-rate su set di seed
  - ELO/TrueSkill (opzionale)
  - Stato attuale:
    - [x] valutazione offline dominio-only (script `scripts/evaluate_agents.py`)
    - [x] lista agenti centralizzata (metadati+factory in `briscola_ai.ai.agents`, riusati da UI/CLI/script)
    - [x] baseline euristica semplice (es. `heuristic_v1`) per confronto vs random
    - [x] taglie benchmark: `small=2000`, `medium=10000`, `big=100000` (tutte seat-fair)
    - [x] supporto a “suite seed” per regressioni ripetibili (seed da file via `--seed-suite-file`)
    - [x] suite canoniche versionate: `small=1000 seed` e `medium=5000 seed` (file in `seed_suites/`)
    - [x] preset `--benchmark` + export risultati JSON (script `scripts/evaluate_agents.py`)
    - [ ] per `big`: decidere se versionare anche 50k seed o usare suite “range()” (generata via CLI)

### Fase 5 — Modelli e training step-by-step (8+ sessioni, incrementale)

Obiettivo: imparare “end‑to‑end” senza saltare subito al deep learning complesso.

Percorso consigliato:
1. **Heuristic agent** (regole semplici) → capire “feature utili” e debugging.
2. **Supervised learning** su dataset di mosse (imitazione):
   - modello piccolo (MLP) → prevedere azione tra `valid_actions`
   - attenzione a mascherare azioni non valide
3. **Reinforcement learning** (quando il dominio è stabile):
   - wrapper stile Gymnasium
   - reward shaping minimo e valutazione robusta

Scelte e stato (Fase 5A):
- [x] definito spazio azioni per BC: **40 carte + action mask** (non "indice nella mano")
- [x] definito encoder observation → feature (v1) e salvata una versione (`src/briscola_ai/ai/training/observation_encoder.py`)
- [x] primo modello BC (baseline lineare) + loop di training riproducibile (`scripts/train_bc.py`)
- [x] integrazione del modello come agente (valutazione con `scripts/evaluate_agents.py`)
- [x] variante BC più espressiva: MLP minimale (1 hidden layer + ReLU) con training in NumPy (`scripts/train_bc.py --model mlp`)
- [x] training RL per superare baseline: policy gradient (REINFORCE) con warm-start da BC (`scripts/train_pg.py`)

Risultati recenti (esempio, artefatti locali in `data/` e JSON in `benchmarks/`):
- BC MLP teacher-only: quasi pari con `heuristic_v1` su `big` (diff punti ≈ -0.6).
  - modello: `data/bc_model_teacher_mlp.npz`
  - benchmark: `benchmarks/bc_teacher_mlp_vs_heuristic_v1_big.json`
- RL (policy gradient) warm-start da BC MLP: supera `heuristic_v1` in modo robusto anche su holdout seed.
  - modello: `data/rl_vs_heuristic_v1_200k.npz` (200k game di training vs `heuristic_v1`)
  - benchmark big: `benchmarks/rl_vs_heuristic_v1_200k_big.json` (diff punti ≈ +5.5)
  - benchmark big holdout: `benchmarks/rl_vs_heuristic_v1_200k_big_holdout_1M.json` (diff punti ≈ +5.3)

Prossime direzioni consigliate (Fase 5B, miglioramenti “algoritmo/setting”):
- [x] Actor-Critic (A2C minimale): aggiungere una value head `V(s)` per ridurre la varianza rispetto a REINFORCE puro.
- [x] Opponent mix: allenare contro un mix di avversari (baseline + snapshot della policy) per robustezza e anti-overfitting.
- [x] Reward shaping leggero: usare reward denso (delta punti per mano) oltre al return finale, mantenendo l’osservazione anti-cheat.
- [ ] Dati umani (opzionale): pipeline di raccolta con consenso UI + tag nel DB + export “human-only” per pretraining/finetune.

Nota (tuning opponent mix):
- in una mini-grid (benchmark `medium` + holdout) la miscela `heuristic_v1:0.7,random:0.2,greedy_points:0.1` ha dato il miglior compromesso
  tra performance vs `heuristic_v1` e robustezza vs baseline più deboli (risultati dettagliati in `README.md`).

### Fase 5B — A2C + reward shaping (prossimo step)

Obiettivo: migliorare stabilità/performance del training RL rispetto a REINFORCE puro, mantenendo anti-cheat.

Piano di lavoro (A2C “minimale” + reward denso):
- [x] Implementare `scripts/train_a2c.py`:
  - policy MLP (1 hidden layer) con action mask (40 carte)
  - value head `V(s)` (critic) per baseline appresa
  - training con Adam (policy + critic)
- [x] Reward shaping “trick delta”:
  - definire il time-step come “turno della policy”
  - reward per step = delta di `(punti_policy - punti_opp)` accumulato fino al prossimo turno della policy
- [x] Supportare `--opponent-mix` anche in A2C (riuso parsing già esistente)
- [x] Validare e benchmarkare:
  - `medium` vs `heuristic_v1` + holdout seed
  - `medium` vs `random` e `greedy_points`
  - (quando promettente) `big` + holdout
- [x] Documentare in `README.md` (didattico):
  - differenza REINFORCE vs A2C
  - perché reward shaping riduce varianza
  - comandi consigliati + note su robustezza (big + holdout)

### Fase 5C — Evaluation matrix (benchmarking ripetibile)

Obiettivo: standardizzare la valutazione di un modello su una “matrice” di match (avversari × seed suite),
per evitare errori manuali e rendere confronti robusti e ripetibili.

Piano di lavoro:
- [x] Implementare `scripts/evaluate_matrix.py`:
  - input: `--model` (path `.npz`)
  - avversari di default: `heuristic_v1`, `random`, `greedy_points` (configurabili)
  - per ogni avversario: `benchmark big` + `big holdout` (configurabili)
  - output: stampa tabella a schermo + `--out-json`
- [x] Migliorare l’output “a schermo” con tabella colorata (Rich) mantenendo fallback CSV-like
- [x] Aggiungere un modulo “core” importabile per test/riuso (es. `src/briscola_ai/ai/evaluation_matrix.py`)
- [x] Aggiungere test (veloci) per parsing/config/output
- [x] Documentare l’uso in `README.md` (didattico): perché serve e comandi consigliati

### Fase 5D — Giocare in UI contro un modello locale (`.npz`)

Obiettivo: permettere all’utente di selezionare un modello addestrato localmente (es. A2C/PG/BC) come avversario
direttamente dalla UI, senza introdurre rischi di sicurezza (path traversal) e mantenendo l’anti-cheat (osservazione parziale).

Piano di lavoro:
- [x] Definire un “catalogo modelli” locale:
  - directory configurabile via env (es. `BRISCOLA_MODELS_DIR`, default sotto `./data/`)
  - lista di file `.npz` con metadati (`metadata_json`) e una descrizione breve in italiano (best effort)
- [x] Standardizzare i metadati UI nei trainer:
  - i modelli salvati da `scripts/train_*.py` includono `label` e `description_it` dentro `metadata_json`
  - la UI li usa per mostrare un dropdown più leggibile (senza euristiche sul filename)
- [x] Esporre un endpoint backend per la UI:
  - `GET /ai/models` → lista `{ id, label, description_it, metadata }` (senza path assoluti)
- [x] Migliorare la robustezza UX:
  - il catalogo indica `is_compatible` + un motivo `compatibility_reason_it` per modelli non caricabili/incompatibili
  - la UI disabilita la selezione di modelli non compatibili e fallisce presto in modo chiaro
- [x] Rifinire la selezione modello in UI:
  - `best_a2c.npz` viene ordinato come scelta consigliata e mostra anche il nome file
  - la descrizione espone i dettagli utili per giocare (file selezionato e guard anti-overkill)
  - `GET /ai/models` sintetizza le metriche lunghe (`metrics_count`) invece di inviare la cronologia completa al browser
- [x] Estendere creazione partita:
  - supportare `ai_agent="bc_model"` + `ai_model_id`

### Fase 5E — Roadmap breve per potenziare i modelli (league/curriculum)

Obiettivo: aumentare la forza/robustezza dei modelli RL **senza perdere riproducibilità** e mantenendo la proprietà anti-cheat
(gli agenti vedono solo `PlayerObservation`).

Roadmap (in ordine):
1. **League training contro un “best” congelato**:
   - introdurre un alias agente `best_a2c` che carica un file locale “campione” (es. `data/models/best_a2c.npz`)
   - usare `best_a2c` dentro `--opponent-mix` (e.g. `best_a2c:0.5,heuristic_v1:0.3,random:0.2`)
   - idea didattica: evitare “chasing” instabile (due policy che cambiano insieme) e ridurre regressioni
2. **Curriculum / mix experiments**:
   - definire 2–3 preset di mix avversari (easy/standard/hard) e farli scalare nel training
   - misurare la generalizzazione con `evaluate_matrix.py` su standard+holdout
3. **(Opzionale) PPO + GAE**:
   - introdurre clipping PPO + advantage con GAE per stabilità su training più lunghi
   - mantenere la stessa observation/action space per confronti “fair”

Stato:
- [x] (5E.1) Alias agente `best_a2c` (file locale) + documentazione
- [x] (5E.2) Preset curriculum + harness “train+eval” riproducibile
  - [x] definire 3 preset opponent mix: `easy`, `standard`, `hard` (quest’ultimo include `best_a2c`)
  - [x] aggiungere una modalità “curriculum” alla pipeline `scripts/run_experiment.py`:
    - eseguire training in 2–3 stage in sequenza (easy → standard → hard)
    - passare `--init` tra stage (warm-start)
    - salvare log per stage + includere i comandi nel `manifest.json`
    - in `--minimal-data`: mantenere solo il modello finale (e rimuovere gli stage intermedi)
  - [x] aggiungere test unit per la logica di split stage (somma `num_games`, rounding deterministico)
  - [x] documentare in `README.md`: quando usare curriculum, esempi, trade-off
- [ ] (5E.3) Spike PPO+GAE (solo se serve)
  - validare che `ai_model_id` punti a un file whitelisted dentro `BRISCOLA_MODELS_DIR` (no `..`, no path arbitrari)
- [x] Aggiornare frontend:
  - mostrare un select “Modello” solo quando l’utente sceglie l’agente `bc_model`
  - visualizzare una descrizione breve del modello selezionato (in italiano) + i metadati utili
- [x] Test:
  - `GET /ai/models` ritorna una lista coerente e non espone path
  - `POST /games` con `bc_model` fallisce senza `ai_model_id` e rifiuta path traversal
- [x] Documentazione:
  - aggiornare `README.md`: dove mettere i modelli, come avviare e giocare contro un modello, note di sicurezza/anti-cheat

### Fase 5E — Pipeline esperimenti (training + evaluation) riproducibile

Obiettivo: rendere facile (e ripetibile) iterare sui modelli senza fare comandi “a mano” e senza perdere traccia dei risultati.

Piano di lavoro:
- [x] Definire un comando unico (script) che:
  - allena un modello (A2C/PG, con warm-start opzionale)
  - esegue una evaluation matrix su `medium` e `big` (incluso holdout) e salva JSON
  - produce un `manifest.json` con: config, comandi, versioni (`code_version`, `rules_version`), percorsi output
- [x] “Best model” locale:
  - scelta metrica: `avg_diff` su suite `holdout` vs `heuristic_v1` (preferibilmente su `big`)
  - salva/aggiorna `data/models/best_<algo>.npz` + JSON di accompagnamento con lo score e la provenienza
- [x] Test:
  - unit test per estrazione metrica da JSON della matrice e per naming deterministico dell’esperimento
- [x] Documentazione:
  - aggiornare `README.md` con un esempio end-to-end e con la struttura cartelle (`data/models`, `benchmarks/experiments/...`)
- [x] Igiene `--minimal-data`:
  - supportare anche `--no-update-best` (screening): mantenere `data/models/` minimale senza forzare l’aggiornamento del best

Workflow consigliato (tuning):
- [x] Mini-sweep “veloce” (no update best):
  - 6 run `--benchmarks medium` con warm-start da `data/models/best_a2c.npz`
  - variando solo `--lr` e `--entropy-beta` via args dopo `--`
  - selezione top-1 per `holdout vs heuristic_v1 avg_diff` (benchmark `medium`)
- [x] Run “definitiva”:
  - stessa configurazione top-1, training più lungo e benchmark `medium,big`
  - aggiornare `best_a2c.npz` solo se migliora lo score su `big holdout vs heuristic_v1`
  - risultato: aggiornato `best_a2c.npz` con `big holdout vs heuristic_v1 avg_diff = +9.71` (seed training=8, `lr=3e-4`, `entropy_beta=1e-3`)

Miglioramenti di ergonomia (pipeline):
- [x] Log “live” durante training/eval:
  - evitare buffering su stdout quando i trainer sono eseguiti via pipe (es. `run_experiment.py`)
  - obiettivo: vedere metriche A2C/PG mentre l’esperimento gira (utile per capire subito se diverge)
- [x] Modalità “data minimale”:
  - mantenere in `data/models/` solo `best_<algo>.npz` + `best_<algo>.json`
  - evitare accumulo di molti `.npz` intermedi (restano i manifest/log in `benchmarks/experiments/`)
- [x] Compattazione automatica dei best model:
  - quando `run_experiment.py` promuove `best_<algo>.npz`, rimuove `metadata_json.metrics` dalla copia runtime
  - conserva `metrics_summary` nel `.npz` e `metadata_compaction` nel JSON laterale
  - il modello completo dell'esperimento resta in `benchmarks/experiments/<name>/model.npz` per audit/riproducibilità

Prossimi esperimenti (A2C):
- [x] Run “lunga” 500k:
  - warm-start da `data/models/best_a2c.npz`
  - config: `lr=3e-4`, `entropy_beta=1e-3`, mix `heuristic_v1:0.7,random:0.2,greedy_points:0.1`, seat-fair
  - benchmark: `medium,big`
  - criterio di successo: aggiornare `best_a2c.npz` se migliora `big holdout vs heuristic_v1 avg_diff`
  - risultato: aggiornato `best_a2c.npz` con `big holdout vs heuristic_v1 avg_diff = +11.19` (seed training=9, 500k game)
- [x] Run “league” 1M contro il best congelato:
  - warm-start da `data/models/best_a2c.npz` precedente
  - config: `lr=1e-4`, `entropy_beta=2e-4`, mix `best_a2c:0.60,heuristic_v1:0.25,greedy_points:0.10,random:0.05`, seat-fair
  - esperimento: `benchmarks/experiments/a2c_league_best60_h25_g10_r05_1m_seed17/`
  - benchmark `medium,big` completati; criterio ufficiale `big holdout vs heuristic_v1`
  - risultato: promosso nuovo `data/models/best_a2c.npz` (`A2C shaped 1.0M game`)
  - score di promozione (policy guard OFF): `big holdout vs heuristic_v1 avg_diff = +13.12788`
  - confronto col best precedente: `+12.69434 -> +13.12788` (`+0.43354`, circa `+3.4%` sul vantaggio medio)
  - head-to-head vs best precedente (`big` holdout, 100k, seat-fair):
    - nuovo `50080` win, vecchio best `47018` win, draw `2902`
    - `avg_point_diff = +0.9772` a favore del nuovo modello
  - robustezza `big holdout`: vs `random` `+43.85694`, vs `greedy_points` `+42.06212`
- [x] Run “league” 1M con pipeline Numba completa:
  - warm-start da `data/models/best_a2c.npz`
  - config: `lr=1e-4`, `entropy_beta=2e-4`, mix `best_a2c:0.60,heuristic_v1:0.25,greedy_points:0.10,random:0.05`, seat-fair
  - esperimento: `benchmarks/experiments/a2c_league_best60_h25_g10_r05_1m_seed18_numba/`
  - pipeline: `--rollout-engine fast --fast-rollout numba --eval-engine numba --minimal-data --no-update-best`
  - tempo misurato con `/usr/bin/time -p`: `real=215.04s` per training 1M + matrix `medium,big`
  - evaluation Numba: `medium=4.175s`, `big=34.771s`
  - score: `big holdout vs heuristic_v1 avg_diff = +13.33254`
  - confronto con run seed17: `+13.12788 -> +13.33254` (`+0.20466`, circa `+1.6%` sul vantaggio medio)
  - robustezza `big holdout`: vs `random` `+43.51`, vs `greedy_points` `+41.94`
- [x] Run “league” 5M con pipeline Numba completa + promozione:
  - warm-start da `data/models/best_a2c.npz`
  - config: `lr=1e-4`, `entropy_beta=2e-4`, mix `best_a2c:0.60,heuristic_v1:0.25,greedy_points:0.10,random:0.05`, seat-fair
  - esperimento: `benchmarks/experiments/a2c_league_best60_h25_g10_r05_5m_seed19_numba/`
  - pipeline: `--rollout-engine fast --fast-rollout numba --eval-engine numba --minimal-data --no-update-best`
  - inferenza: modello salvato con `--inference-overkill-guard` per mantenere il guard anti-overkill anche nel best ufficiale
  - tempo misurato con `/usr/bin/time -p`: `real=930.39s` per training 5M + matrix `medium,big`
  - evaluation Numba: `medium=4.259s`, `big=37.734s`
  - score: `big holdout vs heuristic_v1 avg_diff = +13.91112`
  - confronto col best 1M ufficiale: `+13.12788 -> +13.91112` (`+0.78324`, circa `+6.0%` sul vantaggio medio)
  - head-to-head vs best 1M ufficiale (`big`, 100k, seat-fair): nuovo `50142` win, best precedente `47011` win, draw `2847`,
    `avg_point_diff = +1.1014`
  - decision-quality `big` vs `heuristic_v1`: `avg_diff=+13.9865`, `trump_overkill_rate=0.0%`,
    `trump_overkill_rate_low_lead_points=0.0%`, `trump_waste_rate≈0.08%`
  - decisione: promosso nuovo `data/models/best_a2c.npz` (`A2C shaped 5.0M game`, guard ON)
  - nota storage: il modello completo dell'esperimento contiene `250000` record metriche e pesa circa `244 MB`; il best ufficiale
    è una copia compattata (`metrics` rimossi dai metadati, `metrics_summary` conservato) e pesa circa `138 KB`

### Fase 5F — Comportamenti più “strategici”: storia pubblica + metriche qualità (in progress)

Obiettivo: ridurre comportamenti miopi (es. “spreco briscole alte per prendere scarti”) rendendoli:
1) **misurabili** (metriche qualità decisionale), e
2) **apprendibili** (stato più ricco: card counting lecito tramite storia pubblica).

Piano:
- [x] Metrica qualità v1: `trump_waste_rate` (secondo di mano)
  - definizione: l'agente gioca una briscola pur avendo una risposta vincente non-briscola
  - script: `scripts/evaluate_decision_quality.py`
- [x] Metrica qualità v2: `trump_overkill_rate` (secondo di mano)
  - definizione: quando l'agente vince giocando una briscola, quanto spesso usa una briscola “più costosa del necessario”
    rispetto alla briscola vincente minima disponibile (es. Asso di briscola invece di 2 di briscola)
  - variante: `trump_overkill_rate_low_lead` (solo quando la carta dell'avversario sul tavolo vale pochi punti)
  - scopo: catturare lo stile “butta briscole alte per scarti” che non sempre emerge da `trump_waste_rate`
- [x] Inference: post-processing anti-overkill per `bc_model` (no retrain)
  - regola: 2-player, secondo di mano; se il modello sta per vincere giocando una briscola, sostituisci con la briscola vincente minima
  - attivazione: `metadata.inference_overkill_guard=true` (o env var `BRISCOLA_BC_OVERKILL_GUARD=1`)

Validazione rapida (benchmark decision-quality `medium` vs `heuristic_v1`, seed=0, stesso modello `best_a2c.npz`):
- guard OFF: `avg_diff=+12.89`, `trump_overkill_rate≈20.3%`, low-lead `≈18.4%`
- guard ON (`--force-overkill-guard`): `avg_diff=+12.80`, `trump_overkill_rate=0.0%`, low-lead `=0.0%`
- Nota storica: questi numeri si riferiscono al precedente best v1, prima della promozione del modello 1M
  (`A2C shaped 1.0M game`, seed=17). Il best attuale è il modello 5M seed=19 con
  `inference_overkill_guard=true`.

Operatività (senza env var):
- [x] Abilitare `inference_overkill_guard=true` nei metadati di `data/models/best_a2c.npz`
  - obiettivo: ottenere gli stessi benefici "guard ON" senza dover impostare `BRISCOLA_BC_OVERKILL_GUARD`
  - verifica (`medium` vs `heuristic_v1`, seed=0): `trump_overkill_rate=0.0%` e low-lead `=0.0%` con `avg_diff≈+12.80`
- Nota: questa scelta era valida per il best precedente. Dopo la promozione del modello 1M, il nuovo
  `data/models/best_a2c.npz` non usa il guard in metadati; prima di riattivarlo va rifatta una A/B decision-quality.
  Nota aggiornata: dopo la promozione del modello 5M, il best ufficiale usa di nuovo
  `inference_overkill_guard=true`; questa nota resta solo come storico della decisione sul best 1M.

A/B test storico (evaluation matrix `medium`, seed=0, stessi avversari, prima della promozione del modello 1M):
- modello guard ON: vecchio `data/models/best_a2c.npz` (metadati `inference_overkill_guard=true`)
- modello guard OFF: `benchmarks/ab_overkill_guard/best_a2c_guard_off.npz`
- risultati (avg_diff):
  - vs `heuristic_v1` standard: ON `+12.57` vs OFF `+12.71` (Δ=-0.14)
  - vs `heuristic_v1` holdout:  ON `+12.15` vs OFF `+12.32` (Δ=-0.17)
  - differenze piccole (rumore statistico possibile), ma il guard elimina l’overkill per costruzione
- [x] Stato più ricco (anti-cheat) tramite “storia pubblica”:
  - [x] Definire una mappatura canonica “card -> id” (40 carte) in `domain/` (riusabile da dominio/backend/ai)
  - [x] Aggiungere a `PlayerObservation` `seen_cards_onehot[40]` derivato solo da info pubblica:
    - briscola scoperta (carta sotto il mazzo)
    - carte sul tavolo (in corso)
    - carte già uscite (ricostruite dalle prese/captured)
  - [x] Esporre `seen_cards_onehot` in `ObservationDTO` (UI + dataset logging) e popolarlo dal backend
  - [x] Encoder v2 (2-player) che include `seen_cards_onehot`:
    - mantenere l'ordine feature v1 e aggiungere `seen_cards_onehot[40]` in coda (feature_dim: 248 -> 288)
    - compatibilità: v1 resta default (modelli esistenti)
  - [x] Inference: aggiornare `BCModelAgent` per selezionare l'encoder in base ai metadati del modello
    - regola: `metadata.encoder` (se presente) > fallback su `feature_dim` (248=v1, 288=v2)
  - [x] UI catalog: accettare modelli v1 e v2 (feature_dim coerente) e spiegare la compatibilità in errore
  - [x] Training: aggiungere `--encoder-version {v1,v2}` ai trainer (BC/PG/A2C) + salvare `metadata.encoder`
  - [x] Test: coprire encoder v2 + path inference (BCModelAgent) + compatibilità catalogo
  - [x] Documentazione: in `README.md` spiegare “card counting lecito” (anti-cheat) e come usare v2

Prossimo esperimento (per verificare che v2 sia “meno miope”):
- [x] Addestrare A2C con encoder v2 (seed 6) con warm-start dal best v1:
  - pipeline: `scripts/run_experiment.py`
  - trainer args: `--encoder-version v2 --upgrade-init-v1-to-v2`
  - benchmark: almeno `medium` (poi eventualmente `big`)
- [x] Valutare la qualità decisionale del modello v2 vs `heuristic_v1`:
  - `scripts/evaluate_decision_quality.py` (benchmark `medium`)
  - confronto qualitativo: `trump_waste_rate` del v2 vs `best_a2c` v1
- [x] (Opzionale) Se migliora forza+qualità, promuovere un “best_a2c” v2 (decisione esplicita):
  - aggiornare `data/models/best_a2c.npz` solo se migliora su `holdout vs heuristic_v1` e non peggiora troppo su `trump_waste_rate`

Risultati (screening, seed=6, 200k game, encoder v2):
- esperimento: `benchmarks/experiments/a2c_mix_heuristic_v1_0_7_random_0_2_greedy_points_0_1_200kg_seed6_enc_v2/`
- evaluation matrix `medium`:
  - `holdout vs heuristic_v1 avg_diff = +12.25`
- decision quality `medium` vs `heuristic_v1`:
  - v2 (`model.npz`): `avg_diff=+12.23`, `trump_waste_rate≈0.1%` (55 / 77189)
  - best v1 dell'epoca (`data/models/best_a2c.npz` prima della promozione 1M):
    `avg_diff=+12.89`, `trump_waste_rate≈0.0%` (15 / 77965)
- decision quality “overkill briscola” (stesso match `medium` vs `heuristic_v1`):
  - v2 (`model.npz`): `trump_overkill_rate≈20.6%` (5845 / 28313), low-lead `≈18.5%` (2287 / 12348)
  - best v1 dell'epoca (`data/models/best_a2c.npz` prima della promozione 1M):
    `trump_overkill_rate≈20.3%` (5692 / 27985), low-lead `≈18.4%` (2199 / 11975)
- decisione: NON promuovere a best (in questo screening v2 non migliora né forza né `trump_waste_rate`)

Controllo storico sul best 1M:
- [x] Eseguire `evaluate_decision_quality.py` sul nuovo `data/models/best_a2c.npz`
  - confrontare guard OFF vs guard forzato ON
  - decidere se salvare `inference_overkill_guard=true` anche sul best 1M oppure tenerlo “puro”
  - risultati `medium` vs `heuristic_v1`, seed=0:
    - guard OFF (`data/models/best_a2c.npz`): `avg_diff=+13.21`, `trump_waste_rate≈0.03%`,
      `trump_overkill_rate≈19.3%`, low-lead `≈17.1%`
    - guard ON (copia temporanea con metadato `inference_overkill_guard=true`): `avg_diff=+13.03`,
      `trump_waste_rate≈0.03%`, `trump_overkill_rate=0.0%`, low-lead `=0.0%`
  - decisione tecnica provvisoria: il guard elimina l'overkill con costo piccolo su `medium` (`Δ≈-0.18` punti),
    ma prima di salvarlo nel best ufficiale conviene fare un A/B su `big` oppure almeno un head-to-head rapido.
  - nota tooling: `--force-overkill-guard` non sovrascrive un metadato esplicito
    `inference_overkill_guard=false`; per questo A/B è stata usata una copia temporanea del modello.
- [x] A/B guard su `big` per decisione finale sul best 1M:
  - guard OFF (`data/models/best_a2c.npz` prima della modifica): `avg_diff=+12.9557`,
    `trump_overkill_rate≈19.6%`, low-lead `≈17.0%`, `trump_waste_rate≈0.02%`
  - guard ON (copia temporanea con metadato `inference_overkill_guard=true`): `avg_diff=+12.8115`,
    `trump_overkill_rate=0.0%`, low-lead `=0.0%`, `trump_waste_rate≈0.02%`
  - costo guard su `big`: `Δ≈-0.1442` punti medi vs `heuristic_v1`
  - decisione storica: abilitato `inference_overkill_guard=true` nel best ufficiale dell'epoca
    `data/models/best_a2c.npz` e registrata la decisione in `data/models/best_a2c.json`

Prossimo step (shaping mirato su “spreco briscole alte”):
- [x] A2C: aggiungere shaping opzionale `--overkill-penalty-beta` (penalità flat) quando:
  - siamo secondi di mano
  - vinciamo con una briscola
  - esisteva una briscola vincente più economica
  - (default) carta avversaria sul tavolo vale <=2 punti (`--overkill-low-lead-points-max`)
- [x] Sweep rapido (screening):
  - warm-start da `data/models/best_a2c.npz`
  - 50k game, benchmark `small`
  - provare `beta ∈ {0.0, 0.002, 0.005, 0.01}`
  - criterio: ridurre `trump_overkill_rate_low_lead_points` senza perdere troppo `avg_diff vs heuristic_v1`

Risultati sweep (seed training=6, 50k game, benchmark decision-quality=small vs `heuristic_v1`, seed eval=0):
- `beta=0.0` (`..._okb0`): `avg_diff=+12.63`, overkill `19.58%`, low-lead `16.09%` (407/2530)
- `beta=0.002` (`..._okb2e3`): `avg_diff=+12.64`, overkill `21.31%`, low-lead `18.38%` (462/2513)
- `beta=0.005` (`..._okb5e3`): `avg_diff=+12.26`, overkill `19.56%`, low-lead `16.62%` (413/2485)
- `beta=0.01` (`..._okb1e2`): `avg_diff=+13.56`, overkill `18.87%`, low-lead `17.09%` (422/2469)

Conclusione (per ora):
- La penalità flat, in questo sweep “veloce”, NON riduce in modo affidabile `trump_overkill_rate_low_lead_points`.
- Possibili next step:
  - aumentare durata (es. 200k) e valutare su `medium` (meno rumore);
  - cambiare forma penalità: proporzionale al “gap” (es. differenza strength/punti tra briscole vincenti);
  - introdurre una seconda penalità: “trump_on_low_value_trick” (anche quando non è overkill) se il valore sul tavolo è basso.

Prossimo step (shaping “gap”, più informativo):
- [x] A2C: aggiungere `--overkill-penalty-mode gap`:
  - penalità = `-beta * gap_norm`, dove `gap_norm` misura quanto la briscola scelta è “più costosa” della briscola vincente minima
- [x] Mini-sweep (benchmark `medium`):
  - 50k game, seed training=6, warm-start da `best_a2c`
  - confrontare `beta ∈ {0.0, 0.005, 0.01}` in modalità `gap`
  - criterio: ridurre `trump_overkill_rate_low_lead_points` senza peggiorare troppo `avg_diff vs heuristic_v1`

Risultati mini-sweep “gap” (seed training=6, 50k game, decision-quality `medium` vs `heuristic_v1`, seed eval=0):
- `beta=0.0` (`..._okbgap0`): `avg_diff=+12.21`, overkill `19.16%`, low-lead `16.51%` (2051/12420)
- `beta=0.005` (`..._okbgap5e3`): `avg_diff=+11.86`, overkill `21.33%`, low-lead `19.69%` (2450/12441)
- `beta=0.01` (`..._okbgap1e2`): `avg_diff=+12.29`, overkill `20.65%`, low-lead `17.98%` (2208/12280)

Conclusione:
- Anche la penalità “gap”, in questo setting, NON migliora la metrica `trump_overkill_*` (tende anzi a peggiorarla).
- Quindi conviene cambiare approccio: shaping diverso (es. penalità per “giocare briscola su low-value” anche quando non è overkill)
  oppure intervenire direttamente in inference (post-processing: scegliere la briscola vincente minima tra le top-k azioni del modello).

Roadmap breve (per un modello ancora più “strategico”)
-----------------------------------------------------

Questa roadmap è pensata per fare un passo avanti “vero” rispetto a:
- shaping (che qui non sta funzionando bene), e
- post-processing (utile, ma non sostituisce uno stato/teacher migliori).

Piano (ordine consigliato, 1→2→3):
- [x] (1) Implementare un teacher più forte: `heuristic_v2` (card counting lecito + gestione briscole)
  - usa `PlayerObservation.seen_cards_onehot[40]` (info pubblica) per stimare fase e risorse rimaste
  - regole più “da umano”: conservazione briscole alte, evitare sprechi in early game, aggressività in late game
  - obiettivo: usare `heuristic_v2` come avversario e come teacher per BC
- [x] (2) Generare un dataset BC “pulito” via self-play del teacher:
  - `scripts/self_play_to_db.py` con `--agents heuristic_v2,heuristic_v2` (o mix)
  - `scripts/export_dataset.py` → JSONL
  - `scripts/train_bc.py --encoder-version v2` (consigliato `--model mlp`) per ottenere `bc_teacher_v2.npz`
- [x] (3) Fine-tuning A2C (encoder v2) partendo da BC:
  - init = `bc_teacher_v2.npz`, opponent mix più robusto (`heuristic_v1` + `heuristic_v2` + baseline)
  - valutare con `evaluate_matrix.py` + `evaluate_decision_quality.py` (forza + stile)

Esperimento guidato (A→B→C): “imparare davvero” a ridurre l’overkill (senza guard)
-------------------------------------------------------------------------------

Obiettivo pratico:
- ottenere un `.npz` che riduce `trump_overkill_rate` *perché lo ha imparato* (dataset/ottimizzazione),
  non perché lo forziamo con `inference_overkill_guard`.

Nota importante:
- per misurare l’apprendimento, le valutazioni qui vanno fatte con guard OFF
  (non impostare `BRISCOLA_BC_OVERKILL_GUARD` e non salvare `inference_overkill_guard` nel modello).

Piano:
- [x] (A) Generare un dataset BC da `heuristic_v2` (self-play) e allenare un primo modello BC v2
  - output pesante (DB + JSONL) in temp dir, poi cleanup per mantenere repo “minimale”
  - output modello: `benchmarks/experiments/bc_teacher_v2_seed42/bc_teacher_v2.npz`
- [x] (B) Valutare lo stile del BC (guard OFF) con `evaluate_decision_quality.py` (benchmark `medium`)
  - output JSON: `benchmarks/experiments/bc_teacher_v2_seed42/decision_quality_medium.json`
- [x] (C) Fine-tuning A2C inizializzato dal BC (encoder v2) + evaluation matrix `medium`
  - `scripts/run_experiment.py --algo a2c --init <bc_teacher_v2.npz> --benchmarks medium ...`
  - poi decision quality `medium` (guard OFF) per verificare se RL conserva o peggiora lo stile

Risultati (esecuzione completa A→B→C)
------------------------------------

(A) Self-play + export + BC (teacher `heuristic_v2`)
- self-play: 5000 partite (seed=42) → 200k azioni
- train BC (MLP, encoder v2, 10 epoche): `benchmarks/experiments/bc_teacher_v2_seed42/bc_teacher_v2.npz`
  - training/val acc: ~0.97 / ~0.96 (vedi `train_bc.log`)

(B) Decision quality (BC vs `heuristic_v1`, benchmark `medium`, seed=0, guard OFF)
- match: avg diff punti (A-B) `+2.63`
- `trump_overkill_rate`: `0.4%` (96 / 24144)
- `trump_overkill_rate_low_lead_points`: `0.5%` (68; vedi JSON)

(C) A2C init da BC (encoder v2) + eval matrix `medium` (guard OFF)
- esperimento: `benchmarks/experiments/a2c_mix_heuristic_v2_0_4_heuristic_v1_0_3_random_0_2_greedy_points_0_1_200kg_seed6_from_bc_teacher_v2/`
- evaluation matrix `medium`:
  - vs `heuristic_v1` holdout: `avg_diff=+8.83`
- decision quality `medium` vs `heuristic_v1`:
  - avg diff `+8.65`
  - `trump_overkill_rate`: `1.5%`
  - `trump_overkill_rate_low_lead_points`: `0.3%`

Interpretazione:
- Il BC “impara davvero” lo stile anti-overkill (da ~20% → <1% senza guard).
- Il fine-tuning A2C migliora la forza vs `heuristic_v1`, ma tende a rialzare un po’ l’overkill complessivo.
  Questo è un buon segnale: lo stile è acquisito, ma l’obiettivo RL (reward) può spingere di nuovo verso mosse più “aggressive”.

Prossimo step (D): A2C “ancorato” al BC (stay-close-to-teacher)
---------------------------------------------------------------

Obiettivo:
- mantenere i vantaggi dello stile BC (anti-overkill) durante il fine-tuning RL, senza ricorrere al guard.

Idea:
- aggiungere a `scripts/train_a2c.py` una regolarizzazione opzionale verso un modello BC fisso:
  - loss addizionale (actor): `beta * CE(π_anchor || π_policy)` su azioni valide (action mask)
  - gradiente semplice: `beta * (π_policy - π_anchor)` sui logits

Deliverable:
- [x] flag CLI: `--bc-anchor <path.npz>` + `--bc-anchor-beta <float>`
- [x] test unitari per gradiente CE (`tests/test_policy_regularization.py`)
- [x] run di prova (200k game) init da BC con anchor attivo, e confronto:
  - `evaluate_matrix.py` (forza)
  - `evaluate_decision_quality.py` (stile, guard OFF)

Risultati run di prova (anchor attivo, beta=0.02)
- esperimento: `benchmarks/experiments/a2c_mix_heuristic_v2_0_4_heuristic_v1_0_3_random_0_2_greedy_points_0_1_200kg_seed7_from_bc_teacher_v2_anchor02/`
- evaluation matrix `medium`:
  - vs `heuristic_v1` holdout: `avg_diff=+5.85`
- decision quality `medium` vs `heuristic_v1` (guard OFF):
  - avg diff `+5.55`
  - `trump_overkill_rate`: `0.6%`
  - `trump_overkill_rate_low_lead_points`: `0.5%`

Interpretazione:
- l'anchor aiuta a tenere basso l'overkill (vs A2C non ancorato), ma con questo `beta` sembra “frenare” troppo la policy,
  riducendo la forza vs `heuristic_v1`. Prossimo tuning naturale: provare `beta` più piccoli (es. 0.005–0.01) e confrontare.

Risultati aggiornati (2026-06-08): teacher v2 → BC v2 → A2C v2 strength
-----------------------------------------------------------------------------

Obiettivo:
- verificare se il percorso teacher v2 + encoder v2 può avvicinarsi al best 5M mantenendo migliore stile raw
  (meno overkill anche senza guard).

Dataset/BC:
- smoke: 200 partite `heuristic_v2,heuristic_v2` → 8000 esempi, BC MLP v2 val acc `0.797`
- dataset serio: 10000 partite (seed=43) → 400000 esempi, zero observation mancanti
- BC serio: `benchmarks/experiments/bc_teacher_v2_seed43_10k/bc_teacher_v2_10k.npz`
  - MLP encoder v2, 25 epoche, val acc `0.969`
  - forza: `+2.18` vs `heuristic_v1` medium; circa pari al teacher (`+0.29` vs `heuristic_v2`)
  - qualità: `trump_overkill_rate=0.0%` con guard ON; teacher `heuristic_v2` ha overkill `0.0%` anche nativo

Fine-tuning A2C v2:
- candidato 200k con anchor BC (`beta=0.01`):
  - esperimento: `a2c_v2_from_bc_teacher_v2_10k_seed44_200k_numba`
  - `big holdout vs heuristic_v1 = +6.59`
  - head-to-head vs best 5M: `avg_point_diff=-7.30`
  - decisione: non promuovere, ma conferma che il BC è un init utile
- candidato 1M con anchor leggero (`beta=0.003`):
  - esperimento: `a2c_v2_from_bc_teacher_v2_10k_seed45_1m_numba`
  - `big holdout vs heuristic_v1 = +10.32`
  - head-to-head vs best 5M: `avg_point_diff=-3.74`
  - guard OFF: `trump_overkill_rate=3.6%`, low-lead `1.2%`
  - decisione: non promuovere; buon miglioramento di stile raw
- candidato strength +1M senza anchor:
  - esperimento: `a2c_v2_strength_from_teacher_seed46_1m_numba`
  - `big holdout vs heuristic_v1 = +12.87`
  - head-to-head vs best 5M: `avg_point_diff=-1.22`
  - guard OFF: `trump_overkill_rate=8.8%`, low-lead `4.3%`
  - decisione: non promuovere; recupera forza ma perde parte dello stile raw
- candidato strength +1M con più pressione su `best_a2c`:
  - esperimento: `a2c_v2_strength_from_teacher_seed47_1m_numba`
  - `big holdout vs heuristic_v1 = +13.78` (best 5M ufficiale: `+13.91112`)
  - head-to-head vs best 5M, seat-fair 100k: `avg_point_diff=+0.10`; su suite indipendenti seat-fair:
    `+0.15` e `-0.01` (sostanzialmente pari)
  - guard OFF: `trump_overkill_rate=8.7%`, low-lead `5.7%`
  - decisione: NON promuovere per ora; score ufficiale ancora leggermente sotto al best 5M e head-to-head non abbastanza netto
  - artefatto UI locale: `data/models/a2c_v2_teacher_strength_seed47_1m_candidate.npz`
    (copia compatta, encoder v2, guard ON)
- candidato strength +5M dal checkpoint seed47:
  - esperimento: `benchmarks/experiments/a2c_v2_strength_from_teacher_seed48_5m_numba/`
  - warm-start: `benchmarks/experiments/a2c_v2_strength_from_teacher_seed47_1m_numba/model.npz`
  - config: `lr=1e-4`, `entropy_beta=2e-4`,
    mix `best_a2c:0.65,heuristic_v2:0.20,heuristic_v1:0.05,greedy_points:0.05,random:0.05`, seat-fair,
    encoder v2, rollout/eval Numba, guard ON
  - tempo misurato con `/usr/bin/time -p`: `real=1041.12s` per training 5M + matrix `medium,big`
  - evaluation Numba: `medium=4.692s`, `big=45.118s`
  - score ufficiale: `big holdout vs heuristic_v1 avg_diff = +16.39058`
  - confronto col best precedente seed19: `+13.91112 -> +16.39058` (`+2.47946`, circa `+17.8%`
    sul vantaggio medio)
  - head-to-head vs best precedente seed19 (`big`, 100k, seat-fair): `avg_point_diff=+3.76068`;
    suite indipendenti seat-fair: `+3.66524` e `+3.55502`
  - decision-quality `big` vs `heuristic_v1`:
    - guard ON: `avg_diff=+16.49182`, `trump_overkill_rate=0.0%`, low-lead `0.0%`, `trump_waste_rate≈0.05%`
    - guard OFF: `avg_diff=+16.58308`, `trump_overkill_rate=9.8%`, low-lead `5.1%`, `trump_waste_rate≈0.05%`
  - validazione extra `big` vs `heuristic_v2` (nuovo teacher/avversario più strategico):
    - matrix Numba: standard `avg_diff=+13.26248`, holdout `avg_diff=+13.28656`
    - decision-quality Numba: `avg_diff=+13.39930`, `trump_overkill_rate=0.0%`,
      low-lead `0.0%`, `trump_waste_rate≈0.06%`
  - smoke UI: la UI espone il modello come `A2C v2 strength 5M (best) (best_a2c.npz)`,
    mostra guard attivo/metriche e permette di avviare una partita contro `bc_model` senza errori console
  - decisione: promosso nuovo `data/models/best_a2c.npz` (encoder v2, guard ON, copia compattata)
  - storage: modello completo esperimento `245 MB`; best runtime compattato `157 KB` con `metrics_summary`
    e sidecar `data/models/best_a2c.json`
- replica strength +5M con seed diverso dal checkpoint seed47:
  - primo tentativo: `benchmarks/experiments/a2c_v2_strength_from_teacher_seed49_5m_numba/`
    si è fermato a `4.388M/5M` senza `model.npz`/`manifest.json`; resta solo come log parziale, non come candidato
  - retry completo: `benchmarks/experiments/a2c_v2_strength_from_teacher_seed49_5m_numba_retry1/`
  - stessa config seed48, ma `train_seed=49`; warm-start da
    `benchmarks/experiments/a2c_v2_strength_from_teacher_seed47_1m_numba/model.npz`
  - tempo misurato con `/usr/bin/time -p`: `real=1084.55s` per training 5M + matrix `medium,big`
  - evaluation Numba: `medium=4.877s`, `big=40.001s`
  - score ufficiale: `big holdout vs heuristic_v1 avg_diff = +15.91892`
    (baseline/promozione seed48: `+16.39058`)
  - head-to-head vs best corrente seed48 (`100k`, seat-fair):
    `avg_point_diff=+0.17608`; suite indipendenti: `+0.00512` e `+0.14716`
  - decision-quality `big` vs `heuristic_v1`:
    - guard ON: `avg_diff=+15.92230`, `trump_overkill_rate=0.0%`, low-lead `0.0%`, `trump_waste_rate≈0.06%`
    - guard OFF: `avg_diff=+16.06572`, `trump_overkill_rate=19.0%`, low-lead `11.5%`, `trump_waste_rate≈0.06%`
  - validazione extra `big` vs `heuristic_v2`: `avg_diff=+12.78530`
  - decisione: NON promuovere; il modello è sostanzialmente pari in H2H ma sotto il criterio ufficiale e più miope raw
    senza guard
- fine-tuning mirato anti-overkill dal best seed48:
  - esperimento: `benchmarks/experiments/a2c_v2_best_overkill_gap001_1m_seed50_numba/`
  - warm-start: `data/models/best_a2c.npz` precedente (`A2C v2 strength 5M seed48`)
  - config: `1M` partite, `train_seed=50`, `lr=5e-5`, `entropy_beta=2e-4`,
    mix `best_a2c:0.65,heuristic_v2:0.20,heuristic_v1:0.05,greedy_points:0.05,random:0.05`, seat-fair,
    encoder v2, rollout/eval Numba, guard ON, shaping anti-overkill `mode=gap`, `beta=0.01`
  - tempo misurato con `/usr/bin/time -p`: `real=290.81s` per training 1M + matrix `medium,big`
  - evaluation Numba: `medium=4.907s`, `big=63.416s`
  - score ufficiale: `big holdout vs heuristic_v1 avg_diff = +16.77358`
  - confronto col best seed48: `+16.39058 -> +16.77358` (`+0.38300`, circa `+2.3%`
    sul vantaggio medio)
  - head-to-head vs best seed48 (`100k`, seat-fair): `avg_point_diff=+0.76442`;
    suite indipendenti seat-fair: `+0.63712` e `+0.73228`
  - decision-quality `big` vs `heuristic_v1`:
    - guard ON: `avg_diff=+16.84008`, `trump_overkill_rate=0.0%`, low-lead `0.0%`, `trump_waste_rate≈0.06%`
    - guard OFF: `avg_diff=+16.96330`, `trump_overkill_rate=12.5%`, low-lead `4.7%`, `trump_waste_rate≈0.06%`
  - validazione extra `big` vs `heuristic_v2`:
    - matrix Numba: standard `avg_diff=+13.69230`, holdout `avg_diff=+13.72108`
    - decision-quality Numba: `avg_diff=+13.71820`, `trump_overkill_rate=0.0%`,
      low-lead `0.0%`, `trump_waste_rate≈0.06%`
  - decisione: promosso nuovo `data/models/best_a2c.npz` (`A2C v2 overkill gap 1M (best)`, encoder v2,
    guard ON, copia compattata)
  - nota qualità raw: senza guard l'overkill totale peggiora rispetto al seed48 (`9.8% -> 12.5%`), ma il low-lead
    migliora leggermente (`5.1% -> 4.7%`) e il runtime ufficiale usa guard ON; continuiamo a tracciare entrambe le viste

Conclusione:
- Il percorso teacher v2 funziona anche in forza, non solo nello stile: il run 5M seed48 supera chiaramente il
  precedente best v1/seed19 sia sul criterio ufficiale (`big holdout vs heuristic_v1`) sia in head-to-head.
- Il nuovo best ufficiale è un A2C encoder v2 con guard anti-overkill attivo e shaping gap in training; senza guard il
  modello raw resta lontano dal vecchio best storico (~12.5% overkill vs ~20%), ma il guard resta utile per azzerare
  l'overkill in UI/runtime.
- Baseline congelata: da ora un candidato deve battere almeno `+16.77358` su `big holdout vs heuristic_v1`,
  risultare positivo in head-to-head contro questo `best_a2c` su 100k partite seat-fair più almeno una suite indipendente,
  e non peggiorare materialmente le metriche `trump_waste_rate`/`trump_overkill_rate`.
- Replica 5M con seed diverso completata: conferma che il best seed48 era robusto, ma il fine-tuning mirato seed50 lo supera.
- Prossimo step consigliato: fare esperimenti mirati invece di ripetere run 5M identici, ad esempio PPO/GAE oppure
  un tuning più fine dello shaping anti-overkill (`beta`/scope low-lead) per ridurre la dipendenza dal guard senza perdere forza.

### Fase 5G — Strategia esplicita: endgame esatto + feature strategiche (proposta)

Obiettivo: trasformare alcune euristiche “da giocatore umano” in componenti testabili, senza far barare il modello e senza
buttare via la pipeline A2C/BC già funzionante.

Contesto:
- Il progetto ha già una memoria pubblica lecita (`seen_cards_onehot`) e un encoder v2.
- `seen_cards_onehot` però non è un “cimitero” puro: include anche la briscola scoperta, quindi va bene per card counting
  pubblico, ma non basta da solo per dedurre con precisione quali carte siano definitivamente fuori gioco.
- Il best attuale è forte, ma usa ancora un guard inference anti-overkill: vogliamo ridurre progressivamente la dipendenza
  da post-processing e rendere più spiegabile il comportamento raw.

Piano consigliato (ordine):
1. [x] **Endgame solver esatto (minimax 2-player)**:
   - modulo `src/briscola_ai/ai/endgame_solver.py`: `solve_endgame(state) -> EndgameSolution`
     (con `best_card_index`, `final_delta_p0_p1`, `principal_variation`);
   - minimax completo a `deck_size == 0`: il player 0 massimizza `players[0].points - players[1].points`,
     il player 1 lo minimizza (zero-sum, totale 120); tie-break deterministico sull'indice più basso;
   - tutte le transizioni passano da `domain.step` (nessuna regola duplicata); memoization su `GameState` (frozen/hashable);
   - supporta tavolo vuoto e "secondo di mano" (`len(table_cards) == 1`);
   - guard "strict": 2 giocatori, partita non finita, mazzo vuoto, ≤ 6 carte residue, mani bilanciate;
   - **oracolo di dominio** (vede entrambe le mani): diventerà agent-safe nello step 2 ricostruendo lo stato da `PlayerObservation`;
   - test (`tests/test_endgame_solver.py`): valore esatto a 1 presa, scelta strategica "il tempo conta" (conservare/incassare),
     polarità `current_turn == 1`, pareggio, secondo-di-mano, coerenza principal variation, partita reale fino a mazzo vuoto
     (somma punti = 120), e guard sugli stati fuori scope.
2. **Agente ibrido per UI/evaluation**:
   - [x] agente `hybrid_endgame` registrato in `build_agent`/`list_agent_specs`;
   - [x] early/mid game: fallback `heuristic_v2`;
   - [x] endgame (`deck_size == 0`): ricostruisce uno `GameState` dalla sola `PlayerObservation` e delega al solver esatto;
   - [x] ricostruzione anti-cheat: deduce la mano avversaria da `seen_cards_onehot`, mano propria, tavolo e dimensioni mani;
   - [x] punti/prese azzerati nello stato ricostruito (preserva l'argmax/argmin ed evita il ricalcolo punti errato in `domain.step`);
   - [x] fallback difensivo su osservazioni fuori scope/incoerenti;
   - [x] test (`tests/test_hybrid_endgame_agent.py`): base punti azzerata, briscola in mano avversaria, secondo di mano,
     fallback pre-endgame/incoerente, partita reale fino a endgame, registrazione catalogo agenti;
   - [x] benchmark seat-fair `medium` (engine `domain`, seed suite versionata `medium` = 5000 seed → 10000 partite,
     commit `95d546c`, modello `data/models/best_a2c.npz` encoder v2 + overkill_guard on):
     - vs `heuristic_v2` (proprio fallback): win 5508 / 4236 / 256 draw, **avg diff +4.06** → il solver endgame aggiunge forza reale;
     - vs `best_a2c` (`bc_model`): win 3598 / 6121 / 281 draw, **avg diff -9.04** → il mid-game `heuristic_v2` resta il collo di bottiglia.
   - [x] decision-quality `medium` (engine `domain`, stessa seed suite, metriche sul player A = hybrid):
     - vs `heuristic_v2`: avg diff +4.03; trump_waste 0.1%, trump_overkill 4.1%, overkill low-lead 0.2%;
     - vs `best_a2c`: avg diff -9.21; trump_waste 0.1%, trump_overkill 3.9%, overkill low-lead 2.0%.
   - Output JSON locali (gitignored) in `benchmarks/experiments/hybrid_endgame_*medium.json`.
   - **Conclusione (fallback heuristic_v2)**: il solver endgame migliora nettamente la baseline euristica (+4 vs `heuristic_v2`)
     ma non basta a colmare il gap mid-game con `best_a2c`: il collo di bottiglia è la policy mid-game.
   - [x] variante `hybrid_endgame_best_a2c` (fallback = `best_a2c`, mid-game forte + finale esatto), catalogata a parte
     per non toccare `hybrid_endgame`. Helper condiviso `_load_best_a2c_agent()`; test in `tests/test_hybrid_endgame_agent.py`.
     Benchmark `medium` (engine `domain`, seed suite `medium`, commit successivo a `bbc6460`, modello `data/models/best_a2c.npz`):
     - vs `best_a2c` puro (`medium`): win 5124 / 4557 / 319 draw, **avg diff +1.83** (51.2% win) → il finale esatto aggiunge
       valore anche sopra una policy mid-game forte;
     - decision-quality vs `best_a2c` (`medium`, riferimento): avg diff +1.91; trump_waste 0.2%, trump_overkill 8.0%,
       overkill low-lead 6.0% (l'overkill più alto riflette lo stile raw di `best_a2c` ora usato in mid-game, non il solver).
   - [x] consolidamento `big` (engine `domain`, seed suite `big` via range, 100000 partite seat-fair, ~3m35s):
     - vs `best_a2c` puro: win 51459 / 45431 / 3110 draw, **avg diff +1.90** (51.5% win) → segnale stabile, coerente col `medium`.
   - **Esito**: `hybrid_endgame_best_a2c` è una **candidata chiara** come nuova baseline UI/evaluation (positiva e stabile su 100k).
   - [x] **promossa a baseline consigliata** per UI/evaluation (label "consigliato" + nota nel `description_it` esposto dal catalogo).
     `best_a2c` resta disponibile. Il default server (`_DEFAULT_AI_AGENT_NAME`) resta `random` di proposito: cambiarlo a
     `hybrid_endgame_best_a2c` lo renderebbe dipendente dalla presenza di `best_a2c.npz` → eventuale follow-up opzionale.
   - Possibili next step: eventuale mitigazione dell'overkill mid-game di `best_a2c` (guard/anchor) — non urgente, misura il fallback
     non il solver.
3. [x] **Distinguere memoria pubblica da carte fuori gioco**:
   - [x] campo `out_of_play_cards_onehot[40]` aggiunto a `PlayerObservation` (default 40 zeri, backward-compatible);
   - [x] definizione: SOLO prese (di tutti) + tavolo; la briscola scoperta NON è fuori gioco finché è pescabile o in mano
     (ci finisce solo quando viene catturata/giocata). Invariante: `out_of_play ⊆ seen`;
   - [x] popolato sia nel dominio (`make_player_observation`) sia nel builder DTO (`build_observation_dto`);
   - [x] DTO: `out_of_play_cards_onehot: list[int] = Field(default_factory=list)` (payload/dataset vecchi restano validi);
     export passa l'observation come dict → additivo automatico, nessuna modifica a `export_dataset.py`;
   - [x] encoder v1/v2 **non** modificati (leggono solo `seen`); il campo è preservato nei record per l'encoder v3 futuro;
   - [x] `reconstruct_endgame_state` ora preferisce `out_of_play` (deduzione diretta: `opp = tutte − mia_mano − out_of_play`,
     niente trucco sulla briscola) con **fallback** su `seen` quando il campo è assente/azzerato/incoerente → niente migrazione "tutto o niente";
   - [x] test: semantica nei 5 casi briscola/tavolo/prese (`tests/test_out_of_play_observation.py`), DTO backward-compat
     (`tests/test_dto.py`), ricostruzione via `out_of_play` con `seen` azzerato e fallback su `seen` con `out_of_play` azzerato
     (`tests/test_hybrid_endgame_agent.py`).
   - `seen_cards_onehot` resta invariato come "informazione pubblica vista".
4. **Encoder v3 con feature strategiche aggregate** (domain-first, no promozione modello in questo step):
   - [x] `FEATURE_DIM_2P_V3 = 310` = v2 (288) + **22** feature aggregate, blocco congelato:
     - `unknown_trumps_count_norm` (1) + briscole alte ignote Asso/Tre/Re (3);
     - per seme (×4): `ace_out_of_play`, `three_out_of_play`, `unknown_load_count_norm` (12);
     - fase: `deck_size_norm`, `my_hand_size_norm`, `is_endgame` (3);
     - presa corrente: `current_trick_points_norm`, `current_trick_lead_strength_norm`, `current_trick_lead_is_trump` (3).
   - [x] definizione "ignota" anti-cheat: `unknown = not seen and not in_my_hand`; poiché la briscola scoperta è sempre in
     `seen`, è esclusa correttamente dalle ignote **senza** usarne l'id (parità dict/oggetto garantita anche a mazzo vuoto,
     dove il DTO azzera `trump_card`). Le feature `*_out_of_play` usano `out_of_play_cards_onehot` (step 3).
   - [x] implementato in entrambi i path: `encode_observation_2p_v3` (dict/DTO) e ramo v3 di `encode_player_observation_2p`
     (oggetto), con `_compute_v3_extra_features` condiviso; selettore + `feature_dim_for_encoder_version` aggiornati.
   - [x] compatibilità: `bc_model_agent` (inferenza v3 da metadata/feature_dim + validazione coerenza), `model_catalog`
     (UI accetta 310), `_load_best_a2c_agent` (set feature_dim += v3); label/metadata `encoder=v3`.
   - [x] **guard domain-first**: encoder fast (`fast_observation_encoder`) e numba (`fast_numba_observation`) rifiutano v3
     con errore chiaro (niente fallback a v2); `train_a2c` blocca `--encoder-version v3` con `--rollout-engine fast/numba`.
   - [x] training: `--encoder-version v3` esposto in `train_a2c/train_bc/train_pg` (path domain).
   - [x] test: contratto dim/prefisso v2, parità dict-oggetto (mid-game + endgame), briscola scoperta non-ignota,
     `*_out_of_play` da prese, `is_endgame`, guard fast/numba, roundtrip modello v3 + mismatch metadata
     (`tests/test_observation_encoder_v3.py`).
   - [ ] (prerequisito training v3 BC) re-export dataset con `out_of_play` popolato (dataset vecchi → feature degradate).
   - [ ] confrontare v3 vs v2 con la pipeline (`run_experiment.py`, `evaluate_*`) — quando si addestra un modello v3.
   - [ ] (follow-up) parità v3 su path fast/numba per training/eval ad alto throughput.
5. **Teacher endgame-aware per BC/RL** (primo ciclo eseguito; **nessuna promozione**):
   - teacher = `hybrid_endgame_best_a2c` (mid-game best_a2c + solver esatto nel finale).
   - smoke test pipeline v3 (`tests/test_pipeline_v3_smoke.py`): export preserva `out_of_play` (endgame non banale),
     data path BC v3 = 310 feature, `train_bc --encoder-version v3` → `.npz` caricabile come v3.
   - **Nota dataset**: generato **teacher-vs-teacher** (stesso agente in entrambi i seat), quindi è clonazione del teacher,
     **non** un mix di avversari. Export con `player_index=0` (un seat; entrambi sono comunque il teacher).
   - run di validazione (1k dataset, BC mini, A2C 20k, eval small): metadata/catalogo/load/eval/metriche OK (`encoder=v3`, 310).
   - run serio (engine `domain`, seed=0): dataset 20k partite (800k azioni) → BC v3 MLP h128 20 epoche (val acc ~0.92)
     → fine-tuning A2C v3 200k partite, opponent-mix `heuristic_v1:0.6,heuristic_v2:0.3,random:0.1`, warm-start dal BC.
   - risultati eval (medium/holdout, domain):
     - vs `best_a2c` head-to-head (medium): **avg diff -2.14** (5126/4593/281) → non supera il best;
     - vs `heuristic_v1` (holdout range-start 500000): **+15.82**, contro `best_a2c` **+16.56** sulla stessa suite → leggermente sotto;
     - decision-quality vs `heuristic_v1` (medium): avg diff +16.55; trump_waste 0.2%, **trump_overkill 8.6%** (ereditato dallo stile di best_a2c).
   - **decisione**: criteri di promozione non soddisfatti (head-to-head negativo, holdout sotto il best, overkill più alto).
     `best_a2c` resta il best ufficiale. Atteso: best_a2c viene da training a scala molto maggiore (1M+); 200k domain non colmano il gap.
   - prossimi tentativi possibili: (a) scala maggiore del fine-tuning A2C (richiede parità v3 fast/numba per throughput);
     (b) teacher `hybrid_endgame` (heuristic_v2) per ridurre l'overkill ereditato; (c) BC-anchor nel fine-tuning per restare
     vicino al teacher; (d) opponent mix con `best_a2c` per allenarsi contro il best.

5a. **Follow-up (a) — parità encoder v3 su fast/numba** (commit `e99684a`, `19b19d2`, `1413b12`): COMPLETATO.
   - fast Python e numba ora supportano v3; parità encoder domain==fast (esatta) e domain==numba (30 partite, 1200 stati,
     max diff feature 2.8e-08); rollout fast-Python e collector numba costruiscono `out_of_play` (marca alla giocata, no
     briscola iniziale); guard rimossi per-path solo dopo i test di parità. Smoke: `train_a2c v3 --rollout-engine fast
     --fast-rollout numba` → modello v3/310.

5b. **Benchmark throughput v3** (commit `1413b12`, Mac locale):
   - **A2C training** (20k partite, init BC v3, opponent-mix step 5): `--rollout-engine domain` **~419 games/sec** (47.8s)
     vs `--rollout-engine fast --fast-rollout numba` **~5900 games/sec** (3.4s) → **~14× speedup** (sblocca run ≥1M).
   - **Evaluation domain** (`benchmark_perf.py --mode eval`, 10k×2): encoder v2 (`best_a2c`) **~871 games/sec**
     vs encoder v3 (`a2c_v3`) **~721 games/sec** (v3 ~17% più lento: 22 feature in più + costruzione `out_of_play`).
   - nota: `benchmark_perf.py --mode numba-eval` non supporta `bc_model` (solo rule-based), quindi il throughput numba dei
     modelli è misurato sul training A2C sopra.

5c. **Run sperimentale v3 scalato** (commit `1413b12`; **nessuna promozione**):
   - setup identico allo step 5 tranne scala+engine: stesso dataset teacher 20k → stesso BC v3 (`bc_v3.npz`), fine-tuning A2C v3
     `--rollout-engine fast --fast-rollout numba`, opponent-mix invariato, **1M partite**, seed nuovo 100 (~2m25s).
   - head-to-head vs `best_a2c` (medium, domain): **avg diff -1.83** (5108/4620/272) → migliora il -2.14 dello step 5 di +0.31,
     ma **non diventa positivo**.
   - criterio non soddisfatto (gate "se positivo → big/promozione"): **stop**, niente `big`, niente promozione. `best_a2c` resta il best.
   - lettura: a parità di teacher/mix/encoder, 10× di scala (200k→1M) sposta pochissimo. Le leve da provare separatamente
     restano (b)/(c)/(d) — una variabile alla volta.

5d. **Leva (d) — opponent-mix con `best_a2c`** (1M fast+numba, seed 200; **nessuna promozione**):
   - unica variabile cambiata vs 5c: opponent-mix = `best_a2c:0.4,heuristic_v2:0.3,heuristic_v1:0.2,random:0.1`
     (stesso dataset/BC v3, encoder v3, 1M fast+numba). `best_a2c` come opponent `.npz` MLP nel mix è auto-risolto.
   - head-to-head vs `best_a2c`: medium **+0.13**, big (100k) **-0.17** → **parità** (da -2.14 step 5 / -1.83 in 5c): il gap è chiuso.
   - holdout vs `heuristic_v1` (range 500000): **+15.78** contro `best_a2c` **+16.56** → ancora sotto il best sull'holdout.
   - decision-quality vs `heuristic_v1` (medium): avg diff +16.63; trump_waste 0.2%, **trump_overkill 14.1%** (da 8.6% in 5c,
     vs ~4% di best_a2c), overkill low-lead 8.7%.
   - **esito**: criteri di promozione non soddisfatti — head-to-head non *positivo* (parità), holdout sotto il best, e overkill
     molto peggiore. Allenarsi contro `best_a2c` chiude il testa-a-testa ma rende il modello più aggressivo (overkill).
   - **prossima leva**: (b) teacher `hybrid_endgame` (meno overkill) e/o (c) BC-anchor per controllare l'overkill, mantenendo
     `best_a2c` nel mix (che ha dato la parità head-to-head). Cambiare una variabile alla volta.

5e. **Leva (c) — BC-anchor** (1M fast+numba, seed 300; **decisione promozione al maintainer**):
   - unica variabile cambiata vs 5d: aggiunto `--bc-anchor data/models/bc_v3.npz --bc-anchor-beta 0.01` (anchor = teacher
     distillato v3). Stesso dataset/BC/encoder/mix-con-best_a2c/1M.
   - head-to-head vs `best_a2c`: medium **+0.53**, big (100k) **+0.63** (49547/47458/2995) → **positivo e stabile**: batte il best.
   - holdout vs `heuristic_v1` (range 500000): **+17.23** contro `best_a2c` **+16.56** → **supera il best anche sull'holdout**.
   - decision-quality vs `heuristic_v1` (medium): avg diff +17.27; trump_waste 0.1%, trump_overkill **11.4%** (da 14.1% in 5d),
     overkill **low-lead 1.6%** (da 8.7%): l'overkill "cattivo" crolla; il generico resta sopra il best guarded (~4%).
   - **esito**: primo v3 a battere `best_a2c` su entrambi i criteri di forza (head-to-head + holdout); l'anchor riduce
     materialmente l'overkill low-lead mantenendo/superando la parità. Resta aperto solo il confronto overkill "raw" generico
     (11.4% del candidato senza guard vs ~4% di best_a2c CON guard a inference). Promozione = decisione maintainer
     (eventualmente abilitando l'inference-overkill-guard sul candidato, o un beta anchor più alto come 5f).
6. **PPO/GAE solo dopo baseline ibrida**:
   - mantenere A2C come default, perché è già integrato con Numba, opponent mix, BC-anchor e evaluation matrix;
   - usare PPO/GAE come spike mirato se A2C v3/endgame-aware si stabilizza ma mostra ancora regressioni;
   - non introdurre DQN per ora: action mask, self-play e parziale osservabilità rendono più utile continuare sulla linea policy-gradient già presente.

Criteri di successo:
- solver endgame: test dominio verdi + casi noti spiegabili;
- agente ibrido: nessuna regressione UI/API e benchmark seat-fair ripetibile;
- encoder v3: modello compatibile con catalogo UI e metadati `metadata.encoder="v3"`;
- training: candidato promosso solo se supera il best ufficiale su `big holdout vs heuristic_v1`, è positivo in head-to-head
  contro `best_a2c`, e migliora o non peggiora materialmente `trump_waste_rate`/`trump_overkill_rate` raw.

Risultati tuning anchor più debole (seed training=8, 200k game, benchmark `medium`, guard OFF)
- baseline senza anchor (`..._seed8_from_bc_teacher_v2_no_anchor`):
  - matrix holdout vs `heuristic_v1`: `avg_diff=+9.53`
  - decision quality vs `heuristic_v1`: `avg_diff=+9.87`, `trump_overkill_rate=4.1%`, low-lead `2.0%`
- anchor `beta=0.005` (`..._seed8_from_bc_teacher_v2_anchor005`):
  - matrix holdout vs `heuristic_v1`: `avg_diff=+8.52`
  - decision quality vs `heuristic_v1`: `avg_diff=+8.38`, `trump_overkill_rate=2.1%`, low-lead `1.7%`
- anchor `beta=0.01` (`..._seed8_from_bc_teacher_v2_anchor01`):
  - matrix holdout vs `heuristic_v1`: `avg_diff=+7.24`
  - decision quality vs `heuristic_v1`: `avg_diff=+7.64`, `trump_overkill_rate=1.4%`, low-lead `1.2%`

Decisione provvisoria:
- l'anchor funziona come regolarizzatore di stile, ma il costo in forza è netto anche con `beta=0.005`;
- il modello migliore per forza resta il baseline senza anchor, mentre `beta=0.005` è il compromesso più sensato se si vuole ridurre overkill senza guard;
- nessuno di questi modelli va promosso a `best_a2c` per ora: il best 1M ufficiale resta molto più forte e ha già il guard anti-overkill attivo.

### Fase 6 — Performance simulazioni/training

Obiettivo: aumentare il throughput delle simulazioni senza rompere il dominio didattico/canonico.

Baseline diagnostica (Mac locale, `decision_quality small`, 2000 game, best A2C vs `heuristic_v1`):
- profilo iniziale indicativo: ~4.8s sotto `cProfile`
- hotspot principali:
  - `BCModelAgent.choose_card_index` + encoder osservazioni
  - `make_player_observation`
  - `domain.step` con dataclass/tuple/`replace`

Interventi completati:
- [x] Aggiunto `scripts/benchmark_perf.py` per misurare `games/sec` in modo ripetibile
  - benchmark puro seat-fair 2000 game: circa `990-1000 games/sec` sul Mac locale
  - modalità engine-only aggiunte: `--mode domain-random`, `--mode fast-random` e `--mode numba-random`
- [x] Encoder veloce `PlayerObservation -> feature/mask`
  - evita la conversione intermedia `PlayerObservation -> dict DTO -> feature`
  - test di equivalenza v1/v2 contro il path DTO
- [x] `make_player_observation`: conversione carta->id ottimizzata nel path caldo
  - riduce chiamate/overhead su enum/dict durante costruzione `seen_cards_onehot`

Prossimi step performance (ordine consigliato):
- [x] Parallelizzare `evaluate_decision_quality.py` per seed chunk
  - CLI: `--workers N` (default seriale `1`)
  - benchmark `small` (2000 game): `2.20s` seriale vs `0.80s` con 4 worker (`~2.75x`)
  - benchmark `medium` (10000 game): `10.92s` seriale vs `3.32s` con 4 worker (`~3.3x`)
  - nota RNG: in parallelo usiamo RNG azioni indipendente per coppia seat-fair; per agenti deterministici
    coincide col seriale, per agenti stocastici resta riproducibile ma non byte-identico al vecchio stream seriale
- [x] Estendere la parallelizzazione a `evaluate_matrix.py`
  - CLI: `--workers N` (default seriale `1`)
  - pipeline: `scripts/run_experiment.py --eval-workers N`
  - benchmark `small` matrix completa (6 righe × 2000 game): `11.93s` seriale vs `4.51s` con 4 worker (`~2.65x`)
  - benchmark `medium` matrix completa con 4 worker: `20.47s` (seriale non rilanciato per risparmiare tempo)
- [x] Validare end-to-end `run_experiment.py --eval-workers 4`
  - run: `a2c_mix_best_a2c_0_60_heuristic_v1_0_25_greedy_points_0_10_random_0_05_50kg_seed19_perf_eval_workers4_50k`
  - training: 50k game, warm-start da `data/models/best_a2c.npz`, `--no-update-best --minimal-data`
  - matrix `medium` con 4 worker: `20.34s`
  - risultato holdout vs `heuristic_v1`: `avg_diff=+12.8516`
  - decisione: solo validazione performance; non candidato a promozione (`best_a2c` ufficiale resta invariato)
- [x] Spike iniziale `fast_2p` mutabile/array-based
  - mantenere `domain.step` come fonte canonica per API/test didattici
  - modulo: `src/briscola_ai/ai/fast_2p.py` (carte come `0..39`, stato mutabile, niente `Card/Enum/replace`)
  - test di equivalenza: stesso seed + stesse azioni -> stesso deck, mani, tavolo, turni, punti e vincitore
  - benchmark engine-only random, 20k game × 3 run:
    - dominio canonico: `~4.94k games/sec` medio
    - `fast_2p`: `~22.0k games/sec` medio
    - speedup indicativo: `~4.45x` sul solo motore random
- [x] Integrare `fast_2p` in evaluation dietro flag sperimentale per agenti fast-compatible
  - modulo: `src/briscola_ai/ai/fast_evaluation.py`
  - CLI: `scripts/evaluate_agents.py --engine fast`
  - supporto: `random`, `greedy_points`, `heuristic_v1`, `heuristic_v2` (no modelli `.npz`)
  - test di equivalenza aggregata: fast e dominio producono gli stessi `MatchStats`/`SeatFairStats`
  - benchmark `greedy_points` vs `random`, seat-fair 10k game:
    - dominio canonico: `~3.52s`
    - fast evaluation: `~0.496s`
    - speedup indicativo: `~7.1x`
  - benchmark `heuristic_v2` vs `random`, seat-fair 10k game:
    - dominio canonico: `~4.38s`
    - fast evaluation: `~0.614s`
    - speedup indicativo: `~7.1x`
- [x] Integrare `fast_2p` in self-play summary-only
  - modulo: `src/briscola_ai/ai/fast_self_play.py`
  - CLI: `scripts/fast_self_play.py`
  - supporto iniziale: `random`, `greedy_points`
  - output opzionale JSONL minimale per partita (seed, agenti, punti finali, vincitore), senza osservazioni/azioni step-by-step
  - test di equivalenza per-game contro dominio usando gli stessi `game_seed/action_seed`
  - benchmark `greedy_points` vs `random`, 100k game senza JSONL: `5.478s` (`~18.3k games/sec`)
- [x] Integrare `fast_2p` in training A2C dietro flag sperimentale
  - CLI: `scripts/train_a2c.py --rollout-engine fast`
  - supporto: policy A2C neurale vs `random`/`greedy_points`/`heuristic_v1`/`heuristic_v2`
  - encoder: `Fast2PState -> feature/mask` equivalente al path canonico (`v1`/`v2`)
  - limite storico superato nei passi successivi: modelli `.npz` opponent e shaping overkill sono supportati nel
    fast rollout Numba
  - smoke test: training fast salva modello `.npz` con metadato `rollout_engine=fast`
  - benchmark A2C vs `random`, 5k game, hidden=32:
    - dominio canonico: `7.109s`
    - fast rollout: `5.138s`
    - speedup indicativo: `~1.38x` (il resto del tempo è forward/backprop NumPy)
- [x] Valutare Numba solo sul `fast_2p`
  - Numba ha senso su stato numerico/array, non su dataclass/Enum/oggetti `Card`
  - dipendenze aggiunte: `numba>=0.65.1` + `llvmlite`
  - modulo iniziale: `src/briscola_ai/ai/fast_numba.py`
  - scope iniziale: core random-vs-random 2-player compilato JIT, con seed deterministico e invarianti punti/vincitore
  - test: determinismo per seed, somma punti = 120, conteggi aggregati coerenti
  - benchmark engine-only random, 100k game x 3 run:
    - `fast_2p` Python: `~21.3k games/sec` medio
    - `fast_numba`: `~445.4k games/sec` medio dopo warm-up
    - speedup indicativo: `~20.9x` sul solo core random-vs-random
  - limite: non misura ancora policy neurali/A2C, backprop o opponent `.npz`
- [x] Estendere Numba alle policy fast-compatible
  - agenti tradotti nel core JIT: `random`, `greedy_points`, `heuristic_v1`, `heuristic_v2`
  - modulo: `src/briscola_ai/ai/fast_numba.py`
  - benchmark CLI: `scripts/benchmark_perf.py --mode fast-eval|numba-eval`
  - test: determinismo per seed, invarianti punti/contatori, rifiuto agenti non supportati
  - benchmark seat-fair `heuristic_v2` vs `random`, 100k game x 3 run:
    - fast evaluation Python: `~17.1k games/sec` medio
    - Numba evaluation: `~364.2k games/sec` medio
    - speedup indicativo: `~21.3x`
  - benchmark seat-fair `heuristic_v2` vs `heuristic_v1`, 100k game x 3 run:
    - fast evaluation Python: `~13.3k games/sec` medio
    - Numba evaluation: `~300.3k games/sec` medio
    - speedup indicativo: `~22.6x`
  - limite: RNG Numba separato dal `random.Random` canonico; testiamo determinismo/invarianti, non identità byte-per-byte degli esiti
- [x] Integrare il wrapper encoder Numba nel rollout A2C fast
  - modulo: `src/briscola_ai/ai/fast_numba_observation.py`
  - CLI: `scripts/train_a2c.py --rollout-engine fast --fast-encoder numba`
  - test: equivalenza feature/mask vs `encode_fast_observation_2p` su stati iniziali/intermedi e smoke trainer
  - limite: il wrapper converte ancora `Fast2PState` da liste Python ad array NumPy a ogni decisione, quindi non è ancora lo speedup finale
- [x] Aggiungere rollout inference MLP full-JIT per opponent fast-compatible
  - modulo: `src/briscola_ai/ai/fast_numba_observation.py`
  - CLI benchmark: `scripts/benchmark_perf.py --mode numba-mlp`
  - scope: stato, encoder feature/mask, forward MLP, sampling azione, opponent rule-based e step tutti dentro Numba
  - test: determinismo per seed, invarianti punti/contatori, validazione shape pesi
  - benchmark `zero_mlp_numba(hidden=32)` vs `heuristic_v1`, 100k game x 3 run: `~11.2k games/sec` medio
  - benchmark `zero_mlp_numba(hidden=128)` vs `heuristic_v1`, 20k game x 3 run: `~2.84k games/sec` medio
  - limite: è inference/evaluation, non raccoglie ancora `StepRecord`/traiettorie per backprop A2C
- [x] Integrare Numba full-JIT nel training A2C
  - fare emettere al rollout JIT array di traiettoria (`x`, `z1`, `h`, `mask`, `probs`, `action_id`, reward/value)
  - riusare il backprop NumPy esistente su batch di traiettorie, senza ricostruire `PlayerObservation`
  - CLI: `scripts/train_a2c.py --rollout-engine fast --fast-rollout numba`
  - test: smoke trainer con rollout Python/encoder Python, rollout Python/encoder Numba e rollout Numba
  - benchmark training A2C vs `random`, 5k game, hidden=32:
    - rollout fast Python: `~5.06s`
    - rollout Numba: `~2.52s`
    - speedup indicativo: `~2.0x`
  - benchmark training A2C vs `random`, 5k game, hidden=128:
    - rollout fast Python: `~6.83s`
    - rollout Numba: `~5.04s`
    - speedup indicativo: `~1.36x`
  - limite: il rollout è JIT, ma backprop/Adam e conversione buffer -> `StepRecord` restano Python/NumPy
- [x] Ridurre conversione `NumbaA2CTrajectory -> StepRecord`
  - il path `--fast-rollout numba` ora fa backprop direttamente sui buffer array del rollout JIT
  - benchmark training A2C vs `random`, 5k game:
    - hidden=32: resta `~2.52s` (conversione StepRecord non era il collo principale)
    - hidden=128: `~5.04s -> ~4.93s` (miglioramento piccolo)
- [x] Ottimizzare training A2C oltre il rollout
  - profilo hidden=128 dopo buffer diretti: hotspot principale nel backprop Python con `np.outer` per step
  - il path `--fast-rollout numba` ora accumula i gradienti di ogni traiettoria con moltiplicazioni batch:
    `H.T @ dlogits`, `X.T @ dz1`, `H.T @ dv`
  - logging aggiornato con contatori aggregati (`value_loss_sum`, `grad_step_count`) invece di liste per step
  - test: equivalenza numerica tra nuovo accumulo batch e reference lenta con `np.outer`, incluso BC-anchor
  - benchmark training A2C vs `random`, 5k game, hidden=128: `~4.93s -> ~2.37s`
  - profilo successivo 2k game hidden=128: backprop batch `~0.106s`, collector/wrapper Numba `~0.817s`
- [x] Ridurre overhead per-game del collector A2C Numba
  - aggiunto `collect_a2c_batch_numba_2p`: una chiamata wrapper per batch/update, buffer 3D e `step_counts`
  - il trainer usa il batch per opponent singolo, opponent mix rule-based e mix ibridi con `best_a2c`; nel mix passa
    al JIT un codice opponent e un flag modello per partita
  - il backprop Numba-path appiattisce tutte le righe valide del batch e accumula i gradienti una sola volta per update
  - il loop batch JIT usa `numba.prange`, quindi le partite indipendenti del batch girano in parallelo
  - test: equivalenza batch-vs-single su seed/seat, equivalenza con opponent code per-game, equivalenza mix
    rule-based+MLP, più smoke trainer fast-rollout Numba con opponent singolo/mix
  - benchmark training A2C vs `random`, 5k game:
    - hidden=32: `~2.52s -> ~0.48s`
    - hidden=128: `~2.37s -> ~0.72s`
  - benchmark training A2C vs `best_a2c`, 5k game, hidden=32: `~4.31s -> ~0.82s`
  - benchmark training A2C con mix `heuristic_v1:0.7,random:0.2,greedy_points:0.1`, 5k game, hidden=128: `~0.74s`
  - benchmark training A2C con mix `best_a2c:0.5,random:0.5`, 5k game, hidden=32: `~0.89s`
  - profilo 2k game hidden=128: collector batch parallelo `~0.247s`, batch backprop `~0.042s`
- [x] Integrare reward shaping anti-overkill nel fast rollout Numba
  - supporto CLI: `--rollout-engine fast --fast-rollout numba --overkill-penalty-beta > 0`
  - modalità supportate nel core JIT: `--overkill-penalty-mode flat|gap`
  - il calcolo usa solo dati leciti già nel `Fast2PState` numerico: mano policy, carta sul tavolo, briscola pubblica
  - il fast rollout Python resta senza shaping overkill; per fast + shaping usare Numba
  - test: penalità JIT flat/gap su caso minimale + smoke trainer fast Numba con opponent mix e shaping attivo
  - benchmark training A2C con mix `heuristic_v1:0.7,random:0.2,greedy_points:0.1`, 5k game, hidden=128,
    `--overkill-penalty-beta 0.01 --overkill-penalty-mode gap`: `~0.76s`
- [x] Estendere il rollout fast A2C a opponent `.npz`
  - supporto: `scripts/train_a2c.py --rollout-engine fast --fast-rollout numba --opponent best_a2c`
  - supporto esplicito: `--opponent bc_model --opponent-model path/to/model.npz`
  - scope: opponent `.npz` MLP (`w1/b1/w2/b2`) con forward argmax mascherato nel core JIT
  - supporto mix: `best_a2c` o `bc_model` dentro `--opponent-mix` insieme ad avversari rule-based, con flag opponent per partita
  - supporto guard: `inference_overkill_guard` viene applicato con post-processing numerico anti-overkill
  - test: smoke trainer con opponent `.npz` esplicito e mix ibrido rule-based + `best_a2c`/`bc_model`
  - benchmark training A2C vs `best_a2c`, 5k game, hidden=32: `~4.31s -> ~0.82s`
  - benchmark training A2C con mix `best_a2c:0.5,random:0.5`, 5k game, hidden=32: `~0.89s`
  - limite: nello stesso `--opponent-mix` fast Numba è supportato al massimo un tipo di opponent modello
- [x] Esporre il rollout Numba nella pipeline esperimenti riproducibile
  - CLI: `scripts/run_experiment.py --rollout-engine fast --fast-rollout numba`
  - validazione: i flag fast/Numba sono accettati solo con `--algo a2c` e richiedono `--rollout-engine fast`
  - manifest: `train.rollout` registra engine, encoder e collector usati; `eval.parallelism` distingue
    `numba_threads` da process worker
  - test: help CLI, validazione PG e comando/manifest con `_run` mockato
- [x] Integrare Numba nella evaluation matrix ufficiale
  - CLI: `scripts/evaluate_matrix.py --engine numba --model <model.npz> ...`
  - pipeline: `scripts/run_experiment.py --eval-engine numba` passa il flag alla matrix e lo registra nel manifest
  - semantica: valutazione deterministica argmax come `BCModelAgent`, con seed suite standard/holdout e guard anti-overkill
    letto dai metadati del modello
  - supporto: modelli `.npz` MLP (`w1/b1/w2/b2`) contro opponent fast-compatible (`random`, `greedy_points`,
    `heuristic_v1`, `heuristic_v2`)
  - supporto head-to-head: opponent `bc_model` con `opponent_model_path`, utile per confrontare candidato vs best/precedente
  - performance: con `engine=numba` i process worker della matrix vengono disabilitati per evitare oversubscription;
    il parallelismo è quello interno ai kernel `prange`
  - test: matrix Numba model-vs-baseline e model-vs-model con path opponent esplicito
- [x] Integrare Numba nello script di valutazione singola
  - CLI: `scripts/evaluate_agents.py --engine numba --agent0 bc_model --agent0-model <model.npz> --agent1 heuristic_v1`
  - supporto: `agent0=bc_model` MLP contro baseline fast-compatible oppure `agent1=bc_model` MLP, sia plain sia seat-fair/benchmark
  - uso chiave: head-to-head rapido tra nuovo modello e best/precedente senza tornare al dominio canonico
  - output JSON coerente con gli altri engine (`engine=numba`, `stats` standard)
- [x] Parallelizzare la evaluation MLP Numba
  - kernel separati `prange` per seat fisso e seat-fair, così gli indici restano semplici per Numba
  - `scripts/evaluate_agents.py --engine numba` e `scripts/evaluate_matrix.py --engine numba` usano il path parallelo
  - test: equivalenza seriale/parallelo su policy deterministica argmax con stessi seed
  - benchmark locale `best_a2c.npz` vs `heuristic_v1`, 10k game seat-fair: `3.261s -> 0.550s`
    (`~5.9x`, stesso `avg_diff=+12.4640`)
- [x] Integrare Numba nelle metriche decision-quality
  - CLI: `scripts/evaluate_decision_quality.py --engine numba --agent-a bc_model --agent-a-model <model.npz> --agent-b heuristic_v1`
  - metriche JIT: `trump_waste_rate`, `trump_overkill_rate`, `trump_overkill_rate_low_lead_points`
  - semantica: policy argmax deterministica + guard anti-overkill opzionale letto dai metadati/env
  - supporto: modello A MLP vs baseline fast-compatible oppure vs modello B MLP
  - parallelizzazione: kernel `prange` seat-fair per match + contatori quality, usato dal path pubblico Numba
  - test: equivalenza seriale/parallelo dei contatori quality con stessi seed
  - benchmark locale `best_a2c.npz` vs `heuristic_v1`, 10k game seat-fair: `3.253s -> 0.533s`
    (`~6.1x`, stesso `avg_diff=+12.4640`, `waste=23`, `overkill=0`)
  - il dominio resta fallback per agenti arbitrari/non-MLP

## Deliverable (come sapremo di aver “finito” ogni fase)

- Fase 0: `pytest` verde con test base; script di simulazione che genera partite senza UI.
- Fase 1: dipendenze aggiornate + lock aggiornato + test verdi.
- Fase 2: nuovo layout e motore separato; API che consuma il dominio via interfaccia pulita.
- Fase 3: copertura significativa del dominio (target iniziale: 60–70% sul dominio).
- Fase 4: generazione dataset riproducibile + baseline metriche.
- Fase 5: primo modello addestrato + benchmark ripetibile vs baseline.

## Rischi e decisioni da prendere insieme

- **Compatibilità FastAPI/Pydantic v2**: ✅ già completata (stack aggiornato + DTO Pydantic v2 + test verdi).
- **Modalità 4 giocatori**: l’osservazione parziale e il training a squadre complicano; possiamo partire dal 2‑player per didattica e poi estendere.
- **Persistenza**: SQLite è semplice e “portabile”; Postgres in Docker è più realistico ma aggiunge overhead operativo.
- **Tooling frontend (lint JS)**: decidere se introdurre un linter/formatter JS (es. Biome vs ESLint/Prettier) o mantenere un check minimale (es. `node --check` integrato in `pytest`).
