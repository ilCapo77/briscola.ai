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
- Test attuali: **105** (pytest).
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
  (`A2C shaped 1.0M game`, seed=17). Il best attuale ha `inference_overkill_guard=false`.

Operatività (senza env var):
- [x] Abilitare `inference_overkill_guard=true` nei metadati di `data/models/best_a2c.npz`
  - obiettivo: ottenere gli stessi benefici "guard ON" senza dover impostare `BRISCOLA_BC_OVERKILL_GUARD`
  - verifica (`medium` vs `heuristic_v1`, seed=0): `trump_overkill_rate=0.0%` e low-lead `=0.0%` con `avg_diff≈+12.80`
- Nota: questa scelta era valida per il best precedente. Dopo la promozione del modello 1M, il nuovo
  `data/models/best_a2c.npz` non usa il guard in metadati; prima di riattivarlo va rifatta una A/B decision-quality.

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
- [ ] (Opzionale) Se migliora forza+qualità, promuovere un “best_a2c” v2 (decisione esplicita):
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

Prossimo controllo consigliato sul best attuale 1M:
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
  - decisione: abilitato `inference_overkill_guard=true` nel best ufficiale `data/models/best_a2c.npz`
    e registrata la decisione in `data/models/best_a2c.json`

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
- [ ] (2) Generare un dataset BC “pulito” via self-play del teacher:
  - `scripts/self_play_to_db.py` con `--agents heuristic_v2,heuristic_v2` (o mix)
  - `scripts/export_dataset.py` → JSONL
  - `scripts/train_bc.py --encoder-version v2` (consigliato `--model mlp`) per ottenere `bc_teacher_v2.npz`
- [ ] (3) Fine-tuning A2C (encoder v2) partendo da BC:
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
