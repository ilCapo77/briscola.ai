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
- Motore: `BriscolaGame` (supporta 2 e 4, ma **focus didattico sul 2-player**).
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
    - Contratto WS fissato: gli snapshot includono `type: "observation"` (allineare README/UI e test).
    - Chiarire vincolo attuale UI “umano = player 0” (focus 2-player; da generalizzare se aggiungiamo scelta giocatore/4-player in UI).
  - Timing animazioni (scelta architetturale):
    - Il backend evita `asyncio.sleep()` per ritardi di presentazione (reveal/risultato mano).
    - Il frontend “trattiene” gli snapshot WS per mostrare reveal e risultato con tempi controllati lato UI.
- Test: presenti in `tests/` (unit + integrazione API base).
- Coverage: misurata con `pytest-cov` (attuale ~56% su `briscola_ai`; obiettivo: crescita progressiva).
- Badge coverage: manuale via Shields.io nel `README.md` (niente `coverage.svg` versionato / script di generazione).

Comandi di verifica (sempre validi):
- test: `pytest`
- coverage: `pytest --cov=briscola_ai --cov-report=term-missing`

## Stato iniziale (prima del refactor)

- Backend: `FastAPI` + endpoint HTTP + `WebSocket` (stato partita in memoria).
- Motore: `BriscolaGame` in `src/briscola_ai/game/` (2 e 4 giocatori).
- Frontend: asset statici in `src/briscola_ai/frontend/static/` (HTML/CSS/JS).
- Test: assenti.
- “AI”: bot casuale lato frontend.

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
- tests: verdi (29)
- coverage totale: ~56% (focus prossimo: aumentare copertura di `backend/server.py` e rami non coperti del motore)

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
- [x] Smoke test UI manuale (documentato): passi ripetibili + expected (utile per regressioni).
  - Documentazione: vedi `README.md` → sezione “Smoke test UI (manuale)”.
- [ ] Riprodurre e catalogare eventuali bug UI residui (console JS, tab Network, handshake WebSocket).
- [ ] Allineare contratto dati UI↔API:
  - [ ] definire un DTO stabile per `Card`, `GameObservation`, `GameResult` (idealmente da OpenAPI/Pydantic)
  - [ ] ridurre accoppiamento a stringhe “magiche” (es. `player_0_hand_size`) introducendo campi espliciti
- [ ] Robustezza runtime:
  - [ ] gestione errori in UI (banner/stato connessione, retry/backoff WS, messaggi user-friendly)
  - [ ] fallback senza WebSocket (polling) per debug
- [ ] Test UI:
  - [x] smoke test manuale documentato (passi + expected)
  - [ ] (opzionale) E2E leggero con Playwright quando introduciamo una toolchain JS

- [x] **Refactor IA → modello server-driven (standard)**:
  - Backend: rimuovere endpoint di trigger e far avanzare automaticamente la partita quando tocca all'IA
  - Frontend: mantenere UX invariata con hold/coda eventi (no dipendenza da un trigger client)
  - Obiettivo: UI didattica leggibile senza `asyncio.sleep()` nel backend

Deliverable minimo:
- la UI permette di avviare una partita 2-player, giocare carte e vedere fine partita senza errori in console.

### Fase 2 — Ristrutturazione architetturale (dominio vs adattatori) (3–6 sessioni)

Obiettivo: rendere chiaro “cosa è Briscola” vs “come la servo” vs “come la alleno”.

Stato (Phase 2B, avvio):
- [x] Introdotto motore “funzionale” in parallelo: `GameState + step()` in `src/briscola_ai/domain/`.
- [x] Aggiunti test di parità vs `BriscolaGame` per migrazione sicura (`tests/test_domain_step_parity.py`).
- [ ] Migrare il backend (in-memory store) da `BriscolaGame` a `GameState` + `step`.

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
- API: trasformare oggetti Python in JSON con **Pydantic schema** invece di encoder custom.

### Fase 3 — Test “seri” e qualità (parallela alle fasi 1–2)

Obiettivo: coprire il motore e assicurare stabilità in evoluzione.

- Unit test dominio (priorità alta):
  - regole mano (`who_wins_trick`) con casi noti
  - punteggi (somma punti carte catturate)
  - invarianti: nessuna carta duplicata, mano sempre valida, turni coerenti
- Test di integrazione API (priorità media):
  - crea partita → gioca azione → stato cambia → fine partita
  - WebSocket: connessione + ricezione update (anche test “light”)
- Aggiungere CI (GitHub Actions) per `pytest` e lint (quando il repo è versionato).

### Fase 4 — Data pipeline per ML (didattica) (4–8 sessioni)

Obiettivo: passare da “gioco” a “ambiente addestrabile”.

- Introdurre una persistenza “da laboratorio” (SQLite):
  - tabella partite, step/azioni, osservazioni, metadati (seed, versione regole, versione codice)
  - scrittura append-only (stile event log) per semplificare debug e riproducibilità
- Definire un comando di export dataset (per training):
  - da SQLite → JSONL/Parquet con schema versionato
  - campi minimi: `state` (osservazione), `valid_actions`, `action`, `reward`, `done`, `metadata`
- Implementare un simulatore “self‑play”:
  - due agenti baseline (random + heuristic)
  - generazione di partite in batch con seed
  - scrittura su SQLite + export dataset + metriche
- Introdurre una baseline di valutazione:
  - win-rate su set di seed
  - ELO/TrueSkill (opzionale)

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

## Deliverable (come sapremo di aver “finito” ogni fase)

- Fase 0: `pytest` verde con test base; script di simulazione che genera partite senza UI.
- Fase 1: dipendenze aggiornate + lock aggiornato + test verdi.
- Fase 2: nuovo layout e motore separato; API che consuma il dominio via interfaccia pulita.
- Fase 3: copertura significativa del dominio (target iniziale: 60–70% sul dominio).
- Fase 4: generazione dataset riproducibile + baseline metriche.
- Fase 5: primo modello addestrato + benchmark ripetibile vs baseline.

## Rischi e decisioni da prendere insieme

- **Compatibilità FastAPI/Pydantic v2**: upgrade richiede cambi mirati.
- **Modalità 4 giocatori**: l’osservazione parziale e il training a squadre complicano; possiamo partire dal 2‑player per didattica e poi estendere.
- **Persistenza**: SQLite è semplice e “portabile”; Postgres in Docker è più realistico ma aggiunge overhead operativo.
- **Tooling frontend (lint JS)**: decidere se introdurre un linter/formatter JS (es. Biome vs ESLint/Prettier) o mantenere un check minimale (es. `node --check` integrato in `pytest`).
