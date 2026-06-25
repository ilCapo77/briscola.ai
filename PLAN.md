# Piano operativo — Briscola AI

Questo file deve restare breve e utile per decidere cosa fare dopo. I dettagli storici, i benchmark intermedi non promossi e le prove superate sono stati rimossi: il codice, i test e i commit sono la memoria tecnica completa.

## Stato Corrente

- Versione progetto: `0.8.0` (`pyproject.toml`).
- Runtime/tooling: Python 3.14, FastAPI, Pydantic v2, `ruff`, `mypy`, `pytest`. Deps runtime lazy per il cloud: `redis`, `psycopg`. Dev-only: `fakeredis`, `playwright`.
- Dominio canonico: `src/briscola_ai/domain/`, con `GameState` immutabile e `step()` come transizione pura.
- Backend/UI: HTTP + WebSocket, UI statica servita dal backend. Stato partita in `GameSessionStore` (in-memory in locale, **Redis** se `REDIS_URL`); realtime via **pub/sub** dello store; event log SQLite o **Postgres** (`DATABASE_URL`). **Deployato** su FastAPI Cloud: <https://briscolaai.fastapicloud.dev>.
- Anti-cheat: agenti e modelli ricevono `PlayerObservation`, non `GameState` completo.
- Fast path: 2-player numerico Python/Numba per training/evaluation ad alto throughput.
- Encoder supportati: v1, v2, v3.
- Feature dim:
  - v1: `FEATURE_DIM_2P_V1`
  - v2: `FEATURE_DIM_2P_V2 = 288`
  - v3: `FEATURE_DIM_2P_V3 = 310`
- Modello consigliato: `data/models/best_a2c_v3.npz` (encoder v3, guard anti-overkill ON).
- Modello precedente ancora disponibile: `data/models/best_a2c.npz` (encoder v2).
- Default UI: avversario = modello migliore (`bc_model` + `best_a2c_v3`); `GET /api/ai/agents` espone `available`/`requires_model_id` e la UI disabilita gli avversari il cui modello bundle manca (es. `hybrid_endgame_best_a2c` senza `best_a2c.npz`). Il default lato API resta `random` se non specificato.
- Coverage badge README: manuale via Shields.io.

Comandi quality gate:

```bash
uv run ruff format src tests scripts
uv run ruff check src tests scripts
uv run mypy src
pytest
```

Coverage:

```bash
pytest --cov=briscola_ai --cov-report=term-missing
```

## Architettura Da Mantenere

- `domain/`: regole pure, modelli, stato, osservazioni, mapping carte, serializzazione (`serialization.py`).
- `backend/`: FastAPI, DTO, server, `game_store.py` (GameSessionStore in-memory/Redis + pub/sub), event log (`event_log.py`: SQLite/Postgres).
- `ai/agents/`: agenti baseline, ibridi, factory e catalogo agenti.
- `ai/encoding/observation_encoder.py`: encoder v1/v2/v3 canonici.
- `ai/fast/` e `ai/numba/`: fast path 2-player; deve restare coerente col dominio tramite test di parità.
- `scripts/`: simulazione, export dataset, training, evaluation, benchmark.
- `data/` e `benchmarks/experiments/`: artefatti locali gitignored.

Invarianti importanti:

- Se cambia una regola nel dominio, aggiornare anche fast path e test di parità.
- Non passare mai `GameState` completo a una policy giocante.
- `seen_cards_onehot`: carte pubblicamente viste, inclusa la briscola scoperta.
- `out_of_play_cards_onehot`: carte non più disponibili, cioè prese + tavolo; non include la sola briscola scoperta.
- Modelli `.npz` devono dichiarare metadata coerenti (`encoder_version`, `feature_dim`, label/descrizione quando utile).
- UI/catalogo non devono accettare path arbitrari dal browser.
- Lo stato partita vive solo nel `GameSessionStore`: non reintrodurre dict globali di stato nel server; l'`Agent` non è serializzato (config in sessione, ricostruito con `build_agent`).
- Gli eventi realtime passano per il pub/sub dello store: mantenere la consegna WS indipendente dalla replica (no `connected_clients` globali).

## Baseline AI Ufficiale

### Best Corrente

`best_a2c_v3.npz`

- Encoder: v3 (`feature_dim=310`).
- Guard anti-overkill: ON per runtime/UI.
- Addestramento riproducibile a grandi linee:
  - teacher dataset: `hybrid_endgame_best_a2c` vs `hybrid_endgame_best_a2c`, 20k partite;
  - BC v3: MLP hidden 128, 20 epoche, output `bc_v3.npz`;
  - A2C v3: fast+numba, 1M partite, seed 300;
  - opponent mix: `best_a2c:0.4,heuristic_v2:0.3,heuristic_v1:0.2,random:0.1`;
  - BC anchor: `--bc-anchor data/models/bc_v3.npz --bc-anchor-beta 0.01`.

### Risultati Di Promozione

Confronto contro vecchio `best_a2c`:

- head-to-head raw, big: `+0.63` avg diff;
- head-to-head guarded, medium: `+0.09` avg diff;
- holdout vs `heuristic_v1`: `+17.23` vs `+16.56`;
- overkill raw: `11.4%` vs `12.7%`;
- overkill low-lead raw: `1.6%` vs `4.8%`.

Decisione: `best_a2c_v3.npz` è la baseline consigliata. `best_a2c.npz` resta selezionabile per confronto/regressioni.

## Cosa È Già Chiuso

### Motore, Backend, UI

- Dominio canonico `GameState + step()`.
- Supporto 2-player e 4-player nel dominio, con focus didattico/training sul 2-player.
- DTO Pydantic v2.
- WebSocket + fallback polling UI.
- IA server-driven: il backend avanza automaticamente quando tocca all'IA.
- Event log SQLite append-only.
- Export JSONL.
- Self-play verso DB.
- Evaluation offline seat-fair.
- Package `ai/` riorganizzato per responsabilità (`agents/ encoding/ models/ endgame/ fast/ numba/ evaluation/ training/`); shim legacy rimossi; docstring complete su moduli/pubbliche e nei test (vedi `ai/README.md`).

### Deploy E Infrastruttura (Fase 1)

- Stato partita su `GameSessionStore` (in-memory/Redis) + lock per partita → risolve "partita non trovata" su deploy multi-replica.
- Realtime via **Redis pub/sub**: fan-out WS cross-replica (`ai_card_reveal`/`trick_result`/refresh point-in-time); mantiene REST+WS; `?polling=1` resta fallback.
- Event log **Postgres** opzionale (`PostgresEventLog`, factory `build_event_log` su `DATABASE_URL`); stesso schema dell'SQLite.
- Provisioning modello allo startup (`BRISCOLA_MODEL_URL` + sha256); endpoint `/health` e `/version`; shim `main:app` per `fastapi run`.
- Cache-busting asset automatico (versione + mtime degli static).
- Homepage didattica (tagline + "Cos'è" + link GitHub), punti IA nascosti (fairness), layout mobile fit-to-viewport, nota anti-cheat sotto il bottone, preload immagini carte, footer su una riga con icona GitHub e versione software.
- Suite ermetica (`tests/conftest.py` azzera `REDIS_URL`/`DATABASE_URL`); store/event-log testati con `fakeredis`/connessione fake.
- Repo/release: history senza trailer `Co-Authored-By`; serie completa di tag di versione (`v0.1.0` → corrente) su GitHub; release `v0.5.0` con asset `best_a2c_v3.npz` usata dal provisioning.
- **Deploy COMPLETATO** su FastAPI Cloud (Redis collegato): <https://briscolaai.fastapicloud.dev>. Resta opzionale attivare Postgres/Neon (`DATABASE_URL` + `BRISCOLA_EVENT_LOG_MODE=dataset`) per la raccolta dati.

### Endgame E Strategia

- Solver endgame esatto 2-player a mazzo vuoto.
- `hybrid_endgame`: fallback `heuristic_v2` + solver nel finale.
- `hybrid_endgame_best_a2c`: fallback `best_a2c` + solver nel finale.
- `out_of_play_cards_onehot` aggiunto a osservazioni/DTO in modo backward-compatible.
- Encoder v3 con 22 feature aggregate sopra v2.
- Parità encoder v3 su domain, fast Python e Numba.
- Rollout/evaluation v3 abilitati su fast+numba.
- Pipeline BC/A2C v3 completata e promossa a nuovo best.

### Performance

- Fast path 2-player Python e Numba.
- Evaluation Numba per modelli MLP e baseline fast-compatible.
- Decision-quality Numba.
- A2C fast+numba con batch collector.
- v3 fast+numba sbloccato.
- Benchmark indicativo: A2C v3 20k partite passa da circa `419 games/sec` domain a circa `5900 games/sec` fast+numba.

## Prossime Azioni Consigliate

### 1. Consolidamento Del Nuovo Best

Priorità alta, piccolo rischio. Stato: **in gran parte FATTO** — best v3 in produzione, default UI = best model, opzioni con modello mancante disabilitate. L'evaluation matrix leggera resta utile come check periodico.

- Verificare UI/API con `best_a2c_v3.npz` presente e selezionato.
- Verificare comportamento UI/API quando `best_a2c_v3.npz` manca: non deve rompere creazione partita o catalogo.
- Eseguire evaluation matrix leggera contro:
  - `random`
  - `greedy_points`
  - `heuristic_v1`
  - `heuristic_v2`
  - `hybrid_endgame`
  - `hybrid_endgame_best_a2c`
  - vecchio `best_a2c`
- Salvare JSON locali in `benchmarks/experiments/` e riportare solo sintesi stabile qui, se emergono regressioni.
- Aggiornare README se serve una nota utente sul modello consigliato v3.

### 2. Raccolta Dati In Produzione (Postgres)

Priorità media; serve solo se si vuole costruire il dataset da partite umane reali.

- Collegare Neon all'app (→ `DATABASE_URL`) e impostare `BRISCOLA_EVENT_LOG_MODE=dataset` (il consenso è già gestito: checkbox UI + enforcement backend).
- Aggiornare `scripts/export_dataset.py` per leggere anche da **Postgres** (oggi legge solo SQLite) → export JSONL del dataset umano.
- Verificare privacy: niente PII, `client_id` pseudonimo, solo partite consenzienti/complete.

### 3. Self-Improvement V3 In Stile League

Priorità media, dopo consolidamento.

Obiettivo: usare `best_a2c_v3` come nuovo avversario di lega senza cambiare algoritmo.

Esperimento base:

- warm-start da `best_a2c_v3` o da BC/A2C v3 candidato, da decidere esplicitamente;
- rollout `fast --fast-rollout numba`;
- opponent mix con `best_a2c_v3` incluso;
- mantenere BC-anchor se l'overkill raw peggiora;
- valutare prima medium, poi big solo se medium è promettente.

Criteri di promozione:

- head-to-head positivo contro `best_a2c_v3` su big seat-fair;
- holdout vs `heuristic_v1` non peggiore;
- `trump_waste_rate` e `trump_overkill_rate` non peggiorano materialmente;
- niente promozione se il vantaggio è solo rumore statistico.

### 4. PPO/GAE Solo Se A2C Si Blocca

Priorità bassa per ora.

PPO/GAE ha senso solo se:

- self-improvement A2C v3 non migliora più;
- la baseline v3 è consolidata;
- i test fast/numba restano verdi;
- si definisce un esperimento piccolo e isolato.

Non introdurre DQN per ora: action mask, parziale osservabilità e self-play rendono più coerente continuare con policy-gradient.

### 5. Igiene

- Aggiornare badge coverage README solo dopo `pytest --cov=briscola_ai` se la variazione è materiale.
- Tenere `PLAN.md` breve: risultati intermedi e tentativi falliti vanno rimossi o sintetizzati.
- Non committare artefatti locali in `data/`, `benchmarks/`, `.claude/`.
- Commit in italiano.

## Comandi Utili

Avvio server:

```bash
briscola-server --reload
```

Simulazione headless:

```bash
python scripts/simulate_games.py --num-games 100 --seed 42
```

Evaluation head-to-head:

```bash
uv run python scripts/evaluate_agents.py \
  --benchmark medium \
  --engine domain \
  --agent0 bc_model \
  --agent0-model data/models/best_a2c_v3.npz \
  --agent1 bc_model \
  --agent1-model data/models/best_a2c.npz
```

Decision quality:

```bash
uv run python scripts/evaluate_decision_quality.py \
  --benchmark medium \
  --engine numba \
  --agent-a bc_model \
  --agent-a-model data/models/best_a2c_v3.npz \
  --agent-b heuristic_v1
```

Training A2C v3 fast+numba:

```bash
uv run python scripts/train_a2c.py \
  --encoder-version v3 \
  --rollout-engine fast \
  --fast-rollout numba \
  --init data/models/bc_v3.npz \
  --opponent-mix best_a2c:0.4,heuristic_v2:0.3,heuristic_v1:0.2,random:0.1 \
  --bc-anchor data/models/bc_v3.npz \
  --bc-anchor-beta 0.01
```

Benchmark performance:

```bash
uv run python scripts/benchmark_perf.py
```
