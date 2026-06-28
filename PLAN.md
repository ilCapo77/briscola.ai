# Piano operativo — Briscola AI

Questo file deve restare breve e utile per decidere cosa fare dopo. I dettagli storici, i benchmark intermedi non promossi e le prove superate sono stati rimossi: il codice, i test e i commit sono la memoria tecnica completa.

## Stato Corrente

- Versione progetto: `0.12.1` (`pyproject.toml`).
- Runtime/tooling: Python 3.14, FastAPI, Pydantic v2, `ruff`, `mypy`, `pytest`. Deps runtime lazy per il cloud: `redis`, `psycopg`. Dev-only: `fakeredis`, `playwright`.
- Dominio canonico: `src/briscola_ai/domain/`, con `GameState` immutabile e `step()` come transizione pura.
- Backend/UI: HTTP + WebSocket, UI statica servita dal backend. Stato partita in `GameSessionStore` (in-memory in locale, **Redis** se `REDIS_URL`); realtime via **pub/sub** dello store; event log SQLite o **Postgres** (`DATABASE_URL`). **Deployato** su FastAPI Cloud: <https://briscolaai.fastapicloud.dev>. Release `v0.12.1` deployata con modello v6 come default; rollout cloud v6 verificato in produzione.
- Anti-cheat: agenti e modelli ricevono `PlayerObservation`, non `GameState` completo.
- Fast path: 2-player numerico Python/Numba per training/evaluation ad alto throughput.
- Encoder supportati: v1, v2, v3.
- Feature dim:
  - v1: `FEATURE_DIM_2P_V1`
  - v2: `FEATURE_DIM_2P_V2 = 288`
  - v3: `FEATURE_DIM_2P_V3 = 310`
- Modello consigliato: `data/models/best_a2c_v6.npz` (encoder v3, guard anti-overkill ON).
- Modelli precedenti ancora utili per confronto/regressioni: `data/models/best_a2c_v5.npz`, `data/models/best_a2c_v4.npz`, `data/models/best_a2c_v3.npz`, `data/models/best_a2c.npz` (legacy encoder v2).
- Default UI: avversario = modello consigliato (`bc_model` + `BRISCOLA_DEFAULT_MODEL_ID`, oggi `best_a2c_v6.npz`). `GET /api/ai/models` espone `recommended_model`; la UI seleziona quel modello se compatibile, altrimenti il `best_a2c_vN` più recente disponibile. `GET /api/ai/agents` espone `available`/`requires_model_id` e la UI disabilita gli avversari il cui modello bundle manca. Il default lato API resta `random` se non specificato. In cloud, per restare dentro il limite disco, mantenere solo pochi asset (v6 obbligatorio; v5 opzionale per confronto manuale).
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
- `docs/reports/model_progress.xlsx`: report Excel curato dei modelli significativi; rigenerato da `scripts/build_model_report.py`.
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

`best_a2c_v6.npz`

- Encoder: v3 (`feature_dim=310`).
- Guard anti-overkill: ON per runtime/UI.
- Addestramento riproducibile a grandi linee:
  - teacher dataset: `hybrid_endgame_best_a2c` vs `hybrid_endgame_best_a2c`, 20k partite;
  - BC v3: MLP hidden 128, 20 epoche, output `bc_v3.npz`;
  - A2C v3/v4 league: fast+numba, warm-start da `best_a2c_v3`, 1M partite, seed 301;
  - A2C v5 league: fast+numba, warm-start da `best_a2c_v4`, 1M partite, seed 401;
  - A2C v6 scaling: fast+numba, warm-start da `best_a2c_v5`, 5M partite, seed 501;
  - opponent mix v6: `best_a2c_v5:0.4,heuristic_v2:0.3,heuristic_v1:0.2,random:0.1`;
  - BC anchor: `--bc-anchor data/models/bc_v3.npz --bc-anchor-beta 0.01`.

### Risultati Di Promozione

Confronto v6 contro `best_a2c_v5`:

- head-to-head big guarded seat-fair: `+0.46` su 100k partite;
- holdout head-to-head big guarded seat-fair: `+0.46` su 100k partite;
- big vs `heuristic_v1`: `+18.40` vs `+17.83` del v5;
- decision-quality big vs `heuristic_v1`: `+18.58`, `trump_overkill_rate=0.0%`,
  `trump_waste_rate=0.07%`.

Decisione: `best_a2c_v6.npz` è la baseline consigliata. `best_a2c_v5.npz`, `best_a2c_v4.npz`,
`best_a2c_v3.npz` e `best_a2c.npz` restano selezionabili per confronto/regressioni se presenti nella directory
modelli.

### Report Modelli

Il report curato dei modelli vive in `docs/reports/model_progress.xlsx` ed è generato da
`scripts/build_model_report.py`. Deve restare selettivo: best ufficiali, teacher/anchor importanti e candidati scartati
solo quando spiegano una decisione. Aggiornarlo quando viene promosso un nuovo best o quando un esperimento cambia la
roadmap; non usarlo come archivio di ogni run locale.

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
- Homepage didattica (tagline + "Cos'è" + link GitHub), punti IA nascosti (fairness), layout mobile fit-to-viewport, nota anti-cheat sotto il bottone, preload immagini carte, spinner di avvio partita, footer su una riga con icona GitHub e versione software.
- Suite ermetica (`tests/conftest.py` azzera `REDIS_URL`/`DATABASE_URL`); store/event-log testati con `fakeredis`/connessione fake.
- Repo/release: history senza trailer `Co-Authored-By`; serie completa di tag di versione (`v0.1.0` → corrente) su GitHub; release `v0.12.1` pubblicata e deployata con `best_a2c_v6.npz`.
- **Deploy COMPLETATO** su FastAPI Cloud (Redis collegato): <https://briscolaai.fastapicloud.dev>. Postgres/event log e modalità dataset sono attivabili via `DATABASE_URL` + `BRISCOLA_EVENT_LOG_MODE=dataset` quando serve raccogliere dataset umano.
- Rollout cloud `v0.12.1`/v6 completato e verificato: `/version` espone `recommended_model=best_a2c_v6.npz` e
  `recommended_model_present=true`; la UI mostra la label v6.
- Osservabilità/export cloud completati: produzione verificata con `DATABASE_URL` (Neon/Postgres), `BRISCOLA_EVENT_LOG_MODE=dataset` e `REDIS_URL`; `scripts/report_event_log.py` conta partite/consenso/finestre 24h-7d/qualità `human_action`; `scripts/export_dataset.py` legge SQLite o Postgres mantenendo JSONL v1 e sanifica i nomi giocatore (`player_0`, `player_1`, ...). Smoke export produzione validato: 18 partite complete / 360 record.

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

### 1. Monitoraggio Produzione v6

Obiettivo: osservare `best_a2c_v6.npz` in produzione senza avviare training per inerzia.

- Produzione verificata con modello v6. Per nuovi rollout o per riallineare l'URL asset alla patch release usare:

```text
BRISCOLA_DEFAULT_MODEL_ID=best_a2c_v6.npz
BRISCOLA_MODEL_URL=https://github.com/ilCapo77/briscola.ai/releases/download/v0.12.1/best_a2c_v6.npz
BRISCOLA_MODEL_SHA256=b047a319c3505936d11127a3a2e29b9ca3a2b93676569a2ea8ce186a5e29a951
```

- Monitorare partite reali, errori, abbandoni e feedback qualitativo.
- Usare `scripts/report_event_log.py` per controllare aggregati e qualità del log senza stampare payload sensibili.
- Non usare dati umani per training finché il volume resta basso e la qualità/privacy non sono state riverificate.
- Non avviare v7 solo per inerzia: serve una nuova ipotesi misurabile.

### 2. Prossima Iterazione Modello (`best_a2c_v7`) Solo Con Ipotesi Nuova

Baseline da battere: `best_a2c_v6.npz`.

Storico scaling (seed 501, fast+numba, checkpoint 1M/3M/5M, guard inference ON), promosso a v6:

- 1M vs v5 big: `+0.03`; vs `heuristic_v1`: `+17.93`; overkill `0.0%`, waste `0.07%`;
- 3M vs v5 big: `+0.22`; vs `heuristic_v1`: `+18.18`; overkill `0.0%`, waste `0.09%`;
- 5M vs v5 big: `+0.46`; holdout vs v5 big: `+0.46`; vs `heuristic_v1`: `+18.40`;
  decision-quality vs `heuristic_v1`: `+18.58`, overkill `0.0%`, waste `0.07%`.

#### Ipotesi v7 concordata: league a popolazione (fictitious self-play)

Motivazione (basata su dati, non inerzia):

- **Costo crescente dello scaling puro.** v6 ha speso 5× le partite (5M vs 1M) per un guadagno (`+0.46`) della
  stessa entità dei passi da 1M. La curva (1M `+0.03`, 3M `+0.22`, 5M `+0.46`) è ancora **monotona**, non un
  plateau: il punto è che lo scaling puro ha **costo crescente** e non basta più come ipotesi primaria — "v7 =
  stesso recipe, più partite" è la mossa per inerzia che questo piano sconsiglia.
- **Debolezza metodologica da verificare.** Dalla v3 in poi il recipe è congelato (MLP `310→128→40`, encoder v3,
  opponent-mix `{predecessore 0.4 / h2 0.3 / h1 0.2 / random 0.1}`, warm-start dal predecessore immediato che è
  anche l'unico opponent appreso). Questo ottimizza per "battere il predecessore": **se** i matchup risultano non
  transitivi nel round-robin (A>B, B>C ma C≈A), c'è rischio di **ciclare** anziché crescere in forza assoluta. Il
  segnale che lo suggerisce è che le vittorie H2H sono marginali e ambigue — es. v3 vs v2: big standard `+0.00814`
  (wins `48433` vs `48683`, draws `2884` → in conteggio vittorie v3 *perde di poco*), big holdout `+0.18258`
  (wins `48741` vs `48339`). La decision-quality è ormai satura (overkill `0.0%`, waste `0.07%`): gli errori
  "facili" sono già risolti, i margini residui vengono dalla robustezza, non dalla scala.

Ipotesi misurabile: addestrare v7 contro una **popolazione** dei best storici encoder v3 `{v3, v4, v5, v6}` +
euristiche (invece del solo predecessore) produce una policy con **Elo round-robin** più alto e **matchup peggiore
migliore**, non solo un `+0.x` contro v6. (`best_a2c` v2 legacy è encoder v2/feature_dim 288: resta fuori dalla
popolazione di training ma va incluso almeno in **valutazione** round-robin come ancoraggio storico, dato che a
inference ciascun modello usa il proprio encoder.)

Passi:

- **Pre-validazione economica (prima di addestrare):** round-robin Elo tra i `.npz` esistenti `{v3, v4, v5, v6}` +
  `best_a2c` (v2 legacy, ancoraggio) + `heuristic_v1`, per misurare *quanto* è non transitiva la famiglia attuale.
  Se l'ordine non è coerentemente monotòno (A>B, B>C ma C≈A), l'ipotesi è confermata e vale addestrare.
- **Screening league a popolazione (~200k partite)** con opponent-mix multi-modello, es.
  `{v6:0.3, v5:0.15, v4:0.1, v3:0.05, heuristic_v2:0.2, heuristic_v1:0.1, random:0.1}`, warm-start da v6.
- Solo se lo screening è positivo, run lungo (1M+) e valutazione completa.

Costo implementativo da conoscere: il fast-rollout numba accetta oggi **un solo tipo di opponent modello per batch**
(`scripts/train_a2c.py:1010`), e `evaluate_matrix.py` è single-model. Serve plumbing contenuto per campionare tra
più `.npz` come opponent e per il round-robin. Cambiamento ben circoscritto, da fare con test fast/numba verdi
prima/dopo.

Criteri di promozione (rivisti per l'ipotesi popolazione):

- **Elo round-robin** vs `{v3..v6}` + `heuristic_v1` superiore a v6 (criterio primario, non solo H2H vs v6);
- **matchup peggiore** della popolazione non regredisce (no modello "rock" che batte v6 ma perde contro v4);
- holdout vs `heuristic_v1` non peggiore;
- `trump_waste_rate` e `trump_overkill_rate` non peggiorano materialmente;
- **significatività statistica obbligatoria**: ogni vantaggio (H2H ed Elo) riportato con **intervallo di confidenza
  bootstrap** (ricampionamento sulle partite); niente promozione se il CI tocca lo zero o se il delta è sotto una
  soglia minima predefinita (es. `avg_diff` ≥ una frazione del rumore osservato a parità di modello). Il vantaggio
  deve essere distinguibile dal rumore, non solo positivo in media.

Aggiornare `docs/reports/model_progress.xlsx` solo per candidati significativi.

#### Track parallelo a ceiling più alto: search a inference (PIMC/determinizzazione)

Non è "training di v7" ma probabilmente il guadagno maggiore a lungo termine. Ipotesi: gli errori residui sono linee
tattiche di metà/fine partita che una policy *memoryless* non vede. Una **ricerca determinizzata** (Perfect-Information
Monte Carlo / ISMCTS) con v6 come valutatore alle foglie, attivata quando lo spazio delle carte ignote è piccolo
(ultime ~6–10 carte), batte v6 in head-to-head. Riusa il solver endgame esatto già presente
(`ai/endgame/solver.py`, oggi solo a mazzo vuoto) come caso terminale. Costo: nuovo modulo (determinizer + PIMC) +
costo runtime (rilevante per la webapp). Cantiere separato dal "fai partire v7"; valutare dopo o in parallelo allo
screening league.

### 3. PPO/GAE Solo Dopo Un Blocco Reale Di A2C

Priorità bassa per ora.

- Valutare PPO/GAE solo se A2C league da v6 non produce miglioramenti ripetibili.
- Tenere l'esperimento piccolo e isolato, con test fast/numba verdi prima e dopo.
- Non introdurre DQN per ora: action mask, parziale osservabilità e self-play rendono più coerente continuare con policy-gradient.

### 4. Igiene

- Aggiornare badge coverage README solo dopo `pytest --cov=briscola_ai` se la variazione è materiale.
- Tenere `PLAN.md` breve: risultati intermedi e tentativi falliti vanno rimossi o sintetizzati.
- Non committare artefatti locali in `data/`, `benchmarks/`, `.claude/`.
- Commit in italiano.
- Continuare a non usare dati umani per promuovere modelli finché il volume resta basso; prima dell'uso ML reale verificare ancora privacy/qualità aggregata con `scripts/report_event_log.py`.

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
  --agent0-model data/models/best_a2c_v6.npz \
  --agent1 bc_model \
  --agent1-model data/models/best_a2c_v5.npz
```

Decision quality:

```bash
uv run python scripts/evaluate_decision_quality.py \
  --benchmark medium \
  --engine numba \
  --agent-a bc_model \
  --agent-a-model data/models/best_a2c_v6.npz \
  --agent-b heuristic_v1
```

Training A2C league da v6 fast+numba:

```bash
uv run python scripts/train_a2c.py \
  --encoder-version v3 \
  --rollout-engine fast \
  --fast-rollout numba \
  --init data/models/best_a2c_v6.npz \
  --opponent-mix bc_model:0.4,heuristic_v2:0.3,heuristic_v1:0.2,random:0.1 \
  --opponent-model data/models/best_a2c_v6.npz \
  --bc-anchor data/models/bc_v3.npz \
  --bc-anchor-beta 0.01
```

Benchmark performance:

```bash
uv run python scripts/benchmark_perf.py
```
