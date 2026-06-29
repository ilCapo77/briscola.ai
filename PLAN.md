# Piano operativo — Briscola AI

Questo file deve restare breve e utile per decidere cosa fare dopo. I dettagli storici, i benchmark intermedi non promossi e le prove superate sono stati rimossi: il codice, i test e i commit sono la memoria tecnica completa.

## Stato Corrente

- Versione progetto: `0.15.0` (`pyproject.toml`).
- Runtime/tooling: Python 3.14, FastAPI, Pydantic v2, `ruff`, `mypy`, `pytest`. Deps runtime lazy per il cloud: `redis`, `psycopg`. Dev-only: `fakeredis`, `playwright`.
- Dominio canonico: `src/briscola_ai/domain/`, con `GameState` immutabile e `step()` come transizione pura.
- Backend/UI: HTTP + WebSocket, UI statica servita dal backend. Stato partita in `GameSessionStore` (in-memory in locale, **Redis** se `REDIS_URL`); realtime via **pub/sub** dello store; event log SQLite o **Postgres** (`DATABASE_URL`). **Deployato** su FastAPI Cloud: <https://briscolaai.fastapicloud.dev>. Release cloud corrente verificata con modello v6; `v0.15.0` include default runtime `v6 + solver endgame`, PIMC 16×8 selezionabile, fix contrasto dropdown Windows e audit dataset `ai_action` per mosse IA/PIMC.
- Anti-cheat: agenti e modelli ricevono `PlayerObservation`, non `GameState` completo.
- Fast path: 2-player numerico Python/Numba per training/evaluation ad alto throughput.
- Encoder supportati: v1, v2, v3.
- Feature dim:
  - v1: `FEATURE_DIM_2P_V1`
  - v2: `FEATURE_DIM_2P_V2 = 288`
  - v3: `FEATURE_DIM_2P_V3 = 310`
- Modello consigliato: `data/models/best_a2c_v6.npz` (encoder v3, guard anti-overkill ON).
- Modelli precedenti ancora utili per confronto/regressioni: `data/models/best_a2c_v5.npz`, `data/models/best_a2c_v4.npz`, `data/models/best_a2c_v3.npz`, `data/models/best_a2c.npz` (legacy encoder v2).
- Default UI: avversario = `bc_model_hybrid_endgame` + modello consigliato (`BRISCOLA_DEFAULT_MODEL_ID`, oggi `best_a2c_v6.npz`), cioè v6 durante la partita e solver esatto a mazzo vuoto. Come opzione avanzata è selezionabile `bc_model_pimc_16x8`, cioè PIMC(v6, 16 determinizzazioni, max 8 carte vive ignote) + solver finale. `GET /api/ai/models` espone `recommended_model`; la UI seleziona quel modello se compatibile, altrimenti il `best_a2c_vN` più recente disponibile. `GET /api/ai/agents` espone `available`/`requires_model_id`/`requires_model_selection` e la UI disabilita gli avversari non giocabili. Il default lato API resta `random` se non specificato. In cloud, per restare dentro il limite disco, mantenere solo pochi asset (v6 obbligatorio; v5 opzionale per confronto manuale).
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
- Repo/release: history senza trailer `Co-Authored-By`; serie completa di tag di versione (`v0.1.0` → corrente) su GitHub; release `v0.12.1` pubblicata con `best_a2c_v6.npz`. `v0.13.0` cambia il default runtime a `bc_model_hybrid_endgame`; `v0.14.0` aggiunge `bc_model_pimc_16x8`; `v0.14.1` corregge il contrasto dropdown su Windows; `v0.15.0` aggiunge audit dataset `ai_action`, sempre senza nuovo asset modello.
- **Deploy COMPLETATO** su FastAPI Cloud (Redis collegato): <https://briscolaai.fastapicloud.dev>. Postgres/event log e modalità dataset sono attivabili via `DATABASE_URL` + `BRISCOLA_EVENT_LOG_MODE=dataset` quando serve raccogliere dataset umano.
- Rollout cloud `v0.12.1`/v6 completato e verificato: `/version` espone `recommended_model=best_a2c_v6.npz` e
  `recommended_model_present=true`; la UI mostra la label v6.
- Osservabilità/export cloud completati: produzione verificata con `DATABASE_URL` (Neon/Postgres), `BRISCOLA_EVENT_LOG_MODE=dataset` e `REDIS_URL`; `scripts/report_event_log.py` conta partite/consenso/finestre 24h-7d/qualità `human_action`; `scripts/audit_event_log_games.py` aggrega partite per versione/agente/modello e distingue log PIMC auditabili da log dataset minimali senza eventi IA; in dataset mode il backend salva ora anche `ai_action` self-contained con `decision_trace` minimale per audit IA/PIMC privacy-safe; `scripts/export_dataset.py` legge SQLite o Postgres mantenendo JSONL v1 e sanifica i nomi giocatore (`player_0`, `player_1`, ...). Smoke export produzione validato: 18 partite complete / 360 record.

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
- Usare `scripts/report_event_log.py` per controllare aggregati/qualità del log, `scripts/audit_event_log_games.py`
  per verificare versione/agente/modello/auditabilità e `scripts/export_ai_actions.py` per ispezionare singole
  mosse IA/PIMC auditabili senza stampare payload grezzi o campi sensibili.
- Dopo il deploy della patch `ai_action`, giocare nuove partite PIMC: le partite già raccolte in modalità dataset
  prima della patch identificano l'agente ma non contengono mosse IA auditabili.
- Non usare dati umani per training finché il volume resta basso e la qualità/privacy non sono state riverificate.
- Non avviare v7 solo per inerzia: serve una nuova ipotesi misurabile.

### 2. Prossima Iterazione IA (PIMC Prima Di `best_a2c_v7`)

Baseline da battere: `best_a2c_v6.npz`.

Storico scaling (seed 501, fast+numba, checkpoint 1M/3M/5M, guard inference ON), promosso a v6:

- 1M vs v5 big: `+0.03`; vs `heuristic_v1`: `+17.93`; overkill `0.0%`, waste `0.07%`;
- 3M vs v5 big: `+0.22`; vs `heuristic_v1`: `+18.18`; overkill `0.0%`, waste `0.09%`;
- 5M vs v5 big: `+0.46`; holdout vs v5 big: `+0.46`; vs `heuristic_v1`: `+18.40`;
  decision-quality vs `heuristic_v1`: `+18.58`, overkill `0.0%`, waste `0.07%`.

#### Direzione v7: search a inference prima di altro self-play

Esito della pre-validazione round-robin:

- **Scaling policy-only quasi saturo.** v6 ha speso 5× le partite (5M vs 1M) per arrivare a `+0.46` punti medi su v5
  a big. Il vantaggio esiste, ma è piccolo rispetto al costo.
- **Ipotesi anti-ciclo non confermata.** Aggiunto `scripts/evaluate_round_robin.py` e valutata la famiglia
  `{best_a2c_v2, v3, v4, v5, v6, heuristic_v1}`. Il round-robin mostra una famiglia monotona/transitiva: v6 resta
  primo, v5 secondo, nessun ciclo A>B, B>C, C>A confidente.
- **CI/gate implementati.** Il round-robin riporta CI Wilson sullo score rate, CI analitica sull'`avg_diff` quando è
  disponibile la varianza per-partita, e il detector dei cicli considera un arco A>B solo se la CI è interamente oltre
  `0.5`. Nota di robustezza: con `--suite both`, in futuro aggregare per coppia prima di cercare cicli eviterebbe
  archi contraddittori se standard e holdout divergessero.
- **v6 > v5 è piccolo ma credibile a big.** Follow-up v5-v6 standard+holdout, `100k` partite per suite: score v6
  aggregato `0.5080` (CI95 `0.5058..0.5102`), avg diff `+0.46` (CI95 `+0.34..+0.58`).

Conclusione: non fare `best_a2c_v7` come "stesso recipe, più partite" e non trattare più la population league come
ipotesi primaria. La motivazione specifica anti-ciclo è venuta meno; resta solo un possibile test di robustezza.

Ipotesi primaria misurabile: una **ricerca determinizzata a inference** (PIMC/ISMCTS leggero) con v6 come policy o
valutatore alle foglie, attivata quando lo spazio delle carte ignote è piccolo, batte v6 puro in head-to-head con un
vantaggio statisticamente distinguibile e un costo runtime accettabile per la webapp.

Passi consigliati:

- Prototipo offline PIMC/determinizzazione 2-player: **avviato**.
  - generare determinizzazioni compatibili con informazione pubblica + mano del player;
  - usare v6 per scegliere/valutare rollout o foglie;
  - riusare il solver endgame esatto (`ai/endgame/solver.py`) quando il mazzo è vuoto;
  - attivare la search solo nel finale o semi-finale (es. ultime ~6–10 carte ignote), con budget fisso.
- Implementazione iniziale: `PIMCAgent` in `ai/agents/pimc.py` + `scripts/evaluate_pimc.py`. Smoke run locale
  `PIMC(v6)` vs v6 (`20` partite, `4` determinizzazioni, max unknown `8`) completata: score `0.5250`, avg diff
  `+5.70`, CI95 score `0.3205..0.7215`, CI95 avg diff `-4.60..+16.00`. Non è evidenza statistica, solo verifica
  end-to-end. La strumentazione runtime misura le sole mosse in cui PIMC cerca davvero: in questa smoke `50` search
  move, `~0.020s/search_move` (`~0.05s/game` medio, diluito dal fallback).
- Sweep preliminare con seed fisso `777`, `1000` partite/config:
  - control `max_unknown=0`: score `0.5185` (CI95 `0.4875..0.5493`), avg diff `+1.59` (CI95 `-0.06..+3.23`),
    `search=0`, `coerced_moves=0`. Nota: questo non è un null test puro, perché `PIMCAgent` usa comunque il solver
    esatto a mazzo vuoto prima del gate `max_unknown`; misura quindi `v6 + solver endgame` contro v6 puro. Il `+1.59`
    va letto come beneficio preliminare del solver endgame, non come bias dell'harness;
  - `8×8`: avg diff `+3.89` (CI95 `+2.24..+5.54`), `0.039s/search_move`;
  - `16×8`: avg diff `+4.71` (CI95 `+3.08..+6.34`), `0.079s/search_move`;
  - `16×10`: avg diff `+4.99` (CI95 `+3.37..+6.61`), `0.083s/search_move` → miglior segnale forza finora;
  - `16×12`: avg diff `+4.19` (CI95 `+2.56..+5.83`), `0.118s/search_move` → più costoso e meno forte;
  - blocco `32×*` sospeso dopo costo eccessivo osservato su `32×8` (interrotto senza JSON): non Pareto per una prima
    decisione UI/offline. Nessuna config completata ha avuto `coerced_moves > 0`.
- Esperimento decisivo completato: `PIMC(v6, 16×10)` contro control solver (`v6 + solver endgame`, `max_unknown=0`),
  `2000` partite, seed `777`: score `0.5635` (CI95 `0.5417..0.5851`), avg diff `+3.59` (CI95 `+2.48..+4.70`),
  `0.0805s/search_move`, `coerced_moves=0` su entrambi i lati. La determinizzazione aggiunge valore reale anche al
  netto del solver endgame. Il costo percepito lato utente è probabilmente basso; il vincolo reale è CPU server sotto
  concorrenza.
- Pareto diretto completato: `PIMC(v6, 16×8)` contro `PIMC(v6, 16×10)`, `2000` partite, seed `777`: score `0.5130`
  (CI95 `0.4911..0.5349`), avg diff `+0.25` (CI95 `-0.85..+1.35`) a favore di `16×8`, quindi nessuna evidenza che
  `16×10` sia più forte. `16×8` fa `5013` search move contro `7000` del `16×10` nella stessa run: candidato Pareto
  per un eventuale agente live.
- Caveat: questi vantaggi sono misurati contro v6. In UI contro umani il modello avversario usato nei rollout è
  mis-specificato; non spacciare questi numeri come forza attesa contro giocatori umani.
- Risultato deployabile indipendente: `control_solver(v6)` = `v6 + solver endgame` a runtime. Il solver è esatto,
  anti-cheat, già validato e praticamente gratis; i numeri preliminari indicano un vantaggio di circa `+1.3..+1.6`
  punti medi su v6 puro. Trattarlo come candidato a basso rischio: l'unità deployabile futura diventa **rete + solver
  runtime**, non solo `.npz`. Non serve distillare il solver dentro la rete: la lineage di v6 discende già da teacher
  con endgame solver, ma il vantaggio del solver non è sopravvissuto nei pesi.
- Implementazione app: aggiunto agente `bc_model_hybrid_endgame`, cioè modello `.npz` scelto dalla UI + solver esatto
  a mazzo vuoto. La UI lo usa come default quando esiste un modello compatibile, mantenendo il selettore modello.
  Prossimo passo prima del rollout: validazione seat-fair `bc_model_hybrid_endgame(best_a2c_v6)` vs `bc_model(best_a2c_v6)`
  su `>=2000` partite e smoke manuale in locale/cloud.
- Implementazione PIMC app: aggiunto agente selezionabile `bc_model_pimc_16x8`, cioè modello `.npz` scelto dalla UI
  come fallback/rollout, solver a mazzo vuoto e PIMC con `16` determinizzazioni quando le carte vive ignote sono
  `<=8`. Non è default: serve come modalità avanzata per misurare costo CPU e feedback umano.
- Follow-up "più mosse PIMC" dopo export produzione `ai_action`: in una partita reale `bc_model_pimc_16x8` ha prodotto
  `15` fallback, `2` search e `3` solver. Test diretti con seed `20260629` non supportano l'allargamento della finestra:
  `16×12` vs `16×8`, `500` partite, score `0.4900` (CI95 `0.4464..0.5337`), avg diff `-0.80`
  (CI95 `-3.01..+1.40`), search `4.50` vs `2.50` per partita; `16×16` vs `16×8`, `300` partite, score `0.4917`
  (CI95 `0.4356..0.5480`), avg diff `-0.21` (CI95 `-3.04..+2.61`), search `6.50` vs `2.45` per partita.
  Conclusione operativa: più mosse search aumentano costo CPU ma non mostrano un vantaggio; mantenere `16×8` come
  variante Pareto finché dati più forti non indicano il contrario.
- Asse sperimentale successivo: usare PIMC come **teacher offline** e distillare solo le correzioni di **search** in un
  eventuale `best_a2c_v7` veloce:
  - generare posizioni da self-play/replay v6, soprattutto finale e semi-finale: script
    `scripts/generate_pimc_teacher_dataset.py` implementato;
  - etichettare le mosse con `PIMC(v6, 16×8)` e solver endgame quando applicabile; lo script salva JSONL compatibile
    con `train_bc.py` e, di default, fa avanzare le partite con v6 per mantenere la distribuzione stati del modello base;
  - il dataset deve includere anche esempi fuori finestra search: in quelle posizioni PIMC delega al fallback v6, quindi
    l'obiettivo diventa "v6 ovunque + correzioni PIMC nel finale" invece di sole etichette di finale;
  - aggiornata diagnostica teacher: `PIMCAgent` espone per ogni decisione search valori medi per carta, margine
    best-second, delta paired per determinizzazione, SE e CI95 del margine; il generatore salva questi campi e conta
    i disaccordi `PIMC != v6` forti+affidabili (`margin >= --strong-margin-min` e CI lower bound >=
    `--reliable-margin-ci-low-min`). Prossimo gate: prima di qualunque retrain, generare abbastanza partite per
    misurare se esistono davvero `15k..30k` disaccordi search ad alto margine/affidabilità. Smoke diagnostico
    `5k`, `64` determinizzazioni, `u=8`, seed `20260630`: `2275` search, `2725` solver, `1062` search in disaccordo
    con v6; `424` search forti+affidabili con soglia `margin>=2` e CI low `>=0` (`310` se CI low `>=2`). Segnale
    presente ma non abbondante: per arrivare a `15k` esempi utili servono circa `175k` record in finestra PIMC con
    queste soglie, salvo cambiare soglia/config teacher.
  - mini-confronto teacher `16×8` vs `64×12`, `200` partite, seed `777`: score `0.5150` (CI95 `0.4461..0.5833`),
    avg diff `+0.20` per `16×8` (CI95 `-3.23..+3.63`), `64×12` a `0.3226s/search_move`. Il test è sottodimensionato:
    non dimostra equivalenza e non esclude un vantaggio piccolo ma utile del teacher più pesante. Per il primo dataset
    usare comunque `16×8` come prima passata; rivalutare teacher più costosi solo dopo aver misurato il distillation gap
    di v7, e con un confronto teacher a potenza adeguata se lo studente riesce a copiare bene `16×8`;
  - micro-run completato: `data/pimc_teacher_v7_50k_seed777.jsonl` (`50k` esempi, gitignored) con `36,250` fallback/v6,
    `6,250` search PIMC, `7,500` solver, `coerced_moves=0`; data path BC v3 verificato (`50000×310`);
  - allenare BC/policy distillation warm-start da v6, mantenendo guard inference e feature encoder v3; `train_bc.py`
    supporta ora `--init` MLP e `--bc-anchor ... --bc-anchor-beta ...` per preservare v6 durante il fine-tuning;
  - primo candidato conservativo (`1` epoca, lr `1e-4`, anchor beta `0.05`) e' quasi pari a v6 su medium: avg diff
    `+0.06` vs v6 (`10k` partite), ma perde contro `control_solver(v6)` (`-1.28`, CI95 `-2.44..-0.13`) e resta
    molto sotto il teacher `PIMC(v6,16×8)` (`-3.92`, CI95 `-5.59..-2.26`, `1000` partite). Decision-quality vs
    `heuristic_v1` resta pulita (`+17.96`, waste `0.1%`, overkill `0.0%`). Non promuovere;
  - filtro disaccordi quantificato: solo `5,369/50,000` esempi sono veri `teacher != v6` (`3,017` search, `2,352`
    solver; tutti i `36,250` fallback coincidono con v6). `train_bc.py` supporta `--filter-disagree-with-model` e usa
    la mossa effettiva del modello, incluso overkill guard;
  - retrain solo-disaccordi: recipe aggressiva (`10` epoche, lr `3e-4`, anchor `0.05`) impara disaccordi ma degrada
    forte (`-4.01` vs v6). Recipe conservativa (`5` epoche, lr `1e-4`, anchor `0.20`) resta circa pari a v6 (`+0.17`)
    ma perde ancora contro `control_solver(v6)` (`-1.18`, CI95 `-2.32..-0.04`) e resta molto sotto PIMC (`-3.69`,
    CI95 `-5.30..-2.09`). Non promuovere;
  - conclusione provvisoria: il collo di bottiglia e' la distillazione, non la forza del teacher. In particolare il
    solver non va più trattato come comportamento da comprimere: va eseguito a runtime. La domanda utile diventa se una
    rete `v7-search`, combinata con lo stesso solver runtime, batte `v6 + solver`;
  - prossimo esperimento distillazione: generare un dataset **search-ricco** con più partite e/o `max_unknown` più largo,
    puntando a circa `15k..30k` disaccordi search (`teacher.search != v6`). Tenere esempi v6/fallback come
    copertura eventuale, ma usare soprattutto l'anchor CE a v6 come anti-dimenticanza: le etichette hard v6 rischiano
    di ridiluire il segnale search. Pesare o sovracampionare i disaccordi search; ignorare i disaccordi solver come
    target primario;
  - protocollo pre-registrato per evitare cherry-picking: usare seed/config di validazione per scegliere la recipe, poi
    confermare **un solo** candidato su seed held-out, seat-fair, `>=2000` partite (meglio `4000`). Solo la conferma
    held-out entra nel piano come evidenza di promozione; le run di selezione restano diagnostiche;
  - criterio di promozione v7: valutare `(v7-search + solver)` contro `(v6 + solver)` con CI95 positiva e materiale
    sull'avg diff (lower bound `> +0.5` punti medi, non solo `> 0`), più gap vs PIMC teacher e decision-quality. Se a
    potenza adeguata la CI include `0` o è negativa, chiudere l'idea "v7 distillato" come risultato negativo misurato e
    usare `control_solver(v6)` come baseline deployabile, con PIMC live opzionale.

Screening population league declassato a opzionale:

- Si può ancora fare uno screening economico (~200k partite) come test di robustezza, non come direzione principale.
- Opponent-mix indicativo: `{v6:0.3, v5:0.15, v4:0.1, v3:0.05, heuristic_v2:0.2, heuristic_v1:0.1, random:0.1}`,
  warm-start da v6.
- Kill criterion: fermarsi se non migliora l'Elo round-robin vs `{v3..v6}` + `heuristic_v1` con CI positiva, o se il
  matchup peggiore regredisce. Niente run 1M+ senza segnale nello screening.
- Costo implementativo: il fast-rollout numba accetta oggi **un solo tipo di opponent modello per batch**
  (`scripts/train_a2c.py:1010`). Il plumbing round-robin è stato aggiunto; resta da implementare il campionamento
  multi-`.npz` nel training prima di poter fare davvero fictitious self-play.

Criteri generali di promozione:

- vantaggio riportato con **intervallo di confidenza** coerente con la metrica (Wilson/analitico quando basta,
  bootstrap se servono risultati per-partita);
- niente promozione se il CI tocca lo zero/0.5 o se il delta è sotto una soglia minima predefinita;
- holdout vs `heuristic_v1` non peggiore;
- `trump_waste_rate` e `trump_overkill_rate` non peggiorano materialmente;
- aggiornare `docs/reports/model_progress.xlsx` solo per candidati significativi.

### 3. PPO/GAE Solo Dopo Un Blocco Reale Di A2C

Priorità bassa per ora.

- Valutare PPO/GAE solo se né il search PIMC né un eventuale screening league producono miglioramenti ripetibili.
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
