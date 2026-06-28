# Briscola AI

![Coverage](https://img.shields.io/badge/coverage-67%25-yellow)

Un progetto didattico “end‑to‑end” nato da un’esigenza concreta: **studiare le reti neurali con un progetto reale**, non con esempi astratti.

La Briscola è un ottimo “laboratorio” perché obbliga a mettere insieme tutti i pezzi:
- un **motore di regole** corretto e testabile;
- un **backend** che espone un contratto stabile (API/WS);
- una **UI** che rende leggibile la sequenza delle mani;
- una **pipeline dati** per arrivare a dataset, baseline e training.

Obiettivo: arrivare a un’IA (rete neurale) che impari a giocare in modo riproducibile, misurabile e spiegabile.

> Questo README spiega *come* usare e *perché* è fatto così (parte didattica). Per lo stato corrente e le prossime
> azioni vedi `PLAN.md`; il dettaglio operativo dei comandi è in `--help` di ogni script.

## Funzionalità

- Regole complete della Briscola; motore (`domain/`) con **2 giocatori** e **4 giocatori** (a squadre).
- Interfaccia web con aggiornamenti in tempo reale via WebSocket.
- IA selezionabile, server‑driven (il backend avanza la partita quando tocca all’IA):
  - baseline: `random`, `greedy_points`, `heuristic_v1`, `heuristic_v2`;
  - ibridi endgame: `hybrid_endgame`, `hybrid_endgame_best_a2c` (euristica/modello + solver esatto nel finale);
  - modelli locali `bc_model`: file `.npz` scelto dalla UI da un catalogo server‑side (no path arbitrari dal browser).
- Encoder osservazione **v1 / v2 / v3** (vedi sotto) e fast path numerico Python/Numba per training ed evaluation veloci.
- Pipeline dati completa: event log SQLite → export JSONL → self‑play → valutazione offline → training BC/RL.

Nota didattica:
- la UI è pensata e testata soprattutto in **2 giocatori** (modalità principale);
- il 4‑player è supportato dal motore (usato per regressione) ma **non è ancora pienamente coperto dal frontend**.

## Quick start

Questo progetto usa [uv](https://github.com/astral-sh/uv). Requisiti: Python **3.14** e `uv`.

```bash
uv venv -p python3.14
uv pip install -e ".[dev]"   # runtime + strumenti dev
briscola-server --reload     # UI su http://localhost:8000
```

## Come giocare

1. Inserisci il tuo nome, scegli l’avversario (IA) e premi “Avvia partita” (la UI mostra una descrizione dell’IA dai metadati del backend).
2. Clicca una carta in mano per giocarla.
3. L’IA risponde automaticamente al suo turno.

La UI avvia partite **2‑player**. Per flussi 4‑player (senza UI) usa gli script headless o le API.

### Giocare contro un modello locale (`.npz`)

Se hai addestrato un modello (BC / PG / A2C) salvato in `.npz`, puoi usarlo come avversario:

1. Metti il file in una directory whitelist lato server: consigliato `./data/models/` (oppure imposta `BRISCOLA_MODELS_DIR`).
2. (Ri)avvia il server.
3. Nella UI scegli **“Modello locale (.npz)”** e seleziona il file dal dropdown.

Note:
- il dropdown mostra `metadata_json.label`/`description_it` del `.npz` (i trainer del progetto li salvano in automatico);
- un `.npz` incompatibile (chiavi mancanti o `feature_dim` non supportata) viene segnalato dal catalogo e disabilitato nella UI;
- **sicurezza**: il browser invia solo un `ai_model_id` (path relativo) tra quelli esposti da `GET /api/ai/models`; il backend rifiuta path traversal e carica solo dentro `BRISCOLA_MODELS_DIR`.

## Approccio step‑by‑step (didattico)

L’idea è costruire una pipeline ML “dal basso”, in modo verificabile:

1. **Dominio testabile**: regole e transizioni pure in `domain/` + test su invarianti e casi limite.
2. **Backend/UI**: FastAPI + WS per far giocare umani e rendere osservabile lo stato.
3. **Raccolta dati**: event log SQLite (append‑only).
4. **Export dataset**: SQLite → JSONL con schema versionato.
5. **Self‑play**: generazione rapida di partite dal dominio.
6. **Valutazione**: match offline riproducibili (win‑rate/punti medi).
7. **Training**: imitation/RL quando contratti e pipeline sono stabili.

## Struttura del progetto

- `src/briscola_ai/domain/` – dominio canonico, puro e testabile
  - `models.py` (`Card`/`Suit`/`Rank`), `state.py` (`GameState`), `engine.py` (`step(state, action)`), `rules.py`, `observation.py`, `card_id.py` (mappa carta ↔ id 0–39), `serialization.py` (`GameState` ↔ dict JSON)
- `src/briscola_ai/backend/` – adattatore HTTP/WS (FastAPI): `dto.py` (Pydantic v2), `server.py`, `game_store.py` (stato partita in‑memory/Redis + pub/sub), `event_log.py` (SQLite/Postgres), `observation_builder.py`
- `src/briscola_ai/ai/`
  - `agents/` – baseline, ibridi endgame, factory e catalogo agenti
  - `endgame/` – solver esatto del finale 2‑player
  - `encoding/` – encoder v1/v2/v3 e spazio azioni
  - `models/` – agente modello `.npz`, catalogo per la UI e provisioning modello
  - `fast/` – motore "fast" 2‑player (interi/array NumPy)
  - `numba/` – kernel JIT Numba (vedi nota sotto)
  - `evaluation/` / `training/` – valutazione offline e componenti training condivisi
- `src/briscola_ai/frontend/static/` – UI (HTML/CSS/JS), immagini carte in `assets/cards/`
- `tests/` – unit + integrazione API/WS (pytest)
- `scripts/` – simulazione, self‑play, export, training, evaluation, benchmark
- `docs/reports/model_progress.xlsx` – report Excel curato dei modelli significativi e delle milestone di promozione
- `PLAN.md` – stato corrente e prossime azioni (fonte di verità). `data/` e `benchmarks/` sono artefatti locali (gitignored).

### I tre motori dello stesso gioco (dominio · fast · numba)

Lo **stesso** gioco è implementato a tre livelli, tenuti **in parità dai test** (`tests/test_fast_*`):

- **dominio** (`domain/engine.py`) — il motore "standard": puro, immutabile, leggibile. È la **fonte di verità**, usato da backend, UI e test. Ottimizzato per chiarezza, non per velocità.
- **fast** (`ai/fast/`) — riscrittura 2‑player su **interi/array NumPy** (niente oggetti `Card`/`GameState`): stessa logica, molto più veloce. Serve a self‑play, training ed evaluation massivi.
- **numba** (`ai/numba/`) — gli stessi kernel del fast path compilati **JIT con Numba**: ancora più rapidi.

Negli script si scelgono con `--engine domain|fast|numba` (es. `evaluate_agents.py`, `--rollout-engine`/`--fast-rollout` in `train_a2c.py`). Regola d'oro: il dominio decide la correttezza; fast/numba devono dare **risultati identici** (se cambi una regola nel dominio, aggiorna anche fast/numba e i test di parità). I numeri di throughput sono nella sezione [Performance](#performance-fast-path-pythonnumba).

## Backend (FastAPI + WebSocket)

Architettura ibrida HTTP + WebSocket.

**Perché ibrida?** Il polling REST è inefficiente e ad alta latenza per un gioco real‑time; il solo WebSocket complica azioni puntuali, testing e gestione errori. Quindi:
- **REST** per le *azioni* del client (semantica chiara, stateless, debug facile);
- **WebSocket** per gli *aggiornamenti* dal server (push in tempo reale).

### Endpoint HTTP (REST)

| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| `POST` | `/api/games` | Crea una nuova partita |
| `GET` | `/api/games/{id}` | Stato completo (DTO `type: "game_state"`, debug/spettatori) |
| `GET` | `/api/games/{id}?player_index={i}` | Vista del giocatore `i` (`type: "observation"`) |
| `POST` | `/api/games/{id}/actions` | Il giocatore gioca una carta |
| `GET` | `/api/games/{id}/result` | Risultato finale |

### WebSocket (tempo reale)

Connessione: `ws://host/api/ws/{game_id}/{player_index}`. **Tutti i messaggi includono un campo `type`**:

| Messaggio | `type` | Descrizione |
|----------|---------|-------------|
| Snapshot (observation) | `observation` | Stato della partita per il giocatore del WS (`my_hand`, `my_turn`, `table_cards`, …) |
| Reveal IA | `ai_card_reveal` | L’IA mostra la carta che sta per giocare |
| Risultato mano | `trick_result` | Carte, vincitore, punti della mano |
| Keepalive | `pong` | Risposta ai ping del client |

Gli snapshot includono `server_version` (intero monotono, incrementato a ogni azione) per diagnosticare ordering/reconnect. Regola client: `type === "observation"` → snapshot, altrimenti evento.

### Flusso di gioco e modello server‑driven

```
POST /api/games            -> game_id
WS connect /api/ws/{id}/0  -> snapshot
POST .../actions (gioca)   -> snapshot
                           -> ai_card_reveal -> trick_result -> snapshot
```

Il backend è “server‑authoritative”: dopo la mossa umana, se tocca all’IA gioca da solo ed emette `ai_card_reveal` → `trick_result` → snapshot. Il frontend gestisce solo la **presentazione** (mette in coda gli eventi e applica gli snapshot dopo gli hold), così la **logica di gioco** (backend) resta separata dalla **logica di presentazione** (frontend). Il backend non introduce delay di animazione (niente `asyncio.sleep()`).

**Stato e scalabilità (multi‑replica).** Lo stato delle partite vive in un `GameSessionStore` (`backend/game_store.py`): **in memoria** in locale, **Redis** in cloud quando è impostata `REDIS_URL`. In deploy con più repliche questo evita il “partita non trovata” (azioni/WS che colpiscono repliche diverse). L'architettura REST+WebSocket resta invariata: il fan‑out degli eventi WS passa per il **pub/sub** dello store, così ogni client riceve `ai_card_reveal`/`trick_result`/`observation` da qualsiasi replica. Gli `observation` per‑giocatore sono ricostruiti dal subscriber (anti‑cheat: mai lo stato completo).

## Frontend (UI web)

**Smoke test manuale** (2–3 min): avvia il server, apri `http://localhost:8000` con la Console DevTools aperta, gioca 3 mani complete e verifica che la sequenza sia sempre **reveal → seconda carta → risultato**, senza errori in console, freeze o carte duplicate sul tavolo.

**Debug** quando “si blocca”: controlla la Console (warning su `observation`/`server_version`) e la tab Network → filtro **WS** (connessione attiva, messaggi `observation`/`ai_card_reveal`/`trick_result`). Modalità senza WebSocket (polling) per isolare i bug: apri `http://localhost:8000/?polling=1`.

## Sviluppo (test, lint, typecheck)

Con le dev deps installate (`uv pip install -e ".[dev]"`):

```bash
ruff format src tests scripts
ruff check src tests scripts
mypy src
pytest                       # test
pytest --cov=briscola_ai --cov-report=term-missing   # coverage
```

Il badge coverage in cima è manuale (Shields.io): aggiornalo dopo `pytest --cov` se la variazione è materiale.

## AI & ML

### Anti‑cheat: osservazione parziale (information set)

Nel dominio `GameState` contiene informazione **completa** (ordine del mazzo, mani di tutti). Se un agente la ricevesse, potrebbe “barare” leggendo informazione nascosta, rendendo i benchmark non significativi. Per questo:

- gli agenti ricevono una `PlayerObservation`, vista **parziale e lecita**, costruita con `make_player_observation(state, player_index)`;
- contiene: mano del giocatore, carte sul tavolo, briscola scoperta, `deck_size`, punteggi e dimensioni mani, e due one‑hot pubbliche (sotto);
- **non** contiene: il mazzo come sequenza (`state.deck`) né le carte specifiche degli avversari.

Riferimenti: `domain/observation.py`, test `tests/test_domain_observation.py`.

Due one‑hot pubbliche (entrambe lecite, derivate solo da informazione visibile):
- `seen_cards_onehot[40]`: carte **viste** = briscola scoperta + tavolo + carte uscite nelle prese;
- `out_of_play_cards_onehot[40]`: carte **non più disponibili** = prese + tavolo (la briscola scoperta NON è qui finché è pescabile/in mano). Invariante: `out_of_play ⊆ seen`.

### Encoder: v1, v2, v3

Lo stesso stato lecito può essere codificato a livelli crescenti di “memoria/strategia”:

- **v1** (`feature_dim=248`): vista istantanea (mano, tavolo, briscola, scalari di stato).
- **v2** (`288`): v1 + `seen_cards_onehot[40]` → card counting lecito (storia pubblica).
- **v3** (`310`): v2 + 22 feature **strategiche aggregate**, leggibili: briscole/carichi ignoti, assi/tre usciti per seme, fase partita (`deck_size`, carte in mano, endgame flag), e info sulla presa corrente. Usa `out_of_play_cards_onehot` per distinguere “visto” da “fuori gioco”.

L’encoder canonico vive in `ai/encoding/observation_encoder.py`; esiste in versione **domain** (oggetto), **fast** (Python) e **Numba**, con test di **parità** che garantiscono lo stesso vettore. In partita (`ai_agent=bc_model`) il backend sceglie l’encoder dai metadati del modello (`encoder_version`) o, in fallback, dalla `feature_dim` (248/288/310).

### Agenti disponibili

- `random` – carta casuale (baseline “zero”).
- `greedy_points` – gioca la carta con più punti in mano (spiegabile, spesso sub‑ottimale).
- `heuristic_v1` – euristica 2‑player: prende “a basso costo” quando conviene, scarta in modo economico altrimenti.
- `heuristic_v2` – come v1 ma usa la storia pubblica (`seen_cards_onehot`) per gestire meglio le briscole (buon teacher per BC).
- `hybrid_endgame` – `heuristic_v2` nel mid‑game + **solver esatto** a mazzo vuoto.
- `hybrid_endgame_best_a2c` – modello `best_a2c` nel mid‑game + solver nel finale.
- `bc_model` – modello locale `.npz` (BC/PG/A2C), encoder dedotto dai metadati.

Il **solver endgame** (`ai/endgame/solver.py`) calcola la mossa ottima esatta con minimax a mazzo vuoto; l’agente ibrido lo usa in modo **anti‑cheat** ricostruendo lo stato di finale dalla sola `PlayerObservation`.

### Raccolta dati ed export

Avviando `briscola-server` viene scritto un event log su `./data/briscola_events.sqlite3` (gitignored). Path/modalità configurabili via CLI (`--event-db`, `--event-log-mode`) o env (`BRISCOLA_EVENT_DB_PATH`, `BRISCOLA_EVENT_LOG_MODE`). Per ogni partita si salvano `seed`, `code_version`, `rules_version`. In cloud multi‑replica l'SQLite locale è per‑replica ed effimero: imposta `DATABASE_URL` per usare un event log **Postgres** (Neon) persistente e condiviso (`PostgresEventLog`, stesso schema `games`/`events`).

Due modalità di logging:
- `debug` (default): completa, utile per troubleshooting;
- `dataset`: pensata per raccogliere partite **umane** tenendo il DB piccolo — salva un evento `human_action` self‑contained (observation → action → reward/done → next_observation) + marker `game_finished`, **non** salva `player_names` (privacy), usa un `client_id` pseudonimo per split per‑giocatore, ed esige il **consenso** (checkbox UI; il backend rifiuta `POST /api/games` senza consenso).

Export in JSONL (schema v1, pensato per il 2‑player):

```bash
python scripts/export_dataset.py --db ./data/briscola_events.sqlite3 --out ./data/dataset.jsonl
```

Default: solo azioni del `player_index=0`, escluse quelle IA, solo partite complete. Opzioni utili: `--all-players`, `--include-ai`, `--no-next-state` (supervised), `--include-incomplete`. In modalità `dataset` l’exporter usa preferibilmente `human_action`.

**Deploy (cloud multi‑replica)**: imposta `REDIS_URL` (stato partita condiviso + realtime via pub/sub) e, per la raccolta dati persistente, `DATABASE_URL` (event log Postgres). Restringi le origin con `BRISCOLA_CORS_ALLOW_ORIGINS=https://tuodominio` (default `*`, solo per sviluppo). L'elenco completo delle variabili d'ambiente è in `AGENTS.md`. Sito live: <https://briscolaai.fastapicloud.dev>.

### Simulazioni e self‑play (headless)

```bash
# Simulazione semplice (dominio)
python scripts/simulate_games.py --num-games 100 --seed 42 --num-players 2

# Self-play verso SQLite (scegli gli agenti per seat con --agents)
python scripts/self_play_to_db.py --db ./data/briscola_events.sqlite3 --num-games 100 --seed 42 --agents heuristic_v1,random

# Fast self-play "summary-only" (no DB/DTO): seed/agenti/punti/vincitore, una riga per partita
python scripts/fast_self_play.py --num-games 100000 --seed 0 --agents greedy_points,random --out-jsonl /tmp/fast.jsonl
```

### Valutazione

Confronti riproducibili senza UI/server:

```bash
python scripts/evaluate_agents.py --benchmark medium --engine domain \
  --agent0 bc_model --agent0-model ./data/models/best_a2c_v4.npz --agent1 heuristic_v1
```

Concetti chiave:
- **engine**: `domain` (canonico, supporta tutti gli agenti), `fast`/`numba` (più veloci; `numba` supporta modelli MLP vs baseline fast‑compatible).
- **seat‑fair** (`--seat-fair`): per ogni seed gioca due partite scambiando i posti (rimuove il bias “il player 0 inizia sempre”). Richiede `--num-games` pari.
- **seed suite** riproducibili: `--seed-suite small|medium`, `--seed-suite-file`, oppure `--seed-suite-range-start N` (utile per **holdout**, es. `--seed-suite-range-start 1000000`).
- **preset**: `--benchmark small|medium|big` (= 2000/10000/100000 game, sempre seat‑fair) e `--out-json` per salvare.

Strumenti aggiuntivi:
- `scripts/evaluate_matrix.py` – valuta un modello contro una lista di avversari su due suite (`standard` e `holdout`).
- `scripts/evaluate_decision_quality.py` – metriche di **stile**, non solo forza:
  - `trump_waste_rate`: gioca briscola pur avendo una risposta vincente **non‑briscola**;
  - `trump_overkill_rate`: quando vince con briscola, usa una briscola più costosa del necessario.
- **Guard anti‑overkill** (`inference_overkill_guard`): post‑processing che, da secondo di mano, gioca la briscola vincente **minima**. Attivabile dai metadati del modello (i trainer lo salvano con `--inference-overkill-guard`) o, per A/B, con `BRISCOLA_BC_OVERKILL_GUARD=1`. È deterministico: verifica sempre l’impatto con le metriche.

### Performance (fast path Python/Numba)

Il dominio canonico è la fonte di verità; il fast path 2‑player (`ai/fast/`) replica la stessa logica su interi/array per alzare il throughput, con kernel JIT in `ai/numba/`. È tenuto coerente dai test di parità. Misure con `scripts/benchmark_perf.py` (modi `*-random`, `fast-eval`, `numba-eval`, `numba-mlp`).

Esempio dell’ordine di grandezza: il **training A2C v3** su 20k partite passa da ~419 games/sec (`--rollout-engine domain`) a ~5900 games/sec (`--rollout-engine fast --fast-rollout numba`), ~14×; questo rende fattibili run da 1M partite in pochi minuti.

### Pipeline di training (BC → RL)

Idea didattica: prima un modello supervisionato che **imita** un teacher (Behavior Cloning), poi RL per **superarlo**.

**Spazio azioni**: “40 carte + action mask” (il modello sceglie tra 40 classi; la mask abilita solo le carte in mano).

**Behavior Cloning** (`scripts/train_bc.py`): allena su un JSONL esportato un modello (lineare o MLP) che riproduce le scelte del teacher. Encoder selezionabile con `--encoder-version v1|v2|v3` (v3 richiede dataset con `out_of_play` popolato).

**Reinforcement Learning**: BC tende a *eguagliare* il teacher, non a superarlo. Per superarlo:
- **REINFORCE** (`scripts/train_pg.py`): policy gradient sul return finale. È corretto ma rumoroso.
- **A2C** (`scripts/train_a2c.py`, consigliato): aggiunge un *critic* `V(s)` e usa l’**advantage** `A = G − V(s)` come baseline appresa, con un reward più denso (delta `punti_policy − punti_opp` per turno della policy, senza barare).

Tecniche utili (tutte come flag, vedi `--help`):
- **opponent mix** (`--opponent-mix name:peso,...`) per robustezza (evita overfitting su un avversario);
- **warm‑start** da un BC (`--init`) e **BC‑anchor** (`--bc-anchor ... --bc-anchor-beta`) per restare vicino allo stile del teacher;
- **reward shaping anti‑overkill** (`--overkill-penalty-mode flat|gap --overkill-penalty-beta`);
- **league**: allenare contro un campione congelato. Attenzione: l’alias agente `best_a2c` carica il file **legacy** `best_a2c.npz` (encoder v2), **non** il campione attuale v4. Per allenare contro il best v4 usa `bc_model` con path esplicito nel mix, es. `--opponent-mix bc_model:0.5,heuristic_v1:0.3,random:0.2 --opponent-model ./data/models/best_a2c_v4.npz` (sul fast rollout Numba è supportato al più un tipo di opponent‑modello per mix);
- **curriculum** (`--curriculum easy_standard_hard`) per stage easy→standard→hard.

**Pipeline riproducibile** (`scripts/run_experiment.py`): un comando unico fa training → evaluation matrix → manifest → aggiorna il best locale. Supporta `--rollout-engine fast --fast-rollout numba` e `--eval-engine numba` per i run lunghi. Output in `data/models/` e `benchmarks/experiments/<name>/`.

> I dettagli (decine di varianti di comando e numeri di benchmark) vivono in `--help` degli script e nei commit:
> qui teniamo solo i concetti. La cronologia degli esperimenti e i risultati promossi sono in `PLAN.md`.

### Baseline AI ufficiale

Il modello consigliato è **`data/models/best_a2c_v4.npz`** (encoder v3, guard anti‑overkill ON), promosso perché migliora `best_a2c_v3` nel confronto head‑to‑head big e non peggiora l'holdout vs `heuristic_v1`. `best_a2c_v3.npz` resta selezionabile per confronto se presente nella directory modelli; il vecchio `best_a2c.npz` resta utile per regressioni v2. I file `.npz` sono artefatti **locali** (gitignored): la ricetta di riproduzione del best v4 è in `PLAN.md`.

La release `v0.10.0` usa `best_a2c_v4.npz` come modello consigliato anche in cloud. Per un nuovo deploy/release
di modello, pubblica il `.npz` come release asset e configura:

```text
BRISCOLA_DEFAULT_MODEL_ID=best_a2c_v4.npz
BRISCOLA_MODEL_URL=<URL HTTPS del release asset>
BRISCOLA_MODEL_SHA256=bd3c7612c4a4314f5757472f9fcbffb7c5385cc3661d468a1fafc9341a198827
```

Il provisioning scarica solo il modello consigliato: su ambienti con disco limitato (es. 512 MB) evita di rendere disponibili troppi `.npz` contemporaneamente. Se vuoi mantenere anche `best_a2c_v3.npz` selezionabile online, caricalo nella stessa directory modelli solo se il budget disco lo consente.

### Report progressione modelli

Il report Excel curato vive in `docs/reports/model_progress.xlsx` ed è generato da:

```bash
uv run python scripts/build_model_report.py
```

Serve a tracciare solo i modelli **significativi**: best ufficiali, teacher/anchor importanti e candidati scartati
che spiegano una decisione. La prima tab contiene una dashboard con curva di progressione dei best; le altre tab
riportano milestone, dettagli modello, prove di promozione, decision quality, candidati scartati e fonti dati.

Aggiornalo quando promuovi un nuovo best o quando un esperimento importante cambia una decisione. Non usarlo come dump
di tutti i run: gli esperimenti locali restano in `benchmarks/experiments/` (gitignored), mentre il report conserva la
narrazione sintetica e verificabile.

Esempio di confronto testa‑a‑testa tra due modelli:

```bash
python scripts/evaluate_agents.py --benchmark medium --engine domain \
  --agent0 bc_model --agent0-model ./data/models/best_a2c_v4.npz \
  --agent1 bc_model --agent1-model ./data/models/best_a2c.npz
```

## Stato e roadmap

Il gioco è **online** su <https://briscolaai.fastapicloud.dev> (deploy su FastAPI Cloud: stato partita su Redis, realtime via pub/sub, event log Postgres opzionale). Stato corrente, invarianti da non rompere e prossime azioni: vedi **`PLAN.md`**.

## Licenza

Progetto rilasciato con licenza MIT – vedi `LICENSE`.
