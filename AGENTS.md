# Repository Guidelines

> Questo file è il documento di riferimento condiviso per gli agenti (Claude Code, Codex, …).
> Progetto **didattico**: la chiarezza (docstring, commenti, spiegazioni del "perché") conta quanto la correttezza.

## Big picture (architettura)

Briscola end-to-end: motore di regole puro → backend HTTP/WS → UI → pipeline dati → training/valutazione di IA. Il punto chiave da capire prima di toccare il codice è che **esistono due implementazioni parallele dello stesso gioco**, tenute volutamente in sincronia:

1. **Dominio canonico** (`src/briscola_ai/domain/`) — fonte di verità, puro e testabile.
   - `models.py`: `Card`, `Suit`, `Rank` (carte canoniche).
   - `state.py`: `GameState`/`PlayerState` immutabili (`@dataclass(frozen=True, slots=True)`), serializzabili → abilitano replay deterministico.
   - `engine.py`: transizione pura `step(state, PlayCardAction) -> StepResult`, deterministica dato seme/stato. Supporta **2 e 4 giocatori** (4p a squadre).
   - `rules.py`: regole isolate (es. `who_wins_trick`).
   - `observation.py` + `card_id.py`: `PlayerObservation` (vista parziale lecita) e la mappatura carta ↔ id canonico `[0,39]`.

2. **Fast path** (`src/briscola_ai/ai/fast/` in puro Python su interi/array NumPy, con kernel `@njit` Numba in `src/briscola_ai/ai/numba/`) — reimplementazione 2-player ad alto throughput. Serve per self-play e training massivi. Copre solo gli agenti tradotti nel path fast (`random`, `greedy_points`, `heuristic_v1`, `heuristic_v2`). **Deve restare coerente col dominio**: i test (`tests/test_fast_*`) verificano la parità. Se cambi una regola nel dominio, aggiorna anche il fast path e i suoi test.

**Anti-cheat (invariante centrale):** gli agenti non ricevono mai `GameState` completo, solo `PlayerObservation`. Vale per le baseline (`ai/agents/`), i modelli (`ai/models/bc_model.py`) e il reward shaping (`ai/training/reward_shaping.py` usa solo informazione pubblica + mano del giocatore). Non introdurre scorciatoie che leggano carte nascoste o l'ordine del mazzo.

**Backend** (`src/briscola_ai/backend/`): FastAPI montato sotto `/api` da `main.py` (che serve anche la UI e gli statici ed espone `/health` e `/version`). `dto.py` = contratto Pydantic v2; `server.py` = REST + WebSocket (architettura ibrida: REST per le azioni, WS per gli aggiornamenti). Il backend **avanza automaticamente** quando è il turno dell'IA e **non** introduce delay di presentazione (quello è compito del frontend). Punti chiave del runtime (pensati per il deploy cloud multi-replica):

- **Stato partita**: vive in un `GameSessionStore` (`game_store.py`) — `InMemoryGameSessionStore` in locale, `RedisGameSessionStore` se è impostata `REDIS_URL`. Salva `GameState` (serializzato via `domain/serialization.py`) + la config IA per posto; l'oggetto `Agent` **non** è serializzato (si ricostruisce con `build_agent`, con cache). Lock per partita (asyncio in-memory / lock Redis distribuito) per serializzare le azioni concorrenti. Evita di reintrodurre dict globali di stato (`active_games` & co. sono stati rimossi).
- **Realtime**: il fan-out degli eventi WebSocket passa per il **pub/sub** dello store (Redis in prod, fan-out asyncio in dev), così `ai_card_reveal`/`trick_result`/snapshot raggiungono il client su QUALSIASI replica. Gli snapshot per-giocatore sono ricostruiti dal subscriber (anti-cheat). `?polling=1` resta come fallback di debug.
- **Event log** (`event_log.py`): append-only per dataset. `EventLog` su SQLite (default locale) **oppure** `PostgresEventLog` se è impostata `DATABASE_URL` (deploy persistente/condiviso); il backend è scelto da `build_event_log`. Stesso schema `games`/`events`. Attivo solo se configurato (`BRISCOLA_EVENT_DB_PATH` o `DATABASE_URL`); con modalità `dataset` richiede il consenso utente.
- **Provisioning modello** (`ai/models/provisioning.py`): allo startup, se manca, scarica il modello consigliato da `BRISCOLA_MODEL_URL` (verifica `BRISCOLA_MODEL_SHA256`). Se configurati, scarica anche asset ausiliari come il value model da `BRISCOLA_VALUE_MODEL_URL` (verifica `BRISCOLA_VALUE_MODEL_SHA256`). Best-effort: non blocca l'avvio.

**Modelli locali (`.npz`)**: la UI seleziona un avversario `bc_model` da un catalogo server-side (`ai/models/catalog.py`). Il browser invia solo un `ai_model_id` (path relativo) tra quelli di `GET /api/ai/models`; il backend rifiuta path traversal e carica solo da `BRISCOLA_MODELS_DIR` (default `./data/models/`). I trainer salvano nei `.npz` i metadati (`label`, `description_it`, `feature_dim`).

> **Nota peso `.npz`:** i pesi della MLP sono piccoli (~0.18 MB per v6). Non salvare l'intera storia `metrics`
> nel `metadata_json` dei modelli da pubblicare come release asset o usare in cloud: `np.savez` non comprime e una
> stringa JSON NumPy (`<U...`) può occupare circa 4 byte per carattere. I vecchi `best_a2c_v3/4/5.npz` pesano
> 41–52 MB per questo motivo, mentre `best_a2c_v6.npz` pesa ~0.2 MB con metadata sintetici. Per run lunghi usare
> metadata essenziali o `--metrics-mode summary`. La dimensione del file non indica capacità del modello né quantità
> di training.

**Pipeline ML** (vedi `README.md` e `PLAN.md` per il dettaglio): dominio testabile → backend/UI → event log (SQLite in locale, **Postgres** in cloud) → export JSONL versionato (`export_dataset.py`, oggi legge da SQLite) → self-play → valutazione offline riproducibile → training (BC/PG/A2C). `PLAN.md` è la **fonte di verità su cosa fare dopo**: è volutamente breve (stato corrente + prossime azioni), leggilo per intero prima di pianificare.

## Setup, Build, and Run

Richiede **Python 3.14** e [`uv`](https://github.com/astral-sh/uv).

- Crea env: `uv venv -p python3.14`
- Installa (editable): `uv pip install -e .`
- Dev deps: `uv pip install -e ".[dev]"`
- Avvia server: `briscola-server --reload` (UI su `http://localhost:8000`; `--host`/`--port` supportati)
- Simulazioni headless: `python scripts/simulate_games.py --num-games 100 --seed 42`

Deps di runtime per il cloud: `redis` (game store) e `psycopg` (event log Postgres) sono già in `dependencies`, ma importate **lazy** — usate solo se `REDIS_URL`/`DATABASE_URL` sono impostate (in locale tutto gira in-memory + SQLite). Dev-only: `fakeredis` (test dello store Redis) e `playwright` (ispezione UI/layout; richiede `python -m playwright install chromium`).

### Runtime & deploy (variabili d'ambiente)

Tutte opzionali; in locale i default vanno bene. In cloud (FastAPI Cloud, multi-replica) servono Redis + i provisioning. Il sito è live su `https://briscolaai.fastapicloud.dev`; l'entrypoint per `fastapi run` è lo shim `main:app` nella root.

- `REDIS_URL` (o `BRISCOLA_REDIS_URL`): attiva `RedisGameSessionStore` + pub/sub realtime. Se assente → in-memory + WebSocket diretto.
- `DATABASE_URL` (o `BRISCOLA_DATABASE_URL`): attiva l'event log su Postgres. Se assente → SQLite (`BRISCOLA_EVENT_DB_PATH`) o disabilitato.
- `BRISCOLA_EVENT_LOG_MODE`: `debug` (default) | `dataset` (minimale, richiede consenso) | `off`.
- `BRISCOLA_MODEL_URL` + `BRISCOLA_MODEL_SHA256` (+ `BRISCOLA_DEFAULT_MODEL_ID`): provisioning del modello consigliato allo startup.
- `BRISCOLA_VALUE_MODEL_URL` + `BRISCOLA_VALUE_MODEL_SHA256`: provisioning del value model richiesto da `bc_model_value_lookahead_8x8`.
- `BRISCOLA_MODELS_DIR` (default `./data/models/`); `BRISCOLA_CORS_ALLOW_ORIGINS` (default `*`, restringere in prod).
- `BRISCOLA_REALTIME_MODE` (`ws`|`polling`, override; default `ws`); `BRISCOLA_ASSET_VERSION` (override del cache-busting, che di default deriva da versione + mtime degli static).
- `BRISCOLA_DEBUG_STATE_ENDPOINT`: abilita la vista full-state di `GET /api/games/{id}` senza `player_index` (mani di tutti + `next_deck_card`, per debug/spectator). Default **disabilitata** (403) per l'anti-cheat: non attivarla in produzione pubblica.

### Pipeline AI (script in `scripts/`)

- Self-play → DB: `python scripts/self_play_to_db.py` (verso SQLite, no HTTP)
- Self-play veloce (summary-only, no DB): `python scripts/fast_self_play.py`
- Export dataset: `python scripts/export_dataset.py` (SQLite → JSONL versionato)
- Training BC (imitation): `python scripts/train_bc.py` (legge JSONL, target = action_id `[0,39]` + action mask)
- Training RL: `python scripts/train_a2c.py` / `python scripts/train_pg.py` (salvano `.npz` in `data/models/`)
- Pipeline riproducibile (train + eval matrix + manifest): `python scripts/run_experiment.py`
- Valutazione offline: `python scripts/evaluate_agents.py --num-games 1000 --seed 42 --agent0 heuristic_v1 --agent1 random`
  - engine selezionabile: `--engine domain|fast|numba`; modi seat-fair `--seat-fair --seed-suite small|medium`; benchmark `--benchmark medium|big`; carica modello con `--agent0 bc_model --agent0-model ./data/models/best_a2c.npz`
- Matrix / decision quality: `scripts/evaluate_matrix.py`, `scripts/evaluate_decision_quality.py`
- Benchmark throughput: `python scripts/benchmark_perf.py`

Artefatti in `data/` e `benchmarks/experiments/` sono locali (gitignored).

## Quality gate (prima di ogni commit)

Se le modifiche toccano Python (codice/test), esegui e correggi:

- `ruff format src tests scripts` e `ruff check --fix src tests scripts` (import sorting incluso via regole `I`; **niente** `black`/`isort`)
- `mypy src`
- `pytest`
  - singolo file: `pytest tests/test_trick_rules.py`
  - singolo test: `pytest tests/test_trick_rules.py::test_name` (o `pytest -k "pattern"`)
  - coverage: `pytest --cov=briscola_ai`

La CI GitHub Actions (`.github/workflows/ci.yml`) replica lo stesso gate (ruff format/check, mypy, pytest+coverage) su ogni push/PR: deve restare allineata a questa sezione.

Aggiorna `PLAN.md` se necessario (deve riflettere lo stato reale del repo). Aggiorna il badge coverage Shields.io in `README.md` quando la copertura cambia in modo materiale. Tieni aggiornato questo file quando cambiano tooling/convenzioni.

## Coding Style & Naming

- Indentazione: 4 spazi per Python; JS/CSS coerente coi file esistenti. Line length 120 (ruff, `py314`).
- Naming: `snake_case` per funzioni/variabili, `PascalCase` per le classi; moduli piccoli e dentro il loro layer (`domain/`, `backend/`, `frontend/`, `ai/`).
- **Documentation-first**: docstring chiare e dettagliate per moduli/classi/funzioni pubbliche (intento, input/output, invarianti).
- **Commenting-first**: spiega logica non ovvia, edge case, e cosa asserisce ogni scenario di test.
- **Agent communication**: per ogni iterazione spiega cosa è cambiato e perché in modo didattico (assunzioni, ragionamento, come verificare in locale).

## Testing

`pytest` (dev dep); test in `tests/` come `test_*.py`. Coprono invarianti di dominio, casi limite endgame, parità fast/numba vs dominio, encoder osservazioni, reward shaping, integrazione API/WS, game store (in-memory/Redis via `fakeredis`, incl. pub/sub) e backend event log (SQLite + Postgres con connessione fake). Un fixture autouse in `tests/conftest.py` azzera `REDIS_URL`/`DATABASE_URL` per ogni test, così la suite resta **ermetica** (non contatta servizi reali anche se quelle env sono presenti nell'ambiente).

## Commits & Pull Requests

- Commit (formato): `type(scope): summary` — es. `feat(domain): aggiungi helper punteggio`, `fix(ui): correggi animazione carte`.
  - **IMPORTANTE**: le descrizioni dei commit devono essere SEMPRE in italiano (finalità didattiche).
- Versioning (SemVer, pre-1.0): versione canonica in `pyproject.toml` (`[project].version`).
  - `0.1.x` (patch): fix/refactor/test/doc/performance senza cambiare contratti pubblici.
  - `0.(n+1).0` (minor): nuove feature visibili o cambi rilevanti a componenti/pipeline; anche per cambi **breaking** ai contratti pubblici (API/WS/DTO o schema dataset/export), con nota esplicita.
  - L'agente segnala quando un change merita un bump e propone la versione; la decisione finale resta al maintainer.
  - **Tag git per ogni bump**: dopo aver bumpato `pyproject.toml`, crea il tag annotato corrispondente e pushalo, così la serie `vX.Y.Z` su GitHub resta completa e allineata alla history:
    ```bash
    git tag -a vX.Y.Z -m "Versione X.Y.Z"
    git push <remote> vX.Y.Z
    ```
    Nota: `get_code_version()` (footer UI, `/version`, cache-busting) legge la versione da `pyproject.toml` anche in editable, quindi non serve reinstallare per vederla aggiornata.
  - **Report modelli per ogni release**: a ogni nuova release rigenera `docs/reports/model_progress.xlsx` con
    `uv run python scripts/build_model_report.py`. Controlla esplicitamente il foglio Dashboard e in particolare il
    grafico di progressione: deve includere il nuovo best/versione promossa e il range del grafico deve arrivare
    all'ultima riga dei modelli ufficiali.
- PR: descrivi il change, includi passi di riproduzione per i bug, e screenshot per cambi UI (`src/briscola_ai/frontend/static/`).
