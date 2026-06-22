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

2. **Fast path** (`src/briscola_ai/ai/fast_*.py`) — reimplementazione 2-player ad alto throughput su interi/array NumPy, con kernel `@njit` Numba (`fast_numba.py`). Serve per self-play e training massivi. Copre solo gli agenti tradotti nel path fast (`random`, `greedy_points`, `heuristic_v1`, `heuristic_v2`). **Deve restare coerente col dominio**: i test (`tests/test_fast_*`) verificano la parità. Se cambi una regola nel dominio, aggiorna anche il fast path e i suoi test.

**Anti-cheat (invariante centrale):** gli agenti non ricevono mai `GameState` completo, solo `PlayerObservation`. Vale per le baseline (`ai/agents.py`), i modelli (`ai/bc_model_agent.py`) e il reward shaping (`ai/training/reward_shaping.py` usa solo informazione pubblica + mano del giocatore). Non introdurre scorciatoie che leggano carte nascoste o l'ordine del mazzo.

**Backend** (`src/briscola_ai/backend/`): FastAPI montato sotto `/api` da `main.py`; stato partite **in memoria** (`active_games`, non persistente). `dto.py` = contratto Pydantic v2; `server.py` = REST + WebSocket (architettura ibrida: REST per le azioni, WS per push degli aggiornamenti). Il backend **avanza automaticamente** quando è il turno dell'IA e **non** introduce delay di presentazione (niente `asyncio.sleep` per animazioni — quello è compito del frontend). `event_log.py` scrive un event log SQLite append-only (`data/briscola_events.sqlite3`) usato come base dataset.

**Modelli locali (`.npz`)**: la UI seleziona un avversario `bc_model` da un catalogo server-side (`ai/model_catalog.py`). Il browser invia solo un `ai_model_id` (path relativo) tra quelli di `GET /api/ai/models`; il backend rifiuta path traversal e carica solo da `BRISCOLA_MODELS_DIR` (default `./data/models/`). I trainer salvano nei `.npz` i metadati (`label`, `description_it`, `feature_dim`).

**Pipeline ML** (vedi `README.md` e `PLAN.md` per il dettaglio): dominio testabile → backend/UI → event log SQLite → export JSONL versionato (`export_dataset.py`) → self-play → valutazione offline riproducibile → training (BC/PG/A2C). `PLAN.md` è la **fonte di verità su cosa fare dopo** ed è enorme: leggine la coda per lo stato corrente.

## Setup, Build, and Run

Richiede **Python 3.14** e [`uv`](https://github.com/astral-sh/uv).

- Crea env: `uv venv -p python3.14`
- Installa (editable): `uv pip install -e .`
- Dev deps: `uv pip install -e ".[dev]"`
- Avvia server: `briscola-server --reload` (UI su `http://localhost:8000`; `--host`/`--port` supportati)
- Simulazioni headless: `python scripts/simulate_games.py --num-games 100 --seed 42`

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

Aggiorna `PLAN.md` se necessario (deve riflettere lo stato reale del repo). Aggiorna il badge coverage Shields.io in `README.md` quando la copertura cambia in modo materiale. Tieni aggiornato questo file quando cambiano tooling/convenzioni.

## Coding Style & Naming

- Indentazione: 4 spazi per Python; JS/CSS coerente coi file esistenti. Line length 120 (ruff, `py314`).
- Naming: `snake_case` per funzioni/variabili, `PascalCase` per le classi; moduli piccoli e dentro il loro layer (`domain/`, `backend/`, `frontend/`, `ai/`).
- **Documentation-first**: docstring chiare e dettagliate per moduli/classi/funzioni pubbliche (intento, input/output, invarianti).
- **Commenting-first**: spiega logica non ovvia, edge case, e cosa asserisce ogni scenario di test.
- **Agent communication**: per ogni iterazione spiega cosa è cambiato e perché in modo didattico (assunzioni, ragionamento, come verificare in locale).

## Testing

`pytest` (dev dep); test in `tests/` come `test_*.py`. Coprono invarianti di dominio, casi limite endgame, parità fast/numba vs dominio, encoder osservazioni, reward shaping, e integrazione API/WS.

## Commits & Pull Requests

- Commit (formato): `type(scope): summary` — es. `feat(domain): aggiungi helper punteggio`, `fix(ui): correggi animazione carte`.
  - **IMPORTANTE**: le descrizioni dei commit devono essere SEMPRE in italiano (finalità didattiche).
- Versioning (SemVer, pre-1.0): versione canonica in `pyproject.toml` (`[project].version`).
  - `0.1.x` (patch): fix/refactor/test/doc/performance senza cambiare contratti pubblici.
  - `0.(n+1).0` (minor): nuove feature visibili o cambi rilevanti a componenti/pipeline; anche per cambi **breaking** ai contratti pubblici (API/WS/DTO o schema dataset/export), con nota esplicita.
  - L'agente segnala quando un change merita un bump e propone la versione; la decisione finale resta al maintainer.
- PR: descrivi il change, includi passi di riproduzione per i bug, e screenshot per cambi UI (`src/briscola_ai/frontend/static/`).
