# Repository Guidelines

## Project Structure

- `src/briscola_ai/` – Python package (entrypoint in `main.py`)
  - `domain/` – regole core e modelli canonici (es. `models.py`, `state.py`, `engine.py`, `rules.py`)
  - `backend/` – FastAPI server + WebSocket endpoints (`server.py`)
  - `frontend/static/` – browser UI assets (`index.html`, `css/`, `js/`)
- `ai/` – AI implementations/strategies (currently minimal)
- `scripts/` – local utilities (e.g., `scripts/simulate_games.py`)

## Setup, Build, and Run

This repo uses `uv` for environments and installs.

- Create env: `uv venv`
- Install (editable): `uv pip install -e .`
- Install dev deps: `uv pip install -e ".[dev]"`
- Run server (installed script): `briscola-server --reload`
- Run headless simulations: `python scripts/simulate_games.py --num-games 100 --seed 42`
- Lint (ruff): `ruff check src tests scripts`
- Format (ruff): `ruff format src tests scripts`
- Typecheck (mypy): `mypy src`

The app serves the UI at `http://localhost:8000` by default (`--host`, `--port` supported).

## Coding Style & Naming

- Indentation: 4 spaces for Python; keep JS/CSS consistent with existing files.
- Naming: `snake_case` for Python functions/vars, `PascalCase` for Python classes; keep modules small and scoped to `domain/`, `backend/`, `frontend/`, or `ai/`.
- Documentation-first (didactic repo): add clear, detailed docstrings for all public modules/classes/functions/methods; include intent, inputs/outputs, and invariants where helpful.
- Commenting-first (didactic repo): keep code and tests well commented; explain non-obvious logic, edge cases, and what each test scenario is asserting.
- Agent communication: for each iteration, explain what changed and why in a didactic way (assumptions, reasoning, and how to verify locally).
- Planning hygiene: keep `PLAN.md` continuously updated so it’s always clear what’s done, what’s in progress, and what’s next.
- Planning hygiene (commit gate): before every commit, check `PLAN.md` and update it if needed so the plan stays aligned with the actual repo state.
- Quality gate (commit): before every commit, if the changes affect Python code/tests, run `ruff check src tests scripts`, `pytest` and `mypy src` (and fix issues) before committing.
- README hygiene (coverage badge): when the project coverage changes materially, update the Shields.io badge percentage in `README.md` accordingly.
- Doc hygiene: keep `AGENTS.md` itself updated as the working agreement evolves (update it whenever we introduce new tooling, conventions, or expectations).
- Tooling (dev deps):
  - Canonical: `ruff format src tests scripts` and `ruff check --fix src tests scripts` (includes import sorting via `I` rules), plus `mypy src`.
  - Note: `black` and `isort` are not used; formatting and import sorting are handled by `ruff`.

## Testing Guidelines

`pytest` is included in dev dependencies; tests live under `tests/`.

- Add new tests under `tests/` using `test_*.py` and `pytest` fixtures where helpful.
- Run tests: `pytest`

## Commits & Pull Requests

Git history isn’t available in this workspace snapshot, so there is no established commit style to follow. Prefer a simple conventional format:

- Commits (Italiano): `type(scope): summary` (es. `feat(domain): aggiungi helper punteggio`, `fix(ui): correggi animazione carte`)
  - **IMPORTANTE**: Le descrizioni dei commit devono essere SEMPRE in italiano per finalità didattiche.
- Versioning (SemVer, pre-1.0):
  - La versione canonica del pacchetto vive in `pyproject.toml` (`[project].version`).
  - Finché siamo in `0.x`, usiamo una regola semplice:
    - `0.1.x` (patch): fix/refactor/test/doc/performance senza cambiare contratti pubblici.
    - `0.(n+1).0` (minor): nuove feature “visibili” o cambi rilevanti a componenti/pipeline.
    - Cambi “breaking” a contratti pubblici (API/WS/DTO o schema dataset/export): bump di `minor` e nota esplicita nel commit/README.
  - L’agente deve segnalare quando un change merita bump e proporre la nuova versione, ma la decisione finale resta al maintainer.
- PRs: describe the change, include reproduction steps for bugs, and add screenshots for UI changes (`src/briscola_ai/frontend/static/`).
