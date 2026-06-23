"""
Entry point dell'applicazione web.

Questa app FastAPI:
- monta l'API sotto `/api` (vedi `briscola_ai.backend.server`)
- serve gli asset statici sotto `/static`
- serve la UI principale su `/`

Per avviare in locale:
  - `briscola-server --reload`
  - oppure `python -m briscola_ai.main --reload`
"""

import asyncio
import os
from contextlib import asynccontextmanager
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .backend import server as backend_server
from .backend.event_log import EventLog, EventLogConfig, parse_event_db_path
from .versioning import get_code_version


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown dell'app principale.

    Nota importante:
    l'API backend è montata come sub-app (`/api`). In alcuni setup i mounted sub-app
    non ricevono eventi lifespan. Per evitare che features come cleanup e event log
    restino disabilitate, le inizializziamo esplicitamente qui.
    """
    event_log_created_here = False
    raw_path = parse_event_db_path(os.getenv("BRISCOLA_EVENT_DB_PATH"))

    existing_event_log = getattr(backend_server.app.state, "event_log", None)
    event_log = existing_event_log

    # Se il path cambia tra due startup (tipico nei test), ricreiamo la connessione.
    # Se il path è disabilitato, chiudiamo e azzeriamo.
    if event_log is not None:
        if raw_path is None:
            try:
                event_log.close()
            except Exception:
                pass
            event_log = None
            backend_server.app.state.event_log = None
        elif event_log.path != raw_path:
            try:
                event_log.close()
            except Exception:
                pass
            event_log = None
            backend_server.app.state.event_log = None

    if event_log is None and raw_path is not None:
        try:
            event_log = EventLog(EventLogConfig(path=raw_path))
            backend_server.app.state.event_log = event_log
            event_log_created_here = True
        except Exception:
            print("Event log SQLite: inizializzazione fallita, feature disabilitata.")
            backend_server.app.state.event_log = None

    cleanup_task = asyncio.create_task(backend_server.cleanup_inactive_games())
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        if event_log is not None and event_log_created_here:
            event_log.close()
            backend_server.app.state.event_log = None


# Crea l'applicazione FastAPI principale
#
# Nota:
# usiamo `get_code_version()` per allineare la versione OpenAPI alla versione del pacchetto
# (o all'override via env `BRISCOLA_CODE_VERSION`).
app = FastAPI(title="Briscola AI", version=get_code_version(), lifespan=lifespan)

# Ottiene la directory del file corrente
current_dir = os.path.dirname(os.path.abspath(__file__))

# Monta l'app API sotto /api
app.mount("/api", backend_server.app)

# Monta i file statici
static_dir = os.path.join(current_dir, "frontend", "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


def _asset_version() -> str:
    """
    Versione usata per il cache busting degli asset statici.

    Di default segue la versione del pacchetto; `BRISCOLA_ASSET_VERSION` permette di forzare
    un valore diverso in deploy senza dover fare necessariamente un bump applicativo.
    """
    raw = os.getenv("BRISCOLA_ASSET_VERSION", get_code_version()).strip() or get_code_version()
    return quote(raw, safe="")


# Serve il file HTML principale
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve la pagina HTML principale (single-page UI)."""
    index_path = os.path.join(static_dir, "index.html")
    with open(index_path, encoding="utf-8") as f:
        html = f.read()
    html = html.replace("__BRISCOLA_ASSET_VERSION__", _asset_version())
    return HTMLResponse(content=html, headers={"Cache-Control": "no-cache"})


# Serve la favicon
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve la favicon (esclusa dallo schema OpenAPI)."""
    return FileResponse(os.path.join(static_dir, "favicon.ico"))


def run_server(host="0.0.0.0", port=8000, reload=False):
    """Avvia il server con uvicorn"""
    # Parsing argomenti CLI:
    # - usiamo i parametri della funzione come default (così `run_server(host=..., ...)` resta possibile)
    # - se l'utente passa argomenti, li rispettiamo.
    import argparse

    parser = argparse.ArgumentParser(description="Avvia il server di Briscola AI")
    parser.add_argument("--host", default=host, help="Host su cui esporre il server")
    parser.add_argument("--port", type=int, default=port, help="Porta su cui esporre il server")
    parser.add_argument("--reload", action="store_true", default=reload, help="Abilita auto-reload per lo sviluppo")
    parser.add_argument(
        "--event-db",
        default=os.getenv("BRISCOLA_EVENT_DB_PATH", "./data/briscola_events.sqlite3"),
        help=(
            "Percorso del DB SQLite per l'event log (Phase 4). "
            "Default: ./data/briscola_events.sqlite3. "
            "Usa stringa vuota per disabilitare (es. --event-db '')."
        ),
    )
    parser.add_argument(
        "--event-log-mode",
        default=os.getenv("BRISCOLA_EVENT_LOG_MODE", "debug"),
        choices=["debug", "dataset", "off"],
        help=(
            "Modalità event log: "
            "`debug` (completa), `dataset` (riduce dimensione DB per raccolta umani), `off` (disabilita logging). "
            "Default: debug."
        ),
    )

    args = parser.parse_args()

    print(f"Avvio server Briscola AI su {args.host}:{args.port}")
    print("Premi Ctrl+C per fermare il server")

    host = args.host
    port = args.port
    reload = args.reload

    # Configurazione event log:
    # - CLI è la fonte più esplicita
    # - l'app (main/backend) legge la variabile d'ambiente nel lifespan
    if args.event_db.strip() == "":
        os.environ.pop("BRISCOLA_EVENT_DB_PATH", None)
    else:
        os.environ["BRISCOLA_EVENT_DB_PATH"] = args.event_db
    os.environ["BRISCOLA_EVENT_LOG_MODE"] = args.event_log_mode

    uvicorn.run("briscola_ai.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_server(reload=True)
