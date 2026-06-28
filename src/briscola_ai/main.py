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
from contextlib import asynccontextmanager, suppress
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .backend import server as backend_server
from .backend.event_log import build_event_log, parse_event_db_path, resolve_database_url
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
    # Backend event log: Postgres se `DATABASE_URL` è impostata (cloud multi-replica), altrimenti
    # SQLite locale se è dato un path, altrimenti disabilitato. `desired` è l'identità del backend
    # voluto (per decidere se ricreare la connessione tra due startup, tipico nei test).
    database_url = resolve_database_url()
    sqlite_path = parse_event_db_path(os.getenv("BRISCOLA_EVENT_DB_PATH"))
    desired = database_url or sqlite_path

    existing_event_log = getattr(backend_server.app.state, "event_log", None)
    event_log = existing_event_log

    # Config cambiata o disabilitata: chiudi e azzera.
    if event_log is not None and (desired is None or event_log.path != desired):
        with suppress(Exception):
            event_log.close()
        event_log = None
        backend_server.app.state.event_log = None

    if event_log is None and desired is not None:
        try:
            event_log = build_event_log(sqlite_path=sqlite_path, database_url=database_url)
            backend_server.app.state.event_log = event_log
            event_log_created_here = event_log is not None
        except Exception as exc:
            print(f"Event log: inizializzazione fallita, feature disabilitata ({exc!r}).")
            backend_server.app.state.event_log = None

    # Provisioning modello (best-effort): se manca e `BRISCOLA_MODEL_URL` è impostata, scarica il
    # campione consigliato nella directory modelli. Non blocca l'avvio in caso di errore.
    try:
        from .ai.models import DEFAULT_MODEL_ID, ensure_model_available, get_models_dir_from_env

        _, provisioning_msg = ensure_model_available(
            models_dir=get_models_dir_from_env(),
            model_id=os.getenv("BRISCOLA_DEFAULT_MODEL_ID", DEFAULT_MODEL_ID),
            url=os.getenv("BRISCOLA_MODEL_URL"),
            sha256=os.getenv("BRISCOLA_MODEL_SHA256"),
        )
        print(f"Model provisioning: {provisioning_msg}")
    except Exception as exc:  # difesa extra: il provisioning non deve impedire l'avvio
        print(f"Model provisioning: errore inatteso, ignorato ({exc!r}).")

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
    # Override esplicito (utile in deploy se si vuole forzare un valore preciso).
    override = os.getenv("BRISCOLA_ASSET_VERSION", "").strip()
    if override:
        return quote(override, safe="")

    # Altrimenti deriviamo la versione dal CONTENUTO statico: il `mtime` massimo tra i file CSS/JS.
    # Così ogni modifica a CSS/JS invalida automaticamente la cache del browser, sia in locale
    # (senza dover reinstallare il pacchetto o bumpare la versione) sia tra un deploy e l'altro.
    # Manteniamo `get_code_version()` come prefisso leggibile.
    try:
        latest_mtime_ns = 0
        for root, _dirs, files in os.walk(static_dir):
            for filename in files:
                if filename.endswith((".css", ".js")):
                    mtime_ns = os.stat(os.path.join(root, filename)).st_mtime_ns
                    latest_mtime_ns = max(latest_mtime_ns, mtime_ns)
        if latest_mtime_ns:
            return quote(f"{get_code_version()}-{latest_mtime_ns:x}", safe="-")
    except OSError:
        pass
    return quote(get_code_version(), safe="")


def _realtime_mode() -> str:
    """
    Modalità realtime suggerita al frontend. Default: **WebSocket**.

    Il WebSocket funziona anche in cloud multi-replica perché il fan-out degli eventi passa per
    Redis pub/sub (vedi `backend/game_store.py`): un client su una qualsiasi replica riceve gli
    eventi della partita. Override via `?polling=1` / `?ws=1` nell'URL (il polling resta un
    fallback di debug); forzabile con `BRISCOLA_REALTIME_MODE`.
    """
    forced = os.getenv("BRISCOLA_REALTIME_MODE", "").strip().lower()
    if forced in {"polling", "ws"}:
        return forced
    return "ws"


# Serve il file HTML principale
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve la pagina HTML principale (single-page UI)."""
    index_path = os.path.join(static_dir, "index.html")
    with open(index_path, encoding="utf-8") as f:
        html = f.read()
    html = html.replace("__BRISCOLA_ASSET_VERSION__", _asset_version())
    html = html.replace("__BRISCOLA_REALTIME_MODE_VALUE__", _realtime_mode())
    # Versione "software" (SemVer) mostrata nel footer — distinta dall'asset version (cache-busting).
    html = html.replace("__BRISCOLA_APP_VERSION__", get_code_version())
    return HTMLResponse(content=html, headers={"Cache-Control": "no-cache"})


# Serve la favicon
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve la favicon (esclusa dallo schema OpenAPI)."""
    return FileResponse(os.path.join(static_dir, "favicon.ico"))


@app.get("/health", include_in_schema=False)
async def health():
    """Liveness check minimale (per piattaforme cloud / load balancer)."""
    return {"status": "ok"}


@app.get("/version")
async def version_info():
    """
    Diagnostica deploy: versioni e presenza del modello consigliato.

    Utile in cloud per verificare che il modello consigliato sia risolvibile nella directory modelli
    effettiva (che dipende da `BRISCOLA_MODELS_DIR` o dalla working directory).
    """
    from .ai.models import DEFAULT_MODEL_ID, get_models_dir_from_env
    from .versioning import get_rules_version

    models_dir = get_models_dir_from_env()
    # Coerente col provisioning: stesso `BRISCOLA_DEFAULT_MODEL_ID` usato allo startup.
    recommended_model = os.getenv("BRISCOLA_DEFAULT_MODEL_ID", DEFAULT_MODEL_ID)
    return {
        "code_version": get_code_version(),
        "rules_version": get_rules_version(),
        "models_dir": str(models_dir),
        "recommended_model": recommended_model,
        "recommended_model_present": (models_dir / recommended_model).exists(),
    }


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
