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

import os

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .backend.server import app as api_app

# Crea l'applicazione FastAPI principale
app = FastAPI(title="Briscola AI", version="0.1.0")

# Ottiene la directory del file corrente
current_dir = os.path.dirname(os.path.abspath(__file__))

# Monta l'app API sotto /api
app.mount("/api", api_app)

# Monta i file statici
static_dir = os.path.join(current_dir, "frontend", "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Serve il file HTML principale
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve la pagina HTML principale (single-page UI)."""
    return FileResponse(os.path.join(static_dir, "index.html"))


# Serve la favicon
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve la favicon (esclusa dallo schema OpenAPI)."""
    return FileResponse(os.path.join(static_dir, "favicon.ico"))


def run_server(host="0.0.0.0", port=8000, reload=False):
    """Avvia il server con uvicorn"""
    # Se chiamato direttamente come entry point, fa il parsing degli argomenti da riga di comando
    import sys

    if len(sys.argv) > 1:
        import argparse

        parser = argparse.ArgumentParser(description="Avvia il server di Briscola AI")
        parser.add_argument("--host", default="0.0.0.0", help="Host su cui esporre il server")
        parser.add_argument("--port", type=int, default=8000, help="Porta su cui esporre il server")
        parser.add_argument("--reload", action="store_true", help="Abilita auto-reload per lo sviluppo")

        args = parser.parse_args()

        print(f"Avvio server Briscola AI su {args.host}:{args.port}")
        print("Premi Ctrl+C per fermare il server")

        host = args.host
        port = args.port
        reload = args.reload

    uvicorn.run("briscola_ai.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_server(reload=True)
