"""
Provisioning del modello al deploy: scarica il modello consigliato se assente.

In cloud i file `.npz` non sono nel repo (`data/` è gitignored) e committare un binario da decine
di MB nella git history è poco desiderabile. Per servire comunque il campione, all'avvio scarichiamo
il modello da un URL (es. GitHub Release asset) impostato via env, **solo se non è già presente**.

Principio anti-fragilità: il provisioning non deve MAI impedire l'avvio dell'app. Se l'URL non è
impostato o il download fallisce, l'app parte lo stesso (il default agente è `random` e `/version`
segnala l'assenza del modello).
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

# Id del modello consigliato (coerente con la baseline ufficiale e con `/version`).
DEFAULT_MODEL_ID = "best_a2c_v3.npz"


def ensure_model_available(
    *,
    models_dir: Path,
    model_id: str,
    url: str | None,
    sha256: str | None = None,
) -> tuple[bool, str]:
    """
    Garantisce che `models_dir/model_id` esista, scaricandolo da `url` se assente.

    Ritorna `(disponibile, messaggio)`. Non solleva mai: ogni errore è catturato e riportato nel
    messaggio, così l'avvio dell'app non viene interrotto da un problema di provisioning.

    Se `sha256` è impostato, il file scaricato viene verificato e installato solo se l'hash coincide.
    La scrittura è atomica (file temporaneo nella stessa directory + `os.replace`).
    """
    target = models_dir / model_id
    if target.exists():
        return True, f"modello già presente: {target}"
    if not url:
        return False, "modello assente e nessun BRISCOLA_MODEL_URL impostato"

    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as resp:
            data = resp.read()
    except Exception as exc:  # provisioning best-effort: non deve crashare l'avvio
        return False, f"download fallito: {exc!r}"

    if sha256:
        digest = hashlib.sha256(data).hexdigest()
        if digest.lower() != sha256.strip().lower():
            return False, f"sha256 non corrispondente (atteso {sha256.strip()}, ottenuto {digest})"

    try:
        fd, tmp_name = tempfile.mkstemp(dir=str(models_dir), suffix=".part")
        try:
            with os.fdopen(fd, "wb") as out:
                out.write(data)
            os.replace(tmp_name, target)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
    except Exception as exc:
        return False, f"scrittura modello fallita: {exc!r}"

    return True, f"modello scaricato in {target} ({len(data)} byte)"
