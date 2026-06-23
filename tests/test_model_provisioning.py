"""
Test del provisioning modello allo startup (`ensure_model_available`).

Usiamo URL `file://` per evitare dipendenze di rete: la logica di download/verifica/scrittura
atomica è la stessa di un URL http(s).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from briscola_ai.ai.model_provisioning import ensure_model_available


def _make_source(tmp_path: Path, content: bytes = b"fake-model-bytes") -> tuple[Path, str]:
    src = tmp_path / "source.npz"
    src.write_bytes(content)
    return src, src.as_uri()


def test_returns_true_if_already_present(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "best_a2c_v3.npz").write_bytes(b"x")

    ok, msg = ensure_model_available(models_dir=models_dir, model_id="best_a2c_v3.npz", url=None)
    assert ok is True
    assert "già presente" in msg


def test_returns_false_if_missing_and_no_url(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    ok, msg = ensure_model_available(models_dir=models_dir, model_id="best_a2c_v3.npz", url=None)
    assert ok is False
    assert "nessun BRISCOLA_MODEL_URL" in msg


def test_downloads_when_missing(tmp_path: Path) -> None:
    src, url = _make_source(tmp_path)
    models_dir = tmp_path / "models"  # non esiste ancora: deve crearla

    ok, msg = ensure_model_available(models_dir=models_dir, model_id="best_a2c_v3.npz", url=url)

    assert ok is True
    target = models_dir / "best_a2c_v3.npz"
    assert target.exists()
    assert target.read_bytes() == src.read_bytes()


def test_sha256_match_installs(tmp_path: Path) -> None:
    content = b"model-with-hash"
    src, url = _make_source(tmp_path, content)
    digest = hashlib.sha256(content).hexdigest()
    models_dir = tmp_path / "models"

    ok, _ = ensure_model_available(models_dir=models_dir, model_id="m.npz", url=url, sha256=digest)
    assert ok is True
    assert (models_dir / "m.npz").exists()


def test_sha256_mismatch_does_not_install(tmp_path: Path) -> None:
    src, url = _make_source(tmp_path, b"good-bytes")
    models_dir = tmp_path / "models"

    ok, msg = ensure_model_available(models_dir=models_dir, model_id="m.npz", url=url, sha256="deadbeef")
    assert ok is False
    assert "sha256 non corrispondente" in msg
    assert not (models_dir / "m.npz").exists()  # niente file parziale/sbagliato


def test_download_failure_is_non_fatal(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    ok, msg = ensure_model_available(
        models_dir=models_dir,
        model_id="m.npz",
        url=(tmp_path / "does_not_exist.npz").as_uri(),
    )
    assert ok is False
    assert "download fallito" in msg
    assert not (models_dir / "m.npz").exists()
