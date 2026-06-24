"""
Test dei backend event log e della factory (SQLite vs Postgres).

Postgres non è disponibile nei test: verifichiamo la **logica** di `PostgresEventLog` iniettando
una connessione fake (registra l'SQL eseguito e simula `rowcount` per l'idempotenza), più la
selezione della factory e la conformità al `EventLogProtocol`. La parità di comportamento reale è
demandata al deploy (Neon); qui copriamo dialetto SQL e logica Python.
"""

from __future__ import annotations

import pytest

from briscola_ai.backend.event_log import (
    EventLog,
    EventLogConfig,
    EventLogProtocol,
    PostgresEventLog,
    build_event_log,
    resolve_database_url,
)


class _FakeCursor:
    def __init__(self, conn: "_FakeConn") -> None:
        self._conn = conn
        self.rowcount = conn.rowcount

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, sql: str, params: tuple = ()) -> None:
        # Normalizziamo lo spazio per asserzioni robuste.
        self._conn.executed.append((" ".join(sql.split()), tuple(params)))
        self.rowcount = self._conn.rowcount


class _FakeConn:
    """Connessione psycopg fittizia: registra le execute e restituisce un `rowcount` configurabile."""

    def __init__(self, rowcount: int = 1) -> None:
        self.executed: list[tuple[str, tuple]] = []
        self.rowcount = rowcount
        self.closed = False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def close(self) -> None:
        self.closed = True


def _sqls(conn: _FakeConn) -> str:
    return "\n".join(sql for sql, _ in conn.executed)


def test_postgres_event_log_creates_schema_on_init() -> None:
    conn = _FakeConn()
    PostgresEventLog(conn=conn)
    sqls = _sqls(conn)
    assert "CREATE TABLE IF NOT EXISTS games" in sqls
    assert "CREATE TABLE IF NOT EXISTS events" in sqls
    assert "BIGSERIAL" in sqls  # dialetto Postgres


def test_postgres_event_log_insert_uses_on_conflict_and_placeholders() -> None:
    conn = _FakeConn()
    log = PostgresEventLog(conn=conn)
    log.ensure_game("g1", num_players=2, seed=7, code_version="0.6.0", rules_version="1")
    log.log_event("g1", "action_play_card", {"k": "v"}, server_version=3, player_index=0)

    games_inserts = [(s, p) for s, p in conn.executed if s.startswith("INSERT INTO games")]
    events_inserts = [(s, p) for s, p in conn.executed if s.startswith("INSERT INTO events")]
    assert games_inserts and "ON CONFLICT (game_id) DO NOTHING" in games_inserts[0][0]
    assert "%s" in games_inserts[0][0] and "?" not in games_inserts[0][0]  # placeholder Postgres
    assert events_inserts and events_inserts[0][1][0] == "g1"


def test_postgres_try_mark_finished_idempotent_via_rowcount() -> None:
    conn_ok = _FakeConn(rowcount=1)
    assert PostgresEventLog(conn=conn_ok).try_mark_game_finished("g1") is True

    conn_noop = _FakeConn(rowcount=0)
    assert PostgresEventLog(conn=conn_noop).try_mark_game_finished("g1") is False


def test_postgres_try_mark_aborted_atomic_guard() -> None:
    conn = _FakeConn(rowcount=1)
    log = PostgresEventLog(conn=conn)
    assert log.try_mark_game_aborted("g1", aborted_reason="inactive_timeout") is True
    update = [s for s, _ in conn.executed if s.startswith("UPDATE games SET aborted_at")][0]
    # La guardia idempotente deve stare nel WHERE (UPDATE atomico, niente SELECT-then-UPDATE).
    assert "finished_at IS NULL" in update and "aborted_at IS NULL" in update


def test_postgres_close_is_safe() -> None:
    conn = _FakeConn()
    PostgresEventLog(conn=conn).close()
    assert conn.closed is True


def test_both_backends_satisfy_protocol() -> None:
    assert isinstance(PostgresEventLog(conn=_FakeConn()), EventLogProtocol)
    assert isinstance(EventLog(EventLogConfig(path=":memory:")), EventLogProtocol)


def test_build_event_log_selection() -> None:
    # Solo SQLite (path) → EventLog.
    log = build_event_log(sqlite_path=":memory:", database_url=None)
    assert isinstance(log, EventLog)
    # Niente path né url → disabilitato.
    assert build_event_log(sqlite_path=None, database_url=None) is None


def test_resolve_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BRISCOLA_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert resolve_database_url() is None

    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@host/db")
    assert resolve_database_url() == "postgresql://u:p@host/db"

    # Override esplicito ha priorità.
    monkeypatch.setenv("BRISCOLA_DATABASE_URL", "postgresql://override/db")
    assert resolve_database_url() == "postgresql://override/db"
