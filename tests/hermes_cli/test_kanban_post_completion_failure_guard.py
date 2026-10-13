"""A late non-success signal must not contradict a card that already completed.

The agent finalizer records a terminal ``timed_out`` when a kanban worker runs out of
iteration budget (``_record_kanban_budget_exhausted``). That signal can arrive *after*
the worker already called ``kanban_complete`` — the completion is the real terminal
record, so the failure path must be a strict no-op on a ``done`` card: no trailing
``timed_out`` event, no ``consecutive_failures`` stamp, no ``retry_status: ready``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _events(conn, tid: str) -> list[tuple[str, str | None]]:
    return [
        (r["kind"], r["payload"])
        for r in conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id", (tid,)
        )
    ]


def _task_row(conn, tid: str):
    return conn.execute(
        "SELECT status, consecutive_failures, last_failure_error, current_run_id "
        "FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()


def test_budget_exhausted_after_completion_is_a_noop(kanban_home):
    """Worker completes, then the finalizer reports budget exhaustion: the ``done``
    card keeps its verdict and gains no trailing ``timed_out``."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        run_id = _task_row(conn, tid)["current_run_id"]
        assert run_id is not None
        assert kb.complete_task(conn, tid, result="done ok", expected_run_id=run_id) is True

        before = len(_events(conn, tid))
        tripped = kbd._record_task_failure(
            conn,
            tid,
            error="Iteration budget exhausted (90/90) — task could not complete within the allowed iterations",
            outcome="timed_out",
            release_claim=True,
            end_run=True,
        )
        assert tripped is False
        row = _task_row(conn, tid)
        assert row["status"] == "done"
        assert row["consecutive_failures"] == 0
        assert row["last_failure_error"] is None
        # No trailing failure event: the completion is the last word.
        assert len(_events(conn, tid)) == before
        assert "timed_out" not in [k for k, _ in _events(conn, tid)]


def test_spawn_failure_on_running_card_still_records(kanban_home):
    """The guard is scoped to terminal cards: a genuine failure on a ``running``
    card still closes its run and emits the outcome event."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        assert kb.claim_task(conn, tid) is not None

        tripped = kbd._record_task_failure(
            conn,
            tid,
            error="workspace: boom",
            outcome="spawn_failed",
            release_claim=True,
            end_run=True,
        )
        assert tripped is False
        row = _task_row(conn, tid)
        assert row["status"] == "ready"
        assert row["consecutive_failures"] == 1
        kinds = [k for k, _ in _events(conn, tid)]
        assert "spawn_failed" in kinds
