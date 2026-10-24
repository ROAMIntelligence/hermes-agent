"""Rate-limit requeue bookkeeping (#119070).

Split out of ``test_kanban_db.py`` to keep that facade under its code-health line cap.

The defect: a rate-limited (rc=75) requeue stamped ``tasks.last_failure_error``
with the quota note, and ``check_respawn_guard`` reads that column with the
quota/auth blocker regex. Once a later run existed and was neither ``crashed``
nor ``rate_limited`` (the ``review_requested`` handoff), the guard returned
``blocker_auth`` — and since only a completion clears the column, one quota wall
parked the card forever.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _exited_status(code: int) -> int:
    """Raw wait-status for a WIFEXITED child with the given exit code."""
    return code << 8


def _rate_limit_requeue(conn, tid, pid, claimer):
    """Drive one REAL rate-limited requeue (rc=75) through detect_crashed_workers."""
    kb.claim_task(conn, tid, claimer=claimer)
    conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (pid, tid))
    conn.commit()
    kbd._record_worker_exit(pid, _exited_status(kb.KANBAN_RATE_LIMIT_EXIT_CODE))
    crashed = kbd.detect_crashed_workers(conn)
    assert tid not in crashed
    assert tid in getattr(kbd.detect_crashed_workers, "_last_rate_limited", [])


def _latest_run(conn, tid):
    return conn.execute(
        "SELECT outcome, error, ended_at FROM task_runs WHERE task_id=? "
        "ORDER BY id DESC LIMIT 1",
        (tid,),
    ).fetchone()


def test_rate_limit_requeue_leaves_the_failure_column_empty(kanban_home, monkeypatch):
    """Six quota walls: no stamp, no failure counted, always ready, run row carries the why."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    with kbc.connect() as conn:
        host = kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="rl-empty", assignee="a")
        for i in range(6):
            _rate_limit_requeue(conn, tid, 71000 + i, f"{host}:w{i}")
            task = kb.get_task(conn, tid)
            assert task.last_failure_error is None, (
                f"hit {i}: a rate-limit requeue must not stamp last_failure_error, "
                f"got {task.last_failure_error!r}")
            assert task.consecutive_failures == 0
            assert task.status == "ready"
            run = _latest_run(conn, tid)
            assert run["outcome"] == "rate_limited"
            assert "rate-limited" in (run["error"] or "")


def test_rate_limit_requeue_leaves_prior_failure_text_untouched(kanban_home, monkeypatch):
    """A rate-limit requeue neither overwrites nor clears a genuine prior failure text."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    prior = "genuine prior failure: executor died mid-run"

    with kbc.connect() as conn:
        host = kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="rl-prior", assignee="a")
        conn.execute("UPDATE tasks SET last_failure_error=? WHERE id=?", (prior, tid))
        conn.commit()

        _rate_limit_requeue(conn, tid, 72000, f"{host}:w0")

        task = kb.get_task(conn, tid)
        assert task.last_failure_error == prior
        assert task.status == "ready"
        assert task.consecutive_failures == 0
        run = _latest_run(conn, tid)
        assert run["outcome"] == "rate_limited"
        assert "rate-limited" in (run["error"] or "")


def test_rate_limit_requeues_never_park_the_respawn_guard(kanban_home, monkeypatch):
    """Repeated quota walls space the respawn by the cooldown and never leave the
    guard a ``blocker_auth`` verdict (#119070): one respawn per window, never parked.
    """
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    real_time = kb.time.time

    with kbc.connect() as conn:
        host = kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="rl-guard", assignee="a")

        for i in range(2):
            monkeypatch.setattr(kb.time, "time", real_time)
            _rate_limit_requeue(conn, tid, 73000 + i, f"{host}:w{i}")

            task = kb.get_task(conn, tid)
            assert task.last_failure_error is None, f"requeue {i}"
            assert task.consecutive_failures == 0, f"requeue {i}"
            assert task.status == "ready", f"requeue {i}"

            ended = kb._to_epoch(_latest_run(conn, tid)["ended_at"])
            assert ended is not None

            monkeypatch.setattr(kb.time, "time", lambda: ended + 100)
            assert kbd.check_respawn_guard(conn, tid) == "rate_limit_cooldown"
            monkeypatch.setattr(kb.time, "time", lambda: ended + 301)
            assert kbd.check_respawn_guard(conn, tid) is None
            for delta in (1, 100, 299, 301, 1801, 3600):
                monkeypatch.setattr(kb.time, "time", lambda d=delta: ended + d)
                verdict = kbd.check_respawn_guard(conn, tid)
                assert verdict in {"rate_limit_cooldown", None}, (
                    f"requeue {i}, +{delta}s: got {verdict!r}")
