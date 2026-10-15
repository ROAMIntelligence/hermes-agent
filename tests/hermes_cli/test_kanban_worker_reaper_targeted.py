"""The kanban zombie reaper may reap ONLY the workers this process spawned.

``reap_worker_zombies`` used ``os.waitpid(-1, os.WNOHANG)`` on POSIX: that call
waits for *any* child of the calling process, so the sweep consumed the exit
status of children it never spawned. The owner of such a child — cron's script
runner (``cron/scheduler_script.py``, ``if proc.returncode != 0``), a delegated
subprocess — then read ``returncode == 0`` for a child that FAILED: a script
that exited 3 is booked a clean success, and the dispatcher files the stolen
status under a worker PID it does not own.

The reaper's reach is the ``Popen`` handles ``_default_spawn`` parks in
``_live_worker_procs``; these tests pin both halves of that contract — the
worker is still reaped and classified, and the unrelated child is untouched.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from tests.waiting import wait_until


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fail_after(delay: float, code: int) -> subprocess.Popen:
    """A real child that exits ``code`` after ``delay`` seconds."""
    return subprocess.Popen(  # fixed argv: this interpreter, one -c expression
        [sys.executable, "-c", f"import time; time.sleep({delay}); raise SystemExit({code})"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _is_zombie(pid: int) -> bool:
    """True once ``pid`` has exited and nobody has reaped it yet.

    The steal window IS the zombie window: a child is only reapable by
    ``waitpid`` while it is a zombie, so the tests have to synchronise on that
    state instead of sleeping and hoping.
    """
    if sys.platform == "linux":
        try:
            with open(f"/proc/{pid}/status", "r", encoding="utf-8") as f:
                return "State:\tZ" in f.read()
        except FileNotFoundError:
            return False
    done = subprocess.run(  # fixed argv; a nonzero ps exit just means "no such pid"
        ["ps", "-o", "state=", "-p", str(pid)],
        capture_output=True, text=True, check=False)
    return done.stdout.strip().startswith("Z")


def _await_zombie(pid: int, timeout: float = 15.0) -> None:
    """Block until ``pid`` is a zombie, or fail naming what never happened."""
    wait_until(lambda: _is_zombie(pid), timeout=timeout,
               what=f"pid {pid} to become a zombie (child exited, unreaped)")


@pytest.fixture(autouse=True)
def _clean_reaper_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reap registry is module state shared across a test file."""
    monkeypatch.setattr(kbd, "_live_worker_procs", {})
    monkeypatch.setattr(kbd, "_recent_worker_exits", {})


# ---------------------------------------------------------------------------
# the gate: an unrelated child keeps its exit status
# ---------------------------------------------------------------------------


@pytest.mark.platforms("posix")
def test_control_the_same_failing_child_reads_3_when_no_reap_runs():
    """Control arm for the gate below: identical child, identical read, no
    ``reap_worker_zombies()`` call. It reads 3 — so a 3 in the gate is the
    reaper leaving the status alone, not the child failing to fail."""
    proc = _fail_after(0.0, 3)
    proc.communicate()
    assert proc.returncode == 3


@pytest.mark.platforms("posix")
def test_reaper_does_not_steal_an_unregistered_childs_exit_status():
    """A child of this process that is NOT a kanban worker is none of the
    reaper's business: the sweep returns nothing for it and its owner still
    reads the real exit code. Before the fix ``waitpid(-1)`` claimed it and the
    owner read rc=0 for a child that exited 3."""
    # Repeated, not a ratio: every iteration asserts, so one stolen status fails
    # the test. (An ``n/5`` count cannot distinguish a fix from a lucky run.)
    for _ in range(3):
        proc = _fail_after(0.0, 3)
        _await_zombie(proc.pid)
        assert kbd.reap_worker_zombies() == [], (
            "the reaper claimed a child it did not spawn")
        # Nothing registered: the sweep must be a no-op, not a reaper of record.
        # The zombie stays the owner's until it collects it itself.
        assert _is_zombie(proc.pid), "the sweep consumed an unregistered child's status"
        proc.communicate()
        assert proc.returncode == 3, "the reaper stole the owner's exit status"


@pytest.mark.platforms("posix")
def test_sweep_takes_the_worker_and_leaves_the_cron_shaped_child_alone():
    """The live shape: one tick, two children — the worker this process spawned
    and a sibling subsystem's child. Only the worker is reaped, and the sibling
    books its own failure."""
    worker = _fail_after(0.0, 0)
    other = _fail_after(0.0, 3)
    kbd._live_worker_procs[worker.pid] = worker
    _await_zombie(worker.pid)
    _await_zombie(other.pid)

    assert kbd.reap_worker_zombies() == [worker.pid]
    assert kbd._classify_worker_exit(worker.pid) == ("clean_exit", 0)
    assert not _is_zombie(worker.pid), "the worker must not be left a zombie"
    other.communicate()
    assert other.returncode == 3


# ---------------------------------------------------------------------------
# the worker half: reaping and classification are unchanged
# ---------------------------------------------------------------------------


@pytest.mark.platforms("posix")
def test_registered_worker_nonzero_exit_is_reaped_and_classified():
    proc = _fail_after(0.0, 3)
    kbd._live_worker_procs[proc.pid] = proc
    _await_zombie(proc.pid)

    assert kbd.reap_worker_zombies() == [proc.pid]
    assert kbd._classify_worker_exit(proc.pid) == ("nonzero_exit", 3)
    assert not _is_zombie(proc.pid), "a reaped worker must not be left a zombie"
    assert proc.pid not in kbd._live_worker_procs, "registry must not keep a reaped worker"
    assert kbd.reap_worker_zombies() == [], "a reaped worker must not be reaped twice"


@pytest.mark.platforms("posix")
def test_registered_worker_rate_limit_exit_still_requeues():
    """The rate-limit sentinel (75) must stay a requeue, not a crash — the
    property the POSIX ``waitpid`` path was originally written for."""
    proc = _fail_after(0.0, kb.KANBAN_RATE_LIMIT_EXIT_CODE)
    kbd._live_worker_procs[proc.pid] = proc
    _await_zombie(proc.pid)

    assert kbd.reap_worker_zombies() == [proc.pid]
    assert kbd._classify_worker_exit(proc.pid) == (
        "rate_limited", kb.KANBAN_RATE_LIMIT_EXIT_CODE)


@pytest.mark.platforms("posix")
def test_registered_worker_killed_by_signal_is_classified_signaled():
    """A signalled worker keeps its signal: ``Popen.returncode`` reports
    ``-signum``, and the registry's wait-status layout carries the signal in the
    low 7 bits. Encoding it as an exit code would book a SIGKILLed worker as a
    plain nonzero exit and lose the signal."""
    proc = subprocess.Popen(  # fixed argv: this interpreter, one -c expression
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    kbd._live_worker_procs[proc.pid] = proc
    try:
        os.kill(proc.pid, signal.SIGKILL)
        _await_zombie(proc.pid)

        assert kbd.reap_worker_zombies() == [proc.pid]
        assert kbd._classify_worker_exit(proc.pid) == ("signaled", signal.SIGKILL)
    finally:
        proc.communicate()


# ---------------------------------------------------------------------------
# end to end: cron's script runner around a live reaper
# ---------------------------------------------------------------------------


@pytest.mark.platforms("posix")
def test_failed_cron_script_is_not_booked_clean_while_the_reaper_runs(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
):
    """End-to-end through the REAL cron script runner (``_run_job_script``),
    with the REAL reaper hammering in another thread — the gateway shape, where
    the cron ticker and the kanban dispatcher share one process.

    ``_run_job_script`` decides success on ``proc.returncode``. Before the fix
    the reaper's ``waitpid(-1)`` raced it for the script's status and the run
    came back ``(True, "")`` for a script that exited 3: the job reported
    success and the failure was booked, under a worker pid, as a clean exit.
    """
    from cron.scheduler_script import _run_job_script

    home = tmp_path / ".hermes"
    (home / "scripts").mkdir(parents=True)
    (home / "cron" / "output").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    script = home / "scripts" / "fails.py"
    script.write_text("import sys\nsys.exit(3)\n", encoding="utf-8")

    stop = threading.Event()
    reaps: list[int] = []

    def _hammer() -> None:
        while not stop.is_set():
            reaps.extend(kbd.reap_worker_zombies())

    hammer = threading.Thread(target=_hammer, daemon=True)
    hammer.start()
    try:
        # The script needs no Hermes deps: pin the interpreter to this one so the
        # run exercises the child + status path, not the venv-resolution path.
        results = [_run_job_script(str(script), interpreter=sys.executable) for _ in range(5)]
    finally:
        stop.set()
        hammer.join(timeout=10)

    for success, output in results:
        assert success is False, f"a script that exited 3 was booked clean: {output!r}"
        assert "Script exited with code 3" in output
    assert not reaps, f"the reaper reaped a child it never spawned: {reaps}"


@pytest.mark.platforms("posix")
def test_an_entry_without_a_handle_is_skipped_not_raised(monkeypatch):
    """A registry entry that is not a handle (a test double, a future spawn
    path) is skipped: the sweep runs inside a gateway tick and may not raise."""
    class _NotAHandle:
        pid = 2_147_483_000

    monkeypatch.setattr(kbd, "_live_worker_procs", {2_147_483_000: _NotAHandle()})
    assert kbd.reap_worker_zombies() == []
