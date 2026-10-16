"""Tests for kanban worker in-place turn recovery and the shared exit policy.

Unit contracts first (retry authority, the live run/claim proof, the recovery
loop), then call-site behaviour parametrized over the driver — production
``chat -q`` and quiet ``-Q`` — so each behaviour is stated once, then the
goal-mode veto as one table.

The exit mapping itself is main's ``cli._single_query_exit_code`` (pinned by
``tests/hermes_cli/test_single_query_exit_contract.py``); only the stripped
worker predicate this PR adds to it is pinned here.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.kanban_turn_recovery import (
    DEFAULT_MAX_RECOVERY_ATTEMPTS,
    RECOVERY_DELAYS_SECONDS,
    build_recovery_nudge,
    kanban_task_id,
    kanban_turn_recovery_enabled,
    max_recovery_attempts,
    recover_failed_kanban_turns,
    recovery_delay_seconds,
    should_recover_turn,
    worker_claim_is_live,
)
from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE

KANBAN_ENV = (
    "HERMES_KANBAN_TASK",
    "HERMES_KANBAN_TURN_RECOVERY",
    "HERMES_KANBAN_RUN_ID",
    "HERMES_KANBAN_CLAIM_LOCK",
    "HERMES_KANBAN_DB",
    "HERMES_KANBAN_BOARD",
)

#: The two one-shot routes sharing the exit decision: production ``chat -q`` and quiet ``-Q``.
DRIVERS = ("chat", "quiet")

#: Any "live" lease for these tests (2100-01-01 UTC).
_FAR_FUTURE_EXPIRY = 4_102_444_800


@pytest.fixture
def clear_kanban_env(monkeypatch):
    for var in KANBAN_ENV:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _worker_env(monkeypatch, *, task="t_probe", goal_mode=False, recovery=None):
    """The environment a dispatcher-spawned worker sees (goal mode is its flag)."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", task)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    if goal_mode:
        monkeypatch.setenv("HERMES_KANBAN_GOAL_MODE", "1")
    if recovery is not None:
        monkeypatch.setenv("HERMES_KANBAN_TURN_RECOVERY", str(recovery))


def _failed(*, retryable: bool = True, reason: str = "timeout",
            error: str = "peer closed connection") -> dict:
    return {"failed": True, "failure_retryable": retryable, "failure_reason": reason,
            "error": error, "final_response": f"API call failed after 3 retries: {error}",
            "completed": False, "messages": []}


def _success() -> dict:
    return {"failed": False, "completed": True, "final_response": "done", "messages": []}


def _interrupted() -> dict:
    """``agent/turn_recovery.py::abort_turn_on_interrupt`` already persisted and cleared
    the interrupt — re-entering the model would resurrect cancelled work."""
    return {**_failed(), "interrupted": True, "api_calls": 4}


def _settled() -> dict:
    """#87096: the bounded finalizer recorded a durable ``timed_out`` and released the
    claim while the result still reads ``completed=False`` — and here it ALSO carries a
    retryable failure stamp (the loop exhausted its provider retries, then hit the
    iteration cap). The settlement veto must win over the eligible-looking stamp."""
    return {**_failed(retryable=True), "turn_exit_reason": "max_iterations_reached(60/60)"}


def _partial() -> dict:
    """An outcome, not retry authority: truncation repair belongs in the conversation
    loop (#89289)."""
    return {"partial": True, "completed": False, "final_response": "cut off",
            "error": "truncated", "messages": []}


# ── enablement / budget ──────────────────────────────────────────────


def test_disabled_without_kanban_task(clear_kanban_env):
    assert kanban_task_id() is None
    assert kanban_turn_recovery_enabled() is False
    calls: list[str] = []
    attempts = recover_failed_kanban_turns(
        lambda nudge: calls.append(nudge), lambda: _failed(), sleep_fn=lambda s: None
    )
    assert attempts == 0
    assert calls == []


def test_env_zero_disables_and_blank_ids_are_not_workers(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    clear_kanban_env.setenv("HERMES_KANBAN_TURN_RECOVERY", "0")
    assert kanban_turn_recovery_enabled() is False
    assert should_recover_turn(_failed(), attempt=0) is False

    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "   ")
    assert kanban_task_id() is None  # whitespace-only is NOT a worker, anywhere
    assert kanban_turn_recovery_enabled() is False
    clear_kanban_env.delenv("HERMES_KANBAN_TURN_RECOVERY", raising=False)
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "  t_probe  ")
    assert kanban_task_id() == "t_probe"  # padded ids are the same worker
    assert kanban_turn_recovery_enabled() is True


def test_max_attempts_parsing_and_clamp(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    assert max_recovery_attempts() == DEFAULT_MAX_RECOVERY_ATTEMPTS  # unset
    clear_kanban_env.setenv("HERMES_KANBAN_TURN_RECOVERY", "2")
    assert max_recovery_attempts() == 2
    clear_kanban_env.setenv("HERMES_KANBAN_TURN_RECOVERY", "99")
    assert max_recovery_attempts() == 10  # clamped
    clear_kanban_env.setenv("HERMES_KANBAN_TURN_RECOVERY", "abc")
    assert max_recovery_attempts() == DEFAULT_MAX_RECOVERY_ATTEMPTS
    clear_kanban_env.setenv("HERMES_KANBAN_TURN_RECOVERY", "false")
    assert max_recovery_attempts() == 0


def test_delay_schedule_repeats_last_entry():
    assert RECOVERY_DELAYS_SECONDS == (15.0, 45.0, 90.0)
    assert recovery_delay_seconds(1) == 15.0
    assert recovery_delay_seconds(3) == 90.0
    assert recovery_delay_seconds(9) == 90.0
    assert recovery_delay_seconds(0) == 15.0


# ── retry authority (the unit contract behind both drivers) ──────────


def test_retryable_failed_turn_is_eligible(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    assert should_recover_turn(_failed(reason="timeout"), attempt=0) is True
    assert should_recover_turn(_failed(reason="timeout"), attempt=2) is True  # budget 3
    assert should_recover_turn(_failed(reason="timeout"), attempt=3) is False


@pytest.mark.parametrize(
    "result",
    [
        pytest.param(_failed(retryable=False, reason="auth"), id="not-retryable"),
        pytest.param(_success(), id="success"),
        pytest.param(None, id="nothing-settled"),
        pytest.param(_failed(reason="rate_limit"), id="quota-wall"),
        pytest.param(_failed(reason="billing"), id="billing-wall"),
        pytest.param(_failed(reason="upstream_rate_limit"), id="upstream-quota-wall"),
        pytest.param(_interrupted(), id="interrupt"),
        pytest.param(_settled(), id="terminal-settlement"),
        pytest.param(_partial(), id="unfinished-nonfailed"),
        pytest.param({"completed": False, "final_response": "x"}, id="completed-false"),
    ],
)
def test_should_recover_turn_refuses(result, clear_kanban_env):
    """Nothing but a retryable failed turn is retry authority. The aggregator's upstream
    429 is the same quota-class wall as a direct one — the dispatcher owns the cooldown."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    assert should_recover_turn(result, attempt=0) is False


# ── live run/claim proof (the adversarial boundaries) ────────────────


def _make_board(tmp_path, *, status="running", task_pid=None, run_id=1, lock="lk",
                run_ended=None, run_pid=None, task_expires=_FAR_FUTURE_EXPIRY,
                run_expires=_FAR_FUTURE_EXPIRY):
    """A synthetic board whose task carries a LIVE, unexpired task+run lease by default
    (current-main dispatcher shape: ``claim_task`` sets ``claim_expires`` on both rows
    and ``heartbeat_claim`` mirrors the extension)."""
    db = tmp_path / "kanban.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, status TEXT, worker_pid INTEGER,"
        " current_run_id INTEGER, claim_lock TEXT, claim_expires INTEGER);"
        "CREATE TABLE task_runs (id INTEGER PRIMARY KEY, ended_at INTEGER,"
        " worker_pid INTEGER, claim_expires INTEGER);"
    )
    conn.execute("INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?)",
                 ("t_live", status, task_pid, run_id, lock, task_expires))
    if run_id is not None:
        conn.execute("INSERT INTO task_runs VALUES (?, ?, ?, ?)",
                     (run_id, run_ended, run_pid, run_expires))
    conn.commit()
    conn.close()
    return db


def _pin_carrier(monkeypatch, db, *, task="t_live", run_id="1", lock="lk"):
    """Pin the full dispatcher-spawn carrier: DB + run id + claim lock."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", task)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", run_id)
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", lock)


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({}, id="owned-live"),
        pytest.param({"status": "ready"}, id="released-to-the-board"),
        pytest.param({"task_pid": 999_999_999}, id="other-process-owns-the-claim"),
        pytest.param({"run_ended": 12345}, id="run-already-closed"),
        pytest.param({"run_pid": 999_999_999}, id="run-handed-to-another-pid"),
        pytest.param({"run_id": None}, id="no-live-run-pointer"),
        pytest.param({"run_id": 2}, id="board-moved-to-another-run"),
        pytest.param({"lock": "other-lock"}, id="claim-lock-no-longer-ours"),
        pytest.param({"task_expires": 1}, id="task-lease-expired"),
        pytest.param({"run_expires": 1}, id="run-lease-expired"),
        pytest.param({"task_expires": None}, id="null-task-lease-is-not-unbounded"),
        pytest.param({"run_expires": None}, id="null-run-lease-is-not-unbounded"),
    ],
)
def test_claim_ownership_matrix(clear_kanban_env, tmp_path, overrides):
    """``worker_claim_is_live`` authorises model re-entry, so every broken coordinate must
    deny it — a NULL lease is not "unbounded", and the pinned carrier itself stays intact
    (the denial has to come from the board rows, not from a missing pin)."""
    board = {"task_pid": os.getpid(), "run_pid": os.getpid(), **overrides}
    db = _make_board(tmp_path, **board)
    _pin_carrier(clear_kanban_env, db)
    assert worker_claim_is_live() is (overrides == {})


def test_claim_check_fails_closed_without_a_board(clear_kanban_env, tmp_path):
    """No proof, no retry: a missing board must not authorise model re-entry."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_live")
    clear_kanban_env.setenv("HERMES_KANBAN_DB", str(tmp_path / "missing.db"))
    assert worker_claim_is_live() is False
    clear_kanban_env.delenv("HERMES_KANBAN_DB", raising=False)
    clear_kanban_env.setenv("HERMES_KANBAN_BOARD", "no-such-board-xyz")
    assert worker_claim_is_live() is False  # and no ambient fallback rescues it


def test_missing_pinned_carrier_fails_closed(clear_kanban_env, tmp_path):
    """Round-2 P1: the dispatcher pins DB + run id + claim lock at spawn. Any MISSING
    coordinate means there is no exact authority carrier to re-prove — the proof fails
    closed instead of widening to ambient board state or skipping comparisons."""
    db = _make_board(tmp_path)  # a live board with a live lease
    _pin_carrier(clear_kanban_env, db)
    assert worker_claim_is_live() is True  # sanity: the full carrier proves live

    for missing, value in (("HERMES_KANBAN_DB", str(db)), ("HERMES_KANBAN_RUN_ID", "1"),
                           ("HERMES_KANBAN_CLAIM_LOCK", "lk")):
        clear_kanban_env.delenv(missing, raising=False)
        assert worker_claim_is_live() is False, missing  # a missing pin denies
        clear_kanban_env.setenv(missing, value)


def test_missing_db_pin_never_resolves_an_ambient_board(tmp_path, monkeypatch):
    """Round-2 P1: the old code reconstructed the board from ``HERMES_KANBAN_BOARD`` /
    the default when the pin was absent — the proof could silently move to whatever board
    is ambient NOW. With the pin gone the canonical resolver must not be consulted at all,
    even when it would return a fully live board whose row matches this worker (built with
    the real board API, so this is not a strawman)."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="ambient live task")
        assert kb.claim_task(conn, task_id, claimer="lk", ttl_seconds=3600) is not None
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_CLAIM_LOCK", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)

    import agent.kanban_turn_recovery as rec

    consulted: list = []
    real_resolve = kb.kanban_db_path

    def _spy(board=None):
        consulted.append(board)
        return real_resolve(board=board)

    monkeypatch.setattr(kb, "kanban_db_path", _spy)
    assert rec.worker_claim_is_live() is False
    assert consulted == []  # no ambient resolution was attempted


def test_lease_expired_during_the_backoff_blocks_model_reentry(
    clear_kanban_env, tmp_path, monkeypatch
):
    """Round-2 P1 (the reviewer's exact race): failure near TTL expiry → the pre-backoff
    proof passes → the 15/45/90s sleep crosses expiry → the post-backoff re-proof must FAIL
    on the unexpired-lease check even though status/run-id/lock/pid/ended_at are all
    unchanged. No model turn is made."""
    import agent.kanban_turn_recovery as rec

    db = _make_board(tmp_path, task_expires=1_600, run_expires=1_600)
    _pin_carrier(clear_kanban_env, db, run_id="1", lock="lk")
    clock = iter([1_000, 1_700])  # pre-backoff proof sees a live lease; the re-proof does not
    monkeypatch.setattr(rec, "_now", lambda: next(clock))
    turns: list[str] = []
    emitted: list[str] = []

    attempts = rec.recover_failed_kanban_turns(
        lambda nudge: turns.append(nudge), lambda: _failed(),
        sleep_fn=lambda s: None, emit=emitted.append,  # default claim_check: the real proof
    )
    assert attempts == 1
    assert turns == []  # the expired lease vetoed re-entry before the model call
    assert "no longer holds a live run/claim" in emitted[-1]


# ── the loop ─────────────────────────────────────────────────────────


def _loop_ok(result=None):
    return (lambda: True) if result is None else result


def test_recovery_loop_retries_until_success_and_reports(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    latest = {"r": _failed()}
    turns: list[str] = []
    delays: list[float] = []
    emitted: list[str] = []

    def turn_fn(nudge: str) -> None:
        turns.append(nudge)
        latest["r"] = _success()  # provider recovered

    attempts = recover_failed_kanban_turns(
        turn_fn, lambda: latest["r"], sleep_fn=delays.append, emit=emitted.append,
        claim_check=_loop_ok(),
    )
    assert attempts == 1
    assert delays == [15.0]
    assert "Do NOT start over" in turns[0]
    assert "t_probe" in turns[0]
    assert len(emitted) == 1
    assert "[kanban]" in emitted[0] and "attempt 1/3" in emitted[0]


def test_recovery_loop_bounded_when_result_never_changes(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    delays: list[float] = []
    attempts = recover_failed_kanban_turns(
        lambda nudge: None,
        lambda: _failed(),  # stuck result: must NOT loop forever
        sleep_fn=delays.append, emit=lambda m: None, claim_check=_loop_ok(),
    )
    assert attempts == DEFAULT_MAX_RECOVERY_ATTEMPTS
    assert delays == [15.0, 45.0, 90.0]


def test_recovery_loop_order_is_check_sleep_check_turn(clear_kanban_env):
    """Round-5 finding F1, both halves in one pass: the proof is taken BEFORE the backoff
    and re-taken immediately before model re-entry (check → sleep → check → turn), so a run
    lost during the 15/45/90s sleep stops the loop WITHOUT re-entering the model."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    clear_kanban_env.setenv("HERMES_KANBAN_TURN_RECOVERY", "3")
    events: list[str] = []
    latest = {"r": _failed()}
    proofs = {"n": 0}

    def claim_check() -> bool:
        proofs["n"] += 1
        events.append("check")
        return proofs["n"] == 1  # first proof holds; the run is lost during the backoff

    def turn_fn(nudge: str) -> None:
        events.append("turn")  # a model entry: must NOT happen on the denied path
        latest["r"] = _success()

    attempts = recover_failed_kanban_turns(
        turn_fn, lambda: latest["r"], sleep_fn=lambda s: events.append("sleep"),
        emit=lambda m: None, claim_check=claim_check,
    )
    assert events == ["check", "sleep", "check"]  # the lost run vetoed re-entry
    assert attempts == 1  # the attempt was authorised …

    # …and with the proof intact, the happy path is the full check → sleep → check → turn.
    events.clear()
    latest["r"] = _failed()
    attempts = recover_failed_kanban_turns(
        turn_fn, lambda: latest["r"], sleep_fn=lambda s: events.append("sleep"),
        emit=lambda m: None, claim_check=lambda: (events.append("check"), True)[1],
    )
    assert attempts == 1
    assert events == ["check", "sleep", "check", "turn"]


def test_nudge_terminal_contract(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    nudge = build_recovery_nudge(_failed(error="peer closed connection"), attempt=1, max_attempts=3)
    assert "t_probe" in nudge
    assert "1/3" in nudge
    assert "kanban_complete" in nudge
    assert "kanban_block" in nudge
    assert "Do NOT start over" in nudge
    assert "peer closed connection" in nudge


# ── what this PR adds to the shared exit mapping ─────────────────────


def test_exit_mapping_composes_with_the_stripped_worker_predicate(monkeypatch):
    """``cli._single_query_exit_code`` is main's (its own contract test pins the mapping);
    this PR's contribution is that it asks ``kanban_task_id()`` whether this is a worker.
    Pin exactly that: surrounding whitespace must not decide worker-ness, or a
    bare-whitespace id would hand a quota wall the neutral 75."""
    from cli import _single_query_exit_code

    monkeypatch.setenv("HERMES_KANBAN_TASK", "   ")
    assert _single_query_exit_code(_failed(reason="rate_limit")) == 1
    monkeypatch.setenv("HERMES_KANBAN_TASK", "  t_probe  ")
    assert _single_query_exit_code(_failed(reason="rate_limit")) == KANBAN_RATE_LIMIT_EXIT_CODE


# ── call-site behaviour: one harness for both drivers ────────────────


@pytest.fixture
def cli_harness(monkeypatch):
    """Drive the real ``cli.main`` on either one-shot route through one fake CLI.

    ``script`` is the sequence of settled turn results — or a callable returning one on
    every call. Returns ``(cli_mod, entered, ui)``, where ``entered`` records the user
    message of every model entry, so a test can prove how many turns were taken and what
    the recovery nudge carried, and ``ui`` records non-turn route marks (the exit summary).
    """

    def _install(driver, script):
        import cli as cli_mod

        entered: list[str] = []
        ui: list[str] = []
        seen = {"n": 0}

        class _Feed:
            """Superset of both routes' surfaces: ``chat -q`` drives ``chat()``, the quiet
            route drives ``agent.run_conversation``."""

            def __init__(self, **_kwargs):
                self.console = SimpleNamespace(print=lambda *a, **k: None)
                self.provider = "test-provider"
                self.model = "test-model"
                self.session_id = "single-query-session"
                self.conversation_history = []
                self._active_agent_route_signature = "same-route"
                self.agent = SimpleNamespace(
                    session_id="single-query-session", platform="cli", quiet_mode=False,
                    suppress_status_output=False, stream_delta_callback=object(),
                    tool_gen_callback=object(), run_conversation=self._run_conversation,
                )

            def _next(self):
                result = script() if callable(script) else script[seen["n"]]
                seen["n"] += 1
                return result

            def _run_conversation(self, *, user_message, conversation_history):
                entered.append(user_message)
                return self._next()

            def chat(self, query, images=None):
                entered.append(query)
                result = self._next()
                self._last_turn_result = result
                # Mirror production: chat() returns the settled turn's rendered response —
                # for a failed turn that IS the provider error text.
                return result.get("final_response", "") if isinstance(result, dict) else ""

            # route plumbing
            def _claim_active_session(self, surface, *, stderr=False):
                return True

            def _ensure_runtime_credentials(self):
                return True

            def _resolve_turn_agent_config(self, effective_query):
                return {"signature": "same-route", "model": None, "runtime": None,
                        "request_overrides": None}

            def _init_agent(self, **kwargs):
                return True

            def _show_security_advisories(self):
                pass

            def _print_exit_summary(self, clear_screen=True):
                ui.append("summary")

        monkeypatch.setattr(cli_mod, "HermesCLI", _Feed)
        monkeypatch.setattr(cli_mod.atexit, "register", lambda *a, **k: None)
        monkeypatch.setattr(cli_mod, "_finalize_single_query", lambda fake_cli: None)
        monkeypatch.setattr(cli_mod, "_collect_query_images", lambda q, img: (q, []))
        monkeypatch.setattr(cli_mod, "_collect_kanban_task_images", lambda imgs: [])
        return cli_mod, entered, ui

    return _install


def _drive(cli_mod, driver) -> int:
    """Run one one-shot query on ``driver`` and return the exit code it raised."""
    kwargs: dict = {"query": "hello", "toolsets": "terminal"}
    if driver == "chat":
        kwargs.update(quiet=False, oneshot=True)
    else:
        kwargs.update(quiet=True)
    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(**kwargs)
    return exc_info.value.code


@pytest.fixture
def live_claim(monkeypatch):
    """The board proof is a separate contract (unit-tested above); the call-site tests run
    against a worker that owns its run unless the test overrides it."""
    import agent.kanban_turn_recovery as rec

    monkeypatch.setattr(rec, "worker_claim_is_live", lambda: True)
    return rec


@pytest.fixture
def goal_spy(monkeypatch):
    """Spy on BOTH goal continuation loops at the cli-module seam. The loops are
    pre-existing (their status check sees only run identity, not the claim lease); the
    invariant under test is the call-site veto this PR adds."""
    import cli as cli_mod

    calls: list = []
    monkeypatch.setattr(cli_mod, "_run_kanban_goal_loop_q",
                        lambda c, resp: calls.append((resp, "quiet")))
    monkeypatch.setattr(cli_mod, "_run_kanban_goal_loop_chat",
                        lambda c, resp: calls.append((resp, "chat")))
    return calls


@pytest.mark.parametrize("driver", DRIVERS)
def test_recovers_in_place(driver, cli_harness, live_claim, monkeypatch):
    import agent.kanban_turn_recovery as rec

    _worker_env(monkeypatch, recovery=2)
    monkeypatch.setattr(rec, "RECOVERY_DELAYS_SECONDS", (0.0,))
    cli_mod, entered, _ui = cli_harness(driver, [_failed(), _success()])

    assert _drive(cli_mod, driver) == 0  # settled successfully
    assert len(entered) == 2  # the failed turn, then the recovery turn
    assert "Do NOT start over" in entered[1]
    assert "t_probe" in entered[1]


@pytest.mark.parametrize("driver", DRIVERS)
def test_budget_exhausted_stops_and_releases(driver, cli_harness, live_claim, monkeypatch):
    """Bounded, then released: the loop stops at the cap, and the exhausted transient wall
    still reaches the dispatcher as the neutral code."""
    import agent.kanban_turn_recovery as rec

    _worker_env(monkeypatch, recovery=1)
    monkeypatch.setattr(rec, "RECOVERY_DELAYS_SECONDS", (0.0,))
    cli_mod, entered, _ui = cli_harness(driver, _failed)  # never recovers

    assert _drive(cli_mod, driver) == KANBAN_RATE_LIMIT_EXIT_CODE
    assert len(entered) == 2  # original + one recovery attempt


def _single_turn_cases():
    """One row per single-turn outcome on either route: ``(env, script, exit code)``. Every
    row takes exactly ONE model turn, so what is under test is what the route does AFTER
    that turn: the neutral release for a quota wall (#48000-class boundary, including the
    aggregator's upstream 429) versus an honest 1 for a turn that is not retry authority
    (partial / nothing-settled) or for a run that is not a worker at all."""
    neutral = KANBAN_RATE_LIMIT_EXIT_CODE
    return [
        pytest.param({}, [None], 1, id="nothing-settled"),
        pytest.param({}, [_partial()], 1, id="denial-unfinished-turn"),
        pytest.param({}, [_failed(reason="rate_limit")], neutral, id="quota-wall"),
        pytest.param({}, [_failed(reason="billing")], neutral, id="billing-wall"),
        pytest.param({}, [_failed(reason="upstream_rate_limit")], neutral,
                     id="upstream-quota-wall"),
        pytest.param({"task": "   "}, [_failed(reason="rate_limit")], 1,
                     id="blank-task-id-is-not-a-worker"),
        pytest.param({"task": None}, [_failed()], 1, id="non-worker-failed"),
        pytest.param({"task": None}, [None], 1, id="non-worker-unsettled"),
    ]


@pytest.mark.parametrize("driver", DRIVERS)
@pytest.mark.parametrize(("env", "script", "code"), _single_turn_cases())
def test_single_turn_route_outcomes(driver, env, script, code, cli_harness, live_claim,
                                    monkeypatch):
    """One representative end-to-end denial per driver, plus the quota-wall release and the
    non-worker rows. An unfinished, non-failed turn is not retry authority (the unit table
    above pins interrupt / terminal settlement / partial as well), so exactly one model turn
    is taken and the exit is the honest code rather than a silent rc=0; ``task=None`` means
    no worker env at all, and a blank id is not a worker either (its wall is NOT
    neutralized into 75 — raw env truthiness would)."""
    if "task" in env:
        if env["task"] is None:
            monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
            monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
        else:
            _worker_env(monkeypatch, task=env["task"])
    else:
        _worker_env(monkeypatch)
    cli_mod, entered, _ui = cli_harness(driver, list(script))

    assert _drive(cli_mod, driver) == code
    assert len(entered) == 1


@pytest.mark.parametrize("driver", DRIVERS)
def test_claim_denial_stops_before_model_reentry(driver, cli_harness, capsys, monkeypatch):
    """A worker whose run/claim proof fails gets no recovery nudge, says so, and still
    leaves the dispatcher its neutral release."""
    import agent.kanban_turn_recovery as rec

    _worker_env(monkeypatch)
    monkeypatch.setattr(rec, "worker_claim_is_live", lambda: False)
    cli_mod, entered, _ui = cli_harness(driver, [_failed()])

    assert _drive(cli_mod, driver) == KANBAN_RATE_LIMIT_EXIT_CODE
    assert len(entered) == 1
    assert "no longer holds a live run/claim" in capsys.readouterr().err


def test_non_quiet_route_prints_its_exit_summary(cli_harness, live_claim, monkeypatch):
    """Smoke pin kept from the pre-trim file: a plain (non-worker) failed `chat -q` run still
    prints its exit summary before exiting 1 — the non-quiet tail exits unconditionally, so
    the summary has to be printed on the way out."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    cli_mod, entered, ui = cli_harness("chat", [_failed()])

    assert _drive(cli_mod, "chat") == 1
    assert entered == ["hello"]  # no recovery outside a worker
    assert "summary" in ui


# ── goal mode: a refused recovery stops ALL model entry (one table) ──


def _goal_cases():
    """One row per way model entry can be refused while goal mode is ON:
    ``(id, env setup, script, expected exit code, expected model entries)``. Each row
    carries its own setup, because the refusal reason IS the setup."""
    neutral = KANBAN_RATE_LIMIT_EXIT_CODE
    return [
        pytest.param("recovery-denied", {"live": False}, _failed, neutral, 1),
        pytest.param("lease-expired-during-backoff", {"lease_expiry": True}, _failed, neutral, 1),
        pytest.param("quota-wall-billing", {}, lambda: _failed(reason="billing"), neutral, 1),
        pytest.param("quota-wall-upstream", {}, lambda: _failed(reason="upstream_rate_limit"),
                     neutral, 1),
        pytest.param("interrupt", {}, _interrupted, 130, 1),
        pytest.param("terminal-settlement", {}, _settled, neutral, 1),
        pytest.param("budget-exhausted", {"recovery": 1}, _failed, neutral, 2),
        pytest.param("unfinished-nonfailed", {}, _partial, 1, 1),
    ]


@pytest.mark.parametrize("driver", DRIVERS)
@pytest.mark.parametrize(("case_id", "setup", "script", "code", "entries"), _goal_cases())
def test_goal_mode_does_not_continue(
    case_id, setup, script, code, entries, driver, clear_kanban_env, tmp_path, monkeypatch,
    cli_harness, goal_spy,
):
    """With goal mode ON, every refusal path reaches the shared exit without a judge-driven
    continuation: the goal loop's own status check sees only run identity, not the claim
    lease, so continuing would re-enter the model under an authority this process can no
    longer prove."""
    import agent.kanban_turn_recovery as rec

    clear_kanban_env.setenv("HERMES_KANBAN_GOAL_MODE", "1")
    monkeypatch.setattr(rec, "RECOVERY_DELAYS_SECONDS", (0.0,))
    if setup.get("lease_expiry"):
        db = _make_board(tmp_path, task_expires=1_600, run_expires=1_600)
        _pin_carrier(clear_kanban_env, db, run_id="1", lock="lk")
        clock = iter([1_000, 1_700])  # pre-backoff proof live; the re-proof is not
        monkeypatch.setattr(rec, "_now", lambda: next(clock))
    else:
        clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
        monkeypatch.setattr(rec, "worker_claim_is_live", lambda: setup.get("live", True))
    if setup.get("recovery") is not None:
        clear_kanban_env.setenv("HERMES_KANBAN_TURN_RECOVERY", str(setup["recovery"]))

    cli_mod, entered, _ui = cli_harness(driver, script)

    assert _drive(cli_mod, driver) == code
    assert len(entered) == entries
    assert goal_spy == []  # no continuation after a refused recovery


@pytest.mark.parametrize("driver", DRIVERS)
def test_goal_mode_continues_after_a_successful_authorized_recovery(
    driver, cli_harness, live_claim, goal_spy, monkeypatch,
):
    """Positive control on both drivers: a settled, authorized recovery still continues in
    goal mode — the veto must not swallow the happy path — and the judge receives the
    RECOVERED response, not the initial provider error (re-review P2: the recovery
    callback's return value carries through)."""
    import agent.kanban_turn_recovery as rec

    _worker_env(monkeypatch, goal_mode=True, recovery=2)
    monkeypatch.setattr(rec, "RECOVERY_DELAYS_SECONDS", (0.0,))
    cli_mod, entered, _ui = cli_harness(driver, [_failed(), _success()])

    assert _drive(cli_mod, driver) == 0
    assert len(entered) == 2
    assert goal_spy == [("done", driver)]


# ── guardrail stop on a non-lifecycle tool (run 2648 class) ──────────
#
# A tool-loop guardrail block/stop on ANY tool (not just the five lifecycle
# tools PR #182 covers) used to end a worker run with the interactive canned
# reply and no terminal kanban call: the turn had no failure verdict for
# ``guardrail_halt``, read as completed, and the worker exited rc=0 -> booked
# ``protocol_violation``/``crashed`` with the card still ``running``. These
# tests pin the fix: the seam stamps the stop as a retryable failed turn ONLY
# for a worker, and the bounded in-place recovery drives it to a real terminal
# call. Every fixture is synthetic: dummy paths/text only, no live-tree
# content, no transcript payloads, nothing from the source diagnosis briefs.

import inspect  # noqa: E402  (section-local imports; kept together for the audit)

#: Dummy read result and call arguments for the synthesized block. The SHAPE of
#: run 2648 (same read-only call answered identically five times) with fake values.
_DUMMY_READ_RESULT = "[fixture] notes line one" + chr(10) + "[fixture] notes line two"
_FIXTURE_READ_ARGS = {"path": "/tmp/fixture-scratch/notes.txt"}

#: Canned interactive-reply SHAPE (an offer of ``continue`` + redirect), written for this
#: fixture - deliberately NOT the live text.
_CANNED_HALT_REPLY = (
    "[fixture] the tool loop stopped because identical results kept repeating. "
    "Send `continue` to retry from here, or redirect me."
)


def _blocked_read_decision():
    """Drive the REAL guard controller the way the recorded stop happened: the same
    read-only call answered with the same result five times, then blocked on the sixth
    (the hard-stop promotion for non-interactive platforms is already on it)."""
    from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController

    guard = ToolCallGuardrailController(
        ToolCallGuardrailConfig.from_mapping({}, platform="kanban")
    )
    for _ in range(5):
        guard.before_call("read_file", dict(_FIXTURE_READ_ARGS))
        guard.after_call("read_file", dict(_FIXTURE_READ_ARGS), _DUMMY_READ_RESULT, failed=False)
    decision = guard.before_call("read_file", dict(_FIXTURE_READ_ARGS))
    assert decision.code == "idempotent_no_progress_block"
    assert decision.count == 5
    assert decision.should_halt is True
    return decision


def _stopped_turn_result(decision, *, interrupted: bool = False) -> dict:
    """The settled turn result a guardrail stop produces on the fixed tree, synthesized
    to the recorded shape (canned reply, ``turn_exit_reason``, guardrail decision
    metadata) and stamped through the REAL seam verdict exactly the way
    ``agent/turn_finalizer.py`` consumes it: ``fails_turn`` -> ``failed=True``, then the
    ``completed=`` recompute (which follows the seam) reads False, and the assembly
    stamps the verdict fields. On a non-worker (or interrupted) path the verdict is
    None and the result keeps today's values field-for-field."""
    from agent.turn_failure_copy import stamp_failure, worker_guardrail_halt_failure

    result = {
        "failed": False,
        "completed": True,
        "final_response": _CANNED_HALT_REPLY,
        "error": "",
        "turn_exit_reason": "guardrail_halt",
        "guardrail": decision.to_metadata(),
        "messages": [],
    }
    verdict = worker_guardrail_halt_failure("guardrail_halt", interrupted=interrupted)
    if verdict is not None and verdict.fails_turn:
        result["failed"] = True
        result["completed"] = False
        result["error"] = result["final_response"]
        stamp_failure(result, verdict.reason, verdict.retryable)
    return result


def _live_board(clear_kanban_env, root, monkeypatch):
    """A REAL kanban board (full schema via ``kb.init_db``) with one claimed run, pinned
    as this worker's carrier: task id + run id + claim lock from the actual rows."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    home = root / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: root)
    kb.init_db()
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="guardrail-stop replay fixture")
        claimed = kb.claim_task(conn, task_id, claimer="replay", ttl_seconds=3600)
        assert claimed is not None
    db_path = kb.kanban_db_path()
    with kbc.connect() as conn:
        trow = conn.execute(
            "SELECT claim_lock, current_run_id FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
    assert trow["current_run_id"] is not None
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", task_id)
    clear_kanban_env.setenv("HERMES_KANBAN_DB", str(db_path))
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", str(int(trow["current_run_id"])))
    clear_kanban_env.setenv("HERMES_KANBAN_CLAIM_LOCK", str(trow["claim_lock"]))
    return SimpleNamespace(
        kb=kb, kbc=kbc, root=root, task_id=task_id,
        run_id=int(trow["current_run_id"]), lock=str(trow["claim_lock"]),
    )


def _read_board_state(board):
    with board.kbc.connect() as conn:
        trow = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (board.task_id,)
        ).fetchone()
        runs = conn.execute(
            "SELECT id, outcome, metadata FROM task_runs WHERE task_id = ? ORDER BY id",
            (board.task_id,),
        ).fetchall()
        ev = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (board.task_id,),
        ).fetchone()
    return {
        "task_status": trow["status"],
        "runs": [(int(r["id"]), r["outcome"], r["metadata"]) for r in runs],
        "last_event": ev["kind"] if ev else None,
    }


def _board_script(board, *, terminal, stopped):
    """The cooperating stub on the model seam: turn 1 is the stopped turn (or a clean
    success); on the recovery entry it performs the REAL terminal board call - exactly
    what the kanban_complete / kanban_block tool handlers do, ownership CAS included -
    and then settles successfully."""
    state = {"n": 0, "terminal_ok": None}
    decision = _blocked_read_decision() if stopped else None

    def script():
        state["n"] += 1
        if state["n"] == 1 and stopped:
            return _stopped_turn_result(decision)
        with board.kbc.connect() as conn:
            if terminal == "complete":
                state["terminal_ok"] = board.kb.complete_task(
                    conn, board.task_id,
                    summary="recovered in place after a tool-loop guardrail stop",
                    expected_run_id=board.run_id,
                )
            else:
                state["terminal_ok"] = board.kb.block_task(
                    conn, board.task_id,
                    reason="guardrail stop blocked a repeated call; needs a decision",
                    expected_run_id=board.run_id,
                )
        return _success()

    script.state = state
    return script


def _drive_board_case(clear_kanban_env, root, monkeypatch, cli_harness, driver, *,
                      terminal, stopped, recovery=None):
    board = _live_board(clear_kanban_env, root, monkeypatch)
    if recovery is not None:
        clear_kanban_env.setenv("HERMES_KANBAN_TURN_RECOVERY", str(recovery))
    script = _board_script(board, terminal=terminal, stopped=stopped)
    cli_mod, entered, _ui = cli_harness(driver, script)
    code = _drive(cli_mod, driver)
    return SimpleNamespace(
        board=board, code=code, entered=entered, script=script,
        state=_read_board_state(board),
    )


def test_guardrail_seam_values(clear_kanban_env):
    """The exact verdict the finalizer seam lands, value-for-value: retryable on purpose
    (in-place retry authority), and ONLY for the stripped worker predicate - the shared
    exit-reason map is untouched, so every other surface keeps today's values."""
    from agent.turn_failure_copy import (
        ExitFailure,
        exit_reason_failure,
        worker_guardrail_halt_failure,
    )

    assert exit_reason_failure("guardrail_halt") is None  # nothing added to the shared map

    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    verdict = worker_guardrail_halt_failure("guardrail_halt")
    assert verdict == ExitFailure(reason="guardrail_halt", retryable=True, fails_turn=True)
    stamped = _stopped_turn_result(_blocked_read_decision())
    assert stamped["failed"] is True
    assert stamped["completed"] is False  # a stop never reads as a completed turn
    assert stamped["failure_reason"] == "guardrail_halt"
    assert stamped["failure_retryable"] is True
    assert should_recover_turn(stamped, attempt=0) is True  # -> in-place retry authority
    assert should_recover_turn({**stamped, "interrupted": True}, attempt=0) is False

    clear_kanban_env.delenv("HERMES_KANBAN_TASK", raising=False)
    assert worker_guardrail_halt_failure("guardrail_halt") is None
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "   ")
    assert worker_guardrail_halt_failure("guardrail_halt") is None  # blank is not a worker
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    assert worker_guardrail_halt_failure("timeout") is None
    assert worker_guardrail_halt_failure(None) is None


def test_guardrail_seam_is_wired_into_the_finalizer():
    """If the seam call is ever dropped, the stop silently reads as completed again."""
    from agent.turn_finalizer import finalize_turn

    src = inspect.getsource(finalize_turn)
    assert "worker_guardrail_halt_failure(_turn_exit_reason, interrupted=interrupted)" in src


def test_guardrail_nudge_allowlist_and_leak_check(clear_kanban_env):
    """The nudge may interpolate ONLY tool name / decision code / count. Every other
    guardrail field (message, signature/args hash, path) must NOT cross over, and
    hostile values degrade to the fixed literals instead of echoing."""
    import agent.kanban_turn_recovery as rec

    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    nudge = rec._guardrail_halt_nudge(
        {"guardrail": {
            "tool_name": "read_file", "code": "idempotent_no_progress_block", "count": 5,
            "message": "LEAKCANARY-MESSAGE", "signature": {"args_hash": "LEAKCANARY-HASH"},
            "path": "LEAKCANARY-PATH",
        }},
        attempt=2, max_attempts=3, task_id="t_probe",
    )
    assert "LEAKCANARY" not in nudge
    assert "read_file" in nudge and "idempotent_no_progress_block" in nudge
    assert "count=5" in nudge and "2/3" in nudge
    assert "STOPPED by a tool-loop guardrail" in nudge
    assert "Do NOT start over" in nudge
    assert "API call failed" not in nudge  # never the provider-failure narration
    start = nudge.index("reason: '") + len("reason: '")
    end = nudge.index("').]", start)
    suggested_line = nudge[start:end]
    assert suggested_line and len(suggested_line) <= 300 and chr(10) not in suggested_line

    hostile = rec._guardrail_halt_nudge(
        {"guardrail": {"tool_name": "read_file; hostile-payload", "code": "not_a_real_code",
                       "count": "many"}},
        attempt=1, max_attempts=3, task_id="t_probe",
    )
    assert "hostile-payload" not in hostile and "not_a_real_code" not in hostile
    assert "count=" not in hostile  # non-int count is dropped, not stringified
    assert "Tool `tool` was blocked" in hostile  # literal fallback


def test_build_recovery_nudge_routes_the_guardrail_variant(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    guardrail = build_recovery_nudge(
        _stopped_turn_result(_blocked_read_decision()), attempt=1, max_attempts=3
    )
    assert "STOPPED by a tool-loop guardrail" in guardrail
    assert "Do NOT start over" in guardrail
    assert "kanban_complete" in guardrail and "kanban_block" in guardrail

    provider = build_recovery_nudge(_failed(), attempt=1, max_attempts=3)
    assert "peer closed connection" in provider
    assert "STOPPED by a tool-loop guardrail" not in provider

    reason_only = build_recovery_nudge(
        {"failed": True, "failure_retryable": True, "failure_reason": "guardrail_halt"},
        attempt=1, max_attempts=3,
    )
    assert "STOPPED by a tool-loop guardrail" in reason_only


@pytest.mark.parametrize("driver", DRIVERS)
@pytest.mark.parametrize(
    ("terminal", "expected_status"),
    [("complete", "done"), ("block", "blocked")],
)
def test_guardrail_replay_reaches_a_terminal_call(
    driver, terminal, expected_status, clear_kanban_env, tmp_path, monkeypatch,
    cli_harness, live_claim,
):
    """AC1 forced replay, value-level. Clean-path recording FIRST: the same terminal call
    with no stop anywhere is the expected (status, outcome, event) - then the stopped
    replay must EQUAL it, not merely differ from 'crashed'. The run id survives, no run
    row is added, and the run metadata carries no protocol_violation key."""
    import agent.kanban_turn_recovery as rec

    monkeypatch.setattr(rec, "RECOVERY_DELAYS_SECONDS", (0.0,))

    clean = _drive_board_case(
        clear_kanban_env, tmp_path / "clean", monkeypatch, cli_harness, driver,
        terminal=terminal, stopped=False,
    )
    assert clean.code == 0
    assert len(clean.entered) == 1
    assert clean.state["task_status"] == expected_status
    assert clean.state["last_event"] in ("completed", "blocked")
    assert clean.state["runs"][0][1] is not None  # the clean outcome, recorded first

    stopped = _drive_board_case(
        clear_kanban_env, tmp_path / "stopped", monkeypatch, cli_harness, driver,
        terminal=terminal, stopped=True,
    )
    assert stopped.code == 0  # terminal call made; the stop did not strand the run
    assert len(stopped.entered) == 2  # the stopped turn, then the recovery turn
    nudge = stopped.entered[1]
    assert "STOPPED by a tool-loop guardrail" in nudge
    assert "read_file" in nudge and "idempotent_no_progress_block" in nudge
    assert "count=5" in nudge
    assert "Do NOT start over" in nudge
    assert "kanban_complete" in nudge and "kanban_block" in nudge
    assert "API call failed" not in nudge
    assert stopped.script.state["terminal_ok"] is True

    # equality with the clean path, value-for-value
    assert stopped.state["task_status"] == clean.state["task_status"]
    assert stopped.state["runs"][0][1] == clean.state["runs"][0][1]  # run outcome EQUALS
    assert stopped.state["last_event"] == clean.state["last_event"]  # terminal call
    # run-id unchanged across stop -> recover, zero extra task_runs rows
    assert len(stopped.state["runs"]) == 1 and len(clean.state["runs"]) == 1
    assert stopped.state["runs"][0][0] == stopped.board.run_id
    # no protocol_violation marker in the run metadata
    assert "protocol_violation" not in (stopped.state["runs"][0][2] or "")
    assert "protocol_violation" not in (clean.state["runs"][0][2] or "")


@pytest.mark.parametrize("driver", DRIVERS)
def test_guardrail_budget_exhaustion_and_repeat_exit_nonzero(
    driver, clear_kanban_env, monkeypatch, cli_harness, live_claim,
):
    """AC2: a stub that re-stops on every recovery turn (never terminal) consumes the
    pinned budget monotonically (1 -> 2 -> 3, never more) and then the process exits as a
    REAL failure: rc=1 -> nonzero_exit, never the clean_exit class that books
    protocol_violation."""
    import agent.kanban_turn_recovery as rec
    from hermes_cli.kanban_db_dispatch import _exit_code_kind

    _worker_env(monkeypatch)
    monkeypatch.setattr(rec, "RECOVERY_DELAYS_SECONDS", (0.0,))
    assert rec.max_recovery_attempts() == rec.DEFAULT_MAX_RECOVERY_ATTEMPTS == 3

    ladder: list = []
    real_delay = rec.recovery_delay_seconds
    monkeypatch.setattr(
        rec, "recovery_delay_seconds", lambda a: (ladder.append(a), real_delay(a))[1]
    )

    def restop():
        return _stopped_turn_result(_blocked_read_decision())

    cli_mod, entered, _ui = cli_harness(driver, restop)
    code = _drive(cli_mod, driver)

    assert code == 1
    assert _exit_code_kind(code) == ("nonzero_exit", 1)
    assert len(entered) == 1 + rec.DEFAULT_MAX_RECOVERY_ATTEMPTS  # 1 stop + 3 re-entries
    assert ladder == [1, 2, 3]  # monotone consumption, capped at the budget


@pytest.mark.parametrize("driver", DRIVERS)
def test_guardrail_negative_controls(
    driver, clear_kanban_env, monkeypatch, cli_harness,
):
    """AC3 field-level negative controls: a NON-worker stopped turn keeps every observable
    value it has today (failed False, no stamp, turn_exit_reason intact, rc=0
    clean_exit - cron silence / gateway transcript rows key on exactly these);
    interrupted wins over the seam; a normal completed worker turn takes no extra
    retries."""
    from agent.turn_failure_copy import worker_guardrail_halt_failure
    from cli import _single_query_exit_code
    from hermes_cli.kanban_db_dispatch import _exit_code_kind

    # (a) interrupted wins, even on a worker
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    assert worker_guardrail_halt_failure("guardrail_halt", interrupted=True) is None

    # (b) non-worker: the seam leaves the result exactly as today
    clear_kanban_env.delenv("HERMES_KANBAN_TASK", raising=False)
    result = _stopped_turn_result(_blocked_read_decision())
    assert result["failed"] is False and "failure_reason" not in result
    assert result["turn_exit_reason"] == "guardrail_halt"
    assert _single_query_exit_code(result) == 0
    assert _exit_code_kind(0) == ("clean_exit", 0)
    cli_mod, entered, _ui = cli_harness(driver, [result])
    assert _drive(cli_mod, driver) == 0
    assert len(entered) == 1  # no recovery attempt on a non-worker stop

    # (c) a normal completed worker turn: same single turn, same exit
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    cli_mod, entered, _ui = cli_harness(driver, [_success()])
    assert _drive(cli_mod, driver) == 0
    assert len(entered) == 1
