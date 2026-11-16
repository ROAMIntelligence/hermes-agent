"""One-shot ``chat -q`` runs report their outcome in the exit code, like ``-Q`` always did.

The non-quiet one-shot path used to fall through to an implicit rc=0 for every
outcome, so scripts could not tell a failed run from a good one and the Kanban
dispatcher (which spawns ``chat -q`` workers) booked a provider quota wall as a
protocol violation (#111770, #101800; salvage of #110917 / #97623).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import cli
from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE, KANBAN_TERMINAL_PROVIDER_EXIT_CODE


@pytest.fixture(autouse=True)
def _no_inherited_kanban_env(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)


def _run_non_quiet(monkeypatch, turn_result):
    """Drive ``_run_single_query_mode`` down the non-quiet tail; return the exit code (None = fell through)."""
    monkeypatch.setattr(cli, "_should_seed_interactive", lambda *a, **k: False)
    monkeypatch.setattr(cli, "_collect_query_images", lambda q, i: (q, []))
    monkeypatch.setattr(cli, "_collect_kanban_task_images", lambda imgs: [])
    monkeypatch.setattr(cli, "_finalize_single_query", lambda c: None)
    stub = SimpleNamespace(
        _single_query_mode=False,
        _claim_active_session=lambda *a, **k: True,
        console=SimpleNamespace(print=lambda *a, **k: None),
        _show_security_advisories=lambda: None,
        chat=lambda *a, **k: "response",
        _print_exit_summary=lambda **k: None,
        _last_turn_result=turn_result,
    )
    try:
        cli._run_single_query_mode(stub, "do the thing", None, False, True)
    except SystemExit as exc:
        return exc.code
    return None


def _exit_trailer_lines(err: str) -> list[str]:
    return [ln for ln in err.splitlines() if ln.startswith("[kanban-worker-exit] ")]


@pytest.mark.parametrize(
    "reason", ["rate_limit", "upstream_rate_limit", "billing", "overloaded", "server_error", "timeout"]
)
def test_dispatcher_spawned_worker_signals_a_provider_outage_not_a_protocol_violation(monkeypatch, capsys, reason):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc123")
    code = _run_non_quiet(monkeypatch, {"failed": True, "failure_reason": reason})
    assert code == KANBAN_RATE_LIMIT_EXIT_CODE
    # The 75 carries its own provider reason: the dispatcher refunds a 75 only on this evidence.
    assert _exit_trailer_lines(capsys.readouterr().err) == [f"[kanban-worker-exit] rc=75 reason={reason}"]


@pytest.mark.parametrize(
    "reason", ["auth", "auth_permanent", "model_not_found", "ssl_cert_verification", "upstream_blocked"]
)
def test_dispatcher_spawned_worker_signals_a_terminal_provider_error(monkeypatch, reason):
    """A revoked credential / missing model / WAF User-Agent block cannot be retried into working:
    the worker says so with EX_CONFIG so the dispatcher parks the card after one spawn. A person's
    run keeps 1."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc123")
    assert _run_non_quiet(monkeypatch, {"failed": True, "failure_reason": reason}) == KANBAN_TERMINAL_PROVIDER_EXIT_CODE
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    assert _run_non_quiet(monkeypatch, {"failed": True, "failure_reason": reason}) == 1


def test_dispatcher_spawned_worker_keeps_a_plain_failure_at_one(monkeypatch):
    """Control: a task-level failure (or an unknown reason) is neither transient nor terminal —
    the worker exits 1 and the dispatcher counts it against ``kanban.failure_limit`` as before."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc123")
    assert _run_non_quiet(monkeypatch, {"failed": True, "failure_reason": "some_unknown_reason"}) == 1
    assert _run_non_quiet(monkeypatch, {"failed": True}) == 1


@pytest.mark.parametrize(
    ("turn_result", "expected"),
    [
        ({"final_response": "done", "completed": True}, 0),
        ({"failed": True, "failure_reason": "rate_limit"}, 1),  # a person's run: a wall is just a failure
        ({"final_response": "half", "completed": False, "partial": True}, 1),
        ({"completed": False, "interrupted": True}, 130),
        (None, 1),  # credentials / agent init failed before any turn ran
    ],
)
def test_a_plain_one_shot_run_reports_its_outcome(monkeypatch, turn_result, expected):
    assert _run_non_quiet(monkeypatch, turn_result) == expected


@pytest.mark.parametrize(
    ("rate_limited", "expected", "trailer"),
    [(True, KANBAN_RATE_LIMIT_EXIT_CODE, "[kanban-worker-exit] rc=75 reason=credentials_rate_limited"),
     (False, 1, "[kanban-worker-exit] rc=1")],
)
def test_quiet_kanban_worker_exits_tempfail_when_credentials_are_rate_limited(
        monkeypatch, capsys, rate_limited, expected, trailer):
    """A 429 at startup never produces a turn result; the worker must still exit 75 (#117482),
    naming the wall in its exit trailer. A real credential failure keeps exit 1, with no reason."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc123")
    monkeypatch.setattr(cli, "_should_seed_interactive", lambda *a, **k: False)
    monkeypatch.setattr(cli, "_collect_query_images", lambda q, i: (q, []))
    monkeypatch.setattr(cli, "_collect_kanban_task_images", lambda imgs: [])
    monkeypatch.setattr(cli, "_finalize_single_query", lambda c: None)
    stub = SimpleNamespace(
        _claim_active_session=lambda *a, **k: True,
        _ensure_runtime_credentials=lambda: False,
        _credentials_rate_limited=rate_limited,
        session_id="s1",
        model="gpt-x",
    )
    with pytest.raises(SystemExit) as exc:
        cli._run_single_query_mode(stub, "do the thing", None, True, True)
    assert exc.value.code == expected
    assert _exit_trailer_lines(capsys.readouterr().err) == [trailer]


@pytest.mark.parametrize(
    ("kanban", "result", "credentials_rate_limited", "expected"),
    [
        (True, {"failed": True, "failure_reason": "rate_limit"}, False, (KANBAN_RATE_LIMIT_EXIT_CODE, "rate_limit")),
        (True, {"failed": True, "failure_reason": "auth"}, False, (KANBAN_TERMINAL_PROVIDER_EXIT_CODE, None)),
        (True, None, True, (KANBAN_RATE_LIMIT_EXIT_CODE, "credentials_rate_limited")),
        (True, {"failed": True}, False, (1, None)),
        (False, {"failed": True, "failure_reason": "rate_limit"}, False, (1, None)),
        (False, None, True, (1, None)),
    ],
)
def test_single_query_exit_pairs_a_throttle_code_with_its_reason(
        monkeypatch, kanban, result, credentials_rate_limited, expected):
    """The classifier's reason is the durable half of a 75: only a Kanban worker's provider wall
    has one. ``_single_query_exit_code`` stays the code half of the same decision."""
    from hermes_cli.cli_single_query import _single_query_exit, _single_query_exit_code

    if kanban:
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc123")
    assert _single_query_exit(result, credentials_rate_limited=credentials_rate_limited) == expected
    assert _single_query_exit_code(result, credentials_rate_limited=credentials_rate_limited) == expected[0]


@pytest.mark.parametrize(
    ("result", "credentials_rate_limited"),
    [*[({"failed": True, "failure_reason": r}, False)
       for r in ("rate_limit", "upstream_rate_limit", "billing", "overloaded", "server_error", "timeout")],
     (None, True)],
)
def test_every_reason_a_worker_writes_is_evidence_the_sweep_accepts(monkeypatch, result, credentials_rate_limited):
    """Producer and dispatcher share one allowlist: a reason the worker attaches to its 75 must be
    one the dead-worker sweep accepts as throttle evidence, or a real quota wall gets charged."""
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli.cli_single_query import _single_query_exit

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc123")
    code, reason = _single_query_exit(result, credentials_rate_limited=credentials_rate_limited)
    assert code == KANBAN_RATE_LIMIT_EXIT_CODE
    assert kbd._rate_limit_evidence(reason) is True


@pytest.mark.parametrize(
    ("reason", "trailer"),
    [
        ("rate_limit", "[kanban-worker-exit] rc=75 reason=rate_limit"),
        ("Rate-Limit!", "[kanban-worker-exit] rc=75 reason=ratelimit"),  # sanitized to [a-z0-9_]
        ("--", "[kanban-worker-exit] rc=75"),  # nothing left after sanitizing: bare trailer
        (None, "[kanban-worker-exit] rc=75"),
    ],
)
def test_exit_trailer_carries_a_reason_only_when_one_is_given(monkeypatch, capsys, reason, trailer):
    from hermes_cli.quiet_single_query import exit_single_query_with_reason

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc123")
    with pytest.raises(SystemExit) as exc:
        exit_single_query_with_reason(KANBAN_RATE_LIMIT_EXIT_CODE, reason=reason)
    assert exc.value.code == KANBAN_RATE_LIMIT_EXIT_CODE
    assert _exit_trailer_lines(capsys.readouterr().err) == [trailer]


def test_a_bare_exit_trailer_is_unchanged_for_callers_that_pass_no_reason(monkeypatch, capsys):
    from hermes_cli.quiet_single_query import exit_single_query

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc123")
    with pytest.raises(SystemExit):
        exit_single_query(KANBAN_RATE_LIMIT_EXIT_CODE)
    (line,) = _exit_trailer_lines(capsys.readouterr().err)
    assert line == "[kanban-worker-exit] rc=75" and " reason=" not in line


class _RewritingEmitter:
    """A stream-json emitter whose result record reports a different exit code than it was handed."""

    def __init__(self, code: int) -> None:
        self.code = code

    def emit_result(self, result, session_id="", exit_code=0):
        return self.code


@pytest.mark.parametrize(
    ("emitter", "expected", "trailer"),
    [(None, KANBAN_RATE_LIMIT_EXIT_CODE, "[kanban-worker-exit] rc=75 reason=rate_limit"),
     (_RewritingEmitter(1), 1, "[kanban-worker-exit] rc=1")],
)
def test_quiet_worker_writes_the_reason_of_the_code_it_exits_with(monkeypatch, capsys, emitter, expected, trailer):
    """``-Q`` path: the provider reason rides only the 75 it was classified with — an emitter that
    rewrites the exit code drops it, so a non-75 exit never carries throttle evidence."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc123")
    agent = SimpleNamespace(
        run_conversation=lambda **kw: {"failed": True, "failure_reason": "rate_limit", "error": "HTTP 429"},
        session_id="s-1",
    )
    with pytest.raises(SystemExit) as exc:
        cli._run_quiet_single_query(
            SimpleNamespace(agent=agent, conversation_history=[], session_id="s-1"), "hello", emitter=emitter)
    assert exc.value.code == expected
    assert _exit_trailer_lines(capsys.readouterr().err) == [trailer]
