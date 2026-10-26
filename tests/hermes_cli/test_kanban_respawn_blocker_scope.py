"""Respawn-guard scope: only provider-shaped text parks a card, and a bookkeeping note never can.

Follow-up to #119070 on card t_abd07c8a. ``check_respawn_guard`` reads
``tasks.last_failure_error`` as evidence of a quota/auth wall, and that column also carries the
board's OWN bookkeeping notes: the reclaim note's ``stale_lock=<host>:<pid>`` token, the timeout
template's ``elapsed Ns > limit Ns`` fields, every sweep line naming the worker pid. One of those
must never return ``blocker_auth`` — only a completion clears it, so a bookkeeping note parks the
card until an operator intervenes.

The claim-lock token is opaque to the guard, so a worker pid that happens to read as an HTTP
status does the damage: on the base tree (measured, card t_abd07c8a) ``check_respawn_guard``
returned ``"blocker_auth"`` for ``stale_lock=deadbeef0c1a:403``, for
``stale_lock=1f862eb0d269:429`` and for the sweep line ``pid 403 not alive``. The timeout
template's ``elapsed 429s > limit 3600s`` is redacted **defensively**, not as a measured park: the
base tree already returns ``None`` for it (the template's digits are ``s``-suffixed, so the old
``\\b429\\b`` arm cannot match them).

Both directions are pinned here, each on the exact returned value:

* bookkeeping forms — alone, and behind a later ``review_requested`` handoff, on both the ``ready``
  and the ``review`` lane — return ``None``;
* provider-wall forms return ``"blocker_auth"``, including the underscored provider codes the old
  ``\\b``-bounded list silently missed (``insufficient_quota``, ``rate_limit_error``,
  ``authentication_error``) and ``401``, which matched nothing before;
* a genuine failure note that quotes a claim lock, or carries a bookkeeping prefix, still returns
  ``"blocker_auth"``: the redaction drops identifiers, never diagnosis words;
* the two clauses no other cell isolates are pinned on single inputs, so dropping either cannot
  leave the suite green: the ``elapsed Ns > limit Ns`` redaction (its digits must not survive —
  base returns ``None`` there, so only this cell fails when the clause goes) and the removal of
  bare ``auth`` (``claude auth status`` is benign prose, and it parks again the moment ``auth``
  is re-added to the pattern).
"""

from __future__ import annotations

import time
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


# Bookkeeping notes the board writes into ``last_failure_error`` itself. None of them is evidence
# of a provider wall; the first two carry a pid in a claim-lock token that reads as a status code.
BOOKKEEPING_NOTES = [
    "stale_lock=deadbeef0c1a:403",
    "stale_lock=1f862eb0d269:429",
    "stale_lock=otherhost:4242",
    "elapsed 429s > limit 3600s",
    "pid 403 not alive",
]

# Provider walls. ``429``-only and ``429 quota exceeded`` keep the pre-existing coverage
# (tests/hermes_cli/test_kanban_db.py, tests/hermes_cli/test_kanban_list_undecodable_text.py).
PROVIDER_WALLS = [
    "pid 1059980 exited rate-limited (quota wall) — requeued without counting a failure",
    "429",
    "429 quota exceeded",
    "HTTP 429 Too Many Requests",
    "API Error: 401",
    "HTTP 403 Forbidden",
    "invalid_api_key",
    "incorrect_api_key",
    "Incorrect API key provided",
    "insufficient_quota",
    "rate_limit_error",
    "authentication_error",
    "authentication error from provider",
    "auth failed",
    "auth_error",
    "unauthenticated",
    "permission denied",
    "permission_denied",
    "access denied",
    "Your subscription is past due",
    "insufficient balance",
]

# A real failure note may quote the board's own bookkeeping text. Redacting identifiers must not
# shield the diagnosis behind them — this is what a note-level prefix exemption would have broken.
BOOKKEEPING_PLUS_DIAGNOSIS = [
    "stale_lock=deadbeef0c1a:403 unauthorized",
    "pid 429 exited rate-limited (quota wall) — requeued",
    "elapsed 900s > limit 3600s; provider said HTTP 429",
]

# Benign worker prose, no failure word in it. Bare ``auth`` used to match this
# (``claude auth status`` — #117009's sibling); it is gone, and this cell pins the removal: red on
# the base tree, and red the moment ``auth`` is re-added to the pattern.
BENIGN_PROSE = [
    "claude auth status",
]


def _stamp(
    conn, note: str, *, outcome: str | None = None, ended_at: int | None = None
) -> str:
    """A ``ready`` card whose ``last_failure_error`` is ``note``, its latest run ``outcome``."""
    tid = kb.create_task(conn, title="card", assignee="worker")
    if outcome is not None:
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        conn.execute(
            "UPDATE task_runs SET outcome=?, status=?, ended_at=? WHERE id=?",
            (outcome, outcome, ended_at, run_id),
        )
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, claim_lock=NULL, "
            "claim_expires=NULL, worker_pid=NULL, last_failure_error=? WHERE id=?",
            (note, tid),
        )
    else:
        conn.execute("UPDATE tasks SET last_failure_error=? WHERE id=?", (note, tid))
    conn.commit()
    return tid


@pytest.mark.parametrize("note", BOOKKEEPING_NOTES)
def test_bookkeeping_note_is_not_an_auth_blocker(kanban_home, note):
    """A bookkeeping note alone must not park the card (pid 403 / 429 are identifiers)."""
    with kbc.connect() as conn:
        tid = _stamp(conn, note)
        assert kbd.check_respawn_guard(conn, tid) is None


@pytest.mark.parametrize("note", BOOKKEEPING_NOTES[:3])
def test_bookkeeping_note_behind_a_review_handoff_is_not_an_auth_blocker(
    kanban_home, note
):
    """The #119070 shape with a bookkeeping stamp: stamp, then a later ``review_requested`` run.

    Both lanes must yield exactly ``None`` — ``blocker_auth`` here would strand the review handoff
    (a spawned reviewer is the only thing that clears the column).
    """
    with kbc.connect() as conn:
        tid = _stamp(
            conn, note, outcome="review_requested", ended_at=int(time.time()) - 5
        )
        assert kbd.check_respawn_guard(conn, tid) is None
        assert kbd.check_respawn_guard(conn, tid, lane="review") is None


@pytest.mark.parametrize("note", BENIGN_PROSE)
def test_benign_prose_is_not_an_auth_blocker(kanban_home, note):
    """Ordinary worker prose must not park the card (bare ``auth`` no longer matches)."""
    with kbc.connect() as conn:
        tid = _stamp(conn, note)
        assert kbd.check_respawn_guard(conn, tid) is None


@pytest.mark.parametrize("note", PROVIDER_WALLS)
def test_provider_wall_still_parks_the_card(kanban_home, note):
    """Every provider-wall shape still parks: the scope fix must not become a blanket exemption."""
    with kbc.connect() as conn:
        tid = _stamp(conn, note)
        assert kbd.check_respawn_guard(conn, tid) == "blocker_auth"


@pytest.mark.parametrize("note", BOOKKEEPING_PLUS_DIAGNOSIS)
def test_bookkeeping_text_does_not_shield_a_diagnosis(kanban_home, note):
    """Redaction is token-scoped, not a note-level exemption: the failure word still parks."""
    with kbc.connect() as conn:
        tid = _stamp(conn, note)
        assert kbd.check_respawn_guard(conn, tid) == "blocker_auth"


def test_redaction_drops_identifiers_and_keeps_diagnosis_words():
    """The two halves of the mechanism, on one string each."""
    assert "unauthorized" in kbd._respawn_blocker_text(
        "stale_lock=ab12cd34ef56:403 unauthorized"
    )
    assert "403" not in kbd._respawn_blocker_text("stale_lock=ab12cd34ef56:403")
    assert "rate-limited" in kbd._respawn_blocker_text(
        "pid 429 exited rate-limited (quota wall)"
    )


def test_redaction_drops_the_timeout_templates_own_numbers():
    """The ``elapsed Ns > limit Ns`` clause, on the one input where it is the sole remover.

    Base ``check_respawn_guard`` already returns ``None`` for this note (the template's digits are
    ``s``-suffixed, so the old ``\\b429\\b`` arm cannot match them): the clause is
    defence-in-depth, and without this cell a silent drop would leave every other cell green.
    """
    text = kbd._respawn_blocker_text("elapsed 429s > limit 3600s")
    assert "429" not in text
    assert "3600" not in text
