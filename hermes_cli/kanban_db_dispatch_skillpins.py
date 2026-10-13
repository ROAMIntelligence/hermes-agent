"""Claim-time check that a card's ``skills`` pin resolves in the assignee's profile.

A worker started with ``--skills X`` where ``X`` is not in the assignee's skill
library either dies at startup (every pin missing) or runs without the pin
(partial miss, logged only). The card was never run, so burning the spawn-failure
budget on it, or letting a half-pinned worker proceed, both hide a card-authoring
mistake. The dispatcher instead parks the card ``blocked`` (``needs_input``)
with a named reason and a distinct ``bad_skill_pin`` run outcome / event.

Resolution is delegated to ``build_preloaded_skills_prompt`` under the assignee's
home, i.e. the exact lookup the worker's ``--skills`` preload performs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import sqlite3

    from hermes_cli.kanban_db import Task
    from hermes_cli.kanban_db_dispatch import DispatchResult

_log = logging.getLogger(__name__)


def unresolvable_skill_pins(assignee: Optional[str], skills: Optional[list[str]]) -> list[str]:
    """Pinned skill names that *assignee*'s profile cannot load, in pin order.

    Fails OPEN (returns ``[]``) when the profile home or the skill lookup cannot be
    evaluated: an inspection bug must never stop the board, the worker's own
    preload remains the backstop.
    """
    pins = [s.strip() for s in (skills or ()) if isinstance(s, str) and s.strip()]
    if not pins or not assignee:
        return []
    try:
        from agent.skill_commands import build_preloaded_skills_prompt
        from hermes_cli.profiles import get_profile_dir
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        home = get_profile_dir(assignee)
        token = set_hermes_home_override(str(home))
        try:
            _prompt, _loaded, missing = build_preloaded_skills_prompt(pins)
        finally:
            reset_hermes_home_override(token)
    except Exception:
        _log.warning("kanban dispatcher: skill-pin check skipped for %s", assignee, exc_info=True)
        return []
    return list(missing)


def block_unresolvable_skill_pins(
    conn: sqlite3.Connection, claimed: Task, result: DispatchResult,
) -> bool:
    """Block a just-claimed card whose pins do not resolve; True when blocked.

    The failure counter is untouched (the card never ran). ``block_task`` keeps the
    unblock-loop breaker: re-blocking for the same cause after an unblock routes to
    ``triage`` instead of looping.
    """
    from hermes_cli import kanban_db as _kb

    missing = unresolvable_skill_pins(claimed.assignee, claimed.skills)
    if not missing:
        return False
    names = ", ".join(missing)
    reason = (
        f"unknown skill(s): {names} for assignee {claimed.assignee} (bad skill pin; "
        "never spawned, no failure strike). Fix the card's skills list or reassign, then unblock."
    )
    with _kb.write_txn(conn):
        _kb._append_event(
            conn, claimed.id, "bad_skill_pin",
            {"assignee": claimed.assignee, "missing": missing, "pinned": list(claimed.skills or [])},
            run_id=_kb._current_run_id(conn, claimed.id),
        )
    _kb.block_task(conn, claimed.id, reason=reason, kind="needs_input", run_outcome="bad_skill_pin")
    result.blocked_bad_skill_pin.append((claimed.id, missing))
    return True
