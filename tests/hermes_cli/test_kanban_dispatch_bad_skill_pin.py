"""A card whose ``skills`` pin does not resolve in the assignee's profile must be
parked ``blocked`` at claim time, never spawned and never charged a failure strike.

Without the claim-time check the worker dies at ``--skills`` preload (or silently
runs unpinned on a partial miss) and the card loops ready -> crash -> ready.
"""
from __future__ import annotations

from pathlib import Path

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_dispatch_skillpins as pins


def _write_skill(home: Path, name: str) -> None:
    d = home / "skills" / "demo" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Use when testing {name}.\n---\n\n# {name}\n",
        encoding="utf-8",
    )


def _setup(tmp_path, monkeypatch):
    """Dispatcher home (default) plus a live ``dev`` profile with its own skills."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    dev = home / "profiles" / "dev"
    dev.mkdir(parents=True)
    (dev / "config.yaml").write_text("{}\n", encoding="utf-8")
    _write_skill(dev, "dev-skill")
    _write_skill(home, "default-only-skill")
    kb.init_db()
    return home, dev


def _spawner(spawns):
    def _spawn(task, workspace, board=None):
        spawns.append((task.id, list(task.skills or [])))
    return _spawn


def _kinds(conn, tid):
    return [e.kind for e in kb.list_events(conn, tid)]


def test_unresolvable_pin_blocks_without_spawn_or_strike(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    spawns: list = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="dev", skills=["no-such-skill"])
        res = kbd.dispatch_once(conn, spawn_fn=_spawner(spawns))
        task = kb.get_task(conn, tid)
        events = {e.kind: e.payload for e in kb.list_events(conn, tid)}
        runs = kb.list_runs(conn, tid)
        row = conn.execute("SELECT consecutive_failures, block_kind FROM tasks WHERE id=?", (tid,)).fetchone()
    assert spawns == []
    assert res.spawned == [] and res.blocked_bad_skill_pin == [(tid, ["no-such-skill"])]
    assert task.status == "blocked" and row["block_kind"] == "needs_input"
    assert row["consecutive_failures"] == 0
    assert events["bad_skill_pin"]["missing"] == ["no-such-skill"]
    assert "spawn_failed" not in events and "gave_up" not in events
    assert events["blocked"]["reason"].startswith("unknown skill(s): no-such-skill for assignee dev")
    assert [r.outcome for r in runs] == ["bad_skill_pin"]
    assert task.claim_lock is None


def test_valid_pin_spawns_with_pin_unchanged(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    spawns: list = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="dev", skills=["dev-skill"])
        res = kbd.dispatch_once(conn, spawn_fn=_spawner(spawns))
        assert kb.get_task(conn, tid).status == "running"
    assert spawns == [(tid, ["dev-skill"])]
    assert res.blocked_bad_skill_pin == []


def test_pin_is_resolved_in_assignee_home_not_dispatcher_home(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    spawns: list = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="dev", skills=["default-only-skill"])
        res = kbd.dispatch_once(conn, spawn_fn=_spawner(spawns))
    assert spawns == [] and res.blocked_bad_skill_pin == [(tid, ["default-only-skill"])]


def test_partial_miss_blocks_and_names_only_missing(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    spawns: list = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="dev", skills=["dev-skill", "typo-skill"])
        res = kbd.dispatch_once(conn, spawn_fn=_spawner(spawns))
    assert spawns == [] and res.blocked_bad_skill_pin == [(tid, ["typo-skill"])]


def test_no_pins_spawns_normally(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    spawns: list = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="dev")
        kbd.dispatch_once(conn, spawn_fn=_spawner(spawns))
    assert spawns == [(tid, [])]


def test_fixed_pin_spawns_after_unblock(tmp_path, monkeypatch):
    _, dev = _setup(tmp_path, monkeypatch)
    spawns: list = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="dev", skills=["late-skill"])
        kbd.dispatch_once(conn, spawn_fn=_spawner(spawns))
        assert kb.get_task(conn, tid).status == "blocked"
        _write_skill(dev, "late-skill")
        assert kb.unblock_task(conn, tid)
        kbd.dispatch_once(conn, spawn_fn=_spawner(spawns))
    assert spawns == [(tid, ["late-skill"])]


def test_unfixed_pin_reblocks_into_triage_instead_of_looping(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    spawns: list = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="dev", skills=["still-bad"])
        kbd.dispatch_once(conn, spawn_fn=_spawner(spawns))
        assert kb.unblock_task(conn, tid)
        kbd.dispatch_once(conn, spawn_fn=_spawner(spawns))
        task = kb.get_task(conn, tid)
        row = conn.execute("SELECT consecutive_failures FROM tasks WHERE id=?", (tid,)).fetchone()
    assert spawns == [] and task.status == "triage" and row["consecutive_failures"] == 0


def test_check_fails_open_when_lookup_errors(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    import agent.skill_commands as sc

    def _boom(*a, **k):
        raise RuntimeError("lookup exploded")

    monkeypatch.setattr(sc, "build_preloaded_skills_prompt", _boom)
    assert pins.unresolvable_skill_pins("dev", ["anything"]) == []
    spawns: list = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="dev", skills=["anything"])
        kbd.dispatch_once(conn, spawn_fn=_spawner(spawns))
    assert spawns == [(tid, ["anything"])]
