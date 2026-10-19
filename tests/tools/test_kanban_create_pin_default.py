"""kanban_create materializes the model pin (t_e1b3bd2b).

A card created without explicit ``model``/``provider`` must not land NULL-pinned: each
field resolves explicit arg -> creating card's pin -> the profile's config default.
"""
from __future__ import annotations

import json

import pytest

DEFAULT = ("deepseek/deepseek-v4.1-flash", "nous")
OTHER = ("acme/runner-2", "acme")
THIRD = ("quux/model-7", "quux")


def _make_home(tmp_path, name, pin):
    """Isolated HERMES_HOME with a config.yaml whose default model is ``pin``."""
    home = tmp_path / name
    home.mkdir()
    (home / "config.yaml").write_text(
        f"model:\n  default: {pin[0]}\n  provider: {pin[1]}\n", encoding="utf-8")
    return home


def _activate(monkeypatch, tmp_path, home):
    monkeypatch.setenv("HERMES_HOME", str(home))
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()


def _make_worker(monkeypatch, tmp_path, pin, name="home"):
    home = _make_home(tmp_path, name, pin)
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    _activate(monkeypatch, tmp_path, home)
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="worker-test", assignee="test-worker")
        kb.claim_task(conn, tid)
        run_id = kb._current_run_id(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return tid


@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    """Worker whose (unpinned) card lives in a home with config default OTHER."""
    return _make_worker(monkeypatch, tmp_path, OTHER)


def _pin_parent(parent, pin=DEFAULT):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        kb.set_model_override(conn, parent, *pin)
    finally:
        conn.close()


def _create(**args):
    from tools import kanban_tools as kt
    d = json.loads(kt._handle_create({"title": "child", "assignee": "someone", **args}))
    assert d.get("ok") is True, d
    return d["task_id"]


def _pin_of(tid):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        t = kb.get_task(conn, tid)
        return t.model_override, t.provider_override
    finally:
        conn.close()


def test_no_parent_gets_config_default(monkeypatch, tmp_path):
    _make_worker(monkeypatch, tmp_path, DEFAULT)
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    assert _pin_of(_create()) == DEFAULT


def test_pinned_parent_is_inherited_over_config(worker_env):
    _pin_parent(worker_env, DEFAULT)
    assert _pin_of(_create()) == DEFAULT


def test_created_event_carries_inherited_pin(worker_env):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    _pin_parent(worker_env, DEFAULT)
    tid = _create()
    conn = kbc.connect()
    try:
        created = [e for e in kb.list_events(conn, tid) if e.kind == "created"]
    finally:
        conn.close()
    print("created payload:", json.dumps(created[0].payload, sort_keys=True))
    assert len(created) == 1
    assert (created[0].payload["model_override"],
            created[0].payload["provider_override"]) == DEFAULT


def test_unpinned_parent_falls_through_to_config(worker_env):
    assert _pin_of(_create()) == OTHER


def test_explicit_pair_wins(worker_env):
    _pin_parent(worker_env, DEFAULT)
    assert _pin_of(_create(model="x/explicit", provider="xp")) == ("x/explicit", "xp")


def test_explicit_model_only_takes_parent_provider(worker_env):
    _pin_parent(worker_env, DEFAULT)
    assert _pin_of(_create(model="x/explicit")) == ("x/explicit", DEFAULT[1])


def test_explicit_provider_only_resolves_model(worker_env):
    _pin_parent(worker_env, DEFAULT)
    assert _pin_of(_create(provider="xp")) == (DEFAULT[0], "xp")


def test_explicit_provider_only_resolves_model_from_config(worker_env):
    assert _pin_of(_create(provider="xp")) == (OTHER[0], "xp")


def test_resolves_config_not_a_literal(monkeypatch, tmp_path):
    _make_worker(monkeypatch, tmp_path, THIRD)
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    assert _pin_of(_create()) == THIRD


def test_default_is_read_per_call_across_homes(monkeypatch, tmp_path):
    home_a = _make_home(tmp_path, "a", DEFAULT)
    home_b = _make_home(tmp_path, "b", OTHER)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    got = []
    for home in (home_a, home_b, home_a):
        _activate(monkeypatch, tmp_path, home)
        got.append(_pin_of(_create()))
    assert got == [DEFAULT, OTHER, DEFAULT]


def test_provider_without_any_model_is_refused(monkeypatch, tmp_path):
    from tools import kanban_tools as kt
    home = tmp_path / "bare"
    home.mkdir()
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    _activate(monkeypatch, tmp_path, home)
    out = json.loads(kt._handle_create(
        {"title": "c", "assignee": "someone", "provider": "xp"}))
    assert "error" in out
