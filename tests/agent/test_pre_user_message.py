"""``pre_user_message`` hook (agent/pre_user_message.py): a plugin may rewrite a person's typed prompt.

Driven through the real ``build_turn_context`` prologue and the real plugin registry, so what is
asserted is what the model is sent (``ctx.messages`` / ``ctx.user_message``) and what the transcript
stores (the staged row), not just the helper's return value.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from hermes_cli.plugins import VALID_HOOKS, get_plugin_manager

from tests.agent.test_turn_context import _FakeAgent, _build, _stub_runtime_main


@pytest.fixture
def hook(monkeypatch):
    """Register callbacks on the real registry; yields (register, calls)."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    calls: list[dict] = []

    def register(fn):
        def _cb(**kw):
            calls.append(kw)
            return fn(kw)
        mgr._hooks.setdefault("pre_user_message", []).append(_cb)

    try:
        yield register, calls
    finally:
        mgr._hooks = saved


def _rewrite(text):
    return lambda kw: {"action": "rewrite", "text": text}


def test_hook_is_a_valid_plugin_hook():
    assert "pre_user_message" in VALID_HOOKS


def test_rewrite_replaces_what_the_model_is_sent_and_the_staged_row(hook):
    register, calls = hook
    register(_rewrite("tight prompt"))
    ctx = _build(_FakeAgent(), user_message="a very rambling prompt")
    assert ctx.user_message == "tight prompt"
    assert ctx.messages[-1]["content"] == "tight prompt"
    assert calls[0]["message"] == "a very rambling prompt"
    assert calls[0]["platform"] == "cli" and calls[0]["session_id"] == "sess-1"
    assert calls[0]["model"] == "test/model" and calls[0]["is_first_turn"] is True


@pytest.mark.parametrize("result", [
    None, "allow", {"action": "allow"}, {"action": "skip"}, {"action": "rewrite"},
    {"action": "rewrite", "text": ""}, {"action": "rewrite", "text": "   "},
    {"action": "rewrite", "text": 7}, {"action": "rewrite", "text": "hello"},  # unchanged
])
def test_anything_but_a_real_rewrite_leaves_the_prompt(hook, result):
    register, _ = hook
    register(lambda kw: result)
    ctx = _build(_FakeAgent(), user_message="hello")
    assert ctx.messages[-1]["content"] == "hello"


def test_raising_callback_fails_open(hook):
    register, _ = hook

    def boom(kw):
        raise RuntimeError("plugin bug")
    register(boom)
    assert _build(_FakeAgent(), user_message="hello").messages[-1]["content"] == "hello"


def test_hung_callback_is_bounded_and_fails_open(hook, monkeypatch):
    register, _ = hook
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.3)
    release = threading.Event()
    register(lambda kw: release.wait(10))
    start = time.monotonic()
    try:
        ctx = _build(_FakeAgent(), user_message="hello")
    finally:
        release.set()
    assert time.monotonic() - start < 8
    assert ctx.messages[-1]["content"] == "hello"


def test_first_valid_rewrite_wins(hook):
    register, _ = hook
    register(lambda kw: None)
    register(_rewrite("first"))
    register(_rewrite("second"))
    assert _build(_FakeAgent(), user_message="x y z").messages[-1]["content"] == "first"


@pytest.mark.parametrize("platform", ["cli", "tui", "desktop", "acp"])
def test_fires_on_attended_surfaces(hook, platform):
    register, calls = hook
    register(_rewrite("rewritten"))
    agent = _FakeAgent()
    agent.platform = platform
    assert _build(agent, user_message="typed").messages[-1]["content"] == "rewritten"
    assert calls[0]["platform"] == platform


@pytest.mark.parametrize("platform", ["cron", "subagent", "telegram", "api_server", "discord", ""])
def test_never_fires_on_automation_or_messaging(hook, platform):
    register, calls = hook
    register(_rewrite("rewritten"))
    agent = _FakeAgent()
    agent.platform = platform
    assert _build(agent, user_message="typed").messages[-1]["content"] == "typed"
    assert calls == []


def test_kanban_worker_on_cli_platform_is_not_rewritten(hook, monkeypatch):
    """A kanban worker's agent.platform is "cli"; only its session SOURCE says it is automation."""
    register, calls = hook
    register(_rewrite("rewritten"))
    monkeypatch.setattr("agent.session_source.session_source_for", lambda platform: "kanban")
    assert _build(_FakeAgent(), user_message="work kanban task t_1").messages[-1]["content"] == "work kanban task t_1"
    assert calls == []


def test_synthesized_and_fork_turns_are_skipped(hook):
    register, calls = hook
    register(_rewrite("rewritten"))
    assert _build(_FakeAgent(), user_message="notice", persist_user_display_kind="hidden"
                  ).messages[-1]["content"] == "notice"
    forked = _FakeAgent()
    forked._turn_origin = "background_review"
    assert _build(forked, user_message="review").messages[-1]["content"] == "review"
    assert calls == []


def test_multimodal_prompt_is_left_alone(hook):
    register, calls = hook
    register(_rewrite("rewritten"))
    parts = [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:x"}}]
    ctx = _build(_FakeAgent(), user_message=parts, persist_user_message=parts,
                 summarize_user_message_for_log=str)
    assert ctx.messages[-1]["content"] == parts
    assert calls == []


def test_clean_text_is_rewritten_inside_a_prepared_prompt_and_persisted_clean(hook):
    """@-expansion / voice prefix: model sees the wrapper around the REWRITTEN text; transcript stores it clean."""
    register, calls = hook
    register(_rewrite("short"))
    agent = _FakeAgent()
    ctx = _build(agent, user_message="[NOTE]\n\nlong typed text", persist_user_message="long typed text")
    assert calls[0]["message"] == "long typed text"
    assert ctx.messages[-1]["content"] == "[NOTE]\n\nshort"
    assert agent._persist_user_message_override == "short"


def test_rewrite_skipped_when_typed_text_is_not_in_the_prepared_prompt(hook):
    register, _ = hook
    register(_rewrite("short"))
    ctx = _build(_FakeAgent(), user_message="totally different expansion", persist_user_message="long typed text")
    assert ctx.messages[-1]["content"] == "totally different expansion"


def test_staged_cli_row_is_adopted_not_duplicated(hook):
    """CLI stages the raw text; after a rewrite the turn must still reuse that dict (no second user row)."""
    register, _ = hook
    register(_rewrite("tight"))
    agent = _FakeAgent()
    staged = {"role": "user", "content": "rambling"}
    agent._pending_cli_user_message = staged
    ctx = _build(agent, user_message="rambling")
    assert ctx.messages[-1] is staged
    assert staged["content"] == "tight"
    assert sum(1 for m in ctx.messages if m.get("role") == "user") == 1


def test_desktop_submit_row_is_rewritten_in_the_db_under_its_owner_key(hook):
    """Desktop writes the DB row at submit (raw text); the rewrite must update THAT row, by id and owner."""
    register, _ = hook
    register(_rewrite("tight"))
    agent = _FakeAgent()
    agent._session_db = MagicMock()
    agent._session_db.set_user_message_content.return_value = 1
    staged = {"role": "user", "content": "rambling", "_db_persisted": True, "_row_id": 42,
              "_submit_row_session_id": "rotated-owner"}
    agent._pending_cli_user_message = staged
    ctx = _build(agent, user_message="rambling", persist_user_message="rambling")
    agent._session_db.set_user_message_content.assert_called_once_with("rotated-owner", 42, "tight")
    assert ctx.messages[-1] is staged and staged["content"] == "tight"


def test_rewrite_dropped_when_the_durable_row_cannot_be_updated(hook):
    """Model input and transcript must never disagree: a failed row update drops the rewrite."""
    register, _ = hook
    register(_rewrite("tight"))
    for db_result in (0, RuntimeError("db locked")):
        agent = _FakeAgent()
        agent._session_db = MagicMock()
        if isinstance(db_result, Exception):
            agent._session_db.set_user_message_content.side_effect = db_result
        else:
            agent._session_db.set_user_message_content.return_value = db_result
        staged = {"role": "user", "content": "rambling", "_db_persisted": True, "_row_id": 42}
        agent._pending_cli_user_message = staged
        ctx = _build(agent, user_message="rambling")
        assert ctx.messages[-1]["content"] == "rambling" and staged["content"] == "rambling"


def test_durable_row_without_an_id_drops_the_rewrite(hook):
    register, _ = hook
    register(_rewrite("tight"))
    agent = _FakeAgent()
    agent._pending_cli_user_message = {"role": "user", "content": "rambling", "_db_persisted": True}
    assert _build(agent, user_message="rambling").messages[-1]["content"] == "rambling"


def test_memory_prefetch_and_pre_llm_call_see_the_rewritten_prompt(hook):
    register, _ = hook
    register(_rewrite("tight prompt text"))
    seen: list[str] = []
    mgr = get_plugin_manager()
    mgr._hooks.setdefault("pre_llm_call", []).append(lambda **kw: seen.append(kw["user_message"]))
    ctx = _build(_FakeAgent(), user_message="a very rambling prompt")
    assert seen == ["tight prompt text"] and ctx.user_message == "tight prompt text"
