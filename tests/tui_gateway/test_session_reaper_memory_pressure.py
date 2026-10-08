"""The reaper's opt-in memory-pressure shed, on its own.

``memory_pressure_evict_mb`` bounds DETACHED session accumulation: above the
budget the idle-reaper scan (no new timer) sheds ONE least-recently-active
session that passes the same hard-exemption predicate as the LRU cap. The stamp
it writes is an automatic reclaim, not a deliberate close — a client still
holding the session is told via ``session.reclaimed``.

Lives outside ``test_tui_gateway_server.py`` on purpose: that module is far past
the FILE_LINES target, so new tests go in a new ``test_<stem>_<topic>.py``.
"""

from __future__ import annotations

import threading
import time

import pytest

from tui_gateway import server


def _idle_evictable_session(now):
    """A session that satisfies every eviction precondition."""
    ready = threading.Event()
    ready.set()
    old = now - 10 * 3600  # well past the 6h TTL
    return {
        "running": False,
        "agent_ready": ready,
        "transport": server._detached_ws_transport,  # dead/detached
        "last_active": old,
        "created_at": old,
    }


def _memory_pressure_stubs(monkeypatch, *, anon_mb, closed):
    """Stub the reaper's neighbours and the anon-RSS read so only the pressure path can act."""
    monkeypatch.setattr(server, "_session_pending_kind", lambda sid: "")
    monkeypatch.setattr(server, "_close_session_by_id",
        lambda sid, *, end_reason, predicate=None: closed.append((sid, end_reason)))
    monkeypatch.setattr(server, "_flush_dirty_sessions", lambda *a, **k: 0)
    monkeypatch.setattr(server, "_repair_missing_ws_orphan_reaps", lambda: None)
    monkeypatch.setattr(server, "_enforce_session_cap", lambda: None)
    monkeypatch.setattr(server, "_reclaim_orphaned_leases", lambda: None)
    monkeypatch.setattr(server, "_sessions_quiescent", lambda exclude=None: False)
    import gateway.agent_cache_pressure as acp
    monkeypatch.setattr(acp, "read_anon_rss_mb", lambda: anon_mb)


def _detached_idle(now, idle_s=3600):
    """LRU-eligible (no live client) but NOT past the TTL, so only the pressure path can pick it."""
    return _idle_evictable_session(now) | {"last_active": now - idle_s, "created_at": now - idle_s}


def test_memory_pressure_evicts_lru_detached_session(monkeypatch):
    closed = []
    _memory_pressure_stubs(monkeypatch, anon_mb=900, closed=closed)
    monkeypatch.setattr(server, "_memory_pressure_evict_mb", lambda: 700)
    now = time.time()
    server._sessions.clear()
    server._sessions["old"] = _detached_idle(now, idle_s=4000)
    server._sessions["new"] = _detached_idle(now, idle_s=3600)
    try:
        server._reap_idle_sessions()
        assert closed == [("old", "memory_pressure")]
        # The stamp the reaper emits must be registered in every taxonomy that
        # keys off the automatic-vs-deliberate class: the desktop close path's
        # automatic-cleanup set, the client-reclaim announce set, and the state
        # layer's stamp predicate. Missing from these, a shed session looked
        # like a deliberate close and its clients were never told.
        from hermes_state_common import is_automatic_end_reason

        reason = closed[0][1]
        assert is_automatic_end_reason(reason)
        assert reason in server._AUTOMATIC_SESSION_END_REASONS
        assert reason in server._RECLAIM_END_REASONS
    finally:
        server._sessions.clear()


def test_memory_pressure_pass_noop_when_budget_unset(monkeypatch):
    closed = []
    _memory_pressure_stubs(monkeypatch, anon_mb=9000, closed=closed)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})  # shipped default: key absent -> 0
    now = time.time()
    server._sessions.clear()
    server._sessions["old"] = _detached_idle(now, idle_s=4000)
    server._sessions["newer"] = _detached_idle(now, idle_s=3600)
    try:
        server._reap_idle_sessions()
        assert closed == []
    finally:
        server._sessions.clear()


def test_memory_pressure_spares_live_transport_session(monkeypatch):
    closed = []
    _memory_pressure_stubs(monkeypatch, anon_mb=900, closed=closed)
    monkeypatch.setattr(server, "_memory_pressure_evict_mb", lambda: 700)
    now = time.time()
    live = type("T", (), {"_closed": False})()
    server._sessions.clear()
    server._sessions["attached"] = _detached_idle(now, idle_s=4000) | {"transport": live}
    try:
        server._reap_idle_sessions()
        assert closed == []
    finally:
        server._sessions.clear()


def test_memory_pressure_budget_reads_config_key(monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"memory_pressure_evict_mb": 512})
    assert server._memory_pressure_evict_mb() == 512
    monkeypatch.setattr(server, "_load_cfg",
                        lambda: {"gateway": {"memory_pressure_evict_mb": "256"}})
    assert server._memory_pressure_evict_mb() == 256
    monkeypatch.setattr(server, "_load_cfg", lambda: {"memory_pressure_evict_mb": 0})
    assert server._memory_pressure_evict_mb() == 0
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    assert server._memory_pressure_evict_mb() == 0
