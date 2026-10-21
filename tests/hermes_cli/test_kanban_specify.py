"""Tests for the specifier module + `hermes kanban specify` CLI surface.

The auxiliary LLM client is mocked — these tests don't hit any network or
real provider. They exercise the prompt plumbing, response parsing, DB
writes, and CLI flag surface.
"""

from __future__ import annotations

import argparse
import json as jsonlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_specify as spec


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_aux_response(content: str):
    """Build a minimal object shaped like an OpenAI chat.completions result.

    The specifier only reads ``resp.choices[0].message.content``, so we
    avoid importing the openai SDK and build the tree with MagicMock.
    """
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _mock_client_returning(content: str):
    client = MagicMock()
    client.chat.completions.create = MagicMock(return_value=_fake_aux_response(content))
    return client


def _patch_aux_client(content: str, *, model: str = "test-model"):
    """Patch call_llm at its source module — specify_task now routes through
    it (#35566) instead of building a raw client. Returns (patcher, mock) so
    callers can still assert on the call.
    """
    mock_fn = MagicMock(return_value=_fake_aux_response(content))
    return patch("agent.auxiliary_client.call_llm", mock_fn), mock_fn


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# specify_task (module-level entry point)
# ---------------------------------------------------------------------------

def test_specify_task_happy_path(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough", triage=True)

    content = jsonlib.dumps({
        "title": "Refined rough",
        "body": "**Goal**\nA concrete goal.",
    })
    p, _ = _patch_aux_client(content)
    with p:
        outcome = spec.specify_task(tid, author="ace")

    assert outcome.ok is True
    assert outcome.task_id == tid
    assert outcome.new_title == "Refined rough"

    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    # Parent-free → recompute_ready promotes to ready.
    assert task.status == "ready"
    assert task.title == "Refined rough"
    assert "**Goal**" in (task.body or "")






# ---------------------------------------------------------------------------
# CLI wiring — argparse + _cmd_specify
# ---------------------------------------------------------------------------

def _run_cli(*argv: str) -> int:
    """Invoke the `hermes kanban …` argparse surface directly."""
    root = argparse.ArgumentParser()
    subp = root.add_subparsers(dest="cmd")
    kanban_cli.build_parser(subp)
    ns = root.parse_args(["kanban", *argv])
    return kanban_cli.kanban_command(ns)




def test_cli_specify_tenant_filter(kanban_home, capsys):
    with kbc.connect() as conn:
        outside = kb.create_task(conn, title="outside", triage=True)
        inside = kb.create_task(
            conn, title="inside", triage=True, tenant="proj-a",
        )

    content = jsonlib.dumps({"title": "spec", "body": "body"})
    p, _ = _patch_aux_client(content)
    with p:
        rc = _run_cli("specify", "--all", "--tenant", "proj-a", "--json")
    assert rc == 0
    lines = [
        jsonlib.loads(l)
        for l in capsys.readouterr().out.strip().splitlines()
        if l
    ]
    ids = {row["task_id"] for row in lines}
    assert ids == {inside}

    # The outside task stays in triage.
    with kbc.connect() as conn:
        assert kb.get_task(conn, outside).status == "triage"
        # The inside task was promoted.
        assert kb.get_task(conn, inside).status in {"todo", "ready"}




# ---------------------------------------------------------------------------
# Guardrail preservation — Stack fit line + red-team provenance survive
# ---------------------------------------------------------------------------

_FIXTURE_STACK_FIT = (
    "Stack fit: no MCP fits (no cron/config MCP exists). "
    "Skills: hermes-cron-ops. Lane: software-developer. Rejected: a plugin."
)

_FIXTURE_REDTEAM_BLOCK = (
    "<!-- redteam: v1 lens=security model=deepseek/deepseek-v4.1-flash ts=1791875274 sha=520a16a50a3d -->\n"
    "\n"
    "## Red team (preflight)\n"
    "\n"
    "VERDICT: REVISE\n"
    "WHY: no acceptance criteria anywhere.\n"
    "\n"
    "PRIORITY FIXES:\n"
    "1) Write acceptance criteria a worker can falsify.\n"
    "\n"
    "CONFIDENCE: medium\n"
)

_FIXTURE_SPEC_BODY = (
    "Phase 2 of the live-store reconciler.\n"
    "\n"
    "WHAT IT ADDS\n"
    "1. per-field merge.\n"
    "\n"
    + _FIXTURE_STACK_FIT + "\n"
    "\n"
    + _FIXTURE_REDTEAM_BLOCK
)

_REWRITE_BODY = "**Goal**\nShip the reconciler.\n\n**Approach**\n- merge per field."


def _specify_reply(body: str) -> str:
    return jsonlib.dumps({"title": "Refined reconciler", "body": body})


def test_specify_reattaches_dropped_guardrails_byte_exact(kanban_home):
    """A rewrite that drops both guardrail spans must not lose them (AC a).

    MUST FAIL on the unfixed build — this is the falsifiability anchor.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="reconciler phase 2", body=_FIXTURE_SPEC_BODY, triage=True,
        )

    p, _ = _patch_aux_client(_specify_reply(_REWRITE_BODY))
    with p:
        outcome = spec.specify_task(tid, author="ace")
    assert outcome.ok is True

    with kbc.connect() as conn:
        body = kb.get_task(conn, tid).body or ""

    assert _REWRITE_BODY in body
    assert _FIXTURE_STACK_FIT in body
    assert _FIXTURE_REDTEAM_BLOCK.rstrip("\n") in body
    assert body.count("<!-- redteam:") == 1
    assert body.rstrip().endswith(_FIXTURE_REDTEAM_BLOCK.rstrip("\n"))


def test_specify_kept_guardrails_are_not_duplicated(kanban_home):
    """A rewrite that already echoes the spans must not duplicate them."""
    keep = _REWRITE_BODY + "\n\n" + _FIXTURE_STACK_FIT + "\n\n" + _FIXTURE_REDTEAM_BLOCK
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="reconciler phase 2", body=_FIXTURE_SPEC_BODY, triage=True,
        )

    p, _ = _patch_aux_client(_specify_reply(keep))
    with p:
        assert spec.specify_task(tid, author="ace").ok is True

    with kbc.connect() as conn:
        body = kb.get_task(conn, tid).body or ""
    assert body.count(_FIXTURE_STACK_FIT) == 1
    assert body.count("<!-- redteam:") == 1


def test_specify_without_guardrails_is_byte_identical(kanban_home):
    """Cards without the spans keep the pre-fix rewrite output exactly (AC c)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough", body="rough notes only", triage=True)

    p, _ = _patch_aux_client(_specify_reply(_REWRITE_BODY))
    with p:
        assert spec.specify_task(tid, author="ace").ok is True

    with kbc.connect() as conn:
        body = kb.get_task(conn, tid).body or ""
    assert body == _REWRITE_BODY


def test_specify_title_only_reply_leaves_body_untouched(kanban_home):
    """A title-only reply does not rewrite the body — guardrails stay."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="reconciler phase 2", body=_FIXTURE_SPEC_BODY, triage=True,
        )

    p, _ = _patch_aux_client(jsonlib.dumps({"title": "Refined"}))
    with p:
        assert spec.specify_task(tid, author="ace").ok is True

    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.body == _FIXTURE_SPEC_BODY


def test_guardrail_segments_matches_bullet_and_bold_forms():
    """The extractor catches the gate's bullet/bold-wrapped form + the tail."""
    body = (
        "intro\n"
        "\n"
        "- **Stack fit:** bulleted form.\n"
        "\n"
        "<!-- redteam: v1 lens=x -->\n"
        "\n"
        "## Red team (preflight)\n"
        "\n"
        "VERDICT: OK\n"
    )
    segs = spec._guardrail_segments(body)
    assert segs[0] == "- **Stack fit:** bulleted form."
    assert segs[-1].startswith("<!-- redteam: v1 lens=x -->")
    assert segs[-1].rstrip().endswith("VERDICT: OK")
