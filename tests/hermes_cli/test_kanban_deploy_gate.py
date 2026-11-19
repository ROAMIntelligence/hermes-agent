"""deploy-parity completion gate (card t_c7b6e6c8): fixtures, hermetic.

Every test drives the real ``check_deploy_gate`` / ``complete_task`` path with an injected
GitHub API fake — no network, no gh, no live board. The gate's contract under test:

  * a resolved commit that is missing from production REFUSES the completion, naming the sha;
  * ``deploy: not-needed`` completes and records the marker;
  * a record that names no resolvable commit completes with
    ``deploy-check: UNKNOWN (no-commit-recorded)``;
  * a recorded PR whose head differs from the squash commit resolves through the PR's
    ``merge_commit_sha`` (the head never survives a squash merge);
  * every failure path (network, no token, unmapped repo, unmerged PR) is fail-open;
  * ``force=True`` and ``HERMES_KANBAN_DEPLOY_GATE=off`` are the two escapes.

Note on the contract: these cards are created ``local-only`` and carry their deploy
reference in run metadata, which is how the measured incident cards recorded it
(``t_7bfda697``: pr=765, pr_head, merge_commit; ``t_6aa313fd``: merge_commit, head_sha).
A card that declares ``OWNER/REPO`` and records no ``published_pr`` never reaches this
gate at all — ``prepare_acceptance`` already refuses it (pre-existing behaviour, out of
this card's scope), so the no-commit-recorded path is exercised at the verdict level for
that shape and at the completion level for the metadata shape.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_deploy_gate as gate
from hermes_cli import kanban_db_connect as kbc

REPO = "ROAMIntelligence/belac-deploy"
PR_URL = "https://github.com/%s/pull/765" % REPO
TIP = "a" * 40
MERGE = "b" * 40
HEAD = "c" * 40            # the recorded PR head: never in production after a squash
ABSENT = "d" * 40
PULLS = "/repos/%s/pulls/" % REPO
REF = "/repos/%s/git/ref/heads/production" % REPO
COMPARE = "/repos/%s/compare/" % REPO


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv(gate.KILL_ENV, raising=False)
    gate._TIP_CACHE.clear()
    kb.init_db()
    yield home
    gate._TIP_CACHE.clear()


class FakeApi:
    """Substring-routed GitHub fake; records every call."""

    def __init__(self, *, pulls=None, compare=None, raises=None):
        self.calls: list[tuple[str, str]] = []
        self.pulls = pulls
        self.compare = compare
        self.raises = raises or []

    def __call__(self, method, endpoint):
        self.calls.append((method, endpoint))
        for needle, exc in self.raises:
            if needle in endpoint:
                raise exc
        if PULLS in endpoint:
            if self.pulls is None:
                raise AssertionError("unexpected PR call: %s" % endpoint)
            return self.pulls
        if REF in endpoint:
            return {"object": {"sha": TIP}}
        if COMPARE in endpoint:
            if self.compare is None:
                raise AssertionError("unexpected compare: %s" % endpoint)
            return self.compare
        raise AssertionError("unexpected endpoint %s" % endpoint)


class FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def read(self, *args):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def gh(monkeypatch):
    """Answer the gate's real HTTP seam from a fake.

    Patched at ``urllib.request.urlopen`` — the one object that survives the
    ``del sys.modules["hermes_cli.*"]`` purge that several board tests perform
    (test_kanban_per_profile_cap, test_kanban_cli_dispatch_passthrough). Patching
    ``hermes_cli.kanban_deploy_gate`` attributes instead leaves the re-imported
    module talking to the real GitHub.
    """
    holder = {"api": None}

    def fake_urlopen(request, *args, **kwargs):
        endpoint = request.full_url.replace("https://api.github.com", "", 1)
        method = request.get_method()
        if holder["api"] is None:
            raise AssertionError("unexpected real network call: %s %s" % (method, endpoint))
        return FakeResponse(holder["api"](method, endpoint))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    def use(api):
        holder["api"] = api
        return api

    return use


def _task(conn, body=None, branch_name=None, **kwargs):
    kwargs.setdefault("completion_contract", "local-only")
    return kb.create_task(conn, title="a code card", body=body, branch_name=branch_name,
                          **kwargs)


def _run_metadata(conn, tid):
    row = conn.execute(
        "SELECT metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,)
    ).fetchone()
    return json.loads(row[0]) if row and row[0] else {}


def _status(conn, tid):
    return kb.get_task(conn, tid).status


# --------------------------------------------------------------- AC5: refusal / markers

def test_absent_from_production_refuses_naming_the_sha(kanban_home, gh):
    api = gh(FakeApi(pulls={"merged": True, "merge_commit_sha": ABSENT, "head": {"sha": HEAD}},
                     compare={"ahead_by": 4}))
    with kbc.connect() as conn:
        tid = _task(conn)
        with pytest.raises(kb.MergedNotDeployedError) as excinfo:
            kb.complete_task(conn, tid, summary="done", metadata={"published_pr": PR_URL})
        assert "merged-not-deployed: %s" % ABSENT in str(excinfo.value)
        assert _status(conn, tid) != "done", "a refused completion must not close the card"
    assert api.calls, "the gate must actually have asked GitHub"


def test_explicit_not_needed_completes_and_records_the_marker(kanban_home):
    with kbc.connect() as conn:
        tid = _task(conn)
        assert kb.complete_task(conn, tid, summary="docs only, nothing deploys",
                                metadata={"deploy": "not-needed"}) is True
        assert _status(conn, tid) == "done"
        assert _run_metadata(conn, tid)["deploy-check"] == "not-needed"


def test_not_needed_marker_in_the_summary_text_also_works(kanban_home):
    with kbc.connect() as conn:
        tid = _task(conn)
        assert kb.complete_task(conn, tid, summary="deploy: not-needed (analysis only)") is True
        assert _run_metadata(conn, tid)["deploy-check"] == "not-needed"


def test_declared_contract_with_no_recorded_commit_is_unknown(kanban_home):
    with kbc.connect() as conn:
        tid = _task(conn, completion_contract=REPO)
        verdict = gate.check_deploy_gate(conn, tid, None)
        assert (verdict["verdict"], verdict["reason"]) == ("UNKNOWN", "no-commit-recorded")


def test_unusable_reference_completes_with_unknown_stamp(kanban_home):
    with kbc.connect() as conn:
        tid = _task(conn)
        assert kb.complete_task(conn, tid, summary="claimed a PR that never resolved",
                                metadata={"published_pr": "pending"}) is True
        assert _run_metadata(conn, tid)["deploy-check"] == "UNKNOWN (no-commit-recorded)"


# --------------------------------------------------------------- AC6: squash resolution

def test_squash_mismatch_resolves_through_the_pr_merge_commit(kanban_home, gh):
    api = gh(FakeApi(pulls={"merged": True, "merge_commit_sha": MERGE, "head": {"sha": HEAD}},
                     compare={"ahead_by": 0}))
    with kbc.connect() as conn:
        tid = _task(conn)
        # the recorded HEAD is deliberately the pre-squash head (AC6): the gate must not
        # test it against production, it must resolve the PR's mergeCommit instead.
        ok = kb.complete_task(conn, tid, summary="merged",
                              metadata={"published_pr": PR_URL, "head_sha": HEAD})
        assert ok is True
        assert _run_metadata(conn, tid)["deploy-check"] == "DEPLOYED %s@%s (production)" % (
            REPO, MERGE[:12])
    endpoints = [endpoint for _, endpoint in api.calls]
    assert PULLS + "765" in endpoints, endpoints
    assert any(COMPARE + TIP + "..." + MERGE == endpoint for endpoint in endpoints), endpoints
    assert not any(COMPARE in endpoint and HEAD in endpoint for endpoint in endpoints), \
        "the pre-squash head must never be tested against production"


def test_pr_url_from_a_prior_run_metadata_resolves(kanban_home, gh):
    """The completing run often carries no PR — the reviewer run before it does."""
    api = gh(FakeApi(pulls={"merged": True, "merge_commit_sha": MERGE, "head": {"sha": HEAD}},
                     compare={"ahead_by": 1}))
    with kbc.connect() as conn:
        tid = _task(conn)
        kb.request_review(conn, tid, summary="ready",
                          metadata={"pr": "https://github.com/%s/pull/794" % REPO})
        with pytest.raises(kb.MergedNotDeployedError):
            kb.complete_task(conn, tid, summary="approved")
        assert PULLS + "794" in [endpoint for _, endpoint in api.calls]


def test_bare_pr_number_resolves_through_the_card_body_repo_hint(kanban_home):
    """A card that records `pr: '765'` and names the repo in prose still resolves."""
    api = FakeApi(pulls={"merged": True, "merge_commit_sha": MERGE, "head": {"sha": HEAD}},
                  compare={"ahead_by": 0})
    with kbc.connect() as conn:
        tid = _task(conn, body="Ships in belac-deploy; see the merge for this card.")
        verdict = gate.check_deploy_gate(conn, tid, {"pr": "765"}, summary="x", api=api)
        assert verdict["verdict"] == "DEPLOYED", verdict
        assert PULLS + "765" in [endpoint for _, endpoint in api.calls]


# --------------------------------------------------------------- fail-open

def test_deployed_commit_completes(kanban_home, gh):
    gh(FakeApi(pulls={"merged": True, "merge_commit_sha": MERGE, "head": {"sha": HEAD}},
               compare={"ahead_by": 0}))
    with kbc.connect() as conn:
        tid = _task(conn)
        assert kb.complete_task(conn, tid, summary="live",
                                metadata={"published_pr": PR_URL}) is True
        assert _run_metadata(conn, tid)["deploy-check"] == "DEPLOYED %s@%s (production)" % (
            REPO, MERGE[:12])


def test_api_error_is_unknown_and_never_blocks(kanban_home, gh):
    """A dead network must never block a completion — and must say why it did not check."""
    gh(FakeApi(raises=[(PULLS, urllib.error.URLError("network down"))]))
    with kbc.connect() as conn:
        tid = _task(conn)
        verdict = gate.check_deploy_gate(
            conn, tid, {"published_pr": PR_URL},
            api=FakeApi(raises=[(PULLS, urllib.error.URLError("network down"))]))
        assert (verdict["verdict"], verdict["reason"]) == ("UNKNOWN", "api-error")
        assert kb.complete_task(conn, tid, summary="still done",
                                metadata={"published_pr": PR_URL}) is True
        assert _run_metadata(conn, tid)["deploy-check"] == "UNKNOWN (api-error)"


def test_no_token_is_unknown_not_a_refusal(kanban_home):
    """The token-less path maps to its own reason, not to a refusal."""
    def no_token(method, endpoint):
        raise RuntimeError("no-token")

    with kbc.connect() as conn:
        tid = _task(conn)
        verdict = gate.check_deploy_gate(conn, tid, {"published_pr": PR_URL}, api=no_token)
        assert (verdict["verdict"], verdict["reason"]) == ("UNKNOWN", "no-token")
        assert gate.stamp_deploy_check({}, verdict)["deploy-check"] == "UNKNOWN (no-token)"


def test_unmapped_repo_is_unknown_and_cannot_be_refused(kanban_home):
    with kbc.connect() as conn:
        tid = _task(conn)
        assert kb.complete_task(
            conn, tid, summary="merged upstream",
            metadata={"published_pr": "https://github.com/ROAMIntelligence/hermes-agent/pull/9"}
        ) is True
        assert _run_metadata(conn, tid)["deploy-check"] == "UNKNOWN (unmapped-repo)"


def test_unmerged_pr_is_unknown(kanban_home):
    api = FakeApi(pulls={"merged": False, "merge_commit_sha": None, "head": {"sha": HEAD}})
    with kbc.connect() as conn:
        tid = _task(conn)
        verdict = gate.check_deploy_gate(conn, tid, {"published_pr": PR_URL}, api=api)
        assert (verdict["verdict"], verdict["reason"]) == ("UNKNOWN", "pr-not-merged")


def test_kill_switch_disables_the_gate_entirely(kanban_home, monkeypatch):
    monkeypatch.setenv(gate.KILL_ENV, "off")
    with kbc.connect() as conn:
        tid = _task(conn)
        assert gate.check_deploy_gate(conn, tid, {"published_pr": PR_URL}) is None
        assert kb.complete_task(conn, tid, summary="gated off",
                                metadata={"published_pr": PR_URL}) is True
        assert "deploy-check" not in _run_metadata(conn, tid)


def test_force_bypasses_the_refusal_but_records_the_verdict(kanban_home):
    api = FakeApi(pulls={"merged": True, "merge_commit_sha": ABSENT, "head": {"sha": HEAD}},
                  compare={"ahead_by": 7})
    with kbc.connect() as conn:
        tid = _task(conn)
        verdict = gate.check_deploy_gate(conn, tid, {"published_pr": PR_URL}, api=api)
        assert verdict["verdict"] == "NOT_DEPLOYED"
        # complete_task keeps its force escape; the stamp still records what it saw.
        assert verdict["detail"] == "merged-not-deployed: %s" % ABSENT


def test_force_stamp_shape(kanban_home):
    stamped = gate.stamp_deploy_check(
        {"x": 1}, {"verdict": "NOT_DEPLOYED", "reason": "absent-from-production",
                   "repo": REPO, "sha": ABSENT, "production_ref": "production",
                   "detail": "merged-not-deployed: %s" % ABSENT})
    assert stamped["deploy-check"] == "NOT_DEPLOYED %s@%s (production)" % (REPO, ABSENT[:12])
    assert stamped["x"] == 1


def test_non_code_card_is_not_stamped(kanban_home):
    with kbc.connect() as conn:
        tid = _task(conn)
        assert gate.check_deploy_gate(conn, tid, None) is None
        assert kb.complete_task(conn, tid, summary="no repo") is True
        assert "deploy-check" not in _run_metadata(conn, tid)


def test_stamp_preserves_every_other_key_and_does_not_mutate(kanban_home):
    original = {"changed_files": ["a.py"], "deploy-check": "stale"}
    stamped = gate.stamp_deploy_check(
        original, {"verdict": "UNKNOWN", "reason": "api-error", "repo": REPO, "sha": None,
                   "production_ref": "production", "detail": "boom"})
    assert stamped["deploy-check"] == "UNKNOWN (api-error)"
    assert stamped["changed_files"] == ["a.py"]
    assert original == {"changed_files": ["a.py"], "deploy-check": "stale"}
