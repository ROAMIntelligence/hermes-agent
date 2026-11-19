"""deploy-parity completion gate (card t_c7b6e6c8)

A kanban card may not close while the work it names is absent from its repo's
production ref. 2026-11-18: two cards closed `done` while their bytes were not running
(`t_b077f000` merged 06:08Z, `t_7bfda697` merged 16:52Z; `KANBAN_ATTR_V2` absent from
`hermes_cli` at 18:00Z), and the publish train only fired at 20:17Z through
`overdue-force` — 13 commits late. Nothing on the board noticed.

FAIL-OPEN, BY DESIGN. The gate refuses only on positive evidence that a *resolved*
commit is missing from production. Everything it cannot resolve or read — no repo, an
unmapped repo, no recorded PR/commit, no token, a network error, a PR that is not
merged — completes the card and records `deploy-check: UNKNOWN (<reason>)` in the
completion metadata. A broken check must never block work; a silent wrong answer must
never look like proof.

THE RECORD IS PRE-SQUASH (measurement, not assumption). Card metadata records the PR
*head* (`head_sha`, `reviewed_head`, `pr_head`), and a squash merge never contains that
head: 4 of 4 recent code cards measured `ahead_by > 0` for their recorded head. The
record does carry the PR, and the PR carries `merge_commit_sha` — so the gate resolves
PR -> merge commit and tests *that* against production. The recorded head is never used
for the ancestry test (AC6). A record that carries ONLY a head has nothing testable at
all: it returns `UNKNOWN (head-pre-squash)` — never a refusal, because a squashed head is
absent from production by construction and the retry advice would be unsatisfiable.

Blast-radius fence: only repos whose production ref has been verified appear in
``DEFAULT_PRODUCTION_REFS``. Any other repo (including this repo) is UNKNOWN and cannot
be refused. ``HERMES_KANBAN_DEPLOY_GATE=off`` disables the gate entirely — no stamp, no
refusal — without a redeploy.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request

log = logging.getLogger(__name__)

DEFAULT_PRODUCTION_REFS = {
    "ROAMIntelligence/belac-deploy": "production",
    "ROAMIntelligence/roam-recon-frontend": "main",
    "ROAMIntelligence/ROAM-Intelligence-v2": "main",
}
KILL_ENV = "HERMES_KANBAN_DEPLOY_GATE"
NOT_NEEDED_MARKER = "deploy: not-needed"
TIP_TTL_SECONDS = 120
API_TIMEOUT_SECONDS = 6
PRIOR_RUNS_SCANNED = 5

PR_URL_RE = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/pull/([1-9][0-9]*)")
REPO_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
SHA_RE = re.compile(r"[0-9a-f]{40}")

PR_KEYS = ("published_pr", "pr", "pr_url", "pull_request", "prs")
COMMIT_KEYS = ("merge_commit", "merge_commit_sha", "squash_commit", "published_commit",
               "commit")
# Evidence of a pre-squash PR head. NEVER an ancestry-test candidate: a squash merge does
# not contain the head, so testing it can only produce a false `merged-not-deployed`
# (review round 1, P1: PR #808's head cb336660 was refused while its merge 1c076097 WAS
# production). A head-only record resolves to UNKNOWN (head-pre-squash) instead.
HEAD_KEYS = ("head_sha", "reviewed_head", "pr_head")

_TIP_CACHE: dict[tuple[str, str], tuple[float, str]] = {}


# --------------------------------------------------------------------------- helpers

def _verdict(kind: str, reason: str, *, repo=None, sha=None, production_ref=None,
             detail: str = "") -> dict:
    return {"verdict": kind, "reason": reason, "repo": repo, "sha": sha,
            "production_ref": production_ref, "detail": detail or reason}


def _refs_from_mapping(source: dict, default_repo: str | None) -> list[tuple]:
    """Extract ("pr", repo, number) / ("commit", repo, sha) refs from one metadata dict."""
    refs: list[tuple] = []
    repo = default_repo
    for key in ("repo", "repository"):
        value = source.get(key)
        if isinstance(value, str) and REPO_RE.fullmatch(value.strip()):
            repo = value.strip()
    for key in PR_KEYS:
        value = source.get(key)
        items = value if isinstance(value, (list, tuple)) else [value]
        for item in items:
            if not isinstance(item, str):
                continue
            text = item.strip()
            match = PR_URL_RE.fullmatch(text)
            if match:
                repo = match.group(1)
                refs.append(("pr", match.group(1), int(match.group(2))))
            elif text.isdigit() and int(text) > 0:
                refs.append(("pr", repo, int(text)))
    for key in COMMIT_KEYS:
        value = source.get(key)
        if isinstance(value, str) and SHA_RE.fullmatch(value.strip().lower()):
            refs.append(("commit", repo, value.strip().lower()))
    return refs


def _head_shas_from_mapping(source: dict) -> list[str]:
    """Pre-squash PR heads recorded in one metadata dict — evidence, never a test ref."""
    heads: list[str] = []
    for key in HEAD_KEYS:
        value = source.get(key)
        if isinstance(value, str) and SHA_RE.fullmatch(value.strip().lower()):
            heads.append(value.strip().lower())
    return heads


def _refs_from_contract(contract) -> tuple[list[tuple], str | None]:
    if not isinstance(contract, str):
        return [], None
    text = contract.strip()
    match = PR_URL_RE.fullmatch(text)
    if match:
        return [("pr", match.group(1), int(match.group(2)))], match.group(1)
    if REPO_RE.fullmatch(text):
        return [], text
    return [], None


def _body_pr_refs(text) -> list[tuple]:
    if not isinstance(text, str):
        return []
    return [("pr", match.group(1), int(match.group(2))) for match in PR_URL_RE.finditer(text)]


def _repo_hints(texts) -> set[str]:
    """Repos named in the card's own prose/branch/workspace (never guesses)."""
    found: set[str] = set()
    for text in texts:
        if not isinstance(text, str) or not text:
            continue
        for repo in DEFAULT_PRODUCTION_REFS:
            name = repo.split("/")[1]
            if repo in text or re.search(r"(?<![A-Za-z0-9_-])%s(?![A-Za-z0-9_-])"
                                         % re.escape(name), text):
                found.add(repo)
    return found


def collect_refs(conn, task_id: str, contract, metadata) -> tuple[list[tuple], str | None,
                                                                 set[str], list[str]]:
    """Refs in priority order: this completion's metadata, the contract, prior runs, body.

    Returns (refs, repo, repo_hints, heads). ``repo`` is the strongest repo the record
    names; ``repo_hints`` are repos mentioned by the card's branch/workspace/prose and are
    only used when ``repo`` is None (an ambiguous hint set is never guessed at). ``heads``
    are recorded pre-squash PR heads: they explain a record that has no testable ref, and
    are never returned in ``refs``.
    """
    refs: list[tuple] = []
    heads: list[str] = []
    repo = None
    contract_refs, contract_repo = _refs_from_contract(contract)
    repo = contract_repo
    if isinstance(metadata, dict):
        meta_refs = _refs_from_mapping(metadata, repo)
        heads.extend(_head_shas_from_mapping(metadata))
        if meta_refs:
            repo = meta_refs[-1][1] or repo
        refs.extend(meta_refs)
    refs.extend(contract_refs)
    try:
        rows = conn.execute(
            "SELECT metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT ?",
            (task_id, PRIOR_RUNS_SCANNED),
        ).fetchall()
    except Exception:
        rows = []
    for row in rows:
        raw = row["metadata"] if hasattr(row, "keys") else row[0]
        if not isinstance(raw, str) or not raw:
            continue
        try:
            prior = json.loads(raw)
        except Exception:
            continue
        if isinstance(prior, dict):
            refs.extend(_refs_from_mapping(prior, repo))
            heads.extend(_head_shas_from_mapping(prior))
    hints: set[str] = set()
    try:
        row = conn.execute(
            "SELECT body, branch_name, workspace_path FROM tasks WHERE id=?", (task_id,)
        ).fetchone()
    except Exception:
        row = None
    if row is not None:
        body = row["body"] if hasattr(row, "keys") else row[0]
        branch = row["branch_name"] if hasattr(row, "keys") else row[1]
        workspace = row["workspace_path"] if hasattr(row, "keys") else row[2]
        refs.extend(_body_pr_refs(body))
        hints = _repo_hints((body, branch, workspace))
    # PR refs win: the recorded head does not survive a squash merge.
    prs = [ref for ref in refs if ref[0] == "pr"]
    commits = [ref for ref in refs if ref[0] == "commit"]
    return prs + commits, repo, hints, heads


def _not_needed(metadata, text: str) -> bool:
    if isinstance(metadata, dict):
        flag = metadata.get("deploy")
        if isinstance(flag, str) and flag.strip().lower().replace("_", "-") in {
                "not-needed", "none", "no-deploy"}:
            return True
    return NOT_NEEDED_MARKER in (text or "").lower()


def stamp_deploy_check(metadata, verdict: dict) -> dict:
    """Return a NEW metadata dict carrying the gate's verdict string."""
    out = dict(metadata) if isinstance(metadata, dict) else {}
    kind = verdict.get("verdict")
    repo = verdict.get("repo") or ""
    sha = (verdict.get("sha") or "")[:12]
    ref = verdict.get("production_ref") or ""
    if kind == "DEPLOYED":
        value = "DEPLOYED %s@%s (%s)" % (repo, sha, ref)
    elif kind == "NOT_DEPLOYED":
        value = "NOT_DEPLOYED %s@%s (%s)" % (repo, sha, ref)
    elif kind == "NOT_NEEDED":
        value = "not-needed"
    else:
        value = "UNKNOWN (%s)" % (verdict.get("reason") or "unknown")
    out["deploy-check"] = value
    if kind == "NOT_NEEDED":
        out.setdefault("deploy", "not-needed")
    return out


# --------------------------------------------------------------------------- github

def _github_token(environ) -> str:
    for key in ("GITHUB_TOKEN", "GH_TOKEN"):
        value = environ.get(key)
        if value:
            return value.strip()
    homes = [environ.get("HERMES_HOME"), "/opt/data"]
    for home in [h for h in homes if h]:
        for name in (".env", os.path.join("profiles", "software-developer", ".env")):
            try:
                with open(os.path.join(home, name), "r", encoding="utf-8-sig",
                          errors="replace") as handle:
                    for line in handle:
                        if line.startswith("GITHUB_TOKEN=") or line.startswith("GH_TOKEN="):
                            return line.split("=", 1)[1].strip().strip('"').strip("'")
            except Exception:
                continue
    return ""


def _default_api(method: str, endpoint: str, environ) -> dict:
    token = _github_token(environ)
    if not token:
        raise RuntimeError("no-token")
    request = urllib.request.Request(
        "https://api.github.com" + endpoint,
        headers={"Authorization": "Bearer " + token, "User-Agent": "hermes-deploy-gate",
                 "Accept": "application/vnd.github+json"},
        method=method)
    with urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:
        return json.load(response)


def _production_tip(repo: str, ref: str, api, now: float) -> str:
    key = (repo, ref)
    cached = _TIP_CACHE.get(key)
    if cached and now - cached[0] < TIP_TTL_SECONDS:
        return cached[1]
    payload = api("GET", "/repos/%s/git/ref/heads/%s" % (repo, ref))
    sha = ((payload.get("object") or {}).get("sha") or "")
    if not SHA_RE.fullmatch(sha):
        raise RuntimeError("ref read returned no sha")
    _TIP_CACHE[key] = (now, sha)
    return sha


# --------------------------------------------------------------------------- gate

def check_deploy_gate(conn, task_id: str, metadata=None, *, summary=None, result=None,
                      force=False, environ=None, api=None) -> dict | None:
    """Return a verdict dict, or None when the gate does not apply. Never raises."""
    environ = os.environ if environ is None else environ
    if (environ.get(KILL_ENV) or "").strip().lower() == "off":
        return None
    text = " ".join(value for value in (summary, result) if isinstance(value, str))
    if _not_needed(metadata, text):
        return _verdict("NOT_NEEDED", "explicit-not-needed",
                        detail="the completion carries an explicit deploy: not-needed marker")
    try:
        row = conn.execute("SELECT completion_contract FROM tasks WHERE id=?",
                           (task_id,)).fetchone()
    except Exception:
        return None
    if row is None:
        return None
    contract = row["completion_contract"] if hasattr(row, "keys") else row[0]
    refs, repo, hints, heads = collect_refs(conn, task_id, contract, metadata)
    declared = isinstance(contract, str) and contract not in ("", "local-only")
    intends_deploy = declared or bool(refs)
    if not intends_deploy and isinstance(metadata, dict):
        # The card may claim a deploy artifact that does not resolve to a sha at all
        # (e.g. published_pr still "pending"). That is worth recording as UNKNOWN.
        intends_deploy = any(isinstance(metadata.get(key), str) and metadata.get(key, "").strip()
                             for key in PR_KEYS + COMMIT_KEYS + HEAD_KEYS)
    if not intends_deploy:
        return None

    if not refs:
        if repo is None and len(hints) == 1:
            repo = next(iter(hints))
        if heads:
            # A head-only record has nothing testable: a squash merge never contains the
            # head, so an ancestry test could only refuse a deployed card (review P1).
            log.info("deploy-gate: record carries only a pre-squash head %s; "
                     "nothing testable against production", heads[0][:12])
            return _verdict("UNKNOWN", "head-pre-squash", repo=repo, sha=heads[0],
                            detail="the record names only PR head %s, which a squash merge "
                                   "never contains; record the PR (it carries merge_commit_sha) "
                                   "or the merge commit" % heads[0][:12])
        return _verdict("UNKNOWN", "no-commit-recorded", repo=repo,
                        detail="the card records no PR and no full commit sha")
    kind, ref_repo, value = refs[0]
    repo = ref_repo or repo
    if not repo:
        if len(hints) == 1:
            repo = next(iter(hints))
        elif len(hints) > 1:
            return _verdict("UNKNOWN", "ambiguous-repo",
                            detail="the card mentions several mapped repos")
        else:
            return _verdict("UNKNOWN", "no-repo-recorded",
                            detail="the recorded commit names no repository")

    production_ref = DEFAULT_PRODUCTION_REFS.get(repo)
    if not production_ref:
        return _verdict("UNKNOWN", "unmapped-repo", repo=repo,
                        detail="%s has no verified production ref in this gate" % repo)

    call = api if api is not None else (lambda method, endpoint: _default_api(method, endpoint, environ))
    now = time.time()
    sha = None
    try:
        if kind == "pr":
            payload = call("GET", "/repos/%s/pulls/%d" % (repo, int(value)))
            if not payload.get("merged"):
                return _verdict("UNKNOWN", "pr-not-merged", repo=repo,
                                production_ref=production_ref,
                                detail="PR %s#%s is not merged" % (repo, value))
            sha = (payload.get("merge_commit_sha") or "").lower()
            if not SHA_RE.fullmatch(sha):
                return _verdict("UNKNOWN", "api-error", repo=repo,
                                production_ref=production_ref,
                                detail="PR %s#%s exposes no merge commit" % (repo, value))
            head = (payload.get("head") or {}).get("sha") or "?"
            log.info("deploy-gate: resolved PR %s#%s head %s -> merge %s",
                     repo, value, head[:12], sha[:12])
        else:
            sha = value
        tip = _production_tip(repo, production_ref, call, now)
        compare = call("GET", "/repos/%s/compare/%s...%s" % (repo, tip, sha))
        ahead_by = int(compare.get("ahead_by") or 0)
    except RuntimeError as exc:
        reason = "no-token" if str(exc) == "no-token" else "api-error"
        return _verdict("UNKNOWN", reason, repo=repo, sha=sha, production_ref=production_ref,
                        detail=str(exc)[:120])
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError, KeyError) as exc:
        return _verdict("UNKNOWN", "api-error", repo=repo, sha=sha,
                        production_ref=production_ref,
                        detail="%s: %s" % (exc.__class__.__name__, str(exc)[:120]))
    except Exception as exc:  # noqa: BLE001 - the contract is fail-open
        return _verdict("UNKNOWN", "api-error", repo=repo, sha=sha,
                        production_ref=production_ref,
                        detail="%s: %s" % (exc.__class__.__name__, str(exc)[:120]))

    if ahead_by == 0:
        return _verdict("DEPLOYED", "in-production", repo=repo, sha=sha,
                        production_ref=production_ref,
                        detail="%s@%s is in %s" % (repo, sha[:12], production_ref))
    return _verdict("NOT_DEPLOYED", "absent-from-production", repo=repo, sha=sha,
                    production_ref=production_ref,
                    detail="merged-not-deployed: %s" % sha)
