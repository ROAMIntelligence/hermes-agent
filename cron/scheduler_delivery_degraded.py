"""Degraded Bot Chat delivery: an out-of-band notice plus a record of how long the turn took.

A Bot Chat delivery on the CLI lane is a full agent turn with a wall-clock budget
(``cron.bot_chat_delivery_timeout_seconds``). When that turn times out or fails, the alert body
never reaches the human — and the only notice used to be a deferred marker that replays through
the SAME Bot Chat turn, so an alarm about a failed alarm travelled the path that had just failed.

This sibling owns the two pieces that must not depend on that turn:

* ``announce_degraded`` posts a SHORT notice (headline + the head of the alert) straight to the
  platform home channels through the standalone send lane — no agent, no session, no Bot Chat.
* ``record_turn`` persists the turn's elapsed seconds on the job so the budget is seen being
  approached, not only crossed.

Facade helpers (``_standalone_send`` …) are reached late-bound: this module is imported lazily by
``cron.scheduler_delivery``, never at its module level.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger("cron.scheduler")

# A turn at this fraction of its budget is logged as a warning: the ceiling is being approached.
TURN_WARN_FRACTION = 0.75
# Short form of the alert carried by the notice. Well under every platform's message limit.
EXCERPT_CHARS = 700
_DEFAULT_DELIVER = "all"


@dataclass
class _OobTarget:
    """Duck-types the fields ``_standalone_send`` reads from ``_TargetDelivery``."""

    job: dict
    platform: Any
    pconfig: Any
    chat_id: str
    thread_id: Optional[str]
    where: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def short_form(content: str, limit: int = EXCERPT_CHARS) -> str:
    """Head of the alert, bounded; says how much was cut so the reader knows it is partial."""
    text = (content or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}\n…[{len(text) - limit} more characters in the saved output]"


def marker_text(job_id: str, timeout_s: int, content: str) -> str:
    """Body of the deferred in-chat marker (the secondary, Bot Chat-side copy of the notice)."""
    return (
        f"DELIVERY DEGRADED: this alert's bot-chat turn timed out after "
        f"{timeout_s}s, so the full output could NOT be posted here. Read the "
        f"complete saved output with `hermes cron runs` (job '{job_id}'). "
        f"Excerpt: {content.strip()[:280]}"
    )


def notice_text(job: dict, outcome: str, elapsed_s: float, timeout_s: int, content: str) -> str:
    """The out-of-band notice: what failed, for which job, and the head of the alert itself."""
    name = job.get("name") or job.get("id", "?")
    what = (f"timed out after {timeout_s}s" if outcome == "timeout"
            else f"failed after {elapsed_s:.0f}s")
    return (
        f"DELIVERY DEGRADED: the bot-chat turn for cron job '{name}' ({job.get('id', '?')}) "
        f"{what}, so the full alert may NOT have been posted to Bot Chat. "
        f"Full output is saved: `hermes cron runs`.\n\n"
        f"Alert (short form):\n{short_form(content)}"
    )


def _notice_targets(job: dict, user_cfg: dict) -> list:
    """Home-channel targets for the notice. Bot Chat and ``local`` are excluded by construction:
    the notice exists to avoid the lane that failed."""
    from cron import scheduler_delivery as sd
    raw = str((user_cfg.get("cron") or {}).get("degraded_notice_deliver") or _DEFAULT_DELIVER)
    seen, targets = set(), []
    for token in (part.strip() for part in raw.split(",")):
        if not token or token == "local" or sd.parse_bot_chat_deliver_token(token) is not None:
            continue
        for part in sd._expand_routing_tokens(token):
            target = sd._resolve_single_delivery_target(job, part, from_broadcast=True)
            if not target or target["platform"] == sd.BOT_CHAT_PLATFORM:
                continue
            key = (target["platform"].lower(), str(target["chat_id"]), target.get("thread_id"))
            if key not in seen:
                seen.add(key)
                targets.append(target)
    return targets


def _send_one(job: dict, target: dict, text: str, config, user_cfg: dict) -> tuple[Optional[str], Optional[str]]:
    """``(delivered_where, None)``, ``(None, error)``, or ``(None, None)`` when policy hides it."""
    from cron import scheduler_delivery as sd
    from gateway.config import Platform
    from gateway.warning_notifications import warning_notifications_enabled
    name, chat_id = target["platform"], str(target["chat_id"])
    where = f"{name}:{chat_id}"
    if not warning_notifications_enabled(name, user_cfg):
        logger.info("Job '%s': degraded-delivery notice to %s hidden by suppress_warning_notifications",
                    job.get("id"), where)
        return None, None
    try:
        platform = Platform(name.lower())
    except (ValueError, KeyError):
        return None, f"unknown platform '{name}'"
    resolved, err = sd._resolve_target_transport(job, platform, name, target, None, config)
    if resolved is None:
        return None, err
    pconfig = resolved[1]
    result, send_err = sd._standalone_send(
        _OobTarget(job, platform, pconfig, chat_id, target.get("thread_id"), where), text, [])
    if send_err is None and isinstance(result, dict) and result.get("error"):
        send_err = f"delivery error: {result['error']} (target {where})"
    return (None, send_err) if send_err else (where, None)


def announce_degraded(job: dict, outcome: str, elapsed_s: float, timeout_s: int, content: str,
                      deferred: Optional[dict] = None) -> Optional[dict]:
    """Post the short notice to every home channel, independent of the Bot Chat turn.

    Never raises: the original delivery error is what the caller returns, and an alarm path that
    can throw would mask it. The outcome is stored on the job (``last_delivery_degraded``) so a
    notice that could not be sent anywhere is still visible in ``hermes cron list``.
    """
    if (deferred or {}).get("degraded"):
        return None  # draining the marker itself: the notice for this execution already went out
    delivered: list = []
    errors: list = []
    try:
        from cron import scheduler as sched
        from gateway.config import load_gateway_config
        text = notice_text(job, outcome, elapsed_s, timeout_s, content)
        user_cfg = sched.load_config() or {}
        config = load_gateway_config()
        targets = _notice_targets(job, user_cfg)
        if not targets:
            errors.append("no home channel is configured for the out-of-band notice")
        for target in targets:
            where, err = _send_one(job, target, text, config, user_cfg)
            if where:
                delivered.append(where)
            elif err:
                errors.append(err)
    except Exception as exc:  # health: allow BLE001 -- alarm path must never mask the delivery error; logged with traceback below
        logger.warning("Job '%s': degraded-delivery notice failed", job.get("id"), exc_info=True)
        errors.append(f"{type(exc).__name__}: {exc}")
    for err in errors:
        logger.warning("Job '%s': degraded-delivery notice: %s", job.get("id"), err)
    if delivered:
        logger.info("Job '%s': degraded-delivery notice posted to %s", job.get("id"), ", ".join(delivered))
    record = {"at": _now_iso(), "outcome": outcome, "elapsed_s": round(elapsed_s, 1),
              "timeout_s": timeout_s, "notice_delivered": delivered, "notice_errors": errors}
    _persist(job, {"last_delivery_degraded": record})
    return record


def record_turn(job: dict, elapsed_s: float, timeout_s: int, outcome: str) -> None:
    """Persist the delivery turn's elapsed seconds (last and peak) on the job.

    ``outcome`` is ``ok`` | ``failed`` | ``timeout``. The peak never decreases, so one slow turn
    stays visible after later fast ones; a turn past ``TURN_WARN_FRACTION`` of its budget logs a
    WARNING while it is still inside the ceiling.
    """
    previous = job.get("delivery_turn") if isinstance(job.get("delivery_turn"), dict) else {}
    peak = max(float(previous.get("peak_s") or 0), elapsed_s)
    record = {"last_s": round(elapsed_s, 1), "peak_s": round(peak, 1), "timeout_s": timeout_s,
              "outcome": outcome, "at": _now_iso()}
    log = logger.warning if outcome == "ok" and elapsed_s >= timeout_s * TURN_WARN_FRACTION else logger.info
    log("Job '%s': bot-chat delivery turn %s in %.1fs of the %ss budget",
        job.get("id"), outcome, elapsed_s, timeout_s)
    _persist(job, {"delivery_turn": record})


def _persist(job: dict, values: dict) -> None:
    """Bookkeeping only: set on the in-memory job and store; never raises."""
    job.update(values)
    try:
        from cron.jobs import update_job
        update_job(job["id"], values)
    except Exception as exc:  # health: allow BLE001 -- bookkeeping must not break delivery; logged
        logger.debug("Job '%s': could not record %s: %s", job.get("id"), ", ".join(values), exc,
                     exc_info=True)


def record_exit(job: dict, started: float, timeout_s: int, returncode: int) -> float:
    """Record a turn that exited on its own (``ok`` / ``failed``); returns the elapsed seconds."""
    elapsed = time.monotonic() - started
    record_turn(job, elapsed, timeout_s, "ok" if returncode == 0 else "failed")
    return elapsed


def on_timeout(job: dict, content: str, deferred: Optional[dict], *, key: str, profile: Any,
               home: Any, for_failure: bool, started: float, timeout_s: int) -> str:
    """Everything a timed-out turn owes the human; returns the hint for the delivery error.

    Replaying the full payload risks a duplicate (the killed turn may already have persisted it);
    staying silent loses the alert (2026-09-19 docgen-deadman). So: (1) post the short notice
    out-of-band, then (2) queue a SHORT marker into Bot Chat — once per execution (stable key; the
    re-mark guard reads the record's ``degraded`` flag, never the text).
    """
    job_id = job.get("id", "?")
    elapsed = time.monotonic() - started
    record_turn(job, elapsed, timeout_s, "timeout")
    notice = announce_degraded(job, "timeout", elapsed, timeout_s, content, deferred)
    posted = (f"short notice posted out-of-band to {', '.join(notice['notice_delivered'])}; "
              if notice and notice["notice_delivered"] else "")
    queued = False
    if not (deferred or {}).get("degraded"):
        try:
            from cron.bot_chat_delivery import defer
            # Deferred ids double as live-owner delivery ids (32-64 hex chars,
            # tools.bot_live_delivery._delivery_id): a fresh digest of the execution key.
            marker_key = hashlib.sha256(f"{key}:degraded".encode()).hexdigest()
            defer(marker_key, dict(job), marker_text(job_id, timeout_s, content), profile, home,
                  for_failure=for_failure, degraded=True)
            queued = True
        except Exception as exc:  # health: allow BLE001 -- the marker is best-effort; logged below
            logger.warning("Job '%s': degraded-delivery marker could not be queued: %s",
                           job_id, exc, exc_info=True)
    if queued:
        return (f"{posted}a short degraded-delivery notice was queued to Bot Chat — posted once the "
                f"session frees; full output stays saved, run `hermes cron runs` for job '{job_id}'")
    return (f"{posted}the result is saved; run `hermes cron runs` to see it, or `hermes doctor` "
            "if this keeps happening")
