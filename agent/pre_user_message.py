"""``pre_user_message`` hook: let a plugin rewrite a person's typed prompt before the turn builds on it.

Fired once per turn from the top of ``build_turn_context`` — before the user dict is staged, the
turn identity is bound, history is assembled or ``pre_llm_call`` runs — so the rewritten text is
the one the model is sent AND the one the transcript persists (the two never disagree).

Contract (also documented in ``hermes_cli/plugins.py`` next to ``VALID_HOOKS``):

* kwargs: ``message`` (the clean text the person typed), ``session_id``, ``platform``, ``model``,
  ``is_first_turn``, ``parent_session_id``, ``sender_id``.
* return ``{"action": "rewrite", "text": "<replacement>"}`` to replace the prompt; anything else
  (``None``, ``"allow"``, an empty/unchanged/non-string ``text``) leaves it untouched. First valid
  rewrite wins. Fail-open: a raising, hung (bounded) or malformed callback never blocks the turn.

Which turns fire (default-deny — only text a person typed into a surface they are reading):

* ``is_attended(agent)`` — cli / tui / desktop / acp. Not cron, kanban workers, subagents,
  one-shot runs or messaging chats (those have ``pre_gateway_dispatch``; firing here too would
  rewrite the same message twice).
* not a fork turn (``agent._turn_origin``), not a synthesized turn (``persist_user_display_kind``).
* the prompt is plain text. Multimodal (list) prompts are left alone.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

HOOK_NAME = "pre_user_message"


def _eligible(agent: Any, user_message: Any, persist_user_message: Any, display_kind: Optional[str]) -> bool:
    if display_kind or getattr(agent, "_turn_origin", None):
        return False
    if not isinstance(user_message, str) or not user_message.strip():
        return False
    if persist_user_message is not None and not isinstance(persist_user_message, str):
        return False
    from agent.session_source import is_attended
    return is_attended(agent)


def _first_rewrite(results: list[Any], original: str) -> Optional[str]:
    for result in results:
        if not isinstance(result, dict) or result.get("action") != "rewrite":
            continue
        text = result.get("text")
        if isinstance(text, str) and text.strip() and text != original:
            return text
    return None


def _adopt_rewrite_into_staged_row(agent: Any, original: str, rewritten: str) -> bool:
    """Keep the row the surface already staged for this turn in step with the rewrite.

    CLI/desktop stage the person's raw text (desktop even writes the DB row at submit) and
    ``_stage_turn_user_message`` adopts it only while its content equals the turn's clean text.
    A rewritten turn would otherwise miss the adoption and append a SECOND row. Returns False when
    the durable row cannot be updated — the caller then drops the rewrite so model input and
    transcript still agree.
    """
    pending = getattr(agent, "_pending_cli_user_message", None)
    if not isinstance(pending, dict) or pending.get("content") != original:
        return True  # nothing staged for this text: the turn writes its own row from the rewrite
    from agent.session_persistence import _DB_PERSISTED_MARKER, _persist_lock
    row_id = pending.get("_row_id")
    if isinstance(row_id, int) and not isinstance(row_id, bool):
        db = getattr(agent, "_session_db", None)
        if db is None:
            return False
        # The desktop writes its submit row under a recorded owner key (a rotation may have moved the
        # agent off it); tui_gateway/session_workdir.py::_SUBMIT_ROW_SESSION_KEY stamps it on the dict.
        owner = str(pending.get("_submit_row_session_id") or "") or agent.session_id
        try:
            with _persist_lock(agent):
                if not db.set_user_message_content(owner, row_id, rewritten):
                    return False
        except Exception:
            logger.warning("pre_user_message: staged row update failed for session=%s",
                           getattr(agent, "session_id", None) or "none", exc_info=True)
            return False
    elif pending.get(_DB_PERSISTED_MARKER):
        return False  # durable but unaddressable: cannot be corrected
    pending["content"] = rewritten
    return True


def apply_pre_user_message_hook(
    agent: Any, user_message: Any, persist_user_message: Any, *,
    persist_user_display_kind: Optional[str], conversation_history: Optional[list[Any]],
) -> tuple[Any, Any]:
    """Run the hook; returns ``(user_message, persist_user_message)``, rewritten or unchanged."""
    try:
        if not _eligible(agent, user_message, persist_user_message, persist_user_display_kind):
            return user_message, persist_user_message
        original = persist_user_message if persist_user_message is not None else user_message
        from hermes_cli.lifecycle import invoke_hook
        results = invoke_hook(
            HOOK_NAME,
            message=original,
            session_id=getattr(agent, "session_id", None) or "",
            platform=getattr(agent, "platform", None) or "",
            model=getattr(agent, "model", None) or "",
            is_first_turn=not bool(conversation_history),
            parent_session_id=getattr(agent, "_parent_session_id", None) or "",
            sender_id=getattr(agent, "_user_id", None) or "",
        )
        rewritten = _first_rewrite(results, original)
        if rewritten is None:
            return user_message, persist_user_message
        # ``user_message`` is what the model is sent; ``persist_user_message`` is the clean text when a
        # prologue wrapped it (@-expansion, voice prefix). Rewrite the typed text inside the wrapper.
        if persist_user_message is None or user_message == persist_user_message:
            new_user_message = rewritten
        elif original in user_message:
            new_user_message = user_message.replace(original, rewritten, 1)
        else:
            logger.info("pre_user_message: rewrite skipped — typed text not found in the prepared prompt")
            return user_message, persist_user_message
        if not _adopt_rewrite_into_staged_row(agent, original, rewritten):
            logger.warning("pre_user_message: rewrite dropped — staged transcript row could not be updated")
            return user_message, persist_user_message
        return new_user_message, (rewritten if persist_user_message is not None else None)
    except Exception:
        logger.warning("pre_user_message hook failed; turn continues with the original prompt", exc_info=True)
        return user_message, persist_user_message
