"""A compaction-demoted tool result clears its identical-result no-progress tally once."""

import json

from agent.context_compressor import ContextCompressor
from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController, ToolCallSignature

ARGS_X = {"path": "/x"}
ARGS_Y = {"path": "/y"}
PAYLOAD = "line of file content\n" * 100


def _controller() -> ToolCallGuardrailController:
    return ToolCallGuardrailController(ToolCallGuardrailConfig.from_mapping({"hard_stop_enabled": True}))


def _compressor() -> ContextCompressor:
    return ContextCompressor("test/model", quiet_mode=True)


def _transcript(rounds: int = 8) -> list[dict]:
    messages: list[dict] = [{"role": "user", "content": "go"}]
    for n in range(rounds):
        call = {"id": f"c{n}", "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps(ARGS_X)}}
        messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        messages.append({"role": "tool", "tool_call_id": f"c{n}", "tool_name": "read_file", "content": PAYLOAD})
    return messages


def _read(controller, args=ARGS_X, result=PAYLOAD):
    """One full read: the pre-call decision, then (if allowed) the recorded result."""
    before = controller.before_call("read_file", args)
    after = controller.after_call("read_file", args, result, failed=False) if before.allows_execution else None
    return before, after


def _read_n(controller, n, args=ARGS_X):
    for _ in range(n):
        _read(controller, args)


def _demote(compressor, messages=None):
    messages = messages or _transcript()
    return compressor._demote_stale_tail_tools(messages, 0)


def _bound(controller) -> ContextCompressor:
    compressor = _compressor()
    compressor.bind_tool_guardrails(controller)
    return compressor


def test_demotion_evicts_signature_and_resets_no_progress_tally():
    controller = _controller()
    messages = _transcript()
    pre_demotion = messages[2]["content"]  # the bytes the demotion will replace, captured first

    _read_n(controller, 5)
    out = _demote(_bound(controller), messages)
    stub = out[2]["content"]
    assert stub != pre_demotion  # the oldest result became a recovery stub

    before, after = _read(controller, result=pre_demotion)  # the tool re-reads the unchanged file
    assert before.action == "allow"  # the no-progress block must NOT fire
    assert after.count == 1  # the tally restarted instead of resuming at 5
    assert pre_demotion == PAYLOAD  # byte-for-byte: the re-read value is the pre-demotion content


def test_five_identical_no_demotion_still_block():
    controller = _controller()
    _read_n(controller, 5)
    decision = controller.before_call("read_file", ARGS_X)
    assert (decision.action, decision.code, decision.count) == ("block", "idempotent_no_progress_block", 5)


def test_interleaved_demotions_still_block_within_bound():
    controller = _controller()
    sig = ToolCallSignature.from_call("read_file", ARGS_X)
    _read_n(controller, 1)
    assert controller.note_content_evicted(sig) is True
    _read_n(controller, 2)
    assert controller.note_content_evicted(sig) is False  # one shot, already spent
    _read_n(controller, 3)
    decision = controller.before_call("read_file", ARGS_X)
    assert (decision.action, decision.code, decision.count) == ("block", "idempotent_no_progress_block", 5)


def test_eviction_is_per_signature_and_leaves_exact_failure_untouched():
    controller = _controller()
    sig_a = ToolCallSignature.from_call("read_file", ARGS_X)
    _read_n(controller, 5, ARGS_X)
    _read_n(controller, 5, ARGS_Y)
    assert controller.note_content_evicted(sig_a) is True
    assert controller.before_call("read_file", ARGS_X).action == "allow"
    assert controller.before_call("read_file", ARGS_Y).code == "idempotent_no_progress_block"

    for _ in range(controller.config.exact_failure_block_after):
        controller.after_call("terminal", {"command": "false"}, "boom", failed=True)
    sig_t = ToolCallSignature.from_call("terminal", {"command": "false"})
    controller.note_content_evicted(sig_t)
    assert controller.before_call("terminal", {"command": "false"}).code == "repeated_exact_failure_block"


def test_one_shot_credit_is_per_turn():
    controller = _controller()
    sig = ToolCallSignature.from_call("read_file", ARGS_X)
    assert controller.note_content_evicted(sig) is False  # no tally: credit consumed anyway
    assert controller.note_content_evicted(sig) is False
    controller.reset_for_turn()
    _read_n(controller, 2)
    assert controller.note_content_evicted(sig) is True


def test_compressor_wrapper_notifies_bound_controller():
    controller = _controller()
    _read_n(controller, 5)
    _demote(_bound(controller))
    assert controller.before_call("read_file", ARGS_X).action == "allow"
    sig = ToolCallSignature.from_call("read_file", ARGS_X)
    assert controller.note_content_evicted(sig) is False  # the demotion already spent the credit


def test_unbound_compressor_demotes_without_notifying():
    out = _demote(_compressor())
    assert out[2]["content"] != PAYLOAD
