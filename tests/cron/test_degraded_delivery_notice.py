"""A bot-chat delivery that dies must be announced on a path that does not run through Bot Chat.

Two watchdog alerts hit the 600s bot-chat turn budget; the only notice was a deferred
marker replayed through the same Bot Chat turn, so the human received an excerpt at best. Contract:

1. A timed-out (or failed) bot-chat turn posts a SHORT notice — headline plus the head of the
   alert — straight to the home channels via the standalone send lane, no Bot Chat involved.
2. That notice is sent even when the deferred (Bot Chat) lane is broken, and a failing notice
   path never masks the original delivery error.
3. The turn's elapsed seconds are recorded (last + peak) so the budget is seen being approached.
"""
import subprocess
from unittest.mock import MagicMock, Mock

import pytest

from cron import scheduler_delivery as delivery
from cron import scheduler_delivery_degraded as degraded
from gateway.config import Platform, PlatformConfig

PAYLOAD = "DISK 97% on /opt/data — free 1.1GB\n" + "detail line\n" * 120


@pytest.fixture()
def lane(tmp_path, monkeypatch):
    """CLI fallback lane + one enabled platform with a home channel + a recording standalone sender."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_HOME_CHANNEL", "4242")
    monkeypatch.setattr(delivery, "_run_bot_chat_turn", Mock(
        side_effect=subprocess.TimeoutExpired(["hermes"], 600)))
    config = MagicMock()
    config.platforms = {Platform.DISCORD: PlatformConfig(enabled=True)}
    config.get_home_channel = lambda p: None
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda *a, **k: config)
    monkeypatch.setattr("cron.scheduler.load_config", lambda *a, **k: {"cron": {}})
    sends = []

    def standalone(t, content, media_files):
        sends.append({"where": t.where, "text": content})
        return {"success": True}, None

    monkeypatch.setattr(delivery, "_standalone_send", standalone)
    persisted = {}
    monkeypatch.setattr("cron.jobs.update_job",
                        lambda job_id, values: persisted.setdefault(job_id, {}).update(values))
    return sends, persisted


def _break_deferred_lane(monkeypatch):
    from cron import bot_chat_delivery as queue
    monkeypatch.setattr(queue, "defer", Mock(side_effect=RuntimeError("deferred lane is down")))


def test_timeout_posts_short_notice_without_bot_chat(lane, monkeypatch):
    sends, persisted = lane
    _break_deferred_lane(monkeypatch)
    job = {"id": "3f2a9c1d7e40", "name": "disk-watchdog", "execution_id": "e1"}

    result = delivery._deliver_to_bot_chat(job, PAYLOAD, "")

    assert len(sends) == 1 and sends[0]["where"] == "discord:4242"
    text = sends[0]["text"]
    assert "DELIVERY DEGRADED" in text and "disk-watchdog" in text
    assert "timed out after 600s" in text
    assert "DISK 97% on /opt/data" in text          # the short form of the alert lands...
    assert len(text) < len(PAYLOAD)                 # ...as a short form, not the whole payload
    assert "more characters in the saved output" in text
    assert "posted out-of-band to discord:4242" in result
    assert persisted["3f2a9c1d7e40"]["last_delivery_degraded"]["notice_delivered"] == ["discord:4242"]


def test_failed_turn_also_posts_notice(lane, monkeypatch):
    sends, _ = lane
    monkeypatch.setattr(delivery, "_run_bot_chat_turn", Mock(return_value=subprocess.CompletedProcess(
        ["hermes"], 1, stdout="", stderr="boom")))
    job = {"id": "j-fail", "name": "disk-watchdog", "execution_id": "e2"}

    result = delivery._deliver_to_bot_chat(job, PAYLOAD, "")

    assert "could not deliver" in result
    assert len(sends) == 1 and "failed after" in sends[0]["text"]


def test_notice_failure_never_masks_the_delivery_error(lane, monkeypatch):
    sends, persisted = lane
    monkeypatch.setattr(delivery, "_standalone_send",
                        Mock(side_effect=RuntimeError("discord adapter exploded")))
    job = {"id": "j-oob-down", "name": "w", "execution_id": "e3"}

    result = delivery._deliver_to_bot_chat(job, PAYLOAD, "")

    assert result is not None and "timed out after 600s" in result
    record = persisted["j-oob-down"]["last_delivery_degraded"]
    assert record["notice_delivered"] == [] and "discord adapter exploded" in record["notice_errors"][0]


def test_no_home_channel_is_recorded_not_silent(lane, monkeypatch):
    sends, persisted = lane
    monkeypatch.delenv("DISCORD_HOME_CHANNEL")
    job = {"id": "j-nohome", "name": "w", "execution_id": "e4"}

    delivery._deliver_to_bot_chat(job, PAYLOAD, "")

    assert sends == []
    assert persisted["j-nohome"]["last_delivery_degraded"]["notice_errors"]


def test_draining_the_degraded_marker_does_not_announce_again(lane, tmp_path):
    sends, _ = lane
    from hermes_state import SessionDB
    SessionDB(db_path=tmp_path / "state.db").close()
    record = {"id": "a" * 64, "home": str(tmp_path), "profile": "", "degraded": True}

    delivery._deliver_to_bot_chat({"id": "j-marker", "name": "w"}, "marker", "", deferred=record)

    assert sends == []


def test_turn_elapsed_is_recorded_with_peak(lane, monkeypatch):
    _, persisted = lane
    ticks = iter([100.0, 100.0 + 480.0, 500.0, 500.0 + 12.0])
    monkeypatch.setattr(delivery.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(delivery, "_run_bot_chat_turn", Mock(return_value=subprocess.CompletedProcess(
        ["hermes"], 0, stdout="", stderr="")))
    job = {"id": "j-elapsed", "name": "w", "execution_id": "e5"}

    assert delivery._deliver_to_bot_chat(job, "ok", "") is None
    first = persisted["j-elapsed"]["delivery_turn"]
    assert first["last_s"] == 480.0 and first["peak_s"] == 480.0 and first["timeout_s"] == 600
    assert first["outcome"] == "ok"

    assert delivery._deliver_to_bot_chat(job, "ok", "") is None
    second = persisted["j-elapsed"]["delivery_turn"]
    assert second["last_s"] == 12.0 and second["peak_s"] == 480.0  # a fast turn does not erase the peak


def test_turn_near_budget_logs_a_warning(caplog):
    with caplog.at_level("INFO", logger="cron.scheduler"):
        degraded.record_turn({"id": "j-warn"}, 470.0, 600, "ok")
        degraded.record_turn({"id": "j-fine"}, 20.0, 600, "ok")
    warned = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warned) == 1 and "j-warn" in warned[0].getMessage()


def test_cron_list_flags_a_turn_near_its_budget():
    from hermes_cli.cron import _delivery_turn_warning
    near = {"delivery_turn": {"last_s": 20.0, "peak_s": 470.0, "timeout_s": 600, "outcome": "ok"}}
    fine = {"delivery_turn": {"last_s": 20.0, "peak_s": 60.0, "timeout_s": 600, "outcome": "ok"}}
    assert "470s of its 600s budget" in _delivery_turn_warning(near)
    assert _delivery_turn_warning(fine) is None
    assert _delivery_turn_warning({}) is None


def test_notice_never_routes_through_bot_chat(lane, monkeypatch):
    """A bot-chat token in the notice target list is dropped: that is the lane that failed."""
    monkeypatch.setattr("cron.scheduler.load_config",
                        lambda *a, **k: {"cron": {"degraded_notice_deliver": "bot-chat:default,local,discord"}})
    targets = degraded._notice_targets({"id": "j"}, {"cron": {"degraded_notice_deliver": "bot-chat:default,local,discord"}})
    assert [t["platform"] for t in targets] == ["discord"]


def test_real_standalone_sender_receives_the_notice(tmp_path, monkeypatch):
    """No stub on ``_standalone_send``: the notice reaches ``_send_to_platform`` with the right
    platform/chat/thread, proving the duck-typed target matches what the real sender reads."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_HOME_CHANNEL", "4242")
    config = MagicMock()
    config.platforms = {Platform.DISCORD: PlatformConfig(enabled=True)}
    config.get_home_channel = lambda p: None
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda *a, **k: config)
    monkeypatch.setattr("cron.scheduler.load_config", lambda *a, **k: {"cron": {}})
    monkeypatch.setattr("cron.jobs.update_job", lambda *a, **k: None)
    seen = []

    async def fake_send(platform, pconfig, chat_id, content, thread_id=None, media_files=None):
        seen.append((platform, chat_id, thread_id, content))
        return {"success": True}

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", fake_send)
    record = degraded.announce_degraded({"id": "j-real", "name": "w"}, "timeout", 600.0, 600, PAYLOAD)

    assert record["notice_delivered"] == ["discord:4242"] and record["notice_errors"] == []
    assert seen[0][0] == Platform.DISCORD and seen[0][1] == "4242"
    assert "DELIVERY DEGRADED" in seen[0][3]
