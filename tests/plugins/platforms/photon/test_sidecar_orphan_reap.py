"""A sidecar orphaned to init is reaped on the next start, whatever port it holds.

``_reap_stale_sidecar`` only reaches a sidecar squatting OUR port, so a sidecar leaked by a
killed gateway/test run on another port lived forever with its bearer env. These tests
orphan a REAL node process (SIGKILL its parent) and assert ``_start_sidecar`` removes it,
and pin the guards that keep the scan from touching anything else.
"""
from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon import adapter as photon_adapter
from plugins.platforms.photon import sidecar_paths
from plugins.platforms.photon.adapter import PhotonAdapter

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX process table"),
    pytest.mark.skipif(shutil.which("node") is None, reason="needs node"),
]

# Parent: spawns the fake sidecar detached from its stdio, reports the child pid, then idles.
_PARENT = """
import subprocess, sys, time
c = subprocess.Popen(["node", sys.argv[1]], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
print(c.pid, flush=True)
time.sleep(600)
"""


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    return PhotonAdapter(PlatformConfig(enabled=True, token="", extra={}))


def _state(pid: int) -> str | None:
    """Linux process state letter, or None when the pid is gone."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return None


def _ppid(pid: int) -> int | None:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
    except OSError:
        return None


def _gone(pid: int) -> bool:
    return _state(pid) in (None, "Z")


def _wait(cond: Any, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


@pytest.fixture
def orphan(tmp_path: Path):
    """A live fake sidecar (cmdline ends photon/sidecar/index.mjs) whose parent was SIGKILLed."""
    script = tmp_path / "photon" / "sidecar" / "index.mjs"
    script.parent.mkdir(parents=True)
    script.write_text("setInterval(() => {}, 1000);\n")
    parent = subprocess.Popen(
        [sys.executable, "-c", _PARENT, str(script)], stdout=subprocess.PIPE, text=True
    )
    pid = int(parent.stdout.readline())
    parent.kill()
    parent.wait()
    try:
        if not _wait(lambda: _ppid(pid) == 1):
            pytest.skip("this environment does not reparent orphans to pid 1")
        yield pid
    finally:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


@pytest.mark.live_system_guard_bypass  # fixture reaps an orphan outside the test subtree
def test_orphaned_sidecar_is_found(orphan: int, real_orphan_scan: None) -> None:
    assert orphan in PhotonAdapter._find_orphan_sidecar_pids()


def test_sidecar_with_live_parent_is_not_found(tmp_path: Path, real_orphan_scan: None) -> None:
    script = tmp_path / "photon" / "sidecar" / "index.mjs"
    script.parent.mkdir(parents=True)
    script.write_text("setInterval(() => {}, 1000);\n")
    live = subprocess.Popen(["node", str(script)], stdin=subprocess.DEVNULL)
    try:
        assert _wait(lambda: _state(live.pid) == "S")
        assert live.pid not in PhotonAdapter._find_orphan_sidecar_pids()
    finally:
        live.kill()
        live.wait()


@pytest.mark.parametrize(
    "ps_line, environ_keys, expected",
    [
        ("500 1 /usr/bin/node /x/photon/sidecar/index.mjs", None, True),
        ("500 1 node index.mjs", {"PHOTON_SIDECAR_TOKEN", "PHOTON_SIDECAR_PORT", "PATH"}, True),
        # relative form without the Photon env corroboration: someone else's index.mjs
        ("500 1 node index.mjs", {"PATH"}, False),
        ("500 1 node index.mjs", None, False),
        # parent alive
        ("500 4321 /usr/bin/node /x/photon/sidecar/index.mjs", None, False),
        # not node: an editor/grep merely naming the file
        ("500 1 vim /x/photon/sidecar/index.mjs", None, False),
        ("500 1 node /x/photon/sidecar/index.mjs.bak", None, False),
    ],
)
def test_scan_guards(
    monkeypatch: pytest.MonkeyPatch, ps_line: str, environ_keys: Any, expected: bool,
    real_orphan_scan: None,
) -> None:
    monkeypatch.setattr(PhotonAdapter, "_quick_stdout", staticmethod(lambda cmd: ps_line + "\n"))
    monkeypatch.setattr(PhotonAdapter, "_proc_environ_keys", staticmethod(lambda pid: environ_keys))
    assert (500 in PhotonAdapter._find_orphan_sidecar_pids()) is expected


@pytest.mark.asyncio
@pytest.mark.live_system_guard_bypass  # the reap signals a process outside the test subtree
async def test_start_sidecar_reaps_orphan_on_another_port(
    orphan: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, real_orphan_scan: None
) -> None:
    # `subprocess.run` is faked below (it also backs `ps`), so run the real scan first and
    # only ever signal OUR orphan: a dev box may hold unrelated real ones.
    scanned = [p for p in PhotonAdapter._find_orphan_sidecar_pids() if p == orphan]
    assert scanned == [orphan]
    adapter = _make_adapter(monkeypatch)
    sidecar_dir = tmp_path / "live-sidecar"
    (sidecar_dir / "node_modules" / "spectrum-ts").mkdir(parents=True)
    monkeypatch.setattr(sidecar_paths, "_SIDECAR_DIR", sidecar_dir)
    monkeypatch.setattr(photon_adapter, "_sidecar_deps_stale", lambda: False)

    async def _no_port_reap(self: PhotonAdapter) -> None:
        return None

    async def _no_supervise(self: PhotonAdapter, proc: Any) -> None:
        return None

    monkeypatch.setattr(PhotonAdapter, "_reap_stale_sidecar", _no_port_reap)
    monkeypatch.setattr(PhotonAdapter, "_supervise_sidecar", _no_supervise)
    monkeypatch.setattr(
        photon_adapter.subprocess, "run",
        lambda *a, **k: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
    )

    class _Proc:
        pid = 4242
        stdin = None
        stdout = None

        def poll(self) -> None:
            return None

    monkeypatch.setattr(photon_adapter.subprocess, "Popen", lambda *a, **k: _Proc())

    class _Healthy:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        async def __aenter__(self) -> "_Healthy":
            return self

        async def __aexit__(self, *a: Any) -> bool:
            return False

        async def post(self, *a: Any, **k: Any) -> Any:
            return type("Resp", (), {"status_code": 200})()

    monkeypatch.setattr(photon_adapter.httpx, "AsyncClient", _Healthy)

    monkeypatch.setattr(PhotonAdapter, "_find_orphan_sidecar_pids", staticmethod(lambda: scanned))

    assert not _gone(orphan)
    await adapter._start_sidecar()
    assert _wait(lambda: _gone(orphan)), "orphaned sidecar survived _start_sidecar"
    if adapter._sidecar_supervisor_task is not None:
        adapter._sidecar_supervisor_task.cancel()


@pytest.mark.asyncio
async def test_reap_is_skipped_when_gateway_is_pid_1(
    monkeypatch: pytest.MonkeyPatch, real_orphan_scan: None
) -> None:
    """With a pid-1 gateway every child has ppid 1, so the scan would hit live sidecars."""
    adapter = _make_adapter(monkeypatch)
    monkeypatch.setattr(photon_adapter.os, "getpid", lambda: 1)
    called: List[bool] = []
    monkeypatch.setattr(
        PhotonAdapter, "_find_orphan_sidecar_pids", staticmethod(lambda: called.append(True) or [])
    )
    await adapter._reap_orphan_sidecars()
    assert called == []
