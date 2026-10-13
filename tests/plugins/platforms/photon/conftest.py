"""Photon test isolation: no test may scan (and signal) the HOST's real process table."""
from __future__ import annotations

import pytest

from plugins.platforms.photon.adapter import PhotonAdapter


@pytest.fixture(autouse=True)
def _no_host_orphan_scan(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """``_start_sidecar`` reaps orphaned sidecars found via ``ps``; tests that fake the
    spawn must not kill a real orphan on the machine running them. Tests that exercise
    the scan itself request ``real_orphan_scan``."""
    if "real_orphan_scan" in request.fixturenames:
        return
    monkeypatch.setattr(PhotonAdapter, "_find_orphan_sidecar_pids", staticmethod(lambda: []))


@pytest.fixture
def real_orphan_scan() -> None:
    """Opt out of ``_no_host_orphan_scan``."""
