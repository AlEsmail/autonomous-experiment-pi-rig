# ABOUTME: Tests for the camera driver seam: the fake camera used for all off-Pi development
# ABOUTME: (control state, frame injection, power-cycle reset) and the real driver's refusal off-Pi.

import shutil
import subprocess
import types

import pytest

import rig_service.drivers.camera as camera_mod
from rig_service.drivers.fake import FakeCamera

HAS_V4L2 = shutil.which("v4l2-ctl") is not None

LOCKS = {
    "auto_exposure": 1,
    "exposure_time_absolute": 250,
    "gain": 32,
    "white_balance_automatic": 0,
    "white_balance_temperature": 4600,
    "focus_automatic_continuous": 0,
    "focus_absolute": 85,
    "backlight_compensation": 0,
}

# Firmware defaults measured on the BRIO 4K (046d:086b) after a cold power cycle
# on 2026-07-13: the three auto toggles and their dependent absolutes reset.
BRIO_DEFAULTS = {
    "auto_exposure": 3,
    "exposure_time_absolute": 250,
    "gain": 32,
    "white_balance_automatic": 1,
    "white_balance_temperature": 4000,
    "focus_automatic_continuous": 1,
    "focus_absolute": 0,
    "backlight_compensation": 0,
}


def test_starts_at_defaults():
    camera = FakeCamera(defaults=BRIO_DEFAULTS)
    assert camera.read_controls(BRIO_DEFAULTS.keys()) == BRIO_DEFAULTS


def test_apply_controls_round_trips_through_read():
    camera = FakeCamera(defaults=BRIO_DEFAULTS)
    camera.apply_controls(LOCKS)
    assert camera.read_controls(LOCKS.keys()) == LOCKS


def test_read_controls_returns_only_requested_names():
    camera = FakeCamera(defaults=BRIO_DEFAULTS)
    assert camera.read_controls(["gain"]) == {"gain": 32}


def test_capture_returns_injected_frame():
    camera = FakeCamera(defaults=BRIO_DEFAULTS, frame=b"jpeg-bytes")
    assert camera.capture_jpeg() == b"jpeg-bytes"


def test_power_cycle_resets_controls_to_defaults():
    camera = FakeCamera(defaults=BRIO_DEFAULTS)
    camera.apply_controls(LOCKS)
    camera.power_cycle()
    read = camera.read_controls(LOCKS.keys())
    assert read == BRIO_DEFAULTS
    assert read != LOCKS


@pytest.mark.skipif(HAS_V4L2, reason="with v4l2-ctl present the real driver loads")
def test_real_camera_refuses_to_load_without_v4l2():
    from rig_service.drivers.camera import V4l2Camera

    with pytest.raises(RuntimeError, match="v4l2-ctl"):
        V4l2Camera(device="/dev/video0", width=1920, height=1080)


def test_run_translates_timeout_expired_to_runtimeerror(monkeypatch):
    # Off-Pi there is no v4l2-ctl, so V4l2Camera.__init__ must be shown a truthy path
    # before the _run() behaviour under test can be exercised.
    monkeypatch.setattr(camera_mod.shutil, "which", lambda _name: "/usr/bin/v4l2-ctl")
    camera = camera_mod.V4l2Camera(device="/dev/video0", width=1920, height=1080)

    recorded = {}

    def fake_run(*args, **kwargs):
        recorded["kwargs"] = kwargs
        raise subprocess.TimeoutExpired(cmd=args, timeout=1.0)

    monkeypatch.setattr(camera_mod.subprocess, "run", fake_run)

    # A wedged v4l2-ctl must surface as a camera-level RuntimeError, not leak the raw
    # subprocess exception and hang the caller forever.
    with pytest.raises(RuntimeError):
        camera.capture_jpeg()

    # Must stay below the 60 s measure/blank phase budgets (config/rig.json) so a
    # stuck v4l2-ctl call returns before the phase watchdog fires.
    assert 0 < recorded["kwargs"]["timeout"] < 60


def test_run_bounds_every_subprocess_call(monkeypatch):
    # Even a normal control read must pass a bounded timeout to subprocess.run so a
    # stuck v4l2-ctl cannot hold the camera lock indefinitely.
    monkeypatch.setattr(camera_mod.shutil, "which", lambda _name: "/usr/bin/v4l2-ctl")
    camera = camera_mod.V4l2Camera(device="/dev/video0", width=1920, height=1080)

    recorded = {}

    def fake_run(*args, **kwargs):
        recorded["kwargs"] = kwargs
        return types.SimpleNamespace(stdout="")

    monkeypatch.setattr(camera_mod.subprocess, "run", fake_run)

    camera.read_controls(["gain"])

    # Must stay below the 60 s measure/blank phase budgets (config/rig.json) so a
    # stuck v4l2-ctl call returns before the phase watchdog fires.
    assert 0 < recorded["kwargs"]["timeout"] < 60


def test_timeout_error_names_the_failing_args(monkeypatch):
    # Off-Pi there is no v4l2-ctl, so V4l2Camera.__init__ must be shown a truthy path
    # before the _run() behaviour under test can be exercised.
    monkeypatch.setattr(camera_mod.shutil, "which", lambda _name: "/usr/bin/v4l2-ctl")
    camera = camera_mod.V4l2Camera(device="/dev/video0", width=1920, height=1080)

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=1.0)

    monkeypatch.setattr(camera_mod.subprocess, "run", fake_run)

    # A wedged v4l2-ctl call must name the failing invocation in the error so
    # on-rig diagnosis can distinguish set-ctrl, get-ctrl, and stream hangs.
    with pytest.raises(RuntimeError, match="gain"):
        camera.read_controls(["gain"])
