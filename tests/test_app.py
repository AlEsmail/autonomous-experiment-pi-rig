# ABOUTME: Service-level tests for the rig-service FastAPI app: boot-safe arming, mode
# ABOUTME: enforcement on /mode, /pump and /cycle, volume accounting/budget trips, pump caps,
# ABOUTME: target/lambda updates, the startup camera lock, the history visibility rule, and the
# ABOUTME: console frame/stream/static/route wiring.

import io
import json
import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from rig_service.app import create_app
from rig_service.config import CameraConfig, Channel, CycleConfig, RigConfig, VolumeLimits
from rig_service.drivers.fake import FakeCamera, FakeRelayDriver
from rig_service.frames import FrameSource
from rig_service.patch import MIN_PATCH_SIZE_PX, PatchSpec, validate_patch
from rig_service.store import Store
from rig_service.volume import VolumeState
from shared.color import hex_to_rgb, rgb_to_lab
from shared.measure import lab_from_jpeg

CONSOLE = {"X-Client": "console"}
SCIENTIST = {"X-Client": "scientist"}

# Camera "firmware defaults" a boot-time power loss would leave a real BRIO in --
# deliberately different from every config's lock set below, so CameraLock.apply()
# at startup is observable (non-vacuous), not a no-op.
FAKE_CAMERA_DEFAULTS = {
    "auto_exposure": 3,
    "exposure_time_absolute": 625,
    "gain": 16,
    "white_balance_automatic": 1,
    "white_balance_temperature": 4000,
    "focus_automatic_continuous": 1,
    "focus_absolute": 0,
    "backlight_compensation": 1,
    "power_line_frequency": 60,
}

FAST_LOCK_CONTROLS = {
    "auto_exposure": 1,
    "exposure_time_absolute": 250,
    "white_balance_automatic": 0,
    "white_balance_temperature": 4600,
    "focus_automatic_continuous": 0,
    "focus_absolute": 85,
}

FAST_CAMERA_DEFAULTS = {
    "auto_exposure": 3,
    "exposure_time_absolute": 625,
    "white_balance_automatic": 1,
    "white_balance_temperature": 4000,
    "focus_automatic_continuous": 1,
    "focus_absolute": 0,
}


def _solid_jpeg(color=(200, 60, 30), size=40):
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def make_fast_cycle_config(volumes=None, phase_budgets_s=None):
    """A small/fast RigConfig for HTTP-level /cycle tests: tiny cup/volumes and fast pump
    rates so a full 8-phase cycle completes in well under 5s -- no @pytest.mark.slow needed.
    Kept separate from make_config() below (the pump/mode tests' fixture),
    whose cycle.settle_s (10s, a real sleep) would make every cycle test slow."""
    camera = CameraConfig(
        device="/dev/video0", width=64, height=64, patch_px=20, controls=dict(FAST_LOCK_CONTROLS)
    )
    budgets = {
        "dose": 1.0,
        "refill": 1.0,
        "settle": 1.0,
        "measure": 1.0,
        "drain": 1.0,
        "rinse": 1.0,
        "blank": 1.0,
        "final_drain": 1.0,
    }
    if phase_budgets_s:
        budgets.update(phase_budgets_s)
    cycle = CycleConfig(
        refill_ml=10,
        rinse_ml=10,
        drain_ml=20,
        settle_s=0.01,
        blank_drift_delta_e=5.0,
        phase_budgets_s=budgets,
    )
    return RigConfig(
        channels={
            "A": Channel(gpio=4, role="dye", rate_ml_per_s=1000.0, lag_s=0.001, max_ml_per_dose=5),
            "B": Channel(gpio=5, role="dye", rate_ml_per_s=1000.0, lag_s=0.001, max_ml_per_dose=5),
            "D": Channel(
                gpio=6, role="drain", rate_ml_per_s=1000.0, lag_s=0.001, max_ml_per_dose=20
            ),
            "W": Channel(
                gpio=7, role="water", rate_ml_per_s=1000.0, lag_s=0.001, max_ml_per_dose=20
            ),
        },
        max_relay_on_s=1.0,
        volumes=volumes
        or VolumeLimits(
            cup_ceiling_ml=20, reservoir_ml=6000, waste_capacity_ml=6000, low_margin_ml=20
        ),
        camera=camera,
        cycle=cycle,
    )


def _make_store(tmp_path, volumes=None):
    store = Store(
        str(tmp_path / "rig.db"), str(tmp_path / "frames"), str(tmp_path / "backups")
    )
    if volumes is not None:
        store.set_volumes(volumes)
    return store


def _app_with_store(tmp_path, config=None, camera=None, driver=None, volumes=None, start_run=False):
    """Builds a create_app() instance plus its Store so tests can inspect durable rows."""
    store = _make_store(tmp_path, volumes=volumes)
    if start_run:
        store.start_run(
            target_hex="#3366cc",
            target_lab=rgb_to_lab(hex_to_rgb("#3366cc")),
            initial_lambda=0.0,
            cycle_budget=10,
            model="test",
        )
    app = create_app(
        config=config or make_fast_cycle_config(),
        driver=driver or FakeRelayDriver(),
        camera=camera or FakeCamera(defaults=FAST_CAMERA_DEFAULTS, frame=_solid_jpeg()),
        store=store,
    )
    return app, store


def build_app(tmp_path, config=None, camera=None, driver=None, start_run=False):
    """Assembles a create_app() instance for the /cycle-family tests, independent of
    the make_config()/client fixture the pump/mode tests use."""
    app, _store = _app_with_store(
        tmp_path, config=config, camera=camera, driver=driver,
        volumes=VolumeState(0.0, 0.0, 0.0), start_run=start_run,
    )
    return app


def _walk_payload(obj):
    """Yields every (key, value) pair anywhere in a nested dict/list JSON payload."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key, value
            yield from _walk_payload(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_payload(item)


def make_config(volumes=None, rate_ml_per_s=100.0, lag_s=0.01):
    # Local import: CameraConfig/CycleConfig live in config.py alongside the channel/volume
    # dataclasses.
    from rig_service.config import CameraConfig, CycleConfig

    camera = CameraConfig(
        device="/dev/video0",
        width=1920,
        height=1080,
        patch_px=200,
        controls={
            "auto_exposure": 1,
            "exposure_time_absolute": 250,
            "gain": 32,
            "white_balance_automatic": 0,
            "white_balance_temperature": 4600,
            "focus_automatic_continuous": 0,
            "focus_absolute": 85,
            "backlight_compensation": 0,
            "power_line_frequency": 50,
        },
    )
    # Budgets are generous relative to this fixture's pump rates. The dose budget
    # in particular must cover SLOW (0.1 ml/s, dye role): 10 ml / 0.1 ml/s = 100 s,
    # the worst-case leg of the A+B+SLOW dose-phase sum.
    cycle = CycleConfig(
        refill_ml=100,
        rinse_ml=100,
        drain_ml=150,
        settle_s=10,
        blank_drift_delta_e=5.0,
        phase_budgets_s={
            "dose": 150,
            "refill": 120,
            "settle": 30,
            "measure": 60,
            "drain": 180,
            "rinse": 120,
            "blank": 60,
            "final_drain": 180,
        },
    )
    return RigConfig(
        channels={
            "A": Channel(
                gpio=4, role="dye", rate_ml_per_s=rate_ml_per_s, lag_s=lag_s, max_ml_per_dose=10
            ),
            "B": Channel(
                gpio=5, role="dye", rate_ml_per_s=rate_ml_per_s, lag_s=lag_s, max_ml_per_dose=10
            ),
            "D": Channel(
                gpio=6,
                role="drain",
                rate_ml_per_s=rate_ml_per_s,
                lag_s=lag_s,
                max_ml_per_dose=150,
            ),
            "W": Channel(
                gpio=7,
                role="water",
                rate_ml_per_s=rate_ml_per_s,
                lag_s=lag_s,
                max_ml_per_dose=150,
            ),
            "SLOW": Channel(gpio=8, role="dye", rate_ml_per_s=0.1, lag_s=0.0, max_ml_per_dose=10),
        },
        max_relay_on_s=5.0,
        volumes=volumes
        or VolumeLimits(
            cup_ceiling_ml=150, reservoir_ml=6000, waste_capacity_ml=6000, low_margin_ml=300
        ),
        camera=camera,
        cycle=cycle,
    )


@pytest.fixture()
def driver():
    return FakeRelayDriver()


@pytest.fixture()
def camera():
    # Defaults differ from make_config()'s camera.controls lock set (see
    # FAKE_CAMERA_DEFAULTS above) so startup's CameraLock.apply() is observable.
    return FakeCamera(defaults=FAKE_CAMERA_DEFAULTS, frame=_solid_jpeg())


@pytest.fixture()
def client(driver, camera, tmp_path):
    store = _make_store(tmp_path, volumes=VolumeState(0.0, 0.0, 0.0))
    app = create_app(config=make_config(), driver=driver, camera=camera, store=store)
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def manual(client):
    client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
    return client


def pump(client, channel, ml, headers=CONSOLE):
    return client.post("/pump", json={"channel": channel, "ml": ml}, headers=headers)


def test_startup_arms_all_relays_off(driver, client):
    assert driver.history[0][:2] == ("all_off", None)


def test_status_reports_ok_and_boots_paused(client):
    response = client.get("/status")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["mode"] == "paused"


# --- mode endpoint ---


def test_console_sets_mode(client):
    response = client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
    assert response.status_code == 200
    assert response.json()["mode"] == "manual"
    assert client.get("/status").json()["mode"] == "manual"


def test_scientist_cannot_set_mode(client):
    response = client.post("/mode", json={"mode": "auto"}, headers=SCIENTIST)
    assert response.status_code == 409
    assert client.get("/status").json()["mode"] == "paused"


def test_mode_requires_client_header(client):
    assert client.post("/mode", json={"mode": "manual"}).status_code == 400


def test_invalid_mode_rejected(client):
    assert client.post("/mode", json={"mode": "turbo"}, headers=CONSOLE).status_code == 422


def test_entering_paused_arms_relays_off(driver, manual):
    events_before = len(driver.history)
    manual.post("/mode", json={"mode": "paused"}, headers=CONSOLE)
    assert ("all_off", None) in [e[:2] for e in driver.history[events_before:]]


# --- pump authorization ---


def test_pump_rejected_without_client_header(driver, manual):
    response = manual.post("/pump", json={"channel": "A", "ml": 2})
    assert response.status_code == 400
    assert len(driver.history) == 1


def test_pump_rejected_in_paused(driver, client):
    response = pump(client, "A", 2)
    assert response.status_code == 409
    assert "paused" in response.json()["detail"]
    assert len(driver.history) == 1


def test_pump_rejected_for_scientist(driver, manual):
    response = pump(manual, "A", 2, headers=SCIENTIST)
    assert response.status_code == 409
    assert len(driver.history) == 1


def test_pump_rejected_in_auto(driver, client):
    client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
    response = pump(client, "A", 2)
    assert response.status_code == 409
    assert len(driver.history) == 1


# --- pump behavior (console, manual) ---


def test_pump_fires_relay_on_then_off(driver, manual):
    response = pump(manual, "A", 2)
    assert response.status_code == 200
    body = response.json()
    assert body["channel"] == "A"
    assert body["ml"] == 2
    assert body["seconds"] == pytest.approx(0.03)
    assert [e[:2] for e in driver.history[1:]] == [("on", 4), ("off", 4)]
    assert driver.relays_on() == set()


def test_pump_on_duration_matches_dose_seconds(driver, manual):
    pump(manual, "A", 5)
    t_on = driver.history[1][2]
    t_off = driver.history[2][2]
    assert t_off - t_on == pytest.approx(0.06, abs=0.05)


def test_unknown_channel_rejected(driver, manual):
    response = pump(manual, "X", 1)
    assert response.status_code == 400
    assert "channel" in response.json()["detail"].lower()
    assert len(driver.history) == 1  # only the arming event — relay never touched


def test_dose_over_channel_cap_rejected(driver, manual):
    response = pump(manual, "A", 11)
    assert response.status_code == 400
    assert "cap" in response.json()["detail"].lower()
    assert len(driver.history) == 1


def test_dose_exceeding_relay_on_time_cap_rejected(driver, manual):
    # SLOW channel: 1 ml at 0.1 ml/s = 10 s > max_relay_on_s of 5
    response = pump(manual, "SLOW", 1)
    assert response.status_code == 400
    assert "on-time" in response.json()["detail"].lower()
    assert len(driver.history) == 1


def test_non_positive_ml_rejected(driver, manual):
    response = pump(manual, "A", 0)
    assert response.status_code == 400
    assert len(driver.history) == 1


def test_concurrent_pumps_never_overlap(driver, manual):
    def fire(channel):
        pump(manual, channel, 3)

    threads = [threading.Thread(target=fire, args=(c,)) for c in ("A", "B")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    on_now = set()
    max_simultaneous = 0
    for event, gpio, _ in driver.history:
        if event == "on":
            on_now.add(gpio)
        elif event == "off":
            on_now.discard(gpio)
        elif event == "all_off":
            on_now.clear()
        max_simultaneous = max(max_simultaneous, len(on_now))
    assert max_simultaneous <= 1


# --- volume accounting ---


def volumes(client):
    return client.get("/status", headers=CONSOLE).json()["volumes"]


def test_dye_dose_fills_cup_only(manual):
    pump(manual, "A", 5)
    assert volumes(manual) == {"cup_ml": 5, "water_drawn_ml": 0, "waste_ml": 0}


def test_water_dose_also_draws_from_reservoir(manual):
    pump(manual, "W", 100)
    assert volumes(manual) == {"cup_ml": 100, "water_drawn_ml": 100, "waste_ml": 0}


def test_drain_moves_cup_content_to_waste(manual):
    pump(manual, "W", 100)
    pump(manual, "D", 150)  # pump runs for 150 ml but only 100 ml is in the cup
    assert volumes(manual) == {"cup_ml": 0, "water_drawn_ml": 100, "waste_ml": 100}


def test_overfill_rejected_relay_untouched(driver, manual):
    pump(manual, "W", 140)
    events_before = len(driver.history)
    response = pump(manual, "W", 20)  # 140 + 20 > 150 ceiling
    assert response.status_code == 409
    assert "ceiling" in response.json()["detail"]
    assert volumes(manual)["cup_ml"] == 140
    assert [e[:2] for e in driver.history[events_before:] if e[0] == "on"] == []


def test_budget_low_trips_to_paused_with_alert(tmp_path):
    # Small staged reservoir: one 60 ml refill leaves 140 ml headroom, under the
    # 150 ml margin (which must cover the largest W/D dose for the config to load).
    # cup_ceiling_ml stays at the fixture's default (150, well over the 60 ml dose
    # under test) since make_config's cycle.drain_ml (150) must stay >= it and D's
    # per-dose cap is also 150.
    config = make_config(
        volumes=VolumeLimits(
            cup_ceiling_ml=150, reservoir_ml=200, waste_capacity_ml=6000, low_margin_ml=150
        )
    )
    driver = FakeRelayDriver()
    camera = FakeCamera(defaults=FAKE_CAMERA_DEFAULTS, frame=_solid_jpeg())
    store = _make_store(tmp_path, volumes=VolumeState(0.0, 0.0, 0.0))
    app = create_app(config=config, driver=driver, camera=camera, store=store)
    with TestClient(app) as app_client:
        app_client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
        response = pump(app_client, "W", 60)
        assert response.status_code == 200

        status = app_client.get("/status", headers=CONSOLE).json()
        assert status["mode"] == "paused"
        assert any("reservoir" in alert for alert in status["alerts"])
        assert "budget" in status["trip_reason"]
        assert ("all_off", None) in [e[:2] for e in driver.history[-2:]]

        # Doses stay rejected until the operator intervenes.
        assert pump(app_client, "W", 10).status_code == 409


# --- POST /cycle ---


def _cycle(
    client,
    headers,
    dose_a_ml=1.0,
    dose_b_ml=1.0,
    rationale=None,
    model_id=None,
    latency_s=None,
):
    body = {"dose_a_ml": dose_a_ml, "dose_b_ml": dose_b_ml}
    if rationale is not None:
        body["rationale"] = rationale
    if model_id is not None:
        body["model_id"] = model_id
    if latency_s is not None:
        body["latency_s"] = latency_s
    return client.post("/cycle", json=body, headers=headers)


def _decision_count(store, run_id=None):
    if run_id is not None:
        return store._conn.execute(
            "SELECT COUNT(*) AS c FROM decisions WHERE run_id = ?", (run_id,)
        ).fetchone()["c"]
    return store._conn.execute("SELECT COUNT(*) AS c FROM decisions").fetchone()["c"]


def test_cycle_scientist_in_auto_accepted(tmp_path):
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        response = _cycle(client, SCIENTIST, dose_a_ml=1.0, dose_b_ml=1.0)

        assert response.status_code == 200
        body = response.json()
        assert body["dose_a_ml"] == 1.0
        assert body["dose_b_ml"] == 1.0
        assert body["aborted"] is None
        assert body["score"] is not None
        assert body["delta_e"] is not None
        assert "target_lab" in body
        assert "measured_lab" in body
        assert "blank_lab" in body
        assert "phase_elapsed_s" not in body
        assert "alarms" in body


def test_cycle_console_in_manual_accepted(tmp_path):
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        response = _cycle(client, CONSOLE, dose_a_ml=1.0, dose_b_ml=1.0)

        assert response.status_code == 200
        assert response.json()["aborted"] is None


def test_cycle_rejected_paused_scientist_in_manual_console_in_auto(tmp_path):
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        # paused (boot default): nobody doses, regardless of client.
        assert _cycle(client, SCIENTIST).status_code == 409
        assert _cycle(client, CONSOLE).status_code == 409

        # manual: console commands cycles; scientist is rejected.
        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
        assert _cycle(client, SCIENTIST).status_code == 409

        # auto: scientist commands cycles; console is rejected.
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        assert _cycle(client, CONSOLE).status_code == 409


def test_cycle_without_run_409(tmp_path):
    app = build_app(tmp_path, start_run=False)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)
        # No active run has been started.
        response = _cycle(client, SCIENTIST)
        assert response.status_code == 409
        assert "active run" in response.json()["detail"].lower()


def test_cycle_while_running_409_busy(tmp_path):
    event = threading.Event()

    class BlockingCamera:
        """capture_jpeg blocks on an Event until released -- holds the engine mid-MEASURE
        so a concurrent second /cycle request can observe busy rejection (mirrors
        tests/test_cycle.py's BlockingCamera, adapted to the camera-lock interface)."""

        def __init__(self):
            self._defaults = dict(FAST_CAMERA_DEFAULTS)
            self._controls = dict(self._defaults)
            self._frame = _solid_jpeg()

        def apply_controls(self, controls):
            self._controls.update(controls)

        def read_controls(self, names):
            return {name: self._controls[name] for name in names}

        def capture_jpeg(self):
            event.wait(timeout=5)
            return self._frame

        def power_cycle(self):
            self._controls = dict(self._defaults)

    app = build_app(tmp_path, camera=BlockingCamera(), start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        result = {}

        def fire():
            result["response"] = _cycle(client, SCIENTIST)

        thread = threading.Thread(target=fire)
        thread.start()
        time.sleep(0.3)  # DOSE+REFILL+SETTLE (well under 0.1s) have run; MEASURE now blocks.

        busy_response = _cycle(client, SCIENTIST)
        assert busy_response.status_code == 409
        assert "running" in busy_response.json()["detail"].lower()

        event.set()
        thread.join(timeout=5)
        assert result["response"].status_code == 200
        assert result["response"].json()["aborted"] is None


def test_cycle_negative_dose_rejected_400(tmp_path):
    # Shared design decision 12: validation failures -> 400 (matching /pump's identical
    # PumpError -> 400 mapping), distinct from mode/state conflicts -> 409. A negative
    # dose is a pure validation fault -- nothing has moved a relay -- so it belongs on
    # the 400 side, not lumped in with busy/non-empty-cup/no-target 409s.
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        response = _cycle(client, SCIENTIST, dose_a_ml=-1.0, dose_b_ml=1.0)

        assert response.status_code == 400
        assert "negative" in response.json()["detail"].lower()


def test_cycle_over_channel_cap_rejected_400(tmp_path):
    # Same decision 12 mapping: the PumpError-wrapped "preflight rejected: ..." cap
    # rejection is a validation fault (identical to /pump's PumpError -> 400), not a
    # mode/state conflict.
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        response = _cycle(client, SCIENTIST, dose_a_ml=6.0, dose_b_ml=1.0)  # A's cap is 5 ml

        assert response.status_code == 400
        assert "cap" in response.json()["detail"].lower()


def test_cycle_nonempty_cup_stays_409(tmp_path):
    # Regression guard: a state/trip conflict (decision 12), not a validation fault --
    # must stay 409 even after CycleValidationError -> 400 lands.
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
        pump(client, "A", 1.0)  # leaves the cup non-empty without draining it
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        response = _cycle(client, SCIENTIST)

        assert response.status_code == 409
        assert "cup" in response.json()["detail"].lower()


def test_cycle_low_session_headroom_stays_409(tmp_path):
    # Regression guard: the headroom trip is a state/trip conflict (decision 12,
    # matching /pump's VolumeError -> 409), not a validation fault -- must stay 409.
    low_volumes = VolumeLimits(
        cup_ceiling_ml=20, reservoir_ml=30, waste_capacity_ml=6000, low_margin_ml=20
    )
    config = make_fast_cycle_config(volumes=low_volumes)
    app = build_app(tmp_path, config=config, start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        response = _cycle(client, SCIENTIST, dose_a_ml=1.0, dose_b_ml=1.0)

        assert response.status_code == 409
        detail = response.json()["detail"].lower()
        assert "budget" in detail
        assert "reservoir" in detail


def test_cycle_response_has_no_frame_bytes(tmp_path):
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        response = _cycle(client, SCIENTIST)

        assert response.status_code == 200
        assert b"\xff\xd8\xff" not in response.content  # JPEG SOI/APP0 marker
        body = response.json()
        assert "measurement_jpeg" not in body
        assert "blank_jpeg" not in body


def test_cycle_with_default_seed_matches_centered_behavior(tmp_path):
    # T3 regression net: the wired config seed (patch_center default (0.5, 0.5))
    # reproduces today's hard-centered measurement end-to-end.
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        response = _cycle(client, SCIENTIST)

        assert response.status_code == 200
        body = response.json()
        assert body["aborted"] is None
        measured = tuple(body["measured_lab"])
        blank = tuple(body["blank_lab"])

        # The default FakeCamera frame is _solid_jpeg(); the default config seed
        # is centered, so the cycle's measured/blank LABs must equal an explicit
        # centered reading of that same frame.
        expected_centered = lab_from_jpeg(_solid_jpeg(), 0.5, 0.5, 20)
        assert measured == pytest.approx(expected_centered, abs=0.1)
        assert blank == pytest.approx(expected_centered, abs=0.1)


def test_cycle_and_status_never_expose_executed_doses(tmp_path):
    # The history visibility contract (spec invariant 3), starting before any skew
    # mechanism exists: no payload built from a CycleRecord may name executed doses, and
    # the dose fields it does expose must equal the requested values.
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        cycle_response = _cycle(client, SCIENTIST, dose_a_ml=1.5, dose_b_ml=0.5)
        assert cycle_response.status_code == 200
        cycle_body = cycle_response.json()
        status_body = client.get("/status").json()

        for payload in (cycle_body, status_body):
            for key, _value in _walk_payload(payload):
                assert "executed" not in key.lower(), (key, payload)

        assert cycle_body["dose_a_ml"] == 1.5
        assert cycle_body["dose_b_ml"] == 0.5
        last = status_body["cycle"]["last"]
        assert last["dose_a_ml"] == 1.5
        assert last["dose_b_ml"] == 0.5


# --- POST /target, POST /lambda ---


def test_target_console_only_and_bad_hex_400(tmp_path):
    app = build_app(tmp_path, start_run=False)
    with TestClient(app) as client:
        # paused (boot default): console may still set target -- a legitimate mid-run
        # world change is allowed in any mode.
        response = client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)
        assert response.status_code == 200
        assert client.get("/status").json()["target_hex"] == "#3366cc"

        # console accepted in auto too (any mode).
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        response_in_auto = client.post("/target", json={"hex": "#112233"}, headers=CONSOLE)
        assert response_in_auto.status_code == 200
        assert client.get("/status").json()["target_hex"] == "#112233"

        scientist_rejected = client.post("/target", json={"hex": "#445566"}, headers=SCIENTIST)
        assert scientist_rejected.status_code == 409

        bad_hex = client.post("/target", json={"hex": "not-a-color"}, headers=CONSOLE)
        assert bad_hex.status_code == 400


def test_lambda_console_only_and_negative_400(tmp_path):
    app = build_app(tmp_path, start_run=False)
    with TestClient(app) as client:
        assert client.get("/status").json()["lambda"] == 0.0  # default

        response = client.post("/lambda", json={"value": 0.2}, headers=CONSOLE)
        assert response.status_code == 200
        assert client.get("/status").json()["lambda"] == pytest.approx(0.2)

        rejected = client.post("/lambda", json={"value": 0.3}, headers=SCIENTIST)
        assert rejected.status_code == 409

        negative = client.post("/lambda", json={"value": -0.1}, headers=CONSOLE)
        assert negative.status_code == 400


# --- POST /patch (measurement region) ---


def test_patch_requires_client_header(tmp_path):
    app = build_app(tmp_path, start_run=False)
    with TestClient(app) as client:
        response = client.post("/patch", json={"cx": 0.5, "cy": 0.5, "size_px": 30})
        assert response.status_code == 400
        assert "client" in response.json()["detail"].lower()


def test_patch_rejects_non_console_client(tmp_path):
    app = build_app(tmp_path, start_run=False)
    with TestClient(app) as client:
        response = client.post("/patch", json={"cx": 0.5, "cy": 0.5, "size_px": 30}, headers=SCIENTIST)
        assert response.status_code == 409
        assert "console" in response.json()["detail"].lower()


@pytest.mark.parametrize("cx, cy", [
    (-0.1, 0.5),
    (1.1, 0.5),
    (0.5, -0.1),
    (0.5, 1.1),
])
def test_patch_rejects_center_outside_interval(tmp_path, cx, cy):
    app = build_app(tmp_path, start_run=False)
    with TestClient(app) as client:
        response = client.post(
            "/patch", json={"cx": cx, "cy": cy, "size_px": MIN_PATCH_SIZE_PX}, headers=CONSOLE
        )
        assert response.status_code == 400
        assert "center" in response.json()["detail"].lower()


def test_patch_rejects_size_below_floor(tmp_path):
    app = build_app(tmp_path, start_run=False)
    with TestClient(app) as client:
        too_small = client.post(
            "/patch",
            json={"cx": 0.5, "cy": 0.5, "size_px": MIN_PATCH_SIZE_PX - 1},
            headers=CONSOLE,
        )
        assert too_small.status_code == 400
        assert "floor" in too_small.json()["detail"].lower()

        ok = client.post(
            "/patch",
            json={"cx": 0.5, "cy": 0.5, "size_px": MIN_PATCH_SIZE_PX},
            headers=CONSOLE,
        )
        assert ok.status_code == 200


@pytest.mark.parametrize("cx, cy", [
    (0.0, 0.5),  # left edge: box hangs off the left
    (1.0, 0.5),  # right edge: box hangs off the right
    (0.5, 0.0),  # top edge: box hangs off the top
    (0.5, 1.0),  # bottom edge: box hangs off the bottom
])
def test_patch_rejects_box_not_wholly_inside_frame(tmp_path, cx, cy):
    app = build_app(tmp_path, start_run=False)
    with TestClient(app) as client:
        response = client.post(
            "/patch", json={"cx": cx, "cy": cy, "size_px": MIN_PATCH_SIZE_PX}, headers=CONSOLE
        )
        assert response.status_code == 400
        assert "wholly" in response.json()["detail"].lower()


def test_patch_success_rebinds_and_returns(tmp_path):
    app = build_app(tmp_path, start_run=False)
    with TestClient(app) as client:
        response = client.post(
            "/patch", json={"cx": 0.4, "cy": 0.4, "size_px": 30}, headers=CONSOLE
        )
        assert response.status_code == 200
        body = response.json()
        assert body["cx"] == pytest.approx(0.4)
        assert body["cy"] == pytest.approx(0.4)
        assert body["size_px"] == 30


def test_patch_reflected_on_status(tmp_path):
    app = build_app(tmp_path, start_run=False)
    with TestClient(app) as client:
        client.post("/patch", json={"cx": 0.4, "cy": 0.4, "size_px": 30}, headers=CONSOLE)
        status = client.get("/status").json()
        assert status["patch"]["cx"] == pytest.approx(0.4)
        assert status["patch"]["cy"] == pytest.approx(0.4)
        assert status["patch"]["size_px"] == 30

        # The patch key is required for ALL viewers, not only the scientist/viewer path.
        console_status = client.get("/status", headers=CONSOLE).json()
        assert console_status["patch"]["cx"] == pytest.approx(0.4)
        assert console_status["patch"]["cy"] == pytest.approx(0.4)
        assert console_status["patch"]["size_px"] == 30


def test_patch_rejection_does_not_persist(tmp_path):
    app, store = _app_with_store(tmp_path, volumes=VolumeState(0.0, 0.0, 0.0))
    baseline = PatchSpec(cx=0.4, cy=0.4, size_px=30)
    with TestClient(app) as client:
        valid = client.post(
            "/patch", json={"cx": 0.4, "cy": 0.4, "size_px": 30}, headers=CONSOLE
        )
        assert valid.status_code == 200
        assert store.load_patch() == baseline

        invalid = client.post(
            "/patch",
            json={"cx": 0.5, "cy": 0.5, "size_px": MIN_PATCH_SIZE_PX - 1},
            headers=CONSOLE,
        )
        assert invalid.status_code == 400
        assert "floor" in invalid.json()["detail"].lower()
        assert store.load_patch() == baseline


def test_patch_change_event_scientist_visible(tmp_path):
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        client.post("/patch", json={"cx": 0.4, "cy": 0.4, "size_px": 30}, headers=CONSOLE)
        run_id = client.get("/status").json()["run"]["id"]
        history = client.get(f"/history?run={run_id}", headers=SCIENTIST).json()
        events = [e for e in history["events"] if e["kind"] == "patch_change"]
        assert len(events) == 1
        assert events[0]["detail"]["cx"] == pytest.approx(0.4)
        assert events[0]["detail"]["cy"] == pytest.approx(0.4)
        assert events[0]["detail"]["size_px"] == 30
        assert events[0]["scientist_visible"] == 1


def test_patch_persisted_and_rehydrated_across_restart(tmp_path):
    config = make_fast_cycle_config()
    db_path = tmp_path / "rig.db"
    frames_dir = tmp_path / "frames"
    backups_dir = tmp_path / "backups"

    store1 = Store(str(db_path), str(frames_dir), str(backups_dir))
    store1.set_volumes(VolumeState(0.0, 0.0, 0.0))
    app1 = create_app(
        config=config,
        driver=FakeRelayDriver(),
        camera=FakeCamera(defaults=FAST_CAMERA_DEFAULTS, frame=_solid_jpeg()),
        store=store1,
    )
    with TestClient(app1) as client:
        response = client.post(
            "/patch", json={"cx": 0.4, "cy": 0.4, "size_px": 30}, headers=CONSOLE
        )
        assert response.status_code == 200
    store1.close()

    store2 = Store(str(db_path), str(frames_dir), str(backups_dir))
    app2 = create_app(
        config=config,
        driver=FakeRelayDriver(),
        camera=FakeCamera(defaults=FAST_CAMERA_DEFAULTS, frame=_solid_jpeg()),
        store=store2,
    )
    with TestClient(app2) as client:
        status = client.get("/status").json()
        assert status["patch"]["cx"] == pytest.approx(0.4)
        assert status["patch"]["cy"] == pytest.approx(0.4)
        assert status["patch"]["size_px"] == 30


def test_boot_seeds_patch_from_config_on_fresh_db(tmp_path):
    app = build_app(tmp_path, start_run=False)
    with TestClient(app) as client:
        status = client.get("/status").json()
        assert status["patch"]["cx"] == pytest.approx(0.5)
        assert status["patch"]["cy"] == pytest.approx(0.5)
        assert status["patch"]["size_px"] == 20


def test_boot_falls_back_to_seed_when_persisted_patch_no_longer_fits(tmp_path, caplog):
    config = make_fast_cycle_config()
    store = _make_store(tmp_path, volumes=VolumeState(0.0, 0.0, 0.0))
    # 64x64 frame; a 100 px box cannot fit, so boot must fall back to the config seed.
    store.set_patch(PatchSpec(cx=0.5, cy=0.5, size_px=100))

    with caplog.at_level(logging.WARNING, logger="rig_service.app"):
        app = create_app(
            config=config,
            driver=FakeRelayDriver(),
            camera=FakeCamera(defaults=FAST_CAMERA_DEFAULTS, frame=_solid_jpeg()),
            store=store,
        )

    with TestClient(app) as client:
        status = client.get("/status").json()
        assert status["patch"]["cx"] == pytest.approx(0.5)
        assert status["patch"]["cy"] == pytest.approx(0.5)
        assert status["patch"]["size_px"] == 20

    assert "no longer fits" in caplog.text
    assert "using config seed" in caplog.text


def test_boot_uses_fitting_persisted_patch_without_warning(tmp_path, caplog):
    config = make_fast_cycle_config()
    store = _make_store(tmp_path, volumes=VolumeState(0.0, 0.0, 0.0))
    store.set_patch(PatchSpec(cx=0.4, cy=0.4, size_px=20))

    with caplog.at_level(logging.WARNING, logger="rig_service.app"):
        app = create_app(
            config=config,
            driver=FakeRelayDriver(),
            camera=FakeCamera(defaults=FAST_CAMERA_DEFAULTS, frame=_solid_jpeg()),
            store=store,
        )

    with TestClient(app) as client:
        status = client.get("/status").json()
        assert status["patch"]["cx"] == pytest.approx(0.4)
        assert status["patch"]["cy"] == pytest.approx(0.4)
        assert status["patch"]["size_px"] == 20

    assert "no longer fits" not in caplog.text


# --- GET /status: cycle phase/last, alarms, target_hex, lambda ---


def test_status_reports_cycle_phase_target_lambda_and_alarms(tmp_path):
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        boot_status = client.get("/status", headers=CONSOLE).json()
        assert boot_status["cycle"]["phase"] == "idle"
        assert boot_status["cycle"]["last"] is None
        assert boot_status["alarms"] == []
        assert boot_status["target_hex"] == "#3366cc"
        assert boot_status["lambda"] == 0.0

        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)
        client.post("/lambda", json={"value": 0.1}, headers=CONSOLE)
        response = _cycle(client, SCIENTIST)
        assert response.status_code == 200

        status_after = client.get("/status", headers=CONSOLE).json()
        assert status_after["cycle"]["phase"] == "idle"
        assert status_after["cycle"]["last"]["aborted"] is None
        assert status_after["target_hex"] == "#3366cc"
        assert status_after["lambda"] == pytest.approx(0.1)
        assert status_after["alarms"] == []  # clean camera, locked controls -- no alarms


# --- Startup camera lock ---


def test_startup_applies_camera_lock(tmp_path):
    config = make_fast_cycle_config()
    camera = FakeCamera(defaults=FAST_CAMERA_DEFAULTS, frame=_solid_jpeg())
    camera.power_cycle()  # simulate a power loss before boot (non-vacuous by construction:
    # FAST_CAMERA_DEFAULTS already differs from config.camera.controls, the lock set).
    assert camera.read_controls(config.camera.controls.keys()) != config.camera.controls

    store = _make_store(tmp_path)
    app = create_app(config=config, driver=FakeRelayDriver(), camera=camera, store=store)
    with TestClient(app):
        pass  # lifespan startup runs CameraLock.apply() after relay arming.

    current = camera.read_controls(config.camera.controls.keys())
    assert current == config.camera.controls


# --- Watchdog trip surfaces in status ---


def test_watchdog_trip_surfaces_in_status(tmp_path):
    class HangingCamera:
        """capture_jpeg hangs past the (tiny) measure budget -- forces a real watchdog
        timeout so the resulting paused mode + trip_reason are observable via /status."""

        def __init__(self):
            self._defaults = dict(FAST_CAMERA_DEFAULTS)
            self._controls = dict(self._defaults)
            self._frame = _solid_jpeg()

        def apply_controls(self, controls):
            self._controls.update(controls)

        def read_controls(self, names):
            return {name: self._controls[name] for name in names}

        def capture_jpeg(self):
            time.sleep(0.3)
            return self._frame

        def power_cycle(self):
            self._controls = dict(self._defaults)

    config = make_fast_cycle_config(phase_budgets_s={"measure": 0.05})
    app = build_app(tmp_path, config=config, camera=HangingCamera(), start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        response = _cycle(client, SCIENTIST)
        assert response.status_code == 200
        assert response.json()["aborted"] is not None

        status = client.get("/status").json()
        assert status["mode"] == "paused"
        assert status["trip_reason"] is not None
        assert "measure" in status["trip_reason"].lower()
        assert "measure" in status["cycle"]["last"]["aborted"].lower()


# --- lambda->score HTTP wiring, engine.alarms merge, target_lab fidelity ---


def test_lambda_reaches_cycle_score(tmp_path):
    # Pins that /cycle threads state["lam"] into engine.run_cycle (a dropped-lambda
    # mutation -- state["lam"] -> 0.0 -- would make this fail).
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)
        client.post("/lambda", json={"value": 0.5}, headers=CONSOLE)

        response = _cycle(client, SCIENTIST, dose_a_ml=1.0, dose_b_ml=1.0)

        assert response.status_code == 200
        body = response.json()
        assert body["score"] == pytest.approx(body["delta_e"] + 0.5 * (1.0 + 1.0))


def test_status_alarms_include_engine_alarms(tmp_path):
    # Mirrors tests/test_cycle.py's PowerCyclableCamera pattern, adapted to the HTTP
    # camera-lock interface: startup locks the camera, then a simulated power loss
    # reverts it before BLANK's re-read, so CameraLock.verify() sees a mismatch and
    # engine.alarms becomes non-empty. Pins that /status merges accountant.alerts()
    # + engine.alarms (a mutation dropping the `+ engine.alarms` term must fail this).
    class PowerCyclableCamera:
        def __init__(self):
            self._defaults = dict(FAST_CAMERA_DEFAULTS)
            self._controls = dict(self._defaults)
            self._frame = _solid_jpeg()

        def apply_controls(self, controls):
            self._controls.update(controls)

        def read_controls(self, names):
            return {name: self._controls[name] for name in names}

        def capture_jpeg(self):
            return self._frame

        def power_cycle(self):
            self._controls = dict(self._defaults)

    camera = PowerCyclableCamera()
    app = build_app(tmp_path, camera=camera, start_run=True)
    with TestClient(app) as client:
        # Lifespan startup has just locked the camera (apply_controls ran); simulate a
        # power loss before BLANK's re-read.
        camera.power_cycle()

        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        response = _cycle(client, SCIENTIST)
        assert response.status_code == 200
        assert response.json()["aborted"] is None  # decision 8: alarms never pause

        status = client.get("/status", headers=CONSOLE).json()
        assert status["mode"] != "paused"
        assert any("mismatch" in alarm for alarm in status["alarms"])


def test_status_alarms_include_accountant_alerts(tmp_path):
    # Mirrors test_status_alarms_include_engine_alarms but for the OTHER half of the
    # "alarms": accountant.alerts() + engine.alarms merge in app.py. Drives the
    # accountant into a low-reservoir state via /pump alone -- no cycle ever runs, so
    # engine.alarms stays [] -- so if the `accountant.alerts() +` term were dropped
    # from the merge, "alarms" would stay empty while "alerts" is non-empty.
    config = make_config(
        volumes=VolumeLimits(
            cup_ceiling_ml=150, reservoir_ml=200, waste_capacity_ml=6000, low_margin_ml=150
        )
    )
    driver = FakeRelayDriver()
    camera = FakeCamera(defaults=FAKE_CAMERA_DEFAULTS, frame=_solid_jpeg())
    store = _make_store(tmp_path, volumes=VolumeState(0.0, 0.0, 0.0))
    app = create_app(config=config, driver=driver, camera=camera, store=store)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
        pump(client, "W", 60)  # reservoir headroom 200-60=140 <= 150 margin -> trips paused

        status = client.get("/status", headers=CONSOLE).json()
        assert status["alerts"] != []
        assert any("reservoir" in alert for alert in status["alerts"])
        assert status["alarms"] == status["alerts"]


def test_status_scientist_view_keeps_engine_alarms(tmp_path):
    # Kills a hardcoded-[] scientist "alarms" redaction: app.py's scientist branch
    # sets response["alarms"] = engine.alarms (no accountant merge, unlike console's
    # accountant.alerts() + engine.alarms). Per spec decision 6, engine alarms
    # (camera-lock mismatch, blank drift) ARE scientist-visible; decision 8 says
    # alarms never pause the run. Uses the same PowerCyclableCamera pattern as
    # test_status_alarms_include_engine_alarms to produce a real camera-lock
    # mismatch alarm, then also trips a reservoir accountant alert via /pump so we
    # can prove the scientist alarms carries the engine alarm but NOT the
    # accountant alert -- i.e. engine-alarms-ONLY, not accountant+engine and not a
    # hardcoded empty list.
    class PowerCyclableCamera:
        def __init__(self):
            self._defaults = dict(FAST_CAMERA_DEFAULTS)
            self._controls = dict(self._defaults)
            self._frame = _solid_jpeg()

        def apply_controls(self, controls):
            self._controls.update(controls)

        def read_controls(self, names):
            return {name: self._controls[name] for name in names}

        def capture_jpeg(self):
            return self._frame

        def power_cycle(self):
            self._controls = dict(self._defaults)

    config = make_fast_cycle_config(
        volumes=VolumeLimits(
            cup_ceiling_ml=20, reservoir_ml=50, waste_capacity_ml=6000, low_margin_ml=20
        )
    )
    camera = PowerCyclableCamera()
    app, _store = _app_with_store(
        tmp_path,
        config=config,
        camera=camera,
        volumes=VolumeState(0.0, 0.0, 0.0),
        start_run=True,
    )
    with TestClient(app) as client:
        # Lifespan startup has just locked the camera (apply_controls ran); simulate a
        # power loss before BLANK's re-read so CameraLock.verify() sees a mismatch.
        camera.power_cycle()

        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        response = _cycle(client, SCIENTIST, dose_a_ml=1.0, dose_b_ml=1.0)
        assert response.status_code == 200
        assert response.json()["aborted"] is None  # decision 8: alarms never pause

        # refill(10) + rinse(10) = 20 ml drawn from the reservoir by the cycle above;
        # +15 ml here leaves headroom 50-35=15 <= 20 margin -> trips paused with a
        # reservoir accountant alert (same math as
        # test_status_scientist_view_omits_volumes_relays_alerts_and_timings).
        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
        pump(client, "W", 15)
        con_status = client.get("/status", headers=CONSOLE).json()
        assert con_status["mode"] == "paused"
        assert any("reservoir" in alert for alert in con_status["alerts"])

        sci = client.get("/status").json()
        assert any("mismatch" in alarm for alarm in sci["alarms"])
        assert not any("reservoir" in alarm for alarm in sci["alarms"])
        assert "volumes" not in sci
        assert "relays_on" not in sci
        assert "alerts" not in sci


def test_target_lab_matches_conversion(tmp_path):
    # Pins /target's stored target_lab against the real conversion (a constant or
    # otherwise-wrong stored target_lab is caught, not just "not None").
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        hex_value = "#3366cc"
        client.post("/target", json={"hex": hex_value}, headers=CONSOLE)

        response = _cycle(client, SCIENTIST)
        assert response.status_code == 200
        body = response.json()

        expected_lab = list(rgb_to_lab(hex_to_rgb(hex_value)))
        assert list(body["target_lab"]) == pytest.approx(expected_lab)

        status_last = client.get("/status").json()["cycle"]["last"]
        assert list(status_last["target_lab"]) == pytest.approx(expected_lab)


# --- T5: run lifecycle, /history, events, rehydration ---


def _run_start_json(target_hex="#3366cc", lambda_=0.0, cycle_budget=10, model="test"):
    return {
        "target_hex": target_hex,
        "lambda": lambda_,
        "cycle_budget": cycle_budget,
        "model": model,
    }


def test_run_start_creates_run_sets_target_and_lambda(tmp_path):
    app = build_app(tmp_path, start_run=False)
    with TestClient(app) as client:
        response = client.post(
            "/run/start",
            json=_run_start_json(target_hex="#ff5733", lambda_=0.25, cycle_budget=12, model="claude-opus"),
            headers=CONSOLE,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["run_id"] == 1

        status = client.get("/status").json()
        run = status["run"]
        assert run is not None
        assert run["id"] == 1
        assert run["target_hex"] == "#ff5733"
        assert run["cycle_budget"] == 12
        assert run["model"] == "claude-opus"
        assert run["cycles"] == 0
        assert status["target_hex"] == "#ff5733"
        assert status["lambda"] == pytest.approx(0.25)


def test_run_start_second_run_409(tmp_path):
    app = build_app(tmp_path, start_run=False)
    with TestClient(app) as client:
        first = client.post("/run/start", json=_run_start_json(target_hex="#3366cc"), headers=CONSOLE)
        assert first.status_code == 200

        second = client.post("/run/start", json=_run_start_json(target_hex="#112233"), headers=CONSOLE)
        assert second.status_code == 409
        assert "already active" in second.json()["detail"].lower()


def test_run_start_scientist_409(tmp_path):
    app = build_app(tmp_path, start_run=False)
    with TestClient(app) as client:
        response = client.post("/run/start", json=_run_start_json(), headers=SCIENTIST)
        assert response.status_code == 409
        assert "console" in response.json()["detail"].lower()


def test_run_start_bad_hex_400(tmp_path):
    app = build_app(tmp_path, start_run=False)
    with TestClient(app) as client:
        response = client.post(
            "/run/start",
            json={"target_hex": "not-a-color", "lambda": 0.0, "cycle_budget": 10, "model": "test"},
            headers=CONSOLE,
        )
        assert response.status_code == 400
        assert "hex" in response.json()["detail"].lower()


def test_run_end_closes_run_and_returns_backup_result(tmp_path):
    app, store = _app_with_store(tmp_path, start_run=False)
    with TestClient(app) as client:
        start = client.post("/run/start", json=_run_start_json(), headers=CONSOLE)
        run_id = start.json()["run_id"]

        response = client.post("/run/end", json={"outcome": "ok"}, headers=CONSOLE)
        assert response.status_code == 200
        body = response.json()
        assert body["run_id"] == run_id
        backup = body["backup"]
        assert backup["integrity_ok"] is True
        assert backup["errors"] == []
        assert Path(backup["folder"]).exists()

        status = client.get("/status").json()
        assert status["run"] is None


def test_run_end_without_active_run_409(tmp_path):
    app = build_app(tmp_path, start_run=False)
    with TestClient(app) as client:
        response = client.post("/run/end", json={}, headers=CONSOLE)
        assert response.status_code == 409
        assert "active run" in response.json()["detail"].lower()


def test_run_end_rejected_while_cycle_running_then_ok_when_idle(tmp_path):
    event = threading.Event()

    class BlockingCamera:
        """capture_jpeg blocks on an Event until released -- holds the engine mid-MEASURE
        so POST /run/end can observe the busy rejection, then the run ends cleanly once
        the engine returns to idle."""

        def __init__(self):
            self._defaults = dict(FAST_CAMERA_DEFAULTS)
            self._controls = dict(self._defaults)
            self._frame = _solid_jpeg()

        def apply_controls(self, controls):
            self._controls.update(controls)

        def read_controls(self, names):
            return {name: self._controls[name] for name in names}

        def capture_jpeg(self):
            event.wait(timeout=5)
            return self._frame

        def power_cycle(self):
            self._controls = dict(self._defaults)

    app, store = _app_with_store(
        tmp_path, camera=BlockingCamera(), volumes=VolumeState(0.0, 0.0, 0.0), start_run=True
    )
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        result = {}

        def fire():
            result["response"] = _cycle(client, SCIENTIST)

        thread = threading.Thread(target=fire)
        thread.start()
        time.sleep(0.3)  # DOSE+REFILL+SETTLE run; MEASURE now blocks.

        busy_response = client.post("/run/end", json={"outcome": "ok"}, headers=CONSOLE)
        assert busy_response.status_code == 409
        detail = busy_response.json()["detail"]
        assert "cycle" in detail.lower()
        assert "pause" in detail.lower()

        status = client.get("/status").json()
        assert status["run"] is not None
        assert not store._backup_root.exists() or len(list(store._backup_root.iterdir())) == 0

        event.set()
        thread.join(timeout=5)
        assert result["response"].status_code == 200
        assert result["response"].json()["aborted"] is None

        end_response = client.post("/run/end", json={"outcome": "ok"}, headers=CONSOLE)
        assert end_response.status_code == 200
        assert end_response.json()["run_id"] == status["run"]["id"]
        assert Path(end_response.json()["backup"]["folder"]).exists()

        status_after = client.get("/status").json()
        assert status_after["run"] is None


def test_cycle_persists_row_via_http(tmp_path):
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)
        response = _cycle(client, SCIENTIST, dose_a_ml=1.5, dose_b_ml=0.5)
        assert response.status_code == 200

        status = client.get("/status").json()
        run_id = status["run"]["id"]
        history = client.get(f"/history?run={run_id}", headers=CONSOLE).json()
        assert len(history["cycles"]) == 1
        cycle = history["cycles"][0]
        assert cycle["n"] == 1
        assert cycle["requested_a_ml"] == 1.5
        assert cycle["requested_b_ml"] == 0.5
        assert cycle["executed_a_ml"] == 1.5
        assert cycle["executed_b_ml"] == 0.5
        assert cycle["delta_e"] is not None
        assert cycle["score"] is not None
        assert cycle["completed"] == 1


def test_status_reports_run_and_cycle_count(tmp_path):
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)
        _cycle(client, SCIENTIST)

        status = client.get("/status").json()
        run = status["run"]
        assert run is not None
        assert run["id"] == 1
        assert run["cycles"] == 1


def test_mode_change_writes_visible_event(tmp_path):
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
        run_id = client.get("/status").json()["run"]["id"]
        history = client.get(f"/history?run={run_id}", headers=CONSOLE).json()
        events = [e for e in history["events"] if e["kind"] == "mode_change"]
        assert len(events) == 1
        assert events[0]["detail"]["mode"] == "manual"
        assert events[0]["scientist_visible"] == 1


def test_target_and_lambda_write_visible_events(tmp_path):
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        client.post("/target", json={"hex": "#112233"}, headers=CONSOLE)
        client.post("/lambda", json={"value": 0.42}, headers=CONSOLE)
        run_id = client.get("/status").json()["run"]["id"]
        history = client.get(f"/history?run={run_id}", headers=CONSOLE).json()
        targets = [e for e in history["events"] if e["kind"] == "target_change"]
        lambdas = [e for e in history["events"] if e["kind"] == "lambda_change"]
        assert len(targets) == 1
        assert targets[0]["detail"]["hex"] == "#112233"
        assert targets[0]["scientist_visible"] == 1
        assert len(lambdas) == 1
        assert lambdas[0]["detail"]["value"] == pytest.approx(0.42)
        assert lambdas[0]["scientist_visible"] == 1


def test_pump_writes_operator_event_with_volumes_in_same_tx(tmp_path):
    app, store = _app_with_store(
        tmp_path, config=make_config(), volumes=VolumeState(0.0, 0.0, 0.0), start_run=False
    )
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
        response = pump(client, "W", 50)
        assert response.status_code == 200

        row = store._conn.execute(
            "SELECT detail FROM events WHERE kind = 'pump' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        detail = json.loads(row["detail"])
        assert detail["channel"] == "W"
        assert detail["ml"] == 50

        volumes = store.load_volumes()
        assert volumes.cup_ml == 50.0
        assert volumes.water_drawn_ml == 50.0
        assert volumes.waste_ml == 0.0


def test_trip_writes_event(tmp_path):
    config = make_config(
        volumes=VolumeLimits(
            cup_ceiling_ml=150, reservoir_ml=200, waste_capacity_ml=6000, low_margin_ml=150
        )
    )
    app, store = _app_with_store(
        tmp_path, config=config, volumes=VolumeState(0.0, 0.0, 0.0), start_run=False
    )
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
        pump(client, "W", 60)
        status = client.get("/status").json()
        assert status["mode"] == "paused"
        assert "budget" in status["trip_reason"].lower()

        row = store._conn.execute(
            "SELECT kind, detail FROM events WHERE kind = 'trip' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        assert row["kind"] == "trip"
        detail = json.loads(row["detail"])
        assert "budget" in detail["reason"].lower()


def test_boot_survives_on_trip_write_failure(tmp_path):
    class FailingEventStore(Store):
        def add_event(self, kind, detail, scientist_visible, run_id, volumes):
            raise RuntimeError("disk full")

    store = FailingEventStore(
        str(tmp_path / "rig.db"), str(tmp_path / "frames"), str(tmp_path / "backups")
    )
    store.set_volumes(VolumeState(None, None, None))
    app = create_app(
        config=make_fast_cycle_config(),
        driver=FakeRelayDriver(),
        camera=FakeCamera(defaults=FAST_CAMERA_DEFAULTS, frame=_solid_jpeg()),
        store=store,
    )
    with TestClient(app) as client:
        response = client.get("/status", headers=CONSOLE)
        assert response.status_code == 200
        status = response.json()
        assert status["mode"] == "paused"
        assert any("cup" in alert for alert in status["alerts"])


def test_boot_fresh_db_alerts_unknown_cup(tmp_path):
    store = _make_store(tmp_path, volumes=VolumeState(None, 0.0, 0.0))
    store.start_run(
        target_hex="#3366cc",
        target_lab=rgb_to_lab(hex_to_rgb("#3366cc")),
        initial_lambda=0.0,
        cycle_budget=10,
        model="test",
    )
    app = create_app(
        config=make_fast_cycle_config(),
        driver=FakeRelayDriver(),
        camera=FakeCamera(defaults=FAST_CAMERA_DEFAULTS, frame=_solid_jpeg()),
        store=store,
    )
    with TestClient(app) as client:
        status = client.get("/status", headers=CONSOLE).json()
        assert any("cup volume unknown" in alert for alert in status["alerts"])
        assert "cup" in status["trip_reason"].lower()

        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
        response = _cycle(client, CONSOLE)
        assert response.status_code == 409
        assert "cup" in response.json()["detail"].lower()


def test_boot_fresh_db_alerts_unknown_counters(tmp_path):
    app, store = _app_with_store(tmp_path, volumes=VolumeState(0.0, None, None), start_run=False)
    with TestClient(app) as client:
        status = client.get("/status", headers=CONSOLE).json()
        assert any("water counter unknown" in alert for alert in status["alerts"])
        assert any("waste counter unknown" in alert for alert in status["alerts"])
        assert any("staging" in alert for alert in status["alerts"])
        assert "counter" in status["trip_reason"].lower()

        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
        response = pump(client, "A", 1)
        assert response.status_code == 409
        assert "staging" in response.json()["detail"].lower()


def test_boot_fresh_db_reason_names_both_cup_and_counters(tmp_path):
    app, store = _app_with_store(tmp_path, volumes=VolumeState(None, None, None), start_run=False)
    with TestClient(app) as client:
        status = client.get("/status").json()
        reason = status["trip_reason"].lower()
        assert "cup" in reason
        assert "counter" in reason
        assert "staging" in reason


def test_boot_nonzero_cup_alerts_and_rejects_cycles(tmp_path):
    store = _make_store(tmp_path, volumes=VolumeState(60.0, 0.0, 0.0))
    store.start_run(
        target_hex="#3366cc",
        target_lab=rgb_to_lab(hex_to_rgb("#3366cc")),
        initial_lambda=0.0,
        cycle_budget=10,
        model="test",
    )
    app = create_app(
        config=make_fast_cycle_config(),
        driver=FakeRelayDriver(),
        camera=FakeCamera(defaults=FAST_CAMERA_DEFAULTS, frame=_solid_jpeg()),
        store=store,
    )
    with TestClient(app) as client:
        # A known non-zero cup is surfaced by the boot trip, not by alerts() -- alerts()
        # only flags an UNKNOWN cup (a known non-zero cup legitimately holds liquid
        # mid-cycle, and flagging it there would fire on every normal cycle). Check the
        # trip signal before switching mode, since console mode changes clear trip_reason.
        status = client.get("/status").json()
        assert status["mode"] == "paused"
        assert "cup" in status["trip_reason"].lower()

        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        response = _cycle(client, SCIENTIST)
        assert response.status_code == 409
        assert "cup" in response.json()["detail"].lower()


def test_boot_known_empty_cup_is_clean(tmp_path):
    app, store = _app_with_store(tmp_path, volumes=VolumeState(0.0, 0.0, 0.0), start_run=False)
    with TestClient(app) as client:
        status = client.get("/status", headers=CONSOLE).json()
        assert not any("cup" in alert for alert in status["alerts"])
        assert not any("counter" in alert for alert in status["alerts"])
        assert status["trip_reason"] is None


def test_boot_rehydrates_session_counters(tmp_path):
    app, store = _app_with_store(
        tmp_path, config=make_config(), volumes=VolumeState(0.0, 5800.0, 0.0), start_run=False
    )
    with TestClient(app) as client:
        status = client.get("/status", headers=CONSOLE).json()
        assert status["volumes"]["water_drawn_ml"] == pytest.approx(5800.0)
        assert any("reservoir" in alert for alert in status["alerts"])


def test_boot_rehydrated_counters_still_trip_on_next_dose(tmp_path):
    config = make_config(
        volumes=VolumeLimits(
            cup_ceiling_ml=150, reservoir_ml=6000, waste_capacity_ml=6000, low_margin_ml=300
        )
    )
    app, store = _app_with_store(tmp_path, config=config, volumes=VolumeState(0.0, 5800.0, 0.0), start_run=False)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
        response = pump(client, "W", 10)
        assert response.status_code == 200
        status = client.get("/status", headers=CONSOLE).json()
        assert status["mode"] == "paused"
        assert any("reservoir" in alert for alert in status["alerts"])


def test_staging_then_drain_resolves_fresh_db_end_to_end(tmp_path):
    config = make_config()
    app, store = _app_with_store(
        tmp_path, config=config, volumes=VolumeState(None, None, None), start_run=False
    )
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)

        reject = pump(client, "D", 150)
        assert reject.status_code == 409
        assert "staging" in reject.json()["detail"].lower()

        staging = client.post(
            "/staging",
            json={"water_remaining_ml": config.volumes.reservoir_ml, "waste_ml": 0.0},
            headers=CONSOLE,
        )
        assert staging.status_code == 200
        assert staging.json()["water_drawn_ml"] == 0.0
        assert staging.json()["waste_ml"] == 0.0

        response = pump(client, "D", 150)
        assert response.status_code == 200
        status = client.get("/status", headers=CONSOLE).json()
        assert status["volumes"]["cup_ml"] == 0.0
        assert not any("cup volume unknown" in alert for alert in status["alerts"])


def test_staging_declares_counters_and_writes_visible_event(tmp_path):
    app, store = _app_with_store(tmp_path, volumes=VolumeState(0.0, None, None), start_run=False)
    config = make_fast_cycle_config()
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
        response = client.post(
            "/staging",
            json={"water_remaining_ml": config.volumes.reservoir_ml - 123.0, "waste_ml": 456.0},
            headers=CONSOLE,
        )
        assert response.status_code == 200
        assert response.json()["water_drawn_ml"] == pytest.approx(123.0)
        assert response.json()["waste_ml"] == pytest.approx(456.0)

        status = client.get("/status", headers=CONSOLE).json()
        assert status["volumes"]["water_drawn_ml"] == pytest.approx(123.0)
        assert status["volumes"]["waste_ml"] == pytest.approx(456.0)

        row = store._conn.execute(
            "SELECT detail, scientist_visible FROM events WHERE kind = 'staging'"
        ).fetchone()
        assert row is not None
        assert row["scientist_visible"] == 1
        detail = json.loads(row["detail"])
        assert detail["water_remaining_ml"] == pytest.approx(config.volumes.reservoir_ml - 123.0)
        assert detail["waste_ml"] == pytest.approx(456.0)

        volumes = store.load_volumes()
        assert volumes.water_drawn_ml == pytest.approx(123.0)
        assert volumes.waste_ml == pytest.approx(456.0)


def test_staging_scientist_409(tmp_path):
    app, store = _app_with_store(tmp_path, start_run=False)
    with TestClient(app) as client:
        response = client.post(
            "/staging",
            json={"water_remaining_ml": 100.0, "waste_ml": 0.0},
            headers=SCIENTIST,
        )
        assert response.status_code == 409


def test_staging_in_auto_409(tmp_path):
    app, store = _app_with_store(tmp_path, start_run=False)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        response = client.post(
            "/staging",
            json={"water_remaining_ml": 100.0, "waste_ml": 0.0},
            headers=CONSOLE,
        )
        assert response.status_code == 409


def test_staging_in_paused_409(tmp_path):
    app, store = _app_with_store(tmp_path, start_run=False)
    with TestClient(app) as client:
        response = client.post(
            "/staging",
            json={"water_remaining_ml": 100.0, "waste_ml": 0.0},
            headers=CONSOLE,
        )
        assert response.status_code == 409


def test_staging_out_of_range_400(tmp_path):
    app, store = _app_with_store(tmp_path, volumes=VolumeState(0.0, 0.0, 0.0), start_run=False)
    config = make_fast_cycle_config()
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
        response = client.post(
            "/staging",
            json={"water_remaining_ml": config.volumes.reservoir_ml + 1, "waste_ml": 0.0},
            headers=CONSOLE,
        )
        assert response.status_code == 400

        row = store._conn.execute("SELECT 1 FROM events WHERE kind = 'staging'").fetchone()
        assert row is None
        volumes = store.load_volumes()
        assert volumes.water_drawn_ml == 0.0
        assert volumes.waste_ml == 0.0


def test_restage_after_budget_trip_recovers_without_db_edits(tmp_path):
    config = make_config(
        volumes=VolumeLimits(
            cup_ceiling_ml=150, reservoir_ml=200, waste_capacity_ml=6000, low_margin_ml=150
        )
    )
    app, store = _app_with_store(tmp_path, config=config, volumes=VolumeState(0.0, 0.0, 0.0), start_run=False)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
        pump(client, "W", 60)
        status = client.get("/status", headers=CONSOLE).json()
        assert status["mode"] == "paused"
        assert any("reservoir" in alert for alert in status["alerts"])

        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
        response = client.post(
            "/staging",
            json={"water_remaining_ml": 200.0, "waste_ml": 0.0},
            headers=CONSOLE,
        )
        assert response.status_code == 200

        status = client.get("/status", headers=CONSOLE).json()
        assert not any("reservoir" in alert for alert in status["alerts"])

        response = pump(client, "W", 10)
        assert response.status_code == 200


def test_boot_rehydrates_active_run_target_and_lambda(tmp_path):
    store = _make_store(tmp_path, volumes=VolumeState(0.0, 0.0, 0.0))
    run_id = store.start_run(
        target_hex="#3366cc",
        target_lab=rgb_to_lab(hex_to_rgb("#3366cc")),
        initial_lambda=0.1,
        cycle_budget=10,
        model="test",
    )
    store.add_event(
        "lambda_change",
        {"value": 0.5},
        scientist_visible=True,
        run_id=run_id,
        volumes=VolumeState(0.0, 0.0, 0.0),
    )
    app = create_app(
        config=make_fast_cycle_config(),
        driver=FakeRelayDriver(),
        camera=FakeCamera(defaults=FAST_CAMERA_DEFAULTS, frame=_solid_jpeg()),
        store=store,
    )
    with TestClient(app) as client:
        status = client.get("/status").json()
        assert status["run"] is not None
        assert status["run"]["id"] == run_id
        assert status["lambda"] == pytest.approx(0.5)


def test_boot_marks_dangling_cycles_aborted(tmp_path):
    store = _make_store(tmp_path, volumes=VolumeState(0.0, 0.0, 0.0))
    run_id = store.start_run(
        target_hex="#3366cc",
        target_lab=rgb_to_lab(hex_to_rgb("#3366cc")),
        initial_lambda=0.0,
        cycle_budget=10,
        model="test",
    )
    from rig_service.cycle import CycleRecord

    record = CycleRecord(
        requested_a_ml=1.0,
        requested_b_ml=1.0,
        executed_a_ml=1.0,
        executed_b_ml=1.0,
        lam=0.0,
        target_lab=(50.0, 10.0, -20.0),
    )
    cycle_id = store.start_cycle(run_id, record, VolumeState(0.0, 0.0, 0.0))
    app = create_app(
        config=make_fast_cycle_config(),
        driver=FakeRelayDriver(),
        camera=FakeCamera(defaults=FAST_CAMERA_DEFAULTS, frame=_solid_jpeg()),
        store=store,
    )
    with TestClient(app) as client:
        history = client.get(f"/history?run={run_id}", headers=CONSOLE).json()
        assert len(history["cycles"]) == 1
        assert history["cycles"][0]["aborted"] == "service restart mid-cycle"


def test_history_scientist_vs_console_over_http(tmp_path):
    store = _make_store(tmp_path, volumes=VolumeState(0.0, 0.0, 0.0))
    run_id = store.start_run(
        target_hex="#3366cc",
        target_lab=rgb_to_lab(hex_to_rgb("#3366cc")),
        initial_lambda=0.3,
        cycle_budget=10,
        model="test",
    )
    from rig_service.cycle import CycleRecord

    record = CycleRecord(
        requested_a_ml=1.0,
        requested_b_ml=2.0,
        executed_a_ml=1.5,
        executed_b_ml=2.5,
        lam=0.3,
        target_lab=(50.0, 10.0, -20.0),
        measured_lab=(51.0, 11.0, -19.0),
        blank_lab=(50.5, 10.5, -19.5),
        delta_e=1.5,
        score=2.5,
    )
    cycle_id = store.start_cycle(run_id, record, VolumeState(0.0, 0.0, 0.0))
    store.finish_cycle(cycle_id, record, VolumeState(0.0, 0.0, 0.0))
    store.add_event(
        "disrupt/skew",
        {"channel": "A", "factor": 1.5},
        scientist_visible=False,
        run_id=run_id,
        volumes=VolumeState(0.0, 0.0, 0.0),
    )
    app = create_app(
        config=make_fast_cycle_config(),
        driver=FakeRelayDriver(),
        camera=FakeCamera(defaults=FAST_CAMERA_DEFAULTS, frame=_solid_jpeg()),
        store=store,
    )
    with TestClient(app) as client:
        sci = client.get(f"/history?run={run_id}", headers=SCIENTIST).json()
        assert len(sci["cycles"]) == 1
        assert sci["cycles"][0]["dose_a_ml"] == 1.0
        assert sci["cycles"][0]["dose_b_ml"] == 2.0
        assert "executed_a_ml" not in sci["cycles"][0]
        assert "executed_b_ml" not in sci["cycles"][0]
        assert not any(e["kind"] == "disrupt/skew" for e in sci["events"])

        con = client.get(f"/history?run={run_id}", headers=CONSOLE).json()
        assert con["cycles"][0]["executed_a_ml"] == 1.5
        assert con["cycles"][0]["executed_b_ml"] == 2.5
        assert any(e["kind"] == "disrupt/skew" for e in con["events"])


def test_history_requires_client_header_400(tmp_path):
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        response = client.get("/history?run=1")
        assert response.status_code == 400


def test_history_unknown_run_404(tmp_path):
    app = build_app(tmp_path, start_run=False)
    with TestClient(app) as client:
        response = client.get("/history?run=999", headers=CONSOLE)
        assert response.status_code == 404


def test_staging_blocks_until_in_flight_drain_completes(tmp_path):
    import asyncio as aio
    from rig_service import pumps as pumps_module

    drain_started = threading.Event()
    drain_may_finish = threading.Event()
    original_sleep = pumps_module.asyncio.sleep

    async def blocking_sleep(seconds):
        drain_started.set()
        while not drain_may_finish.is_set():
            await original_sleep(0.01)

    config = make_fast_cycle_config()
    driver = FakeRelayDriver()
    # Cup seeded with contents (not empty): begin_drain charges min(cup_ml, ml), and an
    # empty cup would charge 0 regardless of whether the lock held -- the seeded cup
    # makes the in-flight waste charge (and the pump-lock assertion around it) real.
    app, store = _app_with_store(
        tmp_path, config=config, driver=driver,
        volumes=VolumeState(20.0, 0.0, 0.0), start_run=False,
    )
    d_gpio = config.channels["D"].gpio

    drain_result = {}
    staging_result = {}
    drain_thread = None
    staging_thread = None

    try:
        pumps_module.asyncio.sleep = blocking_sleep
        with TestClient(app) as client:
            client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)

            try:
                def drain():
                    drain_result["response"] = pump(client, "D", 20)

                drain_thread = threading.Thread(target=drain)
                drain_thread.start()
                drain_started.wait(timeout=5)

                # Relay is on and waste has been charged before actuation.
                assert d_gpio in driver.relays_on()
                assert store.load_volumes().waste_ml == 20.0

                def stage():
                    staging_result["response"] = client.post(
                        "/staging",
                        json={"water_remaining_ml": config.volumes.reservoir_ml, "waste_ml": 0.0},
                        headers=CONSOLE,
                    )

                staging_thread = threading.Thread(target=stage)
                staging_thread.start()
                time.sleep(0.2)

                # While the drain relay is still on, staging must still be blocked on the
                # lock and the durable waste must still carry the in-flight drain charge.
                assert d_gpio in driver.relays_on()
                assert store.load_volumes().waste_ml == 20.0
            finally:
                # A failed assertion above must not leave blocking_sleep spinning forever
                # on drain_may_finish -- always release it and rejoin both threads so a
                # red assertion fails the test instead of hanging pytest.
                drain_may_finish.set()
                if drain_thread is not None:
                    drain_thread.join(timeout=5)
                if staging_thread is not None:
                    staging_thread.join(timeout=5)

            assert drain_result["response"].status_code == 200
            assert staging_result["response"].status_code == 200
            assert d_gpio not in driver.relays_on()
            assert store.load_volumes().waste_ml == 0.0
    finally:
        pumps_module.asyncio.sleep = original_sleep


# --- T5 gap-fill: staging's console guard, the boot-gate `or`, target rehydration
#     override, and the run/start reject-path fix ---


def test_staging_scientist_in_manual_409(tmp_path):
    # Every existing staging-rejection test runs in paused mode, so the mode guard
    # (arbiter.mode != "manual") alone accounts for the 409 regardless of who is asking
    # -- deleting `client != "console"` would still pass the whole suite. Switching to
    # manual mode first takes the mode guard out of play: with the mode already correct,
    # only the console-only guard can produce this 409.
    app, store = _app_with_store(
        tmp_path, volumes=VolumeState(0.0, 0.0, 0.0), start_run=False
    )
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)

        response = client.post(
            "/staging",
            json={"water_remaining_ml": 5900.0, "waste_ml": 100.0},
            headers=SCIENTIST,
        )
        assert response.status_code == 409

        row = store._conn.execute("SELECT 1 FROM events WHERE kind = 'staging'").fetchone()
        assert row is None
        volumes_after = store.load_volumes()
        assert volumes_after.cup_ml == 0.0
        assert volumes_after.water_drawn_ml == 0.0
        assert volumes_after.waste_ml == 0.0


@pytest.mark.parametrize(
    "seeded_volumes",
    [
        VolumeState(0.0, 5800.0, None),  # water known, waste unknown
        VolumeState(0.0, None, 5800.0),  # water unknown, waste known
    ],
    ids=["waste_unknown", "water_unknown"],
)
def test_boot_gate_mixed_known_and_unknown_counter_trips(tmp_path, seeded_volumes):
    # Every boot-gate test seeds both session counters known-together or
    # unknown-together, so app.py's `_boot_volume_gate` guard --
    # `water_drawn_ml is None or waste_ml is None` -- never exercises its asymmetric
    # branches; an `or` -> `and` mutation would leave both counters required unknown to
    # trip, silently passing a mixed known/unknown boot as clean. Each parametrization
    # seeds exactly one counter unknown, isolating one branch of the `or` (the cup is
    # known-zero throughout so only the counter guard can be responsible for the trip).
    app, store = _app_with_store(tmp_path, volumes=seeded_volumes, start_run=False)
    with TestClient(app) as client:
        status = client.get("/status", headers=CONSOLE).json()
        assert status["mode"] == "paused"
        assert status["trip_reason"] is not None
        reason = status["trip_reason"].lower()
        assert "counter" in reason
        assert "staging" in reason
        assert any(
            "unknown" in alert and "counter" in alert for alert in status["alerts"]
        )


def test_boot_rehydrates_active_run_target_override(tmp_path):
    # test_boot_rehydrates_active_run_target_and_lambda only seeds a lambda_change event
    # and asserts lambda; the target-override branch (decision 7: the latest
    # target_change event overrides the run row's original target) is never exercised --
    # deleting it would survive the suite. Distinct hexes so the swap is observable.
    store = _make_store(tmp_path, volumes=VolumeState(0.0, 0.0, 0.0))
    run_id = store.start_run(
        target_hex="#3366cc",
        target_lab=rgb_to_lab(hex_to_rgb("#3366cc")),
        initial_lambda=0.0,
        cycle_budget=10,
        model="test",
    )
    store.add_event(
        "target_change",
        {"hex": "#ff0000"},
        scientist_visible=True,
        run_id=run_id,
        volumes=VolumeState(0.0, 0.0, 0.0),
    )
    app = create_app(
        config=make_fast_cycle_config(),
        driver=FakeRelayDriver(),
        camera=FakeCamera(defaults=FAST_CAMERA_DEFAULTS, frame=_solid_jpeg()),
        store=store,
    )
    with TestClient(app) as client:
        status = client.get("/status").json()
        assert status["run"] is not None
        assert status["run"]["id"] == run_id
        assert status["run"]["target_hex"] == "#3366cc"  # the run row's original target
        assert status["target_hex"] == "#ff0000"  # overridden by the latest target_change


def test_run_start_reject_does_not_mutate_active_target(tmp_path):
    # Regression guard for the run/start reject-path fix: a 409 (a run is already
    # active) must not mutate the live target via _set_target before the active-run
    # check is enforced. Two distinct hexes make a leaked mutation observable.
    app, store = _app_with_store(tmp_path, start_run=False)
    with TestClient(app) as client:
        first = client.post(
            "/run/start", json=_run_start_json(target_hex="#3366cc"), headers=CONSOLE
        )
        assert first.status_code == 200

        second = client.post(
            "/run/start", json=_run_start_json(target_hex="#112233"), headers=CONSOLE
        )
        assert second.status_code == 409

        status = client.get("/status").json()
        assert status["target_hex"] == "#3366cc"
        assert status["run"]["target_hex"] == "#3366cc"

        run_count = store._conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()["n"]
        assert run_count == 1


# --- T2: disruption routes, hidden events, rehydration, redaction ---


class _BlockingCamera:
    """Camera whose capture_jpeg blocks on a threading.Event until released."""

    def __init__(self, release_event: threading.Event):
        self._defaults = dict(FAST_CAMERA_DEFAULTS)
        self._controls = dict(self._defaults)
        self._frame = _solid_jpeg()
        self._release = release_event

    def apply_controls(self, controls):
        self._controls.update(controls)

    def read_controls(self, names):
        return {name: self._controls[name] for name in names}

    def capture_jpeg(self):
        self._release.wait(timeout=5)
        return self._frame

    def power_cycle(self):
        self._controls = dict(self._defaults)


@contextmanager
def _patched_sleep_for_phase(phase_name: str, config, release_event: threading.Event):
    """Patch pumps_module.asyncio.sleep so the running cycle blocks in ``phase_name``.

    Yields an Event that is set once the patched sleep is entered.  The caller
    must eventually ``release_event.set()`` so the background cycle can finish.
    """
    import asyncio as aio
    from rig_service import pumps as pumps_module

    original_sleep = pumps_module.asyncio.sleep
    entered = threading.Event()

    if phase_name == "settle":
        threshold = config.cycle.settle_s / 2

        async def blocking_sleep(seconds):
            if seconds >= threshold:
                entered.set()
                while not release_event.is_set():
                    await original_sleep(0.01)
            else:
                await original_sleep(seconds)

    elif phase_name == "dose":

        async def blocking_sleep(seconds):
            entered.set()
            while not release_event.is_set():
                await original_sleep(0.01)

    else:
        raise ValueError(f"unsupported phase: {phase_name}")

    pumps_module.asyncio.sleep = blocking_sleep
    try:
        yield entered
    finally:
        release_event.set()
        pumps_module.asyncio.sleep = original_sleep


# --- Skew route ---


def test_skew_scientist_client_409(tmp_path):
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        response = client.post(
            "/disrupt/skew", json={"channel": "A", "factor": 0.5}, headers=SCIENTIST
        )
        assert response.status_code == 409
        assert "console" in response.json()["detail"].lower()


def test_skew_missing_header_400(tmp_path):
    # A missing X-Client is a malformed request (400), distinct from a scientist
    # attempting a console-only action (409) -- matches require_client's use on every
    # other console-only route (/target, /lambda, /run/start, /run/end, /staging).
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        response = client.post("/disrupt/skew", json={"channel": "A", "factor": 0.5})
        assert response.status_code == 400


def test_skew_no_active_run_409(tmp_path):
    app = build_app(tmp_path, start_run=False)
    with TestClient(app) as client:
        response = client.post(
            "/disrupt/skew", json={"channel": "A", "factor": 0.5}, headers=CONSOLE
        )
        assert response.status_code == 409
        assert "active run" in response.json()["detail"].lower()


def test_skew_paused_with_active_run_accepted(tmp_path):
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "paused"}, headers=CONSOLE)
        response = client.post(
            "/disrupt/skew", json={"channel": "A", "factor": 0.5}, headers=CONSOLE
        )
        assert response.status_code == 200
        body = response.json()
        assert body["channel"] == "A"
        assert body["factor"] == pytest.approx(0.5)
        assert body["factors"] == {"A": 0.5}


def test_skew_bad_channel_400(tmp_path):
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        response = client.post(
            "/disrupt/skew", json={"channel": "X", "factor": 0.5}, headers=CONSOLE
        )
        assert response.status_code == 400
        assert "channel" in response.json()["detail"].lower()


def test_skew_bad_factor_400(tmp_path):
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        response = client.post(
            "/disrupt/skew", json={"channel": "A", "factor": 0.0}, headers=CONSOLE
        )
        assert response.status_code == 400
        assert "factor" in response.json()["detail"].lower()


def test_skew_writes_hidden_event_with_factors_map(tmp_path):
    app, store = _app_with_store(tmp_path, start_run=True)
    with TestClient(app) as client:
        response = client.post(
            "/disrupt/skew",
            json={"channel": "A", "factor": 0.5},
            headers=CONSOLE,
        )
        assert response.status_code == 200
        run_id = store.active_run()["id"]

        con = client.get(f"/history?run={run_id}", headers=CONSOLE).json()
        skew_events = [e for e in con["events"] if e["kind"] == "skew"]
        assert len(skew_events) == 1
        assert skew_events[0]["scientist_visible"] == 0
        detail = skew_events[0]["detail"]
        assert detail["channel"] == "A"
        assert detail["factor"] == pytest.approx(0.5)
        assert detail["factors"] == {"A": 0.5}

        sci = client.get(f"/history?run={run_id}", headers=SCIENTIST).json()
        assert not any(e["kind"] == "skew" for e in sci["events"])


def test_e2e_skewed_cycle_scientist_history_shows_requested_only(tmp_path):
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        skew = client.post(
            "/disrupt/skew",
            json={"channel": "A", "factor": 0.5},
            headers=CONSOLE,
        )
        assert skew.status_code == 200

        response = _cycle(client, SCIENTIST, dose_a_ml=2.0, dose_b_ml=1.5)
        assert response.status_code == 200
        body = response.json()
        assert body["dose_a_ml"] == 2.0
        assert body["dose_b_ml"] == 1.5

        run_id = client.get("/status", headers=CONSOLE).json()["run"]["id"]

        sci = client.get(f"/history?run={run_id}", headers=SCIENTIST).json()
        assert len(sci["cycles"]) == 1
        assert sci["cycles"][0]["dose_a_ml"] == 2.0
        assert sci["cycles"][0]["dose_b_ml"] == 1.5
        assert not any(e["kind"] == "skew" for e in sci["events"])

        con = client.get(f"/history?run={run_id}", headers=CONSOLE).json()
        assert len(con["cycles"]) == 1
        assert con["cycles"][0]["executed_a_ml"] == pytest.approx(1.0)
        assert con["cycles"][0]["executed_b_ml"] == pytest.approx(1.5)
        skew_events = [e for e in con["events"] if e["kind"] == "skew"]
        assert len(skew_events) == 1
        assert skew_events[0]["detail"]["factors"] == {"A": 0.5}


# --- Run-start / rehydration ---


def test_run_start_clears_skew(tmp_path):
    app, store = _app_with_store(
        tmp_path, volumes=VolumeState(0.0, 0.0, 0.0), start_run=True
    )
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        skew = client.post(
            "/disrupt/skew",
            json={"channel": "A", "factor": 0.5},
            headers=CONSOLE,
        )
        assert skew.status_code == 200

        client.post("/run/end", json={"outcome": "ok"}, headers=CONSOLE)
        start = client.post("/run/start", json=_run_start_json(), headers=CONSOLE)
        assert start.status_code == 200

        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)
        response = _cycle(client, SCIENTIST, dose_a_ml=2.0, dose_b_ml=1.5)
        assert response.status_code == 200

        run_id = start.json()["run_id"]
        con = client.get(f"/history?run={run_id}", headers=CONSOLE).json()
        assert len(con["cycles"]) == 1
        assert con["cycles"][0]["executed_a_ml"] == pytest.approx(2.0)
        assert con["cycles"][0]["executed_b_ml"] == pytest.approx(1.5)
        assert not any(e["kind"] == "skew" for e in con["events"])


def test_boot_rehydrates_skew_from_latest_event(tmp_path):
    store = _make_store(tmp_path, volumes=VolumeState(0.0, 0.0, 0.0))
    run_id = store.start_run(
        target_hex="#3366cc",
        target_lab=rgb_to_lab(hex_to_rgb("#3366cc")),
        initial_lambda=0.0,
        cycle_budget=10,
        model="test",
    )
    store.add_event(
        "skew",
        {"channel": "A", "factor": 0.5, "factors": {"A": 0.5}},
        scientist_visible=False,
        run_id=run_id,
        volumes=VolumeState(0.0, 0.0, 0.0),
    )

    app = create_app(
        config=make_fast_cycle_config(),
        driver=FakeRelayDriver(),
        camera=FakeCamera(defaults=FAST_CAMERA_DEFAULTS, frame=_solid_jpeg()),
        store=store,
    )
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        response = _cycle(client, SCIENTIST, dose_a_ml=2.0, dose_b_ml=1.5)
        assert response.status_code == 200

        con = client.get(f"/history?run={run_id}", headers=CONSOLE).json()
        assert con["cycles"][0]["executed_a_ml"] == pytest.approx(1.0)
        assert con["cycles"][0]["executed_b_ml"] == pytest.approx(1.5)


def test_boot_without_skew_events_is_identity(tmp_path):
    store = _make_store(tmp_path, volumes=VolumeState(0.0, 0.0, 0.0))
    run_id = store.start_run(
        target_hex="#3366cc",
        target_lab=rgb_to_lab(hex_to_rgb("#3366cc")),
        initial_lambda=0.0,
        cycle_budget=10,
        model="test",
    )

    app = create_app(
        config=make_fast_cycle_config(),
        driver=FakeRelayDriver(),
        camera=FakeCamera(defaults=FAST_CAMERA_DEFAULTS, frame=_solid_jpeg()),
        store=store,
    )
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        response = _cycle(client, SCIENTIST, dose_a_ml=2.0, dose_b_ml=1.5)
        assert response.status_code == 200

        con = client.get(f"/history?run={run_id}", headers=CONSOLE).json()
        assert con["cycles"][0]["executed_a_ml"] == pytest.approx(2.0)
        assert con["cycles"][0]["executed_b_ml"] == pytest.approx(1.5)


# --- Blip route ---


def test_blip_scientist_client_409(tmp_path):
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        response = client.post(
            "/disrupt/blip", json={"ml_water": 5.0}, headers=SCIENTIST
        )
        assert response.status_code == 409
        assert "console" in response.json()["detail"].lower()


def test_blip_missing_header_400(tmp_path):
    # A missing X-Client is a malformed request (400), distinct from a scientist
    # attempting a console-only action (409) -- matches require_client's use on every
    # other console-only route.
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        response = client.post("/disrupt/blip", json={"ml_water": 5.0})
        assert response.status_code == 400


def test_blip_idle_409(tmp_path):
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        response = client.post(
            "/disrupt/blip", json={"ml_water": 5.0}, headers=CONSOLE
        )
        assert response.status_code == 409


def test_blip_dose_phase_409(tmp_path):
    config = make_fast_cycle_config()
    app = build_app(tmp_path, config=config, start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        release = threading.Event()
        result = {}
        with _patched_sleep_for_phase("dose", config, release) as entered:

            def run():
                result["response"] = _cycle(client, SCIENTIST, dose_a_ml=1.0, dose_b_ml=1.0)

            thread = threading.Thread(target=run)
            thread.start()
            entered.wait(timeout=5)
            assert client.get("/status", headers=CONSOLE).json()["cycle"]["phase"] == "dose"

            blip = client.post(
                "/disrupt/blip", json={"ml_water": 1.0}, headers=CONSOLE
            )
            assert blip.status_code == 409

            release.set()
            thread.join(timeout=5)

        assert result["response"].status_code == 200
        assert result["response"].json()["aborted"] is None


def test_blip_measure_phase_409(tmp_path):
    release = threading.Event()
    app = build_app(tmp_path, camera=_BlockingCamera(release), start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        result = {}

        def run():
            result["response"] = _cycle(client, SCIENTIST, dose_a_ml=1.0, dose_b_ml=1.0)

        thread = threading.Thread(target=run)
        thread.start()
        time.sleep(0.3)  # DOSE/REFILL/SETTLE complete; MEASURE is blocked.

        blip = client.post(
            "/disrupt/blip", json={"ml_water": 1.0}, headers=CONSOLE
        )
        assert blip.status_code == 409

        release.set()
        thread.join(timeout=5)
        assert result["response"].status_code == 200
        assert result["response"].json()["aborted"] is None


def test_blip_during_settle_doses_water_and_hides_event(tmp_path):
    # Fast pump rates: same volumes/settle_s/cup arithmetic as make_config()'s
    # default, but drain/rinse/final_drain (real sleeps outside the patched
    # settle hold) complete in milliseconds instead of seconds.
    config = make_config(rate_ml_per_s=1000.0, lag_s=0.001)
    camera = FakeCamera(defaults=FAKE_CAMERA_DEFAULTS, frame=_solid_jpeg(size=250))
    app = build_app(tmp_path, config=config, camera=camera, start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        release = threading.Event()
        result = {}
        with _patched_sleep_for_phase("settle", config, release) as entered:

            def run():
                result["response"] = _cycle(client, CONSOLE, dose_a_ml=5.0, dose_b_ml=5.0)

            thread = threading.Thread(target=run)
            thread.start()
            entered.wait(timeout=5)
            assert client.get("/status", headers=CONSOLE).json()["cycle"]["phase"] == "settle"

            status_before = client.get("/status", headers=CONSOLE).json()
            assert status_before["volumes"]["cup_ml"] == pytest.approx(110.0)
            water_before = status_before["volumes"]["water_drawn_ml"]

            blip = client.post(
                "/disrupt/blip", json={"ml_water": 5.0}, headers=CONSOLE
            )
            assert blip.status_code == 200
            body = blip.json()
            assert body["ml_water"] == pytest.approx(5.0)
            assert "seconds" in body

            status_during = client.get("/status", headers=CONSOLE).json()
            assert status_during["volumes"]["cup_ml"] == pytest.approx(115.0)
            assert status_during["volumes"]["water_drawn_ml"] == pytest.approx(
                water_before + 5.0
            )

            release.set()
            thread.join(timeout=5)

        assert result["response"].status_code == 200
        assert result["response"].json()["aborted"] is None

        run_id = client.get("/status", headers=CONSOLE).json()["run"]["id"]

        con = client.get(f"/history?run={run_id}", headers=CONSOLE).json()
        blip_events = [e for e in con["events"] if e["kind"] == "blip"]
        assert len(blip_events) == 1
        assert blip_events[0]["scientist_visible"] == 0
        assert blip_events[0]["detail"]["ml_water"] == pytest.approx(5.0)

        sci = client.get(f"/history?run={run_id}", headers=SCIENTIST).json()
        assert not any(e["kind"] == "blip" for e in sci["events"])


def test_blip_over_headroom_400(tmp_path):
    # Fast pump rates: same volumes/settle_s/cup arithmetic as make_config()'s
    # default (two full cycles' worth of drain/rinse/final_drain would otherwise
    # dominate runtime at 100 ml/s).
    config = make_config(rate_ml_per_s=1000.0, lag_s=0.001)
    camera = FakeCamera(defaults=FAKE_CAMERA_DEFAULTS, frame=_solid_jpeg(size=250))
    app = build_app(tmp_path, config=config, camera=camera, start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        # First cycle: cup at 10+10+100 = 120 ml, allowed = 30 ml.
        release1 = threading.Event()
        result1 = {}
        with _patched_sleep_for_phase("settle", config, release1) as entered1:

            def run1():
                result1["response"] = _cycle(client, CONSOLE, dose_a_ml=10.0, dose_b_ml=10.0)

            thread1 = threading.Thread(target=run1)
            thread1.start()
            entered1.wait(timeout=5)
            assert client.get("/status", headers=CONSOLE).json()["cycle"]["phase"] == "settle"

            blip = client.post(
                "/disrupt/blip", json={"ml_water": 30.001}, headers=CONSOLE
            )
            assert blip.status_code == 400
            detail = blip.json()["detail"].lower()
            assert "30" in detail

            release1.set()
            thread1.join(timeout=5)

        assert result1["response"].status_code == 200

        # Second cycle: exactly allowed = 30 ml must be accepted.
        release2 = threading.Event()
        result2 = {}
        with _patched_sleep_for_phase("settle", config, release2) as entered2:

            def run2():
                result2["response"] = _cycle(client, CONSOLE, dose_a_ml=10.0, dose_b_ml=10.0)

            thread2 = threading.Thread(target=run2)
            thread2.start()
            entered2.wait(timeout=5)

            blip = client.post(
                "/disrupt/blip", json={"ml_water": 30.0}, headers=CONSOLE
            )
            assert blip.status_code == 200

            release2.set()
            thread2.join(timeout=5)

        assert result2["response"].status_code == 200


def test_blip_nonpositive_ml_400(tmp_path):
    config = make_config()
    app = build_app(tmp_path, config=config, start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        release = threading.Event()
        result = {}
        with _patched_sleep_for_phase("settle", config, release) as entered:

            def run():
                result["response"] = _cycle(client, CONSOLE, dose_a_ml=1.0, dose_b_ml=1.0)

            thread = threading.Thread(target=run)
            thread.start()
            entered.wait(timeout=5)

            blip = client.post(
                "/disrupt/blip", json={"ml_water": 0.0}, headers=CONSOLE
            )
            assert blip.status_code == 400

            blip_negative = client.post(
                "/disrupt/blip", json={"ml_water": -1.0}, headers=CONSOLE
            )
            assert blip_negative.status_code == 400

            release.set()
            thread.join(timeout=5)

        assert result["response"].status_code == 200


def test_blip_paused_rejected_409(tmp_path):
    config = make_config()
    app = build_app(tmp_path, config=config, start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        release = threading.Event()
        result = {}
        with _patched_sleep_for_phase("settle", config, release) as entered:

            def run():
                result["response"] = _cycle(client, CONSOLE, dose_a_ml=1.0, dose_b_ml=1.0)

            thread = threading.Thread(target=run)
            thread.start()
            entered.wait(timeout=5)
            assert client.get("/status", headers=CONSOLE).json()["cycle"]["phase"] == "settle"

            pause = client.post("/mode", json={"mode": "paused"}, headers=CONSOLE)
            assert pause.status_code == 200

            blip = client.post(
                "/disrupt/blip", json={"ml_water": 1.0}, headers=CONSOLE
            )
            assert blip.status_code == 409
            # Pins the rejection to pumps.dose's paused-floor guard ("mode is
            # paused; dose rejected"), not the route's phase guard ("blip only
            # allowed during settle phase") -- both would 409 here since phase
            # is still settle, but only the paused-floor message names "paused".
            assert "paused" in blip.json()["detail"].lower()

            release.set()
            thread.join(timeout=5)


# --- /status redaction ---


def test_status_scientist_view_omits_volumes_relays_alerts_and_timings(tmp_path):
    # Fast config: one clean cycle draws refill(10) + rinse(10) = 20 ml from the
    # reservoir. With reservoir_ml=50 and low_margin_ml=20, a 15 ml water pump
    # leaves reservoir_left = 50 - (20 + 15) = 15 ml <= 20 ml, tripping paused
    # with a reservoir alert -- same assertion as the slow make_config version.
    config = make_fast_cycle_config(
        volumes=VolumeLimits(
            cup_ceiling_ml=20, reservoir_ml=50, waste_capacity_ml=6000, low_margin_ml=20
        )
    )
    app, store = _app_with_store(
        tmp_path,
        config=config,
        camera=FakeCamera(defaults=FAKE_CAMERA_DEFAULTS, frame=_solid_jpeg(size=250)),
        volumes=VolumeState(0.0, 0.0, 0.0),
        start_run=True,
    )
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)
        response = _cycle(client, SCIENTIST, dose_a_ml=1.0, dose_b_ml=1.0)
        assert response.status_code == 200
        assert response.json()["aborted"] is None

        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
        pump(client, "W", 15)
        con_status = client.get("/status", headers=CONSOLE).json()
        assert con_status["mode"] == "paused"
        assert any("reservoir" in alert for alert in con_status["alerts"])

        sci = client.get("/status").json()
        assert "volumes" not in sci
        assert "relays_on" not in sci
        assert "alerts" not in sci
        assert sci["cycle"]["last"] is not None
        assert "phase_elapsed_s" not in sci["cycle"]["last"]
        assert sci["alarms"] == []


def test_status_console_view_retains_full_payload(tmp_path):
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)
        response = _cycle(client, SCIENTIST, dose_a_ml=1.0, dose_b_ml=1.0)
        assert response.status_code == 200

        con = client.get("/status", headers=CONSOLE).json()
        assert "volumes" in con
        assert "relays_on" in con
        assert "alerts" in con
        assert con["cycle"]["last"] is not None
        assert "phase_elapsed_s" in con["cycle"]["last"]


def test_status_missing_header_gets_scientist_view(tmp_path):
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)
        response = _cycle(client, SCIENTIST, dose_a_ml=1.0, dose_b_ml=1.0)
        assert response.status_code == 200

        status = client.get("/status").json()
        assert "volumes" not in status
        assert "relays_on" not in status
        assert "alerts" not in status


# --- /cycle response redaction ---


def test_cycle_response_scientist_omits_phase_elapsed(tmp_path):
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        response = _cycle(client, SCIENTIST, dose_a_ml=1.0, dose_b_ml=1.0)
        assert response.status_code == 200
        body = response.json()
        assert "phase_elapsed_s" not in body


def test_cycle_response_console_includes_phase_elapsed(tmp_path):
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        response = _cycle(client, CONSOLE, dose_a_ml=1.0, dose_b_ml=1.0)
        assert response.status_code == 200
        body = response.json()
        assert "phase_elapsed_s" in body


# --- T2: decisions writer ---


def test_cycle_with_rationale_writes_decision_row(tmp_path):
    app, store = _app_with_store(
        tmp_path, volumes=VolumeState(0.0, 0.0, 0.0), start_run=True
    )
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        response = _cycle(
            client,
            SCIENTIST,
            dose_a_ml=1.5,
            dose_b_ml=2.5,
            rationale="explore mid-tone",
            model_id="claude-test",
            latency_s=0.42,
        )
        assert response.status_code == 200
        assert response.json()["aborted"] is None

        run_id = client.get("/status", headers=CONSOLE).json()["run"]["id"]
        rows = store._conn.execute(
            "SELECT * FROM decisions WHERE run_id = ?", (run_id,)
        ).fetchall()
        assert len(rows) == 1
        decision = rows[0]
        assert decision["cycle_n"] == 1
        assert decision["proposed_a_ml"] == pytest.approx(1.5)
        assert decision["proposed_b_ml"] == pytest.approx(2.5)
        assert decision["rationale"] == "explore mid-tone"
        assert decision["model_id"] == "claude-test"
        assert decision["latency_s"] == pytest.approx(0.42)


def test_cycle_without_rationale_writes_no_decision(tmp_path):
    app, store = _app_with_store(
        tmp_path, volumes=VolumeState(0.0, 0.0, 0.0), start_run=True
    )
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        before = _decision_count(store)
        response = _cycle(client, SCIENTIST, dose_a_ml=1.0, dose_b_ml=1.0)
        assert response.status_code == 200
        assert _decision_count(store) == before


def test_decision_written_before_actuation(tmp_path):
    driver = FakeRelayDriver()
    camera = FakeCamera(defaults=FAST_CAMERA_DEFAULTS, frame=_solid_jpeg())
    store = _make_store(tmp_path, volumes=VolumeState(0.0, 0.0, 0.0))
    run_id = store.start_run(
        target_hex="#3366cc",
        target_lab=rgb_to_lab(hex_to_rgb("#3366cc")),
        initial_lambda=0.0,
        cycle_budget=10,
        model="test",
    )
    app = create_app(
        config=make_fast_cycle_config(),
        driver=driver,
        camera=camera,
        store=store,
    )
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        relay_history_before = list(driver.history)
        before_decisions = _decision_count(store, run_id)
        before_cycles = store.cycle_count(run_id)

        response = _cycle(
            client,
            SCIENTIST,
            dose_a_ml=-1.0,
            dose_b_ml=1.0,
            rationale="recorded even though the cycle is rejected",
        )
        assert response.status_code == 400

        assert _decision_count(store, run_id) == before_decisions + 1
        assert store.cycle_count(run_id) == before_cycles
        assert list(driver.history) == relay_history_before


def test_decision_defaults_model_unknown_latency_zero(tmp_path):
    app, store = _app_with_store(
        tmp_path, volumes=VolumeState(0.0, 0.0, 0.0), start_run=True
    )
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        response = _cycle(
            client,
            SCIENTIST,
            dose_a_ml=1.0,
            dose_b_ml=1.0,
            rationale="defaults for optional decision fields",
        )
        assert response.status_code == 200

        run_id = client.get("/status", headers=CONSOLE).json()["run"]["id"]
        decision = store._conn.execute(
            "SELECT * FROM decisions WHERE run_id = ?", (run_id,)
        ).fetchone()
        assert decision is not None
        assert decision["model_id"] == "unknown"
        assert decision["latency_s"] == pytest.approx(0.0)


def test_decision_model_id_empty_string_preserved(tmp_path):
    app, store = _app_with_store(
        tmp_path, volumes=VolumeState(0.0, 0.0, 0.0), start_run=True
    )
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        response = _cycle(
            client,
            SCIENTIST,
            dose_a_ml=1.0,
            dose_b_ml=1.0,
            rationale="empty model_id must be preserved",
            model_id="",
        )
        assert response.status_code == 200

        run_id = client.get("/status", headers=CONSOLE).json()["run"]["id"]
        rows = store._conn.execute(
            "SELECT * FROM decisions WHERE run_id = ? ORDER BY cycle_n", (run_id,)
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["model_id"] == ""

        response = _cycle(
            client,
            SCIENTIST,
            dose_a_ml=1.0,
            dose_b_ml=1.0,
            rationale="omitted model_id must default to unknown",
        )
        assert response.status_code == 200

        rows = store._conn.execute(
            "SELECT * FROM decisions WHERE run_id = ? ORDER BY cycle_n", (run_id,)
        ).fetchall()
        assert len(rows) == 2
        assert rows[1]["model_id"] == "unknown"


def test_decision_visible_in_both_history_views(tmp_path):
    app, store = _app_with_store(
        tmp_path, volumes=VolumeState(0.0, 0.0, 0.0), start_run=True
    )
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        response = _cycle(
            client,
            SCIENTIST,
            dose_a_ml=1.0,
            dose_b_ml=1.0,
            rationale="visible to both viewers",
            model_id="claude-x",
            latency_s=0.55,
        )
        assert response.status_code == 200

        run_id = client.get("/status", headers=CONSOLE).json()["run"]["id"]
        for headers in (CONSOLE, SCIENTIST):
            history = client.get(f"/history?run={run_id}", headers=headers).json()
            decisions = history["decisions"]
            assert len(decisions) == 1
            assert decisions[0]["rationale"] == "visible to both viewers"


def test_cycle_authorization_still_precedes_decision_write(tmp_path):
    app, store = _app_with_store(
        tmp_path, volumes=VolumeState(0.0, 0.0, 0.0), start_run=True
    )
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        before = _decision_count(store)
        response = _cycle(
            client,
            SCIENTIST,
            dose_a_ml=1.0,
            dose_b_ml=1.0,
            rationale="must not be written before authorization",
        )
        assert response.status_code == 409
        assert _decision_count(store) == before


def test_busy_engine_409_writes_no_decision(tmp_path):
    release = threading.Event()
    app, store = _app_with_store(
        tmp_path,
        camera=_BlockingCamera(release),
        volumes=VolumeState(0.0, 0.0, 0.0),
        start_run=True,
    )
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        result = {}

        def run_first():
            result["response"] = _cycle(
                client,
                SCIENTIST,
                dose_a_ml=1.0,
                dose_b_ml=1.0,
                rationale="first decision is written before actuation",
                model_id="claude-x",
                latency_s=0.12,
            )

        thread = threading.Thread(target=run_first)
        thread.start()
        time.sleep(0.3)  # first cycle is now in MEASURE

        run_id = client.get("/status", headers=CONSOLE).json()["run"]["id"]
        # The first cycle's decision row was written before run_cycle started.
        assert _decision_count(store, run_id) == 1

        busy = _cycle(
            client,
            SCIENTIST,
            dose_a_ml=1.0,
            dose_b_ml=1.0,
            rationale="must not be written while busy",
        )
        assert busy.status_code == 409
        # The busy second request must not add a second decision row.
        assert _decision_count(store, run_id) == 1

        release.set()
        thread.join(timeout=5)
        assert result["response"].status_code == 200
        assert result["response"].json()["aborted"] is None


def test_busy_guard_before_engine(tmp_path):
    release = threading.Event()
    app, store = _app_with_store(
        tmp_path,
        camera=_BlockingCamera(release),
        volumes=VolumeState(0.0, 0.0, 0.0),
        start_run=True,
    )
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        result = {}

        def run_first():
            result["response"] = _cycle(
                client,
                SCIENTIST,
                dose_a_ml=1.0,
                dose_b_ml=1.0,
                rationale="route guard fires before engine preflight",
                model_id="claude-x",
                latency_s=0.21,
            )

        thread = threading.Thread(target=run_first)
        thread.start()
        time.sleep(0.3)  # first cycle is now in MEASURE

        run_id = client.get("/status", headers=CONSOLE).json()["run"]["id"]
        # The first cycle's decision row was written before run_cycle started.
        assert _decision_count(store, run_id) == 1
        before_cycles = store.cycle_count(run_id)
        before_events = store._conn.execute(
            "SELECT COUNT(*) AS c FROM events WHERE run_id = ?", (run_id,)
        ).fetchone()["c"]

        busy = _cycle(
            client,
            SCIENTIST,
            dose_a_ml=1.0,
            dose_b_ml=1.0,
            rationale="route guard must fire before engine preflight",
        )
        assert busy.status_code == 409
        assert busy.json()["detail"] == "a cycle is already running"
        # No side effect from the second request: route guard rejected it before
        # the engine's preflight or any decision write could run.
        assert _decision_count(store, run_id) == 1
        assert store.cycle_count(run_id) == before_cycles
        assert store._conn.execute(
            "SELECT COUNT(*) AS c FROM events WHERE run_id = ?", (run_id,)
        ).fetchone()["c"] == before_events

        release.set()
        thread.join(timeout=5)
        assert result["response"].status_code == 200
        assert result["response"].json()["aborted"] is None


# --- operate-wall T2: frame/stream routes, static serving, wiring ---


class _CountingCamera:
    """Fake camera that only records how many times capture_jpeg is called --
    used to observe the live-capture loop's periodic behavior without depending
    on any exposed FrameSource/seq internals (create_app doesn't expose them)."""

    def __init__(self, frame=None):
        self._defaults = dict(FAST_CAMERA_DEFAULTS)
        self._controls = dict(self._defaults)
        self._frame = frame or _solid_jpeg()
        self.calls = 0

    def apply_controls(self, controls):
        self._controls.update(controls)

    def read_controls(self, names):
        return {name: self._controls[name] for name in names}

    def capture_jpeg(self):
        self.calls += 1
        return self._frame


def _console_dir(tmp_path):
    """A minimal console/ directory (placeholder operate.html/wall.html -- T4/T5
    write the real content later) for the routes that require console_dir to be set."""
    directory = tmp_path / "console"
    directory.mkdir()
    (directory / "operate.html").write_text(
        "<html>operate placeholder</html>", encoding="utf-8"
    )
    (directory / "wall.html").write_text(
        "<html>wall placeholder</html>", encoding="utf-8"
    )
    return directory


class _FrameSourceSpy(FrameSource):
    """Records every raw payload published so tests can assert the cache stayed raw
    while the HTTP surface served annotated copies (T5 D4)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.published_raw: list[bytes] = []

    def publish(self, jpeg: bytes) -> None:
        self.published_raw.append(jpeg)
        super().publish(jpeg)


def _patch_box(patch: PatchSpec, width: int, height: int):
    """Mirror patch_box math for overlay assertions."""
    left = round(patch.cx * width - patch.size_px / 2)
    top = round(patch.cy * height - patch.size_px / 2)
    return left, top, left + patch.size_px, top + patch.size_px


def _box_edges(img, left, top, right, bottom):
    for x in range(left, right):
        yield img.getpixel((x, top))
        yield img.getpixel((x, bottom - 1))
    for y in range(top, bottom):
        yield img.getpixel((left, y))
        yield img.getpixel((right - 1, y))


def _extract_jpeg_from_part(part: bytes) -> bytes:
    """Pull the JPEG payload out of a multipart/x-mixed-replace part."""
    header_end = part.find(b"\r\n\r\n")
    assert header_end != -1, "multipart part missing header/body separator"
    payload = part[header_end + 4 :]
    if payload.endswith(b"\r\n"):
        payload = payload[:-2]
    return payload


def test_frame_latest_404_before_any_capture(tmp_path):
    # Mutation kill: the `latest is None` guard -- before any capture (measure/blank
    # or the live loop) has ever run, /frame/latest must 404, not serve stale/empty
    # bytes.
    app = build_app(tmp_path, start_run=False)
    with TestClient(app) as client:
        response = client.get("/frame/latest")
        assert response.status_code == 404


def test_frame_latest_serves_annotated_cycle_frame(tmp_path):
    # Exercises the real fake-cycle harness (not a hand-constructed shortcut) so the
    # test actually proves create_app's camera -> FrameSource/SharedCamera wiring:
    # engine captures during MEASURE and BLANK publish to the same cache /frame/latest
    # reads from, with the live-capture loop off (no console_dir).
    # Renamed from test_frame_latest_serves_cycle_frame_bytes because T5 makes
    # /frame/latest serve a request-time annotated copy, so the raw-byte equality
    # assertion is no longer the correct behavior. We now assert the served frame is
    # a same-size annotated JPEG and that an un-annotated corner still matches the
    # source frame, preserving the original intent: the right frame reached the route.
    raw = _solid_jpeg(color=(10, 200, 90), size=64)
    camera = FakeCamera(defaults=FAST_CAMERA_DEFAULTS, frame=raw)
    app = build_app(tmp_path, camera=camera, start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        response = _cycle(client, SCIENTIST, dose_a_ml=1.0, dose_b_ml=1.0)
        assert response.status_code == 200
        assert response.json()["aborted"] is None

        # The fake camera returns the same fixed bytes on every capture, so the
        # assertion holds regardless of whether measure's or blank's capture
        # published last -- both are the blank capture's bytes exactly.
        frame_response = client.get("/frame/latest")
        assert frame_response.status_code == 200
        assert frame_response.headers["content-type"] == "image/jpeg"
        # Under T5 /frame/latest returns a serve-time annotated copy, not raw bytes.
        assert frame_response.content != raw

        served = Image.open(io.BytesIO(frame_response.content))
        assert served.size == (64, 64)
        # Confirm the right frame reached the route by checking a corner well outside
        # the default centered patch box (22, 22, 42, 42) is unchanged from the source.
        raw_img = Image.open(io.BytesIO(raw))
        for point in ((5, 5), (5, 58), (58, 5), (58, 58)):
            assert served.getpixel(point) == raw_img.getpixel(point)


def test_stream_mjpg_route_serves_multipart(tmp_path):
    # The emit-only-on-new-frame behaviour is proved directly against the generator
    # in tests/test_frames.py::test_stream_mjpg_yields_new_frames_only. It cannot be
    # driven over an in-process ASGI transport: both TestClient and httpx's
    # ASGITransport run the app to completion before returning a buffered response,
    # so an endless StreamingResponse deadlocks at stream-open. Here we assert only
    # what an in-process client can observe without consuming the stream -- the route
    # exists, is wired to the frame cache's generator, and advertises the streaming
    # multipart content type.
    import asyncio

    app = build_app(tmp_path)
    endpoint = next(
        (route.endpoint for route in app.routes
         if getattr(route, "path", None) == "/stream.mjpg"),
        None,
    )
    assert endpoint is not None, "no /stream.mjpg route"

    async def call():
        response = await endpoint()
        try:
            assert response.media_type.startswith("multipart/x-mixed-replace")
            assert "boundary=" in response.media_type
        finally:
            # Finalize the un-started generator so the run leaves no warning.
            await response.body_iterator.aclose()

    asyncio.run(call())


def test_stream_mjpg_route_overlay_tracks_patch_change(tmp_path):
    # Regression kill for a one-line bug at /stream.mjpg that hardcodes a PatchSpec
    # instead of wiring patch_holder.current: the existing route test only checks
    # media type without ever driving the generator, and tests/test_frames.py's
    # overlay-tracks-live-patch test builds its own patch_source. This test drives
    # the real route's StreamingResponse and proves POST /patch moves the overlay.
    import asyncio

    raw = _solid_jpeg(color=(10, 200, 90), size=64)
    camera = FakeCamera(defaults=FAST_CAMERA_DEFAULTS, frame=raw)
    app = build_app(tmp_path, camera=camera, start_run=True)

    endpoint = next(
        (
            route.endpoint
            for route in app.routes
            if getattr(route, "path", None) == "/stream.mjpg"
        ),
        None,
    )
    assert endpoint is not None, "no /stream.mjpg route"

    async def call():
        with TestClient(app) as client:
            client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
            client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

            # Publish an initial frame through the real camera/SharedCamera wiring.
            cycle_resp = _cycle(client, SCIENTIST, dose_a_ml=1.0, dose_b_ml=1.0)
            assert cycle_resp.status_code == 200

            response = await endpoint()
            try:
                part1 = await response.body_iterator.__anext__()
                img1 = Image.open(io.BytesIO(_extract_jpeg_from_part(part1)))
                assert img1.size == (64, 64)

                # Move the patch via the real HTTP route.
                patch_resp = client.post(
                    "/patch",
                    json={"cx": 0.84, "cy": 0.16, "size_px": 20},
                    headers=CONSOLE,
                )
                assert patch_resp.status_code == 200

                # Publish another frame so the stream has a new part to emit.
                cycle_resp2 = _cycle(client, SCIENTIST, dose_a_ml=1.0, dose_b_ml=1.0)
                assert cycle_resp2.status_code == 200

                part2 = await response.body_iterator.__anext__()
                img2 = Image.open(io.BytesIO(_extract_jpeg_from_part(part2)))
                assert img2.size == (64, 64)

                bg = (10, 200, 90)
                old_patch = PatchSpec(cx=0.5, cy=0.5, size_px=20)
                old_box = _patch_box(old_patch, 64, 64)
                old_edges = list(_box_edges(img2, *old_box))
                assert all(
                    all(abs(a - b) <= 30 for a, b in zip(pixel, bg))
                    for pixel in old_edges
                ), "streamed frame should show the current patch, not the seed patch"

                new_patch = PatchSpec(cx=0.84, cy=0.16, size_px=20)
                new_box = _patch_box(new_patch, 64, 64)
                new_edges = list(_box_edges(img2, *new_box))
                assert any(
                    any(abs(a - b) > 30 for a, b in zip(pixel, bg))
                    for pixel in new_edges
                ), "current patch box should be visible on the streamed frame"
            finally:
                await response.body_iterator.aclose()

    asyncio.run(call())


def test_frame_latest_serves_annotated_copy_cache_stays_raw(tmp_path, monkeypatch):
    # D4: the cache stores raw frames; /frame/latest draws the overlay on a copy at
    # request time. We spy on FrameSource because create_app does not expose it.
    raw = _solid_jpeg(color=(10, 200, 90), size=64)
    camera = FakeCamera(defaults=FAST_CAMERA_DEFAULTS, frame=raw)

    spies: list[_FrameSourceSpy] = []

    def make_spy(*args, **kwargs):
        spy = _FrameSourceSpy(*args, **kwargs)
        spies.append(spy)
        return spy

    monkeypatch.setattr("rig_service.app.FrameSource", make_spy)
    app = build_app(tmp_path, camera=camera, start_run=True)
    assert len(spies) == 1
    frames = spies[0]

    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        response = _cycle(client, SCIENTIST, dose_a_ml=1.0, dose_b_ml=1.0)
        assert response.status_code == 200

        # Pinned invariant extension: the cached bytes are exactly what the camera captured.
        assert frames.latest == raw

        frame_response = client.get("/frame/latest")
        assert frame_response.status_code == 200
        assert frame_response.headers["content-type"] == "image/jpeg"
        # The served bytes are an annotated copy, not the raw cache contents.
        assert frame_response.content != raw


def test_frame_latest_overlay_uses_current_patch_on_stale_frame(tmp_path):
    # A frame published before the patch move must show the NEW patch when served,
    # proving overlay happens at request time, not publish time.
    raw = _solid_jpeg(color=(10, 200, 90), size=64)
    camera = FakeCamera(defaults=FAST_CAMERA_DEFAULTS, frame=raw)
    app = build_app(tmp_path, camera=camera, start_run=True)

    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        response = _cycle(client, SCIENTIST, dose_a_ml=1.0, dose_b_ml=1.0)
        assert response.status_code == 200

        # Move the patch; do NOT publish a new frame. Coordinates chosen so the new
        # box (44, 0, 64, 20) does not overlap the old seed box (22, 22, 42, 42).
        response = client.post(
            "/patch",
            json={"cx": 0.84, "cy": 0.16, "size_px": 20},
            headers=CONSOLE,
        )
        assert response.status_code == 200

        frame_response = client.get("/frame/latest")
        assert frame_response.status_code == 200
        served = Image.open(io.BytesIO(frame_response.content))
        assert served.size == (64, 64)

        bg = (10, 200, 90)
        # The default-seed box should no longer be annotated on the stale frame.
        old_patch = PatchSpec(cx=0.5, cy=0.5, size_px=20)
        old_box = _patch_box(old_patch, 64, 64)
        old_edges = list(_box_edges(served, *old_box))
        assert all(
            all(abs(a - b) <= 30 for a, b in zip(pixel, bg)) for pixel in old_edges
        ), "stale frame should show the current patch, not the seed patch"

        # The new patch box should be annotated.
        new_patch = PatchSpec(cx=0.84, cy=0.16, size_px=20)
        new_box = _patch_box(new_patch, 64, 64)
        new_edges = list(_box_edges(served, *new_box))
        assert any(
            any(abs(a - b) > 30 for a, b in zip(pixel, bg)) for pixel in new_edges
        ), "current patch box should be visible on the stale frame"


def test_measure_path_never_sees_annotated_bytes(tmp_path):
    # Pinned D4 invariant: measurement/blank frames stored in the cycle record are
    # the raw camera captures, never the serve-time annotated copies.
    raw = _solid_jpeg(color=(10, 200, 90), size=64)
    camera = FakeCamera(defaults=FAST_CAMERA_DEFAULTS, frame=raw)
    app, store = _app_with_store(
        tmp_path,
        config=make_fast_cycle_config(),
        camera=camera,
        driver=FakeRelayDriver(),
        volumes=VolumeState(0.0, 0.0, 0.0),
        start_run=True,
    )

    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        response = _cycle(client, SCIENTIST, dose_a_ml=1.0, dose_b_ml=1.0)
        assert response.status_code == 200

        # A served frame is annotated, so it must differ from raw.
        served = client.get("/frame/latest").content
        assert served != raw

        run_id = client.get("/status", headers=CONSOLE).json()["run"]["id"]
        row = store._conn.execute(
            "SELECT measurement_frame_path, blank_frame_path FROM cycles WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        assert row is not None

        for rel in (row["measurement_frame_path"], row["blank_frame_path"]):
            assert rel is not None
            saved_bytes = (store._frames_dir / rel).read_bytes()
            assert saved_bytes == raw, "cycle-saved frame must be raw camera bytes"
            assert saved_bytes != served, "cycle-saved frame must not match annotated bytes"


def test_status_console_reports_last_frame_age(tmp_path):
    camera = FakeCamera(defaults=FAST_CAMERA_DEFAULTS, frame=_solid_jpeg())
    app = build_app(tmp_path, camera=camera, start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)

        before = client.get("/status", headers=CONSOLE).json()
        assert before["last_frame_age_s"] is None  # no capture has happened yet

        response = _cycle(client, SCIENTIST, dose_a_ml=1.0, dose_b_ml=1.0)
        assert response.status_code == 200

        after = client.get("/status", headers=CONSOLE).json()
        assert after["last_frame_age_s"] is not None
        assert after["last_frame_age_s"] >= 0


def test_status_scientist_omits_last_frame_age(tmp_path):
    # Mutation kill: the viewer branch -- last_frame_age_s is console-only.
    app = build_app(tmp_path, start_run=True)
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
        client.post("/target", json={"hex": "#3366cc"}, headers=CONSOLE)
        response = _cycle(client, SCIENTIST, dose_a_ml=1.0, dose_b_ml=1.0)
        assert response.status_code == 200  # a capture definitely happened

        sci = client.get("/status").json()
        assert "last_frame_age_s" not in sci


def test_operate_wall_routes_serve_html_when_console_dir_set(tmp_path):
    console_dir = _console_dir(tmp_path)
    store = _make_store(tmp_path, volumes=VolumeState(0.0, 0.0, 0.0))
    app = create_app(
        config=make_fast_cycle_config(),
        driver=FakeRelayDriver(),
        camera=FakeCamera(defaults=FAST_CAMERA_DEFAULTS, frame=_solid_jpeg()),
        store=store,
        console_dir=console_dir,
    )
    with TestClient(app) as client:
        operate = client.get("/operate")
        assert operate.status_code == 200
        assert "operate placeholder" in operate.text

        wall = client.get("/wall")
        assert wall.status_code == 200
        assert "wall placeholder" in wall.text


def test_console_routes_404_without_console_dir(tmp_path):
    # Mutation kill: `console_dir is None` -- the default create_app() call (every
    # existing call site in this suite) must not serve /operate or /wall.
    app = build_app(tmp_path, start_run=False)
    with TestClient(app) as client:
        assert client.get("/operate").status_code == 404
        assert client.get("/wall").status_code == 404


def test_static_serves_console_asset_when_console_dir_set(tmp_path):
    # Closes the spec responsibility "rig-service serves the console's static files":
    # with console_dir set, /static/<file> returns the asset's bytes.
    console_dir = _console_dir(tmp_path)
    (console_dir / "common.css").write_text("body{color:#000}", encoding="utf-8")
    store = _make_store(tmp_path, volumes=VolumeState(0.0, 0.0, 0.0))
    app = create_app(
        config=make_fast_cycle_config(),
        driver=FakeRelayDriver(),
        camera=FakeCamera(defaults=FAST_CAMERA_DEFAULTS, frame=_solid_jpeg()),
        store=store,
        console_dir=console_dir,
    )
    with TestClient(app) as client:
        response = client.get("/static/common.css")
        assert response.status_code == 200
        assert response.text == "body{color:#000}"


def test_static_404_without_console_dir(tmp_path):
    # Mutation kill: the /static mount is created only when console_dir is set.
    app = build_app(tmp_path, start_run=False)
    with TestClient(app) as client:
        assert client.get("/static/common.css").status_code == 404


def test_root_redirects_to_operate(tmp_path):
    app = build_app(tmp_path, start_run=False)
    with TestClient(app, follow_redirects=False) as client:
        response = client.get("/")
        assert response.status_code == 307
        assert response.headers["location"] == "/operate"


def test_live_capture_task_runs_only_with_console_dir(tmp_path):
    # Mutation kill: the loop gate -- create_app only starts the live-capture task
    # when console_dir is given. Two isolated apps (distinct data dirs, distinct
    # camera fakes) so neither app's boot/store state can leak into the other's
    # capture count.
    console_dir = _console_dir(tmp_path)

    with_dir = tmp_path / "with"
    with_dir.mkdir()
    with_camera = _CountingCamera()
    with_store = _make_store(with_dir, volumes=VolumeState(0.0, 0.0, 0.0))
    with_app = create_app(
        config=make_fast_cycle_config(),
        driver=FakeRelayDriver(),
        camera=with_camera,
        store=with_store,
        console_dir=console_dir,
    )
    with TestClient(with_app):
        # No /cycle was ever called -- every capture observed here comes from the
        # live loop running unattended inside the lifespan.
        deadline = time.monotonic() + 3.0
        while with_camera.calls < 2:
            assert time.monotonic() < deadline, (
                "live_capture never advanced with console_dir set"
            )
            time.sleep(0.05)

    without_dir = tmp_path / "without"
    without_dir.mkdir()
    without_camera = _CountingCamera()
    without_store = _make_store(without_dir, volumes=VolumeState(0.0, 0.0, 0.0))
    without_app = create_app(
        config=make_fast_cycle_config(),
        driver=FakeRelayDriver(),
        camera=without_camera,
        store=without_store,
        console_dir=None,
    )
    with TestClient(without_app):
        time.sleep(1.2)  # comfortably longer than one LIVE_CAPTURE_INTERVAL_S tick
    assert without_camera.calls == 0


def test_live_capture_task_cancelled_on_shutdown(tmp_path, caplog):
    console_dir = _console_dir(tmp_path)
    camera = _CountingCamera()
    store = _make_store(tmp_path, volumes=VolumeState(0.0, 0.0, 0.0))
    app = create_app(
        config=make_fast_cycle_config(),
        driver=FakeRelayDriver(),
        camera=camera,
        store=store,
        console_dir=console_dir,
    )
    with caplog.at_level(logging.WARNING):
        with TestClient(app):
            deadline = time.monotonic() + 2.5
            while camera.calls < 1:
                assert time.monotonic() < deadline, "live_capture never ran"
                time.sleep(0.05)

        calls_after_shutdown = camera.calls
        time.sleep(1.2)  # comfortably longer than one LIVE_CAPTURE_INTERVAL_S tick

    # The task was cancelled at shutdown, not still running in the background.
    assert camera.calls == calls_after_shutdown

    # A cancelled task whose result/exception is never awaited logs "exception was
    # never retrieved" / "Task was destroyed but it is pending" -- output must stay
    # pristine, so a clean cancellation (awaited inside the lifespan's shutdown path)
    # must produce neither.
    unretrieved = [
        record
        for record in caplog.records
        if "never retrieved" in record.getMessage()
        or "was destroyed" in record.getMessage()
    ]
    assert unretrieved == []


# --- operate-wall T3: calibrate primitive + endpoint ---


def calibrate(client, channel, seconds, headers=CONSOLE):
    return client.post(
        "/calibrate", json={"channel": channel, "seconds": seconds}, headers=headers
    )


def test_calibrate_scientist_client_409(driver, manual):
    # Mutation kill: the client guard -- the detail must name calibrate, not pump
    # (the arbiter's authorize call must carry "calibrate", not a hardcoded "pump").
    response = calibrate(manual, "A", 0.06, headers=SCIENTIST)
    assert response.status_code == 409
    detail = response.json()["detail"].lower()
    assert "calibrate" in detail
    assert "pump" not in detail
    assert len(driver.history) == 1  # only the boot arming event -- relay untouched


def test_calibrate_auto_mode_409(driver, client):
    # Mutation kill: the mode guard -- console in auto (not manual) must still 409,
    # with the calibrate-specific detail.
    client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)
    response = calibrate(client, "A", 0.06)
    assert response.status_code == 409
    detail = response.json()["detail"].lower()
    assert "calibrate" in detail
    assert "pump" not in detail
    assert len(driver.history) == 1


def test_calibrate_returns_est_ml_and_writes_event(tmp_path):
    app, store = _app_with_store(
        tmp_path, config=make_config(), volumes=VolumeState(0.0, 0.0, 0.0), start_run=False
    )
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)

        # W: rate 100 ml/s, lag 0.01 s. seconds=1.01 -> ml = 100*(1.01-0.01) = 100.
        response = calibrate(client, "W", 1.01)

        assert response.status_code == 200
        body = response.json()
        assert body["channel"] == "W"
        assert body["seconds"] == 1.01
        assert body["est_ml"] == pytest.approx(100.0)

        row = store._conn.execute(
            "SELECT detail FROM events WHERE kind = 'calibrate' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        detail = json.loads(row["detail"])
        assert detail["channel"] == "W"
        assert detail["seconds"] == pytest.approx(1.01)
        assert detail["est_ml"] == pytest.approx(100.0)

        volumes_state = store.load_volumes()
        assert volumes_state.cup_ml == pytest.approx(100.0)
        assert volumes_state.water_drawn_ml == pytest.approx(100.0)


def test_calibrate_over_cap_400(driver, manual):
    # A's max_ml_per_dose is 10. seconds=1.01 -> ml = 100*(1.01-0.01) = 100, comfortably
    # over the cap.
    response = calibrate(manual, "A", 1.01)
    assert response.status_code == 400
    assert "cap" in response.json()["detail"].lower()
    assert len(driver.history) == 1


# --- /scientist/state: liveness reporting endpoint ---


def _state_app(tmp_path):
    return _app_with_store(tmp_path, volumes=VolumeState(0.0, 0.0, 0.0))


def test_scientist_state_console_client_409(tmp_path):
    app, _store = _state_app(tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/scientist/state", json={"state": "idle", "detail": ""}, headers=CONSOLE
        )
        assert response.status_code == 409


def test_scientist_state_missing_header_400(tmp_path):
    app, _store = _state_app(tmp_path)
    with TestClient(app) as client:
        response = client.post("/scientist/state", json={"state": "idle", "detail": ""})
        assert response.status_code == 400


def test_scientist_state_valid_report_200(tmp_path):
    app, _store = _state_app(tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/scientist/state",
            json={"state": "thinking", "detail": "asking the model"},
            headers=SCIENTIST,
        )
        assert response.status_code == 200


def test_scientist_state_invalid_state_400_never_coerced(tmp_path):
    app, _store = _state_app(tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/scientist/state", json={"state": "asleep", "detail": ""}, headers=SCIENTIST
        )
        assert response.status_code == 400
        # A rejected report must not disturb the held state.
        body = client.get("/status", headers=CONSOLE).json()
        assert body["scientist"] == {"state": "unknown", "detail": "", "age_s": None}


def test_scientist_state_overlong_detail_400(tmp_path):
    app, _store = _state_app(tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/scientist/state",
            json={"state": "idle", "detail": "x" * 201},
            headers=SCIENTIST,
        )
        assert response.status_code == 400
        response = client.post(
            "/scientist/state",
            json={"state": "idle", "detail": "x" * 200},
            headers=SCIENTIST,
        )
        assert response.status_code == 200


def test_status_console_view_carries_scientist_state(tmp_path):
    app, _store = _state_app(tmp_path)
    with TestClient(app) as client:
        body = client.get("/status", headers=CONSOLE).json()
        assert body["scientist"] == {"state": "unknown", "detail": "", "age_s": None}

        response = client.post(
            "/scientist/state",
            json={"state": "thinking", "detail": "asking the model"},
            headers=SCIENTIST,
        )
        assert response.status_code == 200

        body = client.get("/status", headers=CONSOLE).json()
        assert body["scientist"]["state"] == "thinking"
        assert body["scientist"]["detail"] == "asking the model"
        assert body["scientist"]["age_s"] >= 0.0


def test_status_scientist_view_has_no_scientist_field(tmp_path):
    app, _store = _state_app(tmp_path)
    with TestClient(app) as client:
        client.post(
            "/scientist/state", json={"state": "idle", "detail": ""}, headers=SCIENTIST
        )
        body = client.get("/status", headers=SCIENTIST).json()
        assert "scientist" not in body


def test_scientist_stop_event_written_once_per_transition(tmp_path):
    app, store = _state_app(tmp_path)
    with TestClient(app) as client:
        for _ in range(2):
            response = client.post(
                "/scientist/state",
                json={"state": "stopped_budget", "detail": "budget"},
                headers=SCIENTIST,
            )
            assert response.status_code == 200

        rows = store._conn.execute(
            "SELECT detail, scientist_visible FROM events WHERE kind = 'scientist_stop'"
        ).fetchall()
        assert len(rows) == 1
        assert json.loads(rows[0]["detail"]) == {"state": "stopped_budget", "detail": "budget"}
        assert rows[0]["scientist_visible"] == 1

        # A different state then a fresh stop verdict is a new transition -> new event.
        client.post("/scientist/state", json={"state": "idle", "detail": ""}, headers=SCIENTIST)
        client.post(
            "/scientist/state",
            json={"state": "stopped_converged", "detail": "converged_delta_e"},
            headers=SCIENTIST,
        )
        rows = store._conn.execute(
            "SELECT detail FROM events WHERE kind = 'scientist_stop' ORDER BY id"
        ).fetchall()
        assert len(rows) == 2
        assert json.loads(rows[1]["detail"]) == {
            "state": "stopped_converged",
            "detail": "converged_delta_e",
        }
