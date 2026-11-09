# ABOUTME: Tests for CycleEngine's clean-path sequencing, preflight rejections, and
# ABOUTME: record-population invariants (T5a); T5b owns the failure-mode tests.

import asyncio
import io
import json
import threading
import time
from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image, ImageDraw

from rig_service.camera_lock import CameraLock
from rig_service.config import (
    PHASE_BUDGET_KEYS,
    CameraConfig,
    Channel,
    CycleConfig,
    RigConfig,
    VolumeLimits,
)
from rig_service.cycle import CycleEngine, CycleError, CycleRecord, Phase
from rig_service.disrupt import DisruptError, Disruptions
from rig_service.drivers.fake import FakeRelayDriver
from rig_service.mode import ModeArbiter
from rig_service.patch import PatchSpec
from rig_service.pumps import PumpController
from rig_service.store import Store
from rig_service.volume import VolumeAccountant, VolumeState
from shared.color import delta_e, rgb_to_lab, score

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend():
    return "asyncio"


# The six v4l2 controls the spec names as the asserted lock domains (mirrors
# tests/test_camera_lock.py's LOCK_CONTROLS). The camera doubles below are built holding
# exactly these values, so CameraLock.verify() == [] on the clean path -- no alarms.
LOCK_CONTROLS = {
    "auto_exposure": 1,
    "exposure_time_absolute": 250,
    "white_balance_automatic": 0,
    "white_balance_temperature": 4600,
    "focus_automatic_continuous": 0,
    "focus_absolute": 85,
}


def make_config(volumes=None):
    # Fast rates (1000 ml/s) + tiny lag keep every pump's on-time in the low tens of
    # milliseconds, and settle_s=0.01 keeps SETTLE real but not slow. Phase budgets
    # (1.0 s each) sit comfortably above every worst-case pump second computed below,
    # so nothing watchdog-trips on the clean path:
    #   DOSE worst case  = 2 * (0.001 + 5/1000)  = 0.012 s  (A, B at their 5 ml cap)
    #   REFILL/RINSE     =      0.001 + 10/1000  = 0.011 s  (W at refill_ml/rinse_ml=10)
    #   DRAIN/FINAL_DRAIN=      0.001 + 20/1000  = 0.021 s  (D at drain_ml=20)
    # Cup ceiling (20) == refill_ml(10) + A.max(5) + B.max(5) (the worst-case-fill
    # check) and == drain_ml (the "a full drain always empties the cup" check).
    camera = CameraConfig(
        device="/dev/video0", width=64, height=64, patch_px=32, controls=dict(LOCK_CONTROLS)
    )
    cycle = CycleConfig(
        refill_ml=10,
        rinse_ml=10,
        drain_ml=20,
        settle_s=0.01,
        blank_drift_delta_e=5.0,
        phase_budgets_s={
            "dose": 1.0,
            "refill": 1.0,
            "settle": 1.0,
            "measure": 1.0,
            "drain": 1.0,
            "rinse": 1.0,
            "blank": 1.0,
            "final_drain": 1.0,
        },
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


def tiny_budgets_config(volumes=None, phase_budgets_s=None):
    """make_config() with named phase budgets shrunk to real-but-fast values (0.01-0.05 s)
    -- T5b's watchdog tests need a genuine asyncio.wait_for timeout, not a multi-second
    sleep. Only override phases with no pump-worst-case-time floor (measure/blank); the
    rest keep make_config's comfortable 1.0 s so nothing else spuriously trips."""
    config = make_config(volumes=volumes)
    budgets = dict(config.cycle.phase_budgets_s)
    if phase_budgets_s:
        budgets.update(phase_budgets_s)
    return replace(config, cycle=replace(config.cycle, phase_budgets_s=budgets))


TARGET_LAB = rgb_to_lab((51, 102, 204))
LAM = 0.05


def _solid_jpeg(color, size=40):
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


class CleanCamera:
    """Locked controls, decodable solid frame -- no alarms, no blocking."""

    def __init__(self, controls=None, color=(200, 60, 30), size=40):
        self._controls = dict(controls or LOCK_CONTROLS)
        self._frame = _solid_jpeg(color, size)

    def read_controls(self, names):
        return {name: self._controls[name] for name in names}

    def apply_controls(self, controls):
        self._controls.update(controls)

    def capture_jpeg(self):
        return self._frame


class BlockingCamera(CleanCamera):
    """Blocks capture_jpeg (in the to_thread worker) until released -- used to hold a
    cycle mid-MEASURE so a concurrent second run_cycle can observe busy rejection."""

    def __init__(self, event, **kwargs):
        super().__init__(**kwargs)
        self._event = event

    def capture_jpeg(self):
        self._event.wait(timeout=5)
        return self._frame


class HangingCaptureCamera(CleanCamera):
    """capture_jpeg blocks for hang_s -- forces a real watchdog timeout at MEASURE or
    BLANK. Both call capture_jpeg via asyncio.to_thread, so the event loop stays free
    and asyncio.wait_for's timeout fires on schedule."""

    def __init__(self, hang_s, **kwargs):
        super().__init__(**kwargs)
        self._hang_s = hang_s

    def capture_jpeg(self):
        time.sleep(self._hang_s)
        return self._frame


class HangingReadControlsCamera(CleanCamera):
    """read_controls blocks for hang_s. CameraLock.verify() at BLANK calls this directly
    on the event loop (unlike the adjacent capture_jpeg, it is NOT wrapped in
    asyncio.to_thread) -- hardening target (b): a wedged read_controls freezes the whole
    loop, so the BLANK watchdog cannot fire on time."""

    def __init__(self, hang_s, **kwargs):
        super().__init__(**kwargs)
        self._hang_s = hang_s

    def read_controls(self, names):
        time.sleep(self._hang_s)
        return super().read_controls(names)


class ErroringCamera(CleanCamera):
    """capture_jpeg raises -- a non-timeout phase-body failure (decision 3: one abort
    path handles both timeouts and exceptions)."""

    def capture_jpeg(self):
        raise RuntimeError("camera disconnected")


REVERTED_CONTROLS = {
    "auto_exposure": 3,
    "exposure_time_absolute": 625,
    "white_balance_automatic": 1,
    "white_balance_temperature": 4000,
    "focus_automatic_continuous": 1,
    "focus_absolute": 0,
}


class PowerCyclableCamera(CleanCamera):
    """Like drivers/fake.py's FakeCamera.power_cycle: a USB power loss reverts every
    control to firmware defaults, distinct from the locked values."""

    def __init__(self, reverted_controls=REVERTED_CONTROLS, **kwargs):
        super().__init__(**kwargs)
        self._reverted_controls = dict(reverted_controls)

    def power_cycle(self):
        self._controls = dict(self._reverted_controls)


class DriftingBlankCamera(CleanCamera):
    """MEASURE always sees the fixed measurement frame; each successive BLANK phase
    serves the next queued blank color, in order -- lets a test set up a clean first
    blank (the drift reference, decision 7) then a drifted or clean later blank."""

    def __init__(self, blank_colors, measure_color=(200, 60, 30), **kwargs):
        super().__init__(color=measure_color, **kwargs)
        self._blank_colors = list(blank_colors)
        self._capture_count = 0

    def capture_jpeg(self):
        self._capture_count += 1
        if self._capture_count % 2 == 0:  # even calls are BLANK (odd calls are MEASURE)
            blank_index = self._capture_count // 2 - 1
            return _solid_jpeg(self._blank_colors[blank_index])
        return self._frame


class HangScenarioCamera(CleanCamera):
    """Which single point hangs is chosen by `scenario`; used by the property test that
    sweeps over which phase/call hangs."""

    def __init__(self, scenario, hang_s, **kwargs):
        super().__init__(**kwargs)
        self._scenario = scenario
        self._hang_s = hang_s
        self._capture_count = 0

    def capture_jpeg(self):
        self._capture_count += 1
        is_measure_call = self._capture_count == 1  # MEASURE captures first, always
        if (self._scenario == "measure_capture" and is_measure_call) or (
            self._scenario == "blank_capture" and not is_measure_call
        ):
            time.sleep(self._hang_s)
        return self._frame

    def read_controls(self, names):
        if self._scenario == "blank_verify":
            time.sleep(self._hang_s)
        return super().read_controls(names)


class BudgetInjectingDriver(FakeRelayDriver):
    """Injects an extra, untracked reservoir draw the instant a chosen channel's relay
    first turns on -- simulates a session-level draw the static preflight plan (which
    only reasons about THIS cycle's own planned refill_ml + rinse_ml) could not
    foresee, so it is the phase's OWN post-dose budget check
    (PumpController._trip_if_budget_low), not preflight, that trips mid-cycle.
    holder["accountant"] must be filled in by the caller once build_engine() returns
    the accountant this driver has to share with the engine it drives."""

    def __init__(self, holder, trigger_gpio, extra_ml):
        super().__init__()
        self._holder = holder
        self._trigger_gpio = trigger_gpio
        self._extra_ml = extra_ml
        self._injected = False

    def set(self, gpio, on):
        if on and gpio == self._trigger_gpio and not self._injected:
            self._injected = True
            # Bumps reservoir accounting only (not the cup) -- VolumeAccountant.fill()
            # always fills the cup too, and the cup ceiling is far too small to carry a
            # realistic session-level reservoir draw. Direct state mutation, not a mock:
            # it simulates a real physical event (another draw against the shared
            # session reservoir) the accountant's public API has no other way to model.
            self._holder["accountant"]._water_drawn_ml += self._extra_ml
        super().set(gpio, on)


class RecordingSink:
    """CycleSink-shaped test double that logs every call in order."""

    def __init__(self, cycle_id=1):
        self.calls = []  # (method_name, *args) tuples, in call order
        self._cycle_id = cycle_id

    def cycle_started(self, record):
        self.calls.append(("cycle_started", record))
        return self._cycle_id

    def phase_committed(self, cycle_id, phase, record):
        self.calls.append(("phase_committed", cycle_id, phase, record))

    def cycle_finished(self, cycle_id, record):
        self.calls.append(("cycle_finished", cycle_id, record))


class RaisingStartSink(RecordingSink):
    """Raises from cycle_started to prove the engine rejects before actuation."""

    def cycle_started(self, record):
        raise RuntimeError("sink down at start")


class RaisingPhaseSink(RecordingSink):
    """Raises from phase_committed after a chosen phase to prove the abort path."""

    def __init__(self, raise_after_phase, cycle_id=1):
        super().__init__(cycle_id)
        self._raise_after_phase = raise_after_phase

    def phase_committed(self, cycle_id, phase, record):
        super().phase_committed(cycle_id, phase, record)
        if phase == self._raise_after_phase:
            raise RuntimeError("sink down mid-cycle")


def store_sink(store, run_id, accountant_holder):
    """Local CycleSink adapter around a real Store for the T4 integration tests."""

    class StoreSink:
        def cycle_started(self, record):
            return store.start_cycle(run_id, record, accountant_holder["accountant"].state)

        def phase_committed(self, cycle_id, phase, record):
            store.update_cycle(cycle_id, record, accountant_holder["accountant"].state)

        def cycle_finished(self, cycle_id, record):
            store.finish_cycle(cycle_id, record, accountant_holder["accountant"].state)

    return StoreSink()


_LOW_RESERVOIR_VOLUMES = VolumeLimits(
    cup_ceiling_ml=20, reservoir_ml=200, waste_capacity_ml=6000, low_margin_ml=20
)


def gpio_to_channel(config):
    return {channel.gpio: name for name, channel in config.channels.items()}


def relay_on_sequence(driver, config):
    mapping = gpio_to_channel(config)
    return [mapping[gpio] for event, gpio, _ in driver.history if event == "on"]


def build_engine(config, camera, driver=None, sink=None, disruptions=None, patch_source=None):
    driver = driver or FakeRelayDriver()
    sink = sink if sink is not None else RecordingSink()
    disruptions = disruptions if disruptions is not None else Disruptions(config)
    if patch_source is None:
        patch_source = lambda: PatchSpec(0.5, 0.5, config.camera.patch_px)
    accountant = VolumeAccountant(
        cup_ceiling_ml=config.volumes.cup_ceiling_ml,
        reservoir_ml=config.volumes.reservoir_ml,
        waste_capacity_ml=config.volumes.waste_capacity_ml,
        low_margin_ml=config.volumes.low_margin_ml,
        state=VolumeState(0.0, 0.0, 0.0),
        on_change=lambda _: None,
    )
    arbiter = ModeArbiter()
    arbiter.set_mode("manual", "console")  # any non-paused mode; the paused floor is pumps.py's own
    pumps = PumpController(config, driver, accountant, arbiter)
    camera_lock = CameraLock(camera, controls=config.camera.controls)
    engine = CycleEngine(
        config,
        pumps,
        camera,
        camera_lock,
        arbiter,
        accountant,
        driver,
        sink,
        disruptions,
        patch_source=patch_source,
    )
    return engine, driver, accountant, arbiter


# --- clean path ---


async def test_clean_cycle_phases_in_spec_order():
    config = make_config()
    engine, driver, accountant, arbiter = build_engine(config, CleanCamera())

    record = await engine.run_cycle(3.0, 2.0, TARGET_LAB, LAM)

    assert relay_on_sequence(driver, config) == ["A", "B", "W", "D", "W", "D"]
    assert record.aborted is None
    assert record.measured_lab is not None
    assert record.blank_lab is not None
    assert record.delta_e is not None
    assert record.score is not None
    assert record.alarms == []  # CleanCamera holds exactly the locked controls
    assert engine.phase == Phase.IDLE


def test_phase_enum_matches_config_budget_keys():
    non_idle = {phase.value for phase in Phase if phase != Phase.IDLE}
    assert non_idle == PHASE_BUDGET_KEYS


async def test_zero_dose_skips_that_pump():
    config = make_config()

    engine_a_zero, driver_a_zero, _, _ = build_engine(config, CleanCamera())
    record_a_zero = await engine_a_zero.run_cycle(0.0, 2.0, TARGET_LAB, LAM)
    assert relay_on_sequence(driver_a_zero, config) == ["B", "W", "D", "W", "D"]
    assert record_a_zero.aborted is None

    # decision 6: a pure-water probe (both doses zero) still runs the full plumbing.
    engine_both_zero, driver_both_zero, _, _ = build_engine(config, CleanCamera())
    record_both_zero = await engine_both_zero.run_cycle(0.0, 0.0, TARGET_LAB, LAM)
    assert relay_on_sequence(driver_both_zero, config) == ["W", "D", "W", "D"]
    assert record_both_zero.aborted is None


async def test_negative_dose_rejected_preflight():
    config = make_config()

    engine, driver, _, _ = build_engine(config, CleanCamera())
    with pytest.raises(CycleError):
        await engine.run_cycle(-1.0, 1.0, TARGET_LAB, LAM)
    assert relay_on_sequence(driver, config) == []

    engine2, driver2, _, _ = build_engine(config, CleanCamera())
    with pytest.raises(CycleError):
        await engine2.run_cycle(1.0, -1.0, TARGET_LAB, LAM)
    assert relay_on_sequence(driver2, config) == []


async def test_preflight_cap_violation_fires_no_relay():
    config = make_config()
    engine, driver, _, _ = build_engine(config, CleanCamera())

    with pytest.raises(CycleError):
        await engine.run_cycle(6.0, 1.0, TARGET_LAB, LAM)  # A's cap is 5 ml

    assert relay_on_sequence(driver, config) == []


async def test_preflight_rejects_nonempty_cup():
    config = make_config()
    engine, driver, accountant, _ = build_engine(config, CleanCamera())
    accountant.fill(5.0)  # simulates an abnormal non-drained cup, bypassing the pumps

    with pytest.raises(CycleError):
        await engine.run_cycle(1.0, 1.0, TARGET_LAB, LAM)

    assert relay_on_sequence(driver, config) == []


async def test_preflight_rejection_never_calls_patch_source():
    config = make_config()
    calls = []

    def counting_patch_source():
        calls.append("patch")
        return PatchSpec(0.5, 0.5, config.camera.patch_px)

    engine, driver, accountant, _ = build_engine(
        config, CleanCamera(), patch_source=counting_patch_source
    )
    accountant.fill(5.0)  # preflight-rejecting nonempty cup

    with pytest.raises(CycleError):
        await engine.run_cycle(1.0, 1.0, TARGET_LAB, LAM)

    assert relay_on_sequence(driver, config) == []
    assert calls == []


async def test_preflight_low_session_headroom_trips_paused_dry():
    # Reservoir staged to 30 ml; the cycle's planned water draw (refill_ml + rinse_ml =
    # 10 + 10 = 20 ml, decision 5) leaves headroom of 30 - 20 = 10 ml <= low_margin_ml
    # (20) -- the engine must trip to paused *dry* before any liquid moves.
    low_volumes = VolumeLimits(
        cup_ceiling_ml=20, reservoir_ml=30, waste_capacity_ml=6000, low_margin_ml=20
    )
    config = make_config(volumes=low_volumes)
    engine, driver, accountant, arbiter = build_engine(config, CleanCamera())

    with pytest.raises(CycleError):
        await engine.run_cycle(1.0, 1.0, TARGET_LAB, LAM)

    assert arbiter.mode == "paused"
    assert arbiter.trip_reason is not None
    assert "reservoir" in arbiter.trip_reason.lower()
    assert relay_on_sequence(driver, config) == []


async def test_busy_engine_rejects_second_cycle():
    config = make_config()
    event = threading.Event()
    engine, driver, _, _ = build_engine(config, BlockingCamera(event))

    task = asyncio.create_task(engine.run_cycle(1.0, 1.0, TARGET_LAB, LAM))
    await asyncio.sleep(0.15)  # DOSE+REFILL+SETTLE (well under 0.1 s total) have run; MEASURE blocks

    with pytest.raises(CycleError):
        await engine.run_cycle(1.0, 1.0, TARGET_LAB, LAM)

    event.set()
    record = await task
    assert record.aborted is None


async def test_record_scores_against_target():
    config = make_config()
    a, b = 3.0, 2.0
    engine, driver, _, _ = build_engine(config, CleanCamera(color=(200, 60, 30)))

    record = await engine.run_cycle(a, b, TARGET_LAB, LAM)

    assert record.aborted is None
    expected_delta_e = delta_e(record.measured_lab, TARGET_LAB)
    assert record.delta_e == pytest.approx(expected_delta_e)
    expected_score = score(record.delta_e, a + b, LAM)  # requested ml dye, not executed
    assert record.score == pytest.approx(expected_score)


async def test_doses_recorded_before_any_phase_runs():
    # invariant 5: the record must be published (with requested + executed doses) before
    # the first phase runs, not reconstructed after. Probe by reading engine.last_record
    # from inside the camera's capture_jpeg -- the call MEASURE makes mid-cycle.
    config = make_config()
    a, b = 3.0, 2.0
    engine_holder = {}
    captured = []

    class ProbeCamera(CleanCamera):
        def capture_jpeg(self):
            record = engine_holder["engine"].last_record
            captured.append(
                (
                    record.requested_a_ml,
                    record.requested_b_ml,
                    record.executed_a_ml,
                    record.executed_b_ml,
                )
            )
            return super().capture_jpeg()

    engine, driver, _, _ = build_engine(config, ProbeCamera())
    engine_holder["engine"] = engine

    record = await engine.run_cycle(a, b, TARGET_LAB, LAM)

    assert record.aborted is None
    assert captured  # capture_jpeg ran at least once (MEASURE, then BLANK)
    first_capture = captured[0]  # the MEASURE-phase capture -- the earliest observation
    assert first_capture == (a, b, a, b)  # executed == requested; no disruption exists yet


async def test_preflight_low_session_waste_headroom_trips_paused_dry():
    # Waste capacity staged low so the WASTE branch trips while the reservoir has ample
    # headroom. planned_waste_ml = dose_a + dose_b + refill_ml + rinse_ml = 1+1+10+10 = 22;
    # waste_capacity_ml=30 leaves headroom 30-22=8 <= low_margin_ml (20) -- the engine
    # must trip to paused *dry* before any liquid moves, same as the reservoir branch.
    low_volumes = VolumeLimits(
        cup_ceiling_ml=20, reservoir_ml=6000, waste_capacity_ml=30, low_margin_ml=20
    )
    config = make_config(volumes=low_volumes)
    engine, driver, accountant, arbiter = build_engine(config, CleanCamera())

    with pytest.raises(CycleError):
        await engine.run_cycle(1.0, 1.0, TARGET_LAB, LAM)

    assert arbiter.mode == "paused"
    assert arbiter.trip_reason is not None
    assert "waste" in arbiter.trip_reason.lower()
    assert relay_on_sequence(driver, config) == []


class ProbingRelayDriver(FakeRelayDriver):
    """Snapshots engine.last_record's dose fields the moment the FIRST relay energizes --
    proving invariant 5 (record published before the next phase begins) holds even before
    DOSE, the very first phase, actuates anything."""

    def __init__(self, engine_holder):
        super().__init__()
        self._engine_holder = engine_holder
        self.first_energize_snapshot = None

    def set(self, gpio, on):
        if on and self.first_energize_snapshot is None:
            record = self._engine_holder["engine"].last_record
            self.first_energize_snapshot = (
                record.requested_a_ml,
                record.requested_b_ml,
                record.executed_a_ml,
                record.executed_b_ml,
            )
        super().set(gpio, on)


async def test_doses_recorded_before_dose_actuates():
    config = make_config()
    a, b = 3.0, 2.0
    engine_holder = {}
    driver = ProbingRelayDriver(engine_holder)
    engine, driver, _, _ = build_engine(config, CleanCamera(), driver=driver)
    engine_holder["engine"] = engine

    record = await engine.run_cycle(a, b, TARGET_LAB, LAM)

    assert record.aborted is None
    assert driver.first_energize_snapshot == (a, b, a, b)  # already populated at first "on"


async def test_session_headroom_exact_low_margin_trips_dry():
    # reservoir_headroom - planned_water_ml == low_margin_ml, exactly: the check is
    # "<=", so an exact match must still trip -- this pins the boundary, not just the
    # already-covered comfortably-below case.
    exact_volumes = VolumeLimits(
        cup_ceiling_ml=20, reservoir_ml=40, waste_capacity_ml=6000, low_margin_ml=20
    )
    config = make_config(volumes=exact_volumes)
    engine, driver, accountant, arbiter = build_engine(config, CleanCamera())

    with pytest.raises(CycleError):
        await engine.run_cycle(1.0, 1.0, TARGET_LAB, LAM)

    assert arbiter.mode == "paused"
    assert arbiter.trip_reason is not None
    assert "reservoir" in arbiter.trip_reason.lower()
    assert relay_on_sequence(driver, config) == []

    # One ml of extra headroom (reservoir_ml=41) clears the boundary -- a clean run.
    above_volumes = VolumeLimits(
        cup_ceiling_ml=20, reservoir_ml=41, waste_capacity_ml=6000, low_margin_ml=20
    )
    config_above = make_config(volumes=above_volumes)
    engine_above, driver_above, _, arbiter_above = build_engine(config_above, CleanCamera())

    record_above = await engine_above.run_cycle(1.0, 1.0, TARGET_LAB, LAM)

    assert record_above.aborted is None
    assert arbiter_above.mode != "paused"


# --- T5b: watchdog, aborts, alarms ---


async def test_watchdog_timeout_trips_paused_and_all_off():
    budget = 0.05
    config = tiny_budgets_config(phase_budgets_s={"measure": budget})
    camera = HangingCaptureCamera(hang_s=0.2)
    engine, driver, accountant, arbiter = build_engine(config, camera)

    record = await engine.run_cycle(1.0, 1.0, TARGET_LAB, LAM)

    assert arbiter.mode == "paused"
    assert driver.relays_on() == set()
    assert arbiter.trip_reason is not None
    assert "measure" in arbiter.trip_reason.lower()
    assert "watchdog" in arbiter.trip_reason.lower()
    assert str(budget) in arbiter.trip_reason
    assert record.aborted == arbiter.trip_reason


async def test_watchdog_abort_returns_partial_record():
    budget = 0.05
    config = tiny_budgets_config(phase_budgets_s={"measure": budget})
    camera = HangingCaptureCamera(hang_s=0.2)
    engine, driver, accountant, arbiter = build_engine(config, camera)

    record = await engine.run_cycle(3.0, 2.0, TARGET_LAB, LAM)

    assert isinstance(record, CycleRecord)
    assert record.aborted is not None
    assert "measure" in record.aborted.lower()
    assert record.requested_a_ml == 3.0
    assert record.requested_b_ml == 2.0
    assert record.target_lab == TARGET_LAB
    # earlier-phase fields survive the abort (DOSE, REFILL, SETTLE all completed first).
    assert "dose" in record.phase_elapsed_s
    assert "refill" in record.phase_elapsed_s
    assert "settle" in record.phase_elapsed_s
    assert record.measured_lab is None  # MEASURE itself never completed


async def test_phase_exception_trips_paused_and_returns_aborted_record():
    config = make_config()
    engine, driver, accountant, arbiter = build_engine(config, ErroringCamera())

    record = await engine.run_cycle(1.0, 1.0, TARGET_LAB, LAM)

    assert ("all_off", None) in [event[:2] for event in driver.history]
    assert driver.relays_on() == set()
    assert arbiter.mode == "paused"
    assert record.aborted is not None
    assert "measure" in record.aborted.lower()
    assert "camera disconnected" in record.aborted.lower()


async def test_operator_pause_mid_cycle_aborts_at_phase_boundary():
    # decision 4: the arbiter is flipped to paused from inside MEASURE's own capture
    # (standing in for an operator action landing mid-cycle) -- MEASURE itself still
    # completes (it doesn't check mode mid-phase), but DRAIN's phase-boundary check must
    # abort before firing another relay.
    config = make_config()
    holder = {}

    class PausingCamera(CleanCamera):
        def capture_jpeg(self):
            holder["arbiter"].trip("operator: paused mid-cycle (test)")
            return super().capture_jpeg()

    engine, driver, accountant, arbiter = build_engine(config, PausingCamera())
    holder["arbiter"] = arbiter

    record = await engine.run_cycle(1.0, 1.0, TARGET_LAB, LAM)

    assert record.aborted is not None
    assert "drain" in record.aborted.lower()
    assert driver.relays_on() == set()
    assert arbiter.mode == "paused"
    assert record.measured_lab is not None  # MEASURE completed before the pause landed
    assert record.blank_lab is None  # BLANK never ran
    assert relay_on_sequence(driver, config) == ["A", "B", "W"]  # DRAIN's D never fired


async def test_budget_trip_mid_cycle_aborts():
    # Preflight passes (200 ml reservoir - 20 ml planned water = 180 > 20 ml margin); an
    # untracked extra 175 ml reservoir draw lands the instant REFILL's relay turns on, so
    # by the time REFILL's own post-dose check runs (10 real + 175 injected = 185 drawn,
    # headroom 15 <= 20) it -- not preflight -- trips the cycle mid-run, wet.
    config = make_config(volumes=_LOW_RESERVOIR_VOLUMES)
    holder = {}
    driver = BudgetInjectingDriver(holder, trigger_gpio=config.channels["W"].gpio, extra_ml=175)
    engine, driver, accountant, arbiter = build_engine(config, CleanCamera(), driver=driver)
    holder["accountant"] = accountant

    record = await engine.run_cycle(1.0, 1.0, TARGET_LAB, LAM)

    assert record.aborted is not None
    assert arbiter.mode == "paused"
    assert driver.relays_on() == set()


async def test_budget_trip_after_dye_a_stops_dye_b():
    # Same injected-draw technique, triggered on dye channel A's relay instead: A's own
    # post-dose check (not preflight, not REFILL) trips the budget mid-DOSE, so B's dose
    # call hits the paused floor (decision 10) before its relay ever fires -- the
    # cross-model dose-while-paused scenario.
    config = make_config(volumes=_LOW_RESERVOIR_VOLUMES)
    holder = {}
    driver = BudgetInjectingDriver(holder, trigger_gpio=config.channels["A"].gpio, extra_ml=185)
    engine, driver, accountant, arbiter = build_engine(config, CleanCamera(), driver=driver)
    holder["accountant"] = accountant

    record = await engine.run_cycle(1.0, 1.0, TARGET_LAB, LAM)

    assert record.aborted is not None
    assert arbiter.mode == "paused"
    assert relay_on_sequence(driver, config) == ["A"]  # B's relay never energized

    # Hardening target (a): _abort must not clobber the budget/reservoir root cause that
    # already tripped the arbiter with the paused-floor's generic "dose rejected"
    # message -- the console alert has to keep naming the reservoir, not the symptom.
    assert arbiter.trip_reason is not None
    assert (
        "reservoir" in arbiter.trip_reason.lower() or "budget" in arbiter.trip_reason.lower()
    ), arbiter.trip_reason


async def test_blank_lock_mismatch_sets_alarm():
    config = make_config()
    camera = PowerCyclableCamera()
    camera.power_cycle()  # simulate a power loss before BLANK's re-read
    engine, driver, accountant, arbiter = build_engine(config, camera)

    record = await engine.run_cycle(1.0, 1.0, TARGET_LAB, LAM)

    assert record.aborted is None  # decision 8: alarms never pause
    assert arbiter.mode != "paused"
    for name in LOCK_CONTROLS:
        assert any(name in alarm for alarm in engine.alarms), (name, engine.alarms)
    assert engine.alarms == record.alarms


async def test_blank_drift_sets_alarm():
    config = make_config()
    camera = DriftingBlankCamera(blank_colors=[(100, 150, 200), (250, 10, 10)])
    engine, driver, accountant, arbiter = build_engine(config, camera)

    first = await engine.run_cycle(1.0, 1.0, TARGET_LAB, LAM)
    assert first.aborted is None
    assert engine.alarms == []  # the first blank cleanly sets the reference

    second = await engine.run_cycle(1.0, 1.0, TARGET_LAB, LAM)

    assert second.aborted is None  # decision 8: alarms never pause
    assert arbiter.mode != "paused"
    assert any("drift" in alarm.lower() for alarm in engine.alarms), engine.alarms
    assert engine.alarms == second.alarms


async def test_alarm_clears_on_next_clean_blank():
    config = make_config()
    reference_color = (100, 150, 200)
    camera = DriftingBlankCamera(
        blank_colors=[reference_color, (250, 10, 10), reference_color]
    )
    engine, driver, accountant, arbiter = build_engine(config, camera)

    first = await engine.run_cycle(1.0, 1.0, TARGET_LAB, LAM)
    assert first.aborted is None
    assert engine.alarms == []

    second = await engine.run_cycle(1.0, 1.0, TARGET_LAB, LAM)
    assert second.aborted is None
    assert engine.alarms != []  # alarming after the drifted blank

    third = await engine.run_cycle(1.0, 1.0, TARGET_LAB, LAM)

    assert third.aborted is None
    assert engine.alarms == []  # level-state: a later clean blank clears it (decision 8)
    assert third.alarms == []


async def test_blank_reference_is_first_blank_since_start():
    # decision 7: the drift reference is the FIRST blank measured since engine start --
    # not the target LAB, not the measurement LAB, not a config value.
    config = make_config()
    camera = DriftingBlankCamera(blank_colors=[(100, 150, 200), (250, 10, 10)])
    engine, driver, accountant, arbiter = build_engine(config, camera)

    first = await engine.run_cycle(1.0, 1.0, TARGET_LAB, LAM)
    second = await engine.run_cycle(1.0, 1.0, TARGET_LAB, LAM)

    assert first.aborted is None
    assert second.aborted is None
    expected_drift = delta_e(second.blank_lab, first.blank_lab)
    assert any(f"{expected_drift:.2f}" in alarm for alarm in engine.alarms), (
        expected_drift,
        engine.alarms,
    )


async def test_blank_verify_hang_trips_watchdog():
    # Hardening target (b): CameraLock.verify() at BLANK calls camera.read_controls()
    # directly on the event loop -- unlike the adjacent capture_jpeg, it is NOT wrapped
    # in asyncio.to_thread. A wedged read_controls freezes the whole event loop, so
    # asyncio.wait_for's timeout callback cannot run until the blocking call itself
    # returns -- the watchdog only "trips" AFTER hang_s has elapsed, nowhere near its
    # budget. This test pins TIMING, not just the end state.
    budget = 0.05
    hang_s = 1.0
    config = tiny_budgets_config(phase_budgets_s={"blank": budget})
    camera = HangingReadControlsCamera(hang_s)
    engine, driver, accountant, arbiter = build_engine(config, camera)

    start = time.monotonic()
    record = await engine.run_cycle(1.0, 1.0, TARGET_LAB, LAM)
    elapsed = time.monotonic() - start

    assert record.aborted is not None
    assert "blank" in record.aborted.lower()
    assert arbiter.mode == "paused"
    assert driver.relays_on() == set()
    assert elapsed < budget + 0.5, (
        f"BLANK took {elapsed:.2f}s to abort against a {budget}s budget -- "
        "camera_lock.verify() is blocking the event loop instead of running in a thread"
    )


class SlowRelayWriteDriver(FakeRelayDriver):
    """set() on the chosen gpio commits the pin state immediately (a real relay write
    lands before the bus call returns) but then blocks synchronously for hang_s the first
    time it turns that relay on -- simulates a real-world relay write slower than the
    calibration model assumes (e.g. a laggy relay board), the kind of unmodeled overrun a
    phase-budget watchdog exists to catch. RigConfig.__post_init__ guarantees the DOSE
    phase budget always covers every channel's *calibrated* worst-case pump time, so a
    validated dose can never overrun its budget from the seconds_for_dose model alone --
    only an overrun the model can't see (like this) can trigger the watchdog mid-DOSE."""

    def __init__(self, trigger_gpio, hang_s):
        super().__init__()
        self._trigger_gpio = trigger_gpio
        self._hang_s = hang_s
        self._hung = False

    def set(self, gpio, on):
        super().set(gpio, on)
        if on and gpio == self._trigger_gpio and not self._hung:
            self._hung = True
            time.sleep(self._hang_s)


async def test_watchdog_timeout_during_dose_leaves_relay_off():
    # Engine-level composed path for the mechanism tests/test_pumps.py pins one layer
    # down (test_dose_relay_off_even_if_cancelled): a watchdog timeout firing mid-DOSE,
    # with A's relay actually energized, must still land the driver off, pause the
    # arbiter, and name both the watchdog and the dose phase on the record.
    budget = 0.05
    hang_s = 0.3
    config = tiny_budgets_config(phase_budgets_s={"dose": budget})
    driver = SlowRelayWriteDriver(trigger_gpio=config.channels["A"].gpio, hang_s=hang_s)
    engine, driver, accountant, arbiter = build_engine(config, CleanCamera(), driver=driver)

    record = await engine.run_cycle(1.0, 0.0, TARGET_LAB, LAM)

    assert driver.relays_on() == set()
    assert arbiter.mode == "paused"
    assert record.aborted is not None
    assert "watchdog" in record.aborted.lower()
    assert "dose" in record.aborted.lower()


_HANG_SCENARIOS = ("measure_capture", "blank_capture", "blank_verify")


async def _run_hang_scenario(scenario: str) -> None:
    budget = 0.03
    hang_s = 0.2
    config = tiny_budgets_config(phase_budgets_s={"measure": budget, "blank": budget})
    camera = HangScenarioCamera(scenario, hang_s)
    engine, driver, accountant, arbiter = build_engine(config, camera)

    record = await engine.run_cycle(1.0, 1.0, TARGET_LAB, LAM)

    assert driver.relays_on() == set()
    assert arbiter.mode == "paused"
    assert record.aborted is not None


@given(scenario=st.sampled_from(_HANG_SCENARIOS))
@settings(max_examples=len(_HANG_SCENARIOS), deadline=None)
def test_relays_off_and_paused_after_any_phase_hang(scenario):
    # Property: regardless of WHICH point hangs (a capture_jpeg the wrapper already
    # protects, or the unwrapped camera_lock.verify() read -- hardening target (b)), the
    # engine must end up with every relay off and the arbiter paused.
    asyncio.run(_run_hang_scenario(scenario))


async def test_cancelled_cycle_leaves_engine_clean_for_the_next_cycle():
    # asyncio.CancelledError derives from BaseException (3.8+), so _run_phase's
    # "except asyncio.TimeoutError" / "except Exception" clauses never catch it. A
    # cancelled cycle must still route through _abort (relays off, arbiter tripped, phase
    # reset to IDLE) before the cancellation re-raises -- otherwise _phase sticks non-IDLE
    # and _preflight rejects every later cycle as "already running" until a restart, while
    # /status keeps reporting a healthy, non-paused rig.
    settle_s = 0.3
    config = make_config()
    config = replace(config, cycle=replace(config.cycle, settle_s=settle_s))
    sink = RecordingSink()
    engine, driver, accountant, arbiter = build_engine(config, CleanCamera(), sink=sink)

    task = asyncio.create_task(engine.run_cycle(1.0, 1.0, TARGET_LAB, LAM))
    # DOSE + REFILL (a few ms total, per make_config's rates) have long since finished;
    # SETTLE's real asyncio.sleep(0.3 s) is in progress -- comfortably inside that window.
    await asyncio.sleep(0.1)
    assert engine.phase == Phase.SETTLE

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert driver.relays_on() == set()
    assert arbiter.mode == "paused"
    assert engine.phase == Phase.IDLE

    # A cancellation abort must finalize the durable record via cycle_finished.
    finished = [call for call in sink.calls if call[0] == "cycle_finished"]
    assert len(finished) == 1
    assert finished[0][1] == sink._cycle_id
    assert finished[0][2].aborted is not None
    assert "cancel" in finished[0][2].aborted.lower()

    # The cup carries DOSE+REFILL's liquid (the cycle never reached DRAIN); drain it the
    # way an operator would before the next cycle, then prove the engine itself is not
    # wedged -- a full second cycle must run to completion, not be rejected as busy.
    if accountant.cup_ml:
        accountant.begin_drain(accountant.cup_ml)
        accountant.complete_drain()
    arbiter.set_mode("manual", "console")

    second = await engine.run_cycle(1.0, 1.0, TARGET_LAB, LAM)
    assert second.aborted is None


# --- T4: durable cycle recording ---


async def test_cycle_started_persists_doses_before_any_relay():
    # invariant 5 (durable): the cycle row is created with requested+executed doses
    # before the first relay energizes. A shared event log proves ordering.
    config = make_config()
    a, b = 3.0, 2.0
    event_log = []

    class OrderingSink(RecordingSink):
        def cycle_started(self, record):
            event_log.append(
                (
                    "sink:cycle_started",
                    record.requested_a_ml,
                    record.requested_b_ml,
                    record.executed_a_ml,
                    record.executed_b_ml,
                )
            )
            return super().cycle_started(record)

    class OrderingDriver(FakeRelayDriver):
        def set(self, gpio, on):
            if on:
                event_log.append(("relay:on", gpio))
            super().set(gpio, on)

    sink = OrderingSink()
    driver = OrderingDriver()
    engine, driver, _, _ = build_engine(config, CleanCamera(), driver=driver, sink=sink)

    record = await engine.run_cycle(a, b, TARGET_LAB, LAM)

    assert record.aborted is None
    sink_events = [event for event in event_log if event[0] == "sink:cycle_started"]
    relay_events = [event for event in event_log if event[0] == "relay:on"]
    assert sink_events
    assert relay_events
    first_sink = sink_events[0]
    first_relay = relay_events[0]
    assert event_log.index(first_sink) < event_log.index(first_relay)
    assert first_sink[1:] == (a, b, a, b)


async def test_phase_committed_after_each_phase_in_order():
    config = make_config()
    sink = RecordingSink()
    engine, driver, _, _ = build_engine(config, CleanCamera(), sink=sink)

    record = await engine.run_cycle(3.0, 2.0, TARGET_LAB, LAM)

    assert record.aborted is None
    expected_phases = [
        "dose",
        "refill",
        "settle",
        "measure",
        "drain",
        "rinse",
        "blank",
        "final_drain",
    ]
    committed = [call for call in sink.calls if call[0] == "phase_committed"]
    assert len(committed) == 8
    assert [call[2] for call in committed] == expected_phases
    assert all(call[1] == sink._cycle_id for call in committed)


async def test_cycle_finished_on_clean_completion():
    config = make_config()
    sink = RecordingSink()
    engine, driver, _, _ = build_engine(config, CleanCamera(), sink=sink)

    record = await engine.run_cycle(3.0, 2.0, TARGET_LAB, LAM)

    assert record.aborted is None
    finished = [call for call in sink.calls if call[0] == "cycle_finished"]
    assert len(finished) == 1
    assert finished[0][1] == sink._cycle_id
    assert finished[0][2].aborted is None


async def test_cycle_finished_on_abort_with_reason():
    budget = 0.05
    config = tiny_budgets_config(phase_budgets_s={"measure": budget})
    sink = RecordingSink()
    camera = HangingCaptureCamera(hang_s=0.2)
    engine, driver, _, _ = build_engine(config, camera, sink=sink)

    record = await engine.run_cycle(3.0, 2.0, TARGET_LAB, LAM)

    assert record.aborted is not None
    finished = [call for call in sink.calls if call[0] == "cycle_finished"]
    assert len(finished) == 1
    assert finished[0][1] == sink._cycle_id
    assert finished[0][2].aborted == record.aborted


async def test_sink_failure_at_start_fires_no_relay():
    config = make_config()
    sink = RaisingStartSink()
    engine, driver, _, arbiter = build_engine(config, CleanCamera(), sink=sink)

    with pytest.raises(CycleError):
        await engine.run_cycle(3.0, 2.0, TARGET_LAB, LAM)

    assert relay_on_sequence(driver, config) == []
    assert arbiter.mode != "paused"
    assert not [call for call in sink.calls if call[0] == "cycle_finished"]


async def test_sink_failure_mid_cycle_trips_paused_all_off():
    config = make_config()
    sink = RaisingPhaseSink("dose")
    engine, driver, _, arbiter = build_engine(config, CleanCamera(), sink=sink)

    record = await engine.run_cycle(3.0, 2.0, TARGET_LAB, LAM)

    assert driver.relays_on() == set()
    assert arbiter.mode == "paused"
    assert record.aborted is not None
    assert "sink" in record.aborted.lower()
    assert "down mid-cycle" in record.aborted.lower()
    finished = [call for call in sink.calls if call[0] == "cycle_finished"]
    assert len(finished) == 1
    assert finished[0][2].aborted == record.aborted


async def test_preflight_rejects_unknown_cup():
    config = make_config()
    sink = RecordingSink()
    engine, driver, accountant, _ = build_engine(config, CleanCamera(), sink=sink)
    accountant._cup_ml = None

    with pytest.raises(CycleError) as excinfo:
        await engine.run_cycle(3.0, 2.0, TARGET_LAB, LAM)

    assert "unknown" in str(excinfo.value).lower()
    assert "drain" in str(excinfo.value).lower()
    assert "pump" in str(excinfo.value).lower()
    assert relay_on_sequence(driver, config) == []
    assert sink.calls == []


async def test_preflight_unknown_message_distinct_from_nonempty():
    config = make_config()

    sink_unknown = RecordingSink()
    engine_unknown, driver_unknown, accountant_unknown, _ = build_engine(
        config, CleanCamera(), sink=sink_unknown
    )
    accountant_unknown._cup_ml = None
    with pytest.raises(CycleError) as excinfo_unknown:
        await engine_unknown.run_cycle(3.0, 2.0, TARGET_LAB, LAM)

    sink_nonempty = RecordingSink()
    engine_nonempty, driver_nonempty, accountant_nonempty, _ = build_engine(
        config, CleanCamera(), sink=sink_nonempty
    )
    accountant_nonempty.fill(5.0)
    with pytest.raises(CycleError) as excinfo_nonempty:
        await engine_nonempty.run_cycle(3.0, 2.0, TARGET_LAB, LAM)

    unknown_msg = str(excinfo_unknown.value)
    nonempty_msg = str(excinfo_nonempty.value)
    assert unknown_msg != nonempty_msg
    assert "unknown" in unknown_msg.lower()
    assert "drain" in unknown_msg.lower()
    assert "pump" in unknown_msg.lower()
    assert "not empty" in nonempty_msg.lower()


async def test_preflight_rejects_unknown_water_counter():
    config = make_config()
    sink = RecordingSink()
    engine, driver, accountant, _ = build_engine(config, CleanCamera(), sink=sink)
    accountant._water_drawn_ml = None

    with pytest.raises(CycleError) as excinfo:
        await engine.run_cycle(3.0, 2.0, TARGET_LAB, LAM)

    assert "staging" in str(excinfo.value).lower()
    assert "counter" in str(excinfo.value).lower()
    assert relay_on_sequence(driver, config) == []
    assert sink.calls == []


async def test_preflight_rejects_unknown_waste_counter():
    config = make_config()
    sink = RecordingSink()
    engine, driver, accountant, _ = build_engine(config, CleanCamera(), sink=sink)
    accountant._waste_ml = None

    with pytest.raises(CycleError) as excinfo:
        await engine.run_cycle(3.0, 2.0, TARGET_LAB, LAM)

    assert "staging" in str(excinfo.value).lower()
    assert "counter" in str(excinfo.value).lower()
    assert relay_on_sequence(driver, config) == []
    assert sink.calls == []


def _seed_store_with_throwaway(store):
    """Consume one cycle id in a throwaway run so the real cycle's n differs from its id."""
    dummy = CycleRecord(
        requested_a_ml=0.0,
        requested_b_ml=0.0,
        executed_a_ml=0.0,
        executed_b_ml=0.0,
        lam=LAM,
        target_lab=TARGET_LAB,
    )
    throwaway_run_id = store.start_run("#000000", TARGET_LAB, LAM, 1, "throwaway")
    store.start_cycle(throwaway_run_id, dummy, VolumeState(0.0, 0.0, 0.0))
    store.end_run(throwaway_run_id, "throwaway")


async def test_durable_record_readable_mid_cycle(tmp_path):
    store = Store(tmp_path / "rig.db", tmp_path / "frames", tmp_path / "backups")
    store.set_volumes(VolumeState(0.0, 0.0, 0.0))
    _seed_store_with_throwaway(store)
    run_id = store.start_run("#3266cc", TARGET_LAB, LAM, 10, "test-model")

    config = make_config()
    a, b = 3.0, 2.0
    accountant_holder = {}
    captured_rows = []

    class ProbeStoreCamera(CleanCamera):
        def capture_jpeg(self):
            row = store._conn.execute(
                "SELECT requested_a_ml, requested_b_ml, executed_a_ml, executed_b_ml "
                "FROM cycles WHERE run_id = ? ORDER BY id DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            captured_rows.append(row)
            return super().capture_jpeg()

    sink = store_sink(store, run_id, accountant_holder)
    engine, driver, accountant, _ = build_engine(config, ProbeStoreCamera(), sink=sink)
    accountant_holder["accountant"] = accountant

    record = await engine.run_cycle(a, b, TARGET_LAB, LAM)

    assert record.aborted is None
    assert captured_rows
    first_row = captured_rows[0]
    assert tuple(first_row) == (a, b, a, b)


async def test_aborted_cycle_row_carries_partial_data(tmp_path):
    store = Store(tmp_path / "rig.db", tmp_path / "frames", tmp_path / "backups")
    store.set_volumes(VolumeState(0.0, 0.0, 0.0))
    _seed_store_with_throwaway(store)
    run_id = store.start_run("#3266cc", TARGET_LAB, LAM, 10, "test-model")

    budget = 0.05
    config = tiny_budgets_config(phase_budgets_s={"measure": budget})
    camera = HangingCaptureCamera(hang_s=0.2)
    accountant_holder = {}

    sink = store_sink(store, run_id, accountant_holder)
    engine, driver, accountant, _ = build_engine(config, camera, sink=sink)
    accountant_holder["accountant"] = accountant

    a, b = 3.0, 2.0
    record = await engine.run_cycle(a, b, TARGET_LAB, LAM)

    assert record.aborted is not None

    row = store._conn.execute(
        "SELECT requested_a_ml, requested_b_ml, executed_a_ml, executed_b_ml, "
        "phase_started, phase_elapsed_s, aborted "
        "FROM cycles WHERE run_id = ? ORDER BY id DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    assert row is not None
    assert (
        row["requested_a_ml"],
        row["requested_b_ml"],
        row["executed_a_ml"],
        row["executed_b_ml"],
    ) == (a, b, a, b)

    phase_elapsed_s = json.loads(row["phase_elapsed_s"])
    assert "dose" in phase_elapsed_s
    assert "refill" in phase_elapsed_s
    assert "settle" in phase_elapsed_s
    assert "drain" not in phase_elapsed_s
    assert "rinse" not in phase_elapsed_s
    assert "blank" not in phase_elapsed_s
    assert "final_drain" not in phase_elapsed_s
    assert row["aborted"] is not None


# --- T1: dye-dose skew application ---


async def test_skewed_dose_records_executed_times_factor():
    config = make_config()
    disruptions = Disruptions(config)
    disruptions.set_skew("A", 0.5)
    engine, driver, _, _ = build_engine(config, CleanCamera(), disruptions=disruptions)

    record = await engine.run_cycle(4.0, 3.0, TARGET_LAB, LAM)

    assert record.aborted is None
    assert record.requested_a_ml == 4.0
    assert record.requested_b_ml == 3.0
    assert record.executed_a_ml == 2.0
    assert record.executed_b_ml == 3.0


async def test_skewed_dose_actuates_executed_volume():
    config = make_config()
    disruptions = Disruptions(config)
    disruptions.set_skew("A", 0.5)
    engine, driver, _, _ = build_engine(config, CleanCamera(), disruptions=disruptions)

    dose_calls = []
    original_dose = engine._pumps.dose

    async def recording_dose(channel_name, ml):
        dose_calls.append((channel_name, ml))
        return await original_dose(channel_name, ml)

    engine._pumps.dose = recording_dose

    record = await engine.run_cycle(4.0, 3.0, TARGET_LAB, LAM)

    assert record.aborted is None
    assert record.executed_a_ml == 2.0
    assert record.executed_b_ml == 3.0
    assert ("A", 2.0) in dose_calls
    assert ("A", 4.0) not in dose_calls
    assert ("B", 3.0) in dose_calls


async def test_unskewed_channel_unaffected():
    config = make_config()
    disruptions = Disruptions(config)
    disruptions.set_skew("A", 0.5)
    engine, driver, _, _ = build_engine(config, CleanCamera(), disruptions=disruptions)

    record = await engine.run_cycle(4.0, 3.0, TARGET_LAB, LAM)

    assert record.aborted is None
    assert record.executed_a_ml == 2.0
    assert record.executed_b_ml == 3.0
    assert record.requested_b_ml == 3.0


async def test_score_uses_requested_dye_ml():
    config = make_config()
    disruptions = Disruptions(config)
    disruptions.set_skew("A", 0.5)
    engine, driver, _, _ = build_engine(
        config, CleanCamera(color=(200, 60, 30)), disruptions=disruptions
    )

    record = await engine.run_cycle(4.0, 3.0, TARGET_LAB, LAM)

    assert record.aborted is None
    expected_score = score(record.delta_e, 4.0 + 3.0, LAM)
    assert record.score == pytest.approx(expected_score)


async def test_sink_row_carries_divergent_doses_before_relay():
    config = make_config()
    a, b = 4.0, 3.0
    disruptions = Disruptions(config)
    disruptions.set_skew("A", 0.5)
    event_log = []

    class SkewedOrderingSink(RecordingSink):
        def cycle_started(self, record):
            event_log.append(
                (
                    "sink:cycle_started",
                    record.requested_a_ml,
                    record.requested_b_ml,
                    record.executed_a_ml,
                    record.executed_b_ml,
                )
            )
            return super().cycle_started(record)

    class SkewedOrderingDriver(FakeRelayDriver):
        def set(self, gpio, on):
            if on:
                event_log.append(("relay:on", gpio))
            super().set(gpio, on)

    sink = SkewedOrderingSink()
    driver = SkewedOrderingDriver()
    engine, driver, _, _ = build_engine(
        config, CleanCamera(), driver=driver, sink=sink, disruptions=disruptions
    )

    record = await engine.run_cycle(a, b, TARGET_LAB, LAM)

    assert record.aborted is None
    sink_events = [event for event in event_log if event[0] == "sink:cycle_started"]
    relay_events = [event for event in event_log if event[0] == "relay:on"]
    assert sink_events
    assert relay_events
    first_sink = sink_events[0]
    first_relay = relay_events[0]
    assert event_log.index(first_sink) < event_log.index(first_relay)
    assert first_sink[1:] == (a, b, a * 0.5, b)


async def test_skewed_cycle_persists_divergent_row(tmp_path):
    store = Store(tmp_path / "rig.db", tmp_path / "frames", tmp_path / "backups")
    store.set_volumes(VolumeState(0.0, 0.0, 0.0))
    run_id = store.start_run("#3266cc", TARGET_LAB, LAM, 10, "test-model")

    config = make_config()
    a, b = 4.0, 3.0
    factor = 0.5
    disruptions = Disruptions(config)
    disruptions.set_skew("A", factor)

    accountant_holder = {}
    sink = store_sink(store, run_id, accountant_holder)
    engine, driver, accountant, _ = build_engine(
        config, CleanCamera(), sink=sink, disruptions=disruptions
    )
    accountant_holder["accountant"] = accountant

    record = await engine.run_cycle(a, b, TARGET_LAB, LAM)

    assert record.aborted is None

    row = store._conn.execute(
        "SELECT requested_a_ml, requested_b_ml, executed_a_ml, executed_b_ml "
        "FROM cycles WHERE run_id = ? ORDER BY id DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    assert row is not None
    assert (
        row["requested_a_ml"],
        row["requested_b_ml"],
        row["executed_a_ml"],
        row["executed_b_ml"],
    ) == (a, b, a * factor, b)


# --- T3: per-cycle patch snapshot ---


def _two_field_jpeg(left_color, right_color, size=(64, 64), quality=95):
    """Left half left_color, right half right_color."""
    image = Image.new("RGB", size, left_color)
    width, height = size
    ImageDraw.Draw(image).rectangle((width // 2, 0, width, height), fill=right_color)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, subsampling=0)
    return buf.getvalue()


class TwoFieldCamera(CleanCamera):
    """Renders a frame with distinct left/right color fields."""

    def __init__(self, left_color=(250, 10, 10), right_color=(10, 10, 250), size=64, **kwargs):
        super().__init__(**kwargs)
        self._frame = _two_field_jpeg(left_color, right_color, size=(size, size))


async def test_cycle_fetches_patch_exactly_once_per_cycle():
    config = make_config()
    calls = []

    def counting_patch_source():
        calls.append("patch")
        return PatchSpec(0.5, 0.5, config.camera.patch_px)

    engine, driver, _, _ = build_engine(
        config, CleanCamera(), patch_source=counting_patch_source
    )

    record = await engine.run_cycle(1.0, 1.0, TARGET_LAB, LAM)

    assert record.aborted is None
    assert len(calls) == 1


async def test_cycle_mid_cycle_nudge_does_not_split_measure_and_blank():
    config = make_config()
    left_patch = PatchSpec(0.25, 0.5, 32)
    right_patch = PatchSpec(0.75, 0.5, 32)
    call_count = [0]

    def sequencing_patch_source():
        call_count[0] += 1
        if call_count[0] == 1:
            return left_patch
        return right_patch

    engine, driver, _, _ = build_engine(
        config,
        TwoFieldCamera(left_color=(250, 10, 10), right_color=(10, 10, 250), size=64),
        patch_source=sequencing_patch_source,
    )

    record = await engine.run_cycle(1.0, 1.0, TARGET_LAB, LAM)

    assert record.aborted is None
    assert call_count[0] == 1  # only one fetch happened
    # Both readings must match the left (red) region sampled by left_patch.
    for lab in (record.measured_lab, record.blank_lab):
        assert lab is not None
        assert lab[2] > lab[0]  # blue dominates if split happened; red should dominate


async def test_cycle_measure_and_blank_use_snapshot_values():
    config = make_config()
    off_center = PatchSpec(0.25, 0.5, 32)

    engine, driver, _, _ = build_engine(
        config,
        TwoFieldCamera(left_color=(250, 10, 10), right_color=(10, 10, 250), size=64),
        patch_source=lambda: off_center,
    )

    record = await engine.run_cycle(1.0, 1.0, TARGET_LAB, LAM)

    assert record.aborted is None
    # Off-center snapshot on the left (red) half; center would be a mix.
    for lab in (record.measured_lab, record.blank_lab):
        assert lab is not None
        assert lab[2] > lab[0]
