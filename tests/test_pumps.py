# ABOUTME: Tests for PumpController: the one capped, volume-accounted, serialized dosing
# ABOUTME: path shared by /pump and the cycle engine, including the paused-floor safety gate.

import asyncio
import dataclasses
import time

import pytest

from rig_service.config import CameraConfig, Channel, CycleConfig, RigConfig, VolumeLimits
from rig_service.drivers.fake import FakeRelayDriver
from rig_service.mode import ModeArbiter, ModeError
from rig_service.pumps import PumpController, PumpError
from rig_service.volume import VolumeAccountant, VolumeError, VolumeState

pytestmark = pytest.mark.anyio


def make_config(volumes=None):
    # Mirrors tests/test_app.py's make_config (same channel/cap/volume/camera/cycle
    # values) so pump behavior under the controller matches the app's existing tests.
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
            "A": Channel(gpio=4, role="dye", rate_ml_per_s=100.0, lag_s=0.01, max_ml_per_dose=10),
            "B": Channel(gpio=5, role="dye", rate_ml_per_s=100.0, lag_s=0.01, max_ml_per_dose=10),
            "D": Channel(
                gpio=6, role="drain", rate_ml_per_s=100.0, lag_s=0.01, max_ml_per_dose=150
            ),
            "W": Channel(
                gpio=7, role="water", rate_ml_per_s=100.0, lag_s=0.01, max_ml_per_dose=150
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


def make_shared_log_accountant(config, driver):
    """A durably-backed accountant whose sink writes into the SAME event log as the relay
    driver's own history (`driver.history`), tagged ("sink", state, monotonic-time) --
    so ordering between a persisted write and a relay on/off event is a plain list-index
    comparison, not a timestamp race."""
    return VolumeAccountant(
        cup_ceiling_ml=config.volumes.cup_ceiling_ml,
        reservoir_ml=config.volumes.reservoir_ml,
        waste_capacity_ml=config.volumes.waste_capacity_ml,
        low_margin_ml=config.volumes.low_margin_ml,
        state=VolumeState(0.0, 0.0, 0.0),
        on_change=lambda state: driver.history.append(("sink", state, time.monotonic())),
    )


def make_unknown_cup_accountant(config):
    """An accountant starting with an unknown cup (VolumeState(None, ...)) -- the
    post-restart-with-unresolved-cup world (decision 1), with a trivial no-op sink."""
    return VolumeAccountant(
        cup_ceiling_ml=config.volumes.cup_ceiling_ml,
        reservoir_ml=config.volumes.reservoir_ml,
        waste_capacity_ml=config.volumes.waste_capacity_ml,
        low_margin_ml=config.volumes.low_margin_ml,
        state=VolumeState(None, 0.0, 0.0),
        on_change=lambda _: None,
    )


@pytest.fixture()
def anyio_backend():
    return "asyncio"


@pytest.fixture()
def config():
    return make_config()


@pytest.fixture()
def driver():
    return FakeRelayDriver()


@pytest.fixture()
def accountant(config):
    # These tests are about pump behavior, not durability, so a trivial no-op sink is right.
    return VolumeAccountant(
        cup_ceiling_ml=config.volumes.cup_ceiling_ml,
        reservoir_ml=config.volumes.reservoir_ml,
        waste_capacity_ml=config.volumes.waste_capacity_ml,
        low_margin_ml=config.volumes.low_margin_ml,
        state=VolumeState(0.0, 0.0, 0.0),
        on_change=lambda _: None,
    )


@pytest.fixture()
def arbiter():
    return ModeArbiter()


@pytest.fixture()
def controller(config, driver, accountant, arbiter):
    return PumpController(config, driver, accountant, arbiter)


def unpause(arbiter, mode="manual"):
    arbiter.set_mode(mode, "console")


# --- validate() ---


def test_validate_unknown_channel_raises(controller):
    with pytest.raises(PumpError):
        controller.validate("X", 1)


@pytest.mark.parametrize("ml", [0, -1])
def test_validate_nonpositive_ml_raises(controller, ml):
    with pytest.raises(PumpError):
        controller.validate("A", ml)


def test_validate_over_channel_cap_raises(controller):
    # A's max_ml_per_dose is 10.
    with pytest.raises(PumpError):
        controller.validate("A", 11)


def test_validate_over_on_time_cap_raises(controller):
    # SLOW: 1 ml at 0.1 ml/s = 10 s, over max_relay_on_s (5.0) but under SLOW's
    # own max_ml_per_dose (10) -- isolates the on-time cap from the channel cap.
    with pytest.raises(PumpError):
        controller.validate("SLOW", 1)


def test_validate_at_on_time_cap_accepts(config, driver, accountant, arbiter):
    # The check is "seconds > max_relay_on_s" (strictly greater), so a dose landing
    # exactly ON the cap must be accepted, not rejected. rate=2.0 ml/s, lag=0.0 s and
    # ml=10 make seconds_for_dose == 10.0 / 2.0 == 5.0 -- exactly max_relay_on_s (5.0).
    # All three values (10.0, 2.0, 5.0) are exact powers of two over ten, so this
    # equality holds in real IEEE-754 arithmetic, not by approximation.
    cap_config = dataclasses.replace(
        config,
        channels={
            **config.channels,
            "CAP": dataclasses.replace(
                config.channels["A"], gpio=9, rate_ml_per_s=2.0, lag_s=0.0, max_ml_per_dose=20
            ),
        },
    )
    controller = PumpController(cap_config, driver, accountant, arbiter)

    seconds = controller.validate("CAP", 10)

    assert seconds == 5.0
    assert seconds == cap_config.max_relay_on_s


def test_validate_touches_nothing(controller, driver, accountant):
    cup_before = accountant.cup_ml
    water_before = accountant.water_drawn_ml
    waste_before = accountant.waste_ml
    history_before = list(driver.history)
    relays_before = driver.relays_on()

    seconds = controller.validate("A", 5)
    assert seconds == pytest.approx(0.01 + 5 / 100)

    with pytest.raises(PumpError):
        controller.validate("A", 11)

    assert accountant.cup_ml == cup_before
    assert accountant.water_drawn_ml == water_before
    assert accountant.waste_ml == waste_before
    assert driver.history == history_before
    assert driver.relays_on() == relays_before


# --- dose(): paused floor ---


async def test_paused_landing_while_dose_parked_at_lock_blocks_relay(controller, driver, arbiter, config):
    # The paused floor is checked once, at the TOP of dose(), before the coil-power lock is
    # acquired. If the rig goes paused while a second dose is parked waiting for that lock
    # (mirrors POST /mode {paused}: arbiter.set_mode then driver.all_off(), app.py:114-122),
    # the queued dose resumes on a stale check and energizes its relay while mode == "paused"
    # -- invariant 4 ("all relay GPIOs are off ... at paused") broken by a race, not a bug in
    # the check itself.
    unpause(arbiter)
    w_gpio = config.channels["W"].gpio
    a_gpio = config.channels["A"].gpio

    task1 = asyncio.create_task(controller.dose("W", 10))  # ~0.11 s on-time; holds the lock
    await asyncio.sleep(0)  # let task1 run to its only suspension point: asyncio.sleep(seconds)
    assert driver.relays_on() == {w_gpio}  # sanity: task1 truly holds the lock, relay live

    task2 = asyncio.create_task(controller.dose("A", 5))  # passes its own paused check, then
    await asyncio.sleep(0)  # parks on the held lock -- let it reach that suspension point

    history_at_pause = len(driver.history)
    arbiter.set_mode("paused", "console")  # mirrors POST /mode {paused}
    driver.all_off()

    await task1  # releases the lock; task2's queued acquire() can now proceed

    with pytest.raises(ModeError):
        await task2

    # No relay for A must have energized after the pause landed, regardless of how the
    # ModeError assertion above resolves.
    on_events_after_pause = [
        e for e in driver.history[history_at_pause:] if e[0] == "on" and e[1] == a_gpio
    ]
    assert on_events_after_pause == []
    assert driver.relays_on() == set()


async def test_dose_in_paused_raises_before_any_accounting_or_relay(controller, driver, accountant, arbiter):
    assert arbiter.mode == "paused"
    with pytest.raises(ModeError):
        await controller.dose("A", 5)
    assert accountant.cup_ml == 0
    assert accountant.water_drawn_ml == 0
    assert accountant.waste_ml == 0
    assert driver.history == []
    assert driver.relays_on() == set()


# --- dose(): relay actuation ---


async def test_dose_fires_relay_on_then_off(controller, driver, arbiter):
    unpause(arbiter)
    seconds = await controller.dose("A", 2)
    assert seconds == pytest.approx(0.01 + 2 / 100)
    assert [e[:2] for e in driver.history] == [("on", 4), ("off", 4)]
    assert driver.relays_on() == set()


async def test_dose_relay_off_even_if_cancelled(controller, driver, arbiter):
    unpause(arbiter)
    task = asyncio.create_task(controller.dose("W", 100))  # ~1.01 s on-time
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert driver.relays_on() == set()


# --- dose(): volume accounting ---


async def test_dye_dose_fills_cup_only(controller, accountant, arbiter):
    unpause(arbiter)
    await controller.dose("A", 5)
    assert accountant.cup_ml == 5
    assert accountant.water_drawn_ml == 0


async def test_water_dose_draws_reservoir(controller, accountant, arbiter):
    unpause(arbiter)
    await controller.dose("W", 50)
    assert accountant.cup_ml == 50
    assert accountant.water_drawn_ml == 50


async def test_drain_dose_moves_cup_to_waste(config, driver, arbiter):
    # The end-state pin holds across the two-phase begin_drain/complete_drain split
    # (decision 2).
    accountant = make_shared_log_accountant(config, driver)
    controller = PumpController(config, driver, accountant, arbiter)
    unpause(arbiter)
    await controller.dose("W", 100)
    await controller.dose("D", 150)  # pump runs for 150 ml but only 100 ml is in the cup
    assert accountant.cup_ml == 0
    assert accountant.waste_ml == 100


# --- dose(): durable write ordering across the drain split (decision 2) ---


async def test_water_dose_persists_cup_and_counter_before_relay_on(config, driver, arbiter):
    accountant = make_shared_log_accountant(config, driver)
    controller = PumpController(config, driver, accountant, arbiter)
    unpause(arbiter)

    await controller.dose("W", 50)

    kinds = [e[0] for e in driver.history]
    assert "sink" in kinds
    assert kinds.index("sink") < kinds.index("on")
    sink_state = next(e[1] for e in driver.history if e[0] == "sink")
    assert sink_state.cup_ml == pytest.approx(50)
    assert sink_state.water_drawn_ml == pytest.approx(50)


async def test_drain_persists_waste_before_relay_on(config, driver, arbiter):
    accountant = make_shared_log_accountant(config, driver)
    controller = PumpController(config, driver, accountant, arbiter)
    unpause(arbiter)
    await controller.dose("W", 100)
    driver.history.clear()

    await controller.dose("D", 50)

    kinds = [e[0] for e in driver.history]
    on_index = kinds.index("on")
    sink_before_on = [e[1] for e in driver.history[:on_index] if e[0] == "sink"]
    assert sink_before_on, "begin_drain must persist the waste charge before the relay fires"
    assert sink_before_on[-1].waste_ml == pytest.approx(50)


async def test_drain_persists_cup_after_relay_off(config, driver, arbiter):
    accountant = make_shared_log_accountant(config, driver)
    controller = PumpController(config, driver, accountant, arbiter)
    unpause(arbiter)
    await controller.dose("W", 100)
    driver.history.clear()

    await controller.dose("D", 50)

    kinds = [e[0] for e in driver.history]
    off_index = kinds.index("off")
    sink_after_off = [e[1] for e in driver.history[off_index + 1 :] if e[0] == "sink"]
    assert sink_after_off, "complete_drain must persist the decremented cup after relay-off"
    assert sink_after_off[-1].cup_ml == pytest.approx(50)


async def test_durable_waste_already_charged_while_drain_relay_is_on(config, driver, arbiter):
    accountant = make_shared_log_accountant(config, driver)
    controller = PumpController(config, driver, accountant, arbiter)
    unpause(arbiter)
    await controller.dose("W", 100)
    driver.history.clear()

    await controller.dose("D", 50)

    kinds = [e[0] for e in driver.history]
    on_index = kinds.index("on")
    # A crash exactly at relay-on recovers whatever was last durably persisted before it.
    persisted_before_on = [e[1] for e in driver.history[:on_index] if e[0] == "sink"]
    crash_recovered_state = persisted_before_on[-1]
    assert crash_recovered_state.waste_ml == pytest.approx(50)
    assert crash_recovered_state.cup_ml == pytest.approx(100)  # cup still recorded high


async def test_cancelled_drain_leaves_cup_high_and_waste_charged(config, driver, arbiter):
    accountant = make_shared_log_accountant(config, driver)
    controller = PumpController(config, driver, accountant, arbiter)
    unpause(arbiter)
    await controller.dose("W", 100)

    task = asyncio.create_task(controller.dose("D", 100))  # ~1.01 s on-time
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # complete_drain is deliberately not in the relay-off finally (decision 2): a cancelled
    # drain must not decrement the cup, even though the relay itself still closes.
    assert accountant.cup_ml == pytest.approx(100)
    assert accountant.waste_ml == pytest.approx(100)  # begin_drain already charged waste
    assert driver.relays_on() == set()


async def test_overfill_rejected_relay_untouched(controller, driver, accountant, arbiter):
    unpause(arbiter)
    await controller.dose("W", 140)
    events_before = len(driver.history)
    with pytest.raises(VolumeError):
        await controller.dose("W", 20)  # 140 + 20 > 150 ceiling
    assert accountant.cup_ml == 140
    assert accountant.water_drawn_ml == 140
    assert [e[:2] for e in driver.history[events_before:] if e[0] == "on"] == []


# --- dose(): unknown cup (decision 1's fail-closed path through the real entry point) ---


async def test_water_dose_on_unknown_cup_rejected_relay_untouched(config, driver, arbiter):
    unpause(arbiter)
    accountant = make_unknown_cup_accountant(config)
    controller = PumpController(config, driver, accountant, arbiter)

    with pytest.raises(VolumeError):
        await controller.dose("W", 50)

    assert driver.history == []
    assert driver.relays_on() == set()


async def test_resolving_drain_dose_clears_unknown_cup_without_tripping(config, driver, arbiter):
    unpause(arbiter)
    accountant = make_unknown_cup_accountant(config)
    controller = PumpController(config, driver, accountant, arbiter)

    await controller.dose("D", 150)  # a full-ceiling drain resolves the unknown cup (decision 1)

    assert accountant.cup_ml == 0.0
    assert arbiter.mode == "manual"


async def test_non_resolving_drain_dose_leaves_unknown_and_trips_paused(config, driver, arbiter):
    unpause(arbiter)
    accountant = make_unknown_cup_accountant(config)
    controller = PumpController(config, driver, accountant, arbiter)

    await controller.dose("D", 100)  # short of the 150 ml ceiling: cup stays unknown

    assert accountant.cup_ml is None
    assert arbiter.mode == "paused"
    assert "unknown" in arbiter.trip_reason.lower()


# --- dose(): budget-low trip ---


async def test_budget_low_trips_paused_and_all_off(driver):
    config = make_config(
        volumes=VolumeLimits(
            cup_ceiling_ml=150, reservoir_ml=200, waste_capacity_ml=6000, low_margin_ml=150
        )
    )
    accountant = VolumeAccountant(
        cup_ceiling_ml=config.volumes.cup_ceiling_ml,
        reservoir_ml=config.volumes.reservoir_ml,
        waste_capacity_ml=config.volumes.waste_capacity_ml,
        low_margin_ml=config.volumes.low_margin_ml,
        state=VolumeState(0.0, 0.0, 0.0),
        on_change=lambda _: None,
    )
    arbiter = ModeArbiter()
    unpause(arbiter)
    controller = PumpController(config, driver, accountant, arbiter)

    seconds = await controller.dose("W", 60)  # reservoir headroom 200-60=140 <= 150 margin

    assert seconds == pytest.approx(0.01 + 60 / 100)
    assert arbiter.mode == "paused"
    assert driver.relays_on() == set()
    assert ("all_off", None) in [e[:2] for e in driver.history]


# --- dose(): concurrency ---


async def test_concurrent_doses_never_overlap(controller, driver, arbiter):
    unpause(arbiter)
    await asyncio.gather(controller.dose("A", 3), controller.dose("B", 3))

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


# --- timed_run(): calibration primitive (T3) ---


async def test_timed_run_estimates_ml_from_rate_and_lag(controller, driver, accountant, arbiter):
    unpause(arbiter)
    # A: rate 100 ml/s, lag 0.01 s. seconds=0.06 -> ml = 100*(0.06-0.01) = 5.0, well
    # under A's 10 ml cap and the 5 s relay cap.
    est_ml = await controller.timed_run("A", 0.06)

    assert est_ml == pytest.approx(5.0)
    assert accountant.cup_ml == pytest.approx(5.0)
    assert [e[:2] for e in driver.history] == [("on", 4), ("off", 4)]
    assert driver.relays_on() == set()


@pytest.mark.parametrize("seconds", [0, -1])
async def test_timed_run_rejects_zero_and_negative_seconds(
    controller, driver, accountant, arbiter, seconds
):
    unpause(arbiter)
    with pytest.raises(PumpError):
        await controller.timed_run("A", seconds)
    assert accountant.cup_ml == 0
    assert driver.history == []


async def test_timed_run_rejects_over_dose_cap(controller, driver, accountant, arbiter):
    unpause(arbiter)
    # A's max_ml_per_dose is 10. seconds=1.01 -> ml = 100*(1.01-0.01) = 100, comfortably
    # over the cap regardless of float noise near the boundary.
    with pytest.raises(PumpError) as exc_info:
        await controller.timed_run("A", 1.01)
    assert "cap" in str(exc_info.value).lower()
    assert accountant.cup_ml == 0
    assert driver.history == []


async def test_timed_run_rejects_over_relay_cap(config, driver, accountant, arbiter):
    # With the shipped/test rates the per-dose cap always fires first (card note), so
    # this guard needs a synthetic channel whose per-dose cap is high (200 ml, still
    # under low_margin_ml's 300 ml floor for a water/drain channel) but whose rate
    # (1 ml/s, no lag) makes the 5 s relay on-time cap bind first.
    cap_config = dataclasses.replace(
        config,
        channels={
            **config.channels,
            "RELAYCAP": dataclasses.replace(
                config.channels["D"],
                gpio=10,
                rate_ml_per_s=1.0,
                lag_s=0.0,
                max_ml_per_dose=200.0,
            ),
        },
    )
    controller = PumpController(cap_config, driver, accountant, arbiter)
    unpause(arbiter)

    # ml = 1*(6-0) = 6, far under the 200 ml cap; seconds_for_dose(6, 1, 0) = 6 > the
    # 5 s relay on-time cap.
    with pytest.raises(PumpError) as exc_info:
        await controller.timed_run("RELAYCAP", 6.0)
    assert "on-time" in str(exc_info.value).lower()
    assert accountant.cup_ml == 0
    assert driver.history == []


async def test_timed_run_rejects_cup_overfill(controller, driver, accountant, arbiter):
    unpause(arbiter)
    await controller.dose("W", 149)  # 1 ml of headroom left under the 150 ml ceiling
    events_before = len(driver.history)

    # ml = 100*(0.06-0.01) = 5, over the 1 ml headroom -- whatever volume.py enforces
    # today (not redefined here).
    with pytest.raises(VolumeError):
        await controller.timed_run("W", 0.06)

    assert accountant.cup_ml == 149
    assert [e[:2] for e in driver.history[events_before:] if e[0] == "on"] == []


async def test_timed_run_charges_session_water_before_actuation(config, driver, arbiter):
    accountant = make_shared_log_accountant(config, driver)
    controller = PumpController(config, driver, accountant, arbiter)
    unpause(arbiter)

    est_ml = await controller.timed_run("W", 0.06)  # ml = 100*(0.06-0.01) = 5

    assert est_ml == pytest.approx(5.0)
    kinds = [e[0] for e in driver.history]
    assert "sink" in kinds
    assert kinds.index("sink") < kinds.index("on")
    sink_state = next(e[1] for e in driver.history if e[0] == "sink")
    assert sink_state.cup_ml == pytest.approx(5.0)
    assert sink_state.water_drawn_ml == pytest.approx(5.0)


async def test_timed_run_drain_charges_waste(config, driver, arbiter):
    accountant = make_shared_log_accountant(config, driver)
    controller = PumpController(config, driver, accountant, arbiter)
    unpause(arbiter)
    await controller.dose("W", 100)
    driver.history.clear()

    # ml = 100*(0.51-0.01) = 50; only 100 ml sits in the cup, so waste is charged 50.
    est_ml = await controller.timed_run("D", 0.51)

    assert est_ml == pytest.approx(50.0)
    assert accountant.cup_ml == pytest.approx(50.0)
    assert accountant.waste_ml == pytest.approx(50.0)
    kinds = [e[0] for e in driver.history]
    on_index = kinds.index("on")
    sink_before_on = [e[1] for e in driver.history[:on_index] if e[0] == "sink"]
    assert sink_before_on, "drain's waste charge must persist before the relay fires"
    assert sink_before_on[-1].waste_ml == pytest.approx(50.0)
