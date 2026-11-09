# ABOUTME: Tests for the sim rig seam (SimCup/SimPumps/SimCamera) behind the real
# ABOUTME: pumps/camera hardware boundary, plus full-cycle and disruption runs through it.

import dataclasses
import json
import math
import threading

import pytest
from fastapi.testclient import TestClient

import rig_service.app as app_module
from rig_service.app import create_app
from rig_service.drivers.fake import FakeRelayDriver
from rig_service.mode import ModeArbiter, ModeError
from rig_service.pumps import PumpError
from rig_service.store import Store
from rig_service.volume import VolumeAccountant, VolumeState
from scientist.loop import Scientist
from scientist.settings import ScientistConfig
from shared.color import delta_e, hex_to_rgb, linear_rgb_to_srgb8, rgb_to_lab
from shared.measure import lab_from_jpeg
from sim.rig import SimCamera, SimCup, SimPumps
from sim.world import DYE_A_ABSORBANCE, DYE_B_ABSORBANCE, SimWorld
from tests.test_app import (
    CONSOLE,
    SCIENTIST,
    FAST_CAMERA_DEFAULTS,
    _patched_sleep_for_phase,
    make_fast_cycle_config,
)

# Group-2 SimPumps tests call the real async PumpController.dose path directly (no HTTP);
# the anyio marker lets those be plain `async def` tests in the same file as the sync ones
# (mirrors tests/test_pumps.py -- the plugin no-ops on the sync tests below).
pytestmark = pytest.mark.anyio

WHITE_LAB = (100.0, 0.0, 0.0)  # sRGB (255,255,255) -> CIE LAB, D65


@pytest.fixture()
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------------------
# Shared scaffolding
# ---------------------------------------------------------------------------


def _sim_config():
    """Fast rates + fast phase budgets (K9), reusing make_fast_cycle_config's proven
    tiny-volume shape. 30-cycle arithmetic: each cycle draws refill_ml + rinse_ml =
    20 ml from the reservoir and charges at most dose_a_ml + dose_b_ml + 20 ml <= 30 ml
    of waste (channel caps are 5 ml each). Over 30 cycles that is <= 600 ml reservoir
    draw and <= 900 ml waste -- both far under the 6000/6000 ml reservoir_ml/
    waste_capacity_ml make_fast_cycle_config() already ships, so the session-budget
    trip (K4, ~28.5 cycles on the shipped 100/100/150 ml config/rig.json) is not a
    live constraint here.
    """
    return make_fast_cycle_config()


def _blip_test_config():
    """_sim_config() with a larger real settle_s (0.2s, still comfortably under the
    1.0s settle phase budget) so _patched_sleep_for_phase's "seconds >= threshold"
    heuristic (threshold = settle_s / 2) can tell the SETTLE sleep apart from the fast
    dose/refill/rinse/drain on-time sleeps it also intercepts (it patches asyncio.sleep
    globally, at the actual asyncio module object, not just within rig_service.pumps).
    _sim_config()'s own settle_s (0.01s) is too close to those ~0.006-0.021s on-times
    to discriminate reliably -- confirmed empirically: it falsely fires mid-REFILL."""
    base = _sim_config()
    return dataclasses.replace(base, cycle=dataclasses.replace(base.cycle, settle_s=0.2))


def _expected_dilution_lab(
    dose_a_ml: float, dose_b_ml: float, water_ml: float
) -> tuple[float, float, float]:
    """The camera's exact predicted measurement for a cup holding dose_a_ml/dose_b_ml
    dye in water_ml water: SimCamera.capture_jpeg()'s own Beer-Lambert reflectance model
    (same DYE_*_ABSORBANCE constants) run through the same 8-bit sRGB quantization it
    applies before JPEG-encoding (with noise=False, as in the blip test below). JPEG at
    quality=95 is lossless on a solid-color patch (confirmed empirically: 0.0 delta_e
    against a real capture_jpeg() round trip), so this reproduces the measurement exactly
    rather than approximately."""
    conc_a = dose_a_ml / water_ml
    conc_b = dose_b_ml / water_ml
    reflectance = tuple(
        math.exp(-(conc_a * abs_a + conc_b * abs_b))
        for abs_a, abs_b in zip(DYE_A_ABSORBANCE, DYE_B_ABSORBANCE)
    )
    return rgb_to_lab(linear_rgb_to_srgb8(reflectance))


def _make_pump_scaffold(config):
    """A real (driver, accountant, arbiter) tuple for constructing SimPumps directly,
    no HTTP -- mirrors tests/test_pumps.py's fixtures."""
    driver = FakeRelayDriver()
    arbiter = ModeArbiter()
    accountant = VolumeAccountant(
        cup_ceiling_ml=config.volumes.cup_ceiling_ml,
        reservoir_ml=config.volumes.reservoir_ml,
        waste_capacity_ml=config.volumes.waste_capacity_ml,
        low_margin_ml=config.volumes.low_margin_ml,
        state=VolumeState(0.0, 0.0, 0.0),
        on_change=lambda _: None,
    )
    return driver, accountant, arbiter


def _install_sim_pumps(monkeypatch, world, cup):
    """Substitute SimPumps for the real PumpController inside create_app(), with zero
    changes to app.py: create_app's body resolves `PumpController(...)` as a global name
    lookup in rig_service.app's module namespace AT CALL TIME, so patching that name here
    is enough. Must be called before create_app(...) runs."""

    def factory(config, driver, accountant, arbiter):
        return SimPumps(config, driver, accountant, arbiter, world=world, cup=cup)

    monkeypatch.setattr(app_module, "PumpController", factory)


def _build_sim_app(
    tmp_path,
    monkeypatch,
    world,
    cup,
    config=None,
    start_run=False,
    target_hex="#3366cc",
    cycle_budget=30,
    label="sim",
    noise=False,
):
    """A real create_app() app wired to SimPumps/SimCamera over the given world/cup."""
    config = config or _sim_config()
    _install_sim_pumps(monkeypatch, world, cup)

    store = Store(
        str(tmp_path / f"{label}_rig.db"),
        str(tmp_path / f"{label}_frames"),
        str(tmp_path / f"{label}_backups"),
    )
    store.set_volumes(VolumeState(0.0, 0.0, 0.0))
    if start_run:
        store.start_run(
            target_hex=target_hex,
            target_lab=rgb_to_lab(hex_to_rgb(target_hex)),
            initial_lambda=0.0,
            cycle_budget=cycle_budget,
            model="sim-test",
        )

    camera = SimCamera(
        FAST_CAMERA_DEFAULTS,
        world,
        cup,
        width=config.camera.width,
        height=config.camera.height,
        noise=noise,
    )
    app = create_app(config=config, driver=FakeRelayDriver(), camera=camera, store=store)
    return app, store


def _target_hex_for_dose(world, dose_a_ml, dose_b_ml, water_ml):
    """The 8-bit hex a run's console would set to make target_lab agree with
    world.true_lab(dose_a_ml, dose_b_ml) up to quantization -- same per-channel Beer-
    Lambert formula SimWorld._cup_lab uses internally, from explicit dye/water ml."""
    conc_a = dose_a_ml / water_ml
    conc_b = dose_b_ml / water_ml
    reflectance = tuple(
        math.exp(-(conc_a * abs_a + conc_b * abs_b))
        for abs_a, abs_b in zip(world._dye_a, world._dye_b)
    )
    r, g, b = linear_rgb_to_srgb8(reflectance)
    return f"#{r:02x}{g:02x}{b:02x}"


class GridRefinementAdapter:
    """Deterministic optimizer-shaped ProviderAdapter (no random module, no wall-clock):
    a fixed 4x4 grid over [0, dose_cap]^2 for the first 16 proposals, then axis-aligned
    refinement around the best-scoring COMPLETED cycle so far, read via ITS OWN GET
    /history call -- never by parsing the prompt argument. Cycles through the four
    axis-aligned neighbor directions, halving the step every full pass through them."""

    _DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def __init__(self, client, run_id, dose_cap):
        self._client = client
        self._run_id = run_id
        self._dose_cap = dose_cap
        grid_vals = (0.0, dose_cap / 3, 2 * dose_cap / 3, dose_cap)
        self._grid = [(a, b) for a in grid_vals for b in grid_vals]
        self._calls = 0
        self._step = dose_cap / 3
        self._dir_index = 0

    def propose(self, prompt, model):
        self._calls += 1
        if self._calls <= len(self._grid):
            dose_a, dose_b = self._grid[self._calls - 1]
            rationale = f"grid point {self._calls}"
        else:
            history = self._client.get(
                f"/history?run={self._run_id}", headers={"X-Client": "scientist"}
            ).json()
            completed = [c for c in history["cycles"] if c["aborted"] is None]
            best = min(completed, key=lambda c: c["score"])
            if self._dir_index > 0 and self._dir_index % 4 == 0:
                self._step /= 2
            da, db = self._DIRECTIONS[self._dir_index % 4]
            dose_a = min(max(best["dose_a_ml"] + da * self._step, 0.0), self._dose_cap)
            dose_b = min(max(best["dose_b_ml"] + db * self._step, 0.0), self._dose_cap)
            self._dir_index += 1
            rationale = f"refine step={self._step:.4f} dir=({da},{db})"
        return json.dumps(
            {"dose_a_ml": dose_a, "dose_b_ml": dose_b, "rationale": rationale}
        )


# ---------------------------------------------------------------------------
# Group 1: SimCup
# ---------------------------------------------------------------------------


def test_cup_tracks_fills_by_role_and_drain_empties():
    cup = SimCup()
    cup.fill("dye_a", 3.0)
    cup.fill("dye_b", 2.0)
    cup.fill("water", 50.0)

    assert cup.dye_a_ml == pytest.approx(3.0)
    assert cup.dye_b_ml == pytest.approx(2.0)
    assert cup.water_ml == pytest.approx(50.0)

    cup.drain()

    assert cup.dye_a_ml == 0.0
    assert cup.dye_b_ml == 0.0
    assert cup.water_ml == 0.0


# ---------------------------------------------------------------------------
# Group 2: SimPumps (direct construction, real PumpController path underneath)
# ---------------------------------------------------------------------------


async def test_sim_pumps_updates_cup_only_on_successful_dose():
    # A fresh ModeArbiter starts "paused" -- dose() rejects before touching the cup.
    # Kills an update-before-super reorder in SimPumps.dose.
    config = _sim_config()
    driver, accountant, arbiter = _make_pump_scaffold(config)
    world = SimWorld()
    cup = SimCup()
    sim_pumps = SimPumps(config, driver, accountant, arbiter, world=world, cup=cup)

    with pytest.raises(ModeError):
        await sim_pumps.dose("A", 1.0)

    assert cup.dye_a_ml == 0.0
    assert cup.dye_b_ml == 0.0
    assert cup.water_ml == 0.0


async def test_sim_pumps_preserves_real_cap_rejection():
    config = _sim_config()
    driver, accountant, arbiter = _make_pump_scaffold(config)
    arbiter.set_mode("manual", "console")  # off the paused floor so validate() is reached
    world = SimWorld()
    cup = SimCup()
    sim_pumps = SimPumps(config, driver, accountant, arbiter, world=world, cup=cup)

    # Channel A's max_ml_per_dose is 5 ml (make_fast_cycle_config) -- 1000 ml is over-cap.
    with pytest.raises(PumpError):
        await sim_pumps.dose("A", 1000.0)

    assert cup.dye_a_ml == 0.0
    assert cup.water_ml == 0.0


# ---------------------------------------------------------------------------
# Group 3: SimCamera (direct construction, no HTTP)
# ---------------------------------------------------------------------------


def test_sim_camera_renders_water_as_white_and_dye_shifts_lab():
    world = SimWorld(base_ml=100.0, noise_sigma=0.0, seed=1)
    cup = SimCup()
    cup.fill("water", 100.0)
    camera = SimCamera(FAST_CAMERA_DEFAULTS, world, cup, width=32, height=32, noise=False)

    white_lab = lab_from_jpeg(camera.capture_jpeg(), 0.5, 0.5, 16)
    assert white_lab[0] == pytest.approx(100.0, abs=1.5)
    assert abs(white_lab[1]) < 1.5
    assert abs(white_lab[2]) < 1.5

    cup.fill("dye_a", 3.0)
    cup.fill("dye_b", 1.5)
    dyed_lab = lab_from_jpeg(camera.capture_jpeg(), 0.5, 0.5, 16)

    # Actual delta_e from white, not just "not exactly white" -- a trivially-passing
    # near-white check would survive a broken absorbance formula.
    assert delta_e(dyed_lab, white_lab) > 10.0


def test_sim_camera_matches_world_true_lab_when_noiseless():
    base_ml = 10.0  # small doses keep reflectance mid-range, not saturated near-black
    world = SimWorld(base_ml=base_ml, noise_sigma=0.4, seed=7)
    cup = SimCup()
    cup.fill("water", base_ml)
    cup.fill("dye_a", 0.1)
    cup.fill("dye_b", 0.05)
    camera = SimCamera(FAST_CAMERA_DEFAULTS, world, cup, width=32, height=32, noise=False)

    measured = lab_from_jpeg(camera.capture_jpeg(), 0.5, 0.5, 16)
    expected = world.true_lab(0.1, 0.05)

    # This is the gamma trap's direct detector (W6): without linear_rgb_to_srgb8, the
    # raw reflectance rendered unencoded would be biased by several LAB units, well
    # outside this tolerance.
    for measured_component, expected_component in zip(measured, expected):
        assert measured_component == pytest.approx(expected_component, abs=1.0)


def test_sim_camera_noise_is_seeded_deterministic():
    def _capture():
        world = SimWorld(base_ml=100.0, noise_sigma=0.4, seed=42)
        cup = SimCup()
        cup.fill("water", 100.0)
        cup.fill("dye_a", 2.0)
        cup.fill("dye_b", 1.0)
        camera = SimCamera(FAST_CAMERA_DEFAULTS, world, cup, width=32, height=32, noise=True)
        return camera.capture_jpeg()

    frame1 = _capture()
    frame2 = _capture()
    assert frame1 == frame2


def test_sim_camera_no_dose_no_water_gives_white_frame():
    world = SimWorld(base_ml=100.0, noise_sigma=0.0, seed=3)
    cup = SimCup()  # freshly drained: water_ml == 0, dye ml == 0 -- no division by zero
    camera = SimCamera(FAST_CAMERA_DEFAULTS, world, cup, width=32, height=32, noise=False)

    lab = lab_from_jpeg(camera.capture_jpeg(), 0.5, 0.5, 16)
    assert lab[0] == pytest.approx(100.0, abs=1.0)
    assert abs(lab[1]) < 1.0
    assert abs(lab[2]) < 1.0


# ---------------------------------------------------------------------------
# Group 4: full cycle through the real app (SimPumps + SimCamera)
# ---------------------------------------------------------------------------


def test_full_cycle_through_sim_measures_beer_lambert_lab(tmp_path, monkeypatch):
    config = _sim_config()
    # noise_sigma=0.0: a tight deterministic check against world.true_lab, rather than
    # widening the tolerance to a few multiples of the default noise_sigma.
    world = SimWorld(base_ml=config.cycle.refill_ml, noise_sigma=0.0, seed=11)
    cup = SimCup()
    app, _store = _build_sim_app(
        tmp_path, monkeypatch, world, cup, config=config, start_run=True
    )
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)

        dose_a, dose_b = 1.0, 0.5
        response = client.post(
            "/cycle", json={"dose_a_ml": dose_a, "dose_b_ml": dose_b}, headers=CONSOLE
        )

        assert response.status_code == 200
        measured = response.json()["measured_lab"]
        expected = world.true_lab(dose_a, dose_b)
        for measured_component, expected_component in zip(measured, expected):
            assert measured_component == pytest.approx(expected_component, abs=1.0)


# ---------------------------------------------------------------------------
# Group 5: disruptions through the real endpoints (no injector shortcuts)
# ---------------------------------------------------------------------------


def test_skewed_sim_cycle_shifts_measurement_but_not_history_doses(tmp_path, monkeypatch):
    config = _sim_config()
    world = SimWorld(base_ml=config.cycle.refill_ml, noise_sigma=0.0, seed=13)
    cup = SimCup()
    app, _store = _build_sim_app(
        tmp_path, monkeypatch, world, cup, config=config, start_run=True
    )
    with TestClient(app) as client:
        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
        client.post(
            "/disrupt/skew", json={"channel": "A", "factor": 0.5}, headers=CONSOLE
        )

        dose_a, dose_b = 2.0, 1.0
        response = client.post(
            "/cycle", json={"dose_a_ml": dose_a, "dose_b_ml": dose_b}, headers=CONSOLE
        )
        assert response.status_code == 200

        # Physical half of the visibility contract: measured LAB tracks the executed
        # (halved) dose, via SimPumps having actually doused half as much A into the cup.
        measured = response.json()["measured_lab"]
        expected = world.true_lab(dose_a * 0.5, dose_b)
        for measured_component, expected_component in zip(measured, expected):
            assert measured_component == pytest.approx(expected_component, abs=1.0)

        # Requested-only visibility: history's dose field still shows the unskewed
        # request the scientist sent, not the physically executed one.
        run_id = client.get("/status", headers=CONSOLE).json()["run"]["id"]
        history = client.get(f"/history?run={run_id}", headers=SCIENTIST).json()
        assert history["cycles"][0]["dose_a_ml"] == pytest.approx(dose_a)


def test_blip_through_real_endpoint_dilutes_sim_measurement(tmp_path, monkeypatch):
    config = _blip_test_config()
    dose_a, dose_b = 1.0, 0.5

    # Control cycle: identical doses, no blip.
    control_world = SimWorld(base_ml=config.cycle.refill_ml, noise_sigma=0.0, seed=17)
    control_cup = SimCup()
    control_app, _control_store = _build_sim_app(
        tmp_path, monkeypatch, control_world, control_cup,
        config=config, start_run=True, label="control",
    )
    with TestClient(control_app) as client:
        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)
        control_response = client.post(
            "/cycle", json={"dose_a_ml": dose_a, "dose_b_ml": dose_b}, headers=CONSOLE
        )
        assert control_response.status_code == 200
        control_lab = tuple(control_response.json()["measured_lab"])

    # Blipped cycle: a real water dose injected mid-SETTLE via POST /disrupt/blip.
    blip_world = SimWorld(base_ml=config.cycle.refill_ml, noise_sigma=0.0, seed=17)
    blip_cup = SimCup()
    blip_app, _blip_store = _build_sim_app(
        tmp_path, monkeypatch, blip_world, blip_cup,
        config=config, start_run=True, label="blip",
    )
    with TestClient(blip_app) as client:
        client.post("/mode", json={"mode": "manual"}, headers=CONSOLE)

        release = threading.Event()
        result = {}
        with _patched_sleep_for_phase("settle", config, release) as entered:

            def run():
                result["response"] = client.post(
                    "/cycle",
                    json={"dose_a_ml": dose_a, "dose_b_ml": dose_b},
                    headers=CONSOLE,
                )

            thread = threading.Thread(target=run)
            thread.start()
            entered.wait(timeout=5)
            assert (
                client.get("/status", headers=CONSOLE).json()["cycle"]["phase"]
                == "settle"
            )

            water_before = blip_cup.water_ml
            blip_response = client.post(
                "/disrupt/blip", json={"ml_water": 5.0}, headers=CONSOLE
            )
            assert blip_response.status_code == 200
            # The blip landed through the real SimPumps.dose("W", ...) path -- the cup's
            # water_ml actually rose, which is what must drive the measurement shift.
            assert blip_cup.water_ml == pytest.approx(water_before + 5.0)

            release.set()
            thread.join(timeout=5)

        assert result["response"].status_code == 200
        blipped_lab = tuple(result["response"].json()["measured_lab"])

    assert delta_e(blipped_lab, WHITE_LAB) < delta_e(control_lab, WHITE_LAB)

    # Strengthen beyond "closer to white": the blip must dilute by (close to) the full
    # 5.0 ml it was asked for, not a trivial fraction of it in the right direction. Each
    # measured LAB is checked against the sim's own Beer-Lambert prediction for the exact
    # cup contents a correct blip produces -- control at refill_ml water, blip at
    # refill_ml + 5.0 ml. A blip that actuated e.g. only 0.1 ml of the intended 5.0 ml
    # water would leave blipped_lab within ~0.1 delta_e of control_lab and ~15.9 delta_e
    # of expected_blipped_lab (verified against this exact model), so 0.5 is a tight,
    # stable tolerance that only 8-bit quantization/float error can satisfy.
    expected_control_lab = _expected_dilution_lab(dose_a, dose_b, config.cycle.refill_ml)
    expected_blipped_lab = _expected_dilution_lab(
        dose_a, dose_b, config.cycle.refill_ml + 5.0
    )
    assert delta_e(control_lab, expected_control_lab) < 0.5
    assert delta_e(blipped_lab, expected_blipped_lab) < 0.5


# ---------------------------------------------------------------------------
# Group 6: the real scientist loop converges via the sim (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_scientist_loop_converges_on_reachable_target_via_sim(tmp_path, monkeypatch):
    config = _sim_config()
    dose_cap = config.channels["A"].max_ml_per_dose  # 5.0 ml; B shares the same cap
    cycle_budget = 30

    world = SimWorld(base_ml=config.cycle.refill_ml, noise_sigma=0.4, seed=23)
    cup = SimCup()

    a0, b0 = dose_cap / 3, 2 * dose_cap / 3
    target_hex = _target_hex_for_dose(world, a0, b0, config.cycle.refill_ml)

    app, _store = _build_sim_app(
        tmp_path, monkeypatch, world, cup, config=config, start_run=False, noise=True,
    )
    with TestClient(app) as client:
        start_response = client.post(
            "/run/start",
            json={
                "target_hex": target_hex,
                "lambda": 0.0,
                "cycle_budget": cycle_budget,
                "model": "sim-grid-refine",
            },
            headers=CONSOLE,
        )
        assert start_response.status_code == 200
        run_id = start_response.json()["run_id"]
        client.post("/mode", json={"mode": "auto"}, headers=CONSOLE)

        adapter = GridRefinementAdapter(client, run_id, dose_cap)
        cfg = ScientistConfig(
            base_url="http://testserver",
            poll_idle_s=0.0,
            poll_active_s=0.0,
            request_timeout_s=10.0,
            cycle_timeout_s=900.0,
            llm_max_tokens=2000,
            llm_retries=2,
            dose_cap_hint_ml=dose_cap,
            conv_delta_e=3.0,
            conv_hold=3,
            conv_epsilon=0.5,
            conv_window=5,
            conv_min_cycles=5,
        )
        scientist = Scientist(client=client, adapter=adapter, cfg=cfg, sleep=lambda _: None)

        # One extra tick beyond cycle_budget: should_stop's own budget check runs BEFORE
        # the (cycle_budget)-th proposal is sent, so the "stopped_budget" verdict itself
        # is only observable on the tick immediately after the budget-th cycle completes.
        stopped_action = None
        for _ in range(cycle_budget + 1):
            action = scientist.tick()
            if action.startswith("stopped_"):
                stopped_action = action
                break

        assert stopped_action in {"stopped_converged", "stopped_budget"}

        history = client.get(f"/history?run={run_id}", headers=SCIENTIST).json()
        assert len(history["cycles"]) <= cycle_budget
        completed = [c for c in history["cycles"] if c["aborted"] is None]
        best_delta_e = min(c["delta_e"] for c in completed)
        assert best_delta_e < 5.0
