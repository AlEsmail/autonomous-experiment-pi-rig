# ABOUTME: Tests for rig-service config loading: channel map (GPIO/role/calibration/caps per
# ABOUTME: pump), global caps, and volume limits, from a JSON file — never hard-coded.

import json
from pathlib import Path

import pytest

from rig_service.config import RigConfig, load_config


def write_config(tmp_path, data):
    path = tmp_path / "rig.json"
    path.write_text(json.dumps(data))
    return path


# Camera lock set from the BRIO bring-up (Build Log, 2026-07-13 evening): bench
# placeholders until lamp + shroud + jig calibration.
CAMERA_JSON = {
    "device": "/dev/video0",
    "width": 1920,
    "height": 1080,
    "patch_px": 200,
    "controls": {
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
}

# Pre-bring-up placeholder cycle parameters; budgets get replaced with measured
# phase timings + margin once the cycle engine runs on the rig.
CYCLE_JSON = {
    "refill_ml": 100,
    "rinse_ml": 100,
    "drain_ml": 150,
    "settle_s": 10,
    "blank_drift_delta_e": 5.0,
    "phase_budgets_s": {
        "dose": 30,
        "refill": 120,
        "settle": 30,
        "measure": 60,
        "drain": 180,
        "rinse": 120,
        "blank": 60,
        "final_drain": 180,
    },
}

VALID = {
    "channels": {
        "A": {"gpio": 4, "role": "dye", "rate_ml_per_s": 1.33, "lag_s": 0.5, "max_ml_per_dose": 10},
        "B": {"gpio": 5, "role": "dye", "rate_ml_per_s": 1.33, "lag_s": 0.5, "max_ml_per_dose": 10},
        "D": {"gpio": 6, "role": "drain", "rate_ml_per_s": 1.32, "lag_s": 0.5, "max_ml_per_dose": 150},
        "W": {"gpio": 7, "role": "water", "rate_ml_per_s": 1.33, "lag_s": 0.5, "max_ml_per_dose": 150},
    },
    "caps": {"max_relay_on_s": 120},
    "volumes": {
        "cup_ceiling_ml": 150,
        "reservoir_ml": 6000,
        "waste_capacity_ml": 6000,
        "low_margin_ml": 300,
    },
    "camera": CAMERA_JSON,
    "cycle": CYCLE_JSON,
}


def test_loads_channel_map(tmp_path):
    cfg = load_config(write_config(tmp_path, VALID))
    assert isinstance(cfg, RigConfig)
    assert cfg.channels["A"].gpio == 4
    assert cfg.channels["W"].gpio == 7
    assert cfg.channels["D"].max_ml_per_dose == 150
    assert cfg.max_relay_on_s == 120


def test_all_gpios_lists_every_channel_pin(tmp_path):
    cfg = load_config(write_config(tmp_path, VALID))
    assert sorted(cfg.all_gpios()) == [4, 5, 6, 7]


def test_rejects_duplicate_gpio(tmp_path):
    data = json.loads(json.dumps(VALID))
    data["channels"]["B"]["gpio"] = 4
    with pytest.raises(ValueError, match="gpio"):
        load_config(write_config(tmp_path, data))


def test_rejects_missing_field(tmp_path):
    data = json.loads(json.dumps(VALID))
    del data["channels"]["A"]["rate_ml_per_s"]
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, data))


def test_rejects_empty_channels(tmp_path):
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, {"channels": {}, "caps": {"max_relay_on_s": 120}}))


def test_loads_channel_roles(tmp_path):
    cfg = load_config(write_config(tmp_path, VALID))
    assert cfg.channels["A"].role == "dye"
    assert cfg.channels["W"].role == "water"
    assert cfg.channels["D"].role == "drain"


def test_rejects_unknown_role(tmp_path):
    data = json.loads(json.dumps(VALID))
    data["channels"]["A"]["role"] = "sprinkler"
    with pytest.raises(ValueError, match="role"):
        load_config(write_config(tmp_path, data))


def test_rejects_missing_role(tmp_path):
    data = json.loads(json.dumps(VALID))
    del data["channels"]["A"]["role"]
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, data))


def test_loads_volume_limits(tmp_path):
    cfg = load_config(write_config(tmp_path, VALID))
    assert cfg.volumes.cup_ceiling_ml == 150
    assert cfg.volumes.reservoir_ml == 6000
    assert cfg.volumes.waste_capacity_ml == 6000
    assert cfg.volumes.low_margin_ml == 300


def test_rejects_missing_volumes_section(tmp_path):
    data = json.loads(json.dumps(VALID))
    del data["volumes"]
    with pytest.raises(ValueError, match="volumes"):
        load_config(write_config(tmp_path, data))


def test_rejects_missing_volume_field(tmp_path):
    data = json.loads(json.dumps(VALID))
    del data["volumes"]["cup_ceiling_ml"]
    with pytest.raises(ValueError, match="volumes"):
        load_config(write_config(tmp_path, data))


def test_rejects_margin_smaller_than_largest_water_dose(tmp_path):
    # A single 150 ml water dose could jump past a 100 ml margin — the budget
    # pause would fire after the reservoir crossing, not before.
    data = json.loads(json.dumps(VALID))
    data["volumes"]["low_margin_ml"] = 100
    with pytest.raises(ValueError, match="low_margin_ml"):
        load_config(write_config(tmp_path, data))


def test_rejects_margin_smaller_than_largest_drain_dose(tmp_path):
    data = json.loads(json.dumps(VALID))
    data["channels"]["D"]["max_ml_per_dose"] = 400
    with pytest.raises(ValueError, match="low_margin_ml"):
        load_config(write_config(tmp_path, data))


def test_margin_equal_to_largest_budget_dose_accepted(tmp_path):
    data = json.loads(json.dumps(VALID))
    data["volumes"]["low_margin_ml"] = 150  # equals W and D max_ml_per_dose
    cfg = load_config(write_config(tmp_path, data))
    assert cfg.volumes.low_margin_ml == 150


def test_rejects_reservoir_not_larger_than_margin(tmp_path):
    # Boot headroom must exceed the margin or the very first dose can overdraw
    # the reservoir before the post-dose budget check ever runs.
    data = json.loads(json.dumps(VALID))
    data["volumes"]["reservoir_ml"] = 300  # equals low_margin_ml
    with pytest.raises(ValueError, match="reservoir_ml"):
        load_config(write_config(tmp_path, data))


def test_rejects_waste_capacity_not_larger_than_margin(tmp_path):
    data = json.loads(json.dumps(VALID))
    data["volumes"]["waste_capacity_ml"] = 200
    with pytest.raises(ValueError, match="waste_capacity_ml"):
        load_config(write_config(tmp_path, data))


def test_direct_construction_enforces_budget_coupling():
    # The invariant must hold for every RigConfig, not only ones from load_config.
    # Channels mirror VALID (not just "W") so the camera/cycle cross-checks below have
    # a dye and drain channel to reference; low_margin_ml=50 is the sole violation.
    from rig_service.config import CameraConfig, Channel, CycleConfig, RigConfig, VolumeLimits

    with pytest.raises(ValueError, match="low_margin_ml"):
        RigConfig(
            channels={
                "A": Channel(gpio=4, role="dye", rate_ml_per_s=1.33, lag_s=0.5, max_ml_per_dose=10),
                "B": Channel(gpio=5, role="dye", rate_ml_per_s=1.33, lag_s=0.5, max_ml_per_dose=10),
                "D": Channel(gpio=6, role="drain", rate_ml_per_s=1.32, lag_s=0.5, max_ml_per_dose=150),
                "W": Channel(gpio=7, role="water", rate_ml_per_s=1.33, lag_s=0.5, max_ml_per_dose=150),
            },
            max_relay_on_s=120,
            volumes=VolumeLimits(
                cup_ceiling_ml=150, reservoir_ml=6000, waste_capacity_ml=6000, low_margin_ml=50
            ),
            camera=CameraConfig(**CAMERA_JSON),
            cycle=CycleConfig(**CYCLE_JSON),
        )


# --- camera + cycle sections ---


def test_camera_section_parsed(tmp_path):
    cfg = load_config(write_config(tmp_path, VALID))
    assert cfg.camera.device == "/dev/video0"
    assert cfg.camera.width == 1920
    assert cfg.camera.height == 1080
    assert cfg.camera.patch_px == 200
    assert cfg.camera.controls == CAMERA_JSON["controls"]


def test_cycle_section_parsed(tmp_path):
    cfg = load_config(write_config(tmp_path, VALID))
    assert cfg.cycle.refill_ml == 100
    assert cfg.cycle.rinse_ml == 100
    assert cfg.cycle.drain_ml == 150
    assert cfg.cycle.settle_s == 10
    assert cfg.cycle.blank_drift_delta_e == 5.0
    assert cfg.cycle.phase_budgets_s == CYCLE_JSON["phase_budgets_s"]


@pytest.mark.parametrize("section", ["camera", "cycle"])
def test_missing_camera_or_cycle_section_rejected(tmp_path, section):
    data = json.loads(json.dumps(VALID))
    del data[section]
    with pytest.raises(ValueError, match=section):
        load_config(write_config(tmp_path, data))


def test_missing_phase_budget_rejected(tmp_path):
    data = json.loads(json.dumps(VALID))
    del data["cycle"]["phase_budgets_s"]["dose"]
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, data))


def test_unknown_phase_budget_key_rejected(tmp_path):
    data = json.loads(json.dumps(VALID))
    data["cycle"]["phase_budgets_s"]["extra_phase"] = 10
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, data))


@pytest.mark.parametrize("bad_value", [0, -5])
def test_nonpositive_phase_budget_rejected(tmp_path, bad_value):
    data = json.loads(json.dumps(VALID))
    data["cycle"]["phase_budgets_s"]["dose"] = bad_value
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, data))


def test_settle_time_reaching_settle_budget_rejected(tmp_path):
    # settle_s must be strictly under its budget, not merely at or under it —
    # a settle that reaches the budget leaves no margin before the watchdog fires.
    data = json.loads(json.dumps(VALID))
    data["cycle"]["settle_s"] = data["cycle"]["phase_budgets_s"]["settle"]
    with pytest.raises(ValueError, match="settle"):
        load_config(write_config(tmp_path, data))


def test_refill_over_water_cap_rejected(tmp_path):
    # Lower W's per-dose cap below refill_ml so only the refill-vs-cap check trips;
    # rinse_ml is brought down too so it doesn't independently fail the same check.
    data = json.loads(json.dumps(VALID))
    data["channels"]["W"]["max_ml_per_dose"] = 50
    data["cycle"]["rinse_ml"] = 40
    data["cycle"]["refill_ml"] = 60
    with pytest.raises(ValueError, match="cap"):
        load_config(write_config(tmp_path, data))


def test_max_dose_cycle_over_cup_ceiling_rejected(tmp_path):
    # refill_ml (135) + the two dye channels at their per-dose caps (10 + 10) = 155,
    # over the 150 ml cup ceiling — a worst-case dye+water cycle would overflow the cup.
    data = json.loads(json.dumps(VALID))
    data["cycle"]["refill_ml"] = 135
    with pytest.raises(ValueError, match="ceiling"):
        load_config(write_config(tmp_path, data))


def test_drain_below_cup_ceiling_rejected(tmp_path):
    # drain_ml must be >= cup_ceiling_ml so a full drain always empties the cup.
    data = json.loads(json.dumps(VALID))
    data["cycle"]["drain_ml"] = 100
    with pytest.raises(ValueError, match="ceiling"):
        load_config(write_config(tmp_path, data))


def test_drain_failing_drain_channel_caps_rejected(tmp_path):
    # drain_ml (200) stays >= cup_ceiling (150) but exceeds D's own per-dose cap (150).
    data = json.loads(json.dumps(VALID))
    data["cycle"]["drain_ml"] = 200
    with pytest.raises(ValueError, match="cap"):
        load_config(write_config(tmp_path, data))


def test_refill_over_relay_on_time_cap_rejected(tmp_path):
    # W's per-dose cap (150) comfortably clears refill_ml (100) — only the on-time
    # dimension is violated. Slowing W to 0.5 ml/s makes refill_ml's pump time
    # (0.5 + 100/0.5 = 200.5 s) exceed max_relay_on_s (120 s). rinse_ml is lowered
    # so it doesn't independently trip the same on-time check on the same channel,
    # and the refill phase budget is raised so the (separate) phase-budget-vs-
    # worst-case-pump-time check doesn't fire instead.
    data = json.loads(json.dumps(VALID))
    data["channels"]["W"]["rate_ml_per_s"] = 0.5
    data["cycle"]["rinse_ml"] = 50
    data["cycle"]["phase_budgets_s"]["refill"] = 250
    with pytest.raises(ValueError, match="on-time"):
        load_config(write_config(tmp_path, data))


def test_rinse_over_relay_on_time_cap_rejected(tmp_path):
    # Mirror of the refill case: rinse_ml (100) at W's slowed rate (0.5 ml/s) needs
    # 200.5 s, over max_relay_on_s (120 s), while refill_ml is lowered to stay clear
    # of the same on-time check and the rinse phase budget is raised to stay clear
    # of the phase-budget-vs-worst-case-pump-time check.
    data = json.loads(json.dumps(VALID))
    data["channels"]["W"]["rate_ml_per_s"] = 0.5
    data["cycle"]["refill_ml"] = 50
    data["cycle"]["phase_budgets_s"]["rinse"] = 250
    with pytest.raises(ValueError, match="on-time"):
        load_config(write_config(tmp_path, data))


def test_drain_over_relay_on_time_cap_rejected(tmp_path):
    # drain_ml can't be lowered below cup_ceiling_ml (150) to dodge this, so D is
    # slowed instead: at 1.0 ml/s, drain_ml (150) needs 0.5 + 150/1.0 = 150.5 s,
    # over max_relay_on_s (120 s) — while staying within D's per-dose cap (150,
    # equality) and both drain phase budgets (180 s each, untouched).
    data = json.loads(json.dumps(VALID))
    data["channels"]["D"]["rate_ml_per_s"] = 1.0
    with pytest.raises(ValueError, match="on-time"):
        load_config(write_config(tmp_path, data))


@pytest.mark.parametrize("phase", ["dose", "refill", "drain", "rinse", "final_drain"])
def test_pump_phase_budget_below_worst_case_pump_time_rejected(tmp_path, phase):
    # A 1-second budget is below every phase's worst-case pump time: dose sums A+B
    # at their per-dose caps (~15 s); refill/rinse run W at refill/rinse ml (~75 s);
    # drain/final_drain run D at drain_ml (~114 s). A config that guarantees a
    # watchdog trip must not load.
    data = json.loads(json.dumps(VALID))
    data["cycle"]["phase_budgets_s"][phase] = 1
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, data))


# Mirrors the pinned REQUIRED_LOCK_CONTROLS set (see config.py) — the six controls
# the spec names as the asserted lock domains (exposure/white-balance/focus).
REQUIRED_LOCK_CONTROLS_FOR_TEST = [
    "auto_exposure",
    "exposure_time_absolute",
    "white_balance_automatic",
    "white_balance_temperature",
    "focus_automatic_continuous",
    "focus_absolute",
]


@pytest.mark.parametrize("control", REQUIRED_LOCK_CONTROLS_FOR_TEST)
def test_lock_set_missing_required_control_rejected(tmp_path, control):
    data = json.loads(json.dumps(VALID))
    del data["camera"]["controls"][control]
    with pytest.raises(ValueError, match=control):
        load_config(write_config(tmp_path, data))


def test_shipped_rig_json_loads():
    path = Path(__file__).resolve().parents[1] / "config" / "rig.json"
    cfg = load_config(path)
    assert isinstance(cfg, RigConfig)
    assert cfg.camera.device == "/dev/video0"
    assert set(cfg.camera.controls) >= set(REQUIRED_LOCK_CONTROLS_FOR_TEST)
    assert cfg.cycle.drain_ml >= cfg.volumes.cup_ceiling_ml


# --- cycle-volume bounds: rinse vs. per-dose/on-time caps and the cup ceiling ---


def test_rinse_over_water_cap_rejected(tmp_path):
    # Lower W's per-dose cap below rinse_ml, with refill_ml brought down so only
    # the rinse-vs-cap check trips (mirror of test_refill_over_water_cap_rejected).
    data = json.loads(json.dumps(VALID))
    data["channels"]["W"]["max_ml_per_dose"] = 50
    data["cycle"]["refill_ml"] = 40
    data["cycle"]["rinse_ml"] = 60
    with pytest.raises(ValueError, match="cap"):
        load_config(write_config(tmp_path, data))


def test_rinse_over_cup_ceiling_rejected(tmp_path):
    # rinse_ml (160) exceeds cup_ceiling_ml (150) but W's per-dose cap is raised
    # clear of it and W's rate is raised so the on-time cap isn't independently
    # tripped; refill+dye stays under the ceiling untouched. Only a rinse-vs-
    # ceiling check (invariant 1: RINSE fills a drained cup to rinse_ml alone)
    # should catch this.
    data = json.loads(json.dumps(VALID))
    data["channels"]["W"]["max_ml_per_dose"] = 200
    data["channels"]["W"]["rate_ml_per_s"] = 2.0
    data["cycle"]["rinse_ml"] = 160
    with pytest.raises(ValueError, match="ceiling"):
        load_config(write_config(tmp_path, data))


@pytest.mark.parametrize("bad_value", [0, -5])
def test_nonpositive_patch_px_rejected(tmp_path, bad_value):
    data = json.loads(json.dumps(VALID))
    data["camera"]["patch_px"] = bad_value
    with pytest.raises(ValueError, match="patch_px"):
        load_config(write_config(tmp_path, data))


def test_missing_patch_px_rejected(tmp_path):
    data = json.loads(json.dumps(VALID))
    del data["camera"]["patch_px"]
    with pytest.raises(ValueError, match="patch_px"):
        load_config(write_config(tmp_path, data))


def test_nonpositive_blank_drift_rejected(tmp_path):
    data = json.loads(json.dumps(VALID))
    data["cycle"]["blank_drift_delta_e"] = 0
    with pytest.raises(ValueError, match="blank_drift_delta_e"):
        load_config(write_config(tmp_path, data))


def test_missing_blank_drift_rejected(tmp_path):
    data = json.loads(json.dumps(VALID))
    del data["cycle"]["blank_drift_delta_e"]
    with pytest.raises(ValueError, match="blank_drift_delta_e"):
        load_config(write_config(tmp_path, data))


@pytest.mark.parametrize("bad_value", [0, -5])
def test_nonpositive_refill_ml_rejected(tmp_path, bad_value):
    # No check names refill_ml's own sign directly. On VALID (a water channel present),
    # a non-positive refill_ml is currently only caught as a side effect of the on-time
    # check calling shared.dose.seconds_for_dose, whose message talks about generic "ml",
    # never "refill_ml" -- and that path disappears entirely for a config with no water
    # channel. A dedicated, field-named check is required the way blank_drift_delta_e's is.
    data = json.loads(json.dumps(VALID))
    data["cycle"]["refill_ml"] = bad_value
    with pytest.raises(ValueError, match="refill_ml"):
        load_config(write_config(tmp_path, data))


@pytest.mark.parametrize("bad_value", [0, -5])
def test_nonpositive_rinse_ml_rejected(tmp_path, bad_value):
    # Mirror of test_nonpositive_refill_ml_rejected for rinse_ml.
    data = json.loads(json.dumps(VALID))
    data["cycle"]["rinse_ml"] = bad_value
    with pytest.raises(ValueError, match="rinse_ml"):
        load_config(write_config(tmp_path, data))


def test_negative_settle_s_rejected(tmp_path):
    # A negative settle_s loads clean today and asyncio.sleep(-5) returns immediately --
    # the cycle silently skips settling and MEASURE photographs unmixed dye. Whether
    # settle_s == 0 should be legal is a separate, undecided question -- only the
    # negative case is pinned here.
    data = json.loads(json.dumps(VALID))
    data["cycle"]["settle_s"] = -5
    with pytest.raises(ValueError, match="settle_s"):
        load_config(write_config(tmp_path, data))


@pytest.mark.parametrize("bad_controls", [[], "not-a-dict"])
def test_non_dict_controls_rejected(tmp_path, bad_controls):
    # A non-dict controls section must raise a clean ValueError, not the
    # AttributeError current code produces from calling .keys() on it.
    data = json.loads(json.dumps(VALID))
    data["camera"]["controls"] = bad_controls
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, data))


def test_missing_controls_subsection_rejected(tmp_path):
    data = json.loads(json.dumps(VALID))
    del data["camera"]["controls"]
    with pytest.raises(ValueError, match="controls"):
        load_config(write_config(tmp_path, data))


# --- patch_center config seed (T1) ---


def test_config_patch_center_defaults_to_mid_frame(tmp_path):
    data = json.loads(json.dumps(VALID))
    # patch_center key absent; defaults to a centered seed.
    cfg = load_config(write_config(tmp_path, data))
    assert cfg.camera.patch_center == (0.5, 0.5)


def test_config_patch_center_parsed(tmp_path):
    data = json.loads(json.dumps(VALID))
    data["camera"]["patch_center"] = [0.25, 0.75]
    cfg = load_config(write_config(tmp_path, data))
    assert cfg.camera.patch_center == (0.25, 0.75)


@pytest.mark.parametrize(
    "component,value",
    [
        ("cx", -0.1),
        ("cx", 1.1),
        ("cy", -0.1),
        ("cy", 1.1),
    ],
)
def test_config_rejects_patch_center_outside_unit_interval(tmp_path, component, value):
    data = json.loads(json.dumps(VALID))
    center = [0.5, 0.5]
    index = 0 if component == "cx" else 1
    center[index] = value
    data["camera"]["patch_center"] = center
    with pytest.raises(ValueError, match="center"):
        load_config(write_config(tmp_path, data))

    # Boundary values 0.0/1.0 pass the center check; with the default patch_px they fail
    # the box guard instead, never the center guard (kills a `<=` -> `<` mutation).
    data["camera"]["patch_center"] = [0.0, 0.5]
    with pytest.raises(ValueError, match="not wholly inside"):
        load_config(write_config(tmp_path, data))
    data["camera"]["patch_center"] = [0.5, 0.0]
    with pytest.raises(ValueError, match="not wholly inside"):
        load_config(write_config(tmp_path, data))


def test_config_rejects_seed_patch_not_fitting_frame(tmp_path):
    data = json.loads(json.dumps(VALID))
    # Near-corner center with the fixture's 200 px patch overflows the frame.
    data["camera"]["patch_center"] = [0.01, 0.01]
    with pytest.raises(ValueError, match="not wholly inside"):
        load_config(write_config(tmp_path, data))


def test_config_rejects_seed_patch_below_floor(tmp_path):
    data = json.loads(json.dumps(VALID))
    data["camera"]["patch_px"] = 19
    data["camera"]["patch_center"] = [0.5, 0.5]
    with pytest.raises(ValueError, match="floor"):
        load_config(write_config(tmp_path, data))


@pytest.mark.parametrize(
    "bad_center",
    [
        [0.5],
        [0.5, 0.5, 0.5],
        (0.5,),
        (0.5, 0.5, 0.5),
    ],
)
def test_config_rejects_patch_center_wrong_length(tmp_path, bad_center):
    data = json.loads(json.dumps(VALID))
    data["camera"]["patch_center"] = bad_center
    with pytest.raises(ValueError, match="patch_center"):
        load_config(write_config(tmp_path, data))


@pytest.mark.parametrize(
    "bad_center",
    [
        ["not-a-number", 0.5],
        [0.5, "not-a-number"],
    ],
)
def test_config_rejects_patch_center_non_numeric(tmp_path, bad_center):
    data = json.loads(json.dumps(VALID))
    data["camera"]["patch_center"] = bad_center
    with pytest.raises(ValueError, match="patch_center"):
        load_config(write_config(tmp_path, data))


def test_config_rejects_seed_patch_fitting_default_resolution_but_not_actual(tmp_path):
    # A 500 px box centered at (0.5, 0.5) fits the fixture's default 1920x1080 frame,
    # but not the actual configured 640x480 frame. This proves validate_patch uses the
    # configured width/height, not a hardcoded 1920x1080 pair.
    data = json.loads(json.dumps(VALID))
    data["camera"]["width"] = 640
    data["camera"]["height"] = 480
    data["camera"]["patch_px"] = 500
    data["camera"]["patch_center"] = [0.5, 0.5]
    with pytest.raises(ValueError, match="not wholly inside"):
        load_config(write_config(tmp_path, data))
