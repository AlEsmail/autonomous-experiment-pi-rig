# ABOUTME: Unit tests for the Disruptions per-channel dye-dose skew state (T1).
# ABOUTME: Factors are bounded to (0, 1]; only dye channels may be skewed.

import pytest

from rig_service.config import CameraConfig, Channel, CycleConfig, RigConfig, VolumeLimits
from rig_service.disrupt import DisruptError, Disruptions


def make_config():
    camera = CameraConfig(
        device="/dev/video0", width=64, height=64, patch_px=32, controls={}
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
            "A": Channel(
                gpio=4, role="dye", rate_ml_per_s=1000.0, lag_s=0.001, max_ml_per_dose=5
            ),
            "B": Channel(
                gpio=5, role="dye", rate_ml_per_s=1000.0, lag_s=0.001, max_ml_per_dose=5
            ),
            "D": Channel(
                gpio=6, role="drain", rate_ml_per_s=1000.0, lag_s=0.001, max_ml_per_dose=20
            ),
            "W": Channel(
                gpio=7, role="water", rate_ml_per_s=1000.0, lag_s=0.001, max_ml_per_dose=20
            ),
        },
        max_relay_on_s=1.0,
        volumes=VolumeLimits(
            cup_ceiling_ml=20,
            reservoir_ml=6000,
            waste_capacity_ml=6000,
            low_margin_ml=20,
        ),
        camera=camera,
        cycle=cycle,
    )


def test_factor_defaults_to_identity():
    config = make_config()
    disruptions = Disruptions(config)
    assert disruptions.factor("A") == 1.0
    assert disruptions.factor("B") == 1.0
    assert disruptions.factor("D") == 1.0
    assert disruptions.factor("W") == 1.0


def test_set_skew_stores_factor():
    config = make_config()
    disruptions = Disruptions(config)
    disruptions.set_skew("A", 0.75)
    assert disruptions.factor("A") == 0.75
    assert disruptions.factor("B") == 1.0


def test_set_skew_factor_one_clears_entry():
    config = make_config()
    disruptions = Disruptions(config)
    disruptions.set_skew("A", 0.75)
    assert "A" in disruptions.factors
    disruptions.set_skew("A", 1.0)
    assert "A" not in disruptions.factors
    assert disruptions.factor("A") == 1.0


def test_set_skew_rejects_zero_factor():
    config = make_config()
    disruptions = Disruptions(config)
    with pytest.raises(DisruptError):
        disruptions.set_skew("A", 0.0)


def test_set_skew_rejects_negative_factor():
    config = make_config()
    disruptions = Disruptions(config)
    with pytest.raises(DisruptError):
        disruptions.set_skew("A", -0.1)


def test_set_skew_rejects_factor_above_one():
    config = make_config()
    disruptions = Disruptions(config)
    disruptions.set_skew("A", 1.0)
    assert "A" not in disruptions.factors
    with pytest.raises(DisruptError):
        disruptions.set_skew("A", 1.0 + 1e-9)


def test_set_skew_rejects_unknown_channel():
    config = make_config()
    disruptions = Disruptions(config)
    with pytest.raises(DisruptError):
        disruptions.set_skew("Z", 0.5)


def test_set_skew_rejects_non_dye_channel():
    config = make_config()
    disruptions = Disruptions(config)
    non_dye_names = [
        name for name, channel in config.channels.items() if channel.role != "dye"
    ]
    assert "D" in non_dye_names
    assert "W" in non_dye_names
    for name in non_dye_names:
        with pytest.raises(DisruptError):
            disruptions.set_skew(name, 0.5)


def test_executed_ml_applies_factor():
    config = make_config()
    disruptions = Disruptions(config)
    disruptions.set_skew("A", 0.5)
    assert disruptions.executed_ml("A", 4.0) == 2.0
    assert disruptions.executed_ml("B", 4.0) == 4.0


def test_clear_resets_all_factors():
    config = make_config()
    disruptions = Disruptions(config)
    disruptions.set_skew("A", 0.5)
    disruptions.set_skew("B", 0.75)
    disruptions.clear()
    assert disruptions.factors == {}
    assert disruptions.factor("A") == 1.0
    assert disruptions.factor("B") == 1.0


def test_factors_returns_copy():
    config = make_config()
    disruptions = Disruptions(config)
    disruptions.set_skew("A", 0.5)
    snapshot = disruptions.factors
    snapshot["A"] = 0.25
    snapshot["B"] = 0.8
    assert disruptions.factors == {"A": 0.5}
    assert disruptions.factor("B") == 1.0
