# ABOUTME: Tests for the per-cycle camera-lock watchdog: applying/verifying the v4l2 lock
# ABOUTME: set against a FakeCamera, and the blank-LAB drift alarm against a ΔE threshold.

import re

from hypothesis import given, settings
from hypothesis import strategies as st

from rig_service.camera_lock import CameraLock, blank_drift_alarm
from rig_service.drivers.fake import FakeCamera
from shared.color import delta_e

# The six v4l2 controls the spec names as the asserted lock domains (config.py's
# REQUIRED_LOCK_CONTROLS): each auto toggle + its dependent absolute.
LOCK_CONTROLS = {
    "auto_exposure": 1,
    "exposure_time_absolute": 250,
    "white_balance_automatic": 0,
    "white_balance_temperature": 4600,
    "focus_automatic_continuous": 0,
    "focus_absolute": 85,
}

# Firmware defaults measured on the BRIO 4K after a cold power cycle -- deliberately
# different from every lock value so a power-cycle produces a real mismatch on each
# control (test_camera.py's BRIO_DEFAULTS shape, restricted to the six lock names).
FIRMWARE_DEFAULTS = {
    "auto_exposure": 3,
    "exposure_time_absolute": 625,
    "white_balance_automatic": 1,
    "white_balance_temperature": 4000,
    "focus_automatic_continuous": 1,
    "focus_absolute": 0,
}


def test_apply_sets_all_controls_in_order():
    camera = FakeCamera(defaults=FIRMWARE_DEFAULTS)
    lock = CameraLock(camera, LOCK_CONTROLS)

    lock.apply()

    assert camera.read_controls(LOCK_CONTROLS.keys()) == LOCK_CONTROLS


def test_verify_clean_when_locked():
    camera = FakeCamera(defaults=FIRMWARE_DEFAULTS)
    lock = CameraLock(camera, LOCK_CONTROLS)
    lock.apply()

    assert lock.verify() == []


def test_verify_reports_each_mismatch_by_name():
    camera = FakeCamera(defaults=FIRMWARE_DEFAULTS)
    lock = CameraLock(camera, LOCK_CONTROLS)
    lock.apply()

    camera.power_cycle()
    messages = lock.verify()

    assert len(messages) == len(LOCK_CONTROLS)
    for name in LOCK_CONTROLS:
        assert any(name in message for message in messages), (name, messages)


def test_apply_after_power_cycle_restores_lock():
    camera = FakeCamera(defaults=FIRMWARE_DEFAULTS)
    lock = CameraLock(camera, LOCK_CONTROLS)
    lock.apply()
    camera.power_cycle()

    lock.apply()

    assert lock.verify() == []


_lab_component = st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False)
_threshold = st.floats(min_value=0.5, max_value=20.0, allow_nan=False, allow_infinity=False)


@given(
    reference=st.tuples(_lab_component, _lab_component, _lab_component),
    threshold=_threshold,
    fraction=st.floats(min_value=0.0, max_value=0.9, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=25, deadline=None)
def test_blank_drift_within_threshold_is_none(reference, threshold, fraction):
    # Perturb every axis equally by an amount whose Euclidean norm is a fixed
    # fraction (<1) of the threshold -- guarantees delta_e(blank, reference) < threshold.
    per_axis = (fraction * threshold) / (3**0.5)
    blank = tuple(c + per_axis for c in reference)

    assert blank_drift_alarm(blank, reference, threshold_delta_e=threshold) is None


@given(
    reference=st.tuples(_lab_component, _lab_component, _lab_component),
    threshold=_threshold,
    margin=st.floats(min_value=0.1, max_value=20.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=25, deadline=None)
def test_blank_drift_over_threshold_names_delta_e(reference, threshold, margin):
    # Push only the L axis out by threshold + margin: guarantees
    # delta_e(blank, reference) == threshold + margin > threshold.
    blank = (reference[0] + threshold + margin, reference[1], reference[2])
    expected_delta_e = delta_e(blank, reference)

    message = blank_drift_alarm(blank, reference, threshold_delta_e=threshold)

    assert message is not None
    reported_numbers = [float(match) for match in re.findall(r"[-+]?\d*\.?\d+", message)]
    assert any(abs(number - expected_delta_e) < 0.5 for number in reported_numbers), (
        message,
        expected_delta_e,
    )


def test_verify_reports_only_the_drifted_control():
    camera = FakeCamera(defaults=FIRMWARE_DEFAULTS)
    lock = CameraLock(camera, LOCK_CONTROLS)
    lock.apply()

    # Silently drift exactly one control -- the realistic USB webcam failure
    # mode -- leaving the other five still holding their locked values.
    camera.apply_controls({"white_balance_temperature": 5200})
    messages = lock.verify()

    assert len(messages) == 1
    assert "white_balance_temperature" in messages[0]
    other_controls = [name for name in LOCK_CONTROLS if name != "white_balance_temperature"]
    for name in other_controls:
        assert not any(name in message for message in messages), (name, messages)


def test_blank_drift_zero_drift_is_none():
    lab = (12.5, -3.0, 7.25)
    assert delta_e(lab, lab) == 0.0

    assert blank_drift_alarm(lab, lab, threshold_delta_e=5.0) is None


def test_blank_drift_exactly_at_threshold_is_none():
    reference = (0.0, 0.0, 0.0)
    threshold = 5.0
    blank = (threshold, 0.0, 0.0)
    assert delta_e(blank, reference) == threshold

    assert blank_drift_alarm(blank, reference, threshold_delta_e=threshold) is None


def test_verify_message_reports_expected_and_actual_values():
    camera = FakeCamera(defaults=FIRMWARE_DEFAULTS)
    lock = CameraLock(camera, LOCK_CONTROLS)
    lock.apply()

    camera.apply_controls({"focus_absolute": 40})
    messages = lock.verify()

    assert len(messages) == 1
    message = messages[0]
    assert str(LOCK_CONTROLS["focus_absolute"]) in message
    assert str(40) in message
