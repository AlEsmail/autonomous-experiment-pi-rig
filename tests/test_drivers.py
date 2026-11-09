# ABOUTME: Tests for the relay driver seam: the fake driver used for all off-Pi development
# ABOUTME: and the real driver's refusal to load on a machine without Pi GPIO support.

from pathlib import Path

import pytest

from rig_service.drivers.fake import FakeRelayDriver

ON_PI = Path("/proc/device-tree/model").exists()


def test_starts_with_no_relays_on():
    driver = FakeRelayDriver()
    assert driver.relays_on() == set()


def test_set_on_and_off():
    driver = FakeRelayDriver()
    driver.set(4, True)
    assert driver.relays_on() == {4}
    driver.set(4, False)
    assert driver.relays_on() == set()


def test_all_off_switches_everything_off():
    driver = FakeRelayDriver()
    driver.set(4, True)
    driver.set(5, True)
    driver.all_off()
    assert driver.relays_on() == set()


def test_history_records_ordered_events():
    driver = FakeRelayDriver()
    driver.all_off()
    driver.set(4, True)
    driver.set(4, False)
    assert [e[:2] for e in driver.history] == [
        ("all_off", None),
        ("on", 4),
        ("off", 4),
    ]


def test_history_events_carry_monotonic_timestamps():
    driver = FakeRelayDriver()
    driver.set(4, True)
    driver.set(4, False)
    t_on = driver.history[0][2]
    t_off = driver.history[1][2]
    assert t_off >= t_on


@pytest.mark.skipif(ON_PI, reason="on a Pi the real driver loads and claims pins")
def test_real_driver_refuses_to_load_off_pi():
    from rig_service.drivers.gpio import GpioRelayDriver

    with pytest.raises(RuntimeError, match="Raspberry Pi"):
        GpioRelayDriver(gpios=[4, 5, 6, 7])
