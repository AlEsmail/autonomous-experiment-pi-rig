# ABOUTME: Tests for rig-level mode arbitration: auto/manual/paused ownership, per-client
# ABOUTME: dose-command authorization, and watchdog/budget trips forcing paused.

import pytest

from rig_service.mode import ModeArbiter, ModeError


@pytest.fixture()
def arbiter():
    return ModeArbiter(on_trip=None)


def test_boots_into_paused(arbiter):
    assert arbiter.mode == "paused"


def test_console_can_set_every_mode(arbiter):
    for mode in ("auto", "manual", "paused"):
        arbiter.set_mode(mode, client="console")
        assert arbiter.mode == mode


def test_scientist_cannot_set_mode(arbiter):
    with pytest.raises(ModeError, match="console"):
        arbiter.set_mode("auto", client="scientist")
    assert arbiter.mode == "paused"


def test_invalid_mode_rejected(arbiter):
    with pytest.raises(ModeError, match="mode"):
        arbiter.set_mode("turbo", client="console")


def test_unknown_client_rejected(arbiter):
    arbiter.set_mode("manual", client="console")
    with pytest.raises(ModeError, match="client"):
        arbiter.authorize("intruder", "pump")


def test_pump_allowed_only_for_console_in_manual(arbiter):
    arbiter.set_mode("manual", client="console")
    arbiter.authorize("console", "pump")  # must not raise

    with pytest.raises(ModeError):
        arbiter.authorize("scientist", "pump")

    arbiter.set_mode("auto", client="console")
    with pytest.raises(ModeError, match="manual"):
        arbiter.authorize("console", "pump")


def test_calibrate_follows_pump_rules(arbiter):
    arbiter.set_mode("manual", client="console")
    arbiter.authorize("console", "calibrate")
    with pytest.raises(ModeError):
        arbiter.authorize("scientist", "calibrate")


def test_cycle_authority_follows_mode(arbiter):
    arbiter.set_mode("auto", client="console")
    arbiter.authorize("scientist", "cycle")
    with pytest.raises(ModeError, match="auto"):
        arbiter.authorize("console", "cycle")

    arbiter.set_mode("manual", client="console")
    arbiter.authorize("console", "cycle")
    with pytest.raises(ModeError, match="manual"):
        arbiter.authorize("scientist", "cycle")


def test_nobody_doses_in_paused(arbiter):
    for client in ("scientist", "console"):
        for command in ("pump", "cycle", "calibrate"):
            with pytest.raises(ModeError, match="paused"):
                arbiter.authorize(client, command)


def test_unknown_command_rejected(arbiter):
    arbiter.set_mode("manual", client="console")
    with pytest.raises(ModeError, match="command"):
        arbiter.authorize("console", "explode")


def test_trip_forces_paused_and_records_reason(arbiter):
    arbiter.set_mode("auto", client="console")
    arbiter.trip("watchdog: refill phase exceeded budget")
    assert arbiter.mode == "paused"
    assert "watchdog" in arbiter.trip_reason


def test_operator_mode_change_clears_trip_reason(arbiter):
    arbiter.trip("budget: reservoir low")
    arbiter.set_mode("manual", client="console")
    assert arbiter.trip_reason is None


def test_arbiter_on_trip_called_after_mode_change():
    calls = []

    def on_trip(reason):
        calls.append((arbiter.mode, arbiter.trip_reason, reason))

    arbiter = ModeArbiter(on_trip=on_trip)
    arbiter.set_mode("auto", client="console")
    arbiter.trip("budget: reservoir low")

    assert arbiter.mode == "paused"
    assert len(calls) == 1
    assert calls[0] == ("paused", "budget: reservoir low", "budget: reservoir low")


def test_arbiter_without_on_trip_unchanged():
    arbiter = ModeArbiter()
    arbiter.set_mode("auto", client="console")
    arbiter.trip("watchdog: refill phase exceeded budget")

    assert arbiter.mode == "paused"
    assert arbiter.trip_reason == "watchdog: refill phase exceeded budget"


def test_trip_pauses_even_when_on_trip_raises():
    def on_trip(reason):
        raise RuntimeError("sink write failed")

    arbiter = ModeArbiter(on_trip=on_trip)
    arbiter.set_mode("auto", client="console")

    with pytest.raises(RuntimeError, match="sink write failed"):
        arbiter.trip("budget: waste bucket nearly full")

    assert arbiter.mode == "paused"
    assert arbiter.trip_reason == "budget: waste bucket nearly full"
