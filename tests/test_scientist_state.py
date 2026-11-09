# ABOUTME: Tests for the ScientistState holder: validation, stop-transition detection,
# ABOUTME: and the snapshot/age contract against an injected clock.

import pytest

from rig_service.scientist_state import ScientistState


class FakeClock:
    """Controllable monotonic clock: tests advance .now explicitly."""

    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def test_snapshot_before_any_report_is_unknown():
    holder = ScientistState(clock=FakeClock())
    assert holder.snapshot() == {"state": "unknown", "detail": "", "age_s": None}


def test_valid_update_stores_state_and_detail():
    holder = ScientistState(clock=FakeClock())
    assert holder.update("thinking", "asking the model") is False
    snapshot = holder.snapshot()
    assert snapshot["state"] == "thinking"
    assert snapshot["detail"] == "asking the model"


def test_every_valid_state_accepted():
    holder = ScientistState(clock=FakeClock())
    for state in (
        "idle", "thinking", "waiting", "stopped_budget", "stopped_converged",
        "llm_error", "parse_error", "cycle_refused", "cycle_failed",
    ):
        holder.update(state, "")
    assert holder.snapshot()["state"] == "cycle_failed"


def test_unknown_state_raises_value_error():
    holder = ScientistState(clock=FakeClock())
    with pytest.raises(ValueError):
        holder.update("asleep", "")
    # A rejected update must not disturb the held report.
    assert holder.snapshot() == {"state": "unknown", "detail": "", "age_s": None}


def test_detail_over_200_chars_raises_value_error():
    holder = ScientistState(clock=FakeClock())
    holder.update("idle", "x" * 200)  # exactly at the limit: accepted
    with pytest.raises(ValueError):
        holder.update("idle", "x" * 201)
    assert holder.snapshot()["detail"] == "x" * 200


def test_update_returns_true_only_on_entry_into_stopped_budget():
    holder = ScientistState(clock=FakeClock())
    assert holder.update("waiting", "") is False
    assert holder.update("stopped_budget", "budget") is True
    # Repeat reports of the same stopped state are not transitions.
    assert holder.update("stopped_budget", "budget") is False


def test_update_returns_true_only_on_entry_into_stopped_converged():
    holder = ScientistState(clock=FakeClock())
    assert holder.update("stopped_converged", "converged_delta_e") is True
    assert holder.update("stopped_converged", "converged_delta_e") is False


def test_switching_between_stopped_states_is_a_transition():
    holder = ScientistState(clock=FakeClock())
    assert holder.update("stopped_budget", "budget") is True
    assert holder.update("stopped_converged", "converged_delta_e") is True


def test_reentry_into_stopped_state_after_other_state_is_a_transition():
    holder = ScientistState(clock=FakeClock())
    assert holder.update("stopped_budget", "budget") is True
    assert holder.update("idle", "") is False
    assert holder.update("stopped_budget", "budget") is True


def test_non_stopped_updates_never_return_true():
    holder = ScientistState(clock=FakeClock())
    for state in ("idle", "thinking", "waiting", "llm_error", "parse_error",
                  "cycle_refused", "cycle_failed"):
        assert holder.update(state, "") is False


def test_age_s_counts_seconds_since_last_update():
    clock = FakeClock(now=1000.0)
    holder = ScientistState(clock=clock)
    holder.update("idle", "")
    clock.now = 1002.5
    assert holder.snapshot()["age_s"] == pytest.approx(2.5)


def test_age_s_resets_on_each_update():
    clock = FakeClock(now=1000.0)
    holder = ScientistState(clock=clock)
    holder.update("idle", "")
    clock.now = 1010.0
    holder.update("thinking", "")
    clock.now = 1011.0
    assert holder.snapshot()["age_s"] == pytest.approx(1.0)
