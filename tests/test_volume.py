# ABOUTME: Tests for cup/session volume accounting: fail-closed unknown-cup state, the durable
# ABOUTME: on_change sink's persist-then-mutate ordering, and the begin_drain/complete_drain split.

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rig_service.volume import VolumeAccountant, VolumeError, VolumeState


class RecordingSink:
    """Fakes the durable store: appends every persisted VolumeState. Tests assert on the
    accountant's real logic (ordering, contents) against this recorded log, never on the
    sink's own (trivial) behavior."""

    def __init__(self):
        self.states = []

    def __call__(self, state):
        self.states.append(state)


def make_accountant(state=None, on_change=None, **overrides):
    limits = {
        "cup_ceiling_ml": 150.0,
        "reservoir_ml": 6000.0,
        "waste_capacity_ml": 6000.0,
        "low_margin_ml": 300.0,
    }
    limits.update(overrides)
    if state is None:
        state = VolumeState(0.0, 0.0, 0.0)
    if on_change is None:
        on_change = lambda _: None  # noqa: E731
    return VolumeAccountant(state=state, on_change=on_change, **limits)


# --- constructor: state and on_change have no defaults (invariant 8) ---


def test_constructor_requires_state_and_on_change():
    # A default would re-embed the empty-cup assumption (state) or silently drop
    # durability (on_change); both must be supplied explicitly by every caller.
    with pytest.raises(TypeError):
        VolumeAccountant(
            cup_ceiling_ml=150.0,
            reservoir_ml=6000.0,
            waste_capacity_ml=6000.0,
            low_margin_ml=300.0,
        )


# --- unknown-cup fail-closed semantics (decision 1) ---


def test_constructed_unknown_reports_none():
    acc = make_accountant(state=VolumeState(None, 0.0, 0.0))
    assert acc.cup_ml is None


def test_fill_on_unknown_cup_rejected():
    sink = RecordingSink()
    acc = make_accountant(state=VolumeState(None, 0.0, 0.0), on_change=sink)
    with pytest.raises(VolumeError):
        acc.fill(5)
    assert acc.cup_ml is None
    assert sink.states == []


def test_fill_to_exact_ceiling_accepted():
    acc = make_accountant(cup_ceiling_ml=10.0)
    acc.fill(10.0)
    assert acc.cup_ml == pytest.approx(10.0)


def test_drain_of_exact_ceiling_resolves_unknown_to_zero():
    acc = make_accountant(state=VolumeState(None, 0.0, 0.0), cup_ceiling_ml=10.0)
    acc.begin_drain(10.0)
    acc.complete_drain()
    assert acc.cup_ml == 0.0


def test_drain_below_ceiling_keeps_unknown():
    acc = make_accountant(state=VolumeState(None, 0.0, 0.0), cup_ceiling_ml=10.0)
    acc.begin_drain(9.9)
    acc.complete_drain()
    assert acc.cup_ml is None


def test_unknown_drain_charges_waste_at_most_ceiling():
    acc = make_accountant(state=VolumeState(None, 0.0, 0.0), cup_ceiling_ml=10.0)
    charged = acc.begin_drain(50.0)
    assert charged == pytest.approx(10.0)
    acc.complete_drain()
    assert acc.waste_ml == pytest.approx(10.0)


def test_unknown_drain_below_ceiling_charges_ml():
    acc = make_accountant(state=VolumeState(None, 0.0, 0.0), cup_ceiling_ml=10.0)
    charged = acc.begin_drain(4.0)
    assert charged == pytest.approx(4.0)
    acc.complete_drain()
    assert acc.waste_ml == pytest.approx(4.0)


def test_known_drain_charges_min_of_cup_and_ml():
    dry_run = make_accountant()
    dry_run.fill(30.0)
    charged = dry_run.begin_drain(150.0)  # pump runs dry: min(30, 150) == 30
    assert charged == pytest.approx(30.0)
    dry_run.complete_drain()
    assert dry_run.cup_ml == pytest.approx(0.0)
    assert dry_run.waste_ml == pytest.approx(30.0)

    partial = make_accountant()
    partial.fill(30.0)
    charged = partial.begin_drain(10.0)  # min(30, 10) == 10
    assert charged == pytest.approx(10.0)
    partial.complete_drain()
    assert partial.cup_ml == pytest.approx(20.0)
    assert partial.waste_ml == pytest.approx(10.0)


@pytest.mark.parametrize("ml", [0, -5])
def test_begin_drain_nonpositive_rejected(ml):
    sink = RecordingSink()
    acc = make_accountant(on_change=sink)
    acc.fill(50.0)
    sink.states.clear()

    with pytest.raises(VolumeError):
        acc.begin_drain(ml)

    assert acc.waste_ml == pytest.approx(0.0)
    assert sink.states == []


# --- alerts ---


def test_alerts_report_unknown_cup():
    acc = make_accountant(state=VolumeState(None, 0.0, 0.0), cup_ceiling_ml=37.0)
    alerts = acc.alerts()
    assert any("unknown" in a.lower() and "37" in a for a in alerts)


def test_alert_clears_after_resolving_drain():
    acc = make_accountant(state=VolumeState(None, 0.0, 0.0), cup_ceiling_ml=10.0)
    acc.begin_drain(10.0)
    acc.complete_drain()
    assert not any("unknown" in a.lower() for a in acc.alerts())


# --- unknown session counters: fail-closed semantics (decision 15) ---


def test_constructed_unknown_counters_report_none():
    acc = make_accountant(state=VolumeState(0.0, None, None))
    assert acc.water_drawn_ml is None
    assert acc.waste_ml is None


def test_fill_rejected_while_water_counter_unknown():
    sink = RecordingSink()
    acc = make_accountant(state=VolumeState(0.0, None, 0.0), on_change=sink)

    # A dye fill (not from_reservoir) too: "unknown counters reject dosing" reads as all
    # dosing -- a dye squirt into a cup whose eventual drain can't be charged serves nothing.
    with pytest.raises(VolumeError):
        acc.fill(5.0)

    assert acc.cup_ml == pytest.approx(0.0)
    assert acc.water_drawn_ml is None
    assert acc.waste_ml == pytest.approx(0.0)
    assert sink.states == []


def test_fill_rejected_while_waste_counter_unknown():
    sink = RecordingSink()
    acc = make_accountant(state=VolumeState(0.0, 0.0, None), on_change=sink)

    with pytest.raises(VolumeError):
        acc.fill(5.0)

    assert acc.cup_ml == pytest.approx(0.0)
    assert acc.water_drawn_ml == pytest.approx(0.0)
    assert acc.waste_ml is None
    assert sink.states == []


def test_begin_drain_rejected_while_water_counter_unknown():
    sink = RecordingSink()
    # Cup known and nonzero so this isolates the counter guard from the cup guard.
    acc = make_accountant(state=VolumeState(50.0, None, 0.0), on_change=sink)

    with pytest.raises(VolumeError):
        acc.begin_drain(10.0)

    assert acc.cup_ml == pytest.approx(50.0)
    assert acc.water_drawn_ml is None
    assert acc.waste_ml == pytest.approx(0.0)
    assert sink.states == []

    acc.complete_drain()  # nothing was staged by the rejected begin_drain; stays a no-op
    assert acc.cup_ml == pytest.approx(50.0)
    assert sink.states == []


def test_begin_drain_rejected_while_waste_counter_unknown():
    sink = RecordingSink()
    acc = make_accountant(state=VolumeState(50.0, 0.0, None), on_change=sink)

    with pytest.raises(VolumeError):
        acc.begin_drain(10.0)

    assert acc.cup_ml == pytest.approx(50.0)
    assert acc.water_drawn_ml == pytest.approx(0.0)
    assert acc.waste_ml is None
    assert sink.states == []

    acc.complete_drain()
    assert acc.cup_ml == pytest.approx(50.0)
    assert sink.states == []


def test_alerts_report_unknown_water_counter():
    acc = make_accountant(state=VolumeState(0.0, None, 0.0))
    staging_alerts = [a for a in acc.alerts() if "POST /staging" in a]
    assert len(staging_alerts) == 1  # the known waste side must not also raise one
    assert "water" in staging_alerts[0].lower()
    assert "waste" not in staging_alerts[0].lower()  # the waste branch must not be the source


def test_alerts_report_unknown_waste_counter():
    acc = make_accountant(state=VolumeState(0.0, 0.0, None))
    staging_alerts = [a for a in acc.alerts() if "POST /staging" in a]
    assert len(staging_alerts) == 1  # the known water side must not also raise one
    assert "waste" in staging_alerts[0].lower()
    assert "water" not in staging_alerts[0].lower()  # the water branch must not be the source


def test_alerts_report_both_unknown_counters():
    acc = make_accountant(state=VolumeState(0.0, None, None))
    staging_alerts = [a for a in acc.alerts() if "POST /staging" in a]
    # Both counters unknown must raise two distinct alerts, not one collapsed message.
    assert len(staging_alerts) == 2
    assert any("water" in a.lower() for a in staging_alerts)
    assert any("waste" in a.lower() for a in staging_alerts)


# --- durable sink: what gets persisted, and when ---


def test_water_fill_persists_cup_and_water_drawn_together():
    sink = RecordingSink()
    acc = make_accountant(on_change=sink)
    acc.fill(20.0, from_reservoir=True)
    assert len(sink.states) == 1
    assert sink.states[-1].cup_ml == pytest.approx(20.0)
    assert sink.states[-1].water_drawn_ml == pytest.approx(20.0)


def test_dye_fill_does_not_charge_water_drawn():
    sink = RecordingSink()
    acc = make_accountant(on_change=sink)
    acc.fill(20.0, from_reservoir=False)
    assert acc.water_drawn_ml == pytest.approx(0.0)
    assert sink.states[-1].water_drawn_ml == pytest.approx(0.0)


def test_fill_persists_before_mutating():
    observed = {}
    holder = {}

    def sink(state):
        observed["memory_cup_ml_during_call"] = holder["acc"].cup_ml
        observed["persisted_cup_ml"] = state.cup_ml

    acc = make_accountant(on_change=sink)
    holder["acc"] = acc

    acc.fill(15.0)

    assert observed["memory_cup_ml_during_call"] == pytest.approx(0.0)  # old value: not yet mutated
    assert observed["persisted_cup_ml"] == pytest.approx(15.0)  # sink sees the post-fill state
    assert acc.cup_ml == pytest.approx(15.0)  # mutated once fill() returns


def test_begin_drain_persists_waste_without_touching_cup():
    sink = RecordingSink()
    acc = make_accountant(on_change=sink)
    acc.fill(50.0)
    sink.states.clear()

    acc.begin_drain(20.0)

    assert len(sink.states) == 1
    assert sink.states[-1].waste_ml == pytest.approx(20.0)
    assert sink.states[-1].cup_ml == pytest.approx(50.0)  # cup recorded high until complete_drain
    assert acc.cup_ml == pytest.approx(50.0)  # in-memory cup also untouched


def test_complete_drain_persists_cup_after_mutating():
    observed = []
    holder = {}

    def sink(state):
        observed.append((state.cup_ml, holder["acc"].cup_ml))

    acc = make_accountant(on_change=sink)
    holder["acc"] = acc
    acc.fill(50.0)
    acc.begin_drain(20.0)
    observed.clear()

    acc.complete_drain()

    assert len(observed) == 1
    persisted_cup_ml, memory_cup_ml_during_call = observed[0]
    assert persisted_cup_ml == pytest.approx(30.0)
    assert memory_cup_ml_during_call == pytest.approx(30.0)  # already mutated before the sink call


def test_begin_drain_overwrites_stranded_pending_leaving_cup_high():
    acc = make_accountant()
    acc.fill(100.0)

    acc.begin_drain(30.0)
    assert acc.cup_ml == pytest.approx(100.0)  # begin_drain never touches the cup

    acc.begin_drain(40.0)  # overwrites the stranded first pending decrement (decision 2)
    assert acc.cup_ml == pytest.approx(100.0)  # still high

    acc.complete_drain()
    assert acc.cup_ml == pytest.approx(60.0)  # only the second begin_drain's 40 ml applied
    assert acc.waste_ml == pytest.approx(70.0)  # both charges accumulate (30 + 40)


def test_complete_drain_without_pending_is_noop():
    sink = RecordingSink()
    acc = make_accountant(on_change=sink)
    acc.fill(50.0)
    sink.states.clear()

    acc.complete_drain()  # no begin_drain preceded this

    assert acc.cup_ml == pytest.approx(50.0)
    assert sink.states == []


def test_fill_sink_failure_leaves_state_unchanged():
    def failing_sink(state):
        raise RuntimeError("db down")

    acc = make_accountant(on_change=failing_sink)
    with pytest.raises(VolumeError):
        acc.fill(10.0)
    assert acc.cup_ml == pytest.approx(0.0)


def test_begin_drain_sink_failure_leaves_state_unchanged():
    calls = {"n": 0}

    def sink(state):
        calls["n"] += 1
        if calls["n"] == 2:  # 1st call is the fill below; 2nd is begin_drain's own persist
            raise RuntimeError("db down")

    acc = make_accountant(on_change=sink)
    acc.fill(50.0)

    with pytest.raises(VolumeError):
        acc.begin_drain(20.0)

    assert acc.cup_ml == pytest.approx(50.0)
    assert acc.waste_ml == pytest.approx(0.0)

    calls_before_complete = calls["n"]
    acc.complete_drain()  # a failed begin_drain must leave no pending decrement to apply
    assert acc.cup_ml == pytest.approx(50.0)  # no-op: nothing left to actuate
    assert calls["n"] == calls_before_complete  # no-op: the sink is not called again


def test_complete_drain_sink_failure_keeps_memory_raises_volume_error():
    calls = {"n": 0}

    def sink(state):
        calls["n"] += 1
        if calls["n"] == 3:  # 1st fill, 2nd begin_drain, 3rd complete_drain
            raise RuntimeError("db down")

    acc = make_accountant(on_change=sink)
    acc.fill(50.0)
    acc.begin_drain(20.0)

    with pytest.raises(VolumeError):
        acc.complete_drain()

    assert acc.cup_ml == pytest.approx(30.0)  # memory mutated despite the sink failure


# --- staging declaration: the only reset of the session counters (decision 15) ---


def test_declare_staging_maps_remaining_to_drawn():
    acc = make_accountant(state=VolumeState(0.0, None, None), reservoir_ml=6000.0)
    acc.declare_staging(water_remaining_ml=4000.0, waste_ml=100.0)
    assert acc.water_drawn_ml == pytest.approx(6000.0 - 4000.0)
    assert acc.waste_ml == pytest.approx(100.0)


def test_declare_staging_water_bound_checked_against_reservoir_not_waste_capacity():
    # reservoir_ml and waste_capacity_ml differ; water_remaining_ml sits above the waste
    # capacity but within the reservoir capacity. If the water guard were checked against
    # waste_capacity_ml (rather than reservoir_ml), this would be wrongly rejected.
    acc = make_accountant(state=VolumeState(0.0, None, None), reservoir_ml=6000.0, waste_capacity_ml=4000.0)
    acc.declare_staging(water_remaining_ml=5000.0, waste_ml=0.0)
    assert acc.water_drawn_ml == pytest.approx(1000.0)


def test_declare_staging_waste_bound_checked_against_waste_capacity_not_reservoir():
    # waste_ml sits above the waste capacity but within the reservoir capacity. If the waste
    # guard were checked against reservoir_ml (rather than waste_capacity_ml), this would be
    # wrongly accepted.
    acc = make_accountant(state=VolumeState(0.0, None, None), reservoir_ml=6000.0, waste_capacity_ml=4000.0)
    with pytest.raises(VolumeError):
        acc.declare_staging(water_remaining_ml=0.0, waste_ml=5000.0)
    assert acc.water_drawn_ml is None
    assert acc.waste_ml is None


def test_declare_staging_mapping_uses_reservoir_capacity_not_waste_capacity():
    # reservoir_ml and waste_capacity_ml differ, and water_remaining_ml sits within both
    # capacities so neither guard rejects it: only the mapping arithmetic is under test.
    acc = make_accountant(state=VolumeState(0.0, None, None), reservoir_ml=6000.0, waste_capacity_ml=4000.0)
    acc.declare_staging(water_remaining_ml=1000.0, waste_ml=0.0)
    assert acc.water_drawn_ml == pytest.approx(6000.0 - 1000.0)  # not waste_capacity_ml - 1000.0


def test_declare_staging_reservoir_alert_reports_the_declared_remainder():
    # Decision 15's mapping is justified by this exact round-trip: alerts() must read back
    # the number the operator declared, not the internally stored water_drawn_ml.
    acc = make_accountant(
        state=VolumeState(0.0, None, None), reservoir_ml=6000.0, low_margin_ml=300.0
    )
    acc.declare_staging(water_remaining_ml=250.0, waste_ml=0.0)
    reservoir_alerts = [a for a in acc.alerts() if "reservoir" in a.lower()]
    assert len(reservoir_alerts) == 1
    assert "250" in reservoir_alerts[0]


def test_declare_staging_reservoir_alert_fires_at_exact_margin():
    acc = make_accountant(
        state=VolumeState(0.0, None, None), reservoir_ml=6000.0, low_margin_ml=300.0
    )
    acc.declare_staging(water_remaining_ml=300.0, waste_ml=0.0)  # reservoir_left == margin exactly
    assert any("reservoir" in a.lower() for a in acc.alerts())


def test_declare_staging_waste_alert_fires_at_exact_margin():
    acc = make_accountant(
        state=VolumeState(0.0, None, None), waste_capacity_ml=6000.0, low_margin_ml=300.0
    )
    acc.declare_staging(water_remaining_ml=6000.0, waste_ml=5700.0)  # waste_headroom == margin exactly
    assert any("waste" in a.lower() for a in acc.alerts())


def test_declare_staging_full_reservoir_reads_drawn_zero():
    acc = make_accountant(state=VolumeState(0.0, None, None), reservoir_ml=6000.0)
    acc.declare_staging(water_remaining_ml=6000.0, waste_ml=0.0)  # boundary: accepted
    assert acc.water_drawn_ml == pytest.approx(0.0)


def test_declare_staging_empty_reservoir_accepted_and_alerts():
    acc = make_accountant(state=VolumeState(0.0, None, None), reservoir_ml=6000.0, low_margin_ml=300.0)
    acc.declare_staging(water_remaining_ml=0.0, waste_ml=0.0)  # boundary: a true, alert-raising state
    assert acc.water_drawn_ml == pytest.approx(6000.0)
    assert any("reservoir" in a.lower() for a in acc.alerts())


def test_declare_staging_rejects_remaining_over_capacity():
    acc = make_accountant(state=VolumeState(0.0, None, None), reservoir_ml=6000.0)
    with pytest.raises(VolumeError):
        acc.declare_staging(water_remaining_ml=6000.1, waste_ml=0.0)
    assert acc.water_drawn_ml is None
    assert acc.waste_ml is None


def test_declare_staging_rejects_negative_remaining():
    acc = make_accountant(state=VolumeState(0.0, None, None), reservoir_ml=6000.0)
    with pytest.raises(VolumeError):
        acc.declare_staging(water_remaining_ml=-0.1, waste_ml=0.0)
    assert acc.water_drawn_ml is None
    assert acc.waste_ml is None


def test_declare_staging_rejects_negative_waste():
    acc = make_accountant(state=VolumeState(0.0, None, None), waste_capacity_ml=6000.0)
    with pytest.raises(VolumeError):
        acc.declare_staging(water_remaining_ml=3000.0, waste_ml=-0.1)
    assert acc.water_drawn_ml is None
    assert acc.waste_ml is None


def test_declare_staging_rejects_waste_over_capacity():
    acc = make_accountant(state=VolumeState(0.0, None, None), waste_capacity_ml=6000.0)
    with pytest.raises(VolumeError):
        acc.declare_staging(water_remaining_ml=3000.0, waste_ml=6000.1)
    assert acc.water_drawn_ml is None
    assert acc.waste_ml is None


def test_declare_staging_zero_waste_accepted():
    acc = make_accountant(state=VolumeState(0.0, None, None), waste_capacity_ml=6000.0)
    acc.declare_staging(water_remaining_ml=3000.0, waste_ml=0.0)  # boundary: accepted
    assert acc.waste_ml == pytest.approx(0.0)


def test_declare_staging_full_waste_accepted_and_alerts():
    acc = make_accountant(
        state=VolumeState(0.0, None, None), waste_capacity_ml=6000.0, low_margin_ml=300.0
    )
    acc.declare_staging(water_remaining_ml=3000.0, waste_ml=6000.0)  # boundary: a full bucket is real
    assert acc.waste_ml == pytest.approx(6000.0)
    assert any("waste" in a.lower() for a in acc.alerts())


def test_declare_staging_overwrites_known_counters():
    acc = make_accountant(state=VolumeState(0.0, 500.0, 200.0), reservoir_ml=6000.0)
    acc.declare_staging(water_remaining_ml=1000.0, waste_ml=50.0)  # a repeatable, mid-session reset
    assert acc.water_drawn_ml == pytest.approx(6000.0 - 1000.0)
    assert acc.waste_ml == pytest.approx(50.0)


def test_declare_staging_leaves_cup_untouched():
    known_sink = RecordingSink()
    known_cup = make_accountant(state=VolumeState(42.0, None, None), on_change=known_sink)
    known_cup.declare_staging(water_remaining_ml=1000.0, waste_ml=50.0)
    assert known_cup.cup_ml == pytest.approx(42.0)
    assert known_sink.states[-1].cup_ml == pytest.approx(42.0)  # durable, not just in-memory

    # The cup resolves only by draining (decision 1); a staging declaration must not touch it.
    unknown_sink = RecordingSink()
    unknown_cup = make_accountant(state=VolumeState(None, None, None), on_change=unknown_sink)
    unknown_cup.declare_staging(water_remaining_ml=1000.0, waste_ml=50.0)
    assert unknown_cup.cup_ml is None
    assert unknown_sink.states[-1].cup_ml is None  # durable, not just in-memory


def test_declare_staging_persists_before_mutating():
    observed = {}
    holder = {}

    def sink(state):
        observed["memory_water_drawn_during_call"] = holder["acc"].water_drawn_ml
        observed["memory_waste_during_call"] = holder["acc"].waste_ml
        observed["persisted_water_drawn"] = state.water_drawn_ml
        observed["persisted_waste"] = state.waste_ml

    acc = make_accountant(state=VolumeState(0.0, None, None), on_change=sink, reservoir_ml=6000.0)
    holder["acc"] = acc

    acc.declare_staging(water_remaining_ml=4000.0, waste_ml=100.0)

    assert observed["memory_water_drawn_during_call"] is None  # old value: not yet mutated
    assert observed["memory_waste_during_call"] is None
    assert observed["persisted_water_drawn"] == pytest.approx(2000.0)  # sink sees the declared state
    assert observed["persisted_waste"] == pytest.approx(100.0)
    assert acc.water_drawn_ml == pytest.approx(2000.0)  # mutated once declare_staging returns
    assert acc.waste_ml == pytest.approx(100.0)


def test_declare_staging_sink_failure_leaves_state_unchanged():
    def failing_sink(state):
        raise RuntimeError("db down")

    acc = make_accountant(state=VolumeState(0.0, None, None), on_change=failing_sink, reservoir_ml=6000.0)
    with pytest.raises(VolumeError):
        acc.declare_staging(water_remaining_ml=4000.0, waste_ml=100.0)
    assert acc.water_drawn_ml is None
    assert acc.waste_ml is None


def test_counter_alerts_clear_after_staging():
    acc = make_accountant(state=VolumeState(0.0, None, None))
    acc.declare_staging(water_remaining_ml=4000.0, waste_ml=100.0)
    assert not any("POST /staging" in a for a in acc.alerts())


# --- the known world: behavior unchanged under the durable constructor and the begin/complete
# drain split ---


def test_known_world_behavior_unchanged():
    sink = RecordingSink()
    acc = make_accountant(on_change=sink)
    assert acc.cup_ml == 0
    assert acc.water_drawn_ml == 0
    assert acc.waste_ml == 0

    acc.fill(5)
    acc.fill(3)
    assert acc.cup_ml == pytest.approx(8)
    assert acc.water_drawn_ml == 0

    reservoir_acc = make_accountant()
    reservoir_acc.fill(100, from_reservoir=True)
    assert reservoir_acc.cup_ml == pytest.approx(100)
    assert reservoir_acc.water_drawn_ml == pytest.approx(100)

    ceiling_acc = make_accountant(cup_ceiling_ml=10)
    ceiling_acc.fill(8)
    with pytest.raises(VolumeError, match="ceiling"):
        ceiling_acc.fill(3)
    assert ceiling_acc.cup_ml == pytest.approx(8)  # state untouched by the rejected fill

    nonpositive_acc = make_accountant()
    with pytest.raises(VolumeError):
        nonpositive_acc.fill(0)

    drain_acc = make_accountant()
    drain_acc.fill(50)
    charged = drain_acc.begin_drain(30)
    drain_acc.complete_drain()
    assert charged == pytest.approx(30)
    assert drain_acc.cup_ml == pytest.approx(20)
    drain_acc.begin_drain(150)  # pump can run dry; only 20 ml remains to leave
    drain_acc.complete_drain()
    assert drain_acc.cup_ml == 0
    assert drain_acc.waste_ml == pytest.approx(50)

    headroom_acc = make_accountant()
    headroom_acc.fill(100, from_reservoir=True)
    assert headroom_acc.alerts() == []

    low_reservoir_acc = make_accountant(reservoir_ml=500, low_margin_ml=300, cup_ceiling_ml=1000)
    low_reservoir_acc.fill(150, from_reservoir=True)
    assert low_reservoir_acc.alerts() == []
    low_reservoir_acc.begin_drain(150)
    low_reservoir_acc.complete_drain()
    low_reservoir_acc.fill(60, from_reservoir=True)  # drawn 210, remaining 290 < margin
    assert any("reservoir" in alert for alert in low_reservoir_acc.alerts())

    low_waste_acc = make_accountant(waste_capacity_ml=500, low_margin_ml=300, cup_ceiling_ml=1000)
    low_waste_acc.fill(250)
    low_waste_acc.begin_drain(250)
    low_waste_acc.complete_drain()
    assert any("waste" in alert for alert in low_waste_acc.alerts())


# --- property-based ---


command = st.tuples(
    st.sampled_from(["fill", "fill_reservoir", "drain"]),
    st.floats(min_value=0.1, max_value=500, allow_nan=False, allow_infinity=False),
)


@given(st.lists(command, max_size=100))
def test_unknown_cup_never_fills_until_resolving_drain(commands):
    ceiling = 150.0
    acc = make_accountant(state=VolumeState(None, 0.0, 0.0), cup_ceiling_ml=ceiling)
    resolved = False
    for op, ml in commands:
        if op == "drain":
            acc.begin_drain(ml)
            acc.complete_drain()
            if ml >= ceiling:
                resolved = True
        elif not resolved:
            with pytest.raises(VolumeError):
                acc.fill(ml, from_reservoir=op == "fill_reservoir")
        else:
            try:
                acc.fill(ml, from_reservoir=op == "fill_reservoir")
            except VolumeError:
                pass  # a known cup can still legitimately reject an overfill

        if resolved:
            assert acc.cup_ml is not None
            assert 0 <= acc.cup_ml <= ceiling
        else:
            assert acc.cup_ml is None


split_command = st.tuples(
    st.sampled_from(["fill", "fill_reservoir", "begin_drain", "complete_drain"]),
    st.floats(min_value=0.1, max_value=200, allow_nan=False, allow_infinity=False),
)


@settings(max_examples=200)
@given(st.lists(split_command, max_size=60))
def test_durable_cup_and_counters_never_underreport(commands):
    """The conservative rule as one property: whatever the sink last durably recorded (what a
    crash right after that call would leave behind) never under-reports the true cup content or
    either session counter at that same instant."""
    sink = RecordingSink()
    ceiling = 150.0
    acc = make_accountant(cup_ceiling_ml=ceiling, on_change=sink)

    true_cup = 0.0
    true_water_drawn = 0.0
    true_waste = 0.0
    true_pending = None

    for op, ml in commands:
        if op in ("fill", "fill_reservoir"):
            try:
                acc.fill(ml, from_reservoir=op == "fill_reservoir")
            except VolumeError:
                pass
            else:
                true_cup += ml
                if op == "fill_reservoir":
                    true_water_drawn += ml
        elif op == "begin_drain":
            charged = acc.begin_drain(ml)
            true_pending = min(true_cup, ml)
            true_waste += true_pending
            assert charged == pytest.approx(true_pending)
        else:  # complete_drain
            acc.complete_drain()
            if true_pending is not None:
                true_cup -= true_pending
                true_pending = None

        if sink.states:
            latest = sink.states[-1]
            assert latest.cup_ml is None or latest.cup_ml >= true_cup - 1e-6
            assert latest.water_drawn_ml >= true_water_drawn - 1e-6
            assert latest.waste_ml >= true_waste - 1e-6


@settings(max_examples=200)
@given(st.lists(command, max_size=100))
def test_cup_plus_waste_conserves_accepted_fill_at_drain_boundaries(commands):
    """Two-sided conservation: cup content plus waste equals total accepted fill, checked at
    every drain-completed boundary (each "drain" here runs begin_drain immediately followed by
    complete_drain, so no gap is ever observed). The gap between begin_drain and complete_drain,
    and the stranded-pending overwrite from a second begin_drain before completion, are decision
    2's separately pinned conservative over-reports and must not be asserted away here."""
    ceiling = 150.0
    acc = make_accountant(cup_ceiling_ml=ceiling)
    accepted_fill = 0.0

    for op, ml in commands:
        if op == "drain":
            acc.begin_drain(ml)
            acc.complete_drain()
        else:
            try:
                acc.fill(ml, from_reservoir=op == "fill_reservoir")
            except VolumeError:
                pass
            else:
                accepted_fill += ml

        assert acc.cup_ml + acc.waste_ml == pytest.approx(accepted_fill, abs=1e-6)


staging_command = st.tuples(
    st.sampled_from(["fill", "fill_reservoir", "begin_drain", "complete_drain", "declare_staging"]),
    st.floats(min_value=0.0, max_value=6000, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=6000, allow_nan=False, allow_infinity=False),
)


@settings(max_examples=200)
@given(st.lists(staging_command, max_size=60))
def test_durable_state_never_underreports_any_quantity(commands):
    """test_durable_cup_and_counters_never_underreport's conservative-rule property, extended
    with declare_staging: the session counters start unknown, so every fill and begin_drain is
    rejected (true water_drawn/waste cannot advance) until a declaration resolves them, and each
    successful declaration re-bases "true" to the decision-15 mapping. Whatever the sink last
    durably recorded -- what a crash right after that call would leave behind -- never
    under-reports the true cup content or either session counter at that same instant."""
    sink = RecordingSink()
    ceiling = 150.0
    reservoir_ml = 6000.0
    waste_capacity_ml = 6000.0
    acc = make_accountant(
        state=VolumeState(0.0, None, None),
        on_change=sink,
        cup_ceiling_ml=ceiling,
        reservoir_ml=reservoir_ml,
        waste_capacity_ml=waste_capacity_ml,
    )

    true_cup = 0.0
    true_water_drawn = None
    true_waste = None
    true_pending = None

    for op, a, b in commands:
        if op in ("fill", "fill_reservoir"):
            try:
                acc.fill(a, from_reservoir=op == "fill_reservoir")
            except VolumeError:
                pass
            else:
                true_cup += a
                if op == "fill_reservoir":
                    true_water_drawn += a
        elif op == "begin_drain":
            try:
                charged = acc.begin_drain(a)
            except VolumeError:
                pass
            else:
                true_pending = min(true_cup, a)
                true_waste += true_pending
                assert charged == pytest.approx(true_pending)
        elif op == "complete_drain":
            acc.complete_drain()
            if true_pending is not None:
                true_cup -= true_pending
                true_pending = None
        else:  # declare_staging
            try:
                acc.declare_staging(water_remaining_ml=a, waste_ml=b)
            except VolumeError:
                pass
            else:
                true_water_drawn = reservoir_ml - a
                true_waste = b

        if sink.states:
            latest = sink.states[-1]
            # Nothing can persist before the first successful declare_staging (fill/begin_drain
            # are rejected while either counter is unknown), so a nonempty sink log means both
            # counters are already known -- never re-check against a stale "unknown".
            assert true_water_drawn is not None
            assert true_waste is not None
            assert latest.cup_ml is None or latest.cup_ml >= true_cup - 1e-6
            assert latest.water_drawn_ml >= true_water_drawn - 1e-6
            assert latest.waste_ml >= true_waste - 1e-6
