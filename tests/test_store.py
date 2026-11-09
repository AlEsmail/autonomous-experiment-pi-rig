# ABOUTME: Tests for the SQLite store: schema creation, durable rig_state volumes, run/cycle
# ABOUTME: lifecycle, per-run cycle numbering, frame files on disk, and same-transaction atomicity.

import csv
import io
import json
import shutil
import sqlite3

import pytest
from PIL import Image

from rig_service.cycle import CycleRecord
from rig_service.patch import PatchSpec
from rig_service.store import BackupResult, Store, StoreError, snapshot_integrity_ok
from rig_service.volume import VolumeState


def _connect(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _solid_jpeg(color=(120, 130, 140), size=40):
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _break_rig_state(store):
    # Makes any subsequent _upsert_rig_state raise sqlite3.IntegrityError (CHECK
    # violation) without touching the connection's transaction state -- so a causal
    # write made just before the break is still sitting in an open transaction when
    # the upsert fails.
    store._conn.execute("DROP TABLE rig_state")
    store._conn.execute(
        "CREATE TABLE rig_state (id INTEGER PRIMARY KEY CHECK (id = 1), cup_ml REAL "
        "CHECK (cup_ml < 0), water_drawn_ml REAL, waste_ml REAL)"
    )


def _repair_rig_state(store):
    store._conn.execute("DROP TABLE rig_state")
    store._conn.execute(
        "CREATE TABLE rig_state (id INTEGER PRIMARY KEY CHECK (id = 1), cup_ml REAL, "
        "water_drawn_ml REAL, waste_ml REAL)"
    )


def _break_rig_state_for_patch(store):
    # Makes _upsert_patch raise sqlite3.IntegrityError (CHECK violation on
    # patch_size_px) while leaving the volume upsert path valid, so an event
    # insert that precedes the patch upsert proves same-transaction rollback.
    store._conn.execute("DROP TABLE rig_state")
    store._conn.execute(
        "CREATE TABLE rig_state (id INTEGER PRIMARY KEY CHECK (id = 1), cup_ml REAL, "
        "water_drawn_ml REAL, waste_ml REAL, patch_cx REAL, patch_cy REAL, "
        "patch_size_px INTEGER CHECK (patch_size_px < 0))"
    )


def _all_keys(obj):
    """Yields every dict key found anywhere in a nested JSON-like structure -- the
    recursive walk the redaction tests need, since a leak could hide at any depth."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from _all_keys(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _all_keys(item)


def make_record(**overrides):
    defaults = dict(
        requested_a_ml=2.0,
        requested_b_ml=3.0,
        executed_a_ml=2.0,
        executed_b_ml=3.0,
        lam=0.5,
        target_lab=(50.0, 10.0, -5.0),
    )
    defaults.update(overrides)
    return CycleRecord(**defaults)


@pytest.fixture()
def store_paths(tmp_path):
    return {
        "db_path": tmp_path / "rig.db",
        "frames_dir": tmp_path / "frames",
        "backup_root": tmp_path / "backups",
    }


@pytest.fixture()
def store(store_paths):
    s = Store(**store_paths)
    yield s
    s.close()


# --- schema ---


def test_schema_created_and_reopen_idempotent(store_paths):
    first = Store(**store_paths)
    conn = _connect(store_paths["db_path"])
    tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    first.close()

    assert {"runs", "cycles", "decisions", "events", "rig_state"} <= tables

    # Reopening against the same file (CREATE TABLE IF NOT EXISTS) must not raise.
    second = Store(**store_paths)
    second.close()


# --- rig_state volumes: fresh-DB unknown contract and NULL <-> None roundtrip ---


def test_fresh_db_volumes_all_unknown(store):
    assert store.load_volumes() == VolumeState(None, None, None)


def test_volumes_roundtrip_with_known_cup(store):
    store.set_volumes(VolumeState(82.5, 410.0, 96.0))
    loaded = store.load_volumes()
    assert loaded.cup_ml == pytest.approx(82.5)
    assert loaded.water_drawn_ml == pytest.approx(410.0)
    assert loaded.waste_ml == pytest.approx(96.0)


def test_volumes_roundtrip_with_unknown_cup(store):
    store.set_volumes(VolumeState(None, 410.0, 96.0))
    loaded = store.load_volumes()
    assert loaded.cup_ml is None
    assert loaded.water_drawn_ml == pytest.approx(410.0)
    assert loaded.waste_ml == pytest.approx(96.0)


def test_volumes_roundtrip_with_unknown_counters(store):
    # water_drawn_ml unknown, waste_ml known: the two counters must not be coupled.
    store.set_volumes(VolumeState(82.5, None, 96.0))
    loaded = store.load_volumes()
    assert loaded.cup_ml == pytest.approx(82.5)
    assert loaded.water_drawn_ml is None
    assert loaded.waste_ml == pytest.approx(96.0)

    # waste_ml unknown, water_drawn_ml known: the reverse must also hold independently.
    store.set_volumes(VolumeState(82.5, 410.0, None))
    loaded = store.load_volumes()
    assert loaded.cup_ml == pytest.approx(82.5)
    assert loaded.water_drawn_ml == pytest.approx(410.0)
    assert loaded.waste_ml is None


def test_counters_roundtrip_and_survive_reopen(store_paths):
    first = Store(**store_paths)
    first.set_volumes(VolumeState(31.0, 512.0, 77.0))
    first.close()

    second = Store(**store_paths)
    loaded = second.load_volumes()
    second.close()

    assert loaded.cup_ml == pytest.approx(31.0)
    assert loaded.water_drawn_ml == pytest.approx(512.0)
    assert loaded.waste_ml == pytest.approx(77.0)


def test_rig_state_stays_single_row(store, store_paths):
    store.set_volumes(VolumeState(1.0, 2.0, 3.0))
    store.set_volumes(VolumeState(11.0, 22.0, 33.0))
    store.set_volumes(VolumeState(111.0, 222.0, 333.0))

    conn = _connect(store_paths["db_path"])
    rows = conn.execute("SELECT * FROM rig_state").fetchall()
    conn.close()

    assert len(rows) == 1
    assert rows[0]["cup_ml"] == pytest.approx(111.0)
    assert rows[0]["water_drawn_ml"] == pytest.approx(222.0)
    assert rows[0]["waste_ml"] == pytest.approx(333.0)


# --- runs ---


def test_start_run_returns_id_and_active_run_reports_it(store):
    run_id = store.start_run(
        target_hex="#3355ff",
        target_lab=(55.0, 12.0, -30.0),
        initial_lambda=0.4,
        cycle_budget=25,
        model="claude-x",
    )
    active = store.active_run()
    assert active is not None
    assert active["id"] == run_id
    assert active["ended_at"] is None


def test_start_run_while_active_rejected(store):
    first_id = store.start_run(
        target_hex="#112233", target_lab=(40.0, 5.0, 5.0),
        initial_lambda=0.3, cycle_budget=10, model="m1",
    )
    with pytest.raises(StoreError):
        store.start_run(
            target_hex="#445566", target_lab=(60.0, -5.0, 8.0),
            initial_lambda=0.6, cycle_budget=15, model="m2",
        )

    # The rejected attempt must not disturb the already-active run.
    active = store.active_run()
    assert active["id"] == first_id
    assert active["target_hex"] == "#112233"


def test_end_run_clears_active_and_stores_outcome(store, store_paths):
    run_id = store.start_run(
        target_hex="#abcdef", target_lab=(50.0, 0.0, 0.0),
        initial_lambda=0.5, cycle_budget=12, model="m",
    )
    store.end_run(run_id, outcome="converged")
    assert store.active_run() is None

    conn = _connect(store_paths["db_path"])
    row = conn.execute("SELECT ended_at, outcome FROM runs WHERE id = ?", (run_id,)).fetchone()
    conn.close()
    assert row["ended_at"] is not None
    assert row["outcome"] == "converged"


def test_start_run_after_ended_run_accepted(store):
    first_id = store.start_run(
        target_hex="#111111", target_lab=(10.0, 1.0, 1.0),
        initial_lambda=0.1, cycle_budget=5, model="m1",
    )
    store.end_run(first_id, outcome="budget_reached")

    # The guard is "active" (ended_at IS NULL), not "any run exists" -- a second run must
    # be accepted once the first has ended.
    second_id = store.start_run(
        target_hex="#222222", target_lab=(20.0, 2.0, 2.0),
        initial_lambda=0.2, cycle_budget=8, model="m2",
    )
    assert second_id != first_id
    active = store.active_run()
    assert active["id"] == second_id


# --- cycles ---


def test_cycle_numbers_sequential_per_run(store, store_paths):
    run1 = store.start_run(
        target_hex="#100000", target_lab=(10.0, 0.0, 0.0),
        initial_lambda=0.1, cycle_budget=10, model="m",
    )
    volumes = VolumeState(5.0, 5.0, 5.0)
    ids1 = [store.start_cycle(run1, make_record(), volumes) for _ in range(3)]

    conn = _connect(store_paths["db_path"])
    ns1 = [conn.execute("SELECT n FROM cycles WHERE id = ?", (cid,)).fetchone()["n"] for cid in ids1]
    conn.close()
    assert ns1 == [1, 2, 3]

    store.end_run(run1, outcome="budget_reached")

    run2 = store.start_run(
        target_hex="#200000", target_lab=(20.0, 0.0, 0.0),
        initial_lambda=0.2, cycle_budget=10, model="m",
    )
    cid2 = store.start_cycle(run2, make_record(), volumes)

    conn = _connect(store_paths["db_path"])
    n2 = conn.execute("SELECT n FROM cycles WHERE id = ?", (cid2,)).fetchone()["n"]
    conn.close()
    # A second run restarts numbering at 1 -- a global counter would give 4 here.
    assert n2 == 1


def test_start_cycle_stores_requested_and_executed_doses(store, store_paths):
    run_id = store.start_run(
        target_hex="#303030", target_lab=(30.0, 0.0, 0.0),
        initial_lambda=0.3, cycle_budget=10, model="m",
    )
    record = make_record(
        requested_a_ml=4.0, requested_b_ml=6.0, executed_a_ml=3.2, executed_b_ml=5.4, lam=0.5,
    )
    volumes = VolumeState(12.0, 34.0, 56.0)  # three distinct values, verified below too
    cid = store.start_cycle(run_id, record, volumes)

    conn = _connect(store_paths["db_path"])
    row = conn.execute(
        "SELECT requested_a_ml, requested_b_ml, executed_a_ml, executed_b_ml, "
        "lambda_at_cycle FROM cycles WHERE id = ?",
        (cid,),
    ).fetchone()
    conn.close()

    assert row["requested_a_ml"] == pytest.approx(4.0)
    assert row["requested_b_ml"] == pytest.approx(6.0)
    assert row["executed_a_ml"] == pytest.approx(3.2)  # skewed: ground truth differs from requested
    assert row["executed_b_ml"] == pytest.approx(5.4)  # channel B is skewed too -- not equal to requested
    # run's initial_lambda is 0.3 (distinct from the cycle's 0.5) -- lambda_at_cycle
    # must record the cycle's own lambda, not the run's.
    assert row["lambda_at_cycle"] == pytest.approx(0.5)

    # start_cycle writes the whole rig_state row in the same transaction (decision 3).
    loaded = store.load_volumes()
    assert loaded.cup_ml == pytest.approx(12.0)
    assert loaded.water_drawn_ml == pytest.approx(34.0)
    assert loaded.waste_ml == pytest.approx(56.0)


def test_update_and_finish_cycle_roundtrip(store, store_paths):
    run_id = store.start_run(
        target_hex="#404040", target_lab=(40.0, 0.0, 0.0),
        initial_lambda=0.4, cycle_budget=10, model="m",
    )
    # Four mutually distinct doses -- a requested<->executed swap or a wholesale
    # requested/executed swap must be visible in every one of the four columns.
    record = make_record(
        requested_a_ml=4.0, requested_b_ml=6.5, executed_a_ml=3.2, executed_b_ml=7.1
    )
    cid = store.start_cycle(run_id, record, VolumeState(10.0, 20.0, 30.0))
    assert store.load_volumes().cup_ml == pytest.approx(10.0)

    record.measured_lab = (41.0, 1.5, -2.5)
    record.blank_lab = (95.0, 0.2, -0.3)
    record.delta_e = 3.75
    record.score = 0.62
    record.phase_started = {"dose": 100.0, "measure": 105.0}
    record.phase_elapsed_s = {"dose": 0.02, "measure": 0.5}
    record.alarms = ["blank drift: 4.2 dE"]

    store.update_cycle(cid, record, VolumeState(11.5, 21.5, 31.5))

    conn = _connect(store_paths["db_path"])
    row = conn.execute("SELECT * FROM cycles WHERE id = ?", (cid,)).fetchone()
    conn.close()

    assert row["measured_l"] == pytest.approx(41.0)
    assert row["measured_a"] == pytest.approx(1.5)
    assert row["measured_b"] == pytest.approx(-2.5)
    assert row["blank_l"] == pytest.approx(95.0)
    assert row["blank_a"] == pytest.approx(0.2)
    assert row["blank_b"] == pytest.approx(-0.3)
    assert row["delta_e"] == pytest.approx(3.75)
    assert row["score"] == pytest.approx(0.62)
    assert json.loads(row["phase_started"]) == {"dose": 100.0, "measure": 105.0}
    assert json.loads(row["phase_elapsed_s"]) == {"dose": 0.02, "measure": 0.5}
    assert json.loads(row["alarms"]) == ["blank drift: 4.2 dE"]
    assert row["completed"] == 0  # update_cycle alone must not mark completion

    # update_cycle's dose-column mapping must not skew requested<->executed, and must
    # carry the cycle's own lambda (0.5), distinct from the run's initial_lambda (0.4).
    assert row["requested_a_ml"] == pytest.approx(4.0)
    assert row["requested_b_ml"] == pytest.approx(6.5)
    assert row["executed_a_ml"] == pytest.approx(3.2)
    assert row["executed_b_ml"] == pytest.approx(7.1)
    assert row["lambda_at_cycle"] == pytest.approx(0.5)

    loaded = store.load_volumes()
    assert loaded.cup_ml == pytest.approx(11.5)
    assert loaded.water_drawn_ml == pytest.approx(21.5)
    assert loaded.waste_ml == pytest.approx(31.5)

    store.finish_cycle(cid, record, VolumeState(12.5, 22.5, 32.5))

    conn = _connect(store_paths["db_path"])
    row = conn.execute(
        "SELECT completed, aborted, requested_a_ml, requested_b_ml, executed_a_ml, "
        "executed_b_ml FROM cycles WHERE id = ?",
        (cid,),
    ).fetchone()
    conn.close()
    assert row["completed"] == 1
    assert row["aborted"] is None

    # finish_cycle shares _cycle_field_values with update_cycle -- pin the same mapping
    # here too, since finish_cycle is the funnel a real cycle actually ends through.
    assert row["requested_a_ml"] == pytest.approx(4.0)
    assert row["requested_b_ml"] == pytest.approx(6.5)
    assert row["executed_a_ml"] == pytest.approx(3.2)
    assert row["executed_b_ml"] == pytest.approx(7.1)

    loaded = store.load_volumes()
    assert loaded.cup_ml == pytest.approx(12.5)
    assert loaded.water_drawn_ml == pytest.approx(22.5)
    assert loaded.waste_ml == pytest.approx(32.5)


def test_finish_cycle_writes_frames_to_disk_with_relative_paths(store, store_paths):
    run1 = store.start_run(
        target_hex="#500000", target_lab=(50.0, 0.0, 0.0),
        initial_lambda=0.5, cycle_budget=10, model="m",
    )
    volumes = VolumeState(5.0, 5.0, 5.0)
    store.start_cycle(run1, make_record(), volumes)
    store.start_cycle(run1, make_record(), volumes)

    store.end_run(run1, outcome="budget_reached")

    run2 = store.start_run(
        target_hex="#600000", target_lab=(60.0, 0.0, 0.0),
        initial_lambda=0.6, cycle_budget=10, model="m",
    )
    record = make_record()
    record.measurement_jpeg = _solid_jpeg((10, 20, 30))
    record.blank_jpeg = _solid_jpeg((200, 210, 220))
    # run_id=2, n=1 (run2's first cycle), cycle id=3 -- three distinct values so a swap
    # among them cannot hide in the assertions below.
    cid = store.start_cycle(run2, record, volumes)

    store.finish_cycle(cid, record, volumes)

    frames_dir = store_paths["frames_dir"]
    measurement_path = frames_dir / f"run_{run2}" / "cycle_1_measurement.jpg"
    blank_path = frames_dir / f"run_{run2}" / "cycle_1_blank.jpg"
    assert measurement_path.is_file()
    assert blank_path.is_file()
    Image.open(measurement_path).load()  # decodes cleanly as JPEG
    Image.open(blank_path).load()

    conn = _connect(store_paths["db_path"])
    row = conn.execute(
        "SELECT measurement_frame_path, blank_frame_path FROM cycles WHERE id = ?", (cid,)
    ).fetchone()
    conn.close()
    assert row["measurement_frame_path"] == f"run_{run2}/cycle_1_measurement.jpg"
    assert row["blank_frame_path"] == f"run_{run2}/cycle_1_blank.jpg"


def test_finish_cycle_records_aborted_reason(store, store_paths):
    run_id = store.start_run(
        target_hex="#700000", target_lab=(70.0, 0.0, 0.0),
        initial_lambda=0.7, cycle_budget=10, model="m",
    )
    record = make_record()
    record.measurement_jpeg = _solid_jpeg((5, 5, 5))  # MEASURE completed before the abort
    record.blank_jpeg = None  # BLANK never ran
    record.aborted = "watchdog: blank exceeded 1.0s"
    cid = store.start_cycle(run_id, record, VolumeState(5.0, 5.0, 5.0))

    store.finish_cycle(cid, record, VolumeState(5.0, 5.0, 5.0))

    conn = _connect(store_paths["db_path"])
    row = conn.execute(
        "SELECT completed, aborted, measurement_frame_path, blank_frame_path FROM cycles "
        "WHERE id = ?",
        (cid,),
    ).fetchone()
    conn.close()

    assert row["completed"] == 0
    assert row["aborted"] == "watchdog: blank exceeded 1.0s"
    # A frame captured before the abort is still written to disk (decision 11); the phase
    # that never ran leaves no frame at all.
    assert row["measurement_frame_path"] is not None
    assert row["blank_frame_path"] is None

    frames_dir = store_paths["frames_dir"]
    assert (frames_dir / f"run_{run_id}" / "cycle_1_measurement.jpg").is_file()
    assert not (frames_dir / f"run_{run_id}" / "cycle_1_blank.jpg").exists()


def test_close_dangling_cycles_marks_only_unfinished(store, store_paths):
    run_id = store.start_run(
        target_hex="#800000", target_lab=(80.0, 0.0, 0.0),
        initial_lambda=0.8, cycle_budget=10, model="m",
    )
    volumes = VolumeState(5.0, 5.0, 5.0)

    completed_record = make_record()
    completed_id = store.start_cycle(run_id, completed_record, volumes)
    store.finish_cycle(completed_id, completed_record, volumes)

    already_aborted_record = make_record()
    already_aborted_record.aborted = "watchdog: dose exceeded 1.0s"
    already_aborted_id = store.start_cycle(run_id, already_aborted_record, volumes)
    store.finish_cycle(already_aborted_id, already_aborted_record, volumes)

    # Never finished: this is what "the service died mid-cycle" leaves behind.
    dangling_id = store.start_cycle(run_id, make_record(), volumes)

    count = store.close_dangling_cycles("service restart mid-cycle")
    assert count == 1

    conn = _connect(store_paths["db_path"])
    rows = {
        cid: conn.execute("SELECT completed, aborted FROM cycles WHERE id = ?", (cid,)).fetchone()
        for cid in (completed_id, already_aborted_id, dangling_id)
    }
    conn.close()

    assert rows[completed_id]["completed"] == 1
    assert rows[completed_id]["aborted"] is None

    # Already-aborted rows keep their original reason -- close_dangling_cycles must not
    # overwrite a real abort reason with the generic boot-time one.
    assert rows[already_aborted_id]["completed"] == 0
    assert rows[already_aborted_id]["aborted"] == "watchdog: dose exceeded 1.0s"

    assert rows[dangling_id]["completed"] == 0
    assert rows[dangling_id]["aborted"] == "service restart mid-cycle"


def test_cycle_count(store):
    run1 = store.start_run(
        target_hex="#b10000", target_lab=(11.0, 0.0, 0.0),
        initial_lambda=0.11, cycle_budget=10, model="m",
    )
    volumes = VolumeState(5.0, 5.0, 5.0)
    store.start_cycle(run1, make_record(), volumes)
    store.start_cycle(run1, make_record(), volumes)

    store.end_run(run1, outcome="budget_reached")

    run2 = store.start_run(
        target_hex="#b20000", target_lab=(12.0, 0.0, 0.0),
        initial_lambda=0.12, cycle_budget=10, model="m",
    )
    store.start_cycle(run2, make_record(), volumes)
    store.start_cycle(run2, make_record(), volumes)
    store.start_cycle(run2, make_record(), volumes)

    assert store.cycle_count(run1) == 2
    assert store.cycle_count(run2) == 3


# --- events ---


def test_event_roundtrip_with_visibility_flag_and_null_run(store, store_paths):
    visible_id = store.add_event(
        kind="mode_change", detail={"to": "auto"}, scientist_visible=True,
        run_id=None, volumes=VolumeState(1.5, 2.5, 3.5),
    )
    assert store.load_volumes().cup_ml == pytest.approx(1.5)

    hidden_id = store.add_event(
        kind="disruption", detail={"kind": "skew", "channel": "A"}, scientist_visible=False,
        run_id=None, volumes=VolumeState(4.5, 5.5, 6.5),
    )
    # add_event writes the whole rig_state row in the same transaction (decision 3).
    loaded = store.load_volumes()
    assert loaded.cup_ml == pytest.approx(4.5)
    assert loaded.water_drawn_ml == pytest.approx(5.5)
    assert loaded.waste_ml == pytest.approx(6.5)

    conn = _connect(store_paths["db_path"])
    visible_row = conn.execute("SELECT * FROM events WHERE id = ?", (visible_id,)).fetchone()
    hidden_row = conn.execute("SELECT * FROM events WHERE id = ?", (hidden_id,)).fetchone()
    conn.close()

    assert visible_row["run_id"] is None
    assert visible_row["kind"] == "mode_change"
    assert json.loads(visible_row["detail"]) == {"to": "auto"}
    assert visible_row["scientist_visible"] == 1

    assert hidden_row["run_id"] is None
    assert json.loads(hidden_row["detail"]) == {"kind": "skew", "channel": "A"}
    assert hidden_row["scientist_visible"] == 0


def test_latest_event_detail_returns_newest_of_kind(store):
    run_id = store.start_run(
        target_hex="#900000", target_lab=(90.0, 0.0, 0.0),
        initial_lambda=0.9, cycle_budget=10, model="m",
    )
    volumes = VolumeState(5.0, 5.0, 5.0)
    store.add_event(
        kind="mode_change", detail={"to": "manual"}, scientist_visible=True,
        run_id=run_id, volumes=volumes,
    )
    store.add_event(
        kind="target_change", detail={"target_hex": "#abcabc"}, scientist_visible=True,
        run_id=run_id, volumes=volumes,
    )
    store.add_event(
        kind="mode_change", detail={"to": "auto"}, scientist_visible=True,
        run_id=run_id, volumes=volumes,
    )

    assert store.latest_event_detail(run_id, "mode_change") == {"to": "auto"}
    assert store.latest_event_detail(run_id, "target_change") == {"target_hex": "#abcabc"}
    assert store.latest_event_detail(run_id, "watchdog_trip") is None  # no event of this kind


def test_latest_event_detail_scoped_to_run(store):
    volumes = VolumeState(5.0, 5.0, 5.0)
    run1 = store.start_run(
        target_hex="#910000", target_lab=(91.0, 0.0, 0.0),
        initial_lambda=0.91, cycle_budget=10, model="m",
    )
    store.add_event(
        kind="mode_change", detail={"to": "manual"}, scientist_visible=True,
        run_id=run1, volumes=volumes,
    )
    store.end_run(run1, outcome="budget_reached")  # must end before a second run can start

    run2 = store.start_run(
        target_hex="#920000", target_lab=(92.0, 0.0, 0.0),
        initial_lambda=0.92, cycle_budget=10, model="m",
    )
    # run2's mode_change event is later (higher id) than run1's -- an unfiltered
    # "newest of kind" query would leak it into run1's lookup.
    store.add_event(
        kind="mode_change", detail={"to": "auto"}, scientist_visible=True,
        run_id=run2, volumes=volumes,
    )

    assert store.latest_event_detail(run1, "mode_change") == {"to": "manual"}
    assert store.latest_event_detail(run2, "mode_change") == {"to": "auto"}


# --- same-transaction atomicity (decision 3) ---


def test_cycle_write_tx_atomic_with_volumes(store):
    store.start_run(
        target_hex="#a10000", target_lab=(10.0, 1.0, 1.0),
        initial_lambda=0.15, cycle_budget=10, model="m",
    )
    starting = VolumeState(37.5, 120.0, 45.0)  # three distinct, nonzero values
    store.set_volumes(starting)

    with pytest.raises(StoreError):
        store.update_cycle(
            999999,  # no cycle with this id exists
            make_record(),
            VolumeState(1.0, 2.0, 3.0),  # would-be new state, distinct from `starting`
        )

    # A two-commit implementation would have written the new volumes before discovering
    # the missing cycle row; the same-transaction contract requires both or neither.
    unchanged = store.load_volumes()
    assert unchanged.cup_ml == pytest.approx(starting.cup_ml)
    assert unchanged.water_drawn_ml == pytest.approx(starting.water_drawn_ml)
    assert unchanged.waste_ml == pytest.approx(starting.waste_ml)


def test_event_tx_atomic_with_volumes(store):
    starting = VolumeState(63.0, 210.0, 88.0)
    store.set_volumes(starting)

    with pytest.raises(StoreError):
        store.add_event(
            kind="mode_change", detail={"to": "auto"}, scientist_visible=True,
            run_id=999999,  # FK violation: no such run
            volumes=VolumeState(9.0, 8.0, 7.0),
        )

    unchanged = store.load_volumes()
    assert unchanged.cup_ml == pytest.approx(starting.cup_ml)
    assert unchanged.water_drawn_ml == pytest.approx(starting.water_drawn_ml)
    assert unchanged.waste_ml == pytest.approx(starting.waste_ml)


def test_cycle_row_rolls_back_when_volume_upsert_fails(store, store_paths):
    # The two tests above force the failure *before* any causal write happens (the
    # guard fires, or the FK violation is in the same INSERT). This test forces the
    # failure *after* the causal write, in the volume upsert itself -- the only place
    # a two-commit implementation of decision 3 (commit the causal row, then commit
    # volumes separately) can be told apart from a real single transaction.
    run_id = store.start_run(
        target_hex="#c10000", target_lab=(10.0, 1.0, 1.0),
        initial_lambda=0.15, cycle_budget=10, model="m",
    )
    good_volumes = VolumeState(5.0, 5.0, 5.0)

    _break_rig_state(store)
    with pytest.raises(sqlite3.IntegrityError):
        store.start_cycle(run_id, make_record(), good_volumes)
    conn = _connect(store_paths["db_path"])
    count = conn.execute(
        "SELECT COUNT(*) AS c FROM cycles WHERE run_id = ?", (run_id,)
    ).fetchone()["c"]
    conn.close()
    # A two-commit start_cycle would have left this row behind; the same-transaction
    # contract requires it to be rolled back along with the failed volume write.
    assert count == 0

    # Repair rig_state and seed one real cycle to probe update_cycle / finish_cycle
    # the same way: their causal write is an UPDATE, so "rolled back" means "unchanged".
    _repair_rig_state(store)
    original = make_record(requested_a_ml=1.0, requested_b_ml=1.0)
    cid = store.start_cycle(run_id, original, good_volumes)

    _break_rig_state(store)
    changed = make_record(requested_a_ml=9.0, requested_b_ml=9.0)
    with pytest.raises(sqlite3.IntegrityError):
        store.update_cycle(cid, changed, good_volumes)
    conn = _connect(store_paths["db_path"])
    row = conn.execute(
        "SELECT requested_a_ml, requested_b_ml FROM cycles WHERE id = ?", (cid,)
    ).fetchone()
    conn.close()
    assert row["requested_a_ml"] == pytest.approx(1.0)
    assert row["requested_b_ml"] == pytest.approx(1.0)

    with pytest.raises(sqlite3.IntegrityError):
        store.finish_cycle(cid, changed, good_volumes)
    conn = _connect(store_paths["db_path"])
    row = conn.execute(
        "SELECT requested_a_ml, requested_b_ml, completed FROM cycles WHERE id = ?", (cid,)
    ).fetchone()
    conn.close()
    assert row["requested_a_ml"] == pytest.approx(1.0)
    assert row["requested_b_ml"] == pytest.approx(1.0)
    assert row["completed"] == 0


def test_event_row_rolls_back_when_volume_upsert_fails(store, store_paths):
    run_id = store.start_run(
        target_hex="#c20000", target_lab=(20.0, 1.0, 1.0),
        initial_lambda=0.25, cycle_budget=10, model="m",
    )
    _break_rig_state(store)
    with pytest.raises(StoreError):
        store.add_event(
            kind="mode_change", detail={"to": "auto"}, scientist_visible=True,
            run_id=run_id, volumes=VolumeState(5.0, 5.0, 5.0),
        )

    conn = _connect(store_paths["db_path"])
    count = conn.execute(
        "SELECT COUNT(*) AS c FROM events WHERE run_id = ?", (run_id,)
    ).fetchone()["c"]
    conn.close()
    # A two-commit add_event would have left this row behind.
    assert count == 0


def test_start_cycle_unknown_run_rejected(store):
    # An unknown run_id trips the FK constraint on the cycles row; start_cycle must
    # surface that as StoreError, not a raw sqlite3 exception, so callers on the store
    # seam never see storage-layer exception types. The rejected call must also leave
    # the durable volumes unchanged (decision 3).
    starting = VolumeState(40.0, 41.0, 42.0)  # three distinct, nonzero values
    store.set_volumes(starting)

    with pytest.raises(StoreError):
        store.start_cycle(
            999999,  # no run with this id exists
            make_record(),
            VolumeState(1.0, 2.0, 3.0),
        )

    unchanged = store.load_volumes()
    assert unchanged.cup_ml == pytest.approx(starting.cup_ml)
    assert unchanged.water_drawn_ml == pytest.approx(starting.water_drawn_ml)
    assert unchanged.waste_ml == pytest.approx(starting.waste_ml)


# --- guard coverage and schema constraints ---


def test_finish_cycle_unknown_id_rejected(store, store_paths):
    record = make_record()
    record.measurement_jpeg = _solid_jpeg((1, 2, 3))
    record.blank_jpeg = _solid_jpeg((4, 5, 6))

    with pytest.raises(StoreError):
        store.finish_cycle(999999, record, VolumeState(5.0, 5.0, 5.0))

    # The "no such cycle" guard fires before any frame is written to disk -- a bad id
    # must not leave orphan JPEGs behind.
    assert not store_paths["frames_dir"].exists()


def test_cycle_run_id_n_unique_constraint_enforced(store):
    run_id = store.start_run(
        target_hex="#d10000", target_lab=(30.0, 0.0, 0.0),
        initial_lambda=0.35, cycle_budget=10, model="m",
    )
    store.start_cycle(run_id, make_record(), VolumeState(5.0, 5.0, 5.0))

    # Bypassing start_cycle's own numbering to probe the schema constraint directly --
    # this is what makes "never duplicate or skip a cycle" hold by construction.
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "INSERT INTO cycles (run_id, n, requested_a_ml, requested_b_ml, executed_a_ml, "
            "executed_b_ml, lambda_at_cycle) VALUES (?, 1, 0, 0, 0, 0, 0.5)",
            (run_id,),
        )


# --- history: scientist redaction (decision 5, invariant 3) ---


def test_scientist_history_never_contains_executed_doses(store):
    run_id = store.start_run(
        target_hex="#e00001", target_lab=(50.0, 0.0, 0.0),
        initial_lambda=0.5, cycle_budget=10, model="m",
    )
    volumes = VolumeState(5.0, 5.0, 5.0)
    record = make_record(
        requested_a_ml=2.0, requested_b_ml=3.0, executed_a_ml=1.2, executed_b_ml=2.4,
    )
    cid = store.start_cycle(run_id, record, volumes)
    store.finish_cycle(cid, record, volumes)

    payload = store.history(run_id, viewer="scientist")

    # No key anywhere in the fully-walked payload may reference the executed doses --
    # a flat top-level check would miss a leak nested under "cycles"/"events"/"decisions".
    keys = list(_all_keys(payload))
    assert not any("executed" in key.lower() for key in keys)

    cycle = payload["cycles"][0]
    assert cycle["dose_a_ml"] == pytest.approx(2.0)
    assert cycle["dose_b_ml"] == pytest.approx(3.0)


def test_scientist_history_skewed_vs_unskewed_byte_identical_dose_fields(store):
    run_id = store.start_run(
        target_hex="#e00002", target_lab=(51.0, 0.0, 0.0),
        initial_lambda=0.5, cycle_budget=10, model="m",
    )
    volumes = VolumeState(5.0, 5.0, 5.0)
    unskewed = make_record(
        requested_a_ml=2.0, requested_b_ml=3.0, executed_a_ml=2.0, executed_b_ml=3.0,
    )
    skewed = make_record(
        requested_a_ml=2.0, requested_b_ml=3.0, executed_a_ml=1.4, executed_b_ml=2.1,
    )
    cid1 = store.start_cycle(run_id, unskewed, volumes)
    store.finish_cycle(cid1, unskewed, volumes)
    cid2 = store.start_cycle(run_id, skewed, volumes)
    store.finish_cycle(cid2, skewed, volumes)

    payload = store.history(run_id, viewer="scientist")
    cycles_by_n = {c["n"]: c for c in payload["cycles"]}

    def dose_fields(cycle):
        return {"dose_a_ml": cycle["dose_a_ml"], "dose_b_ml": cycle["dose_b_ml"]}

    unskewed_json = json.dumps(dose_fields(cycles_by_n[1]), sort_keys=True)
    skewed_json = json.dumps(dose_fields(cycles_by_n[2]), sort_keys=True)
    # The contract's exact sentence: identical requested doses must serialize
    # byte-identically regardless of how differently the pumps actually ran.
    assert unskewed_json == skewed_json


def test_scientist_history_excludes_hidden_events(store):
    run_id = store.start_run(
        target_hex="#e00003", target_lab=(52.0, 0.0, 0.0),
        initial_lambda=0.5, cycle_budget=10, model="m",
    )
    volumes = VolumeState(5.0, 5.0, 5.0)
    store.add_event(
        kind="mode_change", detail={"to": "auto"}, scientist_visible=True,
        run_id=run_id, volumes=volumes,
    )
    store.add_event(
        kind="disruption", detail={"channel": "A"}, scientist_visible=False,
        run_id=run_id, volumes=volumes,
    )

    payload = store.history(run_id, viewer="scientist")

    # A flipped scientist_visible comparison would let exactly the hidden event
    # through while excluding the visible one -- a visible event is seeded alongside
    # it so this test cannot pass by accident on an inverted filter.
    kinds = [event["kind"] for event in payload["events"]]
    assert "disruption" not in kinds


def test_scientist_history_includes_visible_events(store):
    run_id = store.start_run(
        target_hex="#e00004", target_lab=(53.0, 0.0, 0.0),
        initial_lambda=0.5, cycle_budget=10, model="m",
    )
    volumes = VolumeState(5.0, 5.0, 5.0)
    store.add_event(
        kind="mode_change", detail={"to": "auto"}, scientist_visible=True,
        run_id=run_id, volumes=volumes,
    )
    store.add_event(
        kind="disruption", detail={"channel": "A"}, scientist_visible=False,
        run_id=run_id, volumes=volumes,
    )

    payload = store.history(run_id, viewer="scientist")

    matches = [event for event in payload["events"] if event["kind"] == "mode_change"]
    assert len(matches) == 1
    assert matches[0]["detail"] == {"to": "auto"}


def test_scientist_history_has_no_timings_frames_or_alarms(store):
    run_id = store.start_run(
        target_hex="#e00005", target_lab=(54.0, 0.0, 0.0),
        initial_lambda=0.5, cycle_budget=10, model="m",
    )
    volumes = VolumeState(5.0, 5.0, 5.0)
    record = make_record()
    record.measurement_jpeg = _solid_jpeg((10, 20, 30))
    record.blank_jpeg = _solid_jpeg((40, 50, 60))
    record.phase_started = {"dose": 100.0, "measure": 105.0}
    record.phase_elapsed_s = {"dose": 0.02, "measure": 0.5}
    record.alarms = ["blank drift: 4.2 dE"]
    cid = store.start_cycle(run_id, record, volumes)
    store.finish_cycle(cid, record, volumes)

    payload = store.history(run_id, viewer="scientist")

    # Decision 5's derived-leak exclusions: phase timings would let the LLM derive the
    # executed dose from DOSE elapsed time; frame paths and alarms are console-only.
    keys = set(_all_keys(payload))
    assert "phase_started" not in keys
    assert "phase_elapsed_s" not in keys
    assert "measurement_frame_path" not in keys
    assert "blank_frame_path" not in keys
    assert "alarms" not in keys


def test_scientist_run_view_excludes_outcome(store):
    run_id = store.start_run(
        target_hex="#e00006", target_lab=(55.0, 0.0, 0.0),
        initial_lambda=0.5, cycle_budget=10, model="m",
    )
    store.end_run(run_id, outcome="disruption injected at cycle 4")

    scientist_payload = store.history(run_id, viewer="scientist")
    assert "outcome" not in scientist_payload["run"]

    # Ground truth is still there for console -- this isn't accidentally dropped
    # everywhere, only redacted from the scientist's view.
    console_payload = store.history(run_id, viewer="console")
    assert console_payload["run"]["outcome"] == "disruption injected at cycle 4"


def test_console_history_includes_executed_doses_hidden_events_and_paths(store, store_paths):
    run_id = store.start_run(
        target_hex="#e00007", target_lab=(56.0, 0.0, 0.0),
        initial_lambda=0.5, cycle_budget=10, model="m",
    )
    volumes = VolumeState(5.0, 5.0, 5.0)
    record = make_record(
        requested_a_ml=2.0, requested_b_ml=3.0, executed_a_ml=1.1, executed_b_ml=2.2,
    )
    record.measurement_jpeg = _solid_jpeg((1, 2, 3))
    record.blank_jpeg = _solid_jpeg((4, 5, 6))
    cid = store.start_cycle(run_id, record, volumes)
    store.finish_cycle(cid, record, volumes)
    store.add_event(
        kind="disruption", detail={"channel": "A"}, scientist_visible=False,
        run_id=run_id, volumes=volumes,
    )

    payload = store.history(run_id, viewer="console")

    cycle = payload["cycles"][0]
    assert cycle["requested_a_ml"] == pytest.approx(2.0)
    assert cycle["requested_b_ml"] == pytest.approx(3.0)
    assert cycle["executed_a_ml"] == pytest.approx(1.1)
    assert cycle["executed_b_ml"] == pytest.approx(2.2)
    assert cycle["measurement_frame_path"] == f"run_{run_id}/cycle_1_measurement.jpg"
    assert cycle["blank_frame_path"] == f"run_{run_id}/cycle_1_blank.jpg"

    hidden = [event for event in payload["events"] if event["kind"] == "disruption"]
    assert len(hidden) == 1
    assert hidden[0]["scientist_visible"] in (0, False)


def test_unknown_viewer_gets_scientist_view(store):
    run_id = store.start_run(
        target_hex="#e00008", target_lab=(57.0, 0.0, 0.0),
        initial_lambda=0.5, cycle_budget=10, model="m",
    )
    volumes = VolumeState(5.0, 5.0, 5.0)
    record = make_record(
        requested_a_ml=2.0, requested_b_ml=3.0, executed_a_ml=1.7, executed_b_ml=2.9,
    )
    cid = store.start_cycle(run_id, record, volumes)
    store.finish_cycle(cid, record, volumes)
    store.add_event(
        kind="disruption", detail={"channel": "A"}, scientist_visible=False,
        run_id=run_id, volumes=volumes,
    )

    # Only the exact string "console" earns the full view -- everything else,
    # including things that merely look like it, gets the redacted default.
    for viewer in ("Console", "", "garbage", "CONSOLE", "consolee"):
        payload = store.history(run_id, viewer=viewer)
        keys = list(_all_keys(payload))
        assert not any("executed" in key.lower() for key in keys), viewer
        kinds = [event["kind"] for event in payload["events"]]
        assert "disruption" not in kinds, viewer


def test_history_unknown_run_raises(store):
    with pytest.raises(StoreError):
        store.history(999999, viewer="console")


# --- decisions writer (T2) ---


def test_add_decision_inserts_row_with_all_fields(store):
    run_id = store.start_run(
        target_hex="#add001", target_lab=(50.0, 10.0, -5.0),
        initial_lambda=0.5, cycle_budget=10, model="m",
    )
    decision_id = store.add_decision(
        run_id=run_id,
        cycle_n=1,
        proposed_a_ml=2.5,
        proposed_b_ml=3.5,
        rationale="moving toward target",
        model_id="claude-x",
        latency_s=1.23,
    )
    assert decision_id is not None

    for viewer in ("console", "scientist"):
        payload = store.history(run_id, viewer=viewer)
        decisions = payload["decisions"]
        assert len(decisions) == 1
        decision = decisions[0]
        assert decision["id"] == decision_id
        assert decision["run_id"] == run_id
        assert decision["cycle_n"] == 1
        assert decision["proposed_a_ml"] == pytest.approx(2.5)
        assert decision["proposed_b_ml"] == pytest.approx(3.5)
        assert decision["rationale"] == "moving toward target"
        assert decision["model_id"] == "claude-x"
        assert decision["latency_s"] == pytest.approx(1.23)
        assert "at" in decision


def test_add_decision_unknown_run_raises_store_error(store):
    with pytest.raises(StoreError):
        store.add_decision(
            run_id=999999,
            cycle_n=1,
            proposed_a_ml=1.0,
            proposed_b_ml=2.0,
            rationale="orphan proposal",
            model_id="unknown",
            latency_s=0.0,
        )


def test_add_decision_does_not_touch_rig_state(store):
    run_id = store.start_run(
        target_hex="#add002", target_lab=(50.0, 10.0, -5.0),
        initial_lambda=0.5, cycle_budget=10, model="m",
    )
    store.set_volumes(VolumeState(10.0, 20.0, 30.0))
    before = store.load_volumes()

    store.add_decision(
        run_id=run_id,
        cycle_n=1,
        proposed_a_ml=1.1,
        proposed_b_ml=2.2,
        rationale="a decision is not a physical actuation",
        model_id="m",
        latency_s=0.42,
    )

    after = store.load_volumes()
    assert after == before


def test_history_includes_decisions(store):
    run_id = store.start_run(
        target_hex="#e00009", target_lab=(58.0, 0.0, 0.0),
        initial_lambda=0.5, cycle_budget=10, model="m",
    )
    store.add_decision(
        run_id=run_id,
        cycle_n=1,
        proposed_a_ml=2.5,
        proposed_b_ml=3.5,
        rationale="moving toward target",
        model_id="claude-x",
        latency_s=1.23,
    )

    for viewer in ("console", "scientist"):
        payload = store.history(run_id, viewer=viewer)
        matches = [d for d in payload["decisions"] if d["rationale"] == "moving toward target"]
        assert len(matches) == 1
        decision = matches[0]
        assert decision["cycle_n"] == 1
        assert decision["proposed_a_ml"] == pytest.approx(2.5)
        assert decision["proposed_b_ml"] == pytest.approx(3.5)
        assert decision["model_id"] == "claude-x"
        assert decision["latency_s"] == pytest.approx(1.23)


def test_history_scientist_cycle_fields_exact(store):
    volumes = VolumeState(5.0, 5.0, 5.0)

    # A first run's cycle consumes cycles.id 1 so that the run under test starts
    # its own per-run numbering ("n") at 1 while its row id is >1 -- otherwise n
    # and id would coincide and a mutation reading row["id"] instead of row["n"]
    # would go undetected.
    other_run_id = store.start_run(
        target_hex="#e0000b", target_lab=(60.0, 0.0, 0.0),
        initial_lambda=0.5, cycle_budget=10, model="m",
    )
    filler_cid = store.start_cycle(other_run_id, make_record(), volumes)
    store.finish_cycle(filler_cid, make_record(), volumes)
    store.end_run(other_run_id, outcome="match")

    run_id = store.start_run(
        target_hex="#e0000a", target_lab=(59.0, 0.0, 0.0),
        initial_lambda=0.5, cycle_budget=10, model="m",
    )
    # Every scientific value is distinct from every other -- a swap (score<->delta_e,
    # or a measured_lab/blank_lab component swap) is only observable if no two
    # seeded values are equal.
    record = make_record(lam=0.83)
    record.measurement_jpeg = _solid_jpeg((10, 20, 30))
    record.blank_jpeg = _solid_jpeg((40, 50, 60))
    record.measured_lab = (41.0, 1.5, -2.5)
    record.blank_lab = (95.0, 0.2, -0.3)
    record.delta_e = 3.75
    record.score = 0.62
    record.phase_started = {"dose": 100.0}
    record.phase_elapsed_s = {"dose": 0.02}
    record.alarms = ["blank drift: 4.2 dE"]
    cid = store.start_cycle(run_id, record, volumes)
    store.finish_cycle(cid, record, volumes)

    assert cid > 1  # this cycle's row id is not 1 -- the filler cycle took it

    payload = store.history(run_id, viewer="scientist")
    cycle = payload["cycles"][0]

    # The exact field set decision 5 names -- no accidental additions (a stray
    # "phase_elapsed_s" or "measurement_frame_path" key would fail this even if every
    # other test above happened to miss it).
    assert set(cycle.keys()) == {
        "n", "dose_a_ml", "dose_b_ml", "measured_lab", "blank_lab",
        "delta_e", "lambda", "score", "aborted",
    }

    # Value mapping -- kills n<-row["id"] (this run's n is 1, its row id is not),
    # a score<->delta_e swap, lambda reading the wrong column, the measured/blank
    # None-guard flipping to "is None", and a measured_lab/blank_lab component swap
    # ([l, a, b] -> [l, b, a]).
    assert cycle["n"] == 1
    assert cycle["n"] != cid
    assert cycle["measured_lab"] == pytest.approx([41.0, 1.5, -2.5])
    assert cycle["blank_lab"] == pytest.approx([95.0, 0.2, -0.3])
    assert cycle["delta_e"] == pytest.approx(3.75)
    assert cycle["lambda"] == pytest.approx(0.83)
    assert cycle["score"] == pytest.approx(0.62)


# --- run backup (invariant 7, Data model "Backup" steps 1-2) ---


def _seed_backup_run(store):
    run_id = store.start_run(
        target_hex="#f00001", target_lab=(70.0, 1.0, 1.0),
        initial_lambda=0.4, cycle_budget=10, model="m",
    )
    volumes = VolumeState(5.0, 5.0, 5.0)
    record = make_record(
        requested_a_ml=2.0, requested_b_ml=3.0, executed_a_ml=1.4, executed_b_ml=2.6,
    )
    record.measurement_jpeg = _solid_jpeg((10, 20, 30))
    record.blank_jpeg = _solid_jpeg((200, 210, 220))
    record.measured_lab = (41.0, 1.5, -2.5)
    record.blank_lab = (95.0, 0.2, -0.3)
    record.delta_e = 3.75
    record.score = 0.62
    cid = store.start_cycle(run_id, record, volumes)
    store.finish_cycle(cid, record, volumes)
    store.add_event(
        kind="mode_change", detail={"to": "auto"}, scientist_visible=True,
        run_id=run_id, volumes=volumes,
    )
    store.add_event(
        kind="disruption", detail={"channel": "A"}, scientist_visible=False,
        run_id=run_id, volumes=volumes,
    )
    store._conn.execute(
        "INSERT INTO decisions (run_id, cycle_n, proposed_a_ml, proposed_b_ml, rationale, "
        "model_id, latency_s, at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, 1, 2.5, 3.5, "toward target", "claude-x", 1.23, 1000.0),
    )
    store._conn.commit()
    store.end_run(run_id, outcome="converged")
    return run_id, cid


def test_backup_creates_dated_folder_with_exports(store, store_paths):
    run_id, cid = _seed_backup_run(store)

    result = store.backup_run(run_id)

    assert isinstance(result, BackupResult)
    assert result.errors == []
    assert result.folder.is_dir()
    assert result.folder.is_relative_to(store_paths["backup_root"])

    run_json = json.loads((result.folder / "run.json").read_text())
    assert run_json == store.history(run_id, viewer="console")

    with open(result.folder / "cycles.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["n"] == "1"
    assert "requested_a_ml" in rows[0]
    assert "executed_a_ml" in rows[0]
    assert len(rows) == 1

    with open(result.folder / "events.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert "kind" in rows[0]

    with open(result.folder / "decisions.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["rationale"] == "toward target"

    measurement_copy = result.folder / "frames" / f"run_{run_id}" / "cycle_1_measurement.jpg"
    blank_copy = result.folder / "frames" / f"run_{run_id}" / "cycle_1_blank.jpg"
    assert measurement_copy.is_file()
    assert blank_copy.is_file()


def test_backup_snapshot_passes_integrity_check(store, store_paths):
    run_id, cid = _seed_backup_run(store)

    result = store.backup_run(run_id)

    assert result.integrity_ok is True

    # A fresh connection onto the snapshot file -- this must be a real, independent
    # SQLite database, not merely a flag copied from the live store's own connection.
    conn = sqlite3.connect(str(result.folder / "snapshot.db"))
    row = conn.execute("PRAGMA integrity_check").fetchone()
    conn.close()
    assert row[0] == "ok"


def test_backup_export_preserves_requested_and_executed(store, store_paths):
    run_id, cid = _seed_backup_run(store)

    result = store.backup_run(run_id)

    run_json = json.loads((result.folder / "run.json").read_text())
    cycle = run_json["cycles"][0]
    # The backup is the graded deliverable -- full fidelity, never redacted.
    assert cycle["requested_a_ml"] == pytest.approx(2.0)
    assert cycle["requested_b_ml"] == pytest.approx(3.0)
    assert cycle["executed_a_ml"] == pytest.approx(1.4)
    assert cycle["executed_b_ml"] == pytest.approx(2.6)


def test_snapshot_integrity_fails_on_corrupt_file(tmp_path):
    corrupt = tmp_path / "corrupt.db"
    corrupt.write_bytes(b"not a sqlite database, just garbage bytes" * 20)

    # A vacuously-true check (e.g. "the file exists" or a swallowed exception
    # defaulting to True) would pass this file; a real integrity check must not.
    assert snapshot_integrity_ok(corrupt) is False


def test_backup_collects_errors_never_raises(store, store_paths):
    run_id, cid = _seed_backup_run(store)
    # The frames referenced by the DB rows no longer exist on disk -- backup_run must
    # collect this as an error, not raise, so a run-end backup can never crash the
    # service or block the operator from ending a run.
    shutil.rmtree(store_paths["frames_dir"])

    result = store.backup_run(run_id)

    assert isinstance(result, BackupResult)
    assert result.errors != []


# --- measurement patch persistence (T2) ---


def _legacy_rig_state_ddl():
    return """
    CREATE TABLE rig_state (
      id INTEGER PRIMARY KEY CHECK (id = 1),
      cup_ml REAL,
      water_drawn_ml REAL,
      waste_ml REAL
    );
    """


def _make_legacy_db(db_path, cup_ml=12.5, water_drawn_ml=400.0, waste_ml=80.0):
    """Create a pre-feature DB with only the original rig_state columns."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_legacy_rig_state_ddl())
    conn.execute(
        "INSERT INTO rig_state (id, cup_ml, water_drawn_ml, waste_ml) VALUES (1, ?, ?, ?)",
        (cup_ml, water_drawn_ml, waste_ml),
    )
    conn.commit()
    conn.close()


def _rig_state_column_names(conn):
    return {row["name"] for row in conn.execute("PRAGMA table_info(rig_state)")}


def test_store_migration_adds_patch_columns_to_old_schema(store_paths):
    _make_legacy_db(store_paths["db_path"])
    store = Store(**store_paths)
    conn = _connect(store_paths["db_path"])
    columns = _rig_state_column_names(conn)
    conn.close()
    store.close()
    assert "patch_cx" in columns
    assert "patch_cy" in columns
    assert "patch_size_px" in columns


def test_store_migration_preserves_existing_volume_data(store_paths):
    _make_legacy_db(store_paths["db_path"], cup_ml=12.5, water_drawn_ml=400.0, waste_ml=80.0)
    store = Store(**store_paths)
    loaded = store.load_volumes()
    store.close()
    assert loaded.cup_ml == pytest.approx(12.5)
    assert loaded.water_drawn_ml == pytest.approx(400.0)
    assert loaded.waste_ml == pytest.approx(80.0)


def test_store_migration_idempotent(store_paths):
    _make_legacy_db(store_paths["db_path"])
    first = Store(**store_paths)
    first.close()
    second = Store(**store_paths)
    conn = _connect(store_paths["db_path"])
    columns = _rig_state_column_names(conn)
    conn.close()
    second.close()
    assert "patch_cx" in columns
    assert "patch_cy" in columns
    assert "patch_size_px" in columns


def test_store_load_patch_none_on_fresh_db(store):
    assert store.load_patch() is None


def test_store_load_patch_none_on_migrated_legacy_row(store_paths):
    _make_legacy_db(store_paths["db_path"])
    store = Store(**store_paths)
    assert store.load_patch() is None
    store.close()


def test_store_set_and_load_patch_roundtrip(store):
    patch = PatchSpec(cx=0.25, cy=0.75, size_px=100)
    store.set_patch(patch)
    loaded = store.load_patch()
    assert loaded == patch


def test_store_set_patch_preserves_volumes(store):
    store.set_volumes(VolumeState(10.0, 20.0, 30.0))
    store.set_patch(PatchSpec(cx=0.1, cy=0.2, size_px=50))
    loaded = store.load_volumes()
    assert loaded.cup_ml == pytest.approx(10.0)
    assert loaded.water_drawn_ml == pytest.approx(20.0)
    assert loaded.waste_ml == pytest.approx(30.0)


def test_store_set_volumes_preserves_patch(store):
    patch = PatchSpec(cx=0.3, cy=0.6, size_px=80)
    store.set_patch(patch)
    store.set_volumes(VolumeState(15.0, 25.0, 35.0))
    loaded = store.load_patch()
    assert loaded == patch


def test_store_add_event_with_patch_persists_in_same_transaction(store, store_paths):
    patch = PatchSpec(cx=0.4, cy=0.5, size_px=60)
    event_id = store.add_event(
        kind="patch_change",
        detail={"cx": 0.4, "cy": 0.5, "size_px": 60},
        scientist_visible=True,
        run_id=None,
        volumes=VolumeState(1.0, 2.0, 3.0),
        patch=patch,
    )
    loaded = store.load_patch()
    assert loaded == patch
    conn = _connect(store_paths["db_path"])
    row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    conn.close()
    assert row is not None


def test_store_add_event_without_patch_leaves_patch_columns(store):
    patch = PatchSpec(cx=0.2, cy=0.3, size_px=70)
    store.set_patch(patch)
    store.add_event(
        kind="mode_change",
        detail={"to": "auto"},
        scientist_visible=True,
        run_id=None,
        volumes=VolumeState(5.0, 6.0, 7.0),
    )
    loaded = store.load_patch()
    assert loaded == patch


def test_event_row_rolls_back_when_patch_upsert_fails(store, store_paths):
    # Mirrors test_event_row_rolls_back_when_volume_upsert_fails, but forces the
    # failure in the PATCH upsert that runs *after* the event insert and volume
    # upsert inside add_event. A two-commit implementation would leave the event
    # row behind; the same-transaction contract requires it to roll back.
    run_id = store.start_run(
        target_hex="#c30000", target_lab=(30.0, 1.0, 1.0),
        initial_lambda=0.35, cycle_budget=10, model="m",
    )
    _break_rig_state_for_patch(store)
    bad_patch = PatchSpec(cx=0.1, cy=0.2, size_px=10)  # violates CHECK (size_px < 0)
    with pytest.raises(StoreError):
        store.add_event(
            kind="patch_change",
            detail={"cx": 0.1, "cy": 0.2, "size_px": 10},
            scientist_visible=True,
            run_id=run_id,
            volumes=VolumeState(5.0, 5.0, 5.0),
            patch=bad_patch,
        )

    conn = _connect(store_paths["db_path"])
    count = conn.execute(
        "SELECT COUNT(*) AS c FROM events WHERE run_id = ?", (run_id,)
    ).fetchone()["c"]
    conn.close()
    assert count == 0


def test_store_migration_backfills_only_missing_patch_columns(store_paths):
    # Simulate a partially-migrated legacy schema: only patch_cx exists before
    # Store.__init__ runs. Opening Store must add the remaining two columns
    # without raising a duplicate-column error.
    conn = sqlite3.connect(str(store_paths["db_path"]))
    conn.executescript(_legacy_rig_state_ddl())
    conn.execute("ALTER TABLE rig_state ADD COLUMN patch_cx REAL")
    conn.execute(
        "INSERT INTO rig_state (id, cup_ml, water_drawn_ml, waste_ml, patch_cx) "
        "VALUES (1, 12.5, 400.0, 80.0, 0.5)",
    )
    conn.commit()
    conn.close()

    store = Store(**store_paths)
    conn = _connect(store_paths["db_path"])
    columns = _rig_state_column_names(conn)
    conn.close()
    store.close()

    assert "patch_cx" in columns
    assert "patch_cy" in columns
    assert "patch_size_px" in columns


def test_store_set_and_load_patch_roundtrip_at_zero_center(store):
    # A legitimate cx/cy of 0.0 must not be mistaken for "unset" by a naive
    # falsiness check on patch_cx.
    patch = PatchSpec(cx=0.0, cy=0.0, size_px=100)
    store.set_patch(patch)
    loaded = store.load_patch()
    assert loaded is not None
    assert loaded == patch
