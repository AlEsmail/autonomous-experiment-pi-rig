# ABOUTME: SQLite store for runs, cycles, decisions, events, and the durable rig_state row
# ABOUTME: (cup + session counters + measurement patch); volume and patch writes share a transaction with their causal row.

import csv
import json
import shutil
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rig_service.cycle import CycleRecord
from rig_service.patch import PatchSpec
from rig_service.volume import VolumeState

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY,
  target_hex TEXT NOT NULL,
  target_l REAL NOT NULL, target_a REAL NOT NULL, target_b REAL NOT NULL,
  initial_lambda REAL NOT NULL,
  cycle_budget INTEGER NOT NULL,
  model TEXT NOT NULL,
  started_at REAL NOT NULL,
  ended_at REAL,
  outcome TEXT
);
CREATE TABLE IF NOT EXISTS cycles (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES runs(id),
  n INTEGER NOT NULL,
  requested_a_ml REAL NOT NULL, requested_b_ml REAL NOT NULL,
  executed_a_ml REAL NOT NULL, executed_b_ml REAL NOT NULL,
  lambda_at_cycle REAL NOT NULL,
  measured_l REAL, measured_a REAL, measured_b REAL,
  blank_l REAL, blank_a REAL, blank_b REAL,
  delta_e REAL, score REAL,
  phase_started TEXT NOT NULL DEFAULT '{}',
  phase_elapsed_s TEXT NOT NULL DEFAULT '{}',
  measurement_frame_path TEXT, blank_frame_path TEXT,
  alarms TEXT NOT NULL DEFAULT '[]',
  aborted TEXT,
  completed INTEGER NOT NULL DEFAULT 0,
  UNIQUE (run_id, n)
);
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES runs(id),
  cycle_n INTEGER NOT NULL,
  proposed_a_ml REAL NOT NULL, proposed_b_ml REAL NOT NULL,
  rationale TEXT NOT NULL,
  model_id TEXT NOT NULL,
  latency_s REAL NOT NULL,
  at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY,
  run_id INTEGER REFERENCES runs(id),
  at REAL NOT NULL,
  kind TEXT NOT NULL,
  detail TEXT NOT NULL,
  scientist_visible INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS rig_state (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  cup_ml REAL,
  water_drawn_ml REAL,
  waste_ml REAL
);
"""

_CYCLE_FIELD_COLUMNS = (
    "requested_a_ml", "requested_b_ml", "executed_a_ml", "executed_b_ml", "lambda_at_cycle",
    "measured_l", "measured_a", "measured_b", "blank_l", "blank_a", "blank_b",
    "delta_e", "score", "phase_started", "phase_elapsed_s", "alarms",
)

_RIG_STATE_PATCH_COLUMNS = (
    ("patch_cx", "REAL"),
    ("patch_cy", "REAL"),
    ("patch_size_px", "INTEGER"),
)


class StoreError(Exception):
    pass


@dataclass(frozen=True)
class BackupResult:
    folder: Path
    integrity_ok: bool
    errors: list[str]


def snapshot_integrity_ok(path: Path) -> bool:
    """Return True iff the SQLite file at ``path`` passes PRAGMA integrity_check."""
    try:
        with sqlite3.connect(str(path)) as conn:
            row = conn.execute("PRAGMA integrity_check").fetchone()
            return row is not None and row[0] == "ok"
    except sqlite3.DatabaseError:
        return False


class Store:
    def __init__(self, db_path: str | Path, frames_dir: str | Path, backup_root: str | Path) -> None:
        self._db_path = Path(db_path)
        self._frames_dir = Path(frames_dir)
        self._backup_root = Path(backup_root)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        with self._conn:
            self._conn.executescript(_SCHEMA_SQL)
            existing_columns = {
                row["name"] for row in self._conn.execute("PRAGMA table_info(rig_state)")
            }
            for name, col_type in _RIG_STATE_PATCH_COLUMNS:
                if name not in existing_columns:
                    self._conn.execute(f"ALTER TABLE rig_state ADD COLUMN {name} {col_type}")

    def close(self) -> None:
        self._conn.close()

    # --- rig_state (durable volumes) ---

    def load_volumes(self) -> VolumeState:
        row = self._conn.execute(
            "SELECT cup_ml, water_drawn_ml, waste_ml FROM rig_state WHERE id = 1"
        ).fetchone()
        if row is None:
            return VolumeState(None, None, None)
        return VolumeState(row["cup_ml"], row["water_drawn_ml"], row["waste_ml"])

    def set_volumes(self, state: VolumeState) -> None:
        with self._conn:
            self._upsert_rig_state(state)

    def _upsert_rig_state(self, state: VolumeState) -> None:
        self._conn.execute(
            "INSERT INTO rig_state (id, cup_ml, water_drawn_ml, waste_ml) VALUES (1, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET cup_ml = excluded.cup_ml, "
            "water_drawn_ml = excluded.water_drawn_ml, waste_ml = excluded.waste_ml",
            (state.cup_ml, state.water_drawn_ml, state.waste_ml),
        )

    def load_patch(self) -> PatchSpec | None:
        row = self._conn.execute(
            "SELECT patch_cx, patch_cy, patch_size_px FROM rig_state WHERE id = 1"
        ).fetchone()
        if row is None or row["patch_cx"] is None:
            return None
        return PatchSpec(
            cx=row["patch_cx"], cy=row["patch_cy"], size_px=row["patch_size_px"]
        )

    def set_patch(self, patch: PatchSpec) -> None:
        with self._conn:
            self._upsert_patch(patch)

    def _upsert_patch(self, patch: PatchSpec) -> None:
        self._conn.execute(
            "INSERT INTO rig_state (id, patch_cx, patch_cy, patch_size_px) VALUES (1, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET patch_cx = excluded.patch_cx, "
            "patch_cy = excluded.patch_cy, patch_size_px = excluded.patch_size_px",
            (patch.cx, patch.cy, patch.size_px),
        )

    # --- runs ---

    def start_run(
        self, target_hex: str, target_lab: tuple[float, float, float],
        initial_lambda: float, cycle_budget: int, model: str,
    ) -> int:
        if self.active_run() is not None:
            raise StoreError("a run is already active")
        target_l, target_a, target_b = target_lab
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO runs (target_hex, target_l, target_a, target_b, initial_lambda, "
                "cycle_budget, model, started_at, ended_at, outcome) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)",
                (target_hex, target_l, target_a, target_b, initial_lambda, cycle_budget, model, time.time()),
            )
        return cur.lastrowid

    def end_run(self, run_id: int, outcome: str | None) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE runs SET ended_at = ?, outcome = ? WHERE id = ?",
                (time.time(), outcome, run_id),
            )

    def active_run(self) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE ended_at IS NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row is not None else None

    # --- cycles ---

    def start_cycle(self, run_id: int, record: CycleRecord, volumes: VolumeState) -> int:
        with self._conn:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(n), 0) + 1 AS n FROM cycles WHERE run_id = ?", (run_id,)
            ).fetchone()
            n = row["n"]
            measured = record.measured_lab or (None, None, None)
            blank = record.blank_lab or (None, None, None)
            try:
                cur = self._conn.execute(
                    "INSERT INTO cycles ("
                    "run_id, n, requested_a_ml, requested_b_ml, executed_a_ml, executed_b_ml, "
                    "lambda_at_cycle, measured_l, measured_a, measured_b, blank_l, blank_a, blank_b, "
                    "delta_e, score, phase_started, phase_elapsed_s, alarms, aborted, completed"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                    (
                        run_id, n,
                        record.requested_a_ml, record.requested_b_ml,
                        record.executed_a_ml, record.executed_b_ml,
                        record.lam,
                        measured[0], measured[1], measured[2],
                        blank[0], blank[1], blank[2],
                        record.delta_e, record.score,
                        json.dumps(record.phase_started), json.dumps(record.phase_elapsed_s),
                        json.dumps(record.alarms), record.aborted,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise StoreError(f"failed to start cycle: {error}") from error
            self._upsert_rig_state(volumes)
        return cur.lastrowid

    def update_cycle(self, cycle_id: int, record: CycleRecord, volumes: VolumeState) -> None:
        values = self._cycle_field_values(record)
        assignments = ", ".join(f"{column} = ?" for column in _CYCLE_FIELD_COLUMNS)
        with self._conn:
            cur = self._conn.execute(
                f"UPDATE cycles SET {assignments} WHERE id = ?", values + (cycle_id,)
            )
            if cur.rowcount == 0:
                raise StoreError(f"no cycle with id {cycle_id}")
            self._upsert_rig_state(volumes)

    def finish_cycle(self, cycle_id: int, record: CycleRecord, volumes: VolumeState) -> None:
        row = self._conn.execute(
            "SELECT run_id, n FROM cycles WHERE id = ?", (cycle_id,)
        ).fetchone()
        if row is None:
            raise StoreError(f"no cycle with id {cycle_id}")
        run_id, n = row["run_id"], row["n"]

        measurement_rel = self._write_frame(run_id, n, "measurement", record.measurement_jpeg)
        blank_rel = self._write_frame(run_id, n, "blank", record.blank_jpeg)

        values = self._cycle_field_values(record)
        assignments = ", ".join(f"{column} = ?" for column in _CYCLE_FIELD_COLUMNS)
        completed = 0 if record.aborted is not None else 1
        with self._conn:
            cur = self._conn.execute(
                f"UPDATE cycles SET {assignments}, measurement_frame_path = ?, "
                "blank_frame_path = ?, aborted = ?, completed = ? WHERE id = ?",
                values + (measurement_rel, blank_rel, record.aborted, completed, cycle_id),
            )
            if cur.rowcount == 0:
                raise StoreError(f"no cycle with id {cycle_id}")
            self._upsert_rig_state(volumes)

    def close_dangling_cycles(self, reason: str) -> int:
        with self._conn:
            cur = self._conn.execute(
                "UPDATE cycles SET aborted = ? WHERE completed = 0 AND aborted IS NULL", (reason,)
            )
        return cur.rowcount

    def cycle_count(self, run_id: int) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS count FROM cycles WHERE run_id = ?", (run_id,)
        ).fetchone()
        return row["count"]

    def _cycle_field_values(self, record: CycleRecord) -> tuple:
        measured = record.measured_lab or (None, None, None)
        blank = record.blank_lab or (None, None, None)
        return (
            record.requested_a_ml, record.requested_b_ml,
            record.executed_a_ml, record.executed_b_ml,
            record.lam,
            measured[0], measured[1], measured[2],
            blank[0], blank[1], blank[2],
            record.delta_e, record.score,
            json.dumps(record.phase_started), json.dumps(record.phase_elapsed_s),
            json.dumps(record.alarms),
        )

    def _write_frame(self, run_id: int, n: int, label: str, jpeg: bytes | None) -> str | None:
        if jpeg is None:
            return None
        # Relative path uses "/" regardless of OS so a backup folder stays self-contained
        # and portable across machines; the on-disk path still goes through Path for
        # correct OS joining.
        relative = f"run_{run_id}/cycle_{n}_{label}.jpg"
        path = self._frames_dir / f"run_{run_id}" / f"cycle_{n}_{label}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(jpeg)
        return relative

    # --- events ---

    def add_event(
        self, kind: str, detail: dict, scientist_visible: bool,
        run_id: int | None, volumes: VolumeState,
        patch: PatchSpec | None = None,
    ) -> int:
        try:
            with self._conn:
                cur = self._conn.execute(
                    "INSERT INTO events (run_id, at, kind, detail, scientist_visible) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (run_id, time.time(), kind, json.dumps(detail), 1 if scientist_visible else 0),
                )
                self._upsert_rig_state(volumes)
                if patch is not None:
                    self._upsert_patch(patch)
        except sqlite3.IntegrityError as error:
            raise StoreError(f"failed to add event: {error}") from error
        return cur.lastrowid

    def latest_event_detail(self, run_id: int, kind: str) -> dict | None:
        row = self._conn.execute(
            "SELECT detail FROM events WHERE run_id = ? AND kind = ? ORDER BY id DESC LIMIT 1",
            (run_id, kind),
        ).fetchone()
        return json.loads(row["detail"]) if row is not None else None

    def add_decision(
        self,
        run_id: int,
        cycle_n: int,
        proposed_a_ml: float,
        proposed_b_ml: float,
        rationale: str,
        model_id: str,
        latency_s: float,
    ) -> int:
        """INSERT INTO decisions (...); StoreError on FK failure (unknown run).

        Unlike cycle/event writers this does NOT touch rig_state -- a decision
        changes no volume.
        """
        try:
            with self._conn:
                cur = self._conn.execute(
                    "INSERT INTO decisions (run_id, cycle_n, proposed_a_ml, proposed_b_ml, "
                    "rationale, model_id, latency_s, at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        cycle_n,
                        proposed_a_ml,
                        proposed_b_ml,
                        rationale,
                        model_id,
                        latency_s,
                        time.time(),
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise StoreError(f"failed to add decision: {error}") from error
        return cur.lastrowid

    # --- history (T3 visibility contract) ---

    def history(self, run_id: int, viewer: str) -> dict:
        """Return the run payload for ``viewer``.

        ``viewer == "console"`` returns the full-fidelity ground truth; any other
        value returns the redacted scientist view (decision 5).
        """
        run_row = self._conn.execute(
            "SELECT * FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        if run_row is None:
            raise StoreError(f"no run with id {run_id}")

        console_view = viewer == "console"
        run = dict(run_row)
        if not console_view:
            run.pop("outcome", None)

        cycles = [
            self._cycle_for_history(dict(row), console_view)
            for row in self._conn.execute(
                "SELECT * FROM cycles WHERE run_id = ? ORDER BY n", (run_id,)
            )
        ]

        if console_view:
            events = [
                self._event_for_history(dict(row))
                for row in self._conn.execute(
                    "SELECT * FROM events WHERE run_id = ? ORDER BY id", (run_id,)
                )
            ]
        else:
            events = [
                self._event_for_history(dict(row))
                for row in self._conn.execute(
                    "SELECT * FROM events WHERE run_id = ? AND scientist_visible = 1 ORDER BY id",
                    (run_id,),
                )
            ]

        decisions = [
            dict(row)
            for row in self._conn.execute(
                "SELECT * FROM decisions WHERE run_id = ? ORDER BY id", (run_id,)
            )
        ]

        return {"run": run, "cycles": cycles, "events": events, "decisions": decisions}

    def _cycle_for_history(self, row: dict, console_view: bool) -> dict:
        if console_view:
            row["phase_started"] = json.loads(row["phase_started"])
            row["phase_elapsed_s"] = json.loads(row["phase_elapsed_s"])
            row["alarms"] = json.loads(row["alarms"])
            return row

        measured_lab = None
        if row["measured_l"] is not None:
            measured_lab = [row["measured_l"], row["measured_a"], row["measured_b"]]
        blank_lab = None
        if row["blank_l"] is not None:
            blank_lab = [row["blank_l"], row["blank_a"], row["blank_b"]]
        return {
            "n": row["n"],
            "dose_a_ml": row["requested_a_ml"],
            "dose_b_ml": row["requested_b_ml"],
            "measured_lab": measured_lab,
            "blank_lab": blank_lab,
            "delta_e": row["delta_e"],
            "lambda": row["lambda_at_cycle"],
            "score": row["score"],
            "aborted": row["aborted"],
        }

    def _event_for_history(self, row: dict) -> dict:
        row["detail"] = json.loads(row["detail"])
        return row

    # --- run backup (T3) ---

    def backup_run(self, run_id: int) -> BackupResult:
        """Export a run to a dated folder under ``backup_root``.

        Writes run.json (console-view history), cycles/events/decisions CSVs, a copy of
        every referenced frame, and an integrity-checked SQLite snapshot.
        """
        folder = self._backup_root
        errors: list[str] = []
        integrity_ok = False
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            folder = self._backup_root / f"run_{run_id}_{timestamp}"
            folder.mkdir(parents=True, exist_ok=True)

            history = self.history(run_id, viewer="console")

            (folder / "run.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            self._write_csv(folder / "cycles.csv", history["cycles"])
            self._write_csv(folder / "events.csv", history["events"])
            self._write_csv(folder / "decisions.csv", history["decisions"])

            for cycle in history["cycles"]:
                for key in ("measurement_frame_path", "blank_frame_path"):
                    rel = cycle.get(key)
                    if rel:
                        src = self._frames_dir / rel
                        dest = folder / "frames" / rel
                        try:
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src, dest)
                        except Exception as error:  # noqa: BLE001
                            errors.append(f"failed to copy frame {rel}: {error}")

            snapshot = folder / "snapshot.db"
            with sqlite3.connect(str(self._db_path), isolation_level=None) as conn:
                conn.execute("VACUUM INTO ?", (str(snapshot),))
            integrity_ok = snapshot_integrity_ok(snapshot)
            if not integrity_ok:
                errors.append("snapshot integrity check failed")
        except Exception as error:  # noqa: BLE001
            errors.append(f"backup failed: {error}")
        return BackupResult(folder=folder, integrity_ok=integrity_ok, errors=errors)

    def _write_csv(self, path: Path, rows: list[dict]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fieldnames = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
