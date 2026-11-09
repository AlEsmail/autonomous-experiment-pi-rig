# ABOUTME: CycleEngine: the full-cycle state machine (dose -> refill -> settle -> measure ->
# ABOUTME: drain -> rinse+blank -> drain), its preflight gate, and the per-phase watchdog/abort path.

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol

from rig_service.camera_lock import CameraLock, blank_drift_alarm
from rig_service.config import RigConfig
from rig_service.disrupt import Disruptions
from rig_service.mode import ModeArbiter
from rig_service.patch import PatchSpec
from rig_service.pumps import PumpController, PumpError
from rig_service.volume import VolumeAccountant
from shared import color
from shared.measure import lab_from_jpeg


class Phase(str, Enum):
    IDLE = "idle"
    DOSE = "dose"
    REFILL = "refill"
    SETTLE = "settle"
    MEASURE = "measure"
    DRAIN = "drain"
    RINSE = "rinse"
    BLANK = "blank"
    FINAL_DRAIN = "final_drain"


@dataclass
class CycleRecord:
    requested_a_ml: float
    requested_b_ml: float
    executed_a_ml: float
    executed_b_ml: float
    lam: float
    target_lab: tuple[float, float, float]
    measured_lab: tuple[float, float, float] | None = None
    blank_lab: tuple[float, float, float] | None = None
    delta_e: float | None = None
    score: float | None = None
    phase_started: dict[str, float] = field(default_factory=dict)  # phase -> epoch s
    phase_elapsed_s: dict[str, float] = field(default_factory=dict)
    measurement_jpeg: bytes | None = None
    blank_jpeg: bytes | None = None
    alarms: list[str] = field(default_factory=list)
    aborted: str | None = None  # None == clean cycle


class CycleError(Exception):
    """Preflight rejection / busy / no target -- nothing has moved a relay."""


class CycleValidationError(CycleError):
    """A pure validation fault (negative dose, over-cap) -- nothing has moved a relay."""


class CycleSink(Protocol):
    """Durable cycle recorder. Every call is a load-bearing safety seam: a sink that
    cannot record must stop the cycle rather than let the rig dose without a trail."""

    def cycle_started(self, record: CycleRecord) -> int:
        """Persist the cycle row with requested + executed doses BEFORE any relay fires.

        Returns the durable cycle id that later phase commits and cycle_finished use.
        """
        ...

    def phase_committed(self, cycle_id: int, phase: str, record: CycleRecord) -> None:
        """Mark a phase as durably committed before the next phase begins."""
        ...

    def cycle_finished(self, cycle_id: int, record: CycleRecord) -> None:
        """Final commit on clean completion or on every abort path."""
        ...


class CycleEngine:
    def __init__(
        self,
        config: RigConfig,
        pumps: PumpController,
        camera,
        camera_lock: CameraLock,
        arbiter: ModeArbiter,
        accountant: VolumeAccountant,
        driver,
        sink: CycleSink,
        disruptions: Disruptions,
        patch_source: Callable[[], PatchSpec],
    ) -> None:
        self._config = config
        self._pumps = pumps
        self._camera = camera
        self._camera_lock = camera_lock
        self._arbiter = arbiter
        self._accountant = accountant
        self._driver = driver
        self._sink = sink
        self._disruptions = disruptions
        self._patch_source = patch_source
        self._phase = Phase.IDLE
        self._last_record: CycleRecord | None = None
        self._alarms: list[str] = []
        self._reference_blank_lab: tuple[float, float, float] | None = None

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def last_record(self) -> CycleRecord | None:
        return self._last_record

    @property
    def alarms(self) -> list[str]:
        return self._alarms

    async def run_cycle(
        self,
        dose_a_ml: float,
        dose_b_ml: float,
        target_lab: tuple[float, float, float],
        lam: float,
    ) -> CycleRecord:
        self._preflight(dose_a_ml, dose_b_ml)

        # Per-cycle snapshot: one patch value for both measure and blank, fetched after
        # preflight and before the first phase runs (D3).
        patch = self._patch_source()

        record = CycleRecord(
            requested_a_ml=dose_a_ml,
            requested_b_ml=dose_b_ml,
            executed_a_ml=self._disruptions.executed_ml("A", dose_a_ml),
            executed_b_ml=self._disruptions.executed_ml("B", dose_b_ml),
            lam=lam,
            target_lab=target_lab,
        )
        # Published before the first phase runs -- invariant 5: a probe reading
        # last_record mid-cycle must already find the doses on it.
        self._last_record = record

        # Invariant 5 (durable): the cycle row with requested + executed doses must be
        # committed BEFORE any relay fires. A sink failure at this seam is a pre-actuation
        # fault: raise CycleError and leave mode/relays untouched.
        try:
            cycle_id = self._sink.cycle_started(record)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise CycleError(f"sink failed to start cycle: {error}") from error

        async def dose_body():
            if record.executed_a_ml > 0:
                await self._pumps.dose("A", record.executed_a_ml)
            if record.executed_b_ml > 0:
                await self._pumps.dose("B", record.executed_b_ml)

        async def refill_body():
            await self._pumps.dose("W", self._config.cycle.refill_ml)

        async def settle_body():
            await asyncio.sleep(self._config.cycle.settle_s)

        async def measure_body():
            jpeg = await asyncio.to_thread(self._camera.capture_jpeg)
            measured_lab = lab_from_jpeg(jpeg, patch.cx, patch.cy, patch.size_px)
            record.measurement_jpeg = jpeg
            record.measured_lab = measured_lab
            record.delta_e = color.delta_e(measured_lab, target_lab)
            record.score = color.score(record.delta_e, dose_a_ml + dose_b_ml, lam)

        async def drain_body():
            await self._pumps.dose("D", self._config.cycle.drain_ml)

        async def rinse_body():
            await self._pumps.dose("W", self._config.cycle.rinse_ml)

        async def blank_body():
            record.alarms += await asyncio.to_thread(self._camera_lock.verify)
            jpeg = await asyncio.to_thread(self._camera.capture_jpeg)
            blank_lab = lab_from_jpeg(jpeg, patch.cx, patch.cy, patch.size_px)
            record.blank_jpeg = jpeg
            record.blank_lab = blank_lab
            if self._reference_blank_lab is None:
                self._reference_blank_lab = blank_lab
            drift_message = blank_drift_alarm(
                blank_lab, self._reference_blank_lab, self._config.cycle.blank_drift_delta_e
            )
            if drift_message is not None:
                record.alarms.append(drift_message)
            self._alarms = record.alarms

        async def final_drain_body():
            await self._pumps.dose("D", self._config.cycle.drain_ml)

        phases = (
            (Phase.DOSE, dose_body),
            (Phase.REFILL, refill_body),
            (Phase.SETTLE, settle_body),
            (Phase.MEASURE, measure_body),
            (Phase.DRAIN, drain_body),
            (Phase.RINSE, rinse_body),
            (Phase.BLANK, blank_body),
            (Phase.FINAL_DRAIN, final_drain_body),
        )
        for phase, body in phases:
            if not await self._run_phase(phase, body, record, cycle_id):
                return record

        self._sink.cycle_finished(cycle_id, record)
        self._phase = Phase.IDLE
        return record

    def _preflight(self, dose_a_ml: float, dose_b_ml: float) -> None:
        if self._phase != Phase.IDLE:
            raise CycleError("a cycle is already running")

        # Unknown cup is rejected FIRST, before the non-empty check: the spec treats
        # unknown as non-empty, and the operator needs a distinct "unknown" message.
        if self._accountant.cup_ml is None:
            raise CycleError(
                "cup volume is unknown; drain the cup via POST /pump (manual) before running"
            )
        if self._accountant.cup_ml != 0:
            raise CycleError(
                f"cup is not empty ({self._accountant.cup_ml} ml); a cycle must start drained"
            )

        if dose_a_ml < 0 or dose_b_ml < 0:
            raise CycleValidationError(
                f"doses must not be negative (dose_a_ml={dose_a_ml}, dose_b_ml={dose_b_ml})"
            )

        cycle = self._config.cycle
        try:
            if dose_a_ml > 0:
                self._pumps.validate("A", dose_a_ml)
            if dose_b_ml > 0:
                self._pumps.validate("B", dose_b_ml)
            self._pumps.validate("W", cycle.refill_ml)
            self._pumps.validate("D", cycle.drain_ml)
            self._pumps.validate("W", cycle.rinse_ml)
            self._pumps.validate("D", cycle.drain_ml)
        except PumpError as error:
            raise CycleValidationError(f"preflight rejected: {error}") from error

        # Unknown session counters reject dosing before the headroom arithmetic, which
        # assumes known counters (decision 15).
        if (
            self._accountant.water_drawn_ml is None
            or self._accountant.waste_ml is None
        ):
            raise CycleError(
                "session water/waste counters are unknown; declare staging via POST /staging"
            )

        self._check_session_headroom(dose_a_ml, dose_b_ml)

    def _check_session_headroom(self, dose_a_ml: float, dose_b_ml: float) -> None:
        volumes = self._config.volumes
        cycle = self._config.cycle
        planned_water_ml = cycle.refill_ml + cycle.rinse_ml
        planned_waste_ml = dose_a_ml + dose_b_ml + planned_water_ml

        reservoir_headroom = volumes.reservoir_ml - self._accountant.water_drawn_ml
        waste_headroom = volumes.waste_capacity_ml - self._accountant.waste_ml

        constrained = []
        if reservoir_headroom - planned_water_ml <= volumes.low_margin_ml:
            constrained.append("reservoir")
        if waste_headroom - planned_waste_ml <= volumes.low_margin_ml:
            constrained.append("waste")

        if constrained:
            reason = (
                "budget: low session headroom for " + " and ".join(constrained) +
                " -- rejecting cycle before any liquid moves"
            )
            self._driver.all_off()
            self._arbiter.trip(reason)
            raise CycleError(reason)

    async def _run_phase(
        self, phase: Phase, body, record: CycleRecord, cycle_id: int
    ) -> bool:
        # Re-check mode at every phase boundary -- an operator pause or a budget trip
        # fired mid-cycle must stop the cycle before the next phase starts.
        if self._arbiter.mode == "paused":
            self._abort(record, f"mode paused before phase {phase.value}", cycle_id)
            return False

        self._phase = phase
        budget = self._config.cycle.phase_budgets_s[phase.value]
        record.phase_started[phase.value] = time.time()
        start = time.monotonic()
        try:
            await asyncio.wait_for(body(), timeout=budget)
        except asyncio.TimeoutError:
            self._abort(record, f"watchdog: {phase.value} exceeded {budget}s", cycle_id)
            return False
        except asyncio.CancelledError:
            # CancelledError is a BaseException, not caught by "except Exception" below --
            # a cancelled cycle still needs relays off, the arbiter tripped, and _phase
            # reset to IDLE before cancellation re-propagates, or the engine sticks
            # non-IDLE and _preflight rejects every later cycle until a restart.
            self._abort(record, f"cancelled during phase {phase.value}", cycle_id)
            raise
        except Exception as error:
            self._abort(record, f"{phase.value}: {error}", cycle_id)
            return False
        finally:
            record.phase_elapsed_s[phase.value] = time.monotonic() - start

        # Commit the phase durably BEFORE the next phase begins. A sink failure here is
        # treated as a phase-body failure: a rig that cannot record must not keep dosing.
        try:
            self._sink.phase_committed(cycle_id, phase.value, record)
        except Exception as error:
            self._abort(record, f"sink: {error}", cycle_id)
            return False

        return True

    def _abort(self, record: CycleRecord, reason: str, cycle_id: int) -> None:
        # One abort path for watchdog timeouts, phase-body exceptions, cancellation, and
        # sink failures -- never a bare exception escaping as a 500 with an ambiguous rig
        # state (cancellation itself still propagates to the caller by design; only the rig
        # state cleanup is shared here).
        self._driver.all_off()
        # If the arbiter is already paused (e.g. a budget/reservoir trip fired mid-phase),
        # keep that root-cause trip_reason -- don't clobber it with this phase's generic
        # "mode paused" symptom. The record still names the proximate phase+cause.
        if self._arbiter.mode != "paused":
            self._arbiter.trip(reason)
        record.aborted = reason
        self._phase = Phase.IDLE
        # Every abort -- including cancellation and sink failure -- finalizes the durable
        # record so the decision trail is complete even when the cycle does not finish.
        self._sink.cycle_finished(cycle_id, record)
