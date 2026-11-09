# ABOUTME: FastAPI app factory for rig-service -- boot-safe relay arming, startup camera lock,
# ABOUTME: mode arbitration, volume accounting, pump/calibrate/cycle/run routes, the scientist-state
# ABOUTME: liveness endpoint, durable event logging, history
# ABOUTME: visibility, and the console's frame/stream/static routes plus the live-capture lifespan.

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rig_service.camera_lock import CameraLock
from rig_service.config import RigConfig
from rig_service.cycle import CycleEngine, CycleError, CycleRecord, CycleSink, CycleValidationError, Phase
from rig_service.disrupt import DisruptError, Disruptions
from rig_service.frames import (
    LIVE_CAPTURE_INTERVAL_S,
    STREAM_BOUNDARY,
    STREAM_POLL_INTERVAL_S,
    FrameSource,
    SharedCamera,
    draw_patch_overlay,
    live_capture,
    mjpeg_stream,
)
from rig_service.mode import ModeArbiter, ModeError
from rig_service.patch import PatchHolder, PatchSpec, validate_patch
from rig_service.pumps import PumpController, PumpError
from rig_service.scientist_state import ScientistState
from rig_service.store import Store, StoreError
from rig_service.volume import VolumeAccountant, VolumeError, VolumeState
from shared.color import hex_to_rgb, rgb_to_lab

logger = logging.getLogger(__name__)


class PumpRequest(BaseModel):
    channel: str
    ml: float


class CalibrateRequest(BaseModel):
    channel: str
    seconds: float


class ModeRequest(BaseModel):
    mode: Literal["auto", "manual", "paused"]


class CycleRequest(BaseModel):
    dose_a_ml: float
    dose_b_ml: float
    rationale: str | None = None
    model_id: str | None = None
    latency_s: float | None = None


class TargetRequest(BaseModel):
    hex: str


class LambdaRequest(BaseModel):
    value: float


class PatchRequest(BaseModel):
    cx: float
    cy: float
    size_px: int


class RunStartRequest(BaseModel):
    target_hex: str
    lambda_: float = Field(alias="lambda")
    cycle_budget: int
    model: str


class RunEndRequest(BaseModel):
    outcome: str | None = None


class StagingRequest(BaseModel):
    water_remaining_ml: float
    waste_ml: float


class SkewRequest(BaseModel):
    channel: str
    factor: float


class BlipRequest(BaseModel):
    ml_water: float


class ScientistStateRequest(BaseModel):
    state: str
    detail: str = ""


def _cycle_summary(record: CycleRecord | None, viewer: str) -> dict | None:
    """Renders a CycleRecord for API payloads (the /cycle response and /status's
    cycle.last): requested doses only -- never frame bytes, never executed_* (the
    history visibility contract binds these payloads; requested-only dose fields
    and, for the scientist viewer, no phase timing)."""
    if record is None:
        return None
    result = {
        "dose_a_ml": record.requested_a_ml,
        "dose_b_ml": record.requested_b_ml,
        "lam": record.lam,
        "target_lab": record.target_lab,
        "measured_lab": record.measured_lab,
        "blank_lab": record.blank_lab,
        "delta_e": record.delta_e,
        "score": record.score,
        "alarms": record.alarms,
        "aborted": record.aborted,
    }
    if viewer == "console":
        result["phase_elapsed_s"] = record.phase_elapsed_s
    return result


class _StoreCycleSink(CycleSink):
    """Durable cycle recorder backed by the SQLite store.

    Resolves the active run id at cycle start (a 409 upstream guarantees one exists)
    and writes every phase commit and cycle finish in the same transaction as the
    accountant's current VolumeState -- invariant 5 and decision 3.
    """

    def __init__(self, store: Store, accountant: VolumeAccountant) -> None:
        self._store = store
        self._accountant = accountant

    def cycle_started(self, record: CycleRecord) -> int:
        run = self._store.active_run()
        # Upstream route guarantees an active run before the engine is invoked.
        assert run is not None, "cycle_started called with no active run"
        return self._store.start_cycle(run["id"], record, self._accountant.state)

    def phase_committed(self, cycle_id: int, phase: str, record: CycleRecord) -> None:
        self._store.update_cycle(cycle_id, record, self._accountant.state)

    def cycle_finished(self, cycle_id: int, record: CycleRecord) -> None:
        self._store.finish_cycle(cycle_id, record, self._accountant.state)


def create_app(
    config: RigConfig, driver, camera, store: Store, console_dir: Path | None = None
) -> FastAPI:
    # Volume state is rehydrated from durable storage; the accountant's on_change sink
    # writes it back. A fresh DB is unknown-everything, which the boot gate below handles.
    accountant = VolumeAccountant(
        cup_ceiling_ml=config.volumes.cup_ceiling_ml,
        reservoir_ml=config.volumes.reservoir_ml,
        waste_capacity_ml=config.volumes.waste_capacity_ml,
        low_margin_ml=config.volumes.low_margin_ml,
        state=store.load_volumes(),
        on_change=store.set_volumes,
    )

    def _current_run_id() -> int | None:
        run = store.active_run()
        return run["id"] if run is not None else None

    def _write_trip_event(reason: str) -> None:
        """Write a durable trip event with the current run + volumes.

        Called by the arbiter after mode/trip_reason are already set. A store-write
        failure degrades to a logged warning so a trip never bricks service start (W1).
        """
        try:
            store.add_event(
                kind="trip",
                detail={"reason": reason},
                scientist_visible=True,
                run_id=_current_run_id(),
                volumes=accountant.state,
            )
        except Exception as error:  # noqa: BLE001
            logger.warning("failed to write trip event: %s", error)

    # D1: every camera operation -- capture, control apply, control read -- is
    # serialized through one lock, and every successful capture publishes to the
    # frame cache (per-cycle stills exist even if the live loop dies).
    frames = FrameSource()
    shared_camera = SharedCamera(camera, frames)

    arbiter = ModeArbiter(on_trip=_write_trip_event)
    pumps = PumpController(config, driver, accountant, arbiter)
    camera_lock = CameraLock(shared_camera, config.camera.controls)
    disruptions = Disruptions(config)
    scientist_state = ScientistState()

    def _initial_patch() -> PatchSpec:
        seed = PatchSpec(*config.camera.patch_center, config.camera.patch_px)
        persisted = store.load_patch()
        if persisted is None:
            return seed
        try:
            return validate_patch(
                persisted.cx,
                persisted.cy,
                persisted.size_px,
                config.camera.width,
                config.camera.height,
            )
        except ValueError:
            logger.warning(
                "persisted patch %s no longer fits %dx%d; using config seed",
                persisted,
                config.camera.width,
                config.camera.height,
            )
            return seed

    patch_holder = PatchHolder(_initial_patch())
    engine = CycleEngine(
        config, pumps, shared_camera, camera_lock, arbiter, accountant, driver,
        _StoreCycleSink(store, accountant), disruptions,
        patch_source=lambda: patch_holder.current,
    )

    # Per-app mutable state for the console-set target/lambda and the active run.
    # A dict, not bare closure variables, so route bodies can mutate it without `nonlocal`.
    state = {"target_hex": None, "target_lab": None, "lam": 0.0}

    def _set_target(hex_value: str) -> None:
        rgb = hex_to_rgb(hex_value)
        state["target_hex"] = hex_value
        state["target_lab"] = rgb_to_lab(rgb)

    def _rehydrate_active_run() -> None:
        """Restore the active run's target/λ and skew state, then override with the latest events."""
        disruptions.clear()
        run = store.active_run()
        if run is None:
            state["target_hex"] = None
            state["target_lab"] = None
            state["lam"] = 0.0
            return
        state["target_hex"] = run["target_hex"]
        state["target_lab"] = (run["target_l"], run["target_a"], run["target_b"])
        state["lam"] = run["initial_lambda"]
        target_detail = store.latest_event_detail(run["id"], "target_change")
        if target_detail is not None:
            _set_target(target_detail["hex"])
        lambda_detail = store.latest_event_detail(run["id"], "lambda_change")
        if lambda_detail is not None:
            state["lam"] = lambda_detail["value"]
        skew_detail = store.latest_event_detail(run["id"], "skew")
        if skew_detail is not None:
            for channel, factor in skew_detail["factors"].items():
                disruptions.set_skew(channel, factor)

    def _boot_volume_gate() -> None:
        """Fail-closed boot gate (decision 9).

        A non-zero or unknown cup, or unknown session counters, trips the arbiter once
        with a reason naming every unresolved quantity. The accountant's existing alerts
        persist in /status so the operator sees the recovery action.
        """
        reasons: list[str] = []
        cup = accountant.cup_ml
        if cup is None:
            reasons.append("cup is unknown; drain before running")
        elif cup != 0.0:
            reasons.append(f"cup holds {cup} ml; drain before running")
        if accountant.water_drawn_ml is None or accountant.waste_ml is None:
            reasons.append("session counters unknown; declare staging (POST /staging)")
        if reasons:
            arbiter.trip("boot gate: " + "; ".join(reasons))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        driver.all_off()
        camera_lock.apply()  # Build Log: rig-service applies the lock set at startup.
        # D2: the live-capture loop only runs with a console to serve it, so no
        # pre-existing test or sim run grows a background task.
        capture_task = None
        if console_dir is not None:
            capture_task = asyncio.create_task(
                live_capture(shared_camera, LIVE_CAPTURE_INTERVAL_S)
            )
        yield
        if capture_task is not None:
            capture_task.cancel()
            try:
                await capture_task
            except asyncio.CancelledError:
                pass
        driver.all_off()

    app = FastAPI(title="rig-service", lifespan=lifespan)

    if console_dir is not None:
        app.mount("/static", StaticFiles(directory=console_dir), name="static")

    # Boot sequence: close any mid-cycle rows from a previous crash, rehydrate the
    # active run, then enforce the fail-closed volume gate.
    store.close_dangling_cycles("service restart mid-cycle")
    _rehydrate_active_run()
    _boot_volume_gate()

    def require_client(x_client: str | None) -> str:
        if x_client is None:
            raise HTTPException(status_code=400, detail="X-Client header required")
        return x_client

    def _active_run_or_409() -> dict:
        run = store.active_run()
        if run is None:
            raise HTTPException(status_code=409, detail="no active run")
        return run

    def _add_event(
        kind: str,
        detail: dict,
        scientist_visible: bool = True,
        patch: PatchSpec | None = None,
    ) -> None:
        """Write an event row with the accountant's current volumes in the same tx."""
        store.add_event(
            kind=kind,
            detail=detail,
            scientist_visible=scientist_visible,
            run_id=_current_run_id(),
            volumes=accountant.state,
            patch=patch,
        )

    @app.get("/status")
    async def status(x_client: str | None = Header(default=None)):
        viewer = "console" if x_client == "console" else "scientist"
        run = store.active_run()
        run_payload = None
        if run is not None:
            run_payload = {
                "id": run["id"],
                "target_hex": run["target_hex"],
                "cycle_budget": run["cycle_budget"],
                "model": run["model"],
                "cycles": store.cycle_count(run["id"]),
            }
        response = {
            "ok": True,
            "mode": arbiter.mode,
            "trip_reason": arbiter.trip_reason,
            "cycle": {
                "phase": engine.phase.value,
                "last": _cycle_summary(engine.last_record, viewer),
            },
            "target_hex": state["target_hex"],
            "lambda": state["lam"],
            "patch": {
                "cx": patch_holder.current.cx,
                "cy": patch_holder.current.cy,
                "size_px": patch_holder.current.size_px,
            },
            "run": run_payload,
        }
        if viewer == "console":
            response["relays_on"] = sorted(driver.relays_on())
            response["volumes"] = {
                "cup_ml": accountant.cup_ml,
                "water_drawn_ml": accountant.water_drawn_ml,
                "waste_ml": accountant.waste_ml,
            }
            response["alerts"] = accountant.alerts()
            response["alarms"] = accountant.alerts() + engine.alarms
            response["last_frame_age_s"] = frames.age_s
            response["scientist"] = scientist_state.snapshot()
        else:
            response["alarms"] = engine.alarms
        return response

    @app.post("/mode")
    async def set_mode(request: ModeRequest, x_client: str | None = Header(default=None)):
        client = require_client(x_client)
        try:
            arbiter.set_mode(request.mode, client)
        except ModeError as error:
            raise HTTPException(status_code=409, detail=str(error))
        if arbiter.mode == "paused":
            driver.all_off()
        _add_event("mode_change", {"mode": request.mode})
        return {"mode": arbiter.mode}

    @app.post("/pump")
    async def pump(request: PumpRequest, x_client: str | None = Header(default=None)):
        client = require_client(x_client)
        try:
            arbiter.authorize(client, "pump")
        except ModeError as error:
            raise HTTPException(status_code=409, detail=str(error))

        try:
            seconds = await pumps.dose(request.channel, request.ml)
        except PumpError as error:
            raise HTTPException(status_code=400, detail=str(error))
        except VolumeError as error:
            raise HTTPException(status_code=409, detail=str(error))
        except ModeError as error:
            raise HTTPException(status_code=409, detail=str(error))

        _add_event("pump", {"channel": request.channel, "ml": request.ml})
        return {"channel": request.channel, "ml": request.ml, "seconds": seconds}

    @app.post("/calibrate")
    async def calibrate(request: CalibrateRequest, x_client: str | None = Header(default=None)):
        client = require_client(x_client)
        try:
            arbiter.authorize(client, "calibrate")
        except ModeError as error:
            raise HTTPException(status_code=409, detail=str(error))

        try:
            est_ml = await pumps.timed_run(request.channel, request.seconds)
        except PumpError as error:
            raise HTTPException(status_code=400, detail=str(error))
        except VolumeError as error:
            raise HTTPException(status_code=409, detail=str(error))
        except ModeError as error:
            raise HTTPException(status_code=409, detail=str(error))

        _add_event(
            "calibrate",
            {"channel": request.channel, "seconds": request.seconds, "est_ml": est_ml},
        )
        return {"channel": request.channel, "seconds": request.seconds, "est_ml": est_ml}

    @app.post("/cycle")
    async def cycle(request: CycleRequest, x_client: str | None = Header(default=None)):
        client = require_client(x_client)
        try:
            arbiter.authorize(client, "cycle")
        except ModeError as error:
            raise HTTPException(status_code=409, detail=str(error))

        # Decision 6: a cycle requires an active run before any target check.
        run = _active_run_or_409()

        if state["target_lab"] is None:
            raise HTTPException(status_code=409, detail="no target set")

        if engine.phase != Phase.IDLE:
            raise HTTPException(status_code=409, detail="a cycle is already running")

        if request.rationale is not None:
            store.add_decision(
                run_id=run["id"],
                cycle_n=store.cycle_count(run["id"]) + 1,
                proposed_a_ml=request.dose_a_ml,
                proposed_b_ml=request.dose_b_ml,
                rationale=request.rationale,
                model_id=request.model_id if request.model_id is not None else "unknown",
                latency_s=request.latency_s if request.latency_s is not None else 0.0,
            )

        try:
            record = await engine.run_cycle(
                request.dose_a_ml, request.dose_b_ml, state["target_lab"], state["lam"]
            )
        except CycleValidationError as error:
            raise HTTPException(status_code=400, detail=str(error))
        except CycleError as error:
            raise HTTPException(status_code=409, detail=str(error))

        viewer = "console" if x_client == "console" else "scientist"
        return _cycle_summary(record, viewer)

    @app.post("/target")
    async def target(request: TargetRequest, x_client: str | None = Header(default=None)):
        client = require_client(x_client)
        if client != "console":
            raise HTTPException(
                status_code=409, detail="target is a console-only action, any mode"
            )
        try:
            _set_target(request.hex)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))

        _add_event("target_change", {"hex": request.hex})
        return {"target_hex": state["target_hex"]}

    @app.post("/lambda")
    async def lambda_route(request: LambdaRequest, x_client: str | None = Header(default=None)):
        client = require_client(x_client)
        if client != "console":
            raise HTTPException(
                status_code=409, detail="lambda is a console-only action, any mode"
            )
        if request.value < 0:
            raise HTTPException(status_code=400, detail="lambda must not be negative")

        state["lam"] = request.value
        _add_event("lambda_change", {"value": request.value})
        return {"lambda": state["lam"]}

    @app.post("/patch")
    async def patch_route(request: PatchRequest, x_client: str | None = Header(default=None)):
        client = require_client(x_client)
        if client != "console":
            raise HTTPException(
                status_code=409, detail="patch is a console-only action, any mode"
            )
        try:
            patch = validate_patch(
                request.cx,
                request.cy,
                request.size_px,
                config.camera.width,
                config.camera.height,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))

        patch_holder.set(patch)
        _add_event(
            "patch_change",
            {"cx": patch.cx, "cy": patch.cy, "size_px": patch.size_px},
            scientist_visible=True,
            patch=patch,
        )
        return {"cx": patch.cx, "cy": patch.cy, "size_px": patch.size_px}

    @app.post("/disrupt/skew")
    async def disrupt_skew(
        request: SkewRequest, x_client: str | None = Header(default=None)
    ):
        client = require_client(x_client)
        if client != "console":
            raise HTTPException(
                status_code=409, detail="skew is a console-only action"
            )
        _active_run_or_409()
        try:
            disruptions.set_skew(request.channel, request.factor)
        except DisruptError as error:
            raise HTTPException(status_code=400, detail=str(error))
        _add_event(
            "skew",
            {
                "channel": request.channel,
                "factor": request.factor,
                "factors": disruptions.factors,
            },
            scientist_visible=False,
        )
        return {
            "channel": request.channel,
            "factor": request.factor,
            "factors": disruptions.factors,
        }

    @app.post("/disrupt/blip")
    async def disrupt_blip(
        request: BlipRequest, x_client: str | None = Header(default=None)
    ):
        client = require_client(x_client)
        if client != "console":
            raise HTTPException(
                status_code=409, detail="blip is a console-only action"
            )
        if engine.phase != Phase.SETTLE:
            raise HTTPException(
                status_code=409, detail="blip only allowed during settle phase"
            )
        allowed = config.volumes.cup_ceiling_ml - accountant.cup_ml
        if request.ml_water > allowed:
            raise HTTPException(
                status_code=400,
                detail=f"blip exceeds headroom: {allowed:.3f} ml allowed",
            )
        try:
            seconds = await pumps.dose("W", request.ml_water)
        except PumpError as error:
            raise HTTPException(status_code=400, detail=str(error))
        except VolumeError as error:
            raise HTTPException(status_code=409, detail=str(error))
        except ModeError as error:
            raise HTTPException(status_code=409, detail=str(error))
        _add_event("blip", {"ml_water": request.ml_water}, scientist_visible=False)
        return {"ml_water": request.ml_water, "seconds": seconds}

    @app.post("/scientist/state")
    async def scientist_state_route(
        request: ScientistStateRequest, x_client: str | None = Header(default=None)
    ):
        client = require_client(x_client)
        # Scientist-only liveness report: any other client is 409.
        if client != "scientist":
            raise HTTPException(
                status_code=409, detail="scientist/state is a scientist-only action"
            )
        try:
            transitioned = scientist_state.update(request.state, request.detail)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))
        if transitioned:
            _add_event(
                "scientist_stop",
                {"state": request.state, "detail": request.detail},
                scientist_visible=True,
            )
        return scientist_state.snapshot()

    @app.post("/run/start")
    async def run_start(
        request: RunStartRequest, x_client: str | None = Header(default=None)
    ):
        client = require_client(x_client)
        if client != "console":
            raise HTTPException(
                status_code=409, detail="run/start is a console-only action, any mode"
            )
        if store.active_run() is not None:
            raise HTTPException(status_code=409, detail="a run is already active")

        try:
            _set_target(request.target_hex)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))

        try:
            run_id = store.start_run(
                target_hex=request.target_hex,
                target_lab=state["target_lab"],
                initial_lambda=request.lambda_,
                cycle_budget=request.cycle_budget,
                model=request.model,
            )
        except StoreError as error:
            raise HTTPException(status_code=409, detail=str(error))

        state["lam"] = request.lambda_
        disruptions.clear()
        _add_event(
            "run_start",
            {
                "target_hex": request.target_hex,
                "lambda": request.lambda_,
                "cycle_budget": request.cycle_budget,
                "model": request.model,
            },
        )
        return {"run_id": run_id}

    @app.post("/run/end")
    async def run_end(
        request: RunEndRequest, x_client: str | None = Header(default=None)
    ):
        client = require_client(x_client)
        if client != "console":
            raise HTTPException(
                status_code=409, detail="run/end is a console-only action, any mode"
            )
        run = _active_run_or_409()
        run_id = run["id"]
        if engine.phase != Phase.IDLE:
            raise HTTPException(
                status_code=409,
                detail="cannot end run while a cycle is in progress; pause (POST /mode {mode: paused}) to abort first",
            )
        _add_event("run_end", {"outcome": request.outcome})
        store.end_run(run_id, request.outcome)
        backup = store.backup_run(run_id)
        return {
            "run_id": run_id,
            "backup": {
                "folder": str(backup.folder),
                "integrity_ok": backup.integrity_ok,
                "errors": backup.errors,
            },
        }

    @app.post("/staging")
    async def staging(
        request: StagingRequest, x_client: str | None = Header(default=None)
    ):
        client = require_client(x_client)
        # Decision 15: console AND manual only; every wrong branch is 409.
        if client != "console":
            raise HTTPException(
                status_code=409, detail="staging is a console-only action in manual mode"
            )
        if arbiter.mode != "manual":
            raise HTTPException(
                status_code=409,
                detail="staging requires manual mode (switch to manual to restage)",
            )

        # Serialize behind any in-flight dose so a declaration cannot overwrite an
        # in-flight waste charge or water-drawn update (open question 7).
        async with pumps.lock:
            try:
                accountant.declare_staging(request.water_remaining_ml, request.waste_ml)
            except VolumeError as error:
                raise HTTPException(status_code=400, detail=str(error))

            _add_event(
                "staging",
                {
                    "water_remaining_ml": request.water_remaining_ml,
                    "waste_ml": request.waste_ml,
                },
            )

        return {
            "water_drawn_ml": accountant.water_drawn_ml,
            "waste_ml": accountant.waste_ml,
        }

    @app.get("/frame/latest")
    async def frame_latest():
        jpeg = frames.latest
        if jpeg is None:
            raise HTTPException(status_code=404, detail="no frame captured yet")
        annotated = await asyncio.to_thread(
            draw_patch_overlay, jpeg, patch_holder.current
        )
        return Response(content=annotated, media_type="image/jpeg")

    @app.get("/stream.mjpg")
    async def stream_mjpg():
        return StreamingResponse(
            mjpeg_stream(frames, patch_source=lambda: patch_holder.current),
            media_type=f"multipart/x-mixed-replace; boundary={STREAM_BOUNDARY}",
        )

    @app.get("/operate")
    async def operate_view():
        if console_dir is None:
            raise HTTPException(status_code=404, detail="console not configured")
        return FileResponse(console_dir / "operate.html")

    @app.get("/wall")
    async def wall_view():
        if console_dir is None:
            raise HTTPException(status_code=404, detail="console not configured")
        return FileResponse(console_dir / "wall.html")

    @app.get("/")
    async def root():
        return RedirectResponse(url="/operate", status_code=307)

    @app.get("/history")
    async def history(
        run: int, x_client: str | None = Header(default=None)
    ):
        require_client(x_client)
        viewer = "console" if x_client == "console" else "scientist"
        try:
            return store.history(run, viewer)
        except StoreError as error:
            raise HTTPException(status_code=404, detail=str(error))

    return app
