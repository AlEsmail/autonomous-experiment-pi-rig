# ABOUTME: Loads and validates rig configuration from JSON: channel map (GPIO, role,
# ABOUTME: calibration, caps), volume limits, camera lock set, and cycle parameters.

import json
from dataclasses import dataclass
from pathlib import Path

from rig_service.patch import validate_patch
from shared.dose import seconds_for_dose

ROLES = frozenset({"dye", "water", "drain"})

PHASE_BUDGET_KEYS = frozenset({
    "dose", "refill", "settle", "measure", "drain", "rinse", "blank", "final_drain",
})

REQUIRED_LOCK_CONTROLS = frozenset({
    "auto_exposure", "exposure_time_absolute",
    "white_balance_automatic", "white_balance_temperature",
    "focus_automatic_continuous", "focus_absolute",
})


@dataclass(frozen=True)
class Channel:
    gpio: int
    role: str
    rate_ml_per_s: float
    lag_s: float
    max_ml_per_dose: float


@dataclass(frozen=True)
class VolumeLimits:
    cup_ceiling_ml: float
    reservoir_ml: float
    waste_capacity_ml: float
    low_margin_ml: float


@dataclass(frozen=True)
class CameraConfig:
    device: str
    width: int
    height: int
    patch_px: int
    controls: dict[str, int]
    patch_center: tuple[float, float] = (0.5, 0.5)


@dataclass(frozen=True)
class CycleConfig:
    refill_ml: float
    rinse_ml: float
    drain_ml: float
    settle_s: float
    blank_drift_delta_e: float
    phase_budgets_s: dict[str, float]


@dataclass(frozen=True)
class RigConfig:
    channels: dict[str, Channel]
    max_relay_on_s: float
    volumes: VolumeLimits
    camera: CameraConfig
    cycle: CycleConfig

    def __post_init__(self) -> None:
        # The budget trip fires when remaining headroom <= low_margin_ml, and only
        # after a dose completes. Two conditions make that sufficient: no single
        # budget-reaching dose may exceed the margin, and boot headroom must start
        # above the margin — otherwise a dose can cross a capacity limit before
        # the pause fires.
        largest_budget_dose = max(
            (c.max_ml_per_dose for c in self.channels.values() if c.role in ("water", "drain")),
            default=0.0,
        )
        if self.volumes.low_margin_ml < largest_budget_dose:
            raise ValueError(
                f"low_margin_ml ({self.volumes.low_margin_ml}) must be at least the "
                f"largest water/drain max_ml_per_dose ({largest_budget_dose})"
            )
        if self.volumes.reservoir_ml <= self.volumes.low_margin_ml:
            raise ValueError(
                f"reservoir_ml ({self.volumes.reservoir_ml}) must exceed "
                f"low_margin_ml ({self.volumes.low_margin_ml})"
            )
        if self.volumes.waste_capacity_ml <= self.volumes.low_margin_ml:
            raise ValueError(
                f"waste_capacity_ml ({self.volumes.waste_capacity_ml}) must exceed "
                f"low_margin_ml ({self.volumes.low_margin_ml})"
            )

        dye_channels = [c for c in self.channels.values() if c.role == "dye"]
        water_channel = next((c for c in self.channels.values() if c.role == "water"), None)
        drain_channel = next((c for c in self.channels.values() if c.role == "drain"), None)

        if self.cycle.refill_ml <= 0:
            raise ValueError(f"cycle.refill_ml ({self.cycle.refill_ml}) must be positive")
        if self.cycle.rinse_ml <= 0:
            raise ValueError(f"cycle.rinse_ml ({self.cycle.rinse_ml}) must be positive")
        if self.cycle.settle_s < 0:
            raise ValueError(f"cycle.settle_s ({self.cycle.settle_s}) must not be negative")

        if self.cycle.settle_s >= self.cycle.phase_budgets_s["settle"]:
            raise ValueError(
                f"settle_s ({self.cycle.settle_s}) must be strictly less than the settle "
                f"phase budget ({self.cycle.phase_budgets_s['settle']})"
            )

        if water_channel is not None:
            if self.cycle.refill_ml > water_channel.max_ml_per_dose:
                raise ValueError(
                    f"cycle.refill_ml ({self.cycle.refill_ml}) exceeds the water channel's "
                    f"max_ml_per_dose cap ({water_channel.max_ml_per_dose})"
                )
            refill_on_time_s = seconds_for_dose(
                self.cycle.refill_ml, water_channel.rate_ml_per_s, water_channel.lag_s
            )
            if refill_on_time_s > self.max_relay_on_s:
                raise ValueError(
                    f"cycle.refill_ml ({self.cycle.refill_ml}) needs {refill_on_time_s} s, "
                    f"exceeds relay on-time cap of {self.max_relay_on_s} s"
                )
            if self.cycle.rinse_ml > water_channel.max_ml_per_dose:
                raise ValueError(
                    f"cycle.rinse_ml ({self.cycle.rinse_ml}) exceeds the water channel's "
                    f"max_ml_per_dose cap ({water_channel.max_ml_per_dose})"
                )
            rinse_on_time_s = seconds_for_dose(
                self.cycle.rinse_ml, water_channel.rate_ml_per_s, water_channel.lag_s
            )
            if rinse_on_time_s > self.max_relay_on_s:
                raise ValueError(
                    f"cycle.rinse_ml ({self.cycle.rinse_ml}) needs {rinse_on_time_s} s, "
                    f"exceeds relay on-time cap of {self.max_relay_on_s} s"
                )

        if self.cycle.drain_ml < self.volumes.cup_ceiling_ml:
            raise ValueError(
                f"cycle.drain_ml ({self.cycle.drain_ml}) must be at least cup_ceiling_ml "
                f"({self.volumes.cup_ceiling_ml}) so a full drain always empties the cup"
            )
        if drain_channel is not None:
            if self.cycle.drain_ml > drain_channel.max_ml_per_dose:
                raise ValueError(
                    f"cycle.drain_ml ({self.cycle.drain_ml}) exceeds the drain channel's "
                    f"max_ml_per_dose cap ({drain_channel.max_ml_per_dose})"
                )
            drain_on_time_s = seconds_for_dose(
                self.cycle.drain_ml, drain_channel.rate_ml_per_s, drain_channel.lag_s
            )
            if drain_on_time_s > self.max_relay_on_s:
                raise ValueError(
                    f"cycle.drain_ml ({self.cycle.drain_ml}) needs {drain_on_time_s} s, "
                    f"exceeds relay on-time cap of {self.max_relay_on_s} s"
                )

        worst_case_fill_ml = self.cycle.refill_ml + sum(c.max_ml_per_dose for c in dye_channels)
        if worst_case_fill_ml > self.volumes.cup_ceiling_ml:
            raise ValueError(
                f"cycle.refill_ml plus worst-case dye doses ({worst_case_fill_ml}) exceeds "
                f"cup_ceiling_ml ({self.volumes.cup_ceiling_ml})"
            )

        if self.cycle.rinse_ml > self.volumes.cup_ceiling_ml:
            raise ValueError(
                f"cycle.rinse_ml ({self.cycle.rinse_ml}) exceeds cup_ceiling_ml "
                f"({self.volumes.cup_ceiling_ml}) — RINSE fills a drained cup to rinse_ml alone"
            )

        if dye_channels:
            dose_worst_case_s = sum(
                seconds_for_dose(c.max_ml_per_dose, c.rate_ml_per_s, c.lag_s)
                for c in dye_channels
            )
            if self.cycle.phase_budgets_s["dose"] < dose_worst_case_s:
                raise ValueError(
                    f"phase_budgets_s['dose'] ({self.cycle.phase_budgets_s['dose']}) is below "
                    f"the worst-case dose pump time ({dose_worst_case_s})"
                )

        if water_channel is not None:
            refill_worst_case_s = seconds_for_dose(
                self.cycle.refill_ml, water_channel.rate_ml_per_s, water_channel.lag_s
            )
            if self.cycle.phase_budgets_s["refill"] < refill_worst_case_s:
                raise ValueError(
                    f"phase_budgets_s['refill'] ({self.cycle.phase_budgets_s['refill']}) is "
                    f"below the worst-case refill pump time ({refill_worst_case_s})"
                )
            rinse_worst_case_s = seconds_for_dose(
                self.cycle.rinse_ml, water_channel.rate_ml_per_s, water_channel.lag_s
            )
            if self.cycle.phase_budgets_s["rinse"] < rinse_worst_case_s:
                raise ValueError(
                    f"phase_budgets_s['rinse'] ({self.cycle.phase_budgets_s['rinse']}) is "
                    f"below the worst-case rinse pump time ({rinse_worst_case_s})"
                )

        if drain_channel is not None:
            drain_worst_case_s = seconds_for_dose(
                self.cycle.drain_ml, drain_channel.rate_ml_per_s, drain_channel.lag_s
            )
            for phase in ("drain", "final_drain"):
                if self.cycle.phase_budgets_s[phase] < drain_worst_case_s:
                    raise ValueError(
                        f"phase_budgets_s[{phase!r}] ({self.cycle.phase_budgets_s[phase]}) is "
                        f"below the worst-case drain pump time ({drain_worst_case_s})"
                    )

    def all_gpios(self) -> list[int]:
        return [channel.gpio for channel in self.channels.values()]


def load_config(path: str | Path) -> RigConfig:
    data = json.loads(Path(path).read_text())

    raw_channels = data.get("channels")
    if not raw_channels:
        raise ValueError("config must define at least one channel")

    channels: dict[str, Channel] = {}
    for name, raw in raw_channels.items():
        try:
            channels[name] = Channel(
                gpio=int(raw["gpio"]),
                role=str(raw["role"]),
                rate_ml_per_s=float(raw["rate_ml_per_s"]),
                lag_s=float(raw["lag_s"]),
                max_ml_per_dose=float(raw["max_ml_per_dose"]),
            )
        except KeyError as missing:
            raise ValueError(f"channel {name!r} is missing field {missing}") from None
        if channels[name].role not in ROLES:
            raise ValueError(
                f"channel {name!r} has unknown role {channels[name].role!r}; "
                f"valid: {sorted(ROLES)}"
            )

    gpios = [c.gpio for c in channels.values()]
    if len(gpios) != len(set(gpios)):
        raise ValueError(f"duplicate gpio in channel map: {sorted(gpios)}")

    try:
        max_relay_on_s = float(data["caps"]["max_relay_on_s"])
    except KeyError as missing:
        raise ValueError(f"caps section is missing field {missing}") from None

    raw_volumes = data.get("volumes")
    if raw_volumes is None:
        raise ValueError("config must define a volumes section")
    try:
        volumes = VolumeLimits(
            cup_ceiling_ml=float(raw_volumes["cup_ceiling_ml"]),
            reservoir_ml=float(raw_volumes["reservoir_ml"]),
            waste_capacity_ml=float(raw_volumes["waste_capacity_ml"]),
            low_margin_ml=float(raw_volumes["low_margin_ml"]),
        )
    except KeyError as missing:
        raise ValueError(f"volumes section is missing field {missing}") from None

    raw_camera = data.get("camera")
    if raw_camera is None:
        raise ValueError("config must define a camera section")
    try:
        raw_controls = raw_camera["controls"]
    except KeyError as missing:
        raise ValueError(f"camera section is missing field {missing}") from None
    if not isinstance(raw_controls, dict):
        raise ValueError("camera controls must be a mapping of control name to value")
    missing_controls = REQUIRED_LOCK_CONTROLS - raw_controls.keys()
    if missing_controls:
        raise ValueError(
            f"camera controls missing required lock control(s): {sorted(missing_controls)}"
        )
    raw_patch_center = raw_camera.get("patch_center", [0.5, 0.5])
    if not isinstance(raw_patch_center, (list, tuple)) or len(raw_patch_center) != 2:
        raise ValueError("camera.patch_center must be a list of two numbers")
    try:
        patch_center = (float(raw_patch_center[0]), float(raw_patch_center[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError("camera.patch_center must be a list of two numbers") from exc

    try:
        camera = CameraConfig(
            device=str(raw_camera["device"]),
            width=int(raw_camera["width"]),
            height=int(raw_camera["height"]),
            patch_px=int(raw_camera["patch_px"]),
            patch_center=patch_center,
            controls=dict(raw_controls),
        )
    except KeyError as missing:
        raise ValueError(f"camera section is missing field {missing}") from None
    if camera.patch_px <= 0:
        raise ValueError(f"camera patch_px ({camera.patch_px}) must be positive")

    # The config seed must fit its own frame at load time (D2).
    validate_patch(camera.patch_center[0], camera.patch_center[1], camera.patch_px, camera.width, camera.height)

    raw_cycle = data.get("cycle")
    if raw_cycle is None:
        raise ValueError("config must define a cycle section")
    try:
        raw_budgets = raw_cycle["phase_budgets_s"]
    except KeyError as missing:
        raise ValueError(f"cycle section is missing field {missing}") from None

    budget_keys = set(raw_budgets.keys())
    missing_budget_keys = PHASE_BUDGET_KEYS - budget_keys
    if missing_budget_keys:
        raise ValueError(f"cycle phase_budgets_s missing phase(s): {sorted(missing_budget_keys)}")
    unknown_budget_keys = budget_keys - PHASE_BUDGET_KEYS
    if unknown_budget_keys:
        raise ValueError(f"cycle phase_budgets_s has unknown phase(s): {sorted(unknown_budget_keys)}")

    phase_budgets_s: dict[str, float] = {}
    for key, value in raw_budgets.items():
        budget = float(value)
        if budget <= 0:
            raise ValueError(f"cycle phase_budgets_s[{key!r}] ({budget}) must be positive")
        phase_budgets_s[key] = budget

    try:
        cycle = CycleConfig(
            refill_ml=float(raw_cycle["refill_ml"]),
            rinse_ml=float(raw_cycle["rinse_ml"]),
            drain_ml=float(raw_cycle["drain_ml"]),
            settle_s=float(raw_cycle["settle_s"]),
            blank_drift_delta_e=float(raw_cycle["blank_drift_delta_e"]),
            phase_budgets_s=phase_budgets_s,
        )
    except KeyError as missing:
        raise ValueError(f"cycle section is missing field {missing}") from None
    if cycle.blank_drift_delta_e <= 0:
        raise ValueError(
            f"cycle blank_drift_delta_e ({cycle.blank_drift_delta_e}) must be positive"
        )

    return RigConfig(
        channels=channels,
        max_relay_on_s=max_relay_on_s,
        volumes=volumes,
        camera=camera,
        cycle=cycle,
    )
