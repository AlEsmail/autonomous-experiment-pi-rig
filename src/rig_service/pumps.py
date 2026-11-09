# ABOUTME: PumpController: the one capped, volume-accounted, serialized dosing path shared
# ABOUTME: by /pump, /calibrate, and the cycle engine, including the paused-floor safety gate.

import asyncio

from rig_service.config import RigConfig
from rig_service.mode import ModeArbiter, ModeError
from rig_service.volume import VolumeAccountant, VolumeError
from shared.dose import seconds_for_dose


class PumpError(Exception):
    pass


class PumpController:
    def __init__(
        self,
        config: RigConfig,
        driver,
        accountant: VolumeAccountant,
        arbiter: ModeArbiter,
    ) -> None:
        self._config = config
        self._driver = driver
        self._accountant = accountant
        self._arbiter = arbiter
        # Coil power feeds one relay at a time (Pi 5V rail): pump actions serialize.
        self._lock = asyncio.Lock()

    @property
    def lock(self) -> asyncio.Lock:
        """Read-only accessor for the coil-serialization lock.

        Callers (e.g. POST /staging) may acquire this lock to serialize vessel
        declarations behind any in-flight dose, preventing a staging declaration
        from overwriting an in-flight waste charge or water-drawn update.
        """
        return self._lock

    def validate(self, channel_name: str, ml: float) -> float:
        channel = self._config.channels.get(channel_name)
        if channel is None:
            raise PumpError(
                f"unknown channel {channel_name!r}; "
                f"configured: {sorted(self._config.channels)}"
            )
        if ml <= 0:
            raise PumpError(f"ml must be positive, got {ml}")
        if ml > channel.max_ml_per_dose:
            raise PumpError(
                f"dose {ml} ml exceeds channel cap of {channel.max_ml_per_dose} ml"
            )
        seconds = seconds_for_dose(ml, channel.rate_ml_per_s, channel.lag_s)
        if seconds > self._config.max_relay_on_s:
            raise PumpError(
                f"dose needs {seconds:.1f} s, exceeds relay on-time cap of "
                f"{self._config.max_relay_on_s} s"
            )
        return seconds

    def _trip_if_budget_low(self) -> None:
        alerts = self._accountant.alerts()
        if alerts and self._arbiter.mode != "paused":
            self._arbiter.trip("budget: " + "; ".join(alerts))
            self._driver.all_off()

    async def dose(self, channel_name: str, ml: float) -> float:
        if self._arbiter.mode == "paused":
            raise ModeError("mode is paused; dose rejected")

        seconds = self.validate(channel_name, ml)
        channel = self._config.channels[channel_name]

        async with self._lock:
            # acquire() is a suspension point: a dose parked waiting for the lock can
            # find the rig paused by the time it resumes (a budget trip from the dose
            # ahead, or an operator POST /mode {paused}). Re-check here, before any
            # accounting or relay write, so a paused landing can't ride a stale check.
            if self._arbiter.mode == "paused":
                raise ModeError("mode is paused; dose rejected")
            if channel.role in ("dye", "water"):
                self._accountant.fill(ml, from_reservoir=channel.role == "water")
            if channel.role == "drain":
                self._accountant.begin_drain(ml)
            self._driver.set(channel.gpio, True)
            try:
                await asyncio.sleep(seconds)
            finally:
                self._driver.set(channel.gpio, False)
            if channel.role == "drain":
                self._accountant.complete_drain()
            self._trip_if_budget_low()

        return seconds

    async def timed_run(self, channel_name: str, seconds: float) -> float:
        """Calibration run: converts an on-time to an estimated ml via the channel's
        configured rate/lag, then delegates to dose() so every cap, accounting path,
        and the relay itself behave exactly as an equivalent manual dose (D3)."""
        if seconds <= 0:
            raise PumpError(f"seconds must be positive, got {seconds}")
        channel = self._config.channels.get(channel_name)
        if channel is None:
            raise PumpError(
                f"unknown channel {channel_name!r}; "
                f"configured: {sorted(self._config.channels)}"
            )
        est_ml = channel.rate_ml_per_s * max(seconds - channel.lag_s, 0.0)
        await self.dose(channel_name, est_ml)
        return est_ml
