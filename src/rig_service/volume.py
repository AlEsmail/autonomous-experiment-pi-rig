# ABOUTME: Cup and session volume accounting: rejects overfills and fails closed on unknown cup
# ABOUTME: or session state; declare_staging is the sole reset; drains persist durably via begin/complete.

from dataclasses import dataclass
from typing import Callable


class VolumeError(Exception):
    pass


@dataclass(frozen=True)
class VolumeState:
    cup_ml: float | None
    water_drawn_ml: float | None
    waste_ml: float | None


class VolumeAccountant:
    def __init__(
        self,
        cup_ceiling_ml: float,
        reservoir_ml: float,
        waste_capacity_ml: float,
        low_margin_ml: float,
        state: VolumeState,
        on_change: Callable[[VolumeState], None],
    ) -> None:
        self._cup_ceiling_ml = cup_ceiling_ml
        self._reservoir_ml = reservoir_ml
        self._waste_capacity_ml = waste_capacity_ml
        self._low_margin_ml = low_margin_ml
        self._cup_ml = state.cup_ml
        self._water_drawn_ml = state.water_drawn_ml
        self._waste_ml = state.waste_ml
        self._on_change = on_change
        # Pending drain decrement: (charged waste ml, drain command ml) stashed by
        # begin_drain and applied by complete_drain. None when no drain is pending.
        self._pending_drain: tuple[float, float] | None = None

    @property
    def cup_ml(self) -> float | None:
        return self._cup_ml

    @property
    def water_drawn_ml(self) -> float | None:
        return self._water_drawn_ml

    @property
    def waste_ml(self) -> float | None:
        return self._waste_ml

    @property
    def state(self) -> VolumeState:
        return VolumeState(self._cup_ml, self._water_drawn_ml, self._waste_ml)

    def _persist(self, state: VolumeState) -> None:
        try:
            self._on_change(state)
        except Exception as error:
            raise VolumeError(f"failed to persist volume state: {error}") from error

    def fill(self, ml: float, from_reservoir: bool = False) -> None:
        if ml <= 0:
            raise VolumeError(f"fill ml must be positive, got {ml}")
        if self._cup_ml is None:
            # Unknown is treated as non-empty (decision 1): any fill could push an
            # already-full cup over the ceiling, so unknown rejects every fill.
            raise VolumeError("cup volume is unknown; a fill could exceed the ceiling")
        if self._water_drawn_ml is None or self._waste_ml is None:
            # Either session counter unknown rejects every fill (decision 15): a rig that
            # cannot charge its accounting must not actuate.
            raise VolumeError(
                "session counters unknown; declare staging via POST /staging before dosing"
            )
        if self._cup_ml + ml > self._cup_ceiling_ml:
            raise VolumeError(
                f"fill of {ml} ml would take the cup to {self._cup_ml + ml:.1f} ml, "
                f"over the {self._cup_ceiling_ml} ml ceiling"
            )
        new_cup_ml = self._cup_ml + ml
        new_water_drawn_ml = self._water_drawn_ml + ml if from_reservoir else self._water_drawn_ml
        # Cup fill and water_drawn both write before the relay fires; persist first so a
        # crash mid-fill is recovered as the post-fill (conservative) state.
        self._persist(VolumeState(new_cup_ml, new_water_drawn_ml, self._waste_ml))
        self._cup_ml = new_cup_ml
        self._water_drawn_ml = new_water_drawn_ml

    def begin_drain(self, ml: float) -> float:
        if ml <= 0:
            raise VolumeError(f"drain ml must be positive, got {ml}")
        if self._water_drawn_ml is None or self._waste_ml is None:
            # Either session counter unknown rejects every drain (decision 15): the drain
            # would charge waste_ml, and unknown waste can't be charged.
            raise VolumeError(
                "session counters unknown; declare staging via POST /staging before draining"
            )
        if self._cup_ml is None:
            # Unknown cup: the tight upper bound on what could physically leave a cup
            # holding at most cup_ceiling_ml.
            charged = min(ml, self._cup_ceiling_ml)
        else:
            charged = min(self._cup_ml, ml)
        new_waste_ml = self._waste_ml + charged
        # waste_ml writes before the relay fires; cup stays recorded high until
        # complete_drain (decision 2).
        self._persist(VolumeState(self._cup_ml, self._water_drawn_ml, new_waste_ml))
        self._waste_ml = new_waste_ml
        # Overwrites any prior pending decrement: a dropped decrement leaves the cup
        # high, which is conservative (decision 2).
        self._pending_drain = (charged, ml)
        return charged

    def complete_drain(self) -> None:
        if self._pending_drain is None:
            return
        charged, command_ml = self._pending_drain
        self._pending_drain = None
        if self._cup_ml is None:
            new_cup_ml = 0.0 if command_ml >= self._cup_ceiling_ml else None
        else:
            new_cup_ml = self._cup_ml - charged
        self._cup_ml = new_cup_ml
        # Cup drain writes after the relay is off; memory is already mutated so a sink
        # failure here leaves memory correct and the durable cup recorded high.
        self._persist(VolumeState(self._cup_ml, self._water_drawn_ml, self._waste_ml))

    def declare_staging(self, water_remaining_ml: float, waste_ml: float) -> None:
        if not (0 <= water_remaining_ml <= self._reservoir_ml):
            raise VolumeError(
                f"water_remaining_ml {water_remaining_ml} out of range "
                f"[0, {self._reservoir_ml}]"
            )
        if not (0 <= waste_ml <= self._waste_capacity_ml):
            raise VolumeError(
                f"waste_ml {waste_ml} out of range [0, {self._waste_capacity_ml}]"
            )
        new_water_drawn_ml = self._reservoir_ml - water_remaining_ml
        # Persist before mutating (fill's pattern); cup_ml is untouched -- the cup
        # resolves only by draining (decision 1), never by declaration.
        self._persist(VolumeState(self._cup_ml, new_water_drawn_ml, waste_ml))
        self._water_drawn_ml = new_water_drawn_ml
        self._waste_ml = waste_ml

    def alerts(self) -> list[str]:
        alerts = []
        if self._cup_ml is None:
            alerts.append(
                f"cup volume unknown; drain at least {self._cup_ceiling_ml:.0f} ml to resolve"
            )
        if self._water_drawn_ml is None:
            alerts.append("water counter unknown; declare staging via POST /staging")
        else:
            reservoir_left = self._reservoir_ml - self._water_drawn_ml
            if reservoir_left <= self._low_margin_ml:
                alerts.append(f"reservoir low: {reservoir_left:.0f} ml remaining")
        if self._waste_ml is None:
            alerts.append("waste counter unknown; declare staging via POST /staging")
        else:
            waste_headroom = self._waste_capacity_ml - self._waste_ml
            if waste_headroom <= self._low_margin_ml:
                alerts.append(f"waste bucket nearly full: {waste_headroom:.0f} ml headroom")
        return alerts
