# ABOUTME: Synthetic dose->LAB world for the simulation harness: Beer-Lambert dye mixing with
# ABOUTME: natural gamut saturation, seeded measurement noise, and injectable skew/blip disruptions.

import math
import random

from shared.color import linear_rgb_to_lab

# Absorbance per unit concentration (ml dye / ml water) for each linear RGB channel.
# Dye A absorbs green+blue (reads red); dye B absorbs red+green (reads blue).
DYE_A_ABSORBANCE = (5.0, 60.0, 60.0)
DYE_B_ABSORBANCE = (60.0, 60.0, 5.0)


class SimWorld:
    def __init__(
        self,
        base_ml: float = 100.0,
        dye_a_absorbance: tuple[float, float, float] = DYE_A_ABSORBANCE,
        dye_b_absorbance: tuple[float, float, float] = DYE_B_ABSORBANCE,
        noise_sigma: float = 0.4,
        seed: int = 0,
    ) -> None:
        self._base_ml = base_ml
        self._dye_a = dye_a_absorbance
        self._dye_b = dye_b_absorbance
        self._noise_sigma = noise_sigma
        self._rng = random.Random(seed)
        self._skew: dict[str, float] = {}
        self._blip_ml_water = 0.0

    def set_skew(self, channel: str, factor: float) -> None:
        if channel not in ("A", "B"):
            raise ValueError(f"skewable channels are A and B, got {channel!r}")
        self._skew[channel] = factor

    def clear_skew(self) -> None:
        self._skew.clear()

    def set_blip(self, ml_water: float) -> None:
        self._blip_ml_water = ml_water

    def true_lab(self, dose_a_ml: float, dose_b_ml: float) -> tuple[float, float, float]:
        return self._cup_lab(dose_a_ml, dose_b_ml, self._base_ml)

    def mix(self, dose_a_ml: float, dose_b_ml: float) -> tuple[float, float, float]:
        executed_a = dose_a_ml * self._skew.get("A", 1.0)
        executed_b = dose_b_ml * self._skew.get("B", 1.0)
        water_ml = self._base_ml + self._blip_ml_water
        self._blip_ml_water = 0.0

        L, a, b = self._cup_lab(executed_a, executed_b, water_ml)
        if self._noise_sigma:
            L += self._rng.gauss(0, self._noise_sigma)
            a += self._rng.gauss(0, self._noise_sigma)
            b += self._rng.gauss(0, self._noise_sigma)
        return (L, a, b)

    def _cup_lab(
        self, dose_a_ml: float, dose_b_ml: float, water_ml: float
    ) -> tuple[float, float, float]:
        conc_a = dose_a_ml / water_ml
        conc_b = dose_b_ml / water_ml
        reflectance = tuple(
            math.exp(-(conc_a * abs_a + conc_b * abs_b))
            for abs_a, abs_b in zip(self._dye_a, self._dye_b)
        )
        return linear_rgb_to_lab(reflectance)
