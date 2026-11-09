# ABOUTME: Simulation harness rig seam (decisions 9-10): SimCup tracks cup contents in
# ABOUTME: dose-domain ml; SimPumps/SimCamera sit behind the real pumps/camera interfaces.

import io
import math

from PIL import Image

from rig_service.drivers.fake import FakeCamera
from rig_service.pumps import PumpController
from shared.color import linear_rgb_to_lab, linear_rgb_to_srgb8

# D65 white point and the exact inverse of shared.color.linear_rgb_to_lab's XYZ<-RGB
# matrix, needed to turn a Lab value (measurement noise applied in Lab, matching
# SimWorld.mix's own noise convention) back into a renderable linear RGB triple.
_D65 = (0.95047, 1.0, 1.08883)
_XYZ_TO_LINEAR_RGB = (
    (3.2404542, -1.5371385, -0.4985314),
    (-0.9692660, 1.8760108, 0.0415560),
    (0.0556434, -0.2040259, 1.0572252),
)


def _lab_to_linear_rgb(lab: tuple[float, float, float]) -> tuple[float, float, float]:
    delta = 6 / 29

    def finv(t: float) -> float:
        return t**3 if t > delta else 3 * delta**2 * (t - 4 / 29)

    L, a, b = lab
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    x = finv(fx) * _D65[0]
    y = finv(fy) * _D65[1]
    z = finv(fz) * _D65[2]
    return tuple(
        row[0] * x + row[1] * y + row[2] * z for row in _XYZ_TO_LINEAR_RGB
    )


class SimCup:
    """Cup contents in dose-domain ml. fill(role, ml): 'dye_a'/'dye_b'/'water' add;
    drain() empties all three. Properties dye_a_ml, dye_b_ml, water_ml."""

    def __init__(self) -> None:
        self.dye_a_ml = 0.0
        self.dye_b_ml = 0.0
        self.water_ml = 0.0

    def fill(self, role: str, ml: float) -> None:
        if role == "dye_a":
            self.dye_a_ml += ml
        elif role == "dye_b":
            self.dye_b_ml += ml
        elif role == "water":
            self.water_ml += ml
        else:
            raise ValueError(f"unknown fill role {role!r}")

    def drain(self) -> None:
        self.dye_a_ml = 0.0
        self.dye_b_ml = 0.0
        self.water_ml = 0.0


class SimPumps(PumpController):
    def __init__(self, config, driver, accountant, arbiter, world, cup: SimCup) -> None:
        super().__init__(config, driver, accountant, arbiter)
        self._world = world
        self._cup = cup

    async def dose(self, channel_name: str, ml: float) -> float:
        # The real capped/locked/paused/accounted path runs first; its exceptions
        # propagate untouched. Only a successful dose updates the cup.
        seconds = await super().dose(channel_name, ml)

        role = self._config.channels[channel_name].role
        if role == "dye":
            self._cup.fill("dye_a" if channel_name == "A" else "dye_b", ml)
        elif role == "water":
            self._cup.fill("water", ml)
        elif role == "drain":
            self._cup.drain()

        return seconds


class SimCamera(FakeCamera):
    def __init__(
        self,
        defaults: dict[str, int],
        world,
        cup: SimCup,
        width: int,
        height: int,
        noise: bool = True,
    ) -> None:
        super().__init__(defaults, frame=b"")
        self._world = world
        self._cup = cup
        self._width = width
        self._height = height
        self._noise = noise

    def capture_jpeg(self) -> bytes:
        if self._cup.water_ml <= 0:
            reflectance = (1.0, 1.0, 1.0)
        else:
            conc_a = self._cup.dye_a_ml / self._cup.water_ml
            conc_b = self._cup.dye_b_ml / self._cup.water_ml
            reflectance = tuple(
                math.exp(-(conc_a * abs_a + conc_b * abs_b))
                for abs_a, abs_b in zip(self._world._dye_a, self._world._dye_b)
            )

        if self._noise:
            # Measurement noise is modeled in Lab space (matching SimWorld.mix's own
            # noise_sigma convention, kept consistent rather than re-derived), then
            # mapped back to a renderable linear RGB triple -- adding noise directly
            # to reflectance would swamp the signal (reflectance lives in [0,1], while
            # noise_sigma=0.4 is calibrated for Lab's roughly 0-100 scale).
            L, a, b = linear_rgb_to_lab(reflectance)
            sigma = self._world._noise_sigma
            noisy_lab = (
                L + self._world._rng.gauss(0, sigma),
                a + self._world._rng.gauss(0, sigma),
                b + self._world._rng.gauss(0, sigma),
            )
            reflectance = _lab_to_linear_rgb(noisy_lab)

        rgb_8bit = linear_rgb_to_srgb8(reflectance)

        image = Image.new("RGB", (self._width, self._height), rgb_8bit)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()
