# ABOUTME: Measurement patch domain module — type, validation, box math, and holder.
# ABOUTME: Standalone domain module: imports nothing from rig_service.* (D6).

from dataclasses import dataclass

MIN_PATCH_SIZE_PX = 20


@dataclass(frozen=True)
class PatchSpec:
    cx: float      # normalized [0,1] fraction of frame width
    cy: float      # normalized [0,1] fraction of frame height
    size_px: int   # box edge length in pixels


def validate_patch(cx: float, cy: float, size_px: int, width: int, height: int) -> PatchSpec:
    """Reject — NEVER clamp (D2). Raises ValueError, checked in this order:
    (1) center outside [0,1]
    (2) size_px below the floor
    (3) box not wholly inside the configured frame
    Returns the frozen PatchSpec on success.
    """
    if not (0.0 <= cx <= 1.0) or not (0.0 <= cy <= 1.0):
        raise ValueError(f"center ({cx}, {cy}) outside [0,1]")
    if size_px < MIN_PATCH_SIZE_PX:
        raise ValueError(f"size_px ({size_px}) below the floor ({MIN_PATCH_SIZE_PX})")

    spec = PatchSpec(cx=cx, cy=cy, size_px=size_px)
    left, top, right, bottom = patch_box(spec, width, height)
    if left < 0 or top < 0 or right > width or bottom > height:
        raise ValueError(
            f"patch box ({left}, {top}, {right}, {bottom}) not wholly inside frame "
            f"({width}x{height})"
        )

    return spec


def patch_box(patch: PatchSpec, width: int, height: int) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) pixel box. Callers validate first;
    a box exactly equal to the frame is valid (edge-inclusive).
    """
    left = round(patch.cx * width - patch.size_px / 2)
    top = round(patch.cy * height - patch.size_px / 2)
    right = left + patch.size_px
    bottom = top + patch.size_px
    return left, top, right, bottom


class PatchHolder:
    """The live measurement patch (D1): holds ONE immutable PatchSpec, only ever
    atomically rebound via set(), never mutated in place. Readers fetch the whole
    spec in a single attribute read, so no lock is needed.
    """

    def __init__(self, initial: PatchSpec) -> None:
        self._current = initial

    @property
    def current(self) -> PatchSpec:
        return self._current

    def set(self, patch: PatchSpec) -> None:
        self._current = patch
