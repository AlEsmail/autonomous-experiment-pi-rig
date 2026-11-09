# ABOUTME: Tests for the measurement patch domain module: validation, box math, and holder.

import pytest

from rig_service.patch import (
    MIN_PATCH_SIZE_PX,
    PatchHolder,
    PatchSpec,
    patch_box,
    validate_patch,
)


def test_validate_patch_accepts_shipped_seed():
    spec = validate_patch(0.5, 0.5, 200, 1920, 1080)
    assert spec == PatchSpec(cx=0.5, cy=0.5, size_px=200)


@pytest.mark.parametrize("cx", [-0.1, 1.1])
def test_validate_patch_rejects_cx_outside_interval(cx):
    with pytest.raises(ValueError, match="center"):
        validate_patch(cx, 0.5, 200, 1920, 1080)


@pytest.mark.parametrize("cy", [-0.1, 1.1])
def test_validate_patch_rejects_cy_outside_interval(cy):
    with pytest.raises(ValueError, match="center"):
        validate_patch(0.5, cy, 200, 1920, 1080)


@pytest.mark.parametrize("cx", [0.0, 1.0])
def test_validate_patch_center_on_boundary_passes_center_check(cx):
    # 0.0 and 1.0 are inside the unit interval; the failure must come from the box
    # guard, not the center guard (kills a `<=` -> `<` mutation).
    with pytest.raises(ValueError, match="not wholly inside"):
        validate_patch(cx, 0.5, MIN_PATCH_SIZE_PX, 1920, 1080)


def test_validate_patch_rejects_size_below_floor():
    with pytest.raises(ValueError, match="floor"):
        validate_patch(0.5, 0.5, MIN_PATCH_SIZE_PX - 1, 1920, 1080)

    # Exact floor with a fitting box is accepted (kills shifted boundary / flipped
    # comparison mutations).
    spec = validate_patch(0.5, 0.5, MIN_PATCH_SIZE_PX, 100, 100)
    assert spec == PatchSpec(cx=0.5, cy=0.5, size_px=MIN_PATCH_SIZE_PX)


def test_validate_patch_rejects_box_off_left_edge():
    with pytest.raises(ValueError, match="not wholly inside"):
        validate_patch(0.05, 0.5, 200, 1920, 1080)


def test_validate_patch_rejects_box_off_top_edge():
    with pytest.raises(ValueError, match="not wholly inside"):
        validate_patch(0.5, 0.05, 200, 1920, 1080)


def test_validate_patch_rejects_box_off_right_edge():
    with pytest.raises(ValueError, match="not wholly inside"):
        validate_patch(0.95, 0.5, 200, 1920, 1080)


def test_validate_patch_rejects_box_off_bottom_edge():
    with pytest.raises(ValueError, match="not wholly inside"):
        validate_patch(0.5, 0.95, 200, 1920, 1080)


def test_validate_patch_accepts_box_exactly_fitting_frame():
    # Edge-inclusive: a 100x100 box on a 100x100 frame centered at (0.5, 0.5).
    spec = validate_patch(0.5, 0.5, 100, 100, 100)
    assert spec == PatchSpec(cx=0.5, cy=0.5, size_px=100)


@pytest.mark.parametrize(
    "cx,cy,size_px,width,height,expected",
    [
        # Even size.
        (0.3, 0.7, 100, 1000, 1000, (250, 650, 350, 750)),
        # Odd size.
        (0.3, 0.7, 101, 1000, 1000, (250, 650, 351, 751)),
    ],
)
def test_patch_box_math(cx, cy, size_px, width, height, expected):
    assert patch_box(PatchSpec(cx, cy, size_px), width, height) == expected


def test_patch_holder_current_returns_seeded_spec():
    spec = PatchSpec(cx=0.5, cy=0.5, size_px=200)
    holder = PatchHolder(spec)
    assert holder.current is spec


def test_patch_holder_set_rebinds_atomically():
    spec1 = PatchSpec(cx=0.5, cy=0.5, size_px=200)
    spec2 = PatchSpec(cx=0.25, cy=0.75, size_px=100)
    holder = PatchHolder(spec1)

    assert holder.current is spec1
    holder.set(spec2)
    assert holder.current is spec2
    assert holder.current is not spec1
