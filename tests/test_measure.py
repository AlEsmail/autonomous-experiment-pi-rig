# ABOUTME: Tests for the measure path from raw camera JPEG bytes to an explicit-patch
# ABOUTME: mean RGB and its CIE LAB conversion (delegated to shared.color, never reimplemented).

import io

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image, ImageDraw, ImageStat

from rig_service.patch import PatchSpec, patch_box, validate_patch
from shared.color import rgb_to_lab
from shared.measure import lab_from_jpeg, patch_rgb


def _solid_jpeg(color, size=(80, 80), quality=95):
    image = Image.new("RGB", size, color)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, subsampling=0)
    return buf.getvalue()


def _bordered_jpeg(border_color, center_color, size, box, quality=95):
    image = Image.new("RGB", size, border_color)
    ImageDraw.Draw(image).rectangle(box, fill=center_color)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, subsampling=0)
    return buf.getvalue()


def _two_field_jpeg(left_color, right_color, size=(80, 80), quality=95):
    """Left half left_color, right half right_color."""
    image = Image.new("RGB", size, left_color)
    width, height = size
    ImageDraw.Draw(image).rectangle((width // 2, 0, width, height), fill=right_color)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, subsampling=0)
    return buf.getvalue()


rgb_channel = st.integers(min_value=0, max_value=255)


@given(st.tuples(rgb_channel, rgb_channel, rgb_channel))
@settings(max_examples=25, deadline=None)
def test_center_patch_mean_of_solid_color(color):
    jpeg = _solid_jpeg(color, size=(80, 80))
    r, g, b = patch_rgb(jpeg, 0.5, 0.5, 30)
    assert r == pytest.approx(color[0], abs=2)
    assert g == pytest.approx(color[1], abs=2)
    assert b == pytest.approx(color[2], abs=2)


def test_patch_ignores_border():
    border_color = (10, 10, 200)
    center_color = (250, 100, 10)
    # 100x100 image, a 60x60 center block (20..80 both axes) leaves 20px margin
    # around a 20px centered patch (40..60) -- comfortably inside the block.
    jpeg = _bordered_jpeg(border_color, center_color, size=(100, 100), box=(20, 20, 80, 80))

    r, g, b = patch_rgb(jpeg, 0.5, 0.5, 20)

    assert r == pytest.approx(center_color[0], abs=2)
    assert g == pytest.approx(center_color[1], abs=2)
    assert b == pytest.approx(center_color[2], abs=2)


def test_lab_from_jpeg_matches_rgb_to_lab():
    jpeg = _solid_jpeg((120, 60, 200), size=(80, 80))

    lab = lab_from_jpeg(jpeg, 0.5, 0.5, 30)
    expected = rgb_to_lab(patch_rgb(jpeg, 0.5, 0.5, 30))

    assert lab == pytest.approx(expected, abs=1e-9)


def test_image_smaller_than_patch_rejected():
    jpeg = _solid_jpeg((10, 10, 10), size=(50, 50))

    with pytest.raises(ValueError):
        patch_rgb(jpeg, 0.5, 0.5, 200)


def test_undecodable_bytes_rejected():
    with pytest.raises(ValueError) as exc_info:
        patch_rgb(b"this is not a jpeg file at all", 0.5, 0.5, 10)

    assert "decode" in str(exc_info.value).lower()


def test_truncated_jpeg_rejected():
    # A valid JPEG header (Image.open succeeds) with its compressed scan data cut short:
    # decoding fails inside image.load(), the OSError branch patch_rgb also catches
    # -- distinct from test_undecodable_bytes_rejected's UnidentifiedImageError branch,
    # which fails at open() on bytes that never look like a JPEG at all.
    jpeg = _solid_jpeg((120, 60, 200), size=(80, 80))
    truncated = jpeg[:-100]

    # Confirms the cut lands past the header (open() itself must not raise here) --
    # otherwise this test would exercise the already-covered open()/UnidentifiedImageError
    # path instead of the load()/OSError one it's meant to pin.
    Image.open(io.BytesIO(truncated))

    with pytest.raises(ValueError) as exc_info:
        patch_rgb(truncated, 0.5, 0.5, 10)

    assert "decode" in str(exc_info.value).lower()


def test_image_smaller_in_width_only_rejected():
    # 150x300: width (150) is below patch_px (200), height (300) is not.
    jpeg = _solid_jpeg((10, 10, 10), size=(150, 300))

    with pytest.raises(ValueError):
        patch_rgb(jpeg, 0.5, 0.5, 200)


def test_image_smaller_in_height_only_rejected():
    # 300x150: height (150) is below patch_px (200), width (300) is not.
    jpeg = _solid_jpeg((10, 10, 10), size=(300, 150))

    with pytest.raises(ValueError):
        patch_rgb(jpeg, 0.5, 0.5, 200)


def test_patch_centered_on_non_square_image():
    border_color = (10, 10, 200)
    center_color = (250, 100, 10)
    # 300x100 image; an 80x80 block centered on the true image center (150, 50)
    # leaves a 20px margin around the 40px centered patch (130..170, 30..70) on
    # both axes. A per-axis-correct centering samples the block. A transposed
    # computation (swapping width/height when computing left/top) would offset
    # the box mostly or entirely out of the 100px-tall image, sampling black
    # padding instead of the block color.
    jpeg = _bordered_jpeg(border_color, center_color, size=(300, 100), box=(110, 10, 190, 90))

    r, g, b = patch_rgb(jpeg, 0.5, 0.5, 40)

    assert r == pytest.approx(center_color[0], abs=2)
    assert g == pytest.approx(center_color[1], abs=2)
    assert b == pytest.approx(center_color[2], abs=2)


def test_image_exactly_patch_size_accepted():
    color = (30, 200, 90)
    jpeg = _solid_jpeg(color, size=(30, 30))

    r, g, b = patch_rgb(jpeg, 0.5, 0.5, 30)

    assert r == pytest.approx(color[0], abs=2)
    assert g == pytest.approx(color[1], abs=2)
    assert b == pytest.approx(color[2], abs=2)


def test_grayscale_jpeg_converted_to_rgb():
    gray = 128
    image = Image.new("L", (80, 80), gray)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95, subsampling=0)
    jpeg = buf.getvalue()

    r, g, b = patch_rgb(jpeg, 0.5, 0.5, 30)

    assert r == pytest.approx(gray, abs=2)
    assert g == pytest.approx(gray, abs=2)
    assert b == pytest.approx(gray, abs=2)


# --- T3: explicit-center interface ---


def test_patch_rgb_off_center_samples_requested_region():
    left_color = (250, 10, 10)
    right_color = (10, 10, 250)
    size = 30
    jpeg = _two_field_jpeg(left_color, right_color, size=(100, 100))

    left_rgb = patch_rgb(jpeg, 0.25, 0.5, size)
    right_rgb = patch_rgb(jpeg, 0.75, 0.5, size)

    assert left_rgb[0] > left_rgb[2]
    assert right_rgb[2] > right_rgb[0]


def test_patch_rgb_midpoint_matches_legacy_center():
    color = (120, 80, 160)
    size = 40
    jpeg = _solid_jpeg(color, size=(100, 100))

    expected_box = (30, 30, 70, 70)
    expected = tuple(ImageStat.Stat(Image.open(io.BytesIO(jpeg)).convert("RGB").crop(expected_box)).mean)

    assert patch_rgb(jpeg, 0.5, 0.5, size) == pytest.approx(expected, abs=1e-9)


def test_patch_rgb_rejects_undecodable():
    with pytest.raises(ValueError) as exc_info:
        patch_rgb(b"not jpeg", 0.5, 0.5, 20)
    assert "decode" in str(exc_info.value).lower()


def test_patch_rgb_rejects_truncated_jpeg():
    jpeg = _solid_jpeg((100, 100, 100), size=(80, 80))
    truncated = jpeg[:-100]
    Image.open(io.BytesIO(truncated))  # header intact

    with pytest.raises(ValueError) as exc_info:
        patch_rgb(truncated, 0.5, 0.5, 20)
    assert "decode" in str(exc_info.value).lower()


@pytest.mark.parametrize(
    "cx,cy,description",
    [
        (0.0, 0.5, "left"),
        (0.5, 0.0, "top"),
        (1.0, 0.5, "right"),
        (0.5, 1.0, "bottom"),
    ],
)
def test_patch_rgb_rejects_box_outside_image(cx, cy, description):
    # 80x80 image, 30x30 patch. Center (0.5, 0.5) fits exactly; each boundary
    # center pushes the box off one edge.
    jpeg = _solid_jpeg((50, 50, 50), size=(80, 80))

    with pytest.raises(ValueError, match="not wholly inside"):
        patch_rgb(jpeg, cx, cy, 30)


@st.composite
def _valid_patch_params(draw):
    """Generate (cx, cy, size_px, width, height) combinations that pass
    rig_service.patch.validate_patch.  The normalized center is offset by a
    signed fractional pixel amount, so the exact value of
    cx*width - size_px/2 can have a negative fractional part; in those
    cases round() and int() disagree by one pixel.
    """
    width = draw(st.integers(min_value=40, max_value=200))
    height = draw(st.integers(min_value=40, max_value=200))
    max_size = min(width, height)
    # Keep size_px at least two pixels smaller than the smaller dimension so
    # there is room for a one-pixel margin on every side.
    size_px = draw(st.integers(min_value=20, max_value=max_size - 2))
    # One-pixel margin guarantees an int()-truncated box is still valid and
    # a one-pixel miss samples the background border.
    left = draw(st.integers(min_value=1, max_value=width - size_px - 1))
    top = draw(st.integers(min_value=1, max_value=height - size_px - 1))
    frac_x = draw(st.floats(min_value=-0.49, max_value=0.49))
    frac_y = draw(st.floats(min_value=-0.49, max_value=0.49))
    cx = (left + size_px / 2 + frac_x) / width
    cy = (top + size_px / 2 + frac_y) / height
    return cx, cy, size_px, width, height


@given(_valid_patch_params())
@settings(max_examples=50, deadline=None)
def test_patch_rgb_box_agrees_with_patch_box(params):
    """patch_rgb must sample exactly the box patch_box (T1) computes.
    We draw a high-contrast foreground rectangle at patch_box's coordinates;
    if patch_rgb's inline math diverged from patch_box by even one pixel it
    would sample the background border and the mean color would shift.
    """
    cx, cy, size_px, width, height = params
    spec = validate_patch(cx, cy, size_px, width, height)
    expected_box = patch_box(spec, width, height)

    border_color = (10, 10, 200)
    center_color = (250, 100, 10)
    jpeg = _bordered_jpeg(
        border_color, center_color, size=(width, height), box=expected_box
    )

    r, g, b = patch_rgb(jpeg, cx, cy, size_px)

    assert r == pytest.approx(center_color[0], abs=2)
    assert g == pytest.approx(center_color[1], abs=2)
    assert b == pytest.approx(center_color[2], abs=2)
