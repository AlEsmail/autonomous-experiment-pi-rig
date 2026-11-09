# ABOUTME: Measure path from raw camera JPEG bytes to an explicit-center patch's mean RGB
# ABOUTME: and its CIE LAB conversion (delegates color math to shared.color).

import io

from PIL import Image, ImageStat, UnidentifiedImageError

from shared.color import rgb_to_lab


def patch_rgb(jpeg: bytes, cx: float, cy: float, size_px: int) -> tuple[float, float, float]:
    """Mean RGB (0-255 floats) over the size_px x size_px box centered at normalized
    (cx, cy) of the frame. Raises ValueError on undecodable bytes or a box not wholly
    inside the image. cx=cy=0.5 reproduces the old centered behavior exactly."""
    try:
        image = Image.open(io.BytesIO(jpeg))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"could not decode JPEG bytes: {exc}") from exc

    image = image.convert("RGB")
    width, height = image.size

    left = round(cx * width - size_px / 2)
    top = round(cy * height - size_px / 2)
    right = left + size_px
    bottom = top + size_px
    if left < 0 or top < 0 or right > width or bottom > height:
        raise ValueError(
            f"patch box ({left}, {top}, {right}, {bottom}) not wholly inside image "
            f"({width}x{height})"
        )

    box = (left, top, right, bottom)
    patch = image.crop(box)

    r, g, b = ImageStat.Stat(patch).mean
    return (r, g, b)


def lab_from_jpeg(jpeg: bytes, cx: float, cy: float, size_px: int) -> tuple[float, float, float]:
    """patch_rgb -> shared.color.rgb_to_lab (sRGB -> CIE LAB, D65)."""
    return rgb_to_lab(patch_rgb(jpeg, cx, cy, size_px))
