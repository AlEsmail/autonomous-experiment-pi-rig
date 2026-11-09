# ABOUTME: Pure color math: hex parsing, sRGB/linear RGB to CIE LAB (D65 white point),
# ABOUTME: CIE76 delta-E, and the run objective score = deltaE + lambda * ml dye.

import math

_D65 = (0.95047, 1.0, 1.08883)


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    digits = value.lstrip("#")
    if len(digits) != 6:
        raise ValueError(f"expected 6 hex digits, got {value!r}")
    try:
        return tuple(int(digits[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        raise ValueError(f"invalid hex color {value!r}") from None


def _srgb_channel_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_channel_to_srgb(c: float) -> float:
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def linear_rgb_to_srgb8(rgb: tuple[float, float, float]) -> tuple[int, int, int]:
    """Linear [0,1] RGB -> gamma-encoded 8-bit sRGB (values clamped to [0,1] first).
    Exact inverse of the module's sRGB-decode path, so
    rgb_to_lab(linear_rgb_to_srgb8(x)) roundtrips linear_rgb_to_lab(x) within 8-bit
    quantization."""
    result = []
    for c in rgb:
        clamped = min(max(c, 0.0), 1.0)
        encoded = _linear_channel_to_srgb(clamped)
        result.append(min(max(round(encoded * 255), 0), 255))
    return tuple(result)


def _f(t: float) -> float:
    delta = 6 / 29
    return t ** (1 / 3) if t > delta**3 else t / (3 * delta**2) + 4 / 29


def linear_rgb_to_lab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    r, g, b = rgb
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    fx, fy, fz = (_f(v / n) for v, n in zip((x, y, z), _D65))
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def rgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    linear = tuple(_srgb_channel_to_linear(c / 255) for c in rgb)
    return linear_rgb_to_lab(linear)


def delta_e(lab1: tuple[float, float, float], lab2: tuple[float, float, float]) -> float:
    return math.dist(lab1, lab2)


def score(delta_e_value: float, ml_dye: float, lam: float) -> float:
    return delta_e_value + lam * ml_dye
