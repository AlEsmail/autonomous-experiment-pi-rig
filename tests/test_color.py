# ABOUTME: Tests for shared color math: hex parsing, sRGB and linear RGB to CIE LAB (D65),
# ABOUTME: CIE76 delta-E, and the run objective score = deltaE + lambda * ml dye.

import pytest

from shared.color import (
    delta_e,
    hex_to_rgb,
    linear_rgb_to_lab,
    linear_rgb_to_srgb8,
    rgb_to_lab,
    score,
)


def test_hex_to_rgb():
    assert hex_to_rgb("#FF8000") == (255, 128, 0)
    assert hex_to_rgb("ff8000") == (255, 128, 0)


def test_hex_to_rgb_rejects_malformed():
    with pytest.raises(ValueError):
        hex_to_rgb("#FF80")
    with pytest.raises(ValueError):
        hex_to_rgb("#GGGGGG")


def test_white_is_L100_neutral():
    L, a, b = rgb_to_lab((255, 255, 255))
    assert L == pytest.approx(100.0, abs=0.01)
    assert a == pytest.approx(0.0, abs=0.01)
    assert b == pytest.approx(0.0, abs=0.01)


def test_black_is_L0():
    L, a, b = rgb_to_lab((0, 0, 0))
    assert L == pytest.approx(0.0, abs=0.01)


def test_srgb_red_reference_values():
    L, a, b = rgb_to_lab((255, 0, 0))
    assert L == pytest.approx(53.24, abs=0.1)
    assert a == pytest.approx(80.09, abs=0.1)
    assert b == pytest.approx(67.20, abs=0.1)


def test_linear_rgb_full_reflectance_is_white():
    L, a, b = linear_rgb_to_lab((1.0, 1.0, 1.0))
    assert L == pytest.approx(100.0, abs=0.01)
    assert a == pytest.approx(0.0, abs=0.01)


def test_delta_e_is_euclidean_in_lab():
    assert delta_e((50, 0, 0), (50, 3, 4)) == pytest.approx(5.0)
    assert delta_e((10, -2, 7), (10, -2, 7)) == 0.0


def test_score_is_delta_e_plus_lambda_ml():
    assert score(delta_e_value=8.0, ml_dye=4.0, lam=0.5) == pytest.approx(10.0)
    assert score(delta_e_value=8.0, ml_dye=4.0, lam=0.0) == pytest.approx(8.0)


# --- T5: linear_rgb_to_srgb8 (the sRGB-encode inverse of _srgb_channel_to_linear) ---


def test_linear_rgb_to_srgb8_roundtrips_through_rgb_to_lab():
    # Dark/mid/light gray plus one non-gray triple -- kills a missing (or wrong-direction)
    # gamma encode: rendering raw linear reflectance as if it were already sRGB-encoded
    # biases LAB by several units, well outside 8-bit quantization tolerance.
    samples = [
        (0.02, 0.02, 0.02),
        (0.18, 0.18, 0.18),
        (0.9, 0.9, 0.9),
        (0.6, 0.2, 0.4),
    ]
    for rgb in samples:
        encoded = linear_rgb_to_srgb8(rgb)
        got = rgb_to_lab(encoded)
        want = linear_rgb_to_lab(rgb)
        for got_component, want_component in zip(got, want):
            assert got_component == pytest.approx(want_component, abs=1.0)


def test_linear_rgb_to_srgb8_clamps_out_of_range():
    encoded = linear_rgb_to_srgb8((-0.5, 1.5, 0.5))
    assert encoded[0] == 0
    assert encoded[1] == 255
    assert 0 <= encoded[2] <= 255
    for channel in encoded:
        assert isinstance(channel, int)
