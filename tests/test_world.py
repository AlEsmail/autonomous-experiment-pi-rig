# ABOUTME: Tests for the synthetic dose->LAB world: deterministic seeded noise, monotone dye
# ABOUTME: response with gamut saturation, skew/blip disruptions, and baseline optimizability.

import pytest

from shared.color import delta_e
from sim.world import SimWorld


def quiet_world(**kwargs):
    kwargs.setdefault("noise_sigma", 0.0)
    return SimWorld(**kwargs)


def test_zero_dye_is_near_white():
    L, a, b = quiet_world().mix(0, 0)
    assert L > 95
    assert abs(a) < 2
    assert abs(b) < 2


def test_same_seed_same_measurement():
    lab1 = SimWorld(noise_sigma=0.5, seed=42).mix(2, 1)
    lab2 = SimWorld(noise_sigma=0.5, seed=42).mix(2, 1)
    assert lab1 == lab2


def test_noise_perturbs_repeated_measurements():
    world = SimWorld(noise_sigma=0.5, seed=42)
    assert world.mix(2, 1) != world.mix(2, 1)


def test_dye_a_moves_color_away_from_white_monotonically():
    world = quiet_world()
    white = world.mix(0, 0)
    distances = [delta_e(world.mix(ml, 0), white) for ml in (0.5, 1, 2, 4)]
    assert distances == sorted(distances)
    assert distances[0] > 0.5


def test_gamut_saturates_at_high_dye():
    world = quiet_world()
    low_step = delta_e(world.mix(2, 0), world.mix(1, 0))
    high_step = delta_e(world.mix(11, 0), world.mix(10, 0))
    assert high_step < low_step / 2


def test_dyes_pull_in_different_directions():
    world = quiet_world()
    _, a_dyeA, b_dyeA = world.mix(3, 0)
    _, a_dyeB, b_dyeB = world.mix(0, 3)
    assert delta_e((0, a_dyeA, b_dyeA), (0, a_dyeB, b_dyeB)) > 10


def test_skew_scales_the_executed_dose():
    world = quiet_world()
    honest = world.mix(1, 0)
    world.set_skew("A", 0.5)
    skewed = world.mix(2, 0)
    assert delta_e(honest, skewed) < 0.01


def test_clear_skew_restores_honesty():
    world = quiet_world()
    honest = world.mix(2, 0)
    world.set_skew("A", 0.5)
    world.clear_skew()
    assert delta_e(world.mix(2, 0), honest) < 0.01


def test_blip_perturbs_exactly_one_cycle():
    world = quiet_world()
    baseline = world.mix(2, 2)
    world.set_blip(ml_water=50)
    blipped = world.mix(2, 2)
    after = world.mix(2, 2)
    assert delta_e(blipped, baseline) > 1
    assert delta_e(after, baseline) < 0.01


def test_reachable_target_found_by_coarse_to_fine_search():
    world = quiet_world()
    target = world.true_lab(3.0, 1.5)

    best_ml, best_de = None, float("inf")
    for a in range(0, 11):
        for b in range(0, 11):
            de = delta_e(world.mix(a, b), target)
            if de < best_de:
                best_ml, best_de = (a, b), de
    step = 0.25
    for a_off in range(-4, 5):
        for b_off in range(-4, 5):
            a = max(0.0, best_ml[0] + a_off * step)
            b = max(0.0, best_ml[1] + b_off * step)
            de = delta_e(world.mix(a, b), target)
            best_de = min(best_de, de)

    assert best_de < 1.0


def test_saturated_green_target_is_unreachable():
    world = quiet_world()
    target = (87.7, -86.2, 83.2)  # LAB of pure sRGB green
    best_de = min(
        delta_e(world.mix(a, b), target) for a in range(0, 13, 2) for b in range(0, 13, 2)
    )
    assert best_de > 20
