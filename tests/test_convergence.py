# ABOUTME: Tests for scientist convergence stop rule.
# ABOUTME: Covers budget, delta-e branch, plateau branch, and floor semantics.

import pytest

from scientist.convergence import StopVerdict, should_stop
from scientist.settings import ScientistConfig


def make_cfg(**overrides):
    defaults = {
        "base_url": "http://127.0.0.1:8000",
        "poll_idle_s": 5.0,
        "poll_active_s": 2.0,
        "request_timeout_s": 10.0,
        "cycle_timeout_s": 900.0,
        "llm_max_tokens": 2000,
        "llm_retries": 2,
        "dose_cap_hint_ml": 10.0,
        "conv_delta_e": 3.0,
        "conv_hold": 3,
        "conv_epsilon": 0.5,
        "conv_window": 5,
        "conv_min_cycles": 5,
    }
    defaults.update(overrides)
    return ScientistConfig(**defaults)


def row(
    n,
    dose_a_ml=1.0,
    dose_b_ml=1.0,
    measured_lab=(50.0, 0.0, 0.0),
    blank_lab=(60.0, 0.0, 0.0),
    delta_e=5.0,
    lam=0.05,
    score=10.0,
    aborted=None,
):
    return {
        "n": n,
        "dose_a_ml": dose_a_ml,
        "dose_b_ml": dose_b_ml,
        "measured_lab": measured_lab,
        "blank_lab": blank_lab,
        "delta_e": delta_e,
        "lambda": lam,
        "score": score,
        "aborted": aborted,
    }


def test_budget_reached_stops():
    cfg = make_cfg()
    budget = 10
    cycles = [row(n, score=float(11 - n), delta_e=float(n)) for n in range(1, budget + 1)]
    verdict = should_stop(cycles, budget, cfg)
    assert verdict.stop is True
    assert verdict.reason == "budget"


def test_budget_not_reached_continues():
    cfg = make_cfg()
    budget = 10
    cycles = [row(n, score=float(11 - n), delta_e=float(n)) for n in range(1, budget)]
    verdict = should_stop(cycles, budget, cfg)
    assert verdict.stop is False
    assert verdict.reason is None


def test_aborted_rows_count_toward_budget_only():
    cfg = make_cfg(conv_min_cycles=2, conv_hold=2)
    budget = 3
    # Two completed rows do not satisfy the delta-e branch (delta_e >= threshold);
    # one aborted row pushes the total count to the budget. Budget must win.
    cycles = [
        row(1, delta_e=5.0, score=10.0),
        row(2, delta_e=5.0, score=9.0),
        row(3, delta_e=1.0, score=8.0, aborted="post_measure"),
    ]
    verdict = should_stop(cycles, budget, cfg)
    assert verdict.stop is True
    assert verdict.reason == "budget"
    # Without the aborted row the budget is not reached.
    verdict2 = should_stop(cycles[:-1], budget, cfg)
    assert verdict2.stop is False


def test_post_measure_abort_excluded_from_convergence():
    cfg = make_cfg(conv_min_cycles=2, conv_hold=2, conv_window=2)
    aborted_row = row(
        1,
        measured_lab=(50.0, 0.0, 0.0),
        blank_lab=(60.0, 0.0, 0.0),
        delta_e=1.0,
        score=5.0,
        aborted="drain timeout",
    )
    completed_rows = [
        row(2, delta_e=5.0, score=4.9),
        row(3, delta_e=5.0, score=4.8),
    ]
    cycles = [aborted_row] + completed_rows
    # If the aborted row counted as completed, the plateau branch would fire.
    verdict = should_stop(cycles, cycle_budget=10, cfg=cfg)
    assert verdict.stop is False
    # Sanity: replacing it with a completed row makes plateau convergence fire.
    cycles_completed = [row(1, delta_e=1.0, score=5.0)] + completed_rows
    verdict2 = should_stop(cycles_completed, cycle_budget=10, cfg=cfg)
    assert verdict2.stop is True


def test_min_cycles_floor_blocks_both_branches():
    cfg = make_cfg(conv_min_cycles=4, conv_hold=3, conv_window=2, conv_epsilon=0.5)

    # Delta-e branch satisfied with 3 qualifying rows, but floor=4 blocks it.
    cycles_delta = [
        row(1, delta_e=1.0, score=10.0),
        row(2, delta_e=1.0, score=9.0),
        row(3, delta_e=1.0, score=8.0),
    ]
    assert should_stop(cycles_delta, 10, cfg).stop is False
    cycles_delta.append(row(4, delta_e=1.0, score=7.0))
    assert should_stop(cycles_delta, 10, cfg).stop is True

    # Plateau branch needs window+1 = 3 completed; floor=4 blocks it.
    cycles_plateau = [
        row(1, score=8.5, delta_e=5.0),
        row(2, score=10.0, delta_e=5.0),
        row(3, score=8.0, delta_e=5.0),
    ]
    assert should_stop(cycles_plateau, 10, cfg).stop is False
    cycles_plateau.append(row(4, score=8.0, delta_e=5.0))
    verdict = should_stop(cycles_plateau, 10, cfg)
    assert verdict.stop is True


def test_delta_e_branch_stops_after_hold_consecutive():
    cfg = make_cfg(conv_hold=3, conv_min_cycles=3)
    cycles = [
        row(1, delta_e=1.0, score=10.0),
        row(2, delta_e=1.0, score=9.0),
    ]
    assert should_stop(cycles, 10, cfg).stop is False
    cycles.append(row(3, delta_e=1.0, score=8.0))
    verdict = should_stop(cycles, 10, cfg)
    assert verdict.stop is True
    assert verdict.reason == "converged_delta_e"


def test_delta_e_at_threshold_continues():
    cfg = make_cfg(conv_hold=3, conv_min_cycles=3)
    cycles = [
        row(1, delta_e=1.0, score=10.0),
        row(2, delta_e=cfg.conv_delta_e, score=9.0),
        row(3, delta_e=1.0, score=8.0),
    ]
    verdict = should_stop(cycles, 10, cfg)
    assert verdict.stop is False


def test_delta_e_break_in_streak_resets_hold():
    cfg = make_cfg(conv_hold=3, conv_min_cycles=3)
    # A noise spike in the third-to-last row breaks the trailing streak even
    # though rows 1, 2, and 4 are all under threshold.
    cycles = [
        row(1, delta_e=1.0, score=10.0),
        row(2, delta_e=1.0, score=9.0),
        row(3, delta_e=cfg.conv_delta_e + 1.0, score=8.0),
        row(4, delta_e=1.0, score=7.0),
    ]
    verdict = should_stop(cycles, 10, cfg)
    assert verdict.stop is False


def test_plateau_branch_stops_when_improvement_at_epsilon():
    cfg = make_cfg(conv_window=2, conv_epsilon=0.5, conv_min_cycles=3)
    cycles = [
        row(1, score=8.5, delta_e=5.0),
        row(2, score=10.0, delta_e=5.0),
        row(3, score=8.0, delta_e=5.0),
        row(4, score=9.0, delta_e=5.0),
    ]
    verdict = should_stop(cycles, 10, cfg)
    assert verdict.stop is True
    assert verdict.reason == "converged_plateau"


def test_plateau_improvement_above_epsilon_continues():
    cfg = make_cfg(conv_window=2, conv_epsilon=0.5, conv_min_cycles=3)
    cycles = [
        row(1, score=10.0, delta_e=5.0),
        row(2, score=9.0, delta_e=5.0),
        row(3, score=8.0, delta_e=5.0),
    ]
    # improvement = best(earlier) - best(all) = 9 - 8 = 1.0 > epsilon.
    verdict = should_stop(cycles, 10, cfg)
    assert verdict.stop is False


def test_plateau_needs_window_plus_one_completed():
    cfg = make_cfg(conv_window=3, conv_epsilon=0.5, conv_min_cycles=3)
    cycles = [
        row(1, score=8.5, delta_e=5.0),
        row(2, score=10.0, delta_e=5.0),
        row(3, score=9.0, delta_e=5.0),
    ]
    assert should_stop(cycles, 10, cfg).stop is False
    cycles.append(row(4, score=8.0, delta_e=5.0))
    verdict = should_stop(cycles, 10, cfg)
    assert verdict.stop is True


def test_or_of_branches():
    cfg = make_cfg(conv_min_cycles=3)

    # Branch A alone (delta-e) stops.
    cycles_a = [
        row(1, delta_e=1.0, score=10.0),
        row(2, delta_e=1.0, score=10.0),
        row(3, delta_e=1.0, score=10.0),
    ]
    verdict_a = should_stop(cycles_a, 10, cfg)
    assert verdict_a.stop is True
    assert verdict_a.reason == "converged_delta_e"

    # Branch B alone (plateau) stops, with delta-e above threshold.
    cycles_b = [
        row(1, delta_e=5.0, score=8.5),
        row(2, delta_e=5.0, score=10.0),
        row(3, delta_e=5.0, score=8.0),
        row(4, delta_e=5.0, score=9.0),
        row(5, delta_e=5.0, score=8.0),
        row(6, delta_e=5.0, score=8.5),
    ]
    verdict_b = should_stop(cycles_b, 10, cfg)
    assert verdict_b.stop is True
    assert verdict_b.reason == "converged_plateau"


def test_budget_checked_before_convergence():
    cfg = make_cfg(conv_min_cycles=3, conv_hold=3)
    budget = 3
    cycles = [
        row(1, delta_e=1.0, score=10.0),
        row(2, delta_e=1.0, score=9.0),
        row(3, delta_e=1.0, score=8.0),
    ]
    verdict = should_stop(cycles, budget, cfg)
    assert verdict.stop is True
    assert verdict.reason == "budget"
