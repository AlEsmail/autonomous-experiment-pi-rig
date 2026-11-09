# ABOUTME: Stop-condition logic for the scientist loop: budget exhaustion, delta-e streak,
# ABOUTME: and score-plateau detection over completed experiment cycles.

from dataclasses import dataclass

from scientist.settings import ScientistConfig


@dataclass(frozen=True)
class StopVerdict:
    stop: bool
    reason: str | None   # "budget" | "converged_delta_e" | "converged_plateau" | None


def should_stop(cycles: list[dict], cycle_budget: int, cfg: ScientistConfig) -> StopVerdict:
    """cycles = scientist-view history rows, ascending n. Budget: len(cycles) >=
    cycle_budget (ALL rows, aborted included -- budget bounds attempts). Convergence uses
    completed rows only, completed == (row["aborted"] is None) -- a post-MEASURE abort
    carries delta_e/score but is NOT completed (decision 3); floor: completed count >=
    conv_min_cycles. Branch A: the last conv_hold completed rows EACH have
    delta_e < conv_delta_e (strict; per-cycle -- decision 3). Branch B:
    improvement = min(score of completed[:-conv_window]) - min(score of all completed)
    (score lower-is-better, so improvement >= 0); stop when improvement <= conv_epsilon;
    requires len(completed) >= conv_window + 1. Budget checked first; then A; then B."""

    # Budget counts every row (attempts), including aborted ones.
    if len(cycles) >= cycle_budget:
        return StopVerdict(True, "budget")

    completed = [row for row in cycles if row.get("aborted") is None]

    if len(completed) < cfg.conv_min_cycles:
        return StopVerdict(False, None)

    # Branch A: per-cycle delta-e streak over the last conv_hold completed rows.
    if len(completed) >= cfg.conv_hold:
        last_hold = completed[-cfg.conv_hold:]
        if all(row["delta_e"] < cfg.conv_delta_e for row in last_hold):
            return StopVerdict(True, "converged_delta_e")

    # Branch B: plateau -- no score improvement > conv_epsilon over conv_window cycles.
    if len(completed) >= cfg.conv_window + 1:
        best_earlier = min(row["score"] for row in completed[:-cfg.conv_window])
        best_all = min(row["score"] for row in completed)
        improvement = best_earlier - best_all
        if improvement <= cfg.conv_epsilon:
            return StopVerdict(True, "converged_plateau")

    return StopVerdict(False, None)
