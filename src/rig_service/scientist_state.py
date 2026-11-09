# ABOUTME: In-memory holder for the scientist's latest reported liveness state.
# ABOUTME: Validates reports, detects stop-verdict transitions, and renders /status snapshots.

import time
from collections.abc import Callable

STATES = frozenset({
    "idle",
    "thinking",
    "waiting",
    "stopped_budget",
    "stopped_converged",
    "llm_error",
    "parse_error",
    "cycle_refused",
    "cycle_failed",
})
STOP_STATES = frozenset({"stopped_budget", "stopped_converged"})
MAX_DETAIL_CHARS = 200


class ScientistState:
    """Holds only the latest scientist report (restart = unknown until the next one).

    update() returns True exactly when the report transitions INTO a stop verdict
    from a different state, so the caller writes one scientist_stop event per
    transition and none on repeat reports.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._state: str | None = None
        self._detail = ""
        self._updated_at: float | None = None

    def update(self, state: str, detail: str = "") -> bool:
        if state not in STATES:
            raise ValueError(f"unknown scientist state: {state!r}")
        if len(detail) > MAX_DETAIL_CHARS:
            raise ValueError(
                f"detail longer than {MAX_DETAIL_CHARS} chars ({len(detail)})"
            )
        transitioned = state in STOP_STATES and state != self._state
        self._state = state
        self._detail = detail
        self._updated_at = self._clock()
        return transitioned

    def snapshot(self) -> dict:
        if self._state is None:
            return {"state": "unknown", "detail": "", "age_s": None}
        return {
            "state": self._state,
            "detail": self._detail,
            "age_s": self._clock() - self._updated_at,
        }
