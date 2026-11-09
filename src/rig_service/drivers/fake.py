# ABOUTME: In-memory fake drivers for off-Pi development and tests: a relay driver tracking
# ABOUTME: on/off state with event history, and a camera holding v4l2-style control state.

from __future__ import annotations

import time
from collections.abc import Iterable


class FakeRelayDriver:
    def __init__(self) -> None:
        self._on: set[int] = set()
        self.history: list[tuple[str, int | None, float]] = []

    def all_off(self) -> None:
        self._on.clear()
        self.history.append(("all_off", None, time.monotonic()))

    def set(self, gpio: int, on: bool) -> None:
        if on:
            self._on.add(gpio)
            self.history.append(("on", gpio, time.monotonic()))
        else:
            self._on.discard(gpio)
            self.history.append(("off", gpio, time.monotonic()))

    def relays_on(self) -> set[int]:
        return set(self._on)


class FakeCamera:
    def __init__(self, defaults: dict[str, int], frame: bytes = b"") -> None:
        self._defaults = dict(defaults)
        self._controls = dict(defaults)
        self.frame = frame

    def apply_controls(self, controls: dict[str, int]) -> None:
        self._controls.update(controls)

    def read_controls(self, names: Iterable[str]) -> dict[str, int]:
        return {name: self._controls[name] for name in names}

    def capture_jpeg(self) -> bytes:
        return self.frame

    def power_cycle(self) -> None:
        # USB webcams revert to firmware defaults when they lose power
        # (measured on the BRIO 4K 2026-07-13) — simulate that world.
        self._controls = dict(self._defaults)
