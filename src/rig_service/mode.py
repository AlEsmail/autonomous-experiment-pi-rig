# ABOUTME: Rig-level mode arbitration (auto/manual/paused), owned by rig-service: decides
# ABOUTME: which client may command doses right now; watchdog/budget trips force paused.

from typing import Callable


MODES = frozenset({"auto", "manual", "paused"})
CLIENTS = frozenset({"scientist", "console"})
COMMANDS = frozenset({"pump", "cycle", "calibrate"})


class ModeError(Exception):
    pass


class ModeArbiter:
    def __init__(self, on_trip: Callable[[str], None] | None = None) -> None:
        self._mode = "paused"
        self._trip_reason: str | None = None
        self._on_trip = on_trip

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def trip_reason(self) -> str | None:
        return self._trip_reason

    def set_mode(self, mode: str, client: str) -> None:
        if client != "console":
            raise ModeError(f"only the console may set mode, not {client!r}")
        if mode not in MODES:
            raise ModeError(f"unknown mode {mode!r}; valid: {sorted(MODES)}")
        self._mode = mode
        self._trip_reason = None

    def trip(self, reason: str) -> None:
        self._mode = "paused"
        self._trip_reason = reason
        if self._on_trip is not None:
            self._on_trip(reason)

    def authorize(self, client: str, command: str) -> None:
        if client not in CLIENTS:
            raise ModeError(f"unknown client {client!r}; valid: {sorted(CLIENTS)}")
        if command not in COMMANDS:
            raise ModeError(f"unknown command {command!r}; valid: {sorted(COMMANDS)}")
        if self._mode == "paused":
            raise ModeError(f"mode is paused; {command} rejected")
        if command in ("pump", "calibrate"):
            if client != "console" or self._mode != "manual":
                raise ModeError(
                    f"{command} requires console in manual mode "
                    f"(client={client!r}, mode={self._mode!r})"
                )
        else:  # cycle
            allowed = (client == "scientist" and self._mode == "auto") or (
                client == "console" and self._mode == "manual"
            )
            if not allowed:
                raise ModeError(
                    f"cycle requires scientist in auto or console in manual mode "
                    f"(client={client!r}, mode={self._mode!r})"
                )
