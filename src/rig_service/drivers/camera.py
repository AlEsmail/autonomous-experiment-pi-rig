# ABOUTME: Real camera driver: captures MJPEG stills and reads/applies v4l2 controls by
# ABOUTME: shelling out to v4l2-ctl. Refuses to load on machines without v4l2 (use FakeCamera).

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

# ~10x a normal ~1s capture (USB re-enumeration margin), under the 60s measure/blank
# phase budgets so a stuck call returns before the phase watchdog fires.
CAPTURE_TIMEOUT_S = 15


class V4l2Camera:
    def __init__(
        self, device: str, width: int, height: int, warmup_frames: int = 5
    ) -> None:
        if shutil.which("v4l2-ctl") is None:
            raise RuntimeError(
                "V4l2Camera requires v4l2-ctl (Linux with v4l-utils); "
                "use FakeCamera (RIG_DRIVER=fake) for development"
            )
        self._device = device
        self._width = width
        self._height = height
        self._warmup_frames = warmup_frames

    def _run(self, *args: str) -> str:
        try:
            result = subprocess.run(
                ["v4l2-ctl", "-d", self._device, *args],
                capture_output=True,
                text=True,
                check=True,
                timeout=CAPTURE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"v4l2-ctl timed out after {CAPTURE_TIMEOUT_S}s on {self._device}: {' '.join(args)}"
            ) from exc
        return result.stdout

    def apply_controls(self, controls: dict[str, int]) -> None:
        # One set-ctrl call per control, in dict order: auto toggles must be
        # switched to manual before their dependent absolute controls accept
        # values (v4l2 marks them "inactive" while auto is on).
        for name, value in controls.items():
            self._run("--set-ctrl", f"{name}={value}")

    def read_controls(self, names: Iterable[str]) -> dict[str, int]:
        output = self._run("--get-ctrl", ",".join(names))
        controls: dict[str, int] = {}
        for line in output.splitlines():
            name, _, rest = line.partition(":")
            if rest:
                # menu controls print e.g. "auto_exposure: 1 (Manual Mode)"
                controls[name.strip()] = int(rest.split()[0])
        return controls

    def capture_jpeg(self) -> bytes:
        with tempfile.TemporaryDirectory() as tmp:
            frame_path = Path(tmp) / "frame.jpg"
            self._run(
                f"--set-fmt-video=width={self._width},height={self._height},pixelformat=MJPG",
                "--stream-mmap",
                "--stream-count=1",
                f"--stream-skip={self._warmup_frames}",
                f"--stream-to={frame_path}",
            )
            return frame_path.read_bytes()
