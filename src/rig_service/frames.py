# ABOUTME: Frame cache (FrameSource) and single-lock camera wrapper (SharedCamera) behind
# ABOUTME: the console's live-video stream and per-cycle-still endpoints.

from __future__ import annotations

import asyncio
import io
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterable

from PIL import Image, ImageDraw

from rig_service.patch import PatchSpec, patch_box

LIVE_CAPTURE_INTERVAL_S = 1.0
STREAM_POLL_INTERVAL_S = 0.2
STREAM_BOUNDARY = "frame"

logger = logging.getLogger(__name__)


class FrameSource:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._latest: bytes | None = None
        self._seq = 0
        self._published_at: float | None = None

    def publish(self, jpeg: bytes) -> None:
        self._latest = jpeg
        self._seq += 1
        self._published_at = self._clock()

    @property
    def latest(self) -> bytes | None:
        return self._latest

    @property
    def seq(self) -> int:
        return self._seq

    @property
    def age_s(self) -> float | None:
        if self._published_at is None:
            return None
        return self._clock() - self._published_at


class SharedCamera:
    """One lock over ALL camera operations (capture_jpeg, apply_controls,
    read_controls) -- exactly one thread touches the device at a time (D1).
    Publishes every successful capture to the FrameSource."""

    def __init__(self, camera, frames: FrameSource) -> None:
        self._camera = camera
        self._frames = frames
        self._lock = threading.Lock()

    def capture_jpeg(self) -> bytes:
        with self._lock:
            jpeg = self._camera.capture_jpeg()
            self._frames.publish(jpeg)
        return jpeg

    def apply_controls(self, controls: dict[str, int]) -> None:
        with self._lock:
            self._camera.apply_controls(controls)

    def read_controls(self, names: Iterable[str]) -> dict[str, int]:
        with self._lock:
            return self._camera.read_controls(names)


OVERLAY_STROKE_PX = 4
OVERLAY_COLOR = (255, 0, 255)  # magenta, high-contrast against cup backgrounds


def draw_patch_overlay(jpeg: bytes, patch: PatchSpec) -> bytes:
    """Serve-time annotation: decode, draw the patch bounding box (PIL rectangle,
    OVERLAY_STROKE_PX stroke, a color contrasting the cup region), re-encode JPEG.

    BEST-EFFORT: any decode/annotate failure returns the ORIGINAL bytes unchanged.
    """
    try:
        image = Image.open(io.BytesIO(jpeg))
        width, height = image.size
        box = patch_box(patch, width, height)
        ImageDraw.Draw(image).rectangle(box, outline=OVERLAY_COLOR, width=OVERLAY_STROKE_PX)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=95)
        return buf.getvalue()
    except Exception:
        return jpeg


async def live_capture(camera: SharedCamera, interval_s: float) -> None:
    """Capture loop body: capture via asyncio.to_thread (NEVER a bare synchronous
    call -- on the Pi a capture takes ~1 s and would stall the event loop, and no
    Windows test can observe the difference against an instant fake; this is a
    pinned property verified on-rig, not by test), sleep interval_s, repeat until
    cancelled. A capture Exception is logged (logger.warning) and the loop
    continues -- never task-fatal (D2); catches Exception, not BaseException, so
    CancelledError escapes and cancellation stays clean."""
    while True:
        try:
            await asyncio.to_thread(camera.capture_jpeg)
        except Exception as exc:
            logger.warning("live capture failed: %s", exc)
        await asyncio.sleep(interval_s)


async def mjpeg_stream(
    frames: FrameSource,
    patch_source: Callable[[], PatchSpec],
    poll_interval_s: float = STREAM_POLL_INTERVAL_S,
) -> AsyncIterator[bytes]:
    """Yield one multipart/x-mixed-replace part per new frame: poll the cache every
    poll_interval_s and emit only when frames.seq advances, so each published frame
    reaches a client exactly once and idle ticks send nothing (D2). The /stream.mjpg
    route wraps this in a StreamingResponse; the client's disconnect finalizes the
    generator (aclose). Iterated directly by its test -- an in-process ASGI transport
    buffers the whole response and so cannot drive an endless stream.

    Each emitted frame is annotated at serve time via draw_patch_overlay, using the
    live value from patch_source() at emit time; the cache stays raw (D4).
    """
    last_seq = 0
    while True:
        await asyncio.sleep(poll_interval_s)
        current = frames.seq
        if current == last_seq:
            continue
        last_seq = current
        jpeg = frames.latest
        if jpeg is None:
            continue
        annotated = await asyncio.to_thread(draw_patch_overlay, jpeg, patch_source())
        yield (
            f"--{STREAM_BOUNDARY}\r\n"
            "Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(annotated)}\r\n\r\n"
        ).encode("ascii") + annotated + b"\r\n"
