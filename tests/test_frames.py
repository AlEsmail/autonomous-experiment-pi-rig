# ABOUTME: Tests for the frame cache (FrameSource) and the single-lock camera wrapper
# ABOUTME: (SharedCamera) behind the console's live-video and per-cycle-still endpoints.

import asyncio
import io
import logging
import threading
import time

import pytest
from PIL import Image

from rig_service.drivers.fake import FakeCamera
from rig_service.frames import FrameSource, SharedCamera, live_capture, mjpeg_stream
from rig_service.patch import PatchSpec

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend():
    return "asyncio"


def _solid_jpeg(color=(100, 150, 200), size=128):
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _box_edges(img, left, top, right, bottom):
    """Yield every pixel on the perimeter of the rectangle (right/bottom exclusive)."""
    for x in range(left, right):
        yield img.getpixel((x, top))
        yield img.getpixel((x, bottom - 1))
    for y in range(top, bottom):
        yield img.getpixel((left, y))
        yield img.getpixel((right - 1, y))


def _patch_box(patch: PatchSpec, width: int, height: int):
    """Mirror patch_box math so tests are self-contained."""
    left = round(patch.cx * width - patch.size_px / 2)
    top = round(patch.cy * height - patch.size_px / 2)
    return left, top, left + patch.size_px, top + patch.size_px


def _extract_jpeg_from_part(part: bytes) -> bytes:
    """Pull the JPEG payload out of a multipart/x-mixed-replace part."""
    header_end = part.find(b"\r\n\r\n")
    assert header_end != -1, "multipart part missing header/body separator"
    payload = part[header_end + 4 :]
    if payload.endswith(b"\r\n"):
        payload = payload[:-2]
    return payload


# --- FrameSource ---


def test_frame_source_latest_and_age_none_before_publish():
    frames = FrameSource()

    assert frames.latest is None
    assert frames.age_s is None


def test_frame_source_seq_increments_per_publish():
    frames = FrameSource()

    # Kills a seq-initialized-at-1 mutation: seq is 0 until the first publish.
    assert frames.seq == 0

    frames.publish(b"a")
    assert frames.seq == 1

    frames.publish(b"b")
    assert frames.seq == 2


def test_frame_source_age_tracks_clock():
    now = [10.0]
    frames = FrameSource(clock=lambda: now[0])

    frames.publish(b"frame")
    assert frames.age_s == pytest.approx(0.0)

    now[0] = 13.25
    assert frames.age_s == pytest.approx(3.25)

    # A republish resets the age against the clock at that moment.
    now[0] = 20.0
    frames.publish(b"frame-2")
    assert frames.age_s == pytest.approx(0.0)


# --- SharedCamera: publish-on-capture, control passthrough ---


def test_shared_camera_publishes_on_capture():
    camera = FakeCamera(defaults={}, frame=b"jpeg-bytes")
    frames = FrameSource()
    shared = SharedCamera(camera, frames)

    result = shared.capture_jpeg()

    assert result == b"jpeg-bytes"
    assert frames.latest == b"jpeg-bytes"
    assert frames.seq == 1


def test_shared_camera_control_passthrough():
    camera = FakeCamera(defaults={"gain": 32})
    frames = FrameSource()
    shared = SharedCamera(camera, frames)

    shared.apply_controls({"gain": 10})

    assert shared.read_controls(["gain"]) == {"gain": 10}


# --- SharedCamera: D1's all-operations lock ---


class TrackingCamera:
    """Fake camera whose every operation sleeps briefly and appends unlocked
    "enter"/"exit" markers to a shared list -- proves (or disproves) that
    SharedCamera's lock keeps calls issued from two real threads from ever
    overlapping. list.append is the only synchronization used here (atomic
    under the GIL), so the marker sequence reflects real thread interleaving
    rather than being hidden by a lock of the test's own making."""

    def __init__(self, delay_s: float = 0.03, frame: bytes = b"jpeg") -> None:
        self._delay_s = delay_s
        self._frame = frame
        self.events: list[str] = []

    def _record(self, result):
        self.events.append("enter")
        time.sleep(self._delay_s)
        self.events.append("exit")
        return result

    def capture_jpeg(self) -> bytes:
        return self._record(self._frame)

    def apply_controls(self, controls: dict[str, int]) -> None:
        self._record(None)

    def read_controls(self, names):
        return self._record({name: 0 for name in names})


def _max_overlap(events: list[str]) -> int:
    depth = 0
    peak = 0
    for event in events:
        if event == "enter":
            depth += 1
            peak = max(peak, depth)
        else:
            depth -= 1
    return peak


def test_shared_camera_serializes_concurrent_captures():
    camera = TrackingCamera(delay_s=0.03)
    frames = FrameSource()
    shared = SharedCamera(camera, frames)

    threads = [threading.Thread(target=shared.capture_jpeg) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(camera.events) == 4
    assert _max_overlap(camera.events) == 1


def test_shared_camera_serializes_capture_against_control_read():
    # Mutation kill: the D1 lock covers ALL three operations, not just capture.
    # A capture-only lock would let this read_controls call run concurrently
    # inside the capture's sleep, pushing overlap depth to 2.
    camera = TrackingCamera(delay_s=0.03)
    frames = FrameSource()
    shared = SharedCamera(camera, frames)

    capture_thread = threading.Thread(target=shared.capture_jpeg)
    read_thread = threading.Thread(target=lambda: shared.read_controls(["gain"]))
    capture_thread.start()
    read_thread.start()
    capture_thread.join()
    read_thread.join()

    assert len(camera.events) == 4
    assert _max_overlap(camera.events) == 1


# --- live_capture loop ---


async def test_live_capture_publishes_frames_periodically():
    camera = FakeCamera(defaults={}, frame=b"frame")
    frames = FrameSource()
    shared = SharedCamera(camera, frames)

    task = asyncio.create_task(live_capture(shared, interval_s=0.01))
    await asyncio.sleep(0.06)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # No test code ever called shared.capture_jpeg() directly -- every publish
    # observed here came from the loop itself running unattended.
    assert frames.seq >= 2


async def test_live_capture_survives_capture_errors(caplog):
    class FlakyCamera:
        def __init__(self) -> None:
            self.calls = 0

        def capture_jpeg(self) -> bytes:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("simulated capture failure")
            return b"frame"

        def apply_controls(self, controls: dict[str, int]) -> None:
            pass

        def read_controls(self, names):
            return {}

    camera = FlakyCamera()
    frames = FrameSource()
    shared = SharedCamera(camera, frames)

    with caplog.at_level(logging.WARNING, logger="rig_service.frames"):
        task = asyncio.create_task(live_capture(shared, interval_s=0.01))
        await asyncio.sleep(0.06)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "simulated capture failure" in warnings[0].getMessage()

    # The loop kept running (and publishing) past the single failure -- never
    # task-fatal (D2).
    assert camera.calls >= 2
    assert frames.seq >= 2


async def test_stream_mjpg_yields_new_frames_only():
    # The /stream.mjpg route serves this generator. An in-process ASGI test
    # transport (TestClient and httpx.ASGITransport alike) runs the app to
    # completion before handing back a buffered response, so it cannot drive an
    # endless stream -- the emit-only-on-new-frame logic is proved by iterating the
    # generator directly. Kills "yield on every tick" and "yield more than once per
    # publish": an idle tick (no seq change) must yield nothing.
    frames = FrameSource()
    marker = b"\xff\xd8\xff"
    # T5: signature ported to pass a live patch source.
    gen = mjpeg_stream(
        frames,
        patch_source=lambda: PatchSpec(cx=0.5, cy=0.5, size_px=20),
        poll_interval_s=0.01,
    )
    try:
        frames.publish(marker + b"AAA")
        part1 = await asyncio.wait_for(gen.__anext__(), timeout=2)
        assert part1.count(marker) == 1
        assert b"AAA" in part1
        assert b"Content-Type: image/jpeg" in part1
        assert f"Content-Length: {len(marker) + 3}".encode("ascii") in part1

        frames.publish(marker + b"BBB")
        part2 = await asyncio.wait_for(gen.__anext__(), timeout=2)
        assert part2.count(marker) == 1
        assert b"BBB" in part2
        assert b"AAA" not in part2

        # No new publish: idle ticks must not yield. A "send every tick" mutation
        # would emit a duplicate here; the correct generator blocks and this times
        # out. (The timeout cancels the pending __anext__, which finalizes the
        # generator -- fine, this is the last read before aclose.)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(gen.__anext__(), timeout=0.1)
    finally:
        await gen.aclose()


# --- draw_patch_overlay (T5) ---


def test_draw_patch_overlay_returns_decodable_jpeg_with_box_drawn():
    # draw_patch_overlay does not exist yet in the current src/ -- this import is
    # intentionally local so the test file collects and the test fails here.
    from rig_service.frames import draw_patch_overlay

    jpeg = _solid_jpeg(color=(100, 150, 200), size=128)
    patch = PatchSpec(cx=0.5, cy=0.5, size_px=20)
    overlay = draw_patch_overlay(jpeg, patch)

    img = Image.open(io.BytesIO(overlay))
    assert img.size == (128, 128)

    left, top, right, bottom = _patch_box(patch, 128, 128)
    bg = (100, 150, 200)
    edge_pixels = list(_box_edges(img, left, top, right, bottom))
    assert any(
        any(abs(a - b) > 30 for a, b in zip(pixel, bg)) for pixel in edge_pixels
    ), "box edge pixels should differ from the uniform background"


def test_draw_patch_overlay_stroke_at_least_3px():
    from rig_service.frames import draw_patch_overlay

    jpeg = _solid_jpeg(color=(60, 60, 60), size=128)
    patch = PatchSpec(cx=0.5, cy=0.5, size_px=20)
    overlay = draw_patch_overlay(jpeg, patch)
    img = Image.open(io.BytesIO(overlay))

    left, top, right, bottom = _patch_box(patch, 128, 128)
    bg = (60, 60, 60)

    # At least the pixels 0, 1, 2 pixels inside and outside the nominal box edge
    # should differ from the background -- killing a stroke-width mutation.
    for offset in (0, 1, 2):
        for x in range(left + offset, right - offset):
            for y in (top + offset, bottom - 1 - offset):
                if any(abs(a - b) > 30 for a, b in zip(img.getpixel((x, y)), bg)):
                    break
            else:
                pytest.fail(f"no stroke visible at offset {offset} on horizontal edge")
        for y in range(top + offset, bottom - offset):
            for x in (left + offset, right - 1 - offset):
                if any(abs(a - b) > 30 for a, b in zip(img.getpixel((x, y)), bg)):
                    break
            else:
                pytest.fail(f"no stroke visible at offset {offset} on vertical edge")


def test_draw_patch_overlay_undecodable_input_returns_raw_bytes():
    from rig_service.frames import draw_patch_overlay

    garbage = b"this is not a jpeg"
    assert draw_patch_overlay(garbage, PatchSpec(cx=0.5, cy=0.5, size_px=20)) == garbage


# --- mjpeg_stream with live patch_source (T5) ---


async def test_mjpeg_stream_parts_carry_overlay():
    from rig_service.frames import draw_patch_overlay

    frames = FrameSource()
    patch = PatchSpec(cx=0.5, cy=0.5, size_px=20)
    gen = mjpeg_stream(frames, patch_source=lambda: patch, poll_interval_s=0.01)
    try:
        raw = _solid_jpeg(color=(80, 120, 160), size=128)
        frames.publish(raw)
        part = await asyncio.wait_for(gen.__anext__(), timeout=2)

        jpeg = _extract_jpeg_from_part(part)
        assert jpeg != raw, "streamed frame should be annotated"
        img = Image.open(io.BytesIO(jpeg))
        assert img.size == (128, 128)

        left, top, right, bottom = _patch_box(patch, 128, 128)
        bg = (80, 120, 160)
        edge_pixels = list(_box_edges(img, left, top, right, bottom))
        assert any(
            any(abs(a - b) > 30 for a, b in zip(pixel, bg)) for pixel in edge_pixels
        )
    finally:
        await gen.aclose()


async def test_mjpeg_stream_undecodable_frame_serves_raw():
    frames = FrameSource()
    gen = mjpeg_stream(
        frames,
        patch_source=lambda: PatchSpec(cx=0.5, cy=0.5, size_px=20),
        poll_interval_s=0.01,
    )
    try:
        garbage = b"not a jpeg"
        frames.publish(garbage)
        part = await asyncio.wait_for(gen.__anext__(), timeout=2)
        assert _extract_jpeg_from_part(part) == garbage
    finally:
        await gen.aclose()


async def test_mjpeg_stream_overlay_tracks_live_patch():
    frames = FrameSource()
    patches = [PatchSpec(cx=0.25, cy=0.25, size_px=20)]
    gen = mjpeg_stream(frames, patch_source=lambda: patches[0], poll_interval_s=0.01)
    try:
        raw = _solid_jpeg(color=(80, 120, 160), size=128)

        frames.publish(raw)
        part1 = await asyncio.wait_for(gen.__anext__(), timeout=2)
        img1 = Image.open(io.BytesIO(_extract_jpeg_from_part(part1)))

        # Rebind the live patch source to a new, non-overlapping region.
        patches[0] = PatchSpec(cx=0.75, cy=0.75, size_px=20)
        frames.publish(raw)
        part2 = await asyncio.wait_for(gen.__anext__(), timeout=2)
        img2 = Image.open(io.BytesIO(_extract_jpeg_from_part(part2)))

        old_box = _patch_box(PatchSpec(cx=0.25, cy=0.25, size_px=20), 128, 128)
        new_box = _patch_box(PatchSpec(cx=0.75, cy=0.75, size_px=20), 128, 128)
        bg = (80, 120, 160)

        # Old box location should have returned to the uniform background.
        old_edges = list(_box_edges(img2, *old_box))
        assert all(
            all(abs(a - b) <= 30 for a, b in zip(pixel, bg)) for pixel in old_edges
        ), "old patch box should not be annotated after rebind"

        # New box location should be drawn.
        new_edges = list(_box_edges(img2, *new_box))
        assert any(
            any(abs(a - b) > 30 for a, b in zip(pixel, bg)) for pixel in new_edges
        ), "new patch box should be annotated after rebind"
    finally:
        await gen.aclose()


async def test_draw_patch_overlay_dispatched_off_thread_in_mjpeg_stream(monkeypatch):
    # Mutation kill: mjpeg_stream must call draw_patch_overlay via asyncio.to_thread.
    # A synchronous call would stall the event loop while the draw blocks; here the
    # draw blocks on a threading.Event and we prove the loop stays responsive.
    entered = threading.Event()
    release = threading.Event()

    def blocking_draw(jpeg: bytes, patch: PatchSpec) -> bytes:
        entered.set()
        release.wait(timeout=5)
        return jpeg

    monkeypatch.setattr("rig_service.frames.draw_patch_overlay", blocking_draw)

    frames = FrameSource()
    gen = mjpeg_stream(
        frames,
        patch_source=lambda: PatchSpec(cx=0.5, cy=0.5, size_px=20),
        poll_interval_s=0.01,
    )

    try:
        frames.publish(b"frame")
        anext_task = asyncio.create_task(gen.__anext__())
        start = time.monotonic()

        # Wait until the blocking draw has been entered in a worker thread.
        deadline = time.monotonic() + 2.0
        while not entered.is_set():
            assert time.monotonic() < deadline, "draw_patch_overlay was never entered"
            await asyncio.sleep(0.001)

        # While the worker thread is blocked, the event loop must stay responsive.
        async def heartbeat():
            for _ in range(5):
                await asyncio.sleep(0.01)

        await asyncio.wait_for(heartbeat(), timeout=0.5)

        # Release the worker and confirm the stream emits the part.
        release.set()
        part = await asyncio.wait_for(anext_task, timeout=2)
        elapsed = time.monotonic() - start
        assert b"frame" in part
        assert elapsed < 1.5, f"off-thread draw should unblock quickly, took {elapsed:.3f}s"
    finally:
        await gen.aclose()


async def test_live_capture_cancel_during_capture_propagates_cancelled_error():
    # Deterministic (not scheduling-lucky) mutation kill for "catches Exception,
    # not BaseException" (T1 pinned docstring). The fake camera blocks on a
    # threading.Event so the test can cancel the task at the exact moment
    # execution is inside the wrapped asyncio.to_thread(camera.capture_jpeg)
    # call, then release the block. Under the correct code the thrown-in
    # CancelledError is not caught there and propagates out of live_capture.
    # Under an `except BaseException` regression it would be caught, logged,
    # and swallowed -- the loop would keep running (capture_jpeg no longer
    # blocks once released) and the task would never complete, so this test
    # bounds the wait instead of using `await task` directly: a regression
    # fails fast (pending after 1 s) rather than hanging the suite.
    entered = threading.Event()
    release = threading.Event()

    class BlockingCamera:
        def capture_jpeg(self) -> bytes:
            entered.set()
            release.wait(timeout=5)
            return b"frame"

        def apply_controls(self, controls: dict[str, int]) -> None:
            pass

        def read_controls(self, names):
            return {}

    camera = BlockingCamera()
    frames = FrameSource()
    shared = SharedCamera(camera, frames)

    task = asyncio.create_task(live_capture(shared, interval_s=0.01))

    deadline = time.monotonic() + 2.0
    while not entered.is_set():
        assert time.monotonic() < deadline, "capture_jpeg was never entered"
        await asyncio.sleep(0.001)

    # The task is now guaranteed suspended inside the wrapped to_thread call
    # (the background thread has started and is blocked on release.wait()).
    # Cancelling here -- before releasing the block -- means the underlying
    # executor future can't be cancelled outright (it's already running), so
    # asyncio guarantees the CancelledError is thrown in at the very next
    # resumption of that same await point once the thread returns, not at
    # some other, unprotected await (e.g. the trailing asyncio.sleep).
    task.cancel()
    release.set()

    done, pending = await asyncio.wait({task}, timeout=1.0)

    if pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        pytest.fail("live_capture did not complete within 1s of cancellation")

    with pytest.raises(asyncio.CancelledError):
        await task
