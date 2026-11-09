# ABOUTME: Entry point for rig-service -- loads config, selects the relay driver and camera
# ABOUTME: (real GPIO/V4l2 by default, fake via RIG_DRIVER=fake), and serves the app with uvicorn.

import io
import os
from pathlib import Path

import uvicorn
from PIL import Image

from rig_service.app import create_app
from rig_service.config import RigConfig, load_config
from rig_service.store import Store

DEFAULT_CONFIG = Path(__file__).parents[2] / "config" / "rig.json"
DEFAULT_DATA_DIR = Path(__file__).parents[2] / "data"
DEFAULT_CONSOLE_DIR = Path(__file__).parents[2] / "console"

# The BRIO's measured power-loss reversion (Build Log 2026-07-13): every control the shipped
# config/rig.json locks, set to its firmware-default value. Deliberately different from that
# lock set so a fake-mode boot exercises the same reset failure mode the real camera has and
# startup's CameraLock.apply() is observable, not vacuous.
FAKE_FIRMWARE_DEFAULTS = {
    "auto_exposure": 3,
    "exposure_time_absolute": 625,
    "gain": 16,
    "white_balance_automatic": 1,
    "white_balance_temperature": 4000,
    "focus_automatic_continuous": 1,
    "focus_absolute": 0,
    "backlight_compensation": 1,
    "power_line_frequency": 60,
}


def _solid_gray_jpeg(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (128, 128, 128)).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def select_driver(gpios: list[int]):
    if os.environ.get("RIG_DRIVER") == "fake":
        from rig_service.drivers.fake import FakeRelayDriver

        return FakeRelayDriver()
    from rig_service.drivers.gpio import GpioRelayDriver

    return GpioRelayDriver(gpios=gpios)


def select_camera(config: RigConfig):
    if os.environ.get("RIG_DRIVER") == "fake":
        from rig_service.drivers.fake import FakeCamera

        # A decodable frame sized to the configured camera resolution, so a fake-mode
        # cycle can complete off-Pi -- the configured patch always fits (FakeCamera's
        # own default frame=b"" would abort every MEASURE, and a frame smaller than
        # the shipped camera resolution would abort at MEASURE the same way).
        frame = _solid_gray_jpeg(config.camera.width, config.camera.height)
        return FakeCamera(defaults=FAKE_FIRMWARE_DEFAULTS, frame=frame)
    from rig_service.drivers.camera import V4l2Camera

    return V4l2Camera(config.camera.device, config.camera.width, config.camera.height)


def main() -> None:
    config = load_config(os.environ.get("RIG_CONFIG", DEFAULT_CONFIG))
    driver = select_driver(config.all_gpios())
    camera = select_camera(config)
    data_dir = Path(os.environ.get("RIG_DATA_DIR", DEFAULT_DATA_DIR))
    frames_dir = data_dir / "frames"
    backups_dir = data_dir / "backups"
    frames_dir.mkdir(parents=True, exist_ok=True)
    backups_dir.mkdir(parents=True, exist_ok=True)
    store = Store(data_dir / "rig.db", frames_dir, backups_dir)
    # console_dir is wired only when the directory exists: StaticFiles rejects a missing
    # directory and would crash boot, so when console/ is absent the console routes stay
    # unwired and return 404.
    console_dir = DEFAULT_CONSOLE_DIR if DEFAULT_CONSOLE_DIR.is_dir() else None
    app = create_app(
        config=config, driver=driver, camera=camera, store=store, console_dir=console_dir
    )
    # An in-flight MJPEG stream otherwise blocks uvicorn's graceful shutdown
    # indefinitely; 5s bounds it.
    uvicorn.run(
        app, host="0.0.0.0", port=8000, timeout_graceful_shutdown=5
    )


if __name__ == "__main__":
    main()
