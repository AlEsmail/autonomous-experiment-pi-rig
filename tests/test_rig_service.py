# ABOUTME: Smoke tests for the rig_service package entry point: importable without side effects,
# ABOUTME: exposing main()/select_driver/select_camera, and the fake-mode boot wiring serving /status.

from fastapi.testclient import TestClient

import rig_service.__main__ as entry
from rig_service.store import Store
from rig_service.volume import VolumeState
from shared.measure import lab_from_jpeg


def test_entry_exposes_main():
    assert callable(entry.main)


def test_fake_driver_selected_by_env(monkeypatch):
    monkeypatch.setenv("RIG_DRIVER", "fake")
    driver = entry.select_driver(gpios=[4, 5])
    from rig_service.drivers.fake import FakeRelayDriver

    assert isinstance(driver, FakeRelayDriver)


def test_select_camera_fake_by_env(monkeypatch):
    monkeypatch.setenv("RIG_DRIVER", "fake")
    from rig_service.config import load_config
    from rig_service.drivers.fake import FakeCamera

    config = load_config(entry.DEFAULT_CONFIG)

    camera = entry.select_camera(config)

    assert isinstance(camera, FakeCamera)
    # Non-vacuous startup lock: the fake's firmware-reset defaults must differ from the
    # config's lock set, or CameraLock.apply() at boot would be a no-op.
    current = camera.read_controls(config.camera.controls.keys())
    assert current != config.camera.controls

    # A fake-mode cycle must be able to complete off-Pi -- the frame must not just decode
    # (FakeCamera's own default frame=b"" would abort every MEASURE) but also fit the
    # shipped config's measurement patch, the actual floor shared.measure.lab_from_jpeg
    # enforces at MEASURE/BLANK.
    frame = camera.capture_jpeg()
    lab_from_jpeg(frame, 0.5, 0.5, config.camera.patch_px)


def test_fake_camera_frame_fits_shipped_patch_px(monkeypatch):
    # Built the way __main__.main() actually builds it, against the real shipped
    # config/rig.json (not a test fixture's smaller patch_px) -- a decodable frame is not
    # enough; shared.measure.lab_from_jpeg rejects any image smaller than patch_px, and
    # every existing /cycle test uses its own patch_px=20 fixture, so this is the only
    # place the shipped config and the fake frame actually meet.
    monkeypatch.setenv("RIG_DRIVER", "fake")
    from rig_service.config import load_config

    config = load_config(entry.DEFAULT_CONFIG)
    camera = entry.select_camera(config)

    lab = lab_from_jpeg(camera.capture_jpeg(), 0.5, 0.5, config.camera.patch_px)

    l, a, b = lab
    assert 40 < l < 65  # mid-gray: neither black nor white
    assert abs(a) < 5
    assert abs(b) < 5


def test_fake_boot_wiring_serves_status(monkeypatch, tmp_path):
    monkeypatch.setenv("RIG_DRIVER", "fake")
    from rig_service.config import load_config

    config = load_config(entry.DEFAULT_CONFIG)
    driver = entry.select_driver(config.all_gpios())
    camera = entry.select_camera(config)
    store = Store(
        str(tmp_path / "rig.db"), str(tmp_path / "frames"), str(tmp_path / "backups")
    )
    app = entry.create_app(config=config, driver=driver, camera=camera, store=store)

    with TestClient(app) as client:
        response = client.get("/status")

    assert response.status_code == 200


def test_main_wiring_builds_store_from_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("RIG_DRIVER", "fake")
    monkeypatch.setenv("RIG_DATA_DIR", str(tmp_path))

    from rig_service.config import load_config

    config = load_config(entry.DEFAULT_CONFIG)
    driver = entry.select_driver(config.all_gpios())
    camera = entry.select_camera(config)

    captured = {}
    original_create_app = entry.create_app

    def fake_create_app(**kwargs):
        captured["app"] = original_create_app(**kwargs)
        captured["kwargs"] = kwargs
        return captured["app"]

    monkeypatch.setattr(entry, "create_app", fake_create_app)
    monkeypatch.setattr(entry.uvicorn, "run", lambda *args, **kwargs: None)

    entry.main()

    assert "store" in captured["kwargs"]
    store = captured["kwargs"]["store"]
    assert isinstance(store, Store)
    assert (tmp_path / "rig.db").exists()
    assert (tmp_path / "frames").exists()
    assert (tmp_path / "backups").exists()

    with TestClient(captured["app"]) as client:
        response = client.get("/status")
    assert response.status_code == 200


def _capture_main_kwargs(monkeypatch, tmp_path):
    """Run main() with uvicorn stubbed and create_app wrapped, returning the kwargs
    it was called with."""
    monkeypatch.setenv("RIG_DRIVER", "fake")
    monkeypatch.setenv("RIG_DATA_DIR", str(tmp_path / "data"))

    captured = {}
    original_create_app = entry.create_app

    def fake_create_app(**kwargs):
        captured["kwargs"] = kwargs
        return original_create_app(**kwargs)

    monkeypatch.setattr(entry, "create_app", fake_create_app)
    monkeypatch.setattr(entry.uvicorn, "run", lambda *args, **kwargs: None)

    entry.main()
    return captured["kwargs"]


def test_main_wires_console_dir_when_present(monkeypatch, tmp_path):
    # __main__.main() passes the repo's console/ dir so the deployed service serves the
    # console (plan T2). Guarded by existence: an existing dir is wired through.
    console_dir = tmp_path / "console"
    console_dir.mkdir()
    monkeypatch.setattr(entry, "DEFAULT_CONSOLE_DIR", console_dir)

    kwargs = _capture_main_kwargs(monkeypatch, tmp_path)

    assert kwargs["console_dir"] == console_dir


def test_main_disables_console_when_dir_absent(monkeypatch, tmp_path):
    # Mutation kill: the .is_dir() guard. Before T4/T5 land console/ the directory does
    # not exist; passing it unguarded would crash boot (StaticFiles rejects a missing
    # dir), so main() passes None and the console routes 404 until the files arrive.
    monkeypatch.setattr(entry, "DEFAULT_CONSOLE_DIR", tmp_path / "nonexistent-console")

    kwargs = _capture_main_kwargs(monkeypatch, tmp_path)

    assert kwargs["console_dir"] is None


def test_main_bounds_graceful_shutdown(monkeypatch, tmp_path):
    # An open MJPEG stream connection must not block process shutdown forever.
    monkeypatch.setenv("RIG_DRIVER", "fake")
    monkeypatch.setenv("RIG_DATA_DIR", str(tmp_path / "data"))

    captured = {}
    original_create_app = entry.create_app

    def fake_create_app(**kwargs):
        captured["kwargs"] = kwargs
        return original_create_app(**kwargs)

    monkeypatch.setattr(entry, "create_app", fake_create_app)

    uvicorn_kwargs = {}

    def fake_uvicorn_run(*args, **kwargs):
        uvicorn_kwargs.update(kwargs)

    monkeypatch.setattr(entry.uvicorn, "run", fake_uvicorn_run)

    entry.main()

    # Must stay below systemd's default 90 s stop timeout so shutdown completes
    # before SIGKILL.
    assert 0 < uvicorn_kwargs["timeout_graceful_shutdown"] < 90
