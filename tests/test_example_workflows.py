"""Offline example workflows: fake devices and windows, real config resolution."""

from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.sensor.camera.geometry import CameraIntrinsics, RGBDGeometry


@pytest.fixture
def diagnostic(monkeypatch, tmp_path):
    import examples.realsense_record_example as diag
    from dexmani_real.sensor.camera import realsense as driver

    real_camera = driver.RealSenseCamera
    state = NS(
        priority=1.0,
        supported=True,
        read_error=False,
        restore=False,
        restore_failure=None,
        configs=[],
        closes=[],
        writes=[],
        window_closes=[],
    )
    intrinsics = CameraIntrinsics(848, 480, 400.0, 400.0, 424.0, 240.0, "none", (0.0,) * 5)
    geometry = RGBDGeometry(intrinsics, intrinsics, np.eye(4))

    class Sensor:
        def get_stream_profiles(self):
            return [NS(stream_type=lambda: driver.rs.stream.color)]

        def supports(self, option):
            return state.supported and not (
                state.restore and state.restore_failure == "unsupported"
            )

        def get_option(self, option):
            if state.read_error:
                raise RuntimeError("unreadable")
            return state.priority

        def set_option(self, option, value):
            if state.restore and state.restore_failure == "write":
                raise OSError("restore write failed")
            if state.restore and state.restore_failure == "mismatch":
                return
            state.priority = value
            state.writes.append(value)

    class Camera:
        active_serial = "requested-camera"

        def __init__(self, config):
            self.config = config
            self.profile = NS(get_device=lambda: NS(query_sensors=lambda: [Sensor()]))
            state.configs.append(config)

        def get_color_sensor(self):
            return real_camera.get_color_sensor(self)

        def connect(self):
            real_camera._apply_color_config(self)
            return True

        def disconnect(self):
            state.closes.append(self)

        def get_device_info(self):
            return {"serial": self.active_serial}

        def get_geometry(self):
            return geometry

        def get_depth_scale(self):
            return 0.001

        def read(self, **kwargs):
            raise RuntimeError("injected read failure")

    monkeypatch.setattr(driver.rs, "context", lambda: pytest.fail("real SDK context created"))
    monkeypatch.setattr(diag, "RealSenseCamera", real_camera, raising=False)
    monkeypatch.setattr(diag, "RealSenseCameraConfig", driver.RealSenseCameraConfig, raising=False)
    monkeypatch.setattr(diag, "rs", driver.rs, raising=False)
    monkeypatch.setattr(driver, "RealSenseCamera", Camera)
    monkeypatch.setattr(diag, "_list_cameras", lambda: [{"serial": Camera.active_serial}])
    monkeypatch.setattr(diag, "CameraExtrinsics", lambda *a: None)
    monkeypatch.setattr(diag, "_compute_base_from_color", lambda *a: np.eye(4))
    monkeypatch.setattr(
        diag,
        "NonBlockingPCDViewer",
        lambda **kw: NS(close=lambda: state.window_closes.append("cloud")),
    )
    monkeypatch.setattr(diag.cv2, "destroyAllWindows", lambda: state.window_closes.append("rgbd"))
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "camera": {
                    "serial": Camera.active_serial,
                    "width": 848,
                    "height": 480,
                    "fps": 15,
                    "warmup_frames": 2,
                    "frame_queue_capacity": 3,
                    "l515_visual_preset": 3,
                    "l515_confidence_threshold": 2,
                },
                "pointcloud": {"remove_table": False},
            }
        )
    )
    return diag, driver, Camera, state, config_path


@pytest.mark.parametrize("original", [0.0, 1.0, None, "unreadable"])
def test_diagnostic_uses_config_and_restores_preexisting_priority(
    diagnostic, monkeypatch, original
):
    diag, _, _, state, path = diagnostic
    state.priority = 1.0 if original in (None, "unreadable") else original
    state.supported = original is not None
    state.read_error = original == "unreadable"

    def live(camera, **kwargs):
        allowed = original not in (None, "unreadable")
        assert kwargs["allow_exposure_changes"] == allowed
        if allowed:
            assert state.priority == 0.0
        # A later readable sensor must not enable changes without an original value.
        state.supported, state.read_error = True, False
        diag._handle_keyboard(
            ord("a"),
            diag.PointCloudDisplayState(),
            NS(),
            camera,
            ExperimentConfig().pointcloud,
            allow_exposure_changes=allowed,
        )
        return "user_exit"

    monkeypatch.setattr(diag, "_run_rgbd_test", live)
    assert diag.main(["--config", str(path)]) == 0
    assert state.priority == (1.0 if original in (None, "unreadable") else original)
    assert len(state.configs) == 2 and len(state.closes) == 3
    for cfg in state.configs:
        assert (cfg.serial, cfg.color_resolution, cfg.depth_resolution, cfg.fps) == (
            "requested-camera",
            (848, 480),
            (848, 480),
            15,
        )
        assert (cfg.warmup_frames, cfg.frame_queue_capacity) == (2, 3)
        assert (
            cfg.l515_depth_config.visual_preset,
            cfg.l515_depth_config.confidence_threshold,
        ) == (3, 2)
        assert cfg.auto_exposure_priority is None
    if original in (None, "unreadable"):
        assert not state.writes


@pytest.mark.parametrize("failure", ["unsupported", "write", "mismatch"])
def test_restore_failure_disconnects(diagnostic, monkeypatch, failure):
    diag, _, _, state, path = diagnostic

    def live(*a, **kw):
        state.restore, state.restore_failure = True, failure
        return "user_exit"

    monkeypatch.setattr(diag, "_run_rgbd_test", live)
    assert diag.main(["--config", str(path)]) == 1
    assert len(state.closes) == 3


@pytest.mark.parametrize("error", [RuntimeError("geometry failed"), KeyboardInterrupt()])
def test_lifecycle_cleanup_on_exception(diagnostic, monkeypatch, error):
    diag, _, camera, state, path = diagnostic

    def fail(self):
        raise error

    monkeypatch.setattr(camera, "get_geometry", fail)
    with pytest.raises(type(error)):
        diag.main(["--config", str(path)])
    assert len(state.closes) == 1
    assert not state.writes


@pytest.mark.parametrize("failure", ["missing", "mismatch", "read", "interrupt"])
def test_live_failure_cleans_windows_and_camera(diagnostic, monkeypatch, failure):
    diag, _, camera, state, path = diagnostic

    def read(self, **kwargs):
        if failure == "interrupt":
            raise KeyboardInterrupt()
        if failure == "read":
            raise OSError("read failed")
        return NS(
            rgb=None if failure == "missing" else np.zeros((2, 2, 3), np.uint8),
            depth_aligned_to_color_raw=np.ones((1, 2), np.uint16),
            depth_aligned_to_color=np.ones((1, 2), np.float32),
        )

    monkeypatch.setattr(camera, "read", read)
    if failure == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            diag.main(["--config", str(path)])
    else:
        assert diag.main(["--config", str(path)]) == 1
    assert state.window_closes == ["cloud", "rgbd"]
    assert len(state.closes) == 3 and state.priority == 1.0


def test_pointcloud_camera_uses_runtime_and_cleans_failed_connect(diagnostic, monkeypatch):
    import examples.pointcloud_process_example as pcd
    from dexmani_real.config.experiment import load_experiment_config

    _, driver, camera, state, path = diagnostic
    monkeypatch.setattr(pcd, "RealSenseCamera", camera, raising=False)
    monkeypatch.setattr(pcd, "RealSenseCameraConfig", driver.RealSenseCameraConfig, raising=False)
    params = load_experiment_config(yaml_path=path).camera
    result = pcd._connect_camera(params)
    assert result.config.serial == params.serial
    assert result.config.fps == 15 and result.config.depth_resolution == (848, 480)
    assert result.config.auto_exposure_priority == 0.0
    monkeypatch.setattr(camera, "connect", lambda self: False)
    with pytest.raises(RuntimeError):
        pcd._connect_camera(params)
    assert len(state.closes) == 1


@pytest.mark.parametrize("cloud", [True, False])
def test_viewer_resolves_current_config_without_unneeded_table_io(tmp_path, monkeypatch, cloud):
    import examples.visualize_episode as viewer

    config = tmp_path / "experiment.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "pointcloud": {"num_points": 512, "remove_table": False},
                "hand": {"T_eef_handbase_pos_xyz": [0.1, 0, 0]},
                "environment": {"table": {"plane_path": "/missing/table.json"}},
            }
        )
    )
    seen = {}

    class Viewer:
        num_steps = 0
        mean_pointcloud_processing_ms = None
        empty_pointcloud_frames = 0

        def __init__(self, *args, **kwargs):
            seen.update(kwargs)

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr(viewer, "EpisodeVisualizer", Viewer)
    monkeypatch.setattr(
        viewer, "resolve_table_plane", lambda *a: pytest.fail("unneeded table read")
    )
    args = [str(tmp_path), "--config", str(config)]
    args += ["--pointcloud-num-points", "256"] if cloud else ["--no-point-cloud"]
    assert viewer.main(args) == 0
    assert seen["handbase_position_eef_m"] == (0.1, 0.0, 0.0) and seen["closed"]
    assert seen["table_plane_abcd"] is None
    assert (
        seen["pointcloud_config"].num_points == 256 if cloud else seen["pointcloud_config"] is None
    )


def test_calibration_records_actual_size_and_requested_fps():
    from dexmani_real.calibration.camera.session import _calibration_capture_metadata

    camera = replace(ExperimentConfig().camera, width=848, fps=15)
    metadata = _calibration_capture_metadata(
        width=camera.width,
        height=camera.height,
        requested_fps=camera.fps,
        intrinsics=np.eye(3),
        distortion=np.zeros(5),
        method="test",
        sample_count=1,
        position_errors_mm=np.zeros(1),
        rotation_errors_deg=np.zeros(1),
    )
    assert (metadata["width"], metadata["height"], metadata["requested_fps"]) == (848, 480, 15)
    assert "fps" not in metadata


def test_viewer_preloads_current_raw_through_episode_reader(tmp_path):
    from test_policy_recording import start_recording

    from dexmani_real.recording import EpisodeReader
    from examples.visualize_episode import EpisodeVisualizer

    recorder = start_recording(tmp_path)
    path = recorder.save_episode(reason="synthetic_fixture")
    viewer = EpisodeVisualizer.__new__(EpisodeVisualizer)
    viewer._T = 1
    hand = ExperimentConfig().hand
    viewer._handbase_position_eef_m = np.asarray(hand.T_eef_handbase_pos_xyz)
    viewer._handbase_quat_eef_wxyz = np.asarray(hand.T_eef_handbase_quat_wxyz)
    with EpisodeReader(path) as reader:
        viewer._reader = reader
        state = viewer._preload_state()
        np.testing.assert_array_equal(state["arm_qpos"], reader["arm_qpos"][:])
        np.testing.assert_array_equal(state["hand_qpos"], reader["hand_qpos"][:])
        assert state["arm_ee"].shape == (1, 9) and np.isfinite(state["arm_ee"]).all()
        assert state["hand_fingertip"].shape == (1, 5, 3)
        assert np.isfinite(state["hand_fingertip"]).all()
        assert np.isnan(state["hand_contact_mag"]).all()
