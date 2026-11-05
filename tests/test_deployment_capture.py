import json
import multiprocessing as mp
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.config.perception import PointCloudConfig
from dexmani_real.config.real import RealConfig
from dexmani_real.deployment import session
from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
from dexmani_real.ipc.sensors import SensorChannels
from dexmani_real.perception import worker
from dexmani_real.perception.pointcloud import PointCloudProcessor
from dexmani_real.runtime.observation import ObservationNeeds, read_observation
from dexmani_real.runtime.state import SessionState


@pytest.mark.parametrize("fields,record,expected", [
    ((), False, None),
    (("finger_tactile_force",), False, None),
    (("rgb",), False, False),
    (("camera_intrinsic", "camera_extrinsic"), False, False),
    (("depth",), False, True),
    (("point_cloud",), False, True),
    (("rgb",), True, True),
])
def test_session_capture_follows_existing_needs(tmp_path, monkeypatch, fields, record, expected):
    monkeypatch.setitem(sys.modules, "dexmani_policy.deployment", SimpleNamespace(load_policy=Mock()))
    monkeypatch.setattr(session, "Keyboard", Mock())
    monkeypatch.setattr(session, "PolicyWorker", Mock())
    monkeypatch.setattr(session, "Robot", Mock())
    monkeypatch.setattr(session.ActionMapper, "for_mode", Mock())
    monkeypatch.setattr(session, "load_table_plane", lambda path: (0., 0., 1., 0.))
    info = SimpleNamespace(
        observation_fields=fields, action_mode="joint", control_action_dim=19,
        control_dt_s=.05, n_action_steps=1, future_steps=1,
        checkpoint_path=Path("checkpoint.pt"), weights="ema", inference_steps=5,
        pointcloud_config=None, depth_config={"shape": [480, 640], "depth_scale_m": .001},
    )
    run = session.PolicySession(
        RealConfig(), "checkpoint.pt", weights=None, inference_steps=None,
        output_root=tmp_path, device="cpu", seed=0, max_running_s=1,
        num_episodes=1, record_raw=record, camera_calibration_path=None,
    )
    monkeypatch.setattr(run, "wait_model", lambda: info)
    monkeypatch.setattr(run, "warmup_policy", Mock())
    monkeypatch.setattr(run, "save_run_config", Mock())
    run.prepare_resources()
    channels = run.sensors.channels
    if expected is None:
        assert channels is None
    else:
        assert ("depth" in channels.camera_ring.field_names) == expected
    started = []
    camera = CameraSnapshot(
        "camera-A", CameraIntrinsics(640, 480, 500., 500., 320., 240., "none", (0.,) * 5),
        .001 if expected else None, np.eye(4),
    )

    def start(process, **kwargs):
        started.append(process.name)
        if process.name == "camera":
            channels.camera_serial.value = camera.serial.encode()
            channels.camera_intrinsics.value = json.dumps(camera.color.to_dict()).encode()
            if expected:
                channels.camera_depth_scale.value = camera.depth_scale_m
            images = {"rgb": np.zeros((480, 640, 3), np.uint8)}
            if expected:
                images["depth"] = np.zeros((480, 640), np.uint16)
            channels.camera_ring.write(timestamp_ns=100, **images)

    monkeypatch.setattr(run.sensors, "start", start)
    shutdown = Mock(wraps=run.sensors.shutdown)
    monkeypatch.setattr(run.sensors, "shutdown", shutdown)
    # 静态参数读取走真实的 shared metadata 与按 serial 匹配的外参读取边界。
    calibration = tmp_path / "cameras.json"
    calibration.write_text(json.dumps({"camera-A": {"T_base_from_camera": np.eye(4).tolist()}}))
    run.camera_calibration_path = calibration
    run.start_devices()
    assert started == ([] if expected is None else ["camera", "pointcloud"] if "point_cloud" in fields else ["camera"])
    static_only = bool(fields) and set(fields) <= {"camera_intrinsic", "camera_extrinsic"}
    assert shutdown.call_count == int(static_only)
    if static_only:
        assert run.sensors.channels is None and not channels.is_running.value
        assert run.camera_snapshot.depth_scale_m is None
        assert set(run.observations.camera_parameters) == set(fields)
    run.warmup_policy.assert_called_once()


def test_pointcloud_keeps_images_and_source_sequence_together(monkeypatch):
    from dexmani_real.config.sensor import CameraConfig

    channels = SensorChannels(
        mp.get_context("spawn"), camera=CameraConfig(width=32, height=32),
        pointcloud_num_points=16, pointcloud_camera_fields=("rgb", "depth"),
    )
    camera = CameraSnapshot(
        "camera-A", CameraIntrinsics(32, 32, 100., 100., 16., 16., "none", (0.,) * 5),
        .001, np.eye(4),
    )
    rgb = np.full((32, 32, 3), 80, np.uint8)
    depth = np.full((32, 32), 500, np.uint16)
    channels.camera_ring.write(timestamp_ns=100, rgb=rgb, depth=depth)
    config = PointCloudConfig(remove_table=False, num_points=16, workspace=(-1., -1., 0., 1., 1., 2.))
    processor = PointCloudProcessor(depth_scale_m=.001, intrinsics=camera.color,
                                    T_xarm_base_from_color=np.eye(4), table_plane_abcd=None, config=config)
    process = processor.process

    def once(depth, rgb):
        channels.is_running.value = False
        return process(depth, rgb)

    monkeypatch.setattr(processor, "process", once)
    monkeypatch.setattr(worker, "PointCloudProcessor", lambda **kwargs: processor)
    worker.run_pointcloud_worker(channels, config, camera, None)
    channels.camera_ring.write(timestamp_ns=200, rgb=rgb + 1, depth=depth + 1)
    row = read_observation(
        channels, SimpleNamespace(read_state=lambda: SimpleNamespace(arm=None, hand=None)),
        needs=ObservationNeeds(camera_fields=("rgb", "depth"), point_cloud=True),
    )
    assert row.point_cloud.shape == (16, 6)
    assert row.camera["ring_sequence"] == 1
    assert row.camera["timestamp_ns"] == row.pointcloud_timestamp_ns == 100
    np.testing.assert_array_equal(row.camera["rgb"], rgb)
    np.testing.assert_array_equal(row.camera["depth"], depth)


def test_depth_consumers_reject_rgb_only_camera():
    from cordvip_geometry.rgbd import RGBDFrame

    from dexmani_real.config.sensor import CameraConfig
    from dexmani_real.sensor.realsense import RealSenseCamera
    from scripts.tools.view_pointcloud import calibrate_table_plane

    camera = CameraSnapshot(
        "camera-A", CameraIntrinsics(32, 32, 100., 100., 16., 16., "none", (0.,) * 5),
        None, np.eye(4),
    )
    with pytest.raises(ValueError, match="depth scale"):
        PointCloudProcessor(
            depth_scale_m=camera.depth_scale_m, intrinsics=camera.color,
            T_xarm_base_from_color=camera.T_xarm_base_from_color,
            table_plane_abcd=None, config=PointCloudConfig(remove_table=False),
        )
    with pytest.raises(ValueError, match="depth scale"):
        RGBDFrame.from_camera(camera, {})
    with pytest.raises(ValueError, match="RGB-D"):
        calibrate_table_plane(camera=RealSenseCamera(CameraConfig(), depth_enabled=False),
                              T_xarm_base_from_color=np.eye(4),
                              config=PointCloudConfig(), plane_path=None)


def test_camera_cleanup_allows_measured_sdk_stop_time():
    process = Mock(pid=1, exitcode=None)
    process.name = "camera"

    def join(timeout):
        if timeout >= 3.2:
            process.exitcode = 0

    process.join.side_effect = join
    process.is_alive.side_effect = lambda: process.exitcode is None
    sensors = session.SensorWorkers()
    sensors.processes.append(process)
    sensors.shutdown(SessionState())
    process.terminate.assert_not_called()
    assert not sensors.processes


@pytest.mark.parametrize("exitcode", [1, -signal.SIGTERM])
def test_sensor_shutdown_preserves_abnormal_exit(exitcode):
    process = Mock(pid=1, exitcode=exitcode)
    process.name = "camera"
    process.is_alive.return_value = False
    sensors = session.SensorWorkers()
    sensors.processes.append(process)
    with pytest.raises(RuntimeError, match=f"exitcode={exitcode}"):
        sensors.shutdown(SessionState())


def test_sensor_shutdown_reports_forced_termination(caplog):
    process = Mock(pid=1, exitcode=None)
    process.name = "camera"
    process.is_alive.side_effect = (True, False)

    def terminate():
        process.exitcode = -signal.SIGTERM

    process.terminate.side_effect = terminate
    sensors = session.SensorWorkers()
    sensors.processes.append(process)
    sensors.shutdown(SessionState())
    process.terminate.assert_called_once()
    assert "terminated during shutdown" in caplog.text
    assert not sensors.processes


@pytest.mark.parametrize("count", [123, 30000, 30001, 59999, 60001])
@pytest.mark.parametrize("publish", [False, True])
def test_table_calibration_samples_full_range(monkeypatch, tmp_path, capsys, count, publish):
    from dexmani_real.geometry import plane
    from dexmani_real.perception import pointcloud
    from dexmani_real.utils import table_plane
    from scripts.tools import view_pointcloud

    points = np.column_stack((np.linspace(.2, .8, count), np.zeros(count), np.full(count, .1)))
    frames = iter([None] + [SimpleNamespace(depth_frame_number=i, depth=np.array([[i]]))
                           for i in (1, 1, 2, 2, 3, 4, 5)])
    camera = SimpleNamespace(depth_scale_m=.001, color_intrinsics=None, read=lambda: next(frames))
    converted = []

    def deproject(**kwargs):
        converted.append(int(kwargs["depth_raw"][0, 0]))
        return points

    batches = []
    fit = SimpleNamespace(plane_abcd=(0., 0., 1., -.1), inlier_points=1000,
                          evaluated_points=1000, inlier_ratio=1., rms_residual_m=0., tilt_deg=0.)
    monkeypatch.setattr(pointcloud, "aligned_depth_points_in_base", deproject)
    monkeypatch.setattr(plane, "fit_table_plane", lambda p: batches.append(p) or fit)
    save = Mock()
    monkeypatch.setattr(table_plane, "save_table_plane", save)
    answers = iter(["", "y" if publish else "n"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    path = tmp_path / "plane.json"
    assert view_pointcloud.calibrate_table_plane(
        camera=camera, T_xarm_base_from_color=np.eye(4), config=PointCloudConfig(),
        plane_path=path) == fit.plane_abcd
    assert converted == [1, 2, 3, 4, 5]
    selected = batches[0].reshape(5, min(count, 30000), 3)
    for batch in selected:
        assert len(np.unique(batch[:, 0])) == min(count, 30000)
        np.testing.assert_array_equal(batch[[0, -1]], points[[0, -1]])
        np.testing.assert_array_equal(batch, selected[0])
        if count <= 30000:
            np.testing.assert_array_equal(batch, points)
    assert save.call_count == int(publish)
    if publish:
        save.assert_called_once_with(path, fit)
    assert "collision" not in capsys.readouterr().out


@pytest.mark.parametrize("slopes,height", [
    ((0., 0.), 0.), ((.08, -.05), .001), ((0., 0.), .8), ((0., 0.), .4),
])
def test_table_calibration_fits_unclipped_height_and_normal(monkeypatch, tmp_path, slopes, height):
    from dexmani_real.perception import pointcloud
    from dexmani_real.utils import table_plane
    from scripts.tools import view_pointcloud

    rng = np.random.default_rng(2026)
    count = 12000
    xy = np.column_stack((rng.uniform(.15, .85, count), rng.uniform(-.5, .5, count)))
    z = height + (xy - (.5, 0.)) @ np.asarray(slopes) + rng.normal(0., .002, count)
    table = np.column_stack((xy, z))
    # 两轴范围外各有一个支持点更多的干扰平面，漏筛任一轴都会选错桌面。
    outside_x = np.column_stack((rng.uniform(.9, 1.2, 30000),
                                 rng.uniform(-.5, .5, 30000), rng.normal(.3, .001, 30000)))
    outside_y = np.column_stack((rng.uniform(.15, .85, 30000),
                                 rng.uniform(.6, .9, 30000), rng.normal(.3, .001, 30000)))
    points = np.concatenate((table, outside_x, outside_y))
    frames = iter(SimpleNamespace(depth_frame_number=i, depth=np.array([[i]])) for i in range(5))
    camera = SimpleNamespace(depth_scale_m=.001, color_intrinsics=None, read=lambda: next(frames))
    monkeypatch.setattr(pointcloud, "aligned_depth_points_in_base", lambda **kwargs: points)
    save = Mock()
    monkeypatch.setattr(table_plane, "save_table_plane", save)
    answers = iter(["", "n"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    # 使用真实 RANSAC 和最小二乘拟合，不替换 fit_table_plane。
    a, b, c, d = view_pointcloud.calibrate_table_plane(
        camera=camera, T_xarm_base_from_color=np.eye(4), config=PointCloudConfig(),
        plane_path=tmp_path / "plane.json",
    )
    fitted_height = -(a * .5 + d) / c
    expected_normal = np.array([-slopes[0], -slopes[1], 1.])
    expected_normal /= np.linalg.norm(expected_normal)
    normal_error = np.degrees(np.arccos(np.clip(np.dot([a, b, c], expected_normal), -1, 1)))
    assert abs(fitted_height - height) < .0002
    assert normal_error < .03
    assert c > 0
    save.assert_not_called()


@pytest.fixture
def viewer_resources(monkeypatch):
    from dexmani_real.config import real
    from dexmani_real.perception import pointcloud
    from dexmani_real.sensor import realsense
    from dexmani_real.utils.camera_calibrate import extrinsics
    from dexmani_real.utils.visualization import image, pointcloud as visualization

    events = []
    camera = Mock(active_serial="camera-A", depth_scale_m=.001)
    camera.connect.side_effect = lambda: events.append("connect")
    camera.disconnect.side_effect = lambda: events.append("disconnect")
    camera.read.return_value = SimpleNamespace(
        rgb=np.zeros((4, 4, 3), np.uint8), depth=np.full((4, 4), 500, np.uint16),
    )
    preview = Mock()
    preview.close.side_effect = lambda: events.append("preview_close")
    cloud = Mock()
    cloud.close.side_effect = lambda: events.append("cloud_close")
    cloud.poll_events.return_value = ["q"]
    processor = Mock()
    processor.process.return_value = np.ones((16, 6), np.float32)

    def open_cloud(**kwargs):
        events.append("cloud_open")
        return cloud

    monkeypatch.setattr(real, "load_config", lambda **kwargs: RealConfig(
        pointcloud=PointCloudConfig(remove_table=False),
    ))
    monkeypatch.setattr(realsense, "RealSenseCamera", lambda *args, **kwargs: camera)
    monkeypatch.setattr(image, "ImageViewer", lambda *args: preview)
    monkeypatch.setattr(visualization, "PointCloudViewer", open_cloud)
    monkeypatch.setattr(extrinsics, "load_camera_extrinsics", lambda *args: np.eye(4))
    monkeypatch.setattr(pointcloud, "PointCloudProcessor", lambda **kwargs: processor)
    return SimpleNamespace(events=events, camera=camera, preview=preview,
                           cloud=cloud, processor=processor)


@pytest.mark.parametrize("quit_preview", [False, True])
def test_static_viewer_closes_camera_before_cloud(monkeypatch, viewer_resources, quit_preview):
    from scripts.tools import view_pointcloud

    resources = viewer_resources
    monkeypatch.setattr("builtins.input", lambda _: "n")
    resources.preview.poll_events.return_value = ["q" if quit_preview else "enter"]
    assert view_pointcloud.main([]) == 0
    expected = ["connect", "preview_close", "disconnect"]
    if not quit_preview:
        expected += ["cloud_open", "cloud_close"]
        resources.processor.process.assert_called_once()
    else:
        resources.processor.process.assert_not_called()
    assert resources.events == expected


def test_static_viewer_preserves_cleanup_exception_chain(monkeypatch, viewer_resources):
    from scripts.tools import view_pointcloud

    resources = viewer_resources
    failure = ValueError("preview failed")
    close_failure = OSError("preview close failed")
    resources.preview.show.side_effect = failure
    resources.preview.close.side_effect = close_failure
    monkeypatch.setattr("builtins.input", lambda _: "n")
    with pytest.raises(OSError, match="preview close failed") as raised:
        view_pointcloud.main([])
    assert raised.value.__context__ is failure
    resources.camera.disconnect.assert_called_once()
    resources.processor.process.assert_not_called()
    assert "cloud_open" not in resources.events


@pytest.mark.parametrize("close_fails", [False, True])
def test_live_viewer_closes_both_windows_and_camera(monkeypatch, viewer_resources, close_fails):
    from dexmani_real.perception import pointcloud
    from scripts.tools.view_realsense import RealSenseViewer

    resources = viewer_resources
    resources.preview.poll_events.return_value = []
    monkeypatch.setattr(pointcloud, "build_raw_point_cloud", lambda **kwargs: np.ones((1, 6)))
    if close_fails:
        def fail_close():
            resources.events.append("cloud_close")
            raise OSError("cloud close failed")
        resources.cloud.close.side_effect = fail_close
    viewer = RealSenseViewer(SimpleNamespace(config=None, camera_calibration=None, cloud="raw"))
    if close_fails:
        with pytest.raises(OSError, match="cloud close failed"):
            viewer.run()
    else:
        assert viewer.run() == 0
    assert resources.events == ["connect", "cloud_open", "cloud_close", "preview_close", "disconnect"]
