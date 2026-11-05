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
