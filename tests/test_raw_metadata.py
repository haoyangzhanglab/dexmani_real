"""Raw 录制元数据的字段白名单与序列化契约。"""

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from dexmani_real.config.real import RealConfig
from dexmani_real.data.raw import (
    RawReader,
    discover_episode_dirs,
    episode_recording_metadata,
    robot_recording_parameters,
)
from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot


@pytest.mark.parametrize("input_kind", ["single", "batch", "relative", "home"])
def test_discover_episode_dirs_uses_only_explicit_root(tmp_path, monkeypatch, input_kind):
    source = tmp_path / "external/pick"
    episodes = tuple(source / name for name in ("episode_001", "episode_002"))
    for episode in episodes:
        episode.mkdir(parents=True)
        (episode / "data.h5").touch()
    # 同名 checkout 数据与更深层数据均不参与发现。
    (tmp_path / "checkout/episodes/pick/episode_003").mkdir(parents=True)
    (source / "nested/episode_004").mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "checkout")
    monkeypatch.setenv("HOME", str(tmp_path))
    requested = {
        "single": episodes[0], "batch": source,
        "relative": "../external/pick", "home": "~/external/pick",
    }[input_kind]
    assert discover_episode_dirs(requested) == (episodes[:1] if input_kind == "single" else episodes)


def test_discover_episode_dirs_reports_invalid_or_empty_inputs(tmp_path):
    with pytest.raises(NotADirectoryError):
        discover_episode_dirs(tmp_path / "missing")
    regular_file = tmp_path / "file"
    regular_file.touch()
    with pytest.raises(NotADirectoryError):
        discover_episode_dirs(regular_file)
    (tmp_path / "nested/episode_001").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no episode directories"):
        discover_episode_dirs(tmp_path)
    unpublished = tmp_path / ".tmp_episode"
    unpublished.mkdir()
    (unpublished / "data.h5").touch()
    with pytest.raises(ValueError, match="unpublished Raw"):
        discover_episode_dirs(unpublished)
    alias = tmp_path / "episode_pending"
    alias.symlink_to(unpublished, target_is_directory=True)
    with pytest.raises(ValueError, match="unpublished Raw"):
        RawReader(alias)


@pytest.mark.parametrize("comm_type,selected,excluded", [
    ("serial", "baudrate", "ethercat_slave_position"),
    ("ethercat", "ethercat_slave_position", "baudrate"),
])
def test_robot_recording_parameters_preserve_hardware_fields(comm_type, selected, excluded):
    runtime = RealConfig()
    runtime = replace(
        runtime,
        arm=replace(runtime.arm, ip="192.168.1.123", homing=object()),
        hand=replace(
            runtime.hand, comm_type=comm_type, home_tor_max=object(),
            **{selected: 123, excluded: object()},
        ),
    )
    parameters = robot_recording_parameters(runtime)
    assert set(parameters) == {"arm", "hand"}
    assert set(parameters["arm"]) == {
        "ip", "collision_sensitivity", "tcp_load_mass_kg", "tcp_load_cog_mm",
        "joint_limit_lower", "joint_limit_upper", "max_joint_velocity_deg_per_s",
        "max_joint_acceleration_deg_per_s2", "max_joint_jerk_deg_per_s3", "servo_hz",
        "feedback_hz", "feedback_max_age_s", "standstill_position_tolerance_rad",
        "standstill_velocity_tolerance_rad_s",
    }
    assert set(parameters["hand"]) == {
        "enabled", "comm_type", "device_name", "device_id", "qpos_min_rad", "qpos_max_rad",
        "kp", "ki", "kd", "tor_max", "feedback_max_age_s", selected,
    }
    for device, values in parameters.items():
        for name, value in values.items():
            original = getattr(getattr(runtime, device), name)
            assert value == original
            assert type(value) is type(original)


@pytest.fixture
def camera_snapshot():
    return CameraSnapshot(
        "camera-123", CameraIntrinsics(640, 480, 600., 601., 320., 240., "brown_conrady", (.1, 0., 0., 0., 0.)),
        .001, np.eye(4),
    )


@pytest.mark.parametrize("full_camera", [False, True])
@pytest.mark.parametrize("tactile_reset", [
    {},
    {"hand_tactile_reset_time_ns": 123456789, "hand_tactile_reset_result": {2: 0, 3: 1}},
])
def test_episode_metadata_preserves_camera_and_input(camera_snapshot, full_camera, tactile_reset):
    if not full_camera:
        camera_snapshot = replace(camera_snapshot, serial=None, depth_scale_m=None, T_xarm_base_from_color=None)
    effective = {"policy": {"weights": "raw"}, "arm": {"ip": "robot"}, "camera_config": {"fps": 30}}
    original = deepcopy(effective)
    metadata = episode_recording_metadata(camera_snapshot, effective, tactile_reset=tactile_reset)
    parameters = json.loads(metadata.pop("effective_parameters"))
    expected_camera = {
        "camera_payload_mode": "depth_to_color_aligned_rgbd" if full_camera else "rgb",
        "camera_color_width": 640,
        "camera_color_height": 480,
        "camera_color_intrinsics": [600., 0., 320., 0., 601., 240., 0., 0., 1.],
        "camera_color_distortion_model": "brown_conrady",
        "camera_color_distortion_coeffs": [.1, 0., 0., 0., 0.],
    }
    if full_camera:
        expected_camera.update(
            camera_serial="camera-123", depth_scale=.001,
            camera_T_xarm_base_from_color=np.eye(4).ravel().tolist(),
        )
    assert metadata == expected_camera
    assert parameters == {
        **original, "tactile_reset": json.loads(json.dumps(tactile_reset)),
    }
    assert effective == original


@pytest.mark.parametrize("effective,tactile_reset,error", [
    ({"invalid": float("nan")}, {}, ValueError),
    ({}, {"invalid": float("nan")}, ValueError),
    ({"invalid": object()}, {}, TypeError),
    ({}, {"invalid": object()}, TypeError),
])
def test_episode_metadata_propagates_serialization_errors(camera_snapshot, effective, tactile_reset, error):
    with pytest.raises(error):
        episode_recording_metadata(camera_snapshot, effective, tactile_reset=tactile_reset)
