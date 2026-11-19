import json
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from unittest.mock import Mock

import h5py
import numpy as np
import pytest
import zarr
from scipy.spatial.transform import Rotation

from dexmani_real.config.real import RealConfig
from dexmani_real.control.actions import ActionMapper
from dexmani_real.data.export import ExportConfig, export_zarr
from dexmani_real.data.raw import (
    ARM_ACTION_SEMANTICS,
    TIMING_ROW_DTYPE,
    RawDataError,
    RawReader,
)
from dexmani_real.geometry import kinematics
from dexmani_real.geometry.kinematics import HandFK, kinematic_urdf, validate_xhand_urdf
from dexmani_real.replay import run as replay
from dexmani_real.replay.trajectory import load_trajectory
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.robot.model import XARM7_JOINT_NAMES, XHAND_SDK_JOINT_NAMES
from dexmani_real.robot.system import Robot
from dexmani_real.robot.xhand.driver import XHand, _parse_joints
from dexmani_real.runtime.state import SessionState


def test_fk_uses_sdk_order_and_signed_right_hand_axes():
    fk = HandFK()
    np.testing.assert_array_equal(fk.urdf_to_sdk, [9, 10, 11, 0, 1, 2, 3, 4, 7, 8, 5, 6])
    axes = ((0, 0, -1), (1, 0, 0), (1, 0, 0), (-1, 0, 0), *((0, 1, 0),) * 8)
    urdf = ET.fromstring(kinematic_urdf("right_hand_link"))
    joints = {joint.get("name"): joint for joint in urdf.findall("joint")}
    for index, (name, axis) in enumerate(zip(XHAND_SDK_JOINT_NAMES, axes, strict=True)):
        frame = fk.model.getFrameId(joints[name].find("child").get("link"))
        qpos = np.zeros(12)
        neutral = fk.forward(qpos, [frame])[0, :3, :3]
        qpos[index] = 0.2
        rotated = fk.forward(qpos, [frame])[0, :3, :3]
        np.testing.assert_allclose(neutral.T @ rotated, Rotation.from_rotvec(0.2 * np.array(axis)).as_matrix(), atol=1e-10)


@pytest.mark.parametrize("fault,match", [
    ("missing", "missing=.*right_hand_thumb_bend_joint"),
    ("duplicate", "duplicate joint name.*right_hand_thumb_bend_joint"),
    ("sign", "signed axis.*differs"),
    ("nonunit", "signed unit axis"),
    ("units", "independent revolute joint in radians"),
])
def test_urdf_validation_rejects_invalid_joint_contract(fault, match):
    root = ET.fromstring(kinematic_urdf("right_hand_link"))
    joint = root.find("joint[@name='right_hand_thumb_bend_joint']")
    if fault == "missing":
        root.remove(joint)
    elif fault == "duplicate":
        root.append(joint)
    elif fault == "sign":
        joint.find("axis").set("xyz", "0 0 1")
    elif fault == "nonunit":
        joint.find("axis").set("xyz", "0 0 2")
    else:
        joint.set("type", "prismatic")
    with pytest.raises(ValueError, match=match):
        validate_xhand_urdf(ET.tostring(root, encoding="unicode"))


@pytest.mark.parametrize("xyz,match", [
    ("0.0096 0.0228 0.0305", "joint frame differs from right-hand URDF"),
    ("0.0095 0.0228", "origin xyz must have three finite values"),
    ("0.0095 nan 0.0305", "origin xyz must have three finite values"),
    ("0.0095 0.0228 inf", "origin xyz must have three finite values"),
])
def test_urdf_validation_rejects_invalid_origin_xyz(xyz, match):
    root = ET.fromstring(kinematic_urdf("right_hand_link"))
    origin = root.find("joint[@name='right_hand_thumb_bend_joint']/origin")
    origin.set("xyz", xyz)
    with pytest.raises(ValueError, match=match):
        validate_xhand_urdf(ET.tostring(root, encoding="unicode"))


def test_combined_urdf_rejects_cycle_before_joint_validation(tmp_path, monkeypatch):
    path = tmp_path / "combined.urdf"
    root = ET.fromstring(kinematic_urdf("right_hand_link"))
    joint = root.find("joint[@name='right_hand_thumb_bend_joint']")
    joint.find("child").set("link", "right_hand_link")
    ET.ElementTree(root).write(path)
    monkeypatch.setattr(kinematics, "XARM7_XHAND_RIGHT_URDF_PATH", path)

    with pytest.raises(ValueError, match="reaches link right_hand_link more than once"):
        validate_xhand_urdf()


def test_sdk_feedback_and_commands_use_ids_and_radians():
    cfg = RealConfig().hand
    positions = cfg.home_qpos_rad
    assert positions[0] == pytest.approx(np.pi / 6)
    np.testing.assert_allclose(np.rad2deg(positions), cfg.home_qpos_deg)
    joints = [SimpleNamespace(id=i, position=value, commboard_err=0, jonitboard_err=0)
              for i, value in enumerate(positions)]
    sdk_state = SimpleNamespace(finger_state=joints[::-1])
    parsed, _ = _parse_joints(sdk_state)
    np.testing.assert_array_equal(parsed, positions)

    hand = XHand(cfg)
    slots = [SimpleNamespace() for _ in range(12)]
    hand.sdk = SimpleNamespace(HandCommand_t=lambda: SimpleNamespace(finger_command=slots))
    command = hand.make_command(parsed)
    assert [joint.id for joint in command.finger_command] == list(range(12))
    np.testing.assert_array_equal([joint.position for joint in command.finger_command], positions)

    slots.pop()
    with pytest.raises(ValueError, match="12 joint slots, got 11"):
        hand.make_command(parsed)
    joints.pop()
    with pytest.raises(ValueError, match="missing .*right_hand_pinky_joint2"):
        _parse_joints(SimpleNamespace(finger_state=joints))
    with pytest.raises(ValueError, match=r"duplicate .*id 0 \(right_hand_thumb_bend_joint\)"):
        _parse_joints(SimpleNamespace(finger_state=[*joints, joints[0]]))


def test_operational_limit_is_checked_before_either_sdk_submission():
    runtime = RealConfig()
    robot = Robot(runtime.arm, runtime.hand, runtime_state=SessionState())
    robot.arm = Mock(is_connected=True)
    robot.hand = Mock(is_connected=True)
    robot.connected = True
    hand = runtime.hand.home_qpos_rad
    hand[2] = 0.0  # URDF 额定范围内，低于本实验 10° 下界。
    command = RobotCommand(np.asarray(runtime.arm.home_qpos), hand)

    with pytest.raises(ValueError, match="right_hand_thumb_rota_joint2"):
        robot.send_action(command, expires_ns=2**62)
    robot.arm.submit.assert_not_called()
    robot.hand.send_action.assert_not_called()
    assert not robot.runtime_state.stop_signal.is_set()

    mapper = ActionMapper.for_mode(
        "joint", arm=runtime.arm, hand=runtime.hand, motion=runtime.motion, environment=runtime.environment,
    )
    with pytest.raises(ValueError, match="right_hand_thumb_rota_joint2"):
        mapper.map_joint(command.arm_qpos, hand)
    mapped = mapper.map_joint(command.arm_qpos, runtime.hand.home_qpos_rad)
    np.testing.assert_array_equal(mapped.hand_qpos, runtime.hand.home_qpos_rad)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_targets_report_joint_name_and_do_not_write(invalid):
    cfg = RealConfig().hand
    hand = XHand(cfg)
    hand.connected_flag = True
    hand.control, hand.command = Mock(), Mock()
    qpos = cfg.home_qpos_rad
    qpos[7] = invalid
    with pytest.raises(ValueError, match="non-finite joints: right_hand_mid_joint2"):
        hand.send_position(qpos, deadline_ns=2**62)
    hand.control.send_command.assert_not_called()


@pytest.mark.parametrize("dof", [11, 13])
def test_sdk_and_fk_reject_wrong_dimensions(dof):
    qpos = np.zeros(dof)
    hand = XHand(RealConfig().hand)
    hand.connected_flag = True
    hand.control, hand.command = Mock(), Mock()
    with pytest.raises(ValueError, match=r"shape \(12,\)"):
        hand.send_position(qpos, deadline_ns=2**62)
    hand.control.send_command.assert_not_called()
    fk = HandFK()
    with pytest.raises(ValueError, match=r"shape \(12,\)"):
        fk.forward(qpos, fk.fingertip_frame_ids)


@pytest.fixture
def joint_raw(tmp_path):
    runtime = RealConfig()

    def write(name, status=(1, 1), *, starts=(0,), status_dtype=np.int8):
        source = tmp_path / f"episode_{name}"
        source.mkdir()
        frames = 2 if status is None else len(status)
        arm = np.tile(runtime.arm.home_qpos, (frames, 1))
        hand = np.tile(runtime.hand.home_qpos_rad, (frames, 1)) + np.arange(12) * 0.007
        timing = np.zeros(frames, dtype=TIMING_ROW_DTYPE)
        for clock in ("control_ns", "observation_ns", "arm_ns", "hand_ns"):
            timing[clock] = 1_000_000_000 + 50_000_000 * np.arange(frames)
        with h5py.File(source / "data.h5", "w") as raw:
            raw.create_group("meta").attrs.update(
                joint_units="rad", arm_joint_names=json.dumps(XARM7_JOINT_NAMES),
                hand_joint_names=json.dumps(XHAND_SDK_JOINT_NAMES), arm_action_semantics=ARM_ACTION_SEMANTICS,
                task_label="mapping", control_hz=20,
            )
            raw["timing/rows"], raw["timing/segment_starts"] = timing, np.array(starts, dtype=np.int64)
            raw["arm_qpos"], raw["action_arm_joint_target"] = arm, arm
            raw["hand_qpos"], raw["action_hand_joint_target"] = hand, hand
            if status is not None:
                raw["action_send_accepted"] = np.asarray(status, dtype=status_dtype)
        return source, hand

    return runtime, write


def test_existing_raw_and_zarr_keep_original_sdk_columns(tmp_path, joint_raw):
    runtime, write = joint_raw
    source, hand = write("original")

    with RawReader(source) as raw:
        np.testing.assert_array_equal(raw["hand_qpos"][:], hand)
    np.testing.assert_array_equal(load_trajectory(source, runtime).action_hand_joint, hand)

    target = tmp_path / "existing_layout.zarr"
    export_zarr(source, target, processing=ExportConfig(), fields=("joint_state", "action", "action_ode", "action_ee"))
    data = zarr.open_group(str(target), mode="r")["data"]
    for field, first_hand in (("joint_state", 7), ("action", 7), ("action_ode", 7), ("action_ee", 9)):
        np.testing.assert_array_equal(data[field][:, first_hand:], hand.astype(np.float32))

    with h5py.File(source / "data.h5", "r+") as raw:
        raw["action_hand_joint_target"][1, 2] = runtime.hand.qpos_min_rad[2] - 1e-8
    with pytest.raises(ValueError, match="row 1: .*right_hand_thumb_rota_joint2"):
        load_trajectory(source, runtime)


@pytest.mark.parametrize("status", [0, 2])
def test_replay_checks_only_selected_segment_before_robot_construction(joint_raw, monkeypatch, status):
    runtime, write = joint_raw
    source, hand = write("segments", (1, 1, 1, status), starts=(0, 2))
    robot = Mock(side_effect=AssertionError("unexpected robot construction"))
    monkeypatch.setattr(replay, "Robot", robot)

    with pytest.raises(RawDataError, match=rf"action_send_accepted={status} at row 3"):
        replay.run_replay(source, runtime, segment=1)
    robot.assert_not_called()
    np.testing.assert_array_equal(load_trajectory(source, runtime, segment=0).action_hand_joint, hand[:2])


@pytest.mark.parametrize("fields", [("action",), ("action_ee",), ("action_ee", "action_ode")])
def test_action_export_rejects_unconfirmed_raw(tmp_path, joint_raw, fields):
    _, write = joint_raw
    write("bad", (1, 0, 2), starts=(0, 1))
    good, hand = write("good", (1, 1, 1), starts=(0, 2))
    with h5py.File(good / "data.h5", "r+") as raw:
        timing = raw["timing/rows"][:]
        timing["arm_ns"] = timing["hand_ns"] = 0
        raw["timing/rows"][:] = timing
    write("missing", None)
    write("wrong_dtype", status_dtype=np.float32)
    write("wrong_shape", ((1,), (1,)))

    target = tmp_path / "actions.zarr"
    result = export_zarr(tmp_path, target, processing=ExportConfig(), fields=fields)

    assert result["source_count"] == 1
    assert result["total_frames"] == len(hand)
    root = zarr.open_group(str(target), mode="r")
    assert root.attrs["source_episodes"] == ["episode_good", "episode_good"]
    assert root.attrs["source_row_ranges"] == [[0, 2], [2, 3]]
    np.testing.assert_array_equal(root["meta/episode_ends"][:], [2, 3])
    assert root.attrs["rejected_episodes"] == result["rejected_episodes"]
    reasons = {item["episode"]: item["reason"] for item in result["rejected_episodes"]}
    assert "action_send_accepted=0 at row 1 (2 row(s) without confirmed submission)" in reasons["episode_bad"]
    assert "missing action_send_accepted" in reasons["episode_missing"]
    assert "dtype float32" in reasons["episode_wrong_dtype"]
    assert "shape (2, 1)" in reasons["episode_wrong_shape"]
    for field in fields:
        np.testing.assert_array_equal(root[f"data/{field}"][:, -12:], hand.astype(np.float32))


def test_observation_export_does_not_require_submission_status(tmp_path, joint_raw):
    _, write = joint_raw
    write("missing", None)
    write("wrong_dtype", status_dtype=np.float32)

    result = export_zarr(tmp_path, tmp_path / "observations.zarr", processing=ExportConfig(), fields=("joint_state",))

    assert result["source_count"] == 2
    assert result["rejected_episodes"] == []


def test_action_ode_export_holds_accepted_targets_within_each_segment(tmp_path, joint_raw):
    _, write = joint_raw
    source, hand = write("ode", (0, 1, 0, 0), starts=(0, 3))
    with h5py.File(source / "data.h5", "r+") as raw:
        raw["action_hand_joint_target"][:] = hand + 0.125
        raw["action_arm_joint_target"][0] = np.nan
        raw["action_hand_joint_target"][0] = np.nan

    target = tmp_path / "ode.zarr"
    export_zarr(source, target, processing=ExportConfig(), fields=("action_ode",))

    data = zarr.open_group(str(target), mode="r")["data/action_ode"]
    hand[1:3] += 0.125
    np.testing.assert_array_equal(data[:, -12:], hand.astype(np.float32))
