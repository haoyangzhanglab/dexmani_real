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
from dexmani_real.data.raw import ARM_ACTION_SEMANTICS, TIMING_ROW_DTYPE, RawReader
from dexmani_real.geometry.kinematics import HandFK, kinematic_urdf, validate_xhand_urdf
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


def test_existing_raw_and_zarr_keep_original_sdk_columns(tmp_path):
    runtime = RealConfig()
    source = tmp_path / "episode_original"
    source.mkdir()
    arm = np.tile(runtime.arm.home_qpos, (2, 1))
    hand = np.tile(runtime.hand.home_qpos_rad, (2, 1)) + np.arange(12) * 0.007
    timing = np.zeros(2, dtype=TIMING_ROW_DTYPE)
    for clock in ("control_ns", "observation_ns", "arm_ns", "hand_ns"):
        timing[clock] = [1_000_000_000, 1_050_000_000]
    raw_path = source / "data.h5"
    with h5py.File(raw_path, "w") as raw:
        raw.create_group("meta").attrs.update(
            joint_units="rad", arm_joint_names=json.dumps(XARM7_JOINT_NAMES),
            hand_joint_names=json.dumps(XHAND_SDK_JOINT_NAMES), arm_action_semantics=ARM_ACTION_SEMANTICS,
            task_label="mapping", control_hz=20,
        )
        raw["timing/rows"], raw["timing/segment_starts"] = timing, np.array([0], dtype=np.int64)
        raw["arm_qpos"], raw["action_arm_joint_target"] = arm, arm
        raw["hand_qpos"], raw["action_hand_joint_target"] = hand, hand
        raw["action_send_accepted"] = np.ones(2, dtype=np.int8)

    with RawReader(source) as raw:
        np.testing.assert_array_equal(raw["hand_qpos"][:], hand)
    np.testing.assert_array_equal(load_trajectory(source, runtime).action_hand_joint, hand)

    target = tmp_path / "existing_layout.zarr"
    export_zarr(source, target, processing=ExportConfig(), fields=("joint_state", "action", "action_ode", "action_ee"))
    data = zarr.open_group(str(target), mode="r")["data"]
    for field, first_hand in (("joint_state", 7), ("action", 7), ("action_ode", 7), ("action_ee", 9)):
        np.testing.assert_array_equal(data[field][:, first_hand:], hand.astype(np.float32))

    with h5py.File(raw_path, "r+") as raw:
        raw["action_hand_joint_target"][1, 2] = runtime.hand.qpos_min_rad[2] - 1e-8
    with pytest.raises(ValueError, match="row 1: .*right_hand_thumb_rota_joint2"):
        load_trajectory(source, runtime)
