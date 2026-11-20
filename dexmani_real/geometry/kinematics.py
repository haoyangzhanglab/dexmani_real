"""xArm 基座系的机械臂 FK，以及手基座系的 XHand link FK。"""

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from math import hypot, isclose, isfinite

import numpy as np

from dexmani_real.robot.commands import validate_hand_qpos
from dexmani_real.robot.model import (
    HAND_DOF,
    XARM7_JOINT_NAMES,
    XARM7_XHAND_RIGHT_URDF_PATH,
    XHAND_FINGERTIP_LINK_NAMES,
    XHAND_RATED_QPOS_MAX_RAD,
    XHAND_RATED_QPOS_MIN_RAD,
    XHAND_RIGHT_URDF_PATH,
    XHAND_SDK_JOINT_NAMES,
)


def kinematic_urdf(root_link: str, *, stop_link: str | None = None) -> str:
    """从完整组合 URDF 提取运动学子树，保留原始 joint 定义并移除网格。"""
    root = ET.parse(XARM7_XHAND_RIGHT_URDF_PATH).getroot()
    link_names = {link.attrib["name"] for link in root.findall("link")}
    children: dict[str, list[tuple[str, str]]] = {}
    joint_names = set()
    for joint in root.findall("joint"):
        name = joint.get("name")
        if not name or name in joint_names:
            raise ValueError(f"robot URDF contains an empty or duplicate joint name {name!r}")
        joint_names.add(name)
        parent, child = joint.find("parent"), joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"URDF joint {name} must name its parent and child")
        children.setdefault(parent.attrib["link"], []).append(
            (name, child.attrib["link"]),
        )
    links, joints = set(), set()
    pending = [root_link]
    while pending:
        name = pending.pop()
        if name not in link_names:
            raise ValueError(f"robot URDF is missing link {name}")
        if name in links:
            raise ValueError(f"robot URDF reaches link {name} more than once")
        links.add(name)
        if name != stop_link:
            for joint_name, child_link in children.get(name, ()):
                joints.add(joint_name)
                pending.append(child_link)
    for node in list(root):
        if node.tag == "joint" and node.attrib["name"] not in joints:
            root.remove(node)
        elif node.tag == "link":
            if node.attrib["name"] not in links:
                root.remove(node)
            else:
                for geometry in (*node.findall("visual"), *node.findall("collision")):
                    node.remove(geometry)
    return ET.tostring(root, encoding="unicode")


def validate_xhand_urdf(hand_urdf: str | None = None) -> None:
    """核对 12 个 SDK 转动关节的轴、坐标系和额定位置限位。"""
    names = XHAND_SDK_JOINT_NAMES
    expected = set(names)
    if len(names) != HAND_DOF or len(expected) != HAND_DOF or "" in expected:
        raise ValueError(f"XHand SDK must define {HAND_DOF} unique joint names")

    def active_joints(root: ET.Element, source: str) -> dict[str, ET.Element]:
        joints: dict[str, ET.Element] = {}
        for joint in root.findall("joint"):
            name = joint.get("name")
            if not name or name in joints:
                raise ValueError(f"{source} contains an empty or duplicate joint name {name!r}")
            joints[name] = joint
        active = {name: joint for name, joint in joints.items() if joint.get("type") != "fixed"}
        missing, extra = expected - active.keys(), active.keys() - expected
        if missing or extra:
            raise ValueError(f"{source} XHand joint mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
        return active

    reference = active_joints(ET.parse(XHAND_RIGHT_URDF_PATH).getroot(), "right-hand URDF")
    if hand_urdf is None:
        hand_urdf = kinematic_urdf("right_hand_link")
    hand = active_joints(ET.fromstring(hand_urdf), "combined URDF")
    for name, lower, upper in zip(names, XHAND_RATED_QPOS_MIN_RAD, XHAND_RATED_QPOS_MAX_RAD, strict=True):
        joint_frames = []
        for joint in (reference[name], hand[name]):
            if joint.get("type") != "revolute" or joint.find("mimic") is not None:
                raise ValueError(f"XHand joint {name} must be an independent revolute joint in radians")
            axis_node, parent, child, limit = (joint.find(tag) for tag in ("axis", "parent", "child", "limit"))
            if axis_node is None or parent is None or child is None or limit is None:
                raise ValueError(f"XHand joint {name} requires axis, parent, child and position limits")
            origin = joint.find("origin")
            origin_attrs = {} if origin is None else origin.attrib
            try:
                axis = tuple(map(float, axis_node.attrib["xyz"].split()))
                xyz = tuple(map(float, origin_attrs.get("xyz", "0 0 0").split()))
                rpy = tuple(map(float, origin_attrs.get("rpy", "0 0 0").split()))
                bounds = float(limit.attrib["lower"]), float(limit.attrib["upper"])
                frame = axis, parent.attrib["link"], child.attrib["link"], xyz, rpy
            except (KeyError, ValueError) as exc:
                raise ValueError(f"XHand joint {name} has invalid axis, frame or position limits") from exc
            if len(axis) != 3 or not isclose(hypot(*axis), 1.0, rel_tol=0, abs_tol=1e-9):
                raise ValueError(f"XHand joint {name} must have a signed unit axis")
            for field, values in (("xyz", xyz), ("rpy", rpy)):
                if len(values) != 3 or not all(map(isfinite, values)):
                    raise ValueError(f"XHand joint {name} origin {field} must have three finite values")
            if bounds != (lower, upper):
                raise ValueError(f"XHand joint {name} URDF limits {bounds} differ from rated limits {(lower, upper)} rad")
            joint_frames.append(frame)
        if joint_frames[0] != joint_frames[1]:
            raise ValueError(f"XHand joint {name} signed axis or joint frame differs from right-hand URDF")


class ArmFK:
    """URDF EEF 与 xArm 固件 EEF 的坐标定义不同。"""

    def __init__(self):
        import pinocchio as pin

        self.pin = pin
        self.model = pin.buildModelFromXML(kinematic_urdf("link_base", stop_link="right_hand_link"))
        active = tuple(self.model.names[1:])
        if self.model.nq != 7 or active != XARM7_JOINT_NAMES:
            raise ValueError("arm FK active q order must match xArm7 URDF joints")
        self.data = self.model.createData()
        self.eef_frame_id = self.model.getFrameId("custom_eef_link")
        handbase_id = self.model.getFrameId("right_hand_link")
        if max(self.eef_frame_id, handbase_id) >= self.model.nframes:
            raise ValueError("arm URDF must include custom_eef_link and right_hand_link")
        pin.forwardKinematics(self.model, self.data, np.zeros(self.model.nq))
        eef = pin.updateFramePlacement(self.model, self.data, self.eef_frame_id)
        handbase = pin.updateFramePlacement(self.model, self.data, handbase_id)
        self.eef_from_handbase = (eef.inverse() * handbase).homogeneous.copy()

    def forward(self, qpos: np.ndarray) -> np.ndarray:
        """返回独立的基座系 EEF SE(3)，后续 FK 不会覆写它。"""
        self.pin.forwardKinematics(self.model, self.data, np.asarray(qpos, dtype=np.float64))
        pose = self.pin.updateFramePlacement(self.model, self.data, self.eef_frame_id)
        return pose.homogeneous.copy()


class HandFK:
    """从组合 URDF 提取手部子树，输入关节及雅可比列均按 SDK 顺序。"""

    def __init__(self):
        import pinocchio as pin

        self.pin = pin
        urdf = kinematic_urdf("right_hand_link")
        validate_xhand_urdf(urdf)
        self.model = pin.buildModelFromXML(urdf)
        self.data = self.model.createData()
        active = tuple(self.model.names[1:])
        if self.model.nq != HAND_DOF or self.model.nv != HAND_DOF or set(active) != set(XHAND_SDK_JOINT_NAMES):
            raise ValueError("hand FK must contain the twelve XHand SDK joints")
        self.sdk_to_urdf = np.array([
            XHAND_SDK_JOINT_NAMES.index(name) for name in active
        ], dtype=np.intp)
        self.fingertip_frame_ids = tuple(self.model.getFrameId(n) for n in XHAND_FINGERTIP_LINK_NAMES)
        if any(fid >= self.model.nframes for fid in self.fingertip_frame_ids):
            raise ValueError("fingertip frame is missing from the hand URDF")

    def neutral_joint_geometry(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 SDK 顺序的中立轴心与手基座系转轴。"""
        self.pin.computeJointJacobians(self.model, self.data, np.zeros(self.model.nq))
        origins, axes = [], []
        for name in XHAND_SDK_JOINT_NAMES:
            jid = self.model.getJointId(name)
            joint = self.model.joints[jid]
            origins.append(self.data.oMi[jid].translation.copy())
            jac = self.pin.getJointJacobian(
                self.model, self.data, jid, self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            axes.append(jac[3:, joint.idx_v].copy())
        return np.array(origins), np.array(axes)

    def forward(
        self, qpos: np.ndarray, frame_ids: Sequence[int], *, out: np.ndarray | None = None,
    ) -> np.ndarray:
        q = validate_hand_qpos(qpos)[self.sdk_to_urdf]
        self.pin.forwardKinematics(self.model, self.data, q)
        return np.stack([
            self.pin.updateFramePlacement(self.model, self.data, fid).homogeneous
            for fid in frame_ids
        ], out=out)


def eef_poses(arm_qpos: np.ndarray, *, arm_fk: ArmFK) -> np.ndarray:
    """用调用方拥有的 FK 返回 float64[T,4,4] 基座系 EEF 变换。"""
    poses = np.empty((len(arm_qpos), 4, 4), dtype=np.float64)
    for index, qpos in enumerate(arm_qpos):
        poses[index] = arm_fk.forward(qpos)
    return poses
