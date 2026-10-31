"""xArm 基座系的机械臂 FK，以及手基座系的 XHand link FK。"""

import xml.etree.ElementTree as ET
from collections.abc import Sequence

import numpy as np

from dexmani_real.robot.model import (
    XARM7_JOINT_NAMES,
    XARM7_XHAND_RIGHT_URDF_PATH,
    XHAND_FINGERTIP_LINK_NAMES,
    XHAND_SDK_JOINT_NAMES,
)


def kinematic_urdf(root_link: str, *, stop_link: str | None = None) -> str:
    """从完整组合 URDF 提取运动学子树，保留原始 joint 定义并移除网格。"""
    root = ET.parse(XARM7_XHAND_RIGHT_URDF_PATH).getroot()
    if root_link not in {link.attrib["name"] for link in root.findall("link")}:
        raise ValueError(f"robot URDF is missing {root_link}")
    children: dict[str, list[tuple[str, str]]] = {}
    for joint in root.findall("joint"):
        parent, child = joint.find("parent"), joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"URDF joint {joint.attrib['name']} must name its parent and child")
        children.setdefault(parent.attrib["link"], []).append(
            (joint.attrib["name"], child.attrib["link"]),
        )
    links, joints = set(), set()
    pending = [root_link]
    while pending:
        name = pending.pop()
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
        self.eef_from_handbase = (eef.inverse() * handbase).homogeneous

    def forward(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.pin.forwardKinematics(self.model, self.data, np.asarray(qpos, dtype=np.float64))
        pose = self.pin.updateFramePlacement(self.model, self.data, self.eef_frame_id)
        rotation = pose.rotation
        return pose.translation.copy(), np.concatenate((rotation[:, 0], rotation[:, 1]))


class HandFK:
    """从组合 URDF 提取手部子树，输入关节及雅可比列均按 SDK 顺序。"""

    def __init__(self, fingertip_link_names: Sequence[str] = XHAND_FINGERTIP_LINK_NAMES):
        import pinocchio as pin

        self.pin = pin
        self.model = pin.buildModelFromXML(kinematic_urdf("right_hand_link"))
        self.data = self.model.createData()
        active = tuple(self.model.names[1:])
        if self.model.nq != 12 or set(active) != set(XHAND_SDK_JOINT_NAMES):
            raise ValueError("hand FK must contain the twelve XHand SDK joints")
        self.sdk_to_urdf = np.array([
            XHAND_SDK_JOINT_NAMES.index(name) for name in active
        ], dtype=np.intp)
        self.urdf_to_sdk = np.argsort(self.sdk_to_urdf)
        if len(fingertip_link_names) != 5 or len(set(fingertip_link_names)) != 5:
            raise ValueError("hand FK requires five distinct fingertip frames")
        self.fingertip_frame_ids = tuple(self.model.getFrameId(n) for n in fingertip_link_names)
        if any(fid >= self.model.nframes for fid in self.fingertip_frame_ids):
            raise ValueError("fingertip frame is missing from the hand URDF")

    def forward(self, qpos: np.ndarray, frame_ids: Sequence[int]) -> np.ndarray:
        q = np.asarray(qpos, dtype=np.float64)[self.sdk_to_urdf]
        self.pin.forwardKinematics(self.model, self.data, q)
        # stack 复制 placement，避免后续 FK 覆盖结果。
        return np.stack([
            self.pin.updateFramePlacement(self.model, self.data, fid).homogeneous
            for fid in frame_ids
        ])

    def positions_and_jacobians(self, qpos: np.ndarray, frame_ids: Sequence[int]):
        """手基座系位置及其对 SDK 顺序关节的解析导数；一次 FK 服务所有目标。"""
        q = np.asarray(qpos, dtype=np.float64)[self.sdk_to_urdf]
        self.pin.computeJointJacobians(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        positions = np.stack([self.data.oMf[fid].translation for fid in frame_ids])
        jacobians = np.stack([
            self.pin.getFrameJacobian(
                self.model, self.data, fid, self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:3]
            for fid in frame_ids
        ])[:, :, self.urdf_to_sdk]
        return positions, jacobians


def eef_poses(arm_qpos: np.ndarray, *, arm_fk: ArmFK | None = None) -> np.ndarray:
    """返回 float64[T,9]，依次为基座系 XYZ 和旋转矩阵前两列。"""
    fk = ArmFK() if arm_fk is None else arm_fk
    poses = np.empty((len(arm_qpos), 9), dtype=np.float64)
    for index, qpos in enumerate(arm_qpos):
        poses[index, :3], poses[index, 3:] = fk.forward(qpos)
    return poses
