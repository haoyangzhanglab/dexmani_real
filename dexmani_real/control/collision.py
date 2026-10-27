"""标称 URDF 的自碰撞、环境碰撞与桌面距离。"""

from itertools import combinations

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.robot.model import (
    HAND_SDK_TO_URDF_IDX,
    XARM7_JOINT_NAMES,
    XARM7_XHAND_RIGHT_URDF_PATH,
    XARM7_XHAND_SRDF_PATH,
    XHAND_MODEL_DIR,
    XHAND_URDF_JOINT_NAMES,
)


class CollisionModel:
    def __init__(self, *, static_boxes=(), table=None, table_plane_abcd=None):
        import pinocchio as pin
        from hppfcl.hppfcl import Box

        self.pin = pin
        urdf = str(XARM7_XHAND_RIGHT_URDF_PATH)
        self.model = pin.buildModelFromUrdf(urdf)
        active = sorted(
            (joint.idx_q, self.model.names[i])
            for i, joint in enumerate(self.model.joints)
            if i and joint.nq > 0
        )
        if self.model.nq != 19 or tuple(name for _, name in active) != (
            *XARM7_JOINT_NAMES,
            *XHAND_URDF_JOINT_NAMES,
        ):
            raise ValueError("collision model joint order must be xArm7 + XHand URDF joints")
        self.data = self.model.createData()
        self.geometry = pin.buildGeomFromUrdf(
            self.model, urdf, pin.GeometryType.COLLISION, package_dirs=[str(XHAND_MODEL_DIR)]
        )
        robot_count = self.geometry.ngeoms
        # 同一 parent joint 的几何也参与检查，仅由 SRDF 排除碰撞对。
        for first, second in combinations(range(robot_count), 2):
            self.geometry.addCollisionPair(pin.CollisionPair(first, second))
        pin.removeCollisionPairs(self.model, self.geometry, str(XARM7_XHAND_SRDF_PATH))
        self.environment_names = {}
        self.table_pair_indices = []
        self.table = table if table is not None and table.enabled else None

        for box in static_boxes:
            rotation = Rotation.from_quat(np.asarray(box.quat_wxyz)[[1, 2, 3, 0]]).as_matrix()
            placement = pin.SE3(rotation, np.asarray(box.center_xyz_m))
            self.add_obstacle(box.name, Box(*box.size_xyz_m), placement, robot_count)

        if self.table is not None:
            plane = np.asarray(table_plane_abcd, dtype=np.float64)
            norm = np.linalg.norm(plane[:3])
            normal = plane[:3] / norm
            origin = -(plane[3] / norm) * normal
            placement = pin.SE3(
                self.table_rotation(normal), origin - 0.5 * table.thickness_m * normal
            )
            first_pair = len(self.geometry.collisionPairs)
            self.add_obstacle(
                "table",
                Box(*table.size_xy_m, table.thickness_m),
                placement,
                robot_count,
                allowed_links=table.allowed_contact_links,
            )
            self.table_pair_indices = list(range(first_pair, len(self.geometry.collisionPairs)))

        self.geometry_data = self.geometry.createData()
        self.hand_frame_ids = [
            index
            for index, frame in enumerate(self.model.frames)
            if frame.name.startswith("right_hand_")
        ]
        self.eef_frame_id = self.model.getFrameId("custom_eef_link")
        self.hand_qpos = None

    def add_obstacle(self, name, shape, placement, robot_count, *, allowed_links=()):
        geometry = self.pin.GeometryObject(f"environment::{name}", 0, 0, shape, placement)
        obstacle_id = self.geometry.addGeometryObject(geometry)
        self.environment_names[obstacle_id] = name
        for robot_id in range(robot_count):
            if self.link_name(robot_id) not in allowed_links:
                self.geometry.addCollisionPair(self.pin.CollisionPair(robot_id, obstacle_id))

    @staticmethod
    def table_rotation(normal: np.ndarray) -> np.ndarray:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        local_x = reference - float(np.dot(reference, normal)) * normal
        if float(np.linalg.norm(local_x)) < 1e-6:
            reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            local_x = reference - float(np.dot(reference, normal)) * normal
        local_x /= np.linalg.norm(local_x)
        local_y = np.cross(normal, local_x)
        return np.column_stack((local_x, local_y, normal))

    def set_hand_qpos(self, hand_qpos):
        hand_qpos = np.asarray(hand_qpos, dtype=np.float64)
        if hand_qpos.shape != (12,) or not np.isfinite(hand_qpos).all():
            raise ValueError("collision hand posture must be finite shape (12,)")
        self.hand_qpos = hand_qpos[list(HAND_SDK_TO_URDF_IDX)]

    def rejection(self, arm_qpos):
        arm_qpos = np.asarray(arm_qpos, dtype=np.float64)
        if arm_qpos.shape != (7,) or not np.isfinite(arm_qpos).all():
            raise ValueError("collision arm posture must be finite shape (7,)")
        if self.hand_qpos is None:
            raise RuntimeError("set the collision hand posture before querying arm poses")
        qpos = np.concatenate((arm_qpos, self.hand_qpos))
        if np.any(qpos < self.model.lowerPositionLimit) or np.any(
            qpos > self.model.upperPositionLimit
        ):
            return "model_joint_limits"
        collided = self.pin.computeCollisions(
            self.model, self.data, self.geometry, self.geometry_data, qpos, True
        )
        if not collided:
            return None
        index = next(
            i
            for i, result in enumerate(self.geometry_data.collisionResults)
            if result.isCollision()
        )
        pair = self.geometry.collisionPairs[index]
        kind = (
            "environment_collision" if pair.second in self.environment_names else "self_collision"
        )
        return f"{kind}: {self.link_name(pair.first)}/{self.link_name(pair.second)}"

    # 以下查询共用最近一次 rejection 更新的运动学。
    def eef_position(self):
        return self.pin.updateFramePlacement(self.model, self.data, self.eef_frame_id).translation

    def minimum_hand_frame_z(self):
        self.pin.updateFramePlacements(self.model, self.data)
        heights = [self.data.oMf[index].translation[2] for index in self.hand_frame_ids]
        if not heights or not np.isfinite(heights).all():
            raise RuntimeError("XHand frame placements unavailable")
        return float(min(heights))

    def minimum_table_distance(self):
        minimum = min(
            float(self.pin.computeDistance(self.geometry, self.geometry_data, index).min_distance)
            for index in self.table_pair_indices
        )
        if not np.isfinite(minimum):
            raise RuntimeError("table distance query returned a non-finite value")
        return minimum

    def link_name(self, geom_id):
        if geom_id in self.environment_names:
            return self.environment_names[geom_id]
        geometry = self.geometry.geometryObjects[geom_id]
        return self.model.frames[geometry.parentFrame].name
