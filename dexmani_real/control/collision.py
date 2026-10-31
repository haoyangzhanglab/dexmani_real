"""组合 URDF 的自碰撞、环境碰撞与桌面距离。"""

from itertools import combinations

import numpy as np

from dexmani_real.robot.model import (
    XARM7_JOINT_NAMES,
    XARM7_XHAND_RIGHT_URDF_PATH,
    XARM7_XHAND_SRDF_PATH,
    XHAND_MODEL_DIR,
    XHAND_SDK_JOINT_NAMES,
)


class CollisionModel:
    def __init__(self, *, static_boxes=(), table=None, table_plane_abcd=None):
        import pinocchio as pin
        from hppfcl.hppfcl import Box

        self.pin = pin
        urdf = str(XARM7_XHAND_RIGHT_URDF_PATH)
        self.model = pin.buildModelFromUrdf(urdf)
        active = set(self.model.names[1:])
        sdk_names = (*XARM7_JOINT_NAMES, *XHAND_SDK_JOINT_NAMES)
        if self.model.nq != 19 or active != set(sdk_names):
            raise ValueError("collision model must contain the xArm7 and XHand SDK joints")
        self.sdk_q_indices = np.array([
            self.model.joints[self.model.getJointId(name)].idx_q for name in sdk_names
        ], dtype=np.intp)
        self.data = self.model.createData()
        self.geometry = pin.buildGeomFromUrdf(
            self.model, urdf, pin.GeometryType.COLLISION, package_dirs=[str(XHAND_MODEL_DIR)]
        )
        robot_count = self.geometry.ngeoms
        hand_geometry_ids = {
            index
            for index, geometry in enumerate(self.geometry.geometryObjects)
            if self.model.frames[geometry.parentFrame].name.startswith("right_hand_")
        }
        # 手内网格尚未验证，统一跳过；其余对保留同 parent joint 检查，再由 SRDF 排除。
        for first, second in combinations(range(robot_count), 2):
            if first in hand_geometry_ids and second in hand_geometry_ids:
                continue
            self.geometry.addCollisionPair(pin.CollisionPair(first, second))
        pin.removeCollisionPairs(self.model, self.geometry, str(XARM7_XHAND_SRDF_PATH))
        self.environment_names = {}
        self.table_pair_indices = []
        self.table = table

        for index, box in enumerate(static_boxes):
            placement = pin.SE3(np.eye(3), np.asarray(box.center_xyz_m))
            self.add_obstacle(f"obstacle_{index}", Box(*box.size_xyz_m), placement, robot_count)

        if self.table is not None:
            unknown_links = set(table.allowed_contact_links) - {
                frame.name for frame in self.model.frames if frame.type == pin.FrameType.BODY
            }
            if unknown_links:
                raise ValueError(f"table allowed_contact_links not found in URDF: {sorted(unknown_links)}")
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
        for index in self.table_pair_indices:
            self.geometry_data.collisionRequests[index].enable_distance_lower_bound = True
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
        # 来源为已投影目标、设备反馈或已校验的 HOME 姿态；隔离后续来源更新。
        self.hand_qpos = np.array(hand_qpos, dtype=np.float64, copy=True)

    def rejection(self, arm_qpos):
        if self.hand_qpos is None:
            raise RuntimeError("set the collision hand posture before querying arm poses")
        qpos = np.empty(self.model.nq)
        qpos[self.sdk_q_indices] = np.concatenate((arm_qpos, self.hand_qpos))
        # 命令限位由调用方检查；手部实测姿态按原值参与几何计算。
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

    def minimum_table_distance(self):
        # 最近一次 rejection 无碰撞时，所有桌面对的下界均已更新。
        results = self.geometry_data.collisionResults
        pairs = sorted(
            (results[index].distance_lower_bound, index)
            for index in self.table_pair_indices
        )
        minimum = np.inf
        for lower_bound, index in pairs:
            if lower_bound >= minimum:
                break
            minimum = min(
                minimum,
                float(self.pin.computeDistance(self.geometry, self.geometry_data, index).min_distance),
            )
        if not np.isfinite(minimum):
            raise RuntimeError("table distance query returned a non-finite value")
        return minimum

    def link_name(self, geom_id):
        if geom_id in self.environment_names:
            return self.environment_names[geom_id]
        geometry = self.geometry.geometryObjects[geom_id]
        return self.model.frames[geometry.parentFrame].name
