"""Nominal-URDF collisions; real-pose mount compensation stays in FK."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from dexmani_real.config.environment import StaticCollisionBox, TableCollisionConfig, validate_table_plane
from dexmani_real.robot.model import (
    ARM_JOINT_SHAPE,
    HAND_DOF,
    HAND_SDK_TO_URDF_IDX,
    XARM7_JOINT_NAMES,
    XARM7_XHAND_RIGHT_URDF_PATH,
    XARM7_XHAND_SRDF_PATH,
    XHAND_MODEL_DIR,
    XHAND_URDF_JOINT_NAMES,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CollisionPair:
    """A collision contact suitable for diagnostics and logs."""

    link_name1: str
    link_name2: str
    object_name1: str
    object_name2: str
    collision_type: str


@dataclass(frozen=True)
class CollisionInfo:
    in_collision: bool
    collision_pairs: tuple[CollisionPair, ...] = ()
    num_contacts: int = 0
    sample_qpos_rad: tuple[float, ...] | None = None

    def __bool__(self) -> bool:
        return self.in_collision


class CollisionModel:
    """19-DOF URDF-order collision model; arm-only queries expand with the supplied SDK-order hand pose."""

    def __init__(
        self,
        *,
        static_boxes: Iterable[StaticCollisionBox] = (),
        table: TableCollisionConfig | None = None,
        table_plane_abcd: tuple[float, float, float, float] | None = None,
    ) -> None:
        import pinocchio as pin

        self.pin = pin
        urdf_path = str(XARM7_XHAND_RIGHT_URDF_PATH)
        package_dirs = [str(XHAND_MODEL_DIR)]

        self.model = pin.buildModelFromUrdf(urdf_path)
        active = sorted(
            (
                (joint.idx_q, self.model.names[i])
                for i, joint in enumerate(self.model.joints)
                if i != 0 and joint.nq > 0
            ),
        )
        if self.model.nq != 19 or tuple(name for _, name in active) != (
            *XARM7_JOINT_NAMES,
            *XHAND_URDF_JOINT_NAMES,
        ):
            raise ValueError("collision model active q order must be xArm7 + XHand URDF joints")
        self.data = self.model.createData()

        self.collision_model = pin.buildGeomFromUrdf(
            self.model, urdf_path, pin.GeometryType.COLLISION, package_dirs=package_dirs
        )

        # Include geometries sharing a parent joint; only SRDF excludes pairs.
        import itertools

        n = self.collision_model.ngeoms
        for i, j in itertools.combinations(range(n), 2):
            self.collision_model.addCollisionPair(pin.CollisionPair(i, j))
        pin.removeCollisionPairs(self.model, self.collision_model, str(XARM7_XHAND_SRDF_PATH))

        self.collision_data = self.collision_model.createData()

        self.environment_collision_model = None
        self.environment_names: dict[int, str] = {}
        self.table_pair_indices: tuple[int, ...] = ()
        self.table = table if table is not None and table.enabled else None
        if self.table is not None:
            self.table.validate()
            table_plane_abcd = validate_table_plane(table_plane_abcd)
        boxes = tuple(static_boxes)
        for box in boxes:
            box.validate()
        box_names = [box.name for box in boxes]
        if len(box_names) != len(set(box_names)):
            raise ValueError("static collision box names must be unique")
        if boxes or self.table is not None:
            self.environment_collision_model = pin.buildGeomFromUrdf(
                self.model, urdf_path, pin.GeometryType.COLLISION, package_dirs=package_dirs
            )
            robot_geom_count = int(self.environment_collision_model.ngeoms)
            from hppfcl.hppfcl import Box

            for box in boxes:
                placement = pin.SE3(
                    self.quat_wxyz_to_matrix(box.quat_wxyz), np.asarray(box.center_xyz_m)
                )
                geometry = pin.GeometryObject(
                    f"environment::{box.name}",
                    0,
                    0,
                    Box(*box.size_xyz_m),
                    placement,
                )
                obstacle_id = int(self.environment_collision_model.addGeometryObject(geometry))
                self.environment_names[obstacle_id] = box.name
                for robot_id in range(robot_geom_count):
                    self.environment_collision_model.addCollisionPair(
                        pin.CollisionPair(robot_id, obstacle_id)
                    )
            if self.table is not None:
                table_config = self.table
                plane = np.asarray(table_plane_abcd, dtype=np.float64)
                normal_norm = np.linalg.norm(plane[:3])
                normal = plane[:3] / normal_norm
                surface_origin = -(plane[3] / normal_norm) * normal
                placement = pin.SE3(
                    self.table_rotation(normal),
                    surface_origin - 0.5 * table_config.thickness_m * normal,
                )
                geometry = pin.GeometryObject(
                    "environment::table",
                    0,
                    0,
                    Box(*table_config.size_xy_m, table_config.thickness_m),
                    placement,
                )
                table_id = int(self.environment_collision_model.addGeometryObject(geometry))
                self.environment_names[table_id] = "table"
                table_pair_indices: list[int] = []
                allowed = table_config.allowed_contact_links
                for robot_id in range(robot_geom_count):
                    link_name = self.get_environment_geom_link_name(robot_id)
                    if link_name in allowed:
                        continue
                    self.environment_collision_model.addCollisionPair(
                        pin.CollisionPair(robot_id, table_id)
                    )
                    table_pair_indices.append(
                        len(self.environment_collision_model.collisionPairs) - 1
                    )
                self.table_pair_indices = tuple(table_pair_indices)
        self.environment_collision_data = (
            self.environment_collision_model.createData()
            if self.environment_collision_model is not None
            else None
        )
        self.nq: int = self.model.nq
        logger.debug(
            "CollisionModel ready: %d DOF (7 arm + 12 hand), %d geometries, %d self pairs, "
            "%d static boxes, table=%s, %d environment pairs",
            self.nq,
            self.collision_model.ngeoms,
            len(self.collision_model.collisionPairs),
            len(boxes),
            "enabled" if self.table is not None else "disabled",
            (
                len(self.environment_collision_model.collisionPairs)
                if self.environment_collision_model is not None
                else 0
            ),
        )

        self.expected_qpos_shape: tuple[int, ...] = (self.nq,)

        # Arm-only queries require an explicit hand posture.
        self.hand_qpos: np.ndarray | None = None

    @staticmethod
    def table_rotation(normal: np.ndarray) -> np.ndarray:
        """Return a stable local-box orientation whose +Z axis is *normal*."""
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        local_x = reference - float(np.dot(reference, normal)) * normal
        if float(np.linalg.norm(local_x)) < 1e-6:
            reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            local_x = reference - float(np.dot(reference, normal)) * normal
        local_x /= np.linalg.norm(local_x)
        local_y = np.cross(normal, local_x)
        return np.column_stack((local_x, local_y, normal))

    @staticmethod
    def quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
        w, x, y, z = (float(value) for value in quat_wxyz)
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def set_hand_qpos(self, hand_qpos: np.ndarray) -> None:
        """Set finite SDK-order (12,) hand angles in radians for arm-only collision queries."""
        hand_qpos = np.asarray(hand_qpos, dtype=np.float64)
        if hand_qpos.shape != (HAND_DOF,):
            raise ValueError(f"Expected hand_qpos shape ({HAND_DOF},), got {hand_qpos.shape}")
        if not np.all(np.isfinite(hand_qpos)):
            raise ValueError("hand_qpos contains NaN or Inf — FK would silently fail")
        self.hand_qpos = hand_qpos[list(HAND_SDK_TO_URDF_IDX)]

    def to_full_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """Expand arm (7,) using the saved hand pose, or accept full URDF (19,)."""
        qpos = np.asarray(qpos, dtype=np.float64)
        if not np.all(np.isfinite(qpos)):
            raise ValueError("qpos contains NaN or Inf — collision FK requires finite values")
        if qpos.shape == ARM_JOINT_SHAPE:
            if self.hand_qpos is None:
                raise RuntimeError(
                    "hand_qpos is required for 19-DOF collision checks; "
                    "call set_hand_qpos() before checking an arm-only qpos"
                )
            return np.concatenate([qpos, self.hand_qpos])
        if qpos.shape != self.expected_qpos_shape:
            raise ValueError(
                f"Expected qpos shape {self.expected_qpos_shape} (or (7,) for auto-expand), "
                f"got {qpos.shape}. Model has {self.nq} DOF."
            )
        return qpos

    def within_joint_limits(self, qpos: np.ndarray) -> bool:
        """Check nominal model limits, including hand joints in URDF ordering."""
        full = self.to_full_qpos(qpos)
        return bool(
            np.all(full >= self.model.lowerPositionLimit)
            and np.all(full <= self.model.upperPositionLimit)
        )

    def pin_update(self, qpos: np.ndarray, stop_at_first: bool = True) -> bool:
        """Compute self-collisions for a validated full configuration."""
        qpos = self.to_full_qpos(qpos)
        return bool(
            self.pin.computeCollisions(
                self.model,
                self.data,
                self.collision_model,
                self.collision_data,
                qpos,
                stop_at_first,
            )
        )

    def check_self_collision(self, qpos: np.ndarray) -> bool:
        """Check one configuration, stopping at the first self-collision."""
        return self.pin_update(qpos, stop_at_first=True)

    def minimum_hand_frame_z(self, arm_qpos: np.ndarray) -> float:
        """Lowest nominal hand-link origin in base; mesh extents require an additional margin."""
        qpos = self.to_full_qpos(arm_qpos)
        self.pin.forwardKinematics(self.model, self.data, qpos)
        self.pin.updateFramePlacements(self.model, self.data)
        z_values = [
            float(self.data.oMf[index].translation[2])
            for index, frame in enumerate(self.model.frames)
            if frame.name.startswith("right_hand_")
        ]
        if not z_values or not np.all(np.isfinite(z_values)):
            raise RuntimeError("XHand frame placements unavailable")
        return min(z_values)

    def check_self_collision_details(self, qpos: np.ndarray) -> "CollisionInfo":
        """Return collision contacts; preserve a positive result even if details are unavailable."""
        has_any = self.pin_update(qpos, stop_at_first=False)
        if not has_any:
            return CollisionInfo(in_collision=False)
        pairs: list[CollisionPair] = []
        try:
            for pair_index, pair in enumerate(self.collision_model.collisionPairs):
                if not self.collision_data.collisionResults[pair_index].isCollision():
                    continue
                pairs.append(
                    CollisionPair(
                        link_name1=self.get_geom_link_name(pair.first),
                        link_name2=self.get_geom_link_name(pair.second),
                        object_name1=self.collision_model.geometryObjects[pair.first].name,
                        object_name2=self.collision_model.geometryObjects[pair.second].name,
                        collision_type="pinocchio",
                    )
                )
        except Exception:
            logger.warning("self collision details unavailable", exc_info=True)
        return CollisionInfo(
            in_collision=True, collision_pairs=tuple(pairs), num_contacts=max(1, len(pairs))
        )

    @property
    def has_table(self) -> bool:
        return self.table is not None

    @property
    def table_soft_clearance_m(self) -> float:
        return 0.0 if self.table is None else self.table.soft_clearance_m

    def minimum_table_distance(self, qpos: np.ndarray) -> float:
        """Return the minimum mesh-to-table distance over non-allowed links."""
        if self.table is None or not self.table_pair_indices:
            raise RuntimeError("calibrated table geometry is disabled")
        full_qpos = self.to_full_qpos(qpos)
        self.pin.forwardKinematics(self.model, self.data, full_qpos)
        self.pin.updateGeometryPlacements(
            self.model,
            self.data,
            self.environment_collision_model,
            self.environment_collision_data,
        )
        minimum_m = float("inf")
        for pair_index in self.table_pair_indices:
            result = self.pin.computeDistance(
                self.environment_collision_model,
                self.environment_collision_data,
                pair_index,
            )
            minimum_m = min(minimum_m, float(result.min_distance))
        if not np.isfinite(minimum_m):
            raise RuntimeError("table distance query returned a non-finite value")
        return minimum_m

    def check_environment_collision_details(self, qpos: np.ndarray) -> "CollisionInfo":
        """Return robot link, obstacle name, and sampled qpos for a scene contact."""
        full_qpos = self.to_full_qpos(qpos)
        if self.environment_collision_model is None:
            return CollisionInfo(in_collision=False)
        collided = bool(
            self.pin.computeCollisions(
                self.model,
                self.data,
                self.environment_collision_model,
                self.environment_collision_data,
                full_qpos,
                False,
            )
        )
        if not collided:
            return CollisionInfo(in_collision=False)
        pairs: list[CollisionPair] = []
        try:
            for pair_index, pair in enumerate(self.environment_collision_model.collisionPairs):
                if not self.environment_collision_data.collisionResults[pair_index].isCollision():
                    continue
                robot_id, obstacle_id = int(pair.first), int(pair.second)
                pairs.append(
                    CollisionPair(
                        link_name1=self.get_environment_geom_link_name(robot_id),
                        link_name2=self.environment_names[obstacle_id],
                        object_name1=self.environment_collision_model.geometryObjects[
                            robot_id
                        ].name,
                        object_name2=self.environment_names[obstacle_id],
                        collision_type="environment",
                    )
                )
        except Exception:
            logger.warning("environment collision details unavailable", exc_info=True)
        return CollisionInfo(
            in_collision=True,
            collision_pairs=tuple(pairs),
            num_contacts=max(1, len(pairs)),
            sample_qpos_rad=tuple(float(value) for value in full_qpos),
        )

    def check_collision_details(self, qpos: np.ndarray) -> "CollisionInfo":
        """Detailed combined query; self collision takes diagnostic precedence."""
        self_info = self.check_self_collision_details(qpos)
        return self_info if self_info else self.check_environment_collision_details(qpos)

    def check_path(
        self,
        path: np.ndarray,
    ) -> tuple[int, CollisionInfo] | None:
        """Check the owner's dense samples and return the first colliding sample index."""
        path = np.asarray(path, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != 7 or not np.isfinite(path).all():
            raise ValueError("collision path must be finite shape (N, 7)")
        for index, sample in enumerate(path):
            info = self.check_collision_details(sample)
            if info:
                return index, info
        return None

    def get_geom_link_name(self, geom_id: int) -> str:
        geom = self.collision_model.geometryObjects[geom_id]
        parent_frame = geom.parentFrame
        if parent_frame < len(self.model.frames):
            return self.model.frames[parent_frame].name
        return geom.name

    def get_environment_geom_link_name(self, geom_id: int) -> str:
        geom = self.environment_collision_model.geometryObjects[geom_id]
        parent_frame = geom.parentFrame
        if parent_frame < len(self.model.frames):
            return self.model.frames[parent_frame].name
        return geom.name
