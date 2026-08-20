"""Standalone Pinocchio collision model for xArm7 + XHand.

Builds a lightweight Pinocchio self-collision model from the collision URDF,
independent of MPlib. Uses T-Rex's ``pin.computeCollisions()`` pattern.

A single unified SRDF controls collision pair filtering:

- ``xarm7_xhand.srdf`` — unified SRDF used by both 7-DOF and 19-DOF modes.
  Enables arm-wrist to hand collision detection while keeping hand self-collision
  disabled. The 19-DOF model retains 255 active pairs: 17 arm-arm and
  238 arm-hand; no hand-hand pairs remain active.

Usage::

    from dexmani_real.planning.collision_model import CollisionModel

    cm = CollisionModel()
    cm.check_self_collision(qpos)           # bool
    cm.check_self_collision_details(qpos)   # CollisionInfo (from types.py)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from dexmani_real.config.defaults import hand
from dexmani_real.planning.constants import (
    HAND_SDK_TO_URDF_IDX,
    XARM7_XHAND_COLLISION_URDF_PATH,
    XARM7_XHAND_RIGHT_URDF_PATH,
    XARM7_XHAND_SRDF_PATH,
    XHAND_MODEL_DIR,
)
from dexmani_real.utils.log import ThrottledWarner, get_logger
from dexmani_real.utils.schema import ARM_JOINT_SHAPE, HAND_DOF

if TYPE_CHECKING:
    from .types import CollisionInfo

logger = get_logger(__name__)

# Home (open-hand) posture — used as the default when _hand_qpos hasn't been set.
# Zero = clenched fist, which would pass collision checks that the actual open
# hand would fail.  Values from XHandParams.home_qpos_deg.
_HAND_HOME_QPOS: np.ndarray = np.deg2rad(np.asarray(hand.home_qpos_deg, dtype=np.float64))
_warn_hand_qpos_unset = ThrottledWarner(interval_s=30.0)  # warn every 30s if hand_qpos never set
_collision_detail_warn = ThrottledWarner(interval_s=60.0)

# Collision model assets.
_COLLISION_URDF = str(XARM7_XHAND_COLLISION_URDF_PATH)  # 7-DOF (hand fixed)
_FULL_URDF = str(XARM7_XHAND_RIGHT_URDF_PATH)  # 19-DOF (7 arm + 12 hand)
_COLLISION_SRDF = str(XARM7_XHAND_SRDF_PATH)  # unified SRDF (single source)

_HAND_DOF_COUNT = HAND_DOF

# User→URDF reorder map for set_hand_qpos().  Defined in planning.constants
# (single source of truth shared with hand_kinematics.py).
_HAND_USER_TO_URDF = HAND_SDK_TO_URDF_IDX
_HAND_HOME_QPOS_URDF: np.ndarray = _HAND_HOME_QPOS[list(_HAND_USER_TO_URDF)]


class CollisionModel:
    """Standalone Pinocchio self-collision model for xArm7 + XHand.

    Two modes, sharing the **same unified SRDF**:

    - **7-DOF** (default, ``hand_dof=False``): arm-only collision model from the
      collision URDF (hand joints merged as fixed).  Accepts ``qpos`` of shape
      ``(7,)``.

    - **19-DOF** (``hand_dof=True``): full arm + hand collision model from the
      full URDF (hand joints active).  Accepts ``qpos`` of shape ``(19,)`` —
      first 7 are arm joints, last 12 are hand joints.

    Both modes use ``xarm7_xhand.srdf``, which enables arm-wrist
    to hand collision detection and keeps hand self-collision disabled.

    Parameters:
        hand_dof: If True, build a 19-DOF model with active hand joints.
        urdf_path: Path to the collision URDF (overrides default).
        srdf_path: Path to the collision SRDF (overrides default).
        package_dir: Mesh resolution directory.
    """

    def __init__(
        self,
        hand_dof: bool = False,
        urdf_path: str | None = None,
        srdf_path: str | None = None,
        package_dir: str | None = None,
        static_boxes: Iterable[Any] = (),
        table: Any | None = None,
    ) -> None:
        import pinocchio as pin

        self._pin = pin
        self._hand_dof = hand_dof

        pkg = package_dir or str(XHAND_MODEL_DIR)
        _urdf = urdf_path or (_FULL_URDF if hand_dof else _COLLISION_URDF)
        _srdf = srdf_path or _COLLISION_SRDF

        # Build model.
        self._model = pin.buildModelFromUrdf(_urdf)
        self._data = self._model.createData()

        # Build collision geometry.
        self._collision_model = pin.buildGeomFromUrdf(
            self._model, _urdf, pin.GeometryType.COLLISION, package_dirs=[pkg]
        )

        # Add all pairs, then apply SRDF exclusions.
        import itertools

        n = self._collision_model.ngeoms
        for i, j in itertools.combinations(range(n), 2):
            self._collision_model.addCollisionPair(pin.CollisionPair(i, j))
        pin.removeCollisionPairs(self._model, self._collision_model, _srdf)

        # Hand self-collision is NOT checked: SRDF filtering leaves no active
        # hand-hand pair, while arm↔hand pairs remain enabled.
        # Arm↔hand collisions (e.g., wrist hitting fingers) remain active.

        self._collision_data = self._collision_model.createData()

        # A separate geometry model contains only robot↔static-obstacle pairs.
        # It deliberately has no robot↔robot or obstacle↔obstacle pairs, so the
        # existing self-collision API and SRDF semantics remain unchanged.
        self._environment_collision_model = pin.buildGeomFromUrdf(
            self._model, _urdf, pin.GeometryType.COLLISION, package_dirs=[pkg]
        )
        self._robot_environment_geom_count = int(self._environment_collision_model.ngeoms)
        self._environment_names: dict[int, str] = {}
        self._table_pair_indices: tuple[int, ...] = ()
        self._table = self._normalize_table(table)
        normalized_boxes = tuple(self._normalize_static_box(box) for box in static_boxes)
        box_names = [str(box["name"]) for box in normalized_boxes]
        if len(box_names) != len(set(box_names)):
            raise ValueError("static collision box names must be unique")
        if normalized_boxes or self._table is not None:
            from hppfcl.hppfcl import Box

            for box in normalized_boxes:
                placement = pin.SE3(self._quat_wxyz_to_matrix(box["quat_wxyz"]), box["center_xyz_m"])
                geometry = pin.GeometryObject(
                    f"environment::{box['name']}",
                    0,
                    0,
                    Box(*box["size_xyz_m"]),
                    placement,
                )
                obstacle_id = int(self._environment_collision_model.addGeometryObject(geometry))
                self._environment_names[obstacle_id] = str(box["name"])
                for robot_id in range(self._robot_environment_geom_count):
                    self._environment_collision_model.addCollisionPair(pin.CollisionPair(robot_id, obstacle_id))
            if self._table is not None:
                table_config = self._table
                normal = table_config["normal"]
                placement = pin.SE3(
                    self._table_rotation(normal),
                    table_config["surface_origin"] - 0.5 * table_config["thickness_m"] * normal,
                )
                geometry = pin.GeometryObject(
                    "environment::table",
                    0,
                    0,
                    Box(*table_config["size_xyz_m"]),
                    placement,
                )
                table_id = int(self._environment_collision_model.addGeometryObject(geometry))
                self._environment_names[table_id] = "table"
                table_pair_indices: list[int] = []
                allowed = table_config["allowed_contact_links"]
                for robot_id in range(self._robot_environment_geom_count):
                    link_name = self._get_environment_geom_link_name(robot_id)
                    if link_name in allowed:
                        continue
                    self._environment_collision_model.addCollisionPair(pin.CollisionPair(robot_id, table_id))
                    table_pair_indices.append(len(self._environment_collision_model.collisionPairs) - 1)
                self._table_pair_indices = tuple(table_pair_indices)
        self._environment_collision_data = self._environment_collision_model.createData()
        self._static_boxes = normalized_boxes
        self._environment_obstacle_count = len(normalized_boxes) + int(self._table is not None)

        self._nq: int = self._model.nq
        logger.info(
            "CollisionModel ready: %d DOF%s, %d geometries, %d self pairs, "
            "%d static boxes, table=%s, %d environment pairs",
            self._nq,
            " (7 arm + 12 hand)" if hand_dof else "",
            self._collision_model.ngeoms,
            len(self._collision_model.collisionPairs),
            len(self._static_boxes),
            "calibrated" if self._table is not None else "disabled",
            len(self._environment_collision_model.collisionPairs),
        )

        # Cache qpos shape for validation.
        self._expected_qpos_shape: tuple[int, ...] = (self._nq,)

        # Hand qpos buffer (used in hand_dof mode to auto-expand 7→19 DOF).
        # None = not yet set by caller; set_hand_qpos() assigns a real array.
        self._hand_qpos: np.ndarray | None = None

    @staticmethod
    def _box_value(box: Any, name: str) -> Any:
        if isinstance(box, Mapping):
            return box[name]
        return getattr(box, name)

    @classmethod
    def _normalize_static_box(cls, box: Any) -> dict[str, Any]:
        """Validate and copy a dataclass, frozen config node, or mapping box."""
        raw_name = cls._box_value(box, "name")
        if not isinstance(raw_name, str):
            raise TypeError("static collision box name must be a string")
        name = raw_name
        center = np.asarray(cls._box_value(box, "center_xyz_m"), dtype=np.float64)
        size = np.asarray(cls._box_value(box, "size_xyz_m"), dtype=np.float64)
        quat = np.asarray(cls._box_value(box, "quat_wxyz"), dtype=np.float64)
        if not name.strip() or name != name.strip() or name == "table":
            raise ValueError("static collision box name must be non-empty and must not use reserved name 'table'")
        if center.shape != (3,) or size.shape != (3,) or quat.shape != (4,):
            raise ValueError("static collision box center/size/quaternion shapes are invalid")
        if not np.all(np.isfinite(np.concatenate((center, size, quat)))) or np.any(size <= 0.0):
            raise ValueError("static collision box values must be finite and sizes positive")
        if not np.isclose(float(np.linalg.norm(quat)), 1.0, rtol=0.0, atol=1e-6):
            raise ValueError("static collision box quaternion must have unit norm")
        return {
            "name": name,
            "center_xyz_m": center.copy(),
            "size_xyz_m": size.copy(),
            "quat_wxyz": quat.copy(),
        }

    @classmethod
    def _normalize_table(cls, table: Any | None) -> dict[str, Any] | None:
        """Validate and normalize a calibrated table configuration."""
        if table is None or not bool(cls._box_value(table, "enabled")):
            return None
        plane = np.asarray(cls._box_value(table, "plane_abcd"), dtype=np.float64)
        size_xy = np.asarray(cls._box_value(table, "size_xy_m"), dtype=np.float64)
        thickness_m = float(cls._box_value(table, "thickness_m"))
        soft_clearance_m = float(cls._box_value(table, "soft_clearance_m"))
        allowed = tuple(str(name) for name in cls._box_value(table, "allowed_contact_links"))
        if plane.shape != (4,) or size_xy.shape != (2,):
            raise ValueError("table plane_abcd and size_xy_m shapes are invalid")
        if not np.all(np.isfinite(np.concatenate((plane, size_xy)))):
            raise ValueError("table plane and size must be finite")
        normal_norm = float(np.linalg.norm(plane[:3]))
        if normal_norm <= 1e-9:
            raise ValueError("table plane normal must be non-zero")
        normal = plane[:3] / normal_norm
        d_normalized = float(plane[3] / normal_norm)
        if normal[2] <= 0.0:
            normal = -normal
            d_normalized = -d_normalized
        if np.any(size_xy <= 0.0) or not np.isfinite(thickness_m) or thickness_m <= 0.0:
            raise ValueError("table size and thickness must be finite and positive")
        if not np.isfinite(soft_clearance_m) or soft_clearance_m < 0.0:
            raise ValueError("table soft clearance must be finite and non-negative")
        return {
            "normal": normal,
            "surface_origin": -d_normalized * normal,
            "size_xyz_m": np.array([size_xy[0], size_xy[1], thickness_m], dtype=np.float64),
            "thickness_m": thickness_m,
            "soft_clearance_m": soft_clearance_m,
            "allowed_contact_links": frozenset(allowed),
        }

    @staticmethod
    def _table_rotation(normal: np.ndarray) -> np.ndarray:
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
    def _quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
        w, x, y, z = (float(value) for value in quat_wxyz)
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    # qpos handling.

    def set_hand_qpos(self, hand_qpos: np.ndarray) -> None:
        """Set the current hand joint configuration for auto-expansion.

        In ``hand_dof`` mode, collision check methods accept 7-DOF arm qpos
        and automatically concatenate with this buffer to form a full 19-DOF
        qpos.  Call this each frame before arm collision checks.

        Accepts hand qpos in **user order** (native hardware order):
        ``[thumb_bend, thumb_rota1, thumb_rota2, index_bend, index_j1, index_j2,
        mid_j1, mid_j2, ring_j1, ring_j2, pinky_j1, pinky_j2]``.

        Internally reorders to URDF joint order for correct Pinocchio FK:
        ``[index_…, mid_…, pinky_…, ring_…, thumb_…]``.

        Args:
            hand_qpos: 12-DOF hand joint angles [rad] in **user order**.

        Raises:
            ValueError: If shape is wrong or values contain NaN/Inf.
        """
        hand_qpos = np.asarray(hand_qpos, dtype=np.float64)
        if hand_qpos.shape != (_HAND_DOF_COUNT,):
            raise ValueError(f"Expected hand_qpos shape ({_HAND_DOF_COUNT},), got {hand_qpos.shape}")
        if not np.all(np.isfinite(hand_qpos)):
            raise ValueError("hand_qpos contains NaN or Inf — FK would silently fail")
        # Reorder user→URDF: _hand_user_to_urdf[i] = which user index maps to URDF slot i
        self._hand_qpos = hand_qpos[list(_HAND_USER_TO_URDF)]

    def _to_full_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """Normalize qpos for internal use, auto-expanding arm→full in hand_dof mode.

        - 7-DOF model: always expects ``(7,)``.
        - 19-DOF model: accepts ``(7,)`` (auto-expands with ``_hand_qpos``) or
          ``(19,)`` (uses as-is).
        """
        qpos = np.asarray(qpos, dtype=np.float64)
        if not np.all(np.isfinite(qpos)):
            raise ValueError("qpos contains NaN or Inf — collision FK requires finite values")
        if self._hand_dof and qpos.shape == ARM_JOINT_SHAPE:
            if self._hand_qpos is None:
                _warn_hand_qpos_unset("hand_qpos not set, using home position. Call set_hand_qpos() first.")
                return np.concatenate([qpos, _HAND_HOME_QPOS_URDF])
            return np.concatenate([qpos, self._hand_qpos])
        if qpos.shape != self._expected_qpos_shape:
            raise ValueError(
                f"Expected qpos shape {self._expected_qpos_shape} (or (7,) for auto-expand), "
                f"got {qpos.shape}. Model has {self._nq} DOF."
            )
        return qpos

    # Core collision evaluation.

    def _pin_update(self, qpos: np.ndarray, stop_at_first: bool = True) -> bool:
        """Compute self-collisions for a validated full configuration."""
        qpos = self._to_full_qpos(qpos)
        return bool(
            self._pin.computeCollisions(
                self._model, self._data, self._collision_model, self._collision_data, qpos, stop_at_first
            )
        )

    # Self-collision.

    def check_self_collision(self, qpos: np.ndarray) -> bool:
        """Check if qpos is in self-collision (fast bool, single point).

        Uses ``stop_at_first_collision=True`` for early exit.
        """
        return self._pin_update(qpos, stop_at_first=True)

    def minimum_hand_frame_z(self, arm_qpos: np.ndarray) -> float:
        """Return the lowest XHand link-frame origin in the robot base frame.

        The active hand configuration comes from ``set_hand_qpos`` in 19-DOF
        mode; the fixed hand posture is used by the 7-DOF model.  Callers apply
        an additional mesh-extent margin because frame origins are not surface
        points.  This is substantially more orientation-aware than subtracting
        a constant distance from the EEF origin.
        """
        qpos = self._to_full_qpos(arm_qpos)
        self._pin.forwardKinematics(self._model, self._data, qpos)
        self._pin.updateFramePlacements(self._model, self._data)
        z_values = [
            float(self._data.oMf[index].translation[2])
            for index, frame in enumerate(self._model.frames)
            if frame.name.startswith("right_hand_")
        ]
        if not z_values or not np.all(np.isfinite(z_values)):
            raise RuntimeError("XHand frame placements unavailable")
        return min(z_values)

    def check_self_collision_details(self, qpos: np.ndarray) -> "CollisionInfo":
        """Check self-collision and return structured ``CollisionInfo``.

        Only allocates on collision; returns cached ``CollisionInfo.no_collision()``
        on the common (no-collision) path.
        """
        from .types import CollisionInfo, CollisionPair

        has_any = self._pin_update(qpos, stop_at_first=False)
        if not has_any:
            return CollisionInfo.no_collision()
        # self._collision_data.collisionResults returns a C++ std::vector that
        # can fail pybind11 type conversion on some hpp-fcl builds.  Fall back
        # to a generic "collision detected" result when individual access isn't
        # available.
        try:
            results = self._collision_data.collisionResults
            pairs: list[CollisionPair] = []
            for i in range(len(results)):
                cr = results[i]
                if cr.isCollision():
                    cp = self._collision_model.collisionPairs[i]
                    pairs.append(
                        CollisionPair(
                            link_name1=self._get_geom_link_name(cp.first),
                            link_name2=self._get_geom_link_name(cp.second),
                            object_name1=self._collision_model.geometryObjects[cp.first].name,
                            object_name2=self._collision_model.geometryObjects[cp.second].name,
                            collision_type="pinocchio",
                        )
                    )
        except TypeError:
            # pybind11 type conversion failure — the collision is real but we
            # can't enumerate which pair(s) triggered it.
            _collision_detail_warn(
                "collisionResults type conversion failed — collision detected "
                "but pair details unavailable (hpp-fcl build limitation)"
            )
            return CollisionInfo(in_collision=True, collision_pairs=(), num_contacts=1)
        if not pairs:
            # ``computeCollisions`` is authoritative. Pair enumeration is
            # diagnostic-only; never turn a real collision into a pass.
            return CollisionInfo(in_collision=True, collision_pairs=(), num_contacts=1)
        return CollisionInfo(in_collision=True, collision_pairs=tuple(pairs), num_contacts=len(pairs))

    # Static environment and combined collision checks.

    @property
    def has_static_environment(self) -> bool:
        return self._environment_obstacle_count > 0

    @property
    def has_table(self) -> bool:
        return self._table is not None

    @property
    def table_soft_clearance_m(self) -> float:
        return 0.0 if self._table is None else float(self._table["soft_clearance_m"])

    def minimum_table_distance(self, qpos: np.ndarray) -> float:
        """Return the minimum mesh-to-table distance over non-allowed links."""
        if self._table is None or not self._table_pair_indices:
            raise RuntimeError("calibrated table geometry is disabled")
        full_qpos = self._to_full_qpos(qpos)
        self._pin.forwardKinematics(self._model, self._data, full_qpos)
        self._pin.updateGeometryPlacements(
            self._model,
            self._data,
            self._environment_collision_model,
            self._environment_collision_data,
        )
        minimum_m = float("inf")
        for pair_index in self._table_pair_indices:
            result = self._pin.computeDistance(
                self._environment_collision_model,
                self._environment_collision_data,
                pair_index,
            )
            minimum_m = min(minimum_m, float(result.min_distance))
        if not np.isfinite(minimum_m):
            raise RuntimeError("table distance query returned a non-finite value")
        return minimum_m

    def check_environment_collision(self, qpos: np.ndarray) -> bool:
        """Return whether robot geometry touches any configured static box."""
        if not self.has_static_environment:
            self._to_full_qpos(qpos)  # preserve finite/shape validation
            return False
        full_qpos = self._to_full_qpos(qpos)
        return bool(
            self._pin.computeCollisions(
                self._model,
                self._data,
                self._environment_collision_model,
                self._environment_collision_data,
                full_qpos,
                True,
            )
        )

    def check_environment_collision_details(self, qpos: np.ndarray) -> "CollisionInfo":
        """Return robot link, obstacle name, and sampled qpos for a scene contact."""
        from .types import CollisionInfo, CollisionPair

        full_qpos = self._to_full_qpos(qpos)
        if not self.has_static_environment:
            return CollisionInfo.no_collision()
        self._pin.forwardKinematics(self._model, self._data, full_qpos)
        self._pin.updateGeometryPlacements(
            self._model,
            self._data,
            self._environment_collision_model,
            self._environment_collision_data,
        )
        pairs: list[CollisionPair] = []
        for pair_index, pair in enumerate(self._environment_collision_model.collisionPairs):
            if not self._pin.computeCollision(
                self._environment_collision_model, self._environment_collision_data, pair_index
            ):
                continue
            robot_id = int(pair.first)
            obstacle_id = int(pair.second)
            pairs.append(
                CollisionPair(
                    link_name1=self._get_environment_geom_link_name(robot_id),
                    link_name2=self._environment_names[obstacle_id],
                    object_name1=self._environment_collision_model.geometryObjects[robot_id].name,
                    object_name2=self._environment_names[obstacle_id],
                    collision_type="environment",
                )
            )
        if not pairs:
            return CollisionInfo.no_collision()
        return CollisionInfo(
            in_collision=True,
            collision_pairs=tuple(pairs),
            num_contacts=len(pairs),
            sample_qpos_rad=tuple(float(value) for value in full_qpos),
        )

    def check_collision(self, qpos: np.ndarray) -> bool:
        """Fast combined self + static-environment collision query."""
        return self.check_self_collision(qpos) or self.check_environment_collision(qpos)

    def check_collision_details(self, qpos: np.ndarray) -> "CollisionInfo":
        """Detailed combined query; self collision takes diagnostic precedence."""
        self_info = self.check_self_collision_details(qpos)
        return self_info if self_info else self.check_environment_collision_details(qpos)

    # Segment collision checking.

    @staticmethod
    def _validate_step_size(step_size: float) -> float:
        step_size = float(step_size)
        if not np.isfinite(step_size) or step_size <= 0.0:
            raise ValueError(f"step_size must be finite and > 0, got {step_size!r}")
        return step_size

    @staticmethod
    def _sample_count(q1: np.ndarray, q2: np.ndarray, step_size: float) -> int:
        return max(1, int(np.ceil(float(np.max(np.abs(q2 - q1))) / step_size)))

    def _check_segment_free(self, q1: np.ndarray, q2: np.ndarray, step_size: float) -> bool:
        """Dense joint-space interpolation with early exit on collision."""
        step_size = self._validate_step_size(step_size)
        q1 = self._to_full_qpos(q1)
        q2 = self._to_full_qpos(q2)
        diff = q2 - q1
        n = self._sample_count(q1, q2, step_size)
        for step in range(n + 1):
            if self.check_self_collision(q1 + (step / n) * diff):
                return False
        return True

    def check_segment_collision_free(
        self,
        q1: np.ndarray,
        q2: np.ndarray,
        step_size: float = 0.02,
    ) -> bool:
        """Check if the linear joint-space segment q1→q2 is self-collision-free."""
        return self._check_segment_free(q1, q2, step_size)

    def check_combined_segment_collision_free(
        self,
        q1: np.ndarray,
        q2: np.ndarray,
        step_size: float = 0.02,
    ) -> bool:
        """Densely check a joint segment against self and static geometry."""
        step_size = self._validate_step_size(step_size)
        q1 = self._to_full_qpos(q1)
        q2 = self._to_full_qpos(q2)
        diff = q2 - q1
        count = self._sample_count(q1, q2, step_size)
        for step in range(count + 1):
            if self.check_collision(q1 + (step / count) * diff):
                return False
        return True

    def check_transition_collision_free(
        self,
        arm_start_qpos: np.ndarray,
        arm_end_qpos: np.ndarray,
        hand_start_qpos: np.ndarray,
        hand_end_qpos: np.ndarray,
        step_size_rad: float = 0.02,
    ) -> bool:
        """Check the conservative envelope of independently executed arm and hand motion."""
        if not self._hand_dof:
            raise RuntimeError("arm/hand transition checks require hand_dof=True")
        step_size = self._validate_step_size(step_size_rad)
        arm_start = self._validate_vector(arm_start_qpos, 7, "arm_start_qpos")
        arm_end = self._validate_vector(arm_end_qpos, 7, "arm_end_qpos")
        hand_start = self._user_hand_to_urdf(hand_start_qpos, "hand_start_qpos")
        hand_end = self._user_hand_to_urdf(hand_end_qpos, "hand_end_qpos")
        arm_steps = int(np.ceil(float(np.max(np.abs(arm_end - arm_start))) / step_size))
        hand_steps = int(np.ceil(float(np.max(np.abs(hand_end - hand_start))) / step_size))
        arm_diff = arm_end - arm_start
        hand_diff = hand_end - hand_start
        arm_alphas = (0.0,) if arm_steps == 0 else np.linspace(0.0, 1.0, arm_steps + 1)
        hand_alphas = (0.0,) if hand_steps == 0 else np.linspace(0.0, 1.0, hand_steps + 1)
        for arm_alpha in arm_alphas:
            arm_qpos = arm_start + arm_alpha * arm_diff
            for hand_alpha in hand_alphas:
                hand_qpos = hand_start + hand_alpha * hand_diff
                if self.check_collision(np.concatenate([arm_qpos, hand_qpos])):
                    return False
        return True

    @staticmethod
    def _validate_vector(values: np.ndarray, size: int, name: str) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (size,):
            raise ValueError(f"Expected {name} shape ({size},), got {result.shape}")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} contains NaN or Inf")
        return result

    def _user_hand_to_urdf(self, hand_qpos: np.ndarray, name: str) -> np.ndarray:
        return self._validate_vector(hand_qpos, _HAND_DOF_COUNT, name)[list(_HAND_USER_TO_URDF)]

    # Helpers.

    def _get_geom_link_name(self, geom_id: int) -> str:
        """Get the link name for a geometry object index."""
        geom = self._collision_model.geometryObjects[geom_id]
        parent_frame = geom.parentFrame
        if parent_frame < len(self._model.frames):
            return self._model.frames[parent_frame].name
        return geom.name

    def _get_environment_geom_link_name(self, geom_id: int) -> str:
        geom = self._environment_collision_model.geometryObjects[geom_id]
        parent_frame = geom.parentFrame
        if parent_frame < len(self._model.frames):
            return self._model.frames[parent_frame].name
        return geom.name

    # Properties.

    @property
    def hand_dof(self) -> bool:
        """Whether this model includes active hand joints (19-DOF vs 7-DOF)."""
        return self._hand_dof
