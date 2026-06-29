"""Standalone Pinocchio collision model for xArm7 + XHand.

Builds a lightweight Pinocchio collision model from the collision URDF, independent of
MPlib. Uses T-Rex's `pin.computeCollisions()` pattern for self-collision and environment
collision detection.

A single unified SRDF controls collision pair filtering for both models:

- ``xarm7_xhand_collision_19dof.srdf`` — unified SRDF used by both 7-DOF (MPlib,
  CollisionModel hand_dof=False) and 19-DOF (CollisionModel hand_dof=True) modes.
  Enables arm-wrist to hand collision detection while keeping hand self-collision
  disabled (291 inter-finger Never rules retained).

Performance: ~0.001ms per single-point check for a 7-DOF / 34-geometry / 141-pair model.

Usage::

    from dexmani_real.planning.collision_model import CollisionModel

    cm = CollisionModel()
    cm.check_self_collision(qpos)           # bool
    cm.check_self_collision_details(qpos)   # CollisionInfo (from types.py)
    cm.add_box_obstacle("table", [1.0,2.0,0.04], [0.5, 0.0, -0.04])
    cm.check_env_collision(qpos)            # bool (obstacles only)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real import ASSET_DIR
from dexmani_real.utils.log import get_logger

if TYPE_CHECKING:
    from .types import CollisionInfo

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Collision URDF / SRDF paths
# ---------------------------------------------------------------------------
_XHAND_DIR = ASSET_DIR / "robots" / "xhand"
_COLLISION_URDF = str(_XHAND_DIR / "xarm7_xhand_collision.urdf")       # 7-DOF (hand fixed)
_FULL_URDF = str(_XHAND_DIR / "xarm7_xhand_right.urdf")                # 19-DOF (7 arm + 12 hand)
_COLLISION_SRDF = str(_XHAND_DIR / "xarm7_xhand_collision_19dof.srdf")  # unified SRDF (single source)

# Hand DOF mapping (qpos indices 7..18 in 19-DOF model)
_HAND_DOF_NAMES: tuple[str, ...] = (
    "right_hand_index_bend_joint",
    "right_hand_index_joint1",
    "right_hand_index_joint2",
    "right_hand_mid_joint1",
    "right_hand_mid_joint2",
    "right_hand_pinky_joint1",
    "right_hand_pinky_joint2",
    "right_hand_ring_joint1",
    "right_hand_ring_joint2",
    "right_hand_thumb_bend_joint",
    "right_hand_thumb_rota_joint1",
    "right_hand_thumb_rota_joint2",
)
_HAND_DOF_COUNT = 12  # number of active hand joints


class CollisionModel:
    """Standalone Pinocchio collision model for xArm7 + XHand.

    Two modes, sharing the **same unified SRDF**:

    - **7-DOF** (default, ``hand_dof=False``): arm-only collision model from the
      collision URDF (hand joints merged as fixed).  Accepts ``qpos`` of shape
      ``(7,)``.  ~1.5μs/check.

    - **19-DOF** (``hand_dof=True``): full arm + hand collision model from the
      full URDF (hand joints active).  Accepts ``qpos`` of shape ``(19,)`` —
      first 7 are arm joints, last 12 are hand joints.  ~64μs/check.

    Both modes use ``xarm7_xhand_collision_19dof.srdf``, which enables arm-wrist
    to hand collision detection and keeps hand self-collision disabled.

    Environment obstacles can be added as ``fcl.Box`` geometry objects, following
    T-Rex's ``add_env_obstacles()`` pattern.

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
    ) -> None:
        import coal as fcl
        import pinocchio as pin

        self._pin = pin
        self._fcl = fcl
        self._hand_dof = hand_dof

        pkg = package_dir or str(_XHAND_DIR)
        _urdf = urdf_path or (_FULL_URDF if hand_dof else _COLLISION_URDF)
        _srdf = srdf_path or _COLLISION_SRDF

        # --- Build model ---
        self._model = pin.buildModelFromUrdf(_urdf)
        self._data = self._model.createData()

        # --- Build collision geometry ---
        self._collision_model = pin.buildGeomFromUrdf(
            self._model, _urdf, pin.GeometryType.COLLISION, package_dirs=[pkg]
        )

        # --- Collision pair setup ---
        # Both modes: URDFs have 0 default collision pairs.  Add all N*(N-1)/2
        # possible pairs, then let the SRDF remove Adjacent + Never pairs.
        # 7-DOF:  34²/2 = 561 → after SRDF: ~0–141  (collision URDF)
        # 19-DOF: 40²/2 = 780 → after SRDF: 255      (full URDF)
        import itertools
        n = self._collision_model.ngeoms
        for i, j in itertools.combinations(range(n), 2):
            self._collision_model.addCollisionPair(pin.CollisionPair(i, j))
        pin.removeCollisionPairs(self._model, self._collision_model, _srdf)

        self._collision_data = self._collision_model.createData()

        # --- Obstacle tracking ---
        self._obstacle_names: set[str] = set()

        self._nq: int = self._model.nq
        self._link_names: list[str] = (
            self._model.names.tolist() if hasattr(self._model.names, "tolist") else list(self._model.names)
        )

        logger.info(
            "CollisionModel ready: %d DOF%s, %d geometries, %d collision pairs",
            self._nq,
            " (7 arm + 12 hand)" if hand_dof else "",
            self._collision_model.ngeoms,
            len(self._collision_model.collisionPairs),
        )

        # --- qpos shape cache for validation ---
        self._expected_qpos_shape: tuple[int, ...] = (self._nq,)
        # Arm-only slice for hand_dof mode (first 7 joints)
        self._arm_slice = slice(0, 7) if hand_dof else slice(0, self._nq)

        # Hand qpos buffer (used in hand_dof mode to auto-expand 7→19 DOF)
        self._hand_qpos: np.ndarray = np.zeros(_HAND_DOF_COUNT, dtype=np.float64)

    # ------------------------------------------------------------------
    # qpos handling
    # ------------------------------------------------------------------

    def set_hand_qpos(self, hand_qpos: np.ndarray) -> None:
        """Set the current hand joint configuration for auto-expansion.

        In ``hand_dof`` mode, collision check methods accept 7-DOF arm qpos
        and automatically concatenate with this buffer to form a full 19-DOF
        qpos.  Call this each frame before arm collision checks.

        Args:
            hand_qpos: 12-DOF hand joint angles [rad].
        """
        hand_qpos = np.asarray(hand_qpos, dtype=np.float64)
        if hand_qpos.shape != (_HAND_DOF_COUNT,):
            raise ValueError(f"Expected hand_qpos shape ({_HAND_DOF_COUNT},), got {hand_qpos.shape}")
        self._hand_qpos = hand_qpos.copy()

    def _to_full_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """Normalize qpos for internal use, auto-expanding arm→full in hand_dof mode.

        - 7-DOF model: always expects ``(7,)``.
        - 19-DOF model: accepts ``(7,)`` (auto-expands with ``_hand_qpos``) or
          ``(19,)`` (uses as-is).
        """
        qpos = np.asarray(qpos, dtype=np.float64)
        if self._hand_dof and qpos.shape == (7,):
            return np.concatenate([qpos, self._hand_qpos])
        if qpos.shape != self._expected_qpos_shape:
            raise ValueError(
                f"Expected qpos shape {self._expected_qpos_shape} (or (7,) for auto-expand), "
                f"got {qpos.shape}. Model has {self._nq} DOF."
            )
        return qpos

    # ------------------------------------------------------------------
    # Core collision evaluation
    # ------------------------------------------------------------------

    def _pin_update(self, qpos: np.ndarray, stop_at_first: bool = True) -> tuple[np.ndarray, bool]:
        """Run FK + update geometry placements + compute collisions.

        Returns ``(full_qpos, has_any_collision)`` where ``has_any_collision`` is
        the return value of ``pin.computeCollisions()`` — True if at least one
        collision pair is in collision, False if all pairs are collision-free.

        Callers use ``has_any_collision`` to skip the per-pair iteration on the
        common collision-free path, avoiding ~233 ``isCollision()`` calls per query.
        """
        qpos = self._to_full_qpos(qpos)
        self._pin.forwardKinematics(self._model, self._data, qpos)
        self._pin.updateGeometryPlacements(
            self._model, self._data, self._collision_model, self._collision_data
        )
        has_any = self._pin.computeCollisions(
            self._model, self._data, self._collision_model, self._collision_data, qpos, stop_at_first
        )
        return qpos, has_any

    # ------------------------------------------------------------------
    # Self-collision
    # ------------------------------------------------------------------

    def check_self_collision(self, qpos: np.ndarray) -> bool:
        """Check if qpos is in self-collision (fast bool, single point).

        Uses ``stop_at_first_collision=True`` for early exit.

        - 7-DOF model: ~1.5μs/check
        - 19-DOF model: ~64μs/check

        This is a drop-in replacement for MPlib's ``has_self_collision(qpos)``
        in the teleop hot path, at ~1/100 the cost.
        """
        _qpos, has_any = self._pin_update(qpos, stop_at_first=True)
        if not has_any:
            return False  # fast path: no collision at all → skip per-pair iteration
        return any(
            self._collision_data.collisionResults[i].isCollision()
            and not self._is_obstacle_pair(self._collision_model.collisionPairs[i])
            for i in range(len(self._collision_data.collisionResults))
        )

    def check_self_collision_details(self, qpos: np.ndarray) -> CollisionInfo:
        """Check self-collision and return structured ``CollisionInfo``.

        Only allocates on collision; returns cached ``CollisionInfo.no_collision()``
        on the common (no-collision) path.
        """
        from .types import CollisionInfo, CollisionPair

        _qpos, has_any = self._pin_update(qpos, stop_at_first=False)
        if not has_any:
            return CollisionInfo.no_collision()  # fast path: no collision at all
        results = self._collision_data.collisionResults
        pairs: list[CollisionPair] = []
        for i in range(len(results)):
            cr = results[i]
            if cr.isCollision():
                cp = self._collision_model.collisionPairs[i]
                if self._is_obstacle_pair(cp):
                    continue  # skip obstacle pairs — these are env, not self
                pairs.append(
                    CollisionPair(
                        link_name1=self._get_geom_link_name(cp.first),
                        link_name2=self._get_geom_link_name(cp.second),
                        object_name1=self._get_geom_object_name(cp.first),
                        object_name2=self._get_geom_object_name(cp.second),
                        collision_type="pinocchio",
                    )
                )
        if not pairs:
            return CollisionInfo.no_collision()
        return CollisionInfo(in_collision=True, collision_pairs=tuple(pairs), num_contacts=len(pairs))

    # ------------------------------------------------------------------
    # Segment collision checking
    # ------------------------------------------------------------------

    def check_segment_collision_free(
        self, q1: np.ndarray, q2: np.ndarray, step_size: float = 0.02,
    ) -> bool:
        """Check if the linear joint-space segment q1→q2 is self-collision-free.

        Interpolates at ``step_size`` (L∞ rad) resolution using the fast
        ``check_self_collision()`` (stop_at_first_collision=True).

        Returns True if all intermediate configurations are collision-free.
        """
        q1 = self._to_full_qpos(q1)
        q2 = self._to_full_qpos(q2)
        diff = q2 - q1
        dist = float(np.max(np.abs(diff)))
        if dist <= step_size:
            return not self.check_self_collision(q2)
        n_steps = int(np.ceil(dist / step_size))
        for step in range(1, n_steps + 1):
            alpha = step / n_steps
            q = q1 + alpha * diff
            if self.check_self_collision(q):
                return False
        return True

    def check_segment_env_collision_free(
        self, q1: np.ndarray, q2: np.ndarray, step_size: float = 0.02,
    ) -> bool:
        """Check if the linear joint-space segment q1→q2 is env-collision-free.

        Returns early (True) when no obstacles are registered.
        Returns True if all intermediate configurations are env-collision-free.
        """
        if not self._obstacle_names:
            return True
        q1 = self._to_full_qpos(q1)
        q2 = self._to_full_qpos(q2)
        diff = q2 - q1
        dist = float(np.max(np.abs(diff)))
        if dist <= step_size:
            return not self.check_env_collision(q2)
        n_steps = int(np.ceil(dist / step_size))
        for step in range(1, n_steps + 1):
            alpha = step / n_steps
            q = q1 + alpha * diff
            if self.check_env_collision(q):
                return False
        return True

    # ------------------------------------------------------------------
    # Environment collision (obstacles only)
    # ------------------------------------------------------------------

    def check_env_collision(self, qpos: np.ndarray) -> bool:
        """Check if qpos collides with any added environment obstacles."""
        if not self._obstacle_names:
            return False
        _qpos, has_any = self._pin_update(qpos, stop_at_first=True)
        if not has_any:
            return False  # fast path: no collision at all → skip per-pair iteration
        for i in range(len(self._collision_data.collisionResults)):
            if self._collision_data.collisionResults[i].isCollision():
                if self._is_obstacle_pair(self._collision_model.collisionPairs[i]):
                    return True
        return False

    # ------------------------------------------------------------------
    # Obstacle management (T-Rex add_env_obstacles pattern)
    # ------------------------------------------------------------------

    def add_box_obstacle(
        self,
        name: str,
        half_extents: tuple[float, float, float],
        position: tuple[float, float, float],
        rotation: np.ndarray | None = None,
    ) -> None:
        """Add a static box obstacle to the collision model.

        Uses ``fcl.Box`` and ``pin.GeometryObject`` with parent frame 0
        (world/universe), matching T-Rex's ``add_env_obstacles()`` pattern.

        Args:
            name: Unique obstacle name (used for later removal).
            half_extents: Box half-extents (hx, hy, hz) in metres.
            position: Centre position [x, y, z] in robot base frame.
            rotation: 3×3 rotation matrix (default: identity).
        """
        if name in self._obstacle_names:
            raise ValueError(f"Obstacle '{name}' already exists.")
        rot = rotation if rotation is not None else np.eye(3)
        # fcl.Box(x, y, z) takes FULL extents (not half), so double them.
        shape = self._fcl.Box(half_extents[0] * 2, half_extents[1] * 2, half_extents[2] * 2)
        pose = self._pin.SE3(rot, np.array(position, dtype=np.float64))
        obj = self._pin.GeometryObject(name, 0, pose, shape)
        obj_id = self._collision_model.addGeometryObject(obj)
        # Register collision pairs: this obstacle × every robot geometry except
        # static base/world-frame geometries (parentJoint == 0 === universe).
        for existing_id in range(self._collision_model.ngeoms - 1):
            if self._collision_model.geometryObjects[existing_id].parentJoint == 0:
                continue  # skip static base geometry (e.g. link_base)
            self._collision_model.addCollisionPair(self._pin.CollisionPair(existing_id, obj_id))
        self._obstacle_names.add(name)
        self._collision_data = self._collision_model.createData()
        logger.info("Added box obstacle '%s' at (%s) half_extents=%s", name, position, half_extents)

    def add_table(
        self,
        table_height: float,
        x_center: float = 0.5,
        half_x: float = 1.0,
        half_y: float = 2.0,
        half_z: float = 0.04,
    ) -> None:
        """Convenience method to add a table obstacle (matches T-Rex convention).

        The table is centered at (x_center, 0, table_height - half_z).
        Per T-Rex convention, subtract half_z from table_height so the top
        surface is at table_height.
        """
        self.add_box_obstacle(
            name="table",
            half_extents=(half_x, half_y, half_z),
            position=(x_center, 0.0, table_height - half_z),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_geom_link_name(self, geom_id: int) -> str:
        """Get the link name for a geometry object index."""
        geom = self._collision_model.geometryObjects[geom_id]
        parent_joint = geom.parentJoint
        if parent_joint < len(self._link_names):
            return self._link_names[parent_joint]
        return geom.name

    def _get_geom_object_name(self, geom_id: int) -> str:
        """Get the geometry object name."""
        return self._collision_model.geometryObjects[geom_id].name

    def _is_obstacle_pair(self, pair) -> bool:
        """Check if a collision pair involves an obstacle geometry."""
        g1 = self._collision_model.geometryObjects[pair.first]
        g2 = self._collision_model.geometryObjects[pair.second]
        return g1.name in self._obstacle_names or g2.name in self._obstacle_names

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def nq(self) -> int:
        return self._nq

    @property
    def hand_dof(self) -> bool:
        """Whether this model includes active hand joints (19-DOF vs 7-DOF)."""
        return self._hand_dof

    @property
    def num_hand_dof(self) -> int:
        """Number of active hand DOFs (0 for 7-DOF, 12 for 19-DOF)."""
        return _HAND_DOF_COUNT if self._hand_dof else 0

    @property
    def arm_slice(self) -> slice:
        """Slice to extract arm-only qpos: ``qpos[cm.arm_slice]`` → ``(7,)``."""
        return self._arm_slice
