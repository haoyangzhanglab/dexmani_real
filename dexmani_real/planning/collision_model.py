"""Standalone Pinocchio collision model for xArm7 + XHand.

Builds a lightweight Pinocchio collision model from the collision URDF, independent of
MPlib. Uses T-Rex's ``pin.computeCollisions()`` pattern for self-collision and environment
collision detection.

A single unified SRDF controls collision pair filtering for both models:

- ``xarm7_xhand_collision_19dof.srdf`` — unified SRDF used by both 7-DOF (MPlib,
  CollisionModel hand_dof=False) and 19-DOF (CollisionModel hand_dof=True) modes.
  Enables arm-wrist to hand collision detection while keeping hand self-collision
  disabled (291 inter-finger Never rules retained).

Performance (post-optimisation, measured on i9-13900K):

=============  ==========  ============  ===========================================
Operation       7-DOF       19-DOF        Notes
=============  ==========  ============  ===========================================
self-collision    ~30 μs     ~35 μs      ``pin.computeCollisions(stop_at_first)``
env (safe)       ~17 μs      ~17 μs      Tier 1 Z-min filter, zero FCL calls
env (near)          —       2–8 ms       Tier 2 FCL mesh-mesh (hand near table)
segment (Δ=0.5)     —      ~870 μs      step_size=0.02 rad, 25 samples × ~35 μs
=============  ==========  ============  ===========================================

Usage::

    from dexmani_real.planning.collision_model import CollisionModel

    cm = CollisionModel()
    cm.check_self_collision(qpos)           # bool
    cm.check_self_collision_details(qpos)   # CollisionInfo (from types.py)
    cm.add_box_obstacle("table", [1.0,2.0,0.04], [0.5, 0.0, -0.04])
    cm.check_env_collision(qpos)            # bool — full two-tier (path planning)
    cm.check_env_collision_fast(qpos)       # bool — Tier 1 only (teleop hot path)
    cm.check_teleop_collision(qpos)         # (has_self, has_env) — single-FK (teleop)
    cm.remove_obstacle("table")             # remove a single obstacle
    cm.clear_obstacles()                    # remove all obstacles
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real import ASSET_DIR
from dexmani_real.utils.log import get_logger

if TYPE_CHECKING:
    from .collision_config import CollisionConfig
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

# User→URDF reorder map for set_hand_qpos().
# Hardware and simulation return hand qpos in user order:
#   [thumb_bend, thumb_rota1, thumb_rota2, index_bend, index_j1, index_j2,
#    mid_j1, mid_j2, ring_j1, ring_j2, pinky_j1, pinky_j2]
# But the URDF (and thus Pinocchio) expects:
#   [index_bend, index_j1, index_j2, mid_j1, mid_j2,
#    pinky_j1, pinky_j2, ring_j1, ring_j2,
#    thumb_bend, thumb_rota1, thumb_rota2]
# _hand_user_to_urdf[i] = user index for URDF slot i.
_HAND_USER_TO_URDF: tuple[int, ...] = (3, 4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2)

# ── Box AABB tuple indices ──
# ``_obstacle_boxes[obj_id]`` stores ``(x_min, x_max, y_min, y_max, z_min, z_max)``.
# These constants name each slot for readable indexing.
_BB_XMIN, _BB_XMAX, _BB_YMIN, _BB_YMAX, _BB_ZMIN, _BB_ZMAX = range(6)

# ── Env collision tier margins ──
# Tier 1: Z-min pre-filter — skip all FCL when lowest robot geometry is safely
# above the highest obstacle.  0.02 m accounts for fingertip mesh extent (~4 cm
# below fingertip centre).
_Z_TIER1_MARGIN = 0.05  # [m] (4 cm fingertip mesh half-extent + 1 cm safety)
# Tier 2: per-geometry Z skip — ignore robot geometries whose FK centre is more
# than 0.25 m above the obstacle (arm base, shoulder, upper arm).  Only hand,
# wrist, and forearm geometries (~15–25 of 39) pass through to FCL.
_Z_TIER2_MARGIN = 0.25  # [m]


class CollisionModel:
    """Standalone Pinocchio collision model for xArm7 + XHand.

    Two modes, sharing the **same unified SRDF**:

    - **7-DOF** (default, ``hand_dof=False``): arm-only collision model from the
      collision URDF (hand joints merged as fixed).  Accepts ``qpos`` of shape
      ``(7,)``.  Self-collision ~30μs, env ~17μs (Tier 1).

    - **19-DOF** (``hand_dof=True``): full arm + hand collision model from the
      full URDF (hand joints active).  Accepts ``qpos`` of shape ``(19,)`` —
      first 7 are arm joints, last 12 are hand joints.  Self-collision ~35μs,
      env Tier 1 ~17μs, Tier 2 2–8ms (hand near table).

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
        collision_config: "CollisionConfig | None" = None,
    ) -> None:
        import coal as fcl
        import pinocchio as pin

        self._pin = pin
        self._fcl = fcl
        self._hand_dof = hand_dof

        # Tier margins from CollisionConfig (A7), with module-constant fallback
        if collision_config is not None:
            self._z_tier1_margin = collision_config.tier1_z_margin
            self._z_tier2_margin = collision_config.tier2_z_margin
        else:
            self._z_tier1_margin = _Z_TIER1_MARGIN
            self._z_tier2_margin = _Z_TIER2_MARGIN

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

        # Selectively re-enable high-risk cross-finger collision pairs (G1).
        # SRDF Never rules disable ALL hand self-collision (483 rules) to avoid
        # the 66 finger-to-finger pair explosion.  We explicitly re-enable only
        # the thumb_tip ↔ index_tip pair — the most common pinch-contact risk.
        # Other cross-finger pairs remain disabled (hardware torque limits +
        # low collision risk in normal operation).
        if hand_dof:
            tip_geom_map: dict[str, int] = {}
            for i in range(self._collision_model.ngeoms):
                tip_geom_map[self._collision_model.geometryObjects[i].name] = i
            thumb_key = "right_hand_thumb_rota_tip_0"
            index_key = "right_hand_index_rota_tip_0"
            if thumb_key in tip_geom_map and index_key in tip_geom_map:
                self._collision_model.addCollisionPair(
                    pin.CollisionPair(tip_geom_map[thumb_key], tip_geom_map[index_key])
                )

        self._collision_data = self._collision_model.createData()

        # --- Obstacle tracking ---
        self._obstacle_names: set[str] = set()
        self._robot_geom_ids: list[int] = []      # non-static robot geometry IDs (cached for env checks)
        self._obstacle_boxes: dict[str, tuple[float, float, float, float, float, float]] = {}
        # name → (x_min, x_max, y_min, y_max, z_min, z_max) in model base frame
        self._obs_z_max: float = float('-inf')    # cached max Z of all obstacles (P6)
        self._fcl_request: "fcl.CollisionRequest" = self._fcl.CollisionRequest()  # reusable (P7)

        self._nq: int = self._model.nq
        self._link_names: list[str] = (
            self._model.names.tolist() if hasattr(self._model.names, "tolist") else list(self._model.names)
        )

        # Cache non-static robot geometry IDs for env collision queries.
        self._robot_geom_ids = [
            i for i in range(self._collision_model.ngeoms)
            if self._collision_model.geometryObjects[i].parentJoint != 0
        ]

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
        self._hand_qpos_initialized: bool = False

    # ------------------------------------------------------------------
    # qpos handling
    # ------------------------------------------------------------------

    def set_hand_qpos(self, hand_qpos: np.ndarray) -> None:
        """Set the current hand joint configuration for auto-expansion.

        In ``hand_dof`` mode, collision check methods accept 7-DOF arm qpos
        and automatically concatenate with this buffer to form a full 19-DOF
        qpos.  Call this each frame before arm collision checks.

        Accepts hand qpos in **user order** (hardware / simulation native):
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
        self._hand_qpos_initialized = True

    def _to_full_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """Normalize qpos for internal use, auto-expanding arm→full in hand_dof mode.

        - 7-DOF model: always expects ``(7,)``.
        - 19-DOF model: accepts ``(7,)`` (auto-expands with ``_hand_qpos``) or
          ``(19,)`` (uses as-is).
        """
        qpos = np.asarray(qpos, dtype=np.float64)
        if self._hand_dof and qpos.shape == (7,):
            if not self._hand_qpos_initialized:
                logger.warning(
                    "hand_qpos not initialized — collision checks use zero (open-hand) pose, "
                    "which may not match actual hand configuration. "
                    "Call set_hand_qpos() before collision checks."
                )
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
        """FK + update geometry placements + compute self-collisions.

        Returns ``(full_qpos, has_any_collision)`` where ``has_any_collision`` is
        the return value of ``pin.computeCollisions()``.
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

    def _update_placements(self, qpos: np.ndarray) -> np.ndarray:
        """FK + update geometry placements only — no collision computation.

        Used by env collision checks (which use direct FCL calls, not
        ``computeCollisions``) and the Tier-1-only fast path.
        """
        qpos = self._to_full_qpos(qpos)
        self._pin.forwardKinematics(self._model, self._data, qpos)
        self._pin.updateGeometryPlacements(
            self._model, self._data, self._collision_model, self._collision_data,
        )
        return qpos

    # ------------------------------------------------------------------
    # Self-collision
    # ------------------------------------------------------------------

    def check_self_collision(self, qpos: np.ndarray) -> bool:
        """Check if qpos is in self-collision (fast bool, single point).

        Uses ``stop_at_first_collision=True`` for early exit.  Obstacle pairs
        are NOT registered in the main model (env checks use direct FCL), so
        every collision result is a genuine self-collision.
        """
        _qpos, has_any = self._pin_update(qpos, stop_at_first=True)
        return has_any

    def check_self_collision_details(self, qpos: np.ndarray) -> CollisionInfo:
        """Check self-collision and return structured ``CollisionInfo``.

        Only allocates on collision; returns cached ``CollisionInfo.no_collision()``
        on the common (no-collision) path.
        """
        from .types import CollisionInfo, CollisionPair

        _qpos, has_any = self._pin_update(qpos, stop_at_first=False)
        if not has_any:
            return CollisionInfo.no_collision()
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

    def _check_segment_free(
        self, q1: np.ndarray, q2: np.ndarray, step_size: float,
        check_fn,
    ) -> bool:
        """Generic dense segment interpolation with early exit on collision.

        Interpolates at ``step_size`` (L∞ rad) resolution and calls ``check_fn(q)``
        at each sample.  Returns True if all samples pass.
        """
        q1 = self._to_full_qpos(q1)
        q2 = self._to_full_qpos(q2)
        diff = q2 - q1
        dist = float(np.max(np.abs(diff)))
        if dist <= step_size:
            return not check_fn(q2)
        n = int(np.ceil(dist / step_size))
        for step in range(1, n + 1):
            if check_fn(q1 + (step / n) * diff):
                return False
        return True

    def check_segment_collision_free(
        self, q1: np.ndarray, q2: np.ndarray, step_size: float = 0.02,
    ) -> bool:
        """Check if the linear joint-space segment q1→q2 is self-collision-free."""
        return self._check_segment_free(q1, q2, step_size, self.check_self_collision)

    def check_segment_env_collision_free(
        self, q1: np.ndarray, q2: np.ndarray, step_size: float = 0.02,
    ) -> bool:
        """Check if the linear joint-space segment q1→q2 is env-collision-free.

        Returns True immediately when no obstacles are registered.
        """
        if not self._obstacle_names:
            return True
        return self._check_segment_free(q1, q2, step_size, self.check_env_collision)

    # ------------------------------------------------------------------
    # Environment collision (obstacles only)
    # ------------------------------------------------------------------

    def check_env_collision(self, qpos: np.ndarray) -> bool:
        """Full two-tier env collision check (path planning, return-to-home).

        Tier 1: Z-min pre-filter (~17 μs, zero FCL calls).
            If the lowest robot geometry is safely above all obstacles, return
            False immediately — the common case in teleop.

        Tier 2: Z-filtered FCL (2–8 ms in 19-DOF mode).
            Only triggers when a robot geometry is within 25 cm of an obstacle.
            Arm base / shoulder / upper arm are skipped by Z filter; only hand,
            wrist, and forearm geometries (~15–25 of 39) pass through to the
            expensive FCL mesh-mesh check.

        Obstacle pairs are NOT registered in the main model, so self-collision
        checks are unaffected.
        """
        if not self._obstacle_names or not self._robot_geom_ids:
            return False

        qpos = self._update_placements(qpos)
        oMg = self._collision_data.oMg

        # ── Tier 1: Z-min filter ──
        robot_z_min = min(oMg[rid].translation[2] for rid in self._robot_geom_ids)
        # Guard against NaN propagation from bad hand_qpos (e.g. uninitialized or NaN input)
        if not np.isfinite(robot_z_min):
            return True  # conservative: assume collision when FK is invalid
        if robot_z_min > self._obs_z_max + self._z_tier1_margin:
            return False

        # ── Tier 2: Z-filtered FCL ──
        result = self._fcl.CollisionResult()
        for name in self._obstacle_names:
            bb = self._obstacle_boxes.get(name)
            if bb is None:
                continue
            obs_id = self._get_obstacle_geom_id(name)
            if obs_id is None:
                continue
            obs_geom = self._collision_model.geometryObjects[obs_id]
            obs_placement = oMg[obs_id]
            obs_z_max_i = bb[_BB_ZMAX]
            for robot_id in self._robot_geom_ids:
                if oMg[robot_id].translation[2] > obs_z_max_i + self._z_tier2_margin:
                    continue
                result.clear()
                try:
                    self._fcl.collide(
                        obs_geom.geometry, obs_placement,
                        self._collision_model.geometryObjects[robot_id].geometry,
                        oMg[robot_id],
                        self._fcl_request, result,
                    )
                except Exception:
                    logger.warning("FCL collide() failed — treating as collision (conservative)", exc_info=True)
                    return True  # conservative: assume collision on FCL error
                if result.isCollision():
                    return True
        return False

    def check_env_collision_fast(self, qpos: np.ndarray) -> bool:
        """Tier-1-only env collision check for the teleop hot path (~17 μs, zero FCL).

        Conservative: returns True when Z-min cannot rule out a collision
        (i.e. some robot geometry is within ``_Z_TIER1_MARGIN`` of an obstacle).
        The full ``check_env_collision()`` with Tier 2 FCL is reserved for path
        planning and return-to-home, where the 2–8 ms penalty is acceptable.

        In practice, teleop operators keep the hand visibly above the table, so
        Tier 1 almost always passes and this returns False.
        """
        if not self._obstacle_names or not self._robot_geom_ids:
            return False
        qpos = self._update_placements(qpos)
        oMg = self._collision_data.oMg
        robot_z_min = min(oMg[rid].translation[2] for rid in self._robot_geom_ids)
        # Guard against NaN propagation from bad hand_qpos
        if not np.isfinite(robot_z_min):
            return True  # conservative: assume collision when FK is invalid
        return robot_z_min <= self._obs_z_max + self._z_tier1_margin

    def check_teleop_collision(self, qpos: np.ndarray) -> tuple[bool, bool]:
        """Single-FK self + env Tier-1 collision check for teleop hot path (~35 μs).

        Replaces two separate FK calls (``check_self_collision`` + ``check_env_collision_fast``,
        ~52 μs total) with a single FK+placements pass.  Self-collision uses
        ``computeCollisions(stop_at_first=True)``; env collision uses the Tier-1
        Z-min pre-filter on the same FK result (zero extra cost).

        Returns ``(has_self_collision, has_env_collision)``.
        """
        qpos_full = self._to_full_qpos(qpos)
        self._pin.forwardKinematics(self._model, self._data, qpos_full)
        self._pin.updateGeometryPlacements(
            self._model, self._data, self._collision_model, self._collision_data,
        )

        # Self-collision: full FCL with early exit on first contact
        has_self = self._pin.computeCollisions(
            self._model, self._data, self._collision_model, self._collision_data,
            qpos_full, True,
        )

        # Env collision: Tier-1 Z-min on the SAME placements (zero extra FK)
        has_env = False
        if self._obstacle_names and self._robot_geom_ids:
            oMg = self._collision_data.oMg
            robot_z_min = min(oMg[rid].translation[2] for rid in self._robot_geom_ids)
            if not np.isfinite(robot_z_min):
                has_env = True  # conservative: assume collision when FK is invalid
            else:
                has_env = robot_z_min <= self._obs_z_max + self._z_tier1_margin

        return has_self, has_env

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
        self._collision_model.addGeometryObject(obj)
        # Do NOT add collision pairs to the main model — env collision uses
        # direct FCL collide() against _robot_geom_ids, keeping the main
        # model's computeCollisions() self-collision-only (~20% faster).
        self._obstacle_names.add(name)

        # Cache world AABB for fast pre-filter in check_env_collision().
        # For identity rotation (the common case), the box axes are aligned
        # with world axes.  For rotated obstacles, use a conservative
        # bounding sphere to avoid recomputing the exact OBB.
        hx, hy, hz = half_extents
        cx, cy, cz = position
        if rotation is None or np.allclose(rot, np.eye(3)):
            self._obstacle_boxes[name] = (
                cx - hx, cx + hx, cy - hy, cy + hy, cz - hz, cz + hz,
            )
        else:
            # Conservative: sphere radius = box half-diagonal
            r = float(np.sqrt(hx * hx + hy * hy + hz * hz))
            self._obstacle_boxes[name] = (
                cx - r, cx + r, cy - r, cy + r, cz - r, cz + r,
            )

        # Update cached obstacle Z-max (P6)
        self._obs_z_max = max(box[_BB_ZMAX] for box in self._obstacle_boxes.values()) if self._obstacle_boxes else float('-inf')
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

    def remove_obstacle(self, name: str) -> bool:
        """Remove a box obstacle by name.

        Returns True if the obstacle was found and removed, False otherwise.
        After removal, the collision data is refreshed and the Z-max cache
        is updated.
        """
        if name not in self._obstacle_names:
            return False
        self._collision_model.removeGeometryObject(name)
        self._obstacle_names.discard(name)
        self._obstacle_boxes.pop(name, None)
        # Update cached Z-max
        self._obs_z_max = max(
            (box[_BB_ZMAX] for box in self._obstacle_boxes.values()),
            default=float('-inf'),
        )
        self._collision_data = self._collision_model.createData()
        logger.info("Removed obstacle '%s'", name)
        return True

    def clear_obstacles(self) -> int:
        """Remove all box obstacles.  Returns the number of obstacles cleared."""
        count = len(self._obstacle_names)
        for name in list(self._obstacle_names):
            self._collision_model.removeGeometryObject(name)
        self._obstacle_names.clear()
        self._obstacle_boxes.clear()
        self._obs_z_max = float('-inf')
        self._collision_data = self._collision_model.createData()
        logger.info("Cleared %d obstacle(s)", count)
        return count

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obstacle_geom_id(self, name: str) -> int | None:
        """Find the geometry object index for an obstacle by name.

        Returns None if the obstacle is not found in the geometry model.
        """
        for i, obj in enumerate(self._collision_model.geometryObjects):
            if obj.name == name:
                return i
        return None

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

    def pad_arm_for_fk(self, qpos_arm: np.ndarray) -> np.ndarray:
        """Pad 7-DOF arm qpos to full model dimension for FK queries.

        Used by external FK consumers (e.g. FingertipDeskSafety) that need
        a full qpos for the Pinocchio model.  In 7-DOF mode the model already
        has hand joints fixed at home, so arm qpos is returned as-is.  In
        19-DOF mode the hand DOFs are taken from the ``_hand_qpos`` buffer
        (set via ``set_hand_qpos()``), falling back to zeros when the buffer
        has not been initialized.

        Args:
            qpos_arm: 7-DOF arm joint angles [rad].

        Returns:
            Full qpos of shape ``(self._nq,)`` suitable for FK.
        """
        qpos = np.asarray(qpos_arm, dtype=np.float64)
        if qpos.shape != (7,):
            raise ValueError(f"Expected arm qpos shape (7,), got {qpos.shape}")
        if not self._hand_dof:
            return qpos
        hand = self._hand_qpos if self._hand_qpos_initialized else np.zeros(_HAND_DOF_COUNT, dtype=np.float64)
        return np.concatenate([qpos, hand])

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

    @property
    def pinocchio_model(self):
        """The underlying Pinocchio model (read-only)."""
        return self._model

    @property
    def pinocchio_data(self):
        """The underlying Pinocchio data (read-only)."""
        return self._data
