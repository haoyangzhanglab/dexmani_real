"""Standalone Pinocchio collision model for xArm7 + XHand.

Builds a lightweight Pinocchio self-collision model from the collision URDF,
independent of MPlib. Uses T-Rex's ``pin.computeCollisions()`` pattern.

A single unified SRDF controls collision pair filtering:

- ``xarm7_xhand.srdf`` — unified SRDF used by both 7-DOF and 19-DOF modes.
  Enables arm-wrist to hand collision detection while keeping hand self-collision
  disabled (291 inter-finger Never rules retained).

Performance (post-optimisation, measured on i9-13900K):

=============  ==========  ============  ===========================================
Operation       7-DOF       19-DOF        Notes
=============  ==========  ============  ===========================================
self-collision    ~30 μs     ~35 μs      ``pin.computeCollisions(stop_at_first)``
segment (Δ=0.5)     —      ~870 μs      step_size=0.02 rad, 25 samples × ~35 μs
=============  ==========  ============  ===========================================

Usage::

    from dexmani_real.planning.collision_model import CollisionModel

    cm = CollisionModel()
    cm.check_self_collision(qpos)           # bool
    cm.check_self_collision_details(qpos)   # CollisionInfo (from types.py)
"""

from __future__ import annotations

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
_COLLISION_URDF = str(_XHAND_DIR / "xarm7_xhand_collision.urdf")  # 7-DOF (hand fixed)
_FULL_URDF = str(_XHAND_DIR / "xarm7_xhand_right.urdf")  # 19-DOF (7 arm + 12 hand)
_COLLISION_SRDF = str(_XHAND_DIR / "xarm7_xhand.srdf")  # unified SRDF (single source)

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


class CollisionModel:
    """Standalone Pinocchio self-collision model for xArm7 + XHand.

    Two modes, sharing the **same unified SRDF**:

    - **7-DOF** (default, ``hand_dof=False``): arm-only collision model from the
      collision URDF (hand joints merged as fixed).  Accepts ``qpos`` of shape
      ``(7,)``.  Self-collision ~30μs.

    - **19-DOF** (``hand_dof=True``): full arm + hand collision model from the
      full URDF (hand joints active).  Accepts ``qpos`` of shape ``(19,)`` —
      first 7 are arm joints, last 12 are hand joints.  Self-collision ~35μs.

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
    ) -> None:
        import pinocchio as pin

        self._pin = pin
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
        # 19-DOF: 40²/2 = 780 → after SRDF: ~254      (full URDF, no hand self-collision)
        import itertools

        n = self._collision_model.ngeoms
        for i, j in itertools.combinations(range(n), 2):
            self._collision_model.addCollisionPair(pin.CollisionPair(i, j))
        pin.removeCollisionPairs(self._model, self._collision_model, _srdf)

        # Hand self-collision is NOT checked — the SRDF Never rules disable
        # all 483 inter-finger pairs, and we intentionally do NOT re-enable any.
        # Arm↔hand collisions (e.g., wrist hitting fingers) remain active.
        self._collision_data = self._collision_model.createData()

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

        # Hand qpos buffer (used in hand_dof mode to auto-expand 7→19 DOF).
        # None = not yet set by caller; set_hand_qpos() assigns a real array.
        self._hand_qpos: np.ndarray | None = None

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

    def _to_full_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """Normalize qpos for internal use, auto-expanding arm→full in hand_dof mode.

        - 7-DOF model: always expects ``(7,)``.
        - 19-DOF model: accepts ``(7,)`` (auto-expands with ``_hand_qpos``) or
          ``(19,)`` (uses as-is).
        """
        qpos = np.asarray(qpos, dtype=np.float64)
        if self._hand_dof and qpos.shape == (7,):
            if self._hand_qpos is None:
                logger.warning(
                    "hand_qpos not initialized — collision checks use zero (open-hand) pose, "
                    "which may not match actual hand configuration. "
                    "Call set_hand_qpos() before collision checks."
                )
                return np.concatenate([qpos, np.zeros(_HAND_DOF_COUNT, dtype=np.float64)])
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
        self._pin.updateGeometryPlacements(self._model, self._data, self._collision_model, self._collision_data)
        has_any = self._pin.computeCollisions(
            self._model, self._data, self._collision_model, self._collision_data, qpos, stop_at_first
        )
        return qpos, has_any

    def _update_placements(self, qpos: np.ndarray) -> np.ndarray:
        """FK + update geometry placements only — no collision computation."""
        qpos = self._to_full_qpos(qpos)
        self._pin.forwardKinematics(self._model, self._data, qpos)
        self._pin.updateGeometryPlacements(
            self._model,
            self._data,
            self._collision_model,
            self._collision_data,
        )
        return qpos

    # ------------------------------------------------------------------
    # Self-collision
    # ------------------------------------------------------------------

    def check_self_collision(self, qpos: np.ndarray) -> bool:
        """Check if qpos is in self-collision (fast bool, single point).

        Uses ``stop_at_first_collision=True`` for early exit.
        """
        _qpos, has_any = self._pin_update(qpos, stop_at_first=True)
        return has_any

    def check_self_collision_details(self, qpos: np.ndarray) -> "CollisionInfo":
        """Check self-collision and return structured ``CollisionInfo``.

        Only allocates on collision; returns cached ``CollisionInfo.no_collision()``
        on the common (no-collision) path.
        """
        from .types import CollisionInfo, CollisionPair

        _qpos, has_any = self._pin_update(qpos, stop_at_first=False)
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
            return CollisionInfo(in_collision=True, collision_pairs=(), num_contacts=1)
        if not pairs:
            return CollisionInfo.no_collision()
        return CollisionInfo(in_collision=True, collision_pairs=tuple(pairs), num_contacts=len(pairs))

    # ------------------------------------------------------------------
    # Segment collision checking
    # ------------------------------------------------------------------

    def _check_segment_free(
        self,
        q1: np.ndarray,
        q2: np.ndarray,
        step_size: float,
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
        self,
        q1: np.ndarray,
        q2: np.ndarray,
        step_size: float = 0.02,
    ) -> bool:
        """Check if the linear joint-space segment q1→q2 is self-collision-free."""
        return self._check_segment_free(q1, q2, step_size, self.check_self_collision)

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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def hand_dof(self) -> bool:
        """Whether this model includes active hand joints (19-DOF vs 7-DOF)."""
        return self._hand_dof
