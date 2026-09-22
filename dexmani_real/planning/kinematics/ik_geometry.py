"""Joint canonicalization, IK geometry and collision queries used by online IK/home."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..collision import CollisionModel
    from ..planner import MotionPlanningConfig
    from .arm_fk import XArm7Kinematics

from ..collision import CollisionInfo
from ..paths import wrap_nearest_equivalent
from .pose import ensure_qpos


class IKGeometry:
    """Joint canonicalization and collision queries shared by online IK and home.

    References:
      - LeFranX weighted_ik.cpp
      - dimos collision_step_size
    """

    def __init__(
        self, kinematics: XArm7Kinematics, collision_model: CollisionModel | None = None
    ) -> None:
        self.kin = kinematics
        self.dof = kinematics.dof
        self.joint_limits = kinematics.joint_limits
        self.equivalent_joint_mask = kinematics.equivalent_joint_mask
        self.mp_planner = kinematics.mp_planner
        self._cm = collision_model
        # Equivalent-joint masks already guarantee a range of at least 2π.
        self._periods_arr = np.where(self.equivalent_joint_mask, 2.0 * np.pi, 1.0)

    def resolve_planning_limits(
        self, profile: MotionPlanningConfig, reference_qpos: np.ndarray | None = None
    ) -> np.ndarray:
        if profile.planning_limits_deg is not None:
            limits = np.deg2rad(np.asarray(profile.planning_limits_deg, dtype=np.float64))
            if limits.shape != self.joint_limits.shape:
                raise ValueError(
                    f"planning_limits_deg must have shape {self.joint_limits.shape}, got {limits.shape}."
                )
            return limits

        limits = self.joint_limits.copy()
        if reference_qpos is None:
            reference_qpos = np.zeros(self.dof, dtype=np.float64)
        reference_qpos = ensure_qpos(reference_qpos, self.dof, "reference_qpos")

        # Allow equivalent solutions near hardware limits by expanding the ±π window.
        mask = self.equivalent_joint_mask
        if np.any(mask):
            limits[mask, 0] = np.maximum(
                self.joint_limits[mask, 0], reference_qpos[mask] - 3.0 * np.pi
            )
            limits[mask, 1] = np.minimum(
                self.joint_limits[mask, 1], reference_qpos[mask] + 3.0 * np.pi
            )
        return limits

    def _periods(self) -> np.ndarray:
        return self._periods_arr

    def nearest_equivalent_qpos(self, qpos: np.ndarray, reference_qpos: np.ndarray) -> np.ndarray:
        return self.canonicalize_qpos(qpos, reference_qpos, limits=self.joint_limits, limit_tol=0.0)

    def canonicalize_qpos(
        self,
        qpos: np.ndarray,
        reference_qpos: np.ndarray,
        limits: np.ndarray | None = None,
        limit_tol: float = 1e-5,
    ) -> np.ndarray:
        qpos = ensure_qpos(qpos, self.dof, "qpos")
        reference_qpos = ensure_qpos(reference_qpos, self.dof, "reference_qpos")
        if limits is None:
            limits = self.joint_limits
        return wrap_nearest_equivalent(
            qpos,
            reference_qpos,
            tuple(np.asarray(limits)[:, 0]),
            tuple(np.asarray(limits)[:, 1]),
        )

    def canonicalize_path_to_planning_limits(
        self, path: np.ndarray, current_qpos: np.ndarray, profile: MotionPlanningConfig
    ) -> np.ndarray:
        path = np.asarray(path, dtype=np.float64).copy()
        if path.ndim != 2 or path.shape[1] != self.dof:
            raise ValueError(f"path must have shape (N, {self.dof}), got {path.shape}.")
        limits = self.resolve_planning_limits(profile, current_qpos)
        path[0] = self.canonicalize_qpos(path[0], current_qpos, limits)
        for index in range(1, len(path)):
            path[index] = self.canonicalize_qpos(path[index], path[index - 1], limits)
        return path

    def snap_path_to_nearest_equivalent(
        self, path: np.ndarray, reference_qpos: np.ndarray
    ) -> np.ndarray:
        path = np.asarray(path, dtype=np.float64).copy()
        if path.ndim != 2 or path.shape[1] != self.dof:
            raise ValueError(f"path must have shape (N, {self.dof}), got {path.shape}.")
        if len(path) == 0:
            return path
        path[0] = self.nearest_equivalent_qpos(path[0], reference_qpos)
        for index in range(1, len(path)):
            path[index] = self.nearest_equivalent_qpos(path[index], path[index - 1])
        return path

    def compute_qpos_delta(self, qpos: np.ndarray, reference_qpos: np.ndarray) -> np.ndarray:
        qpos = ensure_qpos(qpos, self.dof, "qpos")
        reference_qpos = ensure_qpos(reference_qpos, self.dof, "reference_qpos")
        delta = qpos - reference_qpos
        periods = self._periods()
        half = periods[self.equivalent_joint_mask] / 2.0
        delta[self.equivalent_joint_mask] = (delta[self.equivalent_joint_mask] + half) % periods[
            self.equivalent_joint_mask
        ] - half
        return delta

    def limit_violation(
        self, qpos: np.ndarray, limits: np.ndarray, limit_tol: float = 1e-5
    ) -> tuple[np.ndarray, np.ndarray]:
        below = qpos < limits[:, 0] - limit_tol
        above = qpos > limits[:, 1] + limit_tol
        outside = below | above
        lower = np.maximum(limits[:, 0] - qpos, 0.0)
        upper = np.maximum(qpos - limits[:, 1], 0.0)
        return outside, np.maximum(lower, upper)

    def path_limit_violation(
        self, path: np.ndarray, limits: np.ndarray, limit_tol: float = 1e-5
    ) -> tuple[np.ndarray, np.ndarray]:
        below = path < limits[None, :, 0] - limit_tol
        above = path > limits[None, :, 1] + limit_tol
        outside = below | above
        lower = np.maximum(limits[None, :, 0] - path, 0.0)
        upper = np.maximum(path - limits[None, :, 1], 0.0)
        return outside, np.maximum(lower, upper)

    def _require_collision_model(self) -> None:
        if self._cm is None:
            raise RuntimeError(
                "CollisionModel not configured — cannot check collisions. "
                "Pass collision_model=... to IKGeometry constructor."
            )

    def check_self_collision(self, qpos: np.ndarray) -> CollisionInfo:
        self._require_collision_model()
        return self._cm.check_self_collision_details(qpos)  # type: ignore[union-attr]

    def has_self_collision(self, qpos: np.ndarray) -> bool:
        """Robot endpoint self-collision only; excludes environment and table."""
        self._require_collision_model()
        return self._cm.check_self_collision(qpos)  # type: ignore[union-attr]

    def has_collision(self, qpos: np.ndarray) -> bool:
        self._require_collision_model()
        return self._cm.check_collision(qpos)  # type: ignore[union-attr]

    def check_collision(self, qpos: np.ndarray) -> CollisionInfo:
        self._require_collision_model()
        return self._cm.check_collision_details(qpos)  # type: ignore[union-attr]

    def check_path_collisions(
        self,
        path: np.ndarray,
        collision_step_size: float = 0.02,
    ) -> dict[str, Any]:
        """Check self-collision along path with dense interpolation (ref: dimos).

        Linearly interpolates between consecutive waypoints at the given step
        size and checks self-collision at every sampled point.  When a
        collision is found, includes structured ``CollisionInfo`` at the
        violating configuration for root-cause diagnostics.
        """
        self._require_collision_model()
        path = np.asarray(path, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != self.dof:
            raise ValueError(f"path must have shape (N, {self.dof}), got {path.shape}")
        if len(path) == 0:
            return {"path_self_collision": False}
        first_info = self.check_self_collision(path[0])
        if first_info:
            return {
                "path_self_collision": True,
                "collision_waypoint_index": 0,
                "collision_waypoint_count": len(path),
                "collision_step_size": collision_step_size,
                "collision": first_info.to_dict(),
            }
        for i in range(len(path) - 1):
            # Dense segment check — fast bool path for most points.
            if not self._cm.check_segment_collision_free(  # type: ignore[union-attr]  # requires a configured CollisionModel (planner always builds one)
                path[i],
                path[i + 1],
                collision_step_size,
            ):
                # Pinpoint the exact violating configuration and get full details.
                collision_info = self._find_collision_in_segment(
                    path[i],
                    path[i + 1],
                    collision_step_size,
                )
                return {
                    "path_self_collision": True,
                    "collision_waypoint_index": i,
                    "collision_waypoint_count": len(path),
                    "collision_step_size": collision_step_size,
                    "collision": collision_info.to_dict() if collision_info else None,
                }
        return {"path_self_collision": False}

    def check_path_combined_collisions(
        self,
        path: np.ndarray,
        collision_step_size: float = 0.02,
    ) -> dict[str, Any]:
        """Check a path densely against self and configured static geometry."""
        self._require_collision_model()
        path = np.asarray(path, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != self.dof:
            raise ValueError(f"path must have shape (N, {self.dof}), got {path.shape}")
        if len(path) == 0:
            return {"path_collision": False, "path_self_collision": False}
        first = self.check_collision(path[0])
        if first:
            return {
                "path_collision": True,
                "path_self_collision": any(
                    pair.collision_type != "environment" for pair in first.collision_pairs
                ),
                "collision_waypoint_index": 0,
                "collision_waypoint_count": len(path),
                "collision_step_size": collision_step_size,
                "collision": first.to_dict(),
            }
        for index in range(len(path) - 1):
            if self._cm.check_combined_segment_collision_free(  # type: ignore[union-attr]
                path[index], path[index + 1], collision_step_size
            ):
                continue
            info = self._find_combined_collision_in_segment(
                path[index], path[index + 1], collision_step_size
            )
            return {
                "path_collision": True,
                "path_self_collision": info is not None
                and any(pair.collision_type != "environment" for pair in info.collision_pairs),
                "collision_waypoint_index": index,
                "collision_waypoint_count": len(path),
                "collision_step_size": collision_step_size,
                "collision": info.to_dict() if info else None,
            }
        return {"path_collision": False, "path_self_collision": False}

    def _find_combined_collision_in_segment(
        self,
        start: np.ndarray,
        end: np.ndarray,
        step_size: float,
    ) -> CollisionInfo | None:
        diff = end - start
        steps = max(1, int(np.ceil(float(np.max(np.abs(diff))) / step_size)))
        for step in range(steps + 1):
            info = self.check_collision(start + (step / steps) * diff)
            if info:
                return info
        return None

    def _find_collision_in_segment(
        self,
        start: np.ndarray,
        end: np.ndarray,
        step_size: float,
    ) -> CollisionInfo | None:
        """Locate the first self-colliding configuration in segment [start, end].

        Returns structured ``CollisionInfo`` for the first collision found,
        or ``None`` if the segment is collision-free (unexpected caller path).
        """
        diff = end - start
        dist = float(np.max(np.abs(diff)))
        if dist <= step_size:
            info = self.check_self_collision(end)
            return info if info else None
        n_steps = int(np.ceil(dist / step_size))
        for step in range(n_steps + 1):
            alpha = step / n_steps
            q = start + alpha * diff
            info = self.check_self_collision(q)
            if info:
                return info
        return None

    def weighted_joint_distance(
        self,
        qpos: np.ndarray,
        reference_qpos: np.ndarray,
        weights: tuple[float, ...] | np.ndarray,
        delta: np.ndarray | None = None,
    ) -> float:
        """Per-joint weighted, range-normalised L2 distance (ref: LeFranX weighted_ik.cpp:62-69).

        Formula: ``sqrt(Σ wⱼ · (Δqⱼ / rangeⱼ)²)``

        - Δq is the wrapped equivalent-angle delta (handles ±2π ambiguity).
        - Each joint's delta is divided by its hardware range, so 1° on a
          joint with 238° range counts more than 1° on a joint with 720° range.
        - Per-joint weights ``wⱼ`` then scale the normalised squared error:
          higher weight → that joint is "expensive" to move away from current.

        This is the metric that LeFranX uses for ``current_distance`` in their
        multi-objective IK scoring function.  In DexMani it is used by the
        teleop position-IK fallback to rank candidates.

        Args:
            delta: Pre-computed wrapped delta from ``compute_qpos_delta``.
                   When provided, avoids a redundant ``compute_qpos_delta``
                   call (hot-path optimisation).
        """
        qpos = ensure_qpos(qpos, self.dof, "qpos")
        reference_qpos = ensure_qpos(reference_qpos, self.dof, "reference_qpos")
        if delta is None:
            delta = self.compute_qpos_delta(qpos, reference_qpos)
        joint_ranges = self.joint_limits[:, 1] - self.joint_limits[:, 0]
        joint_ranges = np.maximum(joint_ranges, 1e-6)
        weights_arr = self.profile_array(weights, "joint_weights")
        normalized = delta / joint_ranges
        return float(np.sqrt(np.sum(weights_arr * normalized**2)))

    def joint_limit_penalty(self, qpos: np.ndarray, limits: np.ndarray) -> float:
        center = 0.5 * (limits[:, 0] + limits[:, 1])
        half_range = np.maximum(0.5 * (limits[:, 1] - limits[:, 0]), 1e-6)
        return float(np.sum(((qpos - center) / half_range) ** 2))

    def profile_array(self, values: tuple[float, ...] | np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.shape == (1,):
            return np.repeat(array, self.dof)
        if array.shape != (self.dof,):
            raise ValueError(f"{name} must have length 1 or {self.dof}, got {array.shape[0]}.")
        return array
