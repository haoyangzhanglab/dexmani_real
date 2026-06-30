"""IK candidate management — generation, filtering, scoring, canonicalization."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .kinematics import XArm7Kinematics
    from .collision_model import CollisionModel

from .types import CollisionInfo, PlanningProfile, Pose
from .pose_utils import ensure_qpos


# Map freeform reject reason strings → structured diagnostic categories.
# Used in collect_ik_candidates() to produce reject_by_category summary.
# Ref: ssik explain=True pattern — distinguishes unreachable vs all-filtered.
_REJECT_CATEGORY_MAP: dict[str, str] = {
    "mplib_ik_failed": "unreachable",
    "IK candidate outside planning limits.": "limits",
    "IK candidate exceeds max_ik_delta_deg.": "delta",
    "IK candidate pose error exceeds threshold.": "pose_error",
    "IK candidate in self-collision.": "collision",
}


def _categorize_rejects(reject_counts: dict[str, int]) -> dict[str, int]:
    """Aggregate freeform reject reason counts into diagnostic categories."""
    categorized: dict[str, int] = {}
    for reason, count in reject_counts.items():
        category = _REJECT_CATEGORY_MAP.get(reason, "other")
        categorized[category] = categorized.get(category, 0) + count
    return categorized


class IKCandidateManager:
    """IK candidate generation, filtering, scoring, and joint canonicalization.

    References:
      - LeFranX weighted_ik.cpp
      - dimos collision_step_size
    """

    def __init__(self, kinematics: XArm7Kinematics, collision_model: CollisionModel | None = None) -> None:
        self.kin = kinematics
        self.dof = kinematics.dof
        self.joint_limits = kinematics.joint_limits
        self.equivalent_joint_mask = kinematics.equivalent_joint_mask
        self.mp_planner = kinematics.mp_planner
        self._cm = collision_model


    def call_mplib_ik(
        self, target_pose_base: Pose, seed_qpos: np.ndarray, n_init_qpos: int, return_closest: bool
    ) -> tuple[str, Any]:
        return self.mp_planner.IK(
            goal_pose=self.kin.to_mplib_pose(target_pose_base),
            start_qpos=seed_qpos,
            mask=None,
            n_init_qpos=n_init_qpos,
            threshold=1e-3,
            return_closest=return_closest,
        )


    def generate_ik_seeds(self, current_qpos: np.ndarray, profile: PlanningProfile) -> list[np.ndarray]:
        limits = self.resolve_planning_limits(profile, current_qpos)
        current = self.canonicalize_qpos(current_qpos, current_qpos, limits)
        seeds: list[np.ndarray] = [current.copy()]

        rng = np.random.default_rng(profile.random_seed)
        offsets_rad = np.deg2rad(self.profile_array(profile.ik_seed_offsets_deg, "ik_seed_offsets_deg"))
        for _ in range(profile.num_random_ik_seeds):
            seed = current + rng.uniform(-offsets_rad, offsets_rad)
            seed = self.canonicalize_qpos(seed, current, limits)
            seeds.append(seed)
        return self.unique_qpos_list(seeds)


    def collect_ik_candidates(
        self,
        target_eef_pose_world: Pose,
        current_qpos: np.ndarray,
        profile: PlanningProfile,
    ) -> tuple[list[tuple[np.ndarray, dict[str, Any]]], dict[str, Any]]:
        target_pose_base = self.kin.world_to_base_pose(target_eef_pose_world)
        seeds = self.generate_ik_seeds(current_qpos, profile)
        planning_limits = self.resolve_planning_limits(profile, current_qpos)
        candidates: list[tuple[np.ndarray, dict[str, Any]]] = []
        reject_counts: dict[str, int] = {}
        reject_examples: list[dict[str, Any]] = []
        raw_success_count = 0

        for seed_index, seed in enumerate(seeds):
            status, raw_qpos = self.call_mplib_ik(
                target_pose_base, seed, n_init_qpos=profile.n_init_qpos, return_closest=True
            )
            if not status.lower().startswith("success") or raw_qpos is None:
                reason = "mplib_ik_failed"
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
                continue

            raw_success_count += 1
            raw_qpos = np.asarray(raw_qpos, dtype=np.float64)
            qpos = self.canonicalize_qpos(raw_qpos, current_qpos, planning_limits)
            valid, report = self.filter_ik_candidate(
                qpos, raw_qpos, target_eef_pose_world, current_qpos, profile, planning_limits
            )
            if valid:
                report["ik_score"] = self.score_ik_candidate(qpos, current_qpos, report, profile)
                report["seed_index"] = seed_index
                candidates.append((qpos.copy(), report))
                continue

            reason = str(report.get("reason", "rejected"))
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
            if len(reject_examples) < 5:
                reject_examples.append(self.compact_reject_report(seed_index, report))

        candidates.sort(key=lambda item: item[1]["ik_score"])
        reject_by_category = _categorize_rejects(reject_counts)
        summary: dict[str, Any] = {
            "num_seeds": len(seeds),
            "raw_ik_success_count": raw_success_count,
            "valid_candidate_count": len(candidates),
            "returned_candidate_count": min(len(candidates), profile.num_ik_candidates),
            "reject_counts": reject_counts,
            "reject_by_category": reject_by_category,
            "random_seed": profile.random_seed,
        }
        if reject_examples:
            summary["reject_examples"] = reject_examples
        return candidates[: profile.num_ik_candidates], summary


    def filter_ik_candidate(
        self,
        qpos: np.ndarray,
        raw_qpos: np.ndarray,
        target_eef_pose_world: Pose,
        current_qpos: np.ndarray,
        profile: PlanningProfile,
        limits: np.ndarray,
    ) -> tuple[bool, dict[str, Any]]:
        low, high = limits[:, 0], limits[:, 1]
        report: dict[str, Any] = {"raw_qpos": raw_qpos.copy()}

        outside, violation = self.limit_violation(qpos, limits)
        if np.any(outside):
            indices = np.where(outside)[0]
            report.update(
                reason="IK candidate outside planning limits.",
                outside_joint_indices_1based=(indices + 1).tolist(),
                max_limit_violation_deg=float(np.rad2deg(np.max(violation[indices]))),
                qpos_deg=np.rad2deg(qpos.copy()),
                low_deg=np.rad2deg(low.copy()),
                high_deg=np.rad2deg(high.copy()),
            )
            return False, report

        np.clip(qpos, low, high, out=qpos)
        delta = self.compute_qpos_delta(qpos, current_qpos)
        max_delta = np.deg2rad(self.profile_array(profile.max_ik_delta_deg, "max_ik_delta_deg"))
        over_delta = np.abs(delta) > max_delta
        if np.any(over_delta):
            indices = np.where(over_delta)[0]
            violation_delta = np.maximum(np.abs(delta) - max_delta, 0.0)
            report.update(
                reason="IK candidate exceeds max_ik_delta_deg.",
                max_delta_joint_indices_1based=(indices + 1).tolist(),
                max_delta_violation_deg=float(np.rad2deg(np.max(violation_delta[indices]))),
            )
            return False, report

        pose_error_pos, pose_error_rot = self.kin.compute_world_pose_error(target_eef_pose_world, qpos)
        report.update(
            pose_error_pos_m=pose_error_pos,
            pose_error_rot_rad=pose_error_rot,
            qpos_distance=float(np.linalg.norm(delta)),
            max_qpos_delta=float(np.max(np.abs(delta))),
            max_qpos_delta_deg=float(np.rad2deg(np.max(np.abs(delta)))),
            joint_limit_penalty=self.joint_limit_penalty(qpos, limits),
        )

        if pose_error_pos > profile.max_pose_error_pos_m or pose_error_rot > profile.max_pose_error_rot_rad:
            report["reason"] = "IK candidate pose error exceeds threshold."
            return False, report

        if profile.check_self_collision:
            collision_info = self.check_self_collision(qpos)
            if collision_info:
                report["reason"] = "IK candidate in self-collision."
                report["collision"] = collision_info.to_dict()
                return False, report

        return True, report


    def score_ik_candidate(
        self, qpos: np.ndarray, current_qpos: np.ndarray, report: dict[str, Any], profile: PlanningProfile
    ) -> float:
        delta = self.compute_qpos_delta(qpos, current_qpos)
        joint_cost = float(np.linalg.norm(delta) + 0.5 * np.max(np.abs(delta)))
        pose_cost = float(report.get("pose_error_pos_m", 0.0) + report.get("pose_error_rot_rad", 0.0))
        limit_cost = float(report.get("joint_limit_penalty", 0.0))

        score = profile.ik_score_joint_delta_weight * joint_cost
        score += profile.ik_score_pose_error_weight * pose_cost
        score += profile.ik_score_joint_limit_weight * limit_cost

        manipulability = self.kin.compute_manipulability(qpos)
        score -= profile.ik_score_manipulability_weight * manipulability
        report["manipulability"] = float(manipulability)

        if profile.neutral_qpos is not None:
            neutral_dist = self.normalized_joint_distance(qpos, profile.neutral_qpos)
            score += profile.ik_score_neutral_weight * neutral_dist
            report["neutral_distance"] = float(neutral_dist)

        return float(score)


    def resolve_planning_limits(
        self, profile: PlanningProfile, reference_qpos: np.ndarray | None = None
    ) -> np.ndarray:
        if profile.planning_limits_deg is not None:
            limits = np.deg2rad(np.asarray(profile.planning_limits_deg, dtype=np.float64))
            if limits.shape != self.joint_limits.shape:
                raise ValueError(f"planning_limits_deg must have shape {self.joint_limits.shape}, got {limits.shape}.")
            return limits

        limits = self.joint_limits.copy()
        if reference_qpos is None:
            reference_qpos = np.zeros(self.dof, dtype=np.float64)
        reference_qpos = ensure_qpos(reference_qpos, self.dof, "reference_qpos")

        for joint_index in range(self.dof):
            if not self.equivalent_joint_mask[joint_index]:
                continue
            hardware_low = self.joint_limits[joint_index, 0]
            hardware_high = self.joint_limits[joint_index, 1]
            local_low = reference_qpos[joint_index] - np.pi
            local_high = reference_qpos[joint_index] + np.pi
            limits[joint_index, 0] = max(hardware_low, local_low)
            limits[joint_index, 1] = min(hardware_high, local_high)
        return limits

    def _periods(self) -> np.ndarray:
        joint_ranges = self.joint_limits[:, 1] - self.joint_limits[:, 0]
        return np.where(self.equivalent_joint_mask, np.minimum(2.0 * np.pi, joint_ranges), 1.0)

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
        result = qpos.copy()
        mask = self.equivalent_joint_mask
        periods = self._periods()
        low, high = limits[:, 0], limits[:, 1]

        k_min = np.ceil((low[mask] - result[mask] - limit_tol) / periods[mask])
        k_max = np.floor((high[mask] - result[mask] + limit_tol) / periods[mask])
        k = np.round((reference_qpos[mask] - result[mask]) / periods[mask])
        valid = k_min <= k_max
        k = np.where(valid, np.clip(k, k_min, k_max), 0.0)
        result[mask] += k * periods[mask]
        np.clip(result, low, high, out=result)
        return result

    def canonicalize_path_to_planning_limits(
        self, path: np.ndarray, current_qpos: np.ndarray, profile: PlanningProfile
    ) -> np.ndarray:
        path = np.asarray(path, dtype=np.float64).copy()
        if path.ndim != 2 or path.shape[1] != self.dof:
            raise ValueError(f"path must have shape (N, {self.dof}), got {path.shape}.")
        limits = self.resolve_planning_limits(profile, current_qpos)
        path[0] = self.canonicalize_qpos(path[0], current_qpos, limits)
        for index in range(1, len(path)):
            path[index] = self.canonicalize_qpos(path[index], path[index - 1], limits)
        return path

    def snap_path_to_nearest_equivalent(self, path: np.ndarray, reference_qpos: np.ndarray) -> np.ndarray:
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
        delta[self.equivalent_joint_mask] = (delta[self.equivalent_joint_mask] + half) % periods[self.equivalent_joint_mask] - half
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


    # ── Collision check wrappers (delegate to CollisionModel) ──
    # MPlib fallback branches removed — CollisionModel is always available
    # when constructed through XArm7MotionPlanner (the only construction path).

    def has_self_collision(self, qpos: np.ndarray) -> bool:
        return self._cm.check_self_collision(qpos)

    def check_self_collision(self, qpos: np.ndarray) -> CollisionInfo:
        return self._cm.check_self_collision_details(qpos)

    def has_env_collision(self, qpos: np.ndarray) -> bool:
        return self._cm.check_env_collision(qpos)

    def has_env_collision_fast(self, qpos: np.ndarray) -> bool:
        return self._cm.check_env_collision_fast(qpos)

    def check_teleop_collision(self, qpos: np.ndarray) -> tuple[bool, bool]:
        """Single-FK self + env Tier-1 collision check for teleop hot path.

        Returns ``(has_self_collision, has_env_collision)``.
        """
        return self._cm.check_teleop_collision(qpos)

    def _check_segment_collision(
        self, start: np.ndarray, end: np.ndarray, collision_type: str = "self", step_size: float = 0.02,
    ) -> bool:
        """Check if the linear joint-space segment start→end is collision-free."""
        if collision_type == "self":
            return self._cm.check_segment_collision_free(start, end, step_size)
        else:
            return self._cm.check_segment_env_collision_free(start, end, step_size)

    def check_segment_collision_free(
        self, start: np.ndarray, end: np.ndarray, step_size: float = 0.02,
    ) -> bool:
        """Check if the linear joint-space segment start→end is self-collision-free."""
        return self._check_segment_collision(start, end, "self", step_size)

    def check_segment_env_collision_free(
        self, start: np.ndarray, end: np.ndarray, step_size: float = 0.02,
    ) -> bool:
        """Check if the linear joint-space segment start→end is env-collision-free."""
        return self._check_segment_collision(start, end, "env", step_size)

    def check_path_collisions(
        self, path: np.ndarray, collision_step_size: float = 0.02,
    ) -> dict[str, Any]:
        """Check self-collision along path with dense interpolation (ref: dimos).

        Linearly interpolates between consecutive waypoints at the given step
        size and checks self-collision at every sampled point.  When a
        collision is found, includes structured ``CollisionInfo`` at the
        violating configuration for root-cause diagnostics.
        """
        for i in range(len(path) - 1):
            # Dense segment check — fast bool path for most points.
            if not self.check_segment_collision_free(
                path[i], path[i + 1], collision_step_size,
            ):
                # Pinpoint the exact violating configuration and get full details.
                collision_info = self._find_collision_in_segment(
                    path[i], path[i + 1], collision_step_size,
                )
                return {
                    "path_self_collision": True,
                    "collision_waypoint_index": i,
                    "collision_waypoint_count": len(path),
                    "collision_step_size": collision_step_size,
                    "collision": collision_info.to_dict() if collision_info else None,
                }
        return {"path_self_collision": False}

    def _find_collision_in_segment(
        self, start: np.ndarray, end: np.ndarray, step_size: float,
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
        for step in range(1, n_steps + 1):
            alpha = step / n_steps
            q = start + alpha * diff
            info = self.check_self_collision(q)
            if info:
                return info
        return None

    def check_path_env_collisions(
        self, path: np.ndarray, collision_step_size: float = 0.02,
    ) -> dict[str, Any]:
        """Check environment collision along path with dense interpolation."""
        for i in range(len(path) - 1):
            if not self._check_segment_collision(
                path[i], path[i + 1], "env", collision_step_size,
            ):
                return {
                    "path_env_collision": True,
                    "collision_waypoint_index": i,
                    "collision_waypoint_count": len(path),
                    "collision_step_size": collision_step_size,
                }
        return {"path_env_collision": False}


    def normalized_joint_distance(self, qpos: np.ndarray, reference_qpos: np.ndarray) -> float:
        """Per-joint-range normalized Euclidean distance (ref: LeFranX weighted_ik.cpp)."""
        qpos = ensure_qpos(qpos, self.dof, "qpos")
        reference_qpos = ensure_qpos(reference_qpos, self.dof, "reference_qpos")
        joint_ranges = self.joint_limits[:, 1] - self.joint_limits[:, 0]
        joint_ranges = np.maximum(joint_ranges, 1e-6)
        normalized_diff = (qpos - reference_qpos) / joint_ranges
        return float(np.sqrt(np.sum(normalized_diff ** 2)))

    def weighted_joint_distance(
        self, qpos: np.ndarray, reference_qpos: np.ndarray, weights: tuple[float, ...] | np.ndarray,
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
        """
        qpos = ensure_qpos(qpos, self.dof, "qpos")
        reference_qpos = ensure_qpos(reference_qpos, self.dof, "reference_qpos")
        delta = self.compute_qpos_delta(qpos, reference_qpos)
        joint_ranges = self.joint_limits[:, 1] - self.joint_limits[:, 0]
        joint_ranges = np.maximum(joint_ranges, 1e-6)
        weights_arr = self.profile_array(weights, "joint_weights")
        normalized = delta / joint_ranges
        return float(np.sqrt(np.sum(weights_arr * normalized ** 2)))

    def joint_limit_penalty(self, qpos: np.ndarray, limits: np.ndarray) -> float:
        center = 0.5 * (limits[:, 0] + limits[:, 1])
        half_range = np.maximum(0.5 * (limits[:, 1] - limits[:, 0]), 1e-6)
        return float(np.sum(((qpos - center) / half_range) ** 2))


    def profile_array(self, values: tuple[float, ...], name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.shape == (1,):
            return np.repeat(array, self.dof)
        if array.shape != (self.dof,):
            raise ValueError(f"{name} must have length 1 or {self.dof}, got {array.shape[0]}.")
        return array

    def unique_qpos_list(self, qpos_list: list[np.ndarray], atol: float = 1e-8) -> list[np.ndarray]:
        unique: list[np.ndarray] = []
        for qpos in qpos_list:
            if not any(np.allclose(qpos, item, atol=atol, rtol=0.0) for item in unique):
                unique.append(qpos)
        return unique

    def compact_reject_report(self, seed_index: int, report: dict[str, Any]) -> dict[str, Any]:
        compact: dict[str, Any] = {"seed_index": seed_index, "reason": report.get("reason")}
        keys = (
            "outside_joint_indices_1based",
            "max_limit_violation_deg",
            "max_delta_joint_indices_1based",
            "max_delta_violation_deg",
            "pose_error_pos_m",
            "pose_error_rot_rad",
            "max_qpos_delta_deg",
        )
        for key in keys:
            if key in report:
                compact[key] = report[key]
        return compact
