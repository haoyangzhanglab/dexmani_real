"""连续初值 CLIK 与有界零空间探索。"""

from dataclasses import dataclass

import numpy as np

from dexmani_real.control.constraints import (
    equivalent_joint_delta,
    joint_target_rejection,
    wrap_nearest_equivalent,
)
from dexmani_real.geometry.kinematics import kinematic_urdf
from dexmani_real.geometry.pose import Pose, compute_pose_error
from dexmani_real.robot.model import XARM7_JOINT_NAMES

JOINT_WEIGHTS = np.array((4.0, 1.8, 1.2, 0.6, 0.6, 0.9, 0.35))
PREVIOUS_WEIGHTS = np.array((10, 4, 2, 1, 0.5, 0.6, 0.1))
FAST_ACCEPT_RAD = np.deg2rad(8.0)
LIMIT_MARGIN_RAD = np.deg2rad(15.0)
NULLSPACE_STEP_RAD = np.deg2rad(5.0)
SVD_RANK_RTOL = 1e-4
NUMERIC_EPS = 1e-12
MODEL_LIMIT_TOL_RAD = 1e-5


@dataclass(kw_only=True)
class IKResult:
    qpos: np.ndarray | None
    reason: str = ""


@dataclass
class IKCandidate:
    qpos: np.ndarray
    position_error_m: float
    rotation_error_rad: float
    displacement_rad: float
    cost: float
    near_limit: bool
    rejection_reason: str | None


class IKSolver:
    def __init__(self, constraints, *, position_tolerance, rotation_tolerance):
        from mplib.pymp.kinematics.pinocchio import PinocchioModel

        self.constraints = constraints
        self.position_tolerance = position_tolerance
        self.rotation_tolerance = rotation_tolerance
        if any(not np.isfinite(v) or v <= 0 for v in (position_tolerance, rotation_tolerance)):
            raise ValueError("IK pose tolerances must be finite and positive")
        self.model = PinocchioModel.create_from_urdf_string(
            kinematic_urdf("link_base", stop_link="right_hand_link"),
        )
        self.model.set_joint_order(list(XARM7_JOINT_NAMES))
        self.model.set_link_order(self.model.get_link_names(user=False))
        self.eef_link_id = self.model.get_link_names().index("custom_eef_link")
        model_limits = np.concatenate(self.model.get_joint_limits())
        if (
            model_limits.shape != (7, 2)
            or not np.isfinite(model_limits).all()
            or np.any(model_limits[:, 0] >= model_limits[:, 1])
        ):
            raise ValueError("IK requires finite, ordered (7, 2) model limits")
        self.limits = constraints.limits
        if np.any(self.limits[:, 0] < model_limits[:, 0] - MODEL_LIMIT_TOL_RAD) or np.any(
            self.limits[:, 1] > model_limits[:, 1] + MODEL_LIMIT_TOL_RAD
        ):
            raise ValueError("operational limits exceed the loaded model limits")
        self.joint_ranges = model_limits[:, 1] - model_limits[:, 0]
        self.limit_center = self.limits.mean(axis=1)
        self.limit_radius = 0.5 * (self.limits[:, 1] - self.limits[:, 0])

    def compute_eef_pose_base(self, qpos):
        self.model.compute_forward_kinematics(qpos)
        pose = self.model.get_link_pose(self.eef_link_id)
        return Pose(p=np.asarray(pose.p).copy(), q=np.asarray(pose.q))

    def compute_eef_jacobian_base(self, qpos):
        self.model.compute_forward_kinematics(qpos)
        jacobian = np.asarray(
            self.model.compute_single_link_jacobian(qpos, self.eef_link_id, False)
        )[:, :7]
        if jacobian.shape != (6, 7) or not np.isfinite(jacobian).all():
            raise ValueError("IK Jacobian must be finite shape (6, 7)")
        position = np.asarray(self.model.get_link_pose(self.eef_link_id).p)
        # Pinocchio WORLD 的线速度原点在基座，转换到 EEF 原点。
        linear = jacobian[:3] + np.cross(jacobian[3:].T, position).T
        return np.vstack((linear, jacobian[3:]))

    def solve(self, target, current, previous, *, hand_qpos):
        import mplib

        current, previous = (np.asarray(q, dtype=np.float64) for q in (current, previous))
        self.constraints.collision_model.set_hand_qpos(hand_qpos)
        target_base = mplib.Pose(p=target.p, q=target.q)
        pool: list[IKCandidate] = []
        tried: list[np.ndarray] = []
        rejections = []
        converged = False

        def try_seed(seed):
            nonlocal converged
            # 相近但不同的初值仍可能到达不同分支，仅跳过完全相同的初值。
            if any(np.array_equal(seed, old) for old in tried):
                return None
            tried.append(seed.copy())
            candidate, reason = self.try_clik(seed, target_base, target, current, previous, pool)
            if reason != "solver_no_convergence":
                converged = True
            if candidate is not None:
                pool.append(candidate)
            elif reason != "duplicate":
                rejections.append(reason)
            return candidate

        for index, seed in enumerate((previous, current)):
            candidate = try_seed(seed)
            if (
                index == 0
                and candidate is not None
                and candidate.rejection_reason is None
                and not candidate.near_limit
                and candidate.displacement_rad <= FAST_ACCEPT_RAD
                and candidate.position_error_m <= 0.5 * self.position_tolerance
                and candidate.rotation_error_rad <= 0.5 * self.rotation_tolerance
            ):
                return IKResult(qpos=candidate.qpos)

        best = self.select_valid(pool)
        if best is not None and not best.near_limit:
            return IKResult(qpos=best.qpos)
        if pool:
            # 碰撞端点可以作为搜索锚点，但不参与最终解的选择。
            anchor = min(pool, key=lambda c: c.cost)
            direction = self.nullspace_direction(anchor.qpos)
            if direction is not None:
                for seed in self.nullspace_seeds(anchor.qpos, direction):
                    try_seed(seed)
            best = self.select_valid(pool)
            if best is not None:
                return IKResult(qpos=best.qpos)

        rejections.extend(c.rejection_reason for c in pool if c.rejection_reason is not None)
        reason = "solver_no_convergence"
        if converged:
            reason = "no_valid_candidate: " + "; ".join(dict.fromkeys(rejections))
        return IKResult(qpos=None, reason=reason)

    def try_clik(self, seed, target_base, target, current, previous, pool):
        raw, converged, _ = self.model.compute_IK_CLIK(
            self.eef_link_id, target_base, seed.copy(), []
        )
        if raw is not None:
            raw = np.asarray(raw, dtype=np.float64)
            if raw.shape != (7,) or not np.isfinite(raw).all():
                raise ValueError("CLIK output must be finite shape (7,)")
        if not converged:
            return None, "solver_no_convergence"
        if raw is None:
            raise ValueError("CLIK reported convergence without a joint vector")

        lower, upper = self.limits.T
        q = wrap_nearest_equivalent(raw, current, lower, upper)
        reason = joint_target_rejection(q, self.limits)
        if reason is not None:
            return None, reason
        if any(np.max(np.abs(q - c.qpos)) < 1e-4 for c in pool):
            return None, "duplicate"
        pose = self.compute_eef_pose_base(q)
        position_error, rotation_error = compute_pose_error(target, pose)
        if not np.isfinite((position_error, rotation_error)).all():
            raise ValueError("IK pose error is non-finite")
        if position_error > self.position_tolerance or rotation_error > self.rotation_tolerance:
            return None, "pose_error"

        delta = equivalent_joint_delta(q, current)
        previous_delta = equivalent_joint_delta(q, previous)
        cost = (
            np.sqrt(np.sum(JOINT_WEIGHTS * (delta / self.joint_ranges) ** 2))
            + 0.25 * np.sqrt(np.sum(PREVIOUS_WEIGHTS * (previous_delta / self.joint_ranges) ** 2))
            + 0.01 * self.joint_limit_penalty(q)
        )
        candidate = IKCandidate(
            q,
            position_error,
            rotation_error,
            float(np.max(np.abs(delta))),
            float(cost),
            bool(np.min(np.minimum(q - lower, upper - q)) < LIMIT_MARGIN_RAD),
            self.constraints.endpoint_rejection(q, position=pose.p),
        )
        return candidate, ""

    @staticmethod
    def select_valid(candidates):
        return min(
            (c for c in candidates if c.rejection_reason is None),
            key=lambda c: c.cost,
            default=None,
        )

    def joint_limit_penalty(self, qpos):
        return float(np.sum(((qpos - self.limit_center) / self.limit_radius) ** 2))

    def nullspace_direction(self, qpos):
        jacobian = self.compute_eef_jacobian_base(qpos)
        norms = np.array([np.linalg.norm(jacobian[:3]), np.linalg.norm(jacobian[3:])])
        if not np.isfinite(norms).all():
            raise ValueError("IK Jacobian block norms are non-finite")
        if np.any(norms <= NUMERIC_EPS):
            return None
        _, sigma, vh = np.linalg.svd(jacobian / np.repeat(norms, 3)[:, None], full_matrices=True)
        # 秩诊断只限制搜索方向，不额外拒绝已经满足约束的解。
        rank = np.count_nonzero(sigma > SVD_RANK_RTOL * sigma[0])
        return vh[-1] if rank == 6 else None

    def nullspace_seeds(self, anchor, direction):
        desired = NULLSPACE_STEP_RAD * direction / np.max(np.abs(direction))
        seeds = []
        for sign in (1, -1):
            delta = sign * desired
            moving = np.abs(delta) > NUMERIC_EPS
            room = np.where(delta > 0, self.limits[:, 1] - anchor, anchor - self.limits[:, 0])
            fraction = min(1.0, float(np.min(room[moving] / np.abs(delta[moving]))))
            if fraction < 1.0:
                fraction = np.nextafter(fraction, 0.0)
            seed = anchor + fraction * delta
            if np.max(np.abs(seed - anchor)) <= 1e-8:
                continue
            if np.any(seed < self.limits[:, 0]) or np.any(seed > self.limits[:, 1]):
                continue
            seeds.append(seed)
        return sorted(seeds, key=self.joint_limit_penalty)
