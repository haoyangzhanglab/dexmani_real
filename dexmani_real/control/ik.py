"""连续初值 CLIK 与有界零空间探索。"""

from dataclasses import dataclass

import numpy as np

from dexmani_real.control.constraints import (
    joint_target_rejection,
    wrap_nearest_equivalent,
)
from dexmani_real.geometry.kinematics import ArmFK
from dexmani_real.geometry.pose import Pose, compute_pose_error

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
        self.constraints = constraints
        self.position_tolerance = position_tolerance
        self.rotation_tolerance = rotation_tolerance
        if any(not np.isfinite(v) or v <= 0 for v in (position_tolerance, rotation_tolerance)):
            raise ValueError("IK pose tolerances must be finite and positive")
        self.fk = ArmFK()
        frame = self.fk.model.frames[self.fk.eef_frame_id]
        self.eef_joint_id = frame.parent
        self.eef_from_joint = frame.placement.inverse()
        model_limits = np.column_stack(
            (self.fk.model.lowerPositionLimit, self.fk.model.upperPositionLimit)
        )
        if not np.isfinite(model_limits).all() or np.any(model_limits[:, 0] >= model_limits[:, 1]):
            raise ValueError("IK requires finite, ordered model limits")
        self.limits = constraints.limits
        if np.any(self.limits[:, 0] < model_limits[:, 0] - MODEL_LIMIT_TOL_RAD) or np.any(
            self.limits[:, 1] > model_limits[:, 1] + MODEL_LIMIT_TOL_RAD
        ):
            raise ValueError("operational limits exceed the loaded model limits")
        self.joint_ranges = model_limits[:, 1] - model_limits[:, 0]
        self.limit_center = self.limits.mean(axis=1)
        self.limit_radius = 0.5 * (self.limits[:, 1] - self.limits[:, 0])

    def compute_eef_pose_base(self, qpos: np.ndarray) -> Pose:
        transform = self.fk.forward(qpos)
        quaternion = self.fk.pin.Quaternion(transform[:3, :3]).coeffs()[[3, 0, 1, 2]]
        return Pose(p=transform[:3, 3], q=quaternion)

    def compute_eef_jacobian_base(self, qpos: np.ndarray) -> np.ndarray:
        pin = self.fk.pin
        # 基座轴向、EEF 原点，供零空间秩诊断与搜索使用。
        return pin.computeFrameJacobian(
            self.fk.model, self.fk.data, qpos, self.fk.eef_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

    def solve(self, target: Pose, current, previous, *, hand_qpos) -> IKResult:
        current, previous = (np.asarray(q, dtype=np.float64) for q in (current, previous))
        self.constraints.collision_model.set_hand_qpos(hand_qpos)
        pool: list[IKCandidate] = []
        tried: list[np.ndarray] = []
        rejections = []

        def try_seed(seed):
            # 相近但不同的初值仍可能到达不同分支，仅跳过完全相同的初值。
            if any(np.array_equal(seed, old) for old in tried):
                return None
            tried.append(seed)
            candidate, reason = self.try_clik(seed, target, current, previous)
            if candidate is not None:
                pool.append(candidate)
            else:
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
        reason = "no_valid_candidate: " + "; ".join(dict.fromkeys(rejections))
        return IKResult(qpos=None, reason=reason)

    def compute_clik(self, seed: np.ndarray, target: Pose) -> np.ndarray:
        """目标系 log6 误差配合父关节 LOCAL 雅可比，最多更新 50 次。"""
        pin, model, data = self.fk.pin, self.fk.model, self.fk.data
        target_joint = pin.SE3(pin.Quaternion(*target.q).matrix(), target.p) * self.eef_from_joint
        q = seed.copy()
        for iteration in range(51):
            pin.forwardKinematics(model, data, q)
            error = pin.log6(target_joint.actInv(data.oMi[self.eef_joint_id])).vector
            if np.linalg.norm(error) < 1e-5 or iteration == 50:
                break
            jacobian = pin.computeJointJacobian(model, data, q, self.eef_joint_id)
            regularized = jacobian @ jacobian.T
            regularized.flat[::7] += 1e-12
            velocity = -jacobian.T @ np.linalg.solve(regularized, error)
            q = pin.integrate(model, q, velocity * .1)
        return q

    def try_clik(self, seed, target: Pose, current, previous) -> tuple[IKCandidate | None, str]:
        raw = self.compute_clik(seed, target)
        if not np.isfinite(raw).all():
            raise ValueError("IK iterate is non-finite")

        lower, upper = self.limits.T
        q = wrap_nearest_equivalent(raw, current, lower, upper)
        reason = joint_target_rejection(q, self.limits)
        if reason is not None:
            return None, reason
        pose = self.compute_eef_pose_base(q)
        position_error, rotation_error = compute_pose_error(target, pose)
        if not np.isfinite((position_error, rotation_error)).all():
            raise ValueError("IK pose error is non-finite")
        if position_error > self.position_tolerance or rotation_error > self.rotation_tolerance:
            return None, "pose_error"

        delta = q - current
        previous_delta = q - previous
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
