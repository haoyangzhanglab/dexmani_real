"""有界局部 EEF 跟随；返回下一条目标，不表示到达输入位姿。"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from dexmani_real.geometry.kinematics import ArmFK
from dexmani_real.geometry.pose import Pose
from dexmani_real.robot.commands import (
    FEEDBACK_POSITION_ROUNDOFF_RAD,
    arm_reference_from_feedback,
    validate_arm_target,
)

JOINT_SCALE = np.sqrt([4.0, 1.8, 1.2, 0.6, 0.6, 0.9, 0.35])
TASK_SCALE = np.array([1 / 0.3] * 3 + [1.0] * 3)
CONTINUITY_WEIGHT = 0.03
ANCHOR_WEIGHT = 0.03
MAX_EVALUATIONS = 30
# TRF 将边界初值内移约 1e-10，允许相应的目标函数舍入量。
OBJECTIVE_ROUNDOFF = 1e-9


class IKReferenceError(RuntimeError):
    """参考无法支持本段继续运动；调用方须终段。"""


@dataclass(kw_only=True)
class IKResult:
    qpos: np.ndarray | None
    reason: str = ""


class IKSolver:
    def __init__(self, constraints, *, target_speed, max_lead, device_speed):
        if not all(
            np.isfinite(v) and v > 0 for v in (target_speed, max_lead, device_speed)
        ):
            raise ValueError("IK speed and lead must be finite and positive")
        self.constraints = constraints
        self.speed, self.lead = min(target_speed, device_speed), max_lead
        self.limits = constraints.limits
        self.fk = ArmFK()
        model = self.fk.model
        model_limits = np.column_stack(
            (model.lowerPositionLimit, model.upperPositionLimit)
        )
        for limits in (self.limits, model_limits):
            if (
                limits.shape != (7, 2)
                or not np.isfinite(limits).all()
                or np.any(limits[:, 0] >= limits[:, 1])
            ):
                raise ValueError("IK requires finite ordered joint limits")
        if np.any(self.limits[:, 0] < model.lowerPositionLimit - 1e-5) or np.any(
            self.limits[:, 1] > model.upperPositionLimit + 1e-5
        ):
            raise ValueError("operational limits exceed model limits")

    def measured_reference(self, qpos: np.ndarray) -> np.ndarray:
        """有限但越限的实测终止本段；格式与数值错误继续传播。"""
        try:
            return arm_reference_from_feedback(qpos, *self.limits.T)
        except ValueError as exc:
            if np.shape(qpos) != (7,) or not np.isfinite(qpos).all():
                raise
            raise IKReferenceError("measured arm outside operational limits") from exc

    def compute_eef_pose_base(self, qpos: np.ndarray) -> Pose:
        transform = self.fk.forward(qpos)
        return Pose(
            transform[:3, 3],
            self.fk.pin.Quaternion(transform[:3, :3]).coeffs()[[3, 0, 1, 2]],
        )

    def bounds(
        self, measured: np.ndarray, command: np.ndarray, anchor: np.ndarray, h: float
    ):
        if not np.isfinite(h) or h < 0:
            raise ValueError("invalid IK time budget")
        for q in (measured, command, anchor):
            if np.shape(q) != (7,) or not np.isfinite(q).all():
                raise ValueError("IK references must be finite (7,) vectors")
            if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
                raise IKReferenceError("IK reference outside operational limits")
        if np.any(
            np.abs(command - measured) > self.lead + FEEDBACK_POSITION_ROUNDOFF_RAD
        ):
            raise IKReferenceError("IK command exceeds measured lead range")
        lower = np.maximum.reduce(
            (self.limits[:, 0], command - self.speed * h, measured - self.lead)
        )
        upper = np.minimum.reduce(
            (self.limits[:, 1], command + self.speed * h, measured + self.lead)
        )
        if np.any(lower > upper) or (h > 0 and np.any(lower == upper)):
            raise IKReferenceError("IK feasible range unavailable")
        return lower, upper

    def residual_and_jacobian(self, q, goal, reference, weight):
        pin, model, data = self.fk.pin, self.fk.model, self.fk.data
        pin.computeJointJacobians(model, data, q)
        transform = pin.updateFramePlacement(model, data, self.fk.eef_frame_id)
        error = transform.actInv(goal)
        local = pin.getFrameJacobian(
            model, data, self.fk.eef_frame_id, pin.ReferenceFrame.LOCAL
        )
        residual = np.concatenate(
            (
                TASK_SCALE * pin.log6(error).vector,
                weight * JOINT_SCALE * (q - reference),
            )
        )
        jacobian = np.vstack(
            (
                -TASK_SCALE[:, None] * (pin.Jlog6(error.inverse()) @ local),
                np.diag(weight * JOINT_SCALE),
            )
        )
        if not np.isfinite(residual).all() or not np.isfinite(jacobian).all():
            raise ValueError("non-finite IK residual/Jacobian")
        return residual, jacobian

    def solve(
        self, target: Pose, measured, command, anchor, *, h: float, hand_qpos
    ) -> IKResult:
        lower, upper = self.bounds(measured, command, anchor, h)
        seed = np.clip(command, lower, upper)  # 仅初值；最终输出不裁剪。
        weight = np.hypot(CONTINUITY_WEIGHT, ANCHOR_WEIGHT)
        reference = (
            CONTINUITY_WEIGHT**2 * command + ANCHOR_WEIGHT**2 * anchor
        ) / weight**2
        pin = self.fk.pin
        goal = pin.SE3(pin.Quaternion(*target.q).matrix(), target.p)
        cached_q = cached_value = None

        def evaluate(q):
            nonlocal cached_q, cached_value
            if cached_q is None or not np.array_equal(q, cached_q):
                cached_q = q.copy()
                cached_value = self.residual_and_jacobian(q, goal, reference, weight)
            return cached_value

        initial = evaluate(seed)[0]
        initial_cost = initial @ initial
        q = seed
        if h > 0:
            result = least_squares(
                lambda x: evaluate(x)[0],
                seed,
                jac=lambda x: evaluate(x)[1],
                bounds=(lower, upper),
                method="trf",
                tr_solver="exact",
                max_nfev=MAX_EVALUATIONS,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
            )
            if result.status < 0:
                raise RuntimeError(f"IK optimizer failed: {result.message}")
            q = result.x
        # 固定边界已包含操作限位；一次检查覆盖最终数值和全部关节约束。
        validate_arm_target(q, lower, upper)
        final = evaluate(q)[0]
        if final @ final > initial_cost + OBJECTIVE_ROUNDOFF * max(1.0, initial_cost):
            raise RuntimeError("IK objective increased")
        pose = self.compute_eef_pose_base(q)
        self.constraints.collision_model.set_hand_qpos(hand_qpos)
        reason = self.constraints.endpoint_rejection(q, position=pose.p)
        return IKResult(qpos=q.copy() if reason is None else None, reason=reason or "")
