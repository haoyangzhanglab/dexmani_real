"""Pure action-candidate validation before robot command publication."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from dexmani_real.control.action import ActionCandidate
from dexmani_real.robot_spec import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


class GateRejectCode(str, Enum):
    """Stable machine-readable rejection reasons from :class:`SafetyGate`."""

    UNSUPPORTED_CONTRACT = "unsupported representation/units/frame"
    RUN_GENERATION_MISMATCH = "run generation mismatch"
    INVALID_CURRENT_ARM_SHAPE = "invalid current arm joint state shape"
    NONFINITE_CURRENT_ARM = "current arm joint state contains NaN/Inf"
    INVALID_CURRENT_HAND_SHAPE = "invalid current hand joint state shape"
    NONFINITE_CURRENT_HAND = "current hand joint state contains NaN/Inf"
    INVALID_CANDIDATE_SHAPE = "invalid candidate joint shape"
    NONFINITE_CANDIDATE = "candidate contains NaN/Inf"
    ARM_JOINT_LIMIT = "arm joint limit violation"
    HAND_JOINT_LIMIT = "hand joint limit violation"
    ARM_DELTA_LIMIT = "arm per-tick delta limit violation"
    HAND_DELTA_LIMIT = "hand per-tick delta limit violation"
    COLLISION_TRANSITION = "collision on arm/hand transition"
    WORKSPACE = "workspace"
    WORKSPACE_CHECK_FAILED = "workspace check failed"


@dataclass(frozen=True)
class GateResult:
    """Typed outcome of one safety-gate validation."""

    accepted: bool
    code: GateRejectCode | None = None
    detail: str = ""

    @property
    def reason(self) -> str:
        return self.detail or ("" if self.code is None else self.code.value)


class SafetyGate:
    """Fail-closed validation of representation, limits, delta, and collision.

    Delta and collision checks are opt-in (default ``None`` → disabled) so the
    shared gate used by teleop/replay/calibration keeps its existing behavior;
    only the learned-policy deployment enables them.  Delta limits *reject*
    whole (never clip): clipping a learned action silently rewrites the model's
    intent (plan §9).
    """

    def __init__(
        self,
        *,
        arm_joint_lower_rad: tuple[float, ...],
        arm_joint_upper_rad: tuple[float, ...],
        hand_joint_lower_rad: tuple[float, ...],
        hand_joint_upper_rad: tuple[float, ...],
        workspace_check: Callable[[np.ndarray, np.ndarray], bool] | None = None,
        max_arm_delta_rad: Any = None,
        max_hand_delta_rad: Any = None,
        collision_check: (
            Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], bool] | None
        ) = None,
    ) -> None:
        arm_low = np.asarray(arm_joint_lower_rad, dtype=np.float64)
        arm_high = np.asarray(arm_joint_upper_rad, dtype=np.float64)
        hand_low = np.asarray(hand_joint_lower_rad, dtype=np.float64)
        hand_high = np.asarray(hand_joint_upper_rad, dtype=np.float64)
        if arm_low.shape != ARM_JOINT_SHAPE or arm_high.shape != ARM_JOINT_SHAPE:
            raise ValueError("arm joint limits must have seven entries")
        if hand_low.shape != HAND_JOINT_SHAPE or hand_high.shape != HAND_JOINT_SHAPE:
            raise ValueError("hand joint limits must have twelve entries")
        bounds = np.concatenate((arm_low, arm_high, hand_low, hand_high))
        if (
            not np.all(np.isfinite(bounds))
            or np.any(arm_low > arm_high)
            or np.any(hand_low > hand_high)
        ):
            raise ValueError("joint limits must be finite and ordered")
        self.arm_low = arm_low
        self.arm_high = arm_high
        self.hand_low = hand_low
        self.hand_high = hand_high
        self.workspace_check = workspace_check
        self.max_arm_delta_rad = self._coerce_delta(
            max_arm_delta_rad, ARM_JOINT_SHAPE, "max_arm_delta_rad"
        )
        self.max_hand_delta_rad = self._coerce_delta(
            max_hand_delta_rad, HAND_JOINT_SHAPE, "max_hand_delta_rad"
        )
        self.collision_check = collision_check

    @staticmethod
    def _coerce_delta(
        value: Any, shape: tuple[int, ...], name: str
    ) -> np.ndarray | None:
        if value is None:
            return None
        arr = np.broadcast_to(np.asarray(value, dtype=np.float64), shape).copy()
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError(f"{name} must be finite and positive")
        return arr

    def validate(
        self,
        candidate: ActionCandidate,
        *,
        current_arm_qpos: np.ndarray,
        current_hand_qpos: np.ndarray | None = None,
        arm_delta_reference_qpos: np.ndarray | None = None,
        hand_delta_reference_qpos: np.ndarray | None = None,
        run_generation: int,
    ) -> GateResult:
        """Validate one candidate without modifying it or external state.

        Workspace and collision transitions start at measured feedback. Optional
        delta references are the previous published targets, so actuator lag
        cannot turn a command-rate limit into an unintended tracking-error gate.
        """
        if (
            candidate.representation != "joint_position"
            or candidate.units != "rad"
            or candidate.frame != "robot_joint"
        ):
            return GateResult(False, GateRejectCode.UNSUPPORTED_CONTRACT)
        if candidate.run_generation != run_generation:
            return GateResult(False, GateRejectCode.RUN_GENERATION_MISMATCH)

        arm_start = np.asarray(current_arm_qpos, dtype=np.float64)
        if arm_start.shape != ARM_JOINT_SHAPE:
            return GateResult(False, GateRejectCode.INVALID_CURRENT_ARM_SHAPE)
        if not np.all(np.isfinite(arm_start)):
            return GateResult(False, GateRejectCode.NONFINITE_CURRENT_ARM)
        arm_delta_start = arm_start
        if arm_delta_reference_qpos is not None:
            arm_delta_start = np.asarray(arm_delta_reference_qpos, dtype=np.float64)
            if arm_delta_start.shape != ARM_JOINT_SHAPE:
                return GateResult(False, GateRejectCode.INVALID_CURRENT_ARM_SHAPE)
            if not np.all(np.isfinite(arm_delta_start)):
                return GateResult(False, GateRejectCode.NONFINITE_CURRENT_ARM)
        arm_end = (
            arm_start.copy()
            if candidate.arm_qpos is None
            else np.asarray(candidate.arm_qpos, dtype=np.float64).copy()
        )
        hand_end = (
            None
            if candidate.hand_qpos is None
            else np.asarray(candidate.hand_qpos, dtype=np.float64).copy()
        )
        hand_start: np.ndarray | None = None
        hand_delta_start: np.ndarray | None = None
        if hand_end is not None:
            if current_hand_qpos is None:
                # A hand target without a current hand state cannot be delta- or
                # collision-checked; fail closed rather than silently skipping.
                if (
                    self.max_hand_delta_rad is not None
                    or self.collision_check is not None
                ):
                    return GateResult(False, GateRejectCode.INVALID_CURRENT_HAND_SHAPE)
            else:
                hand_start = np.asarray(current_hand_qpos, dtype=np.float64)
                if hand_start.shape != HAND_JOINT_SHAPE:
                    return GateResult(False, GateRejectCode.INVALID_CURRENT_HAND_SHAPE)
                if not np.all(np.isfinite(hand_start)):
                    return GateResult(False, GateRejectCode.NONFINITE_CURRENT_HAND)
            hand_delta_start = hand_start
            if hand_delta_reference_qpos is not None:
                hand_delta_start = np.asarray(
                    hand_delta_reference_qpos, dtype=np.float64
                )
                if hand_delta_start.shape != HAND_JOINT_SHAPE:
                    return GateResult(False, GateRejectCode.INVALID_CURRENT_HAND_SHAPE)
                if not np.all(np.isfinite(hand_delta_start)):
                    return GateResult(False, GateRejectCode.NONFINITE_CURRENT_HAND)
        if arm_end.shape != ARM_JOINT_SHAPE or (
            hand_end is not None and hand_end.shape != HAND_JOINT_SHAPE
        ):
            return GateResult(False, GateRejectCode.INVALID_CANDIDATE_SHAPE)
        if not np.all(np.isfinite(arm_end)) or (
            hand_end is not None and not np.all(np.isfinite(hand_end))
        ):
            return GateResult(False, GateRejectCode.NONFINITE_CANDIDATE)
        if candidate.arm_qpos is not None and (
            np.any(arm_end < self.arm_low) or np.any(arm_end > self.arm_high)
        ):
            return GateResult(False, GateRejectCode.ARM_JOINT_LIMIT)
        if hand_end is not None and (
            np.any(hand_end < self.hand_low - 1e-12)
            or np.any(hand_end > self.hand_high + 1e-12)
        ):
            return GateResult(False, GateRejectCode.HAND_JOINT_LIMIT)
        # Per-tick delta limits (reject, never clip).
        if self.max_arm_delta_rad is not None and candidate.arm_qpos is not None:
            if np.any(np.abs(arm_end - arm_delta_start) > self.max_arm_delta_rad):
                return GateResult(False, GateRejectCode.ARM_DELTA_LIMIT)
        if (
            self.max_hand_delta_rad is not None
            and hand_end is not None
            and hand_delta_start is not None
        ):
            if np.any(np.abs(hand_end - hand_delta_start) > self.max_hand_delta_rad):
                return GateResult(False, GateRejectCode.HAND_DELTA_LIMIT)
        if self.workspace_check is not None and candidate.arm_qpos is not None:
            try:
                if not self.workspace_check(arm_start, arm_end):
                    return GateResult(False, GateRejectCode.WORKSPACE)
            except Exception:
                logger.warning(
                    "SafetyGate: workspace check failed closed", exc_info=True
                )
                return GateResult(False, GateRejectCode.WORKSPACE_CHECK_FAILED)
        # Arm/hand transition collision (requires both current + target hand).
        if (
            self.collision_check is not None
            and candidate.arm_qpos is not None
            and hand_end is not None
            and hand_start is not None
        ):
            try:
                if not self.collision_check(arm_start, arm_end, hand_start, hand_end):
                    return GateResult(False, GateRejectCode.COLLISION_TRANSITION)
            except Exception:
                logger.warning(
                    "SafetyGate: collision transition check failed closed",
                    exc_info=True,
                )
                return GateResult(False, GateRejectCode.COLLISION_TRANSITION)
        return GateResult(True)


def planner_action_safety_gate(
    *,
    planner: Any,
    arm_joint_lower_rad: tuple[float, ...],
    arm_joint_upper_rad: tuple[float, ...],
    hand_joint_lower_rad: tuple[float, ...],
    hand_joint_upper_rad: tuple[float, ...],
    max_arm_delta_rad: Any = None,
    max_hand_delta_rad: Any = None,
    collision_check: (
        Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], bool] | None
    ) = None,
) -> SafetyGate:
    """Build a safety gate using the planner's segment workspace check.

    ``max_arm_delta_rad`` / ``max_hand_delta_rad`` / ``collision_check`` are
    opt-in; each caller enables only the checks owned by its command path.
    """
    return SafetyGate(
        arm_joint_lower_rad=arm_joint_lower_rad,
        arm_joint_upper_rad=arm_joint_upper_rad,
        hand_joint_lower_rad=hand_joint_lower_rad,
        hand_joint_upper_rad=hand_joint_upper_rad,
        workspace_check=planner.is_workspace_segment_safe,
        max_arm_delta_rad=max_arm_delta_rad,
        max_hand_delta_rad=max_hand_delta_rad,
        collision_check=collision_check,
    )
