"""Pure action-candidate validation before robot command publication."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from dexmani_real.policy.runtime import ActionCandidate
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.schema import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE

logger = get_logger(__name__)


class GateRejectCode(str, Enum):
    """Stable machine-readable rejection reasons from :class:`SafetyGate`."""

    UNSUPPORTED_CONTRACT = "unsupported representation/units/frame"
    RUN_GENERATION_MISMATCH = "run generation mismatch"
    INVALID_CURRENT_ARM_SHAPE = "invalid current arm joint state shape"
    NONFINITE_CURRENT_ARM = "current arm joint state contains NaN/Inf"
    INVALID_CANDIDATE_SHAPE = "invalid candidate joint shape"
    NONFINITE_CANDIDATE = "candidate contains NaN/Inf"
    ARM_JOINT_LIMIT = "arm joint limit violation"
    HAND_JOINT_LIMIT = "hand joint limit violation"
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
    """Fail-closed validation of representation, limits, and workspace."""

    def __init__(
        self,
        *,
        arm_joint_lower_rad: tuple[float, ...],
        arm_joint_upper_rad: tuple[float, ...],
        hand_joint_lower_rad: tuple[float, ...],
        hand_joint_upper_rad: tuple[float, ...],
        workspace_check: Callable[[np.ndarray, np.ndarray], bool] | None = None,
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

    def validate(
        self,
        candidate: ActionCandidate,
        *,
        current_arm_qpos: np.ndarray,
        run_generation: int,
    ) -> GateResult:
        """Validate one candidate without modifying it or external state."""
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
        if self.workspace_check is not None and candidate.arm_qpos is not None:
            try:
                if not self.workspace_check(arm_start, arm_end):
                    return GateResult(False, GateRejectCode.WORKSPACE)
            except Exception:
                logger.warning(
                    "SafetyGate: workspace check failed closed", exc_info=True
                )
                return GateResult(False, GateRejectCode.WORKSPACE_CHECK_FAILED)
        return GateResult(True)


def planner_action_safety_gate(
    *,
    planner: Any,
    arm_joint_lower_rad: tuple[float, ...],
    arm_joint_upper_rad: tuple[float, ...],
    hand_joint_lower_rad: tuple[float, ...],
    hand_joint_upper_rad: tuple[float, ...],
) -> SafetyGate:
    """Build a safety gate using the planner's segment workspace check."""
    return SafetyGate(
        arm_joint_lower_rad=arm_joint_lower_rad,
        arm_joint_upper_rad=arm_joint_upper_rad,
        hand_joint_lower_rad=hand_joint_lower_rad,
        hand_joint_upper_rad=hand_joint_upper_rad,
        workspace_check=planner.is_workspace_segment_safe,
    )
