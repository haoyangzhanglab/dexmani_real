"""Pure action-candidate validation before robot command publication."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from dexmani_real.config.defaults import policy as policy_defaults
from dexmani_real.control.action import ActionCandidate
from dexmani_real.robot_spec import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_JOINT_LIMIT_TOLERANCE_RAD = 1e-12


def _hand_joint_limit_detail(
    hand_qpos_rad: np.ndarray,
    lower_rad: np.ndarray,
    upper_rad: np.ndarray,
) -> str:
    """Render the exact hand endpoint components outside the gate envelope.

    This is diagnostic-only: it does not modify the rejected target or relax
    the operational limit.  Keeping the values in the gate result makes a
    shadow log sufficient to distinguish a slightly-open policy endpoint from
    a mechanically unsafe one in the subsequent review.
    """
    below = np.flatnonzero(hand_qpos_rad < lower_rad - _JOINT_LIMIT_TOLERANCE_RAD)
    above = np.flatnonzero(hand_qpos_rad > upper_rad + _JOINT_LIMIT_TOLERANCE_RAD)
    violations: list[str] = []
    violations.extend(
        (
            f"j{index}: target={hand_qpos_rad[index]:.17g}, "
            f"lower={lower_rad[index]:.17g}, "
            f"delta={hand_qpos_rad[index] - lower_rad[index]:+.3e}"
        )
        for index in below
    )
    violations.extend(
        (
            f"j{index}: target={hand_qpos_rad[index]:.17g}, "
            f"upper={upper_rad[index]:.17g}, "
            f"delta={hand_qpos_rad[index] - upper_rad[index]:+.3e}"
        )
        for index in above
    )
    if not violations:
        raise ValueError("hand joint-limit diagnostic requires an out-of-bounds target")
    return "hand joint limit violation (rad): " + ", ".join(violations)


def _joint_delta_limit_detail(
    *,
    joint_group: str,
    target_rad: np.ndarray,
    reference_rad: np.ndarray,
    limit_rad: np.ndarray,
    tolerance_rad: float,
    reference_kind: str,
) -> str:
    """Render the exact components that exceed a reject-only delta envelope."""
    delta_rad = target_rad - reference_rad
    abs_delta_rad = np.abs(delta_rad)
    effective_limit_rad = limit_rad + tolerance_rad
    violating = np.flatnonzero(abs_delta_rad > effective_limit_rad)
    if violating.size == 0:
        raise ValueError("delta-limit diagnostic requires an exceeded component")
    max_index = int(np.argmax(abs_delta_rad))
    violations = ", ".join(
        (
            f"j{index}: reference={reference_rad[index]:.17g}, "
            f"target={target_rad[index]:.17g}, "
            f"delta={delta_rad[index]:+.17g}, "
            f"abs_delta={abs_delta_rad[index]:.17g}, "
            f"limit={limit_rad[index]:.17g}, "
            f"excess={abs_delta_rad[index] - effective_limit_rad[index]:+.3e}"
        )
        for index in violating
    )
    return (
        f"{joint_group} per-tick delta limit violation "
        f"(rad; reference={reference_kind}; "
        f"tolerance={tolerance_rad:.3e}; "
        f"max_abs_delta={abs_delta_rad[max_index]:.17g} at j{max_index}): "
        f"{violations}"
    )


class GateRejectCode(str, Enum):
    """Stable machine-readable rejection reasons from :class:`SafetyGate`."""

    ARM_JOINT_LIMIT = "arm joint limit violation"
    HAND_JOINT_LIMIT = "hand joint limit violation"
    ARM_DELTA_LIMIT = "arm per-tick delta limit violation"
    HAND_DELTA_LIMIT = "hand per-tick delta limit violation"
    COLLISION_TRANSITION = "collision on arm/hand transition"
    COLLISION_CHECK_FAILED = "collision transition check failed"
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
    """Fail-closed validation of physical limits, delta, workspace, and collision.

    Delta and collision checks are opt-in (default ``None`` → disabled) so the
    shared gate used by teleop/replay/calibration keeps its existing behavior;
    only the learned-policy deployment enables them.  Delta limits *reject*
    whole (never clip): clipping a learned action silently rewrites the model's
    intent (plan §9).  ``endpoint_delta_tolerance_rad`` is a shared numerical
    slack applied to both arm and hand endpoint predicates.
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
        endpoint_delta_tolerance_rad: float = (
            policy_defaults.endpoint_delta_tolerance_rad
        ),
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
        if (
            isinstance(endpoint_delta_tolerance_rad, bool)
            or not np.isfinite(endpoint_delta_tolerance_rad)
            or endpoint_delta_tolerance_rad < 0.0
        ):
            raise ValueError(
                "endpoint_delta_tolerance_rad must be finite and non-negative"
            )
        self.endpoint_delta_tolerance_rad = float(endpoint_delta_tolerance_rad)
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
    ) -> GateResult:
        """Validate one candidate without modifying it or external state.

        Workspace and collision transitions start at measured feedback. Optional
        delta references are the previous published targets, so actuator lag
        cannot turn a command-rate limit into an unintended tracking-error gate.
        """
        # ActionCandidate owns command structure. The feedback reader owns the
        # shape/dtype/finite contract of these measured arrays.
        arm_start = current_arm_qpos
        arm_delta_start = arm_start
        arm_delta_reference_kind = "measured_feedback"
        if arm_delta_reference_qpos is not None:
            arm_delta_start = arm_delta_reference_qpos
            arm_delta_reference_kind = "previous_published_target"
        arm_end = arm_start.copy() if candidate.arm_qpos is None else candidate.arm_qpos
        hand_end = candidate.hand_qpos
        hand_start: np.ndarray | None = None
        hand_delta_start: np.ndarray | None = None
        hand_delta_reference_kind = "measured_feedback"
        if hand_end is not None:
            assert current_hand_qpos is not None
            hand_start = current_hand_qpos
            hand_delta_start = hand_start
            if hand_delta_reference_qpos is not None:
                hand_delta_start = hand_delta_reference_qpos
                hand_delta_reference_kind = "previous_published_target"
        if candidate.arm_qpos is not None and (
            np.any(arm_end < self.arm_low) or np.any(arm_end > self.arm_high)
        ):
            return GateResult(False, GateRejectCode.ARM_JOINT_LIMIT)
        if hand_end is not None and (
            np.any(hand_end < self.hand_low - _JOINT_LIMIT_TOLERANCE_RAD)
            or np.any(hand_end > self.hand_high + _JOINT_LIMIT_TOLERANCE_RAD)
        ):
            return GateResult(
                False,
                GateRejectCode.HAND_JOINT_LIMIT,
                _hand_joint_limit_detail(hand_end, self.hand_low, self.hand_high),
            )
        # Per-tick delta limits (reject, never clip).
        if self.max_arm_delta_rad is not None and candidate.arm_qpos is not None:
            if np.any(
                np.abs(arm_end - arm_delta_start)
                > self.max_arm_delta_rad + self.endpoint_delta_tolerance_rad
            ):
                return GateResult(
                    False,
                    GateRejectCode.ARM_DELTA_LIMIT,
                    _joint_delta_limit_detail(
                        joint_group="arm",
                        target_rad=arm_end,
                        reference_rad=arm_delta_start,
                        limit_rad=self.max_arm_delta_rad,
                        tolerance_rad=self.endpoint_delta_tolerance_rad,
                        reference_kind=arm_delta_reference_kind,
                    ),
                )
        if (
            self.max_hand_delta_rad is not None
            and hand_end is not None
            and hand_delta_start is not None
        ):
            if np.any(
                np.abs(hand_end - hand_delta_start)
                > self.max_hand_delta_rad + self.endpoint_delta_tolerance_rad
            ):
                return GateResult(
                    False,
                    GateRejectCode.HAND_DELTA_LIMIT,
                    _joint_delta_limit_detail(
                        joint_group="hand",
                        target_rad=hand_end,
                        reference_rad=hand_delta_start,
                        limit_rad=self.max_hand_delta_rad,
                        tolerance_rad=self.endpoint_delta_tolerance_rad,
                        reference_kind=hand_delta_reference_kind,
                    ),
                )
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
                return GateResult(False, GateRejectCode.COLLISION_CHECK_FAILED)
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
    endpoint_delta_tolerance_rad: float = (
        policy_defaults.endpoint_delta_tolerance_rad
    ),
    collision_check: (
        Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], bool] | None
    ) = None,
) -> SafetyGate:
    """Build a safety gate using the planner's segment workspace check.

    ``max_arm_delta_rad`` / ``max_hand_delta_rad`` / ``collision_check`` are
    opt-in; each caller enables only the checks owned by its command path.
    The endpoint tolerance defaults to the canonical policy runtime default.
    """
    return SafetyGate(
        arm_joint_lower_rad=arm_joint_lower_rad,
        arm_joint_upper_rad=arm_joint_upper_rad,
        hand_joint_lower_rad=hand_joint_lower_rad,
        hand_joint_upper_rad=hand_joint_upper_rad,
        workspace_check=planner.is_workspace_segment_safe,
        max_arm_delta_rad=max_arm_delta_rad,
        max_hand_delta_rad=max_hand_delta_rad,
        endpoint_delta_tolerance_rad=endpoint_delta_tolerance_rad,
        collision_check=collision_check,
    )
