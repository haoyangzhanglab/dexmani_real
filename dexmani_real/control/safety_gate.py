"""Pure action-candidate validation before robot command publication."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from dexmani_real.config.defaults import policy as policy_defaults
from dexmani_real.control.action import ActionCandidate
from dexmani_real.robot.model import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_JOINT_LIMIT_TOLERANCE_RAD = 1e-12


def _hand_joint_limit_detail(
    hand_qpos_rad: np.ndarray, lower_rad: np.ndarray, upper_rad: np.ndarray
) -> str:
    outside = (hand_qpos_rad < lower_rad - _JOINT_LIMIT_TOLERANCE_RAD) | (
        hand_qpos_rad > upper_rad + _JOINT_LIMIT_TOLERANCE_RAD
    )
    return f"hand_joint_limit:j{np.flatnonzero(outside)[0]}"


def _joint_delta_limit_detail(
    *,
    target_rad: np.ndarray,
    reference_rad: np.ndarray,
    limit_rad: np.ndarray,
    tolerance_rad: float,
) -> str:
    delta = np.abs(target_rad - reference_rad)
    index = int(np.flatnonzero(delta > limit_rad + tolerance_rad)[0])
    return f"hand_delta_limit:j{index}:{delta[index]:.3f}>{limit_rad[index]:.3f}"


class GateRejectCode(str, Enum):
    """Stable machine-readable rejection reasons from :class:`SafetyGate`."""

    INVALID_TARGET = "invalid joint target"
    ARM_JOINT_LIMIT = "arm joint limit violation"
    HAND_JOINT_LIMIT = "hand joint limit violation"
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
    """Fail-closed validation of physical limits, workspace, and collision.

    The optional hand delta check rejects the whole coupled endpoint; learned
    arm spike shaping belongs to its producer. ``endpoint_delta_tolerance_rad``
    is numerical slack for the hand endpoint-delta predicate.
    """

    def __init__(
        self,
        *,
        arm_joint_lower_rad: tuple[float, ...],
        arm_joint_upper_rad: tuple[float, ...],
        hand_joint_lower_rad: tuple[float, ...],
        hand_joint_upper_rad: tuple[float, ...],
        workspace_check: Callable[[np.ndarray, np.ndarray], bool] | None = None,
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
        hand_delta_reference_qpos: np.ndarray | None = None,
    ) -> GateResult:
        """Validate one candidate without modifying it or external state.

        Workspace and collision transitions start at measured feedback. The
        optional hand delta reference is the previous published target, so
        actuator lag cannot become an unintended tracking-error gate.
        """
        # Sensor readers own measured feedback; this gate admits outgoing targets.
        if candidate.arm_qpos is None and candidate.hand_qpos is None:
            return GateResult(
                False, GateRejectCode.INVALID_TARGET, "no actuator target"
            )
        for name, target, shape in (
            ("arm", candidate.arm_qpos, ARM_JOINT_SHAPE),
            ("hand", candidate.hand_qpos, HAND_JOINT_SHAPE),
        ):
            if target is not None and (
                target.shape != shape or not np.all(np.isfinite(target))
            ):
                return GateResult(
                    False, GateRejectCode.INVALID_TARGET, f"{name} target shape/finite"
                )
        arm_start = current_arm_qpos
        arm_end = arm_start.copy() if candidate.arm_qpos is None else candidate.arm_qpos
        hand_end = candidate.hand_qpos
        hand_start: np.ndarray | None = None
        hand_delta_start: np.ndarray | None = None
        if hand_end is not None:
            assert current_hand_qpos is not None
            hand_start = current_hand_qpos
            hand_delta_start = hand_start
            if hand_delta_reference_qpos is not None:
                hand_delta_start = hand_delta_reference_qpos
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
                        target_rad=hand_end,
                        reference_rad=hand_delta_start,
                        limit_rad=self.max_hand_delta_rad,
                        tolerance_rad=self.endpoint_delta_tolerance_rad,
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
    max_hand_delta_rad: Any = None,
    endpoint_delta_tolerance_rad: float = (
        policy_defaults.endpoint_delta_tolerance_rad
    ),
    collision_check: (
        Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], bool] | None
    ) = None,
) -> SafetyGate:
    """Build a safety gate using the planner's segment workspace check.

    ``max_hand_delta_rad`` / ``collision_check`` are opt-in; each caller enables
    only the checks owned by its command path.
    The endpoint tolerance defaults to the canonical policy runtime default.
    """
    return SafetyGate(
        arm_joint_lower_rad=arm_joint_lower_rad,
        arm_joint_upper_rad=arm_joint_upper_rad,
        hand_joint_lower_rad=hand_joint_lower_rad,
        hand_joint_upper_rad=hand_joint_upper_rad,
        workspace_check=planner.is_workspace_segment_safe,
        max_hand_delta_rad=max_hand_delta_rad,
        endpoint_delta_tolerance_rad=endpoint_delta_tolerance_rad,
        collision_check=collision_check,
    )
