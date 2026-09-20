"""Single-owner soft command projection shared by robot-command producers.

One place owns the soft command-continuity bound
(``max_servo_command_jump_rad``): :func:`project_arm_command` canonicalizes,
clips to operational joint limits, and bounds the command-step delta exactly
once, including the float64 round-off guard. Producers that deliberately
reject instead of clip (keyboard jog, calibration jog) use
:func:`validate_arm_command` at production time. Hardware workers keep only
the hard physical boundary validation at the SDK fence and never re-reject
the same soft threshold — no two adjacent helpers validate one constraint.

Pure computation only: no device, IPC, or file side effects live here, and
the real hard limits are never widened.
"""

from __future__ import annotations

__all__ = [
    "ARM_COMMAND_JUMP_REJECTION",
    "ArmClipReport",
    "project_arm_command",
    "project_arm_command_reported",
    "project_hand_command",
    "validate_arm_command",
]

from dataclasses import dataclass

import numpy as np

from dexmani_real.planning.paths import wrap_nearest_equivalent

ARM_COMMAND_JUMP_REJECTION = "command jump limit violation"


@dataclass(frozen=True)
class ArmClipReport:
    """Whether one arm projection actually truncated a command step.

    The joint index and the pre-clip magnitude are reported so the producer
    that owns the action identity can print one visible ``[CLIP]`` line per
    really truncated action without re-deriving the clip condition.
    """

    clipped: bool = False
    joint: int = -1
    max_abs_delta_rad: float = 0.0


def project_arm_command(
    target_arm_qpos: np.ndarray,
    reference_arm_qpos: np.ndarray,
    *,
    joint_lower_rad: np.ndarray | tuple[float, ...],
    joint_upper_rad: np.ndarray | tuple[float, ...],
    max_command_jump_rad: float,
) -> np.ndarray:
    """Project one finite arm endpoint exactly once against its reference.

    Canonicalization (nearest 2π-equivalent inside hardware limits), the
    operational joint-limit clip, the soft command-jump clip, and the
    float64 round-off guard all happen here — and only here. The continuity
    ``reference_arm_qpos`` must itself be limit-valid; anything else is a
    producer contract violation, not a recoverable miss. Raises
    ``ValueError`` on non-finite inputs or a broken projection invariant.

    Producers that must report a real truncation visibly use
    :func:`project_arm_command_reported` instead; the projection itself is
    identical.
    """
    return project_arm_command_reported(
        target_arm_qpos,
        reference_arm_qpos,
        joint_lower_rad=joint_lower_rad,
        joint_upper_rad=joint_upper_rad,
        max_command_jump_rad=max_command_jump_rad,
    )[0]


def project_arm_command_reported(
    target_arm_qpos: np.ndarray,
    reference_arm_qpos: np.ndarray,
    *,
    joint_lower_rad: np.ndarray | tuple[float, ...],
    joint_upper_rad: np.ndarray | tuple[float, ...],
    max_command_jump_rad: float,
) -> tuple[np.ndarray, ArmClipReport]:
    """Project one arm endpoint and report whether the soft clip bit."""
    arm = np.asarray(target_arm_qpos, dtype=np.float64)
    reference = np.asarray(reference_arm_qpos, dtype=np.float64)
    for values, shape in ((arm, (7,)), (reference, (7,))):
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError("arm projection requires finite (7,) targets/reference")
    lower = np.asarray(joint_lower_rad, dtype=np.float64)
    upper = np.asarray(joint_upper_rad, dtype=np.float64)
    if np.any(reference < lower) or np.any(reference > upper):
        raise ValueError("arm continuity reference is outside joint limits")
    canonical = wrap_nearest_equivalent(arm, reference, lower, upper)
    arm = np.clip(canonical, lower, upper)
    delta = arm - reference
    limit = float(max_command_jump_rad)
    clipped = np.abs(delta) > limit
    report = ArmClipReport()
    if bool(clipped.any()):
        # The pre-clip delta is the truncation the producer must report.
        joint = int(np.argmax(np.abs(delta)))
        report = ArmClipReport(
            clipped=True,
            joint=joint,
            max_abs_delta_rad=float(np.abs(delta[joint])),
        )
    arm[clipped] = reference[clipped] + np.clip(delta[clipped], -limit, limit)
    # Addition/subtraction can round beyond the strict float64 bound.
    outside = np.abs(arm - reference) > limit
    arm[outside] = np.nextafter(arm[outside], reference[outside])
    if (
        not np.all(np.isfinite(arm))
        or np.any(arm < lower)
        or np.any(arm > upper)
        or np.any(np.abs(arm - reference) > limit)
    ):
        raise ValueError("arm projection violated command invariants")
    return arm, report


def project_hand_command(
    target_hand_qpos: np.ndarray,
    *,
    qpos_min_rad: np.ndarray | tuple[float, ...],
    qpos_max_rad: np.ndarray | tuple[float, ...],
) -> np.ndarray:
    """Clip one finite hand endpoint into its operational command box."""
    hand = np.asarray(target_hand_qpos, dtype=np.float64)
    if hand.shape != (12,) or not np.all(np.isfinite(hand)):
        raise ValueError("hand projection requires a finite (12,) target")
    return np.clip(hand, np.asarray(qpos_min_rad), np.asarray(qpos_max_rad))


def validate_arm_command(
    target_qpos_rad: np.ndarray,
    reference_qpos_rad: np.ndarray,
    *,
    joint_lower_rad: np.ndarray,
    joint_upper_rad: np.ndarray,
    max_command_jump_rad: float,
) -> str | None:
    """Producer-side reject-style validation of one arm command.

    For producers whose interaction model rejects an oversized jog step
    instead of clipping it (operator must release and re-anchor). Returns
    the rejection detail or ``None``. This is the producer's single check;
    the worker's SDK-boundary validation covers only the hard limits.
    """
    target = np.asarray(target_qpos_rad, dtype=np.float64)
    if not np.all(np.isfinite(target)):
        return "non-finite target"
    if np.any(target < joint_lower_rad) or np.any(target > joint_upper_rad):
        return "joint limit violation"
    if np.any(
        np.abs(target - np.asarray(reference_qpos_rad, dtype=np.float64))
        > float(max_command_jump_rad)
    ):
        return ARM_COMMAND_JUMP_REJECTION
    return None
