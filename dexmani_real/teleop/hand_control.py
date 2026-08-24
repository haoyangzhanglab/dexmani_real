"""Hand command generation, observation caching, and retargeter state helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dexmani_real.control.publication import validate_hand_command_bounds
from dexmani_real.robot_spec import HAND_JOINT_SHAPE
from dexmani_real.teleop.retarget.facade import TAGHandRetargeter, XHandRetargeter
from dexmani_real.utils.log import ThrottledWarner, get_logger

logger = get_logger(__name__)


_retarget_fail_warn = ThrottledWarner()


@dataclass
class HandRetargetObservationCache:
    """Cache one solver result per VR ring observation.

    ``observation_id`` is the verified VR ring sequence, not an action id. The
    control grid may causally select that same immutable frame more than once.
    Both success and failure are remembered: a success reuses a copied solved
    endpoint, while a failure holds the caller's *current* previous command.
    This keeps shaping control-rate driven without advancing TAG/DexPilot
    temporal state or retrying a partial failure on duplicate input.

    Teleop clears this cache at command quiescence/reanchor boundaries before
    the reset solver is allowed to process another observation.
    """

    observation_id: int | None = None
    target_qpos: np.ndarray | None = None
    succeeded: bool = False

    def reset(self) -> None:
        self.observation_id = None
        self.target_qpos = None
        self.succeeded = False


def sanitize_hand_command(
    hand_cmd: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    mechanical_lower: np.ndarray,
    mechanical_upper: np.ndarray,
) -> np.ndarray:
    """Backstop validation of an already-clipped hand target; never clips.

    The VR-teleop caller clips the command into the operator-set anti-clogging
    command box (loop.py) before calling here, so the operational bound should
    never fire on that path.  This preflight remains as the *graceful-hold*
    backstop for well-formed shape, finite values, the rated mechanical
    envelope, and limit-array consistency: a violation raises so the loop marks
    ``hand_cmd_valid=False`` and holds arm + hand together, instead of letting
    ``SafetyGate.validate`` turn an out-of-limit command into a sticky fault.
    (Other coupled paths — keyboard, replay, calibrate, return-home — still rely
    on ``validate_hand_command_bounds`` inside ``publish_joint_targets`` to
    reject, not clip.)
    """
    return validate_hand_command_bounds(
        hand_cmd,
        lower,
        upper,
        mechanical_lower,
        mechanical_upper,
    )


def hand_ramp_frame_count(duration_s: float, control_hz: float) -> int:
    if not np.isfinite(duration_s) or duration_s < 0:
        raise ValueError("duration_s must be finite and >= 0")
    if not np.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("control_hz must be finite and > 0")
    return max(0, int(round(duration_s * control_hz)))


def smoothstep_hand_ramp(
    start: np.ndarray,
    target: np.ndarray,
    step_index: int,
    total_steps: int,
) -> np.ndarray:
    """Blend one startup sample; the last configured step reaches the target."""
    start_arr = np.asarray(start, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    if start_arr.shape != target_arr.shape:
        raise ValueError(f"ramp arrays must have matching shapes, got {start_arr.shape} and {target_arr.shape}")
    if total_steps <= 0:
        return target_arr.copy()
    if not 0 <= step_index < total_steps:
        raise ValueError(f"step_index={step_index} must be in [0, {total_steps})")
    progress = (step_index + 1) / total_steps
    smooth = progress * progress * (3.0 - 2.0 * progress)
    return start_arr + smooth * (target_arr - start_arr)


def get_raw_hand_command(
    retargeter: TAGHandRetargeter | XHandRetargeter | None,
    filtered_command: np.ndarray,
    retarget_ok: bool,
) -> np.ndarray:
    """Return the TAG optimizer output when available, otherwise the retargeter output."""
    fallback = np.asarray(filtered_command, dtype=np.float64).copy()
    if not retarget_ok or retargeter is None:
        return fallback
    raw = getattr(retargeter, "last_raw_qpos", None)
    if raw is None:
        return fallback
    raw_arr = np.asarray(raw, dtype=np.float64)
    return raw_arr.copy() if raw_arr.shape == HAND_JOINT_SHAPE and np.all(np.isfinite(raw_arr)) else fallback


def compute_hand_command(
    retargeter: TAGHandRetargeter | XHandRetargeter | None,
    vr_frame: dict | None,
    prev_hand_cmd: np.ndarray,
    hand_available: bool,
    observation_cache: HandRetargetObservationCache,
) -> tuple[np.ndarray, bool]:
    """Compute at most one hand solve per VR ring observation.

    A successful cache hit returns ``retarget_ok=True`` without calling the
    stateful backend again. A cached failure returns the current
    ``prev_hand_cmd`` with ``retarget_ok=False``. New observations are claimed
    before entering the backend so an exception or partial failure is never
    retried on a later control tick.
    """
    if not hand_available:
        return prev_hand_cmd.copy(), False

    if retargeter is None:
        return prev_hand_cmd.copy(), False

    if vr_frame is None:
        return prev_hand_cmd.copy(), False

    landmarks = vr_frame.get("landmarks")
    if landmarks is None:
        return prev_hand_cmd.copy(), False

    observation_id = int(vr_frame.get("ring_sequence", 0))
    if observation_id <= 0:
        _retarget_fail_warn(
            "Hand retargeting: VR frame has invalid ring_sequence=%d", observation_id
        )
        return prev_hand_cmd.copy(), False

    if observation_cache.observation_id == observation_id:
        cached = observation_cache.target_qpos
        if observation_cache.succeeded and cached is not None:
            return cached.copy(), True
        return prev_hand_cmd.copy(), False

    # Claim each observation once; failed solves are not retried on later ticks.
    observation_cache.observation_id = observation_id
    observation_cache.target_qpos = None
    observation_cache.succeeded = False

    try:
        target = retargeter.retarget(landmarks)  # validates shape + finiteness internally
        if target is None:
            _retarget_fail_warn("Hand retargeting: retargeter.retarget() returned None")
            return prev_hand_cmd.copy(), False
        target_arr = np.asarray(target, dtype=np.float64)
        if target_arr.shape == HAND_JOINT_SHAPE and np.all(
            np.isfinite(target_arr)
        ):
            observation_cache.target_qpos = target_arr.copy()
            observation_cache.succeeded = True
            return target_arr, True
        _retarget_fail_warn(
            "Hand retargeting: retargeter.retarget() returned invalid shape/values (%s)",
            target_arr.shape,
        )
    except Exception:
        logger.warning("Hand retargeting failed — holding position", exc_info=True)

    return prev_hand_cmd.copy(), False


def reset_hand_retargeter(
    retargeter: TAGHandRetargeter | XHandRetargeter | None,
    hand_qpos: np.ndarray | None = None,
) -> None:
    """Reset hand retargeter state for a clean teleop start.

    Seeds SLSQP warm-start from actual hardware pose so the first
    retarget() call converges from near-optimum instead of the neutral midpoint.
    The teleop owner must clear its observation cache before retargeting resumes
    with this reset backend.
    """
    if retargeter is not None:
        try:
            retargeter.reset(initial_qpos=hand_qpos)
        except Exception:
            logger.warning("Hand retargeter reset failed — previous optimizer state retained", exc_info=True)


def seed_hand_retargeter(
    retargeter: TAGHandRetargeter | XHandRetargeter | None,
    qpos: np.ndarray | None,
) -> np.ndarray | None:
    """Reset hand retargeter NLP warm-start from *qpos*.

    Returns a copy of *qpos* if valid (for seeding ``prev_hand_qpos``),
    else ``None``.
    """
    if qpos is not None and np.all(np.isfinite(qpos)):
        reset_hand_retargeter(retargeter, qpos.copy())
        return qpos.copy()
    reset_hand_retargeter(retargeter, None)
    return None
