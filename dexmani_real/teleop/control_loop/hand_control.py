"""Hand command generation, observation caching, and retargeter state helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dexmani_real.robot.model import HAND_JOINT_SHAPE
from dexmani_real.teleop.retargeting.retargeter import (
    DexPilotHandRetargeter,
    TAGHandRetargeter,
)
from dexmani_real.utils.log import ThrottledWarner

_retarget_fail_warn = ThrottledWarner()


@dataclass
class HandRetargetObservationCache:
    """Reuse one success or failure per VR sample; clear on pause/re-anchor."""

    observation_id: int | None = None
    target_qpos: np.ndarray | None = None

    def reset(self) -> None:
        self.observation_id = None
        self.target_qpos = None


def compute_hand_command(
    retargeter: TAGHandRetargeter | DexPilotHandRetargeter | None,
    vr_frame: dict | None,
    observation_cache: HandRetargetObservationCache,
) -> np.ndarray | None:
    """Compute at most one hand solve per VR ring observation.

    Failure returns None. Cache both outcomes so repeated input does not
    advance the stateful solver or retry a failed solve.
    """
    if retargeter is None or vr_frame is None:
        return None

    landmarks = vr_frame.get("landmarks")
    if landmarks is None:
        return None

    observation_id = int(vr_frame.get("ring_sequence", 0))
    if observation_id <= 0:
        _retarget_fail_warn(
            "Hand retargeting: VR frame has invalid ring_sequence=%d", observation_id
        )
        return None

    if observation_cache.observation_id == observation_id:
        cached = observation_cache.target_qpos
        return None if cached is None else cached.copy()

    # Claim each observation once; failed solves are not retried on later ticks.
    observation_cache.observation_id = observation_id
    observation_cache.target_qpos = None

    target = retargeter.retarget(landmarks)
    if target is None:
        _retarget_fail_warn("Hand retargeting: retargeter.retarget() returned None")
        return None
    target_arr = np.asarray(target, dtype=np.float64)
    if target_arr.shape != HAND_JOINT_SHAPE or not np.all(np.isfinite(target_arr)):
        raise ValueError(
            "retargeter.retarget() must return a finite hand target with shape "
            f"{HAND_JOINT_SHAPE}, got {target_arr.shape}"
        )
    observation_cache.target_qpos = target_arr.copy()
    return target_arr


def reset_hand_retargeter(
    retargeter: TAGHandRetargeter | DexPilotHandRetargeter | None,
    hand_qpos: np.ndarray | None = None,
) -> None:
    """Reset hand retargeter state for a clean teleop start.

    Seeds SLSQP warm-start from actual hardware pose so the first
    retarget() call converges from near-optimum instead of the neutral midpoint.
    The teleop owner must clear its observation cache before retargeting resumes
    with this reset backend.
    """
    if retargeter is not None:
        retargeter.reset(initial_qpos=hand_qpos)
