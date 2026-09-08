"""Hand-first homing orchestration for teleoperation."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from dexmani_real.control.arm_homing import ArmHomeConfig, execute_arm_home
from dexmani_real.control.hand_homing import publish_hand_home_and_wait_accepted
from dexmani_real.ipc.causal import read_arm_state_causal
from dexmani_real.ipc.channels import RuntimeChannels
from dexmani_real.planning import XArm7MotionPlanner
from dexmani_real.teleop.control_loop.vr_mapping import VRWristMapper
from dexmani_real.teleop.config import TeleopConfig
from dexmani_real.teleop.control_loop.hand_control import reset_hand_retargeter
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


def do_configured_teleop_home(
    shared: RuntimeChannels,
    config: TeleopConfig,
    *,
    hand_available: bool,
    prev_hand_qpos: np.ndarray,
    planner: XArm7MotionPlanner,
    audio: Any,
    estop_requested: Callable[[], bool],
    arm_mapper: VRWristMapper | None = None,
    hand_retargeter: Any = None,
) -> np.ndarray:
    """Apply hand-home, acknowledge its SDK send, then home the arm.

    If *arm_mapper* and *hand_retargeter* are both provided, clears EMA
    state and re-seeds retargeter before homing (active-teleop H path).
    Post-teleop callers pass ``None`` for both — the state is already cleared.
    Hand execution convergence is intentionally not inspected.
    """
    runtime = config.runtime
    hand_home_qpos = np.deg2rad(
        np.asarray(runtime.hand.home_qpos_deg, dtype=np.float64)
    )

    if arm_mapper is not None:
        arm_mapper.clear()
    if hand_retargeter is not None:
        reset_hand_retargeter(hand_retargeter)

    if hand_available and not shared.error_state.value:
        hand_accepted = publish_hand_home_and_wait_accepted(
            shared,
            np.asarray(hand_home_qpos, dtype=np.float64),
            command_lower_rad=np.asarray(runtime.hand.qpos_min_rad, dtype=np.float64),
            command_upper_rad=np.asarray(runtime.hand.qpos_max_rad, dtype=np.float64),
            mechanical_lower_rad=np.asarray(
                runtime.hand.mechanical_qpos_min_rad, dtype=np.float64
            ),
            mechanical_upper_rad=np.asarray(
                runtime.hand.mechanical_qpos_max_rad, dtype=np.float64
            ),
            hand_feedback_max_age_s=float(runtime.safety.heartbeat_timeouts["hand"]),
            timeout_s=runtime.hand.home_command_ack_timeout_s,
            heartbeat=True,
            abort_requested=estop_requested,
        )
        if not hand_accepted:
            logger.warning(
                "arm home cancelled: hand-home command was not accepted by the worker/SDK"
            )
            return prev_hand_qpos
        prev_hand_qpos = np.asarray(hand_home_qpos, dtype=np.float64).copy()
        planner.set_hand_qpos(prev_hand_qpos)
    elif not runtime.policy.hand_enabled:
        prev_hand_qpos = np.asarray(hand_home_qpos, dtype=np.float64).copy()
        planner.set_hand_qpos(prev_hand_qpos)
        print("  hand: using explicitly acknowledged fixed-home geometry", flush=True)
    else:
        print(
            "  hand: not connected — arm home cancelled (hand pose unknown)", flush=True
        )
        return prev_hand_qpos

    _arm_state = read_arm_state_causal(shared)
    if _arm_state is None:
        logger.warning("arm home cancelled: no current arm state")
        return prev_hand_qpos
    _state_age_s = (
        time.monotonic_ns() - int(_arm_state["source_monotonic_ns"][0])
    ) * 1e-9
    if (
        _state_age_s > runtime.arm.homing.state_max_age_s
        or not bool(_arm_state["connected"][0])
        or int(_arm_state["error_code"][0]) != 0
        or not np.all(np.isfinite(_arm_state["qpos"][0]))
    ):
        logger.warning(
            "arm home cancelled: arm state is stale or unhealthy (age=%.3fs)",
            _state_age_s,
        )
        return prev_hand_qpos
    arm_qpos = np.asarray(_arm_state["qpos"][0], dtype=np.float64).copy()
    _home_qpos = np.array(runtime.arm.home_qpos, dtype=np.float64)
    home_result = execute_arm_home(
        shared,
        _home_qpos,
        planner=planner,
        config=ArmHomeConfig.from_runtime(runtime, publish_policy_heartbeat=True),
        table_z_surface_m=runtime.arm.table_z_surface_m,
        current_qpos=arm_qpos,
        estop_requested=estop_requested,
        progress=lambda message: print(f"  {message}", flush=True),
    )
    if home_result.succeeded:
        # Keep the departure cue intact; AudioFeedback.queue() serializes this
        # completion cue after it instead of cancelling it mid-sentence.
        audio.queue("home_done")
        print("  arm: home reached", flush=True)
    else:
        logger.warning("arm home failed or was cancelled")

    return prev_hand_qpos
