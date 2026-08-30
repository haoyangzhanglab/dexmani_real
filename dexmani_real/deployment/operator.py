"""Operator keyboard control for learned-policy deployment (B/S/H/Q/ESC).

Runs as a daemon thread in the Main process.  B/S/Q/ESC translate to the shared
request flags the coordinator and supervisor already consume (``start_request``,
``stop_request``, ``quit_requested``, ``estop_request``). When a caller owns a
home lifecycle, H orchestrates a collision-checked return-home — stop the run
if RUNNING, hand home, then arm home — using a Main-process planner that mirrors
the replay collision setup (hand-dof + table + static boxes). Physical policy
profiles supply that lifecycle; shadow keeps H disabled. The keyboard owns
the emergency-stop latch:
ESC (or a dead listener) sets ``estop_request`` regardless of thread state, so
e-stop never depends on this loop being scheduled.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from dexmani_real.config.runtime import ArmLoopConfig, ResolvedRuntimeConfig
from dexmani_real.control.arm_home import ArmHomeConfig, execute_arm_home
from dexmani_real.control.hand_home import publish_hand_home_and_wait_applied
from dexmani_real.deployment.config import DeploymentConfig
from dexmani_real.ipc.channels import RuntimeChannels
from dexmani_real.planning import (
    Pose,
    TeleopProfile,
    XArm7MotionPlanner,
    XArm7PlannerConfig,
)
from dexmani_real.robot_spec import (
    XARM7_XHAND_COLLISION_URDF_PATH,
    XARM7_XHAND_SRDF_PATH,
)
from dexmani_real.runtime.safety import SafetyState, StopRequest, revoke_motion
from dexmani_real.teleop.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_POLL_S = 0.05
_HOME_WAIT_ARMED_TIMEOUT_S = 5.0


def build_home_planner(runtime: ResolvedRuntimeConfig) -> XArm7MotionPlanner:
    """Construct the collision-checked home planner (mirrors replay setup).

    The coordinator's own planner only carries workspace bounds; a safe
    return-home needs the full collision model (hand-dof + table + static
    boxes), so the Main process builds its own.
    """
    policy = runtime.policy
    workspace = np.array(
        [
            [policy.workspace.x_min, policy.workspace.x_max],
            [policy.workspace.y_min, policy.workspace.y_max],
            [policy.workspace.z_min, policy.workspace.z_max],
        ],
        dtype=np.float64,
    )
    return XArm7MotionPlanner(
        XArm7PlannerConfig(
            urdf_path=str(XARM7_XHAND_COLLISION_URDF_PATH),
            srdf_path=str(XARM7_XHAND_SRDF_PATH),
            base_pose_world=Pose(p=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0])),
            workspace_bounds=workspace,
        ),
        teleop_profile=TeleopProfile(
            max_pose_error_pos_m=float(policy.ik_max_pose_error_pos_m),
            max_pose_error_rot_rad=float(policy.ik_max_pose_error_rot_rad),
        ),
        hand_dof=True,
        static_boxes=tuple(runtime.environment.static_boxes),
        table=runtime.environment.table,
    )


def _stop_to_armed(
    shared: RuntimeChannels,
    *,
    abort_requested,
) -> bool:
    """Stop a RUNNING run and wait for the coordinator to return to ARMED."""
    if int(shared.safety_state.value) != int(SafetyState.RUNNING):
        return int(shared.safety_state.value) == int(SafetyState.ARMED)
    shared.stop_request.value = int(StopRequest.OPERATOR)
    deadline = time.monotonic() + _HOME_WAIT_ARMED_TIMEOUT_S
    while time.monotonic() < deadline:
        if abort_requested():
            return False
        if int(shared.safety_state.value) == int(SafetyState.ARMED):
            return True
        time.sleep(_POLL_S)
    logger.warning("operator: coordinator did not reach ARMED for home")
    return False


def _home(
    shared: RuntimeChannels,
    runtime: ResolvedRuntimeConfig,
    deployment: DeploymentConfig,
    planner: XArm7MotionPlanner,
    *,
    abort_requested,
) -> bool:
    """Stop the run, home hand + arm, and report full-sequence completion."""
    if not _stop_to_armed(shared, abort_requested=abort_requested):
        return False
    if abort_requested():
        return False
    arm_config = ArmLoopConfig.from_runtime(runtime)

    if deployment.hand_enabled:
        hand_home = np.deg2rad(np.asarray(runtime.hand.home_qpos_deg, dtype=np.float64))
        accepted = publish_hand_home_and_wait_applied(
            shared,
            hand_home,
            command_lower_rad=np.asarray(runtime.hand.qpos_min_rad, dtype=np.float64),
            command_upper_rad=np.asarray(runtime.hand.qpos_max_rad, dtype=np.float64),
            mechanical_lower_rad=np.asarray(
                runtime.hand.mechanical_qpos_min_rad, dtype=np.float64
            ),
            mechanical_upper_rad=np.asarray(
                runtime.hand.mechanical_qpos_max_rad, dtype=np.float64
            ),
            hand_feedback_max_age_s=float(runtime.safety.heartbeat_timeouts["hand"]),
            timeout_s=float(runtime.hand.home_command_ack_timeout_s),
            heartbeat=False,
            check_is_running=True,
            verbose=True,
            abort_requested=abort_requested,
        )
        if not accepted:
            logger.warning("operator: hand home not accepted; arm home skipped")
            return False
        planner.set_hand_qpos(hand_home)

    result = execute_arm_home(
        shared,
        np.asarray(arm_config.home_qpos, dtype=np.float64),
        planner=planner,
        config=ArmHomeConfig.from_runtime(runtime, publish_policy_heartbeat=False),
        table_z_surface_m=float(runtime.arm.table_z_surface_m),
        # Only the physical e-stop path may latch ESTOP. Ordinary shutdown,
        # faults, and quit requests are already observed by execute_arm_home.
        estop_requested=lambda: bool(shared.estop_request.value),
        progress=lambda message: print(f"  {message}", flush=True),
    )
    return result.succeeded


def run_operator_control(
    shared: RuntimeChannels,
    runtime: ResolvedRuntimeConfig,
    deployment: DeploymentConfig,
    planner: XArm7MotionPlanner | None,
    *,
    stop_event: threading.Event,
    execution_mode: str,
) -> None:
    """Keyboard thread target: map operator keys to shared flags / home.

    B -> ``start_request``, S -> ``stop_request``, Q -> ``quit_requested``,
    ESC -> ``estop_request``. H is enabled only when a caller supplies a home
    planner. The thread exits when *stop_event*
    is set, when the runtime stops, or after a terminal Q/ESC.
    """
    if execution_mode not in {"shadow", "execute", "task"}:
        raise ValueError("execution_mode must be 'shadow', 'execute', or 'task'")
    keyboard = KeyboardHandler(
        estop_callback=lambda: setattr(shared.estop_request, "value", True)
    )
    try:
        keyboard.start()
    except Exception:
        # Without the e-stop keyboard the deployment must not run: fail closed
        # so the supervisor observes a sticky fault and shuts down.
        logger.error(
            "operator: keyboard failed to start; latching error_state", exc_info=True
        )
        shared.error_state.value = True
        return
    try:
        home_attempted = False
        while not stop_event.is_set() and shared.is_running.value:
            if keyboard.estop_latched or not keyboard.healthy:
                shared.estop_request.value = True
                return
            for signal in keyboard.poll(timeout=_POLL_S):
                if signal is ControlSignal.BEGIN:
                    shared.start_request.value = True
                elif signal is ControlSignal.STOP:
                    shared.stop_request.value = int(StopRequest.OPERATOR)
                elif signal is ControlSignal.HOME:
                    if planner is None:
                        logger.warning("operator: H is disabled in policy deployment")
                        continue
                    if home_attempted:
                        logger.warning(
                            "operator: ignored H after the physical home sequence "
                            "was already attempted in this process"
                        )
                        continue
                    home_attempted = True
                    shared.physical_home_completed.value = False
                    completed = _home(
                        shared,
                        runtime,
                        deployment,
                        planner,
                        abort_requested=lambda: bool(
                            stop_event.is_set()
                            or not shared.is_running.value
                            or shared.quit_requested.value
                            or shared.error_state.value
                            or shared.estop_request.value
                        ),
                    )
                    shared.physical_home_completed.value = bool(completed)
                    if completed:
                        logger.info("operator: physical home sequence completed")
                    else:
                        logger.warning(
                            "operator: physical home sequence did not complete; "
                            "B remains disabled for this process"
                        )
                    # HOME blocks while hand/arm homing completes. Discard only
                    # HOME events accumulated during that interval so one
                    # operator key press cannot trigger repeated home commands;
                    # preserve B/S/Q/ESC events for normal handling.
                    keyboard.drain_signal(ControlSignal.HOME)
                elif signal is ControlSignal.QUIT:
                    if not revoke_motion(shared, SafetyState.ARMED) and int(
                        shared.safety_state.value
                    ) != int(SafetyState.FAULT):
                        shared.error_state.value = True
                    shared.quit_requested.value = True
                    return
                elif signal is ControlSignal.EMERGENCY_STOP:
                    shared.estop_request.value = True
                    return
    finally:
        keyboard.stop()
