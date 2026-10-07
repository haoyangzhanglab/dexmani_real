"""Keyboard Cartesian jogging with persistent command targets."""

import multiprocessing as mp
import os

import numpy as np

from dexmani_real.ipc.channels import RuntimeChannels, RuntimeChannelsConfig
from dexmani_real.planning import XArm7MotionPlanner
from dexmani_real.planning.kinematics.ik import IKFailureKind, make_online_ik_config
from dexmani_real.robot.arm_homing import build_home_planner, home_robot
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.robot.robot import DexManiRobot, DispatchError
from dexmani_real.runtime.observation import feedback_deadline_ns, read_observation
from dexmani_real.runtime.operator_input import KeyboardInput
from dexmani_real.runtime.processes import shutdown_local_runtime
from dexmani_real.runtime.safety import (
    RunEndReason,
    SafetyState,
    begin_motion,
    require_transition,
    revoke_motion,
    revoke_motion_if_run_id,
)
from dexmani_real.runtime.supervisor import RuntimeSupervisor
from dexmani_real.teleop.config import validate_keyboard_workspace
from dexmani_real.teleop.jog import compute_cartesian_jog_delta, propose_cartesian_jog_pose
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate import LoopRate

logger = get_logger(__name__)


def run_keyboard_experiment(runtime, *, no_hand):
    from dexmani_real.config.experiment import resolve_runtime_table, validate_robot_config

    validate_robot_config(runtime)
    runtime = resolve_runtime_table(runtime)
    cfg = runtime.keyboard_teleop
    validate_keyboard_workspace(runtime)
    if not runtime.policy.hand_enabled and not no_hand:
        raise ValueError("hand-disabled operation requires --no-hand")
    ctx = mp.get_context("spawn")
    shared = RuntimeChannels.create(
        prefix=f"keyboard_{os.getpid()}",
        config=RuntimeChannelsConfig.from_runtime(runtime),
        mp_context=ctx,
    )
    supervisor = RuntimeSupervisor(shared, runtime.safety.readiness_timeouts_s)
    robot = DexManiRobot(shared, runtime, check_services=supervisor.check)

    def request_quit():
        revoke_motion(shared, reason=RunEndReason.QUIT)
        shared.quit_requested.value = True

    keys = KeyboardInput(
        suppress_echo=True,
        quit_callback=request_quit,
        capture_commands=False,
        estop_callback=lambda: setattr(shared.estop_request, "value", True),
    )
    cfg = runtime.keyboard_teleop
    workspace = runtime.policy.workspace.as_array()
    home_down = False
    # Keep the operator target independent of tracking lag and IK pose residuals.
    command_qpos = None
    command_pose = None
    clean = False
    try:
        planner = XArm7MotionPlanner.create_default(
            online_ik_profile=make_online_ik_config(
                runtime,
                max_pose_error_pos_m=cfg.ik_max_pose_error_pos_m,
                max_pose_error_rot_rad=cfg.ik_max_pose_error_rot_rad,
            ),
        )
        if not runtime.policy.hand_enabled:
            planner.set_hand_qpos(np.deg2rad(runtime.hand.home_qpos_deg))
        home_planner = build_home_planner(runtime)
        robot.connect()
        require_transition(shared, SafetyState.ARMED)
        keys.start()
        if not keys.healthy:
            raise RuntimeError("keyboard stop listener unavailable")
        robot.check_services = lambda: supervisor.check() and keys.healthy
        print("WASD/arrows and IJKL: jog; R: planned home; Q: exit; ESC: emergency stop")
        rate = LoopRate(cfg.control_hz, label="keyboard_teleop", busy_wait=False)
        while shared.is_running.value and not shared.quit_requested.value and supervisor.check():
            rate.wait()
            if shared.estop_request.value or shared.error_state.value or not keys.healthy:
                break
            if shared.quit_requested.value or keys.is_pressed("q"):
                clean = True
                break
            pressed = keys.is_pressed("r")
            if pressed and not home_down:
                command_qpos = None
                command_pose = None
                revoke_motion(shared)
                robot.stop()
                home_robot(
                    shared,
                    runtime,
                    home_planner,
                    robot=robot,
                    abort_requested=lambda: (
                        bool(shared.estop_request.value or shared.quit_requested.value)
                        or not keys.healthy
                    ),
                )
                rate.reset()
            home_down = pressed
            row = read_observation(shared, runtime, robot, require_hand=runtime.policy.hand_enabled)
            if row is None:
                command_qpos = None
                command_pose = None
                if int(shared.safety_state.value) == int(SafetyState.RUNNING):
                    revoke_motion(shared)
                    robot.stop()
                continue
            dx, drpy = compute_cartesian_jog_delta(keys, cfg.delta_pos_m, cfg.delta_rpy_rad)
            moving = np.any(dx) or np.any(drpy)
            if not moving or pressed:
                # Idle keeps Mode-6 authority and its last dispatched endpoint alive.
                continue
            safety_state = int(shared.safety_state.value)
            epoch = int(shared.run_id.value)
            measured_qpos = row.arm["qpos"][0]
            baseline_qpos = command_qpos
            baseline_pose = command_pose
            if baseline_qpos is None:
                baseline_qpos = measured_qpos.copy()
                baseline_pose = planner.kin.compute_eef_pose_world(measured_qpos)
            proposed_pose, _, changed = propose_cartesian_jog_pose(
                baseline_pose,
                dx,
                drpy,
                workspace[:, 0] + cfg.workspace_command_margin_m,
                workspace[:, 1] - cfg.workspace_command_margin_m,
            )
            if not changed:
                continue
            if row.hand is not None:
                planner.set_hand_qpos(row.hand["qpos"][0])
            result = planner.solve_online_ik(
                proposed_pose,
                measured_qpos,
                baseline_qpos,
            )
            if result.failure_kind == IKFailureKind.INVALID_OUTPUT:
                raise RuntimeError(f"online IK technical failure: {result.reason}")
            if not result.success:
                continue
            if safety_state == int(SafetyState.ARMED):
                with shared.motion_lock:
                    # A revoke during IK must also fence the first jog proposal.
                    if int(shared.run_id.value) != epoch:
                        command_qpos = None
                        command_pose = None
                        continue
                    if not begin_motion(shared):
                        break
                    epoch = int(shared.run_id.value)
            target = result.qpos
            try:
                robot.send_action(
                    RobotCommand(epoch, target), valid_until_ns=feedback_deadline_ns(row, runtime)
                )
            except DispatchError as exc:
                if not exc.revoked:
                    raise
                logger.info("keyboard motion interrupted: dispatch=%s", exc.result)
                revoke_motion_if_run_id(shared, epoch)
                robot.stop()
                command_qpos = command_pose = None
                continue
            command_pose = proposed_pose
            command_qpos = target.copy()
        clean = clean or bool(shared.quit_requested.value)
    except KeyboardInterrupt:
        shared.estop_request.value = True
        raise
    except Exception as exc:
        shared.error_state.value = True
        revoke_motion(
            shared,
            reason=RunEndReason.HARDWARE_FAULT
            if isinstance(exc, DispatchError)
            else RunEndReason.POLICY_FAILURE,
        )
        raise
    finally:
        revoke_motion(shared, reason=RunEndReason.RUNTIME_SHUTDOWN)
        shutdown_clean = shutdown_local_runtime(
            robot, supervisor, keyboard=keys, timeout_s=runtime.safety.shutdown_timeout_s
        )
    return int(
        not clean or not shutdown_clean or shared.error_state.value or shared.estop_request.value
    )
