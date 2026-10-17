"""Keyboard Cartesian jogging with persistent command targets."""

import logging

import numpy as np

from dexmani_real.runtime.state import RuntimeState, RunEndReason, SafetyState
from dexmani_real.planning import XArm7MotionPlanner
from dexmani_real.planning.kinematics.ik import IKFailureKind, make_online_ik_config
from dexmani_real.robot.home import build_home_planner, home_robot
from dexmani_real.robot.commands import RobotCommand, DispatchError
from dexmani_real.robot.robot import DexManiRobot
from dexmani_real.runtime.observation import feedback_deadline_ns, read_observation
from dexmani_real.runtime.operator_input import KeyboardInput
from dexmani_real.runtime.processes import shutdown_local_runtime
from dexmani_real.teleop.config import validate_keyboard_workspace
from dexmani_real.teleop.jog import compute_cartesian_jog_delta, propose_cartesian_jog_pose

from dexmani_real.utils.rate import LoopRate
from dexmani_real.utils.log import configure_logging

logger = logging.getLogger(__name__)


def run_keyboard_experiment(runtime, *, no_hand):
    configure_logging()
    from dexmani_real.config.experiment import resolve_runtime_table, validate_robot_config

    validate_robot_config(runtime)
    runtime = resolve_runtime_table(runtime)
    validate_keyboard_workspace(runtime)
    if not runtime.policy.hand_enabled and not no_hand:
        raise ValueError("hand-disabled operation requires --no-hand")
    shared = RuntimeState()
    robot = DexManiRobot(shared, runtime)

    def request_quit():
        shared.revoke_motion(reason=RunEndReason.QUIT)
        shared.quit_requested = True

    keys = KeyboardInput(
        quit_callback=request_quit,
        capture_commands=False,
        estop_callback=shared.request_emergency_stop,
    )
    cfg = runtime.keyboard_teleop
    workspace = runtime.policy.workspace.as_array()
    home_down = False
    blocked_until_release = False
    # Keep the operator target independent of tracking lag and IK pose residuals.
    command_qpos = None
    command_pose = None
    clean = False
    try:
        planner = XArm7MotionPlanner(
            online_ik_config=make_online_ik_config(
                runtime,
                max_pose_error_pos_m=cfg.ik_max_pose_error_pos_m,
                max_pose_error_rot_rad=cfg.ik_max_pose_error_rot_rad,
            ),
        )
        if not runtime.policy.hand_enabled:
            planner.set_hand_qpos(np.deg2rad(runtime.hand.home_qpos_deg))
        home_planner = build_home_planner(runtime)
        robot.connect()
        shared.arm()
        keys.start()
        if not keys.healthy:
            raise RuntimeError("keyboard stop listener unavailable")
        robot.check_services = lambda: keys.healthy
        print("WASD/arrows and IJKL: jog; R: planned home; Q: exit; ESC: emergency stop")
        rate = LoopRate(cfg.control_hz, label="keyboard_teleop")
        while shared.running and not shared.quit_requested:
            rate.wait()
            if shared.estop_request or shared.error_state or not keys.healthy:
                break
            if shared.quit_requested or keys.is_pressed("q"):
                clean = True
                break
            pressed = keys.is_pressed("r")
            if pressed and not home_down:
                command_qpos = None
                command_pose = None
                shared.revoke_motion()
                robot.stop()
                home_result = home_robot(
                    shared,
                    runtime,
                    home_planner,
                    robot=robot,
                    abort_requested=lambda: (
                        bool(shared.estop_request or shared.quit_requested)
                        or not keys.healthy
                    ),
                )
                if not home_result.ok:
                    if not home_result.interrupted:
                        raise RuntimeError(f"HOME failed: {home_result.reason}")
                    logger.info("HOME cancelled: %s", home_result.reason)
                rate.reset()
                home_down = pressed
                continue
            home_down = pressed
            dx, drpy = compute_cartesian_jog_delta(keys, cfg.delta_pos_m, cfg.delta_rpy_rad)
            moving = np.any(dx) or np.any(drpy)
            if not moving:
                blocked_until_release = False
            if not moving or pressed or blocked_until_release:
                if robot.stop_required:
                    shared.revoke_motion()
                    robot.stop()
                command_qpos = command_pose = None
                robot.service_idle()
                continue
            safety_state = shared.safety_state
            epoch = shared.run_id
            if moving and not pressed and safety_state == SafetyState.ARMED:
                command_qpos = command_pose = None
                if robot.stop_required:
                    robot.stop()
                try:
                    robot.prepare_streaming(epoch)
                except DispatchError as exc:
                    if not exc.revoked:
                        raise
                    robot.stop()
                    command_qpos = command_pose = None
                    continue
            row = read_observation(shared, runtime, robot, require_hand=runtime.policy.hand_enabled)
            if row is None:
                command_qpos = None
                command_pose = None
                if robot.stop_required:
                    shared.revoke_motion()
                    robot.stop()
                continue
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
                blocked_until_release = True
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
                blocked_until_release = True
                continue
            if safety_state == SafetyState.ARMED:
                with shared.motion_lock:
                    # A revoke during IK must also fence the first jog proposal.
                    if shared.run_id != epoch:
                        command_qpos = None
                        command_pose = None
                        continue
                    if not shared.begin_motion():
                        break
                    epoch = shared.run_id
            target = result.qpos
            try:
                robot.send_action(
                    RobotCommand(epoch, target), valid_until_ns=feedback_deadline_ns(row, runtime),
                    execute_until_ns=row.observation_timestamp_ns + int(runtime.policy.vr_mapping.stale_threshold_s * 1e9),
                )
            except DispatchError as exc:
                if not exc.revoked:
                    raise
                logger.info("keyboard motion interrupted: dispatch=%s", exc.result)
                shared.revoke_motion(expected_run_id=epoch)
                robot.stop()
                command_qpos = command_pose = None
                continue
            command_pose = proposed_pose
            command_qpos = target.copy()
        clean = clean or bool(shared.quit_requested)
    except KeyboardInterrupt:
        shared.request_emergency_stop()
        raise
    except Exception as exc:
        shared.error_state = True
        shared.revoke_motion(
            reason=RunEndReason.HARDWARE_FAULT if isinstance(exc, DispatchError)
            else RunEndReason.POLICY_FAILURE,
        )
        raise
    finally:
        shutdown_clean = shutdown_local_runtime(
            shared, robot, keyboard=keys
        )
    return int(not clean or not shutdown_clean)
