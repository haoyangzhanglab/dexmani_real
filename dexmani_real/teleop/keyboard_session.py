"""Keyboard Cartesian jogging with persistent command targets."""

import multiprocessing as mp
import os

import numpy as np

from dexmani_real.ipc.channels import RuntimeChannels, RuntimeChannelsConfig
from dexmani_real.planning import XArm7MotionPlanner
from dexmani_real.planning.kinematics.ik import IKFailureKind, make_online_ik_config
from dexmani_real.robot.arm_homing import build_policy_home_planner, home_policy_robot
from dexmani_real.robot.arm_worker import run_arm_worker
from dexmani_real.robot.commands import RobotCommand, publish_command
from dexmani_real.robot.hand_homing import home_hand
from dexmani_real.robot.hand_worker import run_hand_worker
from dexmani_real.runtime.observation import read_observation
from dexmani_real.runtime.operator_input import KeyboardInput
from dexmani_real.runtime.safety import SafetyState, begin_motion, require_transition, revoke_motion
from dexmani_real.runtime.supervisor import RuntimeSupervisor
from dexmani_real.teleop.jog import compute_cartesian_jog_delta, propose_cartesian_jog_pose
from dexmani_real.utils.rate import LoopRate


def run_keyboard_experiment(runtime, *, no_hand):
    if not runtime.policy.hand_enabled and not no_hand:
        raise ValueError("hand-disabled operation requires --no-hand")
    ctx = mp.get_context("spawn")
    shared = RuntimeChannels.create(
        prefix=f"keyboard_{os.getpid()}",
        config=RuntimeChannelsConfig.from_runtime(runtime),
        mp_context=ctx,
    )
    processes = [ctx.Process(name="arm", target=run_arm_worker, args=(shared, runtime.arm))]
    if runtime.policy.hand_enabled:
        processes.append(
            ctx.Process(name="hand", target=run_hand_worker, args=(shared, runtime.hand))
        )
    supervisor = RuntimeSupervisor(shared, runtime.safety.readiness_timeouts_s)
    keys = KeyboardInput(
        suppress_echo=True,
        capture_commands=False,
        estop_callback=lambda: setattr(shared.estop_request, "value", True),
    )
    cfg = runtime.keyboard_teleop
    planner = XArm7MotionPlanner.create_default(
        online_ik_profile=make_online_ik_config(
            runtime,
            max_pose_error_pos_m=cfg.ik_max_pose_error_pos_m,
            max_pose_error_rot_rad=cfg.ik_max_pose_error_rot_rad,
        )
    )
    if not runtime.policy.hand_enabled:
        planner.set_hand_qpos(np.deg2rad(runtime.hand.home_qpos_deg))
    workspace = runtime.policy.workspace.as_array()
    home_down = False
    # Keep the operator target independent of tracking lag and IK pose residuals.
    command_qpos = None
    command_pose = None
    clean = False
    try:
        supervisor.start(processes)
        require_transition(shared, SafetyState.ARMED)
        home_result = home_hand(shared, runtime)
        if not home_result.ok:
            raise RuntimeError(f"hand home failed: {home_result.reason}")
        keys.start()
        print("WASD/arrows and IJKL: jog; R: planned home; Q: exit; ESC: emergency stop")
        rate = LoopRate(cfg.control_hz, label="keyboard_teleop", busy_wait=False)
        while shared.is_running.value and supervisor.check():
            rate.wait()
            if shared.estop_request.value or shared.error_state.value or not keys.healthy:
                break
            if keys.is_pressed("q"):
                clean = True
                break
            pressed = keys.is_pressed("r")
            if pressed and not home_down:
                command_qpos = None
                command_pose = None
                revoke_motion(shared)
                home_policy_robot(
                    shared,
                    runtime,
                    build_policy_home_planner(runtime),
                    abort_requested=lambda: bool(shared.estop_request.value) or not keys.healthy,
                )
                rate.reset()
            home_down = pressed
            row = read_observation(shared, runtime, require_hand=runtime.policy.hand_enabled)
            if row is None:
                command_qpos = None
                command_pose = None
                if int(shared.safety_state.value) == int(SafetyState.RUNNING):
                    revoke_motion(shared)
                continue
            dx, drpy = compute_cartesian_jog_delta(keys, cfg.delta_pos_m, cfg.delta_rpy_rad)
            moving = np.any(dx) or np.any(drpy)
            if not moving or pressed:
                # Idle keeps Mode-6 authority and its last published endpoint alive.
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
            if publish_command(shared, RobotCommand(epoch, target)):
                command_pose = proposed_pose
                command_qpos = target.copy()
            else:
                command_qpos = None
                command_pose = None
    finally:
        keys.quiesce()
        shutdown_clean = supervisor.shutdown(
            graceful_timeout_s=runtime.safety.shutdown_timeout_s,
        )
        keys.stop()
    return int(
        not clean or not shutdown_clean or shared.error_state.value or shared.estop_request.value
    )
