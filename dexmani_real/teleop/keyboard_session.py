"""Keyboard Cartesian jogging using latest feedback and absolute targets."""
import multiprocessing as mp
import os
import time
import numpy as np
from scipy.spatial.transform import Rotation
from dexmani_real.ipc.channels import RuntimeChannels, RuntimeChannelsConfig
from dexmani_real.robot.arm_worker import arm_loop
from dexmani_real.robot.hand_worker import hand_loop
from dexmani_real.robot.arm_homing import build_policy_home_planner, home_policy_robot
from dexmani_real.robot.hand_homing import home_hand
from dexmani_real.robot.commands import RobotCommand, publish_command
from dexmani_real.robot.projection import project_arm_command
from dexmani_real.planning import Pose, OnlineIKConfig, XArm7MotionPlanner
from dexmani_real.runtime.observation import read_observation
from dexmani_real.runtime.operator_input import KeyboardInput
from dexmani_real.runtime.safety import SafetyState, begin_motion, revoke_motion, require_transition
from dexmani_real.runtime.supervisor import start_processes, check_processes
from dexmani_real.runtime.processes import shutdown_processes_verified
from dexmani_real.teleop.jog import compute_cartesian_jog_delta


def run_keyboard_experiment(runtime, *, no_hand):
    if not runtime.policy.hand_enabled and not no_hand:
        raise ValueError("hand-disabled operation requires --no-hand")
    ctx = mp.get_context("spawn")
    shared = RuntimeChannels.create(prefix=f"keyboard_{os.getpid()}",
        config=RuntimeChannelsConfig.from_runtime(runtime), mp_context=ctx)
    processes = [ctx.Process(name="arm", target=arm_loop, args=(shared, runtime.arm))]
    if runtime.policy.hand_enabled:
        processes.append(ctx.Process(name="hand", target=hand_loop, args=(shared, runtime.hand, runtime.policy.hand_disconnect_timeout_s)))
    started = []
    keys = KeyboardInput(suppress_echo=True, capture_commands=False,
        estop_callback=lambda: setattr(shared.estop_request, "value", True))
    cfg = runtime.keyboard_teleop
    planner = XArm7MotionPlanner.create_default(teleop_profile=OnlineIKConfig(
        max_pose_error_pos_m=cfg.ik_max_pose_error_pos_m,
        max_pose_error_rot_rad=cfg.ik_max_pose_error_rot_rad))
    workspace = runtime.policy.workspace.as_array()
    home_down = False
    clean = False
    try:
        start_processes(shared, processes, runtime.safety.readiness_timeouts_s, started)
        require_transition(shared, SafetyState.ARMED)
        home_result = home_hand(shared, runtime)
        if not home_result.ok:
            raise RuntimeError(f"hand home failed: {home_result.reason}")
        keys.start()
        print("WASD/arrows and IJKL: jog; R: planned home; Q: exit; ESC: emergency stop")
        while shared.is_running.value and check_processes(shared, started):
            if shared.estop_request.value or shared.error_state.value or not keys.healthy:
                break
            if keys.is_pressed("q"):
                clean = True
                break
            pressed = keys.is_pressed("r")
            if pressed and not home_down:
                revoke_motion(shared)
                home_policy_robot(shared, runtime, build_policy_home_planner(runtime),
                    abort_requested=lambda: bool(shared.estop_request.value) or not keys.healthy)
            home_down = pressed
            row = read_observation(shared, runtime, require_hand=runtime.policy.hand_enabled)
            if row is None:
                if int(shared.safety_state.value) == int(SafetyState.RUNNING):
                    revoke_motion(shared)
                time.sleep(0.02)
                continue
            dx, drpy = compute_cartesian_jog_delta(keys, cfg.delta_pos_m, cfg.delta_rpy_rad)
            moving = np.any(dx) or np.any(drpy)
            if not moving or pressed:
                if int(shared.safety_state.value) == int(SafetyState.RUNNING):
                    revoke_motion(shared)
            else:
                if int(shared.safety_state.value) == int(SafetyState.ARMED) and not begin_motion(shared):
                    break
                epoch = int(shared.run_id.value)
                qpos = row.arm["qpos"][0]
                pose = planner.kin.compute_eef_pose_world(qpos)
                pos = np.clip(pose.p+dx, workspace[:,0]+cfg.workspace_command_margin_m,
                              workspace[:,1]-cfg.workspace_command_margin_m)
                quat = (Rotation.from_euler("xyz", drpy)*Rotation.from_quat(pose.q, scalar_first=True)).as_quat(scalar_first=True)
                result = planner.solve_teleop_ik(Pose(p=pos, q=quat), qpos, qpos)
                if result.success:
                    target = project_arm_command(result.qpos, qpos,
                        joint_lower_rad=runtime.arm.joint_limit_lower, joint_upper_rad=runtime.arm.joint_limit_upper)
                    publish_command(shared, RobotCommand(epoch, target))
            time.sleep(1/cfg.control_hz)
    finally:
        keys.quiesce()
        report = shutdown_processes_verified(shared, started, graceful_timeout_s=runtime.safety.shutdown_timeout_s,
                                    disarm_if_clean=clean)
        keys.stop()
    return int(not clean or not report.shared_closed or shared.error_state.value or shared.estop_request.value
               or any(x.exitcode != 0 or x.escalation != "graceful" for x in report.exits))
