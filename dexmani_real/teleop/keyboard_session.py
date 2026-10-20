"""Keyboard Cartesian jogging with persistent command targets."""

import logging

import numpy as np

from dexmani_real.config.experiment import (
    resolve_runtime_table,
    validate_keyboard_workspace,
    validate_robot_config,
)
from dexmani_real.planning.admission import TargetAdmission
from dexmani_real.planning.kinematics.ik import IKFailureKind, make_online_ik_solver
from dexmani_real.robot.commands import DispatchError
from dexmani_real.robot.home import build_home_planner, home_robot
from dexmani_real.robot.robot import DexManiRobot
from dexmani_real.runtime.observation import feedback_deadline_ns, read_observation
from dexmani_real.runtime.operator_input import KeyboardInput
from dexmani_real.runtime.processes import shutdown_local_runtime
from dexmani_real.runtime.state import RunEndReason, RuntimeState, SafetyState
from dexmani_real.teleop.jog import (
    any_jog_key_held,
    compute_cartesian_jog_delta,
    jog_motion_requested,
    propose_cartesian_jog_pose,
    submit_jog_target,
)
from dexmani_real.utils.log import configure_logging
from dexmani_real.utils.timing import LoopRate

logger = logging.getLogger(__name__)


class KeyboardTeleopSession:
    """Own jog input, accepted command references and local robot resources."""

    def __init__(self, runtime, *, no_hand):
        self.runtime = runtime
        self.no_hand = no_hand

    def run(self):
        configure_logging()
        validate_robot_config(self.runtime)
        self.runtime = resolve_runtime_table(self.runtime)
        workspace = validate_keyboard_workspace(self.runtime)
        if not self.runtime.control.hand_enabled and not self.no_hand:
            raise ValueError("hand-disabled operation requires --no-hand")

        runtime_state = RuntimeState()
        robot = DexManiRobot(runtime_state, self.runtime)

        keys = KeyboardInput(
            quit_callback=runtime_state.request_quit,
            capture_commands=False,
            estop_callback=runtime_state.request_emergency_stop,
            jog_change_callback=runtime_state.jog_input_changed,
        )
        jog_config = self.runtime.keyboard_teleop
        home_down = False
        blocked_until_release = False
        # Keep the operator target independent of tracking lag and IK pose residuals.
        command_qpos = None
        command_pose = None
        clean = False
        try:
            solver = make_online_ik_solver(
                self.runtime,
                max_pose_error_pos_m=jog_config.ik_max_pose_error_pos_m,
                max_pose_error_rot_rad=jog_config.ik_max_pose_error_rot_rad,
            )
            if not self.runtime.control.hand_enabled:
                solver.collision_model.set_hand_qpos(np.deg2rad(self.runtime.hand.home_qpos_deg))
            admission = TargetAdmission(
                self.runtime,
                solver.collision_model,
                dynamic_rejection=solver.dynamic_rejection,
            )
            home_planner = build_home_planner(self.runtime)
            robot.connect()
            runtime_state.arm()
            keys.start()
            if not keys.healthy:
                raise RuntimeError("keyboard stop listener unavailable")
            robot.check_services = lambda: keys.healthy
            print("WASD/arrows and IJKL: jog; R: planned home; Q: exit; ESC: emergency stop")
            print("Jog key changes stop and re-prepare motion; release all jog keys after a rejected target")
            rate = LoopRate(jog_config.control_hz, label="keyboard_teleop")

            while runtime_state.running and not runtime_state.quit_requested:
                rate.wait()
                if (
                    runtime_state.estop_request
                    or runtime_state.error_state
                    or not keys.healthy
                ):
                    break
                if runtime_state.quit_requested or keys.is_pressed("q"):
                    clean = True
                    break

                home_pressed = keys.is_pressed("r")
                if home_pressed and not home_down:
                    command_qpos = None
                    command_pose = None
                    runtime_state.revoke_motion()
                    robot.stop()
                    home_result = home_robot(
                        runtime_state,
                        self.runtime,
                        home_planner,
                        robot=robot,
                        abort_requested=lambda: (
                            bool(runtime_state.estop_request or runtime_state.quit_requested)
                            or not keys.healthy
                        ),
                    )
                    if not home_result.ok:
                        if not home_result.interrupted:
                            raise RuntimeError(f"HOME failed: {home_result.reason}")
                        logger.info("HOME cancelled: %s", home_result.reason)
                    rate.reset()
                    home_down = home_pressed
                    continue
                home_down = home_pressed

                pressed_keys, generation = keys.snapshot()
                moving = jog_motion_requested(pressed_keys)
                if not any_jog_key_held(pressed_keys):
                    blocked_until_release = False
                if not moving or home_pressed or blocked_until_release:
                    if robot.stop_required:
                        runtime_state.revoke_motion()
                        robot.stop()
                    command_qpos = command_pose = None
                    robot.service_idle()
                    continue

                with runtime_state.motion_lock:
                    safety_state, epoch = runtime_state.safety_state, runtime_state.run_id
                if safety_state == SafetyState.ARMED:
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
                    if keys.snapshot()[1] != generation:
                        continue

                row = read_observation(
                    runtime_state,
                    self.runtime,
                    robot,
                    require_hand=self.runtime.control.hand_enabled,
                )
                if row is None:
                    command_qpos = None
                    command_pose = None
                    if robot.stop_required:
                        runtime_state.revoke_motion()
                        robot.stop()
                    continue

                measured_qpos = row.arm["qpos"][0]
                baseline_qpos = command_qpos
                baseline_pose = command_pose
                if baseline_qpos is None:
                    baseline_qpos = measured_qpos.copy()
                    baseline_pose = solver.kin.compute_eef_pose_base(measured_qpos)
                dx, drpy = compute_cartesian_jog_delta(
                    pressed_keys, jog_config.delta_pos_m, jog_config.delta_rpy_rad
                )
                proposed_pose, _, changed = propose_cartesian_jog_pose(
                    baseline_pose,
                    dx,
                    drpy,
                    workspace[:, 0] + jog_config.workspace_command_margin_m,
                    workspace[:, 1] - jog_config.workspace_command_margin_m,
                )
                if not changed:
                    blocked_until_release = True
                    continue

                if row.hand is not None:
                    solver.collision_model.set_hand_qpos(row.hand["qpos"][0])
                result = solver.solve(proposed_pose, measured_qpos, baseline_qpos)
                if result.failure_kind == IKFailureKind.INVALID_OUTPUT:
                    raise RuntimeError(f"online IK technical failure: {result.reason}")
                if not result.success:
                    blocked_until_release = True
                    continue

                collision_hand = (
                    row.hand["qpos"][0]
                    if row.hand is not None
                    else np.deg2rad(self.runtime.hand.home_qpos_deg)
                )
                rejection = admission.rejection(
                    result.qpos, measured_qpos, baseline_qpos, collision_hand_qpos=collision_hand
                )
                if rejection is not None:
                    blocked_until_release = True
                    continue

                target = result.qpos
                next_qpos = target.copy()

                def commit_reference():
                    nonlocal command_qpos, command_pose
                    command_pose, command_qpos = proposed_pose, next_qpos

                try:
                    submit_jog_target(
                        keys, generation, runtime_state, robot, target, epoch,
                        valid_until_ns=feedback_deadline_ns(row, self.runtime),
                        execute_until_ns=row.observation_timestamp_ns
                        + int(self.runtime.control.vr_mapping.stale_threshold_s * 1e9),
                        commit_reference=commit_reference,
                    )
                except DispatchError as exc:
                    if not exc.revoked:
                        raise
                    logger.info("keyboard motion interrupted: dispatch=%s", exc.result)
                    robot.stop()
                    command_qpos = command_pose = None
                    continue

            clean = clean or bool(runtime_state.quit_requested)
        except KeyboardInterrupt:
            runtime_state.request_emergency_stop()
            raise
        except Exception as exc:
            runtime_state.mark_error()
            runtime_state.revoke_motion(
                reason=(
                    RunEndReason.HARDWARE_FAULT
                    if isinstance(exc, DispatchError)
                    else RunEndReason.POLICY_FAILURE
                ),
            )
            raise
        finally:
            shutdown_clean = shutdown_local_runtime(runtime_state, robot, keyboard=keys)
        return int(not clean or not shutdown_clean)
