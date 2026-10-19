"""Keyboard Cartesian jogging with persistent command targets."""

import logging

import numpy as np

from dexmani_real.runtime.state import RuntimeState, RunEndReason, SafetyState
from dexmani_real.planning.admission import TargetAdmission
from dexmani_real.planning.kinematics.ik import IKFailureKind, make_online_ik_solver
from dexmani_real.robot.home import build_home_planner, home_robot
from dexmani_real.robot.commands import RobotCommand, DispatchError
from dexmani_real.robot.robot import DexManiRobot
from dexmani_real.runtime.observation import feedback_deadline_ns, read_observation
from dexmani_real.runtime.operator_input import KeyboardInput
from dexmani_real.runtime.processes import shutdown_local_runtime
from dexmani_real.config.experiment import validate_keyboard_workspace
from dexmani_real.teleop.jog import compute_cartesian_jog_delta, propose_cartesian_jog_pose
from dexmani_real.utils.timing import LoopRate
from dexmani_real.utils.log import configure_logging

logger = logging.getLogger(__name__)


class KeyboardTeleopSession:
    """Own jog input, accepted command references and local robot resources."""

    def __init__(self, runtime, *, no_hand):
        self.runtime = runtime
        self.no_hand = no_hand

    def run(self):
        configure_logging()
        from dexmani_real.config.experiment import resolve_runtime_table, validate_robot_config

        validate_robot_config(self.runtime)
        self.runtime = resolve_runtime_table(self.runtime)
        self.workspace = validate_keyboard_workspace(self.runtime)
        if not self.runtime.control.hand_enabled and not self.no_hand:
            raise ValueError("hand-disabled operation requires --no-hand")
        self.runtime_state = RuntimeState()
        self.robot = DexManiRobot(self.runtime_state, self.runtime)

        self.keys = KeyboardInput(
            quit_callback=self.runtime_state.request_quit,
            capture_commands=False,
            estop_callback=self.runtime_state.request_emergency_stop,
            jog_change_callback=self.runtime_state.revoke_motion,
        )
        self.cfg = self.runtime.keyboard_teleop
        self.home_down = False
        self.blocked_until_release = False
        # Keep the operator target independent of tracking lag and IK pose residuals.
        self.command_qpos = None
        self.command_pose = None
        clean = False
        try:
            self.solver = make_online_ik_solver(
                self.runtime,
                max_pose_error_pos_m=self.cfg.ik_max_pose_error_pos_m,
                max_pose_error_rot_rad=self.cfg.ik_max_pose_error_rot_rad,
            )
            if not self.runtime.control.hand_enabled:
                self.solver.collision_model.set_hand_qpos(
                    np.deg2rad(self.runtime.hand.home_qpos_deg)
                )
            self.admission = TargetAdmission(
                self.runtime,
                self.solver.collision_model,
                dynamic_rejection=self.solver.dynamic_rejection,
            )
            self.home_planner = build_home_planner(self.runtime)
            self.robot.connect()
            self.runtime_state.arm()
            self.keys.start()
            if not self.keys.healthy:
                raise RuntimeError("keyboard stop listener unavailable")
            self.robot.check_services = lambda: self.keys.healthy
            print("WASD/arrows and IJKL: jog; R: planned home; Q: exit; ESC: emergency stop")
            self.rate = LoopRate(self.cfg.control_hz, label="keyboard_teleop")
            while self.runtime_state.running and not self.runtime_state.quit_requested:
                self.rate.wait()
                if (
                    self.runtime_state.estop_request
                    or self.runtime_state.error_state
                    or not self.keys.healthy
                ):
                    break
                if self.runtime_state.quit_requested or self.keys.is_pressed("q"):
                    clean = True
                    break
                pressed = self.keys.is_pressed("r")
                if pressed and not self.home_down:
                    self.command_qpos = None
                    self.command_pose = None
                    self.runtime_state.revoke_motion()
                    self.robot.stop()
                    home_result = home_robot(
                        self.runtime_state,
                        self.runtime,
                        self.home_planner,
                        robot=self.robot,
                        abort_requested=lambda: (
                            bool(
                                self.runtime_state.estop_request
                                or self.runtime_state.quit_requested
                            )
                            or not self.keys.healthy
                        ),
                    )
                    if not home_result.ok:
                        if not home_result.interrupted:
                            raise RuntimeError(f"HOME failed: {home_result.reason}")
                        logger.info("HOME cancelled: %s", home_result.reason)
                    self.rate.reset()
                    self.home_down = pressed
                    continue
                self.home_down = pressed
                pressed_keys, generation = self.keys.snapshot()
                dx, drpy = compute_cartesian_jog_delta(
                    pressed_keys, self.cfg.delta_pos_m, self.cfg.delta_rpy_rad
                )
                moving = np.any(dx) or np.any(drpy)
                if not moving:
                    self.blocked_until_release = False
                if not moving or pressed or self.blocked_until_release:
                    if self.robot.stop_required:
                        self.runtime_state.revoke_motion()
                        self.robot.stop()
                    self.command_qpos = self.command_pose = None
                    self.robot.service_idle()
                    continue
                safety_state = self.runtime_state.safety_state
                epoch = self.runtime_state.run_id
                if moving and not pressed and safety_state == SafetyState.ARMED:
                    preparation_generation = generation
                    self.command_qpos = self.command_pose = None
                    if self.robot.stop_required:
                        self.robot.stop()
                    try:
                        self.robot.prepare_streaming(epoch)
                    except DispatchError as exc:
                        if not exc.revoked:
                            raise
                        self.robot.stop()
                        self.command_qpos = self.command_pose = None
                        continue
                    pressed_keys, generation = self.keys.snapshot()
                    if generation != preparation_generation:
                        continue
                    dx, drpy = compute_cartesian_jog_delta(
                        pressed_keys, self.cfg.delta_pos_m, self.cfg.delta_rpy_rad
                    )
                    if not (np.any(dx) or np.any(drpy)) or "r" in pressed_keys:
                        continue
                row = read_observation(
                    self.runtime_state,
                    self.runtime,
                    self.robot,
                    require_hand=self.runtime.control.hand_enabled,
                )
                if row is None:
                    self.command_qpos = None
                    self.command_pose = None
                    if self.robot.stop_required:
                        self.runtime_state.revoke_motion()
                        self.robot.stop()
                    continue
                measured_qpos = row.arm["qpos"][0]
                baseline_qpos = self.command_qpos
                baseline_pose = self.command_pose
                if baseline_qpos is None:
                    baseline_qpos = measured_qpos.copy()
                    baseline_pose = self.solver.kin.compute_eef_pose_world(measured_qpos)
                proposed_pose, _, changed = propose_cartesian_jog_pose(
                    baseline_pose,
                    dx,
                    drpy,
                    self.workspace[:, 0] + self.cfg.workspace_command_margin_m,
                    self.workspace[:, 1] - self.cfg.workspace_command_margin_m,
                )
                if not changed:
                    self.blocked_until_release = True
                    continue
                if row.hand is not None:
                    self.solver.collision_model.set_hand_qpos(row.hand["qpos"][0])
                result = self.solver.solve(
                    proposed_pose,
                    measured_qpos,
                    baseline_qpos,
                )
                if result.failure_kind == IKFailureKind.INVALID_OUTPUT:
                    raise RuntimeError(f"online IK technical failure: {result.reason}")
                if not result.success:
                    self.blocked_until_release = True
                    continue
                collision_hand = (
                    row.hand["qpos"][0]
                    if row.hand is not None
                    else np.deg2rad(self.runtime.hand.home_qpos_deg)
                )
                rejection = self.admission.rejection(
                    result.qpos, measured_qpos, baseline_qpos, collision_hand_qpos=collision_hand
                )
                if rejection is not None:
                    self.blocked_until_release = True
                    continue
                if self.keys.snapshot()[1] != generation:
                    continue
                if safety_state == SafetyState.ARMED:
                    with self.runtime_state.motion_lock:
                        # A revoke during IK must also fence the first jog proposal.
                        if self.runtime_state.run_id != epoch:
                            self.command_qpos = None
                            self.command_pose = None
                            continue
                        if not self.runtime_state.begin_motion():
                            break
                        epoch = self.runtime_state.run_id
                target = result.qpos
                try:
                    self.robot.send_action(
                        RobotCommand(epoch, target),
                        valid_until_ns=feedback_deadline_ns(row, self.runtime),
                        execute_until_ns=row.observation_timestamp_ns
                        + int(self.runtime.control.vr_mapping.stale_threshold_s * 1e9),
                    )
                except DispatchError as exc:
                    if not exc.revoked:
                        raise
                    logger.info("keyboard motion interrupted: dispatch=%s", exc.result)
                    self.runtime_state.revoke_motion(expected_run_id=epoch)
                    self.robot.stop()
                    self.command_qpos = self.command_pose = None
                    continue
                self.command_pose = proposed_pose
                self.command_qpos = target.copy()
            clean = clean or bool(self.runtime_state.quit_requested)
        except KeyboardInterrupt:
            self.runtime_state.request_emergency_stop()
            raise
        except Exception as exc:
            self.runtime_state.mark_error()
            self.runtime_state.revoke_motion(
                reason=(
                    RunEndReason.HARDWARE_FAULT
                    if isinstance(exc, DispatchError)
                    else RunEndReason.POLICY_FAILURE
                ),
            )
            raise
        finally:
            shutdown_clean = shutdown_local_runtime(
                self.runtime_state, self.robot, keyboard=self.keys
            )
        return int(not clean or not shutdown_clean)
