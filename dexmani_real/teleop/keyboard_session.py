"""Keyboard Cartesian jogging with persistent command targets."""

import logging

import numpy as np

from dexmani_real.config.experiment import (
    resolve_runtime_table,
    validate_keyboard_workspace,
    validate_robot_config,
)
from dexmani_real.planning.target import TargetChecker
from dexmani_real.planning.kinematics.ik import OnlineIKConfig, make_online_ik_solver
from dexmani_real.robot.commands import OperationCancelled, RobotError
from dexmani_real.robot.home import build_home_planner, home_robot
from dexmani_real.robot.robot import DexManiRobot
from dexmani_real.runtime.observation import (
    feedback_deadline_ns,
    read_observation, observation_issues, ObservationNeeds, service_feedback,
    control_deadline_ns,
)
from dexmani_real.runtime.operator_input import KeyboardInput
from dexmani_real.runtime.processes import shutdown_local_runtime
from dexmani_real.runtime.state import RunEndReason, RuntimeState, SafetyState
from dexmani_real.runtime.jog import JogController, jog_motion_requested
from dexmani_real.utils.log import configure_logging
from dexmani_real.utils.timing import LoopRate

logger = logging.getLogger(__name__)


class KeyboardTeleopSession:
    """管理 jog 输入、反馈和机器人资源；命令参考由 JogController 持有。"""

    def __init__(self, runtime, *, no_hand):
        self.runtime = runtime
        self.no_hand = no_hand

    def run(self):
        configure_logging()
        validate_robot_config(self.runtime)
        self.runtime = resolve_runtime_table(self.runtime)
        self.workspace = validate_keyboard_workspace(self.runtime)
        if not self.runtime.hand.enabled and not self.no_hand:
            raise ValueError("hand-disabled operation requires --no-hand")

        self.runtime_state = RuntimeState()
        self.robot = DexManiRobot(
            self.runtime, run_signal=self.runtime_state.run_signal,
            stop_signal=self.runtime_state.stop_signal,
            emergency_signal=self.runtime_state.emergency_signal,
            signal_lock=self.runtime_state.motion_lock,
        )

        self.keys = KeyboardInput(
            quit_callback=self.runtime_state.request_quit,
            capture_commands=False,
            estop_callback=self.runtime_state.request_emergency_stop,
        )
        self.jog_config = self.runtime.keyboard_teleop
        self.home_key_down = False
        clean = False
        try:
            self.start()
            clean = self.run_control_loop()
        except KeyboardInterrupt:
            self.runtime_state.request_emergency_stop()
            raise
        except Exception as exc:
            self.runtime_state.mark_error()
            self.runtime_state.revoke_motion(
                reason=(
                    RunEndReason.HARDWARE_FAULT
                    if isinstance(exc, RobotError)
                    else RunEndReason.POLICY_FAILURE
                ),
            )
            raise
        finally:
            shutdown_clean = shutdown_local_runtime(
                self.runtime_state, self.robot, keyboard=self.keys
            )
        return int(not clean or not shutdown_clean)

    def start(self):
        solver = make_online_ik_solver(
            OnlineIKConfig(
                max_pose_error_pos_m=self.jog_config.ik_max_pose_error_pos_m,
                max_pose_error_rot_rad=self.jog_config.ik_max_pose_error_rot_rad,
                operational_joint_lower_rad=tuple(self.runtime.arm.joint_limit_lower),
                operational_joint_upper_rad=tuple(self.runtime.arm.joint_limit_upper),
            ),
            static_boxes=self.runtime.environment.static_boxes,
        )
        target_checker = TargetChecker(
            solver.collision_model,
            joint_lower_rad=self.runtime.arm.joint_limit_lower,
            joint_upper_rad=self.runtime.arm.joint_limit_upper,
            workspace=self.runtime.environment.workspace.as_array(),
            workspace_margin_m=self.runtime.environment.eef_workspace_margin_m,
        )
        self.jog = JogController(solver, target_checker, self.jog_config, self.workspace)
        self.home_planner = build_home_planner(self.runtime)
        self.keys.start()
        if not self.keys.healthy:
            raise RuntimeError("keyboard stop listener unavailable")
        self.robot.check_services = lambda: self.keys.healthy
        self.robot.input_signal = self.keys.generation_signal
        self.robot.connect()
        self.robot.check()
        self.runtime_state.arm()
        print("WASD/arrows and IJKL: jog; R: planned home; Q: exit; ESC: emergency stop")
        print("Jog key changes stop and re-prepare motion; release all jog keys after a rejected target")
        self.rate = LoopRate(self.jog_config.control_hz, label="keyboard_teleop")

    def run_control_loop(self):
        while self.runtime_state.running and not self.runtime_state.quit_requested:
            self.rate.wait()
            if (
                self.runtime_state.estop_request
                or self.runtime_state.error_state
                or not self.keys.healthy
            ):
                break
            if self.runtime_state.quit_requested or self.keys.is_pressed("q"):
                return True

            home_pressed = self.keys.is_pressed("r")
            if self.handle_home_key(home_pressed):
                continue
            self.motion_tick(home_pressed)
        return bool(self.runtime_state.quit_requested)

    def handle_home_key(self, home_pressed):
        if home_pressed and not self.home_key_down:
            self.jog.clear_reference()
            self.runtime_state.revoke_motion()
            self.robot.stop()
            home_result = home_robot(
                self.runtime_state,
                self.runtime,
                self.home_planner,
                robot=self.robot,
                abort_requested=lambda: (
                    bool(self.runtime_state.estop_request or self.runtime_state.quit_requested)
                    or not self.keys.healthy
                ),
            )
            if not home_result.ok:
                if not home_result.interrupted:
                    raise RuntimeError(f"HOME failed: {home_result.reason}")
                logger.info("HOME cancelled: %s", home_result.reason)
            self.rate.reset()
            self.home_key_down = home_pressed
            return True
        self.home_key_down = home_pressed
        return False

    def motion_tick(self, home_pressed):
        pressed_keys, generation = self.keys.snapshot()
        moving = jog_motion_requested(pressed_keys)
        self.jog.release_if_idle(pressed_keys)
        if not moving or home_pressed or self.jog.blocked_until_release:
            if self.robot.stop_required:
                self.runtime_state.revoke_motion()
                self.robot.stop()
            self.jog.clear_reference()
            self.robot.service_idle()
            return

        with self.runtime_state.motion_lock:
            safety_state, epoch = self.runtime_state.safety_state, self.runtime_state.run_id
        if safety_state == SafetyState.ARMED:
            self.jog.clear_reference()
            if self.robot.stop_required:
                self.robot.stop()
            try:
                self.robot.prepare_streaming(epoch, service=lambda: service_feedback(self.robot, self.runtime))
            except OperationCancelled:
                self.robot.stop()
                self.jog.clear_reference()
                return
            if self.keys.snapshot()[1] != generation:
                return

        row = read_observation(None, self.robot)
        issues = observation_issues(
            row, self.runtime, ObservationNeeds(), require_hand=self.runtime.hand.enabled,
        )
        if not issues:
            self.robot.arm.update_health(
                health_until_ns=control_deadline_ns(row, self.runtime, ObservationNeeds()),
            )
        if issues:
            self.jog.clear_reference()
            if self.robot.stop_required:
                self.runtime_state.revoke_motion()
                self.robot.stop()
            return

        proposed_pose, _, result = self.jog.propose(
            pressed_keys, row.arm["qpos"][0],
            hand_qpos=(
                row.hand["qpos"][0] if row.hand is not None
                else np.deg2rad(self.runtime.hand.home_qpos_deg)
            ),
        )
        if self.jog.blocked_until_release:
            return

        try:
            self.jog.submit(
                proposed_pose, result.qpos, self.keys, generation,
                self.runtime_state, self.robot, epoch,
                valid_until_ns=feedback_deadline_ns(row, self.runtime),
                execute_until_ns=row.observation_timestamp_ns
                + int(self.jog_config.target_max_age_s * 1e9),
            )
        except (OperationCancelled, TimeoutError) as exc:
            logger.info("jog interrupted: %s", exc)
            self.robot.stop()
            self.jog.clear_reference()
            return
