"""键盘笛卡尔点动与回零。"""

import logging
import time

from dexmani_real.config.control import HomeConfigurationError
from dexmani_real.control.constraints import MotionConstraints
from dexmani_real.control.homing import HomePathRejected
from dexmani_real.control.ik import IKSolver
from dexmani_real.robot.commands import OperationCancelled, RobotCommand
from dexmani_real.robot.system import Robot
from dexmani_real.runtime.home import build_home_planner, home_robot
from dexmani_real.runtime.jog import CartesianJog
from dexmani_real.runtime.keyboard import Keyboard
from dexmani_real.runtime.lifecycle import close_runtime
from dexmani_real.runtime.preflight import (
    jog_bounds,
    validate_robot_config,
)
from dexmani_real.runtime.state import RunEndReason, SessionState, device_failure_reason
from dexmani_real.utils.log import configure_logging
from dexmani_real.utils.timing import next_tick_ns, sampling_clock_ns

logger = logging.getLogger(__name__)


class KeyboardTeleop:
    def __init__(self, runtime):
        self.runtime = runtime

    def run(self):
        configure_logging()
        validate_robot_config(self.runtime)
        self.home_planner = None
        self.bounds = jog_bounds(self.runtime)
        self.runtime_state = SessionState()
        self.robot = Robot(
            self.runtime.arm, self.runtime.hand, run_signal=self.runtime_state.run_signal,
            stop_signal=self.runtime_state.stop_signal,
            emergency_signal=self.runtime_state.emergency_signal,
        )

        self.keys = Keyboard(
            get_run_id=lambda: self.runtime_state.run_id,
            estop_callback=self.runtime_state.request_emergency_stop,
            bindings={'q': self.runtime_state.request_quit},
        )
        self.jog_config = self.runtime.keyboard_teleop
        clean = False
        shutdown_clean = False
        try:
            try:
                self.start()
                clean = self.run_control_loop()
                self.robot.check()
            except OperationCancelled as exc:
                self.robot.check()
                logger.info("Keyboard teleop cancelled: %s", exc)
                clean = bool(self.runtime_state.quit_requested)
        except KeyboardInterrupt:
            self.runtime_state.request_emergency_stop()
            raise
        except Exception as exc:
            self.runtime_state.mark_error()
            self.runtime_state.revoke_motion(
                reason=device_failure_reason(exc) or RunEndReason.RUNTIME_FAILURE,
            )
            raise
        finally:
            try:
                close_runtime(self.runtime_state, self.robot, keyboard=self.keys)
            except BaseException:
                logger.exception("keyboard runtime cleanup failed")
            else:
                shutdown_clean = True
        return int(
            not clean or not shutdown_clean
            or self.runtime_state.error_state or self.runtime_state.estop_request
        )

    def start(self):
        constraints = MotionConstraints(self.runtime.arm, self.runtime.environment)
        solver = IKSolver(
            constraints,
            position_tolerance=self.jog_config.ik_max_pose_error_pos_m,
            rotation_tolerance=self.jog_config.ik_max_pose_error_rot_rad,
        )
        self.jog = CartesianJog(solver, self.jog_config, self.bounds)
        self.keys.start()
        if not self.keys.check_health():
            raise RuntimeError("keyboard stop listener unavailable")
        self.robot.check_services = lambda: self.keys.check_health()
        self.robot.connect()
        self.robot.check()
        self.prepare_jog(self.runtime_state.arm())
        print("WASD/arrows and IJKL: jog; R: planned home; Q: exit; ESC: emergency stop")
        print(
            "Jog is ready. Release jog keys to brake and hold. "
            "Rejected targets brake and hold; valid targets resume motion."
        )
        self.period_ns, _ = sampling_clock_ns(self.jog_config.control_hz)
        self.tick_due_ns = time.monotonic_ns() + self.period_ns

    def prepare_jog(self, run_id):
        self.jog.clear_reference()
        self.robot.prepare_streaming(run_id)
        with self.runtime_state.motion_lock:
            if self.runtime_state.run_id != run_id or self.runtime_state.begin_motion() is None:
                raise OperationCancelled("jog cancelled during preparation")
        self.robot.send_action(
            RobotCommand(run_id, arm_hold=True),
            expires_ns=time.monotonic_ns() + int(self.jog_config.target_max_age_s * 1e9),
        )
        logger.info("Keyboard jog ready")

    def run_control_loop(self):
        while not self.runtime_state.quit_requested:
            while time.monotonic_ns() < self.tick_due_ns:
                self.robot.service_idle()
                remaining_ns = self.tick_due_ns - time.monotonic_ns()
                if remaining_ns > 0:
                    time.sleep(min(0.005, remaining_ns / 1e9))
            now_ns = time.monotonic_ns()
            self.tick_due_ns += ((now_ns - self.tick_due_ns) // self.period_ns) * self.period_ns
            self.robot.service_feedback()
            events = self.keys.poll(timeout=0)
            if (
                self.runtime_state.estop_request
                or self.runtime_state.error_state
            ):
                break
            if self.runtime_state.quit_requested:
                return True

            home_event = next((event for event in events if event.key == "r"), None)
            if home_event is not None:
                self.run_home(home_event.run_id)
                continue
            self.jog.step(
                self.robot, self.keys, self.runtime_state, self.runtime,
                tick_end_ns=self.tick_due_ns + self.period_ns,
            )
            self.tick_due_ns = next_tick_ns(self.tick_due_ns, self.period_ns, time.monotonic_ns())
        return bool(self.runtime_state.quit_requested)

    def run_home(self, expected_run_id):
        with self.runtime_state.motion_lock:
            if (
                expected_run_id != self.runtime_state.run_id
                or not self.runtime_state.requests_allow_motion()
            ):
                logger.info("Ignored revoked HOME intent")
                return
            self.runtime_state.revoke_motion()
            run_id = self.runtime_state.run_id
        logger.info("Returning home...")

        def cancelled():
            self.robot.check()
            return not self.runtime_state.requests_allow_motion()

        try:
            home_robot(
                self.runtime_state, self.robot, self.get_home_planner,
                run_id=run_id,
                cancel_requested=cancelled,
            )
        except (HomeConfigurationError, HomePathRejected) as exc:
            self.robot.check()
            logger.warning("本次 HOME 未完成（手部可能已回位）：%s", exc)
        else:
            logger.info("HOME completed")
        self.prepare_jog(run_id)
        self.keys.discard_events("r")
        self.tick_due_ns = time.monotonic_ns() + self.period_ns

    def get_home_planner(self):
        if self.home_planner is None:
            self.home_planner = build_home_planner(self.runtime.environment, self.runtime.arm)
        return self.home_planner
