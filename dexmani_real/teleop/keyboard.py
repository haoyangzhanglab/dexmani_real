"""键盘笛卡尔点动与回零。"""

import logging

from dexmani_real.control.constraints import MotionConstraints
from dexmani_real.control.homing import HomePlanner
from dexmani_real.control.ik import IKSolver
from dexmani_real.robot.commands import OperationCancelled
from dexmani_real.robot.system import Robot
from dexmani_real.runtime.home import home_robot
from dexmani_real.runtime.jog import CartesianJog
from dexmani_real.runtime.keyboard import Keyboard
from dexmani_real.runtime.lifecycle import close_runtime
from dexmani_real.runtime.preflight import (
    jog_bounds,
    validate_robot_config,
)
from dexmani_real.runtime.state import RunEndReason, SessionState, device_failure_reason
from dexmani_real.utils.log import configure_logging
from dexmani_real.utils.table_plane import load_table_plane
from dexmani_real.utils.timing import Rate

logger = logging.getLogger(__name__)


class KeyboardTeleop:
    def __init__(self, runtime, *, no_hand):
        self.runtime = runtime
        self.no_hand = no_hand

    def run(self):
        configure_logging()
        validate_robot_config(self.runtime)
        self.table_plane_abcd = (
            load_table_plane(self.runtime.environment.table)
            if self.runtime.environment.table.enabled else None
        )
        self.bounds = jog_bounds(self.runtime)
        if not self.runtime.hand.enabled and not self.no_hand:
            raise ValueError("hand-disabled operation requires --no-hand")

        self.runtime_state = SessionState()
        self.robot = Robot(
            self.runtime.arm, self.runtime.hand, run_signal=self.runtime_state.run_signal,
            stop_signal=self.runtime_state.stop_signal,
            emergency_signal=self.runtime_state.emergency_signal,
            signal_lock=self.runtime_state.motion_lock,
        )

        self.keys = Keyboard(
            estop_callback=self.runtime_state.request_emergency_stop,
            bindings={'q': self.runtime_state.request_quit},
        )
        self.jog_config = self.runtime.keyboard_teleop
        clean = False
        try:
            self.start()
            clean = self.run_control_loop()
        except KeyboardInterrupt:
            self.runtime_state.request_emergency_stop()
            raise
        except OperationCancelled as exc:
            logger.info("Keyboard teleop cancelled: %s", exc)
            clean = bool(self.runtime_state.quit_requested)
        except Exception as exc:
            self.runtime_state.mark_error()
            self.runtime_state.revoke_motion(
                reason=device_failure_reason(exc) or RunEndReason.RUNTIME_FAILURE,
            )
            raise
        finally:
            shutdown_clean = close_runtime(
                self.runtime_state, self.robot, keyboard=self.keys
            )
        return int(not clean or not shutdown_clean)

    def start(self):
        constraints = MotionConstraints(self.runtime.arm, self.runtime.environment)
        solver = IKSolver(
            constraints,
            position_tolerance=self.jog_config.ik_max_pose_error_pos_m,
            rotation_tolerance=self.jog_config.ik_max_pose_error_rot_rad,
        )
        self.jog = CartesianJog(solver, self.jog_config, self.bounds)
        self.home_planner = HomePlanner(
            self.runtime.environment,
            self.runtime.arm,
            table_plane_abcd=self.table_plane_abcd,
        )
        self.keys.start()
        if not self.keys.check_health():
            raise RuntimeError("keyboard stop listener unavailable")
        self.robot.check_services = lambda: self.keys.check_health()
        self.robot.connect()
        self.robot.check()
        self.runtime_state.arm()
        self.jog.prepare(self.robot, self.runtime_state)
        print("WASD/arrows and IJKL: jog; R: planned home; Q: exit; ESC: emergency stop")
        print(
            "Release jog keys to brake and hold; release all jog keys after a rejected target"
        )
        self.rate = Rate(self.jog_config.control_hz)

    def run_control_loop(self):
        while self.runtime_state.running and not self.runtime_state.quit_requested:
            self.rate.wait(service=self.robot.service_idle)
            self.robot.service_feedback()
            events = self.keys.poll(timeout=0)
            if (
                self.runtime_state.estop_request
                or self.runtime_state.error_state
                or not self.keys.check_health()
            ):
                break
            if self.runtime_state.quit_requested or self.keys.is_pressed("q"):
                return True

            if "r" in events:
                self.run_home()
                continue
            self.jog.step(self.robot, self.keys, self.runtime_state, self.runtime)
        return bool(self.runtime_state.quit_requested)

    def run_home(self):
        logger.info("Returning home...")
        self.jog.stop(self.robot, self.runtime_state)
        home_result = home_robot(
            self.runtime_state, self.robot, self.home_planner,
            cancel_requested=lambda: (
                bool(self.runtime_state.estop_request or self.runtime_state.quit_requested)
                or not self.keys.check_health()
            ),
        )
        if not home_result.ok:
            if not home_result.interrupted:
                raise RuntimeError(f"HOME failed: {home_result.reason}")
            logger.info("HOME cancelled: %s", home_result.reason)
        self.keys.discard_events("r")
        self.rate.reset()
