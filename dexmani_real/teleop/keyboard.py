"""键盘笛卡尔点动与回零。"""

import logging
import time

import numpy as np

from dexmani_real.control.ik import IKSolver
from dexmani_real.robot.commands import OperationCancelled, RobotCommand
from dexmani_real.robot.system import Robot
from dexmani_real.runtime.home import (
    HomeConfigurationError,
    HomePathRejected,
    home_robot,
)
from dexmani_real.runtime.jog import CartesianJog
from dexmani_real.runtime.keyboard import Keyboard
from dexmani_real.runtime.lifecycle import close_runtime
from dexmani_real.runtime.preflight import validate_robot_config
from dexmani_real.runtime.state import SessionState
from dexmani_real.runtime.teleop_timing import TeleopTiming
from dexmani_real.utils.log import configure_logging
from dexmani_real.utils.timing import next_tick_ns, sampling_clock_ns

logger = logging.getLogger(__name__)


class KeyboardTeleop:
    def __init__(self, runtime, *, timing_output=None):
        self.runtime = runtime
        self.timing_output = timing_output

    def run(self):
        configure_logging()
        validate_robot_config(self.runtime)
        config = self.runtime.keyboard_teleop
        config.validate()
        self.period_ns = sampling_clock_ns(config.control_hz)
        lease_ns = int(config.target_max_age_s * 1e9)
        if self.period_ns >= lease_ns:
            raise ValueError(
                f"keyboard_teleop.control_hz gives a {self.period_ns} ns period, which must be "
                f"shorter than the keyboard_teleop.target_max_age_s command lease ({lease_ns} ns)"
            )
        solver = IKSolver(
            self.runtime.arm, self.runtime.environment,
            target_speed=np.deg2rad(self.runtime.motion.ik_target_speed_deg_s),
        )
        workspace = solver.workspace
        margin = config.workspace_command_margin_m
        lower, upper = workspace[:, 0] + margin, workspace[:, 1] - margin
        if np.any(lower >= upper):
            raise ValueError("keyboard workspace command margin leaves no interior workspace")
        timing = (
            TeleopTiming(
                self.timing_output, mode="keyboard", control_hz=config.control_hz,
                arm=self.runtime.arm, motion=self.runtime.motion,
            ) if self.timing_output is not None else None
        )
        if timing is not None:
            timing.add(
                "config", delta_pos_m=config.delta_pos_m, delta_rpy_rad=config.delta_rpy_rad,
                target_max_age_s=config.target_max_age_s, hand_enabled=self.runtime.hand.enabled,
            )
        self.jog = CartesianJog(solver, config, (lower, upper), timing=timing)
        self.runtime_state = SessionState()
        self.robot = Robot(
            self.runtime.arm, self.runtime.hand, runtime_state=self.runtime_state,
            timing=timing,
        )

        self.keys = Keyboard(
            estop_callback=self.runtime_state.request_emergency_stop,
            bindings={'q': self.runtime_state.request_quit},
            timing=timing,
        )
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
        finally:
            try:
                close_runtime(self.runtime_state, self.robot, keyboard=self.keys)
            except BaseException:
                logger.exception("keyboard runtime cleanup failed")
            else:
                shutdown_clean = True
            if timing is not None:
                try:
                    timing.write()
                except Exception:
                    logger.exception("keyboard timing output failed")
                    shutdown_clean = False
        return int(
            not clean or not shutdown_clean
            or self.runtime_state.estop_request
        )

    def start(self):
        startup_epoch = self.runtime_state.cancel_epoch
        self.keys.start()
        if not self.keys.check_health():
            raise RuntimeError("keyboard stop listener unavailable")
        self.robot.check_services = self.check_services
        self.robot.connect()
        self.robot.check()
        if not (self.runtime_state.cancel_epoch == startup_epoch and self.runtime_state.requests_allow_motion()):
            raise OperationCancelled("startup cancelled")
        self.prepare_jog(startup_epoch)
        print(
            "WASD、方向键和 IJKL 用于点动。按 Q 退出，按 Esc 急停。"
            + ("按 R 先让灵巧手回零，再让机械臂回零。" if self.runtime.hand.enabled else "按 R 让机械臂回零。")
        )
        print(
            "点动就绪。松开所有点动键或点动目标被拒绝时，机械臂减速停稳并保持位置。"
            "关节反馈或控制参考失效时，程序停止机械臂。"
        )
        self.tick_due_ns = time.monotonic_ns() + self.period_ns

    def check_services(self):
        return self.keys.check_health()

    def prepare_jog(self, cancel_epoch):
        self.jog.clear_reference()
        self.robot.prepare_streaming(cancel_epoch)
        if not (self.runtime_state.cancel_epoch == cancel_epoch and self.runtime_state.requests_allow_motion() and not self.runtime_state.stop_signal.is_set()):
            raise OperationCancelled("jog cancelled during preparation")
        self.robot.send_action(
            RobotCommand(arm_hold=True),
            expires_ns=time.monotonic_ns() + int(self.runtime.keyboard_teleop.target_max_age_s * 1e9),
        )
        logger.info("Keyboard jog ready")

    def run_control_loop(self):
        while not self.runtime_state.quit_requested:
            while time.monotonic_ns() < self.tick_due_ns:
                self.robot.service_idle()
                remaining_ns = self.tick_due_ns - time.monotonic_ns()
                if remaining_ns > 0:
                    time.sleep(min(0.005, remaining_ns / 1e9))
            scheduled_ns = self.tick_due_ns
            self.tick_due_ns = next_tick_ns(scheduled_ns, self.period_ns, time.monotonic_ns())
            self.robot.service_feedback()
            self.keys.allow_start()
            events = self.keys.poll(self.runtime_state, timeout=0)[0]
            if self.runtime_state.estop_request:
                break
            if self.runtime_state.quit_requested:
                return True

            if self.handle_operation(events):
                continue
            if self.runtime_state.stop_signal.is_set():
                continue
            timing = self.jog.timing
            if timing is not None:
                missed_slots = (self.tick_due_ns - scheduled_ns) // self.period_ns - 1
                if missed_slots > 0:
                    timing.add("control_miss", scheduled_ns=scheduled_ns, slots=missed_slots)
            self.jog.step(
                self.robot, self.keys, self.runtime_state, self.runtime,
                scheduled_ns=scheduled_ns,
            )
        return bool(self.runtime_state.quit_requested)

    def handle_operation(self, events) -> bool:
        if "r" not in events:
            return False
        epoch = self.keys.accept_operation(self.runtime_state)
        if epoch is not None:
            self.run_home(epoch)
        return True

    def run_home(self, expected_cancel_epoch):
        with self.runtime_state.motion_lock:
            if (self.runtime_state.cancel_epoch != expected_cancel_epoch
                    or not self.runtime_state.requests_allow_motion()):
                logger.info("Ignored revoked HOME intent")
                return
            self.runtime_state.stop_signal.set()
            cancel_epoch = self.runtime_state.cancel_epoch
        logger.info("Returning home..." if self.runtime.hand.enabled else "Returning arm home...")

        def cancelled():
            self.robot.check()
            return not self.runtime_state.requests_allow_motion()

        try:
            home_robot(
                self.runtime_state, self.robot,
                environment=self.runtime.environment,
                cancel_epoch=cancel_epoch,
                cancel_requested=cancelled,
            )
        except HomeConfigurationError as exc:
            self.robot.check()
            logger.warning("本次 HOME 未完成：%s", exc)
        except HomePathRejected as exc:
            self.robot.check()
            logger.warning(
                "机械臂 HOME 路径不可行（%s）。可使用 Jog 手动调整姿态后再次按 R。",
                exc,
            )
        else:
            logger.info("HOME completed")
        if (self.runtime_state.cancel_epoch != cancel_epoch
                or not self.runtime_state.requests_allow_motion()):
            return False
        self.prepare_jog(cancel_epoch)
        self.tick_due_ns = time.monotonic_ns() + self.period_ns
