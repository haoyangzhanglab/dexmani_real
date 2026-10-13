"""Policy keyboard ownership and HOME/reset/start/stop authorization ordering."""

import logging

from concurrent.futures import CancelledError

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.runtime.state import RuntimeState
from dexmani_real.planning import XArm7MotionPlanner
from dexmani_real.robot.arm_homing import home_robot
from dexmani_real.runtime.operator_input import KeyboardInput, OperatorCommand
from dexmani_real.runtime.safety import (
    RunEndReason,
    SafetyState,
    StopRequest,
    request_policy_start,
    request_policy_stop,
)


logger = logging.getLogger(__name__)


class PolicyOperator:
    """Handle requests on the I/O owner; keyboard callbacks only revoke authority."""

    def __init__(
        self,
        shared: RuntimeState,
        runtime: ExperimentConfig,
        planner: XArm7MotionPlanner,
        *,
        robot,
        idle_for_tactile_reset=None,
    ):
        self.idle_for_tactile_reset = idle_for_tactile_reset
        self.home_results = []
        self.shared = shared
        self.robot = robot
        self.runtime = runtime
        self.planner = planner
        self.keyboard = KeyboardInput(
            estop_callback=lambda: setattr(shared, "estop_request", True),
            stop_callback=self._request_stop,
            quit_callback=self._request_quit,
        )

    def _request_stop(self) -> None:
        """Fence motion even while the I/O owner is blocked in HOME, inference or SDK I/O."""
        if not request_policy_stop(self.shared):
            self.shared.error_state = True

    def _request_quit(self) -> None:
        """Apply Q's motion fence before requesting session shutdown."""
        if not request_policy_stop(self.shared, reason=RunEndReason.QUIT):
            self.shared.error_state = True
        self.shared.quit_requested = True

    def poll(self) -> None:
        if self.keyboard.estop_latched or not self.keyboard.healthy:
            self.shared.estop_request = True
            return
        self._handle_command_batch(self.keyboard.poll(timeout=0))

    def discard_pending_start(self):
        self.keyboard.drain_signal(OperatorCommand.BEGIN)

    def _handle_command_batch(self, signals) -> None:
        # HOME blocks; H/B/T from the same batch need a fresh request afterwards.
        discard_preparation_in_batch = False
        home_handled_in_batch = False
        # S/Q suppress H/B/T throughout the batch; ESC uses the latch/callback.
        stop_in_batch = any(
            signal in {OperatorCommand.STOP, OperatorCommand.QUIT} for signal in signals
        )
        for signal in signals:
            if signal is OperatorCommand.BEGIN:
                if stop_in_batch:
                    logger.warning("operator: ignored B received in the same batch as S/Q")
                    continue
                if discard_preparation_in_batch or not request_policy_start(self.shared):
                    logger.warning(
                        "operator: ignored B while stopped or after preparation in this batch"
                    )
                    continue
            elif signal is OperatorCommand.STOP:
                # Keyboard callbacks already revoked motion; a second revoke could cross a cleared stop request.
                continue
            elif signal is OperatorCommand.PAUSE:
                logger.warning("operator: C is not used in policy deployment; ignored")
            elif signal is OperatorCommand.DISCARD:
                logger.warning("operator: D is not used in policy deployment; ignored")
            elif signal is OperatorCommand.RESET_TACTILE:
                if discard_preparation_in_batch:
                    logger.warning("operator: ignored T after preparation in the same batch")
                    continue
                discard_preparation_in_batch = True
                if stop_in_batch or self.idle_for_tactile_reset is None or not self.idle_for_tactile_reset():
                    logger.warning(
                        "T ignored: tactile reset requires idle, no capture/reset/pending inference"
                    )
                    continue
                if int(self.shared.safety_state) != int(SafetyState.ARMED):
                    continue
                try:
                    logger.info("正在请求触觉传感器复位，请保持手部无接触；S/Q/Esc 可中断后续请求。")
                    self.robot.reset_tactile_sensors(cancel_requested=self._home_abort_requested)
                except CancelledError as exc:
                    logger.info("%s", exc)
                finally:
                    for command in (
                        OperatorCommand.RESET_TACTILE,
                        OperatorCommand.HOME,
                        OperatorCommand.BEGIN,
                    ):
                        self.keyboard.drain_signal(command)
                return
            elif signal is OperatorCommand.HOME:
                if home_handled_in_batch:
                    continue
                if stop_in_batch:
                    logger.warning("operator: ignored H received in the same batch as S/Q")
                    continue
                if self._run_home():
                    home_handled_in_batch = True
                    discard_preparation_in_batch = True
            elif signal is OperatorCommand.QUIT:
                # Listener stays available through final recording cleanup.
                continue
            elif signal is OperatorCommand.EMERGENCY_STOP:
                self.shared.estop_request = True
                return

    def _run_home(self) -> bool:
        with self.shared.motion_lock:
            home_allowed = (
                self.shared.running
                and not self.shared.quit_requested
                and not self.shared.error_state
                and not self.shared.estop_request
                and int(self.shared.stop_request) == int(StopRequest.NONE)
                and int(self.shared.safety_state) == int(SafetyState.ARMED)
            )
            if home_allowed:
                # Only the owner consumes S; HOME never clears a newer stop.
                self.shared.start_request = False
        if not home_allowed:
            logger.warning("operator: ignored H unless ARMED without shutdown, fault or stop")
            return False
        home_result = {"outcome": "incomplete"}
        self.home_results.append(home_result)
        try:
            logger.info("正在回初始位…")
            result = home_robot(
                self.shared,
                self.runtime,
                self.planner,
                robot=self.robot,
                abort_requested=self._home_abort_requested,
            )
        except BaseException as exc:
            home_result.update(outcome="fault", reason=f"{type(exc).__name__}: {exc}")
            raise
        with self.shared.motion_lock:
            completed_without_stop = bool(
                result.ok
                and self.shared.running
                and not self.shared.quit_requested
                and not self.shared.error_state
                and not self.shared.estop_request
                and int(self.shared.stop_request) == int(StopRequest.NONE)
                and int(self.shared.safety_state) == int(SafetyState.ARMED)
            )
        interrupted = bool(
            result.interrupted
            or (result.ok and (
                self.shared.quit_requested or self.shared.estop_request or self.shared.stop_request
            ))
        )
        home_result.update(
            outcome="completed"
            if completed_without_stop
            else "interrupted"
            if interrupted
            else "failed",
            reason=result.reason,
        )
        if completed_without_stop:
            logger.info("已回初始位，按 B 开始。")
        else:
            logger.warning("回初始位未完成，请检查当前姿态：%s", result.reason)
        # Discard stale H/B/T after blocking HOME; preserve S/Q/ESC.
        self.keyboard.drain_signal(OperatorCommand.RESET_TACTILE)
        self.keyboard.drain_signal(OperatorCommand.HOME)
        self.keyboard.drain_signal(OperatorCommand.BEGIN)
        return True

    def _home_abort_requested(self) -> bool:
        self.robot.check()
        return bool(
            not self.shared.running
            or self.shared.quit_requested
            or self.shared.error_state
            or self.shared.estop_request
            or int(self.shared.stop_request) != int(StopRequest.NONE)
        )
