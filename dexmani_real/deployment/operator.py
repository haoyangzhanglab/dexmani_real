"""Policy keyboard ownership and HOME/reset/start/stop authorization ordering."""

import logging
from concurrent.futures import CancelledError

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.runtime.state import RuntimeState, RunEndReason, SafetyState
from dexmani_real.planning.home_planner import HomePlanner
from dexmani_real.robot.home import home_robot
from dexmani_real.runtime.operator_input import KeyboardInput, OperatorCommand

logger = logging.getLogger(__name__)


class PolicyOperator:
    """Handle requests on the I/O owner; keyboard callbacks only revoke authority."""

    def __init__(
        self,
        runtime_state: RuntimeState,
        runtime: ExperimentConfig,
        planner: HomePlanner,
        *,
        robot,
    ):
        self.home_results = []
        self.start_requested = False
        self.runtime_state = runtime_state
        self.robot = robot
        self.runtime = runtime
        self.planner = planner
        self.keyboard = KeyboardInput(
            estop_callback=runtime_state.request_emergency_stop,
            stop_callback=self.request_stop,
            quit_callback=self.request_quit,
        )

    def request_start(self) -> bool:
        with self.runtime_state.motion_lock:
            if not self.runtime_state.command_may_cross_sdk(
                run_id=self.runtime_state.run_id,
                required_safety_state=SafetyState.ARMED,
            ):
                return False
            self.start_requested = True
            return True

    def request_stop(self, *, reason=RunEndReason.OPERATOR) -> None:
        """Fence motion even while the I/O owner is blocked in HOME, inference or SDK I/O."""
        with self.runtime_state.motion_lock:
            self.start_requested = False
            self.runtime_state.request_stop(reason=reason)

    def request_quit(self) -> None:
        """Apply Q's motion fence before requesting session shutdown."""
        with self.runtime_state.motion_lock:
            self.start_requested = False
            self.runtime_state.request_quit()

    def poll(self, *, allow_tactile_reset: bool = False) -> None:
        if self.keyboard.estop_latched or not self.keyboard.healthy:
            self.runtime_state.request_emergency_stop()
            return
        self.handle_command_batch(
            self.keyboard.poll(timeout=0),
            allow_tactile_reset=allow_tactile_reset,
        )

    def discard_pending_start(self):
        self.keyboard.drain_signal(OperatorCommand.BEGIN)

    def handle_command_batch(self, signals, *, allow_tactile_reset=False) -> None:
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
                if discard_preparation_in_batch or not self.request_start():
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
                # B earlier in this batch may have created a pending start.
                if stop_in_batch or self.start_requested or not allow_tactile_reset:
                    logger.warning(
                        "T ignored: tactile reset requires idle, no capture/reset/pending inference"
                    )
                    continue
                if self.runtime_state.safety_state != SafetyState.ARMED:
                    continue
                try:
                    logger.info(
                        "正在请求触觉传感器复位，请保持手部无接触；S/Q/Esc 可中断后续请求。"
                    )
                    self.robot.reset_tactile_sensors(cancel_requested=self.home_abort_requested)
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
                if self.run_home():
                    home_handled_in_batch = True
                    discard_preparation_in_batch = True
            elif signal is OperatorCommand.QUIT:
                # Listener stays available through final recording cleanup.
                continue
            elif signal is OperatorCommand.EMERGENCY_STOP:
                self.runtime_state.request_emergency_stop()
                return

    def run_home(self) -> bool:
        with self.runtime_state.motion_lock:
            home_allowed = self.runtime_state.command_may_cross_sdk(
                run_id=self.runtime_state.run_id,
                required_safety_state=SafetyState.ARMED,
            )
            if home_allowed:
                # Only the owner consumes S; HOME never clears a newer stop.
                self.start_requested = False
        if not home_allowed:
            logger.warning("operator: ignored H unless ARMED without shutdown, fault or stop")
            return False
        home_result = {"outcome": "incomplete"}
        self.home_results.append(home_result)
        try:
            logger.info("正在回初始位…")
            result = home_robot(
                self.runtime_state,
                self.runtime,
                self.planner,
                robot=self.robot,
                abort_requested=self.home_abort_requested,
            )
        except BaseException as exc:
            home_result.update(outcome="fault", reason=f"{type(exc).__name__}: {exc}")
            raise
        with self.runtime_state.motion_lock:
            completed_without_stop = result.ok and self.runtime_state.command_may_cross_sdk(
                run_id=self.runtime_state.run_id,
                required_safety_state=SafetyState.ARMED,
            )
        interrupted = bool(
            result.interrupted
            or (
                result.ok
                and (
                    self.runtime_state.quit_requested
                    or self.runtime_state.estop_request
                    or self.runtime_state.stop_request
                )
            )
        )
        home_result.update(
            outcome=(
                "completed"
                if completed_without_stop
                else "interrupted" if interrupted else "failed"
            ),
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

    def home_abort_requested(self) -> bool:
        self.robot.check()
        return bool(
            not self.runtime_state.running
            or self.runtime_state.quit_requested
            or self.runtime_state.error_state
            or self.runtime_state.estop_request
            or self.runtime_state.stop_request
        )
