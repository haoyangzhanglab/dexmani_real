"""Policy keyboard ownership and HOME/TARE/start/stop authorization ordering."""

from concurrent.futures import CancelledError

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.ipc.channels import RuntimeChannels
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
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


class PolicyOperator:
    """Handle requests on the I/O owner; keyboard callbacks only revoke authority."""

    def __init__(
        self,
        shared: RuntimeChannels,
        runtime: ExperimentConfig,
        planner: XArm7MotionPlanner | None,
        *,
        robot,
        execute: bool,
        idle_for_tare=None,
    ):
        self.idle_for_tare = idle_for_tare
        self.home_results = []
        self.shared = shared
        self.robot = robot
        self.runtime = runtime
        self.planner = planner
        if not isinstance(execute, bool):
            raise TypeError("execute must be a boolean")
        if execute != (planner is not None):
            raise ValueError("execute must match physical home availability")
        self.keyboard = KeyboardInput(
            estop_callback=lambda: setattr(shared.estop_request, "value", True),
            stop_callback=self._request_stop,
            quit_callback=self._request_quit,
        )

    def _request_stop(self) -> None:
        """Fence motion even while the I/O owner is blocked in HOME, inference or SDK I/O."""
        if not request_policy_stop(self.shared):
            self.shared.error_state.value = True

    def _request_quit(self) -> None:
        """Apply Q's motion fence before asking the supervisor to shut down."""
        if not request_policy_stop(self.shared, reason=RunEndReason.QUIT):
            self.shared.error_state.value = True
        self.shared.quit_requested.value = True

    def poll(self) -> None:
        if self.keyboard.estop_latched or not self.keyboard.healthy:
            self.shared.estop_request.value = True
            return
        self._handle_command_batch(self.keyboard.poll(timeout=0))

    def _handle_command_batch(self, signals) -> None:
        # H blocks the I/O owner. H/B/T in that batch require a fresh request
        # after HOME, including when HOME returns incomplete.
        discard_begin_or_tare_in_batch = False
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
                if discard_begin_or_tare_in_batch or not request_policy_start(self.shared):
                    logger.warning(
                        "operator: ignored B while stopped or after preparation in this batch"
                    )
                    continue
            elif signal is OperatorCommand.STOP:
                # The keyboard completed the motion fence before enqueueing.
                # STOP still suppresses HOME/BEGIN/TARE in this batch, but must
                # not revoke again after the policy runner clears the stop request.
                continue
            elif signal is OperatorCommand.PAUSE:
                logger.warning("operator: C is not used in policy deployment; ignored")
            elif signal is OperatorCommand.DISCARD:
                logger.warning("operator: D is not used in policy deployment; ignored")
            elif signal is OperatorCommand.TARE:
                if discard_begin_or_tare_in_batch:
                    logger.warning("operator: ignored T after preparation in the same batch")
                    continue
                discard_begin_or_tare_in_batch = True
                if stop_in_batch or self.idle_for_tare is None or not self.idle_for_tare():
                    logger.warning(
                        "T ignored: tactile tare requires idle, no capture/reset/pending inference"
                    )
                    continue
                if int(self.shared.safety_state.value) != int(SafetyState.ARMED):
                    continue
                try:
                    result = self.robot.tare_tactile(cancel_requested=self._home_abort_requested)
                    logger.info("Explicit tactile baseline aggregate/dense: %s", result)
                except CancelledError:
                    logger.info("Tactile tare cancelled; no new baseline published")
                finally:
                    for command in (
                        OperatorCommand.TARE,
                        OperatorCommand.HOME,
                        OperatorCommand.BEGIN,
                    ):
                        self.keyboard.drain_signal(command)
                return
            elif signal is OperatorCommand.HOME:
                if home_handled_in_batch:
                    continue
                if self.planner is None:
                    logger.warning("operator: H is disabled in policy deployment")
                    continue
                if stop_in_batch:
                    logger.warning("operator: ignored H received in the same batch as S/Q")
                    continue
                if self._run_home():
                    home_handled_in_batch = True
                    discard_begin_or_tare_in_batch = True
            elif signal is OperatorCommand.QUIT:
                # Listener stays available through final recording cleanup.
                continue
            elif signal is OperatorCommand.EMERGENCY_STOP:
                self.shared.estop_request.value = True
                return

    def _run_home(self) -> bool:
        with self.shared.motion_lock:
            home_allowed = (
                self.shared.is_running.value
                and not self.shared.quit_requested.value
                and not self.shared.error_state.value
                and not self.shared.estop_request.value
                and int(self.shared.stop_request.value) == int(StopRequest.NONE)
                and int(self.shared.safety_state.value) == int(SafetyState.ARMED)
            )
            if home_allowed:
                # Only the owner consumes S; HOME never clears a newer stop.
                self.shared.start_request.value = False
        if not home_allowed:
            logger.warning("operator: ignored H unless ARMED without shutdown, fault or stop")
            return False
        home_result = {"outcome": "incomplete"}
        self.home_results.append(home_result)
        try:
            completed = home_robot(
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
                completed
                and self.shared.is_running.value
                and not self.shared.quit_requested.value
                and not self.shared.error_state.value
                and not self.shared.estop_request.value
                and int(self.shared.stop_request.value) == int(StopRequest.NONE)
                and int(self.shared.safety_state.value) == int(SafetyState.ARMED)
            )
        interrupted = bool(
            self.shared.quit_requested.value
            or self.shared.estop_request.value
            or self.shared.stop_request.value
        )
        home_result.update(
            outcome="completed"
            if completed_without_stop
            else "interrupted"
            if interrupted
            else "failed"
        )
        if completed_without_stop:
            logger.info("operator: physical home sequence completed; press B to start")
        else:
            logger.warning("operator: HOME incomplete or interrupted; check current pose before B")
        # HOME blocks while hand/arm homing completes. Drop stale
        # H/B/T events, but preserve S/Q/ESC so an operator can
        # still stop, quit, or e-stop immediately afterwards.
        self.keyboard.drain_signal(OperatorCommand.TARE)
        self.keyboard.drain_signal(OperatorCommand.HOME)
        self.keyboard.drain_signal(OperatorCommand.BEGIN)
        return True

    def _home_abort_requested(self) -> bool:
        self.robot.check()
        return bool(
            not self.shared.is_running.value
            or self.shared.quit_requested.value
            or self.shared.error_state.value
            or self.shared.estop_request.value
            or int(self.shared.stop_request.value) != int(StopRequest.NONE)
        )
