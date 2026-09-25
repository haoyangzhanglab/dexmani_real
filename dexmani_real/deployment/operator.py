"""Policy keyboard ownership and HOME/start/stop authorization ordering."""

import threading

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.ipc.channels import RuntimeChannels
from dexmani_real.planning import XArm7MotionPlanner
from dexmani_real.robot.arm_homing import home_policy_robot
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
_POLL_S = 0.05


class PolicyOperator:
    """Keep immediate motion fences responsive while the operator thread runs HOME."""

    def __init__(
        self,
        shared: RuntimeChannels,
        runtime: ExperimentConfig,
        planner: XArm7MotionPlanner | None,
        *,
        stop_event: threading.Event,
        execute: bool,
    ):
        self.shared = shared
        self.runtime = runtime
        self.planner = planner
        self.stop_event = stop_event
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
        """Fence live motion when S/Q arrives while this thread is blocked by H."""
        if not request_policy_stop(self.shared):
            self.shared.error_state.value = True

    def _request_quit(self) -> None:
        """Apply Q's motion fence before asking the supervisor to shut down."""
        if not request_policy_stop(self.shared, reason=RunEndReason.QUIT):
            self.shared.error_state.value = True
        self.shared.quit_requested.value = True

    def run(self) -> None:
        try:
            self.keyboard.start()
        except Exception:
            # Without the e-stop keyboard the deployment must not run: fail closed
            # so the supervisor observes a sticky fault and shuts down.
            logger.error("operator: keyboard failed to start; latching error_state", exc_info=True)
            self.shared.error_state.value = True
            return
        try:
            while not self.stop_event.is_set() and self.shared.is_running.value:
                if self.keyboard.estop_latched or not self.keyboard.healthy:
                    self.shared.estop_request.value = True
                    return
                if not self._handle_command_batch(self.keyboard.poll(timeout=_POLL_S)):
                    return
        finally:
            self.keyboard.stop()

    def _handle_command_batch(self, signals) -> bool:
        # A physical B must be a fresh, post-home confirmation.  H blocks
        # this thread while the arm moves, so begin events from the same
        # drained batch must not survive a successful home sequence.
        discard_begin_in_batch = False
        # Lifecycle-changing signals suppress Home and Begin in the same batch.
        # C/D (PAUSE/DISCARD) belong to teleop and are true no-ops here, so
        # they must not fence H; ESC is fenced by the estop latch/callback.
        stop_in_batch = any(
            signal in {OperatorCommand.STOP, OperatorCommand.QUIT} for signal in signals
        )
        for signal in signals:
            if signal is OperatorCommand.BEGIN:
                if stop_in_batch:
                    logger.warning("operator: ignored B received in the same batch as S/Q")
                    continue
                if (
                    discard_begin_in_batch
                    or self.shared.workflow_failed.value
                    or not request_policy_start(
                        self.shared,
                        require_physical_home=self.planner is not None,
                    )
                ):
                    logger.warning(
                        "operator: ignored B until a completed physical home "
                        "sequence is followed by a fresh B"
                    )
                    continue
            elif signal is OperatorCommand.STOP:
                # The keyboard completed the motion fence before enqueueing.
                # STOP still suppresses HOME/BEGIN in this batch, but must
                # not revoke again after the policy runner acknowledges it.
                continue
            elif signal is OperatorCommand.PAUSE:
                logger.warning("operator: C is not used in policy deployment; ignored")
            elif signal is OperatorCommand.DISCARD:
                logger.warning("operator: D is not used in policy deployment; ignored")
            elif signal is OperatorCommand.HOME:
                if self.planner is None:
                    logger.warning("operator: H is disabled in policy deployment")
                    continue
                if stop_in_batch:
                    logger.warning("operator: ignored H received in the same batch as S/Q")
                    continue
                if self._run_home():
                    discard_begin_in_batch = True
            elif signal is OperatorCommand.QUIT:
                # Listener stays available through final recording cleanup.
                continue
            elif signal is OperatorCommand.EMERGENCY_STOP:
                self.shared.estop_request.value = True
                return False
        return True

    def _run_home(self) -> bool:
        with self.shared.motion_lock:
            home_allowed = not self.shared.quit_requested.value and int(
                self.shared.safety_state.value
            ) == int(SafetyState.ARMED)
            self.shared.physical_home_completed.value = False
            if home_allowed:
                # H follows any older inactive S but cannot erase an
                # S arriving after this atomic preparation.
                self.shared.start_request.value = False
                self.shared.stop_request.value = int(StopRequest.NONE)
        if not home_allowed:
            logger.warning("operator: ignored H unless safety is ARMED")
            return False
        completed = home_policy_robot(
            self.shared,
            self.runtime,
            self.planner,
            abort_requested=self._home_abort_requested,
        )
        with self.shared.motion_lock:
            authorized = bool(
                completed
                and self.shared.is_running.value
                and not self.shared.quit_requested.value
                and not self.shared.workflow_failed.value
                and not self.shared.error_state.value
                and not self.shared.estop_request.value
                and int(self.shared.stop_request.value) == int(StopRequest.NONE)
                and int(self.shared.safety_state.value) == int(SafetyState.ARMED)
            )
            # Stop/start requests share this lock, so a completed H
            # cannot resurrect authorization after a newer S.
            self.shared.physical_home_completed.value = authorized
        if authorized:
            logger.info("operator: physical home sequence completed; press B to start")
        else:
            logger.warning(
                "operator: physical home sequence did not authorize; "
                "B remains disabled for the next episode"
            )
        # HOME blocks while hand/arm homing completes. Drop stale
        # H and B events, but preserve S/Q/ESC so an operator can
        # still stop, quit, or e-stop immediately afterwards.
        self.keyboard.drain_signal(OperatorCommand.HOME)
        self.keyboard.drain_signal(OperatorCommand.BEGIN)
        return True

    def _home_abort_requested(self) -> bool:
        return bool(
            self.stop_event.is_set()
            or not self.shared.is_running.value
            or self.shared.quit_requested.value
            or self.shared.error_state.value
            or self.shared.estop_request.value
            or int(self.shared.stop_request.value) != int(StopRequest.NONE)
        )
