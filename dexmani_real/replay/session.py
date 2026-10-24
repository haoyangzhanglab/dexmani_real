"""Own local robot I/O, replay scheduling, return-home and replay evaluation."""

import logging
import math
from dataclasses import replace
from pathlib import Path

from dexmani_real.data.results import EpisodeResults, error_detail
from dexmani_real.replay.evaluation import evaluate_replay
from dexmani_real.replay.replayer import ReplayOutcome, ReplayStatus, TrajectoryReplayer
from dexmani_real.replay.trajectory import verify_replay_preflight
from dexmani_real.robot.robot import DexManiRobot
from dexmani_real.runtime.home import build_home_planner, home_robot, home_hand
from dexmani_real.runtime.observation import service_feedback
from dexmani_real.runtime.operator_input import KeyboardInput, OperatorCommand
from dexmani_real.runtime.processes import shutdown_local_runtime
from dexmani_real.runtime.state import RuntimeState
from dexmani_real.utils.log import configure_logging

DEFAULT_OUTPUT_DIR = "replay_results"
logger = logging.getLogger(__name__)


class ReplaySession:
    """Own replay startup, physical evidence, return-home and evaluation."""

    def __init__(self, trajectory, runtime, *, output_dir, hand_start_duration_s=0.5):
        self.trajectory = trajectory
        self.runtime = runtime
        self.output_dir = output_dir
        self.hand_start_duration_s = hand_start_duration_s

    def run(self):
        configure_logging()
        from dexmani_real.config.experiment import (
            resolve_table_plane,
            validate_robot_config,
            validate_control_rate,
        )

        if not math.isfinite(self.hand_start_duration_s) or self.hand_start_duration_s < 0:
            raise ValueError("hand_start_duration_s must be finite and >= 0")
        validate_robot_config(self.runtime)
        validate_control_rate(self.runtime, self.trajectory.fps)
        self.table_plane_abcd = resolve_table_plane(
            self.runtime.environment.table, required=self.runtime.environment.table.enabled,
        )
        target = Path(self.output_dir)
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise ValueError("replay output must be missing or empty")
        verify_replay_preflight(self.trajectory, self.runtime)
        self.results = EpisodeResults(target, "replay")
        self.runtime_state = self.robot = self.keyboard = None
        trajectory_outcome = None
        shutdown_clean = False
        outcome = ReplayOutcome(ReplayStatus.REJECTED, reason="startup failed")
        try:
            self.runtime_state = RuntimeState()
            self.robot = DexManiRobot(
                self.runtime.arm, self.runtime.hand, run_signal=self.runtime_state.run_signal,
                stop_signal=self.runtime_state.stop_signal,
                emergency_signal=self.runtime_state.emergency_signal,
                signal_lock=self.runtime_state.motion_lock,
            )

            self.keyboard = KeyboardInput(
                quit_callback=self.runtime_state.request_quit,
                stop_callback=self.runtime_state.request_stop,
                estop_callback=self.runtime_state.request_emergency_stop,
            )
            self.home_planner = build_home_planner(
                self.runtime.environment, table_plane_abcd=self.table_plane_abcd,
            )
            self.keyboard.start()
            self.robot.check_services = lambda: self.keyboard.healthy
            self.robot.connect()
            self.robot.check()
            self.runtime_state.arm()
            home_result = home_hand(
                self.runtime_state,
                self.runtime.hand,
                robot=self.robot,
                abort_requested=lambda: (
                    bool(self.runtime_state.estop_request or self.runtime_state.quit_requested)
                    or not self.keyboard.healthy
                ),
            )
            if home_result.ok:
                outcome = TrajectoryReplayer(
                    self.runtime_state, self.runtime, self.keyboard, robot=self.robot
                ).run(
                    self.trajectory,
                    hand_start_duration_s=self.hand_start_duration_s,
                    results=self.results,
                )
            else:
                if self.runtime_state.estop_request:
                    status = ReplayStatus.ESTOP
                elif home_result.interrupted and self.runtime_state.quit_requested:
                    status = ReplayStatus.USER_QUIT
                else:
                    status = ReplayStatus.REJECTED
                outcome = ReplayOutcome(
                    status,
                    reason=f"startup hand home incomplete: {home_result.reason}",
                )
            trajectory_outcome = outcome
            if self.results.current is not None:
                self.results.finish_episode(
                    outcome.status.value,
                    details=outcome.reason,
                    replay_rows=(
                        len(outcome.replay_data["arm_qpos"])
                        if outcome.replay_data is not None
                        else 0
                    ),
                )
            if outcome.successful:
                print("H: planned return_home; Q: exit", flush=True)
                while self.runtime_state.running and not self.runtime_state.quit_requested:
                    self.robot.service_idle()
                    if not self.keyboard.healthy:
                        self.runtime_state.request_emergency_stop()
                    if self.runtime_state.error_state or self.runtime_state.estop_request:
                        break
                    signals = self.keyboard.poll(timeout=0.1)
                    if OperatorCommand.QUIT in signals:
                        break
                    if OperatorCommand.HOME in signals:
                        home_result = home_robot(
                            self.runtime_state,
                            self.runtime.arm, self.runtime.hand,
                            self.home_planner,
                            robot=self.robot,
                            service=lambda: service_feedback(self.robot, self.runtime),
                            abort_requested=lambda: (
                                bool(
                                    self.runtime_state.estop_request
                                    or self.runtime_state.quit_requested
                                )
                                or not self.keyboard.healthy
                            ),
                        )
                        self.results.record_home_result(
                            ok=home_result.ok,
                            interrupted=home_result.interrupted,
                            reason=home_result.reason,
                        )
                        if not home_result.ok:
                            if home_result.interrupted and self.runtime_state.quit_requested:
                                status = ReplayStatus.USER_QUIT
                            elif self.runtime_state.estop_request:
                                status = ReplayStatus.ESTOP
                            else:
                                status = ReplayStatus.FAULT
                            outcome = replace(
                                outcome,
                                status=status,
                                reason=f"return_home incomplete: {home_result.reason}",
                            )
                            break
                        print("Return-home completed. H: planned return_home; Q: exit", flush=True)
        except KeyboardInterrupt:
            if self.runtime_state is not None:
                self.runtime_state.request_emergency_stop()
            outcome = replace(outcome, status=ReplayStatus.ESTOP, reason="KeyboardInterrupt")
        except Exception as exc:
            if self.runtime_state is not None:
                self.runtime_state.mark_error()
            logger.exception("replay session failed")
            outcome = replace(
                outcome, status=ReplayStatus.FAULT, reason=f"{type(exc).__name__}: {exc}"
            )
        finally:
            try:
                if self.runtime_state is not None:
                    shutdown_clean = shutdown_local_runtime(
                        self.runtime_state,
                        self.robot,
                        keyboard=self.keyboard,
                    )
            except Exception as exc:
                self.results.record_errors([error_detail("shutdown", exc)])
            if self.runtime_state is not None:
                if self.runtime_state.error_state or (
                    not shutdown_clean and not self.runtime_state.estop_request
                ):
                    outcome = replace(
                        outcome,
                        status=ReplayStatus.FAULT,
                        reason=f"{outcome.reason}; hardware/shutdown failure".lstrip("; "),
                    )
                elif self.runtime_state.estop_request:
                    outcome = replace(
                        outcome,
                        status=ReplayStatus.ESTOP,
                        reason=outcome.reason or "operator emergency stop",
                    )
            try:
                if self.results.current is not None:
                    self.results.finish_episode(outcome.status.value, details=outcome.reason)
                evaluate_replay(
                    self.trajectory,
                    outcome.replay_data,
                    output_dir=self.output_dir,
                    hand_start_duration_s=self.hand_start_duration_s,
                )
            except Exception as exc:
                self.results.record_errors([error_detail("evaluation_save", exc)])
                outcome = replace(
                    outcome,
                    status=ReplayStatus.FAULT,
                    reason=f"{outcome.reason}; {exc}".lstrip("; "),
                )
            try:
                self.results.finish_session(
                    outcome=outcome.status.value,
                    reason=outcome.reason,
                    shutdown_clean=shutdown_clean,
                    trajectory_status=(
                        trajectory_outcome.status.value if trajectory_outcome else None
                    ),
                    trajectory_reason=trajectory_outcome.reason if trajectory_outcome else None,
                )
            except Exception as exc:
                logger.exception("replay final result publication failed")
                outcome = replace(
                    outcome,
                    status=ReplayStatus.FAULT,
                    reason=f"{outcome.reason}; final result: {exc}".lstrip("; "),
                )
        return outcome
