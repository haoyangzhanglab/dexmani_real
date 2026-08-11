"""Live replay worker lifecycle and post-run reporting."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from dexmani_real.config.runtime import ResolvedRuntimeConfig
from dexmani_real.policy.action_protocol import hand_home_converge
from dexmani_real.replay.data import TrajectoryData
from dexmani_real.replay.metrics import ReplayMetrics, compute_metrics, save_replay_data, save_results
from dexmani_real.replay.preflight import verify_live_replay_preflight
from dexmani_real.replay.runner import ReplayOutcome, ReplayStatus, TrajectoryReplayer
from dexmani_real.robot.arm_loop import ArmLoopConfig
from dexmani_real.robot.arm_loop import arm_loop as _arm_loop
from dexmani_real.robot.hand_process import hand_loop as _hand_loop
from dexmani_real.robot.homing import send_arm_home
from dexmani_real.robot.safety import SafetyState, require_transition
from dexmani_real.runtime.processes import ShutdownReport
from dexmani_real.runtime.supervisor import shutdown_processes, wait_subsystem_ready
from dexmani_real.shm.shared_storage import SharedStorage, SharedStorageConfig
from dexmani_real.teleop.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_TRACKING_ERROR_PERCENTILE = 95.0


@dataclass(frozen=True)
class LiveReplayConfig:
    speed: float
    no_hand: bool
    max_frames: int | None
    output_dir: str
    evaluate_consistency: bool


def _latched_fault_status(shared: SharedStorage) -> ReplayStatus | None:
    """Classify sticky runtime state before any transition can mask it."""
    if shared.estop_request.value:
        return ReplayStatus.ESTOP
    if shared.error_state.value or int(shared.safety_state.value) == int(SafetyState.FAULT):
        return ReplayStatus.FAULT
    return None


def _post_shutdown_outcome(outcome: ReplayOutcome, report: ShutdownReport) -> ReplayOutcome:
    """Apply faults observed only after workers have reached a terminal state."""
    if report.estop_requested:
        shutdown_reason = "e-stop latched during replay shutdown"
        reason = f"{outcome.reason}; {shutdown_reason}" if outcome.reason else shutdown_reason
        return ReplayOutcome(
            ReplayStatus.ESTOP,
            outcome.replay_data,
            reason,
        )
    if report.faulted:
        failed = ", ".join(f"{item.name}={item.escalation}:{item.exitcode}" for item in report.abnormal_exits)
        shutdown_reason = (
            f"worker failed during replay shutdown: {failed}" if failed else "fault latched during replay shutdown"
        )
        reason = f"{outcome.reason}; {shutdown_reason}" if outcome.reason else shutdown_reason
        return ReplayOutcome(ReplayStatus.FAULT, outcome.replay_data, reason)
    return outcome


def _live_worker_health_issue(
    shared: SharedStorage,
    processes: list[Any],
    heartbeat_timeouts_s: dict[str, float],
    *,
    now_s: float | None = None,
) -> str | None:
    """Return the first arm/hand worker-health failure, if any."""
    for process in processes:
        if not process.is_alive():
            return f"worker {process.name!r} exited with code {process.exitcode}"

    heartbeat_by_name = {
        "arm": shared.arm_heartbeat_s,
        "hand": shared.hand_heartbeat_s,
    }
    for process in processes:
        heartbeat = heartbeat_by_name.get(process.name)
        if heartbeat is None:
            continue
        last_s = float(heartbeat.value)
        now = time.monotonic() if now_s is None else now_s
        timeout_s = float(heartbeat_timeouts_s[process.name])
        if not np.isfinite(last_s) or last_s <= 0 or last_s > now or now - last_s > timeout_s:
            return f"worker {process.name!r} heartbeat timed out"
    return None


def _report_consistency(metrics: ReplayMetrics) -> None:
    print("\n" + "=" * 60)
    print("Consistency Evaluation")
    print("=" * 60)
    print(f"  Frames: {metrics.replayed_frames} replayed / {metrics.original_frames} original")
    print(
        f"  Arm joint MAE:  {np.round(metrics.arm_joint_mae_deg, 2)} deg  "
        f"(overall: {metrics.arm_joint_mae_overall_deg:.3f} deg)"
    )
    print(
        f"  Arm joint RMSE: {np.round(metrics.arm_joint_rmse_deg, 2)} deg  "
        f"(overall: {metrics.arm_joint_rmse_overall_deg:.3f} deg)"
    )
    if metrics.eef_pos_error_per_frame_mm is not None:
        print(
            f"  EEF pos error:  mean={metrics.eef_pos_error_mean_mm:.1f}mm  "
            f"max={metrics.eef_pos_error_max_mm:.1f}mm  rmse={metrics.eef_pos_error_rmse_mm:.1f}mm"
        )
    if metrics.eef_rot_error_per_frame_deg is not None:
        print(
            f"  EEF rot error:  mean={metrics.eef_rot_error_mean_deg:.2f}°  "
            f"max={metrics.eef_rot_error_max_deg:.2f}°"
        )
    if metrics.hand_joint_mae_overall_deg is not None:
        print(f"  Hand joint MAE: {metrics.hand_joint_mae_overall_deg:.3f} deg")
    print(f"  Tracking lag:  {metrics.tracking_lag_frames} frames ({metrics.tracking_lag_seconds:.3f}s)")
    if metrics.arm_tracking_error_mean_deg > 0:
        print(
            "  Replay tracking error (cmd vs actual): "
            f"mean={metrics.arm_tracking_error_mean_deg:.2f}°  "
            f"p95={metrics.arm_tracking_error_p95_deg:.2f}°  "
            f"max={metrics.arm_tracking_error_max_deg:.2f}°"
        )
    print("=" * 60)


def _evaluate_replay(
    trajectory: TrajectoryData,
    replay_data: dict[str, np.ndarray] | None,
    config: LiveReplayConfig,
) -> None:
    if replay_data is None:
        print("\nNo replay data collected (replay interrupted before any frames captured).")
        return
    if replay_data["arm_qpos"].shape[0] == 0:
        print("\nSkipping metrics: no valid reference or replay data available")
        return
    if not config.evaluate_consistency:
        print("\nSkipping consistency metrics; saving captured replay data.")
        save_replay_data(replay_data, config.output_dir)
        return

    print("\nComputing consistency metrics...")
    try:
        metrics = compute_metrics(
            original_arm_qpos=trajectory.arm_qpos,
            replay_arm_qpos=replay_data["arm_qpos"],
            original_arm_ee=trajectory.arm_ee,
            replay_arm_ee_pos=replay_data["eef_pos"],
            replay_arm_ee_rot6d=replay_data["eef_rot6d"],
            fps=trajectory.fps,
            original_hand_qpos=trajectory.hand_qpos,
            replay_hand_qpos=replay_data.get("hand_qpos"),
            episode_path=trajectory.episode_path,
            task_label=trajectory.task_label,
            speed_factor=config.speed,
        )
    except Exception:
        logger.error("replay consistency evaluation failed; saving raw replay data", exc_info=True)
        save_replay_data(replay_data, config.output_dir)
        raise
    tracking_error = replay_data.get("arm_tracking_error")
    if tracking_error is not None:
        finite = tracking_error[np.isfinite(tracking_error)]
        if finite.size:
            metrics.arm_tracking_error_mean_deg = float(np.rad2deg(np.mean(finite)))
            metrics.arm_tracking_error_p95_deg = float(np.rad2deg(np.percentile(finite, _TRACKING_ERROR_PERCENTILE)))
            metrics.arm_tracking_error_max_deg = float(np.rad2deg(np.max(finite)))

    _report_consistency(metrics)
    save_results(metrics, replay_data, config.output_dir)


def _offer_return_home(
    shared: SharedStorage,
    replayer: TrajectoryReplayer,
    runtime: ResolvedRuntimeConfig,
    arm_config: ArmLoopConfig,
    *,
    hand_available: bool,
    health_check: Callable[[], str | None],
) -> tuple[ReplayStatus, str] | None:
    print("\nPress H to return_home, or Q to exit...")
    keyboard = KeyboardHandler(estop_callback=lambda: setattr(shared.estop_request, "value", True))
    keyboard.start()
    try:
        deadline = time.perf_counter() + float(runtime.policy.post_teleop_timeout_s)
        while time.perf_counter() < deadline:
            signals = set(keyboard.poll(timeout=0.1))
            if ControlSignal.EMERGENCY_STOP in signals or shared.estop_request.value:
                shared.estop_request.value = True
                shared.error_state.value = True
                require_transition(shared, SafetyState.FAULT)
                return ReplayStatus.ESTOP, "operator emergency stop during return-home prompt"
            if shared.error_state.value or int(shared.safety_state.value) == int(SafetyState.FAULT):
                require_transition(shared, SafetyState.FAULT)
                return ReplayStatus.FAULT, "runtime fault during return-home prompt"
            health_issue = health_check()
            if health_issue:
                shared.error_state.value = True
                require_transition(shared, SafetyState.FAULT)
                return ReplayStatus.FAULT, health_issue
            if ControlSignal.QUIT in signals:
                return None
            if ControlSignal.HOME not in signals:
                continue

            if hand_available:
                hand_home = np.deg2rad(np.asarray(runtime.hand.home_qpos_deg, dtype=np.float64))
                hand_reached, _final_hand_qpos = hand_home_converge(
                    shared,
                    hand_home,
                    timeout_s=float(runtime.hand.home_settle_timeout_s),
                    tol_rad=float(runtime.hand.home_settle_tol_rad),
                    heartbeat=False,
                    check_is_running=False,
                    verbose=True,
                    safety_gate=replayer.action_safety_gate,
                    abort_requested=lambda: keyboard.estop_latched or not keyboard.healthy,
                )
                if not hand_reached:
                    logger.warning("arm home cancelled because hand home was not confirmed")
                    continue

            arm_reached = send_arm_home(
                shared,
                np.asarray(arm_config.home_qpos, dtype=np.float64),
                planner=replayer.planner,
                table_z_surface_m=float(runtime.arm.table_z_surface_m),
                queue_timeout=float(runtime.arm.homing.request_queue_timeout_s),
                converge_timeout_s=float(runtime.arm.homing.convergence_timeout_s),
                state_max_age_s=float(runtime.arm.homing.state_max_age_s),
                homing_max_speed_rad_s=float(np.deg2rad(runtime.arm.homing.max_speed_deg_s)),
                homing_target_timeout_s=float(runtime.arm.homing.target_timeout_s),
                preplan_velocity_rad_s=float(runtime.arm.homing.velocity_convergence_rad_s),
                result_tolerance_rad=float(runtime.arm.homing.convergence_rad),
                arm_heartbeat_max_age_s=float(runtime.safety.heartbeat_timeouts["arm"]),
                estop_requested=lambda: keyboard.estop_latched or not keyboard.healthy,
                heartbeat=False,
                verbose=True,
            )
            if not arm_reached:
                if shared.estop_request.value:
                    shared.error_state.value = True
                    require_transition(shared, SafetyState.FAULT)
                    return ReplayStatus.ESTOP, "e-stop requested during return home"
                if shared.error_state.value or int(shared.safety_state.value) == int(SafetyState.FAULT):
                    require_transition(shared, SafetyState.FAULT)
                    return ReplayStatus.FAULT, "runtime fault during return home"
                health_issue = health_check()
                if health_issue:
                    shared.error_state.value = True
                    require_transition(shared, SafetyState.FAULT)
                    return ReplayStatus.FAULT, health_issue
                return ReplayStatus.REJECTED, "return-home request was not completed"
            print("Press Q to exit...")
    finally:
        keyboard.stop()
    if shared.estop_request.value:
        shared.error_state.value = True
        require_transition(shared, SafetyState.FAULT)
        return ReplayStatus.ESTOP, "e-stop requested when return-home prompt expired"
    if shared.error_state.value or int(shared.safety_state.value) == int(SafetyState.FAULT):
        require_transition(shared, SafetyState.FAULT)
        return ReplayStatus.FAULT, "runtime fault when return-home prompt expired"
    health_issue = health_check()
    if health_issue:
        shared.error_state.value = True
        require_transition(shared, SafetyState.FAULT)
        return ReplayStatus.FAULT, health_issue
    return None


def run_live_replay(
    trajectory: TrajectoryData,
    runtime: ResolvedRuntimeConfig,
    config: LiveReplayConfig,
) -> ReplayOutcome:
    """Start arm/hand workers, run one replay, then shut the session down."""
    try:
        verify_live_replay_preflight(
            trajectory,
            runtime,
            no_hand=config.no_hand,
            speed_factor=config.speed,
        )
    except (AttributeError, KeyError, OSError, TypeError, ValueError) as exc:
        return ReplayOutcome(
            ReplayStatus.REJECTED,
            reason=f"live replay preflight rejected: {exc}",
        )

    print("\n" + "=" * 60)
    print("Replay — SharedStorage architecture (arm_loop + hand_loop)")
    print("=" * 60)

    context = mp.get_context("spawn")
    shared = SharedStorage.create(
        prefix=f"dexmani_replay_{os.getpid()}",
        config=SharedStorageConfig.from_runtime(runtime),
        mp_context=context,
    )
    processes: list[Any] = []
    replayer: TrajectoryReplayer | None = None
    outcome = ReplayOutcome(ReplayStatus.REJECTED, reason="replay did not start")
    try:
        arm_config = ArmLoopConfig.from_runtime(runtime)
        hand_available = trajectory.has_hand and not config.no_hand
        processes.append(context.Process(target=_arm_loop, args=(shared, arm_config), name="arm", daemon=False))
        if hand_available:
            from dexmani_real.robot.hand_process import HandProcessConfig

            hand_config = HandProcessConfig.from_runtime(runtime)
            processes.append(context.Process(target=_hand_loop, args=(shared, hand_config), name="hand", daemon=False))

        require_transition(shared, SafetyState.DISARMED)
        for process in processes:
            process.start()

        timeouts = runtime.safety.readiness_timeouts_s
        ready_checks: list[tuple[str, Any, float]] = [("arm", shared.arm_ready, float(timeouts["arm"]))]
        if hand_available:
            ready_checks.append(("hand", shared.hand_ready, float(timeouts["hand"])))
        workers_ready = wait_subsystem_ready(shared, ready_checks, processes)
        if not workers_ready:
            shared.error_state.value = True
            require_transition(shared, SafetyState.FAULT)
            outcome = ReplayOutcome(ReplayStatus.FAULT, reason="worker readiness failed")
        else:
            heartbeat_timeouts = dict(runtime.safety.heartbeat_timeouts)

            def health_check() -> str | None:
                return _live_worker_health_issue(shared, processes, heartbeat_timeouts)

            replayer = TrajectoryReplayer(
                trajectory,
                shared,
                speed=config.speed,
                no_hand=config.no_hand,
                max_frames=config.max_frames,
                runtime=runtime,
                health_check=health_check,
            )
            replayer.setup()
            latched_status = _latched_fault_status(shared)
            health_issue = None if latched_status is not None else health_check()
            if latched_status is not None:
                shared.error_state.value = True
                require_transition(shared, SafetyState.FAULT)
                outcome = ReplayOutcome(latched_status, reason="fault latched before replay could arm")
            elif health_issue:
                shared.error_state.value = True
                require_transition(shared, SafetyState.FAULT)
                outcome = ReplayOutcome(ReplayStatus.FAULT, reason=health_issue)
            else:
                require_transition(shared, SafetyState.ARMED)
                try:
                    outcome = replayer.run()
                except KeyboardInterrupt:
                    print("\nInterrupted by user")
                    if replayer.can_offer_home:
                        shared.error_state.value = True
                        require_transition(shared, SafetyState.FAULT)
                        outcome = ReplayOutcome(
                            ReplayStatus.FAULT,
                            replayer.partial_data,
                            "replay interrupted before a measured hold could be confirmed",
                        )
                    else:
                        outcome = ReplayOutcome(ReplayStatus.USER_QUIT, replayer.partial_data, "KeyboardInterrupt")

                if (
                    outcome.status is ReplayStatus.COMPLETED
                    and not shared.error_state.value
                    and replayer.can_offer_home
                ):
                    home_outcome = _offer_return_home(
                        shared,
                        replayer,
                        runtime,
                        arm_config,
                        hand_available=hand_available,
                        health_check=health_check,
                    )
                    if home_outcome is not None:
                        status, reason = home_outcome
                        outcome = ReplayOutcome(status, outcome.replay_data, reason)
    except Exception:
        logger.error("live replay session failed", exc_info=True)
        shared.error_state.value = True
        require_transition(shared, SafetyState.FAULT)
        raise
    finally:
        if replayer is not None:
            try:
                replayer.shutdown()
            except Exception:
                logger.error("replay controller shutdown failed", exc_info=True)
                shared.error_state.value = True
                require_transition(shared, SafetyState.FAULT)
                outcome = ReplayOutcome(
                    ReplayStatus.FAULT,
                    outcome.replay_data,
                    outcome.reason or "replay controller shutdown failed",
                )

        if outcome.status is ReplayStatus.ESTOP:
            shared.estop_request.value = True
            shared.error_state.value = True
            require_transition(shared, SafetyState.FAULT)
        elif outcome.status is ReplayStatus.FAULT:
            shared.error_state.value = True
            require_transition(shared, SafetyState.FAULT)

        started = [process for process in processes if process.pid is not None]
        try:
            shutdown_report = shutdown_processes(
                shared,
                started,
                graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
                disarm_if_clean=outcome.status not in (ReplayStatus.ESTOP, ReplayStatus.FAULT),
            )
        except RuntimeError:
            logger.critical("child process remains alive; leaving SharedStorage linked", exc_info=True)
            raise
        outcome = _post_shutdown_outcome(outcome, shutdown_report)

    if replayer is not None:
        _evaluate_replay(trajectory, outcome.replay_data, config)
    return outcome
