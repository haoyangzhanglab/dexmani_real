"""Policy deployment process ownership and operator lifecycle."""
import multiprocessing as mp
import math
import os
import threading
import time
from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.deployment.config import (FingertipAssemblerConfig, PolicyRuntimeConfig,
    RolloutRecordingConfig, validate_policy_runtime_compatibility, validate_max_running_s, validate_num_episodes)
from dexmani_real.deployment.runner import policy_runner_loop
from dexmani_real.robot.arm_homing import build_policy_home_planner, home_policy_robot
from dexmani_real.planning import XArm7MotionPlanner
from dexmani_real.runtime.operator_input import KeyboardInput, OperatorCommand
from dexmani_real.runtime.safety import (SafetyState, StopRequest, request_policy_start, request_policy_stop,
    RunEndReason, require_transition, revoke_motion_if_run_id)
from dexmani_real.ipc.channels import RuntimeChannels, RuntimeChannelsConfig
from dexmani_real.recording.io_worker import RecorderIOConfig, recorder_io_loop
from dexmani_real.robot.arm_worker import arm_loop
from dexmani_real.robot.hand_worker import hand_loop
from dexmani_real.runtime.processes import shutdown_processes_verified, stop_processes_verified
from dexmani_real.runtime.supervisor import start_processes, check_processes
from dexmani_real.sensor.camera.worker import CameraLoopConfig, camera_loop
from dexmani_real.sensor.pointcloud_worker import PointCloudLoopConfig, pointcloud_loop
from dexmani_real.utils.log import get_logger
logger = get_logger(__name__)
_POLL_S = 0.05
_ROLLOUT_RECORDER_FRAME_MARGIN = 4

def _rollout_recorder_config(
    runtime: ExperimentConfig,
    rollout: RolloutRecordingConfig,
    worker_config: PolicyRuntimeConfig,
    max_running_s: float,
    num_episodes: int,
) -> RecorderIOConfig:
    """Build the recorder capacity contract for one recorded rollout session.

    The frame capacity is DERIVED from the run-owner episode budget; recording
    does not own or re-validate the run plan. The recorder-owned
    ``provenance_*`` attributes carry the resolved experimental conditions
    (pinned artifact, effective inference steps, seed, budget) so each
    published raw episode is self-describing without any git or checksum
    provenance.
    """
    control_hz = 1.0 / float(worker_config.spec.control_dt_s)
    max_frames = (
        math.ceil(float(max_running_s) * control_hz)
        + _ROLLOUT_RECORDER_FRAME_MARGIN
    )
    return RecorderIOConfig(
        data_dir=rollout.data_dir,
        max_frames=max_frames,
        control_hz=control_hz,
        min_frames=1,
        provenance={
            "workflow": "policy_eval",
            "policy_selector": worker_config.experiment,
            "checkpoint_name": worker_config.artifact,
            "inference_steps": str(worker_config.inference_steps),
            "n_action_steps": str(worker_config.spec.n_action_steps),
            "seed": str(worker_config.seed),
            "max_running_s": f"{float(max_running_s):.17g}",
            "num_episodes": str(int(num_episodes)),
        },
    )


def run_policy_deployment(runtime, policy_spec, worker_config, execute, *, prefix=None,
                          max_running_s=None, num_episodes=1, recording_config=None):
    validate_policy_runtime_compatibility(policy_spec, runtime)
    max_running_s = validate_max_running_s(max_running_s)
    num_episodes = validate_num_episodes(num_episodes)
    if recording_config is not None and (not execute or max_running_s is None):
        raise ValueError("recorded evaluation requires execute and a finite run budget")
    fields = {f.name: f for f in policy_spec.observation_fields}
    cloud = "point_cloud" in fields
    # Cloud production needs a camera worker; pointcloud-only rows need no source-frame lookup.
    camera = cloud or "rgb" in fields or recording_config is not None
    points = fields["point_cloud"].shape[0] if cloud else runtime.pointcloud.num_points
    ctx = mp.get_context("spawn")
    shared = RuntimeChannels.create(prefix=prefix or f"dexmani_policy_{os.getpid()}", mp_context=ctx,
        config=RuntimeChannelsConfig.from_runtime(runtime, pointcloud_num_points=points))
    started = []
    operator_stop = threading.Event()
    operator_thread = None
    try:
        policy = ctx.Process(name="policy", target=policy_runner_loop, args=(shared, runtime, worker_config,
            execute, max_running_s, num_episodes, recording_config,
            FingertipAssemblerConfig.from_runtime(runtime) if "fingertip_points" in fields else None))
        start_processes(shared, [policy], runtime.safety.readiness_timeouts_s, started)
        sensors = [ctx.Process(name="arm", target=arm_loop, args=(shared, runtime.arm)),
                   ctx.Process(name="hand", target=hand_loop, args=(shared, runtime.hand))]
        if camera:
            sensors.append(ctx.Process(name="camera", target=camera_loop, args=(shared, CameraLoopConfig.from_runtime(runtime))))
        if cloud:
            sensors.append(ctx.Process(name="pointcloud", target=pointcloud_loop,
                args=(shared, PointCloudLoopConfig.from_runtime(runtime, num_points=points))))
        if recording_config:
            sensors.append(ctx.Process(name="recorder", target=recorder_io_loop,
                args=(shared, _rollout_recorder_config(runtime, recording_config, worker_config, max_running_s, num_episodes))))
        start_processes(shared, sensors, runtime.safety.readiness_timeouts_s, started)
        require_transition(shared, SafetyState.ARMED)
        planner = build_policy_home_planner(runtime) if execute else None
        operator_thread = threading.Thread(target=run_operator_control,
            args=(shared, runtime, planner), kwargs=dict(stop_event=operator_stop, execute=execute))
        operator_thread.start()
        while shared.is_running.value and not shared.quit_requested.value:
            if shared.error_state.value or shared.estop_request.value or not check_processes(shared, started):
                break
            if not operator_thread.is_alive():
                shared.workflow_failed.value = True
                break
            if max_running_s is not None:
                with shared.motion_lock:
                    epoch = int(shared.run_id.value)
                    start_ns = int(shared.run_started_monotonic_ns.value)
                if start_ns and time.monotonic_ns()-start_ns >= int(max_running_s*1e9):
                    revoke_motion_if_run_id(shared, epoch, reason=RunEndReason.TIMEOUT)
            time.sleep(0.02)
    except KeyboardInterrupt:
        shared.estop_request.value = True
    except Exception:
        shared.workflow_failed.value = True
        logger.exception("policy session failed")
    finally:
        operator_stop.set()
        if operator_thread is not None:
            operator_thread.join(timeout=5)
            if operator_thread.is_alive():
                stop_processes_verified(shared, started, graceful_timeout_s=runtime.safety.shutdown_timeout_s)
                raise RuntimeError("operator thread still uses channels; refusing SHM release")
        report = shutdown_processes_verified(shared, started, disarm_if_clean=True,
            graceful_timeout_s=max(5.0, runtime.safety.shutdown_timeout_s),
            service_process_names={"policy", "camera", "pointcloud", "recorder"})
    return 1 if shared.error_state.value or shared.workflow_failed.value or not report.shared_closed else 0

def _request_immediate_stop(shared: RuntimeChannels) -> None:
    """Fence live motion when S/Q arrives while this thread is blocked by H."""
    if not request_policy_stop(shared):
        shared.error_state.value = True


def _request_immediate_quit(shared: RuntimeChannels) -> None:
    """Apply Q's motion fence before asking the supervisor to shut down."""
    if not request_policy_stop(shared, reason=RunEndReason.QUIT):
        shared.error_state.value = True
    shared.quit_requested.value = True


def run_operator_control(
    shared: RuntimeChannels,
    runtime: ExperimentConfig,
    planner: XArm7MotionPlanner | None,
    *,
    stop_event: threading.Event,
    execute: bool,
) -> None:
    """Keyboard thread target: map operator keys to shared flags / home.

    B -> ``start_request``, S -> ``stop_request`` + motion fence,
    Q -> ``quit_requested``, ESC -> ``estop_request``. S/Q/ESC fire as
    immediate callbacks so they stay responsive while this thread blocks in
    H. H is enabled only when a caller supplies a home planner. The thread
    exits when *stop_event* is set, when the runtime stops, or on ESC.
    Q keeps the listener alive through file finalization. C/D belong to teleop (PAUSE/DISCARD); policy deployment
    ignores them with a warning and assigns no task meaning to any key —
    task success is judged offline from the saved raw episode.
    """
    if not isinstance(execute, bool):
        raise TypeError("execute must be a boolean")
    if execute != (planner is not None):
        raise ValueError("execute must match physical home availability")
    keyboard = KeyboardInput(
        estop_callback=lambda: setattr(shared.estop_request, "value", True),
        stop_callback=lambda: _request_immediate_stop(shared),
        quit_callback=lambda: _request_immediate_quit(shared),
    )
    try:
        keyboard.start()
    except Exception:
        # Without the e-stop keyboard the deployment must not run: fail closed
        # so the supervisor observes a sticky fault and shuts down.
        logger.error(
            "operator: keyboard failed to start; latching error_state", exc_info=True
        )
        shared.error_state.value = True
        return
    try:
        while not stop_event.is_set() and shared.is_running.value:
            if keyboard.estop_latched or not keyboard.healthy:
                shared.estop_request.value = True
                return
            # A physical B must be a fresh, post-home confirmation.  H blocks
            # this thread while the arm moves, so begin events from the same
            # drained batch must not survive a successful home sequence.
            discard_begin_in_batch = False
            signals = keyboard.poll(timeout=_POLL_S)
            # Lifecycle-changing signals suppress Home and Begin in the same batch.
            # C/D (PAUSE/DISCARD) belong to teleop and are true no-ops here, so
            # they must not fence H; ESC is fenced by the estop latch/callback.
            stop_in_batch = any(
                signal in {OperatorCommand.STOP, OperatorCommand.QUIT}
                for signal in signals
            )
            for signal in signals:
                if signal is OperatorCommand.BEGIN:
                    if stop_in_batch:
                        logger.warning(
                            "operator: ignored B received in the same batch as S/Q"
                        )
                        continue
                    if (
                        discard_begin_in_batch
                        or shared.workflow_failed.value
                        or not request_policy_start(
                            shared,
                            require_physical_home=planner is not None,
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
                    logger.warning(
                        "operator: C is not used in policy deployment; ignored"
                    )
                elif signal is OperatorCommand.DISCARD:
                    logger.warning(
                        "operator: D is not used in policy deployment; ignored"
                    )
                elif signal is OperatorCommand.HOME:
                    if planner is None:
                        logger.warning("operator: H is disabled in policy deployment")
                        continue
                    if stop_in_batch:
                        logger.warning(
                            "operator: ignored H received in the same batch as S/Q"
                        )
                        continue
                    with shared.motion_lock:
                        home_allowed = (not shared.quit_requested.value and
                            int(shared.safety_state.value) == int(SafetyState.ARMED))
                        shared.physical_home_completed.value = False
                        if home_allowed:
                            # H follows any older inactive S but cannot erase an
                            # S arriving after this atomic preparation.
                            shared.start_request.value = False
                            shared.stop_request.value = int(StopRequest.NONE)
                    if not home_allowed:
                        logger.warning("operator: ignored H unless safety is ARMED")
                        continue
                    completed = home_policy_robot(
                        shared,
                        runtime,
                        planner,
                        abort_requested=lambda: bool(
                            stop_event.is_set()
                            or not shared.is_running.value
                            or shared.quit_requested.value
                            or shared.error_state.value
                            or shared.estop_request.value
                            or int(shared.stop_request.value) != int(StopRequest.NONE)
                        ),
                    )
                    with shared.motion_lock:
                        authorized = bool(
                            completed
                            and shared.is_running.value
                            and not shared.quit_requested.value
                            and not shared.workflow_failed.value
                            and not shared.error_state.value
                            and not shared.estop_request.value
                            and int(shared.stop_request.value) == int(StopRequest.NONE)
                            and int(shared.safety_state.value) == int(SafetyState.ARMED)
                        )
                        # Stop/start requests share this lock, so a completed H
                        # cannot resurrect authorization after a newer S.
                        shared.physical_home_completed.value = authorized
                    if authorized:
                        logger.info(
                            "operator: physical home sequence completed; "
                            "press B to start"
                        )
                    else:
                        logger.warning(
                            "operator: physical home sequence did not authorize; "
                            "B remains disabled for the next episode"
                        )
                    # HOME blocks while hand/arm homing completes. Drop stale
                    # H and B events, but preserve S/Q/ESC so an operator can
                    # still stop, quit, or e-stop immediately afterwards.
                    keyboard.drain_signal(OperatorCommand.HOME)
                    keyboard.drain_signal(OperatorCommand.BEGIN)
                    discard_begin_in_batch = True
                elif signal is OperatorCommand.QUIT:
                    # Listener stays available through final recording cleanup.
                    continue
                elif signal is OperatorCommand.EMERGENCY_STOP:
                    shared.estop_request.value = True
                    return
    finally:
        keyboard.stop()
