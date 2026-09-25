"""Policy deployment process ownership and operator lifecycle."""

import json
import math
import multiprocessing as mp
import os
import threading
import time

from dexmani_real.calibration.camera.extrinsics import CameraExtrinsics
from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.deployment.config import (
    FingertipAssemblerConfig,
    PolicyRuntimeConfig,
    RolloutRecordingConfig,
    validate_max_running_s,
    validate_num_episodes,
    validate_policy_runtime_compatibility,
)
from dexmani_real.deployment.operator import PolicyOperator
from dexmani_real.deployment.runner import run_policy_worker
from dexmani_real.ipc.channels import RuntimeChannels, RuntimeChannelsConfig
from dexmani_real.recording.io_worker import RecorderWorkerConfig, run_recorder_worker
from dexmani_real.recording.recorder import HARD_MAX_RECORD_FRAMES
from dexmani_real.robot.arm_homing import build_policy_home_planner
from dexmani_real.robot.arm_worker import run_arm_worker
from dexmani_real.robot.hand_worker import run_hand_worker
from dexmani_real.runtime.processes import stop_processes_verified
from dexmani_real.runtime.safety import (
    SafetyState,
    require_transition,
)
from dexmani_real.runtime.supervisor import RuntimeSupervisor
from dexmani_real.sensor.camera.worker import run_camera_worker
from dexmani_real.sensor.pointcloud_worker import PointCloudWorkerConfig, run_pointcloud_worker
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


def _rollout_recorder_config(
    rollout: RolloutRecordingConfig,
    worker_config: PolicyRuntimeConfig,
    max_running_s: float,
    num_episodes: int,
    *,
    camera_calibration: CameraExtrinsics,
    pointcloud_config: PointCloudWorkerConfig | None,
) -> RecorderWorkerConfig:
    """Record resolved policy experiment settings alongside the raw observations."""
    control_hz = 1.0 / float(worker_config.spec.control_dt_s)
    return RecorderWorkerConfig(
        data_dir=rollout.data_dir,
        control_hz=control_hz,
        min_frames=1,
        camera_calibration=camera_calibration,
        provenance={
            "workflow": "policy_eval",
            **(
                {
                    "pointcloud_config_json": json.dumps(pointcloud_config.pointcloud.to_dict()),
                    "pointcloud_table_plane_abcd_json": json.dumps(
                        pointcloud_config.table_plane_abcd
                    ),
                }
                if pointcloud_config is not None
                else {}
            ),
            "policy_selector": worker_config.experiment,
            "checkpoint_name": worker_config.artifact,
            "inference_steps": str(worker_config.inference_steps),
            "n_action_steps": str(worker_config.spec.n_action_steps),
            "seed": str(worker_config.seed),
            "max_running_s": f"{float(max_running_s):.17g}",
            "num_episodes": str(int(num_episodes)),
        },
    )


def run_policy_deployment(
    runtime,
    policy_spec,
    worker_config,
    execute,
    *,
    prefix=None,
    max_running_s=None,
    num_episodes=1,
    recording_config=None,
):
    validate_policy_runtime_compatibility(policy_spec, runtime)
    max_running_s = validate_max_running_s(max_running_s)
    num_episodes = validate_num_episodes(num_episodes)
    if recording_config is not None and (not execute or max_running_s is None):
        raise ValueError("recorded evaluation requires execute and a finite run budget")
    if recording_config is not None:
        control_dt_s = float(worker_config.spec.control_dt_s)
        requested_rows = math.ceil(float(max_running_s) / control_dt_s)
        if requested_rows >= HARD_MAX_RECORD_FRAMES:
            raise ValueError(
                f"policy recording requests {requested_rows} rows; require rows < "
                f"hard limit {HARD_MAX_RECORD_FRAMES} "
                f"(max_running_s={max_running_s}, control_dt_s={control_dt_s}). "
                "Shorten the episode budget instead of increasing the recorder hard guard."
            )
    fields = {f.name: f for f in policy_spec.observation_fields}
    cloud = "point_cloud" in fields
    # Cloud production needs a camera worker; pointcloud-only rows need no source-frame lookup.
    camera = cloud or "rgb" in fields or recording_config is not None
    points = fields["point_cloud"].shape[0] if cloud else runtime.pointcloud.num_points
    camera_calibration = CameraExtrinsics() if cloud or recording_config is not None else None
    pointcloud_config = (
        PointCloudWorkerConfig.from_runtime(
            runtime,
            pointcloud=PointCloudConfig.from_dict(policy_spec.pointcloud_config),
            camera_calibration=camera_calibration,
        )
        if cloud
        else None
    )
    ctx = mp.get_context("spawn")
    shared = RuntimeChannels.create(
        prefix=prefix or f"dexmani_policy_{os.getpid()}",
        mp_context=ctx,
        config=RuntimeChannelsConfig.from_runtime(runtime, pointcloud_num_points=points),
    )
    supervisor = RuntimeSupervisor(shared, runtime.safety.readiness_timeouts_s)
    operator_stop = threading.Event()
    operator_thread = None
    try:
        policy = ctx.Process(
            name="policy",
            target=run_policy_worker,
            args=(
                shared,
                runtime,
                worker_config,
                execute,
                max_running_s,
                num_episodes,
                recording_config,
                FingertipAssemblerConfig.from_runtime(runtime)
                if "fingertip_points" in fields
                else None,
            ),
        )
        supervisor.start([policy])
        sensors = [
            ctx.Process(name="arm", target=run_arm_worker, args=(shared, runtime.arm)),
            ctx.Process(name="hand", target=run_hand_worker, args=(shared, runtime.hand)),
        ]
        if camera:
            sensors.append(
                ctx.Process(
                    name="camera",
                    target=run_camera_worker,
                    args=(shared, runtime.camera),
                )
            )
        if cloud:
            sensors.append(
                ctx.Process(
                    name="pointcloud",
                    target=run_pointcloud_worker,
                    args=(shared, pointcloud_config),
                )
            )
        if recording_config:
            sensors.append(
                ctx.Process(
                    name="recorder",
                    target=run_recorder_worker,
                    args=(
                        shared,
                        _rollout_recorder_config(
                            recording_config,
                            worker_config,
                            max_running_s,
                            num_episodes,
                            camera_calibration=camera_calibration,
                            pointcloud_config=pointcloud_config,
                        ),
                    ),
                )
            )
        supervisor.start(sensors)
        require_transition(shared, SafetyState.ARMED)
        planner = build_policy_home_planner(runtime) if execute else None
        operator = PolicyOperator(
            shared, runtime, planner, stop_event=operator_stop, execute=execute
        )
        operator_thread = threading.Thread(target=operator.run)
        operator_thread.start()
        while shared.is_running.value and not shared.quit_requested.value:
            if shared.error_state.value or shared.estop_request.value or not supervisor.check():
                break
            if not operator_thread.is_alive():
                shared.workflow_failed.value = True
                break
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
                stop_processes_verified(
                    shared,
                    supervisor.started_processes,
                    graceful_timeout_s=runtime.safety.shutdown_timeout_s,
                )
                raise RuntimeError("operator thread still uses channels; refusing SHM release")
        report = supervisor.shutdown(
            disarm_if_clean=True,
            graceful_timeout_s=max(5.0, runtime.safety.shutdown_timeout_s),
            service_process_names={"policy", "camera", "pointcloud", "recorder"},
        )
    return (
        1
        if shared.error_state.value or shared.workflow_failed.value or not report.shared_closed
        else 0
    )
