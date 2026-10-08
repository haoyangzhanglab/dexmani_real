"""Own the local policy/robot, sensor processes and operator lifecycle."""

import logging
import multiprocessing as mp
import os
from dataclasses import replace

from dexmani_real.calibration.camera.extrinsics import (
    CameraExtrinsics,
    load_optional_camera_extrinsics,
)
from dexmani_real.config.experiment import resolve_runtime_table, validate_robot_config
from dexmani_real.deployment.config import (
    validate_max_running_s,
    validate_num_episodes,
    validate_policy_runtime_compatibility,
    validate_warmup_budget,
)
from dexmani_real.deployment.observation import build_observation_kinematics
from dexmani_real.deployment.operator import PolicyOperator
from dexmani_real.deployment.runner import PolicyRunner
from dexmani_real.ipc.channels import SensorChannelsConfig
from dexmani_real.runtime.state import RuntimeState
from dexmani_real.recording.recorder import AsyncEpisodeRecorder
from dexmani_real.recording.results import SessionResults, error_detail
from dexmani_real.robot.action import ActionRealizer
from dexmani_real.robot.arm_homing import build_home_planner
from dexmani_real.robot.robot import DexManiRobot
from dexmani_real.runtime.processes import shutdown_local_runtime
from dexmani_real.runtime.safety import (
    RunEndReason,
    SafetyState,
    require_transition,
    revoke_motion,
)
from dexmani_real.runtime.supervisor import RuntimeSupervisor
from dexmani_real.sensor.camera.worker import run_camera_worker
from dexmani_real.sensor.pointcloud_worker import PointCloudWorkerConfig, run_pointcloud_worker
from dexmani_real.utils.log import configure_logging


logger = logging.getLogger(__name__)


def _load_configured_policy(policy_config, execution_config):
    info = policy_config.info
    from dexmani_policy.deployment import load_policy

    loaded = load_policy(
        policy_config.config, info, device=policy_config.device, seed=policy_config.seed
    )
    try:
        loaded.configure_execution(
            execution_config.execution_mode, execution_config.rtc_guidance_cap
        )
        return loaded
    except BaseException:
        try:
            loaded.close()
        except Exception:
            logger.exception("model cleanup after configure failure also failed")
        raise


def _warmup_policy(model, runtime, info, execution_config, *, max_running_s=None):
    model.submit(
        "load",
        samples=5,
        rgb_hw=(runtime.camera.height, runtime.camera.width),
        rtc_delay=execution_config.prefetch_steps
        if execution_config.execution_mode == "rtc"
        else 0,
    )
    import time

    while (completion := model.poll()) is None:
        time.sleep(0.005)
    if completion[1].error is not None:
        raise completion[1].error
    durations = completion[1].value
    validate_warmup_budget(durations, execution_config, info, max_running_s=max_running_s)
    logger.info("Model warmup durations (not realtime bounds): %s", durations)
    if execution_config.execution_mode != "sync":
        import math

        suggested = math.ceil(max(durations["steady"]) / info.control_dt_s) + 1
        logger.info(
            "Model-only prefetch suggestion d=%d; measure owner prefix/input overhead separately",
            suggested,
        )


def run_policy_deployment(
    runtime,
    policy_config,
    execute,
    *,
    prefix=None,
    max_running_s=None,
    num_episodes=1,
    recording_config=None,
    camera_calibration_path=None,
    save_run_config=None,
):
    """Run one session; without recording_config there are no persistent result artifacts."""
    configure_logging()
    from dexmani_real.deployment.inference import InferenceWorker

    info = policy_config.info
    cloud_recipe = validate_policy_runtime_compatibility(info, runtime)
    max_running_s = validate_max_running_s(max_running_s)
    execution_config = runtime.execution.validate(info, max_running_s=max_running_s)
    validate_robot_config(runtime)
    if cloud_recipe is not None:
        runtime = replace(runtime, pointcloud=cloud_recipe)
    runtime = resolve_runtime_table(runtime, pointcloud=cloud_recipe)
    num_episodes = validate_num_episodes(num_episodes)
    if recording_config is not None and not execute:
        raise ValueError("recorded evaluation requires execute")
    results = (
        SessionResults(recording_config.data_dir, "policy")
        if recording_config is not None
        else None
    )
    shared = supervisor = robot = model = operator = None
    failure = None
    clean = shutdown_clean = home_fault = False
    try:
        fields = set(info.observation_fields)
        cloud = "point_cloud" in fields
        # Cloud production needs a camera worker; recording also retains its source RGB-D.
        camera = cloud or "rgb" in fields or recording_config is not None
        if camera:
            runtime.camera.validate()
        camera_calibration = (
            CameraExtrinsics(camera_calibration_path)
            if cloud
            else load_optional_camera_extrinsics(camera_calibration_path)
            if recording_config is not None
            else None
        )
        pointcloud_config = (
            PointCloudWorkerConfig(
                pointcloud=cloud_recipe,
                camera_calibration=camera_calibration,
                table_plane_abcd=runtime.environment.table.plane_abcd,
            )
            if cloud
            else None
        )
        ctx = mp.get_context("spawn")
        shared = RuntimeState.create(
            prefix=prefix or f"dexmani_policy_{os.getpid()}",
            mp_context=ctx,
            config=SensorChannelsConfig.from_runtime(
                runtime, camera=camera, pointcloud=cloud
            ),
        )
        supervisor = RuntimeSupervisor(shared, runtime.safety.readiness_timeouts_s)
        robot = DexManiRobot(shared, runtime, check_services=supervisor.check)
        model = InferenceWorker(lambda: _load_configured_policy(policy_config, execution_config))
        _warmup_policy(model, runtime, info, execution_config, max_running_s=max_running_s)
        logger.info(
            "Validated policy: inputs=%s action=%s dt=%s N/H/A=%s/%s/%s mode=%s d=%s beta=%s recording=%s; "
            "cadence/cloud recipe from saved Policy, physical configuration from Real",
            info.observation_fields,
            info.action_mode,
            info.control_dt_s,
            info.n_obs_steps,
            info.horizon,
            info.n_action_steps,
            execution_config.execution_mode,
            execution_config.prefetch_steps,
            execution_config.rtc_guidance_cap,
            recording_config is not None,
        )
        kinematics = build_observation_kinematics(info, runtime)
        realizer = ActionRealizer.for_mode(runtime, info.action_mode)
        home_planner = build_home_planner(runtime) if execute else None
        if save_run_config is not None and recording_config is not None:
            save_run_config(runtime, execution_config)
        recorder = (
            AsyncEpisodeRecorder(
                recording_config.data_dir,
                control_hz=1.0 / info.control_dt_s,
                rgb_shape=(runtime.camera.height, runtime.camera.width, 3),
                execution_path=f"worker_grid_{execution_config.execution_mode}_v1",
            )
            if recording_config is not None
            else None
        )
        robot.connect()
        sensors = []
        if camera:
            sensors.append(
                ctx.Process(
                    name="camera", target=run_camera_worker, args=(shared.sensors, runtime.camera)
                )
            )
        if cloud:
            sensors.append(
                ctx.Process(
                    name="pointcloud",
                    target=run_pointcloud_worker,
                    args=(shared.sensors, pointcloud_config),
                )
            )
        supervisor.start(sensors)
        require_transition(shared, SafetyState.ARMED)
        operator = PolicyOperator(
            shared,
            runtime,
            home_planner,
            robot=robot,
            execute=execute,
            idle_for_tare=lambda: (
                runner.run_id is None
                and runner.preparing_epoch is None
                and not shared.start_request
                and model.future is None
                and (recorder is None or not recorder.is_recording)
            ),
        )
        operator.keyboard.start()
        print("空闲且确认手部无接触后按 T 归零触觉；S/Q/ESC 可取消。", flush=True)
        robot.check_services = lambda: supervisor.check() and operator.keyboard.healthy
        runner = PolicyRunner(
            shared,
            runtime,
            info,
            robot=robot,
            realizer=realizer,
            recorder=recorder,
            poll_operator=operator.poll,
            model_runtime=model,
            kinematics=kinematics,
            execute=execute,
            max_running_s=max_running_s,
            num_episodes=num_episodes,
            task_label=recording_config.task_label if recording_config is not None else None,
            camera_calibration=camera_calibration,
            results=results,
        )
        runner.run()
        clean = bool(
            shared.quit_requested
            and not shared.estop_request
            and not shared.error_state
        )
    except KeyboardInterrupt as exc:
        if shared is not None:
            shared.estop_request = True
        failure = exc
    except Exception as exc:
        failure = exc
        if shared is not None:
            shared.error_state = True
        logger.exception("policy session failed")
    finally:
        try:
            if shared is not None:
                revoke_motion(shared, reason=RunEndReason.RUNTIME_SHUTDOWN)
            if supervisor is not None:
                shutdown_clean = shutdown_local_runtime(
                    robot,
                    supervisor,
                    model=model,
                    keyboard=operator.keyboard if operator else None,
                    timeout_s=runtime.safety.shutdown_timeout_s,
                )
            elif shared is not None:
                shutdown_clean = bool(shared.sensors.close())
        except Exception as exc:
            failure = failure or exc
            if results is not None:
                results.session["artifact_errors"].append(error_detail("shutdown", exc))
        finally:
            home_results = operator.home_results if operator is not None else []
            home_fault = any(item["outcome"] in ("failed", "fault") for item in home_results)
            if model is not None and (model.close_error is not None or not model.closed):
                shutdown_clean = False
            if results is not None:
                try:
                    results.finish_session(
                        outcome="finished"
                        if clean and failure is None and shutdown_clean and not home_fault
                        else "fault",
                        reason=str(failure)
                        if failure
                        else "return_home_failed"
                        if home_fault
                        else "shutdown_failed_or_pending"
                        if not shutdown_clean
                        else None,
                        shutdown_clean=shutdown_clean,
                        home_results=home_results,
                    )
                except Exception as exc:
                    failure = failure or exc
                    logger.exception("session result publication failed")
    if model is not None and not model.closed:
        logger.warning("Model cleanup remains pending; Python/CUDA exit is not bounded")
    if model is not None and model.close_error is not None:
        shutdown_clean = False
    return int(not clean or failure is not None or not shutdown_clean or home_fault)
