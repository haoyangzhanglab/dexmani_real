"""Own the local policy/robot, sensor processes and operator lifecycle."""

import logging
import multiprocessing as mp
import os
import time
from dataclasses import replace

from dexmani_real.calibration.camera.extrinsics import (
    CameraExtrinsics,
    load_optional_camera_extrinsics,
)
from dexmani_real.config.experiment import (
    resolve_runtime_table, validate_robot_config, validate_control_rate,
)
from dexmani_real.deployment.config import (
    validate_max_running_s,
    validate_num_episodes,
    validate_policy_runtime_compatibility,
)
from dexmani_real.deployment.observation import build_observation_camera, build_observation_kinematics
from dexmani_real.deployment.operator import PolicyOperator
from dexmani_real.deployment.runner import PolicyRunner
from dexmani_real.ipc.channels import SensorChannelsConfig
from dexmani_real.runtime.state import RuntimeState
from dexmani_real.recording.recorder import AsyncEpisodeRecorder
from dexmani_real.utils.episode_results import EpisodeResults, error_detail
from dexmani_real.robot.action import ActionRealizer
from dexmani_real.robot.arm_homing import build_home_planner
from dexmani_real.robot.robot import DexManiRobot
from dexmani_real.runtime.processes import shutdown_local_runtime
from dexmani_real.runtime.safety import (
    request_emergency_stop,
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


def _warmup_policy(model, runtime, execution_config):
    model.submit(
        "load",
        samples=5,
        rgb_hw=(runtime.camera.height, runtime.camera.width),
        rtc_delay=execution_config.prefetch_steps
        if execution_config.execution_mode == "rtc"
        else 0,
    )
    while (completion := model.poll()) is None:
        time.sleep(0.005)
    if completion[1].error is not None:
        raise completion[1].error
    logger.debug("Model warmup durations: %s", completion[1].value)


def run_policy_deployment(
    runtime,
    policy_config,
    *,
    prefix=None,
    max_running_s=None,
    num_episodes=20,
    recording_config=None,
    results_dir=None,
    camera_calibration_path=None,
    save_run_config=None,
):
    """Run a session with independently optional Raw recording and evaluation summaries."""
    configure_logging()
    from dexmani_real.deployment.inference import InferenceWorker

    info = policy_config.info
    cloud_recipe = validate_policy_runtime_compatibility(info, runtime)
    max_running_s = validate_max_running_s(max_running_s)
    execution_config = runtime.execution.validate(info)
    validate_robot_config(runtime)
    validate_control_rate(runtime, 1.0 / info.control_dt_s)
    if cloud_recipe is not None:
        runtime = replace(runtime, pointcloud=cloud_recipe)
    runtime = resolve_runtime_table(runtime, pointcloud=cloud_recipe)
    num_episodes = validate_num_episodes(num_episodes)
    print(
        f"执行：{info.checkpoint_path.name} ({info.weights}) | NFE={info.inference_steps} | "
        f"A_exec={execution_config.action_steps} | {execution_config.execution_mode} | "
        f"{1.0 / info.control_dt_s:g} Hz"
    )
    print("正在初始化模型和设备…", flush=True)
    results = (
        EpisodeResults(results_dir, "policy")
        if results_dir is not None
        else None
    )
    shared = supervisor = robot = model = operator = recorder = None
    failure = None
    model_close = dict(status="completed", reason="model_not_created")
    clean = shutdown_clean = home_fault = False
    try:
        fields = set(info.observation_fields)
        cloud = "point_cloud" in fields
        # Parameters, cloud production and RGB-D recording share the camera worker.
        camera = bool(
            fields & {"point_cloud", "rgb", "camera_intrinsic", "camera_extrinsic"}
        ) or recording_config is not None
        if camera:
            runtime.camera.validate()
        camera_calibration = (
            CameraExtrinsics(camera_calibration_path)
            if cloud or "camera_extrinsic" in fields
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
        supervisor = (
            RuntimeSupervisor(shared, runtime.safety.readiness_timeouts_s)
            if shared.sensors is not None else None
        )
        robot = DexManiRobot(
            shared, runtime, check_services=supervisor.check if supervisor else None
        )
        model = InferenceWorker(lambda: _load_configured_policy(policy_config, execution_config))
        _warmup_policy(model, runtime, execution_config)
        logger.debug(
            "Validated policy: inputs=%s action=%s dt=%s N/H/A_exec=%s/%s/%s mode=%s d=%s beta=%s recording=%s; "
            "cadence/cloud recipe from saved Policy, physical configuration from Real",
            info.observation_fields,
            info.action_mode,
            info.control_dt_s,
            info.n_obs_steps,
            info.horizon,
            execution_config.action_steps,
            execution_config.execution_mode,
            execution_config.prefetch_steps,
            execution_config.rtc_guidance_cap,
            recording_config is not None,
        )
        kinematics = build_observation_kinematics(info, runtime)
        realizer = ActionRealizer.for_mode(runtime, info.action_mode)
        home_planner = build_home_planner(runtime)
        if save_run_config is not None:
            save_run_config(runtime, execution_config)
        recorder = (
            AsyncEpisodeRecorder(
                recording_config.data_dir,
                control_hz=1.0 / info.control_dt_s,
                rgb_shape=(runtime.camera.height, runtime.camera.width, 3),
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
        if supervisor is not None:
            supervisor.start(sensors)
        camera_parameters = build_observation_camera(
            shared.sensors, runtime.camera, fields, camera_calibration,
        )
        require_transition(shared, SafetyState.ARMED)
        operator = PolicyOperator(
            shared,
            runtime,
            home_planner,
            robot=robot,
            idle_for_tactile_reset=lambda: (
                runner.run_id is None
                and runner.preparing_epoch is None
                and not shared.start_request
                and model.future is None
                and (recorder is None or not recorder.is_recording)
            ),
        )
        operator.keyboard.start()
        robot.check_services = lambda: (
            (supervisor is None or supervisor.check()) and operator.keyboard.healthy
        )
        runner = PolicyRunner(
            shared,
            runtime,
            info,
            robot=robot,
            realizer=realizer,
            recorder=recorder,
            poll_operator=operator.poll,
            discard_pending_start=operator.discard_pending_start,
            model_runtime=model,
            kinematics=kinematics,
            max_running_s=max_running_s,
            num_episodes=num_episodes,
            task_label=recording_config.task_label if recording_config is not None else None,
            camera_calibration=camera_calibration,
            camera_parameters=camera_parameters,
            results=results,
        )
        print(
            "就绪（无需回车）：H 回初始位 → B 开始 | S 结束本轮 | Q 退出 | Esc 急停 | "
            "T 请求复位触觉（空闲且无接触）。",
            flush=True,
        )
        runner.run()
        clean = bool(
            shared.quit_requested
            and not shared.estop_request
            and not shared.error_state
        )
    except KeyboardInterrupt as exc:
        if shared is not None:
            request_emergency_stop(shared)
        failure = exc
    except Exception as exc:
        failure = exc
        if shared is not None:
            shared.error_state = True
        logger.exception("policy session failed")
    finally:
        # Cleanup shares one budget; native calls/threads can still outlive it.
        shutdown_deadline = time.monotonic() + runtime.safety.shutdown_timeout_s
        cleanup_errors = []
        if shared is not None:
            try:
                revoke_motion(shared, reason=RunEndReason.RUNTIME_SHUTDOWN)
            except Exception as exc:
                cleanup_errors.append(error_detail("revoke_motion", exc))
        if robot is not None and robot.stop_required:
            try:
                robot.stop()
            except Exception as exc:
                cleanup_errors.append(error_detail("stop", exc))
        if model is not None:
            try:
                model.close()
            except Exception as exc:
                cleanup_errors.append(error_detail("model_close_submit", exc))
        try:
            if recorder is not None:
                try:
                    recorder.close()
                except Exception as exc:
                    cleanup_errors.append(error_detail("recorder_close", exc))
            if shared is not None:
                shutdown_clean = shutdown_local_runtime(
                    robot, supervisor,
                    shared=shared,
                    keyboard=operator.keyboard if operator else None,
                    timeout_s=max(0.0, shutdown_deadline - time.monotonic()),
                )
            else:
                shutdown_clean = True
        except Exception as exc:
            cleanup_errors.append(error_detail("shutdown", exc))
        finally:
            if model is not None:
                model_close = model.close_result(shutdown_deadline - time.monotonic())
            shutdown_clean = (shutdown_clean and not cleanup_errors
                              and model_close["status"] == "completed")
            if results is not None:
                results.session["errors"].extend(cleanup_errors)
            home_results = operator.home_results if operator is not None else []
            home_fault = any(item["outcome"] in ("failed", "fault") for item in home_results)
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
                        else f"model_close_{model_close['status']}"
                        if model_close["status"] != "completed"
                        else "shutdown_failed"
                        if not shutdown_clean
                        else None,
                        shutdown_clean=shutdown_clean,
                        model_close=model_close,
                        home_results=home_results,
                    )
                except Exception as exc:
                    failure = failure or exc
                    logger.exception("session result publication failed")
    if model_close["status"] == "pending":
        logger.warning("Model cleanup remains pending; Python/CUDA exit is not bounded")
    return int(not clean or failure is not None or not shutdown_clean or home_fault)
