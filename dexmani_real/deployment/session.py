"""Own the local policy/robot, sensor processes and operator lifecycle."""

import logging
import multiprocessing as mp
import time
from dataclasses import replace
from pathlib import Path

from dexmani_real.calibration.camera.extrinsics import (
    CameraExtrinsics,
    load_optional_camera_extrinsics,
)
from dexmani_real.config.experiment import resolve_runtime_table
from dexmani_real.deployment.config import (
    validate_max_running_s,
    validate_num_episodes,
)
from dexmani_real.deployment.startup import validate_deployment_config
from dexmani_real.deployment.observation import (
    build_observation_camera,
    build_observation_kinematics,
)
from dexmani_real.deployment.operator import PolicyOperator
from dexmani_real.deployment.runner import PolicyRunner
from dexmani_real.ipc.channels import SensorChannels
from dexmani_real.runtime.state import RuntimeState, RunEndReason
from dexmani_real.recording.recorder import AsyncEpisodeRecorder
from dexmani_real.utils.episode_results import EpisodeResults, error_detail
from dexmani_real.robot.action import ActionMapper
from dexmani_real.robot.home import build_home_planner
from dexmani_real.robot.robot import DexManiRobot
from dexmani_real.runtime.processes import SensorProcesses, shutdown_local_runtime
from dexmani_real.sensor.camera.worker import run_camera_worker
from dexmani_real.sensor.pointcloud.worker import PointCloudWorkerConfig, run_pointcloud_worker
from dexmani_real.utils.log import configure_logging

logger = logging.getLogger(__name__)


def load_configured_policy(info, execution_config, *, device, seed):
    from dexmani_policy.deployment import load_policy

    loaded = load_policy(info, device=device, seed=seed)
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


class PolicyDeploymentSession:
    """Own deployment resources while the inference worker serializes model work."""

    def __init__(
        self,
        runtime,
        info,
        *,
        device="cuda:0",
        seed=0,
        max_running_s=None,
        num_episodes=20,
        recording_dir=None,
        results_dir=None,
        camera_calibration_path=None,
        save_run_config=None,
    ):
        self.runtime = runtime
        self.info = info
        self.device = device
        self.seed = seed
        self.max_running_s = max_running_s
        self.num_episodes = num_episodes
        self.recording_dir = recording_dir
        self.results_dir = results_dir
        self.camera_calibration_path = camera_calibration_path
        self.save_run_config = save_run_config

    def run(self):
        """Run a session with independently optional Raw recording and evaluation summaries."""
        cloud_recipe = validate_deployment_config(
            self.info, self.runtime, record_raw=self.recording_dir is not None
        )
        self.max_running_s = validate_max_running_s(self.max_running_s)
        self.num_episodes = validate_num_episodes(self.num_episodes)
        configure_logging()

        if cloud_recipe is not None:
            self.runtime = replace(self.runtime, pointcloud=cloud_recipe)
        self.runtime = resolve_runtime_table(self.runtime, pointcloud=cloud_recipe)
        if self.recording_dir is not None:
            self.recording_dir = Path(self.recording_dir).expanduser().resolve()
        print(
            f"执行：{self.info.checkpoint_path.name} ({self.info.weights}) | NFE={self.info.inference_steps} | "
            f"A_exec={self.runtime.execution.action_steps} | {self.runtime.execution.execution_mode} | "
            f"{1.0 / self.info.control_dt_s:g} Hz"
        )
        print("正在初始化模型和设备…", flush=True)
        self.results = (
            EpisodeResults(self.results_dir, "policy") if self.results_dir is not None else None
        )
        self.runtime_state = self.sensors = self.robot = None
        self.model = self.operator = self.recorder = None
        failure = None
        clean = False
        try:
            self.prepare_resources(cloud_recipe)
            self.start_devices()
            self.run_control()
            clean = bool(
                self.runtime_state.quit_requested
                and not self.runtime_state.estop_request
                and not self.runtime_state.error_state
            )
        except KeyboardInterrupt as exc:
            if self.runtime_state is not None:
                self.runtime_state.request_emergency_stop()
            failure = exc
        except Exception as exc:
            failure = exc
            if self.runtime_state is not None:
                self.runtime_state.mark_error()
            logger.exception("policy session failed")
        finally:
            exit_code = self.close(clean=clean, failure=failure)
        return exit_code

    def prepare_resources(self, cloud_recipe):
        from dexmani_real.deployment.inference import InferenceWorker

        fields = set(self.info.observation_fields)
        cloud = "point_cloud" in fields
        # Parameters, cloud production and RGB-D recording share the camera worker.
        camera = (
            bool(fields & {"point_cloud", "rgb", "camera_intrinsic", "camera_extrinsic"})
            or self.recording_dir is not None
        )
        if cloud or "camera_extrinsic" in fields:
            self.camera_calibration = CameraExtrinsics(self.camera_calibration_path)
        elif self.recording_dir is not None:
            self.camera_calibration = load_optional_camera_extrinsics(self.camera_calibration_path)
        else:
            self.camera_calibration = None
        self.pointcloud_config = (
            PointCloudWorkerConfig(
                pointcloud=cloud_recipe,
                camera_calibration=self.camera_calibration,
                table_plane_abcd=self.runtime.environment.table.plane_abcd,
            )
            if cloud
            else None
        )
        self.process_context = mp.get_context("spawn")
        self.runtime_state = RuntimeState(
            SensorChannels(
                self.process_context,
                camera=self.runtime.camera,
                pointcloud_num_points=self.runtime.pointcloud.num_points if cloud else None,
            )
            if camera
            else None
        )
        self.sensors = SensorProcesses(self.runtime_state) if camera else None
        self.robot = DexManiRobot(
            self.runtime_state,
            self.runtime,
            check_services=self.sensors.check if self.sensors else None,
        )
        self.model = InferenceWorker(
            lambda: load_configured_policy(
                self.info,
                self.runtime.execution,
                device=self.device,
                seed=self.seed,
            )
        )
        self.warmup_policy()
        logger.debug(
            "Validated policy: inputs=%s action=%s dt=%s N/H/A_exec=%s/%s/%s "
            "mode=%s d=%s beta=%s recording=%s; "
            "cadence/cloud recipe from saved Policy, physical configuration from Real",
            self.info.observation_fields,
            self.info.action_mode,
            self.info.control_dt_s,
            self.info.n_obs_steps,
            self.info.horizon,
            self.runtime.execution.action_steps,
            self.runtime.execution.execution_mode,
            self.runtime.execution.prefetch_steps,
            self.runtime.execution.rtc_guidance_cap,
            self.recording_dir is not None,
        )
        self.kinematics = build_observation_kinematics(self.info, self.runtime)
        self.action_mapper = ActionMapper.for_mode(self.runtime, self.info.action_mode)
        self.home_planner = build_home_planner(self.runtime)
        if self.save_run_config is not None:
            self.save_run_config(self.runtime)
        self.recorder = (
            AsyncEpisodeRecorder(
                self.recording_dir,
                control_hz=1.0 / self.info.control_dt_s,
            )
            if self.recording_dir is not None
            else None
        )

    def start_devices(self):
        self.robot.connect()
        if self.sensors is not None:
            self.sensors.start(
                self.process_context.Process(
                    name="camera",
                    target=run_camera_worker,
                    args=(self.runtime_state.sensors, self.runtime.camera),
                ),
                ready=self.runtime_state.sensors.camera_ready,
                timeout_s=self.runtime.safety.camera_ready_timeout_s,
            )
        if self.pointcloud_config is not None:
            self.sensors.start(
                self.process_context.Process(
                    name="pointcloud",
                    target=run_pointcloud_worker,
                    args=(self.runtime_state.sensors, self.pointcloud_config),
                ),
                ready=self.runtime_state.sensors.pointcloud_ready,
                timeout_s=self.runtime.safety.pointcloud_ready_timeout_s,
            )
        self.camera_parameters = build_observation_camera(
            self.runtime_state.sensors,
            self.runtime.camera,
            self.info.observation_fields,
            self.camera_calibration,
        )
        self.runtime_state.arm()
        self.operator = PolicyOperator(
            self.runtime_state,
            self.runtime,
            self.home_planner,
            robot=self.robot,
        )
        self.operator.keyboard.start()
        self.robot.check_services = lambda: (
            (self.sensors is None or self.sensors.check()) and self.operator.keyboard.healthy
        )

    def run_control(self):
        runner = PolicyRunner(
            self.runtime_state,
            self.runtime,
            self.info,
            robot=self.robot,
            action_mapper=self.action_mapper,
            recorder=self.recorder,
            operator=self.operator,
            model_runtime=self.model,
            kinematics=self.kinematics,
            max_running_s=self.max_running_s,
            num_episodes=self.num_episodes,
            camera_calibration=self.camera_calibration,
            camera_parameters=self.camera_parameters,
            results=self.results,
        )
        print(
            "就绪（无需回车）：H 回初始位 → B 开始 | S 结束本轮 | Q 退出 | Esc 急停 | "
            "T 请求复位触觉（空闲且无接触）。",
            flush=True,
        )
        runner.run()

    def close(self, *, clean, failure):
        shutdown_clean = False
        model_close = dict(status="completed", reason="model_not_created")
        # Sensor joins and model cleanup share the remaining budget; SDK/recorder calls may exceed it.
        shutdown_deadline = time.monotonic() + self.runtime.safety.shutdown_timeout_s
        cleanup_errors = []
        if self.runtime_state is not None:
            try:
                self.runtime_state.revoke_motion(reason=RunEndReason.RUNTIME_SHUTDOWN)
            except Exception as exc:
                cleanup_errors.append(error_detail("revoke_motion", exc))
        if self.robot is not None and self.robot.stop_required:
            try:
                self.robot.stop()
            except Exception as exc:
                cleanup_errors.append(error_detail("stop", exc))
        if self.model is not None:
            try:
                self.model.close()
            except Exception as exc:
                cleanup_errors.append(error_detail("model_close_submit", exc))
        try:
            if self.recorder is not None:
                try:
                    self.recorder.close()
                except Exception as exc:
                    cleanup_errors.append(error_detail("recorder_close", exc))
            if self.runtime_state is not None:
                shutdown_clean = shutdown_local_runtime(
                    self.runtime_state,
                    self.robot,
                    sensors=self.sensors,
                    keyboard=self.operator.keyboard if self.operator else None,
                    timeout_s=max(0.0, shutdown_deadline - time.monotonic()),
                )
            else:
                shutdown_clean = True
        except Exception as exc:
            cleanup_errors.append(error_detail("shutdown", exc))
        finally:
            if self.model is not None:
                model_close = self.model.close_result(shutdown_deadline - time.monotonic())
            shutdown_clean = (
                shutdown_clean and not cleanup_errors and model_close["status"] == "completed"
            )
            if self.results is not None:
                self.results.record_errors(cleanup_errors)
            home_results = self.operator.home_results if self.operator is not None else []
            home_fault = any(item["outcome"] in ("failed", "fault") for item in home_results)
            if self.results is not None:
                try:
                    self.results.finish_session(
                        outcome=(
                            "finished"
                            if clean and failure is None and shutdown_clean and not home_fault
                            else "fault"
                        ),
                        reason=(
                            str(failure)
                            if failure
                            else (
                                "return_home_failed"
                                if home_fault
                                else (
                                    f"model_close_{model_close['status']}"
                                    if model_close["status"] != "completed"
                                    else "shutdown_failed" if not shutdown_clean else None
                                )
                            )
                        ),
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

    def warmup_policy(self):
        self.model.submit(
            "load",
            samples=5,
            rgb_hw=(self.runtime.camera.height, self.runtime.camera.width),
            rtc_delay=(
                self.runtime.execution.prefetch_steps
                if self.runtime.execution.execution_mode == "rtc"
                else 0
            ),
        )
        while (completion := self.model.poll()) is None:
            time.sleep(0.005)
        if completion[1].error is not None:
            raise completion[1].error
        logger.debug("Model warmup durations: %s", completion[1].value)
