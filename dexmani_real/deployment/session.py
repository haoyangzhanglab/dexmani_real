"""Assemble and close deployment resources; PolicyRunner owns the experiment lifecycle."""

import logging
import multiprocessing as mp
import time

from dexmani_real.calibration.camera.extrinsics import (
    CameraExtrinsics,
    load_optional_camera_extrinsics,
)
from dexmani_real.data.recorder import AsyncEpisodeRecorder
from dexmani_real.data.results import EpisodeResults, error_detail
from dexmani_real.deployment.observation import (
    build_observation_camera,
    build_observation_kinematics,
)
from dexmani_real.deployment.runner import PolicyRunner
from dexmani_real.ipc.channels import SensorChannels
from dexmani_real.planning.action import ActionMapper
from dexmani_real.robot.home import build_home_planner
from dexmani_real.robot.robot import DexManiRobot
from dexmani_real.runtime.processes import SensorProcesses, shutdown_local_runtime
from dexmani_real.runtime.state import RunEndReason, RuntimeState
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


class _PolicySession:
    """Own deployment resources while the inference worker serializes model work."""

    def __init__(
        self,
        runtime,
        info,
        *,
        needs,
        device="cuda:0",
        seed=0,
        max_running_s=None,
        num_episodes=20,
        recording_dir=None,
        results_dir=None,
        camera_calibration_path=None,
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
        self.needs = needs

    def run(self):
        """Run a session with independently optional Raw recording and evaluation summaries."""
        configure_logging()
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
        self.model = self.runner = self.recorder = None
        failure = None
        clean = False
        try:
            self.prepare_resources()
            self.start_devices()
            print(
                "就绪（无需回车）：H 回初始位 → B 开始 | S 结束本轮 | Q 退出 | Esc 急停 | "
                "T 请求复位触觉（空闲且无接触）。",
                flush=True,
            )
            self.runner.run()
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

    def prepare_resources(self):
        from dexmani_real.deployment.inference import InferenceWorker

        fields = set(self.info.observation_fields)
        if self.needs.point_cloud or "camera_extrinsic" in fields:
            self.camera_calibration = CameraExtrinsics(self.camera_calibration_path)
        elif self.recording_dir is not None:
            self.camera_calibration = load_optional_camera_extrinsics(self.camera_calibration_path)
        else:
            self.camera_calibration = None
        self.pointcloud_config = (
            PointCloudWorkerConfig(
                pointcloud=self.runtime.pointcloud,
                camera_calibration=self.camera_calibration,
                table_plane_abcd=self.runtime.environment.table.plane_abcd,
            )
            if self.needs.point_cloud
            else None
        )
        self.process_context = mp.get_context("spawn")
        self.runtime_state = RuntimeState()
        channels = (
            SensorChannels(
                self.process_context,
                camera=self.runtime.camera,
                pointcloud_num_points=(
                    self.runtime.pointcloud.num_points if self.needs.point_cloud else None
                ),
            )
            if self.needs.camera_worker
            else None
        )
        self.sensors = SensorProcesses(self.runtime_state, channels)
        self.robot = DexManiRobot(
            self.runtime_state,
            self.runtime,
            check_services=self.sensors.check,
            required_tactile=self.needs.required_tactile,
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
        kinematics = build_observation_kinematics(self.info, self.runtime)
        action_mapper = ActionMapper.for_mode(self.runtime, self.info.action_mode)
        home_planner = build_home_planner(self.runtime)
        self.recorder = (
            AsyncEpisodeRecorder(
                self.recording_dir,
                control_hz=1.0 / self.info.control_dt_s,
            )
            if self.recording_dir is not None
            else None
        )

        self.runner = PolicyRunner(
            self.runtime_state,
            self.runtime,
            self.info,
            robot=self.robot,
            action_mapper=action_mapper,
            needs=self.needs,
            sensors=self.sensors,
            recorder=self.recorder,
            home_planner=home_planner,
            model_runtime=self.model,
            kinematics=kinematics,
            max_running_s=self.max_running_s,
            num_episodes=self.num_episodes,
            camera_calibration=self.camera_calibration,
            results=self.results,
        )

    def start_devices(self):
        self.runner.keyboard.start()
        self.robot.check_services = lambda: (
            self.runner.keyboard.healthy and self.sensors.check()
        )
        self.robot.connect()
        if self.needs.camera_worker:
            self.sensors.start(
                self.process_context.Process(
                    name="camera",
                    target=run_camera_worker,
                    args=(self.sensors.channels, self.runtime.camera),
                ),
                ready=self.sensors.channels.camera_ready,
                timeout_s=self.runtime.safety.camera_ready_timeout_s,
                check_services=lambda: self.runner.keyboard.healthy,
            )
        if self.pointcloud_config is not None:
            self.sensors.start(
                self.process_context.Process(
                    name="pointcloud",
                    target=run_pointcloud_worker,
                    args=(self.sensors.channels, self.pointcloud_config),
                ),
                ready=self.sensors.channels.pointcloud_ready,
                timeout_s=self.runtime.safety.pointcloud_ready_timeout_s,
                check_services=lambda: self.runner.keyboard.healthy,
            )
        self.runner.camera_parameters = build_observation_camera(
            self.sensors.channels,
            self.runtime.camera,
            self.info.observation_fields,
            self.camera_calibration,
        )
        self.robot.check()
        self.runtime_state.arm()

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
                    keyboard=self.runner.keyboard if self.runner else None,
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
            home_results = self.runner.home_results if self.runner is not None else []
            home_fault = any(item["outcome"] in ("failed", "fault") for item in home_results)
            successful = clean and failure is None and shutdown_clean and not home_fault
            if self.results is not None:
                try:
                    if failure:
                        reason = str(failure)
                    elif home_fault:
                        reason = "return_home_failed"
                    elif model_close["status"] != "completed":
                        reason = f"model_close_{model_close['status']}"
                    elif not shutdown_clean:
                        reason = "shutdown_failed"
                    else:
                        reason = None
                    self.results.finish_session(
                        outcome="finished" if successful else "fault",
                        reason=reason,
                        shutdown_clean=shutdown_clean,
                        model_close=model_close,
                        home_results=home_results,
                    )
                except Exception:
                    successful = False
                    logger.exception("session result publication failed")
        if model_close["status"] == "pending":
            logger.warning("Model cleanup remains pending; Python/CUDA exit is not bounded")
        return int(not successful)

    def warmup_policy(self):
        timeout_s = self.runtime.safety.model_startup_timeout_s
        deadline = time.monotonic() + timeout_s
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
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"model load/warmup timed out after {timeout_s:g}s; status=pending; "
                    "model thread/native call cannot be interrupted"
                )
            time.sleep(min(0.005, remaining))
        if completion[1].error is not None:
            raise completion[1].error
        logger.debug("Model warmup durations: %s", completion[1].value)
