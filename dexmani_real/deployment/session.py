"""Own deployment resources, operator input and the multi-round experiment session."""

import logging
import multiprocessing as mp
import time
from concurrent.futures import CancelledError

from dexmani_real.calibration.camera.extrinsics import (
    CameraExtrinsics,
    load_optional_camera_extrinsics,
)
from dexmani_real.data.recorder import AsyncEpisodeRecorder, RecordingError
from dexmani_real.data.results import EpisodeResults, error_detail
from dexmani_real.deployment.observation import (
    build_observation_camera,
    build_observation_kinematics,
)
from dexmani_real.deployment.runner import PolicyRunner
from dexmani_real.ipc.channels import SensorChannels
from dexmani_real.planning.action import ActionMapper
from dexmani_real.robot.home import build_home_planner, home_robot
from dexmani_real.robot.commands import DispatchError
from dexmani_real.robot.robot import DexManiRobot
from dexmani_real.runtime.processes import SensorProcesses, shutdown_local_runtime
from dexmani_real.runtime.operator_input import KeyboardInput, OperatorCommand
from dexmani_real.runtime.state import RunEndReason, RuntimeState, SafetyState
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
        self.keyboard = None
        self.home_results = []

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
            self.run_control_loop()
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
        self.home_planner = build_home_planner(self.runtime)
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
            model_runtime=self.model,
            kinematics=kinematics,
            max_running_s=self.max_running_s,
            num_episodes=self.num_episodes,
            camera_calibration=self.camera_calibration,
            results=self.results,
        )

        self.keyboard = KeyboardInput(
            estop_callback=self.runtime_state.request_emergency_stop,
            stop_callback=self.request_stop,
            quit_callback=self.request_quit,
        )

    def check_services(self):
        if self.keyboard.estop_latched or not self.keyboard.healthy:
            self.runtime_state.request_emergency_stop()
            return False
        return self.sensors.check()

    def start_devices(self):
        self.keyboard.start()
        self.robot.check_services = self.check_services
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
                check_services=lambda: self.keyboard.healthy,
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
                check_services=lambda: self.keyboard.healthy,
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
                    keyboard=self.keyboard,
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
            home_results = self.home_results
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


    def request_stop(self, *, reason=RunEndReason.OPERATOR) -> None:
        """Fence motion even while the I/O owner is blocked in HOME, inference or SDK I/O."""
        with self.runtime_state.motion_lock:
            self.runner.start_requested = False
            self.runtime_state.request_stop(reason=reason)


    def request_quit(self) -> None:
        """Apply Q's motion fence before requesting session shutdown."""
        with self.runtime_state.motion_lock:
            self.runner.start_requested = False
            self.runtime_state.request_quit()


    def poll_operator(self, *, allow_tactile_reset: bool = False) -> None:
        if self.keyboard.estop_latched or not self.keyboard.healthy:
            self.runtime_state.request_emergency_stop()
            return
        self.handle_command_batch(
            self.keyboard.poll(timeout=0),
            allow_tactile_reset=allow_tactile_reset,
        )


    def handle_command_batch(self, signals, *, allow_tactile_reset=False) -> None:
        # HOME blocks; H/B/T from the same batch need a fresh request afterwards.
        discard_preparation_in_batch = False
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
                if discard_preparation_in_batch or not self.runner.request_start():
                    logger.warning(
                        "operator: ignored B while stopped or after preparation in this batch"
                    )
                    continue
            elif signal is OperatorCommand.STOP:
                # Keyboard callbacks already revoked motion; a second revoke could cross a cleared stop request.
                continue
            elif signal is OperatorCommand.PAUSE:
                logger.warning("operator: C is not used in policy deployment; ignored")
            elif signal is OperatorCommand.DISCARD:
                logger.warning("operator: D is not used in policy deployment; ignored")
            elif signal is OperatorCommand.RESET_TACTILE:
                if discard_preparation_in_batch:
                    logger.warning("operator: ignored T after preparation in the same batch")
                    continue
                discard_preparation_in_batch = True
                # B earlier in this batch may have created a pending start.
                if stop_in_batch or self.runner.start_requested or not allow_tactile_reset:
                    logger.warning(
                        "T ignored: tactile reset requires idle, no capture/reset/pending inference"
                    )
                    continue
                if self.runtime_state.safety_state != SafetyState.ARMED:
                    continue
                try:
                    logger.info(
                        "正在请求触觉传感器复位，请保持手部无接触；S/Q/Esc 可中断后续请求。"
                    )
                    self.robot.reset_tactile_sensors(cancel_requested=self.home_abort_requested)
                except CancelledError as exc:
                    logger.info("%s", exc)
                finally:
                    for command in (
                        OperatorCommand.RESET_TACTILE,
                        OperatorCommand.HOME,
                        OperatorCommand.BEGIN,
                    ):
                        self.keyboard.drain_signal(command)
                return
            elif signal is OperatorCommand.HOME:
                if home_handled_in_batch:
                    continue
                if stop_in_batch:
                    logger.warning("operator: ignored H received in the same batch as S/Q")
                    continue
                if self.run_home():
                    home_handled_in_batch = True
                    discard_preparation_in_batch = True
            elif signal is OperatorCommand.QUIT:
                # Listener stays available through final recording cleanup.
                continue
            elif signal is OperatorCommand.EMERGENCY_STOP:
                self.runtime_state.request_emergency_stop()
                return


    def run_home(self) -> bool:
        with self.runtime_state.motion_lock:
            home_allowed = self.runtime_state.command_may_cross_sdk(
                run_id=self.runtime_state.run_id,
                required_safety_state=SafetyState.ARMED,
            )
            if home_allowed:
                # Only the owner consumes S; HOME never clears a newer stop.
                self.runner.start_requested = False
        if not home_allowed:
            logger.warning("operator: ignored H unless ARMED without shutdown, fault or stop")
            return False
        home_result = {"outcome": "incomplete"}
        self.home_results.append(home_result)
        try:
            logger.info("正在回初始位…")
            result = home_robot(
                self.runtime_state,
                self.runtime,
                self.home_planner,
                robot=self.robot,
                abort_requested=self.home_abort_requested,
            )
        except BaseException as exc:
            home_result.update(outcome="fault", reason=f"{type(exc).__name__}: {exc}")
            raise
        with self.runtime_state.motion_lock:
            completed_without_stop = result.ok and self.runtime_state.command_may_cross_sdk(
                run_id=self.runtime_state.run_id,
                required_safety_state=SafetyState.ARMED,
            )
        interrupted = bool(
            result.interrupted
            or (
                result.ok
                and (
                    self.runtime_state.quit_requested
                    or self.runtime_state.estop_request
                    or self.runtime_state.stop_request
                )
            )
        )
        home_result.update(
            outcome=(
                "completed"
                if completed_without_stop
                else "interrupted" if interrupted else "failed"
            ),
            reason=result.reason,
        )
        if completed_without_stop:
            logger.info("已回初始位，按 B 开始。")
        else:
            logger.warning("回初始位未完成，请检查当前姿态：%s", result.reason)
        # Discard stale H/B/T after blocking HOME; preserve S/Q/ESC.
        self.keyboard.drain_signal(OperatorCommand.RESET_TACTILE)
        self.keyboard.drain_signal(OperatorCommand.HOME)
        self.keyboard.drain_signal(OperatorCommand.BEGIN)
        return True


    def home_abort_requested(self) -> bool:
        self.robot.check()
        return bool(
            not self.runtime_state.running
            or self.runtime_state.quit_requested
            or self.runtime_state.error_state
            or self.runtime_state.estop_request
            or self.runtime_state.stop_request
        )


    def run_control_loop(self):
        failure = None
        reason = RunEndReason.RUNTIME_SHUTDOWN
        try:
            while self.runtime_state.running and not self.runtime_state.quit_requested:
                self.robot.arm.check_error()
                self.poll_operator(
                    allow_tactile_reset=(
                        self.runner.run_id is None
                        and self.runner.preparing_epoch is None
                        and self.model.future is None
                        and (self.recorder is None or not self.recorder.is_recording)
                    )
                )
                if self.runtime_state.estop_request or self.runtime_state.error_state:
                    break
                if self.runner.run_id is None:
                    self.robot.service_idle()
                epoch = self.runner.run_id
                try:
                    self.runner.step()
                finally:
                    if epoch is not None and self.runner.run_id is None:
                        self.keyboard.drain_signal(OperatorCommand.BEGIN)
                time.sleep(0.001 if self.runner.run_id is not None else 0.005)
        except KeyboardInterrupt as exc:
            self.runtime_state.request_emergency_stop()
            failure = exc
            reason = RunEndReason.ESTOP
        except Exception as exc:
            failure = exc
            if isinstance(exc, RecordingError):
                reason = RunEndReason.RECORDING_FAILURE
            elif isinstance(exc, DispatchError):
                reason = RunEndReason.HARDWARE_FAULT
            else:
                reason = RunEndReason.POLICY_FAILURE
            self.runtime_state.mark_error()
            self.sensors.request_shutdown(reason=reason)
        finally:
            try:
                self.runner.finish_episode(str(failure) if failure else "shutdown", run_end_reason=reason)
            except Exception as exc:
                failure = failure or exc
                logger.exception("episode cleanup failed")
            try:
                if self.robot.stop_required:
                    self.robot.stop()
            except Exception as exc:
                failure = failure or exc
                logger.exception(
                    "policy shutdown stop failed before remaining artifact finalization"
                )
            try:
                if self.runner.capture_attempted or (
                    self.results is not None and self.results.current is not None
                ):
                    cancelled = isinstance(failure, KeyboardInterrupt) or (
                        failure is None and (
                            self.runtime_state.stop_request
                            or self.runtime_state.quit_requested
                            or self.runtime_state.estop_request
                        )
                    )
                    preparation_reason = "start_cancelled" if cancelled else "preparation_failed"
                    detail = str(failure) if failure else "shutdown"
                    errors = [error_detail("preparation", failure)] if failure else []
                    self.runner.finalize_outputs(
                        preparation_reason, detail, errors=errors,
                    )
            except Exception as exc:
                failure = failure or exc
        if failure is not None:
            raise failure
