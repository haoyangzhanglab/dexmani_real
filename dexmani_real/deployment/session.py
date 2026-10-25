"""Own deployment resources, operator input and the multi-round experiment session."""

import logging
import multiprocessing as mp
import time
from concurrent.futures import CancelledError
from itertools import count

from dexmani_real.data.recorder import AsyncEpisodeRecorder, RecordingError
from dexmani_real.data.results import EpisodeResults, error_detail
from dexmani_real.deployment.config import DEFAULT_NUM_EPISODES
from dexmani_real.deployment.episode import PolicyEpisode, failure_reason
from dexmani_real.deployment.observation import (
    build_observation_camera,
    build_observation_kinematics,
    build_policy_observation,
)
from dexmani_real.ipc.channels import SensorChannels
from dexmani_real.planning.action import ActionMapper
from dexmani_real.planning.home_planner import HomePlanner
from dexmani_real.robot.commands import RobotError
from dexmani_real.robot.robot import DexManiRobot
from dexmani_real.runtime.home import home_robot
from dexmani_real.runtime.observation import (
    service_feedback,
    snapshot_camera_parameters,
)
from dexmani_real.runtime.operator_input import KeyboardInput, OperatorCommand
from dexmani_real.runtime.processes import SensorProcesses, shutdown_local_runtime
from dexmani_real.runtime.state import RunEndReason, RuntimeState, SafetyState
from dexmani_real.sensor.camera.extrinsics import (
    CameraExtrinsics,
    load_optional_camera_extrinsics,
)
from dexmani_real.sensor.camera.worker import run_camera_worker
from dexmani_real.utils.atomic_io import atomic_yaml_dump
from dexmani_real.utils.log import configure_logging

logger = logging.getLogger(__name__)


class PolicySession:
    """Own deployment resources while the inference worker serializes model work."""

    def __init__(
        self,
        runtime,
        info,
        *,
        control_needs,
        record_needs,
        dependencies,
        results_dir,
        run_config,
        device="cuda:0",
        seed=0,
        max_running_s=None,
        num_episodes=DEFAULT_NUM_EPISODES,
        recording_dir=None,
        camera_calibration_path=None,
        table_plane_abcd=None,
    ):
        self.run_config = run_config
        self.runtime = runtime
        self.table_plane_abcd = table_plane_abcd
        self.info = info
        self.device = device
        self.seed = seed
        self.max_running_s = max_running_s
        self.num_episodes = num_episodes
        self.recording_dir = recording_dir
        self.results_dir = results_dir
        self.camera_calibration_path = camera_calibration_path
        self.control_needs, self.record_needs = control_needs, record_needs
        self.dependencies = dependencies
        self.needs = control_needs.union(record_needs)
        self.keyboard = None
        self.home_results = []
        self.episode = None
        self.start_epoch = None
        self.completed = 0
        self.query_ids = count(1)
        self.capture_ids = count(1)

    def run(self):
        """Run a session with required evaluation results and optional Raw recording."""
        configure_logging()
        print(
            f"执行：{self.info.checkpoint_path.name} ({self.info.weights}) | NFE={self.info.inference_steps} | "
            f"A_exec={self.runtime.execution.action_steps} | {self.runtime.execution.execution_mode} | "
            f"{1.0 / self.info.control_dt_s:g} Hz"
        )
        print("正在初始化模型和设备…", flush=True)
        self.results = EpisodeResults(self.results_dir, "policy")
        self.runtime_state = self.sensors = self.robot = None
        self.model = self.recorder = None
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
        from dexmani_policy.deployment import load_policy

        from dexmani_real.deployment.inference import InferenceWorker

        if self.needs.point_cloud or "camera_extrinsic" in self.info.observation_fields:
            self.camera_calibration = CameraExtrinsics(self.camera_calibration_path)
        elif self.recording_dir is not None:
            self.camera_calibration = load_optional_camera_extrinsics(self.camera_calibration_path)
        else:
            self.camera_calibration = None
        self.camera_snapshot = None
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
        self.sensors = SensorProcesses(channels)
        self.robot = DexManiRobot(
            self.runtime.arm, self.runtime.hand, run_signal=self.runtime_state.run_signal,
            stop_signal=self.runtime_state.stop_signal,
            emergency_signal=self.runtime_state.emergency_signal,
            signal_lock=self.runtime_state.motion_lock,
            check_services=self.sensors.check,
        )

        self.kinematics = build_observation_kinematics(
            self.dependencies, fingertip_link_names=self.info.fingertip_link_names,
            hand=self.runtime.hand, hand_pointcloud=self.runtime.hand_pointcloud,
        )
        self.camera_parameters = None
        self.model = InferenceWorker(
            lambda: load_policy(
                self.info,
                device=self.device,
                seed=self.seed,
            ),
            build_observation=lambda rows: build_policy_observation(
                rows, self.info.observation_fields, self.dependencies, kinematics=self.kinematics,
                camera_parameters=self.camera_parameters,
            ),
            reset_observation=(
                self.kinematics.row_features.clear
                if self.kinematics is not None else None
            ),
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
        self.action_mapper = ActionMapper.for_mode(
            self.info.action_mode, arm=self.runtime.arm, hand=self.runtime.hand,
            motion=self.runtime.motion, environment=self.runtime.environment,
        )
        self.home_planner = HomePlanner(
            self.runtime.environment, table_plane_abcd=self.table_plane_abcd,
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
        control_services = []
        if self.control_needs.camera != "none" or self.control_needs.point_cloud:
            control_services.append("camera")
        if self.control_needs.point_cloud:
            control_services.append("pointcloud")
        if not self.sensors.check(names=control_services):
            raise RobotError("control sensor process exited")
        if self.record_needs.camera != "none" and not self.sensors.check(names=("camera",)):
            raise RecordingError("recording camera process exited")
        return True

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
                may_start=lambda: self.keyboard.healthy and self.runtime_state.requests_allow_motion(),
            )
            self.camera_snapshot = snapshot_camera_parameters(
                self.sensors.channels, self.runtime.camera, self.camera_calibration,
                require_extrinsics=self.needs.point_cloud or "camera_extrinsic" in self.info.observation_fields,
            )
        self.camera_parameters = build_observation_camera(
            self.camera_snapshot, self.info.observation_fields, depth_config=self.info.depth_config,
        )
        if self.camera_snapshot is not None:
            self.run_config["effective_camera"] = {
                "camera_T_xarm_base_from_color": None,
                **self.camera_snapshot.metadata(),
                "transform_direction": "color_optical_to_xarm_base",
                "translation_units": "m",
            }
            atomic_yaml_dump(
                self.run_config, self.results_dir / "run_config.yaml",
            )
        if self.needs.point_cloud:
            from dexmani_real.sensor.pointcloud.worker import run_pointcloud_worker

            self.sensors.start(
                self.process_context.Process(
                    name="pointcloud", target=run_pointcloud_worker,
                    args=(
                        self.sensors.channels, self.runtime.pointcloud,
                        self.camera_snapshot, self.table_plane_abcd,
                    ),
                ),
                ready=self.sensors.channels.pointcloud_ready,
                timeout_s=self.runtime.safety.pointcloud_ready_timeout_s,
                may_start=lambda: self.keyboard.healthy and self.runtime_state.requests_allow_motion(),
            )
        if self.needs.camera_static and self.needs.camera == "none" and not self.needs.point_cloud:
            # Policy 无 VR worker，此分支只持有已初始化的相机。
            if not self.sensors.shutdown(graceful_timeout_s=self.runtime.safety.shutdown_timeout_s):
                raise RuntimeError("static camera failed to close normally")
            self.sensors.channels = None
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
                # close() owns late task reclamation, including the immutable query identity.
                if self.episode is not None and self.episode.termination is not None:
                    self.episode = None
            except Exception as exc:
                cleanup_errors.append(error_detail("model_close_submit", exc))
        try:
            if self.recorder is not None:
                # Episode 已结束也不能丢失仍在退出的 writer。
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
            self.results.record_errors(cleanup_errors)
            home_fault = any(item["outcome"] in ("failed", "fault") for item in self.home_results)
            successful = clean and failure is None and shutdown_clean and not home_fault
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
                    home_results=self.home_results,
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
            rtc_guidance_cap=(
                self.runtime.execution.rtc_guidance_cap
                if self.runtime.execution.execution_mode == "rtc" else None
            ),
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
            self.start_epoch = None
            self.runtime_state.request_stop(reason=reason)

    def request_quit(self) -> None:
        """Apply Q's motion fence before requesting session shutdown."""
        with self.runtime_state.motion_lock:
            self.start_epoch = None
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
                if discard_preparation_in_batch or not self.request_start():
                    logger.warning(
                        "operator: ignored B while stopped or after preparation in this batch"
                    )
                    continue
            elif signal is OperatorCommand.STOP:
                # Keyboard callbacks already revoked motion; a second revoke could cross a cleared stop request.
                continue
            elif signal is OperatorCommand.DISCARD:
                logger.warning("operator: D is not used in policy deployment; ignored")
            elif signal is OperatorCommand.RESET_TACTILE:
                if discard_preparation_in_batch:
                    logger.warning("operator: ignored T after preparation in the same batch")
                    continue
                discard_preparation_in_batch = True
                # B earlier in this batch may have created a pending start.
                if stop_in_batch or self.start_epoch is not None or not allow_tactile_reset:
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
                self.start_epoch = None
        if not home_allowed:
            logger.warning("operator: ignored H unless ARMED without shutdown, fault or stop")
            return False
        home_result = {"outcome": "incomplete"}
        self.home_results.append(home_result)
        try:
            logger.info("正在回初始位…")
            result = home_robot(
                self.runtime_state,
                self.runtime.arm, self.runtime.hand,
                self.home_planner,
                robot=self.robot,
                service=lambda: service_feedback(self.robot, self.runtime),
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

    def request_start(self):
        with self.runtime_state.motion_lock:
            if self.episode is not None or self.model.has_pending_task:
                return False
            epoch = self.runtime_state.run_id
            if not self.runtime_state.command_may_cross_sdk(
                run_id=epoch, required_safety_state=SafetyState.ARMED,
            ):
                return False
            self.start_epoch = epoch
            return True

    def advance_episode(self, *, shutdown=False, failure=None, reason=RunEndReason.RUNTIME_SHUTDOWN):
        if self.episode is None:
            epoch, self.start_epoch = self.start_epoch, None
            if shutdown or epoch is None:
                return
            if not self.runtime_state.command_may_cross_sdk(
                run_id=epoch, required_safety_state=SafetyState.ARMED,
            ):
                return
            if self.recorder is not None:
                self.recorder.check_error()
                if self.recorder.is_recording or self.recorder.writer_pending:
                    raise RecordingError("previous capture still owns resources")
            self.recorder = (
                AsyncEpisodeRecorder(self.recording_dir, control_hz=1.0 / self.info.control_dt_s)
                if self.recording_dir is not None else None
            )
            self.episode = PolicyEpisode(
                self.runtime_state,
                self.runtime,
                self.info,
                preparing_epoch=epoch,
                robot=self.robot,
                sensors=self.sensors,
                control_needs=self.control_needs, record_needs=self.record_needs,
                dependencies=self.dependencies,
                action_mapper=self.action_mapper,
                model_runtime=self.model,
                max_running_s=self.max_running_s,
                query_ids=self.query_ids,
                capture_ids=self.capture_ids,
                recorder=self.recorder,
                results=self.results,
                camera_snapshot=self.camera_snapshot,
            )
        episode = self.episode
        before = episode.termination
        try:
            if shutdown:
                episode.finish(
                    str(failure) if failure else "shutdown", run_end_reason=reason,
                    errors=[error_detail("session", failure)]
                    if failure is not None and not isinstance(failure, KeyboardInterrupt) else [],
                )
            else:
                episode.step()
        finally:
            ended = episode.termination
            if before is None and ended is not None:
                self.start_epoch = None
                self.keyboard.drain_signal(OperatorCommand.BEGIN)
                if ended.run_id is not None:
                    self.completed += 1
                    normal = ended.reason in {"operator", "quit"} or (
                        ended.reason == "timeout" and ended.detail == "duration_timeout"
                    )
                    logger.log(
                        logging.INFO if normal else logging.WARNING,
                        "第 %d 轮结束：%s（%s，%.1f s）。",
                        self.completed,
                        ended.reason,
                        ended.detail,
                        ended.duration_s,
                    )
                    if self.num_episodes is not None and self.completed >= self.num_episodes:
                        self.runtime_state.request_quit()
                        print(f"已达到 {self.num_episodes} 轮上限，正在退出。", flush=True)
                    elif ended.reason in {"operator", "timeout", "executor_boundary"} and (
                        self.runtime_state.requests_allow_motion()
                    ):
                        print("等待下一轮：整理场景后 H 回初始位 → B 开始；Q 退出。", flush=True)
            # A finished attempt still owns reset/predict identity until poll() reclaims it.
            if ended is not None and not self.model.has_pending_task:
                self.episode = None

    def run_control_loop(self):
        failure = None
        reason = RunEndReason.RUNTIME_SHUTDOWN
        try:
            while self.runtime_state.running and not self.runtime_state.quit_requested:
                self.robot.arm.check_error()
                self.poll_operator(
                    allow_tactile_reset=(
                        self.episode is None and not self.model.has_pending_task
                        and (self.recorder is None or not self.recorder.is_recording)
                    )
                )
                if self.runtime_state.estop_request or self.runtime_state.error_state:
                    break
                if self.episode is None or self.episode.run_id is None:
                    self.robot.service_idle()
                    if not self.robot.stop_required and (
                        self.episode is None or self.episode.termination is not None
                    ):
                        self.runtime_state.acknowledge_stop(self.runtime_state.run_id)
                self.advance_episode()
                running = self.episode is not None and self.episode.run_id is not None
                time.sleep(0.001 if running else 0.005)
        except KeyboardInterrupt as exc:
            self.runtime_state.request_emergency_stop()
            failure, reason = exc, RunEndReason.ESTOP
        except Exception as exc:
            failure, reason = exc, failure_reason(exc)
            self.runtime_state.mark_error()
            self.runtime_state.end_session(reason=reason)
            self.sensors.request_shutdown()
        finally:
            try:
                self.advance_episode(shutdown=True, failure=failure, reason=reason)
            except BaseException as exc:
                failure = failure or exc
                logger.exception("episode cleanup failed")
        if failure is not None:
            raise failure
