"""管理模型、设备与多轮策略实验。"""

import logging
import multiprocessing as mp
import time
from collections import Counter
from dataclasses import asdict
from itertools import count

import numpy as np

from dexmani_real.config.deployment import ACTION_TIMEOUT_NS
from dexmani_real.control.actions import ActionMapper
from dexmani_real.control.homing import HomePlanner
from dexmani_real.data.raw import camera_metadata
from dexmani_real.data.recorder import RecordingError
from dexmani_real.deployment.observation import ObservationBuilder
from dexmani_real.deployment.rollout import Rollout, failure_reason, utc_now
from dexmani_real.ipc.sensors import SensorChannels, snapshot_camera_parameters
from dexmani_real.robot.commands import OperationCancelled, RobotError
from dexmani_real.robot.system import Robot
from dexmani_real.runtime.home import home_robot, initialize_hand
from dexmani_real.runtime.keyboard import Keyboard
from dexmani_real.runtime.lifecycle import (
    CAMERA_READY_TIMEOUT_S,
    POINTCLOUD_READY_TIMEOUT_S,
    SHUTDOWN_TIMEOUT_S,
    SensorWorkers,
    close_runtime,
)
from dexmani_real.runtime.observation import check_observation, read_observation
from dexmani_real.runtime.state import RunEndReason, SafetyState, SessionState
from dexmani_real.sensor.realsense import run_camera
from dexmani_real.utils.io import write_json, write_yaml
from dexmani_real.utils.log import configure_logging, error_detail

logger = logging.getLogger(__name__)
MODEL_STARTUP_TIMEOUT_S = 300.0


class PolicySession:
    def __init__(
        self, runtime, info, *, output_dir, run_config, device, seed,
        max_running_s, num_episodes, record_raw, camera_calibration_path, table_plane_abcd,
        dependencies, control_needs, record_needs, needs,
    ):
        self.run_config = run_config
        self.runtime, self.info = runtime, info
        self.table_plane_abcd = table_plane_abcd
        self.device, self.seed = device, seed
        self.max_running_s, self.num_episodes = max_running_s, num_episodes
        self.record_raw = record_raw
        self.output_dir = output_dir
        self.camera_calibration_path = camera_calibration_path
        self.dependencies = dependencies
        self.control_needs, self.record_needs, self.needs = control_needs, record_needs, needs
        self.control_services = tuple(
            name for name, needed in (
                ("camera", bool(self.control_needs.camera_fields) or self.control_needs.point_cloud),
                ("pointcloud", self.control_needs.point_cloud),
            ) if needed
        )
        self.record_services = (
            ("camera",) if self.record_needs.camera_fields and "camera" not in self.control_services else ()
        )
        self.runtime_state = self.sensors = self.robot = self.keyboard = self.model = None
        self.episode = self.start_epoch = None
        self.capture_ids = count(1)
        self.episodes, self.errors = [], []
        self.summary = dict(
            started_utc=utc_now(), ended_utc=None, outcome="incomplete", reason=None, shutdown=None,
        )

    def save_results(self, *, overwrite=True):
        termination_counts = Counter(result.termination_reason for result in self.episodes)
        write_json(
            dict(
                self.summary, errors=self.errors, episodes=[asdict(result) for result in self.episodes],
                started_episode_count=sum(result.started_utc is not None for result in self.episodes),
                technical_termination_count=sum(
                    not result.normal_termination and (
                        result.termination_reason != "start_cancelled" or bool(result.errors)
                    )
                    for result in self.episodes
                ),
                termination_counts=dict(termination_counts),
            ),
            self.output_dir / "session_result.json", overwrite=overwrite,
        )

    def run(self):
        configure_logging()
        print(
            f"执行：{self.info.checkpoint_path.name} ({self.info.weights}) | NFE={self.info.inference_steps} | "
            f"A_exec={self.runtime.execution.action_steps} | {self.runtime.execution.mode} | "
            f"{1.0 / self.info.control_dt_s:g} Hz"
        )
        print("正在初始化模型和设备…", flush=True)
        self.save_results(overwrite=False)
        failure = None
        reason = RunEndReason.RUNTIME_SHUTDOWN
        try:
            self.prepare_resources()
            self.start_devices()
            print("就绪：H 回初始位 → B 开始 | S 结束本轮 | Q 退出 | Esc 急停 | T 复位触觉。", flush=True)
            self.run_control_loop()
        except KeyboardInterrupt as exc:
            failure, reason = exc, RunEndReason.ESTOP
            if self.runtime_state is not None:
                self.runtime_state.request_emergency_stop()
        except Exception as exc:
            if isinstance(exc, OperationCancelled) and self.runtime_state is not None and (
                self.runtime_state.stop_request or self.runtime_state.quit_requested or self.runtime_state.estop_request
            ):
                self.runtime_state.request_quit()
                self.summary["reason"] = "startup_cancelled"
            else:
                failure, reason = exc, failure_reason(exc)
                self.errors.append(error_detail("session", exc))
                if self.runtime_state is not None:
                    self.runtime_state.mark_error()
                    self.runtime_state.end_session(reason=reason)
                logger.exception("policy session failed")
        finally:
            try:
                self.advance_episode(shutdown=True, failure=failure, reason=reason)
            except BaseException as exc:
                failure = failure or exc
                self.errors.append(error_detail("episode_cleanup", exc))
                if self.runtime_state is not None:
                    if isinstance(exc, KeyboardInterrupt):
                        self.runtime_state.request_emergency_stop()
                    else:
                        self.runtime_state.mark_error()
            exit_code = self.close(failure=failure)
        return exit_code

    def prepare_resources(self):
        from dexmani_policy.deployment import load_policy

        from dexmani_real.deployment.inference import PolicyWorker

        self.runtime_state = SessionState()
        self.camera_snapshot = None
        self.observations = ObservationBuilder(
            self.info, self.runtime, dependencies=self.dependencies, needs=self.control_needs,
        )
        self.action_mapper = ActionMapper.for_mode(
            self.info.action_mode, arm=self.runtime.arm, hand=self.runtime.hand,
            motion=self.runtime.motion, environment=self.runtime.environment,
        )
        self.home_planner = HomePlanner(
            self.runtime.environment, self.runtime.arm, table_plane_abcd=self.table_plane_abcd,
        )
        self.keyboard = Keyboard(
            estop_callback=self.runtime_state.request_emergency_stop,
            bindings={"s": self.request_stop, "q": self.request_quit},
        )
        self.keyboard.start()
        self.model = PolicyWorker(
            lambda: load_policy(self.info, device=self.device, seed=self.seed),
            build_observation=self.observations.build,
        )
        self.model.load()

        self.process_context = mp.get_context("spawn")
        channels = (
            SensorChannels(
                self.process_context, camera=self.runtime.camera,
                pointcloud_num_points=self.runtime.pointcloud.num_points if self.needs.point_cloud else None,
            ) if self.needs.camera_worker else None
        )
        self.sensors = SensorWorkers(channels)
        self.robot = Robot(
            self.runtime.arm, self.runtime.hand, run_signal=self.runtime_state.run_signal,
            stop_signal=self.runtime_state.stop_signal, emergency_signal=self.runtime_state.emergency_signal,
            signal_lock=self.runtime_state.motion_lock, check_services=self.check_services,
        )

    def check_services(self):
        if self.keyboard.estop_latched or not self.keyboard.check_health():
            self.runtime_state.request_emergency_stop()
            return False
        if self.control_services and not self.sensors.check(names=self.control_services):
            raise RobotError("control sensor process exited")
        if self.record_services and not self.sensors.check(names=self.record_services):
            raise RecordingError("recording camera process exited")
        return True

    def start_devices(self):
        self.robot.connect()
        if self.needs.camera_worker:
            self.sensors.start(
                self.process_context.Process(
                    name="camera",
                    target=run_camera,
                    args=(self.sensors.channels, self.runtime.camera),
                ),
                ready=self.sensors.channels.camera_ready,
                timeout_s=CAMERA_READY_TIMEOUT_S,
                may_start=lambda: self.keyboard.check_health() and self.runtime_state.requests_allow_motion(),
            )
            needs_extrinsics = self.needs.point_cloud or "camera_extrinsic" in self.info.observation_fields
            self.camera_snapshot = snapshot_camera_parameters(
                self.sensors.channels, self.runtime.camera, self.camera_calibration_path,
                require_extrinsics=needs_extrinsics,
                include_extrinsics=needs_extrinsics or self.record_raw,
            )
        self.observations.configure_camera(self.camera_snapshot)
        if self.camera_snapshot is not None:
            self.run_config["effective_camera"] = {
                "camera_T_xarm_base_from_color": None,
                **camera_metadata(self.camera_snapshot),
                "transform_direction": "color_optical_to_xarm_base",
                "translation_units": "m",
            }
            write_yaml(
                self.run_config, self.output_dir / "run_config.yaml",
            )
        if self.needs.point_cloud:
            from dexmani_real.perception.worker import run_pointcloud_worker

            self.sensors.start(
                self.process_context.Process(
                    name="pointcloud", target=run_pointcloud_worker,
                    args=(
                        self.sensors.channels, self.runtime.pointcloud,
                        self.camera_snapshot, self.table_plane_abcd,
                    ),
                ),
                ready=self.sensors.channels.pointcloud_ready,
                timeout_s=POINTCLOUD_READY_TIMEOUT_S,
                may_start=lambda: self.keyboard.check_health() and self.runtime_state.requests_allow_motion(),
            )
        if self.needs.camera_static and not self.needs.camera_fields and not self.needs.point_cloud:
            if not self.sensors.shutdown(deadline=time.monotonic() + SHUTDOWN_TIMEOUT_S):
                raise RuntimeError("static camera failed to close normally")
            self.sensors.channels = None
        self.robot.check()
        with self.runtime_state.motion_lock:
            if not self.runtime_state.requests_allow_motion():
                raise OperationCancelled("startup cancelled before arming")
            self.runtime_state.arm()
        result = initialize_hand(
            self.runtime_state, self.robot, cancel_requested=self.home_abort_requested,
        )
        if not result.ok:
            if result.interrupted:
                raise OperationCancelled(result.reason)
            raise RuntimeError(f"startup hand initialization failed: {result.reason}")
        self.warmup_policy()
        self.keyboard.discard_events("h", "b", "t")
        self.check_startup()

    def close(self, *, failure=None):
        deadline = time.monotonic() + SHUTDOWN_TIMEOUT_S
        cleanup_errors = []
        if self.runtime_state is not None:
            try:
                self.runtime_state.revoke_motion(reason=RunEndReason.RUNTIME_SHUTDOWN)
            except BaseException as exc:
                cleanup_errors.append(error_detail("revoke_motion", exc))
        if self.robot is not None and self.robot.stop_required:
            try:
                self.robot.stop()
            except BaseException as exc:
                cleanup_errors.append(error_detail("stop", exc))
        model_status = "completed"
        if self.model is not None:
            try:
                self.model.close()
            except BaseException as exc:
                model_status = "failed"
                cleanup_errors.append(error_detail("model_close", exc))
        if self.episode is not None:
            try:
                self.episode.close()
            except BaseException as exc:
                cleanup_errors.append(error_detail("recorder_close", exc))
        runtime_clean = True
        try:
            if self.runtime_state is not None:
                runtime_clean = close_runtime(
                    self.runtime_state, self.robot, sensors=self.sensors,
                    keyboard=self.keyboard, deadline=deadline, errors=cleanup_errors,
                )
        except BaseException as exc:
            runtime_clean = False
            cleanup_errors.append(error_detail("shutdown", exc))
        if self.model is not None and model_status != "failed":
            model_status, errors = self.model.wait_closed(deadline - time.monotonic())
            cleanup_errors.extend(errors)
        writer_pending = False
        if self.episode is not None:
            self.episode.refresh_raw()
            writer_pending = self.episode.writer_pending
        pending = model_status == "pending" or writer_pending
        shutdown = (
            "pending" if pending else "completed"
            if runtime_clean and model_status == "completed" and not cleanup_errors else "failed"
        )
        self.errors.extend(cleanup_errors)
        episode_error = next((error for result in self.episodes for error in result.errors), None)
        successful = (
            failure is None and not self.errors and shutdown == "completed"
            and episode_error is None
            and self.runtime_state is not None and self.runtime_state.quit_requested
            and not self.runtime_state.estop_request and not self.runtime_state.error_state
        )
        reason = self.summary["reason"]
        if failure is not None:
            reason = str(failure) or type(failure).__name__
        elif self.errors:
            reason = self.errors[0]["message"]
        elif episode_error is not None:
            reason = episode_error["message"]
        elif not successful:
            reason = "shutdown_pending" if pending else "session_stopped"
        self.summary.update(ended_utc=utc_now(), outcome="finished" if successful else "fault", reason=reason, shutdown=shutdown)
        try:
            self.save_results()
        except Exception as exc:
            self.errors.append(error_detail("session_save", exc))
            self.summary["outcome"] = "fault"
            if successful:
                self.summary["reason"] = str(exc) or type(exc).__name__
            successful = False
            logger.exception("session result publication failed")
        if pending:
            logger.warning("Cleanup remains pending; native model/writer calls may outlive this wait")
        return int(not successful)

    def check_startup(self):
        if self.keyboard.estop_latched or not self.keyboard.check_health():
            self.runtime_state.request_emergency_stop()
        if not self.runtime_state.requests_allow_motion():
            raise OperationCancelled("startup cancelled")
        self.robot.service_feedback()

    def wait_model(self, deadline):
        while True:
            self.check_startup()
            completion = self.model.poll()
            if completion is not None:
                return completion
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("model load/warmup timed out; native call remains pending")
            time.sleep(min(0.005, remaining))

    def warmup_policy(self):
        deadline = time.monotonic() + MODEL_STARTUP_TIMEOUT_S
        self.wait_model(deadline)
        # 模型/相机启动可能耗尽 connect 的首帧期限；尚未运动，重新建立本次反馈等待。
        self.robot.reset_hand_feedback()
        self.robot.read_initial_state(
            required_tactile=self.control_needs.required_tactile,
            cancel_requested=lambda: not self.runtime_state.requests_allow_motion(),
        )
        rows = []
        dt_ns = round(self.info.control_dt_s * 1e9)
        next_sample_ns = time.monotonic_ns()
        history_deadline_ns = next_sample_ns + (self.info.n_obs_steps - 1) * dt_ns + ACTION_TIMEOUT_NS
        while len(rows) < self.info.n_obs_steps:
            self.check_startup()
            now = time.monotonic_ns()
            if now >= history_deadline_ns:
                raise TimeoutError("warmup observation history timed out")
            if now < next_sample_ns:
                time.sleep(min(0.005, (next_sample_ns - now) / 1e9))
                continue
            row = read_observation(self.sensors.channels, self.robot, needs=self.control_needs)
            issues, _ = check_observation(row, self.runtime, self.control_needs)
            if issues:
                raise ValueError("warmup observation: " + "; ".join(issues))
            rows.append(self.observations.project(row))
            next_sample_ns = time.monotonic_ns() + dt_ns
        context = None
        if self.info.hand_history_steps is not None:
            initial_hand = rows[-1].values["hand_qpos"].astype(np.float32)
            context = {"hand_history": np.repeat(initial_hand[None], self.info.hand_history_steps, axis=0)}
        kwargs = {}
        if self.runtime.execution.mode == "rtc":
            kwargs = dict(
                rtc_guidance_cap=self.runtime.execution.rtc_guidance_cap,
                rtc_delay=self.runtime.execution.prefetch_steps,
            )
        self.model.warmup(tuple(rows), action_context=context, samples=5, **kwargs)
        self.wait_model(time.monotonic() + MODEL_STARTUP_TIMEOUT_S)

    def request_stop(self, *, reason=RunEndReason.OPERATOR) -> None:
        """阻塞操作期间也立即撤销运动权限。"""
        with self.runtime_state.motion_lock:
            self.start_epoch = None
            self.runtime_state.request_stop(reason=reason)

    def request_quit(self) -> None:
        with self.runtime_state.motion_lock:
            self.start_epoch = None
            self.runtime_state.request_quit()

    def poll_operator(self, *, allow_tactile_reset: bool = False) -> None:
        if self.keyboard.estop_latched or not self.keyboard.check_health():
            self.runtime_state.request_emergency_stop()
            return
        self.handle_command_batch(
            self.keyboard.poll(timeout=0),
            allow_tactile_reset=allow_tactile_reset,
        )

    def handle_command_batch(self, signals, *, allow_tactile_reset=False) -> None:
        if "esc" in signals:
            self.runtime_state.request_emergency_stop()
            return
        if any(signal in {"s", "q"} for signal in signals):
            logger.warning("operator: S/Q suppresses preparation commands in this batch")
            return
        preparation_blocked = False
        for signal in signals:
            if signal == "b":
                if preparation_blocked or not self.request_start():
                    logger.warning("operator: ignored B while stopped or preparing")
            elif signal == "h":
                if self.run_home():
                    return
            elif signal == "t":
                if preparation_blocked:
                    continue
                preparation_blocked = True
                if self.start_epoch is not None or not allow_tactile_reset:
                    logger.warning("T ignored: tactile reset requires idle without pending work")
                    continue
                if self.runtime_state.safety_state != SafetyState.ARMED:
                    continue
                try:
                    logger.info("正在复位触觉，请保持手部无接触；S/Q/Esc 可中断。")
                    self.robot.reset_tactile_sensors(cancel_requested=self.home_abort_requested)
                except OperationCancelled as exc:
                    logger.info("%s", exc)
                finally:
                    self.keyboard.discard_events("t", "h", "b")
                return

    def run_home(self) -> bool:
        with self.runtime_state.motion_lock:
            home_allowed = self.runtime_state.can_move(
                run_id=self.runtime_state.run_id,
                required_safety_state=SafetyState.ARMED,
            )
            if home_allowed:
                self.start_epoch = None
        if not home_allowed:
            logger.warning("operator: ignored H unless ARMED without shutdown, fault or stop")
            return False
        logger.info("正在回初始位…")
        result = home_robot(
            self.runtime_state, self.robot, self.home_planner,
            cancel_requested=self.home_abort_requested,
        )
        with self.runtime_state.motion_lock:
            completed_without_stop = result.ok and self.runtime_state.can_move(
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
        if not completed_without_stop and not interrupted:
            self.errors.append(dict(stage="home", exception_type=None, message=result.reason))
        if completed_without_stop:
            logger.info("已回初始位，按 B 开始。")
        else:
            logger.warning("回初始位未完成，请检查当前姿态：%s", result.reason)
        # HOME 阻塞期间积累的准备按键失效，保留停止与退出请求。
        self.keyboard.discard_events("t", "h", "b")
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
            if not self.runtime_state.can_move(
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
            if not self.runtime_state.can_move(run_id=epoch, required_safety_state=SafetyState.ARMED):
                return
            self.episode = Rollout(
                self.runtime_state, self.runtime, self.info, preparing_epoch=epoch,
                robot=self.robot, sensors=self.sensors, observations=self.observations,
                record_needs=self.record_needs, action_mapper=self.action_mapper, model=self.model,
                needs=self.needs,
                max_running_s=self.max_running_s, capture_ids=self.capture_ids,
                index=len(self.episodes) + 1, output_dir=self.output_dir, record_raw=self.record_raw,
                camera_snapshot=self.camera_snapshot,
            )
        episode = self.episode
        error = None
        try:
            if shutdown:
                episode.stop(
                    str(failure) if failure else "shutdown", run_end_reason=reason,
                )
            else:
                episode.step()
        except BaseException as exc:
            error = exc
        finally:
            if episode.finished and episode.result.index > len(self.episodes):
                try:
                    episode.finish()
                except BaseException as exc:
                    error = error or exc
                finally:
                    result = episode.result
                    self.episodes.append(result)
                    self.start_epoch = None
                    self.keyboard.discard_events("b")
                    if result.started_utc is not None:
                        completed = sum(item.started_utc is not None for item in self.episodes)
                        logger.info("第 %d 轮结束：%s（%s，%.1f s）。", completed,
                                    result.termination_reason, result.termination_details, result.duration_s)
                        if self.num_episodes is not None and completed >= self.num_episodes:
                            self.runtime_state.request_quit()
                        elif self.runtime_state.requests_allow_motion():
                            print("等待下一轮：H 回初始位 → B 开始；Q 退出。", flush=True)
                    try:
                        self.save_results()
                    except BaseException as exc:
                        self.errors.append(error_detail("evaluation_save", exc))
                        error = error or exc
            # pending 资源仍由所属轮次持有，下一轮不能覆盖它。
            if episode.finished and not self.model.has_pending_task and not episode.writer_pending:
                episode.refresh_raw()
                self.episode = None
        if error is not None:
            # 单轮已登记主因并完成收尾；外层只登记不属于单轮的新故障。
            logger.error("policy episode failed", exc_info=(type(error), error, error.__traceback__))
            if not isinstance(error, Exception):
                raise error
            self.runtime_state.mark_error()
            self.runtime_state.end_session(reason=failure_reason(error))

    def run_control_loop(self):
        while self.runtime_state.running and not self.runtime_state.quit_requested:
            self.robot.arm.check_error()
            self.poll_operator(
                allow_tactile_reset=(
                    self.episode is None and not self.model.has_pending_task
                )
            )
            if self.runtime_state.estop_request or self.runtime_state.error_state:
                break
            if self.episode is None or self.episode.run_id is None:
                self.robot.service_idle()
                if not self.robot.stop_required and (
                    self.episode is None or self.episode.finished
                ):
                    self.runtime_state.acknowledge_stop(self.runtime_state.run_id)
            else:
                self.robot.service_feedback()
            self.advance_episode()
            running = self.episode is not None and self.episode.run_id is not None
            time.sleep(0.001 if running else 0.005)
