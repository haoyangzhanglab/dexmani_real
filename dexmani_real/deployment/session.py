"""管理模型、设备与多轮策略实验。"""

import logging
import multiprocessing as mp
import sys
import time
from dataclasses import asdict

from dexmani_real.config.control import HomeConfigurationError
from dexmani_real.control.actions import ActionMapper
from dexmani_real.control.homing import HomePathRejected
from dexmani_real.data.raw import camera_metadata
from dexmani_real.deployment.inference import MODEL_SHUTDOWN_TIMEOUT_S, PolicyWorker
from dexmani_real.deployment.rollout import Rollout, failure_reason, utc_now
from dexmani_real.ipc.sensors import SensorChannels, snapshot_camera_parameters
from dexmani_real.robot.commands import OperationCancelled, RobotError
from dexmani_real.robot.system import Robot
from dexmani_real.runtime.home import build_home_planner, home_robot, initialize_hand
from dexmani_real.runtime.keyboard import Keyboard
from dexmani_real.runtime.lifecycle import (
    CAMERA_READY_TIMEOUT_S,
    POINTCLOUD_READY_TIMEOUT_S,
    SensorWorkers,
    close_runtime,
)
from dexmani_real.runtime.observation import (
    ObservationNeeds,
    check_observation,
    read_observation,
)
from dexmani_real.runtime.state import SafetyState, SessionState
from dexmani_real.sensor.realsense import run_camera
from dexmani_real.teleop.audio import AudioFeedback
from dexmani_real.utils.io import write_json, write_yaml
from dexmani_real.utils.log import configure_logging, error_detail
from dexmani_real.utils.table_plane import load_table_plane, resolve_table_plane_path

logger = logging.getLogger(__name__)
MODEL_STARTUP_TIMEOUT_S = 300.0


class PolicySession:
    def __init__(
        self, runtime, info, *, output_dir, run_config, device, seed,
        max_running_s, num_episodes, record_raw, camera_calibration_path,
        observations,
    ):
        self.run_config = run_config
        self.runtime, self.info = runtime, info
        self.table_plane_abcd = None
        self.home_planner = None
        self.device, self.seed = device, seed
        self.max_running_s, self.num_episodes = max_running_s, num_episodes
        self.record_raw = record_raw
        self.output_dir = output_dir
        self.camera_calibration_path = camera_calibration_path
        self.observations = observations
        record_needs = (
            ObservationNeeds(camera_fields=("rgb", "depth"), required_tactile=("aggregate", "dense"))
            if record_raw else ObservationNeeds()
        )
        self.needs = observations.needs.union(record_needs)
        self.runtime_state = SessionState()
        self.sensors = self.robot = self.keyboard = self.model = None
        self.audio = None
        self.episode = None
        self.episodes, self.errors = [], []
        self.summary = dict(
            started_utc=utc_now(), ended_utc=None, outcome="incomplete", reason=None, shutdown=None,
        )

    def save_results(self, *, overwrite=True):
        write_json(
            dict(
                self.summary, errors=self.errors, episodes=[asdict(result) for result in self.episodes],
                emergency_stop_requested=self.runtime_state.estop_request,
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
        failure: BaseException | None = None
        try:
            try:
                self.prepare_resources()
                self.start_devices()
            except OperationCancelled:
                if self.robot is not None:
                    self.robot.check()
                if not (
                    self.runtime_state.stop_signal.is_set() or self.runtime_state.quit_requested
                    or self.runtime_state.estop_request
                ):
                    raise
                self.runtime_state.request_quit()
                self.summary["reason"] = "startup_cancelled"
            else:
                print("就绪：B 新段 | S/C 保存结束 | D 丢弃 | Q 保存退出 | Esc 急停退出 | 空闲 H 回初始位、T 复位触觉。", flush=True)
                self.run_control_loop()
                self.robot.check()
        except KeyboardInterrupt as exc:
            failure = exc
            self.runtime_state.request_emergency_stop()
        except Exception as exc:
            failure = exc
            self.runtime_state.mark_error()
            self.runtime_state.revoke_motion(reason=failure_reason(exc))
            logger.exception("policy session failed")
        finally:
            exit_code = self.close(failure=failure)
        return exit_code

    def prepare_resources(self):
        from dexmani_policy.deployment import load_policy

        needs = self.needs
        camera_required = needs.camera_worker or self.observations.camera_static
        if camera_required:
            self.runtime.camera.validate()
        self.audio = AudioFeedback()
        self.camera_snapshot = None
        self.action_mapper = ActionMapper.for_mode(
            self.info.action_mode, arm=self.runtime.arm, hand=self.runtime.hand,
            motion=self.runtime.motion, environment=self.runtime.environment,
        )
        self.keyboard = Keyboard(
            get_run_id=lambda: self.runtime_state.run_id,
            estop_callback=self.runtime_state.request_emergency_stop,
            bindings={
                "s": self.runtime_state.request_stop, "c": self.runtime_state.request_stop,
                "d": self.request_discard, "q": self.runtime_state.request_quit,
            },
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
                pointcloud_num_points=self.runtime.pointcloud.num_points if needs.point_cloud else None,
                pointcloud_camera_fields=needs.camera_fields,
            ) if camera_required else None
        )
        self.sensors = SensorWorkers(channels)
        self.robot = Robot(
            self.runtime.arm, self.runtime.hand, runtime_state=self.runtime_state,
            check_services=self.check_services,
        )

    def check_services(self):
        if not self.keyboard.check_health():
            self.runtime_state.request_emergency_stop()
            return False
        if not self.sensors.check():
            raise RobotError("required sensor process exited")
        return True

    def start_devices(self):
        needs = self.needs
        self.robot.connect()
        if needs.camera_worker or self.observations.camera_static:
            self.sensors.start(
                self.process_context.Process(
                    name="camera",
                    target=run_camera,
                    args=(self.sensors.channels, self.runtime.camera),
                ),
                ring=self.sensors.channels.camera_ring,
                timeout_s=CAMERA_READY_TIMEOUT_S,
                may_start=lambda: self.keyboard.check_health() and self.runtime_state.requests_allow_motion(),
            )
            needs_extrinsics = needs.point_cloud or "camera_extrinsic" in self.info.observation_fields
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
            }
            write_yaml(
                self.run_config, self.output_dir / "run_config.yaml",
            )
        if needs.point_cloud:
            from dexmani_real.perception.worker import run_pointcloud_worker

            if self.runtime.pointcloud.remove_table and self.table_plane_abcd is None:
                self.table_plane_abcd = load_table_plane(self.runtime.environment.table_plane_path)
                self.record_table_plane()
            self.sensors.start(
                self.process_context.Process(
                    name="pointcloud", target=run_pointcloud_worker,
                    args=(
                        self.sensors.channels, self.runtime.pointcloud,
                        self.camera_snapshot, self.table_plane_abcd,
                    ),
                ),
                ring=self.sensors.channels.pointcloud_ring,
                timeout_s=POINTCLOUD_READY_TIMEOUT_S,
                may_start=lambda: self.keyboard.check_health() and self.runtime_state.requests_allow_motion(),
            )
        if self.observations.camera_static and not needs.camera_worker:
            self.sensors.shutdown(self.runtime_state)
            self.sensors.channels = None
        self.robot.check()
        startup_run_id = self.runtime_state.arm()
        initialize_hand(
            self.runtime_state, self.robot, run_id=startup_run_id,
            required_tactile=needs.required_tactile,
            cancel_requested=self.home_abort_requested,
        )
        self.warmup_policy()
        self.check_startup()
        self.runtime_state.revoke_motion()

    def close(self, *, failure=None):
        close_errors = []
        try:
            close_runtime(
                self.runtime_state, self.robot, sensors=self.sensors, keyboard=self.keyboard,
            )
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                self.runtime_state.request_emergency_stop()
            logger.exception("policy runtime shutdown failed")
            close_errors.append(("shutdown", exc))
        model_status = "completed"
        if self.model is not None:
            stage = "model_close"
            try:
                self.model.close(self.runtime_state)
                stage = "model_close_wait"
                model_status = ("completed" if self.model.wait_closed(MODEL_SHUTDOWN_TIMEOUT_S)
                                else "pending")
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    self.runtime_state.request_emergency_stop()
                logger.exception("policy %s failed", stage)
                model_status = (
                    "failed" if self.model.close_future is not None and self.model.close_future.done()
                    else "pending"
                )
                close_errors.append((stage, exc))
        writer_pending = self.episode is not None and self.episode.writer_pending
        robot_pending = self.robot is not None and (
            self.robot.arm is not None or self.robot.hand is not None
        )
        sensor_pending = self.sensors is not None and bool(self.sensors.processes)
        keyboard_pending = (
            self.keyboard is not None and self.keyboard.listener is not None
            and self.keyboard.listener.is_alive()
        )
        pending = any((model_status == "pending", writer_pending, robot_pending, sensor_pending, keyboard_pending))
        shutdown = (
            "pending" if pending else "completed"
            if model_status == "completed" and not close_errors else "failed"
        )
        failure_detail = None
        if failure is not None:
            # episode 已完成归因；会话只登记独立的启动/运行时错误。
            if self.episode is not None and self.episode.result.errors:
                failure_detail = self.episode.result.errors[0]
            else:
                failure_detail = error_detail("session", failure)
                self.errors.append(failure_detail)
        self.errors.extend(error_detail(stage, error) for stage, error in close_errors)
        successful = (
            not self.errors and shutdown == "completed"
            and failure is None
            and self.runtime_state.quit_requested
            and not self.runtime_state.estop_request and not self.runtime_state.error_state
        )
        if self.audio is not None:
            try:
                self.audio.close()
            except Exception:
                logger.warning("policy audio shutdown failed", exc_info=True)
            except KeyboardInterrupt as exc:
                logger.exception("policy audio shutdown interrupted")
                self.runtime_state.request_emergency_stop()
                self.errors.append(error_detail("audio_close", exc))
                successful = False
                if shutdown != "pending":
                    shutdown = "failed"
        reason = self.summary["reason"]
        reason_error = failure_detail or (self.errors[0] if self.errors else None)
        if reason_error is not None:
            reason = reason_error["message"] or reason_error["exception_type"]
        elif not successful:
            reason = "shutdown_pending" if pending else "session_stopped"
        self.summary.update(ended_utc=utc_now(), outcome="finished" if successful else "fault", reason=reason, shutdown=shutdown)
        try:
            self.save_results()
        except BaseException as exc:
            self.errors.append(error_detail("session_save", exc))
            if isinstance(exc, KeyboardInterrupt):
                self.runtime_state.request_emergency_stop()
            self.summary["outcome"] = "fault"
            if successful:
                self.summary["reason"] = str(exc) or type(exc).__name__
            successful = False
            logger.exception("session result publication failed")
        if pending:
            logger.warning("Cleanup remains pending; retained resources may outlive this wait")
        return int(not successful)

    def check_startup(self):
        self.robot.service_feedback()
        if not self.runtime_state.requests_allow_motion():
            raise OperationCancelled("startup cancelled")

    def wait_model(self, deadline):
        while True:
            self.check_startup()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("model load/warmup deadline expired")
            future = self.model.poll()
            if future is not None:
                future.result()
                return
            time.sleep(min(0.005, remaining))

    def warmup_policy(self):
        deadline = time.monotonic() + MODEL_STARTUP_TIMEOUT_S
        self.wait_model(deadline)
        self.check_startup()
        row = read_observation(self.sensors.channels, self.robot, needs=self.needs)
        issues = check_observation(row, self.runtime, self.needs)
        if issues:
            raise ValueError("warmup observation: " + "; ".join(issues))
        # 仅预热重复一个快照；features 缓存此后由唯一模型线程独占。
        rows = (self.observations.project(row),) * self.info.n_obs_steps
        kwargs = {}
        if self.runtime.execution.mode == "rtc":
            kwargs = dict(
                rtc_guidance_cap=self.runtime.execution.rtc_guidance_cap,
                rtc_delay=self.runtime.execution.prefetch_steps,
            )
        self.model.warmup(rows, samples=1, **kwargs)
        self.wait_model(time.monotonic() + MODEL_STARTUP_TIMEOUT_S)

    def request_discard(self) -> None:
        # 与保存决定排序；决定之后的 D 仍停止，但不逆转局部 save。
        with self.runtime_state.motion_lock:
            if self.episode is not None:
                self.episode.discard_requested = True
            self.runtime_state.request_stop()

    def poll_operator(self, *, allow_tactile_reset: bool = False) -> int | None:
        return self.handle_command_batch(
            self.keyboard.poll(timeout=0),
            allow_tactile_reset=allow_tactile_reset,
        )

    def handle_command_batch(self, signals, *, allow_tactile_reset=False) -> int | None:
        keys = {event.key for event in signals}
        if "esc" in keys:
            self.runtime_state.request_emergency_stop()
            return None
        if "q" in keys:
            self.audio.play("quit")
        if keys.intersection({"s", "c", "d", "q"}):
            logger.debug("operator: S/C/D/Q suppresses preparation commands in this batch")
            return None
        for event in signals:
            signal = event.key
            if signal == "b":
                if self.episode is None and self.runtime_state.can_move(
                    run_id=event.run_id, required_safety_state=SafetyState.ARMED,
                ):
                    return event.run_id
            elif signal == "h":
                if self.run_home(event.run_id):
                    return None
            elif signal == "t":
                if not self.runtime.hand.enabled:
                    logger.info("T ignored: hand is disabled")
                    continue
                if not allow_tactile_reset or self.episode is not None:
                    logger.warning("T ignored: tactile reset requires idle")
                    continue
                if not self.runtime_state.can_move(
                    run_id=event.run_id, required_safety_state=SafetyState.ARMED,
                ):
                    continue
                try:
                    logger.info("正在复位触觉，请保持手部无接触；S/Q/Esc 可中断。")
                    self.robot.reset_tactile_sensors(
                        run_id=event.run_id,
                        cancel_requested=self.home_abort_requested,
                        required_tactile=self.needs.required_tactile,
                    )
                except OperationCancelled as exc:
                    self.robot.check()
                    logger.info("%s", exc)
                finally:
                    self.runtime_state.revoke_motion()
                return None
        return None

    def run_home(self, run_id) -> bool:
        home_allowed = (
            self.episode is None
            and self.runtime_state.can_move(
                run_id=run_id, required_safety_state=SafetyState.ARMED,
            )
        )
        if not home_allowed:
            logger.warning("H 仅限空闲时使用；活动段请先按 S。")
            return False
        logger.info("正在回初始位…")
        self.audio.play("home")
        try:
            home_robot(
                self.runtime_state, self.robot, self.get_home_planner,
                run_id=run_id,
                cancel_requested=self.home_abort_requested,
            )
        except OperationCancelled as exc:
            self.robot.check()
            logger.info("回初始位已中断：%s", exc)
        except (HomeConfigurationError, HomePathRejected) as exc:
            self.robot.check()
            logger.warning("本次 HOME 未完成（手部可能已回位）：%s", exc)
        else:
            self.audio.play("home_done")
            logger.info("已回初始位，按 B 开始。")
        finally:
            self.runtime_state.revoke_motion()
        return True

    def get_home_planner(self):
        if self.home_planner is None:
            self.home_planner = build_home_planner(
                self.runtime.environment, self.runtime.arm, table_plane_abcd=self.table_plane_abcd,
            )
            if self.table_plane_abcd is None:
                self.table_plane_abcd = self.home_planner.table_plane_abcd
                self.record_table_plane()
        return self.home_planner

    def record_table_plane(self):
        self.run_config.update(
            table_plane_abcd=self.table_plane_abcd,
            table_plane_path=str(resolve_table_plane_path(self.runtime.environment.table_plane_path)),
        )
        write_yaml(self.run_config, self.output_dir / "run_config.yaml")

    def home_abort_requested(self) -> bool:
        self.robot.check()
        return not self.runtime_state.requests_allow_motion()

    def run_control_loop(self):
        while not self.runtime_state.quit_requested:
            run_id = self.poll_operator(allow_tactile_reset=True)
            self.robot.service_idle()
            if self.runtime_state.estop_request or self.runtime_state.error_state:
                break
            if run_id is not None and not self.runtime_state.can_move(
                run_id=run_id, required_safety_state=SafetyState.ARMED,
            ):
                run_id = None
            if run_id is None:
                time.sleep(0.005)
                continue
            episode = self.episode = Rollout(
                self.runtime_state, self.runtime, self.info, run_id=run_id,
                robot=self.robot, sensors=self.sensors, observations=self.observations,
                needs=self.needs, action_mapper=self.action_mapper, model=self.model,
                max_running_s=self.max_running_s,
                index=len(self.episodes) + 1, output_dir=self.output_dir, record_raw=self.record_raw,
                camera_snapshot=self.camera_snapshot, audio=self.audio,
            )
            try:
                episode.run(poll_operator=self.poll_operator)
            finally:
                primary_failure = sys.exc_info()[0] is not None
                self.episodes.append(episode.result)
                # 正常返回已同步收尾；失败段保留资源和主错，供最终关闭诊断。
                if not primary_failure:
                    self.episode = None
                try:
                    self.save_results()
                except BaseException as exc:
                    if isinstance(exc, KeyboardInterrupt):
                        self.runtime_state.request_emergency_stop()
                    if not primary_failure:
                        raise
                    logger.exception("episode result write failed")
                    self.errors.append(error_detail("episode_save", exc))
            result = episode.result
            if result.started_utc is not None:
                completed = sum(item.started_utc is not None for item in self.episodes)
                logger.info("第 %d 轮结束：%s（%s，%.1f s）。", completed,
                            result.termination_reason, result.termination_details, result.duration_s)
                if self.num_episodes is not None and completed >= self.num_episodes:
                    self.runtime_state.request_quit()
                elif self.runtime_state.requests_allow_motion():
                    print("等待下一轮：H 回初始位 → B 开始；Q 退出。", flush=True)
