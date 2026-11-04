"""管理模型、设备与多轮策略实验。"""

import logging
import multiprocessing as mp
import sys
import time
from collections.abc import Sequence
from concurrent.futures import wait
from dataclasses import asdict
from typing import cast

from dexmani_real.control.actions import ActionMapper
from dexmani_real.data.raw import camera_metadata
from dexmani_real.deployment.inference import MODEL_SHUTDOWN_TIMEOUT_S, PolicyWorker
from dexmani_real.deployment.rollout import Rollout, utc_now
from dexmani_real.ipc.sensors import SensorChannels, snapshot_camera_parameters
from dexmani_real.robot.commands import OperationCancelled, RobotError
from dexmani_real.robot.system import Robot
from dexmani_real.runtime.home import HomeConfigurationError, home_arm, home_hand
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
from dexmani_real.runtime.state import (
    INITIAL_STATE_TIMEOUT_S,
    RunEndReason,
    SessionState,
)
from dexmani_real.sensor.realsense import run_camera
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
        self.model_needs = observations.needs
        self.record_needs = record_needs
        self.capture_needs = self.model_needs.union(record_needs)
        self.runtime_state = SessionState()
        self.startup_cancel_epoch = self.runtime_state.cancel_epoch
        self.sensors = self.robot = self.keyboard = self.model = None
        self.episode = None
        self.episodes, self.errors = [], []
        self.summary = dict(
            started_utc=utc_now(), ended_utc=None, termination_reason=None,
        )

    def save_auxiliary(self, name, writer, payload):
        if self.output_dir is None:
            return
        path = self.output_dir / name
        try:
            writer(payload, path)
        except Exception:
            logger.exception("辅助输出保存失败：%s", path)

    def save_results(self):
        self.save_auxiliary("session_result.json", write_json, dict(
            self.summary, errors=self.errors, episodes=[asdict(result) for result in self.episodes],
            emergency_stop_requested=self.runtime_state.estop_request,
        ))

    def run(self):
        configure_logging()
        print(
            f"执行：{self.info.checkpoint_path.name} ({self.info.weights}) | NFE={self.info.inference_steps} | "
            f"A_exec={self.runtime.execution.action_steps} | {self.runtime.execution.mode} | "
            f"{1.0 / self.info.control_dt_s:g} Hz"
        )
        print("正在初始化模型和设备…", flush=True)
        failure: BaseException | None = None
        try:
            try:
                self.prepare_resources()
                self.start_devices()
            except OperationCancelled:
                if self.robot is not None:
                    self.robot.check()
                if not (
                    self.runtime_state.cancel_epoch != self.startup_cancel_epoch
                    or self.runtime_state.quit_requested or self.runtime_state.estop_request
                ):
                    raise
                self.runtime_state.request_quit()
                self.summary["termination_reason"] = "startup_cancelled"
            else:
                print("就绪：B 新段 | S/C 保存结束 | D 丢弃 | Q 保存退出 | Esc 急停退出 | 空闲 H 先手后臂回位、T 复位触觉。", flush=True)
                self.run_control_loop()
                self.robot.check()
        except KeyboardInterrupt as exc:
            failure = exc
            self.runtime_state.request_emergency_stop()
        except BaseException as exc:
            failure = exc
            self.runtime_state.stop_signal.set()
            logger.exception("policy session failed")
        finally:
            exit_code = self.close(failure=failure)
        return exit_code

    def prepare_resources(self):
        from dexmani_policy.deployment import load_policy

        needs = self.capture_needs
        camera_required = needs.camera_worker or self.observations.camera_static
        if camera_required:
            self.runtime.camera.validate()
        self.camera_snapshot = None
        self.action_mapper = ActionMapper.for_mode(
            self.info.action_mode, arm=self.runtime.arm, hand=self.runtime.hand,
            motion=self.runtime.motion, environment=self.runtime.environment,
        )
        self.keyboard = Keyboard(
            estop_callback=self.runtime_state.request_emergency_stop,
            bindings={
                "s": self.runtime_state.request_stop, "c": self.runtime_state.request_stop,
                "d": self.runtime_state.request_stop, "q": self.runtime_state.request_quit,
            },
        )
        self.keyboard.start()
        if not self.startup_allowed():
            raise OperationCancelled("startup cancelled before model load")
        self.model = PolicyWorker(
            lambda: load_policy(self.info, device=self.device, seed=self.seed),
            build_observation=self.observations.build,
        )
        self.model.load()
        self.wait_model(time.monotonic() + MODEL_STARTUP_TIMEOUT_S)

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
        needs = self.capture_needs
        self.check_startup()
        self.robot.connect()
        self.check_startup()
        if needs.camera_worker or self.observations.camera_static:
            self.sensors.start(
                self.process_context.Process(
                    name="camera",
                    target=run_camera,
                    args=(self.sensors.channels, self.runtime.camera),
                ),
                ring=self.sensors.channels.camera_ring,
                timeout_s=CAMERA_READY_TIMEOUT_S,
                may_start=self.startup_allowed,
            )
            needs_extrinsics = needs.point_cloud or "camera_extrinsic" in self.info.observation_fields
            self.camera_snapshot = snapshot_camera_parameters(
                self.sensors.channels, self.camera_calibration_path,
                require_extrinsics=needs_extrinsics,
                include_extrinsics=needs_extrinsics or self.record_raw,
            )
        self.observations.configure_camera(self.camera_snapshot)
        if self.camera_snapshot is not None:
            self.run_config["effective_camera"] = {
                "camera_T_xarm_base_from_color": None,
                **camera_metadata(self.camera_snapshot),
            }

        if needs.point_cloud:
            from dexmani_real.perception.worker import run_pointcloud_worker

            if self.runtime.pointcloud.remove_table and self.table_plane_abcd is None:
                self.table_plane_abcd = load_table_plane(self.runtime.environment.table_plane_path)
                self.run_config.update(
                    table_plane_abcd=self.table_plane_abcd,
                    table_plane_path=str(resolve_table_plane_path(self.runtime.environment.table_plane_path)),
                )
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
                may_start=self.startup_allowed,
            )
        if self.observations.camera_static and not needs.camera_worker:
            self.sensors.shutdown(self.runtime_state)
            self.sensors.channels = None
        self.check_startup()
        self.save_auxiliary("run_config.yaml", write_yaml, self.run_config)
        self.warmup_policy()
        self.check_startup()
        self.runtime_state.stop_signal.set()

    def close(self, *, failure=None):
        if failure is not None:
            self.errors.append(error_detail("session", failure))
        try:
            close_runtime(
                self.runtime_state, self.robot, sensors=self.sensors, keyboard=self.keyboard,
            )
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                self.runtime_state.request_emergency_stop()
            logger.exception("policy runtime shutdown failed")
            self.errors.append(error_detail("shutdown", exc))
        if self.episode is not None:
            try:
                self.episode.finish_recording()
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    self.runtime_state.request_emergency_stop()
                logger.exception("episode recording finalization failed")
                self.errors.append(error_detail("recorder_finalize", exc))
        if self.model is not None:
            try:
                closed = self.model.close()
                if closed not in wait((closed,), timeout=MODEL_SHUTDOWN_TIMEOUT_S).done:
                    raise TimeoutError("model close still pending; task handle retained")
                closed.result()
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    self.runtime_state.request_emergency_stop()
                logger.exception("policy model close failed")
                self.errors.append(error_detail("model_close", exc))
        failed = bool(self.errors or self.runtime_state.estop_request)
        if self.runtime_state.estop_request:
            self.summary["termination_reason"] = "estop"
        elif failed:
            self.summary["termination_reason"] = "failed"
        elif self.summary["termination_reason"] is None:
            self.summary["termination_reason"] = "quit"
        self.summary["ended_utc"] = utc_now()
        try:
            self.save_results()
        except KeyboardInterrupt:
            self.runtime_state.request_emergency_stop()
            self.summary["termination_reason"] = "estop"
            raise
        return int(failed)

    def startup_allowed(self):
        return (self.keyboard.check_health() and self.runtime_state.requests_allow_motion()
                and self.runtime_state.cancel_epoch == self.startup_cancel_epoch)

    def check_startup(self):
        if self.robot is not None:
            self.robot.service_feedback()
        if not self.startup_allowed():
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
                self.check_startup()
                return
            time.sleep(min(0.005, remaining))

    def warmup_policy(self):
        self.check_startup()
        observation_deadline = time.monotonic() + INITIAL_STATE_TIMEOUT_S
        while True:
            self.check_startup()
            row = read_observation(self.sensors.channels, self.robot, needs=self.model_needs)
            issues = check_observation(row, self.runtime, self.model_needs)
            if not issues:
                break
            if time.monotonic() >= observation_deadline:
                raise ValueError("warmup observation: " + "; ".join(issues))
            time.sleep(0.005)
        # 首帧填充与正式预测一致。
        rows = (self.observations.project(row),) * self.info.n_obs_steps
        kwargs = {}
        if self.runtime.execution.mode == "rtc":
            kwargs = dict(
                rtc_guidance_cap=self.runtime.execution.rtc_guidance_cap,
                rtc_delay=self.runtime.execution.prefetch_steps,
            )
        self.check_startup()
        self.model.warmup(rows, samples=1, **kwargs)
        self.wait_model(time.monotonic() + MODEL_STARTUP_TIMEOUT_S)

    def poll_operator(self, *, allow_tactile_reset: bool = False) -> int | None:
        return self.handle_command_batch(
            self.keyboard.poll(self.runtime_state, timeout=0)[0],
            allow_tactile_reset=allow_tactile_reset,
        )

    def handle_command_batch(self, signals: Sequence[str], *, allow_tactile_reset=False) -> int | None:
        keys = set(signals)
        if "d" in keys and self.episode is not None and not self.episode.finished:
            self.episode.discard_requested = True
        if self.episode is not None and keys.intersection({"s", "c", "d", "q", "esc"}):
            reason = (RunEndReason.ESTOP if "esc" in keys else
                      RunEndReason.QUIT if "q" in keys else RunEndReason.OPERATOR)
            self.episode.stop("operator input", run_end_reason=reason)
        if "esc" in keys:
            self.runtime_state.request_emergency_stop()
            return None
        if keys.intersection({"s", "c", "d", "q"}):
            logger.debug("operator: S/C/D/Q suppresses preparation commands in this batch")
            return None
        for signal in signals:
            if signal not in {"b", "h", "t"}:
                continue
            cancel_epoch = self.keyboard.accept_operation(self.runtime_state)
            if cancel_epoch is None:
                return None
            if signal == "b":
                if (self.episode is None and self.runtime_state.cancel_epoch == cancel_epoch
                        and self.runtime_state.requests_allow_motion()):
                    return cancel_epoch
            elif signal == "h":
                if self.run_home(cancel_epoch):
                    return None
            elif signal == "t":
                if not allow_tactile_reset or self.episode is not None:
                    logger.warning("T ignored: tactile reset requires idle")
                    continue
                if (self.runtime_state.cancel_epoch != cancel_epoch
                        or not self.runtime_state.requests_allow_motion()):
                    continue
                try:
                    logger.info("正在复位触觉，请保持手部无接触；S/Q/Esc 可中断。")
                    self.robot.reset_tactile_sensors(
                        cancel_epoch=cancel_epoch,
                        cancel_requested=self.home_abort_requested,
                    )
                except OperationCancelled as exc:
                    self.robot.check()
                    logger.info("%s", exc)
                return None
        return None

    def run_home(self, cancel_epoch) -> bool:
        home_allowed = (
            self.episode is None
            and self.runtime_state.cancel_epoch == cancel_epoch
            and self.runtime_state.requests_allow_motion()
        )
        if not home_allowed:
            logger.warning("H 仅限空闲时使用；活动段请先按 S。")
            return False
        # HOME 只由已完成设备初始化的待命循环调用。
        robot = cast(Robot, self.robot)
        logger.info("正在回初始位…")
        try:
            home_hand(
                self.runtime_state, robot,
                cancel_epoch=cancel_epoch,
                cancel_requested=self.home_abort_requested,
            )
            home_arm(
                self.runtime_state, robot,
                cancel_epoch=cancel_epoch,
                cancel_requested=self.home_abort_requested,
            )
        except OperationCancelled as exc:
            robot.stop()
            robot.check()
            if (self.runtime_state.cancel_epoch == cancel_epoch
                    and self.runtime_state.requests_allow_motion()):
                raise
            if robot.arm is None or not robot.arm.is_connected:
                self.runtime_state.request_quit()
            logger.info("回初始位已中断：%s", exc)
        except HomeConfigurationError as exc:
            robot.check()
            logger.warning("本次 HOME 未完成（手部可能已回位）：%s", exc)
        else:
            logger.info("已回初始位，按 B 开始。")
        return True

    def home_abort_requested(self) -> bool:
        self.robot.check()
        return not self.runtime_state.requests_allow_motion()

    def run_control_loop(self):
        while not self.runtime_state.quit_requested:
            self.keyboard.allow_start()
            cancel_epoch = self.poll_operator(allow_tactile_reset=True)
            self.robot.service_idle()
            if self.runtime_state.estop_request:
                break
            if cancel_epoch is not None and (
                self.runtime_state.cancel_epoch != cancel_epoch
                or not self.runtime_state.requests_allow_motion()
            ):
                cancel_epoch = None
            if cancel_epoch is None:
                time.sleep(0.005)
                continue
            episode = self.episode = Rollout(
                self.runtime_state, self.runtime, self.info, cancel_epoch=cancel_epoch,
                robot=self.robot, sensors=self.sensors, observations=self.observations,
                record_needs=self.record_needs, action_mapper=self.action_mapper, model=self.model,
                max_running_s=self.max_running_s,
                index=len(self.episodes) + 1, output_dir=self.output_dir, record_raw=self.record_raw,
                camera_snapshot=self.camera_snapshot,
            )
            try:
                episode.run(poll_operator=self.poll_operator)
            finally:
                primary_failure = sys.exc_info()[0] is not None
                self.episodes.append(episode.result)
                # 正常返回已同步收尾；失败段保留资源和主错，供最终关闭诊断。
                if (not primary_failure and not self.runtime_state.quit_requested
                        and not self.runtime_state.estop_request):
                    self.episode = None
                    self.save_results()
            result = episode.result
            if result.started_utc is not None:
                completed = sum(item.started_utc is not None for item in self.episodes)
                logger.info("第 %d 轮结束：%s（%s，%.1f s）。", completed,
                            result.termination_reason, result.termination_details, result.duration_s)
                if self.num_episodes is not None and completed >= self.num_episodes:
                    self.runtime_state.request_quit()
                elif self.runtime_state.requests_allow_motion():
                    print("等待下一轮：B 开始，按需 H 回初始位；Q 退出。", flush=True)
