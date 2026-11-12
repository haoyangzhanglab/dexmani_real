"""管理模型、设备与多轮策略实验。"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from collections.abc import Sequence
from concurrent.futures import wait
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

from dexmani_real.config.calibration_data import CAMERAS_PATH
from dexmani_real.config.deployment import POLICY_RUNTIME_SECTIONS
from dexmani_real.control.actions import ActionMapper
from dexmani_real.data.raw import camera_metadata
from dexmani_real.data.recorder import format_recording_result
from dexmani_real.deployment.inference import MODEL_SHUTDOWN_TIMEOUT_S, PolicyWorker
from dexmani_real.deployment.observation import ObservationBuilder
from dexmani_real.deployment.preflight import (
    resolve_deployment_config,
    resolve_hoi_inputs,
)
from dexmani_real.deployment.rollout import Rollout, utc_now
from dexmani_real.ipc.sensors import SensorChannels, snapshot_camera_parameters
from dexmani_real.robot.commands import (
    FeedbackUnavailable,
    HomeTimeout,
    OperationCancelled,
    RobotError,
)
from dexmani_real.robot.system import Robot
from dexmani_real.runtime.home import (
    HomeConfigurationError,
    HomePathRejected,
    home_robot,
)
from dexmani_real.runtime.keyboard import Keyboard
from dexmani_real.runtime.lifecycle import (
    CAMERA_READY_TIMEOUT_S,
    POINTCLOUD_READY_TIMEOUT_S,
    SensorWorkers,
    close_runtime,
)
from dexmani_real.runtime.observation import (
    RAW_RECORDING_NEEDS,
    ObservationNeeds,
    check_device_feedback,
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

if TYPE_CHECKING:
    from dexmani_policy.deployment import PolicyInfo


def session_directory(root: Path) -> Path:
    base = root.expanduser().resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = base / f"session_{stamp}"
    dedup = 1
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = base / f"session_{stamp}_{dedup:02d}"
            dedup += 1


class PolicySession:
    def __init__(
        self, runtime, policy, *, weights, inference_steps, output_root, device, seed,
        max_running_s, num_episodes, record_raw, camera_calibration_path,
    ):
        self.runtime, self.policy = runtime, policy
        self.weights, self.inference_steps = weights, inference_steps
        self.info: PolicyInfo
        self.observations: ObservationBuilder
        self.table_plane_abcd = None
        self.device, self.seed = device, seed
        self.max_running_s, self.num_episodes = max_running_s, num_episodes
        self.record_raw = record_raw
        self.output_root = (Path(output_root) if output_root is not None else
                            Path(__file__).resolve().parents[2] / "rollouts" if record_raw else None)
        self.output_dir = None
        self.camera_calibration_path = camera_calibration_path
        self.camera_snapshot = None
        self.runtime_state = SessionState()
        self.startup_cancel_epoch = self.runtime_state.cancel_epoch
        self.sensors = SensorWorkers()
        self.keyboard = Keyboard(
            estop_callback=self.runtime_state.request_emergency_stop,
            bindings={
                "s": self.runtime_state.request_stop, "c": self.runtime_state.request_stop,
                "d": self.runtime_state.request_stop, "q": self.runtime_state.request_quit,
            },
        )
        self.robot: Robot | None = None
        self.model: PolicyWorker | None = None
        self.episode = None
        self.episodes, self.errors = [], []
        self.summary = {"started_utc": utc_now(), "ended_utc": None, "termination_reason": None}

    def save_run_config(self):
        if self.output_dir is None:
            return
        payload = {
            "policy": asdict(self.info),
            "runtime": {name: asdict(getattr(self.runtime, name)) for name in POLICY_RUNTIME_SECTIONS},
            "camera_calibration_path": str(self.camera_calibration_path or CAMERAS_PATH),
            "effective_camera": ({"camera_T_xarm_base_from_color": None,
                                  **camera_metadata(self.camera_snapshot)}
                                 if self.camera_snapshot is not None else None),
            "table_plane_abcd": self.table_plane_abcd,
            "table_plane_path": (str(resolve_table_plane_path(self.runtime.environment.table_plane_path))
                                 if self.table_plane_abcd is not None else None),
            "seed": self.seed, "device": self.device,
            "num_episodes": self.num_episodes, "max_duration_s": self.max_running_s,
            "record_raw": self.record_raw,
        }
        write_yaml(payload, self.output_dir / "run_config.yaml")

    def save_results(self):
        if self.episode is not None and self.episode.recorder is not None:
            self.episode.result.raw = self.episode.recorder.result()
        if self.output_dir is None:
            return
        payload = dict(self.summary, errors=self.errors,
                       episodes=[asdict(result) for result in self.episodes],
                       emergency_stop_requested=self.runtime_state.estop_request)
        write_json(payload, self.output_dir / "session_result.json")

    def run(self):
        configure_logging()
        print("正在加载策略模型…", flush=True)
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
                print("就绪：B 新段 | S/C 保存结束 | D 丢弃 | Q 保存退出 | Esc 急停退出 | 空闲 H 先臂后手回位、T 复位触觉。", flush=True)
                self.run_control_loop()
                cast(Robot, self.robot).check()
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

        self.keyboard.start()
        if not self.startup_allowed():
            raise OperationCancelled("startup cancelled before model load")
        self.model = PolicyWorker(
            lambda: load_policy(self.policy, weights=self.weights, inference_steps=self.inference_steps,
                                device=self.device, seed=self.seed),
            build_observation=lambda rows: self.observations.build(rows),
        )
        self.model.load()
        self.info = self.wait_model()
        self.hoi_inputs = resolve_hoi_inputs(self.info)
        declared_execution = self.runtime.execution
        self.runtime = resolve_deployment_config(self.info, self.runtime)
        self.observations = ObservationBuilder(self.info, self.runtime)
        self.record_needs = RAW_RECORDING_NEEDS if self.record_raw else ObservationNeeds()
        model_needs = self.observations.needs
        self.sensor_needs = model_needs.union(self.record_needs)
        # 派生 Ring 仅携带模型图像；Raw 在 Rollout 独立读取原始 RGB-D。
        self.read_needs = (
            replace(self.sensor_needs, camera_fields=model_needs.camera_fields)
            if model_needs.hoi or model_needs.point_cloud else self.sensor_needs
        )
        needs = self.sensor_needs
        camera_required = needs.camera_worker or self.observations.camera_static
        if camera_required:
            self.runtime.camera.validate()
        self.action_mapper = ActionMapper.for_mode(
            self.info.action_mode, arm=self.runtime.arm, hand=self.runtime.hand,
            motion=self.runtime.motion, environment=self.runtime.environment,
        )
        if self.output_root is not None:
            self.output_dir = session_directory(self.output_root)
        print(
            f"模型：{self.info.checkpoint_path} ({self.info.weights}) | inference_steps={self.info.inference_steps} | "
            f"action_mode={self.info.action_mode}"
        )
        print(
            "执行："
            f"action_steps={self.runtime.execution.action_steps} "
            f"({'Checkpoint' if declared_execution.action_steps is None else 'RealConfig'}) | "
            f"mode={self.runtime.execution.mode} (RealConfig) | "
            f"{1.0 / self.runtime.execution.control_dt_s:g} Hz / "
            f"control_dt_s={self.runtime.execution.control_dt_s:g} "
            f"({'Checkpoint' if declared_execution.control_dt_s is None else 'RealConfig'})"
        )
        duration = f"{self.max_running_s:g} s" if self.max_running_s is not None else "无限制"
        print(f"评估：{self.num_episodes if self.num_episodes is not None else '不限'} 轮 | 每轮预算 {duration} | 输入：{', '.join(self.info.observation_fields)}")
        for name, field in (("pointcloud", "point_cloud"), ("hand_pointcloud", "hand_pointcloud")):
            if field in self.info.observation_fields:
                print(f"{field} ({'RealConfig' if field == 'hand_pointcloud' else 'Checkpoint'}): {asdict(getattr(self.runtime, name))}")
        if "depth" in self.info.observation_fields:
            print(f"depth (Checkpoint): {self.info.depth_config}")
        print(f"输出：{self.output_dir}" if self.output_dir is not None else "输出：仅终端")

        self.process_context = mp.get_context("spawn")
        channels = (
            SensorChannels(
                self.process_context, camera=self.runtime.camera,
                depth_enabled=needs.hoi or needs.point_cloud or "depth" in needs.camera_fields,
                hoi=needs.hoi,
                pointcloud_num_points=self.runtime.pointcloud.num_points if needs.point_cloud else None,
                camera_fields=self.read_needs.camera_fields,
            ) if camera_required else None
        )
        self.sensors.channels = channels
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
        needs = self.sensor_needs
        robot = cast(Robot, self.robot)
        self.check_startup()
        robot.connect()
        # 后续初始化可能阻塞超过反馈 TTL；保持停止，由有界观测等待接纳新帧。
        robot.stop()
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
            needs_extrinsics = needs.hoi or needs.point_cloud or "camera_extrinsic" in self.info.observation_fields
            self.camera_snapshot = snapshot_camera_parameters(
                self.sensors.channels, self.camera_calibration_path,
                require_extrinsics=needs_extrinsics,
                include_extrinsics=needs_extrinsics or self.record_raw,
            )
        self.observations.configure_camera(self.camera_snapshot)
        if needs.hoi:
            from dexmani_real.perception.hoi_worker import run_hoi_worker
            channels = self.sensors.channels
            self.sensors.start(self.process_context.Process(
                name="hoi", target=run_hoi_worker,
                args=(channels, self.camera_snapshot, *self.hoi_inputs),
            ), ring=channels.hoi_ring, may_start=self.startup_allowed)
            deadline = time.monotonic() + 120
            while not channels.hoi_ready.value:
                self.check_startup()
                self.check_services()
                if time.monotonic() > deadline:
                    raise TimeoutError("HOI model startup timeout")
                time.sleep(0.01)

        if needs.point_cloud:
            from dexmani_real.perception.worker import run_pointcloud_worker

            if self.runtime.pointcloud.remove_table:
                self.table_plane_abcd = load_table_plane(self.runtime.environment.table_plane_path)
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
        self.save_run_config()
        if not needs.hoi:
            self.warmup_policy()
        self.check_startup()

    def close(self, *, failure=None):
        exit_signal = failure if isinstance(failure, SystemExit) else None
        if failure is not None:
            self.errors.append(error_detail("session", failure))
        try:
            close_runtime(
                self.runtime_state, self.robot, sensors=self.sensors, keyboard=self.keyboard,
            )
        except BaseException as exc:
            if isinstance(exc, SystemExit):
                exit_signal = exit_signal or exc
            if isinstance(exc, KeyboardInterrupt):
                self.runtime_state.request_emergency_stop()
            logger.exception("policy runtime shutdown failed")
            self.errors.append(error_detail("shutdown", exc))
        if self.episode is not None:
            try:
                self.episode.finish_recording()
            except BaseException as exc:
                if isinstance(exc, SystemExit):
                    exit_signal = exit_signal or exc
                if isinstance(exc, KeyboardInterrupt):
                    self.runtime_state.request_emergency_stop()
                logger.exception("episode recording finalization failed")
                self.episode.fail("recording_finalize", exc)
        if self.model is not None:
            try:
                closed = self.model.close()
                if closed not in wait((closed,), timeout=MODEL_SHUTDOWN_TIMEOUT_S).done:
                    raise TimeoutError("model close still pending; task handle retained")
                closed.result()
            except BaseException as exc:
                if isinstance(exc, SystemExit):
                    exit_signal = exit_signal or exc
                if isinstance(exc, KeyboardInterrupt):
                    self.runtime_state.request_emergency_stop()
                logger.exception("policy model close failed")
                if self.episode is not None:
                    self.episode.fail("model_close", exc)
                else:
                    self.errors.append(error_detail("model_close", exc))
        failed = bool(self.errors or any(result.errors for result in self.episodes)
                      or self.runtime_state.estop_request)
        if self.runtime_state.estop_request:
            self.summary["termination_reason"] = "estop"
        elif failed:
            self.summary["termination_reason"] = "failed"
        elif self.summary["termination_reason"] is None:
            self.summary["termination_reason"] = "quit"
        self.summary["ended_utc"] = utc_now()
        try:
            self.save_results()
        except BaseException as exc:
            if isinstance(exc, SystemExit):
                exit_signal = exit_signal or exc
            if isinstance(exc, KeyboardInterrupt):
                self.runtime_state.request_emergency_stop()
            self.errors.append(error_detail("results_write", exc))
            self.summary["termination_reason"] = "estop" if self.runtime_state.estop_request else "failed"
            failed = True
            logger.exception("session result write failed")
        if exit_signal is not None:
            raise exit_signal
        return int(failed)

    def startup_allowed(self):
        return (self.keyboard.check_health() and self.runtime_state.requests_allow_motion()
                and self.runtime_state.cancel_epoch == self.startup_cancel_epoch)

    def check_startup(self):
        if self.robot is not None:
            self.robot.service_feedback()
        if not self.startup_allowed():
            raise OperationCancelled("startup cancelled")

    def wait_model(self):
        model = cast(PolicyWorker, self.model)
        while True:
            self.check_startup()
            future = model.poll()
            if future is not None:
                result = future.result()
                self.check_startup()
                return result
            time.sleep(0.005)

    def warmup_policy(self):
        self.check_startup()
        observation_deadline = time.monotonic() + INITIAL_STATE_TIMEOUT_S
        while True:
            self.check_startup()
            row = read_observation(self.sensors.channels, self.robot, needs=self.observations.needs)
            issues = (check_device_feedback(row, self.runtime)
                      + check_observation(row, self.runtime, self.observations.needs))
            if not issues:
                break
            if time.monotonic() >= observation_deadline:
                raise ValueError("warmup observation: " + "; ".join(issues))
            time.sleep(0.005)
        # 首帧填充与正式预测一致。
        rows = (self.observations.project(row),) * self.info.n_obs_steps
        kwargs = {}
        if self.runtime.execution.mode == "rtc":
            kwargs = {
                "rtc_guidance_cap": self.runtime.execution.rtc_guidance_cap,
                "rtc_delay": self.runtime.execution.prefetch_steps,
            }
        self.check_startup()
        cast(PolicyWorker, self.model).warmup(rows, **kwargs)
        self.wait_model()

    def poll_operator(self, *, timeout=0) -> int | None:
        return self.handle_command_batch(
            self.keyboard.poll(self.runtime_state, timeout=timeout)[0],
        )

    def handle_command_batch(self, signals: Sequence[str]) -> int | None:
        robot = cast(Robot, self.robot)
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
                if self.episode is not None:
                    logger.warning("T ignored: tactile reset requires idle")
                    continue
                if (self.runtime_state.cancel_epoch != cancel_epoch
                        or not self.runtime_state.requests_allow_motion()):
                    continue
                try:
                    logger.info("正在复位触觉，请保持手部无接触；S/Q/Esc 可中断。")
                    robot.reset_tactile_sensors(
                        cancel_epoch=cancel_epoch,
                        cancel_requested=self.home_abort_requested,
                    )
                except OperationCancelled as exc:
                    robot.check()
                    logger.info("%s", exc)
                return None
        return None

    def run_home(self, cancel_epoch) -> bool:
        """处理空闲态 HOME 请求；返回是否已处理，不表示回位成功。"""
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
            home_robot(
                self.runtime_state, robot,
                environment=self.runtime.environment,
                cancel_epoch=cancel_epoch,
                cancel_requested=self.home_abort_requested,
            )
        except (FeedbackUnavailable, HomeTimeout) as exc:
            # 底层 HOME 已确认停止；这里检查硬故障并保留操作指令优先级。
            robot.check()
            self.poll_operator()
            if (self.runtime_state.cancel_epoch != cancel_epoch
                    or not self.runtime_state.requests_allow_motion()):
                logger.info("回初始位已中断：%s", exc)
            else:
                logger.warning("本次 HOME 未完成：%s", exc)
        except OperationCancelled as exc:
            robot.check()
            if (self.runtime_state.cancel_epoch == cancel_epoch
                    and self.runtime_state.requests_allow_motion()):
                raise
            logger.info("回初始位已中断：%s", exc)
        except (HomeConfigurationError, HomePathRejected) as exc:
            robot.check()
            logger.warning("本次 HOME 未完成：%s", exc)
        else:
            logger.info("已回初始位，按 B 开始。")
        return True

    def home_abort_requested(self) -> bool:
        cast(Robot, self.robot).check()
        return not self.runtime_state.requests_allow_motion()

    def run_control_loop(self):
        robot = cast(Robot, self.robot)
        while not self.runtime_state.quit_requested:
            self.keyboard.allow_start()
            cancel_epoch = self.poll_operator()
            robot.service_idle()
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
                read_needs=self.read_needs, record_needs=self.record_needs,
                action_mapper=self.action_mapper, model=self.model,
                max_running_s=self.max_running_s,
                index=len(self.episodes) + 1, output_dir=self.output_dir, record_raw=self.record_raw,
                camera_snapshot=self.camera_snapshot,
            )
            try:
                episode.run(poll_operator=self.poll_operator)
            finally:
                self.episodes.append(episode.result)
            self.finish_episode()
            result = episode.result
            raw = episode.recorder.result() if episode.recorder is not None else result.raw
            level, text = format_recording_result(raw)
            status = "结束" if self.episode is None else "控制结束，资源收尾未完成"
            logger.info("第 %03d 段%s\nDuration: %.1f s\nEnd reason: %s\nDetails: %s\nSkipped slots: %d",
                        result.index, status, result.duration_s, result.termination_reason,
                        result.termination_details, result.skipped_slots)
            logger.log(level, "%s", text)
            logger.info("Task outcome: unknown")
            if result.started_utc is not None:
                completed = sum(item.started_utc is not None for item in self.episodes)
                if self.num_episodes is not None and completed >= self.num_episodes:
                    self.runtime_state.request_quit()
            if self.runtime_state.quit_requested or self.runtime_state.estop_request:
                break
            if self.episode is None:
                print("停止流程已完成；本轮观测历史与动作计划已清空。检查上方结束原因及录制结果后按 B 开始新轮，按需 H 回初始位；Q 退出。", flush=True)

    def finish_episode(self):
        """停止后回收本段资源；退出意图交由最终关闭先释放设备。"""
        episode = cast(Rollout, self.episode)
        model = cast(PolicyWorker, self.model)
        robot = cast(Robot, self.robot)
        self.poll_operator()
        if self.runtime_state.quit_requested or self.runtime_state.estop_request:
            return
        try:
            while True:
                self.poll_operator()
                robot.service_feedback()
                if self.runtime_state.quit_requested or self.runtime_state.estop_request:
                    return
                if episode.finish_recording(wait=False):
                    break
                time.sleep(0.005)
            deadline = time.monotonic() + MODEL_SHUTDOWN_TIMEOUT_S
            while model.future is not None:
                self.poll_operator()
                robot.service_feedback()
                if self.runtime_state.quit_requested or self.runtime_state.estop_request:
                    return
                future = model.poll()
                if future is not None:
                    future.result()
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("episode model task remains pending")
                time.sleep(0.005)
        except BaseException as exc:
            episode.fail("episode_finalize", exc)
            raise
        self.poll_operator()
        if self.runtime_state.quit_requested or self.runtime_state.estop_request:
            return
        self.episode = None
        self.save_results()
