"""装配遥操作资源并统一启停。"""

import logging
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from dexmani_real.config.calibration_data import VR_TRANSFORM_PATH
from dexmani_real.config.teleop import DEFAULT_TASK_NAME
from dexmani_real.control.actions import ActionMapper
from dexmani_real.data.raw import validate_task_label
from dexmani_real.ipc.sensors import (
    SensorChannels,
    read_vr_frame,
    snapshot_camera_parameters,
)
from dexmani_real.robot.commands import OperationCancelled, RobotError
from dexmani_real.robot.system import Robot
from dexmani_real.runtime.keyboard import Keyboard
from dexmani_real.runtime.lifecycle import (
    CAMERA_READY_TIMEOUT_S,
    VR_READY_TIMEOUT_S,
    SensorWorkers,
    close_runtime,
)
from dexmani_real.runtime.preflight import validate_control_rate, validate_robot_config
from dexmani_real.runtime.state import SessionState
from dexmani_real.sensor.realsense import run_camera
from dexmani_real.sensor.vr import run_vr
from dexmani_real.teleop.audio import AudioFeedback
from dexmani_real.teleop.loop import TeleopLoop
from dexmani_real.teleop.vr import VRController
from dexmani_real.utils.log import configure_logging
from dexmani_real.utils.timing import sample_is_fresh
from dexmani_real.utils.vr_heading import load_vr_transform

logger = logging.getLogger(__name__)


def run_teleop(
    runtime,
    *,
    record: bool,
    episodes_dir: str,
    max_duration_s: float,
    task_name=DEFAULT_TASK_NAME,
    vr_transform_path=VR_TRANSFORM_PATH,
    camera_calibration_path=None,
) -> int:
    """装配遥操作资源并收尾。"""
    configure_logging()
    validate_robot_config(runtime)
    runtime.environment.validate()
    validate_control_rate(runtime, runtime.teleop.control_hz)
    runtime.teleop.validate()
    runtime.vr.validate()
    if record:
        validate_task_label(task_name)
        if task_name in {".", ".."} or "/" in task_name or "\\" in task_name:
            raise ValueError("task_name must be one safe directory component")
        if not isinstance(episodes_dir, str) or not episodes_dir.strip():
            raise ValueError("episodes_dir must be non-empty")
        if not np.isfinite(max_duration_s) or max_duration_s <= 0:
            raise ValueError("max_duration_s must be finite and positive")
        runtime.camera.validate()
    recording_dir = (
        Path(__file__).resolve().parents[2] / episodes_dir / task_name if record else None
    )
    if not runtime.hand.enabled and record:
        raise ValueError("hand-disabled operation cannot record episodes")
    vr_to_robot_rotation = load_vr_transform(vr_transform_path)
    controller = VRController(
        ActionMapper.for_mode(
            "eef",
            arm=runtime.arm,
            hand=runtime.hand,
            motion=runtime.motion,
            environment=runtime.environment,
        ),
        build_hand_retargeter(runtime),
        runtime.teleop,
        vr_to_robot_rotation,
    )
    ctx = mp.get_context("spawn")
    runtime_state = SessionState()
    startup_cancel_epoch = runtime_state.cancel_epoch
    channels = SensorChannels(
        ctx,
        camera=runtime.camera if record else None,
        vr=True,
    )
    sensors = robot = None
    loop = None
    keyboard = audio = None
    primary_failure: BaseException | None = None
    try:
        try:
            sensors = SensorWorkers(channels)
            robot = Robot(
                runtime.arm,
                runtime.hand,
                runtime_state=runtime_state,
            )
            keyboard = Keyboard(
                estop_callback=runtime_state.request_emergency_stop,
                bindings={
                    "s": runtime_state.request_stop,
                    "c": runtime_state.request_stop,
                    "q": runtime_state.request_quit,
                },
            )
            audio = AudioFeedback()
            loop = TeleopLoop(
                runtime_state,
                runtime,
                robot,
                task_label=task_name,
                sensors=sensors,
                controller=controller,
                keyboard=keyboard,
                audio=audio,
                recording_dir=recording_dir,
                max_duration_s=max_duration_s,
            )
            keyboard.bindings["d"] = runtime_state.request_stop
            keyboard.start()

            def check_services():
                if not keyboard.check_health() or not sensors.check():
                    raise RobotError("teleop control service unavailable")
                return True

            robot.check_services = check_services
            robot.connect()
            if record:
                sensors.start(
                    ctx.Process(
                        name="camera",
                        target=run_camera,
                        args=(sensors.channels, runtime.camera),
                    ),
                    ring=sensors.channels.camera_ring,
                    timeout_s=CAMERA_READY_TIMEOUT_S,
                    may_start=lambda: keyboard.check_health() and runtime_state.requests_allow_motion()
                        and runtime_state.cancel_epoch == startup_cancel_epoch,
                )
                loop.camera_snapshot = snapshot_camera_parameters(
                    channels,
                    camera_calibration_path,
                )
            robot.check()
            if (runtime_state.cancel_epoch != startup_cancel_epoch
                    or not runtime_state.requests_allow_motion()):
                raise OperationCancelled("startup cancelled")
            initialize_teleop(
                loop, ctx.Process(name="vr", target=run_vr, args=(channels, runtime.vr)),
                cancel_epoch=startup_cancel_epoch,
            )
        except OperationCancelled:
            if robot is not None:
                robot.check()
            if (runtime_state.cancel_epoch == startup_cancel_epoch
                    and runtime_state.requests_allow_motion()):
                raise
            runtime_state.request_quit()
            logger.info("Teleop startup cancelled")
        else:
            loop.run()
            robot.check()
    except (Exception, KeyboardInterrupt) as exc:
        primary_failure = exc
        if loop is not None:
            loop.record_error("session", exc)
        elif isinstance(exc, KeyboardInterrupt):
            runtime_state.request_emergency_stop()
        else:
            runtime_state.stop_signal.set()
        logger.exception("teleop session failed")
    finally:
        if audio is not None and runtime_state.estop_request:
            audio.play("emergency")
        try:
            close_runtime(runtime_state, robot, sensors=sensors, keyboard=keyboard)
        except BaseException as exc:
            primary_failure = primary_failure or exc
            if loop is not None:
                loop.record_error("runtime_cleanup", exc)
            elif isinstance(exc, KeyboardInterrupt):
                runtime_state.request_emergency_stop()
            logger.exception("teleop runtime cleanup failed")
        try:
            if loop is not None:
                loop.finish_recording()
        except BaseException as exc:
            primary_failure = primary_failure or exc
            if loop is not None:
                loop.record_error("recording_cleanup", exc)
            logger.exception("teleop capture cleanup failed")
        if audio is not None:
            try:
                audio.close()
            except Exception:
                logger.warning("teleop audio cleanup failed", exc_info=True)
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    runtime_state.request_emergency_stop()
                primary_failure = primary_failure or exc
                logger.exception("teleop audio cleanup failed")
    return int(
        not runtime_state.quit_requested or primary_failure is not None or runtime_state.estop_request
    )


def initialize_teleop(loop, vr_process, *, cancel_epoch) -> None:
    loop.robot.check()
    if loop.runtime_state.cancel_epoch != cancel_epoch:
        raise OperationCancelled("teleop startup cancelled")
    if loop.poll_blocking_commands():
        raise OperationCancelled("teleop startup cancelled")
    loop.sensors.start(
        vr_process,
        ring=loop.sensors.channels.vr_ring,
        may_start=lambda: loop.keyboard.check_health() and loop.runtime_state.requests_allow_motion()
            and loop.runtime_state.cancel_epoch == cancel_epoch,
    )
    deadline = time.monotonic() + VR_READY_TIMEOUT_S
    while True:
        if loop.poll_blocking_commands() or loop.runtime_state.cancel_epoch != cancel_epoch:
            raise OperationCancelled("teleop startup cancelled")
        loop.robot.service_idle()
        if loop.sensors.channels.vr_ring.latest_sequence > 0:
            if loop.poll_blocking_commands():
                raise OperationCancelled("teleop startup cancelled")
            if not loop.sensors.check():
                raise RuntimeError("VR worker exited during startup")
            break
        time.sleep(0.05)
        if time.monotonic() >= deadline:
            raise RuntimeError("VR failed to become ready before the startup timeout")
    calibrate_hand(loop, cancel_epoch)
    if loop.poll_blocking_commands() or loop.runtime_state.cancel_epoch != cancel_epoch:
        raise OperationCancelled("teleop startup cancelled")
    loop.runtime_state.stop_signal.set()
    print(
        "就绪：B 空闲时开始新段 | S/C 保存结束本段 | D 丢弃本段 | "
        "Q 保存并退出 | Esc 急停 | H 空闲时先手后臂回位 | "
        "T 空闲时复位触觉。",
        flush=True,
    )


def calibrate_hand(loop, cancel_epoch) -> None:
    """只消费新鲜、来源时间递增的 VR 帧；重复读取缓存不计作标定样本。"""
    retargeter = loop.controller.hand_retargeter
    if retargeter is None:
        return
    from dexmani_real.teleop.retargeting.landmarks import (
        hand_lengths,
        prepare_landmarks,
    )

    config = loop.runtime.hand_retargeting
    print("正在标定手部骨长：请保持右手清晰可见；无需保持伸直。S/Q/Esc 可取消。", flush=True)
    deadline = time.monotonic() + config.calibration_timeout_s
    measurements: list[np.ndarray] = []
    last_source_ns = 0
    last_error = "no fresh hand frames"
    while len(measurements) < config.calibration_samples:
        if loop.poll_blocking_commands() or loop.runtime_state.cancel_epoch != cancel_epoch:
            raise OperationCancelled("hand calibration cancelled")
        loop.robot.service_idle()
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"hand calibration timed out ({len(measurements)}/{config.calibration_samples}): {last_error}"
            )
        frame = read_vr_frame(loop.sensors.channels)
        if frame is not None:
            stamp = int(frame["recv_ts_ns"])
            now_ns = time.monotonic_ns()
            if (
                stamp > last_source_ns
                and sample_is_fresh(stamp, loop.runtime.teleop.vr_stale_threshold_s, now_ns)
            ):
                last_source_ns = stamp
                try:
                    measurement = hand_lengths(prepare_landmarks(frame["landmarks"]))
                except ValueError as exc:
                    last_error = str(exc)
                else:
                    measurements.append(measurement)
        time.sleep(0.005)
    if loop.poll_blocking_commands() or loop.runtime_state.cancel_epoch != cancel_epoch:
        raise OperationCancelled("hand calibration cancelled")
    retargeter.calibrate(measurements)
    logger.info("Hand morphology calibrated from %s distinct frames", len(measurements))


def build_hand_retargeter(runtime):
    if not runtime.hand.enabled:
        return None
    from dexmani_real.teleop.retargeting.adaptive import AdaptiveHandRetargeter

    return AdaptiveHandRetargeter(
        runtime.hand_retargeting, runtime.hand,
    )
