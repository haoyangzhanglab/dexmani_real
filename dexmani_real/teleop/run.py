"""装配遥操作资源并统一启停。"""

import logging
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from dexmani_real.config.calibration_data import VR_TRANSFORM_PATH
from dexmani_real.config.teleop import DEFAULT_TASK_NAME
from dexmani_real.control.actions import ActionMapper
from dexmani_real.data.recorder import RecordingError
from dexmani_real.ipc.sensors import (
    SensorChannels,
    read_vr_frame,
    snapshot_camera_parameters,
)
from dexmani_real.robot.commands import OperationCancelled, RobotError
from dexmani_real.robot.system import Robot
from dexmani_real.runtime.home import initialize_hand
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
    runtime_config_json: str | None,
    task_name=DEFAULT_TASK_NAME,
    vr_transform_path=VR_TRANSFORM_PATH,
    camera_calibration_path=None,
) -> int:
    """装配遥操作资源，运行并按原保存意图完成收尾。"""
    configure_logging()
    name = task_name
    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError("task_name must be a non-empty string without surrounding whitespace")
    if name.startswith("."):
        raise ValueError("task_name must not be a hidden or relative directory name")
    if "/" in name or "\\" in name or any(ord(char) < 32 for char in name):
        raise ValueError("task_name must be one safe directory component")
    validate_robot_config(runtime)
    validate_control_rate(runtime, runtime.teleop.control_hz)
    runtime.teleop.validate()
    runtime.vr.validate()
    if record:
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
    channels = SensorChannels(
        ctx,
        camera=runtime.camera if record else None,
        vr=True,
    )
    sensors = robot = None
    clean = False
    shutdown_clean = False
    loop = None
    keyboard = audio = None
    primary_failure = None
    try:
        sensors = SensorWorkers(channels)
        robot = Robot(
            runtime.arm,
            runtime.hand,
            run_signal=runtime_state.run_signal,
            stop_signal=runtime_state.stop_signal,
            emergency_signal=runtime_state.emergency_signal,
            check_services=sensors.check,
        )
        keyboard = Keyboard(
            get_run_id=lambda: runtime_state.run_id,
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
            runtime_config_json=runtime_config_json,
        )
        keyboard.bindings["d"] = loop.request_discard
        keyboard.start()

        def check_services():
            if not keyboard.check_health() or not sensors.check(names=("vr",)):
                raise RobotError("teleop control service unavailable")
            if record and not sensors.check(names=("camera",)):
                raise RecordingError("recording camera process exited")
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
                may_start=lambda: keyboard.check_health() and runtime_state.requests_allow_motion(),
            )
            loop.camera_snapshot = snapshot_camera_parameters(
                channels,
                runtime.camera,
                camera_calibration_path,
            )
        robot.check()
        startup_run_id = runtime_state.arm()
        if initialize_teleop(
            loop, ctx.Process(name="vr", target=run_vr, args=(channels, runtime.vr)),
            run_id=startup_run_id,
        ):
            loop.run()
        robot.check()
        clean = bool(runtime_state.quit_requested)
    except KeyboardInterrupt as exc:
        primary_failure = exc
        runtime_state.request_emergency_stop()
    except OperationCancelled:
        try:
            if robot is not None:
                robot.check()
            if not (
                runtime_state.stop_signal.is_set() or runtime_state.quit_requested
                or runtime_state.estop_request
            ):
                raise
        except Exception as failure:
            primary_failure = failure
            runtime_state.mark_error()
            if loop is not None:
                loop.record_error("cancellation", failure)
            logger.exception("teleop cancellation also failed")
        else:
            runtime_state.request_quit()
            clean = True
            logger.info("Teleop startup cancelled")
    except Exception as exc:
        primary_failure = exc
        runtime_state.mark_error()
        logger.exception("teleop session failed")
    finally:
        try:
            if loop is not None:
                loop.close()
        except Exception as exc:
            runtime_state.mark_error()
            if exc is not primary_failure:
                logger.exception("teleop capture cleanup failed")
        finally:
            try:
                close_runtime(runtime_state, robot, sensors=sensors, keyboard=keyboard)
            except BaseException as exc:
                if exc is not primary_failure:
                    logger.exception("teleop runtime cleanup failed")
            else:
                shutdown_clean = True
            finally:
                if audio is not None:
                    try:
                        success = (
                            clean and shutdown_clean and not runtime_state.error_state
                            and not runtime_state.estop_request
                        )
                        try:
                            audio.play(
                                "end" if success else "emergency", interrupt=not success,
                            )
                            if not audio.wait_until_idle(timeout_s=5.0):
                                logger.warning("退出提示音未在 5 秒内播放完毕。")
                        finally:
                            audio.close()
                    except Exception:
                        shutdown_clean = False
                        logger.exception("teleop audio cleanup failed")
    return int(
        not clean or not shutdown_clean or runtime_state.error_state or runtime_state.estop_request
    )


def initialize_teleop(loop, vr_process, *, run_id) -> bool:
    # 自动复位前，键盘停止回调已经生效。
    initialize_hand(
        loop.runtime_state,
        loop.robot,
        run_id=run_id,
        required_tactile=loop.needs.required_tactile,
        cancel_requested=loop.poll_blocking_commands,
    )
    if loop.poll_blocking_commands():
        return False
    loop.sensors.start(
        vr_process,
        ring=loop.sensors.channels.vr_ring,
        may_start=lambda: loop.keyboard.check_health() and loop.runtime_state.requests_allow_motion(),
    )
    deadline = time.monotonic() + VR_READY_TIMEOUT_S
    while True:
        if loop.poll_blocking_commands():
            return False
        loop.robot.service_idle()
        if loop.sensors.channels.vr_ring.latest_sequence > 0:
            if loop.poll_blocking_commands():
                return False
            if not loop.sensors.check():
                raise RuntimeError("VR worker exited during startup")
            break
        time.sleep(0.05)
        if time.monotonic() >= deadline:
            raise RuntimeError("VR failed to become ready before the startup timeout")
    if not calibrate_hand(loop):
        return False
    loop.runtime_state.revoke_motion()
    loop.keyboard.discard_events("b", "h", "t")
    print(
        "就绪：B 空闲时开始新段 | S/C 保存结束本段 | D 丢弃本段 | "
        "Q 保存并退出 | Esc 急停 | H 空闲时回初始位 | "
        "T 空闲时复位触觉。",
        flush=True,
    )
    return True


def calibrate_hand(loop):
    """只消费新鲜、来源时间递增的 VR 帧；重复读取缓存不计作标定样本。"""
    retargeter = loop.controller.hand_retargeter
    if retargeter is None:
        return True
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
        if loop.poll_blocking_commands():
            loop.runtime_state.request_quit()
            return False
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
    retargeter.calibrate(measurements)
    logger.info("Hand morphology calibrated from %s distinct frames", len(measurements))
    return True


def build_hand_retargeter(runtime):
    if not runtime.hand.enabled:
        return None
    from dexmani_real.teleop.retargeting.adaptive import AdaptiveHandRetargeter

    return AdaptiveHandRetargeter(
        runtime.hand_retargeting, runtime.hand,
    )
