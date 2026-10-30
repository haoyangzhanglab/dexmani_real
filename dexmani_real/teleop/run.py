"""装配遥操作资源并统一启停。"""

import logging
import multiprocessing as mp
import time
from contextlib import ExitStack
from pathlib import Path

from dexmani_real.config.calibration_data import VR_TRANSFORM_PATH
from dexmani_real.config.teleop import DEFAULT_TASK_NAME
from dexmani_real.control.actions import ActionMapper
from dexmani_real.control.homing import HomePlanner
from dexmani_real.data.recorder import RecordingError
from dexmani_real.ipc.sensors import (
    SensorChannels,
    read_vr_frame,
    snapshot_camera_parameters,
)
from dexmani_real.robot.commands import RobotError
from dexmani_real.robot.system import Robot
from dexmani_real.runtime.home import initialize_hand
from dexmani_real.runtime.keyboard import Keyboard
from dexmani_real.runtime.lifecycle import (
    CAMERA_READY_TIMEOUT_S,
    SHUTDOWN_TIMEOUT_S,
    VR_READY_TIMEOUT_S,
    SensorWorkers,
    close_runtime,
)
from dexmani_real.runtime.observation import ObservationNeeds
from dexmani_real.runtime.preflight import validate_control_rate, validate_robot_config
from dexmani_real.runtime.state import RunEndReason, SessionState
from dexmani_real.sensor.realsense import run_camera
from dexmani_real.sensor.vr import run_vr
from dexmani_real.teleop.audio import AudioFeedback
from dexmani_real.teleop.loop import TeleopLoop, failure_reason
from dexmani_real.teleop.vr import VRController
from dexmani_real.utils.log import configure_logging
from dexmani_real.utils.table_plane import load_table_plane
from dexmani_real.utils.timing import sample_is_fresh
from dexmani_real.utils.vr_heading import load_vr_transform

logger = logging.getLogger(__name__)
END_AUDIO_GRACE_S = 3.0


def run_teleop(
    runtime,
    *,
    task_name=DEFAULT_TASK_NAME,
    allow_no_hand=False,
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
    runtime.collection.validate()
    runtime.vr.validate()
    if runtime.collection.enabled:
        runtime.camera.validate()
    table_plane_abcd = (
        load_table_plane(runtime.environment.table)
        if runtime.environment.table.enabled else None
    )
    if not runtime.hand.enabled and (runtime.collection.enabled or not allow_no_hand):
        raise ValueError("hand-disabled operation requires explicit unrecorded debug mode")
    calibration = load_vr_transform(vr_transform_path)
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
        calibration.transform,
    )
    home_planner = HomePlanner(
        runtime.environment,
        runtime.arm,
        table_plane_abcd=table_plane_abcd,
    )
    control_needs = ObservationNeeds(vr=True)
    record_needs = (
        ObservationNeeds(camera_fields=("rgb", "depth"), required_tactile=("aggregate", "dense"))
        if runtime.collection.enabled
        else ObservationNeeds()
    )
    needs = control_needs.union(record_needs)
    ctx = mp.get_context("spawn")
    runtime_state = SessionState()
    channels = SensorChannels(
        ctx,
        camera=runtime.camera if needs.camera_worker else None,
        vr=needs.vr,
    )
    sensors = robot = None
    clean = False
    loop = None
    keyboard = audio = None
    try:
        sensors = SensorWorkers(channels)
        robot = Robot(
            runtime.arm,
            runtime.hand,
            run_signal=runtime_state.run_signal,
            stop_signal=runtime_state.stop_signal,
            emergency_signal=runtime_state.emergency_signal,
            signal_lock=runtime_state.motion_lock,
            check_services=sensors.check,
        )
        keyboard = Keyboard(
            estop_callback=runtime_state.request_emergency_stop,
            bindings={
                "s": runtime_state.request_stop,
                "q": lambda: runtime_state.request_stop(reason=RunEndReason.QUIT),
            },
        )
        audio = AudioFeedback()
        loop = TeleopLoop(
            runtime_state,
            runtime,
            robot,
            task_label=task_name,
            control_needs=control_needs,
            record_needs=record_needs,
            sensors=sensors,
            controller=controller,
            home_planner=home_planner,
            keyboard=keyboard,
            audio=audio,
            recording_dir=(
                Path(__file__).resolve().parents[2] / runtime.collection.episodes_dir / task_name
                if runtime.collection.enabled
                else None
            ),
        )
        keyboard.bindings["d"] = loop.request_discard
        keyboard.start()

        def check_services():
            if not keyboard.check_health() or not sensors.check(names=("vr",)):
                raise RobotError("teleop control service unavailable")
            if runtime.collection.enabled and not sensors.check(names=("camera",)):
                raise RecordingError("recording camera process exited")
            return True

        robot.check_services = check_services
        robot.connect()
        if needs.camera_worker:
            sensors.start(
                ctx.Process(
                    name="camera",
                    target=run_camera,
                    args=(sensors.channels, runtime.camera),
                ),
                ready=sensors.channels.camera_ready,
                timeout_s=CAMERA_READY_TIMEOUT_S,
                may_start=lambda: keyboard.check_health() and runtime_state.requests_allow_motion(),
            )
            loop.camera_snapshot = snapshot_camera_parameters(
                channels,
                runtime.camera,
                camera_calibration_path,
            )
        robot.check()
        runtime_state.arm()
        try:
            if initialize_teleop(
                loop, ctx.Process(name="vr", target=run_vr, args=(channels, runtime.vr))
            ):
                loop.run()
        except BaseException as exc:
            loop.record_error("runtime", exc)
            runtime_state.revoke_motion(reason=failure_reason(exc))
            sensors.request_shutdown()
            audio.play("emergency")
            raise
        clean = bool(runtime_state.quit_requested)
    except KeyboardInterrupt:
        runtime_state.request_emergency_stop()
    except Exception:
        runtime_state.mark_error()
        logger.exception("teleop session failed")
    finally:

        def close_component(stage, close):
            try:
                close()
            except Exception:
                runtime_state.mark_error()
                logger.exception("teleop %s cleanup failed", stage)

        try:
            with ExitStack() as cleanup:
                if audio is not None:
                    cleanup.callback(close_component, "audio", lambda: finish_audio(audio))
                if loop is not None:
                    cleanup.callback(close_component, "capture", loop.close)
        finally:
            shutdown_clean = close_runtime(
                runtime_state,
                robot,
                sensors=sensors,
                keyboard=keyboard,
                deadline=time.monotonic() + SHUTDOWN_TIMEOUT_S,
            )
    return int(not clean or not shutdown_clean)


def initialize_teleop(loop, vr_process) -> bool:
    # 自动复位前，键盘停止回调已经生效。
    home_result = initialize_hand(
        loop.runtime_state,
        loop.robot,
        cancel_requested=loop.poll_blocking_commands,
    )
    if not home_result.ok:
        if home_result.interrupted:
            loop.runtime_state.request_quit()
            return False
        raise RuntimeError(f"startup hand initialization failed: {home_result.reason}")
    if loop.poll_blocking_commands():
        return False
    loop.sensors.start(
        vr_process,
        ready=loop.sensors.channels.vr_ready,
        may_start=lambda: loop.keyboard.check_health() and loop.runtime_state.requests_allow_motion(),
    )
    deadline = time.monotonic() + VR_READY_TIMEOUT_S
    while True:
        if not loop.keyboard.check_health():
            loop.runtime_state.request_emergency_stop()
        if loop.poll_blocking_commands():
            return False
        loop.robot.service_idle()
        if loop.sensors.channels.vr_ready.wait(timeout=0.05):
            if loop.poll_blocking_commands():
                return False
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("VR failed to become ready before the startup timeout")
    if not calibrate_hand(loop):
        return False
    loop.keyboard.discard_events("b", "c", "h", "t")
    print(
        "就绪：B 开始 | C 暂停/继续（继续时新建录制段） | H 保存并回初始位 | "
        "S 保存 | D 丢弃 | Q 暂停待退出，再按 Q 保存退出 | Esc 急停 | T 空闲时复位触觉。",
        flush=True,
    )
    return True


def finish_audio(audio):
    try:
        if not audio.wait_until_idle(timeout_s=END_AUDIO_GRACE_S):
            logger.warning("Final audio did not finish within %.1fs", END_AUDIO_GRACE_S)
    finally:
        audio.close()


def calibrate_hand(loop):
    """只消费新鲜、来源时间递增的 VR 帧；重复读取缓存不计作标定样本。"""
    retargeter = loop.controller.hand_retargeter
    if retargeter is None:
        return True
    from dexmani_real.teleop.retargeting.landmarks import prepare_landmarks

    config = loop.runtime.hand_retargeting
    print("正在标定手部骨长：请保持右手清晰可见；无需保持伸直。S/Q/Esc 可取消。", flush=True)
    deadline = time.monotonic() + config.calibration_timeout_s
    frames, last_sequence, last_source_ns = [], None, 0
    last_error = "no fresh hand frames"
    while len(frames) < config.calibration_samples:
        if not loop.keyboard.check_health():
            loop.runtime_state.request_emergency_stop()
        if loop.poll_blocking_commands():
            loop.runtime_state.request_quit()
            return False
        loop.robot.service_idle()
        if not loop.sensors.check(names=("vr",)):
            raise RuntimeError("VR worker exited during hand calibration")
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"hand calibration timed out ({len(frames)}/{config.calibration_samples}): {last_error}"
            )
        frame = read_vr_frame(loop.sensors.channels)
        if frame is not None:
            stamp = int(frame["recv_ts_ns"])
            now_ns = time.monotonic_ns()
            if (
                frame["ring_sequence"] != last_sequence and stamp > last_source_ns
                and sample_is_fresh(stamp, loop.runtime.teleop.vr_stale_threshold_s, now_ns)
            ):
                last_sequence, last_source_ns = frame["ring_sequence"], stamp
                try:
                    prepare_landmarks(frame["landmarks"])
                except ValueError as exc:
                    last_error = str(exc)
                else:
                    frames.append(frame["landmarks"].copy())
        time.sleep(0.005)
    retargeter.calibrate(frames)
    logger.info("Hand morphology calibrated from %s distinct frames", len(frames))
    return True


def build_hand_retargeter(runtime):
    if not runtime.hand.enabled:
        return None
    from dexmani_real.teleop.retargeting.adaptive import AdaptiveHandRetargeter

    return AdaptiveHandRetargeter(
        runtime.hand_retargeting, runtime.hand, control_hz=runtime.teleop.control_hz,
    )
