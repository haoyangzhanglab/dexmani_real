"""Own the local teleop/robot and sensor startup; all hardware work is explicit."""

import logging
import multiprocessing as mp
from pathlib import Path

from dexmani_real.calibration import VR_TRANSFORM_PATH
from dexmani_real.calibration.camera.extrinsics import load_optional_camera_extrinsics
from dexmani_real.calibration.table_plane import resolve_table_plane
from dexmani_real.calibration.vr_alignment import load_vr_transform
from dexmani_real.control.action_mapping import ActionMapper
from dexmani_real.control.homing import HomePlanner
from dexmani_real.data.recorder import RecordingError
from dexmani_real.ipc.channels import SensorChannels
from dexmani_real.robot.commands import RobotError
from dexmani_real.robot.model import XHAND_RIGHT_URDF_PATH
from dexmani_real.robot.system import DexManiRobot
from dexmani_real.runtime.observation import (
    ObservationNeeds,
    snapshot_camera_parameters,
)
from dexmani_real.runtime.operator_input import KeyboardInput
from dexmani_real.runtime.preflight import validate_control_rate, validate_robot_config
from dexmani_real.runtime.sensor_processes import SensorProcesses
from dexmani_real.runtime.shutdown import shutdown_local_runtime
from dexmani_real.runtime.state import RunEndReason, RuntimeState
from dexmani_real.sensor.camera.worker import run_camera_worker
from dexmani_real.sensor.vr.worker import run_vr_worker
from dexmani_real.teleop.audio_feedback import AudioFeedback
from dexmani_real.teleop.config import DEFAULT_TASK_NAME
from dexmani_real.teleop.controller import TeleopController
from dexmani_real.teleop.runner import TeleopRunner
from dexmani_real.teleop.vr_mapping import VRWristMapper
from dexmani_real.utils.log import configure_logging

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
    table_plane_abcd = resolve_table_plane(
        runtime.environment.table, required=runtime.environment.table.enabled,
    )
    if not runtime.hand.enabled and (
        runtime.collection.enabled or not allow_no_hand
    ):
        raise ValueError("hand-disabled operation requires explicit unrecorded debug mode")
    calibration = load_vr_transform(vr_transform_path)
    controller = TeleopController(
        ActionMapper.for_mode(
            "eef", arm=runtime.arm, hand=runtime.hand,
            motion=runtime.motion, environment=runtime.environment,
        ),
        VRWristMapper(
            pos_scale=runtime.teleop.vr_pos_scale,
            vr_to_robot_rot=calibration.transform,
        ),
        build_hand_retargeter(runtime),
        ema_alpha_pos=runtime.teleop.ema_alpha_pos,
        ema_alpha_rot=runtime.teleop.ema_alpha_rot,
    )
    home_planner = HomePlanner(
        runtime.environment, table_plane_abcd=table_plane_abcd,
    )
    camera_calibration = (
        load_optional_camera_extrinsics(camera_calibration_path)
        if runtime.collection.enabled
        else None
    )
    control_needs = ObservationNeeds(vr=True)
    record_needs = (
        ObservationNeeds(camera="rgbd", required_tactile=("aggregate", "dense"))
        if runtime.collection.enabled else ObservationNeeds()
    )
    needs = control_needs.union(record_needs)
    ctx = mp.get_context("spawn")
    runtime_state = RuntimeState()
    channels = SensorChannels(
        ctx,
        camera=runtime.camera if needs.camera_worker else None,
        vr=needs.vr,
    )
    sensors = robot = None
    clean = False
    runner = None
    keyboard = audio = None
    try:
        sensors = SensorProcesses(channels)
        robot = DexManiRobot(
            runtime.arm, runtime.hand, run_signal=runtime_state.run_signal,
            stop_signal=runtime_state.stop_signal,
            emergency_signal=runtime_state.emergency_signal,
            signal_lock=runtime_state.motion_lock, check_services=sensors.check,
        )
        keyboard = KeyboardInput(
            estop_callback=runtime_state.request_emergency_stop,
            stop_callback=runtime_state.request_stop,
            quit_callback=lambda: runtime_state.request_stop(reason=RunEndReason.QUIT),
        )
        audio = AudioFeedback()
        runner = TeleopRunner(
            runtime_state,
            runtime,
            robot,
            task_label=task_name,
            control_needs=control_needs, record_needs=record_needs,
            sensors=sensors,
            controller=controller,
            home_planner=home_planner,
            keyboard=keyboard,
            audio=audio,
            recording_dir=(
                Path(__file__).resolve().parents[2]
                / runtime.collection.episodes_dir / task_name
                if runtime.collection.enabled else None
            ),
            vr_process=ctx.Process(
                name="vr",
                target=run_vr_worker,
                args=(channels, runtime.vr),
            ),
        )
        keyboard.start()
        def check_services():
            if not keyboard.healthy or not sensors.check(names=("vr",)):
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
                    target=run_camera_worker,
                    args=(sensors.channels, runtime.camera),
                ),
                ready=sensors.channels.camera_ready,
                timeout_s=runtime.safety.camera_ready_timeout_s,
                may_start=lambda: keyboard.healthy and runtime_state.requests_allow_motion(),
            )
            runner.camera_snapshot = snapshot_camera_parameters(
                channels, runtime.camera, camera_calibration,
            )
        robot.check()
        runtime_state.arm()
        runner.run()
        clean = bool(runtime_state.quit_requested)
    except KeyboardInterrupt:
        runtime_state.request_emergency_stop()
    except Exception:
        runtime_state.mark_error()
        logger.exception("teleop session failed")
    finally:
        try:
            try:
                if runner is not None:
                    runner.close()
            except Exception:
                runtime_state.mark_error()
                logger.exception("teleop capture cleanup failed")
            try:
                # Preserve the final cue after capture finalization and before device shutdown.
                if audio is not None and not audio.wait_until_idle(timeout_s=END_AUDIO_GRACE_S):
                    logger.warning("Final audio did not finish within %.1fs", END_AUDIO_GRACE_S)
            except Exception:
                logger.warning("Final audio unavailable", exc_info=True)
            try:
                if audio is not None:
                    audio.close()
            except Exception:
                runtime_state.mark_error()
                logger.exception("teleop audio cleanup failed")
        finally:
            shutdown_clean = shutdown_local_runtime(
                runtime_state, robot, sensors=sensors, keyboard=keyboard,
                timeout_s=runtime.safety.shutdown_timeout_s,
            )
    return int(not clean or not shutdown_clean)


def build_hand_retargeter(runtime):
    if not runtime.hand.enabled:
        return None
    if runtime.teleop.hand_retargeting_type == "tag":
        from dexmani_real.teleop.retargeting.tag import TAGHandRetargeter

        return TAGHandRetargeter(
            fingertip_link_names=runtime.hand.fingertip_link_names,
            tag_config=runtime.tag_retargeting,
            urdf_path=str(XHAND_RIGHT_URDF_PATH),
        )
    if runtime.teleop.hand_retargeting_type == "dexpilot":
        from dexmani_real.teleop.retargeting.dexpilot import DexPilotHandRetargeter

        return DexPilotHandRetargeter(
            dexpilot_config=runtime.dexpilot_retargeting,
        )
    raise ValueError("hand_retargeting_type must be 'tag' or 'dexpilot'")
