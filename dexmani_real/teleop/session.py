"""Own the local teleop/robot and sensor startup; all hardware work is explicit."""

import logging
import multiprocessing as mp
from pathlib import Path

from dexmani_real.data.recorder import RecordingError
from dexmani_real.ipc.channels import SensorChannels
from dexmani_real.planning.action import ActionMapper
from dexmani_real.planning.home_planner import HomePlanner
from dexmani_real.planning.table_plane import resolve_table_plane
from dexmani_real.robot.commands import RobotError
from dexmani_real.robot.model import XHAND_RIGHT_URDF_PATH
from dexmani_real.robot.robot import DexManiRobot
from dexmani_real.runtime.observation import (
    ObservationNeeds,
    snapshot_camera_parameters,
)
from dexmani_real.runtime.operator_input import KeyboardInput
from dexmani_real.runtime.preflight import validate_control_rate, validate_robot_config
from dexmani_real.runtime.processes import SensorProcesses, shutdown_local_runtime
from dexmani_real.runtime.state import RunEndReason, RuntimeState
from dexmani_real.sensor.camera.extrinsics import load_optional_camera_extrinsics
from dexmani_real.sensor.camera.worker import run_camera_worker
from dexmani_real.sensor.vr.worker import run_vr_worker
from dexmani_real.state import VR_TRANSFORM_PATH
from dexmani_real.teleop.audio_feedback import AudioFeedback
from dexmani_real.teleop.config import DEFAULT_TASK_NAME, validate_task_dir_name
from dexmani_real.teleop.controller import TeleopController
from dexmani_real.teleop.runner import TeleopRunner
from dexmani_real.teleop.vr_alignment import load_vr_transform
from dexmani_real.teleop.vr_mapping import VRWristMapper
from dexmani_real.utils.log import configure_logging

logger = logging.getLogger(__name__)
END_AUDIO_GRACE_S = 3.0


class TeleopSession:
    """Own teleoperation resources, operator input, startup and shutdown."""

    def __init__(
        self,
        runtime,
        *,
        task_name=DEFAULT_TASK_NAME,
        allow_no_hand=False,
        vr_transform_path=VR_TRANSFORM_PATH,
        camera_calibration_path=None,
    ):
        self.runtime = runtime
        self.task_name = task_name
        self.allow_no_hand = allow_no_hand
        self.vr_transform_path = vr_transform_path
        self.camera_calibration_path = camera_calibration_path

    def run(self):
        configure_logging()
        self.task_name = validate_task_dir_name(self.task_name)
        validate_robot_config(self.runtime)
        validate_control_rate(self.runtime, self.runtime.teleop.control_hz)
        self.runtime.teleop.validate()
        self.runtime.collection.validate()
        self.runtime.vr.validate()
        if self.runtime.collection.enabled:
            self.runtime.camera.validate()
        self.table_plane_abcd = resolve_table_plane(
            self.runtime.environment.table, required=self.runtime.environment.table.enabled,
        )
        if not self.runtime.hand.enabled and (
            self.runtime.collection.enabled or not self.allow_no_hand
        ):
            raise ValueError("hand-disabled operation requires explicit unrecorded debug mode")
        calibration = load_vr_transform(self.vr_transform_path)
        mapping = self.runtime.teleop.vr_mapping
        controller = TeleopController(
            ActionMapper.for_mode(
                "eef", arm=self.runtime.arm, hand=self.runtime.hand,
                motion=self.runtime.motion, environment=self.runtime.environment,
            ),
            VRWristMapper(
                pos_scale=mapping.pos_scale,
                vr_to_robot_rot=calibration.transform,
            ),
            self.runtime.teleop.ema,
            self.build_hand_retargeter(),
        )
        home_planner = HomePlanner(
            self.runtime.environment, table_plane_abcd=self.table_plane_abcd,
        )
        camera_calibration = (
            load_optional_camera_extrinsics(self.camera_calibration_path)
            if self.runtime.collection.enabled
            else None
        )
        control_needs = ObservationNeeds(vr=True)
        record_needs = (
            ObservationNeeds(camera="rgbd", required_tactile=("aggregate", "dense"))
            if self.runtime.collection.enabled else ObservationNeeds()
        )
        needs = control_needs.union(record_needs)
        ctx = mp.get_context("spawn")
        self.runtime_state = RuntimeState()
        channels = SensorChannels(
            ctx,
            camera=self.runtime.camera if needs.camera_worker else None,
            vr=needs.vr,
        )
        self.sensors = self.robot = None
        clean = False
        self.runner = None
        keyboard = audio = None
        try:
            self.sensors = SensorProcesses(channels)
            self.robot = DexManiRobot(
                self.runtime.arm, self.runtime.hand, run_signal=self.runtime_state.run_signal,
                stop_signal=self.runtime_state.stop_signal,
                emergency_signal=self.runtime_state.emergency_signal,
                signal_lock=self.runtime_state.motion_lock, check_services=self.sensors.check,
            )
            keyboard = KeyboardInput(
                estop_callback=self.runtime_state.request_emergency_stop,
                stop_callback=self.runtime_state.request_stop,
                quit_callback=lambda: self.runtime_state.request_stop(reason=RunEndReason.QUIT),
            )
            audio = AudioFeedback()
            self.runner = TeleopRunner(
                self.runtime_state,
                self.runtime,
                self.robot,
                task_label=self.task_name,
                control_needs=control_needs, record_needs=record_needs,
                sensors=self.sensors,
                controller=controller,
                home_planner=home_planner,
                keyboard=keyboard,
                audio=audio,
                recording_dir=(
                    Path(__file__).resolve().parents[2]
                    / self.runtime.collection.episodes_dir / self.task_name
                    if self.runtime.collection.enabled else None
                ),
                vr_process=ctx.Process(
                    name="vr",
                    target=run_vr_worker,
                    args=(channels, self.runtime.vr),
                ),
            )
            keyboard.start()
            def check_services():
                if not keyboard.healthy or not self.sensors.check(names=("vr",)):
                    raise RobotError("teleop control service unavailable")
                if self.runtime.collection.enabled and not self.sensors.check(names=("camera",)):
                    raise RecordingError("recording camera process exited")
                return True

            self.robot.check_services = check_services
            self.robot.connect()
            if needs.camera_worker:
                self.sensors.start(
                    ctx.Process(
                        name="camera",
                        target=run_camera_worker,
                        args=(self.sensors.channels, self.runtime.camera),
                    ),
                    ready=self.sensors.channels.camera_ready,
                    timeout_s=self.runtime.safety.camera_ready_timeout_s,
                    may_start=lambda: keyboard.healthy and self.runtime_state.requests_allow_motion(),
                )
                self.runner.camera_snapshot = snapshot_camera_parameters(
                    channels, self.runtime.camera, camera_calibration,
                )
            self.robot.check()
            self.runtime_state.arm()
            self.runner.run()
            clean = bool(self.runtime_state.quit_requested)
        except KeyboardInterrupt:
            self.runtime_state.request_emergency_stop()
        except Exception:
            self.runtime_state.mark_error()
            logger.exception("teleop session failed")
        finally:
            try:
                try:
                    if self.runner is not None:
                        self.runner.close()
                except Exception:
                    self.runtime_state.mark_error()
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
                    self.runtime_state.mark_error()
                    logger.exception("teleop audio cleanup failed")
            finally:
                shutdown_clean = shutdown_local_runtime(
                    self.runtime_state, self.robot, sensors=self.sensors, keyboard=keyboard,
                    timeout_s=self.runtime.safety.shutdown_timeout_s,
                )
        return int(not clean or not shutdown_clean)

    def build_hand_retargeter(self):
        if not self.runtime.hand.enabled:
            return None
        if self.runtime.teleop.hand_retargeting_type == "tag":
            from dexmani_real.teleop.retargeting.tag import TAGHandRetargeter

            return TAGHandRetargeter(
                fingertip_link_names=self.runtime.hand.fingertip_link_names,
                tag_config=self.runtime.tag_retargeting,
                urdf_path=str(XHAND_RIGHT_URDF_PATH),
            )
        if self.runtime.teleop.hand_retargeting_type == "dexpilot":
            from dexmani_real.teleop.retargeting.dexpilot import DexPilotHandRetargeter

            return DexPilotHandRetargeter(
                dexpilot_config=self.runtime.dexpilot_retargeting,
            )
        raise ValueError("hand_retargeting_type must be 'tag' or 'dexpilot'")
