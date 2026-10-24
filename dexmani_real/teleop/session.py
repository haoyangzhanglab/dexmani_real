"""Own the local teleop/robot and sensor startup; all hardware work is explicit."""

import logging
import multiprocessing as mp
from pathlib import Path

import numpy as np

from dexmani_real.calibration import VR_TRANSFORM_PATH
from dexmani_real.calibration.camera.extrinsics import load_optional_camera_extrinsics
from dexmani_real.calibration.vr_heading import load_vr_transform
from dexmani_real.config.control import CollectionOptions
from dexmani_real.config.experiment import (
    DEFAULT_TASK_NAME,
    resolve_runtime_table,
    validate_control_rate,
    validate_robot_config,
    validate_task_dir_name,
)
from dexmani_real.data.recorder import AsyncEpisodeRecorder
from dexmani_real.ipc.channels import SensorChannels
from dexmani_real.planning.action import ActionMapper
from dexmani_real.planning.kinematics.ik import OnlineIKConfig
from dexmani_real.robot.home import build_home_planner
from dexmani_real.robot.model import XHAND_RIGHT_URDF_PATH
from dexmani_real.robot.robot import DexManiRobot
from dexmani_real.runtime.observation import ObservationNeeds
from dexmani_real.runtime.operator_input import KeyboardInput
from dexmani_real.runtime.processes import SensorProcesses, shutdown_local_runtime
from dexmani_real.runtime.state import RunEndReason, RuntimeState
from dexmani_real.sensor.camera.worker import run_camera_worker
from dexmani_real.sensor.vr.worker import run_vr_worker
from dexmani_real.teleop.audio_feedback import AudioFeedback
from dexmani_real.teleop.controller import TeleopController
from dexmani_real.teleop.runner import TeleopRunner
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
        collection=CollectionOptions(),
        task_name=DEFAULT_TASK_NAME,
        allow_no_hand=False,
        vr_transform_path=VR_TRANSFORM_PATH,
        camera_calibration_path=None,
    ):
        self.runtime = runtime
        self.collection = collection
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
        self.collection.validate()
        self.runtime.vr.validate()
        if self.collection.enabled:
            self.runtime.camera.validate()
        self.runtime = resolve_runtime_table(self.runtime)
        if not self.runtime.hand.enabled and (
            self.collection.enabled or not self.allow_no_hand
        ):
            raise ValueError("hand-disabled operation requires explicit unrecorded debug mode")
        calibration = load_vr_transform(self.vr_transform_path)
        mapping = self.runtime.teleop.vr_mapping
        controller = TeleopController(
            ActionMapper.for_mode(
                "eef",
                ik_config=OnlineIKConfig(
                    max_pose_error_pos_m=self.runtime.motion.ik_max_pose_error_pos_m,
                    max_pose_error_rot_rad=self.runtime.motion.ik_max_pose_error_rot_rad,
                    operational_joint_lower_rad=tuple(self.runtime.arm.joint_limit_lower),
                    operational_joint_upper_rad=tuple(self.runtime.arm.joint_limit_upper),
                ),
                hand_enabled=self.runtime.hand.enabled,
                hand_lower_rad=self.runtime.hand.qpos_min_rad,
                hand_upper_rad=self.runtime.hand.qpos_max_rad,
                home_hand_qpos=np.deg2rad(self.runtime.hand.home_qpos_deg),
                workspace=self.runtime.environment.workspace.as_array(),
                workspace_margin_m=self.runtime.environment.eef_workspace_margin_m,
                static_boxes=self.runtime.environment.static_boxes,
            ),
            VRWristMapper(
                pos_scale=mapping.pos_scale,
                vr_to_robot_rot=calibration.transform,
            ),
            self.runtime.teleop.ema,
            self.build_hand_retargeter(),
        )
        home_planner = build_home_planner(self.runtime)
        camera_calibration = (
            load_optional_camera_extrinsics(self.camera_calibration_path)
            if self.collection.enabled
            else None
        )
        needs = ObservationNeeds(
            camera="rgbd" if self.collection.enabled else "none", vr=True,
            required_tactile=("aggregate", "dense") if self.collection.enabled else (),
        )
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
            self.sensors = SensorProcesses(self.runtime_state, channels)
            self.robot = DexManiRobot(
                self.runtime_state, self.runtime, check_services=self.sensors.check,
                required_tactile=needs.required_tactile,
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
                max_record_duration_s=self.collection.max_duration_s,
                needs=needs,
                sensors=self.sensors,
                controller=controller,
                home_planner=home_planner,
                keyboard=keyboard,
                audio=audio,
                recorder=(
                    AsyncEpisodeRecorder(
                        Path(__file__).resolve().parents[2]
                        / self.collection.episodes_dir
                        / self.task_name,
                        control_hz=self.runtime.teleop.control_hz,
                    )
                    if self.collection.enabled
                    else None
                ),
                camera_calibration=camera_calibration,
                vr_process=ctx.Process(
                    name="vr",
                    target=run_vr_worker,
                    args=(channels, self.runtime.vr),
                ),
            )
            keyboard.start()
            self.robot.check_services = lambda: (
                keyboard.healthy and self.sensors.check()
            )
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
                    check_services=lambda: keyboard.healthy,
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
            self.runtime_state.revoke_motion(reason=RunEndReason.RUNTIME_SHUTDOWN)
            cleanup_failed = False
            try:
                try:
                    if self.runner is not None:
                        self.runner.close()
                except Exception:
                    cleanup_failed = True
                    logger.exception("teleop capture cleanup failed")
                try:
                    # Preserve the final cue after capture finalization and before device shutdown.
                    if audio is not None and not audio.wait_until_idle(timeout_s=END_AUDIO_GRACE_S):
                        logger.warning("Final audio did not finish within %.1fs", END_AUDIO_GRACE_S)
                except Exception:
                    logger.warning("Final audio unavailable", exc_info=True)
                for close in (
                    keyboard.stop if keyboard is not None else None,
                    audio.close if audio is not None else None,
                ):
                    if close is None:
                        continue
                    try:
                        close()
                    except Exception:
                        cleanup_failed = True
                        logger.exception("teleop input/audio cleanup failed")
                if cleanup_failed:
                    self.runtime_state.mark_error()
            finally:
                shutdown_clean = shutdown_local_runtime(
                    self.runtime_state,
                    self.robot,
                    sensors=self.sensors,
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
