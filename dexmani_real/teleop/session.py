"""Own the local teleop/robot and sensor startup; all hardware work is explicit."""

import logging
import multiprocessing as mp
from pathlib import Path

from dexmani_real.calibration import VR_TRANSFORM_PATH
from dexmani_real.calibration.camera.extrinsics import load_optional_camera_extrinsics
from dexmani_real.calibration.vr_heading import load_vr_transform
from dexmani_real.config.experiment import (
    DEFAULT_TASK_NAME,
    resolve_runtime_table,
    validate_task_dir_name,
    validate_robot_config,
    validate_control_rate,
)
from dexmani_real.data.recorder import AsyncEpisodeRecorder
from dexmani_real.ipc.channels import SensorChannels
from dexmani_real.planning.action import ActionMapper
from dexmani_real.robot.home import build_home_planner
from dexmani_real.robot.model import XHAND_RIGHT_URDF_PATH
from dexmani_real.robot.robot import DexManiRobot
from dexmani_real.runtime.observation import ObservationNeeds
from dexmani_real.runtime.processes import SensorProcesses, shutdown_local_runtime
from dexmani_real.runtime.state import RuntimeState, RunEndReason
from dexmani_real.sensor.camera.worker import run_camera_worker
from dexmani_real.sensor.vr.worker import run_vr_worker
from dexmani_real.teleop.controller import TeleopController
from dexmani_real.teleop.runner import TeleopRunner
from dexmani_real.teleop.vr_mapping import VRWristMapper
from dexmani_real.utils.log import configure_logging

logger = logging.getLogger(__name__)


class TeleopSession:
    """Own teleoperation components, startup and bounded shutdown."""

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
        self.runtime.control.validate()
        self.runtime.vr.validate()
        if self.runtime.control.recording_enabled:
            self.runtime.camera.validate()
        self.runtime = resolve_runtime_table(self.runtime)
        if not self.runtime.control.hand_enabled and (
            self.runtime.control.recording_enabled or not self.allow_no_hand
        ):
            raise ValueError("hand-disabled operation requires explicit unrecorded debug mode")
        calibration = load_vr_transform(self.vr_transform_path)
        mapping = self.runtime.control.vr_mapping
        controller = TeleopController(
            ActionMapper.for_mode(self.runtime, "eef"),
            VRWristMapper(
                pos_scale=mapping.pos_scale,
                vr_to_robot_rot=calibration.transform,
            ),
            self.runtime,
            self.build_hand_retargeter(),
        )
        home_planner = build_home_planner(self.runtime)
        camera_calibration = (
            load_optional_camera_extrinsics(self.camera_calibration_path)
            if self.runtime.control.recording_enabled
            else None
        )
        needs = ObservationNeeds(
            camera="rgbd" if self.runtime.control.recording_enabled else "none", vr=True
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
        try:
            self.sensors = SensorProcesses(self.runtime_state, channels)
            self.robot = DexManiRobot(
                self.runtime_state, self.runtime, check_services=self.sensors.check,
                required_tactile=needs.required_tactile,
            )
            self.runner = TeleopRunner(
                self.runtime_state,
                self.runtime,
                self.robot,
                task_label=self.task_name,
                needs=needs,
                sensors=self.sensors,
                controller=controller,
                home_planner=home_planner,
                recorder=(
                    AsyncEpisodeRecorder(
                        Path(__file__).resolve().parents[2]
                        / self.runtime.control.episodes_dir
                        / self.task_name,
                        control_hz=self.runtime.teleop.control_hz,
                    )
                    if self.runtime.control.recording_enabled
                    else None
                ),
                camera_calibration=camera_calibration,
                vr_process=ctx.Process(
                    name="vr",
                    target=run_vr_worker,
                    args=(channels, self.runtime.vr),
                ),
            )
            self.runner.keyboard.start()
            self.robot.check_services = lambda: (
                self.runner.keyboard.healthy and self.sensors.check()
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
                    check_services=lambda: self.runner.keyboard.healthy,
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
            try:
                if self.runner is not None:
                    self.runner.close()
            except Exception:
                self.runtime_state.mark_error()
                logger.exception("teleop capture/input cleanup failed")
            finally:
                shutdown_clean = shutdown_local_runtime(
                    self.runtime_state,
                    self.robot,
                    sensors=self.sensors,
                    timeout_s=self.runtime.safety.shutdown_timeout_s,
                )
        return int(not clean or not shutdown_clean)

    def build_hand_retargeter(self):
        if not self.runtime.control.hand_enabled:
            return None
        if self.runtime.control.hand_retargeting_type == "tag":
            from dexmani_real.teleop.retargeting.tag import TAGHandRetargeter

            return TAGHandRetargeter(
                fingertip_link_names=self.runtime.hand.fingertip_link_names,
                tag_config=self.runtime.tag_retargeting,
                urdf_path=str(XHAND_RIGHT_URDF_PATH),
            )
        if self.runtime.control.hand_retargeting_type == "dexpilot":
            from dexmani_real.teleop.retargeting.dexpilot import DexPilotHandRetargeter

            return DexPilotHandRetargeter(
                dexpilot_config=self.runtime.dexpilot_retargeting,
            )
        raise ValueError("hand_retargeting_type must be 'tag' or 'dexpilot'")
