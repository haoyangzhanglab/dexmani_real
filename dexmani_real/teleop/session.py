"""Own the local teleop/robot and sensor startup; all hardware work is explicit."""

import logging
import multiprocessing as mp
from pathlib import Path

from dexmani_real.calibration import VR_TRANSFORM_PATH
from dexmani_real.calibration.camera.extrinsics import load_optional_camera_extrinsics
from dexmani_real.config.experiment import (
    resolve_runtime_table, validate_robot_config, validate_control_rate,
)
from dexmani_real.ipc.channels import SensorChannels
from dexmani_real.runtime.state import RuntimeState, RunEndReason
from dexmani_real.recording.recorder import AsyncEpisodeRecorder
from dexmani_real.robot.action import ActionRealizer
from dexmani_real.robot.home import build_home_planner
from dexmani_real.robot.robot import DexManiRobot
from dexmani_real.robot.model import XHAND_RIGHT_URDF_PATH
from dexmani_real.runtime.processes import SensorProcesses, shutdown_local_runtime
from dexmani_real.sensor.camera.worker import run_camera_worker
from dexmani_real.sensor.vr_worker import run_vr_worker
from dexmani_real.teleop.config import DEFAULT_TASK_NAME, validate_task_dir_name
from dexmani_real.teleop.control.controller import TeleopController
from dexmani_real.teleop.control.vr_mapping import VRWristMapper
from dexmani_real.teleop.runner import TeleopRunner
from dexmani_real.teleop.vr_transform import load_vr_transform
from dexmani_real.utils.log import configure_logging


logger = logging.getLogger(__name__)


def run_teleop_experiment(
    runtime,
    *,
    task_name=DEFAULT_TASK_NAME,
    allow_no_hand=False,
    vr_transform_path=VR_TRANSFORM_PATH,
    camera_calibration_path=None,
):
    configure_logging()
    task_name = validate_task_dir_name(task_name)
    validate_robot_config(runtime)
    validate_control_rate(runtime, runtime.teleop.control_hz)
    for section in (
        runtime.policy,
        runtime.policy.ema,
        runtime.policy.vr_mapping,
        runtime.vr,
    ):
        section.validate()
    if runtime.policy.recording_enabled:
        runtime.camera.validate()
    runtime = resolve_runtime_table(runtime)
    if not runtime.policy.hand_enabled and (runtime.policy.recording_enabled or not allow_no_hand):
        raise ValueError("hand-disabled operation requires explicit unrecorded debug mode")
    calibration = load_vr_transform(vr_transform_path)
    mapping = runtime.policy.vr_mapping
    controller = TeleopController(
        ActionRealizer.for_mode(runtime, "eef"),
        VRWristMapper(
            pos_scale=mapping.pos_scale,
            vr_to_robot_rot=calibration.transform,
        ),
        runtime,
        _build_hand_retargeter(runtime),
    )
    home_planner = build_home_planner(runtime)
    camera_calibration = (
        load_optional_camera_extrinsics(camera_calibration_path)
        if runtime.policy.recording_enabled
        else None
    )
    ctx = mp.get_context("spawn")
    shared = RuntimeState(SensorChannels(
        ctx, camera=runtime.camera if runtime.policy.recording_enabled else None, vr=True,
    ))
    sensors = robot = None
    clean = False
    runner = None
    try:
        sensors = SensorProcesses(shared)
        robot = DexManiRobot(shared, runtime, check_services=sensors.check)
        runner = TeleopRunner(
            shared,
            runtime,
            robot,
            task_label=task_name,
            controller=controller,
            home_planner=home_planner,
            recorder=AsyncEpisodeRecorder(
                Path(__file__).resolve().parents[2]
                / runtime.policy.episodes_dir
                / task_name,
                control_hz=runtime.teleop.control_hz,
            )
            if runtime.policy.recording_enabled
            else None,
            camera_calibration=camera_calibration,
            start_vr=lambda: sensors.start(
                ctx.Process(name="vr", target=run_vr_worker, args=(shared.sensors, runtime.vr)),
                ready=shared.sensors.vr_ready,
            ),
        )
        robot.connect()
        if runtime.policy.recording_enabled:
            sensors.start(
                ctx.Process(
                    name="camera", target=run_camera_worker,
                    args=(shared.sensors, runtime.camera),
                ),
                ready=shared.sensors.camera_ready,
                timeout_s=runtime.safety.readiness_timeouts_s["camera"],
            )
        shared.arm()
        runner.run()
        clean = bool(shared.quit_requested)
    except KeyboardInterrupt:
        shared.request_emergency_stop()
    except Exception:
        shared.error_state = True
        logger.exception("teleop session failed")
    finally:
        shared.revoke_motion(reason=RunEndReason.RUNTIME_SHUTDOWN)
        try:
            if runner is not None:
                runner.close()
        except Exception:
            shared.error_state = True
            logger.exception("teleop capture/input cleanup failed")
        finally:
            shutdown_clean = shutdown_local_runtime(
                shared, robot, sensors=sensors, timeout_s=runtime.safety.shutdown_timeout_s,
            )
    return int(not clean or not shutdown_clean)


def _build_hand_retargeter(runtime):
    if not runtime.policy.hand_enabled:
        return None
    if runtime.policy.hand_retargeting_type == "tag":
        from dexmani_real.teleop.retargeting.tag_optimizer import TAGHandRetargeter

        return TAGHandRetargeter(
            fingertip_link_names=runtime.hand.fingertip_link_names,
            tag_config=runtime.tag_retargeting,
            urdf_path=str(XHAND_RIGHT_URDF_PATH),
        )
    from dexmani_real.teleop.retargeting.dexpilot import DexPilotHandRetargeter

    return DexPilotHandRetargeter(
        dexpilot_config=runtime.dexpilot_retargeting,
    )
