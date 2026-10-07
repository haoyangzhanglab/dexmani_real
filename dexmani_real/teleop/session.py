"""Own the local teleop/robot and sensor startup; all hardware work is explicit."""

import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from dexmani_real.calibration import VR_TRANSFORM_PATH
from dexmani_real.calibration.camera.extrinsics import load_optional_camera_extrinsics
from dexmani_real.config.experiment import resolve_runtime_table, validate_robot_config
from dexmani_real.ipc.channels import RuntimeChannels, RuntimeChannelsConfig
from dexmani_real.recording.recorder import AsyncEpisodeRecorder
from dexmani_real.robot.action import ActionRealizer
from dexmani_real.robot.arm_homing import build_home_planner
from dexmani_real.robot.robot import DexManiRobot
from dexmani_real.runtime.processes import shutdown_local_runtime
from dexmani_real.runtime.safety import RunEndReason, SafetyState, require_transition
from dexmani_real.runtime.supervisor import RuntimeSupervisor
from dexmani_real.sensor.camera.worker import run_camera_worker
from dexmani_real.sensor.vr_worker import run_vr_worker
from dexmani_real.teleop.config import DEFAULT_TASK_NAME, TeleopConfig, validate_task_dir_name
from dexmani_real.teleop.control.controller import TeleopController
from dexmani_real.teleop.control.vr_mapping import VRWristMapper
from dexmani_real.teleop.runner import TeleopRunner
from dexmani_real.teleop.vr_transform import load_vr_transform
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


def run_teleop_experiment(
    runtime,
    *,
    task_name=DEFAULT_TASK_NAME,
    allow_no_hand=False,
    vr_transform_path=VR_TRANSFORM_PATH,
    camera_calibration_path=None,
):
    task_name = validate_task_dir_name(task_name)
    validate_robot_config(runtime)
    for section in (
        runtime.policy,
        runtime.teleop,
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
    if runtime.policy.hand_enabled:
        (
            runtime.tag_retargeting
            if runtime.policy.hand_retargeting_type == "tag"
            else runtime.dexpilot_retargeting
        ).validate()
    calibration = load_vr_transform(vr_transform_path)
    config = TeleopConfig(runtime, task_label=task_name)
    mapping = runtime.policy.vr_mapping
    controller = TeleopController(
        ActionRealizer.for_mode(runtime, "eef"),
        VRWristMapper(
            pos_scale=mapping.pos_scale,
            rot_scale=mapping.rot_scale,
            vr_to_robot_rot=calibration.transform,
            base_to_world_rot=np.eye(3),
        ),
        runtime,
        _build_hand_retargeter(config),
    )
    home_planner = build_home_planner(runtime)
    camera_calibration = (
        load_optional_camera_extrinsics(camera_calibration_path)
        if runtime.policy.recording_enabled
        else None
    )
    ctx = mp.get_context("spawn")
    shared = RuntimeChannels.create(
        prefix=f"dexmani_collect_{os.getpid()}",
        config=RuntimeChannelsConfig.from_runtime(
            runtime, camera=runtime.policy.recording_enabled, vr=True
        ),
        mp_context=ctx,
    )
    supervisor = robot = None
    clean = False
    runner = None
    runner_started = False
    try:
        supervisor = RuntimeSupervisor(shared, runtime.safety.readiness_timeouts_s)
        robot = DexManiRobot(shared, runtime, check_services=supervisor.check)
        runner = TeleopRunner(
            shared,
            config,
            robot,
            controller=controller,
            home_planner=home_planner,
            recorder=AsyncEpisodeRecorder(
                Path(__file__).resolve().parents[2]
                / runtime.policy.episodes_dir
                / config.task_label,
                control_hz=runtime.teleop.control_hz,
                rgb_shape=(runtime.camera.height, runtime.camera.width, 3),
            )
            if runtime.policy.recording_enabled
            else None,
            camera_calibration=camera_calibration,
            start_vr=lambda: supervisor.start(
                [ctx.Process(name="vr", target=run_vr_worker, args=(shared, runtime.vr))],
                wait_ready=False,
            ),
        )
        robot.connect()
        if runtime.policy.recording_enabled:
            supervisor.start(
                [
                    ctx.Process(
                        name="camera", target=run_camera_worker, args=(shared, runtime.camera)
                    )
                ]
            )
        require_transition(shared, SafetyState.ARMED)
        runner_started = True
        runner.run()
        clean = bool(shared.quit_requested.value)
    except KeyboardInterrupt:
        shared.estop_request.value = True
    except Exception:
        shared.error_state.value = True
        logger.exception("teleop session failed")
    finally:
        from dexmani_real.runtime.safety import revoke_motion

        revoke_motion(shared, reason=RunEndReason.RUNTIME_SHUTDOWN)
        if runner is not None and not runner_started:
            cleanup_error = runner._shutdown_capture_and_inputs(None)
            if cleanup_error is not None:
                shared.error_state.value = True
        shutdown_clean = (
            shutdown_local_runtime(robot, supervisor, timeout_s=runtime.safety.shutdown_timeout_s)
            if supervisor is not None
            else shared.close()
        )
    return int(not clean or not shutdown_clean)


def _build_hand_retargeter(config: TeleopConfig):
    """Build the configured hand retargeter without owning runtime state."""
    if not config.runtime.policy.hand_enabled:
        return None
    if config.runtime.policy.hand_retargeting_type == "tag":
        from dexmani_real.teleop.retargeting.tag_optimizer import TAGHandRetargeter

        return TAGHandRetargeter(
            fingertip_link_names=config.runtime.hand.fingertip_link_names,
            tag_config=config.runtime.tag_retargeting,
            urdf_path=config.hand_urdf_path,
        )
    from dexmani_real.teleop.retargeting.dexpilot import DexPilotHandRetargeter

    return DexPilotHandRetargeter(
        hand_type="right",
        retargeting_type=config.runtime.policy.hand_retargeting_type,
        dexpilot_config=config.runtime.dexpilot_retargeting,
    )
