"""Own the local teleop/robot and sensor startup; all hardware work is explicit."""

import multiprocessing as mp
import os

from dexmani_real.calibration import CAMERAS_PATH, VR_TRANSFORM_PATH
from dexmani_real.ipc.channels import RuntimeChannels, RuntimeChannelsConfig
from dexmani_real.recording.recorder import HARD_MAX_RECORD_FRAMES
from dexmani_real.robot.model import (
    XARM7_XHAND_COLLISION_URDF_PATH,
    XARM7_XHAND_RIGHT_URDF_PATH,
    XARM7_XHAND_SRDF_PATH,
)
from dexmani_real.robot.robot import DexManiRobot
from dexmani_real.runtime.processes import shutdown_local_runtime
from dexmani_real.runtime.safety import SafetyState, require_transition
from dexmani_real.runtime.supervisor import RuntimeSupervisor
from dexmani_real.sensor.camera.worker import run_camera_worker
from dexmani_real.sensor.vr_worker import run_vr_worker
from dexmani_real.teleop.config import TeleopConfig
from dexmani_real.teleop.runner import TeleopRunner
from dexmani_real.teleop.vr_transform import load_vr_transform
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)
DEFAULT_TASK_NAME = "test"


def validate_task_dir_name(value: str) -> str:
    """Validate one task name as a safe directory component."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("task_name must be a non-empty string without surrounding whitespace")
    if value in {".", ".."} or value.startswith("."):
        raise ValueError("task_name must not be a hidden or relative directory name")
    if "/" in value or "\\" in value or any(ord(char) < 32 for char in value):
        raise ValueError("task_name must be one safe directory component")
    return value


def _validate_recording_resources() -> None:
    """Ensure files needed by recording and local FK exist before startup."""
    required = (
        XARM7_XHAND_COLLISION_URDF_PATH,
        XARM7_XHAND_RIGHT_URDF_PATH,
        XARM7_XHAND_SRDF_PATH,
        CAMERAS_PATH,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required recording resources are missing: {missing}")


def run_teleop_experiment(runtime, *, task_name=DEFAULT_TASK_NAME, allow_no_hand=False):
    task_name = validate_task_dir_name(task_name)
    if not runtime.policy.hand_enabled and (runtime.policy.recording_enabled or not allow_no_hand):
        raise ValueError("hand-disabled operation requires explicit unrecorded debug mode")
    if runtime.policy.recording_enabled:
        requested_rows = round(runtime.policy.max_record_duration_s * runtime.teleop.control_hz)
        if not 0 < requested_rows < HARD_MAX_RECORD_FRAMES:
            raise ValueError(
                f"teleop recording requests {requested_rows} rows; require 0 < rows < "
                f"hard limit {HARD_MAX_RECORD_FRAMES} "
                f"(max_record_duration_s={runtime.policy.max_record_duration_s}, "
                f"control_hz={runtime.teleop.control_hz}). Configure a positive, shorter "
                "episode budget instead of increasing the recorder hard guard."
            )
    load_vr_transform(VR_TRANSFORM_PATH)
    if runtime.policy.recording_enabled:
        _validate_recording_resources()
    ctx = mp.get_context("spawn")
    shared = RuntimeChannels.create(
        prefix=f"dexmani_collect_{os.getpid()}",
        config=RuntimeChannelsConfig.from_runtime(runtime),
        mp_context=ctx,
    )
    supervisor = RuntimeSupervisor(shared, runtime.safety.readiness_timeouts_s)
    robot = DexManiRobot(shared, runtime, check_services=supervisor.check)
    clean = False
    try:
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
        runner = TeleopRunner(
            shared,
            TeleopConfig(runtime, task_label=task_name),
            robot,
            start_vr=lambda: supervisor.start(
                [ctx.Process(name="vr", target=run_vr_worker, args=(shared, runtime.vr))],
                wait_ready=False,
            ),
        )
        runner.run()
        clean = bool(shared.quit_requested.value)
    except KeyboardInterrupt:
        shared.estop_request.value = True
    except Exception:
        shared.error_state.value = True
        logger.exception("teleop session failed")
    finally:
        from dexmani_real.runtime.safety import revoke_motion

        revoke_motion(shared)
        shutdown_clean = shutdown_local_runtime(
            robot, supervisor, timeout_s=runtime.safety.shutdown_timeout_s
        )
    return int(not clean or not shutdown_clean)
