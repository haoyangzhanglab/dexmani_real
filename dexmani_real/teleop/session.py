"""VR collection process ownership and startup; all hardware work is explicit."""
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any
from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.ipc.channels import RuntimeChannels, RuntimeChannelsConfig
from dexmani_real.recording.io_worker import RecorderIOConfig, recorder_io_loop
from dexmani_real.robot.arm_worker import arm_loop as _arm_loop
from dexmani_real.robot.hand_worker import hand_loop as _hand_loop
from dexmani_real.robot.model import XARM7_XHAND_COLLISION_URDF_PATH, XARM7_XHAND_RIGHT_URDF_PATH, XARM7_XHAND_SRDF_PATH, XHAND_RIGHT_URDF_PATH
from dexmani_real.runtime.safety import SafetyState, require_transition
from dexmani_real.runtime.supervisor import start_processes, run_supervisor
from dexmani_real.runtime.processes import shutdown_processes_verified
from dexmani_real.sensor.camera.worker import CameraLoopConfig, camera_loop as _camera_loop
from dexmani_real.sensor.vr_worker import VRReceiverConfig, vr_loop as _vr_loop
from dexmani_real.teleop.config import TeleopConfig
from dexmani_real.teleop.loop import teleop_loop
from dexmani_real.teleop.vr_transform import load_vr_transform
from dexmani_real.utils.log import get_logger
logger = get_logger(__name__)
DEFAULT_TASK_NAME = "test"

def validate_task_name(value: str) -> str:
    """Validate one task name as a safe directory component."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            "task_name must be a non-empty string without surrounding whitespace"
        )
    if value in {".", ".."} or value.startswith("."):
        raise ValueError("task_name must not be a hidden or relative directory name")
    if "/" in value or "\\" in value or any(ord(char) < 32 for char in value):
        raise ValueError("task_name must be one safe directory component")
    return value


def validate_operator(value: str) -> str:
    """Validate optional operator metadata against the recorder IPC boundary."""
    if not isinstance(value, str):
        raise ValueError("operator must be a string")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("operator must be valid UTF-8 text") from exc
    return value


def _validate_recording_resources(repo_root: Path) -> None:
    """Ensure files needed by recording and FK workers exist before startup."""
    required = (
        XARM7_XHAND_COLLISION_URDF_PATH,
        XARM7_XHAND_RIGHT_URDF_PATH,
        XARM7_XHAND_SRDF_PATH,
        repo_root / "dexmani_real" / "config" / "cameras.json",
        repo_root / "dexmani_real" / "config" / "vr_transform.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required recording resources are missing: {missing}")


def _build_processes(
    context: Any,
    shared: RuntimeChannels,
    runtime: ExperimentConfig,
    *,
    repo_root: Path,
    task_name: str,
    operator: str,
    hand_enabled: bool,
    recording_enabled: bool,
) -> list[Any]:
    policy_config = TeleopConfig.from_runtime(
        runtime,
        task_label=task_name,
        operator=operator,
        hand_urdf_path=str(XHAND_RIGHT_URDF_PATH),
    )
    processes = [
        context.Process(name="arm", target=_arm_loop, args=(shared, runtime.arm)),
        context.Process(name="vr", target=_vr_loop, args=(shared, VRReceiverConfig.from_runtime(runtime))),
        context.Process(name="policy", target=teleop_loop, args=(shared, policy_config)),
    ]
    if recording_enabled:
        camera_config = CameraLoopConfig.from_runtime(runtime)
        processes.append(
            context.Process(name="camera", target=_camera_loop, args=(shared, camera_config))
        )
        # Recorder still owns only episode serialization; the entry point
        # selects the already-validated task parent directory.
        recorder_config = RecorderIOConfig(
            data_dir=str(
                repo_root / policy_config.runtime.policy.episodes_dir / task_name
            ),
            max_frames=int(
                round(
                    policy_config.runtime.policy.max_record_duration_s
                    * policy_config.runtime.teleop.control_hz
                )
            ),
            control_hz=policy_config.runtime.teleop.control_hz,
            min_frames=int(
                round(
                    policy_config.runtime.policy.min_record_duration_s
                    * policy_config.runtime.teleop.control_hz
                )
            ),
            provenance={"workflow": "teleop"},
        )
        processes.append(
            context.Process(name="recorder", target=recorder_io_loop, args=(shared, recorder_config))
        )
    if hand_enabled:
        processes.append(
            context.Process(name="hand", target=_hand_loop, args=(
                    shared,
                    runtime.hand,
                    float(runtime.policy.hand_disconnect_timeout_s),
                ))
        )
    return processes


def run_teleop_experiment(runtime, *, task_name=DEFAULT_TASK_NAME, operator="", allow_no_hand=False):
    task_name, operator = validate_task_name(task_name), validate_operator(operator)
    if not runtime.policy.hand_enabled and (runtime.policy.recording_enabled or not allow_no_hand):
        raise ValueError("hand-disabled operation requires explicit unrecorded debug mode")
    repo_root = Path(__file__).resolve().parents[2]
    load_vr_transform(repo_root / "dexmani_real/config/vr_transform.json")
    if runtime.policy.recording_enabled:
        _validate_recording_resources(repo_root)
    ctx = mp.get_context("spawn")
    shared = RuntimeChannels.create(prefix=f"dexmani_collect_{os.getpid()}",
        config=RuntimeChannelsConfig.from_runtime(runtime, camera_requested=runtime.policy.recording_enabled), mp_context=ctx)
    started = []
    try:
        processes = _build_processes(ctx, shared, runtime, repo_root=repo_root, task_name=task_name,
            operator=operator, hand_enabled=runtime.policy.hand_enabled, recording_enabled=runtime.policy.recording_enabled)
        by_name = {p.name: p for p in processes}
        sensors = [by_name[name] for name in ("arm", "hand", "vr", "camera", "recorder") if name in by_name]
        start_processes(shared, sensors, runtime.safety.readiness_timeouts_s, started)
        require_transition(shared, SafetyState.ARMED)
        start_processes(shared, [by_name["policy"]], runtime.safety.readiness_timeouts_s, started)
        run_supervisor(shared, started)
    except KeyboardInterrupt:
        shared.estop_request.value = True
    except Exception:
        shared.workflow_failed.value = True
        logger.exception("teleop session failed")
    finally:
        report = shutdown_processes_verified(shared, started, disarm_if_clean=True,
            graceful_timeout_s=runtime.safety.shutdown_timeout_s,
            service_process_names={"camera", "recorder", "vr", "policy", "pointcloud"})
    return 1 if shared.error_state.value or shared.workflow_failed.value or not report.shared_closed else 0
