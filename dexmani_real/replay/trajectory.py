"""Load and fail-closed validate trajectories before physical replay starts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from dexmani_real.config.runtime import ResolvedRuntimeConfig
from dexmani_real.ipc.schema import (
    ARM_DOF,
    ARM_EE_SHAPE,
    ARM_JOINT_SHAPE,
    HAND_JOINT_SHAPE,
)
from dexmani_real.planning import Pose, XArm7MotionPlanner, XArm7PlannerConfig
from dexmani_real.recording.reader import EpisodeReader
from dexmani_real.robot_spec import (
    XARM7_XHAND_COLLISION_URDF_PATH,
    XARM7_XHAND_RIGHT_URDF_PATH,
    XARM7_XHAND_SRDF_PATH,
)
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_MIN_EPISODE_RATE_HZ = 1.0
_MAX_EPISODE_RATE_HZ = 100.0
_MODEL_PROVENANCE_KEYS = (
    "arm_hand_collision_urdf_sha256",
    "arm_hand_urdf_sha256",
    "arm_hand_srdf_sha256",
)


def _is_sha256(value: str | None) -> bool:
    """Return True if *value* is a 64-character hex SHA-256 string."""
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def preflight_model_paths() -> tuple[Path, ...]:
    """Return the three URDF/SRDF model paths used for geometry provenance checks."""
    return (
        XARM7_XHAND_COLLISION_URDF_PATH,
        XARM7_XHAND_RIGHT_URDF_PATH,
        XARM7_XHAND_SRDF_PATH,
    )


@dataclass
class TrajectoryData:
    """Preloaded state and command streams used by replay and evaluation."""

    episode_path: str
    num_frames: int
    fps: float
    task_label: str
    action_arm_joint: np.ndarray
    action_hand_joint: np.ndarray | None
    arm_qpos: np.ndarray
    hand_qpos: np.ndarray | None
    arm_ee: np.ndarray | None
    action_source: str | None = None
    resolved_config_sha256: str | None = None
    model_provenance: tuple[tuple[str, str], ...] = ()
    send_mask: np.ndarray | None = None
    processed: bool = False

    @property
    def has_hand(self) -> bool:
        """Whether a fixed-shape hand action stream is present."""
        return self.has_hand_actions

    @property
    def has_hand_actions(self) -> bool:
        """Whether a fixed-shape hand action dataset is present."""
        return self.action_hand_joint is not None


def resolve_episode_path(raw_path: str) -> tuple[str, str]:
    """Validate and name one published episode directory."""
    path = Path(raw_path)
    if not path.is_dir():
        raise ValueError(f"episode must be a published directory: {path}")
    if not (path / "data.h5").is_file():
        raise ValueError(f"episode directory is missing data.h5: {path}")
    return str(path), path.name


def load_trajectory(
    episode_path: str,
) -> TrajectoryData:
    """Load the exact submitted command stream for physical replay."""
    resolved_path, _episode_name = resolve_episode_path(episode_path)
    if not Path(resolved_path).exists():
        raise FileNotFoundError(f"Episode not found: {episode_path}")

    with EpisodeReader(resolved_path) as reader:
        if not reader.meets_min_duration:
            logger.warning(
                "Episode %s is internally readable but below the configured minimum recording duration",
                resolved_path,
            )
        reader.require_valid(purpose="physical replay")
        h5 = reader.h5f
        meta = h5.get("meta")
        num_frames_attr = (
            int(meta.attrs.get("num_frames", 0)) if meta is not None else 0
        )
        fps = float(reader.timing.rate_hz)
        if not _MIN_EPISODE_RATE_HZ <= fps <= _MAX_EPISODE_RATE_HZ:
            raise ValueError(
                f"physical replay requires a valid episode rate, got {fps!r} Hz"
            )
        task_label = str(meta.attrs.get("task_label", "")) if meta is not None else ""
        resolved_config_sha256 = None
        model_provenance: tuple[tuple[str, str], ...] = ()
        if meta is not None:
            raw_hash = meta.attrs.get("resolved_config_sha256")
            resolved_config_sha256 = None if raw_hash is None else str(raw_hash)
            model_provenance = tuple(
                sorted(
                    (name.removeprefix("provenance_"), str(meta.attrs[name]))
                    for name in meta.attrs
                    if name.startswith("provenance_arm_hand_")
                )
            )

        arm_action_key = "action_arm_joint_sent"
        action_source = "sent"
        if arm_action_key not in h5:
            raise ValueError("physical replay requires /action_arm_joint_sent")
        for key in (arm_action_key, "arm_qpos"):
            if key not in h5:
                raise ValueError(f"episode missing required dataset: /{key}")

        source_frames = int(h5[arm_action_key].shape[0])
        total_frames = (
            source_frames
            if num_frames_attr == 0
            else min(source_frames, num_frames_attr)
        )

        action_arm_joint = np.asarray(
            h5[arm_action_key][:total_frames], dtype=np.float64
        )
        arm_qpos = np.asarray(h5["arm_qpos"][:total_frames], dtype=np.float64)
        action_hand_joint = (
            np.asarray(h5["action_hand_joint"][:total_frames], dtype=np.float64)
            if "action_hand_joint" in h5
            else None
        )
        hand_qpos = (
            np.asarray(h5["hand_qpos"][:total_frames], dtype=np.float64)
            if "hand_qpos" in h5
            else None
        )
        arm_ee = (
            np.asarray(h5["arm_ee"][:total_frames], dtype=np.float64)
            if "arm_ee" in h5
            else None
        )
        send_mask = (
            np.asarray(h5["flag_action_queued"][:total_frames], dtype=bool)
            if "flag_action_queued" in h5
            else None
        )

    arrays: dict[str, tuple[np.ndarray, tuple[int, ...]]] = {
        "arm action": (action_arm_joint, (total_frames, *ARM_JOINT_SHAPE)),
        "arm state": (arm_qpos, (total_frames, *ARM_JOINT_SHAPE)),
    }
    if action_hand_joint is not None:
        arrays["hand action"] = (action_hand_joint, (total_frames, *HAND_JOINT_SHAPE))
    if hand_qpos is not None:
        arrays["hand state"] = (hand_qpos, (total_frames, *HAND_JOINT_SHAPE))
    if arm_ee is not None:
        arrays["arm EEF"] = (arm_ee, (total_frames, *ARM_EE_SHAPE))
    if send_mask is not None:
        arrays["send mask"] = (send_mask, (total_frames,))
    for name, (array, expected_shape) in arrays.items():
        if array.shape != expected_shape:
            raise ValueError(
                f"episode {name} has shape {array.shape}, expected {expected_shape}"
            )

    trajectory = TrajectoryData(
        episode_path=resolved_path,
        num_frames=total_frames,
        fps=fps,
        task_label=task_label,
        action_arm_joint=action_arm_joint,
        action_hand_joint=action_hand_joint,
        arm_qpos=arm_qpos,
        hand_qpos=hand_qpos,
        arm_ee=arm_ee,
        action_source=action_source,
        resolved_config_sha256=resolved_config_sha256,
        model_provenance=model_provenance,
        send_mask=send_mask,
    )
    logger.info(
        "Loaded trajectory: %d frames, fps=%.1f, task=%s, hand=%s, ee=%s",
        trajectory.num_frames,
        trajectory.fps,
        trajectory.task_label or "(none)",
        ("yes" if trajectory.has_hand_actions else "no"),
        "yes" if trajectory.arm_ee is not None else "no",
    )
    return trajectory


def load_processed_trajectory(episode_path: str) -> TrajectoryData:
    """Load a processed HDF5 v7 artifact's exact submitted command stream.

    A processed artifact stores the same submitted joint-command stream as its raw
    source episode — ``action[:, :7]`` is ``action_arm_joint_sent`` and
    ``action[:, 7:]`` is ``action_hand_joint`` — but it is row-compacted and does
    not carry the recording's model (URDF/SRDF) provenance.  Replaying one is a
    lower-assurance operation than replaying the raw source: the caller must opt in
    explicitly and must still run the full geometry preflight (see
    :func:`verify_replay_preflight`).
    """
    from dexmani_real.data.process import (
        PROCESSED_SCHEMA_NAME,
        PROCESSED_SCHEMA_VERSION,
    )

    path = Path(episode_path)
    if not path.is_file():
        raise ValueError(f"processed episode must be an HDF5 file: {path}")

    with h5py.File(path, "r") as h5:
        if str(h5.attrs.get("schema_name", "")) != PROCESSED_SCHEMA_NAME:
            raise ValueError(f"not a processed HDF5 artifact: {path.name}")
        if int(h5.attrs.get("schema_version", -1)) != PROCESSED_SCHEMA_VERSION:
            raise ValueError(f"unsupported processed schema version in {path.name}")
        if str(h5.attrs.get("domain", "")) != "real":
            raise ValueError(f"processed episode {path.name} must have domain='real'")

        total_frames = int(h5.attrs.get("episode_steps", -1))
        if total_frames <= 0:
            raise ValueError(f"processed episode {path.name} has invalid episode_steps")
        for key in ("joint_state", "action", "action_ee"):
            if key not in h5:
                raise ValueError(f"processed episode missing required dataset: /{key}")

        action = np.asarray(h5["action"][:], dtype=np.float64)
        joint_state = np.asarray(h5["joint_state"][:], dtype=np.float64)
        action_ee = np.asarray(h5["action_ee"][:], dtype=np.float64)

        dt = float(h5.attrs.get("dt", np.nan))
        task_label = str(h5.attrs.get("task_name", "")).strip()
        resolved_config_sha256 = str(
            h5.attrs.get("source_resolved_config_sha256", "unknown")
        )

        raw_quality = h5.attrs.get("quality_summary_json")
        risky_bridge_count = 0
        if raw_quality is not None:
            try:
                quality = json.loads(str(raw_quality))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"processed episode {path.name} has invalid quality_summary_json"
                ) from exc
            if not isinstance(quality, dict):
                raise ValueError(
                    f"processed episode {path.name} quality_summary_json must encode an object"
                )
            risky_bridge_count = int(quality.get("risky_bridge_count", 0))
        if risky_bridge_count > 0:
            raise ValueError(
                f"processed episode {path.name} has {risky_bridge_count} risky bridge "
                "transition(s): row compaction created abrupt command jumps that the "
                "geometry preflight does not bound. Refusing to physically replay; "
                "reprocess with bridge_policy=reject or replay the raw source episode."
            )

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"processed episode {path.name} has invalid dt")
    fps = 1.0 / dt
    if not _MIN_EPISODE_RATE_HZ <= fps <= _MAX_EPISODE_RATE_HZ:
        raise ValueError(
            f"physical replay requires a valid episode rate, got {fps!r} Hz"
        )

    action_arm_joint = action[:, :ARM_DOF]
    action_hand_joint = action[:, ARM_DOF:]
    arm_qpos = joint_state[:, :ARM_DOF]
    hand_qpos = joint_state[:, ARM_DOF:]
    # action_ee[:, :9] is the commanded (target) EEF; the processed file carries no
    # measured-EEF dataset.  Consistency evaluation therefore compares the replayed
    # measured EEF against this commanded reference rather than a recorded EEF.
    arm_ee = action_ee[:, : ARM_EE_SHAPE[0]]

    arrays: dict[str, tuple[np.ndarray, tuple[int, ...]]] = {
        "arm action": (action_arm_joint, (total_frames, *ARM_JOINT_SHAPE)),
        "arm state": (arm_qpos, (total_frames, *ARM_JOINT_SHAPE)),
        "hand action": (action_hand_joint, (total_frames, *HAND_JOINT_SHAPE)),
        "hand state": (hand_qpos, (total_frames, *HAND_JOINT_SHAPE)),
        "arm EEF": (arm_ee, (total_frames, *ARM_EE_SHAPE)),
    }
    for name, (array, expected_shape) in arrays.items():
        if array.shape != expected_shape:
            raise ValueError(
                f"processed episode {name} has shape {array.shape}, expected {expected_shape}"
            )

    trajectory = TrajectoryData(
        episode_path=str(path),
        num_frames=total_frames,
        fps=fps,
        task_label=task_label,
        action_arm_joint=action_arm_joint,
        action_hand_joint=action_hand_joint,
        arm_qpos=arm_qpos,
        hand_qpos=hand_qpos,
        arm_ee=arm_ee,
        action_source="sent",
        resolved_config_sha256=resolved_config_sha256,
        model_provenance=(),
        send_mask=None,
        processed=True,
    )
    logger.info(
        "Loaded processed trajectory: %d frames, fps=%.1f, task=%s, hand=yes, ee=yes",
        trajectory.num_frames,
        trajectory.fps,
        trajectory.task_label or "(none)",
    )
    return trajectory


def modeled_hand_actions(trajectory: TrajectoryData) -> np.ndarray:
    """Return the recorded hand action stream used for geometry preflight."""
    if trajectory.action_hand_joint is None:
        raise ValueError(
            "episode has no hand action stream; physical replay requires recorded hand data"
        )
    actions = np.asarray(trajectory.action_hand_joint, dtype=np.float64)
    expected_shape = (trajectory.num_frames, *HAND_JOINT_SHAPE)
    if actions.shape != expected_shape or not np.all(np.isfinite(actions)):
        raise ValueError(
            f"physical replay hand actions must be finite shape {expected_shape}"
        )
    return actions


def require_hand_actions(trajectory: TrajectoryData) -> None:
    """Fail closed when an episode has no recorded hand action stream."""
    if not trajectory.has_hand_actions:
        raise ValueError(
            "episode has no hand action stream; physical replay requires recorded hand data"
        )


def _verify_trajectory_provenance(
    trajectory: TrajectoryData,
    *,
    provenance_sha256: str,
    verify_model_provenance: bool = True,
) -> None:
    """Fail-closed provenance gate: source stream, config hash, and model hashes.

    ``verify_model_provenance=False`` is reserved for processed artifacts, which
    do not record the URDF/SRDF hashes of the recording-time collision models.
    The source-stream and config-hash checks above still apply.
    """
    if trajectory.action_source != "sent":
        raise ValueError(
            "physical replay requires the exact submitted action stream ('sent')"
        )
    if trajectory.num_frames <= 0:
        raise ValueError("physical replay trajectory is empty")
    arm_actions = np.asarray(trajectory.action_arm_joint)
    expected_arm_shape = (trajectory.num_frames, *ARM_JOINT_SHAPE)
    if arm_actions.shape != expected_arm_shape or not np.all(np.isfinite(arm_actions)):
        raise ValueError(
            f"physical replay arm actions must be finite shape {expected_arm_shape}"
        )
    if not _is_sha256(trajectory.resolved_config_sha256):
        raise ValueError(
            "physical replay recording provenance lacks a valid resolved_config_sha256"
        )
    if trajectory.resolved_config_sha256 != provenance_sha256:
        raise ValueError(
            "physical replay config provenance mismatch: recorded config differs from replay config"
        )

    if not verify_model_provenance:
        return

    recorded_models = dict(trajectory.model_provenance)
    missing_models = [
        name
        for name in _MODEL_PROVENANCE_KEYS
        if not _is_sha256(recorded_models.get(name))
    ]
    if missing_models:
        raise ValueError(
            f"physical replay model provenance is incomplete: {missing_models}"
        )
    current_models = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in zip(_MODEL_PROVENANCE_KEYS, preflight_model_paths())
    }
    mismatched_models = [
        name
        for name in _MODEL_PROVENANCE_KEYS
        if recorded_models[name] != current_models[name]
    ]
    if mismatched_models:
        raise ValueError(
            f"physical replay model provenance mismatch: {mismatched_models}"
        )


def verify_replay_preflight(
    trajectory: TrajectoryData,
    runtime: ResolvedRuntimeConfig,
    *,
    provenance_sha256: str,
) -> None:
    """Fail-closed validation immediately before spawning hardware workers.

    Checks: hand-data attestation, provenance (source/config/models), and every
    adjacent-frame pair for workspace bounds and collision (self-collision plus
    static obstacle boxes). Robot-table contact is deliberately not a replay
    rejection condition (user_design.md §3): replayed episodes were recorded by
    teleop without table gating, and table clearance remains enforced on the
    return-home path. Called once before worker startup; any rejection prevents
    hardware access entirely.  For a processed artifact (``trajectory.processed``),
    the recording-time model (URDF/SRDF) provenance check is skipped — that hash is
    not carried forward — but the source-stream, config-hash, workspace, and
    collision checks all still apply.
    """
    require_hand_actions(trajectory)
    if not bool(runtime.policy.hand_enabled):
        raise ValueError("physical replay requires policy.hand_enabled=true")
    verify_model_provenance = not trajectory.processed
    _verify_trajectory_provenance(
        trajectory,
        provenance_sha256=provenance_sha256,
        verify_model_provenance=verify_model_provenance,
    )
    if not verify_model_provenance:
        logger.warning(
            "processed replay: model (URDF/SRDF) provenance is absent from the "
            "processed artifact; geometry preflight proceeds against current models "
            "without a recording-time model hash to compare"
        )
    modeled_hand = modeled_hand_actions(trajectory)
    workspace = np.array(
        [
            [runtime.policy.workspace.x_min, runtime.policy.workspace.x_max],
            [runtime.policy.workspace.y_min, runtime.policy.workspace.y_max],
            [runtime.policy.workspace.z_min, runtime.policy.workspace.z_max],
        ],
        dtype=np.float64,
    )
    planner = XArm7MotionPlanner(
        XArm7PlannerConfig(
            urdf_path=str(XARM7_XHAND_COLLISION_URDF_PATH),
            srdf_path=str(XARM7_XHAND_SRDF_PATH),
            base_pose_world=Pose(p=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0])),
            workspace_bounds=workspace,
        ),
        hand_dof=True,
        static_boxes=tuple(runtime.environment.static_boxes),
        # user_design.md §3: replay does not reject on robot-table contact.
        # Table clearance stays enforced on the return-home path, which uses the
        # replay controller's own planner (see replay_controller.setup).
        table=None,
    )
    arm_actions = np.asarray(trajectory.action_arm_joint, dtype=np.float64)
    for index in range(max(1, trajectory.num_frames - 1)):
        start = min(index, trajectory.num_frames - 1)
        end = min(index + 1, trajectory.num_frames - 1)
        arm_start, arm_end = arm_actions[start], arm_actions[end]
        hand_start, hand_end = modeled_hand[start], modeled_hand[end]
        if not planner.is_workspace_segment_safe(arm_start, arm_end):
            raise ValueError(
                f"physical replay workspace rejection at transition {start}->{end}"
            )
        if not planner.collision_model.check_transition_collision_free(
            arm_start, arm_end, hand_start, hand_end
        ):
            raise ValueError(
                f"physical replay collision rejection at transition {start}->{end}"
            )
