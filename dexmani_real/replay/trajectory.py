"""Load exact raw commands and fail-closed validate physical replay trajectories."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.planning import Pose, XArm7MotionPlanner, XArm7PlannerConfig
from dexmani_real.planning.paths import wrap_nearest_equivalent
from dexmani_real.recording.storage.reader import EpisodeReader
from dexmani_real.robot.model import (
    ARM_EE_SHAPE,
    ARM_JOINT_SHAPE,
    HAND_JOINT_SHAPE,
    XARM7_XHAND_COLLISION_URDF_PATH,
    XARM7_XHAND_RIGHT_URDF_PATH,
    XARM7_XHAND_SRDF_PATH,
)
from dexmani_real.utils.atomic_io import sha256_file
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_MIN_EPISODE_RATE_HZ = 1.0
_MAX_EPISODE_RATE_HZ = 100.0
_MODEL_PROVENANCE_KEYS = (
    "arm_hand_collision_urdf_sha256",
    "arm_hand_urdf_sha256",
    "arm_hand_srdf_sha256",
)
_JOINT_LIMIT_TOLERANCE_RAD = 1e-12


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
    provenance_warnings: tuple[str, ...] = ()
    send_mask: np.ndarray | None = None

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


def load_trajectory(episode_path: str) -> TrajectoryData:
    """Load the exact submitted command stream for physical replay."""
    resolved_path, _episode_name = resolve_episode_path(episode_path)
    if not Path(resolved_path).exists():
        raise FileNotFoundError(f"Episode not found: {episode_path}")

    with EpisodeReader(resolved_path) as reader:
        if not reader.min_frames_met:
            logger.warning(
                "Episode %s is internally readable but below the configured minimum recording duration",
                resolved_path,
            )
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


def _processed_replay_source(
    artifact_path: Path,
) -> tuple[Path, np.ndarray, str, int, str]:
    """Read the raw-source identity and retained rows from one processed artifact."""
    from dexmani_real.dataset.processed import (
        PROCESSED_SCHEMA_NAME,
        PROCESSED_SCHEMA_VERSION,
        validate_processed_provenance,
    )

    if not artifact_path.is_file():
        raise ValueError(f"processed episode must be an HDF5 file: {artifact_path}")
    with h5py.File(artifact_path, "r") as artifact:
        if str(artifact.attrs.get("schema_name", "")) != PROCESSED_SCHEMA_NAME:
            raise ValueError(f"not a processed HDF5 artifact: {artifact_path.name}")
        if int(artifact.attrs.get("schema_version", -1)) != PROCESSED_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported processed schema version in {artifact_path.name}"
            )
        if str(artifact.attrs.get("domain", "")) != "real":
            raise ValueError(
                f"processed episode {artifact_path.name} must have domain='real'"
            )
        provenance = validate_processed_provenance(
            artifact,
            label=f"processed episode {artifact_path.name}",
        )

        try:
            decision = json.loads(str(artifact.attrs["source_decision_json"]))
            member_hashes = json.loads(str(artifact.attrs["source_member_sha256_json"]))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"processed episode {artifact_path.name} has invalid source provenance"
            ) from exc
        if not isinstance(decision, dict) or not isinstance(member_hashes, dict):
            raise ValueError(
                f"processed episode {artifact_path.name} has invalid source provenance"
            )
        source_path_text = decision.get("source_path")
        expected_data_sha256 = member_hashes.get("data.h5")
        if not isinstance(source_path_text, str) or not source_path_text:
            raise ValueError(
                f"processed episode {artifact_path.name} lacks its raw source path"
            )
        if not isinstance(expected_data_sha256, str) or not _is_sha256(
            expected_data_sha256
        ):
            raise ValueError(
                f"processed episode {artifact_path.name} lacks a valid raw data.h5 hash"
            )

        source_config_sha256 = str(
            artifact.attrs.get("source_resolved_config_sha256", "")
        )
        retained_rows = provenance.source_rows
        source_frames = int(provenance.keep_mask.shape[0])
        if not np.array_equal(
            provenance.segment_ends,
            np.asarray([retained_rows.size], dtype=np.int64),
        ) or np.any(np.diff(retained_rows) != 1):
            raise ValueError(
                f"processed episode {artifact_path.name} has inconsistent row provenance"
            )

    return (
        Path(source_path_text),
        retained_rows,
        source_config_sha256,
        source_frames,
        expected_data_sha256,
    )


def _select_raw_trajectory_rows(
    raw_trajectory: TrajectoryData,
    retained_rows: np.ndarray,
) -> TrajectoryData:
    """Build a replay trajectory from exact raw commands at retained source rows."""
    rows = np.asarray(retained_rows, dtype=np.int64)
    if (
        rows.ndim != 1
        or rows.size == 0
        or rows[0] < 0
        or rows[-1] >= raw_trajectory.num_frames
        or np.any(np.diff(rows) <= 0)
    ):
        raise ValueError("processed replay rows are invalid for the raw source episode")
    return TrajectoryData(
        episode_path=raw_trajectory.episode_path,
        num_frames=int(rows.size),
        fps=raw_trajectory.fps,
        task_label=raw_trajectory.task_label,
        action_arm_joint=raw_trajectory.action_arm_joint[rows].copy(),
        action_hand_joint=(
            None
            if raw_trajectory.action_hand_joint is None
            else raw_trajectory.action_hand_joint[rows].copy()
        ),
        arm_qpos=raw_trajectory.arm_qpos[rows].copy(),
        hand_qpos=(
            None
            if raw_trajectory.hand_qpos is None
            else raw_trajectory.hand_qpos[rows].copy()
        ),
        arm_ee=(
            None
            if raw_trajectory.arm_ee is None
            else raw_trajectory.arm_ee[rows].copy()
        ),
        action_source=raw_trajectory.action_source,
        resolved_config_sha256=raw_trajectory.resolved_config_sha256,
        model_provenance=raw_trajectory.model_provenance,
        provenance_warnings=raw_trajectory.provenance_warnings,
        send_mask=(
            None
            if raw_trajectory.send_mask is None
            else raw_trajectory.send_mask[rows].copy()
        ),
    )


def load_processed_trajectory(episode_path: str) -> TrajectoryData:
    """Load exact raw commands selected and attested by one processed artifact.

    Processed ``float32`` action arrays are training data, not physical commands.
    This loader uses their provenance only: it hashes the recorded raw ``data.h5``
    and selects the retained rows from the raw ``float64`` submitted command stream.
    """
    artifact_path = Path(episode_path)
    (
        source_path,
        retained_rows,
        source_config_sha256,
        source_frames,
        expected_data_sha256,
    ) = _processed_replay_source(artifact_path)
    source_data_path = source_path / "data.h5"
    if not source_data_path.is_file():
        raise ValueError(
            f"processed episode {artifact_path.name} requires raw source data.h5 at {source_path}"
        )

    if sha256_file(source_data_path) != expected_data_sha256:
        raise ValueError(
            f"processed episode {artifact_path.name} raw source data.h5 hash mismatch"
        )

    raw_trajectory = load_trajectory(str(source_path))
    if raw_trajectory.num_frames != source_frames:
        raise ValueError(
            f"processed episode {artifact_path.name} source frame count does not match raw source"
        )
    if raw_trajectory.resolved_config_sha256 != source_config_sha256:
        raw_trajectory.provenance_warnings += (
            "processed source config hash does not match the raw source",
        )
    trajectory = _select_raw_trajectory_rows(raw_trajectory, retained_rows)
    logger.info(
        "Loaded processed replay selection: %d raw frames from %s (artifact=%s)",
        trajectory.num_frames,
        source_path,
        artifact_path,
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


def replay_start_state(trajectory: TrajectoryData) -> tuple[np.ndarray, np.ndarray]:
    """Return the finite measured arm/hand state at the first replay frame."""
    if trajectory.num_frames <= 0:
        raise ValueError("physical replay trajectory is empty")
    arm_qpos = np.asarray(trajectory.arm_qpos[0], dtype=np.float64)
    if arm_qpos.shape != ARM_JOINT_SHAPE or not np.all(np.isfinite(arm_qpos)):
        raise ValueError("physical replay requires a finite first arm_qpos state")
    if trajectory.hand_qpos is None:
        raise ValueError("physical replay requires a recorded first hand_qpos state")
    hand_qpos = np.asarray(trajectory.hand_qpos[0], dtype=np.float64)
    if hand_qpos.shape != HAND_JOINT_SHAPE or not np.all(np.isfinite(hand_qpos)):
        raise ValueError("physical replay requires a finite first hand_qpos state")
    return arm_qpos.copy(), hand_qpos.copy()


def _verify_trajectory_input(trajectory: TrajectoryData) -> None:
    """Fail closed on the exact source stream needed for physical preflight."""
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


def _canonicalize_replay_arm_actions(
    trajectory: TrajectoryData,
    runtime: ExperimentConfig,
) -> np.ndarray:
    """Return the nearest-equivalent arm stream used for physical preflight.

    Each target is mapped to the nearest limit-valid 2π equivalent relative to
    the preceding target, beginning at the recorded start state.  This models
    replay's measured-pose canonicalization without rejecting equivalent xArm
    angles or checking geometry on discontinuous raw angle representatives.
    """
    arm_actions = np.asarray(trajectory.action_arm_joint, dtype=np.float64)
    canonical_actions = np.empty_like(arm_actions)
    lower = np.asarray(runtime.arm.joint_limit_lower, dtype=np.float64)
    upper = np.asarray(runtime.arm.joint_limit_upper, dtype=np.float64)
    reference = np.asarray(trajectory.arm_qpos[0], dtype=np.float64)
    for frame_index, action in enumerate(arm_actions):
        canonical = wrap_nearest_equivalent(
            action,
            reference,
            tuple(runtime.arm.joint_limit_lower),
            tuple(runtime.arm.joint_limit_upper),
        )
        if np.any(canonical < lower) or np.any(canonical > upper):
            raise ValueError(
                "physical replay arm action at frame "
                f"{frame_index} violates joint limits"
            )
        canonical_actions[frame_index] = canonical
        reference = canonical
    return canonical_actions


def _validate_replay_hand_limits(
    hand_actions: np.ndarray,
    recorded_hand_start: np.ndarray,
    runtime: ExperimentConfig,
) -> None:
    """Reject hand states or commands outside their physical replay envelopes."""
    mechanical_lower = np.asarray(
        runtime.hand.mechanical_qpos_min_rad, dtype=np.float64
    )
    mechanical_upper = np.asarray(
        runtime.hand.mechanical_qpos_max_rad, dtype=np.float64
    )
    if np.any(recorded_hand_start < mechanical_lower) or np.any(
        recorded_hand_start > mechanical_upper
    ):
        raise ValueError("physical replay first hand_qpos violates mechanical limits")

    command_lower = np.asarray(runtime.hand.qpos_min_rad, dtype=np.float64)
    command_upper = np.asarray(runtime.hand.qpos_max_rad, dtype=np.float64)
    violation_rows = np.flatnonzero(
        np.any(
            (hand_actions < command_lower - _JOINT_LIMIT_TOLERANCE_RAD)
            | (hand_actions > command_upper + _JOINT_LIMIT_TOLERANCE_RAD),
            axis=1,
        )
    )
    if violation_rows.size:
        raise ValueError(
            "physical replay hand action at frame "
            f"{int(violation_rows[0])} violates command joint limits"
        )


def _reproducibility_warnings(
    trajectory: TrajectoryData,
    *,
    provenance_sha256: str,
) -> tuple[str, ...]:
    """Return non-safety provenance differences after physical validation.

    The caller must only use this after the complete trajectory has passed the
    current geometry and runtime preflight.  At that point the stored hashes
    remain useful reproducibility evidence, but are not a substitute for the
    successful physical validation that just occurred.
    """
    warnings = list(trajectory.provenance_warnings)
    if not _is_sha256(trajectory.resolved_config_sha256):
        warnings.append("recording lacks a valid resolved_config_sha256")
    elif trajectory.resolved_config_sha256 != provenance_sha256:
        warnings.append("recorded config hash differs from the replay config")
    recorded_models = dict(trajectory.model_provenance)
    missing_models = [
        name
        for name in _MODEL_PROVENANCE_KEYS
        if not _is_sha256(recorded_models.get(name))
    ]
    if missing_models:
        warnings.append(f"recorded model provenance is incomplete: {missing_models}")
    try:
        current_models = {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in zip(_MODEL_PROVENANCE_KEYS, preflight_model_paths())
        }
    except OSError as exc:
        warnings.append(
            f"could not audit current URDF/SRDF provenance: {type(exc).__name__}: {exc}"
        )
        return tuple(warnings)
    mismatched_models = [
        name
        for name in _MODEL_PROVENANCE_KEYS
        if _is_sha256(recorded_models.get(name))
        and recorded_models[name] != current_models[name]
    ]
    if mismatched_models:
        warnings.append(f"recorded model hashes differ: {mismatched_models}")
    return tuple(warnings)


def verify_replay_preflight(
    trajectory: TrajectoryData,
    runtime: ExperimentConfig,
    *,
    provenance_sha256: str,
) -> None:
    """Fail-closed validation immediately before spawning hardware workers.

    Checks: recorded first measured state, hand-data attestation, full arm/hand
    command hard limits, the recorded-state-to-first-command transition, and
    every adjacent command pair for workspace bounds and collision
    (self-collision plus static obstacle boxes). Robot-table contact is
    deliberately not a replay rejection condition (user_design.md §3): replayed
    episodes were recorded by teleop without table gating, and table clearance
    remains enforced on the return-home path. Called once before worker startup;
    any rejection prevents hardware access entirely.
    """
    require_hand_actions(trajectory)
    if not bool(runtime.policy.hand_enabled):
        raise ValueError("physical replay requires policy.hand_enabled=true")
    _verify_trajectory_input(trajectory)
    modeled_hand = modeled_hand_actions(trajectory)
    recorded_arm_start, recorded_hand_start = replay_start_state(trajectory)
    arm_lower = np.asarray(runtime.arm.joint_limit_lower, dtype=np.float64)
    arm_upper = np.asarray(runtime.arm.joint_limit_upper, dtype=np.float64)
    if np.any(recorded_arm_start < arm_lower) or np.any(recorded_arm_start > arm_upper):
        raise ValueError("physical replay first arm_qpos violates joint limits")
    arm_actions = _canonicalize_replay_arm_actions(trajectory, runtime)
    _validate_replay_hand_limits(modeled_hand, recorded_hand_start, runtime)
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
    first_arm_cmd = arm_actions[0]
    if not planner.is_workspace_segment_safe(recorded_arm_start, first_arm_cmd):
        raise ValueError("physical replay workspace rejection at recorded start->0")
    if not planner.collision_model.check_transition_collision_free(
        recorded_arm_start,
        first_arm_cmd,
        recorded_hand_start,
        modeled_hand[0],
    ):
        raise ValueError("physical replay collision rejection at recorded start->0")
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
    for warning in _reproducibility_warnings(
        trajectory,
        provenance_sha256=provenance_sha256,
    ):
        logger.warning(
            "physical replay passed current geometry preflight; reproducibility warning: %s",
            warning,
        )
