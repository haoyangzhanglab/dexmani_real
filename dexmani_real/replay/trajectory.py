"""Load raw arm and logical hand targets and validate physical replay trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.planning import Pose, XArm7MotionPlanner, XArm7PlannerConfig
from dexmani_real.planning.kinematics.arm_fk import compute_eef_pose_history_xarm_base
from dexmani_real.planning.paths import wrap_nearest_equivalent
from dexmani_real.recording.storage.reader import EpisodeReader
from dexmani_real.recording.storage.schema import command_send_mask
from dexmani_real.dataset.provenance import (
    POLICY_EVAL_WORKFLOW,
    read_provenance_workflow,
    supports_fixed_dt_teleop,
)
from dexmani_real.robot.model import (
    ARM_JOINT_SHAPE,
    HAND_JOINT_SHAPE,
    XARM7_XHAND_COLLISION_URDF_PATH,
    XARM7_XHAND_SRDF_PATH,
)
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_MIN_EPISODE_RATE_HZ = 1.0
_MAX_EPISODE_RATE_HZ = 100.0
_JOINT_LIMIT_TOLERANCE_RAD = 1e-12


@dataclass
class TrajectoryData:
    """Required raw-v31 xArm7/XHand streams and derived EEF history for replay."""

    episode_path: str
    num_frames: int
    fps: float
    task_label: str
    action_arm_joint: np.ndarray
    action_hand_joint: np.ndarray
    arm_qpos: np.ndarray
    hand_qpos: np.ndarray
    arm_ee: np.ndarray
    send_mask: np.ndarray
    timestamp_offsets_s: np.ndarray
    arm_present: np.ndarray
    hand_present: np.ndarray


def resolve_episode_path(raw_path: str) -> tuple[str, str]:
    """Validate and name one published episode directory."""
    path = Path(raw_path)
    if not path.is_dir():
        raise ValueError(f"episode must be a published directory: {path}")
    if not (path / "data.h5").is_file():
        raise ValueError(f"episode directory is missing data.h5: {path}")
    return str(path), path.name


def load_trajectory(episode_path: str) -> TrajectoryData:
    """Load recorded published arm targets and logical hand targets for replay."""
    resolved_path, _episode_name = resolve_episode_path(episode_path)

    with EpisodeReader(resolved_path) as reader:
        reader.require_valid(purpose="physical replay")
        if not reader.min_frames_met:
            logger.warning(
                "Episode %s is internally readable but below the configured minimum recording duration",
                resolved_path,
            )
        h5 = reader.h5f
        meta = h5["meta"]
        workflow = read_provenance_workflow(meta.attrs)
        if not supports_fixed_dt_teleop(workflow):
            if workflow == POLICY_EVAL_WORKFLOW:
                raise ValueError(
                    "policy_eval rollout contains synchronous irregular timing; "
                    "current physical replay only admits teleop episodes; it must not silently "
                    "time-compress it"
                )
            raise ValueError(
                f"unsupported provenance_workflow {workflow!r} for physical replay"
            )
        total_frames = int(meta.attrs["num_frames"])
        fps = float(reader.timing.rate_hz)
        if not _MIN_EPISODE_RATE_HZ <= fps <= _MAX_EPISODE_RATE_HZ:
            raise ValueError(
                f"physical replay requires a valid episode rate, got {fps!r} Hz"
            )
        task_label = str(meta.attrs.get("task_label", ""))

        # EpisodeReader validates every v31 dataset's shape, dtype and frame count.
        action_arm_joint = np.asarray(h5["action_arm_joint_sent"][:], dtype=np.float64)
        arm_qpos = np.asarray(h5["arm_qpos"][:], dtype=np.float64)
        action_hand_joint = np.asarray(h5["action_hand_joint"][:], dtype=np.float64)
        hand_qpos = np.asarray(h5["hand_qpos"][:], dtype=np.float64)
        arm_ee = compute_eef_pose_history_xarm_base(arm_qpos)
        send_mask = command_send_mask(h5)
        anchors = np.asarray(h5["observation_anchor_monotonic_ns"][:], dtype=np.uint64)
        if len(anchors) == 0 or np.any(anchors == 0) or np.any(anchors[1:] <= anchors[:-1]):
            raise ValueError("replay anchors must be strictly increasing")
        offsets = (anchors - anchors[0]).astype(np.float64) / 1e9
        arm_present = np.asarray(h5["command_arm_present"][:], dtype=bool)
        hand_present = np.asarray(h5["command_hand_present"][:], dtype=bool)

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
        send_mask=send_mask,
        timestamp_offsets_s=offsets,
        arm_present=arm_present, hand_present=hand_present,
    )
    logger.info(
        "Loaded xArm7/XHand trajectory: %d frames, fps=%.1f, task=%s",
        trajectory.num_frames,
        trajectory.fps,
        trajectory.task_label or "(none)",
    )
    return trajectory


def modeled_hand_actions(trajectory: TrajectoryData) -> np.ndarray:
    """Return recorded logical hand targets used for geometry preflight."""
    actions = np.asarray(trajectory.action_hand_joint, dtype=np.float64)
    expected_shape = (trajectory.num_frames, *HAND_JOINT_SHAPE)
    if actions.shape != expected_shape:
        raise ValueError(f"physical replay hand actions require shape {expected_shape}")
    modeled = np.empty_like(actions)
    reference = np.asarray(trajectory.hand_qpos[0], dtype=np.float64)
    for i, action in enumerate(actions):
        if trajectory.send_mask[i] and trajectory.hand_present[i]:
            if not np.all(np.isfinite(action)):
                raise ValueError(f"non-finite hand target at frame {i}")
            reference = action
        modeled[i] = reference
    return modeled


def replay_start_state(trajectory: TrajectoryData) -> tuple[np.ndarray, np.ndarray]:
    """Return the finite measured arm/hand state at the first replay frame."""
    if trajectory.num_frames <= 0:
        raise ValueError("physical replay trajectory is empty")
    arm_qpos = np.asarray(trajectory.arm_qpos[0], dtype=np.float64)
    if arm_qpos.shape != ARM_JOINT_SHAPE or not np.all(np.isfinite(arm_qpos)):
        raise ValueError("physical replay requires a finite first arm_qpos state")
    hand_qpos = np.asarray(trajectory.hand_qpos[0], dtype=np.float64)
    if hand_qpos.shape != HAND_JOINT_SHAPE or not np.all(np.isfinite(hand_qpos)):
        raise ValueError("physical replay requires a finite first hand_qpos state")
    return arm_qpos.copy(), hand_qpos.copy()


def _verify_trajectory_input(trajectory: TrajectoryData) -> None:
    """Fail closed on the exact source stream needed for physical preflight."""
    if trajectory.num_frames <= 0:
        raise ValueError("physical replay trajectory is empty")
    arm_actions = np.asarray(trajectory.action_arm_joint)
    expected_arm_shape = (trajectory.num_frames, *ARM_JOINT_SHAPE)
    if arm_actions.shape != expected_arm_shape or not np.all(np.isfinite(
            arm_actions[trajectory.send_mask & trajectory.arm_present])):
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
        if not (trajectory.send_mask[frame_index] and trajectory.arm_present[frame_index]):
            canonical_actions[frame_index] = reference
            continue
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


def verify_replay_preflight(
    trajectory: TrajectoryData,
    runtime: ExperimentConfig,
) -> None:
    """Fail-closed validation immediately before spawning hardware workers.

    Checks: recorded first measured state, full arm/hand
    command hard limits, the recorded-state-to-first-command transition, and
    every adjacent command pair for workspace bounds and collision
    (self-collision plus static obstacle boxes). Robot-table contact is
    deliberately not a replay rejection condition because replayed episodes were
    recorded under the same teleop table-contact semantics; table clearance remains
    enforced on the return-home path. Called once before worker startup; any
    rejection prevents hardware access entirely.
    """
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
    _validate_replay_hand_limits(
        modeled_hand[trajectory.send_mask & trajectory.hand_present], recorded_hand_start, runtime
    )
    workspace = runtime.policy.workspace.as_array()
    planner = XArm7MotionPlanner(
        XArm7PlannerConfig(
            urdf_path=str(XARM7_XHAND_COLLISION_URDF_PATH),
            srdf_path=str(XARM7_XHAND_SRDF_PATH),
            base_pose_world=Pose(p=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0])),
            workspace_bounds=workspace,
        ),
        hand_dof=True,
        static_boxes=tuple(runtime.environment.static_boxes),
        # Replay intentionally mirrors teleop's table-contact semantics here.
        # Table clearance stays enforced on the return-home path, which uses the
        # replayer's own planner (see EpisodeReplayer.setup()).
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
