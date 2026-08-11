"""Fail-closed, in-process validation immediately before live replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from dexmani_real import ASSET_DIR
from dexmani_real.config.runtime import ResolvedRuntimeConfig
from dexmani_real.utils.schema import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE
from dexmani_real.planning import Pose, XArm7MotionPlanner, XArm7PlannerConfig
from dexmani_real.replay.data import TrajectoryData

_MODEL_PROVENANCE_KEYS = (
    "arm_hand_collision_urdf_sha256",
    "arm_hand_urdf_sha256",
    "arm_hand_srdf_sha256",
)
_PREFLIGHT_MAX_JOINT_STEP_RAD = 0.02


def replay_runtime_hash(
    canonical_config_yaml: str,
    *,
    source: str,
    speed_factor: float,
    no_hand: bool,
    jerk_management: str,
) -> str:
    """Hash the resolved runtime together with replay-only behavior."""
    speed = float(speed_factor)
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("replay speed factor must be finite and positive")
    if source not in {"cmd", "sent"}:
        raise ValueError(f"unsupported replay action source {source!r}")
    payload = {
        "resolved_config": yaml.safe_load(canonical_config_yaml),
        "replay": {
            "source": source,
            "speed_factor": speed,
            "no_hand": bool(no_hand),
            "jerk_management": jerk_management,
        },
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def preflight_model_paths() -> tuple[Path, ...]:
    model_dir = ASSET_DIR / "robots" / "xhand"
    return (
        model_dir / "xarm7_xhand_collision.urdf",
        model_dir / "xarm7_xhand_right.urdf",
        model_dir / "xarm7_xhand.srdf",
    )


def modeled_hand_actions(
    trajectory: TrajectoryData,
    *,
    no_hand: bool,
    home_qpos_rad: np.ndarray,
) -> np.ndarray:
    if not no_hand and trajectory.has_hand:
        assert trajectory.action_hand_joint is not None
        return np.asarray(trajectory.action_hand_joint, dtype=np.float64)
    home = np.asarray(home_qpos_rad, dtype=np.float64)
    if home.shape != HAND_JOINT_SHAPE or not np.all(np.isfinite(home)):
        raise ValueError(f"configured hand home must be finite shape {HAND_JOINT_SHAPE}")
    return np.repeat(home[None, :], trajectory.num_frames, axis=0)


def require_explicit_hand_mode(trajectory: TrajectoryData, *, no_hand: bool) -> None:
    """Fail closed when an episode cannot prove that live hand data was recorded."""
    if no_hand:
        return
    if trajectory.hand_available is not True:
        detail = "missing hand_available metadata" if trajectory.hand_available is None else "hand_available=false"
        raise ValueError(f"episode reports {detail}; pass --no-hand only after securing the physical hand")
    if not trajectory.has_hand_actions:
        raise ValueError("episode has no hand action stream; pass --no-hand only after securing the physical hand")


def _is_sha256(value: str | None) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _verify_trajectory_provenance(trajectory: TrajectoryData, runtime: ResolvedRuntimeConfig) -> str:
    if trajectory.action_source != "sent":
        raise ValueError("live replay requires the exact submitted action stream ('sent')")
    if trajectory.num_frames <= 0:
        raise ValueError("live replay trajectory is empty")
    arm_actions = np.asarray(trajectory.action_arm_joint)
    expected_arm_shape = (trajectory.num_frames, *ARM_JOINT_SHAPE)
    if arm_actions.shape != expected_arm_shape or not np.all(np.isfinite(arm_actions)):
        raise ValueError(f"live replay arm actions must be finite shape {expected_arm_shape}")
    if not _is_sha256(trajectory.resolved_config_sha256):
        raise ValueError("live replay recording provenance lacks a valid resolved_config_sha256")

    required_controller = {
        "joint_max_acc": trajectory.joint_max_acc,
        "joint_max_speed": trajectory.joint_max_speed,
        "arm_loop_hz": trajectory.arm_loop_hz,
    }
    missing = [name for name, value in required_controller.items() if value is None]
    if trajectory.jerk_management is None:
        missing.append("jerk_management")
    if missing:
        raise ValueError(f"live replay controller provenance is incomplete: {missing}")

    recorded = {name: float(value) for name, value in required_controller.items() if value is not None}
    expected = {
        "joint_max_acc": float(runtime.arm.max_joint_acceleration_deg_per_s2),
        "joint_max_speed": float(runtime.arm.max_joint_velocity_deg_per_s),
        "arm_loop_hz": float(runtime.arm.loop_hz),
    }
    mismatches = [
        name for name, value in recorded.items() if not np.isfinite(value) or not np.isclose(value, expected[name])
    ]
    if mismatches:
        raise ValueError(f"live replay controller provenance mismatch: {mismatches}")
    if trajectory.jerk_management != "unmanaged":
        raise ValueError(f"unsupported recorded jerk management {trajectory.jerk_management!r}")

    recorded_models = dict(trajectory.model_provenance)
    missing_models = [name for name in _MODEL_PROVENANCE_KEYS if not _is_sha256(recorded_models.get(name))]
    if missing_models:
        raise ValueError(f"live replay model provenance is incomplete: {missing_models}")
    current_models = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in zip(_MODEL_PROVENANCE_KEYS, preflight_model_paths())
    }
    mismatched_models = [name for name in _MODEL_PROVENANCE_KEYS if recorded_models[name] != current_models[name]]
    if mismatched_models:
        raise ValueError(f"live replay model provenance mismatch: {mismatched_models}")
    return trajectory.action_source


def verify_live_replay_preflight(
    trajectory: TrajectoryData,
    runtime: ResolvedRuntimeConfig,
    *,
    no_hand: bool,
    speed_factor: float,
) -> None:
    """Validate provenance and every modeled transition before workers start."""
    require_explicit_hand_mode(trajectory, no_hand=no_hand)
    if not no_hand and not bool(runtime.policy.hand_enabled):
        raise ValueError("runtime policy.hand_enabled=false requires explicit no-hand acknowledgement")
    action_source = _verify_trajectory_provenance(trajectory, runtime)
    replay_runtime_hash(
        runtime.canonical_yaml,
        source=action_source,
        speed_factor=speed_factor,
        no_hand=no_hand,
        jerk_management="unmanaged",
    )
    modeled_hand = modeled_hand_actions(
        trajectory,
        no_hand=no_hand,
        home_qpos_rad=np.deg2rad(np.asarray(runtime.hand.home_qpos_deg, dtype=np.float64)),
    )
    workspace = np.array(
        [
            [runtime.policy.workspace.x_min, runtime.policy.workspace.x_max],
            [runtime.policy.workspace.y_min, runtime.policy.workspace.y_max],
            [runtime.policy.workspace.z_min, runtime.policy.workspace.z_max],
        ],
        dtype=np.float64,
    )
    model_dir = ASSET_DIR / "robots" / "xhand"
    planner = XArm7MotionPlanner(
        XArm7PlannerConfig(
            urdf_path=str(model_dir / "xarm7_xhand_collision.urdf"),
            srdf_path=str(model_dir / "xarm7_xhand.srdf"),
            base_pose_world=Pose(p=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0])),
            workspace_bounds=workspace,
        ),
        hand_dof=True,
        static_boxes=tuple(runtime.environment.static_boxes),
    )
    arm_actions = np.asarray(trajectory.action_arm_joint, dtype=np.float64)
    for index in range(max(1, trajectory.num_frames - 1)):
        start = min(index, trajectory.num_frames - 1)
        end = min(index + 1, trajectory.num_frames - 1)
        arm_start, arm_end = arm_actions[start], arm_actions[end]
        hand_start, hand_end = modeled_hand[start], modeled_hand[end]
        if not planner.is_workspace_segment_safe(arm_start, arm_end):
            raise ValueError(f"live replay workspace rejection at transition {start}->{end}")
        if not planner.collision_model.check_transition_collision_free(arm_start, arm_end, hand_start, hand_end):
            raise ValueError(f"live replay collision rejection at transition {start}->{end}")
        max_delta = max(
            float(np.max(np.abs(arm_end - arm_start))),
            float(np.max(np.abs(hand_end - hand_start))),
        )
        steps = max(1, int(np.ceil(max_delta / _PREFLIGHT_MAX_JOINT_STEP_RAD)))
        for alpha in np.linspace(0.0, 1.0, steps + 1):
            arm_qpos = arm_start + alpha * (arm_end - arm_start)
            hand_qpos = hand_start + alpha * (hand_end - hand_start)
            planner.collision_model.set_hand_qpos(hand_qpos)
            lowest_surface_m = planner.collision_model.minimum_hand_frame_z(arm_qpos) - float(
                runtime.arm.hand_safety_margin_m
            )
            if not np.isfinite(lowest_surface_m) or lowest_surface_m < float(runtime.arm.table_z_surface_m):
                raise ValueError(f"live replay table rejection at transition {start}->{end}")
