"""Load one recorded episode into replay-ready NumPy arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dexmani_real.utils.schema import ARM_EE_SHAPE, ARM_JOINT_SHAPE, HAND_JOINT_SHAPE
from dexmani_real.recording.episode_reader import EpisodeReader
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_MIN_EPISODE_RATE_HZ = 1.0
_MAX_EPISODE_RATE_HZ = 100.0
_OFFLINE_FALLBACK_RATE_HZ = 16.0


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
    hand_available: bool | None = None
    joint_max_acc: float | None = None
    joint_max_speed: float | None = None
    arm_loop_hz: float | None = None
    jerk_management: str | None = None
    resolved_config_sha256: str | None = None
    model_provenance: tuple[tuple[str, str], ...] = ()

    @property
    def has_hand(self) -> bool:
        """Whether the episode certifies that hand hardware produced the stream."""
        return self.hand_available is True and self.action_hand_joint is not None

    @property
    def has_hand_actions(self) -> bool:
        """Whether a fixed-shape hand action dataset is present."""
        return self.action_hand_joint is not None


def resolve_episode_path(raw_path: str) -> tuple[str, str]:
    """Resolve an episode directory, its data.h5, or a legacy flat HDF5 file."""
    path = Path(raw_path)
    if path.is_file() and path.name == "data.h5":
        parent = path.parent
        if (parent / "depth.h5").exists() or (parent / "rgb.mp4").exists():
            return str(parent), parent.name
    if path.is_dir():
        return str(path), path.name
    return str(path), path.stem


def _optional_float(meta: Any, name: str) -> float | None:
    raw = meta.attrs.get(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _optional_bool(meta: Any, name: str) -> bool | None:
    raw = meta.attrs.get(name)
    if isinstance(raw, (bool, np.bool_)):
        return bool(raw)
    if isinstance(raw, (int, np.integer)) and int(raw) in (0, 1):
        return bool(raw)
    return None


def load_trajectory(
    episode_path: str,
    max_frames: int | None = None,
    source: str = "cmd",
    *,
    require_live_validity: bool = False,
    require_exact_source: bool = False,
) -> TrajectoryData:
    """Load one command trajectory while keeping older episodes offline-readable."""
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")
    if source not in {"cmd", "sent"}:
        raise ValueError("source must be 'cmd' or 'sent'")

    resolved_path, _episode_name = resolve_episode_path(episode_path)
    if not Path(resolved_path).exists():
        raise FileNotFoundError(f"Episode not found: {episode_path}")

    with EpisodeReader(resolved_path) as reader:
        if require_live_validity:
            reader.require_valid(purpose="live replay")
        h5 = reader.h5f
        meta = h5.get("meta")
        schema = meta.attrs.get("schema_version") if meta is not None else None
        try:
            schema_version = None if schema is None else int(schema)
        except (TypeError, ValueError):
            schema_version = None
        if schema_version is not None and schema_version < 3:
            logger.warning("HDF5 schema v%d < 3; optional replay fields may be absent", schema_version)

        num_frames_attr = int(meta.attrs.get("num_frames", 0)) if meta is not None else 0
        fps = float(reader.timing.rate_hz)
        if not _MIN_EPISODE_RATE_HZ <= fps <= _MAX_EPISODE_RATE_HZ:
            if require_live_validity:
                raise ValueError(f"live replay requires a valid episode rate, got {fps!r} Hz")
            logger.warning(
                "Implausible episode rate %.3f Hz; using %.1f Hz for offline replay",
                fps,
                _OFFLINE_FALLBACK_RATE_HZ,
            )
            fps = _OFFLINE_FALLBACK_RATE_HZ
        task_label = str(meta.attrs.get("task_label", "")) if meta is not None else ""
        hand_available = _optional_bool(meta, "hand_available") if meta is not None else None

        joint_max_acc = _optional_float(meta, "joint_max_acc") if meta is not None else None
        joint_max_speed = _optional_float(meta, "joint_max_speed") if meta is not None else None
        arm_loop_hz = _optional_float(meta, "arm_loop_hz") if meta is not None else None
        jerk_management = None
        resolved_config_sha256 = None
        model_provenance: tuple[tuple[str, str], ...] = ()
        if meta is not None:
            raw_jerk = meta.attrs.get("jerk_management")
            jerk_management = None if raw_jerk is None else str(raw_jerk)
            raw_hash = meta.attrs.get("resolved_config_sha256")
            resolved_config_sha256 = None if raw_hash is None else str(raw_hash)
            model_provenance = tuple(
                sorted(
                    (name.removeprefix("provenance_"), str(meta.attrs[name]))
                    for name in meta.attrs
                    if name.startswith("provenance_arm_hand_")
                )
            )

        arm_action_key = "action_arm_joint_sent" if source == "sent" else "action_arm_joint"
        action_source = source
        if arm_action_key not in h5 and source == "sent":
            if require_live_validity or require_exact_source:
                raise ValueError("the requested replay operation requires /action_arm_joint_sent")
            logger.warning("/action_arm_joint_sent is absent; using /action_arm_joint for offline replay")
            arm_action_key = "action_arm_joint"
            action_source = "cmd"
        for key in (arm_action_key, "arm_qpos"):
            if key not in h5:
                raise ValueError(f"episode missing required dataset: /{key}")

        source_frames = int(h5[arm_action_key].shape[0])
        total_frames = source_frames if num_frames_attr == 0 else min(source_frames, num_frames_attr)
        if max_frames is not None:
            total_frames = min(total_frames, max_frames)

        action_arm_joint = np.asarray(h5[arm_action_key][:total_frames], dtype=np.float64)
        arm_qpos = np.asarray(h5["arm_qpos"][:total_frames], dtype=np.float64)
        action_hand_joint = (
            np.asarray(h5["action_hand_joint"][:total_frames], dtype=np.float64) if "action_hand_joint" in h5 else None
        )
        hand_qpos = np.asarray(h5["hand_qpos"][:total_frames], dtype=np.float64) if "hand_qpos" in h5 else None
        arm_ee = np.asarray(h5["arm_ee"][:total_frames], dtype=np.float64) if "arm_ee" in h5 else None

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
    for name, (array, expected_shape) in arrays.items():
        if array.shape != expected_shape:
            raise ValueError(f"episode {name} has shape {array.shape}, expected {expected_shape}")

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
        hand_available=hand_available,
        joint_max_acc=joint_max_acc,
        joint_max_speed=joint_max_speed,
        arm_loop_hz=arm_loop_hz,
        jerk_management=jerk_management,
        resolved_config_sha256=resolved_config_sha256,
        model_provenance=model_provenance,
    )
    logger.info(
        "Loaded trajectory: %d frames, fps=%.1f, task=%s, hand=%s, ee=%s",
        trajectory.num_frames,
        trajectory.fps,
        trajectory.task_label or "(none)",
        ("yes" if trajectory.has_hand else "dataset-only" if trajectory.has_hand_actions else "no"),
        "yes" if trajectory.arm_ee is not None else "no",
    )
    return trajectory
