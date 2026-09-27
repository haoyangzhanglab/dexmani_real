"""Whole-episode Raw admission and canonical numerical transforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dexmani_real.dataset.contracts import (
    OPTIONAL_TELEMETRY,
    ProcessingConfig,
    validate_task_name,
)
from dexmani_real.dataset.pointcloud import (
    RawEpisodePointCloudDeriver,
    load_raw_episode_base_from_color,
    load_raw_episode_camera_model,
)
from dexmani_real.planning.kinematics.arm_fk import (
    compute_eef_pose_history_xarm_base,
)
from dexmani_real.planning.kinematics.fingertip import compute_fingertip_history_xarm_base
from dexmani_real.planning.kinematics.hand_fk import HandKinematics
from dexmani_real.planning.kinematics.pose import validate_canonical_rot6d
from dexmani_real.recording.storage.reader import EpisodeReader
from dexmani_real.recording.storage.schema import DATASET_SPECS
from dexmani_real.robot.model import XHAND_RIGHT_URDF_PATH


def validate_episode(reader: EpisodeReader) -> int:
    """Admit every row of a complete teleop episode, or reject the whole episode."""
    source = reader
    meta = reader.meta
    frames = reader.num_frames
    if frames <= 0:
        raise ValueError("episode contains no rows")
    task_label = meta["task_label"]
    if isinstance(task_label, bytes):
        task_label = task_label.decode("utf-8")
    validate_task_name(task_label)
    collection_source = meta["collection_source"]
    if isinstance(collection_source, bytes):
        collection_source = collection_source.decode("utf-8")
    if collection_source != "teleop":
        raise ValueError(f"training requires teleop collection_source, got {collection_source!r}")
    reader.require_fields(*DATASET_SPECS, "rgb", "depth")
    reader.require_metadata("handbase_position_eef_m", "handbase_quat_eef_wxyz")
    if reader.is_legacy:
        reader.require_fields("frame_valid")
        if not reader.legacy_episode_valid or not np.all(reader["frame_valid"][:]):
            raise ValueError("legacy episode has invalid episode/frame evidence")
    for name in ("arm_qpos", "hand_qpos", "action_arm_joint_target", "action_hand_joint_target"):
        if not np.isfinite(source[name][:]).all():
            raise ValueError(f"{name}: non-finite core values")
    return frames


def discover_episode_dirs(input_root: str | Path) -> tuple[Path, ...]:
    """Return absolute paths for one episode or a task's episode_* directories.

    Task discovery includes corrupt episode_* directories so downstream checks
    reject them explicitly; it does not filter by data.h5 existence.
    A root containing data.h5 is treated as a single episode.
    """
    root = Path(input_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if (root / "data.h5").is_file():
        return (root,)
    episodes = tuple(
        sorted(
            child
            for child in root.iterdir()
            if child.is_dir() and child.name.startswith("episode_")
        )
    )
    if not episodes:
        raise FileNotFoundError(f"no episode directories found in {root}")
    return episodes


def iter_canonical_blocks(reader: EpisodeReader, config: ProcessingConfig, *, chunk_frames: int):
    """One episode of numeric state; bounded RGB-D/cloud chunks, never a task in RAM."""
    frames = reader.num_frames
    camera_model = load_raw_episode_camera_model(reader)
    geometry = camera_model.geometry
    transform = load_raw_episode_base_from_color(reader)
    arm_action = np.asarray(reader["action_arm_joint_target"][:], dtype=np.float64)
    hand_action = np.asarray(reader["action_hand_joint_target"][:], dtype=np.float64)
    arm_qpos = np.asarray(reader["arm_qpos"][:], dtype=np.float64)
    hand_qpos = np.asarray(reader["hand_qpos"][:], dtype=np.float64)
    action_ee_pose = compute_eef_pose_history_xarm_base(arm_action)
    eef_pose = compute_eef_pose_history_xarm_base(arm_qpos)
    joint_state = np.concatenate((arm_qpos, hand_qpos), axis=1)
    hand_fk = HandKinematics(str(XHAND_RIGHT_URDF_PATH), list(config.fingertip_link_names))
    if not hand_fk.is_ready():
        raise RuntimeError("canonical fingertip FK startup failed")
    mount = reader.meta
    fingertip_points = compute_fingertip_history_xarm_base(
        arm_qpos,
        hand_qpos,
        hand_fk=hand_fk,
        handbase_position_eef_m=np.asarray(mount["handbase_position_eef_m"], dtype=np.float64),
        handbase_quat_eef_wxyz=np.asarray(mount["handbase_quat_eef_wxyz"], dtype=np.float64),
        eef_pose_history=eef_pose,
    )
    values = {
        "joint_state": joint_state.astype(np.float32),
        "action": np.concatenate((arm_action, hand_action), axis=1).astype(np.float32),
        "action_ee": np.concatenate((action_ee_pose, hand_action), axis=1).astype(np.float32),
        "eef_pose": eef_pose.astype(np.float32),
        "fingertip_points": fingertip_points.astype(np.float32),
        "contact_force": np.asarray(reader["hand_contact"][:], dtype=np.float32),
        "tactile_force": np.asarray(reader["hand_tactile_force"][:], dtype=np.float32),
    }
    for name in ("arm_qvel", "arm_effort", "hand_current"):
        values[name] = np.asarray(reader[name][:], dtype=np.float32)
    pointcloud_deriver = RawEpisodePointCloudDeriver(
        reader=reader,
        camera=camera_model,
        T_xarm_base_from_color=transform,
        pointcloud=config.pointcloud,
        table_plane_abcd=config.table_plane_abcd,
    )
    images = iter(reader.iter_camera_frames("rgb"))
    for start in range(0, frames, chunk_frames):
        end = min(frames, start + chunk_frames)
        block = {key: value[start:end] for key, value in values.items()}
        rgb, depth, clouds = [], [], []
        for index in range(start, end):
            image = next(images, None)
            if (
                image is None
                or image.shape != (geometry.color.height, geometry.color.width, 3)
                or image.dtype != np.uint8
            ):
                raise ValueError(f"RGB missing or invalid at row {index}")
            cloud = pointcloud_deriver.derive(index, image)
            if cloud is None:
                raise ValueError(f"derived point cloud empty at source row {index}")
            rgb.append(image)
            depth.append(np.asarray(reader["depth"][index], dtype=np.uint16))
            clouds.append(cloud)
        block.update(
            rgb=np.stack(rgb),
            depth=np.stack(depth),
            point_cloud=np.stack(clouds),
        )
        for key, value in block.items():
            if np.issubdtype(value.dtype, np.floating):
                invalid = (
                    np.isinf(value).any()
                    if key in OPTIONAL_TELEMETRY
                    else not np.isfinite(value).all()
                )
                if invalid:
                    raise ValueError(f"{key}: unsupported non-finite transformed values")
        validate_canonical_rot6d(block["action_ee"][:, 3:9], label="action_ee")
        validate_canonical_rot6d(block["eef_pose"][:, 3:9], label="eef_pose")
        yield block
    if next(images, None) is not None:
        raise ValueError("RGB contains extra frames")
