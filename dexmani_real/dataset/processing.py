"""Dense Raw layout checks and full-modal numerical transforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dexmani_real.dataset.contracts import ProcessingConfig
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
from dexmani_real.recording.storage.reader import EpisodeReader, RawDataError
from dexmani_real.recording.storage.schema import DATASET_SPECS, validate_capture_rows
from dexmani_real.robot.model import XHAND_RIGHT_URDF_PATH


def validate_export_episode(reader: EpisodeReader) -> int:
    """Require complete physical modalities and camera geometry for the whole episode."""
    frames = reader.num_frames
    if frames <= 0:
        raise RawDataError("episode contains no rows")
    reader.require_fields(*DATASET_SPECS, "rgb", "depth")
    reader.require_metadata(
        "camera_payload_mode", "camera_color_height", "camera_color_width", "depth_scale",
        "camera_color_intrinsics", "camera_color_distortion_model",
        "camera_color_distortion_coeffs", "camera_T_xarm_base_from_color",
    )
    scale = float(reader.meta["depth_scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise RawDataError("RGB-D export requires a known positive depth_scale")
    expected = (
        frames,
        int(reader.meta["camera_color_height"]),
        int(reader.meta["camera_color_width"]),
    )
    if reader["depth"].shape != expected or reader["depth"].dtype != np.uint16:
        raise RawDataError(f"depth must be uint16 {expected}")
    for name, spec in DATASET_SPECS.items():
        array = reader[name]
        if np.issubdtype(spec.dtype, np.floating):
            for start in range(0, frames, 256):
                values = array[start:start + 256]
                valid = np.isfinite(values).reshape(len(values), -1).all(axis=1)
                if not valid.all():
                    row = start + int(np.flatnonzero(~valid)[0])
                    raise RawDataError(f"Raw {name}: nonfinite values at row {row}")
    errors = validate_capture_rows(reader["timestamp"][:], reader["dispatch_status"][:])
    if errors:
        raise RawDataError("; ".join(errors))
    try:
        load_raw_episode_camera_model(reader)
        load_raw_episode_base_from_color(reader)
    except (ValueError, TypeError, KeyError) as exc:
        raise RawDataError(f"invalid camera geometry: {exc}") from exc
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


def _iter_rgb_frames(reader):
    try:
        yield from reader.iter_camera_frames("rgb")
    except (ValueError, IndexError, EOFError) as exc:
        raise RawDataError(f"{reader.path / 'rgb.mp4'}: invalid RGB video: {exc}") from exc


def iter_canonical_blocks(
    reader: EpisodeReader, config: ProcessingConfig, *, chunk_frames: int
):
    """Convert a prevalidated episode without repairing or resampling rows."""
    pointcloud_deriver = RawEpisodePointCloudDeriver(
        reader=reader,
        camera=load_raw_episode_camera_model(reader),
        T_xarm_base_from_color=load_raw_episode_base_from_color(reader),
        pointcloud=config.pointcloud,
        table_plane_abcd=config.table_plane_abcd,
    )
    height, width = int(reader.meta["camera_color_height"]), int(reader.meta["camera_color_width"])
    hand_fk = HandKinematics(str(XHAND_RIGHT_URDF_PATH), list(config.fingertip_link_names))
    images = _iter_rgb_frames(reader)
    for start in range(0, reader.num_frames, chunk_frames):
        end = min(reader.num_frames, start + chunk_frames)
        arm = np.asarray(reader["arm_qpos"][start:end], dtype=np.float64)
        hand = np.asarray(reader["hand_qpos"][start:end], dtype=np.float64)
        arm_action = np.asarray(reader["action_arm_joint_target"][start:end], dtype=np.float64)
        hand_action = np.asarray(reader["action_hand_joint_target"][start:end], dtype=np.float64)
        poses = compute_eef_pose_history_xarm_base(arm)
        action_poses = compute_eef_pose_history_xarm_base(arm_action)
        tips = compute_fingertip_history_xarm_base(
            arm, hand, hand_fk=hand_fk,
            handbase_position_eef_m=np.asarray(config.handbase_position_eef_m),
            handbase_quat_eef_wxyz=np.asarray(config.handbase_quat_eef_wxyz),
            eef_pose_history=poses,
        )
        block = {
            "joint_state": np.concatenate((arm, hand), axis=1).astype(np.float32),
            "action": np.concatenate((arm_action, hand_action), axis=1).astype(np.float32),
            "action_ee": np.concatenate((action_poses, hand_action), axis=1).astype(np.float32),
            "eef_pose": poses.astype(np.float32),
            "fingertip_points": tips.astype(np.float32),
            "contact_force": np.asarray(reader["hand_contact"][start:end], dtype=np.float32),
            "tactile_force": np.asarray(reader["hand_tactile_force"][start:end], dtype=np.float32),
        }
        for name in ("arm_qvel", "arm_effort", "hand_current"):
            block[name] = np.asarray(reader[name][start:end], dtype=np.float32)
        rgb, depth, clouds = [], [], []
        for index in range(start, end):
            image = next(images, None)
            if image is None or image.shape != (height, width, 3) or image.dtype != np.uint8:
                raise RawDataError(
                    f"{reader.path / 'rgb.mp4'}: RGB missing or invalid at row {index}"
                )
            cloud = pointcloud_deriver.derive(index, image)
            if cloud is None:
                raise RawDataError(f"point_cloud unavailable at row {index}")
            rgb.append(image)
            depth.append(np.asarray(reader["depth"][index], dtype=np.uint16))
            clouds.append(cloud)
        block.update(rgb=np.stack(rgb), depth=np.stack(depth), point_cloud=np.stack(clouds))
        yield block
    if next(images, None) is not None:
        raise RawDataError(f"{reader.path / 'rgb.mp4'}: RGB contains extra frames")
