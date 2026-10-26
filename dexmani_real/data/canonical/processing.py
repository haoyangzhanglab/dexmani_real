"""Dense Raw layout checks and full-modal numerical transforms."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from dexmani_real.data.canonical.schema import ProcessingConfig
from dexmani_real.data.raw.camera_geometry import (
    RawEpisodeCameraModel,
    load_raw_episode_base_from_color,
    load_raw_episode_camera_model,
)
from dexmani_real.data.raw.reader import EpisodeReader, RawDataError
from dexmani_real.data.raw.schema import DATASET_SPECS
from dexmani_real.geometry.camera import (
    camera_parameters_cv,
    validate_aligned_depth_distortion,
)
from dexmani_real.geometry.kinematics import eef_pose_sequence
from dexmani_real.perception.hand import hand_geometry_sequence
from dexmani_real.perception.pointcloud import build_point_cloud


def validate_export_episode(
    reader: EpisodeReader,
) -> tuple[RawEpisodeCameraModel, np.ndarray, dict[str, np.ndarray]]:
    """Check metadata/layout/timing and prepare static camera geometry once per episode."""
    frames = reader.num_frames
    if frames <= 0:
        raise RawDataError("episode contains no rows")
    reader.require_fields(*DATASET_SPECS, "rgb", "depth")
    try:
        camera = load_raw_episode_camera_model(reader)
        validate_aligned_depth_distortion(camera.intrinsics.distortion_model)
        base_from_color = load_raw_episode_base_from_color(reader)
        camera_parameters = camera_parameters_cv(camera.intrinsics, base_from_color)
    except (ValueError, TypeError, KeyError) as exc:
        raise RawDataError(f"invalid camera geometry: {exc}") from exc
    color = camera.intrinsics
    expected = (frames, color.height, color.width)
    if reader["depth"].shape != expected or reader["depth"].dtype != np.uint16:
        raise RawDataError(f"depth must be uint16 {expected}")
    reader.validate_completion()
    reader.validate_rows()
    if frames == 1:
        logging.getLogger(__name__).warning(
            "%s: single row, no sampling interval to verify", reader.path
        )
    for name in DATASET_SPECS:
        reader[name]  # EpisodeReader validates shape/dtype without reading the array.
    return camera, base_from_color, camera_parameters


def discover_episode_dirs(input_root: str | Path) -> tuple[Path, ...]:
    """Discover episodes, including corrupt episode_* directories for explicit rejection."""
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


def iter_canonical_blocks(
    reader: EpisodeReader,
    config: ProcessingConfig,
    *,
    arm_fk,
    hand_fk,
    hand_cloud,
    chunk_frames: int,
    camera: RawEpisodeCameraModel,
    base_from_color: np.ndarray,
    camera_parameters: dict[str, np.ndarray],
    handbase_rotation_eef: np.ndarray,
):
    """Convert admitted Raw blocks with export-owned FK, without repairing or resampling rows."""
    height, width = camera.intrinsics.height, camera.intrinsics.width
    images = reader.iter_rgb_frames()
    for start in range(0, reader.num_frames, chunk_frames):
        end = min(reader.num_frames, start + chunk_frames)
        raw = {
            name: reader[name][start:end]
            for name in DATASET_SPECS
            if name != "timestamp"
        }
        # Reader 已核对关节数组布局；触觉原值在 Canonical 写入边界检查。
        for name in ("arm_qpos", "hand_qpos", "action_arm_joint_target", "action_hand_joint_target"):
            valid = np.isfinite(raw[name]).all(axis=1)
            if not valid.all():
                row = start + int(np.flatnonzero(~valid)[0])
                raise RawDataError(f"Raw {name}: nonfinite values at row {row}")
        arm, hand = raw["arm_qpos"], raw["hand_qpos"]
        arm_action, hand_action = (
            raw["action_arm_joint_target"],
            raw["action_hand_joint_target"],
        )
        poses = eef_pose_sequence(arm, arm_fk=arm_fk)
        action_poses = eef_pose_sequence(arm_action, arm_fk=arm_fk)
        hand_geometry = hand_geometry_sequence(
            poses,
            hand,
            hand_fk=hand_fk,
            handbase_position_eef_m=config.handbase_position_eef_m,
            handbase_rotation_eef=handbase_rotation_eef,
            hand_cloud=hand_cloud,
        )
        block = {
            "joint_state": np.concatenate((arm, hand), axis=1).astype(np.float32),
            "action": np.concatenate((arm_action, hand_action), axis=1).astype(np.float32),
            "action_ee": np.concatenate((action_poses, hand_action), axis=1).astype(np.float32),
            "eef_pose": poses.astype(np.float32),
            "finger_tactile_force": raw["hand_contact"],
            "taxel_tactile_force": raw["hand_tactile_force"],
            **hand_geometry,
        }
        for name, values in camera_parameters.items():
            block[name] = np.broadcast_to(values, (end - start, values.size))
        # Fill output arrays directly; read depth once per block and retain no decoded-frame list.
        rgb = np.empty((end - start, height, width, 3), dtype=np.uint8)
        depth = reader["depth"][start:end]
        clouds = np.empty((end - start, config.pointcloud.num_points, 6), dtype=np.float32)
        for offset, index in enumerate(range(start, end)):
            image = next(images, None)
            if image is None or image.shape != (height, width, 3) or image.dtype != np.uint8:
                raise RawDataError(
                    f"{reader.path / 'rgb.mp4'}: RGB missing or invalid at row {index}"
                )
            cloud = build_point_cloud(
                depth_raw=depth[offset],
                color=image,
                depth_scale_m=camera.depth_scale_m,
                intrinsics=camera.intrinsics,
                T_xarm_base_from_color=base_from_color,
                table_plane_abcd=config.table_plane_abcd,
                config=config.pointcloud,
            )
            if cloud is None:
                raise RawDataError(f"point_cloud unavailable at row {index}")
            rgb[offset], clouds[offset] = image, cloud
        block.update(rgb=rgb, depth=depth, point_cloud=clouds)
        yield block
    if next(images, None) is not None:
        raise RawDataError(f"{reader.path / 'rgb.mp4'}: RGB contains extra frames")
