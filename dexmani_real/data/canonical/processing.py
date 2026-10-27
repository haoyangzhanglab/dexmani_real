from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from dexmani_real.data.canonical.schema import ExportConfig
from dexmani_real.data.raw.reader import EpisodeReader
from dexmani_real.data.raw.schema import DATASET_SPECS
from dexmani_real.geometry.camera import (
    CameraSnapshot,
    camera_parameters_cv,
    validate_aligned_depth_distortion,
)
from dexmani_real.geometry.kinematics import eef_poses
from dexmani_real.perception.hand_cloud import hand_geometry
from dexmani_real.perception.pointcloud import PointCloudProcessor


def validate_export_episode(reader: EpisodeReader) -> CameraSnapshot:
    frames = reader.num_frames
    if frames <= 0:
        raise ValueError("episode contains no rows")
    reader.require_fields(*DATASET_SPECS, "rgb", "depth")
    camera = reader.read_camera()
    validate_aligned_depth_distortion(camera.color.distortion_model)
    if camera.T_xarm_base_from_color is None:
        raise ValueError("canonical export requires camera extrinsics")
    color = camera.color
    expected = (frames, color.height, color.width)
    if reader["depth"].shape != expected or reader["depth"].dtype != np.uint16:
        raise ValueError(f"depth must be uint16 {expected}")
    reader.validate_completion()
    reader.validate_rows()
    if frames == 1:
        logging.getLogger(__name__).warning(
            "%s: single row, no sampling interval to verify", reader.path
        )
    for name in DATASET_SPECS:
        reader[name]
    return camera


def discover_episode_dirs(input_root: str | Path) -> tuple[Path, ...]:
    """枚举所有 episode 目录，损坏文件由读取边界直接报错。"""
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
    config: ExportConfig,
    *,
    arm_fk,
    hand_fk,
    hand_cloud,
    chunk_frames: int,
    camera: CameraSnapshot,
    handbase_rotation_eef: np.ndarray,
):
    """分块转换，不修补缺失观测或重采样行。"""
    height, width = camera.color.height, camera.color.width
    processor = PointCloudProcessor(
        depth_scale_m=camera.depth_scale_m,
        intrinsics=camera.color,
        T_xarm_base_from_color=camera.T_xarm_base_from_color,
        table_plane_abcd=config.table_plane_abcd,
        config=config.pointcloud,
    )
    camera_parameters = camera_parameters_cv(camera.color, camera.T_xarm_base_from_color)
    images = reader.iter_rgb_frames()
    for start in range(0, reader.num_frames, chunk_frames):
        end = min(reader.num_frames, start + chunk_frames)
        raw = {name: reader[name][start:end] for name in DATASET_SPECS if name != "timestamp"}
        for name in (
            "arm_qpos",
            "hand_qpos",
            "action_arm_joint_target",
            "action_hand_joint_target",
        ):
            valid = np.isfinite(raw[name]).all(axis=1)
            if not valid.all():
                row = start + int(np.flatnonzero(~valid)[0])
                raise ValueError(f"Raw {name}: nonfinite values at row {row}")
        arm, hand = raw["arm_qpos"], raw["hand_qpos"]
        arm_action, hand_action = (
            raw["action_arm_joint_target"],
            raw["action_hand_joint_target"],
        )
        poses = eef_poses(arm, arm_fk=arm_fk)
        action_poses = eef_poses(arm_action, arm_fk=arm_fk)
        hand_features = hand_geometry(
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
            **hand_features,
        }
        for name, values in camera_parameters.items():
            block[name] = np.broadcast_to(values, (end - start, values.size))
        # 深度按块读取，RGB 逐帧解码到输出数组。
        rgb = np.empty((end - start, height, width, 3), dtype=np.uint8)
        depth = reader["depth"][start:end]
        clouds = np.empty((end - start, config.pointcloud.num_points, 6), dtype=np.float32)
        for offset, index in enumerate(range(start, end)):
            image = next(images, None)
            if image is None or image.shape != (height, width, 3) or image.dtype != np.uint8:
                raise ValueError(
                    f"{reader.path / 'rgb.mp4'}: RGB missing or invalid at row {index}"
                )
            cloud = processor.process(depth[offset], image)
            if cloud is None:
                raise ValueError(f"point_cloud unavailable at row {index}")
            rgb[offset], clouds[offset] = image, cloud
        block.update(rgb=rgb, depth=depth, point_cloud=clouds)
        yield block
    if next(images, None) is not None:
        raise ValueError(f"{reader.path / 'rgb.mp4'}: RGB contains extra frames")
