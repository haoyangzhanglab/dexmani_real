"""Dense Raw layout checks and full-modal numerical transforms."""

from __future__ import annotations

from pathlib import Path
import logging

import numpy as np

from dexmani_real.dataset.contracts import ProcessingConfig
from dexmani_real.dataset.pointcloud import (
    RawEpisodeCameraModel,
    load_raw_episode_base_from_color,
    load_raw_episode_camera_model,
)
from dexmani_real.planning.kinematics.arm_fk import (
    compute_eef_pose_history_xarm_base,
)
from dexmani_real.planning.kinematics.fingertip import compute_fingertip_history_xarm_base
from dexmani_real.planning.kinematics.hand_fk import HandKinematics
from dexmani_real.recording.storage.reader import EpisodeReader, RawDataError
from dexmani_real.recording.storage.schema import DATASET_SPECS, validate_training_rows
from dexmani_real.robot.model import XHAND_RIGHT_URDF_PATH
from dexmani_real.sensor.camera.geometry import camera_parameters_cv, validate_aligned_depth_distortion
from dexmani_real.sensor.pointcloud import build_point_cloud


def validate_export_episode(
    reader: EpisodeReader,
) -> tuple[RawEpisodeCameraModel, np.ndarray]:
    """Check metadata/layout/timing and prepare static camera geometry once per episode."""
    frames = reader.num_frames
    if frames <= 0:
        raise RawDataError("episode contains no rows")
    reader.require_fields(*DATASET_SPECS, "rgb", "depth")
    reader.require_metadata(
        "camera_payload_mode", "camera_color_height", "camera_color_width", "depth_scale",
        "camera_color_intrinsics", "camera_color_distortion_model",
        "camera_color_distortion_coeffs", "camera_T_xarm_base_from_color",
    )
    expected = (
        frames,
        int(reader.meta["camera_color_height"]),
        int(reader.meta["camera_color_width"]),
    )
    if reader["depth"].shape != expected or reader["depth"].dtype != np.uint16:
        raise RawDataError(f"depth must be uint16 {expected}")
    errors = validate_training_rows(
        reader["timestamp"][:], reader["dispatch_status"][:], reader.control_hz,
    )
    if errors:
        raise RawDataError("; ".join(errors))
    reader.validate_arm_execution()
    if frames == 1:
        logging.getLogger(__name__).warning("%s: single row, no sampling interval to verify", reader.path)
    for name in DATASET_SPECS:
        reader[name]  # EpisodeReader validates shape/dtype without reading the array.
    try:
        camera = load_raw_episode_camera_model(reader)
        validate_aligned_depth_distortion(camera.geometry.color.distortion_model)
        return camera, load_raw_episode_base_from_color(reader)
    except (ValueError, TypeError, KeyError) as exc:
        raise RawDataError(f"invalid camera geometry: {exc}") from exc


def discover_episode_dirs(input_root: str | Path) -> tuple[Path, ...]:
    """Return absolute episode paths; a data.h5 root is one episode.
    Task discovery includes corrupt episode_* directories for explicit downstream rejection.
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
        yield from reader.iter_rgb_frames()
    except (ValueError, IndexError, EOFError) as exc:
        raise RawDataError(f"{reader.path / 'rgb.mp4'}: invalid RGB video: {exc}") from exc


def iter_canonical_blocks(
    reader: EpisodeReader, config: ProcessingConfig, *,
    camera: RawEpisodeCameraModel, base_from_color: np.ndarray, chunk_frames: int,
):
    """Check and convert each Raw block without repairing or resampling rows."""
    height, width = camera.geometry.color.height, camera.geometry.color.width
    try:
        camera_parameters = camera_parameters_cv(
            camera.geometry.color, base_from_color,
        )
    except ValueError as exc:
        raise RawDataError(f"invalid camera parameters: {exc}") from exc
    hand_fk = HandKinematics(str(XHAND_RIGHT_URDF_PATH), list(config.fingertip_link_names))
    images = _iter_rgb_frames(reader)
    for start in range(0, reader.num_frames, chunk_frames):
        end = min(reader.num_frames, start + chunk_frames)
        raw = {
            name: reader[name][start:end]
            for name in DATASET_SPECS if name not in ("timestamp", "dispatch_status")
        }
        for name, values in raw.items():
            if np.issubdtype(values.dtype, np.floating):
                valid = np.isfinite(values).reshape(len(values), -1).all(axis=1)
                if not valid.all():
                    row = start + int(np.flatnonzero(~valid)[0])
                    raise RawDataError(f"Raw {name}: nonfinite values at row {row}")
        arm, hand = raw["arm_qpos"], raw["hand_qpos"]
        arm_action, hand_action = raw["action_arm_joint_target"], raw["action_hand_joint_target"]
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
            "fingertip_xyz": tips,
            "finger_tactile_force": raw["hand_contact"],
            "taxel_tactile_force": raw["hand_tactile_force"],
        }
        for name in ("arm_qvel", "arm_effort", "hand_current"):
            block[name] = raw[name].astype(np.float32)
        for name, values in camera_parameters.items():
            block[name] = np.broadcast_to(values, (end - start, values.size))
        # Fill final arrays directly; avoid retaining every decoded frame plus
        # a second stacked RGB-D/cloud batch. Depth needs one HDF5 slice per block.
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
                depth_raw=depth[offset], color=image,
                depth_scale_m=camera.depth_scale_m, geometry=camera.geometry,
                T_xarm_base_from_color=base_from_color,
                table_plane_abcd=config.table_plane_abcd, config=config.pointcloud,
            )
            if cloud is None:
                raise RawDataError(f"point_cloud unavailable at row {index}")
            rgb[offset], clouds[offset] = image, cloud
        block.update(rgb=rgb, depth=depth, point_cloud=clouds)
        yield block
    if next(images, None) is not None:
        raise RawDataError(f"{reader.path / 'rgb.mp4'}: RGB contains extra frames")
