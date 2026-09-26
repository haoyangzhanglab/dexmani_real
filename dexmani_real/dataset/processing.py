"""Raw admission and numerical transforms for direct policy export."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from dexmani_real.dataset.contracts import (
    EpisodeAnnotation,
    ProcessingConfig,
    canonical_json,
    validate_task_name,
)
from dexmani_real.dataset.pointcloud import (
    RawEpisodePointCloudDeriver,
    load_raw_episode_base_from_color,
    load_raw_episode_camera_model,
)
from dexmani_real.planning.kinematics.arm_fk import (
    EEF_POSE_COMPONENTS,
    EEF_POSE_FRAME,
    compute_eef_pose_history_xarm_base,
)
from dexmani_real.planning.kinematics.fingertip import compute_fingertip_history_xarm_base
from dexmani_real.planning.kinematics.hand_fk import HandKinematics
from dexmani_real.planning.kinematics.pose import validate_canonical_rot6d
from dexmani_real.recording.storage.reader import EpisodeReader
from dexmani_real.recording.storage.schema import DATASET_SPECS
from dexmani_real.robot.model import (
    CONTACT_FORCE_REPRESENTATION,
    HAND_FINGER_NAMES,
    HAND_FINGER_ORDER_ID,
    ROBOT_JOINT_NAMES,
    TACTILE_FORCE_AXIS_LABELS,
    TACTILE_FORCE_POINT_ORDER,
    TACTILE_FORCE_REPRESENTATION,
    TACTILE_FORCE_SENSOR_ORDER,
    XHAND_SDK_NATIVE_UNKNOWN_SI_UNIT,
    XHAND_SENSOR_NATIVE_AXES_FRAME,
    XHAND_TACTILE_SENSOR_FINGER_IDS,
)


def validate_episode(reader: EpisodeReader) -> int:
    """Admit every row of a complete teleop episode, or reject the whole episode."""
    source = reader.h5f
    meta = source["meta"].attrs
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
    if not reader.episode_valid:
        raise ValueError("episode_valid is false")
    frame_valid = source["frame_valid"][:]
    bad = np.flatnonzero(~frame_valid)
    if len(bad):
        raise ValueError(f"frame_valid is false at row {bad[0]}")
    for name, spec in DATASET_SPECS.items():
        if spec.dtype.kind == "f":
            if not np.isfinite(source[name][:]).all():
                raise ValueError(f"{name}: non-finite values")
    return frames


def load_annotations(path: str | Path | None) -> dict[str, EpisodeAnnotation]:
    """Load whole-episode inclusion and task overrides."""

    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError("annotation YAML root must be a mapping")
    raw_episodes = payload.get("episodes", payload)
    if not isinstance(raw_episodes, dict):
        raise ValueError("annotation episodes must be a mapping")
    result: dict[str, EpisodeAnnotation] = {}
    allowed = {"include", "task_name"}
    for episode_name, raw in raw_episodes.items():
        if not isinstance(episode_name, str) or not episode_name:
            raise ValueError("annotation episode names must be non-empty strings")
        raw = {} if raw is None else raw
        if not isinstance(raw, dict):
            raise ValueError(f"annotation for {episode_name} must be a mapping")
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"annotation for {episode_name} has unknown keys: {sorted(unknown)}")
        include = raw.get("include", True)
        task_name = raw.get("task_name")
        if not isinstance(include, bool):
            raise ValueError(f"{episode_name}.include must be boolean")
        if task_name is not None and not isinstance(task_name, str):
            raise ValueError(f"{episode_name}.task_name must be string or null")
        result[episode_name] = EpisodeAnnotation(
            include=include,
            task_name=task_name,
        )
    return result


def validate_annotation_task_name_override(
    annotations: Mapping[str, EpisodeAnnotation], task_name: str | None
) -> str | None:
    """Reject a batch task override that conflicts with an audited annotation."""

    if task_name is None:
        return None
    resolved_task_name = validate_task_name(task_name)
    for episode_name, annotation in sorted(annotations.items()):
        if annotation.task_name is not None and annotation.task_name != resolved_task_name:
            raise ValueError(
                f"--task-name={resolved_task_name!r} conflicts with annotation task_name="
                f"{annotation.task_name!r} for {episode_name}"
            )
    return resolved_task_name


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


def policy_semantics(reader: EpisodeReader, config: ProcessingConfig) -> dict[str, Any]:
    meta = reader.h5f["meta"].attrs
    return {
        "joint_names": list(ROBOT_JOINT_NAMES),
        "finger_names": list(HAND_FINGER_NAMES),
        "tactile_sensor_ids": list(XHAND_TACTILE_SENSOR_FINGER_IDS),
        "tactile_axis_names": ["fx", "fy", "fz"],
        "tactile_point_indices": list(range(120)),
        "point_cloud_features": ["x", "y", "z", "r", "g", "b"],
        "rgb_channels": ["r", "g", "b"],
        "hand_current_unit": "mA",
        "arm_effort_unit": "sdk_native_unverified",
        "obs_alignment": "obs[t]_before_action[t]",
        "observation_alignment": "control_step_latest_causal",
        "state_alignment": "control_step",
        "action_semantics": "teleop_published_joint_target",
        "fingertip_points_frame": "xarm_base",
        "fingertip_points_unit": "m",
        "fingertip_config_json": canonical_json(
            {
                "fingertip_link_names": list(config.fingertip_link_names),
            }
        ),
        "eef_pose_frame": EEF_POSE_FRAME,
        "eef_pose_components": EEF_POSE_COMPONENTS,
        "action_ee_frame": "xarm_base",
        "action_ee_components": "eef_position_m(3)+eef_rot6d(6)+xhand_target_rad(12)",
        "contact_force_representation": CONTACT_FORCE_REPRESENTATION,
        "contact_force_unit": XHAND_SDK_NATIVE_UNKNOWN_SI_UNIT,
        "contact_force_frame": XHAND_SENSOR_NATIVE_AXES_FRAME,
        "tactile_force_representation": TACTILE_FORCE_REPRESENTATION,
        "tactile_force_finger_order": HAND_FINGER_ORDER_ID,
        "tactile_force_sensor_order": TACTILE_FORCE_SENSOR_ORDER,
        "tactile_force_point_order": TACTILE_FORCE_POINT_ORDER,
        "tactile_force_axis_labels": TACTILE_FORCE_AXIS_LABELS,
        "tactile_force_unit": XHAND_SDK_NATIVE_UNKNOWN_SI_UNIT,
        "depth_scale_m_per_unit": float(meta["depth_scale"]),
        "depth_invalid_value": 0,
        "point_cloud_frame": "xarm_base",
        "pointcloud_config_json": canonical_json(config.pointcloud.to_dict()),
    }


def iter_policy_blocks(reader: EpisodeReader, config: ProcessingConfig, *, chunk_frames: int):
    """One episode of numeric state; bounded RGB-D/cloud chunks, never a task in RAM."""
    frames = reader.num_frames
    camera_model = load_raw_episode_camera_model(reader)
    geometry = camera_model.geometry
    transform = load_raw_episode_base_from_color(reader)
    arm_action = np.asarray(reader.h5f["action_arm_joint_target"][:], dtype=np.float64)
    hand_action = np.asarray(reader.h5f["action_hand_joint_target"][:], dtype=np.float64)
    arm_qpos = np.asarray(reader.h5f["arm_qpos"][:], dtype=np.float64)
    hand_qpos = np.asarray(reader.h5f["hand_qpos"][:], dtype=np.float64)
    action_ee_pose = compute_eef_pose_history_xarm_base(arm_action)
    eef_pose = compute_eef_pose_history_xarm_base(arm_qpos)
    joint_state = np.concatenate((arm_qpos, hand_qpos), axis=1)
    hand_fk = HandKinematics(config.hand_urdf_path, list(config.fingertip_link_names))
    if not hand_fk.is_ready():
        raise RuntimeError("policy fingertip FK startup failed")
    mount = reader.h5f["meta"].attrs
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
        "contact_force": np.asarray(reader.h5f["hand_contact"][:], dtype=np.float32),
        "tactile_force": np.asarray(reader.h5f["hand_tactile_force"][:], dtype=np.float32),
    }
    for name in ("arm_qvel", "arm_effort", "hand_current"):
        values[name] = np.asarray(reader.h5f[name][:], dtype=np.float32)
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
            depth.append(np.asarray(reader.h5f["depth"][index], dtype=np.uint16))
            clouds.append(cloud)
        block.update(
            rgb=np.stack(rgb),
            depth=np.stack(depth),
            point_cloud=np.stack(clouds),
        )
        for key, value in block.items():
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise ValueError(f"{key}: non-finite transformed values")
        validate_canonical_rot6d(block["action_ee"][:, 3:9], label="action_ee")
        validate_canonical_rot6d(block["eef_pose"][:, 3:9], label="eef_pose")
        yield block
    if next(images, None) is not None:
        raise ValueError("RGB contains extra frames")
