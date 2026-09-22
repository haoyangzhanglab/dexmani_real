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
from dexmani_real.dataset.provenance import (
    read_provenance_workflow,
    supports_fixed_dt_teleop,
)
from dexmani_real.planning.kinematics.arm_fk import (
    EEF_POSE_ALGORITHM_ID,
    EEF_POSE_COMPONENTS,
    EEF_POSE_DERIVATION,
    EEF_POSE_FRAME,
    compute_eef_pose_history_xarm_base,
)
from dexmani_real.planning.kinematics.fingertip import (
    FINGERTIP_POINTS_DERIVATION,
    FINGERTIP_POLICY_ID,
    compute_fingertip_history_xarm_base,
)
from dexmani_real.planning.kinematics.hand_fk import HandKinematics
from dexmani_real.planning.kinematics.pose import validate_canonical_rot6d
from dexmani_real.recording.storage.reader import EpisodeReader
from dexmani_real.recording.storage.schema import DATASET_SPECS, FRAME_OK
from dexmani_real.robot.model import (
    CONTACT_FORCE_REPRESENTATION,
    HAND_FINGER_ORDER_ID,
    TACTILE_FORCE_AXIS_LABELS,
    TACTILE_FORCE_POINT_ORDER,
    TACTILE_FORCE_REPRESENTATION,
    TACTILE_FORCE_SENSOR_ORDER,
    XHAND_SDK_NATIVE_UNKNOWN_SI_UNIT,
    XHAND_SENSOR_NATIVE_AXES_FRAME,
)
from dexmani_real.sensor.pointcloud import (
    POINT_CLOUD_COLOR_SOURCE,
    POINT_CLOUD_POLICY_ID,
    POINT_CLOUD_SAMPLING,
    POINT_CLOUD_TRANSFORM,
)


def validate_episode(reader) -> int:
    """Admit every row of a complete teleop episode, or reject the whole episode."""
    reader.require_valid(purpose="training export")
    source = reader.h5f
    if bool(source["meta"].attrs.get("had_pause", False)):
        raise ValueError("episode contains an operator or sensor pause")
    workflow = read_provenance_workflow(source["meta"].attrs)
    if not supports_fixed_dt_teleop(workflow):
        raise ValueError(f"training requires teleop provenance, got {workflow!r}")
    frames = int(source["meta"].attrs["num_frames"])
    if frames <= 0:
        raise ValueError("episode contains no rows")
    status = source["flag_frame_status"][:]
    bad = np.flatnonzero(status != FRAME_OK)
    if len(bad):
        raise ValueError(f"non-OK frame status {status[bad[0]]} at row {bad[0]}")
    for name in ("hand_contact_valid", "hand_tactile_force_valid"):
        bad = np.flatnonzero(~source[name][:])
        if len(bad):
            raise ValueError(f"{name} false at row {bad[0]}")
    for name, spec in DATASET_SPECS.items():
        if spec.dtype.kind == "f" and name not in {"arm_eef_intent", "head_quat_wxyz"}:
            if not np.isfinite(source[name][:]).all():
                raise ValueError(f"{name}: non-finite values")
    dt = reader.timing.grid_dt_s
    for name in ("observation_timestamp_ns", "action_timestamp_ns"):
        stamps = source[name][:]
        if np.any(stamps == 0) or np.any(stamps[1:] <= stamps[:-1]):
            raise ValueError(f"{name}: must be positive and strictly increasing")
        gaps = np.diff(stamps.astype(np.int64))
        if np.any(gaps > round(2 * dt * 1e9)):
            raise ValueError(f"{name}: pause/missing-control gap exceeds 2 nominal periods")
    if np.any(source["action_timestamp_ns"][:] < source["observation_timestamp_ns"][:]):
        raise ValueError("action precedes observation completion")
    for name in ("arm_timestamp_ns", "hand_timestamp_ns", "camera_timestamp_ns", "vr_timestamp_ns"):
        if np.any(source[name][:] == 0):
            raise ValueError(f"{name}: missing source timestamp")
    camera = load_raw_episode_camera_model(reader)
    load_raw_episode_base_from_color(reader)
    geometry = camera.geometry.color
    depth = source["depth"]
    if depth.shape != (frames, geometry.height, geometry.width) or depth.dtype != np.dtype(
        np.uint16
    ):
        raise ValueError("depth shape/dtype/frame count mismatch")
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
    """Accept either one episode directory or a task directory of episodes.

    Task roots discover only explicit ``episode_*`` subdirectories
    (ManiUniCon-style convention). A corrupt ``episode_*`` directory is still
    discovered and must fail loudly downstream; it is never silently filtered
    by a ``data.h5`` existence check. The direct single-episode root
    (``root/data.h5``) path is unchanged. Returned paths are absolute so
    persisted source provenance remains independent of the caller's working
    directory.
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
    camera = load_raw_episode_camera_model(reader)
    return {
        "camera_intrinsic": camera.geometry.color.matrix().reshape(-1).tolist(),
        "camera_extrinsic": load_raw_episode_base_from_color(reader).tolist(),
        "camera_geometry": camera.geometry.to_dict(),
        "joint_order": "xarm7_joint1_to_7+xhand_sdk_12",
        "hand_current_unit": "mA",
        "arm_effort_unit": "sdk_native_unverified",
        "obs_alignment": "obs[t]_before_action[t]",
        "observation_alignment": "control_step_latest_causal",
        "state_alignment": "control_step",
        "action_semantics": "teleop_published_joint_target",
        "fingertip_points_frame": "xarm_base",
        "fingertip_points_unit": "m",
        "fingertip_points_derivation": FINGERTIP_POINTS_DERIVATION,
        "fingertip_points_policy_id": FINGERTIP_POLICY_ID,
        "fingertip_config_json": canonical_json(
            {
                "fingertip_link_names": list(config.fingertip_link_names),
                "handbase_position_eef_m": list(config.handbase_position_eef_m),
                "handbase_quat_eef_wxyz": list(config.handbase_quat_eef_wxyz),
            }
        ),
        "eef_pose_frame": EEF_POSE_FRAME,
        "eef_pose_components": EEF_POSE_COMPONENTS,
        "eef_pose_derivation": EEF_POSE_DERIVATION,
        "eef_pose_algorithm_id": EEF_POSE_ALGORITHM_ID,
        "action_ee_frame": "xarm_base",
        "action_ee_components": "eef_position_m(3)+eef_rot6d(6)+xhand_target_rad(12)",
        "contact_force_source": "raw_hand_contact_control_step",
        "contact_force_representation": CONTACT_FORCE_REPRESENTATION,
        "contact_force_unit": XHAND_SDK_NATIVE_UNKNOWN_SI_UNIT,
        "contact_force_si_verified": False,
        "contact_force_frame": XHAND_SENSOR_NATIVE_AXES_FRAME,
        "tactile_force_representation": TACTILE_FORCE_REPRESENTATION,
        "tactile_force_finger_order": HAND_FINGER_ORDER_ID,
        "tactile_force_sensor_order": TACTILE_FORCE_SENSOR_ORDER,
        "tactile_force_point_order": TACTILE_FORCE_POINT_ORDER,
        "tactile_force_axis_labels": TACTILE_FORCE_AXIS_LABELS,
        "tactile_force_unit": XHAND_SDK_NATIVE_UNKNOWN_SI_UNIT,
        "tactile_force_si_verified": False,
        "tactile_force_spatial_geometry_verified": False,
        "depth_scale_m_per_unit": float(meta["depth_scale"]),
        "depth_invalid_value": 0,
        "camera_intrinsic_semantics": "native_color_intrinsics_for_depth_to_color_aligned_depth",
        "camera_extrinsic_semantics": "T_xarm_base_from_color;native_color_optical_to_xarm_base",
        "point_cloud_frame": "xarm_base",
        "processing_config_json": canonical_json(
            {
                "pointcloud": config.pointcloud.to_dict(),
                "table_plane_abcd": None
                if config.table_plane_abcd is None
                else list(config.table_plane_abcd),
            }
        ),
        "point_cloud_color_source": POINT_CLOUD_COLOR_SOURCE,
        "point_cloud_policy_id": POINT_CLOUD_POLICY_ID,
        "point_cloud_table_plane_abcd_json": canonical_json(
            None if config.table_plane_abcd is None else list(config.table_plane_abcd)
        ),
        "point_cloud_sampling": POINT_CLOUD_SAMPLING,
        "point_cloud_transform": POINT_CLOUD_TRANSFORM,
    }


def iter_policy_blocks(reader: EpisodeReader, config: ProcessingConfig, *, chunk_frames: int):
    """One episode of numeric state; bounded RGB-D/cloud chunks, never a task in RAM."""
    frames = int(reader.h5f["meta"].attrs["num_frames"])
    camera_model = load_raw_episode_camera_model(reader)
    geometry = camera_model.geometry
    transform = load_raw_episode_base_from_color(reader)
    values = {}
    arm_action = np.asarray(reader.h5f["action_arm_joint_target"][:], dtype=np.float32)
    hand_action = np.asarray(reader.h5f["action_hand_joint_target"][:], dtype=np.float32)
    arm_action_ee = compute_eef_pose_history_xarm_base(arm_action).astype(np.float32)
    joint_state = np.concatenate(
        (
            np.asarray(reader.h5f["arm_qpos"][:], dtype=np.float32),
            np.asarray(reader.h5f["hand_qpos"][:], dtype=np.float32),
        ),
        axis=1,
    )
    values["joint_state"] = joint_state
    values["action"] = np.concatenate((arm_action, hand_action), axis=1)
    values["action_ee"] = np.concatenate((arm_action_ee, hand_action), axis=1)
    for name in ("arm_qvel", "arm_effort", "hand_current"):
        values[name] = np.asarray(reader.h5f[name][:], dtype=np.float32)
    values["contact_force"] = np.asarray(reader.h5f["hand_contact"][:], dtype=np.float32)
    values["tactile_force"] = np.asarray(reader.h5f["hand_tactile_force"][:], dtype=np.float32)
    hand_fk = HandKinematics(config.hand_urdf_path, list(config.fingertip_link_names))
    if not hand_fk.is_ready():
        raise RuntimeError("policy fingertip FK startup failed")
    eef_pose = compute_eef_pose_history_xarm_base(joint_state[:, :7])
    values["eef_pose"] = eef_pose.astype(np.float32)
    values["fingertip_points"] = compute_fingertip_history_xarm_base(
        joint_state[:, :7],
        joint_state[:, 7:19],
        hand_fk=hand_fk,
        handbase_position_eef_m=np.asarray(config.handbase_position_eef_m, dtype=np.float64),
        handbase_quat_eef_wxyz=np.asarray(config.handbase_quat_eef_wxyz, dtype=np.float64),
        eef_pose_history=eef_pose,
    )
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
        validate_canonical_rot6d(block["eef_pose"][:, 3:9], label="eef_pose")
        cloud = block["point_cloud"]
        lower, upper = np.asarray(config.pointcloud.workspace).reshape(2, 3)
        if (
            np.any(cloud[..., :3] < lower)
            or np.any(cloud[..., :3] > upper)
            or np.any(cloud[..., 3:] < 0)
            or np.any(cloud[..., 3:] > 1)
            or np.any(~np.any(np.linalg.norm(cloud[..., :3], axis=2) > 0, axis=1))
        ):
            raise ValueError("derived point cloud violates workspace/color contract")
        yield block
    if next(images, None) is not None:
        raise ValueError("RGB contains extra frames")
