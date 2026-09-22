"""Raw admission and numerical transforms for direct policy export."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import yaml

from dexmani_real.dataset.contracts import (
    EpisodeAnnotation,
    EpisodeDecision,
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
    EEF_POSE_ALGORITHM_ID,
    EEF_POSE_COMPONENTS,
    EEF_POSE_DERIVATION,
    EEF_POSE_FRAME,
    compute_eef_pose_history_xarm_base,
)
from dexmani_real.planning.kinematics.pose import validate_canonical_rot6d
from dexmani_real.planning.kinematics.fingertip import (
    FINGERTIP_POINTS_DERIVATION,
    FINGERTIP_POLICY_ID,
    compute_fingertip_history_xarm_base,
)
from dexmani_real.planning.kinematics.hand_fk import HandKinematics
from dexmani_real.recording.storage.reader import EpisodeReader
from dexmani_real.recording.storage.schema import EPISODE_SCHEMA_VERSION
from dexmani_real.dataset.provenance import (
    POLICY_EVAL_WORKFLOW,
    read_provenance_workflow,
    supports_fixed_dt_teleop,
)
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

@contextmanager
def _open_processing_episode(episode: Path):
    """Read the current raw schema without modifying the source."""
    with h5py.File(episode / "data.h5", "r") as source:
        version = int(source["meta"].attrs["schema_version"])
    if version != EPISODE_SCHEMA_VERSION:
        raise ValueError(
            f"offline processing requires current raw v{EPISODE_SCHEMA_VERSION}; "
            f"got v{version}"
        )
    with EpisodeReader(episode) as reader:
        yield reader


def analyze_episode(
    reader: EpisodeReader,
    config: ProcessingConfig,
    annotation: EpisodeAnnotation | None = None,
) -> EpisodeDecision:
    """Technically admit all source rows or fail loudly; never repair rows.

    Shape/dtype, timing, media, masked-tactile, and provenance checks raise on
    corruption. There is no automatic quality rejection: an episode is excluded
    only by explicit operator annotation.
    """
    reader.require_valid(purpose="training export")
    source = reader.h5f
    workflow = read_provenance_workflow(source["meta"].attrs)
    if not supports_fixed_dt_teleop(workflow):
        if workflow == POLICY_EVAL_WORKFLOW:
            raise ValueError(
                "policy_eval rollout has synchronous/irregular execution timing and "
                "cannot enter the current fixed-dt teleop processing pipeline"
            )
        raise ValueError(
            f"unsupported provenance_workflow {workflow!r} for fixed-dt teleop processing"
        )
    frames = int(source["meta"].attrs["num_frames"])
    if frames <= 0:
        raise ValueError("raw episode must contain at least one row")
    if annotation is not None and not annotation.include:
        return EpisodeDecision(reader.h5_path, frames, "excluded by annotation")
    dt = float(reader.timing.grid_dt_s)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("control period must be finite and positive")
    specs = {
        "arm_qpos": ((7,), np.float64),
        "hand_qpos": ((12,), np.float64),
        "action_arm_joint_sent": ((7,), np.float64),
        "action_hand_joint": ((12,), np.float64),
        "action_arm_ee": ((9,), np.float64),
        "hand_contact": ((5, 3), np.float32),
        "hand_contact_valid": ((), np.bool_),
        "hand_tactile_force": ((5, 120, 3), np.float32),
        "hand_tactile_force_valid": ((), np.bool_),
        "timestamp": ((), np.float64),
        "flag_frame_status": ((), np.uint8),
        "observation_anchor_monotonic_ns": ((), np.uint64),
        "arm_source_monotonic_ns": ((), np.uint64),
        "hand_source_monotonic_ns": ((), np.uint64),
        "camera_source_monotonic_ns": ((), np.uint64),
    }
    # Tactile payloads carry an explicit per-row validity mask, so NaN is
    # admissible for invalid rows and never rejects the episode.
    validity_masked = {"hand_contact", "hand_tactile_force"}
    arrays = {}
    for name, (tail, dtype) in specs.items():
        dataset = source[name]
        if dataset.shape != (frames, *tail) or dataset.dtype != np.dtype(dtype):
            raise ValueError(f"{name}: corrupt shape/dtype")
        values = np.asarray(dataset[:])
        if (
            np.issubdtype(dtype, np.floating)
            and name not in validity_masked
            and not np.all(np.isfinite(values))
        ):
            raise ValueError(f"{name}: NaN/Inf")
        arrays[name] = values
    _validate_masked_tactile_rows(
        arrays["hand_contact"],
        arrays["hand_contact_valid"],
        label="raw hand_contact",
    )
    _validate_masked_tactile_rows(
        arrays["hand_tactile_force"],
        arrays["hand_tactile_force_valid"],
        label="raw hand_tactile_force",
    )
    if np.any(np.diff(arrays["timestamp"]) <= 0):
        raise ValueError("raw timestamps must strictly increase")
    anchor = arrays["observation_anchor_monotonic_ns"]
    if np.any(anchor == 0) or np.any(anchor[1:] <= anchor[:-1]):
        raise ValueError("control anchors must be positive and strictly increasing")
    expected_period_ns = int(round(dt * 1e9))
    if np.any(np.diff(anchor.astype(np.int64)) != expected_period_ns):
        raise ValueError("fixed-rate training export rejects missing/irregular grid intervals; raw timing is preserved")
    if np.any(np.asarray(source["command_id"][:]) == 0):
        raise ValueError("training export requires confirmed action labels for every row")
    for name in (
        "arm_source_monotonic_ns",
        "hand_source_monotonic_ns",
        "camera_source_monotonic_ns",
    ):
        if np.any(arrays[name] == 0) or np.any(arrays[name] > anchor):
            raise ValueError(
                f"{name}: source must be positive and causal to control anchor"
            )
    validate_canonical_rot6d(arrays["action_arm_ee"][:, 3:9], label="raw action_arm_ee")
    camera = load_raw_episode_camera_model(reader)
    load_raw_episode_base_from_color(reader)
    depth = source["depth"]
    geometry = camera.geometry.color
    if depth.shape != (
        frames,
        geometry.height,
        geometry.width,
    ) or depth.dtype != np.dtype(np.uint16):
        raise ValueError("depth: corrupt shape/dtype/frame count")
    # Images are decoded and checked once by iter_policy_blocks in both modes.
    # IK-hold rows (flag_frame_status=FRAME_IK_FAIL) are technically valid
    # source rows and stay admitted regardless of run length; quality selection
    # is explicit operator curation (annotation include:false), never an
    # automatic per-episode rejection here.
    return EpisodeDecision(reader.h5_path, frames)


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
            raise ValueError(
                f"annotation for {episode_name} has unknown keys: {sorted(unknown)}"
            )
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
        if (
            annotation.task_name is not None
            and annotation.task_name != resolved_task_name
        ):
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


def _validate_masked_tactile_rows(
    payload: np.ndarray,
    valid: np.ndarray,
    *,
    label: str,
) -> None:
    """Enforce the mask/payload invariant for tactile rows; never repair it.

    ``valid`` rows must be fully finite (a real zero/no-contact reading is finite
    zero + valid); ``invalid`` rows must be all-NaN. Any contradiction is a
    technical contract error and raises.
    """
    axes = tuple(range(1, payload.ndim))
    rows_finite = np.all(np.isfinite(payload), axis=axes)
    rows_all_nan = np.all(np.isnan(payload), axis=axes)
    if np.any(valid & ~rows_finite):
        raise ValueError(f"{label}: non-finite payload on a valid row")
    if np.any(~valid & ~rows_all_nan):
        raise ValueError(f"{label}: finite payload on an invalid row")


def policy_semantics(reader: EpisodeReader, config: ProcessingConfig) -> dict[str, Any]:
    meta = reader.h5f["meta"].attrs
    return {
        "obs_alignment": "obs[t]_before_action[t]",
        "observation_alignment": "control_step_latest_causal",
        "state_alignment": "control_step",
        "action_semantics": "jointly_adopted_robot_target",
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


def iter_policy_blocks(
    reader: EpisodeReader, config: ProcessingConfig, *, chunk_frames: int
):
    """One episode of numeric state; bounded RGB-D/cloud chunks, never a task in RAM."""
    frames = int(reader.h5f["meta"].attrs["num_frames"])
    camera_model = load_raw_episode_camera_model(reader)
    geometry = camera_model.geometry
    transform = load_raw_episode_base_from_color(reader)
    values = {}
    arm_action = np.asarray(reader.h5f["action_arm_joint_sent"][:], dtype=np.float32)
    hand_action = np.asarray(reader.h5f["action_hand_joint"][:], dtype=np.float32)
    arm_action_ee = np.asarray(reader.h5f["action_arm_ee"][:], dtype=np.float32)
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
    # Tactile validity is copied directly from raw; processing never
    # reconstructs measurement truth from calibration/unit/freshness fields.
    values["contact_force"] = np.asarray(
        reader.h5f["hand_contact"][:], dtype=np.float32
    )
    values["contact_force_valid"] = np.asarray(
        reader.h5f["hand_contact_valid"][:], dtype=bool
    )
    values["tactile_force"] = np.asarray(
        reader.h5f["hand_tactile_force"][:], dtype=np.float32
    )
    values["tactile_force_valid"] = np.asarray(
        reader.h5f["hand_tactile_force_valid"][:], dtype=bool
    )
    hand_fk = HandKinematics(config.hand_urdf_path, list(config.fingertip_link_names))
    if not hand_fk.is_ready():
        raise RuntimeError("policy fingertip FK startup failed")
    eef_pose = compute_eef_pose_history_xarm_base(joint_state[:, :7])
    values["eef_pose"] = eef_pose.astype(np.float32)
    values["fingertip_points"] = compute_fingertip_history_xarm_base(
        joint_state[:, :7],
        joint_state[:, 7:19],
        hand_fk=hand_fk,
        handbase_position_eef_m=np.asarray(
            config.handbase_position_eef_m, dtype=np.float64
        ),
        handbase_quat_eef_wxyz=np.asarray(
            config.handbase_quat_eef_wxyz, dtype=np.float64
        ),
        eef_pose_history=eef_pose,
    )
    # Flat timing arrays: one scalar per control row.
    values["observation_anchor_monotonic_ns"] = reader.h5f[
        "observation_anchor_monotonic_ns"
    ][:]
    values["arm_source_monotonic_ns"] = reader.h5f["arm_source_monotonic_ns"][:]
    values["hand_source_monotonic_ns"] = reader.h5f["hand_source_monotonic_ns"][:]
    values["camera_source_monotonic_ns"] = reader.h5f["camera_source_monotonic_ns"][:]

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
            camera_intrinsic=np.broadcast_to(
                geometry.color.matrix().astype(np.float32).reshape(9), (end - start, 9)
            ),
            camera_extrinsic=np.broadcast_to(
                transform.astype(np.float32), (end - start, 4, 4)
            ),
        )
        for key, value in block.items():
            if (
                np.issubdtype(value.dtype, np.floating)
                and key not in {"contact_force", "tactile_force"}
                and not np.isfinite(value).all()
            ):
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
