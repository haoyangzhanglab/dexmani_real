"""Transactional, row-preserving control-step dataset processing."""

from __future__ import annotations

import shutil
import tempfile
from collections import Counter
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
    validate_processed_task_name,
)
from dexmani_real.dataset.pointcloud import (
    RawEpisodePointCloudDeriver,
    load_raw_episode_base_from_color,
    load_raw_episode_camera_model,
)
from dexmani_real.dataset.processed import (
    _ACTION_EE_FRAME,
    _CONTACT_FORCE_SOURCE,
    _CONTACT_FORCE_SI_VERIFIED,
    _FINGERTIP_POINTS_FRAME,
    _FINGERTIP_POINTS_UNIT,
    _FRAME_CHUNKED_DATASETS,
    _TACTILE_FORCE_SI_VERIFIED,
    _TACTILE_FORCE_SPATIAL_GEOMETRY_VERIFIED,
    PROCESSED_SCHEMA_NAME,
    PROCESSED_SCHEMA_VERSION,
    _dataset_row_slices,
    _expected_specs,
    _validate_masked_tactile_rows,
    validate_processed_hdf5,
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
from dexmani_real.utils.atomic_io import atomic_publish

_TASK_NAME_CANDIDATE_UNSET = object()


@contextmanager
def _open_processing_episode(episode: Path):
    """Read current raw only; historical v26 stays frozen (reprocess via old revision)."""
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
    """Admit all source rows or reject the entire episode; never repair rows."""
    source = reader.h5f
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
        "source_sample_index": ((), np.int64),
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
    if not np.array_equal(arrays["source_sample_index"], np.arange(frames)):
        raise ValueError("raw sample identity must match every control row")
    anchor = arrays["observation_anchor_monotonic_ns"]
    if np.any(anchor == 0) or np.any(anchor[1:] <= anchor[:-1]):
        raise ValueError("control anchors must be positive and strictly increasing")
    for name in ("arm_source_monotonic_ns", "hand_source_monotonic_ns", "camera_source_monotonic_ns"):
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
    # Decode all required images during admission, so dry-run can detect
    # technical corruption before any batch is built.
    decoded = 0
    for image in reader.iter_camera_frames("rgb"):
        if (
            image.shape != (geometry.height, geometry.width, 3)
            or image.dtype != np.uint8
        ):
            raise ValueError("RGB: corrupt shape/dtype")
        decoded += 1
    if decoded != frames:
        raise ValueError(f"RGB frame count {decoded} != source frames {frames}")
    for rows in _dataset_row_slices(depth):
        depth[rows]  # Force HDF5 decompression/read errors at admission.
    # Preserve the established transient/persistent boundary: up to four
    # consecutive FRAME_IK_FAIL samples are a short hold, five are persistent.
    consecutive_ik_fail = 0
    for status in arrays["flag_frame_status"]:
        consecutive_ik_fail = consecutive_ik_fail + 1 if status == 2 else 0
        if consecutive_ik_fail > 4:
            return EpisodeDecision(reader.h5_path, frames, "persistent IK failure")
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
    resolved_task_name = validate_processed_task_name(task_name)
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


def _task_name_candidate(
    reader: EpisodeReader,
    annotation: EpisodeAnnotation,
    task_name: str | None,
) -> tuple[Any, bool]:
    """Choose one task source and flag an unvalidated raw fallback."""

    if task_name is not None:
        return task_name, False
    if annotation.task_name is not None:
        return annotation.task_name, False
    return reader.h5f["meta"].attrs.get("task_label", ""), True


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


def _dataset_kwargs(
    config: ProcessingConfig, *, chunks: tuple[int, ...]
) -> dict[str, Any]:
    return {
        "compression": "gzip",
        "compression_opts": config.gzip_level,
        "chunks": chunks,
    }


def _create_data_datasets(
    output: h5py.File,
    length: int,
    config: ProcessingConfig,
    rgb_height: int,
    rgb_width: int,
) -> None:
    numeric_chunk = min(length, 256)
    specs = _expected_specs(length, config.pointcloud.num_points, rgb_height, rgb_width)
    for name, (shape, dtype) in specs.items():
        row_chunk = 1 if name in _FRAME_CHUNKED_DATASETS else numeric_chunk
        output.create_dataset(
            name,
            shape=shape,
            dtype=dtype,
            **_dataset_kwargs(config, chunks=(row_chunk, *shape[1:])),
        )


def _write_attrs(
    output: h5py.File,
    reader: EpisodeReader,
    decision: EpisodeDecision,
    config: ProcessingConfig,
    *,
    task_name: str,
) -> None:
    meta = reader.h5f["meta"].attrs
    output.attrs.update(
        {
            "schema_name": PROCESSED_SCHEMA_NAME,
            "schema_version": PROCESSED_SCHEMA_VERSION,
            "domain": "real",
            "source_path": str(reader.h5_path.resolve()),
            "source_episode": reader.h5_path.name,
            "source_schema_version": int(meta["schema_version"]),
            "source_frames": decision.source_frames,
            "episode_steps": decision.processed_frames,
            "dt": float(reader.timing.grid_dt_s),
            "obs_alignment": "obs[t]_before_action[t]",
            "observation_alignment": "control_step_latest_causal",
            "state_alignment": "control_step",
            "action_semantics": "teleop_published_joint_target",
            "task_name": task_name,
            "fingertip_points_frame": _FINGERTIP_POINTS_FRAME,
            "fingertip_points_unit": _FINGERTIP_POINTS_UNIT,
            "fingertip_points_derivation": FINGERTIP_POINTS_DERIVATION,
            "fingertip_points_policy_id": FINGERTIP_POLICY_ID,
            # Portable numeric FK inputs for the deployment geometry contract;
            # the URDF path stays local and the policy id owns model identity.
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
            "action_ee_frame": _ACTION_EE_FRAME,
            "action_ee_components": "eef_position_m(3)+eef_rot6d(6)+xhand_target_rad(12)",
            "contact_force_source": _CONTACT_FORCE_SOURCE,
            "contact_force_representation": CONTACT_FORCE_REPRESENTATION,
            "contact_force_unit": XHAND_SDK_NATIVE_UNKNOWN_SI_UNIT,
            "contact_force_si_verified": _CONTACT_FORCE_SI_VERIFIED,
            "contact_force_frame": XHAND_SENSOR_NATIVE_AXES_FRAME,
            "tactile_force_representation": TACTILE_FORCE_REPRESENTATION,
            "tactile_force_finger_order": HAND_FINGER_ORDER_ID,
            "tactile_force_sensor_order": TACTILE_FORCE_SENSOR_ORDER,
            "tactile_force_point_order": TACTILE_FORCE_POINT_ORDER,
            "tactile_force_axis_labels": TACTILE_FORCE_AXIS_LABELS,
            "tactile_force_unit": XHAND_SDK_NATIVE_UNKNOWN_SI_UNIT,
            "tactile_force_si_verified": _TACTILE_FORCE_SI_VERIFIED,
            "tactile_force_spatial_geometry_verified": (
                _TACTILE_FORCE_SPATIAL_GEOMETRY_VERIFIED
            ),
            "rgb_transform": "native_color_resolution_no_resize",
            "depth_transform": "depth_to_color_aligned_native_resolution",
            "depth_unit": "sensor_unit",
            "depth_scale_m_per_unit": float(meta["depth_scale"]),
            "depth_invalid_value": 0,
            "camera_intrinsic_semantics": (
                "native_color_intrinsics_for_depth_to_color_aligned_depth"
            ),
            "camera_extrinsic_semantics": (
                "T_xarm_base_from_color;native_color_optical_to_xarm_base"
            ),
            "source_camera_depth_intrinsics_native": np.asarray(
                meta["camera_depth_intrinsics"], dtype=np.float64
            ),
            "source_camera_depth_distortion_model": str(
                meta["camera_depth_distortion_model"]
            ),
            "source_camera_depth_distortion_coeffs": np.asarray(
                meta["camera_depth_distortion_coeffs"], dtype=np.float64
            ),
            "camera_color_distortion_model": str(
                meta["camera_color_distortion_model"]
            ),
            "camera_color_distortion_coeffs": np.asarray(
                meta["camera_color_distortion_coeffs"], dtype=np.float64
            ),
            "camera_T_color_from_depth": np.asarray(
                meta["camera_T_color_from_depth"], dtype=np.float64
            ),
            "point_cloud_frame": "xarm_base",
            "processing_config_json": canonical_json(
                {
                    "pointcloud": config.pointcloud.to_dict(),
                    "table_plane_abcd": (
                        None
                        if config.table_plane_abcd is None
                        else list(config.table_plane_abcd)
                    ),
                }
            ),
            "point_cloud_shape": np.asarray(
                (config.pointcloud.num_points, 6), dtype=np.int64
            ),
            "point_cloud_color_source": POINT_CLOUD_COLOR_SOURCE,
            "point_cloud_policy_id": POINT_CLOUD_POLICY_ID,
            "point_cloud_table_plane_abcd_json": canonical_json(
                None
                if config.table_plane_abcd is None
                else list(config.table_plane_abcd)
            ),
            "point_cloud_sampling": POINT_CLOUD_SAMPLING,
            "point_cloud_transform": POINT_CLOUD_TRANSFORM,
        }
    )


def _write_processed_episode(
    reader: EpisodeReader,
    decision: EpisodeDecision,
    output_root: Path,
    config: ProcessingConfig,
    *,
    task_name: str,
) -> dict[str, Any]:
    path = output_root / f"{reader.h5_path.name}.h5"
    if not decision.accepted:
        raise ValueError("cannot write a rejected episode")
    frames = decision.source_frames
    camera_model = load_raw_episode_camera_model(reader)
    T_xarm_base_from_color = load_raw_episode_base_from_color(reader)
    geometry = camera_model.geometry
    rgb_height = geometry.color.height
    rgb_width = geometry.color.width
    with h5py.File(path, "w") as output:
        _write_attrs(
            output,
            reader,
            decision,
            config,
            task_name=task_name,
        )
        _create_data_datasets(
            output, decision.processed_frames, config, rgb_height, rgb_width
        )
        arm_action = np.asarray(
            reader.h5f["action_arm_joint_sent"][:], dtype=np.float32
        )
        hand_action = np.asarray(reader.h5f["action_hand_joint"][:], dtype=np.float32)
        arm_action_ee = np.asarray(reader.h5f["action_arm_ee"][:], dtype=np.float32)
        joint_state = np.concatenate(
            (
                np.asarray(reader.h5f["arm_qpos"][:], dtype=np.float32),
                np.asarray(reader.h5f["hand_qpos"][:], dtype=np.float32),
            ),
            axis=1,
        )
        output["joint_state"][:] = joint_state
        output["action"][:] = np.concatenate((arm_action, hand_action), axis=1)
        output["action_ee"][:] = np.concatenate((arm_action_ee, hand_action), axis=1)
        # Tactile validity is copied directly from raw; processing never
        # reconstructs measurement truth from calibration/unit/freshness fields.
        output["contact_force"][:] = np.asarray(
            reader.h5f["hand_contact"][:], dtype=np.float32
        )
        output["contact_force_valid"][:] = np.asarray(
            reader.h5f["hand_contact_valid"][:], dtype=bool
        )
        output["tactile_force"][:] = np.asarray(
            reader.h5f["hand_tactile_force"][:], dtype=np.float32
        )
        output["tactile_force_valid"][:] = np.asarray(
            reader.h5f["hand_tactile_force_valid"][:], dtype=bool
        )
        hand_fk = HandKinematics(
            config.hand_urdf_path, list(config.fingertip_link_names)
        )
        if not hand_fk.is_ready():
            raise RuntimeError("processed fingertip FK startup failed")
        eef_pose = compute_eef_pose_history_xarm_base(joint_state[:, :7])
        output["eef_pose"][:] = eef_pose.astype(np.float32)
        output["fingertip_points"][:] = compute_fingertip_history_xarm_base(
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
        output["observation_anchor_monotonic_ns"][:] = reader.h5f[
            "observation_anchor_monotonic_ns"
        ][:]
        output["arm_source_monotonic_ns"][:] = reader.h5f[
            "arm_source_monotonic_ns"
        ][:]
        output["hand_source_monotonic_ns"][:] = reader.h5f[
            "hand_source_monotonic_ns"
        ][:]
        output["camera_source_monotonic_ns"][:] = reader.h5f[
            "camera_source_monotonic_ns"
        ][:]

        # Native aligned RGB-D: store the source color resolution without resize.
        output["camera_intrinsic"][:] = (
            geometry.color.matrix().astype(np.float32).reshape(9)[None, :]
        )
        output["camera_extrinsic"][:] = np.broadcast_to(
            T_xarm_base_from_color,
            output["camera_extrinsic"].shape,
        ).astype(np.float32)
        pointcloud_deriver = RawEpisodePointCloudDeriver(
            reader=reader,
            camera=camera_model,
            T_xarm_base_from_color=T_xarm_base_from_color,
            pointcloud=config.pointcloud,
            table_plane_abcd=config.table_plane_abcd,
        )
        for source_index, frame in enumerate(reader.iter_camera_frames("rgb")):
            if source_index >= frames:
                raise ValueError("RGB contains extra frames")
            output["rgb"][source_index] = frame
            output["depth"][source_index] = np.asarray(
                reader.h5f["depth"][source_index], dtype=np.uint16
            )
            cloud = pointcloud_deriver.derive(source_index, frame)
            if cloud is None:
                raise ValueError(
                    f"derived point cloud empty at source row {source_index}"
                )
            output["point_cloud"][source_index] = cloud
        output.flush()
    return {
        "path": path.name,
        "source_episode": reader.h5_path.name,
        "source_frames": decision.source_frames,
        "frames": decision.processed_frames,
    }


def _rejected_decision(
    episode: Path, config: ProcessingConfig, reason: str
) -> EpisodeDecision:
    return EpisodeDecision(episode, 0, reason)


def _rejection_is_blocking(
    decision: EpisodeDecision,
    annotations: Mapping[str, EpisodeAnnotation],
    *,
    skip_rejected_unannotated: bool,
) -> bool:
    """Keep explicit annotation intent separate from absent annotation metadata."""

    annotation = annotations.get(decision.source_path.name)
    if annotation is None:
        return not skip_rejected_unannotated
    return annotation.include


def process_episode_root(
    input_root: str | Path,
    output_root: str | Path,
    config: ProcessingConfig,
    *,
    annotations_path: str | Path | None = None,
    dry_run: bool = False,
    skip_rejected_unannotated: bool = False,
    task_name: str | None = None,
    expected_task_name: str | None = None,
) -> dict[str, Any]:
    """Publish accepted episodes, preserving explicit annotation intent.

    Rejected unannotated episodes block direct library callers by default. The
    canonical CLI opts into skipping them with ``skip_rejected_unannotated``.
    ``expected_task_name`` is an optional caller-owned output-root invariant;
    direct library callers may leave it unset for arbitrary temporary paths.
    """

    if not isinstance(skip_rejected_unannotated, bool):
        raise TypeError("skip_rejected_unannotated must be boolean")
    episodes = discover_episode_dirs(input_root)
    annotations = load_annotations(annotations_path)
    unknown_annotations = set(annotations) - {episode.name for episode in episodes}
    if unknown_annotations:
        raise ValueError(
            f"annotations reference unknown episodes: {sorted(unknown_annotations)}"
        )
    resolved_task_override = validate_annotation_task_name_override(
        annotations, task_name
    )
    resolved_expected_task_name = (
        None
        if expected_task_name is None
        else validate_processed_task_name(expected_task_name)
    )
    decisions: list[EpisodeDecision] = []
    resolved_episode_task_names: dict[Path, str] = {}
    task_identity_errors: list[str] = []
    for episode in episodes:
        annotation = annotations.get(episode.name)
        if annotation is not None and not annotation.include:
            # Do not require an excluded episode to remain readable. Its
            # exclusion is operator-owned and needs no source inspection.
            decisions.append(
                _rejected_decision(episode, config, "excluded by annotation")
            )
            continue
        analysis_annotation = annotation or EpisodeAnnotation()
        task_name_candidate: Any = _TASK_NAME_CANDIDATE_UNSET
        raw_task_name_requires_validation = False
        # Only explicit behavior/admission decisions become EpisodeDecision
        # results. Technical, source-corruption, and programming failures raise
        # and fail the whole batch: silently training on fewer demonstrations
        # is more dangerous than a loud stop. Known-bad episodes are excluded
        # by the operator through annotation include:false.
        with _open_processing_episode(episode) as reader:
            decision = analyze_episode(
                reader,
                config,
                analysis_annotation,
            )
            if decision.accepted:
                task_name_candidate, raw_task_name_requires_validation = (
                    _task_name_candidate(
                        reader,
                        analysis_annotation,
                        resolved_task_override,
                    )
                )
        decisions.append(decision)
        if task_name_candidate is _TASK_NAME_CANDIDATE_UNSET:
            continue
        if not raw_task_name_requires_validation:
            resolved_episode_task_names[decision.source_path] = task_name_candidate
            continue
        try:
            resolved_episode_task_names[decision.source_path] = (
                validate_processed_task_name(task_name_candidate)
            )
        except (TypeError, ValueError) as exc:
            task_identity_errors.append(f"{episode.name}: {type(exc).__name__}: {exc}")
    if task_identity_errors:
        raise ValueError(
            "processed task identity invalid; no output published: "
            + "; ".join(task_identity_errors)
        )
    accepted_task_names = {
        resolved_episode_task_names[decision.source_path]
        for decision in decisions
        if decision.accepted
    }
    if len(accepted_task_names) > 1:
        raise ValueError(
            "processed batch has multiple task_name values; no output published: "
            + ", ".join(sorted(accepted_task_names))
        )
    resolved_batch_task_name = next(iter(accepted_task_names), None)
    if (
        resolved_batch_task_name is not None
        and resolved_expected_task_name is not None
        and resolved_batch_task_name != resolved_expected_task_name
    ):
        raise ValueError(
            "processed task_name does not match the expected output task identity; "
            f"no output published: {resolved_batch_task_name!r} != "
            f"{resolved_expected_task_name!r}"
        )
    planned_names = [
        f"{decision.source_path.name}.h5" for decision in decisions if decision.accepted
    ]
    collisions = sorted(
        name for name, count in Counter(planned_names).items() if count > 1
    )
    if collisions:
        raise ValueError(f"source names produce colliding outputs: {collisions}")
    report: dict[str, Any] = {
        "schema_name": PROCESSED_SCHEMA_NAME,
        "schema_version": PROCESSED_SCHEMA_VERSION,
        "input_root": str(Path(input_root).resolve()),
        "output_root": str(Path(output_root).resolve()),
        "dry_run": dry_run,
        "config": config.to_dict(),
        "source_episode_count": len(decisions),
        "accepted_source_episode_count": sum(d.accepted for d in decisions),
        "rejected_source_episode_count": sum(not d.accepted for d in decisions),
        "output_episode_count": sum(d.accepted for d in decisions),
        "source_frame_count": sum(d.source_frames for d in decisions),
        "processed_frame_count": sum(d.processed_frames for d in decisions),
        "episodes": [d.to_dict() for d in decisions],
        "outputs": [],
    }
    if dry_run:
        return report
    blocking_rejections = [
        decision
        for decision in decisions
        if not decision.accepted
        and _rejection_is_blocking(
            decision,
            annotations,
            skip_rejected_unannotated=skip_rejected_unannotated,
        )
    ]
    if blocking_rejections:
        details = "; ".join(
            f"{decision.source_path.name}: {decision.rejected_reason}"
            for decision in blocking_rejections
        )
        raise ValueError(f"processing batch rejected; no output published: {details}")
    if not any(decision.accepted for decision in decisions):
        raise ValueError("processing produced no included episodes")
    target = Path(output_root)
    if target.exists():
        raise FileExistsError(
            f"refusing to overwrite existing processed root: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    try:
        outputs: list[dict[str, Any]] = []
        for decision in decisions:
            if not decision.accepted:
                continue
            with _open_processing_episode(decision.source_path) as reader:
                outputs.append(
                    _write_processed_episode(
                        reader,
                        decision,
                        staging,
                        config,
                        task_name=resolved_episode_task_names[decision.source_path],
                    )
                )
        validation = [
            validate_processed_hdf5(staging / item["path"], config) for item in outputs
        ]
        report["outputs"] = outputs
        report["validation"] = validation
        atomic_publish(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report
