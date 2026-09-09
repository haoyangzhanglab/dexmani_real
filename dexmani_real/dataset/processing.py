"""Transactional depth-to-color aligned raw-v25 to processed-v14 processing."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import yaml

from dexmani_real.dataset.clean import (
    analyze_episode,
    select_tactile_rows_to_references,
)
from dexmani_real.dataset.contracts import (
    EpisodeAnnotation,
    EpisodeDecision,
    OutputProfile,
    ProcessingConfig,
    validate_processed_task_name,
)
from dexmani_real.dataset.pointcloud import (
    RawEpisodePointCloudDeriver,
    load_raw_episode_base_from_color,
    load_raw_episode_camera_model,
)
from dexmani_real.dataset.processed import (
    _ACTION_EE_FRAME,
    _CONTACT_FORCE_FRAME,
    _CONTACT_FORCE_SI_VERIFIED,
    _CONTACT_FORCE_UNIT,
    _FINGERTIP_POINTS_FRAME,
    _FINGERTIP_POINTS_UNIT,
    _FRAME_CHUNKED_DATASETS,
    _TACTILE_FORCE_AXIS_LABELS,
    _TACTILE_FORCE_POINT_ORDER,
    _TACTILE_FORCE_REPRESENTATION,
    _TACTILE_FORCE_SENSOR_ORDER,
    _TACTILE_FORCE_SI_VERIFIED,
    _TACTILE_FORCE_SPATIAL_GEOMETRY_VERIFIED,
    _TACTILE_FORCE_UNIT,
    PROCESSED_SCHEMA_NAME,
    PROCESSED_SCHEMA_VERSION,
    _dataset_row_slices,
    _expected_specs,
    _json,
    _validate_processed_output_structure,
    validate_processed_hdf5,
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
from dexmani_real.recording.storage.reader import EpisodeReader
from dexmani_real.sensor.camera.transforms import (
    resize_camera_intrinsic,
    resize_depth,
    resize_rgb,
)
from dexmani_real.sensor.pointcloud import (
    POINT_CLOUD_COLOR_SOURCE,
    POINT_CLOUD_POLICY_ID,
    POINT_CLOUD_SAMPLING,
    POINT_CLOUD_TRANSFORM,
)
from dexmani_real.utils.atomic_io import atomic_publish
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

# Source-facing failures from HDF5/MP4 reading and deterministic cleaning reject
# only the affected episode. Programming errors and process-control exceptions
# remain fatal to the batch.
_ANALYSIS_REJECTION_EXCEPTIONS = (
    FileNotFoundError,
    OSError,
    ValueError,
    KeyError,
    RuntimeError,
    IndexError,
)
_TASK_NAME_CANDIDATE_UNSET = object()


def _derive_depth_valid_mask(reader: EpisodeReader) -> np.ndarray:
    depth = reader.h5f["depth"]
    frame_count = int(reader.h5f["meta"].attrs["num_frames"])
    if depth.shape[0] != frame_count:
        raise ValueError("depth length does not match source grid")
    valid = np.zeros(frame_count, dtype=bool)
    if depth.ndim != 3:
        return valid
    for row_slice in _dataset_row_slices(depth):
        block = np.asarray(depth[row_slice], dtype=np.uint16)
        valid[row_slice] = np.any(block > 0, axis=(1, 2))
    return valid


def _parse_ranges(value: Any, *, label: str) -> tuple[tuple[int, int], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of [start, end] ranges")
    result: list[tuple[int, int]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{label} entries must be [start, end]")
        if any(not isinstance(bound, int) or isinstance(bound, bool) for bound in item):
            raise ValueError(f"{label} bounds must be integers")
        result.append((item[0], item[1]))
    return tuple(result)


def load_annotations(path: str | Path | None) -> dict[str, EpisodeAnnotation]:
    """Load explicit row/task overrides; outcome labels are not accepted."""

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
    allowed = {"include", "task_name", "include_ranges", "exclude_ranges"}
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
            include_ranges=_parse_ranges(
                raw.get("include_ranges"), label=f"{episode_name}.include_ranges"
            ),
            exclude_ranges=_parse_ranges(
                raw.get("exclude_ranges"), label=f"{episode_name}.exclude_ranges"
            ),
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


# Reserved subdirectory names under a task root that are never raw episode
# directories.  ``process_log`` holds per-episode process reports written into
# the processed output root, so it must not be rediscovered as an episode.
_NON_EPISODE_DIR_NAMES = frozenset({"process_log"})


def discover_episode_dirs(input_root: str | Path) -> tuple[Path, ...]:
    """Accept either one episode directory or a task directory of episodes.

    Non-hidden subdirectories of a task root are treated as episodes, except
    reserved names (``process_log``) that hold processed-output sidecars rather
    than raw ``data.h5`` sources. Returned paths are absolute so persisted source
    provenance remains independent of the caller's working directory.
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
            if child.is_dir()
            and not child.name.startswith(".")
            and child.name not in _NON_EPISODE_DIR_NAMES
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
    output: h5py.File, length: int, config: ProcessingConfig
) -> None:
    numeric_chunk = min(length, 256)
    specs = _expected_specs(length, config)
    for name in config.profile.dataset_keys:
        shape, dtype = specs[name]
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
    visual_profile = config.profile.needs_rgb or config.profile.needs_pointcloud
    output.attrs.update(
        {
            "schema_name": PROCESSED_SCHEMA_NAME,
            "schema_version": PROCESSED_SCHEMA_VERSION,
            "domain": "real",
            "source_episode": reader.h5_path.name,
            "source_frames": decision.source_frames,
            "profile": config.profile.value,
            "episode_steps": decision.selected_frames,
            "dt": float(reader.timing.grid_dt_s),
            "time_semantics": "logical_control_grid_after_row_compaction",
            "source_contiguity": "segment_ends_in_provenance",
            "source_contiguity_tolerance_s": max(
                1e-7,
                float(reader.timing.grid_dt_s) * config.grid_dt_relative_tolerance,
            ),
            "action_dim": 19,
            "action_ee_dim": 21,
            "action_space": "joint",
            "obs_alignment": "obs[t]_before_action[t]",
            "observation_reference": (
                "camera_source_monotonic_ns"
                if visual_profile
                else "grid_anchor_monotonic_ns"
            ),
            "state_alignment": (
                "camera_source_aligned_state"
                if visual_profile
                else "control_grid_state"
            ),
            "max_observation_skew_s": config.max_observation_skew_s,
            "action_semantics": "teleop_published_joint_target",
            "task_name": task_name,
            "point_cloud_frame": (
                "xarm_base" if config.profile.needs_pointcloud else "omitted"
            ),
            "fingertip_points_frame": _FINGERTIP_POINTS_FRAME,
            "fingertip_points_unit": _FINGERTIP_POINTS_UNIT,
            "fingertip_points_derivation": FINGERTIP_POINTS_DERIVATION,
            "fingertip_points_policy_id": FINGERTIP_POLICY_ID,
            "action_ee_frame": _ACTION_EE_FRAME,
            "action_ee_components": "eef_position_m(3)+eef_rot6d(6)+xhand_target_rad(12)",
            "eef_pose_frame": EEF_POSE_FRAME,
            "eef_pose_components": EEF_POSE_COMPONENTS,
            "eef_pose_derivation": EEF_POSE_DERIVATION,
            "eef_pose_algorithm_id": EEF_POSE_ALGORITHM_ID,
            "contact_force_source": (
                "camera_causal_tactile_sum"
                if visual_profile
                else "control_grid_tactile_sum"
            ),
            "contact_force_alignment": (
                "newest_source_not_after_camera_within_max_observation_skew"
                if visual_profile
                else "newest_source_not_after_grid_within_max_observation_skew"
            ),
            "contact_force_unit": _CONTACT_FORCE_UNIT,
            "contact_force_si_verified": _CONTACT_FORCE_SI_VERIFIED,
            "contact_force_frame": _CONTACT_FORCE_FRAME,
            "contact_force_fresh_required": True,
            "contact_force_calibrated_required": True,
            "contact_force_unit_code": 0,
            "contact_force_causal_to_reference": True,
            "contact_force_hand_source_match_required": True,
            # Full tactile records only source-provable SDK facts; SI units and
            # taxel spatial geometry stay explicitly unverified.
            "tactile_force_representation": _TACTILE_FORCE_REPRESENTATION,
            "tactile_force_sensor_order": _TACTILE_FORCE_SENSOR_ORDER,
            "tactile_force_point_order": _TACTILE_FORCE_POINT_ORDER,
            "tactile_force_axis_labels": _TACTILE_FORCE_AXIS_LABELS,
            "tactile_force_unit": _TACTILE_FORCE_UNIT,
            "tactile_force_si_verified": _TACTILE_FORCE_SI_VERIFIED,
            "tactile_force_spatial_geometry_verified": (
                _TACTILE_FORCE_SPATIAL_GEOMETRY_VERIFIED
            ),
            "tactile_force_fresh_required": True,
            "tactile_force_calibrated_required": True,
            "tactile_force_unit_code": 0,
            "tactile_force_causal_to_reference": True,
            "tactile_force_hand_source_match_required": True,
            "processing_config_json": _json(config.to_dict()),
            "quality_summary_json": _json(decision.quality),
            "source_decision_json": _json(decision.to_dict()),
        }
    )
    if config.profile.needs_rgb:
        output.attrs.update(
            {
                "rgb_transform": "resize_no_crop",
                "depth_transform": "depth_to_color_aligned_resize_no_crop_nearest",
                "depth_unit": "sensor_unit",
                "depth_scale_m_per_unit": float(meta["depth_scale"]),
                "depth_invalid_value": 0,
                "camera_intrinsic_semantics": (
                    "resized_color_intrinsics_for_depth_to_color_aligned_depth"
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
            }
        )
    if config.profile.needs_pointcloud:
        output.attrs.update(
            {
                "point_cloud_shape": np.asarray(
                    (config.pointcloud.num_points, 6), dtype=np.int64
                ),
                "point_cloud_color_source": POINT_CLOUD_COLOR_SOURCE,
                "point_cloud_policy_id": POINT_CLOUD_POLICY_ID,
                "point_cloud_table_plane_abcd_json": _json(
                    None
                    if config.table_plane_abcd is None
                    else list(config.table_plane_abcd)
                ),
                "point_cloud_sampling": POINT_CLOUD_SAMPLING,
                "point_cloud_transform": POINT_CLOUD_TRANSFORM,
            }
        )


def _processed_joint_state(
    reader: EpisodeReader,
    selected: np.ndarray,
    config: ProcessingConfig,
) -> np.ndarray:
    """Read the state representation selected by the processed profile."""
    visual_profile = config.profile.needs_rgb or config.profile.needs_pointcloud
    if visual_profile:
        arm_key = "policy_observation_arm_qpos"
        hand_key = "policy_observation_hand_qpos"
    else:
        arm_key = "arm_qpos"
        hand_key = "hand_qpos"
    arm_state = np.asarray(reader.h5f[arm_key][selected], dtype=np.float32)
    hand_state = np.asarray(reader.h5f[hand_key][selected], dtype=np.float32)
    return np.concatenate((arm_state, hand_state), axis=1)


def _gather_dataset_rows(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    """Gather raw rows in ``indices`` order, safe for duplicate indices.

    Causal forward-fill repeats raw source rows, and h5py fancy indexing must
    not be relied on for duplicate point selections; unique+inverse keeps the
    read both correct and minimal.
    """
    unique_rows, inverse = np.unique(indices, return_inverse=True)
    values = np.asarray(dataset[unique_rows])
    return values[inverse]


def _write_processed_episode(
    reader: EpisodeReader,
    decision: EpisodeDecision,
    output_root: Path,
    config: ProcessingConfig,
    *,
    task_name: str,
) -> dict[str, Any]:
    path = output_root / f"{reader.h5_path.name}.h5"
    selected = decision.selected_indices
    camera_model = None
    T_xarm_base_from_color = None
    if config.profile.needs_rgb or config.profile.needs_pointcloud:
        # Resolve the raw RGB-D geometry boundary before creating output.
        camera_model = load_raw_episode_camera_model(reader)
        T_xarm_base_from_color = load_raw_episode_base_from_color(reader)
    with h5py.File(path, "w") as output:
        _write_attrs(
            output,
            reader,
            decision,
            config,
            task_name=task_name,
        )
        _create_data_datasets(output, decision.selected_frames, config)
        arm_action = np.asarray(
            reader.h5f["action_arm_joint_sent"][selected], dtype=np.float32
        )
        hand_action = np.asarray(
            reader.h5f["action_hand_joint"][selected], dtype=np.float32
        )
        arm_action_ee = np.asarray(
            reader.h5f["action_arm_ee"][selected], dtype=np.float32
        )
        joint_state = _processed_joint_state(reader, selected, config)
        output["joint_state"][:] = joint_state
        output["action"][:] = np.concatenate((arm_action, hand_action), axis=1)
        output["action_ee"][:] = np.concatenate((arm_action_ee, hand_action), axis=1)
        visual_profile = config.profile.needs_rgb or config.profile.needs_pointcloud
        reference_key = (
            "camera_source_monotonic_ns"
            if visual_profile
            else "observation_anchor_monotonic_ns"
        )
        tactile_source_rows = select_tactile_rows_to_references(
            np.asarray(reader.h5f["hand_source_monotonic_ns"][:], dtype=np.int64),
            np.asarray(reader.h5f["tactile_source_monotonic_ns"][:], dtype=np.int64),
            np.asarray(reader.h5f["tactile_fresh"][:], dtype=bool),
            np.asarray(reader.h5f["tactile_calibrated"][:], dtype=bool),
            np.asarray(reader.h5f["tactile_unit_code"][:], dtype=np.int64),
            np.asarray(reader.h5f[reference_key][:], dtype=np.int64),
            max_observation_skew_s=config.max_observation_skew_s,
        )
        selected_tactile_rows = tactile_source_rows[selected]
        if np.any(selected_tactile_rows < 0):
            raise ValueError("selected row lacks causal tactile provenance")
        # contact_force and tactile_force are gathered from the identical
        # selected raw rows; only those T rows are read, never the full
        # (N,5,120,3) source dataset.
        output["contact_force"][:] = _gather_dataset_rows(
            reader.h5f["hand_contact"], selected_tactile_rows
        ).astype(np.float32, copy=False)
        output["tactile_force"][:] = _gather_dataset_rows(
            reader.h5f["hand_tactile_force"], selected_tactile_rows
        ).astype(np.float32, copy=False)
        # Exactly one canonical Arm FK per processed row; the same EEF history
        # feeds both eef_pose and fingertip_points.
        eef_pose_history = compute_eef_pose_history_xarm_base(joint_state[:, :7])
        output["eef_pose"][:] = eef_pose_history.astype(np.float32, copy=False)
        hand_fk = HandKinematics(
            config.hand_urdf_path, list(config.fingertip_link_names)
        )
        if not hand_fk.is_ready():
            raise RuntimeError("processed fingertip FK startup failed")
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
            eef_pose_history=eef_pose_history,
        )

        provenance = output.create_group("provenance")
        provenance.attrs["drop_reason_bit_names_json"] = _json(
            {str(bit): name for bit, name in enumerate(decision.drop_reason_names)}
        )
        provenance_values = {
            "source_row_index": selected,
            "source_sample_index": np.asarray(
                reader.h5f["source_sample_index"][selected], dtype=np.int64
            ),
            "source_timestamp_s": np.asarray(
                reader.h5f["timestamp"][selected], dtype=np.float64
            ),
            "source_segment_ends": decision.segment_ends,
            "source_keep_mask": decision.keep_mask,
            "source_drop_reason_bits": decision.drop_reason_bits,
            "tactile_source_row_index": np.asarray(
                selected_tactile_rows, dtype=np.int64
            ),
            "observation_reference_monotonic_ns": np.asarray(
                reader.h5f[reference_key][selected], dtype=np.int64
            ),
            "tactile_source_monotonic_ns": _gather_dataset_rows(
                reader.h5f["tactile_source_monotonic_ns"], selected_tactile_rows
            ).astype(np.int64, copy=False),
        }
        for name, values in provenance_values.items():
            provenance.create_dataset(
                name,
                data=values,
                **_dataset_kwargs(config, chunks=(min(len(values), 256),)),
            )

        if config.profile.needs_rgb:
            assert camera_model is not None
            assert T_xarm_base_from_color is not None
            geometry = camera_model.geometry
            camera_k = resize_camera_intrinsic(
                geometry.color.matrix(),
                source_height=geometry.color.height,
                source_width=geometry.color.width,
                target_height=config.target_rgb_height,
                target_width=config.target_rgb_width,
            )
            output["camera_intrinsic"][:] = camera_k[None, :]
            output["camera_extrinsic"][:] = np.broadcast_to(
                T_xarm_base_from_color,
                output["camera_extrinsic"].shape,
            ).astype(np.float32)
            for target_index, source_index in enumerate(selected):
                depth = np.asarray(reader.h5f["depth"][source_index], dtype=np.uint16)
                output["depth"][target_index] = resize_depth(
                    depth,
                    height=config.target_rgb_height,
                    width=config.target_rgb_width,
                )

        if config.profile.needs_pointcloud:
            assert camera_model is not None
            assert T_xarm_base_from_color is not None
            pointcloud_deriver = RawEpisodePointCloudDeriver(
                reader=reader,
                camera=camera_model,
                T_xarm_base_from_color=T_xarm_base_from_color,
                pointcloud=config.pointcloud,
                table_plane_abcd=config.table_plane_abcd,
            )
        else:
            pointcloud_deriver = None
        if config.profile.needs_rgb or config.profile.needs_pointcloud:
            target_by_source = {
                int(source_index): target_index
                for target_index, source_index in enumerate(selected)
            }
            decoded_count = 0
            written_count = 0
            for source_index, frame in enumerate(reader.iter_camera_frames("rgb")):
                decoded_count += 1
                target_row = target_by_source.get(source_index)
                if target_row is None:
                    continue
                if config.profile.needs_rgb:
                    output["rgb"][target_row] = resize_rgb(
                        frame,
                        height=config.target_rgb_height,
                        width=config.target_rgb_width,
                    )
                if pointcloud_deriver is not None:
                    cloud = pointcloud_deriver.derive(source_index, frame)
                    if cloud is None:
                        raise ValueError(
                            f"derived point cloud empty at source row {source_index}"
                        )
                    output["point_cloud"][target_row] = cloud
                written_count += 1
            if decoded_count != decision.source_frames or written_count != len(
                selected
            ):
                raise ValueError(
                    f"RGB alignment mismatch decoded={decoded_count}, written={written_count}, "
                    f"source={decision.source_frames}, selected={len(selected)}"
                )
        output.flush()
    return {
        "path": path.name,
        "source_episode": reader.h5_path.name,
        "source_frames": decision.source_frames,
        "frames": decision.selected_frames,
        "dropped_frames": decision.source_frames - decision.selected_frames,
        "full_window_count": decision.quality["full_window_count"],
    }


def _rejected_decision(
    episode: Path, config: ProcessingConfig, reason: str
) -> EpisodeDecision:
    return EpisodeDecision(
        source_path=episode,
        source_frames=0,
        profile=config.profile,
        selected_indices=np.empty(0, dtype=np.int64),
        keep_mask=np.empty(0, dtype=bool),
        drop_reason_bits=np.empty(0, dtype=np.uint64),
        drop_reason_names=(),
        hard_reason_counts={},
        boundary_counts={},
        selected_frames=0,
        quality={},
        rejected_reason=reason,
    )


def _invalid_frames_report(
    decisions: list[EpisodeDecision], *, task_name: str
) -> dict[str, Any]:
    """Build the concise operator report for genuinely invalid source rows."""

    episodes: list[dict[str, Any]] = []
    for decision in decisions:
        summary = decision.to_dict()
        invalid_count = int(summary["hard_invalid_frame_count"])
        if invalid_count == 0:
            continue
        episodes.append(
            {
                "episode": decision.source_path.name,
                "invalid_frame_count": invalid_count,
                "invalid_ranges": summary["hard_invalid_ranges"],
                "reasons": summary["hard_invalid_reasons"],
            }
        )
    return {
        "schema_name": "dexmani-real-invalid-frames-report",
        "schema_version": 1,
        "task_name": task_name,
        "episodes": episodes,
    }


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
    verify_output: bool = False,
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
        try:
            with EpisodeReader(episode) as reader:
                decision = analyze_episode(
                    reader,
                    config,
                    analysis_annotation,
                    depth_valid_mask=(
                        _derive_depth_valid_mask(reader)
                        if (
                            config.profile.needs_rgb
                            or config.profile.needs_pointcloud
                        )
                        else None
                    ),
                    source_already_validated=True,
                )
                if decision.accepted:
                    task_name_candidate, raw_task_name_requires_validation = (
                        _task_name_candidate(
                            reader,
                            analysis_annotation,
                            resolved_task_override,
                        )
                    )
        except _ANALYSIS_REJECTION_EXCEPTIONS as exc:
            logger.warning("episode analysis rejected %s", episode, exc_info=True)
            decisions.append(
                _rejected_decision(episode, config, f"{type(exc).__name__}: {exc}")
            )
            continue
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
            task_identity_errors.append(
                f"{episode.name}: {type(exc).__name__}: {exc}"
            )
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
        "selected_frame_count": sum(d.selected_frames for d in decisions),
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
            with EpisodeReader(decision.source_path) as reader:
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
            _validate_processed_output_structure(staging / item["path"], config)
            for item in outputs
        ]
        verification = (
            [
                validate_processed_hdf5(staging / item["path"], config)
                for item in outputs
            ]
            if verify_output
            else None
        )
        assert resolved_batch_task_name is not None
        invalid_report = _invalid_frames_report(
            decisions,
            task_name=resolved_batch_task_name,
        )
        process_log = staging / "process_log"
        process_log.mkdir()
        with (process_log / "invalid_frames_report.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(invalid_report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        report["outputs"] = outputs
        report["validation"] = validation
        if verification is not None:
            report["verification"] = verification
        report["invalid_frames_report"] = invalid_report
        atomic_publish(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report
