"""Transactional depth-to-color aligned raw-v24 to processed-v12 processing."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from dexmani_real.dataset.clean import (
    align_tactile_sum_rows_to_references,
    analyze_episode,
)
from dexmani_real.dataset.contracts import (
    EpisodeAnnotation,
    EpisodeDecision,
    OutputProfile,
    ProcessingConfig,
)
from dexmani_real.dataset.pointcloud import (
    RawEpisodePointCloudDeriver,
    load_raw_episode_base_from_color,
    load_raw_episode_camera_model,
)
from dexmani_real.dataset.processed import (
    PROCESSED_SCHEMA_NAME,
    PROCESSED_SCHEMA_VERSION,
    _ACTION_EE_FRAME,
    _CONTACT_FORCE_FRAME,
    _CONTACT_FORCE_SI_VERIFIED,
    _CONTACT_FORCE_UNIT,
    _FRAME_CHUNKED_DATASETS,
    _SOURCE_MEMBERS,
    _FINGERTIP_POINTS_FRAME,
    _FINGERTIP_POINTS_UNIT,
    _dataset_row_slices,
    _expected_specs,
    _json,
    _validate_processed_output_structure,
    validate_processed_hdf5,
)
from dexmani_real.planning.kinematics.arm_fk import make_arm_fk
from dexmani_real.planning.kinematics.fingertip import compute_fingertip_points_xarm_base
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
from dexmani_real.utils.atomic_io import atomic_publish, sha256_file
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


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
    annotation: EpisodeAnnotation,
) -> None:
    meta = reader.h5f["meta"].attrs
    task_name = (
        annotation.task_name or str(meta.get("task_label", "")).strip() or "unknown"
    )
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
            "action_ee_frame": _ACTION_EE_FRAME,
            "action_ee_components": "eef_position_m(3)+eef_rot6d(6)+xhand_target_rad(12)",
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
            "processing_config_json": _json(config.to_dict()),
            "quality_summary_json": _json(decision.quality),
            "source_decision_json": _json(decision.to_dict()),
            "source_member_sha256_json": _json(
                {
                    member: sha256_file(reader.h5_path / member)
                    for member in _SOURCE_MEMBERS
                }
            ),
            "source_resolved_config_sha256": str(
                meta.get("resolved_config_sha256", "unknown")
            ),
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
                "point_cloud_config_sha256": config.pointcloud.sha256,
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


def compute_fingertip_history_xarm_base(
    arm_qpos: np.ndarray,
    hand_qpos: np.ndarray,
    *,
    arm_fk: Any,
    hand_fk: Any,
    handbase_position_eef_m: np.ndarray,
    handbase_quat_eef_wxyz: np.ndarray,
) -> np.ndarray:
    """Recompute camera/control-grid aligned fingertip history from joint state."""
    arm = np.asarray(arm_qpos, dtype=np.float64)
    hand_state = np.asarray(hand_qpos, dtype=np.float64)
    if arm.ndim != 2 or arm.shape[1] != 7 or hand_state.shape != (len(arm), 12):
        raise ValueError("aligned arm/hand qpos histories have invalid shapes")
    return np.asarray(
        [
            compute_fingertip_points_xarm_base(
                arm[index],
                hand_state[index],
                arm_fk=arm_fk,
                hand_fk=hand_fk,
                handbase_position_eef_m=handbase_position_eef_m,
                handbase_quat_eef_wxyz=handbase_quat_eef_wxyz,
            )
            for index in range(len(arm))
        ],
        dtype=np.float32,
    )


def _write_processed_episode(
    reader: EpisodeReader,
    decision: EpisodeDecision,
    output_root: Path,
    config: ProcessingConfig,
    annotation: EpisodeAnnotation,
) -> dict[str, Any]:
    path = output_root / f"{reader.h5_path.name}.h5"
    selected = decision.selected_indices
    camera_model = None
    T_xarm_base_from_color = None
    if config.profile.needs_rgb or config.profile.needs_pointcloud:
        # Resolve the raw-v24 geometry boundary before creating output.
        camera_model = load_raw_episode_camera_model(reader)
        T_xarm_base_from_color = load_raw_episode_base_from_color(reader)
    with h5py.File(path, "w") as output:
        _write_attrs(output, reader, decision, config, annotation)
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
        output["joint_state"][:] = _processed_joint_state(reader, selected, config)
        output["action"][:] = np.concatenate((arm_action, hand_action), axis=1)
        output["action_ee"][:] = np.concatenate((arm_action_ee, hand_action), axis=1)
        visual_profile = config.profile.needs_rgb or config.profile.needs_pointcloud
        all_contact_force = np.asarray(reader.h5f["hand_contact"][:], dtype=np.float64)
        reference_key = (
            "camera_source_monotonic_ns"
            if visual_profile
            else "observation_anchor_monotonic_ns"
        )
        tactile_source_rows = align_tactile_sum_rows_to_references(
            all_contact_force,
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
        contact_force = np.asarray(
            all_contact_force[selected_tactile_rows], dtype=np.float32
        )
        output["contact_force"][:] = contact_force
        if config.profile.needs_rgb or config.profile.needs_pointcloud:
            hand_fk = HandKinematics(
                config.hand_urdf_path, list(config.fingertip_link_names)
            )
            if not hand_fk.is_ready():
                raise RuntimeError("processed fingertip FK startup failed")
            output["fingertip_points"][:] = compute_fingertip_history_xarm_base(
                np.asarray(
                    reader.h5f["policy_observation_arm_qpos"][selected],
                    dtype=np.float64,
                ),
                np.asarray(
                    reader.h5f["policy_observation_hand_qpos"][selected],
                    dtype=np.float64,
                ),
                arm_fk=make_arm_fk(),
                hand_fk=hand_fk,
                handbase_position_eef_m=np.asarray(
                    config.handbase_position_eef_m, dtype=np.float64
                ),
                handbase_quat_eef_wxyz=np.asarray(
                    config.handbase_quat_eef_wxyz, dtype=np.float64
                ),
            )
        else:
            output["fingertip_points"][:] = np.asarray(
                reader.h5f["hand_fingertip"][selected], dtype=np.float32
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


def process_episode_root(
    input_root: str | Path,
    output_root: str | Path,
    config: ProcessingConfig,
    *,
    annotations_path: str | Path | None = None,
    dry_run: bool = False,
    verify_output: bool = False,
) -> dict[str, Any]:
    """Publish a complete one-to-one batch, or publish nothing on rejection."""

    episodes = discover_episode_dirs(input_root)
    annotations = load_annotations(annotations_path)
    unknown_annotations = set(annotations) - {episode.name for episode in episodes}
    if unknown_annotations:
        raise ValueError(
            f"annotations reference unknown episodes: {sorted(unknown_annotations)}"
        )
    decisions: list[EpisodeDecision] = []
    for episode in episodes:
        annotation = annotations.get(episode.name, EpisodeAnnotation())
        try:
            with EpisodeReader(episode) as reader:
                decisions.append(
                    analyze_episode(
                        reader,
                        config,
                        annotation,
                        depth_valid_mask=(
                            _derive_depth_valid_mask(reader)
                            if (
                                config.profile.needs_rgb
                                or config.profile.needs_pointcloud
                            )
                            and annotation.include
                            else None
                        ),
                        source_already_validated=True,
                    )
                )
        except (FileNotFoundError, OSError, ValueError) as exc:
            logger.warning("episode analysis rejected %s", episode, exc_info=True)
            decisions.append(
                _rejected_decision(episode, config, f"{type(exc).__name__}: {exc}")
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
        and annotations.get(decision.source_path.name, EpisodeAnnotation()).include
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
            annotation = annotations.get(decision.source_path.name, EpisodeAnnotation())
            with EpisodeReader(decision.source_path) as reader:
                outputs.append(
                    _write_processed_episode(
                        reader, decision, staging, config, annotation
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
        with h5py.File(staging / outputs[0]["path"], "r") as first_output:
            task_name = str(first_output.attrs["task_name"])
        invalid_report = _invalid_frames_report(decisions, task_name=task_name)
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
