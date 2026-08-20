"""Transactional one-episode-to-one-HDF5 processing for Real schema v17/v18."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
import yaml

from dexmani_real.data_processing.cleaning import analyze_episode
from dexmani_real.data_processing.contracts import (
    EpisodeAnnotation,
    EpisodeDecision,
    ProcessingConfig,
)
from dexmani_real.data_processing.transforms import (
    resize_camera_intrinsic,
    resize_depth,
    resize_point_cloud,
    resize_rgb,
)
from dexmani_real.planning.pose_utils import quat_wxyz_to_rotmat, rot6d_to_quat_wxyz
from dexmani_real.recording.episode_reader import EpisodeReader
from dexmani_real.recording.episode_schema import EPISODE_SCHEMA_VERSION
from dexmani_real.recording.transaction import atomic_publish
from dexmani_real.sensor.pointcloud_processor import (
    PointCloudProcessor,
    PointCloudProcessorConfig,
)
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.pointcloud_utils import depth_to_meters, make_rays

logger = get_logger(__name__)

PROCESSED_SCHEMA_NAME = "dexmani-real-processed-hdf5"
PROCESSED_SCHEMA_VERSION = 3
_SOURCE_MEMBERS = ("data.h5", "depth.h5", "rgb.mp4")
_VALIDATION_CHUNK_BYTES = 64 * 1024 * 1024


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_transform(transform: np.ndarray, *, label: str) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must be finite")
    if not np.allclose(value[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8, rtol=0.0):
        raise ValueError(f"{label} must be homogeneous")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0.0):
        raise ValueError(f"{label} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError(f"{label} rotation determinant must be +1")
    return value


def _camera_transform_for_index(reader: EpisodeReader, source_index: int) -> np.ndarray:
    meta = reader.h5f["meta"].attrs
    camera_type = str(meta.get("camera_type", ""))
    if camera_type == "eye_to_hand":
        return _validate_transform(
            np.asarray(meta["camera_T_world_camera"]),
            label="camera_T_world_camera",
        )
    if camera_type != "eye_in_hand":
        raise ValueError(f"unsupported camera_type {camera_type!r}")
    t_eef_camera = _validate_transform(
        np.asarray(meta["camera_T_eef_camera"]), label="camera_T_eef_camera"
    )
    arm_ee = np.asarray(reader.h5f["arm_ee"][source_index], dtype=np.float64)
    if arm_ee.shape != (9,) or not np.all(np.isfinite(arm_ee)):
        raise ValueError(f"arm_ee[{source_index}] must be finite (9,)")
    t_base_eef = np.eye(4, dtype=np.float64)
    t_base_eef[:3, :3] = quat_wxyz_to_rotmat(rot6d_to_quat_wxyz(arm_ee[3:]))
    t_base_eef[:3, 3] = arm_ee[:3]
    return _validate_transform(t_base_eef @ t_eef_camera, label="T_xarm_base_camera")


@dataclass
class _PointCloudDeriver:
    """Resolved deterministic point-cloud state for one source."""

    reader: EpisodeReader
    processor_config: PointCloudProcessorConfig
    rays: np.ndarray
    depth_scale: float
    target_point_count: int
    static_processor: PointCloudProcessor | None

    @classmethod
    def from_reader(
        cls, reader: EpisodeReader, config: ProcessingConfig
    ) -> _PointCloudDeriver:
        meta = reader.h5f["meta"].attrs
        source_height = int(meta["camera_encoding_height"])
        source_width = int(meta["camera_encoding_width"])
        if source_height <= 0 or source_width <= 0:
            raise ValueError("camera encoding dimensions must be positive")
        source_k = np.asarray(meta["camera_K"], dtype=np.float64).reshape(3, 3)
        if not np.all(np.isfinite(source_k)) or not np.allclose(
            source_k[2], (0.0, 0.0, 1.0), rtol=0.0, atol=1e-9
        ):
            raise ValueError("camera_K must be a finite canonical pinhole matrix")
        raw_pc_meta = meta.get("camera_pointcloud_config_json", "")
        if isinstance(raw_pc_meta, bytes):
            raw_pc_meta = raw_pc_meta.decode("utf-8")
        try:
            pc_meta = json.loads(str(raw_pc_meta))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "camera_pointcloud_config_json must be valid JSON"
            ) from exc
        if not isinstance(pc_meta, dict):
            raise ValueError("camera_pointcloud_config_json must encode an object")
        optional_keys = {"pc_desk_plane", "pc_desk_clearance_m"}
        required_keys = {
            f"pc_{item.name}" for item in fields(PointCloudProcessorConfig)
        } - optional_keys
        missing_keys = sorted(required_keys - set(pc_meta))
        if missing_keys:
            raise ValueError(f"point-cloud metadata missing keys: {missing_keys}")
        processor_config = PointCloudProcessorConfig.from_meta_dict(pc_meta)
        depth_scale = float(meta["depth_scale"])
        if not np.isfinite(depth_scale) or depth_scale <= 0.0:
            raise ValueError("depth_scale must be finite and positive")
        static_processor = None
        if str(meta.get("camera_type", "")) == "eye_to_hand":
            static_processor = PointCloudProcessor(
                _camera_transform_for_index(reader, 0), processor_config
            )
        return cls(
            reader=reader,
            processor_config=processor_config,
            rays=make_rays(source_height, source_width, source_k).numpy(),
            depth_scale=depth_scale,
            target_point_count=config.target_point_count,
            static_processor=static_processor,
        )

    def derive(self, source_index: int, rgb: np.ndarray) -> np.ndarray | None:
        depth_m = depth_to_meters(
            np.asarray(self.reader.h5f["depth"][source_index], dtype=np.uint16),
            depth_scale=self.depth_scale,
        )
        processor = self.static_processor or PointCloudProcessor(
            _camera_transform_for_index(self.reader, source_index),
            self.processor_config,
        )
        cloud = processor.process(depth_m, rgb, self.rays)
        if cloud is None:
            return None
        return resize_point_cloud(
            cloud,
            source_point_count=int(processor.last_source_point_count),
            target_point_count=self.target_point_count,
        )


def _derive_depth_valid_mask(reader: EpisodeReader) -> np.ndarray:
    depth = reader.h5f["depth"]
    frame_count = int(reader.h5f["meta"].attrs["num_frames"])
    if depth.shape[0] != frame_count:
        raise ValueError("depth length does not match source grid")
    valid = np.zeros(frame_count, dtype=bool)
    for index in range(frame_count):
        frame = np.asarray(depth[index], dtype=np.uint16)
        valid[index] = frame.ndim == 2 and bool(np.any(frame > 0))
    return valid


def _derive_pointcloud_valid_mask(
    reader: EpisodeReader, config: ProcessingConfig
) -> np.ndarray:
    frame_count = int(reader.h5f["meta"].attrs["num_frames"])
    valid = np.zeros(frame_count, dtype=bool)
    deriver = _PointCloudDeriver.from_reader(reader, config)
    decoded_count = 0
    logged_value_error = False
    for source_index, rgb in enumerate(reader.iter_camera_frames("rgb")):
        if source_index >= frame_count:
            raise ValueError("decoded RGB frame count exceeds source grid")
        decoded_count += 1
        try:
            valid[source_index] = deriver.derive(source_index, rgb) is not None
        except ValueError:
            if not logged_value_error:
                logger.warning(
                    "point-cloud preflight rejected malformed frame %d in %s",
                    source_index,
                    reader.h5_path,
                    exc_info=True,
                )
                logged_value_error = True
    if decoded_count != frame_count:
        raise ValueError(
            f"decoded RGB frame count {decoded_count} != source grid {frame_count}"
        )
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


def discover_episode_dirs(input_root: str | Path) -> tuple[Path, ...]:
    root = Path(input_root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    episodes = tuple(
        sorted(
            child
            for child in root.iterdir()
            if child.is_dir() and not child.name.startswith(".")
        )
    )
    if not episodes:
        raise FileNotFoundError(f"no episode directories found in {root}")
    return episodes


@contextmanager
def _open_episode(path: Path) -> Iterator[EpisodeReader]:
    with EpisodeReader(path) as reader:
        yield reader


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
    specs: dict[str, tuple[tuple[int, ...], Any, tuple[int, ...]]] = {
        "joint_state": ((length, 19), np.float32, (numeric_chunk, 19)),
        "action": ((length, 19), np.float32, (numeric_chunk, 19)),
        "action_ee": ((length, 21), np.float32, (numeric_chunk, 21)),
        "contact_force": ((length, 5, 3), np.float32, (numeric_chunk, 5, 3)),
        "fingertip_points": ((length, 5, 3), np.float32, (numeric_chunk, 5, 3)),
    }
    if config.profile.needs_rgb:
        specs.update(
            {
                "rgb": (
                    (length, config.target_rgb_height, config.target_rgb_width, 3),
                    np.uint8,
                    (1, config.target_rgb_height, config.target_rgb_width, 3),
                ),
                "depth": (
                    (length, config.target_rgb_height, config.target_rgb_width),
                    np.uint16,
                    (1, config.target_rgb_height, config.target_rgb_width),
                ),
                "camera_intrinsic": ((length, 9), np.float32, (numeric_chunk, 9)),
                "camera_extrinsic": (
                    (length, 4, 4),
                    np.float32,
                    (numeric_chunk, 4, 4),
                ),
            }
        )
    if config.profile.needs_pointcloud:
        specs["point_cloud"] = (
            (length, config.target_point_count, 6),
            np.float32,
            (1, config.target_point_count, 6),
        )
    for name in config.profile.dataset_keys:
        shape, dtype, chunks = specs[name]
        output.create_dataset(
            name,
            shape=shape,
            dtype=dtype,
            **_dataset_kwargs(config, chunks=chunks),
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
    output.attrs.update(
        {
            "schema_name": PROCESSED_SCHEMA_NAME,
            "schema_version": PROCESSED_SCHEMA_VERSION,
            "domain": "real",
            "source_format": "dexmani_v17",
            "source_schema_version": EPISODE_SCHEMA_VERSION,
            "source_episode": reader.h5_path.name,
            "source_frames": decision.source_frames,
            "profile": config.profile.value,
            "episode_steps": decision.selected_frames,
            "dt": float(reader.timing.grid_dt_s),
            "time_semantics": "logical_control_grid_after_row_compaction",
            "action_dim": 19,
            "action_ee_dim": 21,
            "action_space": "joint",
            "obs_alignment": "obs[t]_before_action[t]",
            "task_name": task_name,
            "point_cloud_frame": (
                "xarm_base" if config.profile.needs_pointcloud else "omitted"
            ),
            "fingertip_points_frame": "xarm_base",
            "fingertip_points_unit": "m",
            "action_ee_frame": "xarm_base",
            "action_ee_components": "eef_position_m(3)+eef_rot6d(6)+xhand_target_rad(12)",
            "contact_force_source": "schema_v17.hand_contact",
            "contact_force_unit": str(
                meta.get("tactile_unit", "sdk_scaled_unknown_si")
            ),
            "contact_force_si_verified": bool(
                meta.get("tactile_si_unit_verified", False)
            ),
            "contact_force_frame": "xhand_sensor_native_axes_per_finger",
            "frame_compatibility_with_sim_world": False,
            "processing_config_json": _json(config.to_dict()),
            "quality_summary_json": _json(decision.quality),
            "source_decision_json": _json(decision.to_dict()),
            "source_member_sha256_json": _json(
                {
                    member: _sha256_file(reader.h5_path / member)
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
                "depth_transform": "resize_no_crop_nearest",
                "depth_unit": "sensor_unit",
                "depth_scale_m_per_unit": float(meta["depth_scale"]),
                "depth_invalid_value": 0,
                "camera_extrinsic_semantics": "T_xarm_base_camera;camera_optical_to_xarm_base",
            }
        )
    if config.profile.needs_pointcloud:
        output.attrs["point_cloud_transform"] = (
            "real_rgbd_to_xarm_base_then_deterministic_fps_or_cyclic_pad"
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
    with h5py.File(path, "w") as output:
        _write_attrs(output, reader, decision, config, annotation)
        _create_data_datasets(output, decision.selected_frames, config)
        arm_state = np.asarray(reader.h5f["arm_qpos"][selected], dtype=np.float32)
        hand_state = np.asarray(reader.h5f["hand_qpos"][selected], dtype=np.float32)
        arm_action = np.asarray(
            reader.h5f["action_arm_joint_sent"][selected], dtype=np.float32
        )
        hand_action = np.asarray(
            reader.h5f["action_hand_joint"][selected], dtype=np.float32
        )
        arm_action_ee = np.asarray(
            reader.h5f["action_arm_ee"][selected], dtype=np.float32
        )
        output["joint_state"][:] = np.concatenate((arm_state, hand_state), axis=1)
        output["action"][:] = np.concatenate((arm_action, hand_action), axis=1)
        output["action_ee"][:] = np.concatenate((arm_action_ee, hand_action), axis=1)
        output["contact_force"][:] = np.asarray(
            reader.h5f["hand_contact"][selected], dtype=np.float32
        )
        output["fingertip_points"][:] = np.asarray(
            reader.h5f["hand_fingertip"][selected], dtype=np.float32
        )

        provenance = output.create_group("provenance")
        provenance.attrs["drop_reason_bit_names_json"] = _json(
            {str(bit): name for bit, name in enumerate(decision.drop_reason_names)}
        )
        provenance.create_dataset(
            "source_row_index",
            data=selected,
            **_dataset_kwargs(config, chunks=(min(len(selected), 256),)),
        )
        provenance.create_dataset(
            "source_sample_index",
            data=np.asarray(
                reader.h5f["source_sample_index"][selected], dtype=np.int64
            ),
            **_dataset_kwargs(config, chunks=(min(len(selected), 256),)),
        )
        provenance.create_dataset(
            "source_timestamp_s",
            data=np.asarray(reader.h5f["timestamp"][selected], dtype=np.float64),
            **_dataset_kwargs(config, chunks=(min(len(selected), 256),)),
        )
        provenance.create_dataset(
            "source_keep_mask",
            data=decision.keep_mask,
            **_dataset_kwargs(config, chunks=(min(decision.source_frames, 256),)),
        )
        provenance.create_dataset(
            "source_drop_reason_bits",
            data=decision.drop_reason_bits,
            **_dataset_kwargs(config, chunks=(min(decision.source_frames, 256),)),
        )

        if config.profile.needs_rgb:
            meta = reader.h5f["meta"].attrs
            source_height = int(meta["camera_encoding_height"])
            source_width = int(meta["camera_encoding_width"])
            camera_k = resize_camera_intrinsic(
                np.asarray(meta["camera_K"]),
                source_height=source_height,
                source_width=source_width,
                target_height=config.target_rgb_height,
                target_width=config.target_rgb_width,
            )
            output["camera_intrinsic"][:] = camera_k[None, :]
            output["camera_extrinsic"][:] = np.stack(
                [_camera_transform_for_index(reader, int(index)) for index in selected]
            ).astype(np.float32)
            for target_index, source_index in enumerate(selected):
                depth = np.asarray(reader.h5f["depth"][source_index], dtype=np.uint16)
                output["depth"][target_index] = resize_depth(
                    depth,
                    height=config.target_rgb_height,
                    width=config.target_rgb_width,
                )

        pointcloud_deriver = (
            _PointCloudDeriver.from_reader(reader, config)
            if config.profile.needs_pointcloud
            else None
        )
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


def _dataset_row_slices(dataset: h5py.Dataset) -> Iterator[slice]:
    row_bytes = int(dataset.dtype.itemsize * np.prod(dataset.shape[1:], dtype=np.int64))
    rows_per_chunk = max(1, _VALIDATION_CHUNK_BYTES // max(1, row_bytes))
    for start in range(0, dataset.shape[0], rows_per_chunk):
        yield slice(start, min(dataset.shape[0], start + rows_per_chunk))


def _expected_specs(
    length: int, config: ProcessingConfig
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    specs: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
        "joint_state": ((length, 19), np.dtype(np.float32)),
        "action": ((length, 19), np.dtype(np.float32)),
        "action_ee": ((length, 21), np.dtype(np.float32)),
        "contact_force": ((length, 5, 3), np.dtype(np.float32)),
        "fingertip_points": ((length, 5, 3), np.dtype(np.float32)),
    }
    if config.profile.needs_rgb:
        specs.update(
            {
                "rgb": (
                    (length, config.target_rgb_height, config.target_rgb_width, 3),
                    np.dtype(np.uint8),
                ),
                "depth": (
                    (length, config.target_rgb_height, config.target_rgb_width),
                    np.dtype(np.uint16),
                ),
                "camera_intrinsic": ((length, 9), np.dtype(np.float32)),
                "camera_extrinsic": ((length, 4, 4), np.dtype(np.float32)),
            }
        )
    if config.profile.needs_pointcloud:
        specs["point_cloud"] = (
            (length, config.target_point_count, 6),
            np.dtype(np.float32),
        )
    return specs


def validate_processed_hdf5(
    path: str | Path, config: ProcessingConfig
) -> dict[str, Any]:
    """Fail closed on a processed Real HDF5 v3 artifact."""

    artifact = Path(path)
    with h5py.File(artifact, "r") as source:
        expected_top = set(config.profile.dataset_keys) | {"provenance"}
        if set(source.keys()) != expected_top:
            raise ValueError(f"{artifact.name}: top-level keys do not match v3 profile")
        length = int(source.attrs.get("episode_steps", -1))
        if length < config.min_episode_frames:
            raise ValueError(f"{artifact.name}: episode_steps={length} is too short")
        specs = _expected_specs(length, config)
        for key, (shape, dtype) in specs.items():
            dataset = source[key]
            if (
                not isinstance(dataset, h5py.Dataset)
                or dataset.shape != shape
                or dataset.dtype != dtype
            ):
                raise ValueError(f"{artifact.name}: invalid {key} shape/dtype")
            if (
                dataset.compression != "gzip"
                or int(dataset.compression_opts) != config.gzip_level
            ):
                raise ValueError(f"{artifact.name}: invalid {key} compression")
            if np.issubdtype(dtype, np.floating):
                for row_slice in _dataset_row_slices(dataset):
                    if not np.all(np.isfinite(dataset[row_slice])):
                        raise ValueError(f"{artifact.name}: {key} contains NaN/Inf")
        if config.profile.needs_rgb:
            for row_slice in _dataset_row_slices(source["depth"]):
                depth = np.asarray(source["depth"][row_slice], dtype=np.uint16)
                if np.any(~np.any(depth > 0, axis=(1, 2))):
                    raise ValueError(
                        f"{artifact.name}: depth contains an all-invalid frame"
                    )
            k = np.asarray(source["camera_intrinsic"][:], dtype=np.float32)
            if np.any(k[:, (0, 4)] <= 0.0) or not np.allclose(k[:, 8], 1.0):
                raise ValueError(f"{artifact.name}: invalid camera_intrinsic")
            for transform in source["camera_extrinsic"]:
                _validate_transform(transform, label="camera_extrinsic")
            scale = float(source.attrs.get("depth_scale_m_per_unit", np.nan))
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"{artifact.name}: invalid depth scale")
        if config.profile.needs_pointcloud:
            for row_slice in _dataset_row_slices(source["point_cloud"]):
                cloud = np.asarray(source["point_cloud"][row_slice], dtype=np.float32)
                if np.any(cloud[..., 3:] < 0.0) or np.any(cloud[..., 3:] > 1.0):
                    raise ValueError(f"{artifact.name}: point-cloud RGB outside [0,1]")
                if np.any(
                    ~np.any(np.linalg.norm(cloud[..., :3], axis=2) > 0.0, axis=1)
                ):
                    raise ValueError(f"{artifact.name}: all-zero point-cloud frame")
        if str(source.attrs.get("schema_name", "")) != PROCESSED_SCHEMA_NAME:
            raise ValueError(f"{artifact.name}: invalid schema_name")
        if int(source.attrs.get("schema_version", -1)) != PROCESSED_SCHEMA_VERSION:
            raise ValueError(f"{artifact.name}: invalid schema_version")
        if str(source.attrs.get("domain", "")) != "real":
            raise ValueError(f"{artifact.name}: domain must be real")
        if str(source.attrs.get("profile", "")) != config.profile.value:
            raise ValueError(f"{artifact.name}: profile mismatch")
        if str(source.attrs.get("task_name", "")).strip() in {"", "unknown"}:
            raise ValueError(f"{artifact.name}: explicit task_name required")
        if str(source.attrs.get("obs_alignment", "")) != "obs[t]_before_action[t]":
            raise ValueError(f"{artifact.name}: invalid observation/action alignment")
        for key in (
            "processing_config_json",
            "quality_summary_json",
            "source_decision_json",
            "source_member_sha256_json",
        ):
            try:
                value = json.loads(str(source.attrs[key]))
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{artifact.name}: invalid {key}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{artifact.name}: {key} must encode an object")
        if json.loads(str(source.attrs["processing_config_json"])) != config.to_dict():
            raise ValueError(f"{artifact.name}: processing config mismatch")
        provenance = source["provenance"]
        expected_provenance = {
            "source_row_index",
            "source_sample_index",
            "source_timestamp_s",
            "source_keep_mask",
            "source_drop_reason_bits",
        }
        if set(provenance.keys()) != expected_provenance:
            raise ValueError(f"{artifact.name}: invalid provenance keys")
        source_frames = int(source.attrs.get("source_frames", -1))
        rows = np.asarray(provenance["source_row_index"][:], dtype=np.int64)
        samples = np.asarray(provenance["source_sample_index"][:], dtype=np.int64)
        timestamps = np.asarray(provenance["source_timestamp_s"][:], dtype=np.float64)
        keep = np.asarray(provenance["source_keep_mask"][:], dtype=bool)
        reasons = np.asarray(provenance["source_drop_reason_bits"][:], dtype=np.uint64)
        if (
            rows.shape != (length,)
            or samples.shape != (length,)
            or timestamps.shape != (length,)
            or keep.shape != (source_frames,)
            or reasons.shape != (source_frames,)
            or not np.array_equal(rows, np.flatnonzero(keep))
            or np.any(reasons[keep] != 0)
            or np.any(reasons[~keep] == 0)
            or np.any(np.diff(rows) <= 0)
            or np.any(np.diff(samples) <= 0)
            or not np.all(np.isfinite(timestamps))
            or np.any(np.diff(timestamps) <= 0.0)
        ):
            raise ValueError(f"{artifact.name}: provenance row mapping mismatch")
        try:
            reason_names = json.loads(
                str(provenance.attrs["drop_reason_bit_names_json"])
            )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"{artifact.name}: invalid drop-reason name mapping"
            ) from exc
        if (
            not isinstance(reason_names, dict)
            or set(reason_names) != {str(bit) for bit in range(len(reason_names))}
            or len(reason_names) > 64
            or any(
                not isinstance(name, str) or not name for name in reason_names.values()
            )
        ):
            raise ValueError(f"{artifact.name}: invalid drop-reason name mapping")
        valid_reason_bits = (
            np.uint64((1 << len(reason_names)) - 1)
            if len(reason_names) < 64
            else np.iinfo(np.uint64).max
        )
        if np.any(reasons & ~valid_reason_bits):
            raise ValueError(f"{artifact.name}: unknown provenance reason bit")
        return {"path": artifact.name, "frames": length, "keys": sorted(specs)}


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


def process_episode_root(
    input_root: str | Path,
    output_root: str | Path,
    config: ProcessingConfig,
    *,
    annotations_path: str | Path | None = None,
    dry_run: bool = False,
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
            with _open_episode(episode) as reader:
                reader.require_valid(purpose="offline processing")
                decisions.append(
                    analyze_episode(
                        reader,
                        config,
                        annotation,
                        depth_valid_mask=(
                            _derive_depth_valid_mask(reader)
                            if config.profile.needs_rgb and annotation.include
                            else None
                        ),
                        pointcloud_valid_mask=(
                            _derive_pointcloud_valid_mask(reader, config)
                            if config.profile.needs_pointcloud and annotation.include
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
            with _open_episode(decision.source_path) as reader:
                outputs.append(
                    _write_processed_episode(
                        reader, decision, staging, config, annotation
                    )
                )
        validation = [
            validate_processed_hdf5(staging / item["path"], config) for item in outputs
        ]
        report["outputs"] = outputs
        report["validation"] = validation
        with (staging / "processing_index.json").open("w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.flush()
        atomic_publish(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report
