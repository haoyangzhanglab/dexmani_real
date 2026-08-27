"""Transactional depth-to-color aligned raw-v24 to processed-v10 processing.

Raw v23 remains available only through an explicit ``allow_legacy_v23`` option.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
import yaml

from dexmani_real.data.clean import analyze_episode
from dexmani_real.data.contracts import (
    EpisodeAnnotation,
    EpisodeDecision,
    OutputProfile,
    ProcessingConfig,
)
from dexmani_real.data.raw_pointcloud import (
    RawEpisodePointCloudDeriver,
    load_raw_episode_base_from_color,
    load_raw_episode_camera_model,
    validate_rigid_transform,
)
from dexmani_real.data.transforms import (
    resize_camera_intrinsic,
    resize_depth,
    resize_rgb,
)
from dexmani_real.planning.poses import validate_canonical_rot6d
from dexmani_real.recording.reader import EpisodeReader
from dexmani_real.sensor.pointcloud import (
    POINT_CLOUD_COLOR_SOURCE,
    POINT_CLOUD_POLICY_ID,
    POINT_CLOUD_SAMPLING,
    POINT_CLOUD_TRANSFORM,
)
from dexmani_real.utils.atomic_io import atomic_publish, sha256_file
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

PROCESSED_SCHEMA_NAME = "dexmani-real-processed-hdf5"
PROCESSED_SCHEMA_VERSION = 10
_SOURCE_MEMBERS = ("data.h5", "depth.h5", "rgb.mp4")
_PROVENANCE_DATASETS = (
    "source_row_index",
    "source_sample_index",
    "source_timestamp_s",
    "source_segment_ends",
    "source_keep_mask",
    "source_drop_reason_bits",
)
_PROVENANCE_ATTRS = ("drop_reason_bit_names_json",)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_VALIDATION_CHUNK_BYTES = 64 * 1024 * 1024
_CORE_DATASET_SPECS: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
    "joint_state": ((19,), np.dtype(np.float32)),
    "action": ((19,), np.dtype(np.float32)),
    "action_ee": ((21,), np.dtype(np.float32)),
    "contact_force": ((5, 3), np.dtype(np.float32)),
    "fingertip_points": ((5, 3), np.dtype(np.float32)),
}
_FRAME_CHUNKED_DATASETS = frozenset(("rgb", "depth", "point_cloud"))


@dataclass(frozen=True)
class ProcessedProvenance:
    """Validated compact-row provenance shared by processing and export."""

    source_rows: np.ndarray
    source_samples: np.ndarray
    source_timestamps: np.ndarray
    segment_ends: np.ndarray
    keep_mask: np.ndarray
    drop_reason_bits: np.ndarray
    drop_reason_names: tuple[str, ...]

    @property
    def segment_lengths(self) -> tuple[int, ...]:
        starts = np.concatenate(
            (np.asarray([0], dtype=np.int64), self.segment_ends[:-1])
        )
        return tuple(
            int(end - start)
            for start, end in zip(starts, self.segment_ends, strict=True)
        )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_object_attr(
    source: h5py.File | h5py.Group, key: str, *, label: str
) -> dict[str, Any]:
    try:
        raw = source.attrs[key]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        value = json.loads(str(raw))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}: invalid {key}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label}: {key} must encode an object")
    return value


def _indices_to_ranges(indices: np.ndarray) -> list[list[int]]:
    """Return half-open source-index ranges for a strictly increasing array."""
    values = np.asarray(indices, dtype=np.int64)
    if values.size == 0:
        return []
    starts = np.r_[0, np.flatnonzero(np.diff(values) != 1) + 1]
    ends = np.r_[starts[1:], values.size]
    return [
        [int(values[start]), int(values[end - 1] + 1)]
        for start, end in zip(starts, ends, strict=True)
    ]


def _validate_pointcloud_workspace(
    cloud: np.ndarray,
    workspace: tuple[float, float, float, float, float, float],
    *,
    label: str,
) -> None:
    bounds = np.asarray(workspace, dtype=np.float64)
    if bounds.shape != (6,) or not np.all(np.isfinite(bounds)):
        raise ValueError(f"{label}: persisted point-cloud workspace is invalid")
    lower, upper = bounds[:3], bounds[3:]
    if np.any(lower >= upper):
        raise ValueError(f"{label}: persisted point-cloud workspace is invalid")
    points = cloud[..., :3]
    if np.any(points < lower) or np.any(points > upper):
        raise ValueError(f"{label}: point-cloud XYZ leaves persisted workspace")


def validate_processed_payload(
    source: h5py.File | h5py.Group,
    *,
    expected_specs: Mapping[str, tuple[tuple[int, ...], np.dtype[Any]]],
    length: int,
    label: str,
    validate_rgbd: bool = False,
    pointcloud_workspace: tuple[float, float, float, float, float, float] | None = None,
) -> None:
    """Validate processed datasets before they cross the HDF5/Zarr boundary.

    Dataset dtypes are compared to the actual HDF5 dtype (not a lossy cast),
    all rows must share the declared length, floating payloads must be finite,
    and ``action_ee`` must carry canonical unit/orthogonal rot6d labels.  The
    optional RGB-D and persisted-workspace checks are also admission checks and
    never rebuild a point cloud.
    """
    if length <= 0:
        raise ValueError(f"{label}: processed length must be positive")
    expected_keys = set(expected_specs)
    if set(source.keys()) != expected_keys | {"provenance"}:
        raise ValueError(f"{label}: processed data keys do not match profile")
    for key, (expected_shape, expected_dtype) in expected_specs.items():
        dataset = source.get(key)
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"{label}: {key} is not an HDF5 dataset")
        dtype = np.dtype(expected_dtype)
        if dataset.dtype != dtype:
            raise ValueError(
                f"{label}: {key} dtype must be {dtype}, got {dataset.dtype}"
            )
        if dataset.shape != tuple(expected_shape):
            raise ValueError(
                f"{label}: {key} shape must be {tuple(expected_shape)}, "
                f"got {dataset.shape}"
            )
        if dataset.ndim == 0 or dataset.shape[0] != length:
            raise ValueError(f"{label}: {key} first dimension must be {length}")
        if key == "rgb" and (dataset.ndim != 4 or dataset.shape[-1] != 3):
            raise ValueError(f"{label}: rgb must be (N,H,W,3)")
        if key == "depth" and dataset.ndim != 3:
            raise ValueError(f"{label}: depth must be (N,H,W)")
        if key == "camera_intrinsic" and (dataset.ndim != 2 or dataset.shape[1] != 9):
            raise ValueError(f"{label}: camera_intrinsic must be (N,9)")
        if key == "camera_extrinsic" and (
            dataset.ndim != 3 or dataset.shape[1:] != (4, 4)
        ):
            raise ValueError(f"{label}: camera_extrinsic must be (N,4,4)")
        if key == "point_cloud" and (dataset.ndim != 3 or dataset.shape[-1] != 6):
            raise ValueError(f"{label}: point_cloud must be (N,P,6)")
        if not np.issubdtype(dtype, np.floating) and key != "point_cloud":
            continue
        for row_slice in _dataset_row_slices(dataset):
            block = np.asarray(dataset[row_slice])
            if np.issubdtype(dtype, np.floating) and not np.all(np.isfinite(block)):
                raise ValueError(f"{label}: {key} contains NaN/Inf")
            if key == "action_ee":
                validate_canonical_rot6d(
                    block[:, 3:9], label=f"{label}: action_ee rot6d"
                )
            if key == "point_cloud":
                if np.any(block[..., 3:] < 0.0) or np.any(block[..., 3:] > 1.0):
                    raise ValueError(f"{label}: point-cloud RGB outside [0,1]")
                if np.any(
                    ~np.any(np.linalg.norm(block[..., :3], axis=2) > 0.0, axis=1)
                ):
                    raise ValueError(f"{label}: all-zero point-cloud frame")
                if pointcloud_workspace is not None:
                    _validate_pointcloud_workspace(
                        block, pointcloud_workspace, label=label
                    )

    if validate_rgbd:
        rgb = source.get("rgb")
        depth = source.get("depth")
        intrinsic = source.get("camera_intrinsic")
        extrinsic = source.get("camera_extrinsic")
        if not all(
            isinstance(dataset, h5py.Dataset)
            for dataset in (rgb, depth, intrinsic, extrinsic)
        ):
            raise ValueError(f"{label}: RGB-D datasets are incomplete")
        if rgb.shape[0] != depth.shape[0] or rgb.shape[1:3] != depth.shape[1:3]:
            raise ValueError(
                f"{label}: rgb/depth spatial shape mismatch: "
                f"rgb={rgb.shape}, depth={depth.shape}"
            )
        for row_slice in _dataset_row_slices(depth):
            depth_block = np.asarray(depth[row_slice], dtype=np.uint16)
            if np.any(~np.any(depth_block > 0, axis=(1, 2))):
                raise ValueError(f"{label}: depth contains an all-invalid frame")
        k = np.asarray(intrinsic[:], dtype=np.float64).reshape(length, 3, 3)
        canonical_last_row = np.broadcast_to(
            np.asarray((0.0, 0.0, 1.0), dtype=np.float64), (length, 3)
        )
        if (
            not np.all(np.isfinite(k))
            or np.any(k[:, 0, 0] <= 0.0)
            or np.any(k[:, 1, 1] <= 0.0)
            or not np.allclose(k[:, 0, 1], 0.0, rtol=0.0, atol=1e-7)
            or not np.allclose(k[:, 1, 0], 0.0, rtol=0.0, atol=1e-7)
            or not np.allclose(k[:, 2], canonical_last_row, rtol=0.0, atol=1e-7)
        ):
            raise ValueError(f"{label}: invalid camera_intrinsic")
        for transform in extrinsic:
            validate_rigid_transform(transform, label="camera_extrinsic")


def validate_processed_admission(
    source: h5py.File | h5py.Group, *, label: str = "processed"
) -> ProcessedProvenance:
    """Validate a persisted processed artifact without a runtime config.

    Consumers that only inspect or display processed artifacts do not have a
    resolved processing configuration to compare against.  They still need
    the persisted profile to select the fixed core contract and the optional
    modality contracts before reading any payload.  RGB-D tail shapes are
    intentionally taken from the artifact only after profile/key selection;
    the shared validator then enforces their cross-field geometry.
    """
    schema_name = source.attrs.get("schema_name", "")
    if isinstance(schema_name, bytes):
        schema_name = schema_name.decode("utf-8")
    if str(schema_name) != PROCESSED_SCHEMA_NAME:
        raise ValueError(f"{label}: invalid processed schema_name")
    if int(source.attrs.get("schema_version", -1)) != PROCESSED_SCHEMA_VERSION:
        raise ValueError(f"{label}: invalid processed schema_version")
    profile_value = source.attrs.get("profile", "")
    if isinstance(profile_value, bytes):
        profile_value = profile_value.decode("utf-8")
    try:
        profile = OutputProfile(str(profile_value))
    except ValueError as exc:
        raise ValueError(f"{label}: invalid processed profile") from exc
    length = int(source.attrs.get("episode_steps", -1))
    expected_specs: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
        name: ((length, *tail_shape), dtype)
        for name, (tail_shape, dtype) in _CORE_DATASET_SPECS.items()
    }
    optional_dtypes: dict[str, np.dtype[Any]] = {
        "rgb": np.dtype(np.uint8),
        "depth": np.dtype(np.uint16),
        "camera_intrinsic": np.dtype(np.float32),
        "camera_extrinsic": np.dtype(np.float32),
        "point_cloud": np.dtype(np.float32),
    }
    for key in profile.dataset_keys:
        if key in expected_specs:
            continue
        dataset = source.get(key)
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"{label}: missing processed dataset {key}")
        expected_specs[key] = (
            (length, *tuple(int(value) for value in dataset.shape[1:])),
            optional_dtypes[key],
        )
    validate_processed_payload(
        source,
        expected_specs=expected_specs,
        length=length,
        label=label,
        validate_rgbd=profile.needs_rgb,
    )
    return validate_processed_provenance(
        source,
        length=length,
        expected_profile=profile.value,
        label=label,
    )


def validate_processed_provenance(
    source: h5py.File | h5py.Group,
    *,
    length: int | None = None,
    source_frames: int | None = None,
    dt: float | None = None,
    contiguity_tolerance_s: float | None = None,
    expected_profile: str | None = None,
    label: str = "processed",
) -> ProcessedProvenance:
    """Validate exact processed provenance and return its compact mapping.

    This is intentionally an internal provenance contract, not an authenticity
    mechanism: source member hashes are checked for exact names and hex shape,
    but callers still need a trusted processed artifact or an external raw-data
    attestation before treating those hashes as evidence.
    """
    attrs = source.attrs
    if length is None:
        length = int(attrs.get("episode_steps", -1))
    if source_frames is None:
        source_frames = int(attrs.get("source_frames", -1))
    if dt is None:
        dt = float(attrs.get("dt", np.nan))
    if contiguity_tolerance_s is None:
        contiguity_tolerance_s = float(
            attrs.get("source_contiguity_tolerance_s", np.nan)
        )
    if length <= 0 or source_frames < length:
        raise ValueError(f"{label}: invalid processed/source frame counts")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"{label}: dt must be finite and positive")
    if not np.isfinite(contiguity_tolerance_s) or contiguity_tolerance_s <= 0.0:
        raise ValueError(f"{label}: invalid source contiguity tolerance")

    provenance = source.get("provenance")
    if not isinstance(provenance, h5py.Group):
        raise ValueError(f"{label}: provenance group is missing")
    if set(provenance.keys()) != set(_PROVENANCE_DATASETS):
        raise ValueError(f"{label}: invalid provenance keys")
    if set(provenance.attrs.keys()) != set(_PROVENANCE_ATTRS):
        raise ValueError(f"{label}: invalid provenance attributes")

    expected_dtypes: dict[str, np.dtype[Any]] = {
        "source_row_index": np.dtype(np.int64),
        "source_sample_index": np.dtype(np.int64),
        "source_timestamp_s": np.dtype(np.float64),
        "source_segment_ends": np.dtype(np.int64),
        "source_keep_mask": np.dtype(np.bool_),
        "source_drop_reason_bits": np.dtype(np.uint64),
    }
    expected_shapes = {
        "source_row_index": (length,),
        "source_sample_index": (length,),
        "source_timestamp_s": (length,),
        "source_segment_ends": None,
        "source_keep_mask": (source_frames,),
        "source_drop_reason_bits": (source_frames,),
    }
    values: dict[str, np.ndarray] = {}
    for key in _PROVENANCE_DATASETS:
        dataset = provenance[key]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"{label}: provenance/{key} is not a dataset")
        if dataset.dtype != expected_dtypes[key]:
            raise ValueError(
                f"{label}: provenance/{key} dtype must be "
                f"{expected_dtypes[key]}, got {dataset.dtype}"
            )
        expected_shape = expected_shapes[key]
        if expected_shape is not None and dataset.shape != expected_shape:
            raise ValueError(
                f"{label}: provenance/{key} shape must be {expected_shape}, "
                f"got {dataset.shape}"
            )
        if key == "source_segment_ends" and (
            dataset.ndim != 1 or dataset.shape[0] == 0
        ):
            raise ValueError(f"{label}: provenance/{key} shape is invalid")
        values[key] = np.asarray(dataset[:])

    rows = values["source_row_index"]
    samples = values["source_sample_index"]
    timestamps = values["source_timestamp_s"]
    segment_ends = values["source_segment_ends"]
    keep_mask = values["source_keep_mask"]
    reasons = values["source_drop_reason_bits"]
    if (
        np.any(rows < 0)
        or np.any(samples < 0)
        or np.any(np.diff(rows) <= 0)
        or np.any(np.diff(samples) <= 0)
        or not np.all(np.isfinite(timestamps))
        or np.any(np.diff(timestamps) <= 0.0)
        or np.any(segment_ends <= 0)
        or np.any(np.diff(segment_ends) <= 0)
        or segment_ends[-1] != length
        or not np.array_equal(rows, np.flatnonzero(keep_mask))
        or np.any(reasons[keep_mask] != 0)
        or np.any(reasons[~keep_mask] == 0)
    ):
        raise ValueError(f"{label}: provenance row mapping mismatch")

    reason_names_value = _json_object_attr(
        provenance, "drop_reason_bit_names_json", label=label
    )
    if (
        set(reason_names_value) != {str(bit) for bit in range(len(reason_names_value))}
        or len(reason_names_value) > 64
        or any(
            not isinstance(name, str) or not name
            for name in reason_names_value.values()
        )
        or len(set(reason_names_value.values())) != len(reason_names_value)
    ):
        raise ValueError(f"{label}: invalid drop-reason name mapping")
    reason_names = tuple(
        reason_names_value[str(bit)] for bit in range(len(reason_names_value))
    )
    valid_reason_bits = (
        np.uint64((1 << len(reason_names)) - 1)
        if len(reason_names) < 64
        else np.iinfo(np.uint64).max
    )
    if np.any(reasons & ~valid_reason_bits):
        raise ValueError(f"{label}: unknown provenance reason bit")

    discontinuity = (
        (np.diff(rows) != 1)
        | (np.diff(samples) != 1)
        | (np.abs(np.diff(timestamps) - dt) > contiguity_tolerance_s)
    )
    expected_segment_ends = np.concatenate(
        (np.flatnonzero(discontinuity).astype(np.int64) + 1, [length])
    )
    if not np.array_equal(segment_ends, expected_segment_ends):
        raise ValueError(f"{label}: source segment boundaries mismatch")

    source_decision = _json_object_attr(source, "source_decision_json", label=label)
    if expected_profile is None:
        expected_profile = str(attrs.get("profile", ""))
    expected_decision = {
        "profile": expected_profile,
        "source_frames": source_frames,
        "selected_frames": length,
        "dropped_frames": source_frames - length,
        "accepted": True,
        "rejected_reason": None,
        "selected_source_ranges": _indices_to_ranges(rows),
        "selected_segment_ends": segment_ends.tolist(),
    }
    for key, expected in expected_decision.items():
        if key not in source_decision or source_decision[key] != expected:
            raise ValueError(
                f"{label}: source_decision_json {key!r} disagrees with provenance"
            )

    source_hashes = _json_object_attr(source, "source_member_sha256_json", label=label)
    if set(source_hashes) != set(_SOURCE_MEMBERS) or any(
        not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
        for value in source_hashes.values()
    ):
        raise ValueError(
            f"{label}: source_member_sha256_json must contain exactly "
            f"{_SOURCE_MEMBERS} with 64-hex values"
        )
    return ProcessedProvenance(
        source_rows=rows,
        source_samples=samples,
        source_timestamps=timestamps,
        segment_ends=segment_ends,
        keep_mask=keep_mask,
        drop_reason_bits=reasons,
        drop_reason_names=reason_names,
    )


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
    than raw ``data.h5`` sources.
    """
    root = Path(input_root)
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


@contextmanager
def _open_episode(
    path: Path, *, allow_legacy_v23: bool = False
) -> Iterator[EpisodeReader]:
    """Open one raw episode with an explicit legacy opt-in boundary."""
    with EpisodeReader(path, allow_legacy_v23=allow_legacy_v23) as reader:
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
    endpoint_delta_tolerance_rad = config.endpoint_delta_tolerance_rad
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
            "action_semantics": "deployment_grid_rate_limited_target",
            "arm_max_delta_rad_per_tick": (
                np.nan
                if config.arm_max_delta_rad_per_tick is None
                else float(config.arm_max_delta_rad_per_tick)
            ),
            "hand_max_delta_rad_per_tick": config.hand_max_delta_rad_per_tick,
            "endpoint_delta_tolerance_rad": (
                np.nan
                if endpoint_delta_tolerance_rad is None
                else float(endpoint_delta_tolerance_rad)
            ),
            "deployment_equivalent": bool(config.profile.needs_pointcloud),
            "task_name": task_name,
            "point_cloud_frame": (
                "xarm_base" if config.profile.needs_pointcloud else "omitted"
            ),
            "fingertip_points_frame": "xarm_base",
            "fingertip_points_unit": "m",
            "action_ee_frame": "xarm_base",
            "action_ee_components": "eef_position_m(3)+eef_rot6d(6)+xhand_target_rad(12)",
            "contact_force_source": "raw.hand_contact",
            "contact_force_unit": str(
                meta.get("tactile_unit", "sdk_scaled_unknown_si")
            ),
            "contact_force_si_verified": bool(
                meta.get("tactile_si_unit_verified", False)
            ),
            "contact_force_frame": "xhand_sensor_native_axes_per_finger",
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


def _dataset_row_slices(dataset: h5py.Dataset) -> Iterator[slice]:
    row_bytes = int(dataset.dtype.itemsize * np.prod(dataset.shape[1:], dtype=np.int64))
    rows_per_chunk = max(1, _VALIDATION_CHUNK_BYTES // max(1, row_bytes))
    for start in range(0, dataset.shape[0], rows_per_chunk):
        yield slice(start, min(dataset.shape[0], start + rows_per_chunk))


def _expected_specs(
    length: int, config: ProcessingConfig
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    specs = {
        name: ((length, *tail_shape), dtype)
        for name, (tail_shape, dtype) in _CORE_DATASET_SPECS.items()
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
            (length, config.pointcloud.num_points, 6),
            np.dtype(np.float32),
        )
    return specs


def validate_processed_hdf5(
    path: str | Path, config: ProcessingConfig
) -> dict[str, Any]:
    """Fail closed on a processed Real HDF5 v10 artifact."""

    artifact = Path(path)
    with h5py.File(artifact, "r") as source:
        length = int(source.attrs.get("episode_steps", -1))
        if length < config.min_episode_frames:
            raise ValueError(f"{artifact.name}: episode_steps={length} is too short")
        specs = _expected_specs(length, config)
        validate_processed_payload(
            source,
            expected_specs=specs,
            length=length,
            label=artifact.name,
            validate_rgbd=config.profile.needs_rgb,
            pointcloud_workspace=(
                config.pointcloud.workspace if config.profile.needs_pointcloud else None
            ),
        )
        for key, (shape, dtype) in specs.items():
            dataset = source[key]
            if (
                dataset.compression != "gzip"
                or int(dataset.compression_opts) != config.gzip_level
            ):
                raise ValueError(f"{artifact.name}: invalid {key} compression")
        if config.profile.needs_rgb:
            scale = float(source.attrs.get("depth_scale_m_per_unit", np.nan))
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"{artifact.name}: invalid depth scale")
            if (
                str(source.attrs.get("camera_intrinsic_semantics", ""))
                != "resized_color_intrinsics_for_depth_to_color_aligned_depth"
                or str(source.attrs.get("camera_extrinsic_semantics", ""))
                != "T_xarm_base_from_color;native_color_optical_to_xarm_base"
                or str(source.attrs.get("depth_transform", ""))
                != "depth_to_color_aligned_resize_no_crop_nearest"
            ):
                raise ValueError(f"{artifact.name}: invalid aligned RGB-D semantics")
            depth_k = np.asarray(
                source.attrs.get("source_camera_depth_intrinsics_native", ()),
                dtype=np.float64,
            )
            color_from_depth = np.asarray(
                source.attrs.get("camera_T_color_from_depth", ()),
                dtype=np.float64,
            )
            if depth_k.shape != (9,) or not np.all(np.isfinite(depth_k)):
                raise ValueError(f"{artifact.name}: invalid native depth intrinsics")
            validate_rigid_transform(
                color_from_depth,
                label="camera_T_color_from_depth",
            )
        if config.profile.needs_pointcloud:
            if (
                not np.array_equal(
                    np.asarray(source.attrs.get("point_cloud_shape", ())),
                    np.asarray((config.pointcloud.num_points, 6)),
                )
                or str(source.attrs.get("point_cloud_frame", "")) != "xarm_base"
                or str(source.attrs.get("point_cloud_color_source", ""))
                != POINT_CLOUD_COLOR_SOURCE
                or str(source.attrs.get("point_cloud_policy_id", ""))
                != POINT_CLOUD_POLICY_ID
                or str(source.attrs.get("point_cloud_config_sha256", ""))
                != config.pointcloud.sha256
                or str(source.attrs.get("point_cloud_table_plane_abcd_json", ""))
                != _json(
                    None
                    if config.table_plane_abcd is None
                    else list(config.table_plane_abcd)
                )
                or str(source.attrs.get("point_cloud_sampling", ""))
                != POINT_CLOUD_SAMPLING
                or str(source.attrs.get("point_cloud_transform", ""))
                != POINT_CLOUD_TRANSFORM
            ):
                raise ValueError(f"{artifact.name}: invalid v10 point-cloud semantics")
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
        visual_profile = config.profile.needs_rgb or config.profile.needs_pointcloud
        expected_state_alignment = (
            "camera_source_aligned_state" if visual_profile else "control_grid_state"
        )
        expected_reference = (
            "camera_source_monotonic_ns"
            if visual_profile
            else "grid_anchor_monotonic_ns"
        )
        expected_deployment_equivalent = bool(config.profile.needs_pointcloud)
        endpoint_delta_tolerance_rad = config.endpoint_delta_tolerance_rad
        persisted_endpoint_delta = float(
            source.attrs.get("endpoint_delta_tolerance_rad", np.nan)
        )
        if (
            str(source.attrs.get("state_alignment", "")) != expected_state_alignment
            or str(source.attrs.get("observation_reference", "")) != expected_reference
            or not np.isclose(
                float(source.attrs.get("max_observation_skew_s", np.nan)),
                config.max_observation_skew_s,
                rtol=0.0,
                atol=1e-15,
            )
            or (
                config.arm_max_delta_rad_per_tick is None
                and not np.isnan(
                    float(source.attrs.get("arm_max_delta_rad_per_tick", np.nan))
                )
            )
            or (
                config.arm_max_delta_rad_per_tick is not None
                and not np.isclose(
                    float(source.attrs.get("arm_max_delta_rad_per_tick", np.nan)),
                    config.arm_max_delta_rad_per_tick,
                    rtol=0.0,
                    atol=1e-15,
                )
            )
            or not np.isclose(
                float(source.attrs.get("hand_max_delta_rad_per_tick", np.nan)),
                config.hand_max_delta_rad_per_tick,
                rtol=0.0,
                atol=1e-15,
            )
            or (
                endpoint_delta_tolerance_rad is None
                and not np.isnan(persisted_endpoint_delta)
            )
            or (
                endpoint_delta_tolerance_rad is not None
                and (
                    not np.isfinite(persisted_endpoint_delta)
                    or not np.isclose(
                        persisted_endpoint_delta,
                        endpoint_delta_tolerance_rad,
                        rtol=0.0,
                        atol=1e-15,
                    )
                )
            )
            or str(source.attrs.get("action_semantics", ""))
            != "deployment_grid_rate_limited_target"
            or bool(source.attrs.get("deployment_equivalent", False))
            != expected_deployment_equivalent
        ):
            raise ValueError(f"{artifact.name}: invalid deployment data contract")
        if str(source.attrs.get("source_contiguity", "")) != (
            "segment_ends_in_provenance"
        ):
            raise ValueError(f"{artifact.name}: invalid source-contiguity contract")
        for key in (
            "processing_config_json",
            "quality_summary_json",
        ):
            try:
                value = json.loads(str(source.attrs[key]))
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{artifact.name}: invalid {key}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{artifact.name}: {key} must encode an object")
        if json.loads(str(source.attrs["processing_config_json"])) != config.to_dict():
            raise ValueError(f"{artifact.name}: processing config mismatch")
        provenance = validate_processed_provenance(
            source,
            expected_profile=config.profile.value,
            label=artifact.name,
        )
        segment_ends = provenance.segment_ends
        dt = float(source.attrs.get("dt", np.nan))
        tolerance_s = float(source.attrs.get("source_contiguity_tolerance_s", np.nan))
        if (
            not np.isfinite(tolerance_s)
            or tolerance_s <= 0.0
            or not np.isclose(
                tolerance_s,
                max(1e-7, dt * config.grid_dt_relative_tolerance),
                rtol=0.0,
                atol=1e-15,
            )
        ):
            raise ValueError(f"{artifact.name}: invalid contiguity tolerance")
        segment_starts = np.concatenate(
            (np.asarray([0], dtype=np.int64), segment_ends[:-1])
        )
        full_window_count = sum(
            max(0, int(end - start) - config.horizon + 1)
            for start, end in zip(segment_starts, segment_ends, strict=True)
        )
        if full_window_count < config.min_full_windows:
            raise ValueError(f"{artifact.name}: insufficient source-contiguous windows")
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
    allow_legacy_v23: bool = False,
) -> dict[str, Any]:
    """Publish a complete one-to-one batch, or publish nothing on rejection.

    Legacy raw v23 is rejected by default; callers must opt in explicitly and
    the option is passed to every raw reader opened by this transaction.
    """

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
            with _open_episode(episode, allow_legacy_v23=allow_legacy_v23) as reader:
                reader.require_valid(purpose="offline processing")
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
            with _open_episode(
                decision.source_path, allow_legacy_v23=allow_legacy_v23
            ) as reader:
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
        atomic_publish(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report
