"""Processed-v12 schema, provenance, specifications, and strict validation."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np

from dexmani_real.dataset.contracts import OutputProfile, ProcessingConfig
from dexmani_real.dataset.pointcloud import validate_rigid_transform
from dexmani_real.planning.kinematics.pose import validate_canonical_rot6d
from dexmani_real.recording.storage.schema import SEMANTIC_META_ATTRS


PROCESSED_SCHEMA_NAME = "dexmani-real-processed-hdf5"
PROCESSED_SCHEMA_VERSION = 12
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
_CONTACT_FORCE_UNIT = str(SEMANTIC_META_ATTRS["tactile_unit"])
_CONTACT_FORCE_SI_VERIFIED = bool(SEMANTIC_META_ATTRS["tactile_si_unit_verified"])
_CONTACT_FORCE_FRAME = "xhand_sensor_native_axes_per_finger"
_FINGERTIP_POINTS_FRAME = str(SEMANTIC_META_ATTRS["hand_fingertip_frame"])
_FINGERTIP_POINTS_UNIT = "m"
_ACTION_EE_FRAME = str(SEMANTIC_META_ATTRS["action_arm_ee_frame"])


def _strict_bool_attr(attrs: Any, name: str) -> bool:
    """Read one schema boolean without accepting truthy strings or integers."""
    try:
        value = attrs[name]
    except KeyError as exc:
        raise ValueError(f"{name} is a required boolean HDF5 attribute") from exc
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean HDF5 attribute")
    return bool(value)


def _strict_integer_attr(attrs: Any, name: str) -> int:
    """Read one schema integer without truncating floats or accepting booleans."""
    try:
        value = attrs[name]
    except KeyError as exc:
        raise ValueError(f"{name} is a required integer HDF5 attribute") from exc
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer HDF5 attribute")
    return int(value)


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
    hard_invalid_reason_names: tuple[str, ...]


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
    _validate_processed_structure(
        source,
        expected_specs=expected_specs,
        length=length,
        label=label,
    )
    for key, (_expected_shape, expected_dtype) in expected_specs.items():
        dataset = source[key]
        dtype = np.dtype(expected_dtype)
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


def _validate_processed_structure(
    source: h5py.File | h5py.Group,
    *,
    expected_specs: Mapping[str, tuple[tuple[int, ...], np.dtype[Any]]],
    length: int,
    label: str,
) -> None:
    """Check the HDF5 layout without scanning payload values."""
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
    hard_invalid_value = source_decision.get("hard_invalid_reason_names")
    if (
        not isinstance(hard_invalid_value, list)
        or any(not isinstance(name, str) for name in hard_invalid_value)
        or len(set(hard_invalid_value)) != len(hard_invalid_value)
        or not set(hard_invalid_value).issubset(reason_names)
    ):
        raise ValueError(f"{label}: invalid hard-invalid reason names")
    hard_invalid_reason_names = tuple(hard_invalid_value)

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
        hard_invalid_reason_names=hard_invalid_reason_names,
    )


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


def _validate_processed_output_structure(
    path: str | Path, config: ProcessingConfig
) -> dict[str, Any]:
    """Reopen one just-written output and verify its publishable HDF5 layout.

    Writer-side sanity is intentionally bounded to metadata, keys, dtype, and
    shape.  Payload values and semantic attributes are fully checked only at
    the explicit ``verify_output`` step or by a consumer/export boundary.
    """
    artifact = Path(path)
    with h5py.File(artifact, "r") as source:
        length = int(source.attrs.get("episode_steps", -1))
        specs = _expected_specs(length, config)
        if length < config.min_episode_frames:
            raise ValueError(f"{artifact.name}: episode_steps={length} is too short")
        if str(source.attrs.get("schema_name", "")) != PROCESSED_SCHEMA_NAME:
            raise ValueError(f"{artifact.name}: invalid schema_name")
        if int(source.attrs.get("schema_version", -1)) != PROCESSED_SCHEMA_VERSION:
            raise ValueError(f"{artifact.name}: invalid schema_version")
        if str(source.attrs.get("domain", "")) != "real":
            raise ValueError(f"{artifact.name}: domain must be real")
        if str(source.attrs.get("profile", "")) != config.profile.value:
            raise ValueError(f"{artifact.name}: profile mismatch")
        if not isinstance(source.get("provenance"), h5py.Group):
            raise ValueError(f"{artifact.name}: provenance is not an HDF5 group")
        _validate_processed_structure(
            source,
            expected_specs=specs,
            length=length,
            label=artifact.name,
        )
        for key in specs:
            dataset = source[key]
            if (
                dataset.compression != "gzip"
                or int(dataset.compression_opts) != config.gzip_level
            ):
                raise ValueError(f"{artifact.name}: invalid {key} compression")
    return {
        "path": artifact.name,
        "frames": length,
        "keys": sorted(specs),
        "level": "structural",
    }


def validate_processed_hdf5(
    path: str | Path, config: ProcessingConfig
) -> dict[str, Any]:
    """Fail closed on a processed Real HDF5 v12 artifact."""

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
                raise ValueError(
                    f"{artifact.name}: invalid v{PROCESSED_SCHEMA_VERSION} point-cloud semantics"
                )
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
        expected_contact_source = (
            "camera_causal_tactile_sum"
            if visual_profile
            else "control_grid_tactile_sum"
        )
        expected_contact_alignment = (
            "newest_source_not_after_camera_within_max_observation_skew"
            if visual_profile
            else "newest_source_not_after_grid_within_max_observation_skew"
        )
        contact_force_si_verified = _strict_bool_attr(
            source.attrs, "contact_force_si_verified"
        )
        contact_force_fresh_required = _strict_bool_attr(
            source.attrs, "contact_force_fresh_required"
        )
        contact_force_calibrated_required = _strict_bool_attr(
            source.attrs, "contact_force_calibrated_required"
        )
        contact_force_unit_code = _strict_integer_attr(
            source.attrs, "contact_force_unit_code"
        )
        contact_force_causal_to_reference = _strict_bool_attr(
            source.attrs, "contact_force_causal_to_reference"
        )
        contact_force_hand_source_match_required = _strict_bool_attr(
            source.attrs, "contact_force_hand_source_match_required"
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
            or str(source.attrs.get("action_semantics", ""))
            != "teleop_published_joint_target"
            or str(source.attrs.get("contact_force_unit", "")) != _CONTACT_FORCE_UNIT
            or contact_force_si_verified is not _CONTACT_FORCE_SI_VERIFIED
            or str(source.attrs.get("contact_force_frame", "")) != _CONTACT_FORCE_FRAME
            or str(source.attrs.get("contact_force_source", ""))
            != expected_contact_source
            or str(source.attrs.get("contact_force_alignment", ""))
            != expected_contact_alignment
            or not contact_force_fresh_required
            or not contact_force_calibrated_required
            or contact_force_unit_code != 0
            or not contact_force_causal_to_reference
            or not contact_force_hand_source_match_required
            or str(source.attrs.get("fingertip_points_frame", ""))
            != _FINGERTIP_POINTS_FRAME
            or str(source.attrs.get("fingertip_points_unit", ""))
            != _FINGERTIP_POINTS_UNIT
            or str(source.attrs.get("action_ee_frame", "")) != _ACTION_EE_FRAME
        ):
            raise ValueError(f"{artifact.name}: invalid processed data contract")
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
