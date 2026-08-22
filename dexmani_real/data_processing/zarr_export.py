"""Transactional export of processed Real HDF5 v4 episodes to minimal Zarr."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import zarr

from dexmani_real.data_processing.contracts import OutputProfile
from dexmani_real.data_processing.pipeline import (
    PROCESSED_SCHEMA_NAME,
    PROCESSED_SCHEMA_VERSION,
)
from dexmani_real.recording.transaction import atomic_publish
from dexmani_real.sensor.pointcloud import (
    POINT_CLOUD_COLOR_SOURCE,
    POINT_CLOUD_SAMPLING,
)

POLICY_ZARR_SCHEMA_NAME = "dexmani-real-policy-zarr"
POLICY_ZARR_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class PolicyZarrExportConfig:
    """Resolved storage and task-consistency policy for one Real task store."""

    chunk_frames: int = 100
    compression_level: int = 3
    expected_task_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_frames, int) or self.chunk_frames <= 0:
            raise ValueError("chunk_frames must be a positive integer")
        if (
            not isinstance(self.compression_level, int)
            or not 0 <= self.compression_level <= 9
        ):
            raise ValueError("compression_level must be an integer in [0, 9]")
        if self.expected_task_name is not None and not self.expected_task_name.strip():
            raise ValueError("expected_task_name must be non-empty when provided")

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_frames": self.chunk_frames,
            "compression": {"id": "zstd", "level": self.compression_level},
            "expected_task_name": self.expected_task_name,
        }


@dataclass(frozen=True)
class _Artifact:
    """Validated HDF5 input metadata used only during export, never serialized."""

    path: Path
    length: int
    profile: OutputProfile
    task_name: str
    dt: float
    dataset_shapes: dict[str, tuple[int, ...]]
    dataset_dtypes: dict[str, np.dtype[Any]]
    semantic_attrs: dict[str, Any]


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).strip()


def _validate_dataset(path: Path, key: str, dataset: h5py.Dataset, length: int) -> None:
    expected: dict[str, tuple[tuple[int, ...] | None, np.dtype[Any]]] = {
        "joint_state": ((19,), np.dtype(np.float32)),
        "action": ((19,), np.dtype(np.float32)),
        "action_ee": ((21,), np.dtype(np.float32)),
        "contact_force": ((5, 3), np.dtype(np.float32)),
        "fingertip_points": ((5, 3), np.dtype(np.float32)),
        "rgb": (None, np.dtype(np.uint8)),
        "depth": (None, np.dtype(np.uint16)),
        "camera_intrinsic": ((9,), np.dtype(np.float32)),
        "camera_extrinsic": ((4, 4), np.dtype(np.float32)),
        "point_cloud": (None, np.dtype(np.float32)),
    }
    if dataset.shape[0] != length:
        raise ValueError(f"{path.name}: {key} length mismatch")
    tail, dtype = expected[key]
    if dataset.dtype != dtype:
        raise ValueError(f"{path.name}: {key} dtype must be {dtype}")
    if tail is not None and dataset.shape[1:] != tail:
        raise ValueError(f"{path.name}: {key} tail shape must be {tail}")
    if key == "rgb" and (len(dataset.shape) != 4 or dataset.shape[-1] != 3):
        raise ValueError(f"{path.name}: rgb must be (N,H,W,3)")
    if key == "depth" and len(dataset.shape) != 3:
        raise ValueError(f"{path.name}: depth must be (N,H,W)")
    if key == "point_cloud" and (len(dataset.shape) != 3 or dataset.shape[-1] != 6):
        raise ValueError(f"{path.name}: point_cloud must be (N,P,6)")


def _inspect_artifact(path: Path, config: PolicyZarrExportConfig) -> _Artifact:
    with h5py.File(path, "r") as source:
        if _text(source.attrs.get("schema_name", "")) != PROCESSED_SCHEMA_NAME:
            raise ValueError(f"{path.name}: unsupported processed schema")
        if int(source.attrs.get("schema_version", -1)) != PROCESSED_SCHEMA_VERSION:
            raise ValueError(f"{path.name}: unsupported processed schema version")
        if _text(source.attrs.get("domain", "")) != "real":
            raise ValueError(f"{path.name}: domain must be real")
        try:
            profile = OutputProfile(_text(source.attrs.get("profile", "")))
        except ValueError as exc:
            raise ValueError(f"{path.name}: invalid profile") from exc
        data_keys = {
            key for key, value in source.items() if isinstance(value, h5py.Dataset)
        }
        if data_keys != set(profile.dataset_keys) or set(source.keys()) != data_keys | {
            "provenance"
        }:
            raise ValueError(f"{path.name}: processed data keys do not match profile")
        length = int(source.attrs.get("episode_steps", -1))
        if length <= 0:
            raise ValueError(f"{path.name}: episode_steps must be positive")
        task_name = _text(source.attrs.get("task_name", ""))
        if not task_name or task_name == "unknown":
            raise ValueError(f"{path.name}: explicit task_name is required")
        if (
            config.expected_task_name is not None
            and task_name != config.expected_task_name
        ):
            raise ValueError(
                f"{path.name}: task_name={task_name!r}, expected {config.expected_task_name!r}"
            )
        dt = float(source.attrs.get("dt", np.nan))
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"{path.name}: dt must be finite and positive")
        shapes: dict[str, tuple[int, ...]] = {}
        dtypes: dict[str, np.dtype[Any]] = {}
        for key in profile.dataset_keys:
            _validate_dataset(path, key, source[key], length)
            shapes[key] = tuple(int(value) for value in source[key].shape[1:])
            dtypes[key] = np.dtype(source[key].dtype)
        semantics: dict[str, Any] = {
            "contact_force_unit": _text(source.attrs.get("contact_force_unit", "")),
            "contact_force_si_verified": bool(
                source.attrs.get("contact_force_si_verified", False)
            ),
            "contact_force_frame": _text(source.attrs.get("contact_force_frame", "")),
            "fingertip_points_frame": _text(
                source.attrs.get("fingertip_points_frame", "")
            ),
            "action_ee_frame": _text(source.attrs.get("action_ee_frame", "")),
        }
        if (
            not semantics["contact_force_unit"]
            or not semantics["contact_force_frame"]
            or semantics["fingertip_points_frame"] != "xarm_base"
            or semantics["action_ee_frame"] != "xarm_base"
        ):
            raise ValueError(f"{path.name}: invalid Real core modality semantics")
        if profile.needs_rgb:
            semantics.update(
                {
                    "depth_scale_m_per_unit": float(
                        source.attrs.get("depth_scale_m_per_unit", np.nan)
                    ),
                    "depth_invalid_value": int(
                        source.attrs.get("depth_invalid_value", -1)
                    ),
                    "camera_extrinsic_semantics": _text(
                        source.attrs.get("camera_extrinsic_semantics", "")
                    ),
                }
            )
            if (
                not np.isfinite(semantics["depth_scale_m_per_unit"])
                or semantics["depth_scale_m_per_unit"] <= 0.0
                or semantics["depth_invalid_value"] != 0
                or semantics["camera_extrinsic_semantics"]
                != "T_xarm_base_from_color;native_color_optical_to_xarm_base"
            ):
                raise ValueError(f"{path.name}: invalid Real RGB-D semantics")
        if profile.needs_pointcloud:
            point_cloud_semantics = {
                "frame": _text(source.attrs.get("point_cloud_frame", "")),
                "color_source": _text(source.attrs.get("point_cloud_color_source", "")),
                "sampling": _text(source.attrs.get("point_cloud_sampling", "")),
            }
            if point_cloud_semantics != {
                "frame": "xarm_base",
                "color_source": POINT_CLOUD_COLOR_SOURCE,
                "sampling": POINT_CLOUD_SAMPLING,
            }:
                raise ValueError(f"{path.name}: invalid Real point-cloud semantics")
            semantics.update(
                {
                    f"point_cloud_{key}": value
                    for key, value in point_cloud_semantics.items()
                }
            )
    return _Artifact(
        path=path,
        length=length,
        profile=profile,
        task_name=task_name,
        dt=dt,
        dataset_shapes=shapes,
        dataset_dtypes=dtypes,
        semantic_attrs=semantics,
    )


def _validate_uniform(artifacts: tuple[_Artifact, ...]) -> None:
    first = artifacts[0]
    for artifact in artifacts[1:]:
        if artifact.profile != first.profile:
            raise ValueError("processed HDF5 profiles are not uniform")
        if artifact.task_name != first.task_name:
            raise ValueError("one policy Zarr must contain one task_name")
        if not np.isclose(artifact.dt, first.dt, rtol=0.0, atol=1e-12):
            raise ValueError("processed HDF5 dt values are not uniform")
        if artifact.dataset_shapes != first.dataset_shapes:
            raise ValueError(f"{artifact.path.name}: non-uniform dataset shapes")
        if artifact.dataset_dtypes != first.dataset_dtypes:
            raise ValueError(f"{artifact.path.name}: non-uniform dataset dtypes")
        if artifact.semantic_attrs != first.semantic_attrs:
            raise ValueError(
                f"{artifact.path.name}: non-uniform Real modality semantics"
            )


def _copy_data(
    artifacts: tuple[_Artifact, ...], data_group: zarr.Group, *, chunk_frames: int
) -> dict[str, str]:
    digests = {key: hashlib.sha256() for key in artifacts[0].dataset_shapes}
    offset = 0
    for artifact in artifacts:
        with h5py.File(artifact.path, "r") as source:
            for row_start in range(0, artifact.length, chunk_frames):
                row_end = min(artifact.length, row_start + chunk_frames)
                target_slice = slice(offset + row_start, offset + row_end)
                for key, digest in digests.items():
                    block = np.asarray(source[key][row_start:row_end])
                    if np.issubdtype(block.dtype, np.floating) and not np.all(
                        np.isfinite(block)
                    ):
                        raise ValueError(
                            f"{artifact.path.name}: {key} contains NaN/Inf"
                        )
                    data_group[key][target_slice] = block
                    digest.update(np.ascontiguousarray(block).tobytes())
        offset += artifact.length
    return {key: digest.hexdigest() for key, digest in digests.items()}


def _validate_zarr(
    path: Path,
    artifacts: tuple[_Artifact, ...],
    expected_digests: dict[str, str],
    *,
    chunk_frames: int,
) -> None:
    root = zarr.open_group(str(path), mode="r")
    if set(root.group_keys()) != {"data", "meta"} or set(root.array_keys()):
        raise ValueError("Zarr must contain only data and meta groups")
    if (
        set(root["meta"].array_keys()) != {"episode_ends"}
        or set(root["meta"].group_keys())
        or set(root["data"].group_keys())
    ):
        raise ValueError("Zarr meta must contain only episode_ends")
    first = artifacts[0]
    expected_attrs = {
        "schema_name": POLICY_ZARR_SCHEMA_NAME,
        "schema_version": POLICY_ZARR_SCHEMA_VERSION,
        "domain": "real",
        "profile": first.profile.value,
        "task_name": first.task_name,
        "dt": first.dt,
        **first.semantic_attrs,
    }
    if dict(root.attrs) != expected_attrs:
        raise ValueError("Zarr root semantic attributes mismatch")
    expected_keys = set(artifacts[0].dataset_shapes)
    if set(root["data"].array_keys()) != expected_keys:
        raise ValueError("Zarr data keys do not match processed HDF5")
    expected_ends = np.cumsum(
        [artifact.length for artifact in artifacts], dtype=np.int64
    )
    if not np.array_equal(root["meta"]["episode_ends"][:], expected_ends):
        raise ValueError("Zarr episode_ends mismatch")
    total = int(expected_ends[-1])
    for key in sorted(expected_keys):
        array = root["data"][key]
        expected_shape = (total,) + artifacts[0].dataset_shapes[key]
        if (
            array.shape != expected_shape
            or np.dtype(array.dtype) != artifacts[0].dataset_dtypes[key]
        ):
            raise ValueError(f"Zarr {key} shape/dtype mismatch")
        digest = hashlib.sha256()
        for start in range(0, total, chunk_frames):
            block = np.asarray(array[start : min(total, start + chunk_frames)])
            digest.update(np.ascontiguousarray(block).tobytes())
        if digest.hexdigest() != expected_digests[key]:
            raise ValueError(f"Zarr {key} checksum mismatch")


def export_processed_hdf5_to_zarr(
    input_root: str | Path,
    output_path: str | Path,
    config: PolicyZarrExportConfig | None = None,
) -> dict[str, Any]:
    """Atomically publish data/* + meta/episode_ends, without HDF provenance."""

    resolved = config or PolicyZarrExportConfig()
    source_root = Path(input_root)
    target = Path(output_path)
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing policy Zarr: {target}")
    hdf5_paths = tuple(
        sorted(
            path
            for path in source_root.iterdir()
            if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}
        )
    )
    if not hdf5_paths:
        raise FileNotFoundError(f"no processed HDF5 files found in {source_root}")
    artifacts = tuple(_inspect_artifact(path, resolved) for path in hdf5_paths)
    _validate_uniform(artifacts)
    total_frames = sum(artifact.length for artifact in artifacts)
    episode_ends = np.cumsum(
        [artifact.length for artifact in artifacts], dtype=np.int64
    )
    first = artifacts[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    try:
        root = zarr.open_group(str(staging), mode="w")
        root.attrs.update(
            {
                "schema_name": POLICY_ZARR_SCHEMA_NAME,
                "schema_version": POLICY_ZARR_SCHEMA_VERSION,
                "domain": "real",
                "profile": first.profile.value,
                "task_name": first.task_name,
                "dt": first.dt,
                **first.semantic_attrs,
            }
        )
        data_group = root.create_group("data")
        meta_group = root.create_group("meta")
        compressor = zarr.get_codec({"id": "zstd", "level": resolved.compression_level})
        for key in sorted(first.dataset_shapes):
            tail_shape = first.dataset_shapes[key]
            data_group.create_dataset(
                key,
                shape=(total_frames,) + tail_shape,
                chunks=(min(resolved.chunk_frames, total_frames),) + tail_shape,
                dtype=first.dataset_dtypes[key],
                compressor=compressor,
                overwrite=False,
            )
        meta_group.create_dataset(
            "episode_ends",
            data=episode_ends,
            dtype=np.int64,
            compressor=compressor,
            overwrite=False,
        )
        digests = _copy_data(artifacts, data_group, chunk_frames=resolved.chunk_frames)
        _validate_zarr(
            staging,
            artifacts,
            digests,
            chunk_frames=resolved.chunk_frames,
        )
        atomic_publish(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "output_path": str(target.resolve()),
        "task_name": first.task_name,
        "profile": first.profile.value,
        "episode_count": len(artifacts),
        "total_frames": total_frames,
        "episode_ends": episode_ends.tolist(),
        "dataset_keys": sorted(first.dataset_shapes),
    }
