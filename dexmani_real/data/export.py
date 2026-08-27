"""Transactional export of processed HDF5 v10 episodes to Policy Zarr v5."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import zarr

from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.data.contracts import OutputProfile
from dexmani_real.data.process import (
    PROCESSED_SCHEMA_NAME,
    PROCESSED_SCHEMA_VERSION,
    validate_processed_payload,
    validate_processed_provenance,
)
from dexmani_real.sensor.pointcloud import (
    POINT_CLOUD_COLOR_SOURCE,
    POINT_CLOUD_POLICY_ID,
    POINT_CLOUD_SAMPLING,
    POINT_CLOUD_TRANSFORM,
)
from dexmani_real.utils.atomic_io import atomic_publish, target_is_occupied

POLICY_ZARR_SCHEMA_NAME = "dexmani-real-policy-zarr"
POLICY_ZARR_SCHEMA_VERSION = 5


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
    segment_lengths: tuple[int, ...]


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).strip()


def _discover_processed_hdf5_paths(source_root: Path) -> tuple[Path, ...]:
    """Return direct processed artifacts from one task directory."""

    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    paths = tuple(
        sorted(
            path
            for path in source_root.iterdir()
            if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}
        )
    )
    if not paths:
        raise FileNotFoundError(f"no processed HDF5 files found in {source_root}")
    return paths


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
        if not profile.needs_pointcloud:
            raise ValueError(
                f"{path.name}: Policy Zarr v5 requires a pointcloud processed profile"
            )
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
        dt = float(source.attrs.get("dt", np.nan))
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"{path.name}: dt must be finite and positive")
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
        resolved_pointcloud: PointCloudConfig | None = None
        shapes: dict[str, tuple[int, ...]] = {}
        dtypes: dict[str, np.dtype[Any]] = {}
        semantics: dict[str, Any] = {
            "obs_alignment": _text(source.attrs.get("obs_alignment", "")),
            "observation_reference": _text(
                source.attrs.get("observation_reference", "")
            ),
            "state_alignment": _text(source.attrs.get("state_alignment", "")),
            "max_observation_skew_s": float(
                source.attrs.get("max_observation_skew_s", np.nan)
            ),
            "action_semantics": _text(source.attrs.get("action_semantics", "")),
            "arm_max_delta_rad_per_tick": (
                None
                if np.isnan(
                    float(source.attrs.get("arm_max_delta_rad_per_tick", np.nan))
                )
                else float(source.attrs["arm_max_delta_rad_per_tick"])
            ),
            "hand_max_delta_rad_per_tick": float(
                source.attrs.get("hand_max_delta_rad_per_tick", np.nan)
            ),
            # This field is part of the processed deployment contract.  A
            # missing/NaN value must not silently downgrade the endpoint gate.
            "endpoint_delta_tolerance_rad": float(
                source.attrs.get("endpoint_delta_tolerance_rad", np.nan)
            ),
            "deployment_equivalent": bool(
                source.attrs.get("deployment_equivalent", False)
            ),
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
            semantics["obs_alignment"] != "obs[t]_before_action[t]"
            or semantics["observation_reference"] != "camera_source_monotonic_ns"
            or semantics["state_alignment"] != "camera_source_aligned_state"
            or not np.isfinite(semantics["max_observation_skew_s"])
            or semantics["max_observation_skew_s"] <= 0.0
            or semantics["action_semantics"] != "deployment_grid_rate_limited_target"
            or (
                semantics["arm_max_delta_rad_per_tick"] is not None
                and (
                    not np.isfinite(semantics["arm_max_delta_rad_per_tick"])
                    or semantics["arm_max_delta_rad_per_tick"] <= 0.0
                )
            )
            or not np.isfinite(semantics["hand_max_delta_rad_per_tick"])
            or semantics["hand_max_delta_rad_per_tick"] <= 0.0
            or not np.isfinite(semantics["endpoint_delta_tolerance_rad"])
            or semantics["endpoint_delta_tolerance_rad"] < 0.0
            or not semantics["deployment_equivalent"]
            or not semantics["contact_force_unit"]
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
            try:
                processing_config = json.loads(
                    _text(source.attrs.get("processing_config_json", ""))
                )
                pointcloud_config = processing_config["pointcloud"]
                table_plane_abcd = processing_config["table_plane_abcd"]
                if not isinstance(pointcloud_config, dict):
                    raise TypeError("pointcloud config must be an object")
                resolved_pointcloud = PointCloudConfig(**pointcloud_config)
                if resolved_pointcloud.to_dict() != pointcloud_config:
                    raise ValueError("pointcloud config is not canonical")
                if resolved_pointcloud.num_points != source["point_cloud"].shape[1]:
                    raise ValueError("pointcloud config count does not match dataset")
                if table_plane_abcd is not None:
                    plane = np.asarray(table_plane_abcd, dtype=np.float64)
                    norm = (
                        float(np.linalg.norm(plane[:3])) if plane.shape == (4,) else 0.0
                    )
                    if (
                        plane.shape != (4,)
                        or not np.all(np.isfinite(plane))
                        or norm <= 0.0
                        or plane[2] / norm <= 0.0
                    ):
                        raise ValueError("table plane must be finite and upward")
                canonical_table_plane = json.dumps(
                    table_plane_abcd,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            except (KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"{path.name}: invalid persisted point-cloud config"
                ) from exc
            pointcloud_config_sha256 = resolved_pointcloud.sha256
            point_cloud_semantics = {
                "frame": _text(source.attrs.get("point_cloud_frame", "")),
                "color_source": _text(source.attrs.get("point_cloud_color_source", "")),
                "policy_id": _text(source.attrs.get("point_cloud_policy_id", "")),
                "config_sha256": _text(
                    source.attrs.get("point_cloud_config_sha256", "")
                ),
                "table_plane_abcd_json": _text(
                    source.attrs.get("point_cloud_table_plane_abcd_json", "")
                ),
                "sampling": _text(source.attrs.get("point_cloud_sampling", "")),
                "transform": _text(source.attrs.get("point_cloud_transform", "")),
            }
            if point_cloud_semantics != {
                "frame": "xarm_base",
                "color_source": POINT_CLOUD_COLOR_SOURCE,
                "policy_id": POINT_CLOUD_POLICY_ID,
                "config_sha256": pointcloud_config_sha256,
                "table_plane_abcd_json": canonical_table_plane,
                "sampling": POINT_CLOUD_SAMPLING,
                "transform": POINT_CLOUD_TRANSFORM,
            }:
                raise ValueError(f"{path.name}: invalid Real point-cloud semantics")
            semantics.update(
                {
                    f"point_cloud_{key}": value
                    for key, value in point_cloud_semantics.items()
                }
            )
        expected_specs: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
            "joint_state": ((length, 19), np.dtype(np.float32)),
            "action": ((length, 19), np.dtype(np.float32)),
            "action_ee": ((length, 21), np.dtype(np.float32)),
            "contact_force": ((length, 5, 3), np.dtype(np.float32)),
            "fingertip_points": ((length, 5, 3), np.dtype(np.float32)),
        }
        if profile.needs_rgb:
            expected_specs.update(
                {
                    key: (tuple(int(value) for value in source[key].shape), dtype)
                    for key, dtype in (
                        ("rgb", np.dtype(np.uint8)),
                        ("depth", np.dtype(np.uint16)),
                        ("camera_intrinsic", np.dtype(np.float32)),
                        ("camera_extrinsic", np.dtype(np.float32)),
                    )
                }
            )
        if profile.needs_pointcloud:
            if resolved_pointcloud is None:
                raise ValueError(f"{path.name}: point-cloud config is missing")
            expected_specs["point_cloud"] = (
                (length, resolved_pointcloud.num_points, 6),
                np.dtype(np.float32),
            )
        validate_processed_payload(
            source,
            expected_specs=expected_specs,
            length=length,
            label=path.name,
            validate_rgbd=profile.needs_rgb,
            pointcloud_workspace=(
                None if resolved_pointcloud is None else resolved_pointcloud.workspace
            ),
        )
        provenance = validate_processed_provenance(
            source,
            expected_profile=profile.value,
            label=path.name,
        )
        segment_lengths = provenance.segment_lengths
        for key in profile.dataset_keys:
            shapes[key] = tuple(int(value) for value in source[key].shape[1:])
            dtypes[key] = np.dtype(source[key].dtype)
    return _Artifact(
        path=path,
        length=length,
        profile=profile,
        task_name=task_name,
        dt=dt,
        dataset_shapes=shapes,
        dataset_dtypes=dtypes,
        semantic_attrs=semantics,
        segment_lengths=segment_lengths,
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


def _load_artifacts(
    input_root: str | Path, config: PolicyZarrExportConfig
) -> tuple[_Artifact, ...]:
    """Inspect one complete task input before any Zarr output is created."""

    paths = _discover_processed_hdf5_paths(Path(input_root))
    artifacts = tuple(_inspect_artifact(path, config) for path in paths)
    _validate_uniform(artifacts)
    return artifacts


def _export_plan_report(
    artifacts: tuple[_Artifact, ...], *, input_root: str | Path
) -> dict[str, Any]:
    """Summarize the validated source layout used by preflight and publishing."""

    first = artifacts[0]
    episode_ends = np.cumsum(
        [length for artifact in artifacts for length in artifact.segment_lengths],
        dtype=np.int64,
    )
    return {
        "input_root": str(Path(input_root).resolve()),
        "task_name": first.task_name,
        "profile": first.profile.value,
        "dt": first.dt,
        "source_file_count": len(artifacts),
        "episode_count": len(episode_ends),
        "total_frames": int(episode_ends[-1]),
        "episode_ends": episode_ends.tolist(),
        "dataset_keys": sorted(first.dataset_shapes),
    }


def preflight_processed_hdf5_to_zarr(
    input_root: str | Path,
    config: PolicyZarrExportConfig | None = None,
) -> dict[str, Any]:
    """Read and validate export inputs without creating or modifying a Zarr store.

    The preflight checks the same per-artifact schema, deployment semantics,
    task-level uniformity, and finite floating payload values that publishing
    checks before writing. It deliberately does not create a temporary Zarr
    store, so it is suitable for a quick fail-closed admission check.
    """

    resolved = config or PolicyZarrExportConfig()
    artifacts = _load_artifacts(input_root, resolved)
    return _export_plan_report(artifacts, input_root=input_root)


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
        "episode_start_policy": "full_history",
        **first.semantic_attrs,
    }
    if dict(root.attrs) != expected_attrs:
        raise ValueError("Zarr root semantic attributes mismatch")
    expected_keys = set(artifacts[0].dataset_shapes)
    if set(root["data"].array_keys()) != expected_keys:
        raise ValueError("Zarr data keys do not match processed HDF5")
    expected_ends = np.cumsum(
        [length for artifact in artifacts for length in artifact.segment_lengths],
        dtype=np.int64,
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
    if target_is_occupied(target):
        raise FileExistsError(f"refusing to overwrite existing policy Zarr: {target}")
    artifacts = _load_artifacts(source_root, resolved)
    # All payload admission (including finite checks) completes before a
    # staging directory is created, so a rejected source leaves no partial
    # export artifact behind.
    total_frames = sum(artifact.length for artifact in artifacts)
    episode_ends = np.cumsum(
        [length for artifact in artifacts for length in artifact.segment_lengths],
        dtype=np.int64,
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
                "episode_start_policy": "full_history",
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
        **_export_plan_report(artifacts, input_root=source_root),
    }
