"""Transactional one-processed-file to one-Policy-Zarr-episode export."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import zarr

from dexmani_real.dataset.contracts import validate_processed_task_name
from dexmani_real.dataset.processed import validate_processed_hdf5
from dexmani_real.utils.atomic_io import atomic_publish, target_is_occupied

POLICY_ZARR_SCHEMA_NAME = "dexmani-real-policy-zarr"
POLICY_ZARR_SCHEMA_VERSION = 12
ExportProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class PolicyZarrExportConfig:
    """Resolved storage and task-consistency policy for one Real task store."""

    chunk_frames: int = 100
    compression_level: int = 3
    expected_task_name: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.chunk_frames, bool)
            or not isinstance(self.chunk_frames, int)
            or self.chunk_frames <= 0
        ):
            raise ValueError("chunk_frames must be a positive integer")
        if (
            isinstance(self.compression_level, bool)
            or not isinstance(self.compression_level, int)
            or not 0 <= self.compression_level <= 9
        ):
            raise ValueError("compression_level must be an integer in [0, 9]")
        if self.expected_task_name is not None:
            validate_processed_task_name(self.expected_task_name)


@dataclass(frozen=True)
class _Artifact:
    """Validated HDF5 input metadata used only during export, never serialized."""

    path: Path
    length: int
    task_name: str
    dt: float
    dataset_shapes: dict[str, tuple[int, ...]]
    dataset_dtypes: dict[str, np.dtype[Any]]
    semantic_attrs: dict[str, Any]


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
    # The processed boundary proves payload integrity and row preservation once.
    validation = validate_processed_hdf5(path)
    with h5py.File(path, "r") as source:
        task_name = str(source.attrs["task_name"])
        if (
            config.expected_task_name is not None
            and task_name != config.expected_task_name
        ):
            raise ValueError(
                f"{path.name}: task_name={task_name!r}, expected {config.expected_task_name!r}"
            )
        semantic_keys = [
            "obs_alignment",
            "observation_alignment",
            "state_alignment",
            "action_semantics",
            "action_ee_frame",
            "eef_pose_frame",
            "eef_pose_components",
            "eef_pose_derivation",
            "eef_pose_algorithm_id",
            "contact_force_source",
            "contact_force_representation",
            "contact_force_unit",
            "contact_force_si_verified",
            "contact_force_frame",
            "tactile_force_representation",
            "tactile_force_finger_order",
            "tactile_force_sensor_order",
            "tactile_force_point_order",
            "tactile_force_axis_labels",
            "tactile_force_unit",
            "tactile_force_si_verified",
            "tactile_force_spatial_geometry_verified",
            "fingertip_points_frame",
            "fingertip_points_unit",
            "fingertip_points_derivation",
            "fingertip_points_policy_id",
            "fingertip_config_json",
            "depth_scale_m_per_unit",
            "depth_invalid_value",
            "camera_intrinsic_semantics",
            "camera_extrinsic_semantics",
            "point_cloud_frame",
            "point_cloud_color_source",
            "point_cloud_policy_id",
            "point_cloud_table_plane_abcd_json",
            "point_cloud_sampling",
            "point_cloud_transform",
            "processing_config_json",
        ]
        semantics = {}
        for key in semantic_keys:
            value = source.attrs[key]
            semantics[key] = value.item() if isinstance(value, np.generic) else value
        dataset_keys = tuple(validation["keys"])
        return _Artifact(
            path=path,
            length=validation["frames"],
            task_name=task_name,
            dt=float(source.attrs["dt"]),
            dataset_shapes={key: source[key].shape[1:] for key in dataset_keys},
            dataset_dtypes={key: source[key].dtype for key in dataset_keys},
            semantic_attrs=semantics,
        )


def _validate_uniform(artifacts: tuple[_Artifact, ...]) -> None:
    first = artifacts[0]
    for artifact in artifacts[1:]:
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


def _report_progress(
    callback: ExportProgressCallback | None,
    phase: str,
    completed: int,
    total: int,
) -> None:
    """Report cumulative work from one export phase when a caller requested it."""

    if callback is not None:
        callback(phase, completed, total)


def _load_artifacts(
    input_root: str | Path,
    config: PolicyZarrExportConfig,
    *,
    progress_callback: ExportProgressCallback | None = None,
) -> tuple[_Artifact, ...]:
    """Inspect one complete task input before any Zarr output is created."""

    paths = _discover_processed_hdf5_paths(Path(input_root))
    _report_progress(progress_callback, "validate", 0, len(paths))
    artifacts: list[_Artifact] = []
    for index, path in enumerate(paths, start=1):
        artifacts.append(_inspect_artifact(path, config))
        _report_progress(progress_callback, "validate", index, len(paths))
    _validate_uniform(tuple(artifacts))
    return tuple(artifacts)


def _export_plan_report(
    artifacts: tuple[_Artifact, ...],
    *,
    input_root: str | Path,
) -> dict[str, Any]:
    """Summarize the validated source layout used by preflight and publishing."""

    first = artifacts[0]
    episode_ends = np.cumsum(
        [artifact.length for artifact in artifacts], dtype=np.int64
    )
    return {
        "input_root": str(Path(input_root).resolve()),
        "task_name": first.task_name,
        "dt": first.dt,
        "source_file_count": len(artifacts),
        "episode_count": len(artifacts),
        "total_frames": int(episode_ends[-1]),
        "episode_ends": episode_ends.tolist(),
        "dataset_keys": sorted(first.dataset_shapes),
    }


def preflight_processed_hdf5_to_zarr(
    input_root: str | Path,
    config: PolicyZarrExportConfig | None = None,
    *,
    progress_callback: ExportProgressCallback | None = None,
) -> dict[str, Any]:
    """Read and validate export inputs without creating or modifying a Zarr store.

    The preflight checks the same per-artifact schema, deployment semantics,
    task-level uniformity, and finite floating payload values that publishing
    checks before writing. It deliberately does not create a temporary Zarr
    store, so it is suitable for a quick fail-closed admission check.
    """

    resolved = config or PolicyZarrExportConfig()
    artifacts = _load_artifacts(
        input_root,
        resolved,
        progress_callback=progress_callback,
    )
    return _export_plan_report(
        artifacts,
        input_root=input_root,
    )


def _copy_data(
    artifacts: tuple[_Artifact, ...],
    data_group: zarr.Group,
    *,
    chunk_frames: int,
    progress_callback: ExportProgressCallback | None = None,
) -> None:
    offset = 0
    total_frames = sum(artifact.length for artifact in artifacts)
    _report_progress(progress_callback, "write", 0, total_frames)
    for artifact in artifacts:
        with h5py.File(artifact.path, "r") as source:
            for row_start in range(0, artifact.length, chunk_frames):
                row_end = min(artifact.length, row_start + chunk_frames)
                target_slice = slice(offset + row_start, offset + row_end)
                for key in artifacts[0].dataset_shapes:
                    block = np.asarray(source[key][row_start:row_end])
                    data_group[key][target_slice] = block
                _report_progress(
                    progress_callback,
                    "write",
                    offset + row_end,
                    total_frames,
                )
        offset += artifact.length


def _validate_zarr(
    path: Path,
    artifacts: tuple[_Artifact, ...],
    *,
    progress_callback: ExportProgressCallback | None = None,
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
    if root["meta"]["episode_ends"].dtype != np.dtype(np.int64) or not np.array_equal(
        root["meta"]["episode_ends"][:], expected_ends
    ):
        raise ValueError("Zarr episode_ends mismatch")
    total = int(expected_ends[-1])
    _report_progress(progress_callback, "verify", 0, len(expected_keys))
    completed_keys = 0
    for key in sorted(expected_keys):
        array = root["data"][key]
        expected_shape = (total,) + artifacts[0].dataset_shapes[key]
        if (
            array.shape != expected_shape
            or np.dtype(array.dtype) != artifacts[0].dataset_dtypes[key]
        ):
            raise ValueError(f"Zarr {key} shape/dtype mismatch")
        completed_keys += 1
        _report_progress(
            progress_callback, "verify", completed_keys, len(expected_keys)
        )


def export_processed_hdf5_to_zarr(
    input_root: str | Path,
    output_path: str | Path,
    config: PolicyZarrExportConfig | None = None,
    *,
    progress_callback: ExportProgressCallback | None = None,
) -> dict[str, Any]:
    """Atomically publish one complete Zarr episode for each processed file.

    ``progress_callback`` receives ``(phase, completed, total)`` for validation,
    Zarr writing, and structural verification. It does not affect export
    admission or publication behavior.
    """

    resolved = config or PolicyZarrExportConfig()
    source_root = Path(input_root)
    target = Path(output_path)
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    if target_is_occupied(target):
        raise FileExistsError(f"refusing to overwrite existing policy Zarr: {target}")
    artifacts = _load_artifacts(
        source_root,
        resolved,
        progress_callback=progress_callback,
    )
    # All payload admission (including finite checks) completes before a
    # staging directory is created, so a rejected source leaves no partial
    # export artifact behind.
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
        _copy_data(
            artifacts,
            data_group,
            chunk_frames=resolved.chunk_frames,
            progress_callback=progress_callback,
        )
        _validate_zarr(
            staging,
            artifacts,
            progress_callback=progress_callback,
        )
        atomic_publish(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "output_path": str(target.resolve()),
        **_export_plan_report(
            artifacts,
            input_root=source_root,
        ),
    }
