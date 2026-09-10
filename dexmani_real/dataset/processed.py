"""Processed control-step schema and payload validation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np

from dexmani_real.config.pointcloud import (
    PointCloudConfig,
    POINT_CLOUD_COLOR_SOURCE,
    POINT_CLOUD_POLICY_ID,
    POINT_CLOUD_SAMPLING,
    POINT_CLOUD_TRANSFORM,
)
from dexmani_real.dataset.contracts import (
    OutputProfile,
    ProcessingConfig,
    validate_processed_task_name,
)
from dexmani_real.dataset.pointcloud import validate_rigid_transform
from dexmani_real.planning.kinematics.fingertip import (
    FINGERTIP_POINTS_DERIVATION,
    FINGERTIP_POLICY_ID,
)
from dexmani_real.planning.kinematics.pose import validate_canonical_rot6d

PROCESSED_SCHEMA_NAME = "dexmani-real-processed-hdf5"
PROCESSED_SCHEMA_VERSION = 17
_CONTACT_FORCE_SOURCE = "raw_hand_contact_control_step"
_VALIDATION_CHUNK_BYTES = 64 * 1024 * 1024
_CORE_DATASET_SPECS: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
    "joint_state": ((19,), np.dtype(np.float32)),
    "action": ((19,), np.dtype(np.float32)),
    "action_ee": ((21,), np.dtype(np.float32)),
    "contact_force": ((5, 3), np.dtype(np.float32)),
    "fingertip_points": ((5, 3), np.dtype(np.float32)),
}
_FRAME_CHUNKED_DATASETS = frozenset(("rgb", "depth", "point_cloud"))
# contact_force is the aggregate XHand SDK calc_force per finger, software-bias
# corrected, in the SDK-native numeric scale (SI conversion unverified).
_CONTACT_FORCE_REPRESENTATION = "xhand_sdk_calc_force_fx_fy_fz_bias_corrected"
_CONTACT_FORCE_UNIT = "xhand_sdk_native_unknown_si"
_CONTACT_FORCE_SI_VERIFIED = False
_CONTACT_FORCE_FRAME = "xhand_sensor_native_axes_per_finger"
_FINGERTIP_POINTS_FRAME = "xarm_base"
_FINGERTIP_POINTS_UNIT = "m"
_ACTION_EE_FRAME = "xarm_base"


def _validate_processed_task_name_attr(attrs: Any, *, label: str) -> str:
    """Read and validate the task identity at a persisted-artifact boundary."""

    try:
        return validate_processed_task_name(attrs.get("task_name", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: invalid task_name: {exc}") from exc


def validate_fingertip_points_semantics(
    attrs: Any,
    *,
    label: str,
) -> dict[str, str]:
    """Return the required persisted fingertip derivation and policy identity."""
    values: dict[str, str] = {}
    for key in ("derivation", "policy_id"):
        value = attrs.get(f"fingertip_points_{key}", "")
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        values[key] = str(value).strip()
    if values["derivation"] != FINGERTIP_POINTS_DERIVATION:
        raise ValueError(f"{label}: invalid fingertip_points_derivation")
    if values["policy_id"] != FINGERTIP_POLICY_ID:
        raise ValueError(f"{label}: invalid fingertip_points_policy_id")
    return values


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
        for row_slice in _dataset_row_slices(dataset):
            # Integer image payloads must also be readable. This checks storage
            # integrity without treating sensor pixel values as quality gates.
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
    present_keys = set(source.keys())
    if expected_keys != present_keys:
        raise ValueError(f"{label}: processed data keys are incomplete")
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
    path: str | Path, config: ProcessingConfig | None = None
) -> dict[str, Any]:
    """Validate a complete v17 episode before publishing or consuming it.

    Image dimensions and point-cloud configuration are self-describing. A caller
    may additionally supply its expected preprocessing configuration.
    """
    artifact = Path(path)
    with h5py.File(artifact, "r") as source:
        attrs = source.attrs
        length = _strict_integer_attr(attrs, "episode_steps")
        source_frames = _strict_integer_attr(attrs, "source_frames")
        if length <= 0 or source_frames != length:
            raise ValueError(f"{artifact.name}: source_frames must equal episode_steps")
        if _strict_integer_attr(attrs, "schema_version") != PROCESSED_SCHEMA_VERSION:
            raise ValueError(f"{artifact.name}: invalid schema_version")
        if _strict_integer_attr(attrs, "source_schema_version") <= 0:
            raise ValueError(f"{artifact.name}: invalid source_schema_version")
        for name in ("source_path", "source_episode"):
            if not isinstance(attrs.get(name), str) or not attrs[name]:
                raise ValueError(f"{artifact.name}: missing {name}")
        profile = OutputProfile(str(attrs.get("profile", "")))
        dt = float(attrs.get("dt", np.nan))
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"{artifact.name}: invalid dt")
        _validate_processed_task_name_attr(attrs, label=artifact.name)
        expected_attrs = {
            "schema_name": PROCESSED_SCHEMA_NAME,
            "domain": "real",
            "obs_alignment": "obs[t]_before_action[t]",
            "observation_alignment": "control_step_latest_causal",
            "state_alignment": "control_step",
            "action_semantics": "teleop_published_joint_target",
            "contact_force_source": _CONTACT_FORCE_SOURCE,
            "contact_force_representation": _CONTACT_FORCE_REPRESENTATION,
            "contact_force_unit": _CONTACT_FORCE_UNIT,
            "contact_force_frame": _CONTACT_FORCE_FRAME,
            "fingertip_points_frame": _FINGERTIP_POINTS_FRAME,
            "fingertip_points_unit": _FINGERTIP_POINTS_UNIT,
            "action_ee_frame": _ACTION_EE_FRAME,
            "action_ee_components": "eef_position_m(3)+eef_rot6d(6)+xhand_target_rad(12)",
        }
        validate_fingertip_points_semantics(attrs, label=artifact.name)
        if (
            _strict_bool_attr(attrs, "contact_force_si_verified")
            is not _CONTACT_FORCE_SI_VERIFIED
        ):
            raise ValueError(f"{artifact.name}: invalid contact_force_si_verified")
        specs = {
            name: ((length, *tail), dtype)
            for name, (tail, dtype) in _CORE_DATASET_SPECS.items()
        }
        if profile.needs_rgb:
            rgb, depth = source.get("rgb"), source.get("depth")
            if (
                not isinstance(rgb, h5py.Dataset)
                or rgb.ndim != 4
                or not isinstance(depth, h5py.Dataset)
            ):
                raise ValueError(f"{artifact.name}: RGB-D datasets are incomplete")
            height, width = rgb.shape[1:3]
            if height <= 0 or width <= 0:
                raise ValueError(f"{artifact.name}: invalid image dimensions")
            specs.update(
                {
                    "rgb": ((length, height, width, 3), np.dtype(np.uint8)),
                    "depth": ((length, height, width), np.dtype(np.uint16)),
                    "camera_intrinsic": ((length, 9), np.dtype(np.float32)),
                    "camera_extrinsic": ((length, 4, 4), np.dtype(np.float32)),
                }
            )
            scale = float(attrs.get("depth_scale_m_per_unit", np.nan))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"{artifact.name}: invalid depth scale")
            if _strict_integer_attr(attrs, "depth_invalid_value") != 0:
                raise ValueError(f"{artifact.name}: invalid depth_invalid_value")
            expected_attrs.update(
                {
                    "rgb_transform": "resize_no_crop",
                    "depth_transform": "depth_to_color_aligned_resize_no_crop_nearest",
                    "camera_intrinsic_semantics": "resized_color_intrinsics_for_depth_to_color_aligned_depth",
                    "camera_extrinsic_semantics": "T_xarm_base_from_color;native_color_optical_to_xarm_base",
                }
            )
            depth_k = np.asarray(
                attrs.get("source_camera_depth_intrinsics_native", ()), dtype=np.float64
            )
            if depth_k.shape != (9,) or not np.all(np.isfinite(depth_k)):
                raise ValueError(f"{artifact.name}: invalid native depth intrinsics")
            validate_rigid_transform(
                np.asarray(attrs.get("camera_T_color_from_depth", ())),
                label="camera_T_color_from_depth",
            )
        workspace = None
        if profile.needs_pointcloud:
            persisted = _json_object_attr(
                source, "processing_config_json", label=artifact.name
            )
            if set(persisted) != {"pointcloud", "table_plane_abcd"}:
                raise ValueError(
                    f"{artifact.name}: invalid point-cloud processing config"
                )
            pointcloud = PointCloudConfig(**persisted["pointcloud"])
            table_plane = persisted["table_plane_abcd"]
            if table_plane is not None:
                plane = np.asarray(table_plane, dtype=np.float64)
                if (
                    plane.shape != (4,)
                    or not np.all(np.isfinite(plane))
                    or plane[2] <= 0
                ):
                    raise ValueError(f"{artifact.name}: invalid table plane")
            workspace = pointcloud.workspace
            specs["point_cloud"] = (
                (length, pointcloud.num_points, 6),
                np.dtype(np.float32),
            )
            expected_attrs.update(
                {
                    "point_cloud_frame": "xarm_base",
                    "point_cloud_color_source": POINT_CLOUD_COLOR_SOURCE,
                    "point_cloud_policy_id": POINT_CLOUD_POLICY_ID,
                    "point_cloud_sampling": POINT_CLOUD_SAMPLING,
                    "point_cloud_transform": POINT_CLOUD_TRANSFORM,
                    "point_cloud_table_plane_abcd_json": _json(table_plane),
                }
            )
            if not np.array_equal(
                attrs.get("point_cloud_shape", ()), (pointcloud.num_points, 6)
            ):
                raise ValueError(f"{artifact.name}: invalid point_cloud_shape")
            if config is not None and persisted != {
                "pointcloud": config.pointcloud.to_dict(),
                "table_plane_abcd": (
                    None
                    if config.table_plane_abcd is None
                    else list(config.table_plane_abcd)
                ),
            }:
                raise ValueError(
                    f"{artifact.name}: point-cloud processing config mismatch"
                )
        for name, expected in expected_attrs.items():
            if attrs.get(name) != expected:
                raise ValueError(f"{artifact.name}: invalid {name}")
        if config is not None and (
            profile != config.profile or specs != _expected_specs(length, config)
        ):
            raise ValueError(f"{artifact.name}: processing profile/shape mismatch")
        validate_processed_payload(
            source,
            expected_specs=specs,
            length=length,
            label=artifact.name,
            validate_rgbd=profile.needs_rgb,
            pointcloud_workspace=workspace,
        )
    return {"path": artifact.name, "frames": length, "keys": sorted(specs)}
