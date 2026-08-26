#!/usr/bin/env python3
"""Usage: ``python examples/pointcloud_process_example.py [--save-dir DIR]``.

Self-contained L515 tabletop point-cloud and table-plane diagnostic. It uses
the resolved table plane by default and enters deterministic multi-frame table
calibration only after operator confirmation. A new fit is used immediately;
publishing ``desk_plane.json`` requires separate confirmation and affects both
perception cropping and table-aware collision geometry on the next runtime
resolution. ``--save-dir`` atomically saves the aligned RGB-D source, raw and
processed xArm-base clouds, and complete offline reconstruction metadata.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

from dexmani_real.calibration.table import fit_table_plane, publish_table_plane
from dexmani_real.config.camera_calib import CameraCalib
from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.config.runtime import ResolvedRuntimeConfig, resolve_runtime_config
from dexmani_real.sensor.camera_geometry import RGBDGeometry
from dexmani_real.sensor.pointcloud import (
    POINT_CLOUD_COLOR_SOURCE,
    POINT_CLOUD_POLICY_ID,
    POINT_CLOUD_SAMPLING,
    POINT_CLOUD_TRANSFORM,
    aligned_depth_points_in_base,
    build_point_cloud_with_stats,
    build_raw_point_cloud,
)
from dexmani_real.sensor.realsense import L515DepthConfig, RealSense, RealSenseConfig
from dexmani_real.utils.atomic_io import atomic_json_dump, atomic_publish

if TYPE_CHECKING:
    import open3d as o3d


@dataclass(frozen=True)
class PointCloudDiagnosticConfig:
    """Configuration for the interactive point-cloud diagnostic."""

    rgb_resolution: tuple[int, int] = (640, 480)
    depth_resolution: tuple[int, int] = (640, 480)
    fps: int = 30
    warmup_frames: int = 10
    table_calibration_frames: int = 5

    show_rgbd_panels: bool = True
    show_o3d: bool = True

    vis_depth_min_m: float = 0.3
    vis_depth_max_m: float = 2.5


_SENSOR_OPTIONS: list[tuple[str, rs.option]] = [
    ("visual_preset", rs.option.visual_preset),
    ("laser_power", rs.option.laser_power),
    ("receiver_gain", rs.option.receiver_gain),
    ("confidence_threshold", rs.option.confidence_threshold),
    ("noise_filtering", rs.option.noise_filtering),
    ("digital_gain", rs.option.digital_gain),
    ("depth_offset", rs.option.depth_offset),
    ("min_distance", rs.option.min_distance),
    ("post_processing_sharpening", rs.option.post_processing_sharpening),
    ("pre_processing_sharpening", rs.option.pre_processing_sharpening),
    ("noise_estimation", rs.option.noise_estimation),
]


def _jet_colormap(values: np.ndarray) -> np.ndarray:
    """MATLAB-style jet colormap for normalized values in [0, 1].

    Returns float [..., 3] in [0, 1].
    """
    t = np.clip(values, 0.0, 1.0)
    red = np.where(
        t < 0.375,
        0.0,
        np.where(t < 0.625, 4.0 * t - 1.5, 1.0),
    )
    green = np.where(
        t < 0.125,
        0.0,
        np.where(
            t < 0.375,
            4.0 * t - 0.5,
            np.where(t < 0.625, 1.0, np.where(t < 0.875, 3.5 - 4.0 * t, 0.0)),
        ),
    )
    blue = np.where(
        t < 0.125,
        1.0,
        np.where(t < 0.375, 1.5 - 4.0 * t, 0.0),
    )
    return np.stack([red, green, blue], axis=-1)


def _make_depth_vis(depth_m: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Render metric depth as an 8-bit BGR image for OpenCV."""
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    safe = np.where(valid, depth_m, vmin)
    normalized = (safe - float(vmin)) / (float(vmax) - float(vmin))
    depth_bgr = (_jet_colormap(normalized)[..., ::-1] * 255.0).astype(np.uint8)
    depth_bgr[~valid] = 0
    return depth_bgr


def _show_rgbd_panels(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    cfg: PointCloudDiagnosticConfig,
) -> None:
    """Show same-resolution depth-to-color aligned RGB and depth."""
    if not cfg.show_rgbd_panels:
        return
    try:
        import cv2
    except ImportError:
        print("  RGBD visualization skipped: OpenCV unavailable.")
        return

    # cv2.imshow expects BGR; convert from RGB.
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    panels: list[tuple[str, np.ndarray]] = [
        ("RGB", bgr),
        ("Depth", _make_depth_vis(depth_m, cfg.vis_depth_min_m, cfg.vis_depth_max_m)),
    ]
    labeled = []
    for title, image in panels:
        canvas = image.copy()
        cv2.putText(
            canvas,
            title,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        labeled.append(canvas)

    try:
        print("\nShowing RGBD window -- press any key to continue...")
        cv2.imshow("L515 RGBD diagnostic", np.hstack(labeled))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error as error:
        print(f"  RGBD window skipped (no display?): {error}")


def _print_depth_stats(depth_m: np.ndarray, vmin: float, vmax: float) -> None:
    """Print 2-D depth-gate statistics for the diagnostic view."""
    print("\n" + "=" * 60)
    print("2-D Depth Gate (diagnostic view)")
    print("=" * 60)
    valid = np.isfinite(depth_m) & (depth_m >= float(vmin)) & (depth_m <= float(vmax))
    ratio = float(np.mean(valid))
    print(
        f"  Range [{vmin:.2f}, {vmax:.2f}] m:  {int(valid.sum()):6d} / "
        f"{depth_m.size} pixels  ({ratio * 100:.1f}%)"
    )
    # The production builder applies its own local depth-support and flying-
    # pixel decisions; this panel intentionally reports only the depth gate.


def _tprint(label: str, key: str, timings: dict[str, float]) -> None:
    """Print a single timing line with ASCII bar."""
    ms = timings.get(key, 0.0)
    pipeline_ms = max(timings.get("pipeline_total", 1.0), 0.001)
    if math.isnan(ms):
        print(f"  {label:<30s}     --   (unavailable)")
    elif ms < 0.01:
        print(f"  {label:<30s}     --   (disabled)")
    else:
        pct = ms / pipeline_ms * 100
        bar = "█" * max(1, int(pct / 2.5))
        print(f"  {label:<30s} {ms:6.1f} ms  {bar}")


def _stage_label(key: str) -> str:
    """Human-readable label for a timing stage key."""
    return {
        "capture": "Frame capture",
        "extrinsics": "Extrinsics load",
        "desk_calib": "Table plane calibration",
        "pipeline_p50": "Fresh-frame pipeline p50",
        "pipeline_p95": "Fresh-frame pipeline p95",
        "pipeline_max": "Fresh-frame pipeline max",
        "end_to_end_p50": "Capture-to-cloud p50",
        "end_to_end_p95": "Capture-to-cloud p95",
        "end_to_end_max": "Capture-to-cloud max",
    }.get(key, key)


_BUILD_TIMING_FIELDS = (
    "depth_filter_ms",
    "table_crop_ms",
    "deprojection_ms",
    "base_workspace_ms",
    "voxelization_ms",
    "spatial_outlier_filter_ms",
    "color_sampling_ms",
)

_SNAPSHOT_SCHEMA_NAME = "dexmani-real-pointcloud-diagnostic-snapshot"
_SNAPSHOT_SCHEMA_VERSION = 2


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="L515 tabletop point-cloud processing and calibration diagnostic"
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help=(
            "Atomically save the post-calibration aligned RGB-D frame, raw and "
            "processed clouds, and reconstruction metadata below this directory"
        ),
    )
    return parser.parse_args(argv)


def _validate_snapshot_cloud(cloud: np.ndarray, *, label: str) -> np.ndarray:
    """Return one finite contiguous float32 ``[N,6]`` xyzrgb cloud."""
    value = np.asarray(cloud)
    if value.ndim != 2 or value.shape[1] != 6 or value.dtype != np.float32:
        raise ValueError(
            f"{label} must be float32 [N,6], got shape={value.shape} dtype={value.dtype}"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{label} contains NaN/Inf")
    if value.size and (np.any(value[:, 3:] < 0.0) or np.any(value[:, 3:] > 1.0)):
        raise ValueError(f"{label} RGB values must be in [0,1]")
    return np.ascontiguousarray(value)


def _save_diagnostic_snapshot(
    *,
    output_root: Path,
    rgb: np.ndarray,
    depth_raw: np.ndarray,
    raw_point_cloud: np.ndarray,
    processed_point_cloud: np.ndarray,
    depth_scale_m: float,
    geometry: RGBDGeometry,
    T_xarm_base_from_color: np.ndarray,
    table_plane_abcd: tuple[float, float, float, float] | None,
    table_plane_source: str,
    pointcloud_config: PointCloudConfig,
    runtime_config_sha256: str,
    camera_info: dict,
) -> Path:
    """Durably publish one self-contained offline point-cloud tuning snapshot."""
    color = np.asarray(rgb)
    depth = np.asarray(depth_raw)
    if color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3:
        raise ValueError("rgb must be uint8 [H,W,3]")
    if depth.dtype != np.uint16 or depth.shape != color.shape[:2]:
        raise ValueError("depth_raw must be uint16 [H,W] matching rgb")
    expected_shape = (geometry.color.height, geometry.color.width)
    if (
        color.shape[:2] != expected_shape
        or (
            geometry.depth.height,
            geometry.depth.width,
        )
        != expected_shape
    ):
        raise ValueError("aligned RGB-D arrays must match color-grid geometry")
    if not np.isfinite(depth_scale_m) or depth_scale_m <= 0.0:
        raise ValueError("depth_scale_m must be finite and positive")
    raw_cloud = _validate_snapshot_cloud(raw_point_cloud, label="raw_point_cloud")
    processed_cloud = _validate_snapshot_cloud(
        processed_point_cloud,
        label="processed_point_cloud",
    )
    if processed_cloud.shape[0] not in {0, pointcloud_config.num_points}:
        raise ValueError("processed_point_cloud must be empty or match configured N")
    transform = np.asarray(T_xarm_base_from_color, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("T_xarm_base_from_color must be a finite 4x4 transform")
    if table_plane_abcd is None:
        table_plane_json = None
    else:
        plane = np.asarray(table_plane_abcd, dtype=np.float64)
        if plane.shape != (4,) or not np.all(np.isfinite(plane)):
            raise ValueError("table_plane_abcd must contain four finite values")
        table_plane_json = plane.tolist()
    if table_plane_source not in {
        "calibrated_this_run",
        "resolved_runtime",
        "disabled",
    }:
        raise ValueError(f"invalid table_plane_source {table_plane_source!r}")

    root = output_root.expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(f"snapshot output root is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone()
    artifact_name = timestamp.strftime("pointcloud_%Y%m%d_%H%M%S_%f")
    target = root / artifact_name
    temporary = Path(tempfile.mkdtemp(prefix=".pointcloud-tmp-", dir=root))

    arrays = {
        "rgb.npy": np.ascontiguousarray(color),
        "depth_aligned_to_color_raw.npy": np.ascontiguousarray(depth),
        "raw_point_cloud.npy": raw_cloud,
        "processed_point_cloud.npy": processed_cloud,
        "T_xarm_base_from_color.npy": np.ascontiguousarray(transform),
    }
    metadata = {
        "schema_name": _SNAPSHOT_SCHEMA_NAME,
        "schema_version": _SNAPSHOT_SCHEMA_VERSION,
        "created_at": timestamp.isoformat(),
        "payload_mode": "depth_to_color_aligned_rgbd",
        "depth_payload_frame": "color",
        "depth_scale_m_per_unit": float(depth_scale_m),
        "aligned_geometry": geometry.to_dict(),
        "table_plane_abcd": table_plane_json,
        "table_plane_source": table_plane_source,
        "runtime_config_sha256": runtime_config_sha256,
        "pointcloud_config": pointcloud_config.to_dict(),
        "pointcloud_config_sha256": pointcloud_config.sha256,
        "point_cloud_policy_id": POINT_CLOUD_POLICY_ID,
        "point_cloud_color_source": POINT_CLOUD_COLOR_SOURCE,
        "point_cloud_sampling": POINT_CLOUD_SAMPLING,
        "point_cloud_transform": POINT_CLOUD_TRANSFORM,
        "point_cloud_frame": "xarm_base",
        "point_cloud_columns": ["x_m", "y_m", "z_m", "r", "g", "b"],
        "raw_point_cloud_semantics": "all_finite_nonzero_aligned_depth_pixels",
        "processed_point_cloud_empty": processed_cloud.shape[0] == 0,
        "camera": {
            key: str(camera_info.get(key, "")) for key in ("name", "serial", "firmware")
        },
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        },
    }

    try:
        for name, value in arrays.items():
            np.save(temporary / name, value, allow_pickle=False)
        atomic_json_dump(metadata, temporary / "metadata.json", ensure_ascii=False)
        return atomic_publish(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _print_build_stage_timings(prefix: str, values: dict[str, float]) -> None:
    """Print compact stage timings from one build or benchmark percentile."""
    print(f"  {prefix}:")
    for field in _BUILD_TIMING_FIELDS:
        print(f"    {field.removesuffix('_ms'):<24s} {values[field]:5.1f} ms")


def _connect_camera(cfg: PointCloudDiagnosticConfig) -> RealSense:
    """Connect camera and return it.

    Connection time is deliberately not timed: it is a one-time hardware
    init (pipeline start + warmup frames), not a per-frame pipeline stage, and
    its multi-second duration would dominate the per-stage timing chart.
    """
    camera = RealSense(
        RealSenseConfig(
            depth_resolution=cfg.depth_resolution,
            color_resolution=cfg.rgb_resolution,
            fps=cfg.fps,
            enable_color=True,
            enable_global_time=True,
            warmup_frames=cfg.warmup_frames,
            l515_depth_config=L515DepthConfig(),
        )
    )
    print("Connecting to RealSense...")
    if not camera.connect():
        raise RuntimeError("Failed to connect to RealSense.")
    return camera


def _print_device_info(camera: RealSense) -> dict:
    """Print device info and read back sensor options.  Returns info dict with 'serial'."""
    info = camera.get_device_info()
    print("=" * 60)
    print(f"Device:   {info.get('name', '')}")
    print(f"Serial:   {info.get('serial', '')}")
    print(f"Firmware: {info.get('firmware', '')}")
    print("=" * 60)

    if camera.profile is None:
        raise RuntimeError("Pipeline profile unavailable.")

    device = camera.profile.get_device()
    depth_sensor = device.first_depth_sensor()

    preset = depth_sensor.get_option(rs.option.visual_preset)
    try:
        preset_name = depth_sensor.get_option_value_description(
            rs.option.visual_preset, preset
        )
    except RuntimeError:
        preset_name = "unknown"
    print(f"  Runtime preset: {preset} {preset_name}")
    print(
        f"  Depth scale:    {camera.get_depth_scale():.6f} (raw_uint16 x scale = meters)"
    )

    print("\n  --- Sensor read-back ---")
    for name, opt in _SENSOR_OPTIONS:
        try:
            print(f"  {name}: {depth_sensor.get_option(opt)}")
        except RuntimeError:
            print(f"  {name}: <not readable>")
    return info


def _capture_frame(
    camera: RealSense,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Capture one aligned frame and return RGB, raw/metric depth, and elapsed ms."""
    print("\nCapturing depth-to-color aligned RGBD frame...")
    t0 = time.perf_counter()
    frame = camera.read()
    capture_ms = (time.perf_counter() - t0) * 1000.0

    if frame.rgb is None:
        raise RuntimeError("RGB frame unavailable.")

    rgb = np.ascontiguousarray(frame.rgb)
    if frame.depth_aligned_to_color_raw is None or frame.depth_aligned_to_color is None:
        raise RuntimeError("depth_to_color alignment is unavailable")
    depth_raw = np.ascontiguousarray(frame.depth_aligned_to_color_raw)
    depth_m = np.ascontiguousarray(frame.depth_aligned_to_color, dtype=np.float32)
    if rgb.shape[:2] != depth_raw.shape or depth_raw.shape != depth_m.shape:
        raise RuntimeError("aligned RGB and depth frame dimensions do not match")

    print(f"  RGB:     shape={rgb.shape}, dtype={rgb.dtype}")
    print(
        f"  Depth:   shape={depth_m.shape}, valid={int((depth_m > 0).sum())}/{depth_m.size}"
    )
    print(f"  Depth scale: {float(frame.depth_scale):.6f} m")

    return rgb, depth_raw, depth_m, capture_ms


def _load_extrinsics(camera_info: dict) -> tuple[np.ndarray, float]:
    """Load T_xarm_base_from_color for aligned depth-to-color samples."""
    print("\nLoading extrinsics from cameras.json...")
    t0 = time.perf_counter()
    calib = CameraCalib()
    cam_name = calib.resolve_name_by_serial(str(camera_info.get("serial", "")))
    base_from_color = np.asarray(calib.get_extrinsics(cam_name), dtype=np.float64)

    if base_from_color.shape != (4, 4):
        raise RuntimeError(f"Invalid extrinsic shape: {base_from_color.shape}")
    if not np.allclose(base_from_color[3], [0, 0, 0, 1], atol=1e-6):
        raise RuntimeError("Invalid homogeneous transform last row.")

    elapsed = (time.perf_counter() - t0) * 1000.0
    pos = base_from_color[:3, 3]
    quat_xyzw = R.from_matrix(base_from_color[:3, :3]).as_quat()
    print(f"  Camera '{cam_name}' -> color/aligned-depth frame in xArm-base frame")
    print(f"  pos:         {np.round(pos, 4)} m")
    print(f"  quat (xyzw): {np.round(quat_xyzw, 4)}")
    return base_from_color, elapsed


def _resolve_table_plane_path(runtime: ResolvedRuntimeConfig) -> Path:
    """Resolve the shared table calibration path from runtime configuration."""
    table = runtime.environment.table
    if not table.enabled or table.plane_path is None:
        raise RuntimeError("runtime table calibration is disabled or has no plane_path")
    path = Path(table.plane_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path


def _calibrate_table(
    *,
    camera: RealSense,
    geometry: RGBDGeometry,
    T_xarm_base_from_color: np.ndarray,
    config: PointCloudConfig,
    plane_path: Path,
    frame_count: int,
) -> tuple[tuple[float, float, float, float], float]:
    """Fit a deterministic multi-frame table plane and optionally publish it."""
    print("\n" + "=" * 60)
    print("Table Plane Calibration (multi-frame RANSAC)")
    print("=" * 60)
    print("  Clear movable objects from the visible table before continuing.")
    input("  Press Enter to capture calibration frames...")

    started = time.perf_counter()
    depth_frames: list[np.ndarray] = []
    for _ in range(max(1, frame_count)):
        frame = camera.read()
        if frame.depth_aligned_to_color_raw is None:
            raise RuntimeError("depth_to_color alignment is unavailable")
        depth_frames.append(np.ascontiguousarray(frame.depth_aligned_to_color_raw))

    point_batches: list[np.ndarray] = []
    lower = np.asarray(config.workspace[:3], dtype=np.float32)
    upper = np.asarray(config.workspace[3:], dtype=np.float32)
    for depth_raw in depth_frames:
        points = aligned_depth_points_in_base(
            depth_raw=depth_raw,
            depth_scale_m=camera.get_depth_scale(),
            aligned_depth_intrinsics=geometry.depth,
            T_xarm_base_from_color=T_xarm_base_from_color,
            config=config,
        )
        points = points[np.all((points >= lower) & (points <= upper), axis=1)]
        if points.shape[0] > 30_000:
            step = max(1, points.shape[0] // 30_000)
            points = points[::step][:30_000]
        point_batches.append(points)
    calibration_points = np.concatenate(point_batches, axis=0)
    fit = fit_table_plane(calibration_points)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    a, b, c, d = fit.plane_abcd
    print(f"  Plane: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")
    print(
        f"  Support: {fit.inlier_points}/{fit.evaluated_points} "
        f"({fit.inlier_ratio * 100:.1f}%), RMS={fit.rms_residual_m * 1000:.2f} mm, "
        f"tilt={fit.tilt_deg:.2f} deg"
    )

    print(
        "  WARNING: publishing updates the shared plane used by point-cloud "
        "cropping and table collision geometry."
    )
    confirmed = input(f"Publish to {plane_path}? [y/N] ").strip().lower() in {
        "y",
        "yes",
    }
    if confirmed:
        backup = publish_table_plane(plane_path, fit, confirmed=True)
        if backup is not None:
            print(f"  Backup: {backup}")
        resolved_plane = resolve_runtime_config().environment.table.plane_abcd
        if not np.allclose(resolved_plane, fit.plane_abcd, rtol=0.0, atol=1e-12):
            raise RuntimeError("published plane failed runtime-config round-trip")
        print("  Published and runtime-config round-trip verified.")
    else:
        print("  Calibration file unchanged; using the new fit for this run only.")
    return fit.plane_abcd, elapsed_ms


def _build_cloud(
    *,
    depth_raw: np.ndarray,
    rgb: np.ndarray,
    depth_scale_m: float,
    geometry: RGBDGeometry,
    T_xarm_base_from_color: np.ndarray,
    config: PointCloudConfig,
    table_plane_abcd: tuple[float, float, float, float] | None,
) -> np.ndarray:
    """Build and report one production point cloud."""
    print("\n" + "=" * 60)
    print("Point-Cloud Pipeline (depth_to_color -> xArm-base)")
    print("=" * 60)
    table_state = "ENABLED" if table_plane_abcd is not None else "DISABLED"
    print(f"  num_points={config.num_points}  workspace={config.workspace}")
    print(f"  depth range [{config.depth_min_m}, {config.depth_max_m}] m")
    print(
        "  3x3 support neighbors "
        f"flat={config.depth_support_min_neighbors} "
        f"edge={config.edge_support_min_neighbors}"
    )
    print(
        "  radius-component minimum "
        f"{config.outlier_min_component_points} points; radius neighbors "
        f"{config.outlier_min_neighbors}"
    )
    print(
        "  table height hysteresis "
        f"core={config.table_core_height_m * 1000.0:.1f} mm "
        f"object_seed={config.table_object_seed_height_m * 1000.0:.1f} mm "
        f"x{config.table_object_seed_min_pixels} pixels"
    )
    print(f"  voxel_size_m={config.voxel_size_m}  table crop: {table_state}")

    # Warm up once, then time the public operation.
    _ = build_point_cloud_with_stats(
        depth_raw=depth_raw,
        color=rgb,
        depth_scale_m=depth_scale_m,
        geometry=geometry,
        T_xarm_base_from_color=T_xarm_base_from_color,
        table_plane_abcd=table_plane_abcd,
        config=config,
    )

    t0 = time.perf_counter()
    result, stats = build_point_cloud_with_stats(
        depth_raw=depth_raw,
        color=rgb,
        depth_scale_m=depth_scale_m,
        geometry=geometry,
        T_xarm_base_from_color=T_xarm_base_from_color,
        table_plane_abcd=table_plane_abcd,
        config=config,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    build_stage_timings = {
        field: float(getattr(stats.timings, field)) for field in _BUILD_TIMING_FIELDS
    }

    print(
        "  Stage counts: "
        f"valid={stats.depth_valid_points} -> supported={stats.depth_trusted_points} "
        f"-> table_reject={stats.table_rejected_points} "
        f"-> workspace_reject={stats.workspace_rejected_points} "
        f"-> crop={stats.cropped_points} -> voxel={stats.voxel_points} "
        f"-> density={stats.radius_density_points} "
        f"-> component={stats.spatial_inlier_points} "
        f"-> candidate={stats.candidate_points}"
    )
    _print_build_stage_timings("Single-frame build stages", build_stage_timings)

    if result is not None:
        print(f"\n  Output: {result.shape[0]} points  ({elapsed_ms:.1f} ms)")
    else:
        stage = stats.failure_stage or "unknown"
        print(f"\n  Output: none (pipeline stopped at {stage})")
        result = np.zeros((0, 6), dtype=np.float32)

    return result


def _benchmark_production_pipeline(
    *,
    camera: RealSense,
    geometry: RGBDGeometry,
    T_xarm_base_from_color: np.ndarray,
    config: PointCloudConfig,
    table_plane_abcd: tuple[float, float, float, float] | None,
    frame_count: int = 20,
) -> tuple[np.ndarray, dict[str, float]]:
    """Measure processing and capture-to-cloud p50/p95 on fresh RGB-D frames."""
    elapsed_ms: list[float] = []
    end_to_end_ms: list[float] = []
    stage_samples: dict[str, list[float]] = {
        field: [] for field in _BUILD_TIMING_FIELDS
    }
    latest = np.zeros((0, 6), dtype=np.float32)
    for _ in range(frame_count):
        frame_started = time.perf_counter()
        frame = camera.read()
        if frame.rgb is None or frame.depth_aligned_to_color_raw is None:
            raise RuntimeError("aligned RGB-D frame is unavailable")
        started = time.perf_counter()
        cloud, stats = build_point_cloud_with_stats(
            depth_raw=frame.depth_aligned_to_color_raw,
            color=frame.rgb,
            depth_scale_m=camera.get_depth_scale(),
            geometry=geometry,
            T_xarm_base_from_color=T_xarm_base_from_color,
            table_plane_abcd=table_plane_abcd,
            config=config,
        )
        elapsed_ms.append((time.perf_counter() - started) * 1000.0)
        end_to_end_ms.append((time.perf_counter() - frame_started) * 1000.0)
        for field in _BUILD_TIMING_FIELDS:
            stage_samples[field].append(float(getattr(stats.timings, field)))
        if cloud is not None:
            latest = cloud
    values = np.asarray(elapsed_ms, dtype=np.float64)
    end_to_end_values = np.asarray(end_to_end_ms, dtype=np.float64)
    timings = {
        "pipeline_total": float(np.mean(values)),
        "pipeline_p50": float(np.percentile(values, 50)),
        "pipeline_p95": float(np.percentile(values, 95)),
        "pipeline_max": float(np.max(values)),
        "end_to_end_p50": float(np.percentile(end_to_end_values, 50)),
        "end_to_end_p95": float(np.percentile(end_to_end_values, 95)),
        "end_to_end_max": float(np.max(end_to_end_values)),
    }
    stage_p95 = {
        field: float(np.percentile(stage_samples[field], 95))
        for field in _BUILD_TIMING_FIELDS
    }
    timings.update({f"{field}_p95": value for field, value in stage_p95.items()})
    print(
        "  Fresh-frame benchmark: "
        f"n={frame_count} p50={timings['pipeline_p50']:.1f} ms "
        f"p95={timings['pipeline_p95']:.1f} ms "
        f"max={timings['pipeline_max']:.1f} ms "
        "(target p95 < 40.0 ms)"
    )
    print(
        "  Capture-to-cloud benchmark: "
        f"p50={timings['end_to_end_p50']:.1f} ms "
        f"p95={timings['end_to_end_p95']:.1f} ms "
        f"max={timings['end_to_end_max']:.1f} ms"
    )
    _print_build_stage_timings("Build-stage p95", stage_p95)
    return latest, timings


def _print_timing_summary(timings: dict[str, float]) -> None:
    """Print per-stage timing with ASCII bar charts."""
    print("\n" + "=" * 60)
    print("Per-Stage Timing Summary")
    print("=" * 60)

    sections = [
        ("Setup", ["capture", "extrinsics", "desk_calib"]),
        (
            "Point-Cloud Pipeline (per-frame steady-state)",
            [
                "pipeline_p50",
                "pipeline_p95",
                "pipeline_max",
                "end_to_end_p50",
                "end_to_end_p95",
                "end_to_end_max",
            ],
        ),
    ]

    for title, keys in sections:
        print(f"\n  -- {title} --")
        for key in keys:
            _tprint(f"  {_stage_label(key)}", key, timings)

        if title.startswith("Point-Cloud"):
            pipeline_ms = timings.get("pipeline_total", 0.0)
            print(f"  {'  Point-cloud mean':<30s} {pipeline_ms:6.1f} ms")


def _build_workspace_box(workspace: tuple[float, ...]) -> "o3d.geometry.LineSet":
    """Build a green wireframe box for the workspace crop volume."""
    import open3d as o3d

    ws = workspace
    corners = np.array(
        [
            [ws[0], ws[1], ws[2]],
            [ws[3], ws[1], ws[2]],
            [ws[3], ws[4], ws[2]],
            [ws[0], ws[4], ws[2]],
            [ws[0], ws[1], ws[5]],
            [ws[3], ws[1], ws[5]],
            [ws[3], ws[4], ws[5]],
            [ws[0], ws[4], ws[5]],
        ]
    )
    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    box = o3d.geometry.LineSet()
    box.points = o3d.utility.Vector3dVector(corners)
    box.lines = o3d.utility.Vector2iVector(edges)
    box.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 0.0] for _ in edges])
    return box


def _visualize_result(
    result: np.ndarray,
    T_xarm_base_from_color: np.ndarray,
    config: PointCloudConfig,
) -> None:
    """Visualize final xArm-base point cloud with coordinate frames and crop box."""
    import open3d as o3d

    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    depth_camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    depth_camera_frame.transform(T_xarm_base_from_color)

    crop_box = _build_workspace_box(config.workspace)

    pcd = o3d.geometry.PointCloud()
    if result.shape[0]:
        pcd.points = o3d.utility.Vector3dVector(result[:, :3].astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(result[:, 3:6].astype(np.float64))

    print("\nLaunching open3d visualizer (close window to continue)...")
    print("  xArm-base point cloud + workspace crop box (green) + coordinate frames")
    o3d.visualization.draw_geometries(
        [pcd, crop_box, base_frame, depth_camera_frame],
        window_name="Point Cloud",
        point_show_normal=False,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the hardware diagnostic and return a process exit status."""
    args = _parse_args(argv)
    save_dir = None if args.save_dir is None else args.save_dir.expanduser().resolve()
    if save_dir is not None and save_dir.exists() and not save_dir.is_dir():
        raise NotADirectoryError(f"snapshot output root is not a directory: {save_dir}")
    cfg = PointCloudDiagnosticConfig()
    all_timings: dict[str, float] = {}

    # Resolve and validate file-backed policy before connecting to hardware.
    runtime = resolve_runtime_config()
    pcd_config = runtime.pointcloud

    camera = _connect_camera(cfg)

    try:
        camera_info = _print_device_info(camera)

        rgb, _, depth_m, capture_ms = _capture_frame(camera)
        all_timings["capture"] = capture_ms
        _print_depth_stats(depth_m, cfg.vis_depth_min_m, cfg.vis_depth_max_m)
        _show_rgbd_panels(rgb, depth_m, cfg)

        geometry = camera.get_geometry().aligned_depth_to_color()
        T_xarm_base_from_color, extrinsics_ms = _load_extrinsics(camera_info)
        all_timings["extrinsics"] = extrinsics_ms

        calibrate_table = input("\nRun table calibration? [y/N] ").strip().lower() in {
            "y",
            "yes",
        }
        if calibrate_table:
            # The diagnostic uses the exact runtime point-cloud policy. The
            # new fit is used immediately whether or not it is published.
            table_plane_abcd, calibration_ms = _calibrate_table(
                camera=camera,
                geometry=geometry,
                T_xarm_base_from_color=T_xarm_base_from_color,
                config=pcd_config,
                plane_path=_resolve_table_plane_path(runtime),
                frame_count=cfg.table_calibration_frames,
            )
            all_timings["desk_calib"] = calibration_ms
            table_plane_source = "calibrated_this_run"
        elif runtime.environment.table.enabled:
            table_plane_abcd = runtime.environment.table.plane_abcd
            table_plane_source = "resolved_runtime"
            all_timings["desk_calib"] = math.nan
            print(
                "  Table calibration skipped; using resolved desk_plane.json: "
                f"{np.round(table_plane_abcd, 6)}"
            )
        else:
            table_plane_abcd = None
            table_plane_source = "disabled"
            all_timings["desk_calib"] = math.nan
            print(
                "  Table calibration skipped; table crop is disabled by runtime config."
            )

        # Use a post-calibration frame for the reported production result;
        # never apply a newly fitted plane to a stale pre-calibration frame.
        rgb, depth_raw, _, _ = _capture_frame(camera)

        result = _build_cloud(
            depth_raw=depth_raw,
            rgb=rgb,
            depth_scale_m=camera.get_depth_scale(),
            geometry=geometry,
            T_xarm_base_from_color=T_xarm_base_from_color,
            config=pcd_config,
            table_plane_abcd=table_plane_abcd,
        )
        if save_dir is not None:
            raw_cloud = build_raw_point_cloud(
                depth_raw=depth_raw,
                color=rgb,
                depth_scale_m=camera.get_depth_scale(),
                geometry=geometry,
                T_xarm_base_from_color=T_xarm_base_from_color,
            )
            if raw_cloud is None:
                raw_cloud = np.zeros((0, 6), dtype=np.float32)
            snapshot_path = _save_diagnostic_snapshot(
                output_root=save_dir,
                rgb=rgb,
                depth_raw=depth_raw,
                raw_point_cloud=raw_cloud,
                processed_point_cloud=result,
                depth_scale_m=camera.get_depth_scale(),
                geometry=geometry,
                T_xarm_base_from_color=T_xarm_base_from_color,
                table_plane_abcd=table_plane_abcd,
                table_plane_source=table_plane_source,
                pointcloud_config=pcd_config,
                runtime_config_sha256=runtime.sha256,
                camera_info=camera_info,
            )
            print(f"\nSaved point-cloud diagnostic snapshot: {snapshot_path}")
        benchmark_result, t = _benchmark_production_pipeline(
            camera=camera,
            geometry=geometry,
            T_xarm_base_from_color=T_xarm_base_from_color,
            config=pcd_config,
            table_plane_abcd=table_plane_abcd,
        )
        all_timings.update(t)
        if benchmark_result.shape[0]:
            result = benchmark_result

        _print_timing_summary(all_timings)

        if cfg.show_o3d:
            _visualize_result(result, T_xarm_base_from_color, pcd_config)
        else:
            print("\n  open3d visualizer skipped (show_o3d=False)")

        print("\nDone.")

    finally:
        camera.disconnect()
        print("RealSense pipeline stopped cleanly.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
