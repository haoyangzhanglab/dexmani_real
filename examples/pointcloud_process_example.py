#!/usr/bin/env python3
"""Usage: ``python examples/pointcloud_process_example.py``.

Self-contained L515 tabletop point-cloud and table-plane diagnostic. It fits a
multi-frame deterministic table plane every run, uses that fit immediately,
and publishes ``desk_plane.json`` only after explicit operator confirmation.
Publishing creates a timestamped backup and affects both perception cropping
and table-aware collision geometry on the next runtime resolution.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

from dexmani_real.config.camera_calib import CameraCalib
from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.sensor.camera_geometry import RGBDGeometry
from dexmani_real.sensor.pointcloud import (
    build_point_cloud_with_stats,
    depth_points_in_base,
)
from dexmani_real.sensor.realsense import L515DepthConfig, RealSense, RealSenseConfig
from dexmani_real.sensor.table_calibration import fit_table_plane, publish_table_plane

if TYPE_CHECKING:
    import open3d as o3d


@dataclass(frozen=True)
class PointCloudDiagnosticConfig:
    """Configuration for the interactive point-cloud diagnostic."""

    rgb_resolution: tuple[int, int] = (640, 480)
    depth_resolution: tuple[int, int] = (1024, 768)
    fps: int = 30
    warmup_frames: int = 30
    target_points: int = 2048
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
    """Show native RGB and depth side by side."""
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
    rgb_h, rgb_w = bgr.shape[:2]
    for title, image in panels:
        canvas = image.copy()
        if canvas.shape[:2] != (rgb_h, rgb_w):
            # Depth is native 1024x768 while RGB is 640x480 (both 4:3); resize
            # the depth panel to the RGB size so np.hstack lines up.
            canvas = cv2.resize(canvas, (rgb_w, rgb_h), interpolation=cv2.INTER_LINEAR)
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
        "p_pipeline": "Complete point-cloud pipeline",
    }.get(key, key)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Capture one native RGB-D frame.  Returns (rgb, depth_raw, depth_m, timings)."""
    print("\nCapturing native RGBD frame...")
    t0 = time.perf_counter()
    frame = camera.read()
    capture_ms = (time.perf_counter() - t0) * 1000.0

    if frame.rgb is None:
        raise RuntimeError("RGB frame unavailable.")

    rgb = np.ascontiguousarray(frame.rgb)
    depth_raw = np.ascontiguousarray(frame.depth_raw)
    depth_m = np.ascontiguousarray(frame.depth, dtype=np.float32)

    print(f"  RGB:     shape={rgb.shape}, dtype={rgb.dtype}")
    print(
        f"  Depth:   shape={depth_m.shape}, valid={int((depth_m > 0).sum())}/{depth_m.size}"
    )
    print(f"  Depth scale: {float(frame.depth_scale):.6f} m")

    return rgb, depth_raw, depth_m, {"capture": capture_ms}


def _load_extrinsics(
    camera_info: dict, geometry: RGBDGeometry
) -> tuple[np.ndarray, float]:
    """Load base_from_depth = base_from_color @ T_color_from_depth from cameras.json."""
    print("\nLoading extrinsics from cameras.json...")
    t0 = time.perf_counter()
    calib = CameraCalib()
    cam_name = calib.resolve_name_by_serial(str(camera_info.get("serial", "")))
    base_from_color = np.asarray(calib.get_extrinsics(cam_name), dtype=np.float64)

    if base_from_color.shape != (4, 4):
        raise RuntimeError(f"Invalid extrinsic shape: {base_from_color.shape}")
    if not np.allclose(base_from_color[3], [0, 0, 0, 1], atol=1e-6):
        raise RuntimeError("Invalid homogeneous transform last row.")

    T_xarm_base_from_depth = base_from_color @ geometry.T_color_from_depth

    elapsed = (time.perf_counter() - t0) * 1000.0
    pos = T_xarm_base_from_depth[:3, 3]
    quat_xyzw = R.from_matrix(T_xarm_base_from_depth[:3, :3]).as_quat()
    print(f"  Camera '{cam_name}' -> depth stream in xArm-base frame")
    print(f"  pos:         {np.round(pos, 4)} m")
    print(f"  quat (xyzw): {np.round(quat_xyzw, 4)}")
    return T_xarm_base_from_depth, elapsed


def _production_config(num_points: int) -> PointCloudConfig:
    """Return the canonical runtime policy with only output size overridden."""
    runtime = resolve_runtime_config()
    return replace(runtime.pointcloud, num_points=num_points)


def _table_plane_path() -> Path:
    """Resolve the shared table calibration path from runtime configuration."""
    runtime = resolve_runtime_config()
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
    T_xarm_base_from_depth: np.ndarray,
    config: PointCloudConfig,
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
        depth_frames.append(np.ascontiguousarray(camera.read().depth_raw))

    point_batches: list[np.ndarray] = []
    lower = np.asarray(config.workspace[:3], dtype=np.float32)
    upper = np.asarray(config.workspace[3:], dtype=np.float32)
    for depth_raw in depth_frames:
        points = depth_points_in_base(
            depth_raw=depth_raw,
            depth_scale_m=camera.get_depth_scale(),
            depth_intrinsics=geometry.depth,
            T_xarm_base_from_depth=T_xarm_base_from_depth,
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

    plane_path = _table_plane_path()
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
        runtime_plane = resolve_runtime_config().environment.table.plane_abcd
        if not np.allclose(runtime_plane, fit.plane_abcd, rtol=0.0, atol=1e-12):
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
    T_xarm_base_from_depth: np.ndarray,
    config: PointCloudConfig,
    table_plane_abcd: tuple[float, float, float, float] | None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Run ``build_point_cloud`` once and extract timing."""
    print("\n" + "=" * 60)
    print("Point-Cloud Pipeline (native -> xArm-base)")
    print("=" * 60)
    table_state = "ENABLED" if table_plane_abcd is not None else "DISABLED"
    print(f"  num_points={config.num_points}  workspace={config.workspace}")
    print(f"  depth range [{config.depth_min_m}, {config.depth_max_m}] m")
    print(f"  voxel_size_m={config.voxel_size_m}  table crop: {table_state}")

    # Warm up once, then time the public operation.
    _ = build_point_cloud_with_stats(
        depth_raw=depth_raw,
        color=rgb,
        depth_scale_m=depth_scale_m,
        geometry=geometry,
        T_xarm_base_from_depth=T_xarm_base_from_depth,
        table_plane_abcd=table_plane_abcd,
        config=config,
    )

    t0 = time.perf_counter()
    result, stats = build_point_cloud_with_stats(
        depth_raw=depth_raw,
        color=rgb,
        depth_scale_m=depth_scale_m,
        geometry=geometry,
        T_xarm_base_from_depth=T_xarm_base_from_depth,
        table_plane_abcd=table_plane_abcd,
        config=config,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    timings = {"pipeline_total": elapsed_ms, "p_pipeline": elapsed_ms}

    print(
        "  Stage counts: "
        f"valid={stats.depth_valid_points} -> supported={stats.depth_trusted_points} "
        f"-> crop={stats.cropped_points} -> voxel={stats.voxel_points} "
        f"-> candidate={stats.outlier_candidate_points} "
        f"-> inlier={stats.outlier_inlier_points} -> visible={stats.color_visible_points}"
    )

    if result is not None:
        print(
            f"\n  Output: {result.shape[0]} points  ({timings['pipeline_total']:.1f} ms)"
        )
    else:
        print("\n  Output: none (no in-workspace visible cloud)")
        result = np.zeros((0, 6), dtype=np.float32)

    return result, timings


def _print_timing_summary(timings: dict[str, float]) -> None:
    """Print per-stage timing with ASCII bar charts."""
    print("\n" + "=" * 60)
    print("Per-Stage Timing Summary")
    print("=" * 60)

    sections = [
        ("Setup", ["capture", "extrinsics", "desk_calib"]),
        ("Point-Cloud Pipeline (per-frame steady-state)", ["p_pipeline"]),
    ]

    for title, keys in sections:
        print(f"\n  -- {title} --")
        section_total = 0.0
        for key in keys:
            _tprint(f"  {_stage_label(key)}", key, timings)
            stage_ms = timings.get(key, 0.0)
            if not math.isnan(stage_ms):
                section_total += stage_ms

        if title.startswith("Point-Cloud"):
            pipeline_ms = timings.get("pipeline_total", 0.0)
            print(f"  {'  Point-cloud total':<30s} {pipeline_ms:6.1f} ms")


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
    T_xarm_base_from_depth: np.ndarray,
    config: PointCloudConfig,
) -> None:
    """Visualize final xArm-base point cloud with coordinate frames and crop box."""
    import open3d as o3d

    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    depth_camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    depth_camera_frame.transform(T_xarm_base_from_depth)

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


def main() -> int:
    """Run the hardware diagnostic and return a process exit status."""
    cfg = PointCloudDiagnosticConfig()
    all_timings: dict[str, float] = {}

    camera = _connect_camera(cfg)

    try:
        camera_info = _print_device_info(camera)

        rgb, depth_raw, depth_m, t = _capture_frame(camera)
        all_timings.update(t)
        _print_depth_stats(depth_m, cfg.vis_depth_min_m, cfg.vis_depth_max_m)
        _show_rgbd_panels(rgb, depth_m, cfg)

        geometry = camera.get_geometry()
        T_xarm_base_from_depth, extrinsics_ms = _load_extrinsics(camera_info, geometry)
        all_timings["extrinsics"] = extrinsics_ms

        # The diagnostic uses the exact runtime point-cloud policy. The new
        # fit is used immediately whether or not the operator publishes it.
        pcd_config = _production_config(cfg.target_points)
        table_plane_abcd, calibration_ms = _calibrate_table(
            camera=camera,
            geometry=geometry,
            T_xarm_base_from_depth=T_xarm_base_from_depth,
            config=pcd_config,
            frame_count=cfg.table_calibration_frames,
        )
        all_timings["desk_calib"] = calibration_ms

        result, t = _build_cloud(
            depth_raw=depth_raw,
            rgb=rgb,
            depth_scale_m=camera.get_depth_scale(),
            geometry=geometry,
            T_xarm_base_from_depth=T_xarm_base_from_depth,
            config=pcd_config,
            table_plane_abcd=table_plane_abcd,
        )
        all_timings.update(t)

        _print_timing_summary(all_timings)

        if cfg.show_o3d:
            _visualize_result(result, T_xarm_base_from_depth, pcd_config)
        else:
            print("\n  open3d visualizer skipped (show_o3d=False)")

        print("\nDone.")

    finally:
        camera.disconnect()
        print("RealSense pipeline stopped cleanly.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
