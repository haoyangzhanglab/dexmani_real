#!/usr/bin/env python3
"""Usage: ``python examples/pointcloud_process_example.py``.

Self-contained L515 tabletop point-cloud and desk-plane diagnostic. Connects
RealSense L515 hardware, opens cv2 and open3d GUI windows, and writes
``desk_plane.json`` only after an explicit ``y`` confirmation (with a
timestamped backup). Steps:

  - Connect RealSense L515, read back depth config parameters
  - Capture one aligned RGB-D frame, visualize RGB + depth + edge filter
  - Calibrate desk plane via RANSAC, persist to desk_plane.json
  - Run the PointCloudProcessor pipeline end-to-end
  - Print per-stage timing with ASCII bar charts
  - Visualize final world-frame point cloud (open3d)
"""

from __future__ import annotations

import math
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

from dexmani_real import PACKAGE_DIR
from dexmani_real.config.camera_calib import CameraCalib
from dexmani_real.sensor.pointcloud_processor import (
    PointCloudProcessor,
    PointCloudProcessorConfig,
)
from dexmani_real.sensor.realsense import L515DepthConfig, RealSense, RealSenseConfig
from dexmani_real.utils.pointcloud_utils import make_depth_vis


@dataclass(frozen=True)
class PointCloudDiagnosticConfig:
    """Configuration for the interactive point-cloud diagnostic."""

    rgb_resolution: tuple[int, int] = (640, 480)
    depth_resolution: tuple[int, int] = (1024, 768)
    fps: int = 30
    warmup_frames: int = 30
    target_points: int = 2048

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


def _show_rgbd_panels(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    edge_vis: np.ndarray | None,
    cfg: PointCloudDiagnosticConfig,
) -> None:
    """Show RGB, depth, and optional edge mask side by side."""
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
        ("Depth", make_depth_vis(depth_m, cfg.vis_depth_min_m, cfg.vis_depth_max_m)),
    ]
    if edge_vis is not None:
        panels.append(("Edge mask (red=core, yellow=dilated)", edge_vis))

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
        "connect": "Camera connect",
        "capture": "Frame capture",
        "extrinsics": "Extrinsics load",
        "desk_calib": "Desk plane calibration",
        "2d_median": "Median filter",
        "2d_depth_gate": "Depth gate",
        "2d_edge_filter": "Edge filter (LoG + dilate)",
        "2d_speckle": "Speckle filter",
        "p_pipeline": "Complete point-cloud pipeline",
    }.get(key, key)


def _connect_camera(
    cfg: PointCloudDiagnosticConfig,
) -> tuple[RealSense, dict[str, float]]:
    """Connect camera, return (camera, {timings})."""
    t0 = time.perf_counter()
    camera = RealSense(
        RealSenseConfig(
            depth_resolution=cfg.depth_resolution,
            color_resolution=cfg.rgb_resolution,
            fps=cfg.fps,
            enable_color=True,
            align_mode="depth_to_color",
            enable_global_time=True,
            warmup_frames=cfg.warmup_frames,
            l515_depth_config=L515DepthConfig(),
        )
    )
    print("Connecting to RealSense...")
    if not camera.connect():
        raise RuntimeError("Failed to connect to RealSense.")
    return camera, {"connect": (time.perf_counter() - t0) * 1000.0}


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
    """Capture one aligned RGB-D frame.  Returns (rgb, depth_m, K, timings)."""
    print("\nCapturing aligned RGBD frame...")
    t0 = time.perf_counter()
    frame = camera.read()
    capture_ms = (time.perf_counter() - t0) * 1000.0

    if frame.rgb is None:
        raise RuntimeError("RGB frame unavailable.")
    if frame.align_mode != "depth_to_color":
        raise RuntimeError("This test requires depth_to_color alignment.")

    rgb = np.ascontiguousarray(frame.rgb)
    depth_m = np.ascontiguousarray(frame.depth, dtype=np.float32)
    K = np.asarray(frame.K, dtype=np.float64)

    print(f"  RGB:     shape={rgb.shape}, dtype={rgb.dtype}")
    print(
        f"  Depth:   shape={depth_m.shape}, valid={int((depth_m > 0).sum())}/{depth_m.size}"
    )
    print(
        f"  K:       {depth_m.shape[1]}x{depth_m.shape[0]}  fx={K[0,0]:.1f} fy={K[1,1]:.1f}"
    )

    return rgb, depth_m, K, {"capture": capture_ms}


def _run_2d_filters(
    depth_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, float]]:
    """Run the 2-D depth pre-filtering stages used by the processor.

    Returns (mask, edge_vis, timings).
    """
    import cv2

    cfg = PointCloudProcessorConfig()
    timings: dict[str, float] = {}
    total_pixels = depth_m.size

    print("\n" + "=" * 60)
    print("2-D Depth Filtering (before deprojection)")
    print("=" * 60)

    t0 = time.perf_counter()
    depth_m = PointCloudProcessor.apply_depth_median(depth_m, cfg.depth_median_enabled)
    timings["2d_median"] = (time.perf_counter() - t0) * 1000.0
    print(
        f"  0. Median filter ({'ON' if cfg.depth_median_enabled else 'OFF'}):  "
        f"{timings['2d_median']:.2f}ms"
    )

    t0 = time.perf_counter()
    z_flat = depth_m.ravel()
    mask = np.isfinite(z_flat) & (z_flat > cfg.depth_min_m) & (z_flat < cfg.depth_max_m)
    n_gate = int(mask.sum())
    timings["2d_depth_gate"] = (time.perf_counter() - t0) * 1000.0
    print(
        f"  1. Depth gate [{cfg.depth_min_m}, {cfg.depth_max_m}]m:  "
        f"{n_gate:6d} / {total_pixels} pixels  ({timings['2d_depth_gate']:.2f}ms)"
    )

    t0 = time.perf_counter()
    edge_vis = None
    n_before_edge = 0
    if cfg.depth_edge_threshold_m > 0:
        depth_blur = cv2.GaussianBlur(depth_m, (3, 3), sigmaX=0.8)
        laplacian = cv2.Laplacian(depth_blur, cv2.CV_32F, ksize=3)
        edge_mag = np.abs(laplacian)

        if cfg.depth_edge_relative_ratio > 0:
            thresh = np.maximum(
                cfg.depth_edge_threshold_m, depth_m * cfg.depth_edge_relative_ratio
            )
            edge_raw = edge_mag > thresh
        else:
            edge_raw = edge_mag > cfg.depth_edge_threshold_m

        n_raw_edge = int(edge_raw.sum())

        if cfg.depth_edge_dilate_px > 0:
            k = 2 * cfg.depth_edge_dilate_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k))
            edge_2d = cv2.dilate(edge_raw.astype(np.uint8), kernel).astype(bool)
        else:
            edge_2d = edge_raw

        n_edge_dilated = int(edge_2d.sum())
        n_before_edge = int(mask.sum())
        mask = mask & ~edge_2d.ravel()
        n_after_edge = int(mask.sum())

        edge_vis_arr = np.full((*depth_m.shape, 3), 128, dtype=np.uint8)
        edge_vis_arr[edge_raw] = [0, 0, 255]
        edge_vis_arr[edge_2d & ~edge_raw] = [0, 255, 255]
        edge_vis = edge_vis_arr
    timings["2d_edge_filter"] = (time.perf_counter() - t0) * 1000.0

    if cfg.depth_edge_threshold_m > 0:
        print(
            f"  2. Edge filter (LoG, >{cfg.depth_edge_threshold_m*1000:.0f}mm, "
            f"dilate={cfg.depth_edge_dilate_px}px):  "
            f"{n_before_edge - n_after_edge:6d} removed  "
            f"({n_raw_edge} raw edges, {n_edge_dilated} dilated)  "
            f"({timings['2d_edge_filter']:.2f}ms)"
        )
    else:
        print("  2. Edge filter:  DISABLED")

    t0 = time.perf_counter()
    if cfg.speckle_min_pixels > 0:
        mask_2d = mask.reshape(depth_m.shape).astype(np.uint8)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask_2d, connectivity=8)
        for label_id in range(1, len(stats)):
            if stats[label_id, cv2.CC_STAT_AREA] < cfg.speckle_min_pixels:
                mask_2d[labels == label_id] = 0
        n_before = int(mask.sum())
        mask = mask_2d.ravel().astype(bool)
        n_after = int(mask.sum())
    timings["2d_speckle"] = (time.perf_counter() - t0) * 1000.0

    if cfg.speckle_min_pixels > 0:
        print(
            f"  3. Speckle filter (<{cfg.speckle_min_pixels}px):  "
            f"{n_after:6d} survive  ({n_before - n_after} removed)  "
            f"({timings['2d_speckle']:.2f}ms)"
        )
    else:
        print("  3. Speckle filter:  DISABLED")

    if not np.any(mask):
        raise RuntimeError(
            "No pixels survived 2-D gates -- check depth range or edge threshold."
        )
    return mask, edge_vis, timings


def _load_extrinsics(camera_info: dict) -> tuple[np.ndarray, float]:
    """Load T_world_camera from cameras.json.  Returns (T, timing_ms)."""
    print("\nLoading extrinsics from cameras.json...")
    t0 = time.perf_counter()
    calib = CameraCalib()
    cam_name = calib.resolve_name_by_serial(str(camera_info.get("serial", "")))
    T_world_camera = np.asarray(calib.get_extrinsics(cam_name), dtype=np.float64)

    if T_world_camera.shape != (4, 4):
        raise RuntimeError(f"Invalid extrinsic shape: {T_world_camera.shape}")
    if not np.allclose(T_world_camera[3], [0, 0, 0, 1], atol=1e-6):
        raise RuntimeError("Invalid homogeneous transform last row.")

    elapsed = (time.perf_counter() - t0) * 1000.0
    pos = T_world_camera[:3, 3]
    quat_xyzw = R.from_matrix(T_world_camera[:3, :3]).as_quat()
    print(f"  Camera '{cam_name}' (eye-to-hand)")
    print(f"  pos:         {np.round(pos, 4)} m")
    print(f"  quat (xyzw): {np.round(quat_xyzw, 4)}")
    return T_world_camera, elapsed


def _calibrate_desk(
    depth_m: np.ndarray, rgb: np.ndarray, camera: RealSense, T_world_camera: np.ndarray
) -> tuple[tuple[float, ...], float]:
    """Calibrate desk plane via RANSAC, persist, and round-trip verify.

    Returns (plane, timing_ms).
    """
    print("\n" + "=" * 60)
    print("Desk Plane Calibration (one-shot RANSAC)")
    print("=" * 60)
    print("  Ensure the desk is clear of objects for best results.")

    t0 = time.perf_counter()
    rays = camera.get_rays()
    desk_plane = PointCloudProcessor.calibrate_desk_plane(
        depth_m, rgb, rays, T_world_camera
    )
    elapsed = (time.perf_counter() - t0) * 1000.0

    a, b, c_plane, d_plane = desk_plane
    normal = np.array([a, b, c_plane])
    angle_deg = float(
        np.degrees(np.arccos(np.clip(np.dot(normal, [0.0, 0.0, 1.0]), -1.0, 1.0)))
    )
    print(f"  Plane:  {a:.4f}x + {b:.4f}y + {c_plane:.4f}z + {d_plane:.4f} = 0")
    print(f"  Tilt:   {angle_deg:.1f} deg from horizontal")

    # Persist and verify the shared desk-plane calibration.
    plane_file = PACKAGE_DIR / "config" / "desk_plane.json"
    plane_path = str(plane_file)

    answer = input(f"Overwrite {plane_path}? [y/N] ")
    if answer.strip().lower() not in ("y", "yes"):
        print("  Skipped -- desk_plane.json left unchanged.")
        return desk_plane, elapsed

    if plane_file.exists():
        backup = plane_file.with_suffix(
            f".json.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        shutil.copy2(plane_file, backup)
        print(f"  backed up previous plane → {backup.name}")

    PointCloudProcessor.save_desk_plane(desk_plane, plane_path)
    loaded = PointCloudProcessor.load_desk_plane(plane_path)
    print(
        f"  Saved + round-trip OK: a={loaded[0]:.4f} b={loaded[1]:.4f} "
        f"c={loaded[2]:.4f} d={loaded[3]:.4f}"
    )

    return desk_plane, elapsed


def _run_pipeline(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    T_world_camera: np.ndarray,
    desk_plane: tuple[float, ...],
    camera: RealSense,
) -> tuple[np.ndarray, dict[str, float]]:
    """Run PointCloudProcessor and extract stage timings."""
    print("\n" + "=" * 60)
    print("3-D Outlier Removal Pipeline (post-deprojection)")
    print("=" * 60)

    processor = PointCloudProcessor(T_world_camera)
    cfg = processor.config

    if processor.desk_plane is not None:
        print(
            f"  Desk plane: auto-loaded  (clearance={cfg.desk_clearance_m*1000:.0f}mm)"
        )
    else:
        print(
            f"  Desk plane: NONE -- falling back to workspace z_min={cfg.workspace[2]*1000:.0f}mm"
        )

    a, b, c_plane, _d = desk_plane
    normal = np.array([a, b, c_plane])
    angle_deg = float(
        np.degrees(np.arccos(np.clip(np.dot(normal, [0.0, 0.0, 1.0]), -1.0, 1.0)))
    )
    print(
        f"  1. Desk plane removal      -> drops points <{cfg.desk_clearance_m*1000:.0f}mm above desk"
    )
    print(f"                              (tilt-aware: {angle_deg:.1f} deg slope)")
    print(
        f"  2. Workspace crop          -> x in [{cfg.workspace[0]:.1f},{cfg.workspace[3]:.1f}] "
        f"y in [{cfg.workspace[1]:.1f},{cfg.workspace[4]:.1f}] "
        f"z in [{cfg.workspace[2]:.3f},{cfg.workspace[5]:.1f}]"
    )
    print("  3. 5mm voxel downsample    -> averages points per grid cell")
    if cfg.dbscan_min_cluster_size > 0:
        print(
            f"  4. DBSCAN two-in-one       -> noise + clusters <{cfg.dbscan_min_cluster_size} pts  "
            f"(eps={cfg.dbscan_eps*1000:.0f}mm, min={cfg.dbscan_min_points})"
        )
    else:
        print("  4. DBSCAN two-in-one       -> DISABLED")
    if cfg.radius_outlier_min_points > 0:
        print(
            f"  5. Radius outlier          -> <{cfg.radius_outlier_min_points} neighbours "
            f"in {cfg.radius_outlier_radius*1000:.0f}mm"
        )
    else:
        print("  5. Radius outlier          -> DISABLED")
    if cfg.stat_outlier_nb_neighbors > 0:
        print(
            f"  6. Statistical outlier     -> k={cfg.stat_outlier_nb_neighbors} "
            f"deviation >{cfg.stat_outlier_std_ratio} sigma"
        )
    else:
        print("  6. Statistical outlier     -> DISABLED")
    print(f"  7. FPS sampling            -> {cfg.num_points} points")

    # Warm up once, then time the public operation.
    rays = camera.get_rays()
    rays_2d = rays.reshape(depth_m.shape[0], depth_m.shape[1], 3)
    _ = processor.process(depth_m, rgb, rays_2d)

    t0 = time.perf_counter()
    result = processor.process(depth_m, rgb, rays_2d)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    timings = {"pipeline_total": elapsed_ms, "p_pipeline": elapsed_ms}

    if result is not None:
        print(
            f"\n  Output: {result.shape[0]} points  ({timings['pipeline_total']:.1f} ms)"
        )
    else:
        result = np.zeros((0, 6), dtype=np.float32)

    return result, timings


def _print_timing_summary(timings: dict[str, float]) -> None:
    """Print per-stage timing with ASCII bar charts."""
    print("\n" + "=" * 60)
    print("Per-Stage Timing Summary")
    print("=" * 60)

    sections = [
        ("Setup", ["connect", "capture", "extrinsics", "desk_calib"]),
        (
            "2-D Pre-deprojection Filters",
            ["2d_median", "2d_depth_gate", "2d_edge_filter", "2d_speckle"],
        ),
        ("3-D Pipeline (per-frame steady-state)", ["p_pipeline"]),
    ]

    for title, keys in sections:
        print(f"\n  -- {title} --")
        section_total = 0.0
        for key in keys:
            _tprint(f"  {_stage_label(key)}", key, timings)
            stage_ms = timings.get(key, 0.0)
            if not math.isnan(stage_ms):
                section_total += stage_ms

        if title.startswith("3-D"):
            pipeline_ms = timings.get("pipeline_total", 0.0)
            print(f"  {'  3-D subtotal':<30s} {section_total:6.1f} ms")
            print(f"  {'Pipeline total':<30s} {pipeline_ms:6.1f} ms")
        elif title.startswith("2-D"):
            print(f"  {'2-D total':<30s} {section_total:6.1f} ms")


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
    result: np.ndarray, T_world_camera: np.ndarray, cfg: PointCloudProcessorConfig
) -> None:
    """Visualize final world-frame point cloud with coordinate frames and crop box."""
    import open3d as o3d

    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    camera_frame.transform(T_world_camera)

    crop_box = _build_workspace_box(cfg.workspace)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(result[:, :3].astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(result[:, 3:6].astype(np.float64))

    print("\nLaunching open3d visualizer (close window to continue)...")
    print("  World-frame point cloud + workspace crop box (green) + coordinate frames")
    o3d.visualization.draw_geometries(
        [pcd, crop_box, world_frame, camera_frame],
        window_name="Point Cloud",
        point_show_normal=False,
    )


def main() -> int:
    """Run the hardware diagnostic and return a process exit status."""
    cfg = PointCloudDiagnosticConfig()
    all_timings: dict[str, float] = {}

    camera, t = _connect_camera(cfg)
    all_timings.update(t)

    try:
        camera_info = _print_device_info(camera)

        rgb, depth_m, K, t = _capture_frame(camera)
        all_timings.update(t)

        _mask, edge_vis, t = _run_2d_filters(depth_m)
        all_timings.update(t)
        _show_rgbd_panels(rgb, depth_m, edge_vis, cfg)

        T_world_camera, extrinsics_ms = _load_extrinsics(camera_info)
        all_timings["extrinsics"] = extrinsics_ms

        desk_plane, desk_ms = _calibrate_desk(depth_m, rgb, camera, T_world_camera)
        all_timings["desk_calib"] = desk_ms

        result, t = _run_pipeline(depth_m, rgb, T_world_camera, desk_plane, camera)
        all_timings.update(t)

        _print_timing_summary(all_timings)

        if cfg.show_o3d:
            _visualize_result(result, T_world_camera, PointCloudProcessorConfig())
        else:
            print("\n  open3d visualizer skipped (show_o3d=False)")

        print("\nDone.")

    finally:
        camera.disconnect()
        print("RealSense pipeline stopped cleanly.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
