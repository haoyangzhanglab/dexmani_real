#!/usr/bin/env python3
"""Usage: ``python examples/pointcloud_process_example.py``.

Self-contained L515 tabletop point-cloud and depth diagnostic. Connects a
RealSense L515, opens cv2 and open3d GUI windows, and prints per-stage timing.

This file was restored against the NEW native point-cloud API
(``dexmani_real.sensor.pointcloud.build_point_cloud``). The old modules
``dexmani_real.utils.pointcloud_utils`` and
``dexmani_real.sensor.pointcloud_processor`` were deleted in commit 749fe38;
this example no longer imports them.

What changed vs the old deleted version:

  - ``PointCloudProcessor`` (class) -> ``build_point_cloud`` (function).
  - ``PointCloudProcessorConfig`` / ``PointCloudConfig`` now live in
    ``dexmani_real.sensor.pointcloud`` with renamed fields
    (num_points / depth_min_m / depth_max_m / voxel_size_m).
  - The new pipeline is FIXED: valid-mask -> flying-pixel reject -> deproject
    -> transform to xArm-base -> crop(workspace + table) -> voxel
    representatives -> color projection -> fixed-size sample.
  - REMOVED (no new equivalent): the 2-D median / LoG-edge / speckle
    pre-filters, the interactive RANSAC desk-plane calibration plus the
    desk_plane.json save/load round-trip, DBSCAN / radius / statistical
    outlier removal, pytorch3d-FPS sampling, and the sampling-mode /
    voxel-cycle / config-variant comparison modes.  ``table_plane_abcd`` is
    passed as ``None`` here (desk-plane RANSAC was removed), so only the
    workspace crop is active.
  - Output is float32 [N, 6] = xyz in the XARM-BASE frame + rgb(0..1), instead
    of the old camera-frame output.
  - ``make_depth_vis`` / ``depth_valid_ratio`` no longer exist; small jet
    colormap / valid-ratio helpers are inlined below.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

from dexmani_real.config.camera_calib import CameraCalib
from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.sensor.camera_geometry import RGBDGeometry
from dexmani_real.sensor.pointcloud import PointCloudConfig, build_point_cloud
from dexmani_real.sensor.realsense import L515DepthConfig, RealSense, RealSenseConfig


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
    """Render a float depth map (meters) as a uint8 BGR-compatible RGB image.

    # NOTE: removed -- make_depth_vis was deleted with pointcloud_utils; the
    jet-colormap helper is inlined here.
    """
    safe = np.where(np.isfinite(depth_m), depth_m, 0.0)
    normalized = (safe - float(vmin)) / (float(vmax) - float(vmin))
    colored = _jet_colormap(normalized)
    return (colored * 255.0).astype(np.uint8)


def _depth_valid_ratio(
    depth_m: np.ndarray, vmin: float, vmax: float
) -> float:
    """Fraction of pixels with finite depth inside [vmin, vmax] meters.

    # NOTE: removed -- depth_valid_ratio was deleted with pointcloud_utils;
    the mean-of-valid-mask helper is inlined here.
    """
    valid = np.isfinite(depth_m) & (depth_m >= float(vmin)) & (depth_m <= float(vmax))
    return float(np.mean(valid))


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
        ("Depth", _make_depth_vis(depth_m, cfg.vis_depth_min_m, cfg.vis_depth_max_m)),
    ]
    # NOTE: removed -- the LoG+dilate edge-mask visualization came from the old
    # 2-D edge filter, which no longer exists; edge_vis is always None now.
    if edge_vis is not None:
        panels.append(("Edge mask (red=core, yellow=dilated)", edge_vis))

    labeled = []
    rgb_h, rgb_w = bgr.shape[:2]
    for title, image in panels:
        canvas = image.copy()
        if canvas.shape[:2] != (rgb_h, rgb_w):
            # Depth is native 1024x768 while RGB is 640x480 (both 4:3); resize
            # the depth/edge panels to the RGB size so np.hstack lines up.
            canvas = cv2.resize(
                canvas, (rgb_w, rgb_h), interpolation=cv2.INTER_LINEAR
            )
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


def _print_depth_stats(
    depth_m: np.ndarray, vmin: float, vmax: float
) -> float:
    """Print 2-D depth-gate statistics for the diagnostic view."""
    print("\n" + "=" * 60)
    print("2-D Depth Gate (diagnostic view)")
    print("=" * 60)
    valid = np.isfinite(depth_m) & (depth_m >= float(vmin)) & (depth_m <= float(vmax))
    ratio = _depth_valid_ratio(depth_m, vmin, vmax)
    print(
        f"  Range [{vmin:.2f}, {vmax:.2f}] m:  {int(valid.sum()):6d} / "
        f"{depth_m.size} pixels  ({ratio * 100:.1f}%)"
    )
    # NOTE: removed -- the old 2-D median / LoG-edge / speckle pre-filters had
    # no equivalent in the new fixed pipeline; build_point_cloud applies a
    # single internal flying-pixel reject instead.
    return ratio


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
        "p_pipeline": "Complete point-cloud pipeline",
    }.get(key, key)


def _connect_camera(cfg: PointCloudDiagnosticConfig) -> RealSense:
    """Connect camera and return it.

    Connection time is deliberately not timed: it is a one-time hardware
    init (pipeline start + warmup frames), not a per-frame pipeline stage, and
    its multi-second duration would dominate the per-stage timing chart.
    """
    # NOTE: removed -- RealSenseConfig no longer has an ``align_mode`` field;
    # the driver keeps the native depth/color streams and maps them with
    # T_color_from_depth (see get_geometry).
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

    # NOTE: removed -- CameraFrame no longer carries ``align_mode`` or ``K``;
    # the native depth/color streams are mapped by the driver's RGBDGeometry.
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
    """PointCloudConfig matching the production realtime worker.

    Mirrors ``PointCloudLoopConfig.from_runtime`` in
    ``dexmani_real/sensor/pointcloud_process.py``: the workspace crop volume is
    taken from the active runtime policy, not the PointCloudConfig default.
    """
    runtime = resolve_runtime_config()
    ws = runtime.policy.workspace
    return PointCloudConfig(
        num_points=num_points,
        workspace=(
            float(ws.x_min),
            float(ws.y_min),
            float(ws.z_min),
            float(ws.x_max),
            float(ws.y_max),
            float(ws.z_max),
        ),
    )


def _table_plane() -> tuple[float, float, float, float] | None:
    """Active table plane from the runtime environment config, matching production."""
    runtime = resolve_runtime_config()
    table = runtime.environment.table
    return table.plane_abcd if table.enabled else None


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
    # NOTE: removed -- the old PointCloudProcessor supported median / LoG-edge /
    # speckle pre-filters, RANSAC desk-plane removal, DBSCAN / radius /
    # statistical outlier removal, and pytorch3d-FPS sampling.  The new
    # pipeline is FIXED: valid-mask -> flying-pixel reject -> deproject ->
    # transform to xArm-base -> crop(workspace + table) -> voxel
    # representatives -> color projection -> fixed-size sample.
    # NOTE: removed -- interactive desk-plane RANSAC calibration and the
    # desk_plane.json save/load round-trip have no equivalent; the table plane
    # is now loaded from the runtime environment config (same as production).
    table_state = "ENABLED" if table_plane_abcd is not None else "DISABLED"
    print(f"  num_points={config.num_points}  workspace={config.workspace}")
    print(f"  depth range [{config.depth_min_m}, {config.depth_max_m}] m")
    print(f"  voxel_size_m={config.voxel_size_m}  table crop: {table_state}")

    build_kwargs = dict(
        depth_raw=depth_raw,
        color=rgb,
        depth_scale_m=depth_scale_m,
        geometry=geometry,
        T_xarm_base_from_depth=T_xarm_base_from_depth,
        table_plane_abcd=table_plane_abcd,
        config=config,
    )

    # Warm up once, then time the public operation.
    _ = build_point_cloud(**build_kwargs)

    t0 = time.perf_counter()
    result = build_point_cloud(**build_kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    timings = {"pipeline_total": elapsed_ms, "p_pipeline": elapsed_ms}

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
        ("Setup", ["capture", "extrinsics"]),
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
        _show_rgbd_panels(rgb, depth_m, None, cfg)

        geometry = camera.get_geometry()
        T_xarm_base_from_depth, extrinsics_ms = _load_extrinsics(camera_info, geometry)
        all_timings["extrinsics"] = extrinsics_ms

        # Match the production realtime worker: workspace from the runtime
        # policy, table plane from the runtime environment config.
        pcd_config = _production_config(cfg.target_points)
        table_plane_abcd = _table_plane()

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
