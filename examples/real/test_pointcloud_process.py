#!/usr/bin/env python3
"""L515 tabletop point-cloud diagnostic + desk-plane calibration.

  - Connects RealSense L515, reads back all 11 depth config parameters
  - Captures one aligned RGB-D frame, visualises RGB + depth + edge filter
  - Calibrates the desk plane via RANSAC, persists to desk_plane.json
  - Runs the production PointCloudProcessor pipeline end-to-end
  - Prints per-stage timing with ASCII bar charts for bottleneck analysis
  - Visualises the final world-frame point cloud (open3d)

Usage:
  conda activate real_robot
  python examples/real/test_pointcloud_process.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dexmani_real.config.camera_calib import CameraCalib
from dexmani_real.sensor.pointcloud_processor import PointCloudProcessor, PointCloudProcessorConfig
from dexmani_real.sensor.realsense import L515DepthConfig, RealSense, RealSenseConfig
from dexmani_real.utils.pointcloud_utils import make_depth_vis

RGB_RESOLUTION = (640, 480)
DEPTH_RESOLUTION = (1024, 768)
FPS = 30
TARGET_POINTS = 2048

USE_RGBD_VIS = True
USE_O3D_VIS = True


def _show_rgbd_panels(
    rgb_bgr: np.ndarray,
    depth_m: np.ndarray,
    edge_vis: np.ndarray | None = None,
) -> None:
    """Show RGB, depth, and optional edge mask side by side."""
    if not USE_RGBD_VIS:
        return

    try:
        import cv2
    except ImportError:
        print("  RGBD visualization skipped: OpenCV is unavailable.")
        return

    panels: list[tuple[str, np.ndarray]] = [
        ("RGB", rgb_bgr),
        ("Depth", make_depth_vis(depth_m, 0.3, 2.5)),
    ]
    if edge_vis is not None:
        panels.append(("Edge mask (red=core, yellow=dilated)", edge_vis))

    labeled: list[np.ndarray] = []
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
        print("\nShowing RGBD window — press any key to continue...")
        cv2.imshow("L515 RGBD diagnostic", np.hstack(labeled))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error as error:
        print(f"  RGBD window skipped (no display?): {error}")


def main() -> None:
    # ═══════════════════════════════════════════════════════════
    # Timing accumulators
    # ═══════════════════════════════════════════════════════════
    _timings: dict[str, float] = {}

    # ── Connect ──
    t_connect_0 = time.perf_counter()
    camera = RealSense(
        RealSenseConfig(
            camera_name="realsense",
            depth_resolution=DEPTH_RESOLUTION,
            color_resolution=RGB_RESOLUTION,
            fps=FPS,
            enable_color=True,
            align_mode="depth_to_color",
            enable_global_time=True,
            warmup_frames=30,
            # Use the production defaults (Short Range preset, tuned for
            # close-range dexterous manipulation).  See L515DepthConfig
            # docstring in dexmani_real/sensor/realsense.py for rationale.
            l515_depth_config=L515DepthConfig(),
        )
    )

    print("Connecting to RealSense...")
    if not camera.connect():
        raise RuntimeError("Failed to connect to RealSense.")
    _timings["connect"] = (time.perf_counter() - t_connect_0) * 1000.0

    try:
        info = camera.get_device_info()
        print("=" * 60)
        print("Device:", info.get("name", ""))
        print("Serial:", info.get("serial", ""))
        print("Firmware:", info.get("firmware", ""))
        print("=" * 60)

        if camera.profile is None:
            raise RuntimeError("Pipeline profile is unavailable.")

        active_device = camera.profile.get_device()
        depth_sensor = active_device.first_depth_sensor()
        depth_scale = camera.get_depth_scale()

        preset = depth_sensor.get_option(rs.option.visual_preset)
        try:
            preset_name = depth_sensor.get_option_value_description(rs.option.visual_preset, preset)
        except RuntimeError:
            preset_name = "unknown"

        print(f"  Runtime preset: {preset} {preset_name}")
        print(f"  Depth scale:    {depth_scale:.6f} (raw_uint16 x scale = meters)")

        # --- Sensor read-back ---
        readable_opts = [
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
        print("\n  --- Sensor read-back ---")
        for name, opt in readable_opts:
            try:
                print(f"  {name}: {depth_sensor.get_option(opt)}")
            except RuntimeError:
                print(f"  {name}: <not readable>")

        # ── Capture one aligned RGB-D frame ──
        print("\nCapturing aligned RGBD frame...")
        t_cap_0 = time.perf_counter()
        frame = camera.read()
        _timings["capture"] = (time.perf_counter() - t_cap_0) * 1000.0

        if frame.rgb is None:
            raise RuntimeError("RGB frame is unavailable.")
        if frame.align_mode != "depth_to_color":
            raise RuntimeError("This test requires depth_to_color alignment.")

        rgb = np.ascontiguousarray(frame.rgb)
        rgb_bgr = np.ascontiguousarray(rgb[..., ::-1])
        depth_m = np.ascontiguousarray(frame.depth, dtype=np.float32)
        K = np.asarray(frame.K, dtype=np.float64)
        total_pixels = depth_m.size

        print(f"  RGB:            shape={rgb.shape}, dtype={rgb.dtype}")
        print(f"  Depth:          shape={depth_m.shape}, valid={int((depth_m > 0).sum())}/{total_pixels}")
        print(f"  Intrinsics:     {depth_m.shape[1]}x{depth_m.shape[0]} fx={K[0,0]:.1f} fy={K[1,1]:.1f}")

        # ═══════════════════════════════════════════════════════════
        # 2-D depth filtering (pre-deprojection, production-identical)
        # ═══════════════════════════════════════════════════════════
        print("\n" + "=" * 60)
        print("2-D Depth Filtering (before deprojection)")
        print("=" * 60)

        cfg_default = PointCloudProcessorConfig()

        # Warm up lazy imports so timing below reflects steady-state.
        import cv2  # noqa: F811

        # --- 1. Depth range gate ---
        t_gate_0 = time.perf_counter()
        z_flat = depth_m.ravel()
        mask = np.isfinite(z_flat) & (z_flat > cfg_default.depth_min_m) & (z_flat < cfg_default.depth_max_m)
        n_depth_gate = int(mask.sum())
        _timings["2d_depth_gate"] = (time.perf_counter() - t_gate_0) * 1000.0
        print(
            f"  1. Depth gate [{cfg_default.depth_min_m}, {cfg_default.depth_max_m}]m:       "
            f"{n_depth_gate:6d} / {total_pixels} pixels  ({_timings['2d_depth_gate']:.2f}ms)"
        )

        # --- 2. Depth edge filter (LoG, byte-identical to production) ---
        t_edge_0 = time.perf_counter()
        edge_vis: np.ndarray | None = None
        if cfg_default.depth_edge_threshold_m > 0:

            depth_blur = cv2.GaussianBlur(depth_m, (3, 3), sigmaX=0.8)
            laplacian = cv2.Laplacian(depth_blur, cv2.CV_32F, ksize=3)
            edge_mag = np.abs(laplacian)

            if cfg_default.depth_edge_relative_ratio > 0:
                thresh = np.maximum(cfg_default.depth_edge_threshold_m, depth_m * cfg_default.depth_edge_relative_ratio)
                edge_raw = edge_mag > thresh
            else:
                edge_raw = edge_mag > cfg_default.depth_edge_threshold_m

            n_raw_edge = int(edge_raw.sum())

            # Dilate with cross kernel (same as production).
            if cfg_default.depth_edge_dilate_px > 0:
                k = 2 * cfg_default.depth_edge_dilate_px + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k))
                edge_2d = cv2.dilate(edge_raw.astype(np.uint8), kernel).astype(bool)
            else:
                edge_2d = edge_raw

            n_edge_dilated = int(edge_2d.sum())
            n_before_edge = int(mask.sum())
            mask = mask & ~edge_2d.ravel()
            n_after_edge = int(mask.sum())

            # Build edge overlay: grey = kept, red = raw edge core, yellow = dilation band.
            edge_vis = np.full((*depth_m.shape, 3), 128, dtype=np.uint8)
            edge_vis[edge_raw] = [0, 0, 255]
            edge_vis[edge_2d & ~edge_raw] = [0, 255, 255]
        _timings["2d_edge_filter"] = (time.perf_counter() - t_edge_0) * 1000.0

        if cfg_default.depth_edge_threshold_m > 0:
            print(
                f"  2. Edge filter (LoG, >{cfg_default.depth_edge_threshold_m*1000:.0f}mm abs, "
                f"dilate={cfg_default.depth_edge_dilate_px}): "
                f"{n_before_edge - n_after_edge:6d} removed  "
                f"(raw edges={n_raw_edge}, dilated={n_edge_dilated}), "
                f"{n_after_edge:6d} survive  ({_timings['2d_edge_filter']:.2f}ms)"
            )
        else:
            print(f"  2. Edge filter:                                      DISABLED")

        _show_rgbd_panels(rgb_bgr, depth_m, edge_vis)

        if not np.any(mask):
            raise RuntimeError("No pixels survived 2-D gates — check depth range or edge threshold.")

        # --- 3. Speckle filter ---
        t_speckle_0 = time.perf_counter()
        if cfg_default.speckle_min_pixels > 0:
            mask_2d = mask.reshape(depth_m.shape).astype(np.uint8)
            _, labels, stats, _ = cv2.connectedComponentsWithStats(mask_2d, connectivity=8)
            for label_id in range(1, len(stats)):
                if stats[label_id, cv2.CC_STAT_AREA] < cfg_default.speckle_min_pixels:
                    mask_2d[labels == label_id] = 0
            n_before_speckle = int(mask.sum())
            mask = mask_2d.ravel().astype(bool)
            n_after_speckle = int(mask.sum())
        _timings["2d_speckle"] = (time.perf_counter() - t_speckle_0) * 1000.0

        if cfg_default.speckle_min_pixels > 0:
            print(
                f"  3. Speckle filter (<{cfg_default.speckle_min_pixels}px components):     "
                f"{n_after_speckle:6d} survive  ({n_before_speckle - n_after_speckle} removed)"
                f"  ({_timings['2d_speckle']:.2f}ms)"
            )
        else:
            print(f"  3. Speckle filter:                                    DISABLED")

        print(
            f"     (median filter: {'ON' if cfg_default.depth_median_enabled else 'OFF'} "
            f"— applied before all gates above)"
        )

        # ═══════════════════════════════════════════════════════════
        # Load extrinsics
        # ═══════════════════════════════════════════════════════════
        print("\nLoading extrinsics from cameras.json...")
        t_ext_0 = time.perf_counter()
        serial = str(info.get("serial", ""))
        calib = CameraCalib()
        cam_name = calib.resolve_name_by_serial(serial)
        T_world_camera = np.asarray(calib.get_extrinsics(cam_name), dtype=np.float64)

        if T_world_camera.shape != (4, 4):
            raise RuntimeError(f"Invalid extrinsic shape: {T_world_camera.shape}")
        if not np.allclose(T_world_camera[3], [0, 0, 0, 1], atol=1e-6):
            raise RuntimeError("Invalid homogeneous transform last row.")
        _timings["extrinsics"] = (time.perf_counter() - t_ext_0) * 1000.0

        pos = T_world_camera[:3, 3]
        quat_xyzw = R.from_matrix(T_world_camera[:3, :3]).as_quat()
        print(f"  Camera '{cam_name}' (eye-to-hand)")
        print(f"  T_world_camera pos:         {np.round(pos, 4)} m")
        print(f"  T_world_camera quat (xyzw): {np.round(quat_xyzw, 4)}")

        # ═══════════════════════════════════════════════════════════
        # Calibrate desk plane (one-shot RANSAC)
        # ═══════════════════════════════════════════════════════════
        rays = camera.get_rays()
        print("\n" + "=" * 60)
        print("Desk Plane Calibration (one-shot RANSAC)")
        print("=" * 60)
        print("  Make sure the desk is clear of objects for best results.")

        t_desk_0 = time.perf_counter()
        desk_plane = PointCloudProcessor.calibrate_desk_plane(depth_m, rgb, rays, T_world_camera)
        _timings["desk_calib"] = (time.perf_counter() - t_desk_0) * 1000.0

        a, b, c_plane, d_plane = desk_plane
        normal = np.array([a, b, c_plane])
        angle_deg = float(np.degrees(np.arccos(np.clip(np.dot(normal, [0.0, 0.0, 1.0]), -1.0, 1.0))))
        print(f"  Plane:  {a:.4f}x + {b:.4f}y + {c_plane:.4f}z + {d_plane:.4f} = 0")
        print(f"  Tilt:   {angle_deg:.1f} deg from horizontal")

        # Persist.
        _repo_root = Path(__file__).resolve().parents[2]
        _desk_plane_path = str(_repo_root / "dexmani_real" / "config" / "desk_plane.json")
        PointCloudProcessor.save_desk_plane(desk_plane, _desk_plane_path)
        _loaded = PointCloudProcessor.load_desk_plane(_desk_plane_path)
        print(f"  Saved + round-trip OK: a={_loaded[0]:.4f} b={_loaded[1]:.4f} c={_loaded[2]:.4f} d={_loaded[3]:.4f}")

        # ═══════════════════════════════════════════════════════════
        # Production pipeline with per-stage point counts
        # ═══════════════════════════════════════════════════════════
        print("\n" + "=" * 60)
        print("3-D Outlier Removal Pipeline (post-deprojection)")
        print("=" * 60)
        print("  Each stage below removes a different noise class.")
        print()

        processor = PointCloudProcessor(T_world_camera)
        cfg = processor.config

        if processor._desk_plane is not None:
            print(f"  Desk plane: auto-loaded from file  (clearance={cfg.desk_clearance_m*1000:.0f}mm)")
        else:
            print(f"  Desk plane: NONE — falling back to workspace z_min={cfg.workspace[2]*1000:.0f}mm")

        # --- Run the full pipeline once (warm-up) ---
        rays_2d = rays.reshape(depth_m.shape[0], depth_m.shape[1], 3)
        _ = processor.process(depth_m, rgb, rays_2d)

        # --- Describe each stage ---
        print(f"  1. Desk plane removal       → drops points <{cfg.desk_clearance_m*1000:.0f}mm above desk")
        print(f"                              (tilt-aware: handles {angle_deg:.1f} deg desk slope)")
        print(
            f"  2. Workspace crop           → keeps only x in [{cfg.workspace[0]:.1f},{cfg.workspace[3]:.1f}] "
            f"y in [{cfg.workspace[1]:.1f},{cfg.workspace[4]:.1f}] "
            f"z in [{cfg.workspace[2]:.3f},{cfg.workspace[5]:.1f}]"
        )
        print(f"  3. 5mm voxel downsample     → averages points in each 5mm grid cell")
        if cfg.dbscan_min_cluster_size > 0:
            print(
                f"  4. DBSCAN two-in-one filter  → noise pts (label=-1) + clusters <{cfg.dbscan_min_cluster_size} pts"
            )
            print(
                f"                              (eps={cfg.dbscan_eps*1000:.0f}mm, min_points={cfg.dbscan_min_points})"
            )
        else:
            print(f"  4. DBSCAN two-in-one filter  → DISABLED")
        if cfg.radius_outlier_min_points > 0:
            print(
                f"  5. Radius outlier removal   → drops points with <{cfg.radius_outlier_min_points} "
                f"neighbours in {cfg.radius_outlier_radius*1000:.0f}mm"
            )
        else:
            print(f"  5. Radius outlier removal   → DISABLED (DBSCAN noise removal covers it)")
        if cfg.stat_outlier_nb_neighbors > 0:
            print(
                f"  6. Statistical outlier      → drops points whose mean k-NN distance "
                f"(k={cfg.stat_outlier_nb_neighbors}) deviates >{cfg.stat_outlier_std_ratio} sigma"
            )
        else:
            print(f"  6. Statistical outlier      → DISABLED (DBSCAN noise removal covers it)")
        print(f"  7. Farthest-point sampling  → selects {cfg.num_points} points with max spatial coverage")

        # --- Measure a single-frame timing breakdown by temporarily lower the log interval ---
        # Save original accumulators, reset, run one frame, read back.
        saved = (
            processor._t_numpy,
            processor._t_voxel,
            processor._t_dbscan,
            processor._t_radius,
            processor._t_stat,
            processor._t_fps,
            processor._t_in_n,
            processor._t_voxel_n,
            processor._t_radius_n,
            processor._t_n,
        )
        processor._t_numpy = processor._t_voxel = processor._t_dbscan = 0.0
        processor._t_radius = processor._t_stat = processor._t_fps = 0.0
        processor._t_in_n = processor._t_voxel_n = processor._t_radius_n = 0
        processor._t_n = 0

        t_prod_0 = time.perf_counter()
        result = processor.process(depth_m, rgb, rays_2d)
        _timings["pipeline_total"] = (time.perf_counter() - t_prod_0) * 1000.0

        # Extract single-frame stage timings from processor accumulators.
        _timings["p_numpy"] = processor._t_numpy  # 2-D gates + deproject + crop
        _timings["p_voxel"] = processor._t_voxel
        _timings["p_dbscan"] = processor._t_dbscan
        _timings["p_radius"] = processor._t_radius
        _timings["p_stat"] = processor._t_stat
        _timings["p_fps"] = processor._t_fps

        # Restore.
        (
            processor._t_numpy,
            processor._t_voxel,
            processor._t_dbscan,
            processor._t_radius,
            processor._t_stat,
            processor._t_fps,
            processor._t_in_n,
            processor._t_voxel_n,
            processor._t_radius_n,
            processor._t_n,
        ) = saved

        if result is not None:
            print(f"\n  Output: {result.shape[0]} points  ({_timings['pipeline_total']:.1f} ms)")
        else:
            print(f"\n  Output: no points survived  ({_timings['pipeline_total']:.1f} ms)")
            result = np.zeros((0, 6), dtype=np.float32)

        # ═══════════════════════════════════════════════════════════
        # Per-stage timing summary
        # ═══════════════════════════════════════════════════════════
        print("\n" + "=" * 60)
        print("Per-Stage Timing Summary")
        print("=" * 60)

        def _tprint(label: str, key: str) -> None:
            ms = _timings.get(key, 0.0)
            pct = ms / max(_timings.get("pipeline_total", 1.0), 0.001) * 100
            if ms < 0.01:
                print(f"  {label:<30s}     —   (disabled)")
            else:
                bar = "█" * max(1, int(pct / 2.5))
                print(f"  {label:<30s} {ms:6.1f} ms  {bar}")

        print("\n  ── Setup ──")
        _tprint("Camera connect", "connect")
        _tprint("Frame capture", "capture")
        _tprint("Extrinsics load", "extrinsics")
        _tprint("Desk plane calibration", "desk_calib")

        print("\n  ── 2-D Pre-deprojection Filters ──")
        _tprint("Depth gate", "2d_depth_gate")
        _tprint("Edge filter (LoG + dilate)", "2d_edge_filter")
        _tprint("Speckle filter", "2d_speckle")
        _2d_total = _timings.get("2d_depth_gate", 0) + _timings.get("2d_edge_filter", 0) + _timings.get("2d_speckle", 0)
        print(f"  {'2-D total':<30s} {_2d_total:6.1f} ms")

        print("\n  ── 3-D Pipeline (per-frame steady-state) ──")
        _tprint("  2-D gates + deproject + crop", "p_numpy")
        _tprint("  5mm voxel downsample", "p_voxel")
        _tprint("  DBSCAN two-in-one filter", "p_dbscan")
        _tprint("  Radius outlier removal", "p_radius")
        _tprint("  Statistical outlier", "p_stat")
        _tprint("  FPS to 2048", "p_fps")
        _p3d_total = sum(_timings.get(k, 0) for k in ["p_numpy", "p_voxel", "p_dbscan", "p_radius", "p_stat", "p_fps"])
        print(f"  {'  3-D subtotal':<30s} {_p3d_total:6.1f} ms")
        print(f"  {'Pipeline total':<30s} {_timings.get('pipeline_total', 0):6.1f} ms  (budget: 62.5ms @ 16Hz)")

        # ═══════════════════════════════════════════════════════════
        # Visualise final point cloud
        # ═══════════════════════════════════════════════════════════
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
        camera_frame.transform(T_world_camera)

        ws = cfg.workspace
        crop_corners = np.array(
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
        crop_edges = [
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
        crop_box = o3d.geometry.LineSet()
        crop_box.points = o3d.utility.Vector3dVector(crop_corners)
        crop_box.lines = o3d.utility.Vector2iVector(crop_edges)
        crop_box.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 0.0] for _ in crop_edges])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(result[:, :3].astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(result[:, 3:6].astype(np.float64))

        if USE_O3D_VIS:
            print("\nLaunching open3d visualizer (close window to continue)...")
            print("  World-frame point cloud + workspace crop box (green) + coordinate frames")
            o3d.visualization.draw_geometries(
                [pcd, crop_box, world_frame, camera_frame],
                window_name="Point Cloud (production pipeline)",
                point_show_normal=False,
            )
        else:
            print("\n  open3d visualizer skipped (USE_O3D_VIS=False)")

        print("\nDone.")

    finally:
        camera.disconnect()
        print("RealSense pipeline stopped cleanly.")


if __name__ == "__main__":
    main()
