#!/usr/bin/env python3
"""L515 tabletop point-cloud diagnostic.

  - RGB:   640 x 480 @ 30 FPS
  - Depth: 1024 x 768 @ 30 FPS, aligned to RGB (output 640 x 480)
  - Camera-frame valid depth: [0.3, 1.5] m
  - Workspace crop, desk RANSAC, 5 mm voxel, cluster outlier removal, FPS to 2048

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
import torch
from pytorch3d.ops import sample_farthest_points

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dexmani_real.config.camera_calib import CameraCalib
from dexmani_real.sensor.realsense import (
    L515DepthConfig,
    RealSense,
    RealSenseConfig,
)
from dexmani_real.utils.pointcloud_utils import DepthEdgeConfig, DepthValidityConfig, make_depth_vis


# Acquisition settings: intentionally unchanged.
RGB_RESOLUTION = (640, 480)
DEPTH_RESOLUTION = (1024, 768)
FPS = 30

USE_RGBD_VIS = True

# Single-pass outlier removal (policy-inference latency budget): Euclidean
# clustering at eps = 2 x voxel, then drop connected components smaller than
# CLUSTER_MIN_SIZE points (~7.5 cm^2 of surface at 5 mm voxel). One pass covers
# every artifact class — isolated specks/strings become tiny clusters, dense
# clumps stay below the size threshold — while anything resting on / touching
# another structure merges into that structure's cluster, so on-desk objects
# survive (verified: 2x2 cm on-desk object kept 16/16; 300 specks, 10-pt
# string, 8-pt clump all removed; ~1.2 ms @ 3k post-voxel points).
CLUSTER_EPS_M = 0.010
CLUSTER_MIN_SIZE = 30

# Fixed-size policy input. >= TARGET points: farthest point sampling
# (pytorch3d GPU ~9 ms @ 5.3k pts — B=1 underutilizes the GPU but keeps the
# CPU free; open3d CPU FPS is ~7 ms if no CUDA). < TARGET: random duplicate
# padding of existing points.
TARGET_POINTS = 2048


def _show_rgbd_panels(
    rgb_bgr: np.ndarray,
    depth_m: np.ndarray,
) -> None:
    """Show RGB and depth side by side."""
    if not USE_RGBD_VIS:
        return

    try:
        import cv2
    except ImportError:
        print("  RGBD visualization skipped: OpenCV is unavailable.")
        return

    panels = [
        ("RGB", rgb_bgr),
        ("Depth", make_depth_vis(depth_m, 0.3, 2.5)),
    ]

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
        print(
            "\nShowing RGBD window (RGB | depth) — press any key to continue..."
        )
        cv2.imshow("L515 RGBD diagnostic", np.hstack(labeled))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error as error:
        print(f"  RGBD window skipped (no display?): {error}")



def main() -> None:
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
            l515_depth_config=L515DepthConfig(
                enabled=True,
                # Low Ambient Light (runtime enum 3) — dense indoor base preset.
                # 2026-07-15 A/B on this scene: Low-Ambient base + conf=1 gave
                # 95% valid depth; Short-Range base (5) + conf=3 gave 48%.
                visual_preset=3,
                min_distance=190,
                laser_power=100,
                receiver_gain=12,
                # Firmware confidence cull (0-3). At 3 (max) roughly half the
                # image was invalidated; 2 keeps marginal pixels for the
                # driver's image-domain gate + cluster filter to judge.
                confidence_threshold=2,
                noise_filtering=2,  # runtime scale 0-6
            ),
            # Image-domain validity gate: confidence + IR streams, raw depth
            # masked before depth_to_color alignment (specular/overexposure spikes
            # cannot be cleaned by 3-D outlier removal alone).
            depth_validity=DepthValidityConfig(
                confidence_min=2,
                ir_min=2,
                ir_saturation=250,
                saturation_dilate_px=3,
                # Discontinuity band: T(z) = max(5*sigma_z(z), 10mm).
                # sigma_poly calibrated 2026-07-15 (SN f1382055, warm camera, via
                # calibrate_l515_depth.py): sigma_z = -0.94 + 2.93*z mm. Negative
                # below z~0.32m is clamped by t_min; within the workspace the
                # 10mm floor dominates up to ~1.0m. Cold-run confirmation pending.
                edge=DepthEdgeConfig(
                    sigma_poly=(-0.00094, 0.00293),
                    n_sigma=5.0,
                    t_min=0.010,
                    t_max=None,
                    dilate_px=0,
                ),
            ),
        )
    )

    print("Connecting to RealSense...")
    if not camera.connect():
        raise RuntimeError("Failed to connect to RealSense.")

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
            preset_name = depth_sensor.get_option_value_description(
                rs.option.visual_preset,
                preset,
            )
        except RuntimeError:
            preset_name = "unknown"

        print(f"  Runtime preset: {preset} {preset_name}")
        print(f"  Depth scale:    {depth_scale:.6f} (raw_uint16 x scale = meters)")

        # --- Verify key L515 parameters actually applied ---
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
                val = depth_sensor.get_option(opt)
                print(f"  {name}: {val}")
            except RuntimeError:
                print(f"  {name}: <not readable>")

        # Capture a single aligned RGBD frame.
        print("\nCapturing aligned RGBD frame...")
        frame = camera.read()

        if frame.rgb is None:
            raise RuntimeError("RGB frame is unavailable.")
        if frame.align_mode != "depth_to_color":
            raise RuntimeError("This test requires depth_to_color alignment.")

        rgb = np.ascontiguousarray(frame.rgb)
        rgb_bgr = np.ascontiguousarray(rgb[..., ::-1])
        depth_m = np.ascontiguousarray(frame.depth, dtype=np.float32)
        K = np.asarray(frame.K, dtype=np.float64)

        print(f"  RGB:            shape={rgb.shape}, dtype={rgb.dtype}")
        print(
            "  Depth:          "
            f"shape={depth_m.shape}, valid={int((depth_m > 0).sum())}/"
            f"{depth_m.size}"
        )
        print(
            f"  Intrinsics:     {depth_m.shape[1]}x{depth_m.shape[0]} "
            f"fx={K[0, 0]:.1f} fy={K[1, 1]:.1f}"
        )

        depth_for_pointcloud = depth_m

        _show_rgbd_panels(rgb_bgr, depth_m)

        # Camera-frame point cloud.
        print("\nGenerating camera-frame point cloud...")
        # Unit rays precomputed by the driver at connect (edge-LUT pattern):
        # per-frame deprojection is a single multiply — ~10x cheaper than
        # rebuilding meshgrid + inv(K) every frame.
        rays_cam = camera.get_rays()
        if rays_cam.shape[:2] != depth_for_pointcloud.shape:
            raise RuntimeError(
                f"rays shape {rays_cam.shape[:2]} does not match depth {depth_for_pointcloud.shape}."
            )
        points_cam = rays_cam * depth_for_pointcloud[..., None]

        points_flat = points_cam.reshape(-1, 3)
        colors_flat = (rgb.astype(np.float64) / 255.0).reshape(-1, 3)
        z_cam = points_flat[:, 2]

        # Camera-frame depth gate.
        # Lower bound 0.3 m: sensor physics — L515 spec min-Z is 0.25 m (+ margin);
        # closer returns are unreliable ToF data regardless of workspace.
        # Upper bound 1.5 m: workspace geometry — farthest workspace-crop corner is
        # z_cam = 1.34 m under current cameras.json extrinsics (+ ~0.15 m margin);
        # points beyond can never survive the world-frame crop. Re-derive if the
        # camera pose or crop box changes.
        valid = (
            np.isfinite(points_flat).all(axis=1)
            & (z_cam > 0.3)
            & (z_cam < 1.5)
        )
        pts_cam = points_flat[valid]
        col_cam = colors_flat[valid]

        print(f"  Valid depth range [0.3, 1.5]m: {int(valid.sum())} / {valid.size} points")
        print(f"  Camera-frame points: {pts_cam.shape[0]}")
        if pts_cam.shape[0] == 0:
            raise RuntimeError("No valid camera-frame points remain.")

        # Transform to world frame.
        print("\nLoading extrinsics from cameras.json...")
        serial = str(info.get("serial", ""))
        calib = CameraCalib()
        cam_name = calib.resolve_name_by_serial(serial)
        T_world_camera = np.asarray(calib.get_extrinsics(cam_name), dtype=np.float64)

        if T_world_camera.shape != (4, 4):
            raise RuntimeError(f"Invalid extrinsic shape: {T_world_camera.shape}")
        if not np.allclose(T_world_camera[3], [0, 0, 0, 1], atol=1e-6):
            raise RuntimeError("Invalid homogeneous transform last row.")

        from scipy.spatial.transform import Rotation as R

        pos = T_world_camera[:3, 3]
        quat_xyzw = R.from_matrix(T_world_camera[:3, :3]).as_quat()
        print(f"  Camera '{cam_name}' (eye-to-hand)")
        print(f"  T_world_camera pos:         {np.round(pos, 4)} m")
        print(f"  T_world_camera quat (xyzw): {np.round(quat_xyzw, 4)}")

        ones = np.ones((pts_cam.shape[0], 1), dtype=np.float64)
        pts_world = (
            np.concatenate([pts_cam, ones], axis=1) @ T_world_camera.T
        )[:, :3]
        print(f"  World-frame points: {pts_world.shape[0]}")

        # Spatial crop: intentionally unchanged.
        x_min, x_max = 0.0, 0.8
        y_min, y_max = -0.6, 0.6
        z_guard_lo, z_guard_hi = -0.2, 0.8
        crop_mask = (
            (pts_world[:, 0] >= x_min)
            & (pts_world[:, 0] <= x_max)
            & (pts_world[:, 1] >= y_min)
            & (pts_world[:, 1] <= y_max)
            & (pts_world[:, 2] >= z_guard_lo)
            & (pts_world[:, 2] <= z_guard_hi)
        )
        pts_world = pts_world[crop_mask]
        col_cam = col_cam[crop_mask]
        print(
            f"  After X/Y crop (x in [{x_min},{x_max}], y in [{y_min},{y_max}], "
            f"z guard in [{z_guard_lo},{z_guard_hi}]): {pts_world.shape[0]} points"
        )
        if pts_world.shape[0] < 3:
            raise RuntimeError("Too few workspace points for plane fitting.")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_world.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(col_cam.astype(np.float64))

        # Desk plane fitting: original 1 cm threshold retained.
        print("\nFitting desk plane (RANSAC, distance threshold = 1 cm)...")
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.01,
            ransac_n=3,
            num_iterations=1000,
        )
        a, b, c, d = plane_model
        if c < 0:
            a, b, c, d = -a, -b, -c, -d

        inlier_pts_full = np.asarray(pcd.points)[inliers]
        if inlier_pts_full.shape[0] == 0:
            raise RuntimeError("Desk plane has no inliers.")

        desk_z_mean = float(inlier_pts_full[:, 2].mean())
        desk_z_std = float(inlier_pts_full[:, 2].std())
        normal = np.asarray([a, b, c], dtype=np.float64)
        normal /= max(np.linalg.norm(normal), 1e-12)
        angle_deg = float(
            np.degrees(
                np.arccos(
                    np.clip(np.dot(normal, [0.0, 0.0, 1.0]), -1.0, 1.0)
                )
            )
        )

        print(f"  Plane equation:   {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
        print(f"  Plane normal:     [{a:.4f}, {b:.4f}, {c:.4f}]")
        print(f"  Inlier count:     {len(inliers)} / {len(pcd.points)}")
        print(f"  Desk Z (world):   mean={desk_z_mean:.4f} m  std={desk_z_std:.4f} m")
        print(f"  Tilt from horiz:  {angle_deg:.1f} deg")

        # Fixed Z crop.
        z_lo, z_hi = 0.0, 0.8
        pts_all = np.asarray(pcd.points)
        col_all = np.asarray(pcd.colors)
        z_mask = (pts_all[:, 2] >= z_lo) & (pts_all[:, 2] <= z_hi)

        pcd.points = o3d.utility.Vector3dVector(pts_all[z_mask])
        pcd.colors = o3d.utility.Vector3dVector(col_all[z_mask])
        print(
            f"  After fixed Z crop (z in [{z_lo},{z_hi}]): "
            f"{int(z_mask.sum())} points"
        )

        # 5 mm voxel downsample.
        pcd_ds = pcd.voxel_down_sample(voxel_size=0.005)
        print(f"  Downsampled: {len(pcd_ds.points)} points (5mm voxel)")

        # Single-pass outlier removal (see constants at top).
        n_before = len(pcd_ds.points)
        t_start = time.perf_counter()
        labels = np.asarray(pcd_ds.cluster_dbscan(eps=CLUSTER_EPS_M, min_points=1))
        cluster_sizes = np.bincount(labels)
        keep_idx = np.flatnonzero(cluster_sizes[labels] >= CLUSTER_MIN_SIZE)
        pcd_ds = pcd_ds.select_by_index(keep_idx)
        elapsed_ms = (time.perf_counter() - t_start) * 1e3
        print(
            f"  Cluster outlier removal (eps={CLUSTER_EPS_M * 1000:.0f}mm, "
            f"min_size={CLUSTER_MIN_SIZE}): "
            f"removed {n_before - len(pcd_ds.points)}, kept {len(pcd_ds.points)} "
            f"in {int((cluster_sizes >= CLUSTER_MIN_SIZE).sum())}/{len(cluster_sizes)} "
            f"clusters ({elapsed_ms:.1f} ms)"
        )

        # Fixed-size downsample for policy input (see constants at top).
        pts_in = np.asarray(pcd_ds.points, dtype=np.float32)
        col_in = np.asarray(pcd_ds.colors)
        n_in = pts_in.shape[0]
        if n_in == 0:
            raise RuntimeError("No points left before fixed-size downsampling.")
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            # Warm up CUDA so the timing below reflects steady state.
            sample_farthest_points(torch.zeros((1, 8, 3), device="cuda"), K=2)
        t_start = time.perf_counter()
        if n_in >= TARGET_POINTS:
            pts_t = torch.from_numpy(pts_in)[None].to("cuda" if use_cuda else "cpu")
            _, idx_t = sample_farthest_points(pts_t, K=TARGET_POINTS)
            idx = idx_t[0].cpu().numpy()
            method = "FPS/" + ("cuda" if use_cuda else "cpu")
        else:
            pad = np.random.default_rng().integers(0, n_in, TARGET_POINTS - n_in)
            idx = np.concatenate([np.arange(n_in), pad])
            method = "random pad"
        elapsed_ms = (time.perf_counter() - t_start) * 1e3
        pcd_ds = o3d.geometry.PointCloud()
        pcd_ds.points = o3d.utility.Vector3dVector(pts_in[idx].astype(np.float64))
        pcd_ds.colors = o3d.utility.Vector3dVector(col_in[idx])
        print(
            f"  Fixed-size downsample ({method}): {n_in} -> {len(idx)} points "
            f"({elapsed_ms:.1f} ms)"
        )

        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
        camera_frame.transform(T_world_camera)

        # Workspace crop box wireframe (effective bounds after all crop stages).
        crop_corners = np.array([
            [x_min, y_min, z_lo], [x_max, y_min, z_lo],
            [x_max, y_max, z_lo], [x_min, y_max, z_lo],
            [x_min, y_min, z_hi], [x_max, y_min, z_hi],
            [x_max, y_max, z_hi], [x_min, y_max, z_hi],
        ])
        crop_edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ]
        crop_box = o3d.geometry.LineSet()
        crop_box.points = o3d.utility.Vector3dVector(crop_corners)
        crop_box.lines = o3d.utility.Vector2iVector(crop_edges)
        crop_box.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 0.0] for _ in crop_edges])

        print("\nLaunching open3d visualizer (close window to continue)...")
        print(
            "  World-frame point cloud (RGB) + workspace crop box (green) + "
            "coordinate frames (world=large, camera=small)"
        )
        o3d.visualization.draw_geometries(
            [pcd_ds, crop_box, world_frame, camera_frame],
            window_name="World-frame Point Cloud",
            point_show_normal=False,
        )

        print("\nDone.")

    finally:
        camera.disconnect()
        print("RealSense pipeline stopped cleanly.")


if __name__ == "__main__":
    main()