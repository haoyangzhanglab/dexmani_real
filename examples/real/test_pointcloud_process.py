#!/usr/bin/env python3
"""Test: capture a 640x480 RGBD frame → camera-frame point cloud → world-frame
point cloud → open3d visualization → desk plane fitting.

Usage:
  conda activate real_robot
  python examples/real/test_pointcloud_process.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pyrealsense2 as rs

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dexmani_real.config.camera_calib import CameraCalib


def main() -> None:
    # ── 1. Connect to RealSense ───────────────────────────────────────────────
    print("Connecting to RealSense...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)

    # Device info
    device = profile.get_device()
    serial = device.get_info(rs.camera_info.serial_number)
    depth_sensor = device.first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    print(f"  Serial:      {serial}")
    print(f"  Depth scale: {depth_scale:.6f} (raw_uint16 × scale = meters)")

    # Color intrinsics
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    K = np.array(
        [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64
    )
    print(f"  Intrinsics:  {intr.width}x{intr.height}  fx={intr.fx:.1f} fy={intr.fy:.1f}")

    # Align depth → color
    align = rs.align(rs.stream.color)

    # Warm up
    print("\nWarming up (30 frames)...")
    for i in range(30):
        pipeline.wait_for_frames()
    print("  Done.")

    # ── 2. Capture frames (with optional post-processing) ─────────────────────
    # L515 "edge elongation" = ToF flying pixels: a boundary pixel's FoV straddles
    # foreground + background, so its depth is a blend that bridges object → bg.
    # SDK filter chain to attenuate it (spatial in disparity space = edge-preserving):
    #     disparity → spatial → temporal → depth
    # Temporal is an EMA across frames, so it is a no-op on a single frame — we feed
    # it a short burst of the (static) scene so both spatial + temporal are effective.
    USE_SDK_FILTERS = True
    BURST = 15 if USE_SDK_FILTERS else 1

    to_disparity = rs.disparity_transform(True)
    to_depth = rs.disparity_transform(False)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)  # low delta = preserve edges
    spatial.set_option(rs.option.holes_fill, 0)            # don't invent depth
    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.3)
    temporal.set_option(rs.option.filter_smooth_delta, 20)

    def postprocess(df: rs.depth_frame) -> rs.depth_frame:
        f = to_disparity.process(df)
        f = spatial.process(f)
        f = temporal.process(f)
        f = to_depth.process(f)
        return f.as_depth_frame()

    print(f"\nCapturing {'a burst of ' + str(BURST) if USE_SDK_FILTERS else 'one'} "
          f"aligned RGBD frame(s) (SDK filters={'ON' if USE_SDK_FILTERS else 'OFF'})...")
    frames = None
    aligned = None
    aligned_depth_frame = None
    for _ in range(BURST):
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        df = aligned.get_depth_frame()
        aligned_depth_frame = postprocess(df) if (USE_SDK_FILTERS and df) else df

    # --- Diagnose raw depth (before alignment, last frame of burst) ---
    raw_depth_frame = frames.get_depth_frame()
    if raw_depth_frame:
        raw_depth = np.asanyarray(raw_depth_frame.get_data())
        raw_valid = (raw_depth > 0).sum()
        print(f"  Raw depth (unaligned):  shape={raw_depth.shape}, nonzero={raw_valid}/{raw_depth.size}")
        if raw_valid > 0:
            print(f"    raw values: min={raw_depth[raw_depth > 0].min()}, "
                  f"max={raw_depth.max()}, "
                  f"mean={raw_depth[raw_depth > 0].mean():.0f}")
            print(f"    → min dist = {raw_depth[raw_depth > 0].min() * depth_scale:.3f} m, "
                  f"max dist = {raw_depth.max() * depth_scale:.3f} m")
    else:
        print("  Raw depth frame: NONE (sensor not producing depth!)")

    color_frame = aligned.get_color_frame()
    if not color_frame or not aligned_depth_frame:
        raise RuntimeError("Failed to capture aligned RGBD frame.")

    rgb_bgr = np.asanyarray(color_frame.get_data())  # (480, 640, 3) BGR uint8
    aligned_depth = np.asanyarray(aligned_depth_frame.get_data())  # (480, 640) uint16

    aligned_valid = (aligned_depth > 0).sum()
    print(f"  Aligned depth:          shape={aligned_depth.shape}, nonzero={aligned_valid}/{aligned_depth.size}")
    if aligned_valid > 0:
        print(f"    aligned values: min={aligned_depth[aligned_depth > 0].min()}, "
              f"max={aligned_depth.max()}, "
              f"mean={aligned_depth[aligned_depth > 0].mean():.0f}")
        print(f"    → min dist = {aligned_depth[aligned_depth > 0].min() * depth_scale:.3f} m, "
              f"max dist = {aligned_depth.max() * depth_scale:.3f} m")

    # Fall back to raw depth if aligned is all zeros
    if aligned_valid == 0 and raw_depth_frame:
        print("\n  ⚠ Aligned depth is all zeros — falling back to raw (unaligned) depth.")
        depth = raw_depth
        depth_frame = raw_depth_frame
        # For raw depth we need depth-stream intrinsics
        depth_profile = depth_frame.get_profile().as_video_stream_profile()
        depth_intr = depth_profile.get_intrinsics()
        K = np.array(
            [[depth_intr.fx, 0, depth_intr.ppx],
             [0, depth_intr.fy, depth_intr.ppy],
             [0, 0, 1]], dtype=np.float64
        )
    else:
        depth = aligned_depth
        depth_frame = aligned_depth_frame

    # RGB: BGR→RGB
    rgb = rgb_bgr[..., ::-1]
    print(f"  RGB:  {rgb.shape} {rgb.dtype}")
    print(f"  Depth: {depth.shape} {depth.dtype}  nonzero={aligned_valid if 'aligned_valid' in dir() else (depth > 0).sum()}")

    # ── 3. Camera-frame point cloud ───────────────────────────────────────────
    print("\nGenerating camera-frame point cloud...")
    depth_m = depth.astype(np.float64) * depth_scale  # raw uint16 → meters

    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    K_inv = np.linalg.inv(K)
    pixels_h = np.stack([u, v, np.ones_like(u)], axis=-1)  # (H, W, 3)
    rays_cam = pixels_h @ K_inv.T  # (H, W, 3) — unit rays in camera frame
    points_cam = rays_cam * depth_m[..., None]  # (H, W, 3)

    # Flatten
    points_flat = points_cam.reshape(-1, 3)
    colors_flat = (rgb.astype(np.float64) / 255.0).reshape(-1, 3)

    # Filter: finite + depth in [0.05, 2.0] m
    z_cam = points_flat[:, 2]  # depth = Z in camera frame
    valid = (
        np.isfinite(points_flat).all(axis=1)
        & (z_cam > 0.05)
        & (z_cam < 2.0)
    )
    pts_cam = points_flat[valid]
    col_cam = colors_flat[valid]
    print(f"  Valid depth range [0.05, 2.0]m: {valid.sum()} / {valid.size} points")
    print(f"  Camera-frame points: {pts_cam.shape[0]}")

    if pts_cam.shape[0] == 0:
        print("\n  ❌ Still no valid points. Raw depth diagnostics:")
        print(f"     depth_scale = {depth_scale}")
        print(f"     depth min/max/mean = {depth.min()}/{depth.max()}/{depth.mean():.1f}")
        z_min, z_max = z_cam.min(), z_cam.max()
        print(f"     depth_m min/max = {z_min:.4f}/{z_max:.4f} m")
        n_neg = (z_cam <= 0).sum()
        n_lo = ((z_cam > 0) & (z_cam <= 0.05)).sum()
        n_ok = ((z_cam > 0.05) & (z_cam < 2.0)).sum()
        n_hi = (z_cam >= 2.0).sum()
        print(f"     z<=0: {n_neg}, 0<z<=0.05: {n_lo}, 0.05<z<2.0: {n_ok}, z>=2.0: {n_hi}")
        pipeline.stop()
        return

    # ── 4. Transform to world frame ───────────────────────────────────────────
    print("\nLoading extrinsics from cameras.json...")
    calib = CameraCalib()
    cam_name = calib.resolve_name_by_serial(serial)
    T_world_camera = calib.get_extrinsics(cam_name)

    from scipy.spatial.transform import Rotation as R

    pos = T_world_camera[:3, 3]
    quat_xyzw = R.from_matrix(T_world_camera[:3, :3]).as_quat()
    print(f"  Camera '{cam_name}' (eye-to-hand)")
    print(f"  T_world_camera pos:         {np.round(pos, 4)} m")
    print(f"  T_world_camera quat (xyzw): {np.round(quat_xyzw, 4)}")

    # p_world = T_world_camera @ p_camera
    ones = np.ones((pts_cam.shape[0], 1), dtype=np.float64)
    pts_cam_h = np.concatenate([pts_cam, ones], axis=1)
    pts_world = (pts_cam_h @ T_world_camera.T)[:, :3]
    print(f"  World-frame points: {pts_world.shape[0]}")

    # ── 4.1 Spatial crop in world frame ────────────────────────────────────
    x_min, x_max = 0.0, 0.8
    y_min, y_max = -0.5, 0.5
    z_min, z_max = 0.0, 0.6
    crop_mask = (
        (pts_world[:, 0] >= x_min) & (pts_world[:, 0] <= x_max)
        & (pts_world[:, 1] >= y_min) & (pts_world[:, 1] <= y_max)
        & (pts_world[:, 2] >= z_min) & (pts_world[:, 2] <= z_max)
    )
    pts_world = pts_world[crop_mask]
    col_cam = col_cam[crop_mask]
    print(f"  After crop (x∈[{x_min},{x_max}], y∈[{y_min},{y_max}], z∈[{z_min},{z_max}]): "
          f"{pts_world.shape[0]} points")

    # ── 5. open3d visualization ───────────────────────────────────────────────
    print("\nLaunching open3d visualizer (close window to continue)...")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(col_cam.astype(np.float64))

    # ── 4.3 Remove residual flying pixels / edge tails ────────────────────────
    # SDK spatial/temporal attenuates but does not delete the interpolated edge
    # tails (they can carry high confidence). Statistical outlier removal drops
    # points whose mean k-NN distance is an outlier — i.e. the sparse points
    # bridging foreground → background. (Alternative: remove_radius_outlier.)
    USE_OUTLIER_REMOVAL = True
    if USE_OUTLIER_REMOVAL and len(pcd.points) > 0:
        n_before = len(pcd.points)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"  Statistical outlier removal: {n_before} → {len(pcd.points)} points "
              f"(removed {n_before - len(pcd.points)} flying pixels)")

    # Downsample for faster rendering
    pcd_ds = pcd.voxel_down_sample(voxel_size=0.003)
    print(f"  Downsampled: {len(pcd_ds.points)} points (3mm voxel)")

    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    camera_frame.transform(T_world_camera)

    print("  Window 1: world-frame point cloud + coordinate frames")
    o3d.visualization.draw_geometries(
        [pcd_ds, world_frame, camera_frame],
        window_name="World-frame Point Cloud",
        point_show_normal=False,
    )

    # ── 6. Fit desk plane (RANSAC on FULL pcd for accuracy) ───────────────────
    print("\nFitting desk plane (RANSAC, distance threshold = 1 cm)...")
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=1000
    )
    a, b, c, d = plane_model

    # Orient normal upward (positive world Z)
    if c < 0:
        a, b, c, d = -a, -b, -c, -d

    inlier_pts_full = np.asarray(pcd.points)[inliers]
    desk_z_mean = float(inlier_pts_full[:, 2].mean())
    desk_z_std = float(inlier_pts_full[:, 2].std())

    normal = np.array([a, b, c])
    z_axis = np.array([0.0, 0.0, 1.0])
    angle_deg = float(np.degrees(np.arccos(np.clip(np.dot(normal, z_axis), -1.0, 1.0))))

    print(f"  Plane equation:   {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    print(f"  Plane normal:     [{a:.4f}, {b:.4f}, {c:.4f}]")
    print(f"  Inlier count:     {len(inliers)} / {len(pcd.points)}")
    print(f"  Desk Z (world):   mean={desk_z_mean:.4f} m  std={desk_z_std:.4f} m")
    print(f"  Tilt from horiz:  {angle_deg:.1f}°")

    # ── 7. Color-coded desk segmentation on downsampled pcd ───────────────────
    # Map inlier status from full pcd → downsampled pcd via KD-tree
    inlier_pcd = o3d.geometry.PointCloud()
    inlier_pcd.points = o3d.utility.Vector3dVector(inlier_pts_full.astype(np.float64))
    inlier_tree = o3d.geometry.KDTreeFlann(inlier_pcd)

    ds_pts = np.asarray(pcd_ds.points)
    colors_viz = np.zeros((len(ds_pts), 3), dtype=np.float64)
    for i, pt in enumerate(ds_pts):
        _, idx, _ = inlier_tree.search_knn_vector_3d(pt, 1)
        if len(idx) > 0 and np.linalg.norm(pt - inlier_pts_full[idx[0]]) < 0.005:
            colors_viz[i] = [0.0, 1.0, 0.0]  # green = desk
        else:
            colors_viz[i] = [1.0, 0.0, 0.0]  # red = non-desk
    pcd_ds.colors = o3d.utility.Vector3dVector(colors_viz)

    print("\n  Window 2: desk segmentation (green=desk, red=other, close to exit)")
    o3d.visualization.draw_geometries(
        [pcd_ds, world_frame, camera_frame],
        window_name="Desk Segmentation: green=desk, red=other",
        point_show_normal=False,
    )

    pipeline.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
