#!/usr/bin/env python3
"""L515 tabletop point-cloud diagnostic.

  - RGB:   640 x 480 @ 30 FPS
  - Depth: 640 x 480 @ 30 FPS
  - Depth is aligned to RGB
  - Camera-frame valid depth: [0.3, 2.5] m
  - Workspace crop, desk RANSAC, 3 mm voxel

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

# Conservative residual 3-D tail removal after 3 mm voxelization.
USE_RADIUS_OUTLIER_REMOVAL = True
RADIUS_OUTLIER_RADIUS_M = 0.010
RADIUS_OUTLIER_MIN_NEIGHBORS = 6

USE_RGBD_VIS = True





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
                visual_preset=5,
                depth_units=0.000250000011874363,
                depth_offset=4.5,
                min_distance=190,
                laser_power=100,
                receiver_gain=18,
                confidence_threshold=3,
                digital_gain=2,
                noise_filtering=4,
                noise_estimation=0.0,
                pre_processing_sharpening=0.0,
                post_processing_sharpening=1,
                alternate_ir=0.0,
                enable_ir_reflectivity=0.0,
                enable_max_usable_range=0.0,
                error_polling_enabled=1,
                frames_queue_size=16,
                freefall_detection_enabled=1,
                global_time_enabled=0.0,
                host_performance=0.0,
                inter_cam_sync_mode=0.0,
                invalidation_bypass=0.0,
                reset_camera_accuracy_health=0.0,
                sensor_mode=0.0,
                trigger_camera_accuracy_health=0.0,
            ),
            # Image-domain validity gate: confidence + IR streams, raw depth
            # masked before depth_to_color alignment (specular/overexposure spikes
            # cannot be cleaned by 3-D outlier removal alone).
            depth_validity=DepthValidityConfig(
                confidence_min=2,
                ir_min=2,
                ir_saturation=250,
                saturation_dilate_px=3,
                # Discontinuity band: T(z) = max(5*sigma_z(z), 8mm), sigma_z = 1.0 + 1.2*z mm
                # -> 8mm @0.5m, 11mm @1.0m. Calibrate sigma_poly from plane temporal std.
                edge=DepthEdgeConfig(
                    sigma_poly=(0.0010, 0.0012),
                    n_sigma=5.0,
                    t_min=0.008,
                    t_max=None,
                    dilate_px=1,
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

        # Camera-frame point cloud. Depth range intentionally unchanged.
        print("\nGenerating camera-frame point cloud...")
        height, width = depth_for_pointcloud.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        pixels_h = np.stack([u, v, np.ones_like(u)], axis=-1)
        rays_cam = pixels_h @ np.linalg.inv(K).T
        points_cam = rays_cam * depth_for_pointcloud[..., None]

        points_flat = points_cam.reshape(-1, 3)
        colors_flat = (rgb.astype(np.float64) / 255.0).reshape(-1, 3)
        z_cam = points_flat[:, 2]

        valid = (
            np.isfinite(points_flat).all(axis=1)
            & (z_cam > 0.3)
            & (z_cam < 2.5)
        )
        pts_cam = points_flat[valid]
        col_cam = colors_flat[valid]

        print(f"  Valid depth range [0.3, 2.5]m: {int(valid.sum())} / {valid.size} points")
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

        # Tight Z crop: intentionally unchanged.
        z_below_desk, z_above_desk = 0.03, 0.6
        z_lo = desk_z_mean - z_below_desk
        z_hi = desk_z_mean + z_above_desk
        pts_all = np.asarray(pcd.points)
        col_all = np.asarray(pcd.colors)
        z_mask = (pts_all[:, 2] >= z_lo) & (pts_all[:, 2] <= z_hi)

        pcd.points = o3d.utility.Vector3dVector(pts_all[z_mask])
        pcd.colors = o3d.utility.Vector3dVector(col_all[z_mask])
        print(
            f"  After desk-anchored Z crop (z in [{z_lo:.3f},{z_hi:.3f}]): "
            f"{int(z_mask.sum())} points"
        )

        # Original 3 mm voxel retained.
        pcd_ds = pcd.voxel_down_sample(voxel_size=0.003)
        print(f"  Downsampled: {len(pcd_ds.points)} points (3mm voxel)")

        # Radius outlier filter (disabled).
        # if USE_RADIUS_OUTLIER_REMOVAL and len(pcd_ds.points) > 0:
        #     before = len(pcd_ds.points)
        #     pcd_ds, _ = pcd_ds.remove_radius_outlier(
        #         nb_points=RADIUS_OUTLIER_MIN_NEIGHBORS,
        #         radius=RADIUS_OUTLIER_RADIUS_M,
        #     )
        #     removed_3d = before - len(pcd_ds.points)
        #     print(
        #         "  Radius outlier filter: "
        #         f"removed {removed_3d}/{before} points "
        #         f"({100.0 * removed_3d / max(before, 1):.2f}%)"
        #     )

        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
        camera_frame.transform(T_world_camera)

        print("\nLaunching open3d visualizer (close window to continue)...")
        print("  Window 1: cleaned world-frame point cloud + coordinate frames")
        o3d.visualization.draw_geometries(
            [pcd_ds, world_frame, camera_frame],
            window_name="World-frame Point Cloud",
            point_show_normal=False,
        )

        # Desk color coding on the cleaned/downsampled point cloud.
        inlier_pcd = o3d.geometry.PointCloud()
        inlier_pcd.points = o3d.utility.Vector3dVector(
            inlier_pts_full.astype(np.float64)
        )
        inlier_tree = o3d.geometry.KDTreeFlann(inlier_pcd)

        ds_pts = np.asarray(pcd_ds.points)
        colors_viz = np.zeros((len(ds_pts), 3), dtype=np.float64)
        for index, point in enumerate(ds_pts):
            _, nearest, _ = inlier_tree.search_knn_vector_3d(point, 1)
            if (
                len(nearest) > 0
                and np.linalg.norm(point - inlier_pts_full[nearest[0]]) < 0.005
            ):
                colors_viz[index] = [0.0, 1.0, 0.0]
            else:
                colors_viz[index] = [1.0, 0.0, 0.0]
        pcd_ds.colors = o3d.utility.Vector3dVector(colors_viz)

        print("\n  Window 2: desk segmentation (green=desk, red=other)")
        o3d.visualization.draw_geometries(
            [pcd_ds, world_frame, camera_frame],
            window_name="Desk Segmentation: green=desk, red=other",
            point_show_normal=False,
        )

        print("\nDone.")

    finally:
        camera.disconnect()
        print("RealSense pipeline stopped cleanly.")


if __name__ == "__main__":
    main()