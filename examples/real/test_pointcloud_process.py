#!/usr/bin/env python3
"""L515 tabletop point-cloud diagnostic.

The script keeps the original acquisition and workspace settings:
  - RGB:   640 x 480 @ 30 FPS
  - Depth: 640 x 480 @ 30 FPS
  - Depth is aligned to RGB
  - Camera-frame valid depth: [0.3, 2.5] m
  - Original world workspace, desk crop, RANSAC threshold and 3 mm voxel

Compared with the previous version, it uses the production RealSense driver,
a conservative 5x5 mixed-edge filter, a removal-ratio safety valve, and a
lightweight 3-D radius filter after voxel downsampling.

Usage:
  conda activate real_robot
  python examples/real/test_pointcloud_process.py
"""

from __future__ import annotations

import sys
import warnings
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
    remove_l515_mixed_edge_depth,
)
from dexmani_real.utils.pointcloud_utils import make_depth_vis


# Acquisition settings: intentionally unchanged.
RGB_RESOLUTION = (640, 480)
DEPTH_RESOLUTION = (640, 480)
FPS = 30

# Static diagnostic only. Set to 1 to disable temporal median.
TEMPORAL_MEDIAN_FRAMES = 5

# Conservative 2-D L515 mixed-edge filter.
EDGE_JUMP_THRESHOLD_M = 0.020
EDGE_SURFACE_MARGIN_M = 0.004
EDGE_FILTER_RADIUS = 2
EDGE_MIN_VALID_NEIGHBORS = 6
EDGE_MAX_REMOVED_RATIO = 0.08

# Conservative residual 3-D tail removal after 3 mm voxelization.
USE_RADIUS_OUTLIER_REMOVAL = True
RADIUS_OUTLIER_RADIUS_M = 0.010
RADIUS_OUTLIER_MIN_NEIGHBORS = 6
RADIUS_OUTLIER_MAX_REMOVED_RATIO = 0.15

USE_RGBD_VIS = True



def _median_valid_depth(depth_frames: list[np.ndarray]) -> np.ndarray:
    """Compute a valid-only temporal median; zero remains invalid."""
    if not depth_frames:
        raise ValueError("depth_frames must not be empty")

    stack = np.stack(
        [
            np.where(
                np.isfinite(depth) & (depth > 0),
                depth,
                np.nan,
            )
            for depth in depth_frames
        ],
        axis=0,
    ).astype(np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(stack, axis=0)

    return np.nan_to_num(median, nan=0.0).astype(np.float32)



def _show_rgbd_panels(
    rgb_bgr: np.ndarray,
    depth_single_m: np.ndarray,
    depth_median_m: np.ndarray,
    depth_filtered_m: np.ndarray,
    *,
    filter_accepted: bool,
) -> None:
    """Show the same [0.3, 2.5] m depth visualization range as before."""
    if not USE_RGBD_VIS:
        return

    try:
        import cv2
    except ImportError:
        print("  RGBD visualization skipped: OpenCV is unavailable.")
        return

    status = "accepted" if filter_accepted else "rejected"
    panels = [
        ("RGB", rgb_bgr),
        ("Depth single", make_depth_vis(depth_single_m, 0.3, 2.5)),
        ("Depth median", make_depth_vis(depth_median_m, 0.3, 2.5)),
        (f"Depth edge filter ({status})", make_depth_vis(depth_filtered_m, 0.3, 2.5)),
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
            "\nShowing RGBD window (RGB | single | median | edge filter) "
            "— press any key to continue..."
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
            depth_hole_filling=False,
            enable_sdk_spatial_filter=False,
            # This test calls the same filter explicitly so it can visualize the
            # candidate and report whether the safety valve accepted it.
            enable_l515_flying_pixel_filter=False,
            enable_global_time=True,
            warmup_frames=30,
            l515_depth_config=L515DepthConfig(
                enabled=True,
                load_json_before_stream=False,
                visual_preset=5,
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

        # Capture several already-aligned RGBD frames. This is a static quality
        # diagnostic; temporal median is not recommended during fast motion.
        print(f"\nCapturing {TEMPORAL_MEDIAN_FRAMES} aligned RGBD frame(s)...")
        frames = [camera.read() for _ in range(TEMPORAL_MEDIAN_FRAMES)]
        frame = frames[-1]

        if frame.rgb is None:
            raise RuntimeError("RGB frame is unavailable.")
        if frame.align_mode != "depth_to_color":
            raise RuntimeError("This test requires depth_to_color alignment.")

        rgb = np.ascontiguousarray(frame.rgb)
        rgb_bgr = np.ascontiguousarray(rgb[..., ::-1])
        depth_single_m = np.ascontiguousarray(frame.depth, dtype=np.float32)
        depth_median_m = _median_valid_depth([item.depth for item in frames])
        K = np.asarray(frame.K, dtype=np.float64)

        print(f"  RGB:            shape={rgb.shape}, dtype={rgb.dtype}")
        print(
            "  Depth single:   "
            f"shape={depth_single_m.shape}, valid={int((depth_single_m > 0).sum())}/"
            f"{depth_single_m.size}"
        )
        print(
            "  Depth median:   "
            f"valid={int((depth_median_m > 0).sum())}/{depth_median_m.size}"
        )
        print(
            f"  Intrinsics:     {depth_single_m.shape[1]}x{depth_single_m.shape[0]} "
            f"fx={K[0, 0]:.1f} fy={K[1, 1]:.1f}"
        )

        # Conservative edge-ramp candidate.
        depth_filtered_candidate, removed_mask = remove_l515_mixed_edge_depth(
            depth_median_m,
            jump_threshold_m=EDGE_JUMP_THRESHOLD_M,
            surface_margin_m=EDGE_SURFACE_MARGIN_M,
            radius=EDGE_FILTER_RADIUS,
            min_valid_neighbors=EDGE_MIN_VALID_NEIGHBORS,
        )

        valid_before = int(np.count_nonzero(depth_median_m > 0))
        removed_count = int(np.count_nonzero(removed_mask))
        removed_ratio = removed_count / max(valid_before, 1)
        filter_accepted = removed_ratio <= EDGE_MAX_REMOVED_RATIO

        if filter_accepted:
            depth_for_pointcloud = depth_filtered_candidate
        else:
            depth_for_pointcloud = depth_median_m

        print(
            "  Mixed-edge filter: "
            f"removed {removed_count}/{valid_before} valid px "
            f"({100.0 * removed_ratio:.2f}%), "
            f"status={'ACCEPTED' if filter_accepted else 'REJECTED'}"
        )
        if not filter_accepted:
            print(
                "  The candidate exceeded the 8% safety limit; the point cloud "
                "will use median depth instead."
            )

        _show_rgbd_panels(
            rgb_bgr,
            depth_single_m,
            depth_median_m,
            depth_filtered_candidate,
            filter_accepted=filter_accepted,
        )

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

        if USE_RADIUS_OUTLIER_REMOVAL and len(pcd_ds.points) > 0:
            before = len(pcd_ds.points)
            candidate, _ = pcd_ds.remove_radius_outlier(
                nb_points=RADIUS_OUTLIER_MIN_NEIGHBORS,
                radius=RADIUS_OUTLIER_RADIUS_M,
            )
            removed_3d = before - len(candidate.points)
            removed_3d_ratio = removed_3d / max(before, 1)

            if removed_3d_ratio <= RADIUS_OUTLIER_MAX_REMOVED_RATIO:
                pcd_ds = candidate
                status = "ACCEPTED"
            else:
                status = "REJECTED"

            print(
                "  Radius outlier filter: "
                f"removed {removed_3d}/{before} points "
                f"({100.0 * removed_3d_ratio:.2f}%), status={status}"
            )

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