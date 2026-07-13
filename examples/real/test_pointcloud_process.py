#!/usr/bin/env python3
"""Test: capture a 640x480 RGBD frame → camera-frame point cloud → world-frame
point cloud → open3d visualization → desk plane fitting.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dexmani_real.config.camera_calib import CameraCalib


def _depth_intrinsics_ok(profile: rs.pipeline_profile) -> bool:
    """The L515 can get stuck in a state where depth intrinsics for VGA/XGA
    (640x480 / 1024x768) fail to load — only QVGA (320x240) survives — which
    then makes rs.align throw "intrinsics for resolution 640,480 doesn't exist".
    The calibration is intact; a hardware_reset() reloads it. Returns whether the
    started depth stream actually has intrinsics."""
    try:
        profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        return True
    except RuntimeError:
        return False


def _apply_l515_preset() -> None:
    """Apply the production L515 depth preset to the device BEFORE streaming,
    reusing sensor/realsense.py so the diagnostic matches production. On hosts
    whose kernel blocks the load_json/XU path, this falls back to the UVC-safe
    options (noise_filtering/confidence/min_distance) via set_option.
    Single-camera setup: uses the first enumerated device."""
    from dexmani_real.sensor.realsense import L515DepthConfig, apply_l515_depth_config

    devices = rs.context().query_devices()
    if len(devices) == 0:
        print("  L515 preset skipped: no device found.")
        return
    status = apply_l515_depth_config(devices[0], L515DepthConfig(), depth_resolution=(640, 480), fps=30)
    print(
        {
            "json": "  L515 preset applied via load_json (short-range).",
            "fallback": "  L515 preset: load_json blocked by kernel UVC; applied UVC-safe subset "
            "(noise_filtering/confidence/min_distance) via set_option.",
            "failed": "  L515 preset could not be applied — using firmware defaults.",
            "disabled": "  L515 preset disabled.",
        }.get(status, f"  L515 preset status: {status}")
    )


def main() -> None:
    # ── 1. Connect to RealSense ───────────────────────────────────────────────
    print("Connecting to RealSense...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Apply the production L515 depth preset (short-range visual preset, confidence
    # threshold, noise filtering, min-distance) at the firmware source — attenuates
    # edge flying-line *generation*. Must load on the device BEFORE streaming: the
    # preset's sensor-mode/stream keys can't change mid-stream.
    USE_L515_PRESET = True
    if USE_L515_PRESET:
        _apply_l515_preset()

    # If the L515 is stuck without VGA depth intrinsics, rs.align later throws.
    # Detect it right after start and recover with a one-shot hardware_reset.
    profile = pipeline.start(config)
    if not _depth_intrinsics_ok(profile):
        print("  Depth intrinsics missing (L515 bad state) — hardware_reset() and retry...")
        profile.get_device().hardware_reset()
        time.sleep(12)
        if USE_L515_PRESET:
            _apply_l515_preset()  # reset reloaded firmware defaults
        pipeline = rs.pipeline()
        profile = pipeline.start(config)
        if not _depth_intrinsics_ok(profile):
            raise RuntimeError("L515 depth intrinsics still missing after hardware_reset.")

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

    # ── 2. Capture one aligned RGBD frame ─────────────────────────────────────
    print("\nCapturing one aligned RGBD frame...")
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    aligned_depth_frame = aligned.get_depth_frame()

    # --- Diagnose raw depth (before alignment) ---
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
    print(f"  Depth: {depth.shape} {depth.dtype}  nonzero={(depth > 0).sum()}")

    # ── 3. Camera-frame point cloud ───────────────────────────────────────────
    print("\nGenerating camera-frame point cloud...")
    depth_m = depth.astype(np.float64) * depth_scale  # raw uint16 → meters

    # Remove ToF flying pixels at depth edges: deletes the 1-3px "ramp" that a
    # LiDAR spot straddling an object boundary paints between foreground and
    # background. Conservative — only sets pixels to 0, never fills/interpolates.
    #
    # gap=3      : samples each side from k = gap+1, skipping the ramp body so it
    #              no longer contaminates the clean-surface mean/std (this is the
    #              fix for the old k=1 sampling ceiling); pair with sample_radius=9
    #              so enough clean samples remain past the dead-zone.
    # adaptive   : with beta = 1/fx, edge/noise/margin become dimensionless
    #              multipliers of the local lateral spacing d·β (η = k·d·β), so one
    #              setting covers 0.3–2.5 m (≈4mm @0.5m … ≈19mm @2.5m). Set
    #              USE_ADAPTIVE_THRESHOLDS=False for fixed absolute-meter thresholds.
    # edge gate  : USE_EDGE_GATE requires an aligned-color edge to co-occur with the
    #              depth-gradient candidate (experimental; assumes aligned depth).
    USE_FLYING_PIXEL_REMOVAL = True
    USE_ADAPTIVE_THRESHOLDS = True  # η = k·d·β (P1-4); False → fixed absolute meters
    USE_EDGE_GATE = False  # experimental color-edge AND-gate (P1-5)
    if USE_FLYING_PIXEL_REMOVAL:
        import cv2

        from dexmani_real.utils import remove_flying_pixels_at_edges

        # Optional edge gate: keep a candidate only where the aligned color image
        # also shows an edge — an independent second cue against false deletion.
        edge_gate = None
        if USE_EDGE_GATE:
            gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
            sx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
            sy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
            color_edges = np.sqrt(sx**2 + sy**2) > 60.0
            edge_gate = cv2.dilate(color_edges.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

        n_before = int((depth_m > 0).sum())
        if USE_ADAPTIVE_THRESHOLDS:
            depth_m = remove_flying_pixels_at_edges(
                depth_m.astype(np.float32),
                edge_threshold=4.0,  # ×(d·β) multipliers, not meters
                noise_threshold=10.0,
                margin=6.0,
                sample_radius=9,
                gap=3,
                beta=1.0 / float(K[0, 0]),
                edge_gate_mask=edge_gate,
            ).astype(np.float64)
        else:
            depth_m = remove_flying_pixels_at_edges(
                depth_m.astype(np.float32),
                edge_threshold=0.008,
                noise_threshold=0.008,  # tightened vs old 0.015 now the ramp is skipped
                margin=0.009,
                sample_radius=9,
                gap=3,
                edge_gate_mask=edge_gate,
            ).astype(np.float64)
        n_after = int((depth_m > 0).sum())
        print(f"  Flying-pixel removal (adaptive={USE_ADAPTIVE_THRESHOLDS}, gate={USE_EDGE_GATE}): "
              f"{n_before} → {n_after} valid depth px (removed {n_before - n_after} edge flyers)")

    # ── 3.5 RGBD 2D visualization (RGB | depth raw | depth filtered) ──────────
    # Side-by-side so the flying-pixel removal is visible in 2D: the raw depth
    # still carries the edge ramps; the filtered depth has them zeroed (black).
    # Colormap spans the workspace range [0.3, 2.5] m.
    USE_RGBD_VIS = True
    if USE_RGBD_VIS:
        import cv2

        from dexmani_real.utils.pointcloud_utils import make_depth_vis

        raw_depth_m = depth.astype(np.float64) * depth_scale  # pre-filter meters
        panels = [
            ("RGB", rgb_bgr),
            ("Depth raw", make_depth_vis(raw_depth_m, 0.3, 2.5)),
            ("Depth filtered", make_depth_vis(depth_m, 0.3, 2.5)),
        ]
        labeled = []
        for title, img in panels:
            img = img.copy()
            cv2.putText(img, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            labeled.append(img)
        try:
            print("\nShowing RGBD window (RGB | depth raw | depth filtered) — press any key to continue...")
            cv2.imshow("RGBD: RGB | depth raw | depth filtered", np.hstack(labeled))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as e:
            print(f"  RGBD window skipped (no display?): {e}")

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
        & (z_cam > 0.3)
        & (z_cam < 2.5)
    )
    pts_cam = points_flat[valid]
    col_cam = colors_flat[valid]
    print(f"  Valid depth range [0.3, 2.5]m: {valid.sum()} / {valid.size} points")
    print(f"  Camera-frame points: {pts_cam.shape[0]}")

    if pts_cam.shape[0] == 0:
        print("\n  ❌ Still no valid points. Raw depth diagnostics:")
        print(f"     depth_scale = {depth_scale}")
        print(f"     depth min/max/mean = {depth.min()}/{depth.max()}/{depth.mean():.1f}")
        z_min, z_max = z_cam.min(), z_cam.max()
        print(f"     depth_m min/max = {z_min:.4f}/{z_max:.4f} m")
        n_neg = (z_cam <= 0).sum()
        n_lo = ((z_cam > 0) & (z_cam <= 0.3)).sum()
        n_ok = ((z_cam > 0.3) & (z_cam < 2.5)).sum()
        n_hi = (z_cam >= 2.5).sum()
        print(f"     z<=0: {n_neg}, 0<z<=0.3: {n_lo}, 0.3<z<2.5: {n_ok}, z>=2.5: {n_hi}")
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

    # ── 4.1 Spatial crop (X/Y workspace + loose Z guard) ─────────────────────
    # X/Y limits the cloud to the workspace footprint so the desk (not the floor
    # or a wall) is the dominant plane for RANSAC. The Z here is only a loose
    # guard against gross outliers (ceiling / deep background) — intentionally
    # wide so it does NOT pre-assume the desk height; the tight desk-anchored Z
    # crop happens in §6.1 AFTER the plane is fit.
    x_min, x_max = 0.0, 0.8
    y_min, y_max = -0.6, 0.6
    z_guard_lo, z_guard_hi = -0.2, 0.8
    crop_mask = (
        (pts_world[:, 0] >= x_min) & (pts_world[:, 0] <= x_max)
        & (pts_world[:, 1] >= y_min) & (pts_world[:, 1] <= y_max)
        & (pts_world[:, 2] >= z_guard_lo) & (pts_world[:, 2] <= z_guard_hi)
    )
    pts_world = pts_world[crop_mask]
    col_cam = col_cam[crop_mask]
    print(f"  After X/Y crop (x∈[{x_min},{x_max}], y∈[{y_min},{y_max}], "
          f"z guard∈[{z_guard_lo},{z_guard_hi}]): {pts_world.shape[0]} points")

    # ── 5. Build point cloud + clean residual flyers ─────────────────────────
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(col_cam.astype(np.float64))

    # ── 5.1 Remove residual flying pixels / edge tails ────────────────────────
    # The 2D depth-edge filter misses flying lines that are structured in the
    # depth map but sparse in 3D. In the point cloud those bridging points are
    # exactly radius-outliers: a surface point has dozens–hundreds of neighbours
    # within 1cm (full-res spacing ≈ depth/fx ≈ 1mm @0.6m … 3mm @2m), a flying
    # line point has far fewer. remove_radius_outlier is the targeted tool.
    USE_OUTLIER_REMOVAL = False
    if USE_OUTLIER_REMOVAL and len(pcd.points) > 0:
        n_before = len(pcd.points)
        pcd, _ = pcd.remove_radius_outlier(nb_points=40, radius=0.01)
        print(f"  Radius outlier removal: {n_before} → {len(pcd.points)} points "
              f"(removed {n_before - len(pcd.points)} flying-line points)")

    # ── 6. Fit desk plane (RANSAC) — before the tight Z crop ──────────────────
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

    # ── 6.1 Tight Z crop anchored at the measured desk plane ──────────────────
    # Now that the desk height is known, keep a band from just below the desk
    # surface up to workspace height — no dependence on a pre-assumed desk Z.
    Z_BELOW_DESK, Z_ABOVE_DESK = 0.03, 0.6
    z_lo, z_hi = desk_z_mean - Z_BELOW_DESK, desk_z_mean + Z_ABOVE_DESK
    pts_all = np.asarray(pcd.points)
    col_all = np.asarray(pcd.colors)
    zmask = (pts_all[:, 2] >= z_lo) & (pts_all[:, 2] <= z_hi)
    pcd.points = o3d.utility.Vector3dVector(pts_all[zmask])
    pcd.colors = o3d.utility.Vector3dVector(col_all[zmask])
    print(f"  After desk-anchored Z crop (z∈[{z_lo:.3f},{z_hi:.3f}]): {int(zmask.sum())} points")

    # ── 7. Downsample + visualize ─────────────────────────────────────────────
    pcd_ds = pcd.voxel_down_sample(voxel_size=0.003)
    print(f"  Downsampled: {len(pcd_ds.points)} points (3mm voxel)")

    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    camera_frame.transform(T_world_camera)

    print("\nLaunching open3d visualizer (close window to continue)...")
    print("  Window 1: world-frame point cloud + coordinate frames")
    o3d.visualization.draw_geometries(
        [pcd_ds, world_frame, camera_frame],
        window_name="World-frame Point Cloud",
        point_show_normal=False,
    )

    # ── 8. Color-coded desk segmentation on downsampled pcd ───────────────────
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
