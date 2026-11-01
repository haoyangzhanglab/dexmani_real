#!/usr/bin/env python3
"""预览 RealSense RGB-D、可选拟合桌面，处理一帧点云后关闭相机静态查看。"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.utils.log import attach_cleanup_errors

if TYPE_CHECKING:
    from dexmani_real.config.perception import PointCloudConfig
    from dexmani_real.sensor.realsense import RealSenseCamera

logger = logging.getLogger(__name__)
VIS_DEPTH_RANGE_M = (0.3, 2.5)
TABLE_CALIBRATION_FRAMES = 5


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Trusted Python experiment file exporting CONFIG: RealConfig")
    parser.add_argument("--camera-calibration", type=Path)
    return parser.parse_args(argv)


def capture_frame(camera: RealSenseCamera):
    """等待相机读取边界校验过的对齐帧；停流错误由相机报告。"""
    while (frame := camera.read()) is None:
        pass
    return frame


def calibrate_table_plane(
    *,
    camera: RealSenseCamera,
    T_xarm_base_from_color: np.ndarray,
    config: PointCloudConfig,
    plane_path: Path,
) -> tuple[float, float, float, float]:
    from dexmani_real.geometry.plane import fit_table_plane
    from dexmani_real.perception.pointcloud import aligned_depth_points_in_base
    from dexmani_real.utils.table_plane import save_table_plane

    lower, upper = config.workspace_bounds()
    print("\n" + "=" * 60)
    print("Table Plane Calibration (multi-frame RANSAC)")
    print("=" * 60)
    print("  Clear movable objects from the visible table before continuing.")
    input("  Press Enter to capture calibration frames...")

    depth_frames: list[np.ndarray] = []
    last_depth_frame_number = None
    while len(depth_frames) < TABLE_CALIBRATION_FRAMES:
        frame = camera.read()
        if frame is None or frame.depth_frame_number == last_depth_frame_number:
            continue
        last_depth_frame_number = frame.depth_frame_number
        depth_frames.append(frame.depth)

    point_batches: list[np.ndarray] = []
    for depth_raw in depth_frames:
        points = aligned_depth_points_in_base(
            depth_raw=depth_raw,
            depth_scale_m=camera.depth_scale_m,
            aligned_depth_intrinsics=camera.color_intrinsics,
            T_xarm_base_from_color=T_xarm_base_from_color,
        )
        points = points[np.all((points >= lower) & (points <= upper), axis=1)]
        if points.shape[0] > 30_000:
            step = max(1, points.shape[0] // 30_000)
            points = points[::step][:30_000]
        point_batches.append(points)
    calibration_points = np.concatenate(point_batches, axis=0)
    fit = fit_table_plane(calibration_points)

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
        save_table_plane(plane_path, fit)
        print("  Published table calibration.")
    else:
        print("  Calibration file unchanged.")
    return fit.plane_abcd


def run_pointcloud_diagnostic(args):
    from dexmani_real.config.real import load_config
    from dexmani_real.perception.pointcloud import PointCloudProcessor
    from dexmani_real.sensor.realsense import RealSenseCamera
    from dexmani_real.utils.camera_calibrate.extrinsics import load_camera_extrinsics
    from dexmani_real.utils.log import configure_logging
    from dexmani_real.utils.table_plane import (
        load_table_plane,
        resolve_table_plane_path,
    )
    from dexmani_real.utils.visualization.image import ImageViewer, rgbd_panel

    fit_requested = input("Run table calibration? [y/N] ").strip().lower() in {"y", "yes"}
    runtime = load_config(path=args.config)
    runtime.camera.validate()
    config = runtime.pointcloud
    config.validate()
    table_plane = (
        load_table_plane(runtime.environment.table_plane_path)
        if config.remove_table and not fit_requested else None
    )
    configure_logging()
    camera = RealSenseCamera(runtime.camera, auto_exposure_priority=None)
    try:
        camera.connect()
        print(f"Connected: {camera.active_serial}")
        frame = capture_frame(camera)
        depth_m = frame.depth.astype(np.float32) * camera.depth_scale_m
        lower, upper = VIS_DEPTH_RANGE_M
        valid = np.isfinite(depth_m) & (depth_m >= lower) & (depth_m <= upper)
        print(f"Depth display range [{lower:.2f}, {upper:.2f}] m: {int(valid.sum())}/{depth_m.size} pixels")
        preview = ImageViewer("RGB-D diagnostic | Enter=continue Q/Esc=quit")
        try:
            preview.show(rgbd_panel(frame.rgb, depth_m, depth_range_m=VIS_DEPTH_RANGE_M))
            while True:
                events = preview.poll_events()
                if any(key in {"q", "esc", "closed"} for key in events):
                    return 0
                if "enter" in events:
                    break
                time.sleep(0.01)
        finally:
            original_error = sys.exc_info()[1]
            try:
                preview.close()
            except BaseException as exc:
                if original_error is None:
                    raise
                attach_cleanup_errors(original_error, [("image_viewer_close", exc)])
                logger.exception("RGB-D preview close also failed")

        intrinsics = camera.color_intrinsics
        base_from_color = load_camera_extrinsics(camera.active_serial, args.camera_calibration)
        if fit_requested:
            fitted = calibrate_table_plane(
                camera=camera, T_xarm_base_from_color=base_from_color,
                config=config, plane_path=resolve_table_plane_path(runtime.environment.table_plane_path),
            )
            table_plane = fitted if config.remove_table else None

        # 标定与预览后重新采集，用于点云处理与静态查看。
        frame = capture_frame(camera)
        processor = PointCloudProcessor(
            depth_scale_m=camera.depth_scale_m, intrinsics=intrinsics,
            T_xarm_base_from_color=base_from_color, config=config, table_plane_abcd=table_plane,
        )
        started = time.perf_counter()
        result = processor.process(frame.depth, frame.rgb)
        elapsed_ms = (time.perf_counter() - started) * 1000
        print(f"Point-cloud processing: {config}")
        print(f"Output: {len(result)} points ({elapsed_ms:.1f} ms)")
    finally:
        original_error = sys.exc_info()[1]
        try:
            camera.disconnect()
        except BaseException as exc:
            if original_error is None:
                raise
            attach_cleanup_errors(original_error, [("camera_disconnect", exc)])
            logger.exception("camera disconnect also failed")

    # 采集全部结束且相机成功关闭后才创建静态点云窗口。
    from dexmani_real.utils.visualization.pointcloud import PointCloudViewer

    viewer = PointCloudViewer(workspace=config.workspace, base_from_color=base_from_color)
    try:
        viewer.update(result)
        while not any(key in {"q", "esc", "closed"} for key in viewer.poll_events()):
            time.sleep(0.01)
    finally:
        original_error = sys.exc_info()[1]
        try:
            viewer.close()
        except BaseException as exc:
            if original_error is None:
                raise
            attach_cleanup_errors(original_error, [("cloud_viewer_close", exc)])
            logger.exception("point-cloud window close also failed")
    return 0


def main(argv=None):
    return run_pointcloud_diagnostic(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
