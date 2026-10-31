#!/usr/bin/env python3
"""预览 RealSense RGB-D 和固定类型点云，或执行一次快照、测量、桌面拟合。"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.config.perception import DEPTH_RANGE_M
from dexmani_real.geometry.camera import CameraIntrinsics
from dexmani_real.utils.log import configure_logging

if TYPE_CHECKING:
    from dexmani_real.config.perception import PointCloudConfig
    from dexmani_real.perception.pointcloud import PointCloudProcessor
    from dexmani_real.sensor.realsense import RealSenseCamera

TABLE_CALIBRATION_FRAMES = 5
logger = logging.getLogger(__name__)


def read_rgbd(camera):
    """检查 SDK 帧的网格和类型，保留首次主机接收时间与设备帧号。"""
    frame = camera.read()
    if frame is None:
        return None
    shape = (camera.color_intrinsics.height, camera.color_intrinsics.width)
    if frame.rgb.dtype != np.uint8 or frame.rgb.shape != (*shape, 3):
        raise ValueError("camera RGB must be uint8 [H,W,3] matching color intrinsics")
    if frame.depth.dtype != np.uint16 or frame.depth.shape != shape:
        raise ValueError("aligned camera depth must be uint16 [H,W] matching color intrinsics")
    return frame


def save_rgbd_snapshot(
    *,
    output_root: Path,
    frame,
    depth_scale_m: float,
    intrinsics: CameraIntrinsics,
    serial: str,
) -> Path:
    """仅保存原始对齐 RGB-D 和来源事实；原始深度计数乘 scale 后为米。"""
    timestamp = datetime.now().astimezone()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / timestamp.strftime("rgbd_%Y%m%d_%H%M%S_%f.npz")
    stream = target.open("xb")
    try:
        try:
            np.savez_compressed(
                stream,
                rgb=frame.rgb,
                depth_aligned_to_color_raw=frame.depth,
                K=intrinsics.matrix(),
                distortion_model=intrinsics.distortion_model,
                distortion_coeffs=np.asarray(intrinsics.distortion_coeffs, dtype=np.float64),
                depth_scale_m_per_unit=float(depth_scale_m),
                camera_serial=serial,
                source_timestamp_ns=int(frame.timestamp_ns),
                source_clock="host_monotonic_oldest_rgbd_receive",
                depth_frame_number=int(frame.depth_frame_number),
                color_frame_number=int(frame.color_frame_number),
            )
        except BaseException:
            try:
                stream.close()
            except BaseException:
                logger.exception("RGB-D snapshot close also failed: %s", target)
            raise
        else:
            stream.close()
    except BaseException:
        try:
            target.unlink()
        except BaseException:
            logger.exception("failed to remove incomplete RGB-D snapshot: %s", target)
        raise
    return target


def benchmark_production_pipeline(
    *,
    camera: RealSenseCamera,
    processor: PointCloudProcessor,
    frame_count: int = 20,
) -> None:
    elapsed_ms: list[float] = []
    end_to_end_ms: list[float] = []
    valid_clouds = 0
    last_depth_frame_number = None
    while len(elapsed_ms) < frame_count:
        frame_started = time.perf_counter()
        frame = read_rgbd(camera)
        if frame is None or frame.depth_frame_number == last_depth_frame_number:
            continue
        last_depth_frame_number = frame.depth_frame_number
        started = time.perf_counter()
        cloud = processor.process(frame.depth, frame.rgb)
        elapsed_ms.append((time.perf_counter() - started) * 1000.0)
        end_to_end_ms.append((time.perf_counter() - frame_started) * 1000.0)
        valid_clouds += len(cloud) > 0
    values = np.asarray(elapsed_ms, dtype=np.float64)
    end_to_end_values = np.asarray(end_to_end_ms, dtype=np.float64)
    print(
        f"  Fresh-frame benchmark: n={frame_count} "
        f"p50={np.percentile(values, 50):.1f} ms "
        f"p95={np.percentile(values, 95):.1f} ms; valid clouds={valid_clouds}/{frame_count}"
    )
    print(
        f"  Capture-to-cloud benchmark: p50={np.percentile(end_to_end_values, 50):.1f} ms "
        f"p95={np.percentile(end_to_end_values, 95):.1f} ms"
    )


def calibrate_table_plane(
    *,
    camera: RealSenseCamera,
    intrinsics: CameraIntrinsics,
    T_xarm_base_from_color: np.ndarray,
    config: PointCloudConfig,
    plane_path: Path,
    frame_count: int,
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
    while len(depth_frames) < frame_count:
        frame = read_rgbd(camera)
        if frame is None or frame.depth_frame_number == last_depth_frame_number:
            continue
        last_depth_frame_number = frame.depth_frame_number
        depth_frames.append(frame.depth)

    point_batches: list[np.ndarray] = []
    for depth_raw in depth_frames:
        points = aligned_depth_points_in_base(
            depth_raw=depth_raw,
            depth_scale_m=camera.depth_scale_m,
            aligned_depth_intrinsics=intrinsics,
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


class RealSenseViewer:
    """一次相机连接用于预览或一个显式的一次性操作。"""

    def __init__(self, args):
        self.args = args

    def run(self) -> int:
        from dexmani_real.config.real import load_config
        from dexmani_real.sensor.realsense import RealSenseCamera

        runtime = load_config(path=self.args.config)
        if self.args.save_dir is not None and self.args.save_dir.exists() and not self.args.save_dir.is_dir():
            raise NotADirectoryError(f"snapshot output root is not a directory: {self.args.save_dir}")
        self.pointcloud_config = runtime.pointcloud
        self.table_plane_path = runtime.environment.table_plane_path
        self.base_from_color: np.ndarray | None = None
        self.processor: PointCloudProcessor | None = None
        runtime.camera.validate()
        configure_logging()
        # 预览不额外修改曝光优先级；其余启动设置使用实验配置。
        self.camera = RealSenseCamera(runtime.camera, auto_exposure_priority=None)
        try:
            self.camera.connect()
            print(f"Connected: {self.camera.active_serial}")
            if self.args.calibrate_table:
                from dexmani_real.utils.table_plane import resolve_table_plane_path

                calibrate_table_plane(
                    camera=self.camera, intrinsics=self.camera.color_intrinsics,
                    T_xarm_base_from_color=self.ensure_cloud_transform(), config=self.pointcloud_config,
                    plane_path=resolve_table_plane_path(self.table_plane_path),
                    frame_count=TABLE_CALIBRATION_FRAMES,
                )
            elif self.args.benchmark_frames is not None:
                benchmark_production_pipeline(
                    camera=self.camera, processor=self.ensure_processor(),
                    frame_count=self.args.benchmark_frames,
                )
            elif self.args.save_dir is not None:
                self.save_snapshot()
            else:
                self.capture()
        finally:
            original_error = sys.exc_info()[1]
            try:
                self.camera.disconnect()
            except BaseException:
                if original_error is None:
                    raise
                logger.exception("camera disconnect failed")
        return 0

    def ensure_cloud_transform(self) -> np.ndarray:
        if self.base_from_color is None:
            from dexmani_real.utils.camera_calibrate.extrinsics import (
                load_camera_extrinsics,
            )

            self.base_from_color = load_camera_extrinsics(
                self.camera.active_serial, self.args.camera_calibration,
            )
        return self.base_from_color

    def ensure_processor(self):
        if self.processor is None:
            from dexmani_real.perception.pointcloud import PointCloudProcessor

            base_from_color = self.ensure_cloud_transform()
            table_plane = None
            if self.pointcloud_config.remove_table:
                from dexmani_real.utils.table_plane import load_table_plane

                table_plane = load_table_plane(self.table_plane_path)
            self.processor = PointCloudProcessor(
                depth_scale_m=self.camera.depth_scale_m, intrinsics=self.camera.color_intrinsics,
                T_xarm_base_from_color=base_from_color, table_plane_abcd=table_plane,
                config=self.pointcloud_config,
            )
        return self.processor

    def save_snapshot(self):
        while (frame := read_rgbd(self.camera)) is None:
            pass
        target = save_rgbd_snapshot(
            output_root=self.args.save_dir, frame=frame,
            depth_scale_m=self.camera.depth_scale_m, intrinsics=self.camera.color_intrinsics,
            serial=self.camera.active_serial,
        )
        print(f"Snapshot: {target}")

    def capture(self) -> None:
        from dexmani_real.utils.visualization.image import ImageViewer, rgbd_panel

        print("Q/Esc 或关闭任一窗口退出；点云类型由 --cloud raw/processed 选择。")
        image_viewer = cloud_viewer = None
        depth_scale_m = self.camera.depth_scale_m
        try:
            image_viewer = ImageViewer("RealSense RGB-D | RGB / Depth")
            if self.args.cloud is not None:
                from dexmani_real.utils.visualization.pointcloud import PointCloudViewer

                self.pointcloud_config.workspace_bounds()
                cloud_viewer = PointCloudViewer(
                    workspace=self.pointcloud_config.workspace,
                    base_from_color=self.ensure_cloud_transform(),
                )
            while True:
                events = image_viewer.poll_events()
                if cloud_viewer is not None:
                    events.extend(cloud_viewer.poll_events())
                if any(key in {"q", "esc", "closed"} for key in events):
                    return
                frame = read_rgbd(self.camera)
                if frame is None:
                    continue
                if cloud_viewer is not None:
                    if self.args.cloud == "processed":
                        cloud = self.ensure_processor().process(frame.depth, frame.rgb)
                    else:
                        from dexmani_real.perception.pointcloud import (
                            build_raw_point_cloud,
                        )

                        cloud = build_raw_point_cloud(
                            depth_raw=frame.depth, color=frame.rgb, depth_scale_m=depth_scale_m,
                            intrinsics=self.camera.color_intrinsics,
                            T_xarm_base_from_color=self.ensure_cloud_transform(),
                        )
                    cloud_viewer.update(cloud)
                image_viewer.show(rgbd_panel(
                    frame.rgb, frame.depth.astype(np.float32) * depth_scale_m,
                    depth_range_m=DEPTH_RANGE_M,
                ))
        finally:
            original_error = sys.exc_info()[1]
            cleanup_error = None
            for resource in (cloud_viewer, image_viewer):
                if resource is not None:
                    try:
                        resource.close()
                    except BaseException as exc:
                        logger.exception("preview window close failed")
                        cleanup_error = cleanup_error or exc
            if original_error is None and cleanup_error is not None:
                raise cleanup_error


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Trusted Python experiment file exporting CONFIG: RealConfig")
    parser.add_argument("--camera-calibration")
    parser.add_argument("--cloud", choices=("raw", "processed"), help="启动时打开所选点云视图")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--calibrate-table", action="store_true", help="采集五帧拟合桌面，确认发布后退出")
    operation.add_argument("--save-dir", type=Path, help="保存一帧原始 RGB-D 与来源事实为单个 NPZ 后退出，无需外参或桌面标定")
    operation.add_argument("--benchmark-frames", type=int, metavar="N", help="测量 N 个新深度帧后退出")
    args = parser.parse_args(argv)
    if args.benchmark_frames is not None and args.benchmark_frames < 1:
        parser.error("--benchmark-frames 必须为正整数")
    return RealSenseViewer(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
