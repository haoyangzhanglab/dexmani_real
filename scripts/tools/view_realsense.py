#!/usr/bin/env python3
"""实时查看 RealSense RGB-D/点云，退出时恢复原曝光优先级。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.config.real import load_config
from dexmani_real.config.perception import PointCloudConfig
from dexmani_real.geometry.camera import CameraIntrinsics
from dexmani_real.geometry.plane import fit_table_plane
from dexmani_real.perception.pointcloud import (
    DEPTH_RANGE_M,
    PointCloudProcessor,
    aligned_depth_points_in_base,
    build_raw_point_cloud,
)
from dexmani_real.sensor.realsense import get_ae_priority, set_ae_priority
from dexmani_real.utils.camera_calibrate.extrinsics import load_camera_extrinsics
from dexmani_real.utils.log import configure_logging
from dexmani_real.utils.table_plane import load_table_plane, resolve_table_plane_path, save_table_plane
from dexmani_real.utils.visualization.image import ImageViewer, rgbd_panel
from dexmani_real.utils.visualization.pointcloud import PointCloudViewer

if TYPE_CHECKING:
    from dexmani_real.sensor.realsense import RealSenseCamera

FPS_WINDOW = 100
TABLE_CALIBRATION_FRAMES = 5
logger = logging.getLogger(__name__)


def depth_valid_ratio(depth: np.ndarray, min_d: float, max_d: float) -> float:
    """Return the fraction of depth pixels inside the display band."""
    depth_f32 = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth_f32) & (depth_f32 > 0.0) & (depth_f32 >= min_d) & (depth_f32 <= max_d)
    return float(valid.mean()) if depth_f32.size else 0.0


def read_rgbd(camera):
    """检查 SDK 帧的网格和类型，保留设备侧来源时间与帧号。"""
    frame = camera.read()
    if frame is None:
        return None
    shape = (camera.color_intrinsics.height, camera.color_intrinsics.width)
    if frame.rgb.dtype != np.uint8 or frame.rgb.shape != (*shape, 3):
        raise ValueError("camera RGB must be uint8 [H,W,3] matching color intrinsics")
    if frame.depth.dtype != np.uint16 or frame.depth.shape != shape:
        raise ValueError("aligned camera depth must be uint16 [H,W] matching color intrinsics")
    return frame


def toggle_ae_priority(camera: RealSenseCamera) -> None:
    """Toggle auto_exposure_priority; OFF caps exposure at the requested frame period."""
    current = get_ae_priority(camera)
    if current is None:
        print("  auto_exposure_priority: not supported on this color sensor")
        return
    target = 0.0 if current >= 0.5 else 1.0
    readback = set_ae_priority(camera, target)
    if readback is None:
        print("  auto_exposure_priority: set failed (unsupported)")
        return
    label = "OFF (cap exposure at frame period)" if target == 0.0 else "ON (may reduce FPS)"
    print(f"  auto_exposure_priority: {current:g} -> {readback:g}  [{label}]")


def save_diagnostic_snapshot(
    *,
    output_root: Path,
    frame,
    raw_point_cloud: np.ndarray,
    processed_point_cloud: np.ndarray,
    depth_scale_m: float,
    intrinsics: CameraIntrinsics,
    T_xarm_base_from_color: np.ndarray,
    table_plane_abcd: tuple[float, float, float, float] | None,
    table_plane_source: str,
    pointcloud_config: PointCloudConfig,
    camera_info: dict,
) -> Path:
    timestamp = datetime.now().astimezone()
    target = output_root.expanduser().resolve() / timestamp.strftime("pointcloud_%Y%m%d_%H%M%S_%f")
    target.mkdir(parents=True)

    arrays = {
        "rgb.npy": frame.rgb,
        "depth_aligned_to_color_raw.npy": frame.depth,
        "raw_point_cloud.npy": raw_point_cloud,
        "processed_point_cloud.npy": processed_point_cloud,
        "T_xarm_base_from_color.npy": T_xarm_base_from_color,
    }
    metadata = {
        "created_at": timestamp.isoformat(),
        "source_timestamp_ns": int(frame.timestamp_ns),
        "depth_frame_number": int(frame.depth_frame_number),
        "color_frame_number": int(frame.color_frame_number),
        "source_clock": "host_monotonic_oldest_rgbd_receive",
        "payload_mode": "depth_to_color_aligned_rgbd",
        "depth_payload_frame": "color",
        "depth_scale_m_per_unit": float(depth_scale_m),
        "color_intrinsics": intrinsics.to_dict(),
        "table_plane_abcd": None if table_plane_abcd is None else list(table_plane_abcd),
        "table_plane_source": table_plane_source,
        "pointcloud_config": asdict(pointcloud_config),
        "point_cloud_frame": "xarm_base",
        "point_cloud_columns": ["x_m", "y_m", "z_m", "r", "g", "b"],
        "raw_point_cloud_semantics": "all_finite_nonzero_aligned_depth_pixels",
        "processed_point_cloud_empty": processed_point_cloud.shape[0] == 0,
        "camera": {key: camera_info[key] for key in ("name", "serial", "firmware")},
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        },
    }

    for name, value in arrays.items():
        np.save(target / name, value, allow_pickle=False)
    with (target / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
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
    lower = np.asarray(config.workspace[:3], dtype=np.float32)
    upper = np.asarray(config.workspace[3:], dtype=np.float32)
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
        print("  Calibration file unchanged; using the new fit for this run only.")
    return fit.plane_abcd


class RealSenseViewer:
    """Own live preview state and restore the captured exposure priority on exit."""

    def __init__(self, args):
        self.args = args

    def run(self) -> int:
        # Resolve experiment settings before enumerating or opening a camera.
        runtime = load_config(path=self.args.config)
        if self.args.save_dir is not None and self.args.save_dir.exists() and not self.args.save_dir.is_dir():
            raise NotADirectoryError(f"snapshot output root is not a directory: {self.args.save_dir}")
        self.pointcloud_config = runtime.pointcloud
        self.table_plane_path = runtime.environment.table_plane_path
        self.base_from_color = self.processor = self.table_plane = None
        self.table_plane_source = "disabled"
        self.fitted_plane = None

        from dexmani_real.sensor.realsense import RealSenseCamera

        # Leave exposure priority untouched until it is snapshotted.
        config = runtime.camera
        config.validate()
        configure_logging()

        self.camera = RealSenseCamera(config, auto_exposure_priority=None)
        original_priority = None
        failure = None
        try:
            self.camera.connect()
            self.camera_info = self.camera.get_device_info()
            print(f"Connected: {self.camera_info}")
            original_priority = get_ae_priority(self.camera)
            if original_priority is not None:
                print(f"  auto_exposure_priority (original): {original_priority:g}")
                if original_priority != 0.0:
                    applied = set_ae_priority(self.camera, 0.0)
                    if applied is None or not np.isclose(applied, 0.0):
                        raise RuntimeError("could not apply diagnostic exposure priority")
            else:
                print("  auto_exposure_priority unavailable; leaving it unchanged")
            if self.args.calibrate_table:
                self.ensure_cloud_transform()
                self.fitted_plane = calibrate_table_plane(
                    camera=self.camera, intrinsics=self.camera.color_intrinsics,
                    T_xarm_base_from_color=self.base_from_color, config=self.pointcloud_config,
                    plane_path=resolve_table_plane_path(self.table_plane_path),
                    frame_count=TABLE_CALIBRATION_FRAMES,
                )
            if self.args.benchmark_frames:
                benchmark_production_pipeline(
                    camera=self.camera, processor=self.ensure_processor(),
                    frame_count=self.args.benchmark_frames,
                )
            self.capture(allow_exposure_changes=original_priority is not None)
        except BaseException as exc:
            failure = exc
        finally:
            try:
                if original_priority is not None:
                    try:
                        restored = set_ae_priority(self.camera, original_priority)
                        if restored is None or not np.isclose(restored, original_priority):
                            raise RuntimeError(f"exposure priority readback mismatch: {restored}")
                        print(f"  [restore] auto_exposure_priority -> {restored:g}")
                    except BaseException as exc:
                        logger.exception("exposure priority restore failed")
                        failure = failure or exc
            finally:
                try:
                    self.camera.disconnect()
                except BaseException as exc:
                    logger.exception("camera disconnect failed")
                    failure = failure or exc
        if failure is not None:
            raise failure
        return 0

    def ensure_cloud_transform(self):
        if self.base_from_color is None:
            self.base_from_color = load_camera_extrinsics(
                self.camera.active_serial, self.args.camera_calibration,
            )

    def ensure_processor(self):
        if self.processor is None:
            self.ensure_cloud_transform()
            if self.pointcloud_config.remove_table:
                self.table_plane = self.fitted_plane if self.args.calibrate_table else load_table_plane(self.table_plane_path)
                self.table_plane_source = "calibrated_this_run" if self.args.calibrate_table else "resolved_runtime"
            self.processor = PointCloudProcessor(
                depth_scale_m=self.camera.depth_scale_m, intrinsics=self.camera.color_intrinsics,
                T_xarm_base_from_color=self.base_from_color, table_plane_abcd=self.table_plane,
                config=self.pointcloud_config,
            )
        return self.processor

    def capture(self, *, allow_exposure_changes: bool) -> None:
        print("\nQ/Esc=退出 P=点云 S=raw/processed F=冻结 R=重置 C=配置 D=配色 A=曝光优先级")
        print("快捷键在诊断窗口内生效；关闭 RGB-D 退出，关闭点云窗口仅关闭点云。")
        intervals: deque[float] = deque(maxlen=FPS_WINDOW)
        previous_frame_start = None
        frame_count = 0
        enabled, frozen = self.args.cloud is not None, False
        mode, colormap = self.args.cloud or "processed", "jet"
        cached_cloud = np.empty((0, 6), dtype=np.float32)
        image_viewer = cloud_viewer = None
        intrinsics = self.camera.color_intrinsics
        depth_scale_m = self.camera.depth_scale_m
        snapshot_pending = self.args.save_dir is not None
        try:
            image_viewer = ImageViewer("RealSense RGB-D | RGB / Depth")
            while True:
                events = image_viewer.poll_events()
                if "closed" in events:
                    return
                if cloud_viewer is not None:
                    cloud_events = cloud_viewer.poll_events()
                    if "closed" in cloud_events:
                        cloud_viewer.close()
                        cloud_viewer = None
                        enabled, frozen = False, False
                        cached_cloud = np.empty((0, 6), dtype=np.float32)
                    events.extend(key for key in cloud_events if key != "closed")
                for key in events:
                    if key in {"q", "esc"}:
                        return
                    if key == "a" and allow_exposure_changes:
                        toggle_ae_priority(self.camera)
                    elif key == "a":
                        print("Exposure changes disabled: priority is unsupported")
                    elif key == "c":
                        print(f"Point cloud: enabled={enabled} mode={mode} config={self.pointcloud_config}")
                    elif key == "d":
                        colormap = "gray" if colormap == "jet" else "jet"
                    elif key == "f":
                        if enabled and len(cached_cloud):
                            frozen = not frozen
                        else:
                            print("Freeze unavailable until a nonempty point cloud is displayed")
                    elif key in {"p", "s", "r"}:
                        frozen = False
                        cached_cloud = np.empty((0, 6), dtype=np.float32)
                        if key == "p":
                            enabled = not enabled
                        elif key == "s":
                            mode = "raw" if mode == "processed" else "processed"
                        else:
                            enabled, mode, colormap = False, "processed", "jet"
                        if cloud_viewer is not None:
                            if enabled:
                                cloud_viewer.update(cached_cloud)
                            else:
                                cloud_viewer.close()
                                cloud_viewer = None

                started = time.perf_counter()
                frame = read_rgbd(self.camera)
                if frame is None:
                    continue
                read_ms = (time.perf_counter() - started) * 1000
                if previous_frame_start is not None:
                    intervals.append(started - previous_frame_start)
                previous_frame_start = started
                frame_count += 1
                rgb, depth = frame.rgb, frame.depth
                if snapshot_pending:
                    processed = self.ensure_processor().process(depth, rgb)
                    raw = build_raw_point_cloud(
                        depth_raw=depth, color=rgb, depth_scale_m=depth_scale_m,
                        intrinsics=intrinsics, T_xarm_base_from_color=self.base_from_color,
                    )
                    target = save_diagnostic_snapshot(
                        output_root=self.args.save_dir, frame=frame,
                        raw_point_cloud=raw, processed_point_cloud=processed,
                        depth_scale_m=depth_scale_m, intrinsics=intrinsics,
                        T_xarm_base_from_color=self.base_from_color,
                        table_plane_abcd=self.table_plane, table_plane_source=self.table_plane_source,
                        pointcloud_config=self.pointcloud_config, camera_info=self.camera_info,
                    )
                    print(f"Snapshot: {target}")
                    snapshot_pending = False
                pcd_ms = 0.0
                if enabled and not frozen:
                    self.ensure_cloud_transform()
                    if cloud_viewer is None:
                        cloud_viewer = PointCloudViewer(
                            workspace=self.pointcloud_config.workspace,
                            base_from_color=self.base_from_color, keys="psfrcda",
                        )
                    started = time.perf_counter()
                    if mode == "processed":
                        cached_cloud = self.ensure_processor().process(depth, rgb)
                    else:
                        cached_cloud = build_raw_point_cloud(
                            depth_raw=depth, color=rgb, depth_scale_m=depth_scale_m,
                            intrinsics=intrinsics, T_xarm_base_from_color=self.base_from_color,
                        )
                    pcd_ms = (time.perf_counter() - started) * 1000
                    cloud_viewer.update(cached_cloud)
                depth_m = depth.astype(np.float32) * depth_scale_m
                fps = len(intervals) / sum(intervals) if intervals else 0.0
                valid = depth_valid_ratio(depth_m, *DEPTH_RANGE_M)
                point_count = len(cached_cloud)
                lines = [
                    f"frame={frame_count} fps={fps:.1f} read={read_ms:.1f}ms pcd={pcd_ms:.1f}ms",
                    f"valid={valid:.3f} points={point_count} PCD={mode if enabled else 'OFF'} frozen={frozen}",
                    f"K=[{intrinsics.fx:.1f},{intrinsics.fy:.1f},{intrinsics.ppx:.1f},{intrinsics.ppy:.1f}]",
                    f"depth={DEPTH_RANGE_M}m cmap={colormap}",
                    "P=cloud S=raw/processed F=freeze D=colormap R=reset C=config A=AE Q/Esc=quit",
                ]
                image_viewer.show(rgbd_panel(
                    rgb, depth_m, depth_range_m=DEPTH_RANGE_M, colormap=colormap, lines=lines,
                ))
        finally:
            print(f"Captured {frame_count} frames")
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
    parser.add_argument("--calibrate-table", action="store_true", help="预览前采集五帧拟合桌面；终端确认后才发布")
    parser.add_argument("--save-dir", type=Path, help="保存首个预览帧的 RGB-D、raw/processed 点云与来源事实")
    parser.add_argument("--benchmark-frames", type=int, default=0, metavar="N", help="显式测量 N 个新深度帧；默认不运行")
    args = parser.parse_args(argv)
    if args.benchmark_frames < 0:
        parser.error("--benchmark-frames 必须非负")
    return RealSenseViewer(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
