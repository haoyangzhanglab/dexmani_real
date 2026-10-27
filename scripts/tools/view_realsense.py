#!/usr/bin/env python3
"""实时查看 RealSense RGB-D/点云，退出时恢复原曝光优先级。"""

from __future__ import annotations

import argparse
import time
from collections import deque
from typing import TYPE_CHECKING, cast

import numpy as np

from dexmani_real.calibration.camera.extrinsics import load_camera_extrinsics
from dexmani_real.calibration.table_plane import load_table_plane
from dexmani_real.config import load_experiment_config
from dexmani_real.geometry.camera import validate_aligned_depth_distortion
from dexmani_real.perception.pointcloud import (
    DEPTH_RANGE_M,
    PointCloudProcessor,
    build_raw_point_cloud,
)
from dexmani_real.runtime.keyboard import Keyboard
from dexmani_real.utils.log import configure_logging
from dexmani_real.utils.visualization import (
    log_camera_geometry,
    log_points,
    log_rgbd,
    log_series,
    viewer_blueprint,
)

if TYPE_CHECKING:
    from dexmani_real.sensor.realsense import RealSenseCamera

FPS_WINDOW = 100


def depth_valid_ratio(depth: np.ndarray, min_d: float, max_d: float) -> float:
    """Return the fraction of depth pixels inside the display band."""
    depth_f32 = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth_f32) & (depth_f32 > 0.0) & (depth_f32 >= min_d) & (depth_f32 <= max_d)
    return float(valid.mean()) if depth_f32.size else 0.0


def list_cameras() -> list[dict[str, str]]:
    from dexmani_real.sensor.realsense import RealSenseCamera

    cameras = RealSenseCamera.list_cameras()
    if not cameras:
        print("No RealSense camera detected. Check USB connection.")
        return []
    print(f"Detected {len(cameras)} camera(s):")
    for i, cam in enumerate(cameras):
        print(
            f"  [{i}] {cam['name']:28s} "
            f"SN={cam['serial']:16s} "
            f"FW={cam['firmware']}  "
            f"PL={cam['product_line']}"
        )
    return cameras


def get_ae_priority(camera: RealSenseCamera) -> float | None:
    """Read auto_exposure_priority (0=OFF, 1=ON); None if unsupported."""
    import pyrealsense2 as rs

    sensor = camera.color_sensor
    option = rs.option.auto_exposure_priority
    if sensor.supports(option):
        return float(sensor.get_option(option))
    return None


def set_ae_priority(camera: RealSenseCamera, value: float) -> float | None:
    """Set auto_exposure_priority and return the readback; None if unsupported."""
    import pyrealsense2 as rs

    sensor = camera.color_sensor
    option = rs.option.auto_exposure_priority
    if not sensor.supports(option):
        return None
    sensor.set_option(option, float(value))
    return float(sensor.get_option(option))


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


class RealSenseViewer:
    """Own live preview state and restore the captured exposure priority on exit."""

    def __init__(self, args):
        self.args = args

    def run(self) -> int:
        # Resolve experiment settings before enumerating or opening a camera.
        runtime = load_experiment_config(yaml_path=self.args.config)
        self.pointcloud_config = runtime.pointcloud
        self.table = runtime.environment.table
        self.calibration: dict[str, np.ndarray] | None = None
        self.table_plane_abcd: tuple[float, float, float, float] | None = None

        import rerun as rr

        from dexmani_real.sensor.realsense import RealSenseCamera

        self.rr = rr
        # Leave exposure priority untouched until it is snapshotted.
        config = runtime.camera
        config.validate()
        configure_logging()

        print("=" * 60)
        print("RealSense Test -- RGB-D Live Capture + Real-time Point Cloud")
        print("=" * 60)
        print(f"NumPy        : {np.__version__}")
        print("pyrealsense2 : installed")

        cameras = list_cameras()
        if not cameras:
            return 1

        self.camera = RealSenseCamera(config, auto_exposure_priority=None)
        original_priority = None
        failed = False
        try:
            self.camera.connect()
            print(f"Connected: {self.camera.get_device_info()}")
            original_priority = get_ae_priority(self.camera)
            if original_priority is not None:
                print(f"  auto_exposure_priority (original): {original_priority:g}")
                if original_priority != 0.0:
                    applied = set_ae_priority(self.camera, 0.0)
                    if applied is None or not np.isclose(applied, 0.0):
                        raise RuntimeError("could not apply diagnostic exposure priority")
            else:
                print("  auto_exposure_priority unavailable; leaving it unchanged")
            self.capture(allow_exposure_changes=original_priority is not None)
        except (RuntimeError, OSError) as exc:
            failed = True
            print(f"Diagnostic failed: {exc}")
        finally:
            try:
                if original_priority is not None:
                    try:
                        restored = set_ae_priority(self.camera, original_priority)
                        if restored is None or not np.isclose(restored, original_priority):
                            raise RuntimeError(f"exposure priority readback mismatch: {restored}")
                        print(f"  [restore] auto_exposure_priority -> {restored:g}")
                    except (RuntimeError, OSError) as exc:
                        failed = True
                        print(f"  [restore] auto_exposure_priority FAILED: {exc}")
            finally:
                self.camera.disconnect()
                print("disconnect OK")

        print("\n" + "=" * 60)
        print("Test failed" if failed else "Test complete")
        print("=" * 60)
        return int(failed)

    def capture(self, *, allow_exposure_changes: bool) -> None:
        print("\nQ/Esc=退出 P=点云计算 S=raw/processed R=重置计算 C=配置 A=曝光优先级")
        print("冻结、时间浏览、配色和实体显隐使用 Rerun；关闭 viewer 不会停止相机。")
        rr = self.rr
        keys = Keyboard()
        intervals: deque[float] = deque(maxlen=FPS_WINDOW)
        previous_frame_start = None
        frame_count = 0
        mode = "processed"
        base_from_color: np.ndarray | None = None
        processor = None
        metric_names = ("fps", "read_ms", "pcd_ms", "valid_ratio", "point_count")
        series = [("Diagnostic", [f"diagnostic/{name}" for name in metric_names])]
        try:
            rr.init("DexMani RealSense", spawn=True)
            rr.send_blueprint(viewer_blueprint(series))
            keys.start()
            intrinsics = self.camera.color_intrinsics
            depth_scale_m = self.camera.depth_scale_m
            serial = cast(str, self.camera.active_serial)
            log_camera_geometry(intrinsics)
            while True:
                if not keys.healthy:
                    raise RuntimeError("diagnostic keyboard listener unavailable")
                if keys.estop_latched:
                    return
                for key in keys.poll(timeout=0):
                    if key == "q":
                        return
                    if key == "a":
                        if allow_exposure_changes:
                            toggle_ae_priority(self.camera)
                        else:
                            print("Exposure changes disabled: priority is unsupported")
                    elif key == "c":
                        print(f"Point cloud: enabled={base_from_color is not None} mode={mode} config={self.pointcloud_config}")
                    elif key in {"p", "s", "r"}:
                        enable_cloud = base_from_color is not None
                        if key == "p":
                            enable_cloud = not enable_cloud
                        elif key == "s":
                            mode = "raw" if mode == "processed" else "processed"
                        else:
                            enable_cloud, mode = False, "processed"
                        base_from_color = None
                        processor = None
                        log_points("pcd", None)
                        if enable_cloud:
                            validate_aligned_depth_distortion(intrinsics.distortion_model)
                            base_from_color, table_plane_abcd = self.load_pointcloud_geometry(
                                mode, serial,
                            )
                            if mode == "processed":
                                processor = PointCloudProcessor(
                                    depth_scale_m=depth_scale_m,
                                    intrinsics=intrinsics,
                                    T_xarm_base_from_color=base_from_color,
                                    table_plane_abcd=table_plane_abcd,
                                    config=self.pointcloud_config,
                                )
                            log_camera_geometry(intrinsics, base_from_color)
                            rr.send_blueprint(viewer_blueprint(series, camera_in_base=True))
                        print(f"Point cloud: enabled={base_from_color is not None} mode={mode}")

                started = time.perf_counter()
                if previous_frame_start is not None:
                    intervals.append(started - previous_frame_start)
                previous_frame_start = started
                frame = self.camera.read()
                if frame is None:
                    continue
                read_ms = (time.perf_counter() - started) * 1000
                frame_count += 1
                rgb, depth = frame.rgb, frame.depth
                if rgb.shape[:2] != depth.shape:
                    raise ValueError("aligned RGB-D dimensions do not match")
                rr.set_time_sequence("frame", frame_count)
                rr.set_time_nanos("camera_time", frame.timestamp_ns)
                log_rgbd(
                    rgb, depth, depth_scale_m=depth_scale_m,
                    depth_range_m=DEPTH_RANGE_M,
                )
                cloud, pcd_ms = None, 0.0
                if base_from_color is not None:
                    started = time.perf_counter()
                    if processor is not None:
                        cloud = processor.process(depth, rgb)
                    else:
                        cloud = build_raw_point_cloud(
                            depth_raw=depth, color=rgb, depth_scale_m=depth_scale_m,
                            intrinsics=intrinsics, T_xarm_base_from_color=base_from_color,
                        )
                    pcd_ms = (time.perf_counter() - started) * 1000
                    log_points("pcd", cloud)
                metrics = (
                    len(intervals) / sum(intervals) if intervals else 0.0,
                    read_ms, pcd_ms,
                    depth_valid_ratio(
                        depth.astype(np.float32) * depth_scale_m,
                        *DEPTH_RANGE_M,
                    ),
                    len(cloud) if cloud is not None else 0,
                )
                for name, value in zip(metric_names, metrics):
                    log_series(f"diagnostic/{name}", value)
        finally:
            print(f"Captured {frame_count} frames")
            try:
                keys.stop()
            finally:
                rr.disconnect()

    def load_pointcloud_geometry(
        self, mode: str, serial: str,
    ) -> tuple[np.ndarray, tuple[float, float, float, float] | None]:
        if self.calibration is None:
            self.calibration = load_camera_extrinsics(self.args.camera_calibration)
        transform = self.calibration[serial]
        if mode == "processed" and self.pointcloud_config.remove_table and self.table_plane_abcd is None:
            self.table_plane_abcd = load_table_plane(self.table)
        return transform, self.table_plane_abcd


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional experiment YAML overrides")
    parser.add_argument("--camera-calibration")
    return RealSenseViewer(parser.parse_args(argv)).run()


if __name__ == "__main__":
    raise SystemExit(main())
