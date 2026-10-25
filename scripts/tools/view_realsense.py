#!/usr/bin/env python3
"""实时查看 RealSense RGB-D/点云，退出时恢复原曝光优先级。"""

from __future__ import annotations

import argparse
import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.calibration.camera.extrinsics import CameraExtrinsics
from dexmani_real.config.experiment import load_experiment_config, resolve_table_plane
from dexmani_real.runtime.operator_input import KeyboardInput
from dexmani_real.sensor.camera.geometry import validate_aligned_depth_distortion
from dexmani_real.sensor.pointcloud.processing import build_point_cloud, build_raw_point_cloud
from dexmani_real.utils.log import configure_logging
from dexmani_real.utils.visualization import (
    log_camera_geometry, log_points, log_rgbd, log_series, viewer_blueprint,
)

if TYPE_CHECKING:
    from dexmani_real.sensor.camera.realsense import RealSenseCamera

FPS_WINDOW = 100


def depth_valid_ratio(depth: np.ndarray, min_d: float, max_d: float) -> float:
    """Return the fraction of depth pixels inside the display band."""
    depth_f32 = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth_f32) & (depth_f32 > 0.0) & (depth_f32 >= min_d) & (depth_f32 <= max_d)
    return float(valid.mean()) if depth_f32.size else 0.0


def list_cameras() -> list[dict[str, str]]:
    from dexmani_real.sensor.camera.realsense import RealSenseCamera

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
    """Read auto_exposure_priority (0=OFF, 1=ON); None if unsupported or unreadable."""
    import pyrealsense2 as rs

    try:
        sensor = camera.get_color_sensor()
        option = rs.option.auto_exposure_priority
        if sensor.supports(option):
            return float(sensor.get_option(option))
    except (RuntimeError, OSError):
        pass
    return None


def set_ae_priority(camera: RealSenseCamera, value: float) -> float | None:
    """Set auto_exposure_priority and return the readback; None if unsupported."""
    import pyrealsense2 as rs

    sensor = camera.get_color_sensor()
    option = rs.option.auto_exposure_priority
    if not sensor.supports(option):
        return None
    sensor.set_option(option, float(value))
    return float(sensor.get_option(option))


def toggle_ae_priority(camera: RealSenseCamera) -> None:
    """Toggle auto_exposure_priority; OFF caps exposure at the requested frame period."""
    current = get_ae_priority(camera)
    if current is None:
        print("  auto_exposure_priority: not supported/readable on this color sensor")
        return
    target = 0.0 if current >= 0.5 else 1.0
    readback = set_ae_priority(camera, target)
    if readback is None:
        print("  auto_exposure_priority: set failed (unsupported)")
        return
    label = "OFF (cap exposure at frame period)" if target == 0.0 else "ON (may reduce FPS)"
    print(f"  auto_exposure_priority: {current:g} -> {readback:g}  [{label}]")


class RealSenseExample:
    """Own live preview state and restore the captured exposure priority on exit."""

    def __init__(self, args):
        self.args = args

    def run(self) -> int:
        # Resolve experiment settings before enumerating or opening a camera.
        self.runtime = load_experiment_config(yaml_path=self.args.config)
        self.production = self.runtime.pointcloud
        if not np.isfinite([self.production.depth_min_m, self.production.depth_max_m]).all() or not (
            0.0 < self.production.depth_min_m < self.production.depth_max_m
        ):
            raise ValueError("display depth range must be finite and satisfy 0 < min < max")
        self.table = self.runtime.environment.table
        self.calibration = self.table_plane_abcd = None

        import rerun as rr
        from dexmani_real.sensor.camera.realsense import RealSenseCamera

        self.rr = rr
        # Leave exposure priority untouched until it is snapshotted.
        config = self.runtime.camera
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
        self.original_priority = None
        restore_failed = False
        outcome = "read_failed"
        try:
            self.camera.connect()
            print(f"Connected: {self.camera.get_device_info()}")
            self.original_priority = get_ae_priority(self.camera)
            if self.original_priority is not None:
                print(f"  auto_exposure_priority (original): {self.original_priority:g}")
                if self.original_priority != 0.0:
                    applied = set_ae_priority(self.camera, 0.0)
                    if applied is None or not np.isclose(applied, 0.0):
                        raise RuntimeError("could not apply diagnostic exposure priority")
            else:
                print("  auto_exposure_priority unavailable; leaving it unchanged")
            outcome = self.capture(allow_exposure_changes=self.original_priority is not None)
        except (RuntimeError, OSError) as exc:
            print(f"Diagnostic failed: {exc}")
        finally:
            try:
                if self.original_priority is not None:
                    try:
                        restored = set_ae_priority(self.camera, self.original_priority)
                        if restored is None or not np.isclose(restored, self.original_priority):
                            raise RuntimeError(f"exposure priority readback mismatch: {restored}")
                        print(f"  [restore] auto_exposure_priority -> {restored:g}")
                    except (RuntimeError, OSError) as exc:
                        restore_failed = True
                        print(f"  [restore] auto_exposure_priority FAILED: {exc}")
            finally:
                self.camera.disconnect()
                print("disconnect OK")

        failed = outcome == "read_failed" or restore_failed
        print("\n" + "=" * 60)
        print("Test failed" if failed else "Test complete")
        print("=" * 60)
        return int(failed)

    def capture(self, *, allow_exposure_changes: bool) -> str:
        print("\nQ/Esc=退出 P=点云计算 S=raw/processed R=重置计算 C=配置 A=曝光优先级")
        print("冻结、时间浏览、配色和实体显隐使用 Rerun；关闭 viewer 不会停止相机。")
        rr = self.rr
        keys = KeyboardInput(
            capture_commands=False, capture_raw_events=True,
            raw_event_keys=frozenset({"q", "p", "s", "r", "c", "a"}),
        )
        intervals = deque(maxlen=FPS_WINDOW)
        previous_frame_start = None
        frame_count = total_dropped = 0
        show_cloud, mode = False, "processed"
        base_from_color = table_plane_abcd = None
        metric_names = ("fps", "read_ms", "pcd_ms", "valid_ratio", "point_count", "dropped")
        series = [("Diagnostic", [f"diagnostic/{name}" for name in metric_names])]
        try:
            rr.init("DexMani RealSense", spawn=True)
            rr.send_blueprint(viewer_blueprint(series))
            keys.start()
            intrinsics = self.camera.get_color_intrinsics()
            depth_scale_m = self.camera.get_depth_scale()
            if not np.isfinite(depth_scale_m) or depth_scale_m <= 0:
                raise ValueError("display depth scale must be finite and positive")
            log_camera_geometry(intrinsics)
            while True:
                if not keys.healthy:
                    raise RuntimeError("diagnostic keyboard listener unavailable")
                if keys.estop_latched:
                    return "user_exit"
                while (key := keys.pop_event()) is not None:
                    if key == "q":
                        return "user_exit"
                    if key == "a":
                        if allow_exposure_changes:
                            toggle_ae_priority(self.camera)
                        else:
                            print("Exposure changes disabled: original priority could not be read")
                    elif key == "c":
                        print(f"Point cloud: enabled={show_cloud} mode={mode} config={self.production}")
                    elif key in {"p", "s", "r"}:
                        if key == "p":
                            show_cloud = not show_cloud
                        elif key == "s":
                            mode = "raw" if mode == "processed" else "processed"
                        else:
                            show_cloud, mode = False, "processed"
                        log_points("pcd", None)
                        if show_cloud:
                            try:
                                validate_aligned_depth_distortion(intrinsics.distortion_model)
                                base_from_color, table_plane_abcd = self.load_pointcloud_geometry(
                                    mode, self.camera.active_serial,
                                )
                                log_camera_geometry(intrinsics, base_from_color)
                                rr.send_blueprint(viewer_blueprint(series, camera_in_base=True))
                            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                                show_cloud = False
                                print(f"PCD {mode} unavailable for {self.camera.active_serial!r}: {exc}")
                        print(f"Point cloud: enabled={show_cloud} mode={mode}")

                started = time.perf_counter()
                if previous_frame_start is not None:
                    intervals.append(started - previous_frame_start)
                previous_frame_start = started
                try:
                    frame = self.camera.read(timeout_ms=5000)
                except (RuntimeError, OSError) as exc:
                    print(f"read() failed: {exc}")
                    return "read_failed"
                read_ms = (time.perf_counter() - started) * 1000
                frame_count += 1
                rgb, depth = frame.rgb, frame.depth_aligned_to_color_raw
                if rgb.shape[:2] != depth.shape:
                    raise ValueError("aligned RGB-D dimensions do not match")
                rr.set_time_sequence("frame", frame_count)
                rr.set_time_nanos("camera_time", frame.timestamp_ns)
                log_rgbd(
                    rgb, depth, depth_scale_m=depth_scale_m,
                    depth_range_m=(self.production.depth_min_m, self.production.depth_max_m),
                )
                cloud, pcd_ms = None, 0.0
                if show_cloud:
                    started = time.perf_counter()
                    try:
                        kwargs = dict(
                            depth_raw=depth, color=rgb, depth_scale_m=depth_scale_m,
                            intrinsics=intrinsics, T_xarm_base_from_color=base_from_color,
                        )
                        cloud = (
                            build_raw_point_cloud(**kwargs) if mode == "raw" else build_point_cloud(
                                **kwargs, table_plane_abcd=table_plane_abcd, config=self.production,
                            )
                        )
                    except (ValueError, RuntimeError) as exc:
                        total_dropped += 1
                        if total_dropped <= 3:
                            print(f"PCD generation failed (frame {frame_count}): {exc}")
                    pcd_ms = (time.perf_counter() - started) * 1000
                    log_points("pcd", cloud)
                metrics = (
                    len(intervals) / sum(intervals) if intervals else 0.0,
                    read_ms, pcd_ms,
                    depth_valid_ratio(
                        depth.astype(np.float32) * depth_scale_m,
                        self.production.depth_min_m, self.production.depth_max_m,
                    ),
                    len(cloud) if cloud is not None else 0, total_dropped,
                )
                for name, value in zip(metric_names, metrics):
                    log_series(f"diagnostic/{name}", value)
        finally:
            print(f"Captured {frame_count} frames; point-cloud failures: {total_dropped}")
            try:
                keys.stop()
            finally:
                rr.disconnect()

    def load_pointcloud_geometry(self, mode, serial):
        if mode == "processed":
            self.production.validate()
        if self.calibration is None:
            self.calibration = CameraExtrinsics(self.args.camera_calibration)
        transform = self.calibration.for_serial(serial)
        if mode == "processed" and self.production.remove_table and self.table_plane_abcd is None:
            self.table_plane_abcd = resolve_table_plane(self.table)
        return transform, self.table_plane_abcd


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional experiment YAML overrides")
    parser.add_argument("--camera-calibration")
    return RealSenseExample(parser.parse_args(argv)).run()


if __name__ == "__main__":
    raise SystemExit(main())
