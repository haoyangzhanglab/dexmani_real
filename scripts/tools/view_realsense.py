#!/usr/bin/env python3
"""实时查看 RealSense RGB-D/点云，退出时恢复原曝光优先级。"""

from __future__ import annotations

import argparse
import time
from collections import deque
from typing import TYPE_CHECKING, cast

import numpy as np

from dexmani_real.config.real import load_config
from dexmani_real.perception.pointcloud import (
    DEPTH_RANGE_M,
    PointCloudProcessor,
    build_raw_point_cloud,
)
from dexmani_real.utils.camera_calibrate.extrinsics import load_camera_extrinsics
from dexmani_real.utils.log import configure_logging
from dexmani_real.utils.table_plane import load_table_plane
from dexmani_real.utils.visualization.image import ImageViewer, rgbd_panel
from dexmani_real.utils.visualization.pointcloud import PointCloudViewer

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
        runtime = load_config(path=self.args.config)
        self.pointcloud_config = runtime.pointcloud
        self.table = runtime.environment.table

        from dexmani_real.sensor.realsense import RealSenseCamera

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
        print("\nQ/Esc=退出 P=点云 S=raw/processed F=冻结 R=重置 C=配置 D=配色 A=曝光优先级")
        print("快捷键在诊断窗口内生效；关闭 RGB-D 退出，关闭点云窗口仅关闭点云。")
        intervals: deque[float] = deque(maxlen=FPS_WINDOW)
        previous_frame_start = None
        frame_count = 0
        enabled, frozen = False, False
        mode, colormap = "processed", "jet"
        cached_cloud = None
        base_from_color = None
        processor = None
        image_viewer = cloud_viewer = None
        intrinsics = self.camera.color_intrinsics
        depth_scale_m = self.camera.depth_scale_m
        serial = cast(str, self.camera.active_serial)
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
                        enabled, frozen, cached_cloud = False, False, None
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
                        if enabled and cached_cloud is not None and len(cached_cloud):
                            frozen = not frozen
                        else:
                            print("Freeze unavailable until a nonempty point cloud is displayed")
                    elif key in {"p", "s", "r"}:
                        frozen, cached_cloud = False, None
                        if key == "p":
                            enabled = not enabled
                        elif key == "s":
                            mode = "raw" if mode == "processed" else "processed"
                        else:
                            enabled, mode, colormap = False, "processed", "jet"
                        if cloud_viewer is not None:
                            if enabled:
                                cloud_viewer.update(None)
                            else:
                                cloud_viewer.close()
                                cloud_viewer = None

                started = time.perf_counter()
                frame = self.camera.read()
                if frame is None:
                    continue
                read_ms = (time.perf_counter() - started) * 1000
                if previous_frame_start is not None:
                    intervals.append(started - previous_frame_start)
                previous_frame_start = started
                frame_count += 1
                rgb, depth = frame.rgb, frame.depth
                if rgb.shape[:2] != depth.shape:
                    raise ValueError("aligned RGB-D dimensions do not match")
                pcd_ms = 0.0
                if enabled and not frozen:
                    if base_from_color is None:
                        base_from_color = load_camera_extrinsics(self.args.camera_calibration)[serial]
                    if cloud_viewer is None:
                        cloud_viewer = PointCloudViewer(
                            workspace=self.pointcloud_config.workspace,
                            base_from_color=base_from_color, keys="psfrcda",
                        )
                    started = time.perf_counter()
                    if mode == "processed":
                        if processor is None:
                            processor = PointCloudProcessor(
                                depth_scale_m=depth_scale_m, intrinsics=intrinsics,
                                T_xarm_base_from_color=base_from_color,
                                table_plane_abcd=(
                                    load_table_plane(self.table)
                                    if self.pointcloud_config.remove_table else None
                                ),
                                config=self.pointcloud_config,
                            )
                        cached_cloud = processor.process(depth, rgb)
                    else:
                        cached_cloud = build_raw_point_cloud(
                            depth_raw=depth, color=rgb, depth_scale_m=depth_scale_m,
                            intrinsics=intrinsics, T_xarm_base_from_color=base_from_color,
                        )
                    pcd_ms = (time.perf_counter() - started) * 1000
                    cloud_viewer.update(cached_cloud)
                depth_m = depth.astype(np.float32) * depth_scale_m
                fps = len(intervals) / sum(intervals) if intervals else 0.0
                valid = depth_valid_ratio(depth_m, *DEPTH_RANGE_M)
                point_count = len(cached_cloud) if cached_cloud is not None else 0
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
            try:
                if cloud_viewer is not None:
                    cloud_viewer.close()
            finally:
                if image_viewer is not None:
                    image_viewer.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional experiment YAML overrides")
    parser.add_argument("--camera-calibration")
    return RealSenseViewer(parser.parse_args(argv)).run()


if __name__ == "__main__":
    raise SystemExit(main())
