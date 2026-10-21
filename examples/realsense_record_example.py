#!/usr/bin/env python3
"""实时查看 RealSense RGB-D/点云，退出时恢复原曝光优先级。"""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from dexmani_real.calibration.camera.extrinsics import CameraExtrinsics
from dexmani_real.config.experiment import load_experiment_config, resolve_table_plane
from dexmani_real.sensor.pointcloud.config import PointCloudConfig
from dexmani_real.sensor.camera.geometry import CameraIntrinsics, validate_aligned_depth_distortion
from dexmani_real.sensor.pointcloud.processing import build_point_cloud, build_raw_point_cloud
from dexmani_real.utils.log import configure_logging

if TYPE_CHECKING:
    from dexmani_real.sensor.camera.realsense import RealSenseCamera

WINDOW_NAME = "RealSense Test | RGB(left) Depth(right)"
FPS_WINDOW = 100


@dataclass
class PointCloudDisplayState:
    """Keyboard-controlled point-cloud display state."""

    show: bool = False
    colormap: str = "jet"
    cloud_mode: str = "processed"
    frozen: bool = False
    cached_cloud: np.ndarray | None = None

    def toggle_colormap(self) -> None:
        self.colormap = "gray" if self.colormap == "jet" else "jet"

    def reset(self) -> None:
        self.show = False
        self.colormap = "jet"
        self.cloud_mode = "processed"
        self.frozen = False
        self.cached_cloud = None


class NonBlockingPCDViewer:
    def __init__(self, point_size: float = 3.0) -> None:
        self.vis: Any | None = None
        self.pcd: Any | None = None
        self.frame: Any | None = None
        self.point_size = point_size

    def update(self, points: np.ndarray) -> bool:
        import open3d as o3d

        if points.dtype != np.float32 or points.ndim != 2 or points.shape[1] != 6:
            raise ValueError("viewer points must be float32 [N,6]")
        if not np.all(np.isfinite(points)):
            raise ValueError("viewer points must be finite")
        if points.size and np.any((points[:, 3:] < 0.0) | (points[:, 3:] > 1.0)):
            raise ValueError("viewer RGB values must be within [0,1]")

        if self.pcd is None:
            self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
        self.pcd.colors = o3d.utility.Vector3dVector(points[:, 3:].astype(np.float64))

        if self.vis is None:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name="Point Cloud", width=800, height=600)
            self.vis.add_geometry(self.pcd)
            self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            self.vis.add_geometry(self.frame)
            opt = self.vis.get_render_option()
            opt.point_size = float(self.point_size)
            opt.background_color = np.array([1.0, 1.0, 1.0])
        else:
            self.vis.update_geometry(self.pcd)

        if not self.vis.poll_events():
            return False
        self.vis.update_renderer()
        return True

    def close(self) -> None:
        if self.vis is not None:
            self.vis.destroy_window()
        self.vis = None
        self.pcd = None
        self.frame = None


def overlay_text(
    img: np.ndarray, lines: list[str], x: int = 10, y_start: int = 22, step: int = 22
) -> None:
    for i, line in enumerate(lines):
        cv2.putText(
            img,
            line,
            (x, y_start + i * step),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )


def make_jet_depth_vis(depth: np.ndarray, min_d: float, max_d: float) -> np.ndarray:
    """Render metric depth with near points at the hot end of the colormap."""
    depth_f32 = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth_f32) & (depth_f32 > 0.0)
    depth_clip = np.clip(depth_f32, min_d, max_d)
    depth_norm = (depth_clip - min_d) / max(max_d - min_d, 1e-6)
    depth_uint8 = (255.0 * (1.0 - depth_norm)).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    depth_vis[~valid] = 0
    return depth_vis


def depth_valid_ratio(depth: np.ndarray, min_d: float, max_d: float) -> float:
    """Return the fraction of depth pixels inside the display band."""
    depth_f32 = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth_f32) & (depth_f32 > 0.0) & (depth_f32 >= min_d) & (depth_f32 <= max_d)
    return float(valid.mean()) if depth_f32.size else 0.0


def make_gray_depth_vis(depth: np.ndarray, min_d: float, max_d: float) -> np.ndarray:
    """Grayscale metric depth: near=white, far=black."""
    depth_f32 = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth_f32) & (depth_f32 > 0)
    depth_clip = np.clip(depth_f32, min_d, max_d)
    depth_norm = (depth_clip - min_d) / max(max_d - min_d, 1e-6)
    gray = (255.0 * (1.0 - depth_norm)).astype(np.uint8)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    bgr[~valid] = 0
    return bgr


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


def build_hud_lines(
    frame_count: int,
    fps: float,
    read_ms: float,
    pcd_ms: float,
    valid_ratio: float,
    point_count: int,
    state: PointCloudDisplayState,
    total_dropped: int,
    intrinsics: CameraIntrinsics,
    production: PointCloudConfig,
) -> list[str]:
    lines = [
        f"frame={frame_count}  fps={fps:.1f}",
        f"read={read_ms:.1f}ms  pcd={pcd_ms:.1f}ms  valid={valid_ratio:.3f}",
        f"colorK=[{intrinsics.fx:.0f},{intrinsics.fy:.0f},"
        f"{intrinsics.ppx:.0f},{intrinsics.ppy:.0f}]",
    ]
    if state.show:
        if state.cloud_mode == "raw":
            lines.append(
                f"PCD RAW FULL  n={point_count}  "
                f"filter/crop=OFF freeze={state.frozen} drop={total_dropped}"
            )
        else:
            lines.append(
                f"PCD PROCESSED  n={point_count}  "
                f"vox={production.voxel_size_m:.3f} crop=production "
                f"freeze={state.frozen} drop={total_dropped}"
            )
    else:
        lines.append("PCD OFF  [p]toggle [s]raw/processed [f]freeze")
    lines.append(
        f"cmap={state.colormap}  depth=[{production.depth_min_m:.2f},{production.depth_max_m:.2f}]"
    )
    return lines


def handle_keyboard(
    key: int,
    state: PointCloudDisplayState,
    viewer: NonBlockingPCDViewer,
    camera: RealSenseCamera,
    production: PointCloudConfig,
    *,
    allow_exposure_changes: bool,
) -> bool:
    if key in (ord("q"), 27):
        return False

    if key == ord("a"):
        if allow_exposure_changes:
            toggle_ae_priority(camera)
        else:
            print("  Exposure changes disabled: original priority could not be read")
        return True

    if key == ord("p"):
        toggle_pcd(state, viewer)
    elif key == ord("s"):
        toggle_cloud_mode(state)
    elif key == ord("f"):
        toggle_freeze(state)
    elif key == ord("r"):
        reset_state(state, viewer)
    elif key == ord("c"):
        print_state(state, production)
    elif key == ord("d"):
        state.toggle_colormap()
        print(f"  Colormap: {state.colormap}")
    return True


def toggle_pcd(state: PointCloudDisplayState, viewer: NonBlockingPCDViewer) -> None:
    state.show = not state.show
    if not state.show:
        viewer.close()
    print(f"  Point cloud: {'ON' if state.show else 'OFF'}")


def toggle_cloud_mode(state: PointCloudDisplayState) -> None:
    state.cloud_mode = "raw" if state.cloud_mode == "processed" else "processed"
    state.frozen = False
    state.cached_cloud = None
    print(f"  Cloud mode: {state.cloud_mode.upper()}")


def toggle_freeze(state: PointCloudDisplayState) -> None:
    if state.cached_cloud is None:
        print("  Freeze unavailable until a point-cloud frame has been built")
        return
    state.frozen = not state.frozen
    print(f"  Point-cloud frame: {'FROZEN' if state.frozen else 'LIVE'}")


def reset_state(
    state: PointCloudDisplayState,
    viewer: NonBlockingPCDViewer,
) -> None:
    state.reset()
    viewer.close()
    print("  Display state reset")


def print_state(state: PointCloudDisplayState, production: PointCloudConfig) -> None:
    print(
        f"  mode={state.cloud_mode}  num_points={production.num_points}  "
        f"voxel={production.voxel_size_m}  crop=production  frozen={state.frozen}  "
        f"depth=[{production.depth_min_m},{production.depth_max_m}]  "
        f"workspace={production.workspace}  "
        f"pcd={'ON' if state.show else 'OFF'}"
    )


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

        from dexmani_real.sensor.camera.realsense import RealSenseCamera

        # Leave exposure priority untouched until it is snapshotted.
        config = self.runtime.camera
        configure_logging()

        print("=" * 60)
        print("RealSense Test -- RGB-D Live Capture + Real-time Point Cloud")
        print("=" * 60)
        print(f"OpenCV       : {cv2.__version__}")
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
        print("\n-- RGB-D live capture + point cloud --")
        print(
            "   q/Esc=quit  p=pcd  s=raw/processed  f=freeze "
            "r=reset  c=config  d=colormap  a=AE-priority"
        )

        self.state = PointCloudDisplayState()
        self.viewer = NonBlockingPCDViewer(point_size=3.0)
        self.frame_intervals: deque[float] = deque(maxlen=FPS_WINDOW)
        previous_frame_start = None
        frame_count = 0
        total_dropped = 0

        try:
            intrinsics = self.camera.get_color_intrinsics()
            depth_scale_m = self.camera.get_depth_scale()
            if not np.isfinite(depth_scale_m) or depth_scale_m <= 0.0:
                raise ValueError("display depth scale must be finite and positive")
            print(
                f"  Start: num_points={self.production.num_points}  "
                f"depth=[{self.production.depth_min_m}, {self.production.depth_max_m}]  "
                f"voxel={self.production.voxel_size_m}  workspace={self.production.workspace}"
            )

            while True:
                loop_start = time.perf_counter()
                if previous_frame_start is not None:
                    self.frame_intervals.append(loop_start - previous_frame_start)
                previous_frame_start = loop_start

                t0 = time.perf_counter()
                try:
                    frame = self.camera.read(timeout_ms=5000)
                except (RuntimeError, OSError) as e:
                    print(f"  read() failed: {e}")
                    outcome = "read_failed"
                    break
                read_ms = (time.perf_counter() - t0) * 1000.0
                frame_count += 1
                frame_rgb = frame.rgb
                aligned_depth_raw = frame.depth_aligned_to_color_raw
                aligned_depth_m = aligned_depth_raw.astype(np.float32) * depth_scale_m
                if frame_rgb.shape[:2] != aligned_depth_raw.shape:
                    print("  aligned RGB-D dimensions do not match; stopping diagnostic")
                    outcome = "read_failed"
                    break

                pcd_ms = 0.0
                point_count = 0
                pcd_array = None
                if self.state.show:
                    t0 = time.perf_counter()
                    try:
                        if self.state.frozen and self.state.cached_cloud is not None:
                            pcd_array = self.state.cached_cloud
                        elif self.state.cloud_mode == "raw":
                            pcd_array = build_raw_point_cloud(
                                depth_raw=aligned_depth_raw,
                                color=frame_rgb,
                                depth_scale_m=depth_scale_m,
                                intrinsics=intrinsics,
                                T_xarm_base_from_color=base_from_color,
                            )
                            self.state.cached_cloud = pcd_array
                        else:
                            pcd_array = build_point_cloud(
                                depth_raw=aligned_depth_raw,
                                color=frame_rgb,
                                depth_scale_m=depth_scale_m,
                                intrinsics=intrinsics,
                                T_xarm_base_from_color=base_from_color,
                                table_plane_abcd=table_plane_abcd,
                                config=self.production,
                            )
                            self.state.cached_cloud = pcd_array
                        point_count = pcd_array.shape[0] if pcd_array is not None else 0
                    except (ValueError, RuntimeError) as e:
                        if not self.state.frozen:
                            self.state.cached_cloud = None
                        total_dropped += 1
                        if total_dropped <= 3:
                            print(f"  PCD generation failed (frame {frame_count}): {e}")
                    pcd_ms = (time.perf_counter() - t0) * 1000.0

                if self.state.colormap == "jet":
                    depth_vis = make_jet_depth_vis(
                        aligned_depth_m,
                        self.production.depth_min_m,
                        self.production.depth_max_m,
                    )
                else:
                    depth_vis = make_gray_depth_vis(
                        aligned_depth_m,
                        self.production.depth_min_m,
                        self.production.depth_max_m,
                    )

                valid_ratio = depth_valid_ratio(
                    aligned_depth_m,
                    self.production.depth_min_m,
                    self.production.depth_max_m,
                )

                color_bgr = np.ascontiguousarray(frame_rgb[..., ::-1])
                panel = np.concatenate([color_bgr, depth_vis], axis=1)

                fps = (
                    len(self.frame_intervals) / sum(self.frame_intervals)
                    if self.frame_intervals
                    else 0.0
                )
                lines = build_hud_lines(
                    frame_count,
                    fps,
                    read_ms,
                    pcd_ms,
                    valid_ratio,
                    point_count,
                    self.state,
                    total_dropped,
                    intrinsics,
                    self.production,
                )
                overlay_text(panel, lines)
                cv2.imshow(WINDOW_NAME, panel)

                if self.state.show:
                    live_cloud = (
                        pcd_array if pcd_array is not None else np.empty((0, 6), dtype=np.float32)
                    )
                    if not self.viewer.update(live_cloud):
                        self.state.show = False
                        self.viewer.close()
                        print("  Point cloud window closed")

                key = cv2.waitKey(1) & 0xFF
                if not handle_keyboard(
                    key,
                    self.state,
                    self.viewer,
                    self.camera,
                    self.production,
                    allow_exposure_changes=allow_exposure_changes,
                ):
                    outcome = "user_exit"
                    break
                if self.state.show and key in (ord("p"), ord("s")):
                    try:
                        validate_aligned_depth_distortion(intrinsics.distortion_model)
                        base_from_color, table_plane_abcd = self.load_pointcloud_geometry(
                            self.state.cloud_mode,
                            self.camera.active_serial,
                        )
                    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                        self.state.show = False
                        self.state.frozen = False
                        self.state.cached_cloud = None
                        self.viewer.close()
                        print(
                            f"  PCD {self.state.cloud_mode} unavailable "
                            f"for serial {self.camera.active_serial!r}: {exc}"
                        )

            print(f"Captured {frame_count} frames; point-cloud failures: {total_dropped}")
            return outcome
        finally:
            try:
                self.viewer.close()
            finally:
                cv2.destroyAllWindows()

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
