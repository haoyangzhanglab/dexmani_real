#!/usr/bin/env python3
"""Usage: ``python examples/realsense_record_example.py``.

Self-contained interactive RealSense RGB-D and point-cloud diagnostic. The
point-cloud window switches explicitly between the complete unprocessed native
depth cloud and the canonical processed production cloud. Diagnostic crop and
voxel overrides never mutate the runtime production policy.

Keyboard controls::

    q / Esc    Quit
    p          Toggle point cloud display
    s          Switch RAW full <-> PROCESSED
    v          Cycle diagnostic processed voxel size
    w          Toggle processed table/workspace crop
    f          Freeze/unfreeze the point-cloud frame
    r          Reset to defaults
    c          Print current config
    d          Toggle depth colormap (jet <-> gray)
    a          Toggle auto-exposure priority ON <-> OFF (Option A test)

The ``a`` key toggles ``auto_exposure_priority`` live (Auto Exposure stays ON),
so you can watch the RGB brighten/darken and the fps jump between ~16.7 Hz
(priority ON, exposure ~60 ms in a dark scene) and ~30 Hz (priority OFF). The
pre-test priority value is restored automatically on exit.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pyrealsense2 as rs

from dexmani_real.config.camera_calib import CameraCalib
from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.sensor.camera_geometry import RGBDGeometry
from dexmani_real.sensor.pointcloud import build_point_cloud, build_raw_point_cloud
from dexmani_real.sensor.realsense import CameraFrame, RealSense, RealSenseConfig

_DIAGNOSTIC_VOXEL_CYCLE_M = (0.005, 0.010)
_FULL_VIEW_WORKSPACE: tuple[float, float, float, float, float, float] = (
    -0.3,
    -1.0,
    -0.5,  # xyz_min
    2.0,
    1.0,
    1.5,  # xyz_max
)

_WINDOW_NAME = "RealSense Test | RGB(left) Depth(right)"


@dataclass(frozen=True)
class RealSenseDiagnosticConfig:
    """Configuration for camera connection and interactive display."""

    fps: int = 30
    depth_resolution: tuple[int, int] = (640, 480)
    color_resolution: tuple[int, int] = (640, 480)
    warmup_frames: int = 10
    stats_window: int = 100  # rolling-average window (frames)


@dataclass
class PointCloudDisplayState:
    """Mutable point-cloud display state toggled via keyboard.

    Not frozen -- state is mutated in place by key handlers.
    """

    show: bool = False
    colormap: str = "jet"
    cloud_mode: str = "processed"
    voxel_size_m: float = 0.005
    crop_on: bool = True
    frozen: bool = False
    cached_cloud: np.ndarray | None = None

    def toggle_colormap(self) -> None:
        self.colormap = "gray" if self.colormap == "jet" else "jet"

    def reset(self, production: PointCloudConfig) -> None:
        self.show = False
        self.colormap = "jet"
        self.cloud_mode = "processed"
        self.voxel_size_m = production.voxel_size_m
        self.crop_on = True
        self.frozen = False
        self.cached_cloud = None


@dataclass
class FrameStats:
    """Per-frame timing and quality metrics."""

    read_ms: float = 0.0
    pcd_ms: float = 0.0
    valid_ratio: float = 0.0
    point_count: int = 0
    total_ms: float = 0.0


def _make_pcd_config(
    state: PointCloudDisplayState, production: PointCloudConfig
) -> PointCloudConfig:
    """Apply explicit diagnostic overrides to the immutable production policy."""
    return replace(
        production,
        voxel_size_m=state.voxel_size_m,
        workspace=production.workspace if state.crop_on else _FULL_VIEW_WORKSPACE,
    )


class NonBlockingPCDViewer:
    """Non-blocking open3d point-cloud window -- create on first frame, update thereafter."""

    def __init__(self, point_size: float = 3.0) -> None:
        # Keep optional Open3D lifecycle state explicit because its stubs are incomplete.
        self._vis: Any | None = None
        self._pcd: Any | None = None
        self._frame: Any | None = None
        self._created: bool = False
        self.point_size = point_size

    def update(self, points: np.ndarray) -> bool:
        """Update point cloud.  *points*: (N, 3) or (N, 6).  Returns False if window closed."""
        import open3d as o3d

        pcd_array = points.astype(np.float32)
        if pcd_array.ndim != 2 or pcd_array.shape[1] not in (3, 6):
            return self._created

        if self._pcd is None:
            self._pcd = o3d.geometry.PointCloud()
        self._pcd.points = o3d.utility.Vector3dVector(
            pcd_array[:, :3].astype(np.float64)
        )

        if pcd_array.shape[1] == 6:
            colors = pcd_array[:, 3:].astype(np.float64)
            if colors.size and colors.max() > 1.0:
                colors /= 255.0
            self._pcd.colors = o3d.utility.Vector3dVector(colors)

        if not self._created:
            self._vis = o3d.visualization.Visualizer()
            self._vis.create_window(window_name="Point Cloud", width=800, height=600)
            self._vis.add_geometry(self._pcd)
            self._frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            self._vis.add_geometry(self._frame)
            opt = self._vis.get_render_option()
            opt.point_size = float(self.point_size)
            opt.background_color = np.array([1.0, 1.0, 1.0])
            self._created = True
        else:
            assert self._vis is not None
            self._vis.update_geometry(self._pcd)

        assert self._vis is not None
        if not self._vis.poll_events():
            self._created = False
            return False
        self._vis.update_renderer()
        return True

    def close(self) -> None:
        if self._vis is not None:
            self._vis.destroy_window()
            self._vis = None
            self._pcd = None
            self._frame = None
            self._created = False


def _overlay_text(
    img: np.ndarray, lines: list[str], x: int = 10, y_start: int = 22, step: int = 22
) -> None:
    """Draw green HUD text lines on an image."""
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


def _make_jet_depth_vis(depth: np.ndarray, min_d: float, max_d: float) -> np.ndarray:
    """Render metric depth with near points at the hot end of the colormap."""
    depth_f32 = depth.astype(np.float32)
    valid = np.isfinite(depth_f32) & (depth_f32 > 0.0)
    depth_clip = np.clip(depth_f32, min_d, max_d)
    depth_norm = (depth_clip - min_d) / max(max_d - min_d, 1e-6)
    depth_uint8 = (255.0 * (1.0 - depth_norm)).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    depth_vis[~valid] = 0
    return depth_vis


def _depth_valid_ratio(depth: np.ndarray, min_d: float, max_d: float) -> float:
    """Return the fraction of depth pixels inside the display band."""
    depth_f32 = depth.astype(np.float32)
    valid = (
        np.isfinite(depth_f32)
        & (depth_f32 > 0.0)
        & (depth_f32 >= min_d)
        & (depth_f32 <= max_d)
    )
    return float(valid.mean()) if depth_f32.size else 0.0


def _make_gray_depth_vis(depth: np.ndarray, min_d: float, max_d: float) -> np.ndarray:
    """Grayscale depth visualization (near=white, far=black).

    Complements ``_make_jet_depth_vis`` which only produces jet colormap.
    """
    depth_f32 = depth.astype(np.float32)
    valid = np.isfinite(depth_f32) & (depth_f32 > 0)
    depth_clip = np.clip(depth_f32, min_d, max_d)
    depth_norm = (depth_clip - min_d) / max(max_d - min_d, 1e-6)
    gray = (255.0 * (1.0 - depth_norm)).astype(np.uint8)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    bgr[~valid] = 0
    return bgr


def _list_cameras() -> list[dict[str, str]]:
    """Enumerate connected RealSense cameras with retry."""
    for attempt in range(3):
        try:
            cameras = RealSense.list_cameras()
        except RuntimeError as e:
            if attempt < 2:
                print(f"  Enumeration failed ({e}), retry {attempt + 2}/3...")
                time.sleep(1.0)
                continue
            print(f"  Enumeration failed: {e}")
            return []
        break

    if not cameras:
        print("No RealSense camera detected. Check USB connection.")
        return []
    print(f"Detected {len(cameras)} camera(s):")
    for i, cam in enumerate(cameras):
        print(
            f"  [{i}] {cam.get('name', 'unknown'):28s} "
            f"SN={cam.get('serial', ''):16s} "
            f"FW={cam.get('firmware', '')}  "
            f"PL={cam.get('product_line', '')}"
        )
    return cameras


def _test_lifecycle(test_cfg: RealSenseDiagnosticConfig) -> bool:
    """Connect, verify intrinsics, disconnect (idempotent checks included)."""
    print("\n-- 1. connect/disconnect lifecycle --")

    config = RealSenseConfig(
        depth_resolution=test_cfg.depth_resolution,
        color_resolution=test_cfg.color_resolution,
        fps=test_cfg.fps,
        warmup_frames=test_cfg.warmup_frames,
    )
    camera = RealSense(config)

    # Connect with retry (L515 USB state can be flaky).
    ok = False
    for attempt in range(3):
        ok = camera.connect()
        print(
            f"  connect() -> {ok}"
            + (f" (attempt {attempt + 1}/3)" if attempt > 0 else "")
        )
        if ok:
            break
        time.sleep(1.0)
    if not ok:
        print("  FAIL: connect failed after 3 retries")
        return False

    print(f"  connect() idempotent -> {camera.connect()}")

    geometry = camera.get_geometry()
    K = geometry.depth.matrix()
    print(f"  K =\n{K}")
    print(f"  depth_scale = {camera.get_depth_scale()}")
    depth_intrinsics = geometry.depth
    print(
        f"  depth intrinsics = {depth_intrinsics.width}x{depth_intrinsics.height}  "
        f"fx={depth_intrinsics.fx:.1f} fy={depth_intrinsics.fy:.1f} "
        f"cx={depth_intrinsics.ppx:.1f} cy={depth_intrinsics.ppy:.1f}"
    )
    print(f"  color intrinsics = {geometry.color.width}x{geometry.color.height}")

    camera.disconnect()
    print("  disconnect() -> OK")
    camera.disconnect()  # idempotent
    print("  disconnect() idempotent -> OK")
    print("  Lifecycle test passed")
    return True


def _compute_rolling_stats(history: deque[FrameStats]) -> dict[str, float]:
    """Compute rolling averages from frame stats history."""
    if not history:
        return dict(fps=0, read_ms=0, pcd_ms=0, total_ms=0, valid=0, points=0)
    mean_total_ms = float(np.mean([sample.total_ms for sample in history]))
    return dict(
        fps=1000.0 / max(mean_total_ms, 0.001),
        read_ms=float(np.mean([s.read_ms for s in history])),
        pcd_ms=float(np.mean([s.pcd_ms for s in history])),
        total_ms=float(np.mean([s.total_ms for s in history])),
        valid=float(np.mean([s.valid_ratio for s in history])),
        points=float(np.mean([s.point_count for s in history])),
    )


def _build_hud_lines(
    frame: CameraFrame,
    frame_count: int,
    stats: dict[str, float],
    state: PointCloudDisplayState,
    total_dropped: int,
    geometry: RGBDGeometry,
    production: PointCloudConfig,
) -> list[str]:
    """Build HUD overlay text lines."""
    depth_intrinsics = geometry.depth
    lines = [
        f"frame={frame_count}  fps={stats['fps']:.1f}  total={stats['total_ms']:.1f}ms",
        f"read={stats['read_ms']:.1f}ms  pcd={stats['pcd_ms']:.1f}ms  valid={stats['valid']:.3f}",
        f"depthK=[{depth_intrinsics.fx:.0f},{depth_intrinsics.fy:.0f},"
        f"{depth_intrinsics.ppx:.0f},{depth_intrinsics.ppy:.0f}]",
    ]
    if state.show:
        if state.cloud_mode == "raw":
            lines.append(
                f"PCD RAW FULL  n={stats['points']:.0f}  "
                f"filter/crop=OFF freeze={state.frozen} drop={total_dropped}"
            )
        else:
            lines.append(
                f"PCD PROCESSED  n={stats['points']:.0f}  "
                f"vox={state.voxel_size_m:.3f} crop={state.crop_on} "
                f"freeze={state.frozen} drop={total_dropped}"
            )
    else:
        lines.append("PCD OFF  [p]toggle [s]raw/processed [v]voxel [w]crop [f]freeze")
    lines.append(
        f"cmap={state.colormap}  depth=[{production.depth_min_m:.2f},"
        f"{production.depth_max_m:.2f}]"
    )
    return lines


def _build_key_actions(
    state: PointCloudDisplayState,
    viewer: NonBlockingPCDViewer,
    production: PointCloudConfig,
) -> dict[int, Callable[[], None]]:
    """Build the key-action dispatch table for the current loop state."""
    return {
        ord("p"): lambda: _toggle_pcd(state, viewer),
        ord("s"): lambda: _toggle_cloud_mode(state),
        ord("v"): lambda: _cycle_voxel(state),
        ord("w"): lambda: _toggle_crop(state),
        ord("f"): lambda: _toggle_freeze(state),
        ord("r"): lambda: _reset_state(state, viewer, production),
        ord("c"): lambda: _print_state(state, production),
        ord("d"): lambda: _toggle_cmap(state),
    }


def _handle_keyboard(
    key: int,
    state: PointCloudDisplayState,
    viewer: NonBlockingPCDViewer,
    camera: RealSense,
    production: PointCloudConfig,
) -> bool:
    """Process keyboard input.  Returns False if quit requested."""
    if key in (ord("q"), 27):
        return False

    # 'a' toggles auto_exposure_priority live (Option A test: OFF -> ~30 Hz).
    if key == ord("a"):
        _toggle_ae_priority(camera)
        return True

    action = _build_key_actions(state, viewer, production).get(key)
    if action:
        action()
    return True


def _toggle_pcd(state: PointCloudDisplayState, viewer: NonBlockingPCDViewer) -> None:
    state.show = not state.show
    if not state.show:
        viewer.close()
    print(f"  Point cloud: {'ON' if state.show else 'OFF'}")


def _toggle_cloud_mode(state: PointCloudDisplayState) -> None:
    state.cloud_mode = "raw" if state.cloud_mode == "processed" else "processed"
    state.frozen = False
    state.cached_cloud = None
    print(f"  Cloud mode: {state.cloud_mode.upper()}")


def _cycle_voxel(state: PointCloudDisplayState) -> None:
    try:
        index = _DIAGNOSTIC_VOXEL_CYCLE_M.index(state.voxel_size_m)
        next_index = (index + 1) % len(_DIAGNOSTIC_VOXEL_CYCLE_M)
    except ValueError:
        next_index = 0
    state.voxel_size_m = _DIAGNOSTIC_VOXEL_CYCLE_M[next_index]
    state.frozen = False
    state.cached_cloud = None
    print(f"  Processed voxel: {state.voxel_size_m:.3f} m")


def _toggle_crop(state: PointCloudDisplayState) -> None:
    state.crop_on = not state.crop_on
    state.frozen = False
    state.cached_cloud = None
    print(f"  Processed table/workspace crop: {'ON' if state.crop_on else 'OFF'}")


def _toggle_freeze(state: PointCloudDisplayState) -> None:
    if state.cached_cloud is None:
        print("  Freeze unavailable until a point-cloud frame has been built")
        return
    state.frozen = not state.frozen
    print(f"  Point-cloud frame: {'FROZEN' if state.frozen else 'LIVE'}")


def _reset_state(
    state: PointCloudDisplayState,
    viewer: NonBlockingPCDViewer,
    production: PointCloudConfig,
) -> None:
    state.reset(production)
    viewer.close()
    print("  Config reset to defaults")


def _print_state(state: PointCloudDisplayState, production: PointCloudConfig) -> None:
    pcd = _make_pcd_config(state, production)
    print(
        f"  mode={state.cloud_mode}  num_points={pcd.num_points}  "
        f"voxel={pcd.voxel_size_m}  crop={state.crop_on}  frozen={state.frozen}  "
        f"depth=[{pcd.depth_min_m},{pcd.depth_max_m}]  workspace={pcd.workspace}  "
        f"pcd={'ON' if state.show else 'OFF'}"
    )


def _toggle_cmap(state: PointCloudDisplayState) -> None:
    state.toggle_colormap()
    print(f"  Colormap: {state.colormap}")


def _compute_base_from_depth(geometry: RGBDGeometry, serial: str | None) -> np.ndarray:
    """Return T_xarm_base_from_depth from calibration and live geometry.

    Same wiring as ``examples/check_l515_native_shadow.py``: resolve the
    calibration entry by serial, then compose the static camera extrinsic with
    the live depth->color transform. Assumes an eye-to-hand camera (a static
    extrinsic); an eye-in-hand entry would additionally need the live eef FK.
    """
    if not serial:
        raise RuntimeError("connected camera did not publish a serial")
    calibration = CameraCalib()
    camera_name = calibration.resolve_name_by_serial(serial)
    base_from_color = calibration.get_extrinsics(camera_name)
    return base_from_color @ geometry.T_color_from_depth


def _find_color_sensor(camera: RealSense) -> Any:
    """Return the color sensor from the live pipeline profile."""
    if camera.profile is None:
        raise RuntimeError("camera pipeline profile is unavailable")
    device = camera.profile.get_device()
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            try:
                if profile.stream_type() == rs.stream.color:
                    return sensor
            except RuntimeError:
                continue
    raise RuntimeError("no color sensor with a color stream profile found")


def _get_ae_priority(camera: RealSense) -> float | None:
    """Read auto_exposure_priority (0=OFF, 1=ON); None if unsupported."""
    try:
        sensor = _find_color_sensor(camera)
        option = rs.option.auto_exposure_priority
        if sensor.supports(option):
            return float(sensor.get_option(option))
    except RuntimeError:
        pass
    return None


def _set_ae_priority(camera: RealSense, value: float) -> float | None:
    """Set auto_exposure_priority and return the readback; None if unsupported."""
    sensor = _find_color_sensor(camera)
    option = rs.option.auto_exposure_priority
    if not sensor.supports(option):
        return None
    sensor.set_option(option, float(value))
    return float(sensor.get_option(option))


def _toggle_ae_priority(camera: RealSense) -> None:
    """Toggle auto_exposure_priority ON<->OFF (Option A: OFF keeps ~30 Hz)."""
    current = _get_ae_priority(camera)
    if current is None:
        print("  auto_exposure_priority: not supported/readable on this color sensor")
        return
    target = 0.0 if current >= 0.5 else 1.0
    readback = _set_ae_priority(camera, target)
    if readback is None:
        print("  auto_exposure_priority: set failed (unsupported)")
        return
    label = "OFF (keep ~30 Hz)" if target == 0.0 else "ON (default)"
    print(f"  auto_exposure_priority: {current:g} -> {readback:g}  [{label}]")


def _run_rgbd_test(camera: RealSense, test_cfg: RealSenseDiagnosticConfig) -> dict:
    """Run interactive RGB-D live capture + point cloud visualization."""
    print("\n-- 2. RGB-D live capture + point cloud --")
    print(
        "   q/Esc=quit  p=pcd  s=raw/processed  v=voxel  w=crop  f=freeze "
        "r=reset  c=config  d=colormap  a=AE-priority"
    )

    runtime = resolve_runtime_config()
    production = runtime.pointcloud
    table = runtime.environment.table
    table_plane_abcd = table.plane_abcd if table.enabled else None
    state = PointCloudDisplayState(voxel_size_m=production.voxel_size_m)
    viewer = NonBlockingPCDViewer(point_size=3.0)
    stats_history: deque[FrameStats] = deque(maxlen=test_cfg.stats_window)
    frame_count = 0
    total_dropped = 0

    geometry = camera.get_geometry()
    depth_scale_m = camera.get_depth_scale()
    base_from_depth = _compute_base_from_depth(geometry, camera.active_serial)
    pcd_config = _make_pcd_config(state, production)
    print(
        f"  Start: num_points={pcd_config.num_points}  "
        f"depth=[{pcd_config.depth_min_m}, {pcd_config.depth_max_m}]  "
        f"voxel={pcd_config.voxel_size_m}  workspace={pcd_config.workspace}"
    )

    while True:
        loop_start = time.perf_counter()

        t0 = time.perf_counter()
        try:
            frame = camera.read(timeout_ms=5000)
        except RuntimeError as e:
            print(f"  read() failed: {e}")
            break
        read_ms = (time.perf_counter() - t0) * 1000.0
        frame_count += 1
        frame_rgb = frame.rgb
        if frame_rgb is None:
            print("  RGB frame unavailable; stopping diagnostic")
            break

        pcd_ms = 0.0
        point_count = 0
        pcd_array = None
        if state.show:
            t0 = time.perf_counter()
            try:
                if state.frozen and state.cached_cloud is not None:
                    pcd_array = state.cached_cloud
                elif state.cloud_mode == "raw":
                    pcd_array = build_raw_point_cloud(
                        depth_raw=frame.depth_raw,
                        color=frame_rgb,
                        depth_scale_m=depth_scale_m,
                        geometry=geometry,
                        T_xarm_base_from_depth=base_from_depth,
                    )
                    state.cached_cloud = pcd_array
                else:
                    pcd_config = _make_pcd_config(state, production)
                    pcd_array = build_point_cloud(
                        depth_raw=frame.depth_raw,
                        color=frame_rgb,
                        depth_scale_m=depth_scale_m,
                        geometry=geometry,
                        T_xarm_base_from_depth=base_from_depth,
                        table_plane_abcd=(table_plane_abcd if state.crop_on else None),
                        config=pcd_config,
                    )
                    state.cached_cloud = pcd_array
                point_count = pcd_array.shape[0] if pcd_array is not None else 0
            except (ValueError, RuntimeError) as e:
                total_dropped += 1
                if total_dropped <= 3:
                    print(f"  PCD generation failed (frame {frame_count}): {e}")
            pcd_ms = (time.perf_counter() - t0) * 1000.0

        if state.colormap == "jet":
            depth_vis = _make_jet_depth_vis(
                frame.depth, production.depth_min_m, production.depth_max_m
            )
        else:
            depth_vis = _make_gray_depth_vis(
                frame.depth, production.depth_min_m, production.depth_max_m
            )

        valid_ratio = _depth_valid_ratio(
            frame.depth, production.depth_min_m, production.depth_max_m
        )

        color_bgr = np.ascontiguousarray(frame_rgb[..., ::-1])
        if color_bgr.shape[:2] != depth_vis.shape[:2]:
            depth_vis = cv2.resize(depth_vis, (color_bgr.shape[1], color_bgr.shape[0]))
        panel = np.concatenate([color_bgr, depth_vis], axis=1)

        total_ms = (time.perf_counter() - loop_start) * 1000.0
        stats_history.append(
            FrameStats(
                read_ms=read_ms,
                pcd_ms=pcd_ms,
                valid_ratio=valid_ratio,
                point_count=point_count,
                total_ms=total_ms,
            )
        )

        stats = _compute_rolling_stats(stats_history)
        lines = _build_hud_lines(
            frame, frame_count, stats, state, total_dropped, geometry, production
        )
        _overlay_text(panel, lines)
        cv2.imshow(_WINDOW_NAME, panel)

        if state.show and pcd_array is not None and pcd_array.shape[0] > 0:
            if not viewer.update(pcd_array):
                state.show = False
                viewer.close()
                print("  Point cloud window closed")

        if not _handle_keyboard(
            cv2.waitKey(1) & 0xFF, state, viewer, camera, production
        ):
            break

    viewer.close()
    cv2.destroyAllWindows()

    if stats_history:
        reads = np.array([s.read_ms for s in stats_history])
        totals = np.array([s.total_ms for s in stats_history])
        pcds = np.array([s.pcd_ms for s in stats_history])
        valids = np.array([s.valid_ratio for s in stats_history])
        avg_total = float(totals.mean())
        fps_avg = 1000.0 / avg_total if avg_total > 0 else 0.0

        print(f"\n  -- Performance ({frame_count} frames) --")
        print(f"  avg fps:          {fps_avg:.1f}")
        print(f"  avg frame total:  {avg_total:.1f} ms  (max {totals.max():.1f} ms)")
        print(
            f"  avg read(grab):   {float(reads.mean()):.1f} ms  (max {reads.max():.1f} ms)"
        )
        print(f"  avg pcd:          {float(pcds.mean()):.1f} ms")
        print(f"  avg valid depth:  {float(valids.mean()):.3f}")
        if total_dropped:
            print(f"  pcd drops:        {total_dropped}")
        return dict(
            frames=frame_count,
            avg_fps=fps_avg,
            avg_read_ms=float(reads.mean()),
            avg_pcd_ms=float(pcds.mean()),
            avg_total_ms=avg_total,
            avg_valid_ratio=float(valids.mean()),
            drops=total_dropped,
        )
    return dict(
        frames=0,
        avg_fps=0,
        avg_read_ms=0,
        avg_pcd_ms=0,
        avg_total_ms=0,
        avg_valid_ratio=0,
        drops=0,
    )


def main() -> int:
    """Run the hardware diagnostic and return a process exit status."""
    test_cfg = RealSenseDiagnosticConfig()

    print("=" * 60)
    print("RealSense Test -- RGB-D Live Capture + Real-time Point Cloud")
    print("=" * 60)
    print(f"OpenCV       : {cv2.__version__}")
    print(f"NumPy        : {np.__version__}")
    try:
        import pyrealsense2 as rs  # noqa: F401

        print("pyrealsense2 : installed")
    except ImportError:
        print("pyrealsense2 : NOT INSTALLED")
        return 1

    cameras = _list_cameras()
    if not cameras:
        return 1

    if not _test_lifecycle(test_cfg):
        print("Lifecycle test failed, exiting.")
        return 1

    config = RealSenseConfig(
        depth_resolution=test_cfg.depth_resolution,
        color_resolution=test_cfg.color_resolution,
        fps=test_cfg.fps,
        warmup_frames=test_cfg.warmup_frames,
    )
    camera = RealSense(config)
    connect_ok = False
    for attempt in range(3):
        connect_ok = camera.connect()
        if connect_ok:
            break
        if attempt < 2:
            delay = 1.0 * (attempt + 1)
            print(f"  Connection failed, retrying in {delay:.0f}s...")
            time.sleep(delay)
    if not connect_ok:
        print("Camera connection failed (3 retries).")
        return 1
    print(f"Connected: {camera.get_device_info()}")

    # Capture the pre-test auto_exposure_priority so we can restore it on exit,
    # even if 'a' was toggled during the session.
    original_priority = _get_ae_priority(camera)
    if original_priority is not None:
        state_name = "OFF" if original_priority < 0.5 else "ON"
        print(
            f"  auto_exposure_priority (initial): {original_priority:g} [{state_name}] "
            "-- press 'a' to toggle"
        )

    result: dict = {}
    try:
        result = _run_rgbd_test(camera, test_cfg)
    finally:
        if original_priority is not None:
            try:
                restored = _set_ae_priority(camera, original_priority)
                print(f"  [restore] auto_exposure_priority -> {restored:g}")
            except (RuntimeError, OSError) as exc:
                print(f"  [restore] auto_exposure_priority FAILED: {exc}")
        camera.disconnect()
        print("disconnect OK")

    print("\n" + "=" * 60)
    print("Test complete")
    print(f"  Total frames:    {result.get('frames', 0)}")
    print(f"  Avg fps:         {result.get('avg_fps', 0):.1f}")
    print(f"  Avg latency:     {result.get('avg_total_ms', 0):.1f} ms")
    print(f"  PCD drops:       {result.get('drops', 0)}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
