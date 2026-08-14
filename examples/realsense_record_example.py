#!/usr/bin/env python3
"""RealSense camera test -- RGB-D live capture, real-time point cloud, config comparison.

Usage::

    conda activate real_robot
    python examples/realsense_record_example.py

Test flow: enumerate -> lifecycle + intrinsics -> RGB-D live capture (cv2 window,
fps/latency/valid_ratio HUD) -> real-time point cloud (open3d non-blocking) ->
performance summary -> config variant comparison.

Keyboard controls::

    q / Esc    Quit
    p          Toggle point cloud display
    v          Cycle voxel downsample (5 mm -> 10 mm -> off)
    w          Toggle workspace crop
    s          Cycle sampling mode (random -> fps -> none)
    r          Reset to defaults
    c          Print current config
    d          Toggle depth colormap (jet <-> gray)
"""

from __future__ import annotations

import sys
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dexmani_real.sensor.realsense import CameraFrame, RealSense, RealSenseConfig
from dexmani_real.utils.pointcloud_utils import (
    PointCloudConfig,
    depth_valid_ratio,
    make_depth_vis,
    rgbd_to_pointcloud,
)

# ── Constants ──

# Point-cloud defaults.
_DEFAULT_PCD_NPOINTS = 1024
_DEFAULT_PCD_MIN_DEPTH = 0.05
_DEFAULT_PCD_MAX_DEPTH = 1.5
_DEFAULT_WORKSPACE: tuple[float, float, float, float, float, float] = (
    -0.3, -1.0, -0.5,  # xyz_min
    2.0, 1.0, 1.5,     # xyz_max
)

# Voxel cycle order for 'v' key.
_VOXEL_CYCLE: tuple[float | None, ...] = (None, 0.005, 0.01)

# Sampling mode cycle order for 's' key.
_SAMPLING_MODES: tuple[str, ...] = ("random", "fps", "none")

_WINDOW_NAME = "RealSense Test | RGB(left) Depth(right)"


# ═══════════════════════════════════════════════════════════════════════
# Configuration dataclasses
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CameraTestConfig:
    """Test parameters for camera connection and display."""

    fps: int = 30
    depth_resolution: tuple[int, int] = (640, 480)
    color_resolution: tuple[int, int] = (640, 480)
    warmup_frames: int = 10
    stats_window: int = 100  # rolling-average window (frames)


@dataclass
class PCDDisplayState:
    """Mutable point-cloud display state toggled via keyboard.

    Not frozen -- state is mutated in place by key handlers.
    """

    show: bool = False
    voxel_size: float | None = 0.005
    workspace_on: bool = True
    sampling_mode: str = "random"
    colormap: str = "jet"

    def cycle_voxel(self) -> None:
        idx = _VOXEL_CYCLE.index(self.voxel_size)
        self.voxel_size = _VOXEL_CYCLE[(idx + 1) % len(_VOXEL_CYCLE)]

    def cycle_sampling(self) -> None:
        idx = _SAMPLING_MODES.index(self.sampling_mode)
        self.sampling_mode = _SAMPLING_MODES[(idx + 1) % len(_SAMPLING_MODES)]

    def toggle_workspace(self) -> None:
        self.workspace_on = not self.workspace_on

    def toggle_colormap(self) -> None:
        self.colormap = "gray" if self.colormap == "jet" else "jet"

    def reset(self) -> None:
        self.show = False
        self.voxel_size = 0.005
        self.workspace_on = True
        self.sampling_mode = "random"
        self.colormap = "jet"


@dataclass
class FrameStats:
    """Per-frame timing and quality metrics."""

    read_ms: float = 0.0
    pcd_ms: float = 0.0
    valid_ratio: float = 0.0
    point_count: int = 0
    total_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════
# Point-cloud config builder
# ═══════════════════════════════════════════════════════════════════════

def _make_pcd_config(state: PCDDisplayState) -> PointCloudConfig:
    """Build PointCloudConfig from current display state."""
    return PointCloudConfig(
        npoints=_DEFAULT_PCD_NPOINTS,
        min_depth=_DEFAULT_PCD_MIN_DEPTH,
        max_depth=_DEFAULT_PCD_MAX_DEPTH,
        sampling=state.sampling_mode,  # type: ignore[arg-type]
        voxel_size=state.voxel_size,
        workspace=_DEFAULT_WORKSPACE if state.workspace_on else None,
        device="cpu",
        return_tensor=False,
    )


# ═══════════════════════════════════════════════════════════════════════
# Non-blocking point cloud viewer (open3d)
# ═══════════════════════════════════════════════════════════════════════

class NonBlockingPCDViewer:
    """Non-blocking open3d point-cloud window -- create on first frame, update thereafter."""

    def __init__(self, point_size: float = 3.0) -> None:
        self._vis = None
        self._pcd = None
        self._frame = None
        self._created = False
        self.point_size = point_size

    def update(self, points: np.ndarray) -> bool:
        """Update point cloud.  *points*: (N, 3) or (N, 6).  Returns False if window closed."""
        import open3d as o3d

        pcd_array = points.astype(np.float32)
        if pcd_array.ndim != 2 or pcd_array.shape[1] not in (3, 6):
            return self._created

        if self._pcd is None:
            self._pcd = o3d.geometry.PointCloud()
        self._pcd.points = o3d.utility.Vector3dVector(pcd_array[:, :3].astype(np.float64))

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
            self._vis.update_geometry(self._pcd)

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


# ═══════════════════════════════════════════════════════════════════════
# Display utilities
# ═══════════════════════════════════════════════════════════════════════

def _overlay_text(img: np.ndarray, lines: list[str],
                  x: int = 10, y_start: int = 22, step: int = 22) -> None:
    """Draw green HUD text lines on an image."""
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y_start + i * step),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)


def _make_gray_depth_vis(depth: np.ndarray, min_d: float, max_d: float) -> np.ndarray:
    """Grayscale depth visualization (near=white, far=black).

    Complements ``make_depth_vis`` which only produces jet colormap.
    """
    depth_f32 = depth.astype(np.float32)
    valid = np.isfinite(depth_f32) & (depth_f32 > 0)
    depth_clip = np.clip(depth_f32, min_d, max_d)
    depth_norm = (depth_clip - min_d) / max(max_d - min_d, 1e-6)
    vis = (255.0 * (1.0 - depth_norm)).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    vis[~valid] = 0
    return vis


# ═══════════════════════════════════════════════════════════════════════
# Step 0: enumerate cameras
# ═══════════════════════════════════════════════════════════════════════

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
        print(f"  [{i}] {cam.get('name', 'unknown'):28s} "
              f"SN={cam.get('serial', ''):16s} "
              f"FW={cam.get('firmware', '')}  "
              f"PL={cam.get('product_line', '')}")
    return cameras


# ═══════════════════════════════════════════════════════════════════════
# Step 1: lifecycle test
# ═══════════════════════════════════════════════════════════════════════

def _test_lifecycle(test_cfg: CameraTestConfig) -> bool:
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
        print(f"  connect() -> {ok}" + (f" (attempt {attempt + 1}/3)" if attempt > 0 else ""))
        if ok:
            break
        time.sleep(1.0)
    if not ok:
        print("  FAIL: connect failed after 3 retries")
        return False

    print(f"  connect() idempotent -> {camera.connect()}")

    K = camera.get_intrinsics()
    print(f"  K =\n{K}")
    print(f"  depth_scale = {camera.get_depth_scale()}")
    info = camera.get_intrinsics_info()
    print(f"  resolution = {info.get('width')}x{info.get('height')}  "
          f"fx={info.get('fx'):.1f} fy={info.get('fy'):.1f} "
          f"cx={info.get('cx'):.1f} cy={info.get('cy'):.1f}")

    camera.disconnect()
    print("  disconnect() -> OK")
    camera.disconnect()  # idempotent
    print("  disconnect() idempotent -> OK")
    print("  Lifecycle test passed")
    return True


# ═══════════════════════════════════════════════════════════════════════
# Step 2: RGB-D live capture + interactive point cloud
# ═══════════════════════════════════════════════════════════════════════

def _compute_rolling_stats(history: deque[FrameStats]) -> dict[str, float]:
    """Compute rolling averages from frame stats history."""
    if not history:
        return dict(fps=0, read_ms=0, pcd_ms=0, total_ms=0, valid=0, points=0)
    return dict(
        fps=1000.0 / max(np.mean([s.total_ms for s in history]), 0.001),
        read_ms=float(np.mean([s.read_ms for s in history])),
        pcd_ms=float(np.mean([s.pcd_ms for s in history])),
        total_ms=float(np.mean([s.total_ms for s in history])),
        valid=float(np.mean([s.valid_ratio for s in history])),
        points=float(np.mean([s.point_count for s in history])),
    )


def _build_hud_lines(frame: "CameraFrame", frame_count: int, stats: dict[str, float],
                     state: PCDDisplayState, total_dropped: int) -> list[str]:
    """Build HUD overlay text lines."""
    lines = [
        f"frame={frame_count}  fps={stats['fps']:.1f}  total={stats['total_ms']:.1f}ms",
        f"read={stats['read_ms']:.1f}ms  pcd={stats['pcd_ms']:.1f}ms  valid={stats['valid']:.3f}",
        f"align={frame.align_mode}  K=[{frame.K[0,0]:.0f},{frame.K[1,1]:.0f},"
        f"{frame.K[0,2]:.0f},{frame.K[1,2]:.0f}]",
    ]
    if state.show:
        lines.append(f"PCD ON  n={stats['points']:.0f}  vox={state.voxel_size}  "
                     f"ws={state.workspace_on}  samp={state.sampling_mode}  drop={total_dropped}")
    else:
        lines.append("PCD OFF  [p]toggle [v]voxel [w]workspace [s]sampling [r]reset")
    lines.append(f"cmap={state.colormap}  depth=[{_DEFAULT_PCD_MIN_DEPTH:.2f},{_DEFAULT_PCD_MAX_DEPTH:.2f}]")
    return lines


# ── Keyboard action handlers ──

def _build_key_actions(state: PCDDisplayState, viewer: NonBlockingPCDViewer) -> dict[int, Callable[[], None]]:
    """Build the key-action dispatch table for the current loop state."""
    return {
        ord("p"): lambda: _toggle_pcd(state, viewer),
        ord("v"): lambda: _cycle_voxel(state),
        ord("w"): lambda: _toggle_ws(state),
        ord("s"): lambda: _cycle_sampling(state),
        ord("r"): lambda: _reset_state(state, viewer),
        ord("c"): lambda: _print_state(state),
        ord("d"): lambda: _toggle_cmap(state),
    }


def _handle_keyboard(key: int, state: PCDDisplayState, viewer: NonBlockingPCDViewer) -> bool:
    """Process keyboard input.  Returns False if quit requested."""
    if key in (ord("q"), 27):
        return False

    action = _build_key_actions(state, viewer).get(key)
    if action:
        action()
    return True


def _toggle_pcd(state: PCDDisplayState, viewer: NonBlockingPCDViewer) -> None:
    state.show = not state.show
    if not state.show:
        viewer.close()
    print(f"  Point cloud: {'ON' if state.show else 'OFF'}")


def _cycle_voxel(state: PCDDisplayState) -> None:
    state.cycle_voxel()
    print(f"  Voxel: {state.voxel_size}")


def _toggle_ws(state: PCDDisplayState) -> None:
    state.toggle_workspace()
    print(f"  Workspace: {'ON' if state.workspace_on else 'OFF'}")


def _cycle_sampling(state: PCDDisplayState) -> None:
    state.cycle_sampling()
    print(f"  Sampling: {state.sampling_mode}")


def _reset_state(state: PCDDisplayState, viewer: NonBlockingPCDViewer) -> None:
    state.reset()
    viewer.close()
    print("  Config reset to defaults")


def _print_state(state: PCDDisplayState) -> None:
    pcd = _make_pcd_config(state)
    print(f"  npoints={pcd.npoints}  sampling={pcd.sampling}  voxel={pcd.voxel_size}  "
          f"depth=[{pcd.min_depth},{pcd.max_depth}]  workspace={pcd.workspace}  "
          f"pcd={'ON' if state.show else 'OFF'}")


def _toggle_cmap(state: PCDDisplayState) -> None:
    state.toggle_colormap()
    print(f"  Colormap: {state.colormap}")


def _run_rgbd_test(camera: RealSense, test_cfg: CameraTestConfig) -> dict:
    """Run interactive RGB-D live capture + point cloud visualization."""
    print("\n-- 2. RGB-D live capture + point cloud --")
    print("   q/Esc=quit  p=pcd  v=voxel  w=workspace  s=sampling  r=reset  c=config  d=colormap")

    state = PCDDisplayState()
    viewer = NonBlockingPCDViewer(point_size=3.0)
    stats_history: deque[FrameStats] = deque(maxlen=test_cfg.stats_window)
    frame_count = 0
    total_dropped = 0

    pcd_config = _make_pcd_config(state)
    print(f"  Start: npoints={pcd_config.npoints}  sampling={pcd_config.sampling}  "
          f"depth=[{pcd_config.min_depth}, {pcd_config.max_depth}]  "
          f"voxel={pcd_config.voxel_size}  workspace={state.workspace_on}")

    while True:
        loop_start = time.perf_counter()

        # Read frame.
        t0 = time.perf_counter()
        try:
            frame = camera.read(timeout_ms=5000)
        except RuntimeError as e:
            print(f"  read() failed: {e}")
            break
        read_ms = (time.perf_counter() - t0) * 1000.0
        frame_count += 1

        # Point cloud generation.
        pcd_ms = 0.0
        point_count = 0
        pcd_array = None
        if state.show:
            t0 = time.perf_counter()
            try:
                pcd_config = _make_pcd_config(state)
                pcd_array = rgbd_to_pointcloud(depth=frame.depth, K=frame.K,
                                              rgb=frame.rgb, config=pcd_config)
                point_count = pcd_array.shape[0]
            except ValueError as e:
                total_dropped += 1
                if total_dropped <= 3:
                    print(f"  PCD generation failed (frame {frame_count}): {e}")
            pcd_ms = (time.perf_counter() - t0) * 1000.0

        # Depth visualization.
        if state.colormap == "jet":
            depth_vis = make_depth_vis(frame.depth, _DEFAULT_PCD_MIN_DEPTH, _DEFAULT_PCD_MAX_DEPTH)
        else:
            depth_vis = _make_gray_depth_vis(frame.depth, _DEFAULT_PCD_MIN_DEPTH, _DEFAULT_PCD_MAX_DEPTH)

        valid_ratio = depth_valid_ratio(frame.depth, _DEFAULT_PCD_MIN_DEPTH, _DEFAULT_PCD_MAX_DEPTH)

        # Panel assembly.
        if frame.rgb is not None:
            color_bgr = np.ascontiguousarray(frame.rgb[..., ::-1])
            if color_bgr.shape[:2] != depth_vis.shape[:2]:
                depth_vis = cv2.resize(depth_vis, (color_bgr.shape[1], color_bgr.shape[0]))
            panel = np.concatenate([color_bgr, depth_vis], axis=1)
        else:
            panel = depth_vis

        total_ms = (time.perf_counter() - loop_start) * 1000.0
        stats_history.append(FrameStats(
            read_ms=read_ms, pcd_ms=pcd_ms, valid_ratio=valid_ratio,
            point_count=point_count, total_ms=total_ms,
        ))

        stats = _compute_rolling_stats(stats_history)
        lines = _build_hud_lines(frame, frame_count, stats, state, total_dropped)
        _overlay_text(panel, lines)
        cv2.imshow(_WINDOW_NAME, panel)

        # Point cloud window update.
        if state.show and pcd_array is not None and pcd_array.shape[0] > 0:
            if not viewer.update(pcd_array):
                state.show = False
                viewer.close()
                print("  Point cloud window closed")

        # Keyboard.
        if not _handle_keyboard(cv2.waitKey(1) & 0xFF, state, viewer):
            break

    viewer.close()
    cv2.destroyAllWindows()

    # Performance summary.
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
        print(f"  avg read(grab):   {float(reads.mean()):.1f} ms  (max {reads.max():.1f} ms)")
        print(f"  avg pcd:          {float(pcds.mean()):.1f} ms")
        print(f"  avg valid depth:  {float(valids.mean()):.3f}")
        if total_dropped:
            print(f"  pcd drops:        {total_dropped}")
        return dict(
            frames=frame_count, avg_fps=fps_avg, avg_read_ms=float(reads.mean()),
            avg_pcd_ms=float(pcds.mean()), avg_total_ms=avg_total,
            avg_valid_ratio=float(valids.mean()), drops=total_dropped,
        )
    return dict(frames=0, avg_fps=0, avg_read_ms=0, avg_pcd_ms=0,
                avg_total_ms=0, avg_valid_ratio=0, drops=0)


# ═══════════════════════════════════════════════════════════════════════
# Step 3: config variant comparison
# ═══════════════════════════════════════════════════════════════════════

def _build_variants() -> list[tuple[str, PointCloudConfig]]:
    """Build the set of point-cloud config variants for comparison."""
    base = dict(min_depth=0.05, max_depth=1.5, return_tensor=False)
    return [
        ("random 1024 (default)",
         PointCloudConfig(npoints=1024, sampling="random", **base)),  # type: ignore[arg-type]
        ("voxel 5mm + fps 1024",
         PointCloudConfig(npoints=1024, sampling="fps", voxel_size=0.005, **base)),  # type: ignore[arg-type]
        ("voxel 10mm + fps 1024",
         PointCloudConfig(npoints=1024, sampling="fps", voxel_size=0.01, **base)),  # type: ignore[arg-type]
        ("no sampling (full)",
         PointCloudConfig(sampling="none", **base)),  # type: ignore[arg-type]
        ("fps 2048 (no voxel)",
         PointCloudConfig(npoints=2048, sampling="fps", **base)),  # type: ignore[arg-type]
        ("narrow depth [0.1, 0.8]m",
         PointCloudConfig(npoints=1024, sampling="random", min_depth=0.1, max_depth=0.8,
                          return_tensor=False)),  # type: ignore[arg-type]
        ("far depth [0.5, 2.0]m",
         PointCloudConfig(npoints=1024, sampling="random", min_depth=0.5, max_depth=2.0,
                          return_tensor=False)),  # type: ignore[arg-type]
        ("with workspace crop",
         PointCloudConfig(npoints=1024, sampling="random", workspace=_DEFAULT_WORKSPACE, **base)),  # type: ignore[arg-type]
        ("dense random 4096",
         PointCloudConfig(npoints=4096, sampling="random", **base)),  # type: ignore[arg-type]
    ]


def _run_pcd_variants(camera: RealSense) -> None:
    """Compare point-cloud config variants on one frame."""
    print("\n-- 3. Point cloud config variant comparison --")

    try:
        frame = camera.read()
    except RuntimeError as e:
        print(f"  camera.read() failed: {e}")
        print("  Skipping variant comparison (camera unavailable)")
        return

    for label, cfg in _build_variants():
        t0 = time.perf_counter()
        try:
            pcd = rgbd_to_pointcloud(depth=frame.depth, K=frame.K, rgb=frame.rgb, config=cfg)
            elapsed = (time.perf_counter() - t0) * 1000.0
            print(f"  {label:30s} -> {pcd.shape[0]:6d} pts  {elapsed:.1f}ms")
        except ValueError as e:
            print(f"  {label:30s} -> ERROR: {e}")

    print("  Variant comparison complete")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    test_cfg = CameraTestConfig()

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
        sys.exit(1)

    # 0. Enumerate.
    cameras = _list_cameras()
    if not cameras:
        sys.exit(1)

    # 1. Lifecycle.
    if not _test_lifecycle(test_cfg):
        print("Lifecycle test failed, exiting.")
        sys.exit(1)

    # 2. Main test.
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
        print(f"  Connection failed, retrying in {1.0 * (attempt + 1):.0f}s...")
        time.sleep(1.0)
    if not connect_ok:
        print("Camera connection failed (3 retries).")
        sys.exit(1)
    print(f"Connected: {camera.get_device_info()}")

    result: dict = {}
    try:
        result = _run_rgbd_test(camera, test_cfg)
        _run_pcd_variants(camera)
    finally:
        camera.disconnect()
        print("disconnect OK")

    print("\n" + "=" * 60)
    print("Test complete")
    print(f"  Total frames:    {result.get('frames', 0)}")
    print(f"  Avg fps:         {result.get('avg_fps', 0):.1f}")
    print(f"  Avg latency:     {result.get('avg_total_ms', 0):.1f} ms")
    print(f"  PCD drops:       {result.get('drops', 0)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
