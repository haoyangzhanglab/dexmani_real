#!/usr/bin/env python3
"""RealSense 相机测试 — RGB-D 实时采集 + 点云实时生成。

用法:
    conda activate real
    python test_realsense.py

测试流程:
    0. 列出可用相机
    1. connect/disconnect 生命周期 + 内参验证
    2. RGB-D 实时采集（cv2 窗口，含 fps/latency/valid_ratio HUD）
    3. 实时点云生成（open3d 非阻塞窗口，可切换降采样/采样模式/workspace）
    4. 退出时打印性能汇总
    5. 点云配置变体对比（各采一帧，对比点数与耗时）

键盘控制:
    q / Esc    退出
    p          切换点云显示（开/关）
    v          切换点云降采样 (0.005m → 0.01m → 关闭)
    w          切换 workspace 裁剪 (开/关)
    s          切换点云采样模式 (random → fps → none)
    r          重置点云配置为默认值
    c          打印当前配置
    d          切换深度色彩映射 (jet → gray)
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import time
from collections import deque
from dataclasses import dataclass

import cv2
import open3d as o3d
import numpy as np

from dexmani_real.sensor.realsense import RealSense, RealSenseConfig
from dexmani_real.utils.pointcloud_utils import (
    PointCloudConfig,
    depth_valid_ratio,
    make_depth_vis,
    rgbd_to_pointcloud,
)

# ═══════════════════════════════════════════════ 配置

FPS = 30
DEPTH_RESOLUTION = (640, 480)
COLOR_RESOLUTION = (640, 480)
WARMUP_FRAMES = 10

# 点云默认配置
DEFAULT_PCD_NPOINTS = 1024
DEFAULT_PCD_MIN_DEPTH = 0.05
DEFAULT_PCD_MAX_DEPTH = 1.5
DEFAULT_PCD_SAMPLING: str = "random"
DEFAULT_PCD_WORKSPACE: tuple[float, float, float, float, float, float] = (
    -0.3, -1.0, -0.5,   # x_min, y_min, z_min
     2.0,  1.0,  1.5,   # x_max, y_max, z_max
)

# 显示
WINDOW_NAME = "RealSense Test | RGB(left) Depth(right)"
STATS_WINDOW_SIZE = 100


# ═══════════════════════════════════════════════ open3d 非阻塞点云窗口

class NonBlockingPCDViewer:
    """open3d 非阻塞点云窗口 — 首帧创建窗口，后续帧更新几何体。

    点云数据已经过 pipeline 处理（体素降采样 + FPS），
    viewer 只负责渲染，不再做额外降采样。
    """

    def __init__(self, point_size: float = 3.0):
        self._vis: "o3d.visualization.Visualizer | None" = None
        self._pcd: "o3d.geometry.PointCloud | None" = None
        self._frame: "o3d.geometry.TriangleMesh | None" = None
        self._created = False
        self.point_size = point_size

    @property
    def is_open(self) -> bool:
        return self._created

    def update(self, points: np.ndarray) -> bool:
        """更新点云数据。points: (N, 3) 或 (N, 6) [xyz | xyzrgb]。

        返回 False 表示窗口已关闭。
        """
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


# ═══════════════════════════════════════════════ 工具


def overlay_text(img: np.ndarray, lines: list[str], x: int = 10, y_start: int = 22, step: int = 22) -> None:
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y_start + i * step), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════ 步骤 0: 枚举相机


def list_available_cameras() -> list[dict[str, str]]:
    for attempt in range(3):
        try:
            cameras = RealSense.list_cameras()
        except RuntimeError as e:
            if attempt < 2:
                print(f"  枚举相机失败 ({e})，{1.0 * (attempt + 1):.0f}s 后重试...")
                time.sleep(1.0)
                continue
            print(f"  枚举相机失败: {e}")
            return []
        break

    if not cameras:
        print("未检测到 RealSense 相机。请检查 USB 连接。")
        return []
    print(f"检测到 {len(cameras)} 个 RealSense 相机:")
    for i, cam in enumerate(cameras):
        print(f"  [{i}] {cam.get('name', 'unknown'):28s} "
              f"SN={cam.get('serial', ''):16s} "
              f"FW={cam.get('firmware', '')} "
              f"PL={cam.get('product_line', '')}")
    return cameras


# ═══════════════════════════════════════════════ 步骤 1: 生命周期测试


def test_lifecycle() -> bool:
    print("\n── 1. connect/disconnect 生命周期 ──")

    config = RealSenseConfig(
        depth_resolution=DEPTH_RESOLUTION,
        color_resolution=COLOR_RESOLUTION,
        fps=FPS,
        warmup_frames=WARMUP_FRAMES,
    )
    camera = RealSense(config)

    # 带重试的 connect（L515 偶发 USB 状态不稳定）
    ok = False
    for attempt in range(3):
        ok = camera.connect()
        print(f"  connect() → {ok}" + (f" (attempt {attempt + 1}/3)" if attempt > 0 else ""))
        if ok:
            break
        time.sleep(1.0)
    if not ok:
        print("  ❌ connect 失败（重试 3 次后仍失败）")
        return False

    ok2 = camera.connect()
    print(f"  connect() 幂等 → {ok2}")

    K = camera.get_intrinsics()
    print(f"  K =\n{K}")
    print(f"  depth_scale = {camera.get_depth_scale()}")
    info = camera.get_intrinsics_info()
    print(f"  resolution = {info.get('width')}x{info.get('height')}  "
          f"fx={info.get('fx'):.1f} fy={info.get('fy'):.1f} "
          f"cx={info.get('cx'):.1f} cy={info.get('cy'):.1f}")

    camera.disconnect()
    print("  disconnect() → OK")

    camera.disconnect()
    print("  disconnect() 幂等 → OK")

    print("  生命周期测试通过")
    return True


# ═══════════════════════════════════════════════ 步骤 2: RGB-D 实时采集 + 点云


@dataclass
class FrameStats:
    read_ms: float = 0.0
    pcd_ms: float = 0.0
    valid_ratio: float = 0.0
    point_count: int = 0
    total_ms: float = 0.0


def _make_gray_depth_vis(depth: np.ndarray, min_d: float, max_d: float) -> np.ndarray:
    depth_m = depth.astype(np.float32)
    valid = np.isfinite(depth_m) & (depth_m > 0)
    depth_clip = np.clip(depth_m, min_d, max_d)
    depth_norm = (depth_clip - min_d) / max(max_d - min_d, 1e-6)
    vis = (255.0 * (1.0 - depth_norm)).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    vis[~valid] = 0
    return vis


def _build_pcd_config(base: PointCloudConfig, *, workspace: tuple | None,
                      sampling: str, voxel_size: float | None) -> PointCloudConfig:
    return PointCloudConfig(
        npoints=base.npoints,
        min_depth=base.min_depth,
        max_depth=base.max_depth,
        sampling=sampling,  # type: ignore[arg-type]
        voxel_size=voxel_size,
        workspace=workspace,
        device="cpu",
        return_tensor=False,
    )


def run_rgbd_test(camera: RealSense, pcd_config: PointCloudConfig) -> dict:
    print("\n── 2. RGB-D 实时采集 + 点云生成 ──")
    print("   q/Esc=退出 p=点云 v=体素降采样 w=workspace s=采样 r=重置 c=配置 d=色彩")

    show_pointcloud = False
    vox_size: float | None = 0.005
    workspace_on = True
    colormap: str = "jet"
    sample_modes = ["random", "fps", "none"]
    sample_idx = 0

    stats_history: deque[FrameStats] = deque(maxlen=STATS_WINDOW_SIZE)
    frame_count = 0
    total_dropped = 0
    pcd_viewer = NonBlockingPCDViewer(point_size=3.0)

    current_pcd_config = _build_pcd_config(
        pcd_config,
        workspace=DEFAULT_PCD_WORKSPACE if workspace_on else None,
        sampling=sample_modes[sample_idx],
        voxel_size=vox_size,
    )

    print(f"  起始: npoints={current_pcd_config.npoints}  "
          f"sampling={current_pcd_config.sampling}  "
          f"depth=[{current_pcd_config.min_depth}, {current_pcd_config.max_depth}]  "
          f"voxel={current_pcd_config.voxel_size}  "
          f"workspace={workspace_on}")

    while True:
        loop_start = time.perf_counter()

        # ── 读取帧 ──
        t0 = time.perf_counter()
        try:
            frame = camera.read(timeout_ms=1000)
        except RuntimeError as e:
            print(f"  read() 失败: {e}")
            break
        read_ms = (time.perf_counter() - t0) * 1000.0
        frame_count += 1

        # ── 点云生成 ──
        pcd_ms = 0.0
        point_count = 0
        pcd_array = None
        if show_pointcloud:
            t0 = time.perf_counter()
            try:
                pcd_array = rgbd_to_pointcloud(
                    depth=frame.depth, K=frame.K, rgb=frame.rgb,
                    config=current_pcd_config,
                )
                point_count = pcd_array.shape[0]
            except ValueError as e:
                total_dropped += 1
                if total_dropped <= 3:
                    print(f"  点云生成失败 (frame {frame_count}): {e}")
            pcd_ms = (time.perf_counter() - t0) * 1000.0

        # ── 深度可视化 ──
        min_d = current_pcd_config.min_depth or 0.05
        max_d = current_pcd_config.max_depth or 1.5
        if colormap == "jet":
            depth_vis = make_depth_vis(frame.depth, min_d, max_d)
        else:
            depth_vis = _make_gray_depth_vis(frame.depth, min_d, max_d)

        valid_ratio = depth_valid_ratio(frame.depth, min_d, max_d)

        # ── 拼接面板 ──
        if frame.rgb is not None:
            color_bgr = np.ascontiguousarray(frame.rgb[..., ::-1])
            if color_bgr.shape[:2] != depth_vis.shape[:2]:
                depth_vis = cv2.resize(depth_vis, (color_bgr.shape[1], color_bgr.shape[0]))
            panel = np.concatenate([color_bgr, depth_vis], axis=1)
        else:
            panel = depth_vis

        total_ms = (time.perf_counter() - loop_start) * 1000.0
        stats_history.append(FrameStats(
            read_ms=read_ms, pcd_ms=pcd_ms,
            valid_ratio=valid_ratio, point_count=point_count,
            total_ms=total_ms,
        ))

        # ── 滑动统计 ──
        avg_read = float(np.mean([s.read_ms for s in stats_history])) if stats_history else 0.0
        avg_pcd = float(np.mean([s.pcd_ms for s in stats_history])) if stats_history else 0.0
        avg_total = float(np.mean([s.total_ms for s in stats_history])) if stats_history else 0.0
        avg_fps = 1000.0 / avg_total if avg_total > 0 else 0.0
        avg_valid = float(np.mean([s.valid_ratio for s in stats_history])) if stats_history else 0.0
        avg_points = float(np.mean([s.point_count for s in stats_history])) if (stats_history and show_pointcloud) else 0.0

        # ── HUD ──
        lines = [
            f"frame={frame_count}  fps={avg_fps:.1f}  total={avg_total:.1f}ms",
            f"read={avg_read:.1f}ms  pcd={avg_pcd:.1f}ms  valid={avg_valid:.3f}",
            f"align={frame.align_mode}  K=[{frame.K[0,0]:.0f},{frame.K[1,1]:.0f},"
            f"{frame.K[0,2]:.0f},{frame.K[1,2]:.0f}]",
        ]
        if show_pointcloud:
            lines.append(f"PCD ON  n={avg_points:.0f}  vox={current_pcd_config.voxel_size}  "
                         f"ws={workspace_on}  samp={current_pcd_config.sampling}  "
                         f"drop={total_dropped}")
        else:
            lines.append("PCD OFF  [p]toggle [v]voxel [w]workspace [s]sampling [r]reset")
        lines.append(f"cmap={colormap}  "
                     f"depth=[{min_d:.2f},{max_d:.2f}]")
        overlay_text(panel, lines)

        cv2.imshow(WINDOW_NAME, panel)

        # ── 点云窗口更新 ──
        if show_pointcloud and pcd_array is not None and pcd_array.shape[0] > 0:
            if not pcd_viewer.update(pcd_array):
                show_pointcloud = False
                pcd_viewer.close()
                print("  点云窗口已关闭")

        # ── 键盘 ──
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or Esc
            break

        if key == ord("p"):
            show_pointcloud = not show_pointcloud
            if not show_pointcloud:
                pcd_viewer.close()
            print(f"  点云显示: {'ON' if show_pointcloud else 'OFF'}")

        elif key == ord("v"):
            if vox_size is None:
                vox_size = 0.005
            elif vox_size == 0.005:
                vox_size = 0.01
            else:
                vox_size = None
            current_pcd_config = _build_pcd_config(
                current_pcd_config,
                workspace=current_pcd_config.workspace,
                sampling=current_pcd_config.sampling,
                voxel_size=vox_size,
            )
            print(f"  体素降采样: {vox_size}")

        elif key == ord("w"):
            workspace_on = not workspace_on
            current_pcd_config = _build_pcd_config(
                current_pcd_config,
                workspace=DEFAULT_PCD_WORKSPACE if workspace_on else None,
                sampling=current_pcd_config.sampling,
                voxel_size=current_pcd_config.voxel_size,
            )
            print(f"  workspace: {'ON' if workspace_on else 'OFF'}")

        elif key == ord("s"):
            sample_idx = (sample_idx + 1) % 3
            current_pcd_config = _build_pcd_config(
                current_pcd_config,
                workspace=current_pcd_config.workspace,
                sampling=sample_modes[sample_idx],
                voxel_size=current_pcd_config.voxel_size,
            )
            print(f"  采样模式: {current_pcd_config.sampling}")

        elif key == ord("r"):
            workspace_on = True
            vox_size = 0.005
            sample_idx = 0
            show_pointcloud = False
            pcd_viewer.close()
            current_pcd_config = PointCloudConfig(
                npoints=DEFAULT_PCD_NPOINTS,
                min_depth=DEFAULT_PCD_MIN_DEPTH,
                max_depth=DEFAULT_PCD_MAX_DEPTH,
                sampling=DEFAULT_PCD_SAMPLING,  # type: ignore[arg-type]
                voxel_size=vox_size,
                workspace=DEFAULT_PCD_WORKSPACE,
                device="cpu",
                return_tensor=False,
            )
            print("  配置已重置为默认值")

        elif key == ord("c"):
            print(f"  npoints={current_pcd_config.npoints}  "
                  f"sampling={current_pcd_config.sampling}  "
                  f"voxel={current_pcd_config.voxel_size}  "
                  f"depth=[{current_pcd_config.min_depth},{current_pcd_config.max_depth}]  "
                  f"workspace={current_pcd_config.workspace}  "
                  f"pcd={'ON' if show_pointcloud else 'OFF'}")

        elif key == ord("d"):
            colormap = "gray" if colormap == "jet" else "jet"
            print(f"  深度色彩: {colormap}")

    pcd_viewer.close()
    cv2.destroyAllWindows()

    # ── 性能汇总 ──
    if stats_history:
        avg_read = float(np.mean([s.read_ms for s in stats_history]))
        avg_pcd = float(np.mean([s.pcd_ms for s in stats_history]))
        avg_total = float(np.mean([s.total_ms for s in stats_history]))
        avg_valid = float(np.mean([s.valid_ratio for s in stats_history]))
        fps_avg = 1000.0 / avg_total if avg_total > 0 else 0.0
        read_max = float(np.max([s.read_ms for s in stats_history]))
        total_max = float(np.max([s.total_ms for s in stats_history]))

        print(f"\n  ── 性能汇总 ({frame_count} 帧) ──")
        print(f"  avg fps:          {fps_avg:.1f}")
        print(f"  avg frame total:  {avg_total:.1f} ms  (max {total_max:.1f} ms)")
        print(f"  avg read(grab):   {avg_read:.1f} ms  (max {read_max:.1f} ms)")
        print(f"  avg pcd:          {avg_pcd:.1f} ms")
        print(f"  avg valid depth:  {avg_valid:.3f}")
        if total_dropped:
            print(f"  pcd drops:        {total_dropped}")
    else:
        avg_read = avg_pcd = avg_total = avg_valid = fps_avg = 0.0

    return {
        "frames": frame_count,
        "avg_fps": fps_avg,
        "avg_read_ms": avg_read,
        "avg_pcd_ms": avg_pcd,
        "avg_total_ms": avg_total,
        "avg_valid_ratio": avg_valid,
        "drops": total_dropped,
    }


# ═══════════════════════════════════════════════ 步骤 3: 点云配置变体对比


def run_pcd_variants(camera: RealSense) -> None:
    print("\n── 3. 点云配置变体对比 ──")

    frame = camera.read()

    Variant = tuple[str, PointCloudConfig]  # type: ignore[no-redef]

    variants = [
    ("random 1024 (default)", PointCloudConfig(
        npoints=1024, sampling="random", min_depth=0.05, max_depth=1.5,
        return_tensor=False)),
    ("voxel 5mm + fps 1024", PointCloudConfig(
        npoints=1024, sampling="fps", voxel_size=0.005,
        min_depth=0.05, max_depth=1.5, return_tensor=False)),
    ("voxel 10mm + fps 1024", PointCloudConfig(
        npoints=1024, sampling="fps", voxel_size=0.01,
        min_depth=0.05, max_depth=1.5, return_tensor=False)),
    ("no sampling (full)", PointCloudConfig(
        sampling="none", min_depth=0.05, max_depth=1.5,
        return_tensor=False)),
    ("fps 2048 (no voxel)", PointCloudConfig(
        npoints=2048, sampling="fps", min_depth=0.05, max_depth=1.5,
        return_tensor=False)),
    ("narrow depth [0.1, 0.8]m", PointCloudConfig(
        npoints=1024, sampling="random", min_depth=0.1, max_depth=0.8,
        return_tensor=False)),
    ("far depth [0.5, 2.0]m", PointCloudConfig(
        npoints=1024, sampling="random", min_depth=0.5, max_depth=2.0,
        return_tensor=False)),
    ("with workspace crop", PointCloudConfig(
        npoints=1024, sampling="random", min_depth=0.05, max_depth=1.5,
        workspace=DEFAULT_PCD_WORKSPACE, return_tensor=False)),
    ("dense random 4096", PointCloudConfig(
        npoints=4096, sampling="random", min_depth=0.05, max_depth=1.5,
        return_tensor=False)),
]

    for label, cfg in variants:
        t0 = time.perf_counter()
        try:
            pcd = rgbd_to_pointcloud(
                depth=frame.depth, K=frame.K, rgb=frame.rgb, config=cfg,
            )
            elapsed = (time.perf_counter() - t0) * 1000.0
            print(f"  {label:30s} → {pcd.shape[0]:6d} pts  {elapsed:.1f}ms")
        except ValueError as e:
            print(f"  {label:30s} → ERROR: {e}")

    print("  变体对比完成")


# ═══════════════════════════════════════════════ main


def main() -> None:
    print("=" * 60)
    print("RealSense 测试 — RGB-D 实时采集 + 点云实时生成")
    print("=" * 60)
    print(f"OpenCV       : {cv2.__version__}")
    print(f"NumPy        : {np.__version__}")
    try:
        import pyrealsense2 as rs  # noqa: F401
        print("pyrealsense2 : installed")
    except ImportError:
        print("pyrealsense2 : NOT INSTALLED")
        sys.exit(1)

    # ── 0. 枚举 ──
    cameras = list_available_cameras()
    if not cameras:
        sys.exit(1)

    # ── 1. 生命周期 ──
    if not test_lifecycle():
        print("生命周期测试失败，退出。")
        sys.exit(1)

    # ── 2. 主测试 ──
    config = RealSenseConfig(
        depth_resolution=DEPTH_RESOLUTION,
        color_resolution=COLOR_RESOLUTION,
        fps=FPS,
        warmup_frames=WARMUP_FRAMES,
    )
    pcd_config = PointCloudConfig(
        npoints=DEFAULT_PCD_NPOINTS,
        min_depth=DEFAULT_PCD_MIN_DEPTH,
        max_depth=DEFAULT_PCD_MAX_DEPTH,
        sampling=DEFAULT_PCD_SAMPLING,  # type: ignore[arg-type]
        voxel_size=0.005,
        workspace=DEFAULT_PCD_WORKSPACE,
        device="cpu",
        return_tensor=False,
    )

    camera = RealSense(config)
    connect_ok = False
    for attempt in range(3):
        connect_ok = camera.connect()
        if connect_ok:
            break
        print(f"  连接失败，{1.0 * (attempt + 1):.0f}s 后重试...")
        time.sleep(1.0)
    if not connect_ok:
        print("相机连接失败（重试 3 次）。")
        sys.exit(1)
    print(f"连接成功: {camera.get_device_info()}")

    result = {}
    try:
        result = run_rgbd_test(camera, pcd_config)
        run_pcd_variants(camera)
    finally:
        camera.disconnect()
        print("disconnect ✓")

    print("\n" + "=" * 60)
    print("测试完成")
    print(f"  总帧数:    {result.get('frames', 0)}")
    print(f"  平均 fps:  {result.get('avg_fps', 0):.1f}")
    print(f"  平均延迟:  {result.get('avg_total_ms', 0):.1f} ms")
    print(f"  pcd丢帧:   {result.get('drops', 0)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
