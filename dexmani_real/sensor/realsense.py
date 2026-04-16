import time
import argparse
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np
import pyrealsense2 as rs

from dexmani_real.sensor.pcd_utils import get_pointcloud

try:
    from dexmani_real.sensor.pcd_utils import vis_point_cloud
except ImportError:
    vis_point_cloud = None


class RealSense:
    def __init__(
        self,
        serial: str,
        depth_resolution: Tuple[int, int] = (640, 480),
        color_resolution: Tuple[int, int] = (1280, 720),
        fps: int = 30,
        depth_hole_filling: bool = False,
        pose: Optional[np.ndarray] = None,   # camera -> world, 4x4
    ):
        self.serial = serial
        self.depth_resolution = depth_resolution
        self.color_resolution = color_resolution
        self.fps = fps
        self.depth_hole_filling = depth_hole_filling
        self.pose = pose

        self.pipeline: Optional[rs.pipeline] = None
        self.config: Optional[rs.config] = None
        self.aligner: Optional[rs.align] = None
        self.hole_filling = None

        self.depth_scale: Optional[float] = None
        self.intrinsics: Optional[np.ndarray] = None   # depth intrinsics, (3, 3)

    def start(self) -> None:
        if self.pipeline is not None:
            return

        dw, dh = self.depth_resolution
        cw, ch = self.color_resolution

        self.config = rs.config()
        self.config.enable_device(self.serial)
        self.config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, self.fps)

        self.pipeline = rs.pipeline()

        try:
            self.config.resolve(rs.pipeline_wrapper(self.pipeline))
        except RuntimeError as e:
            raise RuntimeError(
                "Failed to resolve RealSense stream config. "
                f"serial={self.serial}, "
                f"depth={dw}x{dh}@{self.fps}, "
                f"color={cw}x{ch}@{self.fps}"
            ) from e

        profile = self.pipeline.start(self.config)

        self.aligner = rs.align(rs.stream.depth)

        if self.depth_hole_filling:
            self.hole_filling = rs.hole_filling_filter(mode=2)

        self.depth_scale = (
            profile.get_device()
            .first_depth_sensor()
            .get_depth_scale()
        )

        di = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        self.intrinsics = np.array([
            [di.fx, 0.0, di.ppx],
            [0.0, di.fy, di.ppy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        try:
            profile.get_device().first_color_sensor().set_option(
                rs.option.global_time_enabled, 1
            )
        except Exception:
            pass

        for _ in range(5):
            self.pipeline.wait_for_frames()

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None

        self.config = None
        self.aligner = None
        self.hole_filling = None
        self.depth_scale = None
        self.intrinsics = None

    def __enter__(self) -> "RealSense":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def get_frame(self, timeout_ms: int = 5000) -> Tuple[np.ndarray, np.ndarray, float]:
        if self.pipeline is None or self.aligner is None:
            raise RuntimeError("RealSense is not started")

        frameset = self.pipeline.wait_for_frames(timeout_ms)
        frameset = self.aligner.process(frameset)

        depth_frame = frameset.get_depth_frame()
        color_frame = frameset.get_color_frame()

        if not depth_frame or not color_frame:
            raise RuntimeError("Failed to get aligned depth/color frame")

        if self.hole_filling is not None:
            depth_frame = self.hole_filling.process(depth_frame)

        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth *= self.depth_scale

        color_bgr = np.asanyarray(color_frame.get_data())
        color = np.ascontiguousarray(color_bgr[..., ::-1])   # RGB, contiguous

        timestamp = depth_frame.get_timestamp() / 1000.0
        return color, depth, timestamp

    def get_intrinsics(self) -> np.ndarray:
        if self.intrinsics is None:
            raise RuntimeError("RealSense is not started")
        return self.intrinsics.copy()

    def get_depth_scale(self) -> float:
        if self.depth_scale is None:
            raise RuntimeError("RealSense is not started")
        return float(self.depth_scale)

    def frames_to_pointcloud(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        *,
        bound: Optional[list[float]] = None,
        npoints: int = 1024,
        min_depth: float = 0.1,
        max_depth: float = 5.0,
        device: str = "cpu",
    ) -> Dict[str, np.ndarray]:
        return get_pointcloud(
            depth=depth,
            intr=self.get_intrinsics(),
            color=color,
            bound=bound,
            npoints=npoints,
            min_depth=min_depth,
            max_depth=max_depth,
            device=device,
            transform=self.pose,
        )

    def get_pointcloud(
        self,
        *,
        bound: Optional[list[float]] = None,
        npoints: int = 1024,
        min_depth: float = 0.1,
        max_depth: float = 5.0,
        device: str = "cpu",
    ) -> Dict[str, np.ndarray]:
        color, depth, _ = self.get_frame()
        return self.frames_to_pointcloud(
            color=color,
            depth=depth,
            bound=bound,
            npoints=npoints,
            min_depth=min_depth,
            max_depth=max_depth,
            device=device,
        )

    @staticmethod
    def list_cameras() -> List[Dict[str, str]]:
        ctx = rs.context()
        cameras = []
        for dev in ctx.query_devices():
            cameras.append({
                "serial": dev.get_info(rs.camera_info.serial_number),
                "name": dev.get_info(rs.camera_info.name),
                "firmware": dev.get_info(rs.camera_info.firmware_version),
            })
        return cameras


def _parse_resolution(text: str) -> Tuple[int, int]:
    w, h = text.lower().split("x")
    return int(w), int(h)


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _make_depth_vis(
    depth: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 1.5,
) -> np.ndarray:
    valid = (depth > 0.0) & np.isfinite(depth)

    depth_clip = np.clip(depth, min_depth, max_depth)
    depth_norm = (depth_clip - min_depth) / max(max_depth - min_depth, 1e-6)
    depth_u8 = (255.0 * (1.0 - depth_norm)).astype(np.uint8)

    depth_vis = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
    depth_vis[~valid] = 0
    return depth_vis


def example(
    serial: Optional[str] = None,
    depth_resolution: Tuple[int, int] = (640, 480),
    color_resolution: Tuple[int, int] = (1280, 720),
    fps: int = 30,
    npoints: int = 1024,
    bound: Optional[list[float]] = None,
    min_depth: float = 0.1,
    max_depth: float = 1.5,
    device: str = "cpu",
    depth_hole_filling: bool = False,
) -> None:
    cams = RealSense.list_cameras()
    if not cams:
        print("[ERROR] 未检测到 RealSense 相机")
        return

    print("检测到相机:")
    for cam in cams:
        print(f"  {cam['name']:30s} SN={cam['serial']}  FW={cam['firmware']}")

    serial = serial or cams[0]["serial"]
    print(f"\n使用设备: {serial}")
    print(f"depth: {depth_resolution[0]}x{depth_resolution[1]} @ {fps}fps")
    print(f"color: {color_resolution[0]}x{color_resolution[1]} @ {fps}fps")
    print("align: color -> depth")
    print(f"npoints: {npoints}, device={device}")
    if bound is not None:
        print(f"bound: {bound}")
    print()

    with RealSense(
        serial=serial,
        depth_resolution=depth_resolution,
        color_resolution=color_resolution,
        fps=fps,
        depth_hole_filling=depth_hole_filling,
    ) as cam:
        intr = cam.get_intrinsics()
        print("depth intrinsics:")
        print(f"  {intr[0]}")
        print(f"  {intr[1]}")
        print(f"  {intr[2]}")
        print(f"  depth_scale = {cam.get_depth_scale()}")
        print()
        print("按键: q 退出, p 查看当前点云")
        print()

        try:
            while True:
                t0 = time.perf_counter()
                color, depth, ts = cam.get_frame()
                grab_ms = (time.perf_counter() - t0) * 1000.0

                t1 = time.perf_counter()
                pcd = cam.frames_to_pointcloud(
                    color=color,
                    depth=depth,
                    bound=bound,
                    npoints=npoints,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    device=device,
                )
                pcd_ms = (time.perf_counter() - t1) * 1000.0

                num_points = int(pcd["pos"].shape[0])

                color_bgr = np.ascontiguousarray(color[..., ::-1])
                depth_vis = _make_depth_vis(depth, min_depth=min_depth, max_depth=max_depth)

                panel = np.concatenate([color_bgr, depth_vis], axis=1)
                panel = np.ascontiguousarray(panel)

                cv2.putText(
                    panel,
                    f"ts={ts:.3f}s  grab={grab_ms:.1f}ms  pcd={pcd_ms:.1f}ms",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    panel,
                    f"aligned_to_depth  points={num_points}  device={device}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("RealSense | RGB(left) Depth(right)", panel)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if key == ord("p"):
                    if vis_point_cloud is None:
                        print("[WARN] vis_point_cloud 不可用")
                    else:
                        pos = _to_numpy(pcd["pos"])
                        print(f"Visualizing point cloud with {pos.shape[0]} points...")
                        vis_point_cloud(pos, voxel_size=0.005)

        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RealSense RGBD example (color aligned to depth)")
    parser.add_argument("--serial", type=str, default=None, help="相机序列号")
    parser.add_argument("--fps", type=int, default=30, help="帧率")
    parser.add_argument("--depth-res", type=str, default="640x480", help="深度分辨率 WxH")
    parser.add_argument("--color-res", type=str, default="1280x720", help="彩色分辨率 WxH")
    parser.add_argument("--npoints", type=int, default=1024, help="点云下采样点数")
    parser.add_argument("--device", type=str, default="cpu", help="pointcloud device")
    parser.add_argument("--hole-filling", action="store_true", help="启用 depth hole filling")
    parser.add_argument(
        "--bound",
        type=float,
        nargs=6,
        default=None,
        help="workspace crop: x_min x_max y_min y_max z_min z_max",
    )
    parser.add_argument("--min-depth", type=float, default=0.1, help="最小深度")
    parser.add_argument("--max-depth", type=float, default=1.5, help="最大深度")

    args = parser.parse_args()

    example(
        serial=args.serial,
        depth_resolution=_parse_resolution(args.depth_res),
        color_resolution=_parse_resolution(args.color_res),
        fps=args.fps,
        npoints=args.npoints,
        bound=args.bound,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        device=args.device,
        depth_hole_filling=args.hole_filling,
    )