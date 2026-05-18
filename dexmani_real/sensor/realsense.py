import cv2
import time
import numpy as np
import argparse
import pyrealsense2 as rs
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dexmani_real.sensor.pcd_utils import (
    blend_overlay,
    check_transform,
    depth_meters_to_mm,
    depth_valid_ratio,
    intrinsics_to_dict,
    intrinsics_to_matrix,
    make_depth_vis,
    make_rays,
    pack_obs,
    parse_resolution,
    rgbd_to_pointcloud,
    vis_point_cloud,
)


class RealSense:
    """Minimal RealSense RGB-D camera wrapper.

    Coordinate convention:
    - align_to='depth': color is aligned to depth, K is depth intrinsics.
    - align_to='color': depth is aligned to color, K is color intrinsics.
    - align_to='none': no SDK alignment, K is depth intrinsics.

    Point cloud convention:
    - If T_out_camera is None, point clouds are in current camera frame.
    - If T_out_camera is provided, point clouds are transformed before workspace crop.
      Therefore workspace should be expressed in the output/world/base frame.
    """

    def __init__(
        self,
        serial: Optional[str] = None,
        depth_resolution: Tuple[int, int] = (640, 480),
        color_resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
        enable_color: bool = True,
        align_to: str = "depth",
        depth_hole_filling: bool = False,
        T_out_camera: Optional[np.ndarray] = None,
        enable_global_time: bool = True,
        warmup_frames: int = 10,
    ) -> None:
        if align_to not in ("depth", "color", "none"):
            raise ValueError("align_to must be one of: 'depth', 'color', 'none'.")
        if not enable_color and align_to == "color":
            raise ValueError("align_to='color' requires enable_color=True.")

        self.serial = serial
        self.depth_resolution = depth_resolution
        self.color_resolution = color_resolution
        self.fps = int(fps)
        self.enable_color = bool(enable_color)
        self.align_to = align_to
        self.depth_hole_filling = bool(depth_hole_filling)
        self.T_out_camera = check_transform(T_out_camera)
        self.enable_global_time = bool(enable_global_time)
        self.warmup_frames = int(warmup_frames)

        self.pipeline: Optional[rs.pipeline] = None
        self.config: Optional[rs.config] = None
        self.pipeline_profile: Optional[rs.pipeline_profile] = None
        self.align: Optional[rs.align] = None
        self.hole_filling_filter: Optional[rs.hole_filling_filter] = None

        self.depth_scale: Optional[float] = None
        self.depth_intrinsics: Optional[np.ndarray] = None
        self.color_intrinsics: Optional[np.ndarray] = None
        self.active_intrinsics: Optional[np.ndarray] = None
        self.depth_intrinsics_info: Optional[dict] = None
        self.color_intrinsics_info: Optional[dict] = None
        self.active_intrinsics_info: Optional[dict] = None

        self.rays_cache: Dict[Tuple[int, int, str], Any] = {}
        self.frame_id = 0
        self.last_timestamp: Optional[float] = None
        self.last_host_time: Optional[float] = None

    def start(self) -> None:
        """Start the RealSense pipeline."""
        if self.pipeline is not None:
            return

        if self.serial is None:
            cameras = self.list_cameras()
            if len(cameras) == 0:
                raise RuntimeError("No RealSense camera found.")
            self.serial = cameras[0]["serial"]

        depth_width, depth_height = self.depth_resolution
        color_width, color_height = self.color_resolution

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(self.serial)
        self.config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, self.fps)
        if self.enable_color:
            self.config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, self.fps)

        try:
            self.config.resolve(rs.pipeline_wrapper(self.pipeline))
        except RuntimeError as error:
            raise RuntimeError(
                "Failed to resolve RealSense stream config. "
                f"serial={self.serial}, "
                f"depth={depth_width}x{depth_height}@{self.fps}, "
                f"color={color_width}x{color_height}@{self.fps}, "
                f"enable_color={self.enable_color}, align_to={self.align_to}"
            ) from error

        self.pipeline_profile = self.pipeline.start(self.config)
        self.align = self.create_aligner()
        self.hole_filling_filter = rs.hole_filling_filter(2) if self.depth_hole_filling else None

        self.set_global_time()
        self.depth_scale = float(self.pipeline_profile.get_device().first_depth_sensor().get_depth_scale())
        self.update_intrinsics_from_profile()
        self.frame_id = 0
        self.last_timestamp = None
        self.last_host_time = None
        self.rays_cache.clear()

        for warmup_index in range(max(self.warmup_frames, 0)):
            self.pipeline.wait_for_frames()

    def stop(self) -> None:
        """Stop the RealSense pipeline. Safe to call repeatedly."""
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            finally:
                self.pipeline = None
                self.config = None
                self.pipeline_profile = None
                self.align = None
                self.hole_filling_filter = None
                self.depth_scale = None
                self.depth_intrinsics = None
                self.color_intrinsics = None
                self.active_intrinsics = None
                self.depth_intrinsics_info = None
                self.color_intrinsics_info = None
                self.active_intrinsics_info = None
                self.rays_cache.clear()
                self.last_timestamp = None
                self.last_host_time = None

    def __enter__(self) -> "RealSense":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.stop()

    def create_aligner(self) -> Optional[rs.align]:
        """Create an SDK aligner according to align_to."""
        if not self.enable_color or self.align_to == "none":
            return None
        if self.align_to == "depth":
            return rs.align(rs.stream.depth)
        if self.align_to == "color":
            return rs.align(rs.stream.color)
        return None

    def set_global_time(self) -> None:
        """Try to enable RealSense global time on all sensors that support it."""
        if not self.enable_global_time or self.pipeline_profile is None:
            return
        device = self.pipeline_profile.get_device()
        for sensor in device.query_sensors():
            try:
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1)
            except Exception:
                pass

    def update_intrinsics_from_profile(self) -> None:
        """Read active intrinsics from the started pipeline profile."""
        if self.pipeline_profile is None:
            raise RuntimeError("RealSense is not started.")

        depth_profile = self.pipeline_profile.get_stream(rs.stream.depth).as_video_stream_profile()
        depth_intrinsics = depth_profile.get_intrinsics()
        self.depth_intrinsics = intrinsics_to_matrix(depth_intrinsics)
        self.depth_intrinsics_info = intrinsics_to_dict(depth_intrinsics)

        if self.enable_color:
            color_profile = self.pipeline_profile.get_stream(rs.stream.color).as_video_stream_profile()
            color_intrinsics = color_profile.get_intrinsics()
            self.color_intrinsics = intrinsics_to_matrix(color_intrinsics)
            self.color_intrinsics_info = intrinsics_to_dict(color_intrinsics)
        else:
            self.color_intrinsics = None
            self.color_intrinsics_info = None

        if self.align_to == "color" and self.color_intrinsics is not None:
            self.active_intrinsics = self.color_intrinsics.copy()
            self.active_intrinsics_info = dict(self.color_intrinsics_info)
        else:
            self.active_intrinsics = self.depth_intrinsics.copy()
            self.active_intrinsics_info = dict(self.depth_intrinsics_info)
        self.rays_cache.clear()

    def update_active_intrinsics_from_depth_frame(self, depth_frame: rs.depth_frame) -> None:
        """Update active intrinsics from the current depth frame profile."""
        video_profile = depth_frame.get_profile().as_video_stream_profile()
        intrinsics = video_profile.get_intrinsics()
        active_intrinsics = intrinsics_to_matrix(intrinsics)
        active_intrinsics_info = intrinsics_to_dict(intrinsics)
        if self.active_intrinsics is None or not np.allclose(active_intrinsics, self.active_intrinsics):
            self.active_intrinsics = active_intrinsics
            self.active_intrinsics_info = active_intrinsics_info
            self.rays_cache.clear()

    def read(self, timeout_ms: int = 5000, return_dict: bool = False):
        """Read one RGB-D frame.

        Default return:
            color, depth, timestamp

        Data format:
            color: np.uint8 RGB, shape (H, W, 3), value range [0, 255].
            depth: np.uint16 millimeters, shape (H, W).
            timestamp: RealSense frame timestamp in seconds.

        return_dict=True:
            plain dict with rgb, depth, timestamp, host_time, intrinsics, depth_scale and meta.
        """
        if self.pipeline is None:
            raise RuntimeError("RealSense is not started. Call start() first.")
        if self.depth_scale is None:
            raise RuntimeError("RealSense depth_scale is unavailable. Call start() first.")

        frames = self.pipeline.wait_for_frames(timeout_ms)
        if self.align is not None:
            frames = self.align.process(frames)

        host_time = time.time()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame() if self.enable_color else None

        if not depth_frame:
            raise RuntimeError("Failed to get depth frame from RealSense.")
        if self.enable_color and not color_frame:
            raise RuntimeError("Failed to get color frame from RealSense.")

        if self.hole_filling_filter is not None:
            depth_frame = self.hole_filling_filter.process(depth_frame).as_depth_frame()

        self.update_active_intrinsics_from_depth_frame(depth_frame)

        depth_meters = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
        depth = depth_meters_to_mm(depth_meters)
        color = None
        if color_frame is not None:
            color_bgr = np.asanyarray(color_frame.get_data())
            color = np.ascontiguousarray(color_bgr[..., ::-1])

        timestamp = float(depth_frame.get_timestamp()) * 1e-3
        self.frame_id += 1
        self.last_timestamp = timestamp
        self.last_host_time = host_time

        if return_dict:
            return pack_obs(
                color=color,
                depth=depth,
                timestamp=timestamp,
                host_time=host_time,
                intrinsics=self.get_intrinsics(),
                intrinsics_info=self.get_intrinsics_info(),
                depth_scale=self.get_depth_scale(),
                serial=self.serial,
                frame_id=self.frame_id,
                align_to=self.align_to,
                pointcloud_frame="out" if self.T_out_camera is not None else "camera",
                valid_ratio=depth_valid_ratio(depth),
                mode="rgbd",
            )
        return color, depth, timestamp

    def get_intrinsics(self) -> np.ndarray:
        """Return a copy of active intrinsics matrix K for current depth image."""
        if self.active_intrinsics is None:
            raise RuntimeError("RealSense is not started or active intrinsics are unavailable.")
        return self.active_intrinsics.copy()

    def get_intrinsics_info(self) -> dict:
        """Return active intrinsics as fx/fy/cx/cy/width/height dict."""
        if self.active_intrinsics_info is None:
            raise RuntimeError("RealSense is not started or active intrinsics info is unavailable.")
        return dict(self.active_intrinsics_info)

    def get_depth_scale(self) -> float:
        """Return RealSense raw-depth to meter scale."""
        if self.depth_scale is None:
            raise RuntimeError("RealSense is not started or depth scale is unavailable.")
        return float(self.depth_scale)

    def get_device_info(self) -> dict:
        """Return basic device information for the active camera."""
        if self.pipeline_profile is None:
            raise RuntimeError("RealSense is not started.")
        device = self.pipeline_profile.get_device()
        info = {}
        for camera_info in [
            rs.camera_info.name,
            rs.camera_info.serial_number,
            rs.camera_info.firmware_version,
            rs.camera_info.product_line,
        ]:
            try:
                if device.supports(camera_info):
                    info[str(camera_info)] = device.get_info(camera_info)
            except Exception:
                pass
        return info

    def rays(self, shape: Tuple[int, int], device: str = "cpu"):
        """Return cached camera rays for a given depth shape and torch device."""
        if self.active_intrinsics is None:
            raise RuntimeError("RealSense is not started or active intrinsics are unavailable.")
        height, width = int(shape[0]), int(shape[1])
        cache_key = (height, width, str(device))
        if cache_key not in self.rays_cache:
            self.rays_cache[cache_key] = make_rays(height, width, self.active_intrinsics, device=device)
        return self.rays_cache[cache_key]

    def pointcloud_from_frame(
        self,
        color: Optional[np.ndarray],
        depth: np.ndarray,
        *,
        workspace: Optional[Sequence[float]] = None,
        bound: Optional[Sequence[float]] = None,
        npoints: Optional[int] = 1024,
        min_depth: Optional[float] = 0.05,
        max_depth: Optional[float] = 2.0,
        sampling: str = "random",
        device: str = "cpu",
        return_tensor: bool = True,
    ):
        """Convert one RGB-D frame to point cloud.

        workspace is applied after T_out_camera. If T_out_camera is None, workspace is in camera frame.
        bound is kept as a backward-compatible alias for workspace.
        """
        if workspace is not None and bound is not None:
            raise ValueError("Use only one of workspace or bound, not both.")
        if workspace is None:
            workspace = bound

        return rgbd_to_pointcloud(
            depth=depth,
            color=color,
            rays=self.rays(depth.shape, device=device),
            T_out_camera=self.T_out_camera,
            workspace=workspace,
            npoints=npoints,
            min_depth=min_depth,
            max_depth=max_depth,
            sampling=sampling,
            device=device,
            return_tensor=return_tensor,
        )

    def pointcloud(self, **kwargs):
        """Read one frame and return packed XYZRGB point cloud.

        Return shape is (N, 6), dtype float32. Columns 0:3 are XYZ in meters;
        columns 3:6 are normalized RGB in [0, 1].
        """
        color, depth, timestamp = self.read()
        return self.pointcloud_from_frame(color, depth, **kwargs)

    def get_obs(
        self,
        *,
        mode: str = "full",
        workspace: Optional[Sequence[float]] = None,
        bound: Optional[Sequence[float]] = None,
        npoints: Optional[int] = 1024,
        min_depth: Optional[float] = 0.05,
        max_depth: Optional[float] = 2.0,
        sampling: str = "random",
        device: str = "cpu",
        return_tensor: bool = True,
    ) -> dict:
        """Read one frame and pack an observation dict.

        mode:
        - 'rgbd': RGB-D only, no point cloud generation.
        - 'pointcloud': point cloud and metadata only.
        - 'full': RGB-D + point cloud + metadata.
        """
        if mode not in ("rgbd", "pointcloud", "full"):
            raise ValueError("mode must be one of: 'rgbd', 'pointcloud', 'full'.")
        if workspace is not None and bound is not None:
            raise ValueError("Use only one of workspace or bound, not both.")
        if workspace is None:
            workspace = bound

        color, depth, timestamp = self.read()
        host_time = self.last_host_time
        pointcloud = None

        if mode in ("pointcloud", "full"):
            pointcloud = self.pointcloud_from_frame(
                color,
                depth,
                workspace=workspace,
                npoints=npoints,
                min_depth=min_depth,
                max_depth=max_depth,
                sampling=sampling,
                device=device,
                return_tensor=return_tensor,
            )

        return pack_obs(
            color=color,
            depth=depth,
            timestamp=timestamp,
            host_time=host_time,
            pointcloud=pointcloud,
            intrinsics=self.get_intrinsics(),
            intrinsics_info=self.get_intrinsics_info(),
            depth_scale=self.get_depth_scale(),
            serial=self.serial,
            frame_id=self.frame_id,
            align_to=self.align_to,
            pointcloud_frame="out" if self.T_out_camera is not None else "camera",
            workspace=workspace,
            npoints=npoints,
            sampling=sampling,
            min_depth=min_depth,
            max_depth=max_depth,
            valid_ratio=depth_valid_ratio(depth, min_depth=min_depth, max_depth=max_depth),
            mode=mode,
        )

    @staticmethod
    def list_cameras() -> List[Dict[str, str]]:
        """List connected RealSense devices."""
        context = rs.context()
        cameras = []
        for device in context.query_devices():
            camera = {}
            for key, value in [
                ("serial", rs.camera_info.serial_number),
                ("name", rs.camera_info.name),
                ("firmware", rs.camera_info.firmware_version),
                ("product_line", rs.camera_info.product_line),
            ]:
                try:
                    camera[key] = device.get_info(value) if device.supports(value) else ""
                except Exception:
                    camera[key] = ""
            cameras.append(camera)
        return cameras


def example(
    serial: Optional[str] = None,
    depth_resolution: Tuple[int, int] = (640, 480),
    color_resolution: Tuple[int, int] = (640, 480),
    fps: int = 30,
    enable_color: bool = True,
    align_to: str = "depth",
    npoints: Optional[int] = 1024,
    workspace: Optional[Sequence[float]] = None,
    min_depth: float = 0.05,
    max_depth: float = 1.5,
    sampling: str = "random",
    device: str = "cpu",
    depth_hole_filling: bool = False,
    overlay_path: Optional[str] = None,
    overlay_alpha: float = 0.5,
) -> None:
    """Interactive RealSense RGB-D and packed XYZRGB point cloud smoke test."""
    cameras = RealSense.list_cameras()
    if len(cameras) == 0:
        print("[ERROR] No RealSense camera found.")
        return

    print("Detected RealSense cameras:")
    for camera in cameras:
        print(
            f"  {camera.get('name', ''):30s} SN={camera.get('serial', '')} "
            f"FW={camera.get('firmware', '')} LINE={camera.get('product_line', '')}"
        )

    serial = serial or cameras[0]["serial"]
    print()
    print(f"Using serial: {serial}")
    print(f"depth: {depth_resolution[0]}x{depth_resolution[1]} @ {fps}fps")
    print(f"color: {color_resolution[0]}x{color_resolution[1]} @ {fps}fps, enable_color={enable_color}")
    print(f"align_to: {align_to}")
    print(f"npoints: {npoints}, sampling={sampling}, device={device}")
    print(f"workspace: {workspace}")
    print()

    overlay_bgr = cv2.imread(overlay_path) if overlay_path else None
    if overlay_path and overlay_bgr is None:
        print(f"[WARN] Failed to read overlay image: {overlay_path}")

    with RealSense(
        serial=serial,
        depth_resolution=depth_resolution,
        color_resolution=color_resolution,
        fps=fps,
        enable_color=enable_color,
        align_to=align_to,
        depth_hole_filling=depth_hole_filling,
        warmup_frames=10,
    ) as camera:
        print("Active intrinsics K:")
        print(camera.get_intrinsics())
        print("Active intrinsics info:")
        print(camera.get_intrinsics_info())
        print(f"depth_scale = {camera.get_depth_scale()}")
        print()
        print("Keys: q quit | p show current point cloud")

        while True:
            grab_start = time.perf_counter()
            color, depth, timestamp = camera.read()
            grab_ms = (time.perf_counter() - grab_start) * 1000.0

            pcd_start = time.perf_counter()
            try:
                pointcloud = camera.pointcloud_from_frame(
                    color,
                    depth,
                    workspace=workspace,
                    npoints=npoints,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    sampling=sampling,
                    device=device,
                    return_tensor=False,
                )
            except ValueError as error:
                print(f"[WARN] {error}")
                pointcloud = np.empty((0, 6), dtype=np.float32)
            pcd_ms = (time.perf_counter() - pcd_start) * 1000.0

            depth_vis = make_depth_vis(depth, min_depth=min_depth, max_depth=max_depth)
            if color is not None:
                color_bgr = np.ascontiguousarray(color[..., ::-1])
                color_bgr = blend_overlay(color_bgr, overlay_bgr, alpha=overlay_alpha)
                if color_bgr.shape[:2] == depth_vis.shape[:2]:
                    panel = np.concatenate([color_bgr, depth_vis], axis=1)
                else:
                    depth_resized = cv2.resize(depth_vis, (color_bgr.shape[1], color_bgr.shape[0]))
                    panel = np.concatenate([color_bgr, depth_resized], axis=1)
            else:
                panel = depth_vis

            valid_ratio = depth_valid_ratio(depth, min_depth=min_depth, max_depth=max_depth)
            cv2.putText(
                panel,
                f"ts={timestamp:.3f}s grab={grab_ms:.1f}ms pcd={pcd_ms:.1f}ms points={pointcloud.shape[0]}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                f"align_to={align_to} valid_depth={valid_ratio:.3f} sampling={sampling} device={device}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            title = "RealSense V2 | RGB(left) Depth(right)" if color is not None else "RealSense V2 | Depth"
            cv2.imshow(title, panel)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("p") and pointcloud.shape[0] > 0:
                try:
                    vis_point_cloud(pointcloud, voxel_size=0.005)
                except ImportError:
                    print("[WARN] open3d is not installed; cannot visualize point cloud.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RealSense V2 RGB-D / point cloud demo")
    parser.add_argument("--serial", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--depth-res", type=str, default="640x480")
    parser.add_argument("--color-res", type=str, default="640x480")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--align-to", type=str, default="depth", choices=["depth", "color", "none"])
    parser.add_argument("--npoints", type=int, default=1024)
    parser.add_argument("--all-points", action="store_true")
    parser.add_argument("--sampling", type=str, default="random", choices=["none", "random", "fps", "first"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hole-filling", action="store_true")
    parser.add_argument("--workspace", type=float, nargs=6, default=[-0.5, -0.5, 0.0, 0.5, 0.5, 1.5], help="Workspace XYZ min/max in meters: --workspace x_min y_min z_min x_max y_max z_max")
    parser.add_argument("--bound", type=float, nargs=6, default=None, help="Alias of --workspace")
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=1.5)
    parser.add_argument("--overlay", type=str, default=None, help="Optional BGR/RGB image path for alpha-blended display overlay")
    parser.add_argument("--overlay-alpha", type=float, default=0.5)
    args = parser.parse_args()

    workspace_arg = args.workspace if args.workspace is not None else args.bound
    npoints_arg = None if args.all_points else args.npoints

    example(
        serial=args.serial,
        depth_resolution=parse_resolution(args.depth_res),
        color_resolution=parse_resolution(args.color_res),
        fps=args.fps,
        enable_color=not args.no_color,
        align_to=args.align_to,
        npoints=npoints_arg,
        workspace=workspace_arg,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        sampling=args.sampling,
        device=args.device,
        depth_hole_filling=args.hole_filling,
        overlay_path=args.overlay,
        overlay_alpha=args.overlay_alpha,
    )