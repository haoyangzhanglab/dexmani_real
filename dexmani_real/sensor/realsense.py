from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

import cv2
import numpy as np
import pyrealsense2 as rs

from dexmani_real.sensor.pointcloud_utils import (
    PointCloudConfig,
    depth_valid_ratio,
    intrinsics_to_dict,
    intrinsics_to_matrix,
    intrinsics_to_vector,
    make_depth_vis,
    make_rays,
    rgbd_to_pointcloud,
    vis_point_cloud,
)


AlignMode = Literal["depth_to_color", "color_to_depth", "none"]
ALIGN_MODE_ALIASES = {
    "depth_to_color": "depth_to_color",
    "color_to_depth": "color_to_depth",
    "none": "none",
    "color": "depth_to_color",
    "depth": "color_to_depth",
}


@dataclass(frozen=True)
class RealSenseConfig:
    camera_name: str = "realsense"
    serial: str | None = None
    depth_resolution: tuple[int, int] = (640, 480)
    color_resolution: tuple[int, int] = (640, 480)
    fps: int = 30
    enable_color: bool = True
    align_mode: AlignMode = "depth_to_color"
    depth_hole_filling: bool = False
    enable_global_time: bool = True
    warmup_frames: int = 10
    frame_name: str | None = None

    def __post_init__(self) -> None:
        mode = normalize_align_mode(self.align_mode)
        object.__setattr__(self, "align_mode", mode)
        if mode != "none" and not self.enable_color:
            raise ValueError("alignment requires enable_color=True.")
        if self.frame_name is None:
            frame_name = "camera_color_optical" if mode == "depth_to_color" else "camera_depth_optical"
            object.__setattr__(self, "frame_name", frame_name)


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray | None
    depth: np.ndarray
    depth_raw: np.ndarray
    timestamp: float
    host_time: float
    frame_id: int
    K: np.ndarray
    intr: np.ndarray
    intrinsics_info: dict
    depth_scale: float
    camera_name: str
    serial: str | None
    align_mode: AlignMode
    frame_name: str

    def to_dict(self) -> dict:
        return {
            "rgb": self.rgb,
            "depth": self.depth,
            "depth_raw": self.depth_raw,
            "timestamp": self.timestamp,
            "host_time": self.host_time,
            "frame_id": self.frame_id,
            "K": self.K,
            "intr": self.intr,
            "intrinsics_info": self.intrinsics_info,
            "depth_scale": self.depth_scale,
            "camera_name": self.camera_name,
            "serial": self.serial,
            "align_mode": self.align_mode,
            "frame_name": self.frame_name,
            "meta": {
                "rgb_order": "RGB",
                "rgb_dtype": "uint8" if self.rgb is not None else None,
                "depth_unit": "m",
                "depth_dtype": "float32",
                "depth_raw_unit": "raw_z16",
                "depth_raw_dtype": "uint16",
            },
        }


def normalize_align_mode(mode: str) -> AlignMode:
    key = str(mode).lower()
    if key not in ALIGN_MODE_ALIASES:
        valid = ", ".join(ALIGN_MODE_ALIASES.keys())
        raise ValueError(f"align_mode must be one of: {valid}.")
    return ALIGN_MODE_ALIASES[key]  # type: ignore[return-value]


class RealSense:
    def __init__(self, config: RealSenseConfig = RealSenseConfig()) -> None:
        self.config = config
        self.active_serial: str | None = config.serial

        self.pipeline: rs.pipeline | None = None
        self.profile: rs.pipeline_profile | None = None
        self.aligner: rs.align | None = None
        self.hole_filling_filter: rs.hole_filling_filter | None = None

        self.depth_scale: float | None = None
        self.K: np.ndarray | None = None
        self.intr: np.ndarray | None = None
        self.intrinsics_info: dict | None = None
        self.rays_cache: dict[tuple[int, int, str], Any] = {}

        self.frame_id = 0
        self.last_frame: CameraFrame | None = None

    def start(self) -> None:
        if self.pipeline is not None:
            return

        self.active_serial = self.config.serial or self.find_default_serial()
        self.pipeline = rs.pipeline()
        rs_config = self.create_rs_config()

        try:
            rs_config.resolve(rs.pipeline_wrapper(self.pipeline))
            self.profile = self.pipeline.start(rs_config)
        except RuntimeError:
            self.pipeline = None
            self.profile = None
            raise

        self.aligner = self.create_aligner()
        self.hole_filling_filter = rs.hole_filling_filter(2) if self.config.depth_hole_filling else None
        self.set_global_time()
        self.depth_scale = float(self.profile.get_device().first_depth_sensor().get_depth_scale())
        self.update_intrinsics_from_profile()
        self.frame_id = 0
        self.last_frame = None
        self.rays_cache.clear()

        for _ in range(max(self.config.warmup_frames, 0)):
            self.pipeline.wait_for_frames()

    def stop(self) -> None:
        if self.pipeline is None:
            return
        try:
            self.pipeline.stop()
        finally:
            self.pipeline = None
            self.profile = None
            self.aligner = None
            self.hole_filling_filter = None
            self.depth_scale = None
            self.K = None
            self.intr = None
            self.intrinsics_info = None
            self.rays_cache.clear()
            self.last_frame = None

    def __enter__(self) -> "RealSense":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.stop()

    def create_rs_config(self) -> rs.config:
        depth_width, depth_height = self.config.depth_resolution
        color_width, color_height = self.config.color_resolution

        rs_config = rs.config()
        rs_config.enable_device(self.active_serial)
        rs_config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, self.config.fps)
        if self.config.enable_color:
            rs_config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, self.config.fps)
        return rs_config

    def create_aligner(self) -> rs.align | None:
        if self.config.align_mode == "none":
            return None
        target = rs.stream.color if self.config.align_mode == "depth_to_color" else rs.stream.depth
        return rs.align(target)

    def set_global_time(self) -> None:
        if not self.config.enable_global_time or self.profile is None:
            return
        for sensor in self.profile.get_device().query_sensors():
            try:
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1)
            except RuntimeError:
                pass

    def update_intrinsics_from_profile(self) -> None:
        if self.profile is None:
            raise RuntimeError("RealSense is not started.")
        stream = rs.stream.color if self.config.align_mode == "depth_to_color" else rs.stream.depth
        video_profile = self.profile.get_stream(stream).as_video_stream_profile()
        self.set_intrinsics(video_profile.get_intrinsics())

    def update_intrinsics_from_depth_frame(self, depth_frame: rs.depth_frame) -> None:
        video_profile = depth_frame.get_profile().as_video_stream_profile()
        self.set_intrinsics(video_profile.get_intrinsics())

    def set_intrinsics(self, intrinsics: Any) -> None:
        K = intrinsics_to_matrix(intrinsics)
        if self.K is None or not np.allclose(K, self.K):
            self.K = K
            self.intr = intrinsics_to_vector(K)
            self.intrinsics_info = intrinsics_to_dict(intrinsics)
            self.rays_cache.clear()

    def read(self, timeout_ms: int = 5000) -> CameraFrame:
        if self.pipeline is None:
            raise RuntimeError("RealSense is not started. Call start() first.")
        if self.depth_scale is None:
            raise RuntimeError("RealSense depth_scale is unavailable.")

        frames = self.pipeline.wait_for_frames(timeout_ms)
        if self.aligner is not None:
            frames = self.aligner.process(frames)

        host_time = time.time()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame() if self.config.enable_color else None
        if not depth_frame:
            raise RuntimeError("Failed to get depth frame.")
        if self.config.enable_color and not color_frame:
            raise RuntimeError("Failed to get color frame.")

        if self.hole_filling_filter is not None:
            depth_frame = self.hole_filling_filter.process(depth_frame).as_depth_frame()

        self.update_intrinsics_from_depth_frame(depth_frame)
        if self.K is None or self.intr is None or self.intrinsics_info is None:
            raise RuntimeError("RealSense intrinsics are unavailable.")

        depth_raw = np.ascontiguousarray(np.asanyarray(depth_frame.get_data()))
        depth = depth_raw.astype(np.float32) * float(self.depth_scale)

        rgb = None
        if color_frame is not None:
            bgr = np.asanyarray(color_frame.get_data())
            rgb = np.ascontiguousarray(bgr[..., ::-1])

        self.frame_id += 1
        frame = CameraFrame(
            rgb=rgb,
            depth=depth,
            depth_raw=depth_raw,
            timestamp=float(depth_frame.get_timestamp()) * 1e-3,
            host_time=host_time,
            frame_id=self.frame_id,
            K=self.K.copy(),
            intr=self.intr.copy(),
            intrinsics_info=dict(self.intrinsics_info),
            depth_scale=float(self.depth_scale),
            camera_name=self.config.camera_name,
            serial=self.active_serial,
            align_mode=self.config.align_mode,
            frame_name=str(self.config.frame_name),
        )
        self.last_frame = frame
        return frame

    def read_rgbd(self, timeout_ms: int = 5000) -> tuple[np.ndarray | None, np.ndarray, float]:
        frame = self.read(timeout_ms=timeout_ms)
        return frame.rgb, frame.depth, frame.timestamp

    def get_obs(
        self,
        *,
        mode: Literal["rgbd", "pointcloud", "full"] = "full",
        pcd_config: PointCloudConfig | None = None,
        T_out_camera: Optional[np.ndarray] = None,
        timeout_ms: int = 5000,
    ) -> dict:
        if mode not in ("rgbd", "pointcloud", "full"):
            raise ValueError("mode must be one of: 'rgbd', 'pointcloud', 'full'.")

        frame = self.read(timeout_ms=timeout_ms)
        obs = frame.to_dict()
        config = pcd_config or PointCloudConfig()
        obs["meta"].update(
            {
                "valid_depth_ratio": depth_valid_ratio(frame.depth, config.min_depth, config.max_depth),
                "pointcloud_frame": "out" if T_out_camera is not None else frame.frame_name,
            }
        )

        if mode == "pointcloud":
            obs.pop("rgb", None)
            obs.pop("depth", None)
            obs.pop("depth_raw", None)

        if mode in ("pointcloud", "full"):
            pointcloud = self.pointcloud_from_frame(frame, config, T_out_camera=T_out_camera)
            obs["pointcloud"] = pointcloud
            obs["meta"].update(
                {
                    "pointcloud_format": "xyzrgb",
                    "pointcloud_xyz_unit": "m",
                    "pointcloud_rgb_range": [0.0, 1.0],
                    "pointcloud_dtype": "float32",
                    "point_count": int(pointcloud.shape[0]),
                    "workspace": list(config.workspace) if config.workspace is not None else None,
                    "npoints": config.npoints,
                    "sampling": config.sampling,
                    "min_depth": config.min_depth,
                    "max_depth": config.max_depth,
                }
            )
        return obs

    def pointcloud_from_frame(
        self,
        frame: CameraFrame,
        config: PointCloudConfig | None = None,
        *,
        T_out_camera: Optional[np.ndarray] = None,
    ):
        return rgbd_to_pointcloud(
            depth=frame.depth,
            K=frame.K,
            rgb=frame.rgb if frame.align_mode != "none" else None,
            config=config or PointCloudConfig(),
            T_out_camera=T_out_camera,
        )

    def pointcloud(self, config: PointCloudConfig | None = None, *, T_out_camera: Optional[np.ndarray] = None):
        frame = self.read()
        return self.pointcloud_from_frame(frame, config, T_out_camera=T_out_camera)

    def get_intrinsics(self) -> np.ndarray:
        if self.K is None:
            raise RuntimeError("RealSense is not started or intrinsics are unavailable.")
        return self.K.copy()

    def get_intrinsics_info(self) -> dict:
        if self.intrinsics_info is None:
            raise RuntimeError("RealSense is not started or intrinsics info is unavailable.")
        return dict(self.intrinsics_info)

    def get_depth_scale(self) -> float:
        if self.depth_scale is None:
            raise RuntimeError("RealSense is not started or depth_scale is unavailable.")
        return float(self.depth_scale)

    def get_device_info(self) -> dict:
        if self.profile is None:
            raise RuntimeError("RealSense is not started.")
        device = self.profile.get_device()
        info = {}
        for name, key in [
            ("name", rs.camera_info.name),
            ("serial", rs.camera_info.serial_number),
            ("firmware", rs.camera_info.firmware_version),
            ("product_line", rs.camera_info.product_line),
        ]:
            try:
                info[name] = device.get_info(key) if device.supports(key) else ""
            except RuntimeError:
                info[name] = ""
        return info

    def get_rays(self, shape: tuple[int, int], device: str = "cpu"):
        if self.K is None:
            raise RuntimeError("RealSense is not started or intrinsics are unavailable.")
        height, width = int(shape[0]), int(shape[1])
        key = (height, width, str(device))
        if key not in self.rays_cache:
            self.rays_cache[key] = make_rays(height, width, self.K, device=device)
        return self.rays_cache[key]

    def find_default_serial(self) -> str:
        cameras = self.list_cameras()
        if len(cameras) == 0:
            raise RuntimeError("No RealSense camera found.")
        return cameras[0]["serial"]

    @staticmethod
    def list_cameras() -> list[dict[str, str]]:
        context = rs.context()
        cameras = []
        for device in context.query_devices():
            item = {}
            for name, key in [
                ("serial", rs.camera_info.serial_number),
                ("name", rs.camera_info.name),
                ("firmware", rs.camera_info.firmware_version),
                ("product_line", rs.camera_info.product_line),
            ]:
                try:
                    item[name] = device.get_info(key) if device.supports(key) else ""
                except RuntimeError:
                    item[name] = ""
            cameras.append(item)
        return cameras


EXAMPLE_CAMERA_CONFIG = RealSenseConfig()
EXAMPLE_POINTCLOUD_CONFIG = PointCloudConfig(return_tensor=False)


def example(
    camera_config: RealSenseConfig = EXAMPLE_CAMERA_CONFIG,
    pointcloud_config: PointCloudConfig = EXAMPLE_POINTCLOUD_CONFIG,
) -> None:
    cameras = RealSense.list_cameras()
    if len(cameras) == 0:
        print("No RealSense camera found.")
        return

    print("Detected RealSense cameras:")
    for camera in cameras:
        print(f"  {camera.get('name', ''):28s} SN={camera.get('serial', '')} FW={camera.get('firmware', '')}")
    print(f"Camera config: {asdict(camera_config)}")
    print(f"PointCloud config: {asdict(pointcloud_config)}")
    print("Keys: q quit | p show current point cloud")

    with RealSense(camera_config) as camera:
        print("Active K:")
        print(camera.get_intrinsics())
        print(f"depth_scale: {camera.get_depth_scale()}")

        last_pointcloud = None
        while True:
            start_time = time.perf_counter()
            frame = camera.read()
            grab_ms = (time.perf_counter() - start_time) * 1000.0

            depth_vis = make_depth_vis(frame.depth, pointcloud_config.min_depth or 0.05, pointcloud_config.max_depth or 1.5)
            if frame.rgb is not None:
                color_bgr = np.ascontiguousarray(frame.rgb[..., ::-1])
                if color_bgr.shape[:2] != depth_vis.shape[:2]:
                    depth_vis = cv2.resize(depth_vis, (color_bgr.shape[1], color_bgr.shape[0]))
                panel = np.concatenate([color_bgr, depth_vis], axis=1)
            else:
                panel = depth_vis

            valid_ratio = depth_valid_ratio(frame.depth, pointcloud_config.min_depth, pointcloud_config.max_depth)
            text = f"id={frame.frame_id} ts={frame.timestamp:.3f}s grab={grab_ms:.1f}ms valid={valid_ratio:.3f}"
            cv2.putText(panel, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(panel, f"align={frame.align_mode}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("RealSense | RGB(left) Depth(right)", panel)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                try:
                    last_pointcloud = camera.pointcloud_from_frame(frame, pointcloud_config)
                    vis_point_cloud(last_pointcloud, voxel_size=0.005)
                except ImportError:
                    print("open3d is not installed; cannot visualize point cloud.")
                except ValueError as error:
                    print(error)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    example()