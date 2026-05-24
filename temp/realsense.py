import argparse
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from pcd_utils_refactor_minfix import make_rays, rgbd_to_pointcloud

try:
    from pcd_utils_refactor_minfix import vis_point_cloud
except ImportError:
    vis_point_cloud = None


def intrinsics_to_matrix(intr):
    return np.array(
        [
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def intrinsics_from_frame(frame):
    profile = frame.get_profile().as_video_stream_profile()
    return intrinsics_to_matrix(profile.get_intrinsics())


class RealSense:
    def __init__(
        self,
        serial,
        depth_resolution=(640, 480),
        color_resolution=(1280, 720),
        fps=30,
        depth_hole_filling=False,
        pose=None,
        enable_color=True,
    ):
        self.serial = serial
        self.depth_resolution = depth_resolution
        self.color_resolution = color_resolution
        self.fps = fps
        self.depth_hole_filling = depth_hole_filling
        self.pose = pose
        self.enable_color = enable_color

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.depth) if enable_color else None
        self.hole_filling = rs.hole_filling_filter(2) if depth_hole_filling else None

        self.depth_scale = None
        self.depth_intr = None
        self.rays_cache = {}

    def start(self):
        dw, dh = self.depth_resolution
        cw, ch = self.color_resolution

        self.config.enable_device(self.serial)
        self.config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, self.fps)
        if self.enable_color:
            self.config.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, self.fps)

        profile = self.pipeline.start(self.config)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        self.depth_intr = intrinsics_to_matrix(depth_stream.get_intrinsics())
        self.rays_cache.clear()

        for _ in range(5):
            self.pipeline.wait_for_frames()

    def stop(self):
        self.pipeline.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def update_depth_intrinsics(self, depth_frame):
        depth_intr = intrinsics_from_frame(depth_frame)
        if self.depth_intr is None or not np.allclose(depth_intr, self.depth_intr):
            self.depth_intr = depth_intr
            self.rays_cache.clear()

    def read(self, timeout_ms=5000):
        frames = self.pipeline.wait_for_frames(timeout_ms)
        frames = self.align.process(frames) if self.align is not None else frames

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame() if self.enable_color else None

        if not depth_frame:
            raise RuntimeError("Failed to get depth frame from RealSense.")
        if self.enable_color and not color_frame:
            raise RuntimeError("Failed to get color frame from RealSense.")

        if self.hole_filling is not None:
            depth_frame = self.hole_filling.process(depth_frame).as_depth_frame()

        self.update_depth_intrinsics(depth_frame)

        depth = np.asarray(depth_frame.get_data(), dtype=np.float32) * self.depth_scale
        color = None
        if color_frame is not None:
            color = np.ascontiguousarray(np.asarray(color_frame.get_data())[..., ::-1])

        timestamp = depth_frame.get_timestamp() * 1e-3
        return color, depth, timestamp

    def rays(self, shape, device="cpu"):
        if self.depth_intr is None:
            raise RuntimeError("RealSense is not started or depth intrinsics are unavailable.")

        key = (shape[0], shape[1], str(device))
        if key not in self.rays_cache:
            self.rays_cache[key] = make_rays(shape[0], shape[1], self.depth_intr, device=device)
        return self.rays_cache[key]

    def pointcloud_from_frame(
        self,
        color,
        depth,
        *,
        bound=None,
        npoints=1024,
        min_depth=0.1,
        max_depth=5.0,
        device="cpu",
        return_tensor=True,
    ):
        return rgbd_to_pointcloud(
            depth=depth,
            color=color,
            rays=self.rays(depth.shape, device=device),
            bound=bound,
            npoints=npoints,
            min_depth=min_depth,
            max_depth=max_depth,
            transform=self.pose,
            device=device,
            return_tensor=return_tensor,
        )

    def pointcloud(
        self,
        *,
        bound=None,
        npoints=1024,
        min_depth=0.1,
        max_depth=5.0,
        device="cpu",
        return_tensor=True,
    ):
        color, depth, _ = self.read()
        return self.pointcloud_from_frame(
            color,
            depth,
            bound=bound,
            npoints=npoints,
            min_depth=min_depth,
            max_depth=max_depth,
            device=device,
            return_tensor=return_tensor,
        )

    @staticmethod
    def list_cameras():
        ctx = rs.context()
        return [
            {
                "serial": dev.get_info(rs.camera_info.serial_number),
                "name": dev.get_info(rs.camera_info.name),
                "firmware": dev.get_info(rs.camera_info.firmware_version),
            }
            for dev in ctx.query_devices()
        ]


def parse_resolution(text):
    w, h = text.lower().split("x")
    return int(w), int(h)


def make_depth_vis(depth, min_depth=0.1, max_depth=1.5):
    valid = (depth > 0.0) & np.isfinite(depth)
    depth = np.clip(depth, min_depth, max_depth)
    depth = (depth - min_depth) / max(max_depth - min_depth, 1e-6)
    depth = (255.0 * (1.0 - depth)).astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    depth[~valid] = 0
    return depth


def example(
    serial=None,
    depth_resolution=(640, 480),
    color_resolution=(1280, 720),
    fps=30,
    npoints=1024,
    bound=None,
    min_depth=0.1,
    max_depth=1.5,
    device="cpu",
    depth_hole_filling=False,
    enable_color=True,
):
    cams = RealSense.list_cameras()
    if serial is None:
        if len(cams) == 0:
            raise RuntimeError("No RealSense camera found.")
        serial = cams[0]["serial"]

    with RealSense(
        serial=serial,
        depth_resolution=depth_resolution,
        color_resolution=color_resolution,
        fps=fps,
        depth_hole_filling=depth_hole_filling,
        enable_color=enable_color,
    ) as cam:
        while True:
            t0 = time.perf_counter()
            color, depth, ts = cam.read()

            try:
                points, colors = cam.pointcloud_from_frame(
                    color,
                    depth,
                    bound=bound,
                    npoints=npoints,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    device=device,
                    return_tensor=False,
                )
            except ValueError:
                points = np.empty((0, 3), dtype=np.float32)
                colors = None

            dt = (time.perf_counter() - t0) * 1000.0

            depth_vis = make_depth_vis(depth, min_depth, max_depth)
            panel = depth_vis
            if color is not None:
                panel = np.concatenate([np.ascontiguousarray(color[..., ::-1]), depth_vis], axis=1)

            cv2.putText(
                panel,
                f"ts={ts:.3f}s  total={dt:.1f}ms  points={points.shape[0]}  device={device}  color={int(color is not None)}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            title = "RealSense | RGB(left) Depth(right)" if color is not None else "RealSense | Depth"
            cv2.imshow(title, panel)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("p") and vis_point_cloud is not None and points.shape[0] > 0:
                vis = np.concatenate([points, colors.astype(np.float32)], axis=1) if colors is not None else points
                vis_point_cloud(vis, voxel_size=0.005)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--depth-res", type=str, default="640x480")
    parser.add_argument("--color-res", type=str, default="1280x720")
    parser.add_argument("--npoints", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hole-filling", action="store_true")
    parser.add_argument("--bound", type=float, nargs=6, default=None)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=1.5)
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    example(
        serial=args.serial,
        depth_resolution=parse_resolution(args.depth_res),
        color_resolution=parse_resolution(args.color_res),
        fps=args.fps,
        npoints=args.npoints,
        bound=args.bound,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        device=args.device,
        depth_hole_filling=args.hole_filling,
        enable_color=not args.no_color,
    )
