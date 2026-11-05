#!/usr/bin/env python3
"""预览 RealSense RGB-D，按 P 开关点云、S 切换 raw/processed。"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, cast

import numpy as np

from dexmani_real.utils.log import configure_logging

if TYPE_CHECKING:
    from dexmani_real.perception.pointcloud import PointCloudProcessor


class RealSenseViewer:
    """一次相机连接用于 RGB-D 与可切换类型的点云预览。"""

    def __init__(self, args):
        self.args = args

    def run(self) -> int:
        from dexmani_real.config.real import load_config
        from dexmani_real.sensor.realsense import RealSenseCamera

        runtime = load_config(path=self.args.config)
        self.pointcloud_config = runtime.pointcloud
        self.table_plane_path = runtime.environment.table_plane_path
        self.base_from_color: np.ndarray | None = None
        self.processor: PointCloudProcessor | None = None
        runtime.camera.validate()
        configure_logging()
        # 预览不额外修改曝光优先级；其余启动设置使用实验配置。
        self.camera = RealSenseCamera(runtime.camera, depth_enabled=True, auto_exposure_priority=None)
        try:
            self.camera.connect()
            print(f"Connected: {self.camera.active_serial}")
            self.capture()
        finally:
            self.camera.disconnect()
        return 0

    def ensure_cloud_transform(self) -> np.ndarray:
        if self.base_from_color is None:
            from dexmani_real.utils.camera_calibrate.extrinsics import (
                load_camera_extrinsics,
            )

            self.base_from_color = load_camera_extrinsics(
                self.camera.active_serial, self.args.camera_calibration,
            )
        return self.base_from_color

    def ensure_processor(self):
        if self.processor is None:
            from dexmani_real.perception.pointcloud import PointCloudProcessor

            base_from_color = self.ensure_cloud_transform()
            table_plane = None
            if self.pointcloud_config.remove_table:
                from dexmani_real.utils.table_plane import load_table_plane

                table_plane = load_table_plane(self.table_plane_path)
            self.processor = PointCloudProcessor(
                depth_scale_m=self.camera.depth_scale_m, intrinsics=self.camera.color_intrinsics,
                T_xarm_base_from_color=base_from_color, table_plane_abcd=table_plane,
                config=self.pointcloud_config,
            )
        return self.processor

    def capture(self) -> None:
        from dexmani_real.perception.pointcloud import (
            DEPTH_RANGE_M,
            build_raw_point_cloud,
        )
        from dexmani_real.utils.visualization.image import ImageViewer, rgbd_panel

        print("P=开关点云 S=切换 raw/processed Q/Esc=退出；关闭点云窗口后 RGB-D 预览继续。")
        cloud_enabled = self.args.cloud is not None
        mode = self.args.cloud or "processed"
        image_viewer = cloud_viewer = None
        depth_scale_m = cast(float, self.camera.depth_scale_m)
        try:
            image_viewer = ImageViewer("RealSense RGB-D | P=cloud S=raw/processed Q/Esc=quit")
            while True:
                events = image_viewer.poll_events()
                if "closed" in events:
                    return
                if cloud_viewer is not None:
                    cloud_events = cloud_viewer.poll_events()
                    if "closed" in cloud_events:
                        cloud_viewer = None
                        cloud_enabled = False
                    events.extend(key for key in cloud_events if key != "closed")
                if any(key in {"q", "esc"} for key in events):
                    return
                for key in events:
                    if key == "p":
                        cloud_enabled = not cloud_enabled
                    elif key == "s":
                        mode = "raw" if mode == "processed" else "processed"
                    else:
                        continue
                    print(f"Point cloud: {'on' if cloud_enabled else 'off'}, mode={mode}")
                if not cloud_enabled and cloud_viewer is not None:
                    cloud_viewer.close()
                    cloud_viewer = None
                if cloud_enabled and cloud_viewer is None:
                    from dexmani_real.utils.visualization.pointcloud import (
                        PointCloudViewer,
                    )

                    self.pointcloud_config.workspace_bounds()
                    cloud_viewer = PointCloudViewer(
                        workspace=self.pointcloud_config.workspace,
                        base_from_color=self.ensure_cloud_transform(), keys="ps",
                        title="RealSense Point Cloud | P=cloud S=raw/processed Q/Esc=quit",
                    )
                frame = self.camera.read()
                if frame is None:
                    continue
                depth = cast(np.ndarray, frame.depth)
                if cloud_viewer is not None:
                    if mode == "processed":
                        cloud = self.ensure_processor().process(depth, frame.rgb)
                    else:
                        cloud = build_raw_point_cloud(
                            depth_raw=depth, color=frame.rgb, depth_scale_m=depth_scale_m,
                            intrinsics=self.camera.color_intrinsics,
                            T_xarm_base_from_color=self.ensure_cloud_transform(),
                        )
                    cloud_viewer.update(cloud)
                image_viewer.show(rgbd_panel(
                    frame.rgb, depth.astype(np.float32) * depth_scale_m,
                    depth_range_m=DEPTH_RANGE_M,
                ))
        finally:
            try:
                if cloud_viewer is not None:
                    cloud_viewer.close()
            finally:
                if image_viewer is not None:
                    image_viewer.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Trusted Python experiment file exporting CONFIG: RealConfig")
    parser.add_argument("--camera-calibration")
    parser.add_argument("--cloud", choices=("raw", "processed"), help="启动时打开点云并指定初始类型；P 开关，S 切换")
    args = parser.parse_args(argv)
    return RealSenseViewer(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
