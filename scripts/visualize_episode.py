#!/usr/bin/env python3
"""离线查看 Raw episode 的观测与点云，或输出数据概要。"""
# Usage: python scripts/visualize_episode.py episodes/pnp_apple_banana/episode_20261101_165909
# 已有 /hoi 时自动叠加物体点云；追加 --info 仅输出数据概要。

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from itertools import islice
from pathlib import Path
from typing import cast

os.environ.setdefault("RUST_LOG", "error")

import numpy as np

from dexmani_real.config.perception import HandCloudConfig, PointCloudConfig
from dexmani_real.config.real import load_config
from dexmani_real.data.raw import TACTILE_FINGER_ORDER, RawReader
from dexmani_real.geometry.kinematics import ArmFK, HandFK
from dexmani_real.perception.hand_cloud import HandPointCloud, hand_geometry
from dexmani_real.perception.pointcloud import PointCloudProcessor
from dexmani_real.utils.table_plane import load_table_plane
from dexmani_real.utils.visualization.rerun import (
    episode_blueprint,
    log_camera_geometry,
    log_points,
    log_rgbd,
    log_series,
)

logger = logging.getLogger("dexmani_real.cli.visualize_episode")


RAW_SERIES_FIELDS = (
    "arm_qpos", "hand_qpos", "action_arm_joint_target", "action_hand_joint_target",
)

FINGERTIP_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 60, 60),  # thumb  — red
    (60, 255, 60),  # index  — green
    (60, 120, 255),  # middle — blue
    (255, 200, 40),  # ring   — gold
    (220, 60, 255),  # pinky  — magenta
)
# Fixed, easily distinguishable from the five finger colors.
EEF_COLOR = (255, 255, 255)


def print_episode_info(episode_path: str) -> None:
    import h5py

    path = Path(episode_path).expanduser()
    print(f"Episode: {path} -> {path.resolve()}")
    print(f"RGB file exists: {(path / 'rgb.mp4').is_file()}")
    with h5py.File(path / "data.h5", "r") as data:
        if "meta" in data:
            for name, value in data["meta"].attrs.items():
                print(f"  {name}: {value}")
        def describe(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
        data.visititems(describe)


class EpisodeVisualizer:
    def __init__(
        self,
        episode_path: str,
        *,
        table_plane_path: str | Path | None,
        max_frames: int | None = None,
        pointcloud_config: PointCloudConfig | None = None,
        hand_pointcloud_config: HandCloudConfig = HandCloudConfig(),
    ):
        import rerun as rr

        self.rr = rr

        episode_name = Path(episode_path).stem
        try:
            self.reader = RawReader(episode_path)
            self.timestamps_s = self.reader.timestamps_s
            self.object_colors = None
            if self.reader.hoi_metadata is not None:
                import colorsys
                offsets = self.reader.data_file["hoi"].attrs["offsets"]
                self.object_colors = np.concatenate([
                    np.tile(np.asarray(colorsys.hsv_to_rgb(i * 0.61803398875 % 1, 0.75, 1)) * 255,
                            (int(count), 1))
                    for i, count in enumerate(np.diff(offsets))
                ]).astype(np.uint8)

            self.reader.require_fields(
                "rgb",
                "depth",
                "arm_qpos",
                "hand_qpos",
                "hand_contact",
                "action_arm_joint_target",
                "action_hand_joint_target",
            )

            self.frame_count = self.reader.num_frames
            if self.frame_count <= 0:
                raise ValueError("episode must contain at least one row")
            if max_frames is not None:
                self.frame_count = min(self.frame_count, max_frames)
            self.camera_model = self.reader.read_camera()
            self.T_xarm_base_from_color = self.camera_model.T_xarm_base_from_color
            if pointcloud_config is not None and self.T_xarm_base_from_color is None:
                logger.warning("No camera extrinsics: camera point cloud disabled")
            self.pointcloud_processor = None
            self.empty_cloud_count = 0
            self.pointcloud_processing_ns = 0
            self.pointcloud_processed_frames = 0
            if pointcloud_config is not None and self.T_xarm_base_from_color is not None:
                table_plane = load_table_plane(table_plane_path) if pointcloud_config.remove_table else None
                self.pointcloud_processor = PointCloudProcessor(
                    depth_scale_m=self.camera_model.depth_scale_m,
                    intrinsics=self.camera_model.color,
                    T_xarm_base_from_color=self.T_xarm_base_from_color,
                    table_plane_abcd=table_plane,
                    config=pointcloud_config,
                )
                logger.info(
                    "Canonical point cloud enabled: N=%d table=%s",
                    pointcloud_config.num_points,
                    "enabled" if table_plane is not None else "disabled",
                )

            logger.info("Frames=%d", self.frame_count)

            self.arm_fk = ArmFK()
            self.hand_fk = HandFK()
            self.hand_cloud = HandPointCloud(self.hand_fk, hand_pointcloud_config.num_points)
            self.state = self.preload_state()

            app_id = f"DexMani - {episode_name}"
            rec_id = f"{episode_name}-{time.time_ns()}"
            rr.init(
                app_id,
                recording_id=rec_id,
                spawn=True,
            )
            # 同一份定义供布局、标签和批量发送使用，数组仅保留引用。
            self.series = []
            force_labels = tuple(f"{axis} (SDK / 255)" for axis in ("Fx", "Fy", "Fz"))
            for group, path, key, labels in (
                ("State", "state/arm_qpos", "arm_qpos", None),
                ("State", "state/hand_qpos", "hand_qpos", None),
                ("Action", "action/action_arm_joint_target", "action_arm_joint_target", None),
                ("Action", "action/action_hand_joint_target", "action_hand_joint_target", None),
                ("Contact Force", "state/hand_contact_mag", "hand_contact_mag",
                 [f"{finger} (SDK / 255)" for finger in TACTILE_FINGER_ORDER.split(",")]),
                ("Contact Force", "state/hand_force_thumb", "hand_force_thumb", force_labels),
                ("Contact Force", "state/hand_force_index", "hand_force_index", force_labels),
            ):
                values = self.state[key]
                if labels is None:
                    labels = [str(i) for i in range(values.shape[1])]
                self.series.append((group, path, values, labels))
            series_groups: dict[str, list[str]] = {}
            for group, path, _, _ in self.series:
                series_groups.setdefault(group, []).append(path)
            # 当前布局覆盖 viewer 记忆的旧布局。
            rr.send_blueprint(
                episode_blueprint(
                    series_groups.items(),
                    camera_in_base=self.T_xarm_base_from_color is not None,
                ),
                make_active=True, make_default=True,
            )
            self.log_recording_data()
        except BaseException as exc:
            self.close(exc)
            raise

    def preload_state(self) -> dict[str, np.ndarray]:
        state = {
            key: self.reader[key][: self.frame_count] for key in RAW_SERIES_FIELDS
        }

        contact = self.reader["hand_contact"][: self.frame_count]
        state["hand_contact_mag"] = np.linalg.norm(contact, axis=2)
        state["hand_force_thumb"] = contact[:, 0, :]
        state["hand_force_index"] = contact[:, 1, :]

        return state

    def log_recording_data(self) -> None:
        steps = np.arange(self.frame_count)
        for _, path, values, labels in self.series:
            log_series(
                path, values, labels=labels, steps=steps,
                timestamps_s=self.timestamps_s[:self.frame_count],
            )
        log_camera_geometry(self.camera_model.color, self.T_xarm_base_from_color)

    def log_step(self, step_idx: int, rgb: np.ndarray) -> None:
        self.rr.set_time_sequence("step", step_idx)
        self.rr.set_time_seconds("time", float(self.timestamps_s[step_idx]))
        depth = self.log_camera(step_idx, rgb)
        self.log_pointcloud(rgb, depth)
        self.log_hand_geometry(step_idx)
        if self.reader.hoi_metadata is not None:
            points = self.reader["object_pointcloud"][step_idx] if self.reader["object_pointcloud_valid"][step_idx] else None
            if points is not None and not np.isfinite(points).all():
                raise ValueError(f"object_pointcloud contains nonfinite coordinates at row {step_idx}")
            log_points("object_pointcloud", points, colors=self.object_colors)

    def log_hand_geometry(self, step_idx: int) -> None:
        arm = self.state["arm_qpos"][step_idx]
        hand = self.state["hand_qpos"][step_idx]
        pose = self.arm_fk.forward(arm) if np.isfinite(arm).all() else None
        fingertips, cloud = None, None
        if pose is not None and np.isfinite(hand).all():
            # 指尖和表面点共用一次手部 FK，逐帧发送避免缓存整段稠密点云。
            geometry = hand_geometry(
                pose[None], hand[None],
                arm_fk=self.arm_fk, hand_fk=self.hand_fk, hand_cloud=self.hand_cloud,
            )
            fingertips = geometry["fingertip_xyz"][0]
            cloud = geometry["hand_pointcloud"][0]
        log_points("hand_pointcloud", cloud)
        log_points(
            "fingertips", fingertips,
            colors=FINGERTIP_COLORS, radii=0.012,
        )
        log_points(
            "eef", pose[None, :3, 3] if pose is not None else None,
            colors=[EEF_COLOR], radii=0.020,
        )

    def log_camera(self, step_idx: int, rgb: np.ndarray) -> np.ndarray:
        color = self.camera_model.color
        if rgb.dtype != np.uint8 or rgb.shape != (color.height, color.width, 3):
            raise ValueError(f"RGB shape or dtype invalid at frame {step_idx}")
        depth = np.asarray(self.reader["depth"][step_idx])
        depth_scale_m = cast(float, self.camera_model.depth_scale_m)
        log_rgbd(
            rgb, depth, depth_scale_m=depth_scale_m,
            depth_range_m=(0, 10000 * depth_scale_m),
        )
        return depth

    def log_pointcloud(self, rgb: np.ndarray, depth: np.ndarray) -> None:
        if self.pointcloud_processor is None:
            return
        started_ns = time.perf_counter_ns()
        cloud = self.pointcloud_processor.process(depth, rgb)
        self.pointcloud_processing_ns += time.perf_counter_ns() - started_ns
        self.pointcloud_processed_frames += 1
        if not len(cloud):
            self.empty_cloud_count += 1
        log_points("pcd", cloud)

    @property
    def mean_pointcloud_processing_ms(self) -> float | None:
        """Mean cloud derivation time, excluding depth reads and viewer logging."""
        if self.pointcloud_processed_frames == 0:
            return None
        return self.pointcloud_processing_ns / self.pointcloud_processed_frames / 1e6

    def close(self, error: BaseException | None = None) -> None:
        failure = error
        for attribute, method, stage in (
            ("reader", "close", "reader_close"), ("rr", "disconnect", "rerun_disconnect"),
        ):
            resource = getattr(self, attribute, None)
            if resource is None:
                continue
            try:
                getattr(resource, method)()
            except BaseException as exc:
                if failure is None:
                    failure = exc
                else:
                    logger.exception("%s also failed", stage)
            else:
                setattr(self, attribute, None)
        if failure is not None and error is None:
            raise failure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Visualize Raw episodes with Rerun 3D."
    )
    parser.add_argument(
        "episode",
        type=str,
        help="Path to a current Raw episode directory (data.h5/rgb.mp4).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit number of state frames to load.",
    )
    parser.add_argument(
        "--config", type=Path, help="Trusted Python experiment file exporting CONFIG: RealConfig"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print structure summary and exit without starting a Rerun viewer.",
    )
    parser.add_argument(
        "--point-cloud",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display the camera point cloud; FK hand cloud is always shown (default: enabled).",
    )
    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be a positive integer")
    episode_path = Path(args.episode).expanduser()
    if not episode_path.is_dir():
        logger.error("Episode not found: %s", episode_path)
        return 1

    if args.info:
        print_episode_info(str(episode_path))
        return 0

    runtime = load_config(path=args.config)
    from dexmani_real.utils.log import configure_logging

    configure_logging()
    pointcloud_config = None
    if args.point_cloud:
        pointcloud_config = runtime.pointcloud

    viz = EpisodeVisualizer(
        str(episode_path),
        table_plane_path=runtime.environment.table_plane_path,
        max_frames=args.max_frames,
        pointcloud_config=pointcloud_config,
        hand_pointcloud_config=runtime.hand_pointcloud,
    )
    try:
        logger.info("Logging %d frames to Rerun...", viz.frame_count)
        logged_frames = 0
        for step, rgb in enumerate(islice(viz.reader.iter_rgb_frames(), viz.frame_count)):
            viz.log_step(step, rgb)
            logged_frames += 1
            if step % 500 == 0:
                logger.info("  frame %d/%d", step, viz.frame_count)
        if logged_frames != viz.frame_count:
            raise ValueError(f"RGB missing at frame {logged_frames}")
        mean_pointcloud_ms = viz.mean_pointcloud_processing_ms
        if mean_pointcloud_ms is not None:
            logger.info(
                "Point-cloud derivation average (excluding RGB-D I/O and rendering): %.2f ms/frame",
                mean_pointcloud_ms,
            )
        if viz.empty_cloud_count:
            logger.warning(
                "%d point-cloud frames were empty and explicitly cleared",
                viz.empty_cloud_count,
            )
        logger.info("数据已发送，脚本结束；可在 Rerun 中继续浏览。")
    finally:
        viz.close(sys.exc_info()[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
