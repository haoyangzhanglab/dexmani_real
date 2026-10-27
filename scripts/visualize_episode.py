#!/usr/bin/env python3
"""离线查看 Raw episode 的观测与点云，或输出数据概要。"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("RUST_LOG", "error")

import numpy as np

from dexmani_real.calibration.table_plane import load_table_plane
from dexmani_real.config import load_experiment_config
from dexmani_real.control.config import TableCollisionConfig
from dexmani_real.data.raw import RawReader
from dexmani_real.geometry.kinematics import HandFK, eef_poses
from dexmani_real.geometry.pose import (
    quat_wxyz_to_rotmat,
    validate_unit_quaternion_wxyz,
)
from dexmani_real.perception.config import PointCloudConfig
from dexmani_real.perception.hand_cloud import hand_geometry
from dexmani_real.perception.pointcloud import PointCloudProcessor
from dexmani_real.robot.model import XHAND_FINGERTIP_LINK_NAMES
from dexmani_real.utils.visualization import (
    log_camera_geometry,
    log_points,
    log_rgbd,
    log_series,
    viewer_blueprint,
)

logger = logging.getLogger("dexmani_real.cli.visualize_episode")


SERIES_GROUPS = {
    "state": ("arm_qpos", "hand_qpos"),
    "action": ("action_arm_joint_target", "action_hand_joint_target"),
}
FORCE_SERIES = {
    "hand_contact_mag": (
        "thumb (SDK native)",
        "index (SDK native)",
        "middle (SDK native)",
        "ring (SDK native)",
        "pinky (SDK native)",
    ),
    "hand_force_thumb": ("Fx", "Fy", "Fz"),
    "hand_force_index": ("Fx", "Fy", "Fz"),
}

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
    with RawReader(episode_path) as reader:
        keys = sorted(reader.fields - {"rgb"})
        print(f"Episode:    {episode_path}")
        print(f"Control/depth rows: {reader.num_frames}")
        print(
            "Time:",
            f"relative seconds ({reader.meta['timestamp_source']})",
        )
        print()

        print("Meta:")
        for attr in sorted(reader.meta.keys()):
            print(f"  {attr}: {reader.meta[attr]}")
        print()

        print(f"Datasets ({len(keys)}):")
        for key in keys:
            ds = reader[key]
            print(f"  {key:<28s} shape={str(ds.shape):<22s} dtype={str(ds.dtype):<10s}")

        print()
        if reader.num_frames:
            for key in ("arm_qpos", "hand_qpos"):
                if key not in reader.fields:
                    continue
                q = reader[key][:]
                low = np.array2string(q.min(axis=0), precision=3, suppress_small=True)
                high = np.array2string(q.max(axis=0), precision=3, suppress_small=True)
                print(f"{key} range: {low} .. {high}")


class EpisodeVisualizer:
    def __init__(
        self,
        episode_path: str,
        *,
        handbase_position_eef_m: tuple[float, float, float],
        handbase_quat_eef_wxyz: tuple[float, float, float, float],
        fingertip_link_names: tuple[str, ...],
        table_config: TableCollisionConfig,
        max_frames: int | None = None,
        pointcloud_config: PointCloudConfig | None = None,
    ):
        import rerun as rr

        self.rr = rr

        episode_name = Path(episode_path).stem
        self.fingertip_link_names = fingertip_link_names
        self.reader = RawReader(episode_path)
        try:
            self.timestamps_s = self.reader.timestamps_s
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
            self.rgb_frames = iter(self.reader.iter_rgb_frames())
            self.rgb_index = -1
            self.camera_model = self.reader.read_camera()
            self.T_xarm_base_from_color = self.camera_model.T_xarm_base_from_color
            if self.T_xarm_base_from_color is None:
                logger.warning("No camera extrinsics: displaying pixels without point cloud")
            self.handbase_position_eef_m = np.asarray(handbase_position_eef_m, dtype=np.float64)
            if self.handbase_position_eef_m.shape != (3,) or not np.isfinite(
                self.handbase_position_eef_m
            ).all():
                raise ValueError("hand base position must be a finite (3,) vector")
            handbase_quat = validate_unit_quaternion_wxyz(
                handbase_quat_eef_wxyz, name="hand base quaternion"
            )
            self.handbase_rotation_eef = quat_wxyz_to_rotmat(handbase_quat)
            self.pointcloud_processor = None
            self.empty_cloud_count = 0
            self.pointcloud_processing_ns = 0
            self.pointcloud_processed_frames = 0
            if pointcloud_config is not None and self.T_xarm_base_from_color is not None:
                table_plane = load_table_plane(table_config) if pointcloud_config.remove_table else None
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

            self.state = self.preload_state()

            app_id = f"DexMani - {episode_name}"
            rec_id = f"{episode_name}-{time.time_ns()}"
            rr.init(
                app_id,
                recording_id=rec_id,
                spawn=True,
            )
            series_groups = []
            for category, keys in SERIES_GROUPS.items():
                series_keys = (*keys, *FORCE_SERIES) if category == "state" else keys
                series_groups.append((category.title(), [f"{category}/{key}" for key in series_keys]))
            # 当前布局覆盖 viewer 记忆的旧布局。
            rr.send_blueprint(
                viewer_blueprint(
                    series_groups,
                    camera_in_base=self.T_xarm_base_from_color is not None,
                ),
                make_active=True, make_default=True,
            )
            self.log_static()
        except BaseException:
            self.close()
            raise

    def preload_state(self) -> dict[str, np.ndarray]:
        state = {
            key: self.reader[key][: self.frame_count] for keys in SERIES_GROUPS.values() for key in keys
        }

        arm = state["arm_qpos"]
        hand = state["hand_qpos"]
        arm_valid = np.all(np.isfinite(arm), axis=1)
        hand_valid = arm_valid & np.all(np.isfinite(hand), axis=1)
        state["arm_ee"] = np.full((self.frame_count, 9), np.nan)
        state["hand_fingertip"] = np.full((self.frame_count, 5, 3), np.nan, dtype=np.float32)
        if np.any(arm_valid):
            state["arm_ee"][arm_valid] = eef_poses(arm[arm_valid])
        if np.any(hand_valid):
            hand_fk = HandFK(self.fingertip_link_names)
            state["hand_fingertip"][hand_valid] = hand_geometry(
                state["arm_ee"][hand_valid],
                hand[hand_valid],
                hand_fk=hand_fk,
                handbase_position_eef_m=self.handbase_position_eef_m,
                handbase_rotation_eef=self.handbase_rotation_eef,
            )["fingertip_xyz"]

        contact = self.reader["hand_contact"][: self.frame_count]
        state["hand_contact_mag"] = np.linalg.norm(contact, axis=2)
        state["hand_force_thumb"] = contact[:, 0, :]
        state["hand_force_index"] = contact[:, 1, :]

        return state

    def log_static(self) -> None:
        for category, keys in SERIES_GROUPS.items():
            for key in keys:
                log_series(
                    f"{category}/{key}", labels=[str(i) for i in range(self.state[key].shape[1])],
                )
        for key, labels in FORCE_SERIES.items():
            log_series(f"state/{key}", labels=labels)
        log_camera_geometry(self.camera_model.color, self.T_xarm_base_from_color)

    def log_step(self, step_idx: int) -> None:
        self.rr.set_time_sequence("step", step_idx)
        self.rr.set_time_seconds("time", float(self.timestamps_s[step_idx]))
        rgb, depth = self.log_camera(step_idx)
        self.log_pointcloud(rgb, depth)
        log_points(
            "fingertips", self.state["hand_fingertip"][step_idx],
            colors=FINGERTIP_COLORS, radii=0.012,
        )
        pose = self.state["arm_ee"][step_idx]
        log_points(
            "eef", pose[None, :3] if np.isfinite(pose).all() else None,
            colors=[EEF_COLOR], radii=0.020,
        )
        for category, keys in SERIES_GROUPS.items():
            for key in keys:
                log_series(f"{category}/{key}", self.state[key][step_idx])
        for key in FORCE_SERIES:
            log_series(f"state/{key}", self.state[key][step_idx])

    def log_camera(self, step_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if step_idx != self.rgb_index + 1:
            raise ValueError("viewer reads frames sequentially")
        rgb = next(self.rgb_frames, None)
        if rgb is None:
            raise ValueError(f"RGB missing at frame {step_idx}")
        self.rgb_index = step_idx
        depth = np.asarray(self.reader["depth"][step_idx])
        log_rgbd(
            rgb, depth, depth_scale_m=self.camera_model.depth_scale_m,
            depth_range_m=(0, 10000 * self.camera_model.depth_scale_m),
        )
        return rgb, depth

    def log_pointcloud(self, rgb: np.ndarray, depth: np.ndarray) -> None:
        if self.pointcloud_processor is None:
            return
        started_ns = time.perf_counter_ns()
        cloud = self.pointcloud_processor.process(depth, rgb)
        self.pointcloud_processing_ns += time.perf_counter_ns() - started_ns
        self.pointcloud_processed_frames += 1
        if cloud is None:
            self.empty_cloud_count += 1
        log_points("pcd", cloud)

    @property
    def mean_pointcloud_processing_ms(self) -> float | None:
        """Mean cloud derivation time, excluding depth reads and viewer logging."""
        if self.pointcloud_processed_frames == 0:
            return None
        return self.pointcloud_processing_ns / self.pointcloud_processed_frames / 1e6

    def close(self) -> None:
        try:
            if self.reader is not None:
                self.reader.close()
                self.reader = None  # type: ignore[assignment]
        finally:
            try:
                self.rr.disconnect()
            except Exception:
                logger.warning("Rerun disconnect failed", exc_info=True)


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
        "--config", type=Path, help="Optional experiment YAML for current cloud and hand geometry"
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
        help=("Display the canonical point cloud (default: enabled)."),
    )
    parser.add_argument(
        "--pointcloud-num-points",
        type=int,
        default=None,
        help="Override the point-cloud count used for this offline view.",
    )
    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be a positive integer")
    if args.point_cloud is False and args.pointcloud_num_points is not None:
        parser.error("--pointcloud-num-points requires --point-cloud")
    episode_path = Path(args.episode).expanduser().resolve()
    if not episode_path.is_dir():
        logger.error("Episode not found: %s", episode_path)
        return 1

    if args.info:
        print_episode_info(str(episode_path))
        return 0

    runtime = load_experiment_config(yaml_path=args.config)
    from dexmani_real.utils.log import configure_logging

    configure_logging()
    pointcloud_config = None
    if args.point_cloud:
        pointcloud_config = runtime.pointcloud
        if args.pointcloud_num_points is not None:
            pointcloud_config = replace(pointcloud_config, num_points=args.pointcloud_num_points)

    viz = EpisodeVisualizer(
        str(episode_path),
        handbase_position_eef_m=runtime.hand.T_eef_handbase_pos_xyz,
        handbase_quat_eef_wxyz=runtime.hand.T_eef_handbase_quat_wxyz,
        fingertip_link_names=XHAND_FINGERTIP_LINK_NAMES,
        table_config=runtime.environment.table,
        max_frames=args.max_frames,
        pointcloud_config=pointcloud_config,
    )
    try:
        logger.info("Logging %d frames to Rerun...", viz.frame_count)
        for step in range(viz.frame_count):
            viz.log_step(step)
            if step % 500 == 0:
                logger.info("  frame %d/%d", step, viz.frame_count)
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
        viz.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
