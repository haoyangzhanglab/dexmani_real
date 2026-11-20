"""离线查看 Raw episode 的观测与点云，或输出数据概要。"""

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

import cv2
import numpy as np

from dexmani_real.config.perception import HandCloudConfig, PointCloudConfig
from dexmani_real.config.real import load_config
from dexmani_real.data.raw import (
    TACTILE_FINGER_ORDER,
    RawReader,
)
from dexmani_real.geometry.camera import init_undistort_maps, remap_image
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


RAW_JOINT_SERIES = (
    ("State", "arm_qpos"),
    ("State", "hand_qpos"),
    ("Action", "action_arm_joint_target"),
    ("Action", "action_hand_joint_target"),
)

FINGERTIP_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 60, 60),  # thumb  — red
    (60, 255, 60),  # index  — green
    (60, 120, 255),  # middle — blue
    (255, 200, 40),  # ring   — gold
    (220, 60, 255),  # pinky  — magenta
)
# 白色末端标记与五指颜色区分。
EEF_COLOR = (255, 255, 255)


def print_episode_info(episode_path: str, *, hoi: bool = True) -> None:
    import h5py

    requested = Path(episode_path).expanduser()
    with RawReader(requested) as reader:
        print(f"Episode: {requested} -> {reader.path}")
        print(f"RGB file exists: {reader.rgb_path.is_file()}")
        for name, value in reader.meta.items():
            print(f"  {name}: {value}")

        def describe(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")

        reader.data_file.visititems(describe)
        if hoi and (reader.path / "hoi.h5").is_file():
            print("HOI: hoi.h5")
            reader.hoi_file.visititems(describe)


class EpisodeVisualizer:
    def __init__(
        self,
        episode_path: str,
        *,
        table_plane_path: str | Path | None,
        max_frames: int | None = None,
        hoi: bool = True,
        pointcloud_config: PointCloudConfig | None = None,
        hand_pointcloud_config: HandCloudConfig = HandCloudConfig(),  # noqa: B008 — 不可变配置可共享。
    ):
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive")

        episode_name = Path(episode_path).stem
        try:
            self.reader = RawReader(episode_path)
            self.frame_count = self.reader.num_frames
            if self.frame_count <= 0:
                raise ValueError("episode must contain at least one row")
            if max_frames is not None:
                self.frame_count = min(self.frame_count, max_frames)
            show_hoi = hoi and (self.reader.path / "hoi.h5").is_file()
            source_clocks = ("camera_ns", "arm_ns", "hand_ns") if show_hoi else ()
            timing = self.reader.read_timing(stop=self.frame_count, source_clocks=source_clocks)
            stamps = timing["observation_ns"]
            self.time_ns = stamps - stamps[0]

            self.object_cloud = None
            if show_hoi:
                self.object_cloud = self.reader["object_pointcloud"]
                self.hoi_valid = self.reader.read_hoi_validity(stop=self.frame_count)

            if not self.reader.rgb_path.is_file():
                raise FileNotFoundError(self.reader.rgb_path)
            self.depth = self.reader["depth"]

            self.camera_model = self.reader.read_camera()
            self.image_maps = init_undistort_maps(self.camera_model.color)
            self.T_xarm_base_from_color = self.camera_model.T_xarm_base_from_color
            if pointcloud_config is not None and self.T_xarm_base_from_color is None:
                logger.warning("No camera extrinsics: camera point cloud disabled")
            self.pointcloud_processor = None
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

            import rerun as rr

            app_id = f"DexMani - {episode_name}"
            rec_id = f"{episode_name}-{time.time_ns()}"
            rr.init(app_id, recording_id=rec_id)
            self.rr = rr
            rr.spawn()
            series = []
            series_groups: dict[str, list[str]] = {}
            force_labels = tuple(f"{axis} (SDK / 255)" for axis in ("Fx", "Fy", "Fz"))
            joint_series = (
                (group, f"{group.lower()}/{field}", field, None) for group, field in RAW_JOINT_SERIES
            )
            for group, path, key, labels in (
                *joint_series,
                ("Contact Force", "state/hand_contact_mag", "hand_contact_mag",
                 tuple(f"{finger} (SDK / 255)" for finger in TACTILE_FINGER_ORDER.split(","))),
                ("Contact Force", "state/hand_force_thumb", "hand_force_thumb", force_labels),
                ("Contact Force", "state/hand_force_index", "hand_force_index", force_labels),
            ):
                values = self.state[key]
                if labels is None:
                    labels = tuple(str(i) for i in range(values.shape[1]))
                series.append((path, values, labels))
                series_groups.setdefault(group, []).append(path)
            hoi_series = self.log_recording_data(series, timing if show_hoi else None)
            if hoi_series:
                series_groups["HOI"] = hoi_series
            rr.send_blueprint(
                episode_blueprint(
                    series_groups.items(),
                    camera_in_base=self.T_xarm_base_from_color is not None,
                ),
                make_active=True, make_default=True,
            )
        except BaseException as exc:
            self.close(exc)
            raise

    def preload_state(self) -> dict[str, np.ndarray]:
        state = {
            field: self.reader[field][: self.frame_count] for _, field in RAW_JOINT_SERIES
        }

        contact = self.reader["hand_contact"][: self.frame_count]
        state["hand_contact_mag"] = np.hypot.reduce(contact, axis=2, dtype=np.float64)
        state["hand_force_thumb"] = contact[:, 0, :]
        state["hand_force_index"] = contact[:, 1, :]

        return state

    def log_recording_data(
        self, series: list[tuple[str, np.ndarray, tuple[str, ...]]], hoi_timing: np.ndarray | None,
    ) -> list[str]:
        steps = np.arange(self.frame_count)
        indexes = [
            self.rr.TimeSequenceColumn("step", steps),
            self.rr.TimeNanosColumn("time", self.time_ns),
        ]
        for path, values, labels in series:
            log_series(path, values, labels=labels, indexes=indexes)
        log_camera_geometry(self.camera_model.color, self.T_xarm_base_from_color)
        if hoi_timing is None:
            return []

        diagnostics = {"hoi/valid": self.hoi_valid}
        camera_ns = hoi_timing["camera_ns"]
        for device in ("arm", "hand"):
            diagnostics[f"hoi/{device}_source_offset_ms"] = (hoi_timing[f"{device}_ns"] - camera_ns) / 1e6

        annotations = self.reader.hoi_file
        expected = (self.frame_count, annotations["object_valid"].shape[1])
        for name, dataset in annotations["quality"].items():
            quality = dataset[: self.frame_count]
            if quality.shape != expected:
                raise ValueError(f"Invalid HOI quality/{name}: expected selected shape {expected}")
            for index, values in enumerate(quality.T):
                # 跟踪后端以 -1 标记未计算的指标。
                diagnostics[f"hoi/object_{index}/{name}"] = np.where(values == -1, np.nan, values)

        for entity, values in diagnostics.items():
            self.rr.send_columns(entity, indexes=indexes, columns=self.rr.Scalar.columns(scalar=values))
        return list(diagnostics)

    def log_step(self, step_idx: int, rgb: np.ndarray) -> None:
        self.rr.set_time_sequence("step", step_idx)
        self.rr.set_time_nanos("time", int(self.time_ns[step_idx]))
        depth = self.log_camera(step_idx, rgb)
        self.log_pointcloud(rgb, depth)
        self.log_hand_geometry(step_idx)
        if self.object_cloud is not None:
            points = None
            if self.hoi_valid[step_idx]:
                points = self.object_cloud[step_idx]
                if not np.isfinite(points).all():
                    raise ValueError(
                        f"object_pointcloud contains nonfinite coordinates at row {step_idx}"
                    )
            log_points("object_pointcloud", points, colors=[255, 170, 0])

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
        log_points("hand_pointcloud", cloud, colors=[100, 180, 255])
        log_points(
            "fingertips", fingertips,
            colors=FINGERTIP_COLORS, radii=0.012,
        )
        log_points(
            "eef", pose[None, :3, 3] if pose is not None else None,
            colors=[EEF_COLOR], radii=0.020,
        )

    def log_camera(self, step_idx: int, rgb: np.ndarray) -> np.ndarray:
        depth = self.depth[step_idx]
        depth_scale_m = cast(float, self.camera_model.depth_scale_m)
        log_rgbd(
            remap_image(rgb, self.image_maps, interpolation=cv2.INTER_LINEAR),
            remap_image(depth, self.image_maps, interpolation=cv2.INTER_NEAREST),
            depth_scale_m=depth_scale_m,
            depth_range_m=(0, 10000 * depth_scale_m),
        )
        # 后续点云仍按原始像素配对取色。
        return depth

    def log_pointcloud(self, rgb: np.ndarray, depth: np.ndarray) -> None:
        if self.pointcloud_processor is None:
            return
        log_points("pcd", self.pointcloud_processor.process(depth, rgb))

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
    parser.add_argument("--no-hoi", action="store_true", help="View Raw without opening HOI annotations.")
    args = parser.parse_args(argv)
    episode_path = Path(args.episode).expanduser()
    if not episode_path.is_dir():
        logger.error("Episode not found: %s", episode_path)
        return 1

    if args.info:
        print_episode_info(str(episode_path), hoi=not args.no_hoi)
        return 0

    runtime = load_config(path=args.config)
    from dexmani_real.utils.log import configure_logging

    configure_logging()
    viz = EpisodeVisualizer(
        str(episode_path),
        table_plane_path=runtime.environment.table_plane_path,
        max_frames=args.max_frames,
        hoi=not args.no_hoi,
        pointcloud_config=runtime.pointcloud if args.point_cloud else None,
        hand_pointcloud_config=runtime.hand_pointcloud,
    )
    try:
        logger.info("Logging %d frames to Rerun...", viz.frame_count)
        stop = viz.frame_count if viz.frame_count < viz.reader.num_frames else None
        for step, rgb in enumerate(islice(viz.reader.iter_rgb_frames(), stop)):
            viz.log_step(step, rgb)
    finally:
        viz.close(sys.exc_info()[1])
    logger.info("数据已发送，脚本结束；可在 Rerun 中继续浏览。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
