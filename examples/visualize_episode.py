#!/usr/bin/env python3
"""Usage: python examples/visualize_episode.py EPISODE [--info] [--max-frames N]

Offline raw-episode Rerun viewer with production point clouds; --info prints a summary.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("RUST_LOG", "error")

import h5py
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from dexmani_real.config.hardware import HandParams
from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.dataset.pointcloud import (
    RawEpisodePointCloudDeriver,
    load_raw_episode_base_from_color,
    load_raw_episode_camera_model,
)
from dexmani_real.ipc.schema import validate_point_cloud_array
from dexmani_real.planning.kinematics.arm_fk import compute_eef_pose_history_xarm_base
from dexmani_real.planning.kinematics.fingertip import compute_fingertip_history_xarm_base
from dexmani_real.planning.kinematics.hand_fk import HandKinematics
from dexmani_real.recording import EpisodeReader
from dexmani_real.robot.model import HAND_FINGERTIP_SHAPE, XHAND_RIGHT_URDF_PATH
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


_SERIES_GROUPS = {
    "state": ("arm_effort", "arm_qpos", "arm_qvel", "hand_current", "hand_qpos"),
    "action": ("action_arm_joint_target", "action_hand_joint_target"),
    "flags": ("frame_valid",),
}
_FORCE_SERIES = {
    "hand_contact_mag": (
        "thumb (SDK-scaled)",
        "index (SDK-scaled)",
        "middle (SDK-scaled)",
        "ring (SDK-scaled)",
        "pinky (SDK-scaled)",
    ),
    "hand_force_thumb": ("Fx", "Fy", "Fz"),
    "hand_force_index": ("Fx", "Fy", "Fz"),
}

_FINGERTIP_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 60, 60),  # thumb  — red
    (60, 255, 60),  # index  — green
    (60, 120, 255),  # middle — blue
    (255, 200, 40),  # ring   — gold
    (220, 60, 255),  # pinky  — magenta
)
# Fixed, easily distinguishable from the five finger colors.
_EEF_COLOR = (255, 255, 255)


def _eef_position_or_none(eef_row: np.ndarray) -> np.ndarray | None:
    row = np.asarray(eef_row, dtype=np.float64)
    if row.shape != (9,) or not np.all(np.isfinite(row)):
        return None
    return row[:3]


def _fingertip_positions_or_none(fingertip_row: np.ndarray) -> np.ndarray | None:
    row = np.asarray(fingertip_row, dtype=np.float32)
    if row.ndim != 2 or row.shape != HAND_FINGERTIP_SHAPE or not np.all(np.isfinite(row)):
        return None
    return row


def print_episode_info(h5_path: str) -> None:
    with EpisodeReader(h5_path) as reader:
        f = reader.h5f
        keys = sorted(k for k in f.keys() if isinstance(f[k], h5py.Dataset))
        print(f"Episode:    {h5_path}")
        print(f"Control/depth rows: {reader.num_frames}")
        print()

        print("Meta:")
        for attr in sorted(f["meta"].attrs.keys()):
            print(f"  {attr}: {f['meta'].attrs[attr]}")
        print()

        print(f"Datasets ({len(keys)}):")
        for key in keys:
            ds = f[key]
            print(f"  {key:<28s} shape={str(ds.shape):<22s} dtype={str(ds.dtype):<10s}")

        print()
        if reader.num_frames:
            for key in ("arm_qpos", "hand_qpos"):
                q = f[key][:]
                low = np.array2string(q.min(axis=0), precision=3, suppress_small=True)
                high = np.array2string(q.max(axis=0), precision=3, suppress_small=True)
                print(f"{key} range: {low} .. {high}")
        print(f"episode valid: {reader.episode_valid}")
        frame_valid = f["frame_valid"][:]
        rate = f"{np.mean(frame_valid):.2%}" if reader.num_frames else "n/a (empty episode)"
        print(f"frame valid rate: {rate}")


class EpisodeVisualizer:
    def __init__(
        self,
        h5_path: str,
        max_frames: int | None = None,
        point_cloud: bool = True,
        pointcloud_config: PointCloudConfig | None = None,
        table_plane_abcd: tuple[float, float, float, float] | None = None,
    ):
        self._h5_path = Path(h5_path)
        self._reader = EpisodeReader(h5_path)
        try:
            self._h5f = self._reader.h5f
            self._logical_dt_s = self._reader.dt

            self._rgb_cache = self._reader.read_camera_all("rgb")
            self._depth_cache = self._reader.read_camera_all("depth")
            logger.info("Pre-decoded %d RGB-D frames", self._rgb_cache.shape[0])

            meta = self._h5f["meta"].attrs
            self._camera_model = load_raw_episode_camera_model(self._reader)
            self._camera_K = self._camera_model.geometry.color.matrix()
            self._depth_meter = 1.0 / self._camera_model.depth_scale_m
            self._T_xarm_base_from_color = load_raw_episode_base_from_color(self._reader)
            self._handbase_position_eef_m = np.asarray(
                meta["handbase_position_eef_m"], dtype=np.float64
            )
            self._handbase_quat_eef_wxyz = np.asarray(
                meta["handbase_quat_eef_wxyz"], dtype=np.float64
            )
            self._pointcloud_deriver: RawEpisodePointCloudDeriver | None = None
            self._empty_pointcloud_frames = 0
            self._pointcloud_processing_ns = 0
            self._pointcloud_processed_frames = 0
            if point_cloud:
                if pointcloud_config is None:
                    raise ValueError("point-cloud visualization requires resolved config")
                self._pointcloud_deriver = RawEpisodePointCloudDeriver(
                    reader=self._reader,
                    camera=self._camera_model,
                    T_xarm_base_from_color=self._T_xarm_base_from_color,
                    pointcloud=pointcloud_config,
                    table_plane_abcd=table_plane_abcd,
                )
                logger.info(
                    "Canonical point cloud enabled: N=%d table=%s",
                    pointcloud_config.num_points,
                    "enabled" if table_plane_abcd is not None else "disabled",
                )

            self._T = self._resolve_frame_count(max_frames)
            logger.info("Frames=%d", self._T)

            self._state = self._preload_state()

            self._blueprint = self._build_blueprint()
            app_id = f"DexMani - {self._h5_path.stem}"
            rec_id = f"{self._h5_path.stem}-{time.time_ns()}"
            rr.init(
                app_id,
                recording_id=rec_id,
                spawn=True,
                default_blueprint=self._blueprint,
            )
            rr.send_blueprint(
                blueprint=self._blueprint
            )  # force-override any cached blueprint for this app_id
            self._log_static()
        except BaseException:
            self.close()
            raise

    def _resolve_frame_count(self, max_frames: int | None) -> int:
        raw = self._reader.num_frames
        if raw <= 0:
            raise ValueError("episode /meta num_frames must be positive")
        camera_counts = [self._rgb_cache.shape[0], self._depth_cache.shape[0]]
        if any(count != raw for count in camera_counts):
            raise ValueError(f"camera frame counts {camera_counts} do not match grid length {raw}")
        if max_frames is not None:
            return min(raw, max_frames)
        return raw

    def _preload_state(self) -> dict[str, np.ndarray]:
        state = {key: self._h5f[key][: self._T] for keys in _SERIES_GROUPS.values() for key in keys}

        arm = state["arm_qpos"]
        hand = state["hand_qpos"]
        arm_valid = np.all(np.isfinite(arm), axis=1)
        hand_valid = arm_valid & np.all(np.isfinite(hand), axis=1)
        state["arm_ee"] = np.full((self._T, 9), np.nan)
        state["hand_fingertip"] = np.full((self._T, 5, 3), np.nan, dtype=np.float32)
        if np.any(arm_valid):
            state["arm_ee"][arm_valid] = compute_eef_pose_history_xarm_base(arm[arm_valid])
        if np.any(hand_valid):
            hand_fk = HandKinematics(
                str(XHAND_RIGHT_URDF_PATH), list(HandParams.fingertip_link_names)
            )
            state["hand_fingertip"][hand_valid] = compute_fingertip_history_xarm_base(
                arm[hand_valid],
                hand[hand_valid],
                hand_fk=hand_fk,
                handbase_position_eef_m=self._handbase_position_eef_m,
                handbase_quat_eef_wxyz=self._handbase_quat_eef_wxyz,
                eef_pose_history=state["arm_ee"][hand_valid],
            )

        contact = self._h5f["hand_contact"][: self._T]
        state["hand_contact_mag"] = np.linalg.norm(contact, axis=2)
        state["hand_force_thumb"] = contact[:, 0, :]
        state["hand_force_index"] = contact[:, 1, :]

        return state

    def _build_blueprint(self) -> rrb.Blueprint:
        ts_verticals = []
        for category, keys in _SERIES_GROUPS.items():
            if category == "state":
                keys = (*keys, *_FORCE_SERIES)
            views = [rrb.TimeSeriesView(origin=f"{category}/{key}", name=key) for key in keys]
            ts_verticals.append(rrb.Vertical(contents=views, name=category.title()))
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial2DView(origin="camera/color/rgb", name="RGB"),
                    rrb.Spatial2DView(origin="depth/image", name="Depth"),
                    name="Camera",
                ),
                rrb.Spatial3DView(origin="/", name="Point Cloud", background=[0.12, 0.12, 0.14]),
                rrb.Tabs(contents=ts_verticals, active_tab=0, name="Time Series"),
            )
        )

    def _log_static(self) -> None:
        for category, keys in _SERIES_GROUPS.items():
            for key in keys:
                arr = self._state[key]
                base = f"{category}/{key}"
                if arr.ndim == 1:
                    rr.log(base, rr.SeriesLine(name=key), static=True)
                else:
                    for i in range(arr.shape[1]):
                        rr.log(f"{base}/{i}", rr.SeriesLine(name=f"{i}"), static=True)

        h, w = self._rgb_cache.shape[1:3]
        rr.log(
            "camera/color",
            rr.Pinhole(
                image_from_camera=self._camera_K,
                resolution=[w, h],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=1.25,
            ),
            static=True,
        )
        logger.info("Camera pinhole logged (%dx%d)", w, h)
        rr.log(
            "camera/color",
            rr.Transform3D(
                translation=self._T_xarm_base_from_color[:3, 3],
                mat3x3=self._T_xarm_base_from_color[:3, :3],
            ),
            static=True,
        )

        for key, labels in _FORCE_SERIES.items():
            for i, label in enumerate(labels):
                rr.log(f"state/{key}/{i}", rr.SeriesLine(name=label), static=True)

    def log_step(self, step_idx: int) -> None:
        rr.set_time_sequence("step", step_idx)
        rr.set_time_seconds("time", step_idx * self._logical_dt_s)
        self._log_camera(step_idx)
        self._log_pointcloud(step_idx)
        self._log_fingertips(step_idx)
        self._log_eef(step_idx)
        self._log_time_series(step_idx)

    def _log_camera(self, step_idx: int) -> None:
        rr.log("camera/color/rgb", rr.Image(self._rgb_cache[step_idx]))
        rr.log(
            "depth/image",
            rr.DepthImage(
                self._depth_cache[step_idx],
                meter=self._depth_meter,
                depth_range=(0, 10000),  # Clamp outliers to stabilize the colormap.
            ),
        )

    def _log_pointcloud(self, step_idx: int) -> None:
        if self._pointcloud_deriver is None:
            return
        started_ns = time.perf_counter_ns()
        cloud = self._pointcloud_deriver.derive(step_idx, self._rgb_cache[step_idx])
        self._pointcloud_processing_ns += time.perf_counter_ns() - started_ns
        self._pointcloud_processed_frames += 1
        if cloud is None:
            self._empty_pointcloud_frames += 1
            rr.log("pcd", rr.Clear(recursive=False))
            return
        cloud = validate_point_cloud_array(
            cloud,
            num_points=self._pointcloud_deriver.pointcloud.num_points,
            label=f"point_cloud[{step_idx}]",
        )
        colors = (cloud[:, 3:] * 255.0).astype(np.uint8)
        rr.log(
            "pcd",
            rr.Points3D(positions=cloud[:, :3], colors=colors, radii=0.003),
        )

    def _log_fingertips(self, step_idx: int) -> None:
        fp = _fingertip_positions_or_none(self._state["hand_fingertip"][step_idx])
        if fp is None:
            # Clear stale geometry on invalid rows.
            rr.log("fingertips", rr.Clear(recursive=False))
            return

        rr.log(
            "fingertips",
            rr.Points3D(
                positions=fp,
                colors=np.array(_FINGERTIP_COLORS, dtype=np.uint8),
                radii=0.012,
            ),
        )

    def _log_eef(self, step_idx: int) -> None:
        position = _eef_position_or_none(self._state["arm_ee"][step_idx])
        if position is None:
            # Clear stale geometry on invalid rows.
            rr.log("eef", rr.Clear(recursive=False))
            return
        rr.log(
            "eef",
            rr.Points3D(
                positions=position[None, :],
                colors=np.array([_EEF_COLOR], dtype=np.uint8),
                radii=0.020,  # EEF sphere is intentionally larger than fingertips
            ),
        )

    def _log_time_series(self, step_idx: int) -> None:
        for category, keys in _SERIES_GROUPS.items():
            for key in keys:
                arr = self._state[key]
                base = f"{category}/{key}"
                if arr.ndim == 1:
                    rr.log(base, rr.Scalar(float(arr[step_idx])))
                else:
                    for i in range(arr.shape[1]):
                        rr.log(f"{base}/{i}", rr.Scalar(float(arr[step_idx, i])))

        for key in _FORCE_SERIES:
            arr = self._state[key]
            for i in range(arr.shape[1]):
                rr.log(f"state/{key}/{i}", rr.Scalar(float(arr[step_idx, i])))

    @property
    def num_steps(self) -> int:
        return self._T

    @property
    def empty_pointcloud_frames(self) -> int:
        return self._empty_pointcloud_frames

    @property
    def mean_pointcloud_processing_ms(self) -> float | None:
        if self._pointcloud_processed_frames == 0:
            return None
        return self._pointcloud_processing_ns / self._pointcloud_processed_frames / 1e6

    def close(self) -> None:
        if hasattr(self, "_reader") and self._reader is not None:
            self._reader.close()
            self._reader = None  # type: ignore[assignment]
            self._h5f = None  # type: ignore[assignment]
        try:
            rr.disconnect()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Visualize DexMani HDF5 teleop episodes with Rerun 3D."
    )
    parser.add_argument(
        "episode",
        type=str,
        help="Path to a current raw episode directory (data.h5/depth.h5/rgb.mp4).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit number of state frames to load.",
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
        help="Override the no-table point-cloud count used for this offline view.",
    )
    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be a positive integer")
    if args.point_cloud is False and args.pointcloud_num_points is not None:
        parser.error("--pointcloud-num-points requires --point-cloud")
    h5_path = Path(args.episode).expanduser().resolve()
    if not h5_path.is_dir():
        logger.error("Episode not found: %s", h5_path)
        return 1

    if args.info:
        print_episode_info(str(h5_path))
        return 0

    pointcloud_config = None
    if args.point_cloud:
        # Raw stores no historical table plane. Keep table removal disabled
        # rather than reading today's runtime calibration for an offline episode.
        pointcloud_config = PointCloudConfig(remove_table=False)
        if args.pointcloud_num_points is not None:
            pointcloud_config = replace(pointcloud_config, num_points=args.pointcloud_num_points)

    viz = EpisodeVisualizer(
        str(h5_path),
        max_frames=args.max_frames,
        point_cloud=args.point_cloud,
        pointcloud_config=pointcloud_config,
    )
    try:
        logger.info("Logging %d frames to Rerun...", viz.num_steps)
        for step in range(viz.num_steps):
            viz.log_step(step)
            if step % 500 == 0:
                logger.info("  frame %d/%d", step, viz.num_steps)
        mean_pointcloud_ms = viz.mean_pointcloud_processing_ms
        if mean_pointcloud_ms is not None:
            logger.info(
                "Point-cloud processing average: %.2f ms/frame",
                mean_pointcloud_ms,
            )
        if viz.empty_pointcloud_frames:
            logger.warning(
                "%d point-cloud frames were empty and explicitly cleared",
                viz.empty_pointcloud_frames,
            )
        logger.info("Done. Close the Rerun window to exit.")
    finally:
        viz.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
