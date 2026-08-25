#!/usr/bin/env python3
"""Usage: ``python examples/visualize_episode_processed.py PROCESSED.h5 [--info] [--max-frames N]``.

Self-contained Rerun-based visualizer for processed HDF5 v7
(``dexmani-real-processed-hdf5``) artifacts written by
``examples/process_episodes.py``.  Offline only: connects to no hardware, writes
no files; opens a Rerun viewer window (or prints a structure summary with
``--info``).

Unlike ``examples/visualize_episode.py``, which derives a current-config point
cloud preview from raw v22 RGB-D, this reads a single ``.h5`` file whose RGB,
depth, and point cloud are already stored grid-aligned at ``(T, ...)`` with
processing provenance. The point cloud is precomputed in the xArm base frame,
so nothing is back-projected here.

Examples::

  python examples/visualize_episode_processed.py episodes_processed/<task>/episode_<ts>.h5
  python examples/visualize_episode_processed.py episodes_processed/<task>/episode_<ts>.h5 --info
  python examples/visualize_episode_processed.py episodes_processed/<task>/episode_<ts>.h5 --max-frames 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Set a quiet Rerun logging default unless the operator already configured one.
os.environ.setdefault("RUST_LOG", "error")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import h5py
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.data.process import PROCESSED_SCHEMA_NAME, PROCESSED_SCHEMA_VERSION
from dexmani_real.ipc.schema import POINT_CLOUD_FEATURE_DIM, validate_point_cloud_array
from dexmani_real.sensor.pointcloud import (
    POINT_CLOUD_COLOR_SOURCE,
    POINT_CLOUD_POLICY_ID,
    POINT_CLOUD_SAMPLING,
    POINT_CLOUD_TRANSFORM,
)
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


_FINGERTIP_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 60, 60),  # thumb  — red
    (60, 255, 60),  # index  — green
    (60, 120, 255),  # middle — blue
    (255, 200, 40),  # ring   — gold
    (220, 60, 255),  # pinky  — magenta
)

_FINGER_NAMES: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")

# Processed core modalities are fixed by the v7 contract: joint_state/action are
# arm (7) + hand (12), action_ee is eef_position (3) + eef_rot6d (6) + hand (12),
# and contact_force is one native-axis 3-vector per finger.
_ARM_JOINT_LABELS = tuple(f"arm_j{i}" for i in range(7))
_HAND_JOINT_LABELS = tuple(f"hand_j{i}" for i in range(12))


def _series_labels(key: str, dim: int) -> list[str]:
    """Human channel labels for a time-series dataset, else numeric indices."""
    if key in ("joint_state", "action") and dim == 19:
        return list(_ARM_JOINT_LABELS) + list(_HAND_JOINT_LABELS)
    if key == "action_ee" and dim == 21:
        return (
            ["ee_x", "ee_y", "ee_z"]
            + [f"ee_r{i}" for i in range(6)]
            + list(_HAND_JOINT_LABELS)
        )
    if key == "contact_force_mag" and dim == 5:
        return list(_FINGER_NAMES)
    return [str(i) for i in range(dim)]


def _present_keys(h5f: h5py.File) -> set[str]:
    """Top-level HDF5 datasets (excludes the provenance group)."""
    return {k for k in h5f.keys() if isinstance(h5f[k], h5py.Dataset)}


def _validate_pointcloud_dataset(h5f: h5py.File) -> int | None:
    """Validate the processed-file point-cloud boundary and return its N."""
    if "point_cloud" not in h5f:
        return None
    dataset = h5f["point_cloud"]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError("point_cloud must be an HDF5 dataset")
    if (
        dataset.ndim != 3
        or dataset.shape[1] <= 0
        or dataset.shape[2] != POINT_CLOUD_FEATURE_DIM
    ):
        raise ValueError(
            f"point_cloud must have shape (T, N, {POINT_CLOUD_FEATURE_DIM}), "
            f"got {dataset.shape}"
        )
    if dataset.dtype != np.dtype(np.float32):
        raise ValueError(f"point_cloud must be float32, got {dataset.dtype}")

    episode_steps = int(h5f.attrs.get("episode_steps", -1))
    if episode_steps <= 0:
        raise ValueError("episode_steps must be positive when point_cloud is present")
    if dataset.shape[0] != episode_steps:
        raise ValueError(
            "point_cloud frame count must match episode_steps, "
            f"got {dataset.shape[0]} and {episode_steps}"
        )

    num_points = int(dataset.shape[1])
    declared_shape = np.asarray(h5f.attrs.get("point_cloud_shape", ()))
    expected_shape = (num_points, POINT_CLOUD_FEATURE_DIM)
    if not np.array_equal(declared_shape, np.asarray(expected_shape)):
        raise ValueError(
            "point_cloud_shape must match the dataset, "
            f"got {declared_shape.tolist()} for {expected_shape}"
        )
    expected_attrs = {
        "point_cloud_frame": "xarm_base",
        "point_cloud_color_source": POINT_CLOUD_COLOR_SOURCE,
        "point_cloud_policy_id": POINT_CLOUD_POLICY_ID,
        "point_cloud_sampling": POINT_CLOUD_SAMPLING,
        "point_cloud_transform": POINT_CLOUD_TRANSFORM,
    }
    for name, expected in expected_attrs.items():
        actual = str(h5f.attrs.get(name, ""))
        if actual != expected:
            raise ValueError(f"{name} must be {expected!r}, got {actual!r}")
    try:
        processing_config = json.loads(str(h5f.attrs.get("processing_config_json", "")))
        pointcloud_config = processing_config["pointcloud"]
        table_plane_abcd = processing_config["table_plane_abcd"]
        if not isinstance(pointcloud_config, dict):
            raise TypeError("pointcloud config must be an object")
        resolved_pointcloud = PointCloudConfig(**pointcloud_config)
        if resolved_pointcloud.to_dict() != pointcloud_config:
            raise ValueError("persisted point-cloud config is not canonical")
        if resolved_pointcloud.num_points != num_points:
            raise ValueError("persisted point-cloud count does not match dataset")
        if table_plane_abcd is not None:
            plane = np.asarray(table_plane_abcd, dtype=np.float64)
            norm = float(np.linalg.norm(plane[:3])) if plane.shape == (4,) else 0.0
            if (
                plane.shape != (4,)
                or not np.all(np.isfinite(plane))
                or norm <= 0.0
                or plane[2] / norm <= 0.0
            ):
                raise ValueError("persisted point-cloud table plane is invalid")
        canonical_table_plane = json.dumps(
            table_plane_abcd,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            "processing_config_json has no valid pointcloud policy"
        ) from exc
    expected_hash = resolved_pointcloud.sha256
    actual_hash = str(h5f.attrs.get("point_cloud_config_sha256", ""))
    if actual_hash != expected_hash:
        raise ValueError("point_cloud_config_sha256 does not match persisted policy")
    actual_table_plane = str(h5f.attrs.get("point_cloud_table_plane_abcd_json", ""))
    if actual_table_plane != canonical_table_plane:
        raise ValueError("persisted point-cloud table plane is inconsistent")
    return num_points


def print_episode_info(h5_path: str) -> None:
    """Print a human-readable summary of the processed file without opening Rerun."""
    with h5py.File(h5_path, "r") as f:
        attrs = f.attrs
        steps = int(attrs.get("episode_steps", -1))
        source_frames = int(attrs.get("source_frames", -1))

        print(f"Processed: {h5_path}")
        print(
            f"schema: {attrs.get('schema_name', '?')} v{attrs.get('schema_version', '?')}  "
            f"profile={attrs.get('profile', '?')}  domain={attrs.get('domain', '?')}"
        )
        print(
            f"task_name: {attrs.get('task_name', '?')}   "
            f"source_episode: {attrs.get('source_episode', '?')}"
        )
        print(
            f"steps (T): {steps}   source_frames: {source_frames}   dt: {attrs.get('dt', '?')}"
        )
        if "depth_scale_m_per_unit" in attrs:
            print(
                f"depth_scale_m_per_unit: {attrs['depth_scale_m_per_unit']}   "
                f"depth_invalid_value: {attrs.get('depth_invalid_value', '?')}"
            )
        print()

        keys = sorted(_present_keys(f))
        print(f"Datasets ({len(keys)}):")
        for key in keys:
            ds = f[key]
            print(f"  {key:<22s} shape={str(ds.shape):<24s} dtype={str(ds.dtype):<10s}")
        print()

        num_points = _validate_pointcloud_dataset(f)
        if num_points is not None:
            print(
                f"point_cloud: ({num_points}, 6) float32, "
                f"frame={attrs['point_cloud_frame']}, "
                f"color={attrs['point_cloud_color_source']}"
            )
            print(f"sampling: {attrs['point_cloud_sampling']}")
            print()

        if "provenance" in f:
            prov = f["provenance"]
            keep = np.asarray(prov["source_keep_mask"][:], dtype=bool)
            names = prov.attrs.get("drop_reason_bit_names_json", "")
            print(
                f"provenance: retained {keep.sum()}/{keep.size} source rows "
                f"(drop reasons: {names})"
            )
            print()

        if "joint_state" in f:
            q = f["joint_state"][:]
            print(
                f"joint_state (arm7+hand12) range: "
                f"[{np.array2string(q.min(axis=0), precision=3, suppress_small=True)}]"
            )
            print(
                f"                             [{np.array2string(q.max(axis=0), precision=3, suppress_small=True)}]"
            )
        if "action_ee" in f:
            ee = f["action_ee"][:, :3]
            print(
                f"action_ee pos range (m): "
                f"[{np.array2string(ee.min(axis=0), precision=3, suppress_small=True)}]"
            )
            print(
                f"                          [{np.array2string(ee.max(axis=0), precision=3, suppress_small=True)}]"
            )
        if "contact_force" in f:
            mag = np.linalg.norm(f["contact_force"][:], axis=2)
            print(
                f"contact_force mag range: [{np.array2string(mag.min(axis=0), precision=3, suppress_small=True)}]"
            )
        if "fingertip_points" in f:
            fp = f["fingertip_points"][:]
            print(
                f"fingertip_points range (m): "
                f"[{np.array2string(fp.min(axis=0).ravel(), precision=3, suppress_small=True)}]"
            )


class ProcessedEpisodeVisualizer:
    """Load a processed HDF5 v7 file and stream it into Rerun for interactive viewing."""

    def __init__(
        self,
        h5_path: str,
        max_frames: int | None = None,
        point_cloud: bool = True,
    ):
        self._h5_path = Path(h5_path)
        self._h5f = h5py.File(h5_path, "r")
        try:
            if (
                str(self._h5f.attrs.get("schema_name", "")) != PROCESSED_SCHEMA_NAME
                or int(self._h5f.attrs.get("schema_version", -1))
                != PROCESSED_SCHEMA_VERSION
            ):
                raise ValueError(
                    f"{self._h5_path.name} is not a processed HDF5 v7 artifact"
                )
            self._keys = _present_keys(self._h5f)
            if "joint_state" not in self._keys and "action" not in self._keys:
                raise ValueError(
                    f"{self._h5_path.name} has no joint_state/action datasets"
                )

            self._T = self._resolve_frame_count(max_frames)
            self._has_rgb = {"rgb", "depth"} <= self._keys
            self._pointcloud_num_points = _validate_pointcloud_dataset(self._h5f)
            self._has_pointcloud = self._pointcloud_num_points is not None

            # Preload the small scalar/tactile/fingertip modalities; the larger
            # rgb/depth/point_cloud arrays are sliced per frame in log_step.
            self._state = self._preload_state()
            self._timestamps = self._preload_timestamps()

            self._depth_meter = self._resolve_depth_meter()
            self._K, self._h, self._w = self._resolve_intrinsics()

            self._pc_enabled = point_cloud and self._has_pointcloud

            self._blueprint = self._build_blueprint()
            app_id = f"DexMani Processed - {self._h5_path.stem}"
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
        steps = int(self._h5f.attrs.get("episode_steps", -1))
        if steps <= 0:
            ref = next(iter(sorted(self._keys)), None)
            steps = int(self._h5f[ref].shape[0]) if ref is not None else 0
        if steps <= 0:
            raise ValueError(f"{self._h5_path.name} has no frames")
        if max_frames is not None:
            if max_frames <= 0:
                raise ValueError("max_frames must be a positive integer")
            return min(steps, max_frames)
        return steps

    def _preload_state(self) -> dict[str, np.ndarray]:
        """Read small non-camera, non-pointcloud datasets into memory, truncated to T."""
        state: dict[str, np.ndarray] = {}
        for key in ("joint_state", "action", "action_ee", "fingertip_points"):
            if key in self._keys:
                state[key] = np.asarray(self._h5f[key][: self._T])
        if "contact_force" in self._keys:
            contact = np.asarray(
                self._h5f["contact_force"][: self._T], dtype=np.float32
            )
            state["contact_force_mag"] = np.linalg.norm(contact, axis=2)  # (T, 5)
        return state

    def _preload_timestamps(self) -> np.ndarray | None:
        prov = self._h5f.get("provenance")
        if prov is None or "source_timestamp_s" not in prov:
            return None
        return np.asarray(prov["source_timestamp_s"][: self._T], dtype=np.float64)

    def _resolve_depth_meter(self) -> float | None:
        """Rerun ``DepthImage`` meter = raw units per meter = 1 / meters-per-unit."""
        if "depth" not in self._keys:
            return None
        scale = float(self._h5f.attrs.get("depth_scale_m_per_unit", 0.0))
        if not np.isfinite(scale) or scale <= 0.0:
            logger.warning("missing/invalid depth_scale_m_per_unit; depth not rendered")
            return None
        return 1.0 / scale

    def _resolve_intrinsics(self) -> tuple[np.ndarray | None, int, int]:
        if not self._has_rgb or "camera_intrinsic" not in self._keys:
            return None, 0, 0
        K = np.asarray(self._h5f["camera_intrinsic"][0], dtype=float).reshape(3, 3)
        rgb_shape = self._h5f["rgb"].shape
        h, w = int(rgb_shape[1]), int(rgb_shape[2])
        return K, h, w

    def _series_groups(self) -> list[tuple[str, str]]:
        """Time-series datasets as (group, key) in fixed view order."""
        groups: list[tuple[str, str]] = []
        if "joint_state" in self._state:
            groups.append(("state", "joint_state"))
        for key in ("action", "action_ee"):
            if key in self._state:
                groups.append(("action", key))
        if "contact_force_mag" in self._state:
            groups.append(("contact", "contact_force_mag"))
        return groups

    def _build_blueprint(self) -> rrb.Blueprint:
        columns: list[rrb.Container | rrb.View] = []

        cam_views = []
        if "rgb" in self._keys:
            cam_views.append(rrb.Spatial2DView(origin="camera/color/rgb", name="RGB"))
        if "depth" in self._keys and self._depth_meter is not None:
            # Processed depth is aligned to the color pinhole.
            cam_views.append(rrb.Spatial2DView(origin="depth/image", name="Depth"))
        if cam_views:
            columns.append(rrb.Vertical(contents=cam_views, name="Camera"))

        has_3d = self._pc_enabled or "fingertip_points" in self._state
        if has_3d:
            columns.append(
                rrb.Spatial3DView(
                    origin="/",
                    name="Point Cloud",
                    background=[0.12, 0.12, 0.14],
                )
            )

        group_to_views: dict[str, list[rrb.TimeSeriesView]] = {}
        for group, key in self._series_groups():
            group_to_views.setdefault(group, []).append(
                rrb.TimeSeriesView(origin=f"{group}/{key}", name=key)
            )
        group_order = ("state", "action", "contact")
        ts_verticals = [
            rrb.Vertical(contents=group_to_views[group], name=group.capitalize())
            for group in group_order
            if group in group_to_views
        ]
        if ts_verticals:
            if len(ts_verticals) == 1:
                columns.append(ts_verticals[0])
            else:
                columns.append(
                    rrb.Tabs(contents=ts_verticals, active_tab=0, name="Time Series")
                )

        return rrb.Blueprint(rrb.Horizontal(contents=columns))

    def _log_static(self) -> None:
        """Log per-series labels and the camera pinhole once."""
        for group, key in self._series_groups():
            arr = self._state[key]
            base = f"{group}/{key}"
            labels = _series_labels(key, arr.shape[1])
            for i, label in enumerate(labels):
                rr.log(f"{base}/{i}", rr.SeriesLine(name=label), static=True)

        if self._K is not None and self._h > 0 and self._w > 0:
            rr.log(
                "camera/color",
                rr.Pinhole(
                    image_from_camera=self._K,
                    resolution=[self._w, self._h],
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=1.25,
                ),
                static=True,
            )
            logger.info("Camera pinhole logged (%dx%d)", self._w, self._h)

    def log_step(self, step_idx: int) -> None:
        """Log camera, point cloud, fingertips, and time series for one timestep."""
        rr.set_time_sequence("step", step_idx)
        if self._timestamps is not None:
            rr.set_time_seconds("time", float(self._timestamps[step_idx]))
        self._log_camera(step_idx)
        self._log_pointcloud(step_idx)
        self._log_fingertips(step_idx)
        self._log_time_series(step_idx)

    def _log_camera(self, step_idx: int) -> None:
        if not self._has_rgb:
            return
        if "rgb" in self._keys:
            rr.log("camera/color/rgb", rr.Image(self._h5f["rgb"][step_idx]))
        if "depth" in self._keys and self._depth_meter is not None:
            rr.log(
                "depth/image",
                rr.DepthImage(
                    self._h5f["depth"][step_idx],
                    meter=self._depth_meter,
                    depth_range=(0, 10000),
                ),
            )
        if "camera_extrinsic" in self._keys:
            # T_xarm_base_from_color maps aligned color/depth optical coordinates
            # into xArm base.
            transform = np.asarray(self._h5f["camera_extrinsic"][step_idx], dtype=float)
            rr.log(
                "camera/color",
                rr.Transform3D(translation=transform[:3, 3], mat3x3=transform[:3, :3]),
            )

    def _log_pointcloud(self, step_idx: int) -> None:
        if not self._pc_enabled:
            return
        assert self._pointcloud_num_points is not None
        cloud = validate_point_cloud_array(
            np.asarray(self._h5f["point_cloud"][step_idx]),
            num_points=self._pointcloud_num_points,
            label=f"point_cloud[{step_idx}]",
        )
        if not np.any(np.linalg.norm(cloud[:, :3], axis=1) > 0.0):
            raise ValueError(f"point_cloud[{step_idx}] is an all-zero frame")
        colors = (cloud[:, 3:] * 255.0).astype(np.uint8)
        # Already in xArm base (world); no extrinsic transform is applied.
        rr.log("pcd", rr.Points3D(positions=cloud[:, :3], colors=colors, radii=0.003))

    def _log_fingertips(self, step_idx: int) -> None:
        fp_data = self._state.get("fingertip_points")
        if fp_data is None:
            return
        fp = np.asarray(fp_data[step_idx], dtype=np.float32)
        if fp.ndim != 2 or fp.shape != (5, 3) or not np.all(np.isfinite(fp)):
            return
        rr.log(
            "fingertips",
            rr.Points3D(
                positions=fp,
                colors=np.array(_FINGERTIP_COLORS, dtype=np.uint8),
                radii=0.012,
            ),
        )

    def _log_time_series(self, step_idx: int) -> None:
        for group, key in self._series_groups():
            arr = self._state[key]
            base = f"{group}/{key}"
            for i in range(arr.shape[1]):
                rr.log(f"{base}/{i}", rr.Scalar(float(arr[step_idx, i])))

    @property
    def num_steps(self) -> int:
        return self._T

    def close(self) -> None:
        """Release the HDF5 handle and disconnect from Rerun."""
        if hasattr(self, "_h5f") and self._h5f is not None:
            self._h5f.close()
            self._h5f = None  # type: ignore[assignment]
        try:
            rr.disconnect()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Visualize processed DexMani HDF5 v7 episodes with Rerun 3D."
    )
    parser.add_argument(
        "episode",
        type=str,
        help="Path to a processed .h5 file, e.g. episodes_processed/<task>/episode_<ts>.h5.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit number of frames to load.",
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
        help="Enable the 3D point cloud view (default: on).",
    )

    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be a positive integer")
    h5_path = Path(args.episode).expanduser().resolve()
    if not h5_path.is_file():
        logger.error("Processed file not found: %s", h5_path)
        return 1

    if args.info:
        print_episode_info(str(h5_path))
        return 0

    viz = ProcessedEpisodeVisualizer(
        str(h5_path),
        max_frames=args.max_frames,
        point_cloud=args.point_cloud,
    )
    try:
        logger.info("Logging %d frames to Rerun...", viz.num_steps)
        for step in range(viz.num_steps):
            viz.log_step(step)
            if step % 500 == 0:
                logger.info("  frame %d/%d", step, viz.num_steps)
        logger.info("Done. Close the Rerun window to exit.")
    finally:
        viz.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
