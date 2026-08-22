#!/usr/bin/env python3
"""Usage: ``python examples/visualize_episode_processed.py PROCESSED.h5 [--info] [--max-frames N]``.

Self-contained Rerun-based visualizer for processed HDF5 v3
(``dexmani-real-processed-hdf5``) artifacts written by
``examples/process_episodes.py``.  Offline only: connects to no hardware, writes
no files; opens a Rerun viewer window (or prints a structure summary with
``--info``).

Unlike ``examples/visualize_episode.py``, which reads a raw episode directory
(RGB in the MP4 sidecar, depth back-projected into a point cloud), this reads a
single ``.h5`` file whose RGB, depth, and point cloud are already stored
grid-aligned at ``(T, ...)``.  The point cloud is precomputed in the xArm base
frame, so nothing is back-projected here.

Examples::

  python examples/visualize_episode_processed.py episodes_processed/<task>/episode_<ts>.h5
  python examples/visualize_episode_processed.py episodes_processed/<task>/episode_<ts>.h5 --info
  python examples/visualize_episode_processed.py episodes_processed/<task>/episode_<ts>.h5 --max-frames 300
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Set a quiet Rerun logging default unless the operator already configured one.
os.environ.setdefault("RUST_LOG", "error")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

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

# Processed core modalities are fixed by the v3 contract: joint_state/action are
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
        print(f"task_name: {attrs.get('task_name', '?')}   "
              f"source_episode: {attrs.get('source_episode', '?')}")
        print(f"steps (T): {steps}   source_frames: {source_frames}   dt: {attrs.get('dt', '?')}")
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
    """Load a processed HDF5 v3 file and stream it into Rerun for interactive viewing."""

    def __init__(
        self,
        h5_path: str,
        max_frames: int | None = None,
        point_cloud: bool = True,
    ):
        self._h5_path = Path(h5_path)
        self._h5f = h5py.File(h5_path, "r")
        try:
            self._keys = _present_keys(self._h5f)
            if "joint_state" not in self._keys and "action" not in self._keys:
                raise ValueError(f"{self._h5_path.name} has no joint_state/action datasets")

            self._T = self._resolve_frame_count(max_frames)
            self._has_rgb = {"rgb", "depth"} <= self._keys
            self._has_pointcloud = "point_cloud" in self._keys

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
            contact = np.asarray(self._h5f["contact_force"][: self._T], dtype=np.float32)
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
            cam_views.append(rrb.Spatial2DView(origin="camera/rgb", name="RGB"))
        if "depth" in self._keys and self._depth_meter is not None:
            cam_views.append(rrb.Spatial2DView(origin="camera/depth", name="Depth"))
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
                "camera",
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
            rr.log("camera/rgb", rr.Image(self._h5f["rgb"][step_idx]))
        if "depth" in self._keys and self._depth_meter is not None:
            rr.log(
                "camera/depth",
                rr.DepthImage(
                    self._h5f["depth"][step_idx],
                    meter=self._depth_meter,
                    depth_range=(0, 10000),
                ),
            )
        if "camera_extrinsic" in self._keys:
            # T_xarm_base_camera: maps the camera-optical frame into xArm base,
            # which is the world frame of the processed data.  Logged per frame
            # because eye-in-hand extrinsics change with the arm; eye-to-hand is
            # simply re-logged unchanged.
            transform = np.asarray(
                self._h5f["camera_extrinsic"][step_idx], dtype=float
            )
            rr.log(
                "camera",
                rr.Transform3D(translation=transform[:3, 3], mat3x3=transform[:3, :3]),
            )

    def _log_pointcloud(self, step_idx: int) -> None:
        if not self._pc_enabled:
            return
        cloud = np.asarray(self._h5f["point_cloud"][step_idx], dtype=np.float32)
        valid = np.linalg.norm(cloud[:, :3], axis=1) > 1e-6
        if not valid.any():
            return
        positions = cloud[valid, :3]
        colors = (np.clip(cloud[valid, 3:], 0.0, 1.0) * 255.0).astype(np.uint8)
        # Already in xArm base (world); no extrinsic transform is applied.
        rr.log("pcd", rr.Points3D(positions=positions, colors=colors, radii=0.003))

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
        description="Visualize processed DexMani HDF5 v3 episodes with Rerun 3D."
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
