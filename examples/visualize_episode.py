#!/usr/bin/env python3
"""Usage: ``python examples/visualize_episode.py EPISODE [--info] [--max-frames N]``.

Self-contained Rerun visualizer for raw schema-v24 DexMani episodes. Offline
only: connects to no hardware and writes no files. Episodes display a canonical
fixed-size ``(N, 6)`` point cloud derived with the same production implementation
used by offline processing and deployment.

Examples::

  python examples/visualize_episode.py episodes/<task_name>/<episode_dir>
  python examples/visualize_episode.py episodes/<task_name>/<episode_dir> --info
  python examples/visualize_episode.py episodes/<task_name>/<episode_dir> --max-frames 500
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
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
from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.data.raw_pointcloud import (
    RawEpisodeCameraModel,
    RawEpisodePointCloudDeriver,
    load_raw_episode_base_from_color,
    load_raw_episode_camera_model,
)
from dexmani_real.ipc.schema import (
    SUPPORTED_POINT_CLOUD_COUNTS,
    validate_point_cloud_array,
)
from dexmani_real.recording import EpisodeReader, MergedH5File
from dexmani_real.robot_spec import HAND_FINGERTIP_SHAPE
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


_KNOWN_CATEGORIES: dict[str, set[str]] = {
    "arm": {"arm_qpos", "arm_ee"},
    "hand": {"hand_qpos", "hand_fingertip", "hand_contact"},
    "action": {"action_arm_joint", "action_arm_ee", "action_hand_joint"},
    "vr": {"vr_wrist_pos", "vr_wrist_rot6d", "vr_landmarks"},
    "camera": {"rgb", "depth"},
    "flags": {
        "flag_ik_ok",
        "flag_retarget_ok",
        "flag_held",
        "flag_camera_fresh",
        "camera_age_s",
        "camera_frame_number",
    },
    "meta": {"timestamp"},
}

_FINGERTIP_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 60, 60),  # thumb  — red
    (60, 255, 60),  # index  — green
    (60, 120, 255),  # middle — blue
    (255, 200, 40),  # ring   — gold
    (220, 60, 255),  # pinky  — magenta
)


def _classify_datasets(h5f: MergedH5File) -> dict[str, list[str]]:
    """Group top-level HDF5 datasets by category (arm, hand, action, vr, camera, flags, meta)."""
    available_keys = {k for k in h5f.keys() if isinstance(h5f[k], h5py.Dataset)}
    classified: dict[str, list[str]] = {}

    for category, known_keys in _KNOWN_CATEGORIES.items():
        found = sorted(known_keys & available_keys)
        if found:
            classified[category] = found

    return classified


def print_episode_info(h5_path: str) -> None:
    """Print a human-readable summary of the episode structure without opening Rerun."""
    with EpisodeReader(h5_path) as reader:
        if not reader.meets_min_duration:
            print(
                "WARNING: episode is below the configured minimum recording duration (quality label only)"
            )
        f = reader.h5f
        keys = sorted(k for k in f.keys() if isinstance(f[k], h5py.Dataset))
        groups = sorted(k for k in f.keys() if not isinstance(f[k], h5py.Dataset))

        t_key = next((k for k in keys if k == "arm_qpos"), None)
        t_key = t_key or next((k for k in keys if k == "timestamp"), keys[0])
        t_frames = f[t_key].shape[0]
        c_key = "rgb" if "rgb" in f else ("depth" if "depth" in f else None)
        if c_key:
            c_frames = f[c_key].shape[0]
        else:
            c_frames = None

        print(f"Episode:    {h5_path}")
        print(f"State frames (T): {t_frames}")
        if c_frames is not None:
            print(f"Camera frames (C): {c_frames}  (ratio={c_frames / t_frames:.2f})")
        print()

        if "meta" in groups:
            print("Meta:")
            for attr in sorted(f["meta"].attrs.keys()):
                print(f"  {attr}: {f['meta'].attrs[attr]}")
            print()

        print(f"Datasets ({len(keys)}):")
        for key in sorted(keys):
            ds = f[key]
            print(f"  {key:<28s} shape={str(ds.shape):<22s} dtype={str(ds.dtype):<10s}")

        print()
        if "arm_qpos" in f:
            q = f["arm_qpos"][:]
            print(
                f"arm_qpos  range: [{np.array2string(q.min(axis=0), precision=3, suppress_small=True)}]"
            )
            print(
                f"                    [{np.array2string(q.max(axis=0), precision=3, suppress_small=True)}]"
            )
        if "arm_ee" in f:
            ee = f["arm_ee"][:]
            print(
                f"arm_ee    pos range (m):  [{np.array2string(ee[:, :3].min(axis=0), precision=3, suppress_small=True)}]"
            )
            print(
                f"                    [{np.array2string(ee[:, :3].max(axis=0), precision=3, suppress_small=True)}]"
            )
        if "hand_qpos" in f:
            hq = f["hand_qpos"][:]
            print(
                f"hand_qpos range: [{np.array2string(hq.min(axis=0), precision=3, suppress_small=True)}]"
            )
            print(
                f"                    [{np.array2string(hq.max(axis=0), precision=3, suppress_small=True)}]"
            )
        if "flag_ik_ok" in f:
            ik = f["flag_ik_ok"][:]
            print(f"flag_ik_ok success rate: {ik.mean():.2%}")
        if "flag_held" in f:
            held = f["flag_held"][:]
            print(f"flag_held  engaged rate: {held.mean():.2%}")
        if "flag_camera_fresh" in f:
            fresh = f["flag_camera_fresh"][:]
            print(f"flag_camera_fresh rate: {fresh.mean():.2%}")


class EpisodeVisualizer:
    """Load an HDF5 episode and stream it into Rerun for interactive 3D viewing."""

    def __init__(
        self,
        h5_path: str,
        max_frames: int | None = None,
        point_cloud: bool = True,
        pointcloud_config: PointCloudConfig | None = None,
        table_plane_abcd: tuple[float, float, float, float] | None = None,
        runtime_config_sha256: str | None = None,
    ):
        self._h5_path = Path(h5_path)
        self._reader = EpisodeReader(h5_path)
        try:
            if not self._reader.meets_min_duration:
                logger.warning(
                    "Episode is below the configured minimum recording duration"
                )
            self._h5f = self._reader.h5f

            # Preload the RGB-D sidecars once for Rerun.
            self._rgb_cache = self._reader.read_camera_all("rgb")
            self._depth_cache = self._reader.read_camera_all("depth")
            logger.info("Pre-decoded %d RGB-D frames", self._rgb_cache.shape[0])

            self._available = _classify_datasets(self._h5f)
            # _classify_datasets only scans HDF5 keys; RGB lives in the MP4 sidecar.
            if "rgb" not in self._available.get("camera", []):
                self._available.setdefault("camera", []).append("rgb")
            logger.info(
                "Detected %d categories: %s",
                len(self._available),
                sorted(self._available.keys()),
            )

            meta = self._h5f.get("meta")
            if meta is None:
                raise ValueError("episode is missing /meta")
            self._camera_model = load_raw_episode_camera_model(self._reader)
            self._camera_K = self._camera_model.geometry.color.matrix()
            self._depth_meter = 1.0 / self._camera_model.depth_scale_m
            self._T_xarm_base_from_color: np.ndarray | None = None
            self._pc_enabled = point_cloud
            camera_type = str(meta.attrs.get("camera_type", ""))
            if camera_type == "eye_to_hand" or self._pc_enabled:
                self._T_xarm_base_from_color = load_raw_episode_base_from_color(
                    self._reader
                )

            self._pointcloud_deriver: RawEpisodePointCloudDeriver | None = None
            self._empty_pointcloud_frames = 0
            self._pointcloud_processing_ns = 0
            self._pointcloud_processed_frames = 0
            if self._pc_enabled:
                if pointcloud_config is None:
                    raise ValueError(
                        "point-cloud visualization requires resolved config"
                    )
                assert self._T_xarm_base_from_color is not None
                self._pointcloud_deriver = RawEpisodePointCloudDeriver(
                    reader=self._reader,
                    camera=self._camera_model,
                    T_xarm_base_from_color=self._T_xarm_base_from_color,
                    pointcloud=pointcloud_config,
                    table_plane_abcd=table_plane_abcd,
                )
                source_config_sha256 = str(meta.attrs.get("resolved_config_sha256", ""))
                if (
                    runtime_config_sha256 is not None
                    and source_config_sha256 != runtime_config_sha256
                ):
                    logger.warning(
                        "Point cloud uses current runtime config %s, which differs "
                        "from recorded config %s",
                        runtime_config_sha256,
                        source_config_sha256 or "missing",
                    )
                logger.info(
                    "Canonical point cloud enabled: N=%d config=%s table=%s",
                    pointcloud_config.num_points,
                    pointcloud_config.sha256,
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
        """Resolve the fixed-grid frame count and validate camera alignment."""
        raw = int(self._h5f["meta"].attrs.get("num_frames", 0))
        if raw <= 0:
            raise ValueError("episode /meta num_frames must be positive")
        camera_counts = [self._rgb_cache.shape[0], self._depth_cache.shape[0]]
        if any(count != raw for count in camera_counts):
            raise ValueError(
                f"camera frame counts {camera_counts} do not match grid length {raw}"
            )
        if max_frames is not None:
            return min(raw, max_frames)
        return raw

    def _preload_state(self) -> dict[str, np.ndarray]:
        """Read all non-camera datasets into memory, truncated to T frames."""
        state: dict[str, np.ndarray] = {}
        for _category, keys in self._available.items():
            if _category == "camera":
                continue
            for key in keys:
                data = self._h5f[key][: self._T]
                if data.ndim == 0:
                    data = data[()]
                state[key] = np.asarray(data)

        if "hand_contact" in state:
            contact = state["hand_contact"]
            state["hand_contact_mag"] = np.linalg.norm(contact, axis=2)  # (T, 5)
            state["hand_force_thumb"] = contact[:, 0, :].copy()  # (T, 3)
            state["hand_force_index"] = contact[:, 1, :].copy()  # (T, 3)

        return state

    def _build_blueprint(self) -> rrb.Blueprint:
        """Build Rerun view layout from detected data categories."""
        has_state = bool(self._available.get("arm") or self._available.get("hand"))
        has_action = bool(self._available.get("action"))
        has_flags = bool(self._available.get("flags"))

        columns: list[rrb.Container | rrb.View] = []

        cam_views = []
        if "rgb" in (self._available.get("camera") or []):
            cam_views.append(rrb.Spatial2DView(origin="camera/color/rgb", name="RGB"))
        if "depth" in (self._available.get("camera") or []):
            cam_views.append(rrb.Spatial2DView(origin="depth/image", name="Depth"))
        if cam_views:
            columns.append(rrb.Vertical(contents=cam_views, name="Camera"))

        if self._pc_enabled or "hand_fingertip" in self._state:
            columns.append(
                rrb.Spatial3DView(
                    origin="/",
                    name="Point Cloud",
                    background=[0.12, 0.12, 0.14],
                )
            )

        ts_verticals = []

        if has_state:
            state_views = []
            for key in self._available.get("arm", []) + self._available.get("hand", []):
                if key in self._state and 1 <= self._state[key].ndim <= 2:
                    state_views.append(
                        rrb.TimeSeriesView(origin=f"state/{key}", name=key)
                    )
            for fkey in ("hand_contact_mag", "hand_force_thumb", "hand_force_index"):
                if fkey in self._state:
                    state_views.append(
                        rrb.TimeSeriesView(origin=f"state/{fkey}", name=fkey)
                    )
            if state_views:
                ts_verticals.append(rrb.Vertical(contents=state_views, name="State"))

        if has_action:
            action_views = []
            for key in self._available.get("action", []):
                if key in self._state and 1 <= self._state[key].ndim <= 2:
                    action_views.append(
                        rrb.TimeSeriesView(origin=f"action/{key}", name=key)
                    )
            if action_views:
                ts_verticals.append(rrb.Vertical(contents=action_views, name="Action"))

        if has_flags:
            flag_views = [
                rrb.TimeSeriesView(origin=f"flags/{key}", name=key)
                for key in self._available.get("flags", [])
            ]
            ts_verticals.append(rrb.Vertical(contents=flag_views, name="Flags"))

        if ts_verticals:
            if len(ts_verticals) == 1:
                columns.append(ts_verticals[0])
            else:
                columns.append(
                    rrb.Tabs(contents=ts_verticals, active_tab=0, name="Time Series")
                )

        return rrb.Blueprint(rrb.Horizontal(contents=columns))

    def _log_static(self) -> None:
        """Log per-series labels, camera pinhole, and extrinsics once."""
        for category, keys in self._available.items():
            if category == "camera":
                continue
            for key in keys:
                if key not in self._state:
                    continue
                arr = self._state[key]
                if arr.ndim > 2:
                    continue  # skip 3D+ arrays
                base = self._series_origin(category, key)
                if arr.ndim <= 1:
                    rr.log(base, rr.SeriesLine(name=key), static=True)
                else:
                    for i in range(arr.shape[1]):
                        rr.log(f"{base}/{i}", rr.SeriesLine(name=f"{i}"), static=True)

        if self._camera_K is not None:
            rgb_shape = self._rgb_cache.shape
            h, w = rgb_shape[1], rgb_shape[2]
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

        if self._T_xarm_base_from_color is not None:
            rr.log(
                "camera/color",
                rr.Transform3D(
                    translation=self._T_xarm_base_from_color[:3, 3],
                    mat3x3=self._T_xarm_base_from_color[:3, :3],
                ),
                static=True,
            )

        _force_series = {
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
        for fkey, labels in _force_series.items():
            if fkey in self._state:
                base = f"state/{fkey}"
                for i, label in enumerate(labels):
                    rr.log(f"{base}/{i}", rr.SeriesLine(name=label), static=True)

    @staticmethod
    def _series_origin(category: str, key: str) -> str:
        """Entity path: arm/hand → state/<key>, others → <category>/<key>."""
        if category in ("arm", "hand"):
            return f"state/{key}"
        return f"{category}/{key}"

    def log_step(self, step_idx: int) -> None:
        """Log camera, canonical point cloud, fingertips, and state."""
        rr.set_time_sequence("step", step_idx)
        if "timestamp" in self._state:
            rr.set_time_seconds("time", float(self._state["timestamp"][step_idx]))
        self._log_camera(step_idx)
        self._log_pointcloud(step_idx)
        self._log_fingertips(step_idx)
        self._log_time_series(step_idx)

    def _log_camera(self, step_idx: int) -> None:
        camera_keys = self._available.get("camera", [])
        if not camera_keys:
            return

        if "rgb" in camera_keys:
            rr.log("camera/color/rgb", rr.Image(self._rgb_cache[step_idx]))
        if "depth" in camera_keys:
            rr.log(
                "depth/image",
                rr.DepthImage(
                    self._depth_cache[step_idx],
                    meter=self._depth_meter,
                    depth_range=(0, 10000),
                ),
            )  # clamp outliers to stabilize colormap

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
        """Render hand_fingertip FK positions as per-finger colored keypoints."""
        fp_data = self._state.get("hand_fingertip")
        if fp_data is None:
            return
        fp = np.asarray(fp_data[step_idx], dtype=np.float32)
        if fp.ndim != 2 or fp.shape != HAND_FINGERTIP_SHAPE:
            return
        if not np.all(np.isfinite(fp)):
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
        for category, keys in self._available.items():
            if category == "camera":
                continue
            for key in keys:
                if key not in self._state:
                    continue
                arr = self._state[key]
                if arr.ndim > 2:
                    continue  # skip 3D+ arrays
                base = self._series_origin(category, key)
                if arr.ndim <= 1:
                    rr.log(base, rr.Scalar(float(arr[step_idx])))
                else:
                    for i in range(arr.shape[1]):
                        rr.log(f"{base}/{i}", rr.Scalar(float(arr[step_idx, i])))

        for fkey in ("hand_contact_mag", "hand_force_thumb", "hand_force_index"):
            if fkey in self._state:
                arr = self._state[fkey]
                for i in range(arr.shape[1]):
                    rr.log(f"state/{fkey}/{i}", rr.Scalar(float(arr[step_idx, i])))

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
        """Release HDF5/video resources and disconnect from Rerun."""
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
        help="Path to a raw schema-v24 episodes/<task_name>/episode_* directory.",
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
        default=None,
        help=("Display the canonical point cloud (default: enabled)."),
    )
    parser.add_argument(
        "--pointcloud-num-points",
        type=int,
        choices=sorted(SUPPORTED_POINT_CLOUD_COUNTS),
        default=None,
        help="Override the current runtime point-cloud count.",
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

    point_cloud_enabled = True if args.point_cloud is None else args.point_cloud
    if args.pointcloud_num_points is not None and not point_cloud_enabled:
        parser.error("--pointcloud-num-points requires an enabled point cloud")

    runtime = resolve_runtime_config() if point_cloud_enabled else None
    pointcloud_config = None
    table_plane_abcd = None
    runtime_config_sha256 = None
    if runtime is not None:
        pointcloud_config = runtime.pointcloud
        if args.pointcloud_num_points is not None:
            pointcloud_config = replace(
                pointcloud_config, num_points=args.pointcloud_num_points
            )
        table = runtime.environment.table
        table_plane_abcd = table.plane_abcd if table.enabled else None
        runtime_config_sha256 = runtime.sha256

    viz = EpisodeVisualizer(
        str(h5_path),
        max_frames=args.max_frames,
        point_cloud=point_cloud_enabled,
        pointcloud_config=pointcloud_config,
        table_plane_abcd=table_plane_abcd,
        runtime_config_sha256=runtime_config_sha256,
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
