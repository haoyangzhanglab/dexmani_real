#!/usr/bin/env python3
"""Usage: ``python examples/visualize_episode.py EPISODE [--info] [--max-frames N]``.

Self-contained Rerun-based visualizer for supported schema-v17/v18/v19
DexMani episodes. Offline only: connects to no hardware, writes no files;
opens a Rerun viewer window (or prints a structure summary with ``--info``).

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
from pathlib import Path

# Set a quiet Rerun logging default unless the operator already configured one.
os.environ.setdefault("RUST_LOG", "error")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from dexmani_real.recording.episode_reader import EpisodeReader, MergedH5File
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.pointcloud_utils import depth_to_meters
from dexmani_real.utils.schema import HAND_FINGERTIP_SHAPE

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
        depth_scale: float | None = None,
        point_cloud: bool = True,
        pc_stride: int = 4,
        pc_min_depth: float = 0.1,
        pc_max_depth: float = 2.0,
    ):
        self._h5_path = Path(h5_path)
        self._reader = EpisodeReader(h5_path)
        try:
            if not self._reader.meets_min_duration:
                logger.warning(
                    "Episode is below the configured minimum recording duration"
                )
            self._h5f = self._reader.h5f

            self._rgb_cache: np.ndarray | None = None
            self._depth_cache: np.ndarray | None = None
            # RGB lives in the MP4 sidecar; query the reader rather than HDF5.
            try:
                self._rgb_cache = self._reader.read_camera_all("rgb")
                logger.info("Pre-decoded %d rgb frames", self._rgb_cache.shape[0])
            except KeyError:
                pass
            if "depth" in self._h5f:
                self._depth_cache = self._reader.read_camera_all("depth")
                logger.info("Pre-decoded %d depth frames", self._depth_cache.shape[0])

            self._available = _classify_datasets(self._h5f)
            # _classify_datasets only scans HDF5 keys — inject MP4 RGB when present.
            if self._rgb_cache is not None and "rgb" not in self._available.get(
                "camera", []
            ):
                self._available.setdefault("camera", []).append("rgb")
            logger.info(
                "Detected %d categories: %s",
                len(self._available),
                sorted(self._available.keys()),
            )

            # Depth units: CLI > the required /meta depth_scale.
            if depth_scale is None:
                meta = self._h5f.get("meta")
                if meta is not None and "depth_scale" in meta.attrs:
                    depth_scale = float(meta.attrs["depth_scale"])
                elif "depth" in self._h5f:
                    raise ValueError("episode is missing /meta depth_scale")
            self._depth_meter = 1.0 / (depth_scale if depth_scale else 0.001)
            self._depth_scale = (
                depth_scale if depth_scale else 0.001
            )  # meters per raw unit

            # Camera extrinsics: camera_T_world_camera (4x4 row-major) maps camera → world.
            self._cam_R: np.ndarray | None = None
            self._cam_t: np.ndarray | None = None
            meta = self._h5f.get("meta")
            if meta is not None and "camera_T_world_camera" in meta.attrs:
                T_cw = np.asarray(
                    meta.attrs["camera_T_world_camera"], dtype=float
                ).reshape(4, 4)
                self._cam_R = T_cw[:3, :3].copy()
                self._cam_t = T_cw[:3, 3].copy()
                logger.info(
                    "Camera extrinsics loaded: t=[%.3f, %.3f, %.3f]", *self._cam_t
                )
            else:
                logger.info(
                    "No camera extrinsics in /meta — camera frame = world frame (identity)"
                )

            self._pc_enabled = point_cloud and "depth" in (
                self._available.get("camera") or []
            )
            self._pc_stride = max(1, pc_stride)
            self._pc_min_depth = pc_min_depth
            self._pc_max_depth = pc_max_depth
            self._pc_cache: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}

            self._pc_K: np.ndarray | None = None
            self._pc_rays: tuple[np.ndarray, np.ndarray] | None = (
                None  # (u_strided, v_strided)
            )
            if self._pc_enabled:
                meta = self._h5f.get("meta")
                if meta is not None and "camera_K" in meta.attrs:
                    self._pc_K = np.asarray(
                        meta.attrs["camera_K"], dtype=float
                    ).reshape(3, 3)
                    depth_shape = (
                        self._depth_cache.shape
                        if self._depth_cache is not None
                        else self._h5f["depth"].shape
                    )
                    h, w = depth_shape[1], depth_shape[2]
                    self._pc_h, self._pc_w = h, w
                    v, u = np.mgrid[0 : h : self._pc_stride, 0 : w : self._pc_stride]
                    self._pc_rays = (u.astype(np.float32), v.astype(np.float32))
                    logger.info(
                        "Point cloud enabled: stride=%d → ~%d points/frame, depth=[%.2f, %.2f]m",
                        self._pc_stride,
                        u.size,
                        self._pc_min_depth,
                        self._pc_max_depth,
                    )
                else:
                    logger.warning("Point cloud disabled: no camera_K in /meta")
                    self._pc_enabled = False

            if self._pc_enabled:
                logger.info(
                    "Point cloud derived from depth back-projection + camera_K."
                )

            self._T = self._resolve_frame_count(max_frames)
            self._C = self._resolve_camera_count()
            logger.info("State frames=%d, Camera frames=%d", self._T, self._C or 0)

            self._state = self._preload_state()

            self._cam_idx: np.ndarray | None = None
            if self._C is not None and self._C > 0:
                if self._C < self._T:
                    self._cam_idx = np.minimum(
                        (np.arange(self._T) * self._C / self._T).astype(int),
                        self._C - 1,
                    )
                else:
                    self._cam_idx = np.arange(self._T, dtype=int)

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
        """Return T = min(arm_qpos.shape[0], max_frames) falling back through meta keys."""
        arm_keys = self._available.get("arm", [])
        meta_keys = self._available.get("meta", [])
        ref_key = next(iter(arm_keys), None) or next(iter(meta_keys), None)
        if ref_key is None:
            all_keys = [k for v in self._available.values() for k in v]
            ref_key = all_keys[0] if all_keys else None
        if ref_key is None:
            self._h5f.close()
            raise ValueError("No datasets found in HDF5 file")

        raw = self._h5f[ref_key].shape[0]
        if max_frames is not None:
            return min(raw, max_frames)
        return raw

    def _resolve_camera_count(self) -> int | None:
        """Return C from pre-decoded cache or first camera dataset in HDF5."""
        if self._rgb_cache is not None:
            return self._rgb_cache.shape[0]
        if self._depth_cache is not None:
            return self._depth_cache.shape[0]
        for cam_key in self._available.get("camera", []):
            return self._h5f[cam_key].shape[0]
        return None

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

    @staticmethod
    def _depth_to_pointcloud(
        depth: np.ndarray,
        K: np.ndarray,
        u_strided: np.ndarray,
        v_strided: np.ndarray,
        rgb: np.ndarray | None,
        depth_scale: float,
        stride: int,
        min_depth: float,
        max_depth: float,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Back-project a strided depth frame to camera-frame points (N,3) + optional colors (N,3)."""
        depth_m = depth_to_meters(depth, depth_scale=depth_scale)
        depth_strided = depth_m[::stride, ::stride]
        z = depth_strided.astype(np.float32)

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        x = (u_strided - cx) * z / fx
        y = (v_strided - cy) * z / fy

        valid = (z > min_depth) & (z < max_depth) & np.isfinite(z)
        if not valid.any():
            return np.zeros((0, 3), dtype=np.float32), None

        points = np.stack([x[valid], y[valid], z[valid]], axis=-1).astype(np.float32)

        colors = None
        if rgb is not None:
            colors = rgb[::stride, ::stride][valid]

        return points, colors

    def _build_blueprint(self) -> rrb.Blueprint:
        """Build Rerun view layout from detected data categories."""
        has_state = bool(self._available.get("arm") or self._available.get("hand"))
        has_action = bool(self._available.get("action"))
        has_flags = bool(self._available.get("flags"))

        columns: list[rrb.Container | rrb.View] = []

        cam_views = []
        if "rgb" in (self._available.get("camera") or []):
            cam_views.append(rrb.Spatial2DView(origin="camera/rgb", name="RGB"))
        if "depth" in (self._available.get("camera") or []):
            cam_views.append(rrb.Spatial2DView(origin="camera/depth", name="Depth"))
        if cam_views:
            columns.append(rrb.Vertical(contents=cam_views, name="Camera"))

        _has_3d = self._pc_enabled or "hand_fingertip" in self._state
        if _has_3d:
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

        meta = self._h5f.get("meta")
        has_rgb = "rgb" in self._h5f or self._rgb_cache is not None
        if meta is not None and "camera_K" in meta.attrs and has_rgb:
            K = np.asarray(meta.attrs["camera_K"], dtype=float).reshape(3, 3)
            rgb_shape = (
                self._rgb_cache.shape
                if self._rgb_cache is not None
                else self._h5f["rgb"].shape
            )
            h, w = rgb_shape[1], rgb_shape[2]
            rr.log(
                "camera",
                rr.Pinhole(
                    image_from_camera=K,
                    resolution=[w, h],
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=1.25,
                ),
                static=True,
            )
            logger.info("Camera pinhole logged (%dx%d)", w, h)

        if self._cam_R is not None and self._cam_t is not None:
            rr.log(
                "camera",
                rr.Transform3D(translation=self._cam_t, mat3x3=self._cam_R),
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
        """Log camera, fingertips, and time series for one timestep."""
        rr.set_time_sequence("step", step_idx)
        if "timestamp" in self._state:
            rr.set_time_seconds("time", float(self._state["timestamp"][step_idx]))
        self._log_camera(step_idx)
        self._log_fingertips(step_idx)
        self._log_time_series(step_idx)

    def _log_camera(self, step_idx: int) -> None:
        camera_keys = self._available.get("camera", [])
        if not camera_keys:
            return

        cam_idx = (
            int(self._cam_idx[step_idx]) if self._cam_idx is not None else step_idx
        )

        if "rgb" in camera_keys:
            rgb = (
                self._rgb_cache[cam_idx]
                if self._rgb_cache is not None
                else self._h5f["rgb"][cam_idx]
            )
            rr.log("camera/rgb", rr.Image(rgb))
        if "depth" in camera_keys:
            depth = (
                self._depth_cache[cam_idx]
                if self._depth_cache is not None
                else self._h5f["depth"][cam_idx]
            )
            rr.log(
                "camera/depth",
                rr.DepthImage(depth, meter=self._depth_meter, depth_range=(0, 10000)),
            )  # clamp outliers to stabilize colormap

        if self._pc_enabled and self._pc_K is not None and self._pc_rays is not None:
            if cam_idx not in self._pc_cache:
                depth = (
                    self._depth_cache[cam_idx]
                    if self._depth_cache is not None
                    else self._h5f["depth"][cam_idx]
                )
                rgb = (
                    self._rgb_cache[cam_idx]
                    if self._rgb_cache is not None
                    else (self._h5f["rgb"][cam_idx] if "rgb" in camera_keys else None)
                )
                u_strided, v_strided = self._pc_rays
                self._pc_cache[cam_idx] = self._depth_to_pointcloud(
                    depth=depth,
                    K=self._pc_K,
                    u_strided=u_strided,
                    v_strided=v_strided,
                    rgb=rgb,
                    depth_scale=self._depth_scale,
                    stride=self._pc_stride,
                    min_depth=self._pc_min_depth,
                    max_depth=self._pc_max_depth,
                )
            points, colors = self._pc_cache[cam_idx]
            if points.shape[0] > 0:
                if self._cam_R is not None:
                    points = points @ self._cam_R.T + self._cam_t
                rr.log("pcd", rr.Points3D(positions=points, colors=colors, radii=0.003))

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
        help="Path to a supported schema-v17/v18/v19 episodes/<task_name>/episode_* directory.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit number of state frames to load.",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=None,
        help="Raw depth units in meters (overrides /meta depth_scale).",
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
        help="Enable 3D point cloud from depth.",
    )
    parser.add_argument(
        "--pc-stride",
        type=int,
        default=4,
        help="Pixel stride for point cloud downsampling (default: 4).",
    )
    parser.add_argument(
        "--pc-min-depth",
        type=float,
        default=0.1,
        help="Min depth for point cloud filtering in meters (default: 0.1).",
    )
    parser.add_argument(
        "--pc-max-depth",
        type=float,
        default=2.0,
        help="Max depth for point cloud filtering in meters (default: 2.0).",
    )

    args = parser.parse_args(argv)
    h5_path = Path(args.episode).expanduser().resolve()
    if not h5_path.exists():
        logger.error("Episode not found: %s", h5_path)
        return 1

    if args.info:
        print_episode_info(str(h5_path))
        return 0

    viz = EpisodeVisualizer(
        str(h5_path),
        max_frames=args.max_frames,
        depth_scale=args.depth_scale,
        point_cloud=args.point_cloud,
        pc_stride=args.pc_stride,
        pc_min_depth=args.pc_min_depth,
        pc_max_depth=args.pc_max_depth,
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
