#!/usr/bin/env python3
"""Rerun-based HDF5 episode visualizer for DexMani teleop data.

Auto-detects available datasets in flat-key HDF5 episodes:
  State:   arm_qpos(7), arm_ee(9), arm_qvel(7), arm_tau(7),
           hand_qpos(12), hand_fingertip(5,3), hand_contact(5,3)
  Action:  action_arm_joint(7), action_arm_ee(9), action_hand_joint(12)
  VR:      vr_wrist_pos(3), vr_wrist_rot6d(6), vr_landmarks(21,3)
  Flags:   flag_ik_ok, flag_retarget_ok, flag_held
  Camera:  rgb(C,H,W,3), depth(C,H,W)  -- C may be < T (forward-filled)

Usage:
  python -m dexmani_real.tools.visualize_episode episode.h5
  python -m dexmani_real.tools.visualize_episode episode.h5 --info
  python -m dexmani_real.tools.visualize_episode episode.h5 --max-frames 500
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Known dataset categories for auto-detection
# ---------------------------------------------------------------------------

_KNOWN_CATEGORIES: dict[str, set[str]] = {
    "arm": {"arm_qpos", "arm_ee"},
    "hand": {"hand_qpos", "hand_fingertip", "hand_contact"},
    "action": {"action_arm_joint", "action_arm_ee", "action_hand_joint"},
    "vr": {"vr_wrist_pos", "vr_wrist_rot6d", "vr_landmarks"},
    "camera": {"rgb", "depth"},
    "flags": {"flag_ik_ok", "flag_retarget_ok", "flag_held"},
    "meta": {"timestamp"},
}


def _classify_datasets(h5f: h5py.File) -> dict[str, list[str]]:
    """Scan top-level HDF5 datasets and group them by category.

    Returns:
        Dict mapping category name -> list of available keys.
        Categories with no matches are omitted.
    """
    available_keys = {k for k in h5f.keys() if isinstance(h5f[k], h5py.Dataset)}
    classified: dict[str, list[str]] = {}

    for category, known_keys in _KNOWN_CATEGORIES.items():
        found = sorted(known_keys & available_keys)
        if found:
            classified[category] = found

    return classified


# ---------------------------------------------------------------------------
# Info mode (no Rerun needed)
# ---------------------------------------------------------------------------


def print_episode_info(h5_path: str) -> None:
    """Print a human-readable summary of the episode structure without opening Rerun."""
    with h5py.File(h5_path, "r") as f:
        keys = sorted(k for k in f.keys() if isinstance(f[k], h5py.Dataset))
        groups = sorted(k for k in f.keys() if not isinstance(f[k], h5py.Dataset))

        # Determine frame counts
        t_key = next((k for k in keys if k == "arm_qpos"), None)
        t_key = t_key or next((k for k in keys if k == "timestamp"), keys[0])
        t_frames = f[t_key].shape[0]
        c_key = "rgb" if "rgb" in f else ("depth" if "depth" in f else None)
        c_frames = f[c_key].shape[0] if c_key else None

        print(f"File:       {h5_path}")
        print(f"State frames (T): {t_frames}")
        if c_frames is not None:
            print(f"Camera frames (C): {c_frames}  (ratio={c_frames / t_frames:.2f})")
        print()

        # Metadata
        if "meta" in groups:
            print("Meta:")
            for attr in sorted(f["meta"].attrs.keys()):
                print(f"  {attr}: {f['meta'].attrs[attr]}")
            print()

        # Datasets
        print(f"Datasets ({len(keys)}):")
        for key in sorted(keys):
            ds = f[key]
            print(f"  {key:<28s} shape={str(ds.shape):<22s} dtype={str(ds.dtype):<10s}")

        # Quick stats
        print()
        if "arm_qpos" in f:
            q = f["arm_qpos"][:]
            print(f"arm_qpos  range: [{np.array2string(q.min(axis=0), precision=3, suppress_small=True)}]")
            print(f"                    [{np.array2string(q.max(axis=0), precision=3, suppress_small=True)}]")
        if "arm_ee" in f:
            ee = f["arm_ee"][:]
            print(f"arm_ee    pos range (m):  [{np.array2string(ee[:, :3].min(axis=0), precision=3, suppress_small=True)}]")
            print(f"                    [{np.array2string(ee[:, :3].max(axis=0), precision=3, suppress_small=True)}]")
        if "hand_qpos" in f:
            hq = f["hand_qpos"][:]
            print(f"hand_qpos range: [{np.array2string(hq.min(axis=0), precision=3, suppress_small=True)}]")
            print(f"                    [{np.array2string(hq.max(axis=0), precision=3, suppress_small=True)}]")
        if "flag_ik_ok" in f:
            ik = f["flag_ik_ok"][:]
            print(f"flag_ik_ok success rate: {ik.mean():.2%}")
        if "flag_held" in f:
            held = f["flag_held"][:]
            print(f"flag_held  engaged rate: {held.mean():.2%}")


# ---------------------------------------------------------------------------
# Main visualizer
# ---------------------------------------------------------------------------


class EpisodeVisualizer:
    """Load and stream a DexMani HDF5 episode into Rerun for interactive viewing."""

    def __init__(self, h5_path: str, max_frames: int | None = None):
        self._h5_path = Path(h5_path)
        self._h5f = h5py.File(h5_path, "r")

        # Discover datasets
        self._available = _classify_datasets(self._h5f)
        logger.info("Detected %d categories: %s", len(self._available), sorted(self._available.keys()))

        # Determine T (state frames) and C (camera frames)
        self._T = self._resolve_frame_count(max_frames)
        self._C = self._resolve_camera_count()
        logger.info("State frames=%d, Camera frames=%d", self._T, self._C or 0)

        # Preload non-camera data (small, fits in memory)
        self._state = self._preload_state()

        # Camera forward-fill index: for each state step t, which camera frame to show
        self._cam_idx: np.ndarray | None = None
        if self._C is not None and self._C > 0:
            if self._C < self._T:
                self._cam_idx = np.minimum((np.arange(self._T) * self._C / self._T).astype(int), self._C - 1)
            else:
                self._cam_idx = np.arange(self._T, dtype=int)

        # Init Rerun — app_id includes episode name for the window title;
        # recording_id is made unique per invocation so re-running the same
        # file always creates a fresh recording (no stale-data merge).
        self._blueprint = self._build_blueprint()
        _app_id = f"DexMani - {self._h5_path.stem}"
        _rec_id = f"{self._h5_path.stem}-{time.time_ns()}"
        rr.init(_app_id, recording_id=_rec_id, spawn=True, default_blueprint=self._blueprint)
        self._log_static()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _resolve_frame_count(self, max_frames: int | None) -> int:
        """Determine T from the first available arm or meta key."""
        arm_keys = self._available.get("arm", [])
        meta_keys = self._available.get("meta", [])
        ref_key = next(iter(arm_keys), None) or next(iter(meta_keys), None)
        if ref_key is None:
            # Fallback: pick any dataset
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
        """Determine C from rgb or depth dataset."""
        for cam_key in self._available.get("camera", []):
            return self._h5f[cam_key].shape[0]
        return None

    def _preload_state(self) -> dict[str, np.ndarray]:
        """Read all non-camera datasets into a dict, truncated to T frames."""
        state: dict[str, np.ndarray] = {}
        for _category, keys in self._available.items():
            if _category == "camera":
                continue
            for key in keys:
                data = self._h5f[key][: self._T]
                # HDF5 scalars may return shape (); ensure at least 1-d for flags
                if data.ndim == 0:
                    data = data[()]
                state[key] = np.asarray(data)

        # ── Derived 2-D time series from 3-D hand data ──
        # hand_contact (T,5,3) → force magnitude (T,5) + per-finger Fx/Fy/Fz
        if "hand_contact" in state:
            contact = state["hand_contact"]  # (T, 5, 3)
            state["hand_contact_mag"] = np.linalg.norm(contact, axis=2)  # (T, 5)
            # First two fingers: thumb (0) and index (1) per-axis forces
            state["hand_force_thumb"] = contact[:, 0, :].copy()   # (T, 3) Fx,Fy,Fz
            state["hand_force_index"] = contact[:, 1, :].copy()   # (T, 3) Fx,Fy,Fz

        return state

    # ------------------------------------------------------------------
    # Blueprint
    # ------------------------------------------------------------------

    def _build_blueprint(self) -> rrb.Blueprint:
        """Build a dynamic layout based on available data categories."""
        has_state = bool(self._available.get("arm") or self._available.get("hand"))
        has_action = bool(self._available.get("action"))
        has_flags = bool(self._available.get("flags"))

        columns: list[rrb.Container] = []

        # Camera column
        cam_views = []
        if "rgb" in (self._available.get("camera") or []):
            cam_views.append(rrb.Spatial2DView(origin="camera/rgb", name="RGB"))
        if "depth" in (self._available.get("camera") or []):
            cam_views.append(rrb.Spatial2DView(origin="camera/depth", name="Depth"))
        if cam_views:
            columns.append(rrb.Vertical(contents=cam_views, name="Camera"))

        # Time-series column
        ts_verticals = []

        if has_state:
            state_views = []
            for key in self._available.get("arm", []) + self._available.get("hand", []):
                if key in self._state and 1 <= self._state[key].ndim <= 2:
                    state_views.append(rrb.TimeSeriesView(origin=f"state/{key}", name=key))
            # Derived force views (computed from 3-D hand_contact in _preload_state)
            for fkey in ("hand_contact_mag", "hand_force_thumb", "hand_force_index"):
                if fkey in self._state:
                    state_views.append(rrb.TimeSeriesView(origin=f"state/{fkey}", name=fkey))
            if state_views:
                ts_verticals.append(rrb.Vertical(contents=state_views, name="State"))

        if has_action:
            action_views = []
            for key in self._available.get("action", []):
                if key in self._state and 1 <= self._state[key].ndim <= 2:
                    action_views.append(rrb.TimeSeriesView(origin=f"action/{key}", name=key))
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
                columns.append(rrb.Tabs(contents=ts_verticals, active_tab=0, name="Time Series"))

        return rrb.Blueprint(rrb.Horizontal(contents=columns))

    # ------------------------------------------------------------------
    # Static metadata
    # ------------------------------------------------------------------

    def _log_static(self) -> None:
        """Log series-line labels and camera intrinsics (if available)."""
        for category, keys in self._available.items():
            if category == "camera":
                continue
            for key in keys:
                if key not in self._state:
                    continue
                arr = self._state[key]
                # Skip 3D+ data (can't display as time series)
                if arr.ndim > 2:
                    continue
                base = self._series_origin(category, key)
                if arr.ndim <= 1:
                    # Single series (scalar flag, 1-D timestamp, etc.)
                    rr.log(base, rr.SeriesLine(name=key), static=True)
                else:
                    # Multi-dimensional: one entity path per dimension.
                    # TimeSeriesView(origin="state/arm_qpos") collects children.
                    for i in range(arr.shape[1]):
                        rr.log(f"{base}/{i}", rr.SeriesLine(name=f"{i}"), static=True)

        # Camera pinhole from /meta attrs (optional)
        meta = self._h5f.get("meta")
        if meta is not None and "camera_K" in meta.attrs and "rgb" in self._h5f:
            K = np.asarray(meta.attrs["camera_K"], dtype=float).reshape(3, 3)
            rgb_shape = self._h5f["rgb"].shape
            h, w = rgb_shape[1], rgb_shape[2]
            rr.log("camera", rr.Pinhole(
                image_from_camera=K,
                resolution=[w, h],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=1.25,
            ), static=True)
            logger.info("Camera pinhole logged (%dx%d)", w, h)

        # ── Derived force series labels ──
        _force_series = {
            "hand_contact_mag": ("thumb", "index", "middle", "ring", "pinky"),
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
        """Entity path for a time-series dataset.

        Arm and hand state are grouped under ``state/`` so the blueprint's
        single ``TimeSeriesView(origin="state/<key>")`` collects both.
        """
        if category in ("arm", "hand"):
            return f"state/{key}"
        return f"{category}/{key}"

    # ------------------------------------------------------------------
    # Per-step logging
    # ------------------------------------------------------------------

    def log_step(self, step_idx: int) -> None:
        """Log all data for one timestep."""
        rr.set_time_sequence("step", step_idx)
        # Also set real time if timestamp dataset is available
        if "timestamp" in self._state:
            rr.set_time_seconds("time", float(self._state["timestamp"][step_idx]))
        self._log_camera(step_idx)
        self._log_time_series(step_idx)

    # ------------------------------------------------------------------
    # Camera logging
    # ------------------------------------------------------------------

    def _log_camera(self, step_idx: int) -> None:
        camera_keys = self._available.get("camera", [])
        if not camera_keys:
            return

        cam_idx = int(self._cam_idx[step_idx]) if self._cam_idx is not None else step_idx

        if "rgb" in camera_keys:
            rr.log("camera/rgb", rr.Image(self._h5f["rgb"][cam_idx]))
        if "depth" in camera_keys:
            rr.log("camera/depth", rr.DepthImage(self._h5f["depth"][cam_idx], meter=1000.0))

    # ------------------------------------------------------------------
    # Time series logging
    # ------------------------------------------------------------------

    def _log_time_series(self, step_idx: int) -> None:
        for category, keys in self._available.items():
            if category == "camera":
                continue
            for key in keys:
                if key not in self._state:
                    continue
                arr = self._state[key]
                # Skip 3D+ data (can't display as time series)
                if arr.ndim > 2:
                    continue
                base = self._series_origin(category, key)
                if arr.ndim <= 1:
                    rr.log(base, rr.Scalar(float(arr[step_idx])))
                else:
                    for i in range(arr.shape[1]):
                        rr.log(f"{base}/{i}", rr.Scalar(float(arr[step_idx, i])))

        # ── Derived force scalars (computed from 3-D hand_contact) ──
        for fkey in ("hand_contact_mag", "hand_force_thumb", "hand_force_index"):
            if fkey in self._state:
                arr = self._state[fkey]
                for i in range(arr.shape[1]):
                    rr.log(f"state/{fkey}/{i}", rr.Scalar(float(arr[step_idx, i])))

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def num_steps(self) -> int:
        return self._T

    def close(self) -> None:
        """Release the HDF5 file handle and finalise the Rerun recording."""
        if self._h5f is not None:
            self._h5f.close()
            self._h5f = None  # type: ignore[assignment]
        try:
            rr.disconnect()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a DexMani HDF5 teleop episode with Rerun."
    )
    parser.add_argument(
        "episode",
        type=str,
        help="Path to HDF5 episode file (.h5)",
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
        help="Print HDF5 structure summary and exit (no Rerun needed).",
    )
    args = parser.parse_args()

    h5_path = Path(args.episode).expanduser().resolve()
    if not h5_path.is_file():
        logger.error("File not found: %s", h5_path)
        sys.exit(1)

    if args.info:
        print_episode_info(str(h5_path))
        return

    viz = EpisodeVisualizer(str(h5_path), max_frames=args.max_frames)
    try:
        logger.info("Logging %d frames to Rerun...", viz.num_steps)
        for step in range(viz.num_steps):
            viz.log_step(step)
            if step % 500 == 0:
                logger.info("  frame %d/%d", step, viz.num_steps)
        logger.info("Done. Close the Rerun window to exit.")
    finally:
        viz.close()


if __name__ == "__main__":
    main()
