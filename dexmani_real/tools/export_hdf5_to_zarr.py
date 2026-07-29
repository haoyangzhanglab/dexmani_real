#!/usr/bin/env python3
"""Export HDF5 teleop episodes to Zarr format compatible with Diffusion Policy.

Ref: ManiUniCon Zarr exporter + LeRobot v3.0 dataset format.

Output structure (single zarr):
    <output_dir>/
      data.zarr/
        data/
          obs      (total_frames, obs_dim)   float32
          action   (total_frames, action_dim) float32
          rgb      (total_frames, H, W, 3)   uint8    (if camera present)
          depth    (total_frames, H, W)      uint16   (if camera present)
        meta/
          episode_ends  (num_episodes,) int64
          camera/                                    (if camera present)
            K               (3,3) float64
            T_world_camera  (4,4) float64
          norm_stats/
            obs_mean      (obs_dim,) float32
            obs_std       (obs_dim,) float32
            action_mean   (action_dim,) float32
            action_std    (action_dim,) float32

Output structure (with --train_val_split 0.8):
    <output_dir>/
      train.zarr/    (80% episodes)
      val.zarr/      (20% episodes)
      norm_stats.json            # train-only stats (no leakage)
      validation_report.json     # DataValidator results (if --validate)

Usage:
    # Basic export
    python -m dexmani_real.tools.export_hdf5_to_zarr --data_dir ./episodes/ --output ./zarr_data/

    # Train/val split + task filter + validation
    python -m dexmani_real.tools.export_hdf5_to_zarr \\
        --data_dir data/episodes/ --output data/export/ \\
        --validate --train_val_split 0.8 \\
        --filter_success true --min_frames 50

    # Timestamp alignment (post-process multi-rate streams to unified grid)
    python -m dexmani_real.tools.export_hdf5_to_zarr \\
        --data_dir data/episodes/ --output data/export/ \\
        --align --align_dt 0.020 --align_method linear
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import zarr
from numcodecs import Blosc

from dexmani_real.recording.episode_reader import EpisodeReader

# Default observation / action keys to read from HDF5.
# Maps HDF5 dataset path → target dimension per frame.
_OBS_KEYS: list[tuple[str, int]] = [
    ("arm_qpos", 7),
    ("arm_ee", 9),
    ("hand_qpos", 12),
]
_ACTION_KEYS: list[tuple[str, int]] = [
    ("action_arm_joint", 7),
    ("action_hand_joint", 12),
]


def _detect_camera_keys(reader: EpisodeReader) -> list[tuple[str, str, str]]:
    """Return list of (rgb_key, depth_key, label) for all cameras in the file.

    Detects HDF5 datasets.
    """
    f = reader.h5f
    pairs: list[tuple[str, str, str]] = []
    if "rgb" in f:
        pairs.append(("rgb", "depth", ""))
    for key in sorted(f.keys()):
        if key.endswith("_rgb") and key != "rgb":
            label = key[:-4]  # strip "_rgb"
            depth_key = f"{label}_depth"
            if depth_key in f:
                pairs.append((key, depth_key, label))
    return pairs


def _read_camera_meta(f: h5py.File) -> dict | None:
    """Read camera intrinsics/extrinsics from /meta attrs.  Returns None if absent."""
    if "meta" not in f:
        return None
    meta = dict(f["meta"].attrs)
    has_cam = meta.get("has_camera", False)
    if not has_cam:
        return None
    try:
        return {
            "serial": str(meta.get("camera_serial", "")),
            "type": str(meta.get("camera_type", "")),
            "depth_scale": float(meta["depth_scale"]) if "depth_scale" in meta else None,
            "K": np.asarray(meta["camera_K"], dtype=np.float64).reshape(3, 3),
            "T_world_camera": np.asarray(meta["camera_T_world_camera"], dtype=np.float64).reshape(4, 4),
        }
    except (KeyError, ValueError):
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Episode loading with metadata filtering
# ═══════════════════════════════════════════════════════════════════════════════


def _read_episode_meta(h5_path: Path) -> dict:
    """Read metadata attributes from an HDF5 episode file.

    Adds a derived ``held_ratio`` key (fraction of frames where the command
    was held — no fresh VR/IK result) computed from /flag_held when present.
    """
    try:
        with EpisodeReader(h5_path) as reader:
            f = reader.h5f
            if "meta" not in f:
                return {}
            meta = dict(f["meta"].attrs)
            if "flag_held" in f:
                held = np.asarray(f["flag_held"][:], dtype=bool)
                if held.size > 0:
                    meta["held_ratio"] = float(held.mean())
            return meta
    except (OSError, KeyError):
        return {}


def _episode_passes_filters(
    meta: dict,
    filter_task: str | None = None,
    filter_success: bool | None = None,
    filter_tags: str | None = None,
    min_frames: int | None = None,
    max_held_ratio: float | None = None,
) -> bool:
    """Check if episode metadata passes all filter criteria."""
    if filter_task is not None:
        task = str(meta.get("task_label", ""))
        if filter_task not in task:
            return False

    if filter_success is not None:
        success_val = meta.get("success", None)
        if isinstance(success_val, (np.bool_, np.integer)):
            success_val = bool(success_val)
        if success_val != filter_success:
            return False

    if filter_tags is not None:
        tags_str = str(meta.get("tags", ""))
        if filter_tags not in tags_str:
            return False

    if min_frames is not None:
        nf = int(meta.get("num_frames", 0))
        if nf < min_frames:
            return False

    if max_held_ratio is not None:
        held_ratio = meta.get("held_ratio", None)
        if held_ratio is not None and float(held_ratio) > max_held_ratio:
            return False

    return True


def load_episodes(
    data_dir: Path,
    filter_task: str | None = None,
    filter_success: bool | None = None,
    filter_tags: str | None = None,
    min_frames: int | None = None,
    max_held_ratio: float | None = None,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[int],
    list[Path],
    list[np.ndarray | None],
    list[np.ndarray | None],
    dict | None,
]:
    """Load all HDF5 episodes from data_dir with optional metadata filtering.

    Args:
        data_dir: Directory containing episode_*.h5 files.
        filter_task: Episode-level filter: substring match on task_label.
        filter_success: Episode-level filter: exact match on success flag.
        filter_tags: Episode-level filter: substring match on tags.
        min_frames: Episode-level filter: minimum frame count.
        max_held_ratio: Episode-level filter: exclude episodes whose fraction
            of held frames (/flag_held) exceeds this value.

    Returns:
        (obs_list, action_list, episode_lengths, episode_paths,
         rgb_list, depth_list, camera_meta).

        rgb_list / depth_list entries are None when the episode has no camera.
        camera_meta is from the first episode with camera data (or None).
    """
    data_dir = Path(data_dir)

    obs_list: list[np.ndarray] = []
    action_list: list[np.ndarray] = []
    episode_lengths: list[int] = []
    episode_paths: list[Path] = []
    rgb_list: list[np.ndarray | None] = []
    depth_list: list[np.ndarray | None] = []
    camera_meta: dict | None = None

    h5_paths = sorted(data_dir.glob("episode_*.h5"))
    if not h5_paths:
        print(f"[WARN] No episode_*.h5 files found in {data_dir}")
        return [], [], [], [], [], [], None

    print(f"Found {len(h5_paths)} episode(s) in {data_dir}")

    skipped_meta = 0
    for h5_path in h5_paths:
        try:
            # ── Episode-level metadata filtering ──
            meta = _read_episode_meta(h5_path)
            if not _episode_passes_filters(
                meta,
                filter_task,
                filter_success,
                filter_tags,
                min_frames,
                max_held_ratio,
            ):
                skipped_meta += 1
                continue

            with EpisodeReader(h5_path) as reader:
                f = reader.h5f
                n_frames = f["arm_qpos"].shape[0]
                valid = np.ones(n_frames, dtype=bool)

                num_kept = int(np.sum(valid))
                if num_kept == 0:
                    print(f"  [SKIP] {h5_path.name}: 0 valid frames")
                    continue

                # Read and concatenate observation channels
                obs_parts = []
                for key, dim in _OBS_KEYS:
                    if key in f:
                        arr = np.asarray(f[key][:], dtype=np.float32)
                        if arr.ndim == 1:
                            arr = arr[:, np.newaxis]
                        assert arr.shape[1] == dim, f"{h5_path.name}/{key}: expected dim={dim}, got {arr.shape[1]}"
                        obs_parts.append(arr[valid])
                    else:
                        print(f"  [WARN] {h5_path.name}/{key} not found — skipping")
                obs = np.concatenate(obs_parts, axis=1)

                # Read and concatenate action channels
                act_parts = []
                for key, dim in _ACTION_KEYS:
                    if key in f:
                        arr = np.asarray(f[key][:], dtype=np.float32)
                        if arr.ndim == 1:
                            arr = arr[:, np.newaxis]
                        assert arr.shape[1] == dim, f"{h5_path.name}/{key}: expected dim={dim}, got {arr.shape[1]}"
                        act_parts.append(arr[valid])
                    else:
                        print(f"  [WARN] {h5_path.name}/{key} not found — skipping")
                action = np.concatenate(act_parts, axis=1)

                assert (
                    obs.shape[0] == action.shape[0]
                ), f"obs ({obs.shape[0]}) and action ({action.shape[0]}) frame count mismatch"

                # ── Camera frames ──
                episode_rgb: np.ndarray | None = None
                episode_depth: np.ndarray | None = None
                cam_keys = _detect_camera_keys(reader)
                if cam_keys:
                    # Use first camera only (additional cameras stored under data/{label}_rgb)
                    rgb_k, depth_k, _label = cam_keys[0]
                    has_rgb = "rgb" in f
                    has_depth = "depth" in f
                    if has_rgb and has_depth:
                        episode_rgb = reader.read_camera_all(rgb_k)[valid]
                        episode_depth = reader.read_camera_all(depth_k)[valid]
                        # Read camera metadata from first episode that has it
                        if camera_meta is None:
                            camera_meta = _read_camera_meta(f)

                obs_list.append(obs)
                action_list.append(action)
                episode_lengths.append(num_kept)
                episode_paths.append(h5_path)
                rgb_list.append(episode_rgb)
                depth_list.append(episode_depth)

                total = f["arm_qpos"].shape[0]
                print(f"  {h5_path.name}: {num_kept}/{total} frames kept " f"({100 * num_kept / max(total, 1):.1f}%)")

        except (OSError, KeyError, AssertionError) as e:
            print(f"  [ERROR] {h5_path.name}: {e}")

    if skipped_meta > 0:
        print(f"  Filtered out {skipped_meta} episode(s) by metadata filters")

    has_cam = any(r is not None for r in rgb_list)
    print(
        f"Loaded {len(obs_list)} episodes, "
        f"{sum(episode_lengths)} total frames "
        f"(obs_dim={obs_list[0].shape[1] if obs_list else '?'}, "
        f"action_dim={action_list[0].shape[1] if action_list else '?'}"
        f"{', camera=yes' if has_cam else ', camera=no'})"
    )

    return obs_list, action_list, episode_lengths, episode_paths, rgb_list, depth_list, camera_meta


# ═══════════════════════════════════════════════════════════════════════════════
# Train/val split (episode level)
# ═══════════════════════════════════════════════════════════════════════════════


def split_train_val(
    obs_list: list[np.ndarray],
    action_list: list[np.ndarray],
    episode_lengths: list[int],
    train_ratio: float = 0.8,
    seed: int = 42,
    rgb_list: list[np.ndarray | None] | None = None,
    depth_list: list[np.ndarray | None] | None = None,
) -> tuple[list, list, list, list, list, list, list | None, list | None, list | None, list | None]:
    """Split episodes into train/val sets at the episode level.

    Args:
        obs_list, action_list, episode_lengths: Per-episode data.
        train_ratio: Fraction of episodes for training (default 0.8).
        seed: Random seed for reproducible splits.
        rgb_list, depth_list: Optional per-episode camera data.

    Returns:
        (train_obs, train_act, train_lengths, val_obs, val_act, val_lengths,
         train_rgb, train_depth, val_rgb, val_depth).
    """
    if rgb_list is None:
        rgb_list = []
    if depth_list is None:
        depth_list = []
    n = len(obs_list)
    if n == 0:
        return [], [], [], [], [], [], [], [], [], []

    if n == 1:
        print("[WARN] Only 1 episode — placing it in train set.")
        return (
            obs_list,
            action_list,
            episode_lengths,
            [],
            [],
            [],
            rgb_list if rgb_list else None,
            depth_list if depth_list else None,
            None,
            None,
        )

    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    split = max(1, int(n * train_ratio))
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_obs = [obs_list[i] for i in train_idx]
    train_act = [action_list[i] for i in train_idx]
    train_lengths = [episode_lengths[i] for i in train_idx]

    val_obs = [obs_list[i] for i in val_idx] if val_idx else []
    val_act = [action_list[i] for i in val_idx] if val_idx else []
    val_lengths = [episode_lengths[i] for i in val_idx] if val_idx else []

    train_rgb = [rgb_list[i] for i in train_idx] if rgb_list else None
    train_depth = [depth_list[i] for i in train_idx] if depth_list else None
    val_rgb = [rgb_list[i] for i in val_idx] if val_idx and rgb_list else None
    val_depth = [depth_list[i] for i in val_idx] if val_idx and depth_list else None

    train_frames = sum(train_lengths)
    val_frames = sum(val_lengths)
    has_cam = rgb_list and any(r is not None for r in rgb_list)
    print(f"\nTrain/val split (ratio={train_ratio}, seed={seed}):")
    print(
        f"  Train: {len(train_obs)} episodes, {train_frames} frames"
        f"{', +camera' if has_cam and train_rgb and any(r is not None for r in train_rgb) else ''}"
    )
    print(
        f"  Val:   {len(val_obs)} episodes, {val_frames} frames"
        f"{', +camera' if has_cam and val_rgb and any(r is not None for r in val_rgb) else ''}"
    )

    return (
        train_obs,
        train_act,
        train_lengths,
        val_obs,
        val_act,
        val_lengths,
        train_rgb,
        train_depth,
        val_rgb,
        val_depth,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Timestamp alignment
# ═══════════════════════════════════════════════════════════════════════════════


def _align_all_episodes(
    episode_paths: list[Path],
    obs_list: list[np.ndarray],
    action_list: list[np.ndarray],
    episode_lengths: list[int],
    dt: float = 0.020,
    method: str = "linear",
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Align all episodes using TimestampAligner and reconstruct obs/action.

    Returns updated (obs_list, action_list, episode_lengths).
    """
    from dexmani_real.recording.post_processor import TimestampAligner

    aligner = TimestampAligner(dt=dt, method=method, max_gap_s=2.5 * dt)  # gap threshold scales with grid rate

    new_obs_list: list[np.ndarray] = []
    new_action_list: list[np.ndarray] = []
    new_lengths: list[int] = []

    for i, h5_path in enumerate(episode_paths):
        try:
            aligned = aligner.align(str(h5_path), dt=dt)
            if aligned is None:
                print(f"  [SKIP] {h5_path.name}: alignment failed")
                # Keep original
                new_obs_list.append(obs_list[i])
                new_action_list.append(action_list[i])
                new_lengths.append(episode_lengths[i])
                continue

            # Reconstruct obs from aligned data
            obs_parts = []
            for key, dim in _OBS_KEYS:
                if key in aligned:
                    arr = aligned[key]
                    if arr.ndim == 1:
                        arr = arr[:, np.newaxis]
                    obs_parts.append(arr)
            if obs_parts:
                new_obs = np.concatenate(obs_parts, axis=1).astype(np.float32)
            else:
                new_obs = obs_list[i]

            # Reconstruct action from aligned data
            act_parts = []
            for key, dim in _ACTION_KEYS:
                if key in aligned:
                    arr = aligned[key]
                    if arr.ndim == 1:
                        arr = arr[:, np.newaxis]
                    act_parts.append(arr)
            if act_parts:
                new_act = np.concatenate(act_parts, axis=1).astype(np.float32)
            else:
                new_act = action_list[i]

            # Remove NaN frames (gaps beyond max_gap)
            valid_mask = ~np.isnan(new_obs).any(axis=1)
            new_obs = new_obs[valid_mask]
            new_act = new_act[valid_mask]

            original_len = len(obs_list[i])
            aligned_len = len(new_obs)
            print(
                f"  {h5_path.name}: aligned {original_len}→{aligned_len} frames "
                f"(removed {original_len - aligned_len} NaN-gap frames)"
            )

            new_obs_list.append(new_obs)
            new_action_list.append(new_act)
            new_lengths.append(aligned_len)

        except (OSError, KeyError, ValueError) as e:
            print(f"  [ERROR] {h5_path.name}: alignment error: {e}")
            # Keep original
            new_obs_list.append(obs_list[i])
            new_action_list.append(action_list[i])
            new_lengths.append(episode_lengths[i])

    return new_obs_list, new_action_list, new_lengths


# ═══════════════════════════════════════════════════════════════════════════════
# Norm stats (train-only to avoid leakage)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_norm_stats(
    obs_list: list[np.ndarray],
    action_list: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute per-dimension mean and std across all frames.

    Ref: ManiUniCon pre-computed obs_mean/std and action_mean/std.
    Uses Welford-style incremental computation for numerical stability.
    """
    if not obs_list:
        return {}

    all_obs = np.concatenate(obs_list, axis=0)
    all_act = np.concatenate(action_list, axis=0)

    # Guard against zero std (constant dimensions)
    obs_std = np.std(all_obs, axis=0)
    act_std = np.std(all_act, axis=0)
    obs_std = np.where(obs_std < 1e-8, 1.0, obs_std)
    act_std = np.where(act_std < 1e-8, 1.0, act_std)

    return {
        "obs_mean": np.mean(all_obs, axis=0).astype(np.float32),
        "obs_std": obs_std.astype(np.float32),
        "action_mean": np.mean(all_act, axis=0).astype(np.float32),
        "action_std": act_std.astype(np.float32),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Zarr writer
# ═══════════════════════════════════════════════════════════════════════════════


def _episode_control_rates(paths: list) -> list[float]:
    """Nominal grid rate per episode (schema v7: control_hz; older files: fps; 50.0 if absent)."""
    rates: list[float] = []
    for p in paths:
        try:
            with EpisodeReader(p) as reader:
                f = reader.h5f
                meta = f.get("meta")
                rates.append(float(meta.attrs.get("control_hz", meta.attrs.get("fps", 50.0))) if meta else 50.0)
        except OSError:
            rates.append(50.0)
    return rates


def write_zarr(
    output_dir: Path,
    obs_list: list[np.ndarray],
    action_list: list[np.ndarray],
    episode_lengths: list[int],
    norm_stats: dict[str, np.ndarray],
    name: str = "data",
    compressor: Any | None = None,
    rgb_list: list[np.ndarray | None] | None = None,
    depth_list: list[np.ndarray | None] | None = None,
    camera_meta: dict | None = None,
    control_hz: float | None = None,
) -> None:
    """Write concatenated data to Zarr format.

    Args:
        output_dir: Target directory.
        obs_list, action_list, episode_lengths: Per-episode data.
        norm_stats: Normalization statistics dict.
        name: Zarr store name (e.g. "data", "train", "val").
        compressor: numcodecs compressor (default: Blosc zstd).
        rgb_list, depth_list: Optional per-episode camera frames.
        camera_meta: Optional camera intrinsics/extrinsics dict.
        control_hz: Uniform nominal grid rate of all episodes (written to
            meta attrs for downstream dt derivation); None when mixed/unknown.
    """
    if compressor is None:
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

    if not obs_list:
        print(f"[WARN] No episodes for {name}.zarr — skipping.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    store_path = output_dir / f"{name}.zarr"

    all_obs = np.concatenate(obs_list, axis=0)
    all_act = np.concatenate(action_list, axis=0)

    total_frames = all_obs.shape[0]
    obs_dim = all_obs.shape[1]
    action_dim = all_act.shape[1]

    # Cumulative episode ends
    episode_ends = np.cumsum(episode_lengths, dtype=np.int64)

    root = zarr.open_group(str(store_path), mode="w")

    # Write data arrays
    data_grp = root.create_group("data")
    data_grp.create_dataset(
        "obs",
        data=all_obs,
        chunks=(min(1000, total_frames), obs_dim),
        compressor=compressor,
        dtype=np.float32,
    )
    data_grp.create_dataset(
        "action",
        data=all_act,
        chunks=(min(1000, total_frames), action_dim),
        compressor=compressor,
        dtype=np.float32,
    )

    # Write meta
    meta_grp = root.create_group("meta")
    meta_grp.create_dataset(
        "episode_ends",
        data=episode_ends,
        dtype=np.int64,
    )
    if control_hz is not None:
        # Nominal grid rate — downstream consumers derive dt = 1/control_hz
        meta_grp.attrs["control_hz"] = float(control_hz)

    if norm_stats:
        stats_grp = meta_grp.create_group("norm_stats")
        for key, val in norm_stats.items():
            stats_grp.create_dataset(key, data=val, dtype=np.float32)

    # ── Camera frames ──
    has_cam = rgb_list is not None and any(r is not None for r in rgb_list)
    all_rgb = None
    all_depth = None
    if has_cam and rgb_list is not None and depth_list is not None:
        all_rgb = np.concatenate([r for r in rgb_list if r is not None], axis=0)
        all_depth = np.concatenate([d for d in depth_list if d is not None], axis=0)

        # Image compressor: zstd without bitshuffle (images don't benefit from it)
        img_compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.NOSHUFFLE)

        data_grp.create_dataset(
            "rgb",
            data=all_rgb,
            chunks=(1,) + all_rgb.shape[1:],
            compressor=img_compressor,
            dtype=np.uint8,
        )
        data_grp.create_dataset(
            "depth",
            data=all_depth,
            chunks=(1,) + all_depth.shape[1:],
            compressor=img_compressor,
            dtype=np.uint16,
        )

        # Camera metadata
        if camera_meta is not None:
            cam_grp = meta_grp.create_group("camera")
            cam_grp.create_dataset("K", data=camera_meta["K"], dtype=np.float64)
            cam_grp.create_dataset("T_world_camera", data=camera_meta["T_world_camera"], dtype=np.float64)
            cam_grp.attrs["serial"] = camera_meta["serial"]
            cam_grp.attrs["type"] = camera_meta["type"]
            if camera_meta.get("depth_scale") is not None:
                cam_grp.attrs["depth_scale"] = camera_meta["depth_scale"]

    # Summary
    print(f"\n{name}.zarr export complete:")
    print(f"  {store_path}")
    print(f"  data/obs:      {all_obs.shape} {all_obs.dtype}")
    print(f"  data/action:   {all_act.shape} {all_act.dtype}")
    print(f"  meta/episode_ends: {len(episode_ends)} episodes, total={total_frames}")
    if control_hz is not None:
        print(f"  meta/control_hz:  {control_hz:g} Hz")
    if norm_stats:
        print(f"  meta/norm_stats:  obs({obs_dim}d), action({action_dim}d)")
    if has_cam and all_rgb is not None and all_depth is not None:
        print(f"  data/rgb:      {all_rgb.shape} {all_rgb.dtype}")
        print(f"  data/depth:    {all_depth.shape} {all_depth.dtype}")
        if camera_meta:
            print(f"  meta/camera:   serial={camera_meta['serial']}")
    print(f"  compression:   Blosc zstd level=3 bitshuffle")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HDF5 teleop episodes to Zarr (Diffusion Policy compatible).")
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing episode_*.h5 files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory (zarr files will be created inside).",
    )
    # ── Train/val split ──
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=None,
        help="Train ratio for episode-level split (e.g. 0.8 = 80%% train, 20%% val).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42).",
    )
    # ── Episode metadata filters ──
    parser.add_argument(
        "--filter_task",
        type=str,
        default=None,
        help="Only include episodes whose task_label contains this string.",
    )
    parser.add_argument(
        "--filter_success",
        type=lambda x: x.lower() == "true" if x.lower() in ("true", "false") else None,
        default=None,
        help="Only include episodes with success=true or success=false.",
    )
    parser.add_argument(
        "--filter_tags",
        type=str,
        default=None,
        help="Only include episodes whose tags contain this string.",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=None,
        help="Minimum frame count per episode.",
    )
    parser.add_argument(
        "--max_held_ratio",
        type=float,
        default=None,
        help="Exclude episodes whose held-frame ratio (/flag_held mean) exceeds this value (e.g. 0.2).",
    )
    # ── Validation ──
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run DataValidator on episodes before export.",
    )
    # ── Timestamp alignment ──
    parser.add_argument(
        "--align",
        action="store_true",
        help="Post-process timestamp alignment: interpolate all streams to a unified time grid.",
    )
    parser.add_argument(
        "--align_dt",
        type=float,
        default=None,
        help="Target dt for aligned grid in seconds (default: derived from each "
        "episode's /meta control_hz|fps; 0.020 if absent).",
    )
    parser.add_argument(
        "--align_method",
        type=str,
        default="linear",
        choices=["linear", "nearest"],
        help="Interpolation method for alignment (default: linear).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        print(f"ERROR: data_dir not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output).expanduser().resolve()

    # ── Validation (if requested) ──
    if args.validate:
        from dexmani_real.recording.data_validator import DataValidator

        validator = DataValidator(
            min_frames=args.min_frames if args.min_frames is not None else 50,
        )
        print("Running DataValidator...")
        reports = validator.validate_directory(data_dir)
        valid_count = sum(1 for r in reports if r.is_valid)
        print(f"Validation: {valid_count}/{len(reports)} episodes passed")
        DataValidator.save_reports(reports, output_dir / "validation_report.json")

    # ── Load episodes ──
    obs_list, action_list, episode_lengths, episode_paths, rgb_list, depth_list, camera_meta = load_episodes(
        data_dir,
        filter_task=args.filter_task,
        filter_success=args.filter_success,
        filter_tags=args.filter_tags,
        min_frames=args.min_frames,
        max_held_ratio=args.max_held_ratio,
    )

    if not obs_list:
        print("No valid episodes to export.", file=sys.stderr)
        sys.exit(1)

    # ── Rate consistency across the selected episodes ──
    rates = _episode_control_rates(episode_paths)
    unique_rates = sorted({round(r, 3) for r in rates})
    control_hz: float | None = unique_rates[0] if len(unique_rates) == 1 else None

    # ── Timestamp alignment (if requested) ──
    if args.align:
        if args.align_dt is None:
            if control_hz is None:
                print(
                    f"ERROR: mixed control rates {unique_rates} Hz across episodes — "
                    f"pass an explicit --align_dt to choose the target grid.",
                    file=sys.stderr,
                )
                sys.exit(1)
            # Nominal grid rate shared by all selected episodes (schema v7:
            # control_hz; older files: fps attr).
            args.align_dt = 1.0 / control_hz
        control_hz = 1.0 / args.align_dt
        print(f"\nTimestamp alignment: dt={args.align_dt*1000:.1f}ms method={args.align_method}")
        if any(r is not None for r in rgb_list):
            print("[WARN] Camera frames dropped during timestamp alignment (not interpolatable).")
            rgb_list = [None] * len(rgb_list)
            depth_list = [None] * len(depth_list)
        obs_list, action_list, episode_lengths = _align_all_episodes(
            episode_paths,
            obs_list,
            action_list,
            episode_lengths,
            dt=args.align_dt,
            method=args.align_method,
        )
    elif control_hz is None:
        print(
            f"[WARN] Mixed control rates {unique_rates} Hz concatenated WITHOUT alignment — "
            f"per-step action magnitudes differ across episodes and the zarr carries "
            f"no control_hz. Use --align (with --align_dt) to unify the grid."
        )

    # ── Train/val split ──
    if args.train_val_split is not None:
        (
            train_obs,
            train_act,
            train_lengths,
            val_obs,
            val_act,
            val_lengths,
            train_rgb,
            train_depth,
            val_rgb,
            val_depth,
        ) = split_train_val(
            obs_list, action_list, episode_lengths, args.train_val_split, args.seed, rgb_list, depth_list
        )

        # Compute norm_stats from TRAIN ONLY (no leakage)
        norm_stats = compute_norm_stats(train_obs, train_act)

        # Write train
        write_zarr(
            output_dir,
            train_obs,
            train_act,
            train_lengths,
            norm_stats,
            name="train",
            rgb_list=train_rgb,
            depth_list=train_depth,
            camera_meta=camera_meta,
            control_hz=control_hz,
        )

        # Write val (use train stats — no separate stats for val, prevents leakage)
        write_zarr(
            output_dir,
            val_obs,
            val_act,
            val_lengths,
            norm_stats,
            name="val",
            rgb_list=val_rgb,
            depth_list=val_depth,
            camera_meta=camera_meta,
            control_hz=control_hz,
        )

        # Save norm_stats as human-readable JSON
        with open(output_dir / "norm_stats.json", "w") as f:
            json.dump({k: v.tolist() for k, v in norm_stats.items()}, f, indent=2)

        print(f"\nExport complete: {output_dir}")
        print(f"  train.zarr + val.zarr + norm_stats.json")
    else:
        # Single zarr (no split)
        norm_stats = compute_norm_stats(obs_list, action_list)
        write_zarr(
            output_dir,
            obs_list,
            action_list,
            episode_lengths,
            norm_stats,
            rgb_list=rgb_list,
            depth_list=depth_list,
            camera_meta=camera_meta,
            control_hz=control_hz,
        )

        # Save norm_stats
        with open(output_dir / "norm_stats.json", "w") as f:
            json.dump({k: v.tolist() for k, v in norm_stats.items()}, f, indent=2)

        print(f"\nExport complete: {output_dir}")


if __name__ == "__main__":
    main()
