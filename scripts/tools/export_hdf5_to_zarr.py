#!/usr/bin/env python3
"""Export HDF5 teleop episodes to Zarr format compatible with Diffusion Policy.

Ref: ManiUniCon Zarr exporter + LeRobot v3.0 dataset format.

Output structure:
    <output_dir>/
      data.zarr/
        data/
          obs      (total_frames, obs_dim)   float32
          action   (total_frames, action_dim) float32
        meta/
          episode_ends  (num_episodes,) int64
          norm_stats/
            obs_mean      (obs_dim,) float32
            obs_std       (obs_dim,) float32
            action_mean   (action_dim,) float32
            action_std    (action_dim,) float32

Usage:
    python scripts/tools/export_hdf5_to_zarr.py --data_dir ./recordings/ --output ./zarr_data/

    # Filter only high-quality frames (ALL_GOOD_MASK = 0x07BF)
    python scripts/tools/export_hdf5_to_zarr.py --data_dir ./recordings/ --output ./zarr_data/ --quality_filter

    # Custom quality mask
    python scripts/tools/export_hdf5_to_zarr.py --data_dir ./recordings/ --output ./zarr_data/ --quality_mask 0x0700
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from numcodecs import Blosc

# Default observation / action keys to read from HDF5.
# Maps HDF5 dataset path → target dimension per frame.
_OBS_KEYS: list[tuple[str, int]] = [
    ("obs/arm_qpos", 7),
    ("obs/eef_pos", 3),
    ("obs/eef_quat", 4),
    ("obs/hand_qpos", 12),
]
_ACTION_KEYS: list[tuple[str, int]] = [
    ("action/arm_qpos", 7),
    ("action/hand_qpos", 12),
]

from dexmani_real.recording.quality_flags import ALL_GOOD_MASK


def load_episodes(
    data_dir: Path,
    quality_mask: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Load all HDF5 episodes from data_dir.

    Args:
        data_dir: Directory containing episode_*.h5 files.
        quality_mask: If set, only frames where (flags & mask) == mask are kept.
                      None keeps all frames.

    Returns:
        (obs_list, action_list, episode_ends) where:
          - obs_list[i]  = (num_valid_frames_in_ep_i, obs_dim)
          - action_list[i] = (num_valid_frames_in_ep_i, action_dim)
          - episode_ends  = cumulative frame counts per episode
    """
    import h5py

    data_dir = Path(data_dir)

    obs_list: list[np.ndarray] = []
    action_list: list[np.ndarray] = []
    episode_lengths: list[int] = []

    h5_paths = sorted(data_dir.glob("episode_*.h5"))
    if not h5_paths:
        print(f"[WARN] No episode_*.h5 files found in {data_dir}")
        return [], [], []

    print(f"Found {len(h5_paths)} episode(s) in {data_dir}")

    for h5_path in h5_paths:
        try:
            with h5py.File(str(h5_path), "r") as f:
                # Read quality flags and build frame mask
                if "quality_flags" in f:
                    flags = np.asarray(f["quality_flags"][:], dtype=np.uint16)
                    if quality_mask is not None:
                        valid = (flags & np.uint16(quality_mask)) == np.uint16(quality_mask)
                    else:
                        valid = np.ones(len(flags), dtype=bool)
                else:
                    # No quality flags → keep all frames
                    n_frames = f["obs/arm_qpos"].shape[0]
                    valid = np.ones(n_frames, dtype=bool)

                num_kept = int(np.sum(valid))
                if num_kept == 0:
                    print(f"  [SKIP] {h5_path.name}: 0 valid frames after quality filter")
                    continue

                # Read and concatenate observation channels
                obs_parts = []
                for key, dim in _OBS_KEYS:
                    if key in f:
                        arr = np.asarray(f[key][:], dtype=np.float32)
                        if arr.ndim == 1:
                            arr = arr[:, np.newaxis]
                        assert arr.shape[1] == dim, (
                            f"{h5_path.name}/{key}: expected dim={dim}, got {arr.shape[1]}"
                        )
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
                        assert arr.shape[1] == dim, (
                            f"{h5_path.name}/{key}: expected dim={dim}, got {arr.shape[1]}"
                        )
                        act_parts.append(arr[valid])
                    else:
                        print(f"  [WARN] {h5_path.name}/{key} not found — skipping")
                action = np.concatenate(act_parts, axis=1)

                assert obs.shape[0] == action.shape[0], (
                    f"obs ({obs.shape[0]}) and action ({action.shape[0]}) frame count mismatch"
                )

                obs_list.append(obs)
                action_list.append(action)
                episode_lengths.append(num_kept)

                total = f["obs/arm_qpos"].shape[0]
                print(f"  {h5_path.name}: {num_kept}/{total} frames kept "
                      f"({100 * num_kept / max(total, 1):.1f}%)")

        except (OSError, KeyError, AssertionError) as e:
            print(f"  [ERROR] {h5_path.name}: {e}")

    print(f"Loaded {len(obs_list)} episodes, "
          f"{sum(episode_lengths)} total frames "
          f"(obs_dim={obs_list[0].shape[1] if obs_list else '?'}, "
          f"action_dim={action_list[0].shape[1] if action_list else '?'})")

    return obs_list, action_list, episode_lengths


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


def write_zarr(
    output_dir: Path,
    obs_list: list[np.ndarray],
    action_list: list[np.ndarray],
    episode_lengths: list[int],
    norm_stats: dict[str, np.ndarray],
    compressor: Any | None = None,
) -> None:
    """Write concatenated data to Zarr format.

    Args:
        output_dir: Target directory (data.zarr will be created inside).
        obs_list: Per-episode observation arrays.
        action_list: Per-episode action arrays.
        episode_lengths: Number of valid frames per episode.
        norm_stats: Normalization statistics dict.
        compressor: numcodecs compressor (default: Blosc zstd).
    """
    if compressor is None:
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

    output_dir.mkdir(parents=True, exist_ok=True)
    store_path = output_dir / "data.zarr"

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
        "obs", data=all_obs,
        chunks=(min(1000, total_frames), obs_dim),
        compressor=compressor, dtype=np.float32,
    )
    data_grp.create_dataset(
        "action", data=all_act,
        chunks=(min(1000, total_frames), action_dim),
        compressor=compressor, dtype=np.float32,
    )

    # Write meta
    meta_grp = root.create_group("meta")
    meta_grp.create_dataset(
        "episode_ends", data=episode_ends, dtype=np.int64,
    )

    stats_grp = meta_grp.create_group("norm_stats")
    for key, val in norm_stats.items():
        stats_grp.create_dataset(key, data=val, dtype=np.float32)

    # Write human-readable copy of norm_stats as JSON
    with open(output_dir / "norm_stats.json", "w") as f:
        json.dump({k: v.tolist() for k, v in norm_stats.items()}, f, indent=2)

    # Summary
    print(f"\nZarr export complete:")
    print(f"  {store_path}")
    print(f"  data/obs:      {all_obs.shape} {all_obs.dtype}")
    print(f"  data/action:   {all_act.shape} {all_act.dtype}")
    print(f"  meta/episode_ends: {len(episode_ends)} episodes, total={total_frames}")
    print(f"  meta/norm_stats:  obs({obs_dim}d), action({action_dim}d)")
    print(f"  compression:   Blosc zstd level=3 bitshuffle")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export HDF5 teleop episodes to Zarr (Diffusion Policy compatible)."
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Directory containing episode_*.h5 files.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory (data.zarr will be created inside).",
    )
    parser.add_argument(
        "--quality_filter", action="store_true",
        help=f"Keep only frames with ALL_GOOD_MASK (0x{ALL_GOOD_MASK:04X}) quality flags set.",
    )
    parser.add_argument(
        "--quality_mask", type=lambda x: int(x, 0), default=None,
        help="Custom quality mask (e.g. 0x07BF). Overrides --quality_filter.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        print(f"ERROR: data_dir not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output).expanduser().resolve()

    # Determine quality mask
    quality_mask: int | None = None
    if args.quality_mask is not None:
        quality_mask = args.quality_mask
    elif args.quality_filter:
        quality_mask = ALL_GOOD_MASK

    if quality_mask is not None:
        print(f"Quality filter: mask=0x{quality_mask:04X} (ALL_GOOD=0x{ALL_GOOD_MASK:04X})")

    obs_list, action_list, episode_lengths = load_episodes(data_dir, quality_mask)

    if not obs_list:
        print("No valid episodes to export.", file=sys.stderr)
        sys.exit(1)

    norm_stats = compute_norm_stats(obs_list, action_list)
    write_zarr(output_dir, obs_list, action_list, episode_lengths, norm_stats)


if __name__ == "__main__":
    main()
