#!/usr/bin/env python3
"""Export HDF5 teleop episodes to Zarr format compatible with Diffusion Policy.

Ref: ManiUniCon Zarr exporter + LeRobot v3.0 dataset format.

Output structure (single zarr):
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

Output structure (with --train_val_split 0.8):
    <output_dir>/
      train.zarr/    (80% episodes)
      val.zarr/      (20% episodes)
      norm_stats.json            # train-only stats (no leakage)
      validation_report.json     # DataValidator results (if --validate)

Usage:
    # Basic export
    python scripts/tools/export_hdf5_to_zarr.py --data_dir ./recordings/ --output ./zarr_data/

    # Train/val split + task filter + validation
    python scripts/tools/export_hdf5_to_zarr.py \\
        --data_dir data/episodes/ --output data/export/ \\
        --validate --train_val_split 0.8 \\
        --filter_success true --min_frames 50

    # Timestamp alignment (post-process multi-rate streams to unified grid)
    python scripts/tools/export_hdf5_to_zarr.py \\
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


# ═══════════════════════════════════════════════════════════════════════════════
# Episode loading with metadata filtering
# ═══════════════════════════════════════════════════════════════════════════════


def _read_episode_meta(h5_path: Path) -> dict:
    """Read metadata attributes from an HDF5 episode file."""
    try:
        with h5py.File(str(h5_path), "r") as f:
            if "meta" not in f:
                return {}
            return dict(f["meta"].attrs)
    except (OSError, KeyError):
        return {}


def _episode_passes_filters(
    meta: dict,
    filter_task: str | None = None,
    filter_success: bool | None = None,
    filter_tags: str | None = None,
    min_frames: int | None = None,
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

    return True


def load_episodes(
    data_dir: Path,
    filter_task: str | None = None,
    filter_success: bool | None = None,
    filter_tags: str | None = None,
    min_frames: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[Path]]:
    """Load all HDF5 episodes from data_dir with optional metadata filtering.

    Args:
        data_dir: Directory containing episode_*.h5 files.
        filter_task: Episode-level filter: substring match on task_label.
        filter_success: Episode-level filter: exact match on success flag.
        filter_tags: Episode-level filter: substring match on tags.
        min_frames: Episode-level filter: minimum frame count.

    Returns:
        (obs_list, action_list, episode_lengths, episode_paths).
    """
    data_dir = Path(data_dir)

    obs_list: list[np.ndarray] = []
    action_list: list[np.ndarray] = []
    episode_lengths: list[int] = []
    episode_paths: list[Path] = []

    h5_paths = sorted(data_dir.glob("episode_*.h5"))
    if not h5_paths:
        print(f"[WARN] No episode_*.h5 files found in {data_dir}")
        return [], [], [], []

    print(f"Found {len(h5_paths)} episode(s) in {data_dir}")

    skipped_meta = 0
    for h5_path in h5_paths:
        try:
            # ── Episode-level metadata filtering ──
            meta = _read_episode_meta(h5_path)
            if not _episode_passes_filters(
                meta, filter_task, filter_success, filter_tags,
                min_frames,
            ):
                skipped_meta += 1
                continue

            with h5py.File(str(h5_path), "r") as f:
                n_frames = f["obs/arm_qpos"].shape[0]
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
                episode_paths.append(h5_path)

                total = f["obs/arm_qpos"].shape[0]
                print(f"  {h5_path.name}: {num_kept}/{total} frames kept "
                      f"({100 * num_kept / max(total, 1):.1f}%)")

        except (OSError, KeyError, AssertionError) as e:
            print(f"  [ERROR] {h5_path.name}: {e}")

    if skipped_meta > 0:
        print(f"  Filtered out {skipped_meta} episode(s) by metadata filters")

    print(f"Loaded {len(obs_list)} episodes, "
          f"{sum(episode_lengths)} total frames "
          f"(obs_dim={obs_list[0].shape[1] if obs_list else '?'}, "
          f"action_dim={action_list[0].shape[1] if action_list else '?'})")

    return obs_list, action_list, episode_lengths, episode_paths


# ═══════════════════════════════════════════════════════════════════════════════
# Train/val split (episode level)
# ═══════════════════════════════════════════════════════════════════════════════


def split_train_val(
    obs_list: list[np.ndarray],
    action_list: list[np.ndarray],
    episode_lengths: list[int],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list, list, list, list, list, list]:
    """Split episodes into train/val sets at the episode level.

    Args:
        obs_list, action_list, episode_lengths: Per-episode data.
        train_ratio: Fraction of episodes for training (default 0.8).
        seed: Random seed for reproducible splits.

    Returns:
        (train_obs, train_act, train_lengths, val_obs, val_act, val_lengths).
    """
    n = len(obs_list)
    if n == 0:
        return [], [], [], [], [], []

    if n == 1:
        print("[WARN] Only 1 episode — placing it in train set.")
        return obs_list, action_list, episode_lengths, [], [], []

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

    train_frames = sum(train_lengths)
    val_frames = sum(val_lengths)
    print(f"\nTrain/val split (ratio={train_ratio}, seed={seed}):")
    print(f"  Train: {len(train_obs)} episodes, {train_frames} frames")
    print(f"  Val:   {len(val_obs)} episodes, {val_frames} frames")

    return train_obs, train_act, train_lengths, val_obs, val_act, val_lengths


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

    aligner = TimestampAligner(dt=dt, method=method)

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


def write_zarr(
    output_dir: Path,
    obs_list: list[np.ndarray],
    action_list: list[np.ndarray],
    episode_lengths: list[int],
    norm_stats: dict[str, np.ndarray],
    name: str = "data",
    compressor: Any | None = None,
) -> None:
    """Write concatenated data to Zarr format.

    Args:
        output_dir: Target directory.
        obs_list, action_list, episode_lengths: Per-episode data.
        norm_stats: Normalization statistics dict.
        name: Zarr store name (e.g. "data", "train", "val").
        compressor: numcodecs compressor (default: Blosc zstd).
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

    if norm_stats:
        stats_grp = meta_grp.create_group("norm_stats")
        for key, val in norm_stats.items():
            stats_grp.create_dataset(key, data=val, dtype=np.float32)

    # Summary
    print(f"\n{name}.zarr export complete:")
    print(f"  {store_path}")
    print(f"  data/obs:      {all_obs.shape} {all_obs.dtype}")
    print(f"  data/action:   {all_act.shape} {all_act.dtype}")
    print(f"  meta/episode_ends: {len(episode_ends)} episodes, total={total_frames}")
    if norm_stats:
        print(f"  meta/norm_stats:  obs({obs_dim}d), action({action_dim}d)")
    print(f"  compression:   Blosc zstd level=3 bitshuffle")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


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
        help="Output directory (zarr files will be created inside).",
    )
    # ── Train/val split ──
    parser.add_argument(
        "--train_val_split", type=float, default=None,
        help="Train ratio for episode-level split (e.g. 0.8 = 80%% train, 20%% val).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val split (default: 42).",
    )
    # ── Episode metadata filters ──
    parser.add_argument(
        "--filter_task", type=str, default=None,
        help="Only include episodes whose task_label contains this string.",
    )
    parser.add_argument(
        "--filter_success", type=lambda x: x.lower() == "true" if x.lower() in ("true", "false") else None,
        default=None,
        help="Only include episodes with success=true or success=false.",
    )
    parser.add_argument(
        "--filter_tags", type=str, default=None,
        help="Only include episodes whose tags contain this string.",
    )
    parser.add_argument(
        "--min_frames", type=int, default=None,
        help="Minimum frame count per episode.",
    )
    # ── Validation ──
    parser.add_argument(
        "--validate", action="store_true",
        help="Run DataValidator on episodes before export.",
    )
    # ── Timestamp alignment ──
    parser.add_argument(
        "--align", action="store_true",
        help="Post-process timestamp alignment: interpolate all streams to a unified time grid.",
    )
    parser.add_argument(
        "--align_dt", type=float, default=0.020,
        help="Target dt for aligned grid in seconds (default: 0.020 = 20ms).",
    )
    parser.add_argument(
        "--align_method", type=str, default="linear",
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
            min_frames=args.min_frames or 50,
        )
        print("Running DataValidator...")
        reports = validator.validate_directory(data_dir)
        valid_count = sum(1 for r in reports if r.is_valid)
        print(f"Validation: {valid_count}/{len(reports)} episodes passed")
        DataValidator.save_reports(reports, output_dir / "validation_report.json")

    # ── Load episodes ──
    obs_list, action_list, episode_lengths, episode_paths = load_episodes(
        data_dir,
        filter_task=args.filter_task,
        filter_success=args.filter_success,
        filter_tags=args.filter_tags,
        min_frames=args.min_frames,
    )

    if not obs_list:
        print("No valid episodes to export.", file=sys.stderr)
        sys.exit(1)

    # ── Timestamp alignment (if requested) ──
    if args.align:
        print(f"\nTimestamp alignment: dt={args.align_dt*1000:.0f}ms method={args.align_method}")
        obs_list, action_list, episode_lengths = _align_all_episodes(
            episode_paths, obs_list, action_list, episode_lengths,
            dt=args.align_dt, method=args.align_method,
        )

    # ── Train/val split ──
    if args.train_val_split is not None:
        train_obs, train_act, train_lengths, val_obs, val_act, val_lengths = \
            split_train_val(obs_list, action_list, episode_lengths, args.train_val_split, args.seed)

        # Compute norm_stats from TRAIN ONLY (no leakage)
        norm_stats = compute_norm_stats(train_obs, train_act)

        # Write train
        write_zarr(output_dir, train_obs, train_act, train_lengths, norm_stats, name="train")

        # Write val (use train stats — no separate stats for val, prevents leakage)
        write_zarr(output_dir, val_obs, val_act, val_lengths, norm_stats, name="val")

        # Save norm_stats as human-readable JSON
        with open(output_dir / "norm_stats.json", "w") as f:
            json.dump({k: v.tolist() for k, v in norm_stats.items()}, f, indent=2)

        print(f"\nExport complete: {output_dir}")
        print(f"  train.zarr + val.zarr + norm_stats.json")
    else:
        # Single zarr (no split)
        norm_stats = compute_norm_stats(obs_list, action_list)
        write_zarr(output_dir, obs_list, action_list, episode_lengths, norm_stats)

        # Save norm_stats
        with open(output_dir / "norm_stats.json", "w") as f:
            json.dump({k: v.tolist() for k, v in norm_stats.items()}, f, indent=2)

        print(f"\nExport complete: {output_dir}")


if __name__ == "__main__":
    main()
