#!/usr/bin/env python3
"""Filter training frames from DexMani HDF5 episodes by quality flags.

Removes frames matching any enabled criteria:
  - Held frames (flag_held == True): forward-filled grid slots,
    no fresh VR/IK result.
  - IK-fail frames (flag_frame_status == 2): IK solver failed.
  - Safety-reject frames (flag_frame_status == 3): action rejected
    by pre-send gate.
  - High tracking error (tracking_error > threshold, rad): arm
    lagging behind commanded position.

Usage:
    # Stats only (no output file)
    python tools/filter_training_frames.py episodes/episode_20260803_120000/ --stats

    # Drop held frames, write filtered copy
    python tools/filter_training_frames.py episodes/episode_20260803_120000/ \\
        --drop-held --output filtered_episodes/

    # Drop held + high tracking error frames
    python tools/filter_training_frames.py episodes/episode_20260803_120000/ \\
        --drop-held --max-tracking-error 0.35 --output filtered_episodes/

    # Batch process a directory of episodes
    python tools/filter_training_frames.py episodes/ --drop-held --output filtered_episodes/
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

# ── Frame quality codes (schema v11, matches policy/vr_teleop_policy.py) ──
_FRAME_OK = 0
_FRAME_HELD = 1
_FRAME_IK_FAIL = 2
_FRAME_SAFETY_REJECT = 3
_FRAME_RETARGET_FAIL = 4


def _is_episode_dir(path: Path) -> bool:
    """Check if path is an episode directory (contains data.h5)."""
    return path.is_dir() and (path / "data.h5").exists()


def _is_legacy_episode(path: Path) -> bool:
    """Check if path is a legacy single-file HDF5 episode."""
    return path.is_file() and path.suffix == ".h5" and not path.name.startswith("depth")


def _find_episodes(data_dir: Path) -> list[Path]:
    """Find all episode directories/legacy files under data_dir."""
    episodes: list[Path] = []
    for entry in sorted(data_dir.iterdir()):
        if _is_episode_dir(entry):
            episodes.append(entry)
        elif _is_legacy_episode(entry):
            episodes.append(entry)
    return episodes


def _build_mask(h5f: h5py.File, num_frames: int, args: argparse.Namespace) -> np.ndarray:
    """Build a boolean mask of frames to KEEP.

    Returns a 1-D boolean array where True = keep this frame.
    """
    mask = np.ones(num_frames, dtype=bool)

    # ── Held frames ──
    if args.drop_held:
        if "flag_held" in h5f:
            held = np.asarray(h5f["flag_held"][:num_frames], dtype=bool)
            n_held = int(np.sum(held))
            mask &= ~held
            print(f"  held frames: {n_held} (dropped)")
        else:
            print("  WARNING: /flag_held dataset not found — skipping held filter")

    # ── Frame status codes ──
    if args.drop_ik_fail:
        if "flag_frame_status" in h5f:
            status = np.asarray(h5f["flag_frame_status"][:num_frames], dtype=np.int32)
            n_ik_fail = int(np.sum(status == _FRAME_IK_FAIL))
            mask &= (status != _FRAME_IK_FAIL)
            print(f"  IK-fail frames: {n_ik_fail} (dropped)")
        else:
            print("  WARNING: /flag_frame_status dataset not found — skipping IK-fail filter")

    if args.drop_safety_reject:
        if "flag_frame_status" in h5f:
            status = np.asarray(h5f["flag_frame_status"][:num_frames], dtype=np.int32)
            n_rej = int(np.sum(status == _FRAME_SAFETY_REJECT))
            mask &= (status != _FRAME_SAFETY_REJECT)
            print(f"  safety-reject frames: {n_rej} (dropped)")
        else:
            print("  WARNING: /flag_frame_status dataset not found — skipping safety-reject filter")

    if args.drop_retarget_fail:
        if "flag_frame_status" in h5f:
            status = np.asarray(h5f["flag_frame_status"][:num_frames], dtype=np.int32)
            n_rf = int(np.sum(status == _FRAME_RETARGET_FAIL))
            mask &= (status != _FRAME_RETARGET_FAIL)
            print(f"  retarget-fail frames: {n_rf} (dropped)")
        else:
            print("  WARNING: /flag_frame_status dataset not found — skipping retarget-fail filter")

    # ── Tracking error ──
    if args.max_tracking_error is not None:
        threshold = float(args.max_tracking_error)
        if "tracking_error" in h5f:
            te = np.asarray(h5f["tracking_error"][:num_frames], dtype=np.float64)
            valid = np.isfinite(te)
            n_high = int(np.sum(valid & (te > threshold)))
            mask &= ~(valid & (te > threshold))
            print(f"  tracking_error > {threshold:.3f} rad: {n_high} (dropped)")
        else:
            print("  WARNING: /tracking_error dataset not found — skipping tracking-error filter")

    return mask


def _filter_episode(
    input_path: Path, output_dir: Path | None, args: argparse.Namespace,
) -> tuple[int, int]:
    """Filter one episode. Returns (total_frames, kept_frames)."""
    print(f"\n── {input_path.name} ──")

    h5_path = input_path / "data.h5" if input_path.is_dir() else input_path

    with h5py.File(h5_path, "r") as h5f:
        # Get frame count from arm_qpos dataset.
        if "arm_qpos" not in h5f:
            print("  ERROR: /arm_qpos dataset not found — skipping")
            return 0, 0
        num_frames = h5f["arm_qpos"].shape[0]
        print(f"  total frames: {num_frames}")

        mask = _build_mask(h5f, num_frames, args)
        kept = int(np.sum(mask))
        dropped = num_frames - kept

        if kept == 0:
            print(f"  ALL frames dropped ({dropped}) — skipping output")
            return num_frames, 0

        # Determine which datasets to copy (all 1-D time-series datasets).
        time_series_keys: list[str] = []
        for key in sorted(h5f.keys()):
            if key == "meta":
                continue
            ds = h5f[key]
            if isinstance(ds, h5py.Dataset) and ds.ndim >= 1 and ds.shape[0] == num_frames:
                time_series_keys.append(key)

        if output_dir is not None:
            out_name = input_path.name
            out_path = output_dir / out_name
            out_h5_path = out_path / "data.h5" if input_path.is_dir() else out_path.with_suffix(".h5")

            out_h5_path.parent.mkdir(parents=True, exist_ok=True)

            with h5py.File(out_h5_path, "w") as out_f:
                # Copy /meta group if present.
                if "meta" in h5f:
                    h5f.copy("meta", out_f)

                # Copy & filter time-series datasets.
                for key in time_series_keys:
                    data = np.asarray(h5f[key][:num_frames])
                    out_f.create_dataset(key, data=data[mask], compression="gzip", compression_opts=4)

                # Update num_frames in meta.
                if "meta" in out_f:
                    out_f["meta"].attrs["num_frames"] = kept
                    # Add filtering metadata.
                    out_f["meta"].attrs["filter_original_frames"] = num_frames
                    out_f["meta"].attrs["filter_kept_frames"] = kept
                    out_f["meta"].attrs["filter_drop_held"] = args.drop_held
                    out_f["meta"].attrs["filter_max_tracking_error"] = (
                        args.max_tracking_error if args.max_tracking_error is not None else "off"
                    )

            # Copy sidecar files (depth.h5, rgb.mp4) if present.
            if input_path.is_dir():
                for sidecar in ("depth.h5", "rgb.mp4"):
                    src = input_path / sidecar
                    if src.exists():
                        shutil.copy2(src, out_path / sidecar)

            print(f"  → {out_h5_path}  ({kept} frames)")

    return num_frames, kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter training frames from DexMani HDF5 episodes")
    parser.add_argument(
        "input", type=str,
        help="Episode directory, legacy .h5 file, or directory of episodes",
    )
    parser.add_argument("--output", type=str, default=None, help="Output directory for filtered episodes")
    parser.add_argument("--stats", action="store_true", help="Print statistics only (no output)")
    parser.add_argument(
        "--drop-held", action="store_true", default=True,
        help="Drop held frames (flag_held == True). Default: ON.",
    )
    parser.add_argument(
        "--keep-held", action="store_true",
        help="Keep held frames (overrides default drop-held behaviour)",
    )
    parser.add_argument("--drop-ik-fail", action="store_true", help="Drop IK-fail frames")
    parser.add_argument("--drop-safety-reject", action="store_true", help="Drop safety-reject frames")
    parser.add_argument("--drop-retarget-fail", action="store_true", help="Drop retarget-fail frames")
    parser.add_argument(
        "--max-tracking-error", type=float, default=None,
        help="Drop frames with tracking_error > THRESHOLD (rad). e.g. 0.35",
    )
    args = parser.parse_args()

    # --keep-held overrides default --drop-held
    if args.keep_held:
        args.drop_held = False

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        sys.exit(1)

    output_dir = None if args.output is None else Path(args.output)

    # Find episodes.
    if _is_episode_dir(input_path) or _is_legacy_episode(input_path):
        episodes = [input_path]
    elif input_path.is_dir():
        episodes = _find_episodes(input_path)
        if not episodes:
            print(f"No episodes found in {input_path}")
            sys.exit(1)
        print(f"Found {len(episodes)} episode(s) in {input_path}")
    else:
        print(f"ERROR: {input_path} is not an episode or directory")
        sys.exit(1)

    total_frames = 0
    total_kept = 0

    for ep_path in episodes:
        n_total, n_kept = _filter_episode(ep_path, output_dir, args)
        total_frames += n_total
        total_kept += n_kept

    print(f"\n{'='*50}")
    print(f"Total: {total_frames} frames → {total_kept} kept ({total_frames - total_kept} dropped)")
    if total_frames > 0:
        print(f"Keep rate: {100.0 * total_kept / total_frames:.1f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
