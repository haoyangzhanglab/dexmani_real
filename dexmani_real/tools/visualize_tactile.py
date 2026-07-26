#!/usr/bin/env python3
"""Tactile force visualisation for recorded episodes.

Reads /hand_tactile_force (T, 5, 120, 3) from an HDF5 episode and produces:

  plot force_timeline  — per-finger total force (L2-norm) over time
  plot force_heatmap   — force magnitude heatmap for a single frame (5 fingers × 120 sensors)
  plot force_3axis     — fx/fy/fz breakdown for one finger over time
  dump contact_frames  — list frames where per-finger force exceeds a threshold

Usage:
  python tools/visualize_tactile.py episodes/episode_20260725_120000.h5 force_timeline
  python tools/visualize_tactile.py episodes/episode_20260725_120000.h5 force_heatmap --frame 100
  python tools/visualize_tactile.py episodes/episode_20260725_120000.h5 force_3axis --finger thumb
  python tools/visualize_tactile.py episodes/episode_20260725_120000.h5 contact_frames --threshold 5.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]


def _load_tactile(path: str) -> tuple[np.ndarray, dict]:
    with h5py.File(path, "r") as f:
        if "hand_tactile_force" not in f:
            raise KeyError("hand_tactile_force dataset not found — episode recorded before tactile was saved")
        tactile = f["hand_tactile_force"][:]  # (T, 5, 120, 3)
        meta = dict(f["meta"].attrs)
    return tactile, meta


def _force_magnitude(tactile: np.ndarray) -> np.ndarray:
    """L2 norm across 3 force axes → (T, 5, 120)."""
    return np.linalg.norm(tactile, axis=-1)  # (T, 5, 120)


def cmd_timeline(args: argparse.Namespace) -> None:
    """Per-finger total force (sum over 120 sensors) over time."""
    import matplotlib.pyplot as plt

    tactile, meta = _load_tactile(args.episode)
    mag = _force_magnitude(tactile)  # (T, 5, 120)
    total = mag.sum(axis=-1)  # (T, 5)

    fps = float(meta.get("fps", meta.get("control_hz", 16.0)))
    t = np.arange(len(total)) / fps

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, name in enumerate(FINGER_NAMES):
        ax.plot(t, total[:, i], label=name, linewidth=0.8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Total force (raw sensor units)")
    ax.set_title(f"Per-finger Tactile Force — {Path(args.episode).name}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Mark frames with significant contact
    if args.mark_contacts:
        threshold = args.threshold if args.threshold > 0 else np.percentile(total[total > 0], 50) if np.any(total > 0) else 10.0
        for i in range(5):
            contact = total[:, i] > threshold
            changes = np.diff(contact.astype(int))
            starts = np.where(changes == 1)[0] + 1
            for s in starts:
                ax.axvline(t[s], color=f"C{i}", alpha=0.15, linewidth=0.5)

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Saved {args.output}")
    else:
        plt.show()


def cmd_heatmap(args: argparse.Namespace) -> None:
    """Force magnitude heatmap for one frame (5 fingers × 120 sensors)."""
    import matplotlib.pyplot as plt

    tactile, meta = _load_tactile(args.episode)
    mag = _force_magnitude(tactile)  # (T, 5, 120)

    frame = args.frame if args.frame is not None else len(mag) // 2
    if frame < 0 or frame >= len(mag):
        raise ValueError(f"frame {frame} out of range [0, {len(mag) - 1}]")
    frame_data = mag[frame]  # (5, 120)

    fig, axes = plt.subplots(5, 1, figsize=(14, 8), sharex=True)
    vmax = max(np.percentile(frame_data[frame_data > 0], 95) if np.any(frame_data > 0) else 1.0, 1.0)

    for i, (ax, name) in enumerate(zip(axes, FINGER_NAMES)):
        row = frame_data[i].reshape(1, -1)
        im = ax.imshow(row, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set_ylabel(name, rotation=0, labelpad=25, va="center")

    axes[-1].set_xlabel("Sensor index (0–119)")
    fig.suptitle(f"Tactile Force Heatmap — frame {frame} — {Path(args.episode).name}", fontsize=11)
    fig.colorbar(im, ax=axes, label="Force magnitude (raw units)", shrink=0.6)

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Saved {args.output}")
    else:
        plt.show()


def cmd_3axis(args: argparse.Namespace) -> None:
    """fx/fy/fz breakdown for one finger over time."""
    import matplotlib.pyplot as plt

    tactile, meta = _load_tactile(args.episode)
    finger = args.finger if args.finger in FINGER_NAMES else "thumb"
    idx = FINGER_NAMES.index(finger)
    fps = float(meta.get("fps", meta.get("control_hz", 16.0)))

    # Sum over sensors → (T, 3) per finger
    finger_total = tactile[:, idx, :, :].sum(axis=1)  # (T, 3)
    t = np.arange(len(finger_total)) / fps

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for ax, axis_name, color in zip(axes, ["fx", "fy", "fz"], ["#d62728", "#2ca02c", "#1f77b4"]):
        ax.plot(t, finger_total[:, ["fx", "fy", "fz"].index(axis_name)], color=color, linewidth=0.6)
        ax.set_ylabel(axis_name)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linewidth=0.5)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"3-Axis Force — {finger} — {Path(args.episode).name}", fontsize=11)

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Saved {args.output}")
    else:
        plt.show()


def cmd_contact_frames(args: argparse.Namespace) -> None:
    """List frames where per-finger force exceeds threshold."""
    tactile, meta = _load_tactile(args.episode)
    mag = _force_magnitude(tactile)  # (T, 5, 120)
    total = mag.sum(axis=-1)  # (T, 5)

    threshold = args.threshold if args.threshold > 0 else 5.0
    fps = float(meta.get("fps", meta.get("control_hz", 16.0)))

    print(f"{'frame':>6s}  {'time_s':>8s}  " + "  ".join(f"{n:>8s}" for n in FINGER_NAMES))
    print("-" * (6 + 1 + 8 + 1 + 5 * 9))

    in_contact = np.any(total > threshold, axis=1)
    transitions = np.diff(in_contact.astype(int))
    starts = np.where(transitions == 1)[0] + 1
    ends = np.where(transitions == -1)[0] + 1

    if len(starts) == 0:
        print("(no contact detected at threshold {:.1f})".format(threshold))
        return

    print(f"Contact events (threshold={threshold:.1f}):")
    for i, (s, e) in enumerate(zip(starts, ends[:len(starts)])):
        duration = (e - s) / fps
        fingers_in_contact = [FINGER_NAMES[j] for j in range(5) if np.any(total[s:e, j] > threshold)]
        print(f"  #{i+1}: frame {s:>5d}–{e:>5d}  ({duration:.1f}s)  fingers: {', '.join(fingers_in_contact)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise hand tactile force from recorded episodes")
    parser.add_argument("episode", help="Path to .h5 episode file")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("force_timeline", help="Per-finger total force over time")
    p1.add_argument("--threshold", type=float, default=0.0, help="Contact threshold (raw units; default=auto)")
    p1.add_argument("--mark-contacts", action="store_true", help="Mark contact start frames")
    p1.add_argument("-o", "--output", help="Save to file instead of showing")

    p2 = sub.add_parser("force_heatmap", help="Force magnitude heatmap for one frame")
    p2.add_argument("--frame", type=int, help="Frame index (default=midpoint)")
    p2.add_argument("-o", "--output", help="Save to file instead of showing")

    p3 = sub.add_parser("force_3axis", help="fx/fy/fz breakdown for one finger")
    p3.add_argument("--finger", choices=FINGER_NAMES, default="thumb")
    p3.add_argument("-o", "--output", help="Save to file instead of showing")

    p4 = sub.add_parser("contact_frames", help="List contact events")
    p4.add_argument("--threshold", type=float, default=5.0, help="Contact threshold (raw units)")

    args = parser.parse_args()
    {"force_timeline": cmd_timeline, "force_heatmap": cmd_heatmap, "force_3axis": cmd_3axis, "contact_frames": cmd_contact_frames}[args.command](args)


if __name__ == "__main__":
    main()
