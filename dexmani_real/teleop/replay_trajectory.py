"""Replay a recorded trajectory through the Rerun visualizer.

Usage:
  python replay_trajectory.py                           # latest episode
  python replay_trajectory.py episode_000/trajectory.npz
  python replay_trajectory.py recordings/episode_001/trajectory.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from dexmani_real.teleop.quest_hand_visualizer import QuestHandVisualizer
from dexmani_real.teleop.trajectory_buffer import (
    DEFAULT_RECORD_DIR,
    TrajectoryBuffer,
    get_latest_episode,
    list_episodes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a recorded trajectory")
    parser.add_argument(
        "path", nargs="?",
        help="Path to trajectory .npz file (default: latest episode)",
    )
    parser.add_argument(
        "--speed", "-s", type=float, default=1.0,
        help="Playback speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--list", "-l", action="store_true",
        help="List available episodes and exit",
    )
    args = parser.parse_args()

    if args.list:
        episodes = list_episodes()
        if not episodes:
            print("No episodes found.")
            return
        for ep in episodes:
            buf = TrajectoryBuffer.load(ep)
            print(f"  {ep.parent.name}  —  {len(buf)} frames, {buf.duration:.1f}s")
        return

    traj_path = _resolve_path(args.path)
    if traj_path is None:
        print("No trajectory file found.")
        sys.exit(1)

    buffer = TrajectoryBuffer.load(traj_path)
    if not buffer:
        print(f"Empty trajectory: {traj_path}")
        sys.exit(1)

    print(f"Loaded {len(buffer)} frames ({buffer.duration:.1f}s) from {traj_path.name}")
    print("Press Ctrl+C to stop.")

    vis = QuestHandVisualizer(show_axes=True)

    timestamps = buffer.get_array("timestamp")
    t0 = time.time()
    start_ts = float(timestamps[0])

    try:
        idx = 0
        while idx < len(buffer):
            elapsed = (time.time() - t0) * args.speed
            frame_ts = float(timestamps[idx]) - start_ts

            if frame_ts <= elapsed:
                frame = buffer.get_frame(idx)
                vis.log_frame(frame, path="replay/right_hand")
                if "eef_pos" in frame and "eef_quat_wxyz" in frame:
                    vis.log_axes(
                        "replay_ee_target/right_hand",
                        frame["eef_pos"],
                        frame["eef_quat_wxyz"],
                    )
                idx += 1
            else:
                time.sleep(0.002)

    except KeyboardInterrupt:
        print(f"\nStopped at frame {idx}/{len(buffer)}")

    print("Done.")


def _resolve_path(path: str | None) -> Path | None:
    if path is not None:
        p = Path(path)
        if p.is_file():
            return p
        # Try relative to DEFAULT_RECORD_DIR
        p = DEFAULT_RECORD_DIR / path
        if p.is_file():
            return p
        print(f"File not found: {path}")
        return None
    return get_latest_episode()


if __name__ == "__main__":
    main()
