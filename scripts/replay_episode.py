#!/usr/bin/env python3
"""在真机上按标称节拍回放 teleop Raw 关节目标，并保存回放结果。"""

from __future__ import annotations

import argparse
from pathlib import Path

from dexmani_real.config.real import load_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Physically replay a recorded trajectory on the xArm7 and XHand.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/replay_episode.py episodes/<task_name>/<episode_dir>
  python scripts/replay_episode.py episodes/<task_name>/<episode_dir> --config local.yaml
  python scripts/replay_episode.py episodes/<task_name>/<episode_dir> --output replay_results/my_replay/

Controls:
  Q     clean exit (save partial results)
  H     planned robot return-home, then wait for Q to exit (post-replay prompt)
  S     stop current motion
  ESC   emergency stop
        """,
    )
    parser.add_argument(
        "episode",
        type=str,
        help=(
            "Published Raw teleop episode directory (episodes/<task_name>/episode_*) "
            "with a technically normal capture and complete finite submission-attempt targets."
        ),
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Optional experiment YAML overrides"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Missing or empty directory for replay samples and control result.",
    )
    parser.add_argument(
        "--hand-start-duration",
        type=float,
        default=0.5,
        help=(
            "Seconds to ramp XHand to the first recorded measured posture before replay (default: 0.5). "
            "Only small start differences are allowed; 0 skips the ramp. Startup preserves the current hand posture."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from dexmani_real.replay.joint_replay import ReplayStatus
    from dexmani_real.replay.run import (
        DEFAULT_OUTPUT_DIR,
        run_replay,
    )
    from dexmani_real.replay.trajectory import load_trajectory

    trajectory = load_trajectory(args.episode)

    runtime = load_config(path=args.config)

    print(f"Trajectory: {trajectory.episode_path}")
    print(
        f"  Frames: {len(trajectory)}  Nominal FPS: {trajectory.fps:.1f}  "
        f"Nominal duration: {len(trajectory) / trajectory.fps:.1f}s"
    )
    print(f"  Task: {trajectory.task_label}")
    print(f"  Acc: {runtime.arm.max_joint_acceleration_deg_per_s2:.0f}°/s²")
    print(f"  Joint speed: {runtime.arm.max_joint_velocity_deg_per_s:.0f}°/s")

    if args.output is None:
        episode_name = Path(trajectory.episode_path).name
        output_dir = str(Path(DEFAULT_OUTPUT_DIR) / f"{episode_name}_replay")
    else:
        output_dir = args.output
    print(f"Output: {output_dir}")

    outcome = run_replay(
        trajectory,
        runtime,
        output_dir=output_dir,
        hand_start_duration_s=args.hand_start_duration,
    )

    if not outcome.successful:
        print(f"Replay stopped: {outcome.status.value}: {outcome.reason}")
        return 1
    if outcome.status is ReplayStatus.COMPLETED:
        print("Replay completed.")
    else:
        print("Replay exited cleanly; partial results were retained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
