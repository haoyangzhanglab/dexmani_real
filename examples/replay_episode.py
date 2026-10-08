#!/usr/bin/env python3
"""在真机上按标称节拍回放 teleop Raw 关节目标，并保存回放结果。

python examples/replay_episode.py episodes/test/episode_001
python examples/replay_episode.py episodes/test/episode_001 --config local.yaml --output replay_results/test

参数：episode 是已发布 Raw 目录；--config 覆盖配置，--output 指定新建或空结果目录。
"""

from __future__ import annotations

import logging
import argparse
import math
from pathlib import Path

import yaml

from dexmani_real.config.experiment import load_experiment_config


logger = logging.getLogger("dexmani_real.cli.replay_episode")


def _nonnegative_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError(f"must be finite and >= 0, got {value}")
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Physically replay a recorded trajectory on the xArm7 and XHand.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/replay_episode.py episodes/<task_name>/<episode_dir>
  python examples/replay_episode.py episodes/<task_name>/<episode_dir> --config local.yaml
  python examples/replay_episode.py episodes/<task_name>/<episode_dir> --output replay_results/my_replay/

Controls:
  Q     clean exit (save partial results)
  H     planned robot return-home, then wait for Q to exit (post-replay prompt)
  ESC   emergency stop
        """,
    )
    parser.add_argument(
        "episode",
        type=str,
        help=(
            "Published teleop Raw episode directory (episodes/<task_name>/episode_*) "
            "with finite joint targets and continued dispatch status for both devices."
        ),
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Optional experiment YAML overrides"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Missing or empty directory for replay data and consistency metrics.",
    )
    parser.add_argument(
        "--hand-start-duration",
        type=_nonnegative_float,
        default=0.5,
        help=(
            "Seconds to ramp XHand to the first target before replay (default: 0.5). "
            "Only small start differences are allowed; 0 disables preparation."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from dexmani_real.replay.replayer import ReplayStatus
    from dexmani_real.replay.session import (
        DEFAULT_OUTPUT_DIR,
        EpisodeReplayConfig,
        replay_episode,
    )
    from dexmani_real.replay.trajectory import load_trajectory, resolve_episode_path

    try:
        trajectory = load_trajectory(args.episode)
    except (OSError, ValueError) as exc:
        print(f"Error loading episode: {exc}")
        return 1

    try:
        runtime = load_experiment_config(yaml_path=args.config)
        if not runtime.policy.hand_enabled:
            raise ValueError("physical replay requires policy.hand_enabled=true")
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"Error resolving replay config: {exc}")
        return 1

    print(f"Trajectory: {trajectory.episode_path}")
    print(
        f"  Frames: {trajectory.num_frames}  Nominal FPS: {trajectory.fps:.1f}  "
        f"Nominal duration: {trajectory.num_frames / trajectory.fps:.1f}s"
    )
    print(f"  Task: {trajectory.task_label or '(none)'}")
    print(f"  Acc: {runtime.arm.max_joint_acceleration_deg_per_s2:.0f}°/s²")
    print(f"  Joint speed: {runtime.arm.max_joint_velocity_deg_per_s:.0f}°/s")

    if args.output is None:
        episode_name = resolve_episode_path(args.episode)[1]
        output_dir = str(Path(DEFAULT_OUTPUT_DIR) / f"{episode_name}_replay")
    else:
        output_dir = args.output
    print(f"Output: {output_dir}")

    try:
        outcome = replay_episode(
            trajectory,
            runtime,
            EpisodeReplayConfig(
                output_dir=output_dir,
                evaluate_consistency=True,
                hand_start_duration_s=args.hand_start_duration,
            ),
        )
    except Exception as exc:
        logger.error("physical replay failed", exc_info=True)
        print(f"\nPhysical replay failed: {exc}")
        return 1

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
