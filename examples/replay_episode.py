#!/usr/bin/env python3
"""Usage: ``python examples/replay_episode.py EPISODE [--config FILE]``.

Physically replay one recorded episode on the real robot.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from dexmani_real.config.runtime import ResolvedRuntimeConfig, resolve_runtime_config
from dexmani_real.robot.episode_replay import (
    DEFAULT_OUTPUT_DIR,
    EpisodeReplayConfig,
    ReplayStatus,
    load_trajectory,
    replay_episode,
    resolve_episode_path,
)
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ReplayRuntimeSelection:
    """Resolved runtime config and physical replay provenance."""

    runtime: ResolvedRuntimeConfig
    acceleration_deg_s2: float
    joint_speed_deg_s: float
    config_sha256: str


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError(f"must be finite and > 0, got {value}")
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Physically replay a recorded trajectory on the xArm7 and XHand.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/replay_episode.py episodes/<task_name>/<episode_dir>
  python examples/replay_episode.py episodes/<task_name>/<episode_dir> --acc 900 --speed 120
  python examples/replay_episode.py episodes/<task_name>/<episode_dir> --output results/my_replay/

Controls:
  Q     clean exit (save partial results)
  H     return arm to home (post-replay prompt)
  ESC   emergency stop
        """,
    )
    parser.add_argument(
        "episode",
        type=str,
        help="Published episode directory: episodes/<task_name>/episode_*.",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Validated experiment YAML overrides."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory for captured replay data and consistency metrics.",
    )
    parser.add_argument(
        "--acc",
        type=_positive_float,
        default=None,
        help="Joint max acceleration (°/s²); must match the recording's resolved config.",
    )
    parser.add_argument(
        "--speed",
        type=_positive_float,
        default=None,
        help="Joint max speed (°/s); must match the recording's resolved config.",
    )
    return parser.parse_args(argv)


def _resolve_replay_runtime(args: argparse.Namespace) -> ReplayRuntimeSelection:
    runtime = resolve_runtime_config(
        yaml_path=args.config,
        cli_overrides={
            "arm.max_joint_acceleration_deg_per_s2": args.acc,
            "arm.max_joint_velocity_deg_per_s": args.speed,
        },
    )
    if not bool(runtime.policy.hand_enabled):
        raise ValueError("physical replay requires policy.hand_enabled=true")
    return ReplayRuntimeSelection(
        runtime=runtime,
        acceleration_deg_s2=float(runtime.arm.max_joint_acceleration_deg_per_s2),
        joint_speed_deg_s=float(runtime.arm.max_joint_velocity_deg_per_s),
        config_sha256=runtime.sha256,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        trajectory = load_trajectory(args.episode)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error loading episode: {exc}")
        return 1

    try:
        selection = _resolve_replay_runtime(args)
    except (FileNotFoundError, OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"Error resolving replay config: {exc}")
        return 1

    print(f"Trajectory: {trajectory.episode_path}")
    print(
        f"  Frames: {trajectory.num_frames}  FPS: {trajectory.fps:.1f}  "
        f"Duration: {trajectory.num_frames / trajectory.fps:.1f}s"
    )
    print(f"  Task: {trajectory.task_label or '(none)'}")
    print(f"  Hand action stream: {'yes' if trajectory.has_hand_actions else 'no'}")
    print(f"  EE data: {'yes' if trajectory.arm_ee is not None else 'no'}")
    print(f"  Acc: {selection.acceleration_deg_s2:.0f}°/s²")
    print(f"  Joint speed: {selection.joint_speed_deg_s:.0f}°/s")
    print(f"  Replay config: {selection.config_sha256[:12]}")

    evaluate_consistency = bool(np.all(np.isfinite(trajectory.arm_qpos)))
    if not evaluate_consistency:
        print("Warning: arm_qpos is invalid; consistency metrics will be skipped.")
    if args.output is None:
        _, episode_name = resolve_episode_path(args.episode)
        output_dir = str(Path(DEFAULT_OUTPUT_DIR) / f"{episode_name}_replay")
    else:
        output_dir = args.output
    print(f"Output: {output_dir}")

    try:
        outcome = replay_episode(
            trajectory,
            selection.runtime,
            EpisodeReplayConfig(
                output_dir=output_dir,
                evaluate_consistency=evaluate_consistency,
                config_sha256=selection.config_sha256,
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
