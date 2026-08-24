#!/usr/bin/env python3
"""Physically replay one recorded episode on xArm7 and optional XHand.

This is a hardware-affecting entry point and writes replay evaluation results.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import yaml

from dexmani_real.config.runtime import ResolvedRuntimeConfig, resolve_runtime_config
from dexmani_real.replay.controller import ReplayStatus
from dexmani_real.replay.session import (
    DEFAULT_OUTPUT_DIR,
    EpisodeReplayConfig,
    replay_episode,
)
from dexmani_real.replay.trajectory import (
    load_processed_trajectory,
    load_trajectory,
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
  python examples/replay_episode.py episodes/<task_name>/<episode_dir> --output replay_results/my_replay/
  python examples/replay_episode.py episodes_processed/<task>/episode_<timestamp>.h5 --processed

Controls:
  Q     clean exit (save partial results)
  H     return arm to home (post-replay prompt)
  ESC   emergency stop
        """,
    )
    parser.add_argument(
        "episode",
        type=str,
        help=(
            "Published episode directory (episodes/<task_name>/episode_*) or, with "
            "--processed, a processed HDF5 file (episodes_processed/<task>/episode_*.h5)."
        ),
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
    parser.add_argument(
        "--processed",
        action="store_true",
        help=(
            "Replay a processed HDF5 artifact (episodes_processed/<task>/episode_*.h5) "
            "instead of a raw episode directory. The processed file lacks recording-time "
            "model (URDF/SRDF) provenance; the full workspace and collision preflight "
            "still runs against current models."
        ),
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
        if args.processed:
            trajectory = load_processed_trajectory(args.episode)
        else:
            trajectory = load_trajectory(args.episode)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error loading episode: {exc}")
        if not args.processed and Path(args.episode).is_file():
            print(
                "Hint: this path is a file, not a raw episode directory; use "
                "--processed to replay a processed HDF5."
            )
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

    if args.processed:
        print(
            "Warning: replaying a processed HDF5. Model (URDF/SRDF) provenance is "
            "unavailable; workspace and collision preflight run against current models."
        )

    evaluate_consistency = bool(np.all(np.isfinite(trajectory.arm_qpos)))
    if not evaluate_consistency:
        print("Warning: arm_qpos is invalid; consistency metrics will be skipped.")
    if args.output is None:
        if args.processed:
            episode_name = Path(args.episode).stem
        else:
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
