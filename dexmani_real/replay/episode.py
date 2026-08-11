"""Offline episode inspection and fail-closed hardware replay.

Live commands reach arm/hand workers only through SharedStorage and the
geometry-aware prepare/commit protocol. Dry-run is the default.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np

from dexmani_real.config.runtime import ResolvedRuntimeConfig, resolve_runtime_config
from dexmani_real.replay.data import TrajectoryData, load_trajectory
from dexmani_real.replay.data import resolve_episode_path as _resolve_episode_path
from dexmani_real.replay.preflight import replay_runtime_hash
from dexmani_real.replay.runner import ReplayStatus, TrajectoryReplayer
from dexmani_real.replay.session import LiveReplayConfig, run_live_replay
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

DEFAULT_OUTPUT_DIR = "replay_results"


def _positive_finite_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be finite and > 0, got {value!r}")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {value!r}")
    return parsed


@dataclass(frozen=True)
class ReplayRuntimeSelection:
    runtime: ResolvedRuntimeConfig
    acceleration_deg_s2: float
    joint_speed_deg_s: float
    acceleration_source: str
    config_sha256: str


def _resolve_replay_runtime(args: argparse.Namespace, trajectory: TrajectoryData) -> ReplayRuntimeSelection:
    base_runtime = resolve_runtime_config(yaml_path=args.config)
    if args.acc is not None:
        acceleration = float(args.acc)
        acceleration_source = " (--acc override)"
    elif trajectory.joint_max_acc is not None:
        acceleration = float(trajectory.joint_max_acc)
        acceleration_source = " (from episode metadata)"
    else:
        acceleration = float(base_runtime.arm.max_joint_acceleration_deg_per_s2)
        acceleration_source = " (from runtime config)"

    joint_speed = float(
        args.joint_speed
        if args.joint_speed is not None
        else (
            trajectory.joint_max_speed
            if trajectory.joint_max_speed is not None
            else base_runtime.arm.max_joint_velocity_deg_per_s
        )
    )
    runtime = resolve_runtime_config(
        yaml_path=args.config,
        cli_overrides={
            "arm.ip": args.arm_ip,
            "arm.max_joint_acceleration_deg_per_s2": acceleration,
            "arm.max_joint_velocity_deg_per_s": joint_speed,
            "policy.hand_enabled": False if args.no_hand else None,
        },
    )
    if not args.no_hand and not bool(runtime.policy.hand_enabled):
        raise ValueError("policy.hand_enabled=false requires explicit --no-hand confirmation")
    config_sha256 = replay_runtime_hash(
        runtime.canonical_yaml,
        source=args.source,
        speed_factor=args.speed,
        no_hand=args.no_hand,
        jerk_management="unmanaged",
    )
    return ReplayRuntimeSelection(runtime, acceleration, joint_speed, acceleration_source, config_sha256)


def _run_offline(
    args: argparse.Namespace,
    trajectory: TrajectoryData,
    selection: ReplayRuntimeSelection,
) -> None:
    replayer = TrajectoryReplayer(
        trajectory,
        None,
        speed=args.speed,
        dry_run=True,
        no_hand=args.no_hand,
        max_frames=args.max_frames,
        runtime=selection.runtime,
    )
    replayer.validate_offline()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a DexMani trajectory offline or run a fail-closed live replay.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Offline inspection is the default; --live is the hardware boundary.
  python examples/replay_episode.py episodes/episode_20260729_213332

  # Validate only a bounded prefix
  python examples/replay_episode.py episodes/episode_20260729_213332 --max-frames 200

  # Validate trajectory without hardware (dry-run)
  python examples/replay_episode.py episodes/episode_20260729_213332 --dry-run

  # Arm-only (skip hand even if episode has hand data)
  python examples/replay_episode.py episodes/episode_20260729_213332 --no-hand

  # Live replay re-runs dense geometry and provenance checks before workers start
  python examples/replay_episode.py episodes/episode_20260729_213332 --source sent --live

  # Live replay with a custom capture/metrics directory
  python examples/replay_episode.py episodes/episode_20260729_213332 --source sent --live \
    --output results/my_replay/

Live replay controls:
  Q     clean exit (save partial results)
  H     return arm to home (post-replay prompt)
  ESC   emergency stop
        """,
    )
    parser.add_argument(
        "episode",
        nargs="?",
        type=str,
        help="Episode directory, data.h5, or legacy flat HDF5 file.",
    )
    parser.add_argument(
        "--h5",
        dest="episode_h5",
        metavar="PATH",
        help="Legacy alias for the episode path; use the positional path in new commands.",
    )
    parser.add_argument(
        "--speed",
        type=_positive_finite_float,
        default=1.0,
        help="Speed factor used by live replay (1.0=recorded speed).",
    )
    parser.add_argument(
        "--max-frames",
        type=_positive_int,
        default=None,
        help="Maximum number of frames to replay (default: all).",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Offline validation only (default; retained for explicit scripts).",
    )
    mode_group.add_argument(
        "--live",
        action="store_true",
        help="Run dense preflight checks, then cross the hardware boundary.",
    )
    parser.add_argument("--config", type=str, default=None, help="Validated experiment YAML overrides.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Live-only directory for captured replay data and optional consistency metrics.",
    )
    parser.add_argument(
        "--no-hand",
        action="store_true",
        help="Skip hand commands; live use requires the physical hand to be absent or secured at its configured home.",
    )
    parser.add_argument(
        "--arm-ip",
        type=str,
        default=None,
        help="XArm controller IP (experiment YAML/defaults when omitted).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="cmd",
        choices=["cmd", "sent"],
        help="Arm stream: policy target (cmd) or exact submitted target (sent).",
    )
    parser.add_argument(
        "--joint-speed",
        type=_positive_finite_float,
        default=None,
        metavar="DEG_S",
        help="Mode-6 joint speed; defaults to recorded provenance, then runtime config.",
    )
    parser.add_argument(
        "--acc",
        type=_positive_finite_float,
        default=None,
        metavar="DEG_S2",
        help="Joint max acceleration in °/s²; CLI overrides episode metadata and runtime config.",
    )
    args = parser.parse_args(argv)
    args.dry_run = not args.live

    if args.episode is not None and args.episode_h5 is not None:
        parser.error("provide the episode path either positionally or with --h5, not both")
    args.episode = args.episode if args.episode is not None else args.episode_h5
    if args.episode is None:
        parser.error("an episode path is required (positional path or --h5 PATH)")
    if not args.live and args.output is not None:
        parser.error("--output is only used with --live; offline validation does not produce replay metrics")
    if args.live and args.source != "sent":
        parser.error("--live requires --source sent; raw policy candidates are never replayed on hardware")
    if args.live and args.speed > 1.0:
        print("Warning: speed-up increases commanded joint velocities; live safety gates may reject the replay")

    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        traj = load_trajectory(
            args.episode,
            max_frames=args.max_frames,
            source=args.source,
            require_live_validity=args.live,
            require_exact_source=args.live,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error loading episode: {exc}")
        return 1

    if traj.num_frames == 0:
        print("Error: trajectory has 0 frames")
        return 1

    try:
        selection = _resolve_replay_runtime(args, traj)
    except (FileNotFoundError, TypeError, ValueError, OSError) as exc:
        print(f"Error resolving replay config: {exc}")
        return 1

    print(f"Trajectory: {traj.episode_path}")
    print(f"  Frames: {traj.num_frames}  FPS: {traj.fps:.1f}  Duration: {traj.num_frames/traj.fps:.1f}s")
    print(f"  Task: {traj.task_label or '(none)'}")
    hand_metadata = "yes" if traj.hand_available is True else "no" if traj.hand_available is False else "unknown"
    print(f"  Hand available: {hand_metadata}  action dataset: {'yes' if traj.has_hand_actions else 'no'}")
    print(f"  EE data: {'yes' if traj.arm_ee is not None else 'no'}")
    print(f"  Acc: {selection.acceleration_deg_s2:.0f}°/s²{selection.acceleration_source}")
    print(f"  Joint speed: {selection.joint_speed_deg_s:.0f}°/s")
    print(f"  Replay config: {selection.config_sha256[:12]}")

    if args.dry_run:
        try:
            _run_offline(args, traj, selection)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"Error: dry-run validation failed: {exc}")
            return 1
        print("Done.")
        return 0

    eval_available = bool(np.all(np.isfinite(traj.arm_qpos)))
    if not eval_available:
        print("Warning: arm_qpos missing/invalid in episode, cannot evaluate consistency.")
    if args.output is not None:
        output_dir = args.output
    else:
        _, episode_name = _resolve_episode_path(args.episode)
        output_dir = os.path.join(DEFAULT_OUTPUT_DIR, f"{episode_name}_replay")
    print(f"Output: {output_dir}")

    try:
        outcome = run_live_replay(
            traj,
            selection.runtime,
            LiveReplayConfig(
                speed=args.speed,
                no_hand=args.no_hand,
                max_frames=args.max_frames,
                output_dir=output_dir,
                evaluate_consistency=bool(eval_available),
            ),
        )
    except Exception as exc:
        logger.error("live replay failed", exc_info=True)
        print(f"\nLive replay failed: {exc}")
        return 1

    if not outcome.successful:
        print(f"Live replay stopped: {outcome.status.value}: {outcome.reason}")
        return 1
    if outcome.status is ReplayStatus.COMPLETED:
        print("Replay completed.")
    else:
        print("Replay exited cleanly; partial results were retained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
