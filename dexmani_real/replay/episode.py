"""Offline episode inspection and certificate-gated hardware replay.

Live commands reach arm/hand workers only through SharedStorage and the
geometry-aware prepare/commit protocol. Dry-run is the default.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np

from dexmani_real import ASSET_DIR
from dexmani_real.config.runtime import ResolvedRuntimeConfig, resolve_runtime_config
from dexmani_real.planning import Pose, XArm7MotionPlanner, XArm7PlannerConfig
from dexmani_real.planning.preflight import PreflightCertificate, create_preflight_certificate
from dexmani_real.replay.data import TrajectoryData, load_trajectory
from dexmani_real.replay.data import resolve_episode_path as _resolve_episode_path
from dexmani_real.replay.preflight import (
    LiveReplayAuthorization,
    modeled_hand_actions,
    preflight_model_paths,
    replay_runtime_hash,
    require_explicit_hand_mode,
)
from dexmani_real.replay.runner import ReplayStatus, TrajectoryReplayer
from dexmani_real.replay.session import LiveReplayConfig, run_live_replay
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

DEFAULT_OUTPUT_DIR = "replay_results"
_PREFLIGHT_MAX_JOINT_STEP_RAD = 0.02


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


def _build_preflight_certificate(
    traj: "TrajectoryData",
    *,
    runtime: ResolvedRuntimeConfig,
    replay_runtime_sha256: str,
    no_hand: bool,
) -> PreflightCertificate:
    runtime_arm = runtime.arm
    runtime_hand = runtime.hand
    runtime_policy = runtime.policy
    workspace = np.array(
        [
            [runtime_policy.workspace.x_min, runtime_policy.workspace.x_max],
            [runtime_policy.workspace.y_min, runtime_policy.workspace.y_max],
            [runtime_policy.workspace.z_min, runtime_policy.workspace.z_max],
        ],
        dtype=np.float64,
    )
    model_dir = ASSET_DIR / "robots" / "xhand"
    planner = XArm7MotionPlanner(
        XArm7PlannerConfig(
            urdf_path=str(model_dir / "xarm7_xhand_collision.urdf"),
            srdf_path=str(model_dir / "xarm7_xhand.srdf"),
            base_pose_world=Pose(p=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0])),
            workspace_bounds=workspace,
        ),
        hand_dof=True,
        static_boxes=tuple(runtime.environment.static_boxes),
    )
    modeled_hand = modeled_hand_actions(
        traj,
        no_hand=no_hand,
        home_qpos_rad=np.deg2rad(np.asarray(runtime_hand.home_qpos_deg, dtype=np.float64)),
    )

    def table_check(arm_start: np.ndarray, arm_end: np.ndarray, hand_start: np.ndarray, hand_end: np.ndarray) -> bool:
        max_delta = max(float(np.max(np.abs(arm_end - arm_start))), float(np.max(np.abs(hand_end - hand_start))))
        steps = max(1, int(np.ceil(max_delta / _PREFLIGHT_MAX_JOINT_STEP_RAD)))
        for alpha in np.linspace(0.0, 1.0, steps + 1):
            arm_qpos = arm_start + alpha * (arm_end - arm_start)
            hand_qpos = hand_start + alpha * (hand_end - hand_start)
            planner.collision_model.set_hand_qpos(hand_qpos)
            lowest_surface_m = planner.collision_model.minimum_hand_frame_z(arm_qpos) - float(
                runtime_arm.hand_safety_margin_m
            )
            if lowest_surface_m < float(runtime_arm.table_z_surface_m):
                return False
        return True

    return create_preflight_certificate(
        source_episode=traj.episode_path,
        arm_actions=traj.action_arm_joint,
        hand_actions=modeled_hand,
        collision_model_paths=preflight_model_paths(),
        workspace_bounds_m=workspace,
        resolved_config_sha256=replay_runtime_sha256,
        transition_check=planner.collision_model.check_transition_collision_free,
        workspace_check=planner.is_workspace_segment_safe,
        table_check=table_check,
        hand_enabled=not no_hand and traj.has_hand,
        static_boxes=tuple(runtime.environment.static_boxes),
    )


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
    if args.write_certificate is not None:
        require_explicit_hand_mode(trajectory, no_hand=args.no_hand)
        certificate = _build_preflight_certificate(
            trajectory,
            runtime=selection.runtime,
            replay_runtime_sha256=selection.config_sha256,
            no_hand=args.no_hand,
        )
        path = certificate.write(args.write_certificate)
        print(f"Preflight certificate: {path}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a DexMani trajectory offline or run a certified live replay.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Offline inspection is the default; --live is the hardware boundary.
  python examples/real/replay_episode.py episodes/episode_20260729_213332

  # Validate only a bounded prefix
  python examples/real/replay_episode.py episodes/episode_20260729_213332 --max-frames 200

  # Validate trajectory without hardware (dry-run)
  python examples/real/replay_episode.py episodes/episode_20260729_213332 --dry-run

  # Arm-only (skip hand even if episode has hand data)
  python examples/real/replay_episode.py episodes/episode_20260729_213332 --no-hand

  # Generate a certificate, then separately authorize live replay
  python examples/real/replay_episode.py episodes/episode_20260729_213332 --source sent --write-certificate preflight.json
  python examples/real/replay_episode.py episodes/episode_20260729_213332 --source sent --live --certificate preflight.json

  # Live replay with a custom capture/metrics directory
  python examples/real/replay_episode.py episodes/episode_20260729_213332 --source sent --live \
    --certificate preflight.json --output results/my_replay/

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
        help="Speed factor bound into a certificate and used by live replay (1.0=recorded speed).",
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
        help="Authorize hardware replay after a matching preflight certificate is supplied.",
    )
    parser.add_argument(
        "--certificate", type=str, default=None, help="Existing preflight certificate required by --live."
    )
    parser.add_argument(
        "--write-certificate",
        type=str,
        default=None,
        help="In dry-run mode, run dense checks and atomically write a new certificate.",
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
    if args.live and args.certificate is None:
        parser.error("--live requires --certificate from a successful dry-run preflight")
    if not args.live and args.certificate is not None:
        parser.error("--certificate is only used with --live")
    if not args.live and args.output is not None:
        parser.error("--output is only used with --live; offline validation does not produce replay metrics")
    if args.live and args.write_certificate is not None:
        parser.error("--write-certificate is dry-run only")
    if args.live and args.source != "sent":
        parser.error("--live requires --source sent; raw policy candidates are never replayed on hardware")
    if args.write_certificate is not None and args.source != "sent":
        parser.error("--write-certificate requires --source sent so its binding can authorize live replay")
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
            require_exact_source=args.write_certificate is not None,
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
                authorization=LiveReplayAuthorization(
                    certificate_path=args.certificate,
                    replay_runtime_sha256=selection.config_sha256,
                ),
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
