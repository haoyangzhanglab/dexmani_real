"""Command-line surface for offline episode processing."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from dexmani_real.data_processing.contracts import (
    BridgePolicy,
    OutputProfile,
    ProcessingConfig,
    QualityPolicy,
    TemporalQualityConfig,
)
from dexmani_real.data_processing.pipeline import process_episode_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compact supported schema-v17/v18 Real episodes into one processed HDF5 per source."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help=(
            "Directory whose direct children are episodes for one task, e.g. "
            "episodes/pick_apple_messy; use episodes only for legacy flat recordings."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Published batch directory; defaults to "
            "episodes_processed/<input-root-name>."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=[profile.value for profile in OutputProfile],
        help="Select the required modalities and their shared hard-valid mask.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help=(
            "Optional audited include/task/range overrides; task outcome labels "
            "are rejected."
        ),
    )
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--min-full-windows", type=int, default=1)
    parser.add_argument("--max-camera-age-s", type=float, default=0.25)
    parser.add_argument(
        "--quality-policy",
        choices=[policy.value for policy in QualityPolicy],
        default=QualityPolicy.AUDIT.value,
        help="hard_only disables temporal detectors; audit reports findings; strict excludes only high-confidence findings",
    )
    parser.add_argument(
        "--bridge-policy",
        choices=[policy.value for policy in BridgePolicy],
        default=BridgePolicy.REJECT.value,
        help=(
            "reject (default) blocks compaction that creates an abrupt transition; "
            "audit permits it only for an explicitly reviewed batch"
        ),
    )
    parser.add_argument(
        "--abrupt-arm-step-rad",
        type=float,
        default=float(TemporalQualityConfig().abrupt_arm_step_rad),
        help="Bridge/suspect arm action threshold in radians.",
    )
    parser.add_argument(
        "--abrupt-hand-step-rad",
        type=float,
        default=TemporalQualityConfig().abrupt_hand_step_rad,
        help="Bridge/suspect hand action threshold in radians.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print decisions without creating episodes_processed output.",
    )
    parser.add_argument(
        "--compare-profiles",
        action="store_true",
        help="dry-run all profiles so modality-dependent retention can be compared",
    )
    return parser


def _config(args: argparse.Namespace, profile: OutputProfile) -> ProcessingConfig:
    return ProcessingConfig(
        profile=profile,
        horizon=args.horizon,
        min_full_windows=args.min_full_windows,
        max_camera_age_s=args.max_camera_age_s,
        temporal_quality=TemporalQualityConfig(
            policy=QualityPolicy(args.quality_policy),
            abrupt_arm_step_rad=args.abrupt_arm_step_rad,
            abrupt_hand_step_rad=args.abrupt_hand_step_rad,
        ),
        bridge_policy=BridgePolicy(args.bridge_policy),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    output_root = args.output_root or Path("episodes_processed") / args.input_root.name
    if args.compare_profiles:
        reports = {
            profile.value: process_episode_root(
                args.input_root,
                output_root,
                _config(args, profile),
                annotations_path=args.annotations,
                dry_run=True,
            )
            for profile in OutputProfile
        }
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return
    if args.profile is None:
        raise SystemExit("--profile is required unless --compare-profiles is used")
    report = process_episode_root(
        args.input_root,
        output_root,
        _config(args, OutputProfile(args.profile)),
        annotations_path=args.annotations,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
