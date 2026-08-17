"""Command-line surface for offline episode processing."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from dexmani_real.data_processing.contracts import OutputProfile, ProcessingConfig
from dexmani_real.data_processing.pipeline import process_episode_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean Real v16 episodes and write real-domain Sim-label HDF5 segments."
    )
    parser.add_argument("--input-root", type=Path, default=Path("episodes"))
    parser.add_argument("--output-root", type=Path, default=Path("episode_processed"))
    parser.add_argument("--profile", choices=[profile.value for profile in OutputProfile])
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--min-full-windows", type=int, default=1)
    parser.add_argument("--max-camera-age-s", type=float, default=0.25)
    parser.add_argument("--dry-run", action="store_true")
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
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.compare_profiles:
        reports = {
            profile.value: process_episode_root(
                args.input_root,
                args.output_root,
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
        args.output_root,
        _config(args, OutputProfile(args.profile)),
        annotations_path=args.annotations,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
