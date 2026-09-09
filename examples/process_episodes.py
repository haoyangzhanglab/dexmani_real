#!/usr/bin/env python3
"""Usage: ``python examples/process_episodes.py INPUT_ROOT [--profile PROFILE]``.

Offline CLI that audits and compacts one task's depth-to-color aligned raw-v26
episodes into processed HDF5 v15 files, one per source episode.

Directory mapping: ``episodes/<task>/episode_*`` (raw) is published to
``episodes_processed/<task>/episode_*.h5``.  Passing a single episode
directory resolves the task level from its parent directory, so the same
mapping holds for one episode.

Each profile is analyzed once per invocation. Unannotated episodes rejected by
the library are skipped with a warning; an explicit annotation whose ``include``
value is true remains batch-blocking. Explicit ``include: false`` skips its
episode before raw files are opened.

Connects to no hardware, opens no GUI, and writes only the resolved
``episodes_processed/`` output.  No JSON is printed to stdout; progress,
warnings, and a concise summary go to stderr. A concise
``process_log/invalid_frames_report.json`` is always published from staging; it
lists only episodes with genuinely invalid source frames. Exit codes: 0 at least
one episode was published or an audit completed; 1 nothing was published or the
publish failed; 2 usage or environment error (bad input root, existing output
root, unreadable annotations).

Writer publication reopens each output for bounded HDF5 structural sanity.
Pass ``--verify-output`` to also rescan every written payload and semantic
contract before the batch is published.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import logging
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml

from dexmani_real.config.experiment import resolve_experiment_config
from dexmani_real.dataset.contracts import (
    EpisodeAnnotation,
    OutputProfile,
    ProcessingConfig,
    QualityPolicy,
    TemporalQualityConfig,
    validate_processed_task_name,
)
from dexmani_real.dataset.processing import (
    discover_episode_dirs,
    load_annotations,
    process_episode_root,
    validate_annotation_task_name_override,
)
from dexmani_real.ipc.schema import SUPPORTED_POINT_CLOUD_COUNTS


def _route_library_logging_to_stderr() -> None:
    """Keep the terminal concise: silence sub-error library chatter on streams.

    ``get_logger`` attaches a stdout ``StreamHandler`` (plus a file handler) to
    each ``dexmani_real`` logger.  The CLI reports its own progress, skip
    warnings, and summary, so the library's per-episode rejection warnings and
    point-cloud preflight warnings (each with a traceback) would only duplicate
    that on the terminal.  Route any stdout stream to stderr and hold those
    stream handlers at ERROR so the terminal shows the CLI summary instead;
    full detail stays on the on-disk session log.
    """

    for obj in list(logging.Logger.manager.loggerDict.values()):
        if not isinstance(obj, logging.Logger):
            continue
        if obj.name != "dexmani_real" and not obj.name.startswith("dexmani_real."):
            continue
        for handler in obj.handlers:
            if isinstance(handler, logging.StreamHandler):
                if handler.stream is sys.stdout:
                    handler.setStream(sys.stderr)
                handler.setLevel(logging.ERROR)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and compact depth-to-color aligned raw-v26 Real episodes into one "
            "processed-v15 "
            "HDF5 per source; seriously broken episodes are skipped with a warning."
        )
    )
    parser.add_argument(
        "input_root",
        type=Path,
        help=(
            "One episode directory or a task directory whose direct children are "
            "episodes, e.g. episodes/pick_apple_messy/episode_*."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Published batch directory; defaults to episodes_processed/<task>, "
            "where <task> is the task directory name (or the parent directory "
            "name of a single episode directory)."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=[profile.value for profile in OutputProfile],
        default=OutputProfile.RGB_PC.value,
        help="Select the required modalities and their shared hard-valid mask.",
    )
    parser.add_argument(
        "--pointcloud-num-points",
        type=int,
        choices=sorted(SUPPORTED_POINT_CLOUD_COUNTS),
        default=1024,
        help=(
            "Fixed (N,6) point-cloud size for pointcloud/rgb_pc profiles "
            "(default: 1024)."
        ),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help=(
            "Optional audited include/task/range overrides; task outcome labels "
            "are rejected. Entries for episodes outside this input are ignored."
        ),
    )
    parser.add_argument(
        "--task-name",
        help=(
            "Write this one task_name into every processed episode, overriding raw "
            "task_label. Refuses conflicts with per-episode annotation task_name."
        ),
    )
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--min-full-windows", type=int, default=1)
    parser.add_argument(
        "--max-camera-age-s",
        type=float,
        default=None,
        help=(
            "Override the visual camera-age limit; it may not exceed "
            "runtime.policy.max_input_age_s."
        ),
    )
    parser.add_argument(
        "--max-observation-skew-s",
        type=float,
        default=None,
        help=(
            "Override runtime.policy.max_observation_skew_s for visual observations."
        ),
    )
    parser.add_argument(
        "--quality-policy",
        choices=[policy.value for policy in QualityPolicy],
        default=None,
        help="hard_only disables temporal detectors; audit reports findings (default audit).",
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
        default=float(TemporalQualityConfig().abrupt_hand_step_rad),
        help="Bridge/suspect hand action threshold in radians.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit only; print decisions without creating episodes_processed output.",
    )
    parser.add_argument(
        "--compare-profiles",
        action="store_true",
        help="Audit all profiles so modality-dependent retention can be compared.",
    )
    parser.add_argument(
        "--verify-output",
        action="store_true",
        help=(
            "Before publication, fully rescan each written processed HDF5 payload "
            "and semantic contract (slow; normal consumers/exporters remain strict)."
        ),
    )
    return parser


def _config(
    args: argparse.Namespace,
    profile: OutputProfile,
    policy: QualityPolicy,
    runtime: Any,
) -> ProcessingConfig:
    overrides: dict[str, object] = {
        "horizon": args.horizon,
        "min_full_windows": args.min_full_windows,
        "temporal_quality": TemporalQualityConfig(
            policy=policy,
            abrupt_arm_step_rad=args.abrupt_arm_step_rad,
            abrupt_hand_step_rad=args.abrupt_hand_step_rad,
        ),
        "pointcloud": dataclasses.replace(
            runtime.pointcloud, num_points=args.pointcloud_num_points
        ),
    }
    if args.max_camera_age_s is not None:
        overrides["max_camera_age_s"] = args.max_camera_age_s
    if args.max_observation_skew_s is not None:
        overrides["max_observation_skew_s"] = args.max_observation_skew_s
    return ProcessingConfig.from_runtime(
        runtime,
        profile=profile,
        **overrides,
    )


def _resolve_default_output_root(input_root: Path) -> Path:
    """Map both a task directory and one episode directory to episodes_processed/<task>."""

    if (input_root / "data.h5").is_file():
        return Path("episodes_processed") / input_root.parent.name
    return Path("episodes_processed") / input_root.name


def _validate_task_name(
    parser: argparse.ArgumentParser,
    task_name: str | None,
    *,
    label: str = "--task-name",
) -> str | None:
    """Reject task labels that cannot identify one processed dataset."""

    if task_name is None:
        return None
    try:
        return validate_processed_task_name(task_name)
    except (TypeError, ValueError) as exc:
        parser.error(f"{label}: {exc}")


def _write_annotations_yaml(
    annotations: dict[str, EpisodeAnnotation], directory: Path
) -> Path | None:
    if not annotations:
        return None
    payload = {
        "episodes": {
            name: dataclasses.asdict(annotation)
            for name, annotation in sorted(annotations.items())
        }
    }
    fd, name = tempfile.mkstemp(
        suffix=".yml", prefix="annotations-", dir=str(directory)
    )
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, allow_unicode=True, sort_keys=False)
    return Path(name)


def _print_report_summary(report: dict) -> None:
    """Print the one completed batch decision without reanalyzing sources."""

    source_count = int(report["source_episode_count"])
    accepted = int(report["accepted_source_episode_count"])
    rejected = int(report["rejected_source_episode_count"])
    excluded = sum(
        1
        for decision in report["episodes"]
        if decision["rejected_reason"] == "excluded by annotation"
    )
    skipped = rejected - excluded
    parts = [f"{accepted} accepted"]
    if skipped:
        parts.append(f"{skipped} skipped")
    if excluded:
        parts.append(f"{excluded} user-excluded")
    print(
        f"processing: {source_count} episode(s) -> {', '.join(parts)}", file=sys.stderr
    )
    for decision in report["episodes"]:
        reason = decision["rejected_reason"]
        if reason is not None and reason != "excluded by annotation":
            print(
                f"WARNING: skipping episode {decision['source_episode']}: {reason}",
                file=sys.stderr,
            )


def _filtered_annotations_path(
    annotations: dict[str, EpisodeAnnotation],
    *,
    original_path: Path | None,
    has_unknown_entries: bool,
    temporary_directory: Path | None,
) -> Path | None:
    """Keep CLI-only unknown-entry filtering out of the library boundary."""

    if not has_unknown_entries:
        return original_path
    assert temporary_directory is not None
    return _write_annotations_yaml(annotations, temporary_directory)


def main(argv: Sequence[str] | None = None) -> int:
    _route_library_logging_to_stderr()
    parser = _parser()
    args = parser.parse_args(argv)
    task_name = _validate_task_name(parser, args.task_name)
    selected_profile = OutputProfile(args.profile)
    policy = QualityPolicy(args.quality_policy or QualityPolicy.AUDIT.value)

    input_root = args.input_root
    if not input_root.is_dir():
        print(f"error: input root is not a directory: {input_root}", file=sys.stderr)
        return 2
    output_root = args.output_root or _resolve_default_output_root(input_root)
    expected_task_name = _validate_task_name(
        parser,
        output_root.name,
        label="--output-root basename",
    )
    assert expected_task_name is not None
    if task_name is not None and task_name != expected_task_name:
        parser.error("--task-name must match the --output-root basename")
    try:
        episodes = discover_episode_dirs(input_root)
        annotations = load_annotations(args.annotations)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    known = {episode.name for episode in episodes}
    unknown = sorted(set(annotations) - known)
    if unknown:
        print(
            f"note: ignoring annotations for {len(unknown)} episode(s) outside this input: "
            + ", ".join(unknown),
            file=sys.stderr,
        )
    user_annotations = {
        name: annotation for name, annotation in annotations.items() if name in known
    }
    try:
        validate_annotation_task_name_override(user_annotations, task_name)
    except (TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not args.dry_run and not args.compare_profiles and output_root.exists():
        print(
            f"error: output root already exists: {output_root}; "
            "remove it or pass a different --output-root",
            file=sys.stderr,
        )
        return 2

    try:
        runtime = resolve_experiment_config()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: failed to resolve runtime config: {exc}", file=sys.stderr)
        return 2

    temporary_context = (
        tempfile.TemporaryDirectory(prefix="process_episodes-")
        if unknown
        else contextlib.nullcontext(None)
    )
    with temporary_context as temporary_name:
        annotations_path = _filtered_annotations_path(
            user_annotations,
            original_path=args.annotations,
            has_unknown_entries=bool(unknown),
            temporary_directory=(
                Path(temporary_name) if temporary_name is not None else None
            ),
        )
        if args.compare_profiles:
            print("profile retention (selected/source frames):", file=sys.stderr)
            for profile in OutputProfile:
                try:
                    profile_config = _config(args, profile, policy, runtime)
                except (TypeError, ValueError) as exc:
                    print(f"error: invalid processing config: {exc}", file=sys.stderr)
                    return 2
                try:
                    report = process_episode_root(
                        input_root,
                        output_root,
                        profile_config,
                        annotations_path=annotations_path,
                        dry_run=True,
                        skip_rejected_unannotated=True,
                        task_name=task_name,
                        expected_task_name=expected_task_name,
                    )
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    print(f"error: profile audit failed: {exc}", file=sys.stderr)
                    return 1
                source = int(report["source_frame_count"])
                selected = int(report["selected_frame_count"])
                retention = 100.0 * selected / source if source else 0.0
                print(
                    f"  {profile.value:<11} {selected}/{source} ({retention:.1f}%)",
                    file=sys.stderr,
                )
            return 0

        try:
            config = _config(args, selected_profile, policy, runtime)
        except (TypeError, ValueError) as exc:
            print(f"error: invalid processing config: {exc}", file=sys.stderr)
            return 2
        try:
            report = process_episode_root(
                input_root,
                output_root,
                config,
                annotations_path=annotations_path,
                dry_run=args.dry_run,
                verify_output=args.verify_output,
                skip_rejected_unannotated=True,
                task_name=task_name,
                expected_task_name=expected_task_name,
            )
        except Exception as exc:
            print(f"error: batch processing failed: {exc}", file=sys.stderr)
            return 1

    _print_report_summary(report)
    if args.dry_run:
        return 0
    print(
        f"published {report['output_episode_count']} episode(s) -> {output_root}",
        file=sys.stderr,
    )
    invalid_episodes = report["invalid_frames_report"]["episodes"]
    if invalid_episodes:
        print(
            f"invalid-frame report: {len(invalid_episodes)} episode(s) -> "
            f"{output_root / 'process_log' / 'invalid_frames_report.json'}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
