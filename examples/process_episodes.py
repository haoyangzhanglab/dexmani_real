"""Usage: ``python examples/process_episodes.py INPUT_ROOT [--profile PROFILE]``.

Offline CLI that audits and compacts one task's schema-v17/v18/v19 episodes
into processed HDF5 files, one per source episode.

Directory mapping: ``episodes/<task>/episode_*`` (raw) is published to
``episodes_processed/<task>/episode_*.h5``.  Passing a single episode
directory resolves the task level from its parent directory, so the same
mapping holds for one episode.

Processing is two-pass.  A per-episode audit pass (a dry-run analysis per
episode, progress on stderr via tqdm) decides accept or skip per episode.
Episodes with serious problems — unreadable files, schema validation failure,
hard row loss, or compaction that would create risky action jumps — are then
automatically skipped with a warning and the reason.  The surviving batch is
published in one transactional pass.  An episode whose annotation explicitly
sets ``include: true`` is never auto-skipped: its rejection stays blocking and
aborts the batch, so explicit operator intent is not silently overridden.

Connects to no hardware, opens no GUI, and writes only the resolved
``episodes_processed/`` output.  No JSON is printed to stdout; progress,
warnings, and a concise summary go to stderr.  With ``--write-report``, one JSON
per source episode is written under ``episodes_processed/<task>/process_log/``
after a successful publish.  Exit codes: 0 at least one episode was published
or an audit completed; 1 nothing was published or the publish failed; 2 usage
or environment error (bad input root, existing output root, unreadable
annotations).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import yaml
from tqdm import tqdm

from dexmani_real.data_processing.contracts import (
    BridgePolicy,
    EpisodeAnnotation,
    OutputProfile,
    ProcessingConfig,
    QualityPolicy,
    TemporalQualityConfig,
)
from dexmani_real.data_processing.pipeline import (
    PROCESSED_SCHEMA_NAME,
    PROCESSED_SCHEMA_VERSION,
    discover_episode_dirs,
    load_annotations,
    process_episode_root,
)

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
            "Audit and compact supported schema-v17/v18/v19 Real episodes into one "
            "processed HDF5 per source; seriously broken episodes are skipped with a warning."
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
        "--annotations",
        type=Path,
        help=(
            "Optional audited include/task/range overrides; task outcome labels "
            "are rejected. Entries for episodes outside this input are ignored."
        ),
    )
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--min-full-windows", type=int, default=1)
    parser.add_argument("--max-camera-age-s", type=float, default=0.25)
    parser.add_argument(
        "--quality-policy",
        choices=[policy.value for policy in QualityPolicy],
        default=None,
        help="hard_only disables temporal detectors; audit reports findings; strict excludes only high-confidence findings (default audit)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Shorthand for --quality-policy strict.",
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
        "--write-report",
        action="store_true",
        help=(
            "Write one JSON per source episode to <output_root>/process_log/"
            "episode_<name>.json after a successful publish (off by default; "
            "ignored in --dry-run and --compare-profiles)."
        ),
    )
    return parser


def _config(
    args: argparse.Namespace, profile: OutputProfile, policy: QualityPolicy
) -> ProcessingConfig:
    return ProcessingConfig(
        profile=profile,
        horizon=args.horizon,
        min_full_windows=args.min_full_windows,
        max_camera_age_s=args.max_camera_age_s,
        temporal_quality=TemporalQualityConfig(
            policy=policy,
            abrupt_arm_step_rad=args.abrupt_arm_step_rad,
            abrupt_hand_step_rad=args.abrupt_hand_step_rad,
        ),
        bridge_policy=BridgePolicy(args.bridge_policy),
    )


def _resolve_default_output_root(input_root: Path) -> Path:
    """Map both a task directory and one episode directory to episodes_processed/<task>."""

    if (input_root / "data.h5").is_file():
        return Path("episodes_processed") / input_root.parent.name
    return Path("episodes_processed") / input_root.name


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
    fd, name = tempfile.mkstemp(suffix=".yml", prefix="annotations-", dir=str(directory))
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, allow_unicode=True, sort_keys=False)
    return Path(name)


def _audit_episode(
    episode: Path,
    output_root: Path,
    config: ProcessingConfig,
    annotation: EpisodeAnnotation | None,
    tmp_dir: Path,
) -> tuple[dict, str | None] | tuple[None, str]:
    """Analyze one episode without writing; return (decision, error)."""

    annotations_path = None
    if annotation is not None:
        annotations_path = _write_annotations_yaml({episode.name: annotation}, tmp_dir)
    try:
        report = process_episode_root(
            episode,
            output_root,
            config,
            annotations_path=annotations_path,
            dry_run=True,
        )
        return report["episodes"][0], None
    except Exception as exc:
        # Isolate any per-episode analysis failure (missing dataset KeyError,
        # unexpected h5py/transform errors) so one bad episode never crashes
        # the whole audit; it is reported as a skipped "error" episode instead.
        return None, f"{type(exc).__name__}: {exc}"


def _audit_episodes(
    episodes: tuple[Path, ...],
    config: ProcessingConfig,
    user_annotations: dict[str, EpisodeAnnotation],
    output_root: Path,
    tmp_dir: Path,
) -> list[dict]:
    results: list[dict] = []
    bar = tqdm(
        episodes,
        desc=f"audit [{config.profile.value}]",
        unit="ep",
        file=sys.stderr,
        dynamic_ncols=True,
    )
    for episode in bar:
        annotation = user_annotations.get(episode.name)
        bar.set_postfix_str(episode.name)
        decision, error = _audit_episode(episode, output_root, config, annotation, tmp_dir)
        if annotation is not None and not annotation.include:
            status = "user-excluded"
        elif error is not None:
            status = "error"
            tqdm.write(
                f"WARNING: skipping episode {episode.name}: analysis failed: {error}",
                file=sys.stderr,
            )
        elif decision["accepted"]:
            status = "ok"
        else:
            status = "SKIP"
            tqdm.write(
                f"WARNING: skipping episode {episode.name}: {decision['rejected_reason']}",
                file=sys.stderr,
            )
        results.append(
            {
                "name": episode.name,
                "path": episode,
                "decision": decision,
                "error": error,
                "status": status,
            }
        )
        bar.set_postfix_str(status)
    bar.close()
    return results


def _print_audit_summary(results: list[dict]) -> None:
    """One concise per-batch line; full detail is in per-episode reports (--write-report)."""

    accepted = sum(1 for item in results if item["status"] == "ok")
    skipped = sum(1 for item in results if item["status"] in ("SKIP", "error"))
    user_excluded = sum(1 for item in results if item["status"] == "user-excluded")
    parts = [f"{accepted} accepted"]
    if skipped:
        parts.append(f"{skipped} skipped")
    if user_excluded:
        parts.append(f"{user_excluded} user-excluded")
    print(f"audit: {len(results)} episode(s) -> {', '.join(parts)}", file=sys.stderr)


def _write_process_logs(
    output_root: Path, results: list[dict], report: dict
) -> None:
    """Write one JSON per source episode under ``output_root/process_log/``.

    Runs after a successful publish, so the batch config, the per-episode
    decision (including the real skip reason for auto-skipped episodes), and
    the written output/validation entries are all available.  Rejected episodes
    keep their ``decision`` (or ``error``) and a null ``output``/``validation``.
    """

    log_dir = output_root / "process_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    outputs_by_episode = {
        item["source_episode"]: item for item in report.get("outputs", [])
    }
    validation_by_episode = {
        Path(item["path"]).stem: item for item in report.get("validation", [])
    }
    for item in results:
        entry = {
            "schema_name": PROCESSED_SCHEMA_NAME,
            "schema_version": PROCESSED_SCHEMA_VERSION,
            "task_name": output_root.name,
            "dry_run": False,
            "config": report["config"],
            "source_episode": item["name"],
            "status": item["status"],
            "decision": item["decision"],
            "error": item["error"],
            "output": outputs_by_episode.get(item["name"]),
            "validation": validation_by_episode.get(item["name"]),
        }
        with (log_dir / f"{item['name']}.json").open("w", encoding="utf-8") as stream:
            json.dump(entry, stream, ensure_ascii=False, indent=2)
            stream.flush()


def _merge_exclusions(
    user_annotations: dict[str, EpisodeAnnotation], results: list[dict]
) -> tuple[dict[str, EpisodeAnnotation], list[tuple[str, str]]]:
    """Add include=False for auto-skipped episodes; keep explicit includes blocking."""

    merged = dict(user_annotations)
    kept_blocking: list[tuple[str, str]] = []
    for item in results:
        if item["status"] not in ("SKIP", "error"):
            continue
        reason = (
            item["decision"]["rejected_reason"]
            if item["decision"] is not None
            else f"analysis failed: {item['error']}"
        )
        annotation = merged.get(item["name"])
        if annotation is not None and annotation.include:
            kept_blocking.append((item["name"], reason))
            continue
        merged[item["name"]] = EpisodeAnnotation(include=False)
    return merged, kept_blocking


def main(argv: Sequence[str] | None = None) -> int:
    _route_library_logging_to_stderr()
    parser = _parser()
    args = parser.parse_args(argv)
    if args.strict and args.quality_policy is not None:
        parser.error("--strict is mutually exclusive with --quality-policy")
    policy = (
        QualityPolicy.STRICT
        if args.strict
        else QualityPolicy(args.quality_policy or QualityPolicy.AUDIT.value)
    )

    input_root = args.input_root
    if not input_root.is_dir():
        print(f"error: input root is not a directory: {input_root}", file=sys.stderr)
        return 2
    output_root = args.output_root or _resolve_default_output_root(input_root)
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
    user_annotations = {name: annotation for name, annotation in annotations.items() if name in known}
    if not args.dry_run and not args.compare_profiles and output_root.exists():
        print(
            f"error: output root already exists: {output_root}; "
            "remove it or pass a different --output-root",
            file=sys.stderr,
        )
        return 2

    if args.compare_profiles:
        print("profile retention (selected/source frames):", file=sys.stderr)
        with tempfile.TemporaryDirectory(prefix="process_episodes-") as tmp_name:
            tmp_dir = Path(tmp_name)
            for profile in OutputProfile:
                profile_config = _config(args, profile, policy)
                results = _audit_episodes(
                    episodes, profile_config, user_annotations, output_root, tmp_dir
                )
                decided = [
                    item["decision"]
                    for item in results
                    if item["decision"] is not None and item["status"] != "user-excluded"
                ]
                source = sum(decision["source_frames"] for decision in decided)
                selected = sum(decision["selected_frames"] for decision in decided)
                retention = 100.0 * selected / source if source else 0.0
                print(
                    f"  {profile.value:<11} {selected}/{source} ({retention:.1f}%)",
                    file=sys.stderr,
                )
        return 0

    config = _config(args, OutputProfile(args.profile), policy)
    with tempfile.TemporaryDirectory(prefix="process_episodes-") as tmp_name:
        tmp_dir = Path(tmp_name)
        results = _audit_episodes(episodes, config, user_annotations, output_root, tmp_dir)
        _print_audit_summary(results)
        if args.dry_run:
            return 0
        accepted_count = sum(1 for item in results if item["status"] == "ok")
        if accepted_count == 0:
            print("error: no episodes accepted; nothing to publish", file=sys.stderr)
            return 1
        merged, kept_blocking = _merge_exclusions(user_annotations, results)
        for name, reason in kept_blocking:
            print(
                f"WARNING: episode {name} is explicitly included via annotations but fails "
                f"analysis ({reason}); the batch will abort unless this is resolved",
                file=sys.stderr,
            )
        merged_path = _write_annotations_yaml(merged, tmp_dir)
        print(f"publishing {accepted_count} episode(s) to {output_root} ...", file=sys.stderr)
        try:
            report = process_episode_root(
                input_root, output_root, config, annotations_path=merged_path
            )
        except Exception as exc:
            # A publish-only failure (validation, transform, or h5py error)
            # aborts cleanly; staging was already removed by the pipeline.
            print(f"error: batch publish failed: {exc}", file=sys.stderr)
            return 1
        if args.write_report:
            try:
                _write_process_logs(output_root, results, report)
            except OSError as exc:
                print(f"warning: failed to write process reports: {exc}", file=sys.stderr)
    print(f"published {report['output_episode_count']} episode(s) -> {output_root}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
