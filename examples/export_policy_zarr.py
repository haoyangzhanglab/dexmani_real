#!/usr/bin/env python3
"""Export raw episodes directly to policy Zarr; --dry-run executes the same transforms."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

from tqdm import tqdm

from dexmani_real.config.experiment import resolve_experiment_config
from dexmani_real.dataset.contracts import ProcessingConfig, validate_task_name
from dexmani_real.dataset.export import (
    PolicyZarrExportConfig,
    export_raw_to_zarr,
)
from dexmani_real.utils.atomic_io import target_is_occupied


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transform raw Real episodes to dexmani_policy Zarr."
    )
    parser.add_argument(
        "input_root",
        type=Path,
        metavar="episodes/<task_name>",
        help=(
            "One raw task or episode directory. Exports to "
            "datasets/<task_name>.zarr by default (see --output); existing "
            "output paths are refused."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Alternative Zarr output path for a NEW generation of this task "
            "(default: datasets/<task_name>.zarr). The task identity always "
            "comes from the input directory. Existing outputs are refused, "
            "and the resolved target (symlinks followed) must not fall "
            "inside the protected sources: episodes/, "
            "rollouts/, the input root, or an existing .zarr store."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Experiment YAML; otherwise use the default experiment configuration.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Whole-episode include/task annotations; unknown episodes are rejected.",
    )
    parser.add_argument(
        "--task-name",
        help="Explicit task label override; must match the input task directory.",
    )
    parser.add_argument(
        "--pointcloud-num-points",
        type=int,
        default=None,
    )
    parser.add_argument("--chunk-frames", type=int, default=100)
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Compute and validate raw inputs without creating a Zarr output. "
            "Use this before a large export."
        ),
    )
    return parser


# Source roots whose contents must never receive derived export output. They
# are anchored to the REPOSITORY root, not the caller's working directory, so
# the guard holds no matter where the tool is invoked from.
# Historical data directories remain write-protected even though their loader is gone.
_PROTECTED_SOURCE_ROOTS = ("episodes", "episodes_processed", "rollouts")


def _resolve_output_path(
    output: Path | None,
    default_path: Path,
    input_root: Path,
) -> Path:
    """Resolve the export target and keep it outside every protected source.

    Symlinks are followed, so a link that escapes into raw/rollout
    data is refused by its resolved location. Overwriting any existing output
    is refused by the occupied-target check before either mode runs. The
    returned path is the same one the checks resolved (``~`` expanded), so
    safety, the occupied-target refusal, and the real export all agree.
    """
    candidate = default_path if output is None else output
    resolved = candidate.expanduser().resolve(strict=False)
    protected = [(_REPO_ROOT / name).resolve(strict=False) for name in _PROTECTED_SOURCE_ROOTS]
    protected.append(input_root.expanduser().resolve(strict=False))
    for root in protected:
        if resolved == root or root in resolved.parents:
            raise ValueError(
                f"output target {resolved} must not resolve inside the protected source {root}"
            )
    for parent in resolved.parents:
        if parent.suffix == ".zarr" and parent.exists():
            raise ValueError(
                f"output target {resolved} must not resolve inside the existing Zarr store {parent}"
            )
    # Return the path the checks actually resolved. Without the expansion a
    # ``~``-prefixed --output would be validated at the home directory and
    # then written to a literal ``~/`` tree under the working directory.
    return candidate.expanduser()


def _resolve_task_paths(input_root: Path) -> tuple[Path, str]:
    """Derive the policy store path and required task name from one input directory."""

    task_name = (
        input_root.parent.name
        if (input_root / "data.h5").exists() or input_root.name.startswith("episode_")
        else input_root.name
    )
    if not task_name or task_name in {".", ".."}:
        raise ValueError("input_root must name one task directory, e.g. episodes/pick_place_toy")
    try:
        task_name = validate_task_name(task_name)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "input_root must name one valid task directory, e.g. episodes/pick_place_toy"
        ) from exc
    return Path("datasets") / f"{task_name}.zarr", task_name


class _ExportProgress:
    """Render the data-layer's cumulative progress events as one bar per phase."""

    _PHASE_LABELS = {"convert": ("raw to policy Zarr", "episode")}

    def __init__(self) -> None:
        self._phase: str | None = None
        self._bar: tqdm | None = None

    def update(self, phase: str, completed: int, total: int) -> None:
        if phase != self._phase:
            self.close()
            label, unit = self._PHASE_LABELS[phase]
            self._bar = tqdm(total=total, desc=label, unit=unit, file=sys.stderr)
            self._phase = phase
        assert self._bar is not None
        self._bar.update(completed - self._bar.n)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        default_output_path, task_name = _resolve_task_paths(args.input_root)
        # Preflight and real export share the identical output contract.
        output_path = _resolve_output_path(args.output, default_output_path, args.input_root)
        config = PolicyZarrExportConfig(
            chunk_frames=args.chunk_frames,
            compression_level=args.compression_level,
            expected_task_name=task_name,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    progress = _ExportProgress()
    report: dict
    # Preflight and real export share the identical output contract: an
    # occupied target is refused up front in BOTH modes, exactly as the
    # export transaction would refuse it.
    if target_is_occupied(output_path):
        print(
            f"error: refusing to overwrite existing output: {output_path}",
            file=sys.stderr,
        )
        return 2
    try:
        runtime = resolve_experiment_config(yaml_path=args.config)
        processing = ProcessingConfig.from_runtime(runtime)
        if args.pointcloud_num_points is not None:
            processing = replace(
                processing,
                pointcloud=replace(processing.pointcloud, num_points=args.pointcloud_num_points),
            )
        report = export_raw_to_zarr(
            args.input_root,
            output_path,
            config,
            processing=processing,
            annotations_path=args.annotations,
            task_name=args.task_name,
            dry_run=args.dry_run,
            progress_callback=progress.update,
        )
    except (FileExistsError, FileNotFoundError, NotADirectoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        UnicodeError,
        ValueError,
    ) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    finally:
        progress.close()
    verb = "Validated" if args.dry_run else "Exported"
    print(
        f"{verb} {report['episode_count']} episode(s), {report['total_frames']} frames.",
        file=sys.stderr,
    )
    print(
        f"Rejected {len(report['rejected_episodes'])} episode(s); excluded {len(report['excluded_episodes'])} by annotation.",
        file=sys.stderr,
    )
    for rejected in report["rejected_episodes"]:
        print(f"  {rejected['episode']}: {rejected['reason']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
