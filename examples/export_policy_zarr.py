#!/usr/bin/env python3
"""Usage: python examples/export_policy_zarr.py episodes/<task_name>

Exports complete Raw episodes to a rebuildable canonical Zarr cache.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import yaml
from tqdm import tqdm

from dexmani_real.config.experiment import load_experiment_config, resolve_table_plane
from dexmani_real.dataset.contracts import ProcessingConfig, validate_task_identity
from dexmani_real.dataset.export import (
    CanonicalExportConfig,
    export_raw_to_zarr,
)
from dexmani_real.utils.atomic_io import target_is_occupied

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transform Raw Real episodes to a canonical multimodal Zarr."
    )
    parser.add_argument(
        "input_root",
        type=Path,
        metavar="episodes/<task_name>",
        help=(
            "One raw task or episode directory. Exports to "
            "datasets/<task_name>.zarr by default (see --output); existing "
            "output paths require --overwrite."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Alternative Zarr cache output path for this task "
            "(default: datasets/<task_name>.zarr). The default cache name "
            "comes from the input directory; task labels come from Raw. Existing caches require --overwrite, "
            "and the resolved target (symlinks followed) must not fall "
            "inside the protected sources: episodes/, episodes_processed/, "
            "rollouts/, the input root, or an existing .zarr store."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional processing YAML; defaults to project settings and current table calibration.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="EPISODE",
        help="Exclude a whole episode by directory name; may be repeated.",
    )
    parser.add_argument(
        "--pointcloud-num-points",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Complete conversion in staging before replacing an existing canonical cache.",
    )
    parser.add_argument("--chunk-frames", type=int, default=100)
    parser.add_argument("--compression-level", type=int, default=3)
    return parser


# Protect repository-relative raw, legacy processed and rollout data.
_PROTECTED_SOURCE_ROOTS = ("episodes", "episodes_processed", "rollouts")


def _resolve_output_path(
    output: Path | None,
    default_path: Path,
    input_root: Path,
) -> Path:
    """Reject outputs inside protected sources or existing Zarr stores, following symlinks."""
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
    # Expand ~ for writing as well as checking.
    return candidate.expanduser()


def _default_output_path(input_root: Path) -> Path:
    task_name = (
        input_root.parent.name
        if (input_root / "data.h5").exists() or input_root.name.startswith("episode_")
        else input_root.name
    )
    if not task_name or task_name in {".", ".."}:
        raise ValueError("input_root must name one task directory, e.g. episodes/pick_place_toy")
    try:
        task_name = validate_task_identity(task_name)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "input_root must name one valid task directory, e.g. episodes/pick_place_toy"
        ) from exc
    return Path("datasets") / f"{task_name}.zarr"


def _load_processing_config(path: Path | None) -> ProcessingConfig:
    """Load processing values; read the current table plane only for table removal."""
    runtime = load_experiment_config(yaml_path=path)
    plane = (
        resolve_table_plane(runtime.environment.table) if runtime.pointcloud.remove_table else None
    )
    return ProcessingConfig.from_runtime(runtime, table_plane_abcd=plane)


class _ExportProgress:
    def __init__(self) -> None:
        self._bar: tqdm | None = None

    def update(self, completed: int, total: int) -> None:
        if self._bar is None:
            self._bar = tqdm(
                total=total, desc="Raw to canonical Zarr", unit="episode", file=sys.stderr
            )
        self._bar.update(completed - self._bar.n)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        args.input_root = args.input_root.expanduser().resolve()
        default_output_path = _default_output_path(args.input_root)
        output_path = _resolve_output_path(args.output, default_output_path, args.input_root)
        config = CanonicalExportConfig(
            chunk_frames=args.chunk_frames,
            compression_level=args.compression_level,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    progress = _ExportProgress()
    report: dict
    if target_is_occupied(output_path) and not args.overwrite:
        print(
            f"error: refusing to overwrite existing output: {output_path}",
            file=sys.stderr,
        )
        return 2
    try:
        processing = _load_processing_config(args.config)
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
            exclude=tuple(args.exclude),
            overwrite=args.overwrite,
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
        yaml.YAMLError,
    ) as exc:
        traceback.print_exc()
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    finally:
        progress.close()
    print(
        f"Exported {report['episode_count']} episode(s), {report['total_frames']} frames.",
        file=sys.stderr,
    )
    print(
        f"Automatically rejected {len(report['rejected_episodes'])} episode(s).",
        file=sys.stderr,
    )
    print(
        f"Explicitly excluded {len(report['excluded_episodes'])} episode(s).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
