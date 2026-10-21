#!/usr/bin/env python3
"""离线将完整 Raw episode 导出为可重建的全模态 Canonical Zarr 缓存。"""

from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

import yaml
from tqdm import tqdm

from dexmani_real.config.experiment import load_experiment_config, resolve_table_plane
from dexmani_real.data.canonical.export import CanonicalExporter
from dexmani_real.data.canonical.schema import ProcessingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export complete Raw episodes to a canonical multimodal Zarr; "
            "reject incomplete episodes as a whole."
        )
    )
    parser.add_argument(
        "input_root",
        type=Path,
        metavar="episodes/<task_name>",
        help=(
            "One raw task or episode directory. Exports to "
            "datasets/<task_name>.zarr by default (see --output). "
            "All modalities must be complete and finite. The destination must not already exist."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Alternative Zarr cache output path for this task "
            "(default: datasets/<task_name>.zarr). The default cache name "
            "comes from the input directory; task labels come from Raw. "
            "The destination must be new and must not be a symlink. Its resolved path must not fall "
            "inside episodes/, rollouts/, the input root, or an existing .zarr store."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional experiment YAML overrides for offline processing",
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
        "--chunk-frames", type=int, default=100,
        help="Maximum processing batch rows; storage chunks are sized independently per field",
    )
    parser.add_argument("--compression-level", type=int, default=3)
    return parser


def default_output_path(input_root: Path) -> Path:
    task_name = (
        input_root.parent.name
        if (input_root / "data.h5").exists() or input_root.name.startswith("episode_")
        else input_root.name
    )
    if not task_name:
        raise ValueError("input_root must name one task directory, e.g. episodes/pick_place_toy")
    return Path("datasets") / f"{task_name}.zarr"


def load_processing_config(path: Path | None, *, num_points: int | None) -> ProcessingConfig:
    """Load processing values; read the current table plane only for table removal."""
    runtime = load_experiment_config(
        yaml_path=path,
        cli_overrides={"pointcloud.num_points": num_points},
    )
    plane = (
        resolve_table_plane(runtime.environment.table) if runtime.pointcloud.remove_table else None
    )
    return ProcessingConfig.from_runtime(runtime, table_plane_abcd=plane)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.input_root = args.input_root.expanduser().resolve()
        output_path = (
            args.output if args.output is not None else default_output_path(args.input_root)
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    try:
        processing = load_processing_config(args.config, num_points=args.pointcloud_num_points)
        from dexmani_real.utils.log import configure_logging

        configure_logging()
        with tqdm(desc="Raw to canonical Zarr", unit="episode", file=sys.stderr) as progress:

            def update_progress(completed, total):
                progress.total = total
                progress.update(completed - progress.n)

            result = CanonicalExporter(
                processing=processing,
                chunk_frames=args.chunk_frames,
                compression_level=args.compression_level,
            ).export(
                args.input_root,
                output_path,
                exclude=tuple(args.exclude),
                progress_callback=update_progress,
            )
    except (FileExistsError, FileNotFoundError, NotADirectoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as exc:
        traceback.print_exc()
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Exported {result['episode_count']} episode(s), {result['total_frames']} frames.",
        file=sys.stderr,
    )
    print(
        f"Automatically rejected {len(result['rejected_episodes'])} episode(s).",
        file=sys.stderr,
    )
    print(
        f"Explicitly excluded {len(result['excluded_episodes'])} episode(s).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
