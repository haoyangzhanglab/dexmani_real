#!/usr/bin/env python3
"""将 Raw 的有效连续前缀导出为全模态 Canonical Zarr。"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from tqdm import tqdm

from dexmani_real.config.real import load_config
from dexmani_real.data.export import ExportConfig, export_zarr
from dexmani_real.utils.table_plane import load_table_plane


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export one valid continuous prefix per Raw episode to a canonical multimodal Zarr. "
            "Report trimmed tails; reject structural corruption or an empty valid prefix."
        )
    )
    parser.add_argument(
        "input_root",
        type=Path,
        metavar="episodes/<task_name>",
        help=(
            "One raw task or episode directory. Exports to "
            "datasets/<task_name>.zarr by default (see --output). "
            "All retained modalities share the same row range and must be valid. "
            "The destination must not already exist."
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
            "The destination must be new and must not overlap the input directory."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Trusted Python experiment file exporting CONFIG: RealConfig",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="EPISODE",
        help="Exclude a whole episode by directory name; may be repeated.",
    )
    return parser


def default_output_path(input_root: Path) -> Path:
    task_name = (
        input_root.parent.name
        if (input_root / "data.h5").is_file()
        else input_root.name
    )
    if not task_name:
        raise ValueError("input_root must name one task directory, e.g. episodes/pick_place_toy")
    return Path("datasets") / f"{task_name}.zarr"


def load_processing_config(path: Path | None) -> ExportConfig:
    """Load processing values; read the current table plane only for table removal."""
    runtime = load_config(path)
    plane = (
        load_table_plane(runtime.environment.table_plane_path) if runtime.pointcloud.remove_table else None
    )
    return ExportConfig(
        pointcloud=runtime.pointcloud,
        hand_pointcloud=runtime.hand_pointcloud,
        table_plane_abcd=plane,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.input_root = args.input_root.expanduser()
    output_path = (
        args.output if args.output is not None else default_output_path(args.input_root)
    )
    processing = load_processing_config(args.config)
    from dexmani_real.utils.log import configure_logging

    configure_logging()
    with tqdm(desc="Raw to canonical Zarr", unit="episode", file=sys.stderr) as progress:

        def update_progress(completed, total):
            progress.total = total
            progress.update(completed - progress.n)

        result = export_zarr(
            args.input_root,
            output_path,
            processing=processing,
            exclude=tuple(args.exclude),
            progress_callback=update_progress,
        )
    print(
        f"Exported {result['episode_count']} episode(s), {result['total_frames']} frames.",
        file=sys.stderr,
    )
    print(
        f"Explicitly excluded {len(result['excluded_episodes'])} episode(s).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
