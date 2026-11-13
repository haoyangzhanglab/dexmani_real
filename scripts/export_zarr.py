#!/usr/bin/env python3
"""将 Raw episode 的所选字段导出为 Canonical Zarr。"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from tqdm import tqdm

from dexmani_real.config.real import load_config
from dexmani_real.data.export import (
    FIELD_INPUTS,
    ExportConfig,
    export_zarr,
    resolve_fields,
)
from dexmani_real.utils.table_plane import load_table_plane


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export selected fields from published Raw episodes of one task to canonical Zarr. "
            "Preserve recorded continuous segments as separate episodes without resampling. "
            "Reject missing/nonfinite selected data and publish only after all episodes succeed."
        )
    )
    parser.add_argument(
        "input_root",
        type=Path,
        metavar="INPUT_ROOT",
        help="One Raw task directory containing episode_* subdirectories, or one episode directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Alternative canonical Zarr output path for this task "
            "(default: datasets/<input-directory>.zarr, using the parent for a single episode). "
            "Stored task_name metadata comes from Raw meta.task_label. "
            "The destination must be new; a new target inside the input directory is allowed."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Trusted Python experiment file exporting CONFIG: RealConfig; execution.hoi_max_skew_s optionally limits joint HOI source skew (default None/off)",
    )
    parser.add_argument(
        "--fields", nargs="+", choices=tuple(FIELD_INPUTS),
        help="Canonical fields to export; omitted means all fields except object_pointcloud.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="EPISODE",
        help="Exclude a whole episode by directory name; may be repeated. Unknown names are rejected.",
    )
    return parser


def default_output_path(input_root: Path) -> Path:
    output_stem = (
        input_root.parent.name
        if (input_root / "data.h5").is_file()
        else input_root.name
    )
    if not output_stem:
        raise ValueError("input_root must name one task directory, e.g. episodes/pick_place_toy")
    return Path("datasets") / f"{output_stem}.zarr"


def load_processing_config(path: Path | None, fields) -> ExportConfig:
    """Load processing values; read the current table plane only for table removal."""
    runtime = load_config(path)
    plane = (
        load_table_plane(runtime.environment.table_plane_path)
        if "point_cloud" in fields and runtime.pointcloud.remove_table else None
    )
    return ExportConfig(
        pointcloud=runtime.pointcloud,
        hand_pointcloud=runtime.hand_pointcloud,
        table_plane_abcd=plane,
        hoi_max_skew_s=runtime.execution.hoi_max_skew_s,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.input_root = args.input_root.expanduser()
    output_path = (
        args.output if args.output is not None else default_output_path(args.input_root)
    )
    fields = resolve_fields(args.fields)
    processing = load_processing_config(args.config, fields)
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
            fields=fields,
            exclude=tuple(args.exclude),
            progress_callback=update_progress,
        )
    print(
        f"Exported {result['raw_episode_count']} Raw file(s) as "
        f"{result['episode_count']} continuous episode(s), {result['total_frames']} frames.",
        file=sys.stderr,
    )
    print(
        f"Explicitly excluded {len(result['excluded_episodes'])} episode(s).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
