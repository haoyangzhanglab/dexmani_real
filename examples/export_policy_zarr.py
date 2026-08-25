#!/usr/bin/env python3
"""Usage: ``python examples/export_policy_zarr.py --input-root DIR --output PATH``.

Offline CLI that exports validated processed task episodes to a minimal
dexmani_policy Zarr. Connects to no hardware, opens no GUI, and writes only
the requested ``dataset/<task>.zarr`` output. Argument parsing and JSON
report printing live here; the export transaction itself stays in
``dexmani_real.data.export``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dexmani_real.data.export import (
    PolicyZarrExportConfig,
    export_processed_hdf5_to_zarr,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export validated Real processed HDF5 episodes to dexmani_policy Zarr."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="One task directory, e.g. episodes_processed/<task_name>.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New dataset/<task_name>.zarr path; files, directories, and symlinks are refused.",
    )
    parser.add_argument(
        "--task-name",
        help="Require this dataset-level task_name in every processed HDF5.",
    )
    parser.add_argument("--chunk-frames", type=int, default=100)
    parser.add_argument("--compression-level", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = export_processed_hdf5_to_zarr(
            args.input_root,
            args.output,
            PolicyZarrExportConfig(
                chunk_frames=args.chunk_frames,
                compression_level=args.compression_level,
                expected_task_name=args.task_name,
            ),
        )
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
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
