#!/usr/bin/env python3
"""Export or preflight processed Real episodes for dexmani_policy.

Offline CLI that exports validated processed task episodes to a minimal
dexmani_policy Zarr. Connects to no hardware, opens no GUI, and writes only
the requested ``dataset/<task>.zarr`` output. ``--dry-run`` performs the same
input-contract and finite-payload checks without creating an output store.
Argument parsing and JSON report printing live here; the export transaction
itself stays in ``dexmani_real.data.export``.
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
    preflight_processed_hdf5_to_zarr,
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
        help=(
            "New dataset/<task_name>.zarr path; required unless --dry-run. Files, "
            "directories, and symlinks are refused."
        ),
    )
    parser.add_argument(
        "--task-name",
        help="Require this dataset-level task_name in every processed HDF5.",
    )
    parser.add_argument("--chunk-frames", type=int, default=100)
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Read and validate the processed inputs without creating a Zarr output. "
            "Use this before a large export."
        ),
    )
    return parser


def _format_export_failure(exc: Exception) -> str:
    """Add the recovery hint for a processed artifact rejected by policy export."""

    message = str(exc)
    if "invalid Real core modality semantics" in message:
        return (
            f"{message}\n"
            "hint: Policy Zarr v5 requires deployment-equivalent point-cloud "
            "processed v10 data; reprocess raw v23 with --task-name <task>"
        )
    return message


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.dry_run and args.output is not None:
        parser.error("--output cannot be used with --dry-run")
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run")
    try:
        config = PolicyZarrExportConfig(
            chunk_frames=args.chunk_frames,
            compression_level=args.compression_level,
            expected_task_name=args.task_name,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    try:
        if args.dry_run:
            report = preflight_processed_hdf5_to_zarr(args.input_root, config)
        else:
            assert args.output is not None
            report = export_processed_hdf5_to_zarr(
                args.input_root,
                args.output,
                config,
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
        print(f"Export failed: {_format_export_failure(exc)}", file=sys.stderr)
        return 1
    report["dry_run"] = args.dry_run
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
