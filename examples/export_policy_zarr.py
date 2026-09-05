#!/usr/bin/env python3
"""Export or preflight processed Real episodes for dexmani_policy.

Offline CLI that exports validated processed task episodes to a minimal
dexmani_policy Zarr. Connects to no hardware, opens no GUI, and writes only
the derived ``datasets/<task>.zarr`` output. The positional input
``episodes_processed/<task>`` determines both paths and the required dataset
task name. ``--dry-run`` performs the same input-contract and finite-payload
checks without creating an output store. Progress bars and errors go to stderr;
stdout stays empty. Argument parsing and terminal presentation live here; the
export transaction itself stays in ``dexmani_real.data.export``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tqdm import tqdm

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
        "input_root",
        type=Path,
        metavar="episodes_processed/<task_name>",
        help=(
            "One processed task directory. Exports to "
            "datasets/<task_name>.zarr; existing output paths are refused."
        ),
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


def _resolve_task_paths(input_root: Path) -> tuple[Path, str]:
    """Derive the policy store path and required task name from one input directory."""

    task_name = input_root.name
    if not task_name or task_name in {".", ".."}:
        raise ValueError(
            "input_root must name one task directory, e.g. "
            "episodes_processed/pick_place_toy"
        )
    return Path("datasets") / f"{task_name}.zarr", task_name


def _format_export_failure(exc: Exception) -> str:
    """Add the recovery hint for a processed artifact rejected by policy export."""

    message = str(exc)
    if "invalid Real core modality semantics" in message:
        return (
            f"{message}\n"
            "hint: Policy Zarr v6 requires teleop-published-target processed "
            "v12 data; reprocess raw v24 with --task-name <task>"
        )
    return message


def _print_export_admission(report: dict) -> None:
    """Print whole-episode rejections and one concise batch summary."""

    for rejection in report["rejected_episodes"]:
        print(f"REJECT {rejection['episode']}:", file=sys.stderr)
        print(
            f"  {rejection['invalid_frame_count']} invalid frame(s); "
            f"rows: {rejection['invalid_ranges']}",
            file=sys.stderr,
        )
        for reason in rejection["reasons"]:
            print(
                f"  reason: {reason['reason']} "
                f"({reason['frame_count']} frame(s), rows {reason['ranges']})",
                file=sys.stderr,
            )
    print(
        f"Exported {report['episode_count']}/{report['source_file_count']} episode(s); "
        f"rejected {report['rejected_episode_count']} episode(s).",
        file=sys.stderr,
    )


class _ExportProgress:
    """Render the data-layer's cumulative progress events as one bar per phase."""

    _PHASE_LABELS = {
        "validate": ("validate processed episodes", "file"),
        "write": ("write policy Zarr", "frame"),
        "verify": ("verify policy Zarr", "chunk"),
    }

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
        output_path, task_name = _resolve_task_paths(args.input_root)
        config = PolicyZarrExportConfig(
            chunk_frames=args.chunk_frames,
            compression_level=args.compression_level,
            expected_task_name=task_name,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    progress = _ExportProgress()
    report: dict
    try:
        if args.dry_run:
            report = preflight_processed_hdf5_to_zarr(
                args.input_root,
                config,
                progress_callback=progress.update,
            )
        else:
            report = export_processed_hdf5_to_zarr(
                args.input_root,
                output_path,
                config,
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
        print(f"Export failed: {_format_export_failure(exc)}", file=sys.stderr)
        return 1
    finally:
        progress.close()
    _print_export_admission(report)
    return 0 if report["episode_count"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
