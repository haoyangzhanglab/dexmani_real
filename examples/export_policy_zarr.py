#!/usr/bin/env python3
"""Export or preflight processed Real episodes for dexmani_policy.

Offline CLI that exports validated processed task episodes to a minimal
dexmani_policy Zarr. Connects to no hardware, opens no GUI, and writes only
the derived ``datasets/<task>.zarr`` output. The positional input
``episodes_processed/<task>`` determines both paths and the required dataset
task name. ``--dry-run`` performs the same input-contract and finite-payload
checks without creating an output store. Progress bars and errors go to stderr;
stdout stays empty. Argument parsing and terminal presentation live here; the
export transaction itself stays in ``dexmani_real.dataset.export``.
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

from dexmani_real.dataset.contracts import validate_processed_task_name
from dexmani_real.dataset.export import (
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
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Alternative Zarr output path for a NEW generation of this task "
            "(default: datasets/<task_name>.zarr). The task identity always "
            "comes from the input directory. Existing outputs are refused, "
            "and the resolved target (symlinks followed) must not fall "
            "inside the protected sources: episodes/, episodes_processed/, "
            "rollouts/, the input root, or an existing .zarr store."
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


# Source roots whose contents must never receive derived export output.
_PROTECTED_SOURCE_ROOTS = ("episodes", "episodes_processed", "rollouts")


def _resolve_output_path(
    output: Path | None,
    default_path: Path,
    input_root: Path,
) -> Path:
    """Resolve the export target and keep it outside every protected source.

    Symlinks are followed, so a link that escapes into raw/processed/rollout
    data is refused by its resolved location. Overwriting any existing output
    is refused later by the export transaction itself.
    """
    candidate = default_path if output is None else output
    resolved = candidate.expanduser().resolve(strict=False)
    protected = [
        Path(name).resolve(strict=False) for name in _PROTECTED_SOURCE_ROOTS
    ]
    protected.append(input_root.expanduser().resolve(strict=False))
    for root in protected:
        if resolved == root or root in resolved.parents:
            raise ValueError(
                f"output target {resolved} must not resolve inside the "
                f"protected source {root}"
            )
    for parent in resolved.parents:
        if parent.suffix == ".zarr" and parent.exists():
            raise ValueError(
                f"output target {resolved} must not resolve inside the "
                f"existing Zarr store {parent}"
            )
    return candidate


def _resolve_task_paths(input_root: Path) -> tuple[Path, str]:
    """Derive the policy store path and required task name from one input directory."""

    task_name = input_root.name
    if not task_name or task_name in {".", ".."}:
        raise ValueError(
            "input_root must name one task directory, e.g. "
            "episodes_processed/pick_place_toy"
        )
    try:
        task_name = validate_processed_task_name(task_name)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "input_root must name one valid task directory, e.g. "
            "episodes_processed/pick_place_toy"
        ) from exc
    return Path("datasets") / f"{task_name}.zarr", task_name


class _ExportProgress:
    """Render the data-layer's cumulative progress events as one bar per phase."""

    _PHASE_LABELS = {
        "validate": ("validate processed episodes", "file"),
        "write": ("write policy Zarr", "frame"),
        "verify": ("verify policy Zarr", "array"),
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
        default_output_path, task_name = _resolve_task_paths(args.input_root)
        # Preflight and real export share the identical output contract.
        output_path = _resolve_output_path(
            args.output, default_output_path, args.input_root
        )
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
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    finally:
        progress.close()
    verb = "Validated" if args.dry_run else "Exported"
    print(
        f"{verb} {report['episode_count']} episode(s), {report['total_frames']} frames.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
