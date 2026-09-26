#!/usr/bin/env python3
"""Usage: python examples/export_policy_zarr.py episodes/<task_name> --config export.yaml

Exports raw episodes to canonical Zarr; --dry-run runs the same transforms without writing.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

import yaml
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
            "inside the protected sources: episodes/, episodes_processed/, "
            "rollouts/, the input root, or an existing .zarr store."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit processing YAML; table removal requires an inline audited table plane.",
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


def _resolve_task_paths(input_root: Path) -> tuple[Path, str]:
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


def _load_processing_config(path: Path) -> ProcessingConfig:
    """Resolve export parameters without reading today's table calibration."""
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("export config must use a .yaml or .yml suffix")
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or not isinstance(data.get("pointcloud", {}), dict):
        raise ValueError("export YAML and its pointcloud section must be mappings")
    remove_table = data.get("pointcloud", {}).get("remove_table", True)
    if type(remove_table) is not bool:
        raise ValueError("pointcloud.remove_table must be boolean")
    plane = None
    if remove_table:
        environment = data.get("environment", {})
        table = environment.get("table", {}) if isinstance(environment, dict) else {}
        if (
            not isinstance(table, dict)
            or "plane_abcd" not in table
            or "plane_path" not in table
            or table["plane_path"] is not None
        ):
            raise ValueError(
                "table removal requires explicit environment.table.plane_abcd and "
                "plane_path: null; use pointcloud.remove_table: false when unproven"
            )
        plane = table["plane_abcd"]
    runtime = resolve_experiment_config(
        data=data,
        cli_overrides={"environment.table.enabled": False},
    )
    return ProcessingConfig.from_runtime(runtime, table_plane_abcd=plane)


class _ExportProgress:
    def __init__(self) -> None:
        self._bar: tqdm | None = None

    def update(self, completed: int, total: int) -> None:
        if self._bar is None:
            self._bar = tqdm(
                total=total, desc="raw to policy Zarr", unit="episode", file=sys.stderr
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
        default_output_path, task_name = _resolve_task_paths(args.input_root)
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
    # Dry runs also reject occupied targets.
    if target_is_occupied(output_path):
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
        yaml.YAMLError,
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
