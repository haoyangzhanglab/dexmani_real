#!/usr/bin/env python3
"""Batch-run the L515 RGB timing diagnostic suite (guide §29) as one command.

Sequences ``runs`` copies of RGB-D baseline followed by ``runs`` copies of the
color-only control, each invoking ``examples/diagnose_l515_rgb_timing.py`` with
the exact arguments the manual protocol prescribes:

    RGB-D   -> depth(Z16)+color(BGR8), queue-capacity 2
    color   -> color(BGR8) only,       queue-capacity 1

The produced label / output-directory names match the guide (§29.2/§29.3):

    diagnostics/dark_rgbd_baseline_01/ … _03
    diagnostics/dark_color_only_01/    … _03

Each run prints one concise summary line; the full detail stays in that run's
``report.json``. After all runs, the wrapper writes a self-describing
``suite_summary.json`` manifest under ``--output-root`` that aggregates every
run's key metrics and status (``ok`` / ``empty_capture`` / ``failed``), so a
human can eyeball the whole suite at once. The manifest is written on both the
success and the stop-on-failure paths, atomically.

Safety: this wrapper only *launches* the diagnostic as a subprocess (capturing
its stdout); it never connects to the RealSense camera itself, never commands
the robot, and never writes camera options — the diagnostic is non-mutating
(``get_option`` only, no ``set_option``). Prefer running with ``--dry-run``
first to preview the exact commands. Re-running with the same ``--output-root``
and labels fails closed *before starting the pipeline*: each diagnostic refuses
a non-empty output directory, and this wrapper refuses an existing manifest, so
nothing is overwritten.

Example::

    # Preview the full six-command sequence without touching hardware:
    python examples/run_l515_timing_suite.py --dry-run

    # Real dark-room run (serial auto-detected if exactly one camera):
    python examples/run_l515_timing_suite.py --serial f1382055
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# Path to the underlying diagnostic, resolved relative to this file so the
# suite works regardless of the caller's working directory.
_DIAGNOSTIC = Path(__file__).resolve().with_name("diagnose_l515_rgb_timing.py")

# Guide §29 labels and per-mode queue capacities.
_MODES: dict[str, dict[str, str | int]] = {
    "rgbd": {"label_stem": "dark_rgbd_baseline", "queue_capacity": 2},
    "color": {"label_stem": "dark_color_only", "queue_capacity": 1},
}
# Order the guide runs: RGB-D baseline first, then color-only control.
_MODE_ORDER = ("rgbd", "color")

_SUITE_SCHEMA_VERSION = 1
_MANIFEST_FILENAME = "suite_summary.json"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run the L515 RGB timing diagnostic suite (guide §29)."
    )
    parser.add_argument(
        "--serial",
        default=None,
        help="RealSense serial. Omitted -> diagnostic auto-selects if exactly one camera.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    parser.add_argument("--duration-seconds", type=float, default=20.0)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of repetitions per mode (default 3).",
    )
    parser.add_argument(
        "--output-root",
        default="diagnostics",
        help="Directory under which each run's folder and suite_summary.json are created.",
    )
    parser.add_argument(
        "--only",
        choices=("rgbd", "color"),
        default=None,
        help="Run only one mode (default: both, RGB-D then color).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact commands without launching anything.",
    )
    args = parser.parse_args(argv)
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("width, height, and fps must be positive")
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.warmup_seconds < 0 or args.duration_seconds <= 0 or args.timeout_ms <= 0:
        parser.error("warmup must be non-negative; duration and timeout must be positive")
    return args


def _build_command(
    args: argparse.Namespace, mode: str, index: int
) -> tuple[list[str], Path, str]:
    label = f"{_MODES[mode]['label_stem']}_{index:02d}"
    output_dir = Path(args.output_root) / label
    cmd = [
        sys.executable,
        str(_DIAGNOSTIC),
        "--mode", mode,
        "--width", str(args.width),
        "--height", str(args.height),
        "--fps", str(args.fps),
        "--queue-capacity", str(_MODES[mode]["queue_capacity"]),
        "--warmup-seconds", str(args.warmup_seconds),
        "--duration-seconds", str(args.duration_seconds),
        "--timeout-ms", str(args.timeout_ms),
        "--label", label,
        "--output-dir", str(output_dir),
    ]
    if args.serial:
        cmd += ["--serial", args.serial]
    return cmd, output_dir, label


def _fmt(value: Any, spec: str) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and not math.isfinite(value):
        return "-"
    return format(value, spec)


def _period_class(evidence: dict[str, Any]) -> str:
    if evidence.get("color_normalized_period_near_33ms"):
        return "33ms"
    if evidence.get("color_normalized_period_near_60ms"):
        return "60ms"
    return "other"


def _read_report_entry(mode: str, label: str, output_dir: Path) -> dict[str, Any] | None:
    """Read a just-written report.json and reduce it to the manifest entry."""
    try:
        report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    color_unique_count = report.get("color_unique_count")
    evidence = report.get("evidence") or {}
    return {
        "mode": mode,
        "label": label,
        "output_dir": str(output_dir),
        "status": "empty_capture" if color_unique_count in (None, 0) else "ok",
        "exit_code": 0,
        "color_unique_count": color_unique_count,
        "color_unique_rate_hz": report.get("color_unique_rate_hz"),
        "color_period_p50_ms": (report.get("color_normalized_period_ms") or {}).get(
            "p50"
        ),
        "color_period_class": _period_class(evidence),
        "color_repeat_ratio": report.get("color_repeat_ratio"),
        "color_frame_gap_event_count": report.get("color_frame_gap_event_count"),
        "luma_mean": (report.get("luminance") or {}).get("mean"),
        "evidence": evidence,
    }


def _print_run_summary(entry: dict[str, Any]) -> None:
    tag = "[empty]" if entry["status"] == "empty_capture" else "[ok]"
    parts = [
        f"{tag} {entry['label']}",
        f"color={_fmt(entry['color_unique_rate_hz'], '.2f')}Hz",
        f"period_p50={_fmt(entry['color_period_p50_ms'], '.1f')}ms",
        f"band={entry['color_period_class']}",
    ]
    repeat = entry["color_repeat_ratio"]
    if entry["mode"] == "rgbd" and isinstance(repeat, (int, float)) and math.isfinite(repeat):
        parts.append(f"repeat={repeat * 100:.1f}%")
    gaps = entry["color_frame_gap_event_count"]
    if gaps:
        parts.append(f"gaps={gaps}")
    print("  " + " | ".join(parts))
    if entry["status"] == "empty_capture":
        print(
            f"  WARNING: {entry['label']} captured 0 color frames; "
            "report.json is all-None/NaN — recorded as empty_capture.",
            file=sys.stderr,
        )


def _failed_entry(
    mode: str, label: str, output_dir: Path, exit_code: int, stderr: str
) -> dict[str, Any]:
    return {
        "mode": mode,
        "label": label,
        "output_dir": str(output_dir),
        "status": "failed",
        "exit_code": exit_code,
        "stderr_tail": (stderr or "")[-2000:],
    }


def _write_manifest(
    manifest_path: Path,
    *,
    args: argparse.Namespace,
    total: int,
    started_utc: str,
    runs: list[dict[str, Any]],
    status: str,
    stopped_at: dict[str, Any] | None,
) -> None:
    manifest = {
        "suite": "l515_rgb_timing_suite",
        "schema_version": _SUITE_SCHEMA_VERSION,
        "generated_by": "run_l515_timing_suite.py",
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "command": " ".join(shlex.quote(p) for p in sys.argv[1:]),
        "serial": args.serial,
        "modes": [args.only] if args.only is not None else list(_MODE_ORDER),
        "runs_per_mode": args.runs,
        "run_count_total": total,
        "runs": runs,
        "stopped_at": stopped_at,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    tmp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, manifest_path)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    modes = [args.only] if args.only is not None else list(_MODE_ORDER)

    jobs: list[tuple[str, int, list[str], Path, str]] = []
    for mode in modes:
        for index in range(1, args.runs + 1):
            cmd, output_dir, label = _build_command(args, mode, index)
            jobs.append((mode, index, cmd, output_dir, label))
    total = len(jobs)

    manifest_path = Path(args.output_root) / _MANIFEST_FILENAME

    if args.dry_run:
        for _mode, _index, cmd, _out, _label in jobs:
            print(" ".join(shlex.quote(part) for part in cmd))
        print(
            f"[dry-run] {total} command(s); would write manifest to {manifest_path}; "
            "nothing executed."
        )
        return 0

    # Fail closed: never overwrite a manifest from a previous suite run.
    if manifest_path.exists():
        print(f"refusing to overwrite existing manifest: {manifest_path}", file=sys.stderr)
        return 1

    started_utc = datetime.now(timezone.utc).isoformat()
    runs: list[dict[str, Any]] = []

    for run_no, (mode, index, cmd, output_dir, label) in enumerate(jobs, start=1):
        print(f"\n[{run_no}/{total}] {mode} {label} -> {output_dir}", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            sys.stderr.write(result.stderr or "")
            print(
                f"[{run_no}/{total}] FAILED (exit {result.returncode}): {label}; "
                "nothing overwritten.",
                file=sys.stderr,
            )
            runs.append(
                _failed_entry(mode, label, output_dir, result.returncode, result.stderr)
            )
            _write_manifest(
                manifest_path,
                args=args,
                total=total,
                started_utc=started_utc,
                runs=runs,
                status="stopped",
                stopped_at={"label": label, "exit_code": result.returncode},
            )
            return result.returncode

        entry = _read_report_entry(mode, label, output_dir)
        if entry is None:
            print(
                f"[{run_no}/{total}] FAILED: {label} exited 0 but report.json is "
                "unreadable; treating as failure (nothing overwritten).",
                file=sys.stderr,
            )
            runs.append(
                _failed_entry(mode, label, output_dir, -1, "report.json unreadable after exit 0")
            )
            _write_manifest(
                manifest_path,
                args=args,
                total=total,
                started_utc=started_utc,
                runs=runs,
                status="stopped",
                stopped_at={"label": label, "exit_code": -1},
            )
            return 1

        runs.append(entry)
        _print_run_summary(entry)

    _write_manifest(
        manifest_path,
        args=args,
        total=total,
        started_utc=started_utc,
        runs=runs,
        status="complete",
        stopped_at=None,
    )
    ok = sum(1 for r in runs if r["status"] == "ok")
    empty = sum(1 for r in runs if r["status"] == "empty_capture")
    print(
        f"\nSuite done: {ok} ok, {empty} empty_capture, {total} total; "
        f"manifest: {manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
