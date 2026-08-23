#!/usr/bin/env python3
"""Run the guide §32 Branch A controlled ablation: Auto-Exposure Priority ON -> OFF.

This is the *mutating* counterpart to the non-mutating diagnostic. The first
round established "strongly supports exposure-limited sensor cadence" (Case 1:
contiguous frame numbers, ~60 ms normalized period, exposure ~60 ms, AE ON,
AE-Priority ON). Branch A then asks a controlled second round that keeps
``Auto Exposure = ON`` but turns ``Auto-Exposure Priority = OFF``, to see
whether the RGB stream recovers toward 30 Hz and how exposure/gain/luminance
move.

What this wrapper does, in order:

    1. Resolve the camera serial (auto-detect if exactly one device).
    2. Open a *short* color-only stream, snapshot the four ablation-relevant
       options (enable_auto_exposure, auto_exposure_priority, exposure, gain),
       and fail closed unless the precondition holds:
       ``enable_auto_exposure == 1`` and ``auto_exposure_priority == 1``.
    3. Set ``auto_exposure_priority -> 0`` while streaming, read it back, and
       stop. Setting while streaming is the most reliable way to make the
       value stick on the device.
    4. Launch ``run_l515_timing_suite.py`` as a subprocess (inherit stdout so
       the per-run progress is visible), writing to a *separate* output root so
       it never collides with the round-1 manifest.
    5. Read the suite manifest and cross-check that each run actually observed
       ``auto_exposure_priority ≈ 0`` during capture. Because the device is
       handed off to a subprocess between set and capture, this readback is the
       honest, fail-closed proof that the option persisted — if any run reports
       priority still ON, the ablation is flagged ``ablation_valid = false``.
    6. Restore ``auto_exposure_priority`` to its pre-ablation value, read it
       back, and record the restore status. Restore runs in a ``finally`` and
       therefore happens on every path: suite success, suite failure, an
       exception, or Ctrl-C.
    7. Atomically write ``ablation_summary.json`` next to the suite manifest.

Safety notes (read before running):

    - This script *does* call ``sensor.set_option(...)`` on
      ``auto_exposure_priority`` — a §8-forbidden option for the non-mutating
      diagnostic. It is deliberately a separate tool, and it always restores
      the pre-ablation value. It never touches enable_auto_exposure, exposure,
      gain, or any other option.
    - It never commands the robot, never writes calibration, and never alters
      production config. The only hardware it touches is the RealSense color
      sensor's auto_exposure_priority, set OFF then put back ON.
    - Restore runs on every handled exit path (suite success/failure, an
      exception, Ctrl-C, and SIGTERM). It is interrupt-tolerant and retries a
      few times. SIGKILL and power loss cannot be handled — if either lands
      mid-ablation the camera is left at priority OFF until manually restored
      (re-run the script once the precondition is back to 1.0, or set it in
      realsense-viewer). If the precondition fails, the script aborts *before*
      changing anything and writes no manifest, so the output root stays clean.
    - Prefer ``--dry-run`` first: it prints the exact plan and touches nothing.

Example::

    # Preview the plan without touching hardware:
    python examples/run_l515_ablation.py --dry-run

    # Real dark-room ablation (separate root, mirrors round-1 duration/fps):
    python examples/run_l515_ablation.py --serial f1382055

    # Include a side-by-side comparison against the round-1 manifest:
    python examples/run_l515_ablation.py --serial f1382055 \
        --baseline-manifest diagnostics/suite_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pyrealsense2 as rs

# Path to the suite wrapper, resolved relative to this file so the ablation
# works regardless of the caller's working directory.
_SUITE = Path(__file__).resolve().with_name("run_l515_timing_suite.py")

# Options snapshotted before/after the ablation (guide §32 Branch A compares
# RGB FPS, actual exposure, gain, and luminance; enable_auto_exposure and
# auto_exposure_priority are the load-bearing precondition/ablation controls).
_ABLATION_OPTIONS = (
    "enable_auto_exposure",
    "auto_exposure_priority",
    "exposure",
    "gain",
)

# The only option this tool ever writes.
_TARGET_OPTION = "auto_exposure_priority"
_ABLATION_OFF = 0.0  # Branch A: priority ON -> OFF
# A priority readback < 0.5 is treated as "held OFF" (the option is a 0/1 float).
_PRIORITY_OFF_THRESHOLD = 0.5

_SCHEMA_VERSION = 1
_MANIFEST_FILENAME = "ablation_summary.json"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run guide §32 Branch A ablation: Auto-Exposure Priority ON -> OFF."
    )
    parser.add_argument(
        "--serial",
        default=None,
        help="RealSense serial. Omitted -> auto-detect if exactly one camera.",
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
        help="Repetitions per mode for the suite (default 3, matching round 1).",
    )
    parser.add_argument(
        "--output-root",
        default="diagnostics/ablation_prio_off",
        help="Separate root for this round (default keeps round 1 untouched).",
    )
    parser.add_argument(
        "--only",
        choices=("rgbd", "color"),
        default=None,
        help="Restrict the suite to one mode (default: both).",
    )
    parser.add_argument(
        "--baseline-manifest",
        default=None,
        help="Path to round-1 suite_summary.json for a side-by-side comparison.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without touching the camera or running anything.",
    )
    args = parser.parse_args(argv)
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("width, height, and fps must be positive")
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.warmup_seconds < 0 or args.duration_seconds <= 0 or args.timeout_ms <= 0:
        parser.error("warmup must be non-negative; duration and timeout must be positive")
    if not math.isfinite(args.warmup_seconds) or not math.isfinite(args.duration_seconds):
        parser.error("warmup and duration must be finite numbers")
    return args


def _device_info(device: Any, key: Any) -> str:
    if key is None:
        return ""
    try:
        return str(device.get_info(key)) if device.supports(key) else ""
    except RuntimeError:
        return ""


def _select_serial(context: Any, requested: str | None) -> str:
    devices = context.query_devices()
    serials = [_device_info(d, rs.camera_info.serial_number) for d in devices]
    serials = [s for s in serials if s]
    if requested is not None:
        if requested not in serials:
            raise RuntimeError(
                f"configured serial {requested!r} is not connected; connected={serials}"
            )
        return requested
    if len(serials) != 1:
        raise RuntimeError(
            f"connect exactly one RealSense or pass --serial; connected={serials}"
        )
    return serials[0]


def _find_color_sensor(device: Any) -> Any:
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            try:
                if profile.stream_type() == rs.stream.color:
                    return sensor
            except RuntimeError:
                continue
    raise RuntimeError("no color sensor with a color stream profile found")


def _snapshot_options(profile: Any) -> dict[str, float | None]:
    """Read the four ablation options from the color sensor; None if unsupported."""
    sensor = _find_color_sensor(profile.get_device())
    result: dict[str, float | None] = {}
    for name in _ABLATION_OPTIONS:
        option = getattr(rs.option, name, None)
        try:
            if option is not None and sensor.supports(option):
                result[name] = float(sensor.get_option(option))
            else:
                result[name] = None
        except RuntimeError:
            result[name] = None
    return result


def _set_priority(profile: Any, value: float) -> float:
    """Set auto_exposure_priority while streaming and return the readback."""
    sensor = _find_color_sensor(profile.get_device())
    option = getattr(rs.option, _TARGET_OPTION, None)
    if option is None or not sensor.supports(option):
        raise RuntimeError(f"{_TARGET_OPTION} not supported by the color sensor")
    sensor.set_option(option, float(value))
    return float(sensor.get_option(option))


def _open_color_session(serial: str, args: argparse.Namespace):
    """Open a short color-only stream and return (pipeline, profile)."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(
        rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps
    )
    profile = pipeline.start(config)
    return pipeline, profile


def _build_suite_command(args: argparse.Namespace, serial: str, output_root: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(_SUITE),
        "--serial", serial,
        "--width", str(args.width),
        "--height", str(args.height),
        "--fps", str(args.fps),
        "--warmup-seconds", str(args.warmup_seconds),
        "--duration-seconds", str(args.duration_seconds),
        "--timeout-ms", str(args.timeout_ms),
        "--runs", str(args.runs),
        "--output-root", str(output_root),
    ]
    if args.only is not None:
        cmd += ["--only", args.only]
    return cmd


def _fmt(value: Any, spec: str = ".3g") -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and not math.isfinite(value):
        return "-"
    return format(value, spec)


def _run_priority_readback(serial: str, args: argparse.Namespace, target: float) -> float:
    """Open a fresh stream, set priority to ``target``, return readback, stop."""
    pipeline, profile = _open_color_session(serial, args)
    try:
        return _set_priority(profile, target)
    finally:
        try:
            pipeline.stop()
        except RuntimeError:
            pass


def _restore_with_retry(
    serial: str, args: argparse.Namespace, target: float, attempts: int = 3
) -> tuple[float | None, str | None]:
    """Restore priority to ``target``, retrying a few times.

    Returns ``(readback, error)``; ``error is None`` iff the readback matched
    ``target``. Retrying covers a transient comm error on the device write —
    the exact failure class the readback-verification exists to catch.
    """
    last_err: str | None = "restore not attempted"
    for _ in range(attempts):
        try:
            readback = _run_priority_readback(serial, args, target)
            if readback == target:
                return readback, None
            last_err = f"readback={_fmt(readback)} != target={_fmt(target)}"
        except (OSError, RuntimeError, ValueError) as exc:
            last_err = str(exc)
    return None, last_err


def _cross_check_runs(suite_manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], bool, int]:
    """Verify each successful suite run observed priority OFF during capture.

    Returns ``(rows, all_ok, ok_count)`` where ``all_ok`` is False if any ok
    run saw priority still ON, or if there were no ok runs to verify against.
    """
    rows: list[dict[str, Any]] = []
    all_ok = True
    ok_count = 0
    for run in suite_manifest.get("runs", []):
        priority = (run.get("evidence") or {}).get("auto_exposure_priority")
        held_off = priority is not None and priority < _PRIORITY_OFF_THRESHOLD
        if run.get("status") == "ok":
            ok_count += 1
            if not held_off:
                all_ok = False
        rows.append(
            {
                "label": run.get("label"),
                "mode": run.get("mode"),
                "status": run.get("status"),
                "auto_exposure_priority_during_capture": priority,
                "priority_held_off": held_off,
            }
        )
    if ok_count == 0:
        all_ok = False
    return rows, all_ok, ok_count


def _build_comparison(
    baseline_path: str | None, ablation_suite: dict[str, Any]
) -> list[dict[str, Any]] | None:
    """Side-by-side per-mode comparison of the manifest-level metrics."""
    if baseline_path is None:
        return None
    try:
        baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [{"error": f"unreadable baseline manifest: {exc}"}]

    def aggregate(suite: dict[str, Any]) -> dict[str, dict[str, float | None]]:
        per: dict[str, list[dict[str, Any]]] = {}
        for run in suite.get("runs", []):
            if run.get("status") != "ok":
                continue
            per.setdefault(run.get("mode"), []).append(run)
        out: dict[str, dict[str, float | None]] = {}
        for mode, rows in per.items():
            def med(key: str) -> float | None:
                vals = [r.get(key) for r in rows]
                vals = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
                if not vals:
                    return None
                return float(statistics.median(vals))
            out[mode] = {
                "color_unique_rate_hz": med("color_unique_rate_hz"),
                "color_period_p50_ms": med("color_period_p50_ms"),
                "luma_mean": med("luma_mean"),
            }
        return out

    base = aggregate(baseline)
    abla = aggregate(ablation_suite)
    modes = sorted(set(base) | set(abla))
    comparison = []
    for mode in modes:
        b = base.get(mode, {})
        a = abla.get(mode, {})
        row: dict[str, Any] = {"mode": mode, "baseline": b, "ablation": a}
        for key in ("color_unique_rate_hz", "color_period_p50_ms", "luma_mean"):
            bv, av = b.get(key), a.get(key)
            if bv is not None and av is not None:
                row[f"delta_{key}"] = av - bv
        comparison.append(row)
    return comparison


def _write_manifest(manifest_path: Path, state: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    tmp.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, manifest_path)


def _print_plan(args: argparse.Namespace, serial: str, output_root: Path) -> None:
    print("[dry-run] guide §32 Branch A ablation plan (no hardware touched)")
    print(f"  serial:              {serial}")
    print(f"  precondition:        enable_auto_exposure=1, auto_exposure_priority=1")
    print(f"  ablation:            auto_exposure_priority 1 -> 0 (verified while streaming)")
    print("  suite:               " + " ".join(
        shlex.quote(p) for p in _build_suite_command(args, serial, output_root)
    ))
    print(f"  restore:             auto_exposure_priority 0 -> 1 (verified, always)")
    print(f"  ablation manifest:   {output_root / _MANIFEST_FILENAME}")
    if args.baseline_manifest:
        print(f"  baseline manifest:   {args.baseline_manifest}")
    print("  nothing executed.")


def _sigterm_as_interrupt(signum: int, frame: Any) -> None:
    # Convert SIGTERM into KeyboardInterrupt so the finally-block restore still
    # runs when the process is `kill`ed (SIGKILL/power-loss cannot be handled).
    raise KeyboardInterrupt


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root = Path(args.output_root)
    manifest_path = output_root / _MANIFEST_FILENAME

    if args.dry_run:
        _print_plan(args, args.serial or "<auto-detect>", output_root)
        return 0

    # Resolve serial (auto-detect) for the run.
    try:
        serial = _select_serial(rs.context(), args.serial)
    except RuntimeError as exc:
        print(f"ablation aborted: {exc}", file=sys.stderr)
        return 1

    # A `kill` (SIGTERM) should still restore the option, so route it through
    # the same KeyboardInterrupt path the finally-block restore already handles.
    signal.signal(signal.SIGTERM, _sigterm_as_interrupt)

    # Fail closed: require the output root to be absent or empty before doing
    # anything, so a re-run never collides with round-1 output, a stale
    # suite_summary.json, or a previous ablation's per-run directories. Name
    # exactly what to delete rather than leaving the operator to guess.
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        print(
            f"output root must be absent or empty: {output_root} "
            "(delete it, or move it aside, to re-run)",
            file=sys.stderr,
        )
        return 1

    started_utc = _now_utc()
    state: dict[str, Any] = {
        "tool": "run_l515_ablation",
        "schema_version": _SCHEMA_VERSION,
        "ablation": "auto_exposure_priority 1 -> 0 (guide §32 Branch A)",
        "started_utc": started_utc,
        "command": " ".join(shlex.quote(p) for p in sys.argv[1:]),
        "serial": serial,
        "output_root": str(output_root),
        "baseline_manifest": args.baseline_manifest,
        "precondition": {},
        "set_to": {},
        "suite": {},
        "restore": {},
        "ablation_valid": None,
        "ablation_valid_message": "",
        "comparison": None,
    }

    pre_priority: float | None = None
    attempted_set = False  # True only once _set_priority(OFF) is actually attempted
    exit_code = 1

    try:
        # 1. Precondition + set priority OFF (one short streaming session).
        pipeline, profile = _open_color_session(serial, args)
        try:
            pre = _snapshot_options(profile)
            pre_priority = pre.get("auto_exposure_priority")
            pre_ae = pre.get("enable_auto_exposure")
            state["precondition"]["snapshot"] = pre
            if pre_priority is None or pre_ae is None:
                state["precondition"]["ok"] = False
                state["precondition"]["message"] = (
                    "auto_exposure_priority / enable_auto_exposure not readable; "
                    "cannot run a controlled ablation."
                )
            elif pre_ae != 1.0:
                state["precondition"]["ok"] = False
                state["precondition"]["message"] = (
                    f"precondition not met: enable_auto_exposure={pre_ae} (expected 1.0, "
                    "the Branch A ablation keeps AE ON)."
                )
            elif pre_priority != 1.0:
                state["precondition"]["ok"] = False
                state["precondition"]["message"] = (
                    f"precondition not met: auto_exposure_priority={pre_priority} "
                    "(expected 1.0 before the ON->OFF ablation)."
                )
            else:
                state["precondition"]["ok"] = True
                state["precondition"]["message"] = (
                    f"enable_auto_exposure={pre_ae}, auto_exposure_priority={pre_priority}"
                )

            if state["precondition"]["ok"]:
                attempted_set = True
                readback = _set_priority(profile, _ABLATION_OFF)
                state["set_to"] = {"target": _ABLATION_OFF, "readback": readback}
                if readback != _ABLATION_OFF:
                    raise RuntimeError(
                        f"set auto_exposure_priority -> {_ABLATION_OFF} but readback={readback}"
                    )
        finally:
            try:
                pipeline.stop()
            except RuntimeError:
                pass

        # 2. Abort (before running the suite) if the precondition failed.
        if not state["precondition"]["ok"]:
            print(state["precondition"]["message"], file=sys.stderr)
            exit_code = 1
            return exit_code

        print(
            f"[set] {_TARGET_OPTION} -> {_ABLATION_OFF} "
            f"(readback={_fmt(state['set_to'].get('readback'))})",
            flush=True,
        )

        # 3. Run the suite (live progress, separate output root).
        suite_cmd = _build_suite_command(args, serial, output_root)
        suite_exit = subprocess.run(suite_cmd).returncode
        state["suite"]["exit_code"] = suite_exit
        state["suite"]["manifest"] = str(output_root / "suite_summary.json")

        # 4. Cross-check the ablation actually held during capture.
        suite_manifest_path = output_root / "suite_summary.json"
        suite_manifest: dict[str, Any] | None = None
        if suite_exit == 0 and suite_manifest_path.exists():
            try:
                loaded = json.loads(suite_manifest_path.read_text(encoding="utf-8"))
                suite_manifest = loaded if isinstance(loaded, dict) else None
            except (OSError, ValueError):
                suite_manifest = None
        if suite_manifest is not None:
            rows, all_held, ok_count = _cross_check_runs(suite_manifest)
            state["suite"]["runs"] = rows
            state["ablation_valid"] = all_held
            if all_held:
                state["ablation_valid_message"] = (
                    f"all {ok_count} ok run(s) observed auto_exposure_priority OFF "
                    "during capture"
                )
            else:
                state["ablation_valid_message"] = (
                    "no ok run observed auto_exposure_priority OFF during capture — "
                    "either a run saw it still ON (option did not persist across the "
                    "device handoff) or every capture was empty; this ablation is not a "
                    "valid controlled comparison."
                )
            state["comparison"] = _build_comparison(args.baseline_manifest, suite_manifest)
        else:
            state["ablation_valid"] = False
            state["ablation_valid_message"] = (
                f"suite exited {suite_exit} or wrote no readable manifest; "
                "capture results unavailable."
            )

        if state["ablation_valid"] and suite_exit == 0:
            exit_code = 0
        elif suite_exit == 0:
            exit_code = 2  # ran, but the ablation did not hold
        else:
            exit_code = 1

    except KeyboardInterrupt:
        print("\nablation interrupted; restoring auto_exposure_priority ...", file=sys.stderr)
        exit_code = 1
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ablation failed: {exc}", file=sys.stderr)
        exit_code = 1

    # 5. Restore + write manifest — on every path where the option was actually
    # changed. The restore write is interrupt-tolerant: a second Ctrl-C/SIGTERM
    # landing mid-restore is caught so the failure is still recorded and the
    # manifest is still written, instead of silently skipping the write.
    finally:
        if attempted_set and serial is not None:
            try:
                restored, restore_err = _restore_with_retry(serial, args, pre_priority)
                state["restore"] = {
                    "target": pre_priority,
                    "readback": restored,
                    "ok": restore_err is None and restored == pre_priority,
                    "message": (
                        f"restored -> {pre_priority} (readback={_fmt(restored)})"
                        if restore_err is None
                        else restore_err
                    ),
                }
                if state["restore"]["ok"]:
                    print(f"[restore] {_TARGET_OPTION} -> {pre_priority} (readback={_fmt(restored)})")
                else:
                    print(f"[restore] FAILED: {state['restore']['message']}", file=sys.stderr)
                    exit_code = 1
            except BaseException as exc:
                state["restore"] = {"ok": False, "message": f"restore interrupted: {exc!r}"}
                print(f"[restore] FAILED (interrupted): {exc!r}", file=sys.stderr)
                exit_code = 1

        if attempted_set:
            state["finished_utc"] = _now_utc()
            try:
                _write_manifest(manifest_path, state)
            except OSError as exc:
                print(f"failed to write ablation manifest: {exc}", file=sys.stderr)
                exit_code = 1
            print(f"[manifest] {manifest_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
