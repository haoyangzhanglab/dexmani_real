"""VR receiver lifecycle and sample collection for heading calibration."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import yaml

from dexmani_real import ASSET_DIR, PACKAGE_DIR
from dexmani_real.calibration.vr_heading import (
    HeadingEstimate,
    build_heading_config,
    estimate_heading,
    forward_from_quat_wxyz,
    reference_sample_from_vr_frame,
    timestamp_is_fresh,
    write_json_atomic_with_backup,
)
from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.runtime.supervisor import shutdown_processes, wait_subsystem_ready
from dexmani_real.sensor.vr_receiver_process import VRReceiverConfig, vr_loop
from dexmani_real.shm.shared_storage import SharedStorage, SharedStorageConfig
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

OUTPUT_PATH = PACKAGE_DIR / "config" / "vr_transform.json"
AUDIO_PATH = ASSET_DIR / "audio" / "轴向已标定.wav"
_TRACKING_READY_TIMEOUT_S = 15.0
_OPERATOR_SETTLE_S = 3.0
_COUNTDOWN_S = 3
_POLL_INTERVAL_S = 0.01
_STATUS_INTERVAL_S = 1.0
_AUDIO_TIMEOUT_S = 10.0


@dataclass
class CollectionStats:
    repeated_reads: int = 0
    stale_frames: int = 0
    invalid_frames: int = 0


def _positive_finite_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("duration must be a number") from exc
    if not np.isfinite(seconds) or seconds <= 0.0:
        raise argparse.ArgumentTypeError("duration must be finite and positive")
    return seconds


def _latest_vr_fault(shared: SharedStorage) -> str | None:
    # Worker liveness, heartbeat, and sticky error_state are the authoritative
    # health channels. The former multi-producer status ring was not seqlock-safe.
    del shared
    return None


def _require_vr_health(shared: SharedStorage, process: Any, heartbeat_timeout_s: float) -> None:
    if bool(shared.estop_request.value):
        raise RuntimeError("e-stop was requested")
    if bool(shared.error_state.value):
        raise RuntimeError("shared error_state is set")
    fault = _latest_vr_fault(shared)
    if fault is not None:
        raise RuntimeError(fault)
    if process.exitcode is not None or not process.is_alive():
        raise RuntimeError(f"VR worker exited unexpectedly (exitcode={process.exitcode})")
    heartbeat_s = float(shared.vr_heartbeat_s.value)
    now_s = time.monotonic()
    if not np.isfinite(heartbeat_s) or heartbeat_s <= 0.0:
        raise RuntimeError("VR worker heartbeat is missing or non-finite")
    heartbeat_age_s = now_s - heartbeat_s
    if heartbeat_age_s < 0.0 or heartbeat_age_s > heartbeat_timeout_s:
        raise RuntimeError(f"VR heartbeat age {heartbeat_age_s:.2f}s exceeds the {heartbeat_timeout_s:.2f}s limit")


def _sleep_with_health(
    duration_s: float,
    shared: SharedStorage,
    process: Any,
    heartbeat_timeout_s: float,
) -> None:
    deadline = time.monotonic() + duration_s
    while True:
        _require_vr_health(shared, process, heartbeat_timeout_s)
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0.0:
            return
        time.sleep(min(_POLL_INTERVAL_S, remaining_s))


def _wait_for_reference_tracking(
    shared: SharedStorage,
    process: Any,
    *,
    reference: str,
    max_age_s: float,
    heartbeat_timeout_s: float,
) -> None:
    deadline = time.monotonic() + _TRACKING_READY_TIMEOUT_S
    last_status_s = 0.0
    while time.monotonic() < deadline:
        _require_vr_health(shared, process, heartbeat_timeout_s)
        result = shared.vr_ring.read_latest()
        if result is not None:
            data, _publish_ns, ring_sequence = result
            _source_sequence, source_timestamp_ns, quat_wxyz = reference_sample_from_vr_frame(
                data,
                ring_sequence,
                reference,
            )
            if timestamp_is_fresh(source_timestamp_ns, time.monotonic_ns(), max_age_s):
                try:
                    forward_from_quat_wxyz(quat_wxyz)
                except ValueError:
                    pass
                else:
                    return
        now_s = time.monotonic()
        if now_s - last_status_s >= _STATUS_INTERVAL_S:
            print("    waiting for a fresh reference pose...", flush=True)
            last_status_s = now_s
        time.sleep(_POLL_INTERVAL_S)
    raise RuntimeError(f"no fresh {reference} tracking pose after {_TRACKING_READY_TIMEOUT_S:.0f}s")


def _collect_forwards(
    shared: SharedStorage,
    process: Any,
    *,
    reference: str,
    duration_s: float,
    max_age_s: float,
    heartbeat_timeout_s: float,
) -> tuple[np.ndarray, CollectionStats]:
    forwards: list[np.ndarray] = []
    stats = CollectionStats()
    previous_source_sequence: int | None = None
    deadline = time.monotonic() + duration_s
    last_status_s = 0.0

    while time.monotonic() < deadline:
        _require_vr_health(shared, process, heartbeat_timeout_s)
        result = shared.vr_ring.read_latest()
        if result is None:
            time.sleep(_POLL_INTERVAL_S)
            continue
        data, _publish_ns, ring_sequence = result
        source_sequence, source_timestamp_ns, quat_wxyz = reference_sample_from_vr_frame(
            data,
            ring_sequence,
            reference,
        )
        if source_sequence == previous_source_sequence:
            stats.repeated_reads += 1
            time.sleep(_POLL_INTERVAL_S)
            continue
        previous_source_sequence = source_sequence

        if not timestamp_is_fresh(source_timestamp_ns, time.monotonic_ns(), max_age_s):
            stats.stale_frames += 1
            forwards.append(np.full(3, np.nan))
            time.sleep(_POLL_INTERVAL_S)
            continue
        try:
            forwards.append(forward_from_quat_wxyz(quat_wxyz))
        except ValueError:
            stats.invalid_frames += 1
            forwards.append(np.full(3, np.nan))

        now_s = time.monotonic()
        if now_s - last_status_s >= _STATUS_INTERVAL_S:
            print(f"    collected {len(forwards)} unique source frames...", flush=True)
            last_status_s = now_s
        time.sleep(_POLL_INTERVAL_S)

    return np.asarray(forwards, dtype=np.float64).reshape((-1, 3)), stats


def _play_completion_audio() -> None:
    if not AUDIO_PATH.exists():
        return
    player_name = "aplay" if sys.platform.startswith("linux") else "afplay" if sys.platform == "darwin" else None
    player = shutil.which(player_name) if player_name is not None else None
    if player is None:
        return
    try:
        subprocess.run(
            [player, str(AUDIO_PATH)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_AUDIO_TIMEOUT_S,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        logger.warning("calibration saved, but completion audio failed", exc_info=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VR-to-robot heading calibration")
    parser.add_argument(
        "--duration",
        type=_positive_finite_seconds,
        default=10.0,
        help="fresh-pose collection duration in seconds (default: 10)",
    )
    parser.add_argument("--port", type=int, default=None, help="VR TCP port (CLI > YAML > defaults)")
    parser.add_argument("--config", default=None, help="experiment config YAML")
    parser.add_argument(
        "--ref",
        choices=("head", "wrist"),
        default="head",
        help="head: face robot +X (default); wrist: point right-hand +X toward robot +X",
    )
    return parser


def _capture_samples(runtime: Any, *, reference: str, duration_s: float) -> tuple[np.ndarray, CollectionStats]:
    heartbeat_timeout_s = float(runtime.safety.heartbeat_timeouts["vr"])
    max_age_s = float(runtime.policy.vr_mapping.stale_threshold_s)
    readiness_timeout_s = float(runtime.safety.readiness_timeouts_s["vr"])
    ctx = mp.get_context("spawn")
    shared = SharedStorage.create(
        prefix=f"dexmani_vr_calib_{os.getpid()}",
        config=SharedStorageConfig.from_runtime(runtime),
        mp_context=ctx,
    )
    process: Any | None = None
    try:
        process = ctx.Process(
            target=vr_loop,
            args=(shared, VRReceiverConfig.from_runtime(runtime)),
            name="vr-calibration",
            daemon=False,
        )
        process.start()
        print(f"  waiting for VR data (up to {readiness_timeout_s:.0f}s); put on the headset...", flush=True)
        if not wait_subsystem_ready(
            shared,
            [("vr", shared.vr_ready, readiness_timeout_s)],
            [process],
        ):
            raise RuntimeError("VR worker did not become ready")
        _require_vr_health(shared, process, heartbeat_timeout_s)

        print("  waiting for fresh reference tracking...", flush=True)
        _wait_for_reference_tracking(
            shared,
            process,
            reference=reference,
            max_age_s=max_age_s,
            heartbeat_timeout_s=heartbeat_timeout_s,
        )
        print(f"  settle for {_OPERATOR_SETTLE_S:.0f}s and hold the reference pose...", flush=True)
        _sleep_with_health(_OPERATOR_SETTLE_S, shared, process, heartbeat_timeout_s)
        for remaining_s in range(_COUNTDOWN_S, 0, -1):
            print(f"  {remaining_s}...", flush=True)
            _sleep_with_health(1.0, shared, process, heartbeat_timeout_s)

        print(f"  collecting fresh {reference} poses...", flush=True)
        return _collect_forwards(
            shared,
            process,
            reference=reference,
            duration_s=duration_s,
            max_age_s=max_age_s,
            heartbeat_timeout_s=heartbeat_timeout_s,
        )
    finally:
        if process is None or process.pid is None:
            if not shared.close():
                raise RuntimeError("SharedStorage cleanup was incomplete")
        else:
            report = shutdown_processes(
                shared,
                [process],
                graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
            )
            if not report.shared_closed or not report.clean:
                raise RuntimeError(f"VR calibration worker did not shut down cleanly: {report}")


def _print_estimate(forwards: np.ndarray, stats: CollectionStats) -> HeadingEstimate:
    estimate = estimate_heading(forwards)
    print(
        "\nCalibration result\n"
        f"  samples:    {estimate.total_count} unique source frames\n"
        f"  projected:  {estimate.valid_count}\n"
        f"  inliers:    {estimate.inlier_count} ({estimate.inlier_ratio:.1%})\n"
        f"  skipped:    repeated_reads={stats.repeated_reads}, "
        f"stale={stats.stale_frames}, invalid={stats.invalid_frames}\n"
        f"  quality:    {estimate.quality_text}",
        flush=True,
    )
    return estimate


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        runtime = resolve_runtime_config(yaml_path=args.config, cli_overrides={"vr.port": args.port})
    except (KeyError, OSError, TypeError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        parser.error(f"invalid experiment config: {exc}")
    reference_label = "head facing robot +X" if args.ref == "head" else "right hand pointing toward robot +X"
    print(
        "\nVR heading calibration\n"
        f"  reference: {reference_label}\n"
        f"  duration:  {args.duration:.1f}s\n"
        f"  port:      {int(runtime.vr.port)}",
        flush=True,
    )
    try:
        forwards, stats = _capture_samples(runtime, reference=args.ref, duration_s=args.duration)
    except KeyboardInterrupt:
        print("\nCalibration cancelled; no file was written.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError):
        logger.error("VR heading calibration aborted; no file was written", exc_info=True)
        return 1

    estimate = _print_estimate(forwards, stats)
    if not estimate.accepted:
        for reason in estimate.reasons:
            print(f"  reject:     {reason}", file=sys.stderr)
        print(f"POOR calibration: {OUTPUT_PATH} was not changed.", file=sys.stderr)
        return 2

    assert estimate.theta_deg is not None
    assert estimate.mean_forward_xy is not None
    assert estimate.rotation_vr_to_robot is not None
    corrected = estimate.rotation_vr_to_robot @ np.array([*estimate.mean_forward_xy, 0.0])
    print(
        f"  heading:    {estimate.theta_deg:.2f}°\n"
        f"  check:      T @ forward = [{corrected[0]:.4f}, {corrected[1]:.4f}]",
        flush=True,
    )
    try:
        backup_path = write_json_atomic_with_backup(OUTPUT_PATH, build_heading_config(estimate, args.ref))
    except (OSError, TypeError, ValueError):
        logger.error("failed to publish VR heading calibration", exc_info=True)
        return 1

    print(f"  saved:      {OUTPUT_PATH}")
    if backup_path is not None:
        print(f"  backup:     {backup_path}")
    _play_completion_audio()
    return 0


__all__ = ["CollectionStats", "main"]
