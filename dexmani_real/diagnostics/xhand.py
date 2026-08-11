"""Bounded, read-only XHand diagnostics through the production hand worker."""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_MONITOR_POLL_S = 0.05


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(f"must be finite and > 0, got {text!r}")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only XHand worker diagnostic")
    parser.add_argument("--config", type=Path, default=None, help="Experiment YAML")
    parser.add_argument("--duration-s", type=_positive_float, default=10.0, help="Bounded monitoring duration")
    parser.add_argument("--status-interval-s", type=_positive_float, default=1.0, help="Status print interval")
    return parser


def _decode_identity(shared: Any) -> str:
    raw = bytes(shared.hand_device_identity.value).rstrip(b"\x00")
    return raw.decode("utf-8", errors="replace") if raw else "unknown"


def _heartbeat_age_s(heartbeat_s: float, now_s: float) -> float:
    """Return heartbeat age, rejecting invalid or future timestamps."""
    if not math.isfinite(heartbeat_s) or heartbeat_s <= 0.0:
        raise ValueError("heartbeat timestamp is non-finite or non-positive")
    if not math.isfinite(now_s):
        raise ValueError("heartbeat check time is non-finite")
    age_s = now_s - heartbeat_s
    if age_s < 0.0:
        raise ValueError(f"heartbeat timestamp is in the future by {-age_s:.6f}s")
    return age_s


def _validate_state(state: np.ndarray, *, now_ns: int, max_age_s: float) -> str | None:
    record = state[0]
    if not bool(record["connected"]):
        return "hand feedback reports disconnected"
    if not bool(record["state_valid"]) or not bool(record["read_healthy"]) or not bool(record["send_healthy"]):
        return "hand feedback is invalid or worker health is degraded"
    if bool(record["error_state"]):
        return "hand feedback reports an SDK/device error"
    if bool(record["qpos_stale"]):
        return "hand joint feedback is stale"
    source_ns = int(record["source_monotonic_ns"])
    age_ns = now_ns - source_ns
    if source_ns <= 0 or age_ns < 0 or age_ns > int(max_age_s * 1e9):
        return f"hand feedback timestamp is stale or invalid (age={age_ns / 1e9:.3f}s)"
    for name in ("qpos", "current", "tactile_sum"):
        if not np.all(np.isfinite(np.asarray(record[name], dtype=np.float64))):
            return f"hand feedback field {name} contains NaN/Inf"
    for name in ("commboard_err", "jointboard_err", "tipboard_err"):
        if np.any(np.asarray(record[name], dtype=np.int64) != 0):
            return f"hand feedback field {name} contains a board fault"
    return None


def _monitor(
    shared: Any, process: Any, *, duration_s: float, status_interval_s: float, heartbeat_timeout_s: float
) -> int:
    from dexmani_real.robot.safety import SafetyState

    if not math.isfinite(heartbeat_timeout_s) or heartbeat_timeout_s <= 0.0:
        raise ValueError("heartbeat_timeout_s must be finite and positive")
    started_s = time.monotonic()
    deadline_s = started_s + duration_s
    next_status_s = started_s
    samples = 0
    while time.monotonic() < deadline_s:
        now_s = time.monotonic()
        if not process.is_alive():
            logger.error("XHand worker exited unexpectedly with code %s", process.exitcode)
            return 1
        if bool(shared.error_state.value):
            logger.error("XHand worker latched error_state")
            return 1
        if bool(shared.estop_request.value):
            logger.error("Unexpected e-stop request during read-only diagnostic")
            return 1
        if int(shared.safety_state.value) != int(SafetyState.DISARMED):
            logger.error("Read-only diagnostic left DISARMED state")
            return 1

        heartbeat_s = float(shared.hand_heartbeat_s.value)
        heartbeat_checked_s = time.monotonic()
        try:
            heartbeat_age_s = _heartbeat_age_s(heartbeat_s, heartbeat_checked_s)
        except ValueError as exc:
            logger.error("XHand heartbeat invalid: %s", exc)
            return 1
        if heartbeat_age_s > heartbeat_timeout_s:
            logger.error("XHand heartbeat stale (age=%.3fs)", heartbeat_age_s)
            return 1

        result = shared.hand_state_ring.read_latest()
        if result is None:
            logger.error("XHand ready worker has no state frame")
            return 1
        state = result[0]
        state_error = _validate_state(state, now_ns=time.monotonic_ns(), max_age_s=heartbeat_timeout_s)
        if state_error is not None:
            logger.error("XHand diagnostic failed: %s", state_error)
            return 1
        samples += 1

        if now_s >= next_status_s:
            record = state[0]
            qpos_deg = np.rad2deg(np.asarray(record["qpos"], dtype=np.float64))
            contact = np.asarray(record["tactile_contact"], dtype=bool)
            print(
                f"t={now_s - started_s:5.1f}s heartbeat_age={heartbeat_age_s:.3f}s "
                f"qpos_deg={np.round(qpos_deg, 1).tolist()} contact={contact.astype(int).tolist()}",
                flush=True,
            )
            next_status_s = now_s + status_interval_s
        time.sleep(min(_MONITOR_POLL_S, max(0.0, deadline_s - time.monotonic())))

    if samples == 0:
        logger.error("XHand diagnostic received no valid state samples")
        return 1
    print(f"Read-only XHand diagnostic passed: samples={samples} duration={time.monotonic() - started_s:.2f}s")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    runtime: Any | None = None
    shared: Any | None = None
    process: Any | None = None
    exit_code = 1
    try:
        from dexmani_real.config.runtime import resolve_runtime_config
        from dexmani_real.robot.hand_process import HandProcessConfig, hand_loop
        from dexmani_real.runtime.supervisor import shutdown_processes, wait_subsystem_ready
        from dexmani_real.shm.shared_storage import SharedStorage, SharedStorageConfig

        runtime = resolve_runtime_config(yaml_path=args.config)
        ctx = mp.get_context("spawn")
        shared = SharedStorage.create(
            prefix=f"dexmani_xhand_diag_{os.getpid()}",
            config=SharedStorageConfig.from_runtime(runtime),
            mp_context=ctx,
        )
        config = HandProcessConfig.from_runtime(runtime, startup_failure_is_fatal=True)
        process = ctx.Process(target=hand_loop, args=(shared, config), name="hand-diagnostic", daemon=False)
        process.start()
        ready_timeout_s = float(runtime.safety.readiness_timeouts_s["hand"])
        if wait_subsystem_ready(shared, [("hand", shared.hand_ready, ready_timeout_s)], [process]):
            print(f"XHand identity: {_decode_identity(shared)}")
            print("Safety state remains DISARMED; no joint target will be published.")
            exit_code = _monitor(
                shared,
                process,
                duration_s=args.duration_s,
                status_interval_s=args.status_interval_s,
                heartbeat_timeout_s=float(runtime.safety.heartbeat_timeouts["hand"]),
            )
    except KeyboardInterrupt:
        print("Interrupted; stopping the read-only worker.")
        exit_code = 130
    except Exception:
        logger.error("XHand diagnostic failed", exc_info=True)
        exit_code = 1
    finally:
        if shared is not None:
            if process is not None and process.pid is not None:
                assert runtime is not None
                try:
                    report = shutdown_processes(
                        shared,
                        [process],
                        graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
                    )
                    if not report.clean:
                        logger.error("XHand diagnostic shutdown was not clean: %s", report)
                        exit_code = 1
                except Exception:
                    logger.critical(
                        "XHand worker could not be confirmed stopped; shared memory remains linked", exc_info=True
                    )
                    exit_code = 1
            else:
                try:
                    if not shared.close():
                        logger.error("SharedStorage cleanup was incomplete")
                        exit_code = 1
                except Exception:
                    logger.warning("SharedStorage cleanup failed", exc_info=True)
                    exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
