#!/usr/bin/env python3
"""Hardware-gated XHand EtherCAT reconnect soak — Phase B A/B instrument (B1).

Exercises the PRODUCTION driver's ``connect → fresh-read → disconnect``
lifecycle repeatedly, with NO finger motion, to detect a reconnect regression
from the B1 single-controller connect change.  The target regression signal is
a reproducible ``write sdo failed`` / stale-OP failure on reconnect.

The script is **branch-agnostic**: run the identical command on ``main``
(baseline, two-phase discovery) and on ``b1-single-controller`` (experiment,
single-controller-per-attempt).  The branch — not this script — is the A/B
variable.  Use a distinct ``--out`` file per branch so the two runs are
comparable and never append into each other.

THIS IS A HARDWARE TOOL, NOT AN OFFLINE CHECK.  It opens the real EtherCAT
slave.  Run only with explicit hardware authorization, the hand clear of
obstacles, and no finger-motion command anywhere in the loop::

    # baseline
    git checkout main
    conda run -n real_robot python checks/hardware/xhand_reconnect_soak.py \
        --cycles 100 --out main_soak.jsonl
    # experiment (B1)
    git checkout b1-single-controller
    conda run -n real_robot python checks/hardware/xhand_reconnect_soak.py \
        --cycles 100 --out b1_soak.jsonl

Each run writes its own log directory under ``./xhand_soak_logs/`` so vendor
"write sdo failed" chatter is scoped to that run (never leaked across branches),
and every cycle records its own ``sdo_failures`` / ``open_retries`` deltas.

Failure semantics (matches §8.1's "next-session reconnect" metric):

- A single connect failure is recorded, and the next cycle is treated as a
  fresh "next-session reconnect" — exactly the self-recovery behaviour the A/B
  needs to observe, so a transient SDO glitch that recovers is counted, not
  hidden.
- After ``--max-consecutive-failures`` consecutive failures the slave is
  assumed wedged (persistent CoE dictionary lock / slave stuck in OP).  The run
  STOPS and prints a power-cycle instruction, because a 24V power-cycle is a
  human action this script cannot perform.  Resume with ``--start-cycle <n>``.

Known limitations (observed, not fixed — do not mask them for the A/B):

- ``connect_latency_ms`` includes tactile-sensor init and any retry backoff,
  not just the EtherCAT open; the open-retry count is recorded separately as
  ``open_retries`` per cycle.
- A repeated exception *during* ``open_ethercat`` (open-raise, distinct from a
  returned error code) may accumulate native handles — a pre-existing driver
  behaviour outside B1's single-variable scope.  §8.3's fd/socket observation
  covers it; watch `ls /proc/<pid>/fd | wc -l` if open-raise recurs.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

# Per-process log directory so each soak run's vendor chatter is scoped to THIS
# run and never leaked across baseline/experiment invocations.  Must be set
# before the driver import: utils.log creates its file handler lazily at the
# first get_logger() call (driver-module import time), reading DEXMANI_LOG_DIR.
# Override rather than setdefault — a pre-set DEXMANI_LOG_DIR would scatter the
# "write sdo failed" lines and invalidate the per-cycle counts below.
_RUN_LOG_DIR = Path.cwd() / "xhand_soak_logs" / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
os.environ["DEXMANI_LOG_DIR"] = str(_RUN_LOG_DIR)

import numpy as np  # noqa: E402

from dexmani_real.robot.xhand import XHand, XHandConfig  # noqa: E402


def _build_config(args: argparse.Namespace) -> XHandConfig:
    """Build the driver config; device_name=None keeps discovery mode (B1 path)."""
    return XHandConfig(
        comm_type=args.comm_type,
        device_name=args.device_name or None,
        ethercat_slave_position=args.ethercat_slave_position,
    )


def _run_cycle(config: XHandConfig, cycle: int, interval_s: float) -> dict[str, object]:
    """One create→connect→fresh-read→disconnect→destroy cycle.  No motion."""
    result: dict[str, object] = {
        "cycle": cycle,
        "connect_ok": False,
        "read_ok": False,
        "sdo_failures": 0,
        "open_retries": 0,
        "connect_latency_ms": None,
        "disconnect_latency_ms": None,
        "error_code": None,
        "error_message": "",
        "device_name": None,
        "sdk_version": None,
        "serial_number": None,
        "hand_type": None,
    }

    hand = XHand(config)
    t0 = time.monotonic()
    try:
        connect_ok = hand.connect()
    except Exception as exc:  # discovery-raise / open-raise propagate out of connect()
        result["connect_latency_ms"] = round((time.monotonic() - t0) * 1000.0, 1)
        result["error_code"] = hand.last_error_code
        result["error_message"] = f"connect raised: {exc}"
        del hand
        return result
    result["connect_latency_ms"] = round((time.monotonic() - t0) * 1000.0, 1)
    result["connect_ok"] = bool(connect_ok)

    if not connect_ok:
        result["error_code"] = hand.last_error_code
        result["error_message"] = hand.last_error_message
        del hand
        return result

    # Identity / environment record (§8.1) — stable across cycles.
    result["device_name"] = hand.device_name
    result["sdk_version"] = hand.device_identity.get("sdk_version")
    result["serial_number"] = hand.device_identity.get("serial_number")
    result["hand_type"] = hand.device_identity.get("hand_type")

    # One fresh read (read-only health check, §8.2).  No send_command / motion.
    try:
        state = hand.get_state(full=False, force_update=True)
        qpos = np.asarray(state.get("qpos"), dtype=np.float64)
        result["read_ok"] = bool(qpos.size == 12 and np.all(np.isfinite(qpos)))
    except Exception as exc:
        result["read_ok"] = False
        result["error_message"] = f"fresh read raised: {exc}"

    t1 = time.monotonic()
    try:
        hand.disconnect()
    except Exception as exc:
        result["error_message"] = f"disconnect raised: {exc}"
    result["disconnect_latency_ms"] = round((time.monotonic() - t1) * 1000.0, 1)

    del hand
    if interval_s > 0:
        time.sleep(interval_s)
    return result


def _read_log_text() -> str:
    """Concatenate this run's driver log files (one file per run, in practice)."""
    parts: list[str] = []
    for path in sorted(glob.glob(str(_RUN_LOG_DIR / "*.log"))):
        try:
            parts.append(Path(path).read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n".join(parts)


def _existing_cycles(out_path: Path) -> set[int]:
    """Cycle IDs already present in the output file (for a resume-overlap guard)."""
    if not out_path.exists():
        return set()
    ids: set[int] = set()
    try:
        lines = out_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ids
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        cycle = rec.get("cycle")
        if isinstance(cycle, int):
            ids.add(cycle)
    return ids


def _summarize(results: list[dict[str, object]], stop_reason: str) -> None:
    if not results:
        return
    ok = [r for r in results if r["connect_ok"]]
    fail = [r for r in results if not r["connect_ok"]]
    read_fail = [r for r in results if r["connect_ok"] and not r["read_ok"]]
    conn_lat = [r["connect_latency_ms"] for r in results if r["connect_latency_ms"] is not None]
    disc_lat = [r["disconnect_latency_ms"] for r in results if r["disconnect_latency_ms"] is not None]
    codes = Counter(str(r["error_code"]) for r in fail)
    sdo_cycles = sorted(r["cycle"] for r in results if r["sdo_failures"] > 0)
    retry_cycles = sorted(r["cycle"] for r in results if r["open_retries"] > 0)
    total_sdo = sum(int(r["sdo_failures"]) for r in results)
    total_retry = sum(int(r["open_retries"]) for r in results)

    print("\n=== SOAK SUMMARY ===")
    print(f"stop reason           : {stop_reason}")
    print(f"cycles attempted      : {len(results)}")
    print(f"connect success       : {len(ok)}")
    print(f"connect fail          : {len(fail)}  (codes {dict(codes)})")
    print(f"fresh-read fail       : {len(read_fail)}")
    print(f"'write sdo failed'    : {total_sdo}  (cycles {sdo_cycles or 'none'})")
    print(f"open retries          : {total_retry}  (cycles {retry_cycles or 'none'})")
    if conn_lat:
        print(f"connect latency ms    : avg={sum(conn_lat) / len(conn_lat):.1f} max={max(conn_lat):.1f}")
    if disc_lat:
        print(f"disconnect latency ms : avg={sum(disc_lat) / len(disc_lat):.1f} max={max(disc_lat):.1f}")
    if ok:
        first = ok[0]
        print(
            "identity              : "
            f"sdk={first.get('sdk_version')} type={first.get('hand_type')} "
            f"serial={first.get('serial_number')} device={first.get('device_name')}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cycles", type=int, default=100, help="target cycle count (default 100)")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between cycles (default 1.0)")
    parser.add_argument("--device-name", default=None, help="configured EtherCAT device name; omit for discovery")
    parser.add_argument("--comm-type", default="EtherCAT", help="EtherCAT or RS485 (default EtherCAT)")
    parser.add_argument("--ethercat-slave-position", type=int, default=-1, help="slave position or -1 (default)")
    parser.add_argument("--out", default="xhand_soak.jsonl", help="JSONL results path (append mode)")
    parser.add_argument("--start-cycle", type=int, default=1, help="resume cycle index after a power-cycle")
    parser.add_argument("--max-consecutive-failures", type=int, default=3, help="wedge threshold (default 3)")
    args = parser.parse_args(argv)

    if args.cycles < 1 or args.max_consecutive_failures < 1 or args.start_cycle < 1:
        parser.error("--cycles, --start-cycle and --max-consecutive-failures must be >= 1")

    config = _build_config(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume-overlap guard: refuse to silently double-count cycles already in
    # --out (also catches accidentally reusing the same file across branches).
    overlap = sorted(c for c in range(args.start_cycle, args.start_cycle + args.cycles) if c in _existing_cycles(out_path))
    if overlap:
        parser.error(
            f"--start-cycle {args.start_cycle} overlaps cycles already recorded in {out_path}: "
            f"{overlap[:8]}{'...' if len(overlap) > 8 else ''}. Use a higher --start-cycle, or a "
            f"fresh --out (one file per branch)."
        )

    print(
        f"# XHand reconnect soak — {args.comm_type} / device_name={args.device_name or '(discovery)'} "
        f"/ cycles={args.cycles}"
    )
    print(f"# results -> {out_path}  driver log dir -> {_RUN_LOG_DIR}")

    results: list[dict[str, object]] = []
    consecutive_failures = 0
    prev_sdo = 0
    prev_retry = 0
    stop_reason = "completed"

    with out_path.open("a", encoding="utf-8") as fh:
        for cycle in range(args.start_cycle, args.start_cycle + args.cycles):
            result = _run_cycle(config, cycle, args.interval)

            # Attribute this cycle's share of the monotonically-growing driver
            # log (write sdo failed / succeeded-on-attempt) for per-cycle signal.
            log_text = _read_log_text().lower()
            sdo_now = log_text.count("write sdo failed")
            retry_now = log_text.count("succeeded on attempt")
            result["sdo_failures"] = max(0, sdo_now - prev_sdo)
            result["open_retries"] = max(0, retry_now - prev_retry)
            prev_sdo, prev_retry = sdo_now, retry_now

            results.append(result)
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            fh.flush()

            status = "OK" if result["connect_ok"] else f"FAIL(code={result['error_code']})"
            read = "read-ok" if result["read_ok"] else "read-fail"
            print(
                f"[{cycle}] connect={status} {read} "
                f"t_conn={result['connect_latency_ms']}ms t_disc={result['disconnect_latency_ms']}ms "
                f"sdo={result['sdo_failures']} rtry={result['open_retries']}"
            )

            if result["connect_ok"]:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= args.max_consecutive_failures:
                    stop_reason = (
                        f"wedged (>= {args.max_consecutive_failures} consecutive connect failures)"
                    )
                    print(
                        "\n!! Slave appears wedged — power-cycle the XHand "
                        "(disconnect/reconnect 24V, wait >=5s), then resume with "
                        f"--start-cycle {cycle + 1}.\n"
                    )
                    break

    _summarize(results, stop_reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
