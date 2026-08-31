#!/usr/bin/env python3
"""Seal one completed H4 runtime receipt with its terminal transcript.

This is a post-shutdown, no-hardware step.  It binds the coordinator's atomic
receipt to the immutable terminal log and a separately prepared operator
checklist, so a future review never needs to infer which transcript belonged to
which physical publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dexmani_real.utils.log import write_json_evidence_manifest

_REQUIRED_LOG_MARKERS = {
    "startup_reset_home_accepted": "hand_loop: startup reset-home command accepted",
    "hand_home_accepted": "hand: home command accepted (action_id=",
    "arm_home_path_selected": "arm: home path selected=",
    "arm_home_reached": "arm: home reached",
    "physical_home_completed": "operator: physical home sequence completed; press B to start",
    "running": "coordinator_loop: RUNNING",
    "h4_publication": "coordinator: execute published action_id=",
    "bounded_stop": "coordinator: policy run stopped: H4 publication bound reached",
    "session_end": "── Session End ──",
    "disarmed": "safety=DISARMED",
    "supervisor_normal": "supervisor_normal=True",
    "completed": "execute_completed=True",
    "clean_shutdown": "clean=True",
}
_ARM_EXIT_RE = re.compile(
    r"arm_loop: exited \(servo_calls=(\d+), duplicate_skips=(\d+)\)"
)
_HAND_EXIT_RE = re.compile(
    r"hand_loop: exited \(sdk_send_attempts=(\d+), exact_target_accepts=(\d+), "
    r"crc_unconfirmed=(\d+), duplicate_skips=(\d+), sdk_rejections=(\d+)\)"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _validate_runtime_receipt(receipt: dict[str, Any]) -> None:
    if receipt.get("execution_mode") != "execute":
        raise ValueError("H4 evidence requires an execute runtime receipt")
    if receipt.get("completed") is not True:
        raise ValueError("H4 runtime receipt is not completed")
    if receipt.get("max_published_endpoints") != 1:
        raise ValueError("H4 runtime receipt does not retain the one-publication bound")
    if receipt.get("coupled_command_writes") != 1:
        raise ValueError("H4 runtime receipt does not prove exactly one coupled write")
    if receipt.get("within_publication_bound") is not True:
        raise ValueError("H4 runtime receipt exceeds its publication bound")
    if not isinstance(receipt.get("acknowledged_action_id"), int):
        raise ValueError("H4 runtime receipt has no worker-acknowledged action")
    metrics = receipt.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("physical_home_completed") != 1:
        raise ValueError("H4 runtime receipt does not prove the physical home gate")
    timeline = receipt.get("timeline_monotonic_ns")
    if not isinstance(timeline, dict):
        raise ValueError("H4 runtime receipt has no monotonic execution timeline")
    required = (
        "run_started",
        "first_publication",
        "last_publication",
        "receipt_emitted",
    )
    values: list[int] = []
    for name in required:
        value = timeline.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("H4 runtime receipt has an incomplete monotonic timeline")
        values.append(value)
    if values != sorted(values):
        raise ValueError("H4 runtime receipt timeline is not monotonic")


def _marker_lines(log_text: str) -> dict[str, int]:
    markers: dict[str, int] = {}
    for line_number, line in enumerate(log_text.splitlines(), start=1):
        for name, marker in _REQUIRED_LOG_MARKERS.items():
            if name not in markers and marker in line:
                markers[name] = line_number
    missing = sorted(set(_REQUIRED_LOG_MARKERS) - set(markers))
    if missing:
        raise ValueError(
            "terminal log is missing required H4 markers: " + ", ".join(missing)
        )
    return markers


def _worker_command_counts(log_text: str) -> dict[str, dict[str, int]]:
    arm_matches = _ARM_EXIT_RE.findall(log_text)
    hand_matches = _HAND_EXIT_RE.findall(log_text)
    if len(arm_matches) != 1 or len(hand_matches) != 1:
        raise ValueError(
            "terminal log must contain exactly one arm and hand worker exit summary"
        )
    arm_servo_calls, arm_duplicate_skips = (int(value) for value in arm_matches[0])
    (
        hand_send_attempts,
        hand_exact_target_accepts,
        hand_crc_unconfirmed,
        hand_duplicate_skips,
        hand_sdk_rejections,
    ) = (int(value) for value in hand_matches[0])
    if (
        arm_servo_calls != 1
        or hand_exact_target_accepts != 1
        or hand_sdk_rejections != 0
    ):
        raise ValueError(
            "worker SDK evidence does not prove one successful coupled H4 endpoint"
        )
    return {
        "arm": {
            "sdk_servo_calls": arm_servo_calls,
            "duplicate_skips": arm_duplicate_skips,
        },
        "hand": {
            "sdk_send_attempts": hand_send_attempts,
            "exact_target_accepts": hand_exact_target_accepts,
            "crc_unconfirmed": hand_crc_unconfirmed,
            "duplicate_skips": hand_duplicate_skips,
            "sdk_rejections": hand_sdk_rejections,
        },
    }


def _validate_operator_record(record: dict[str, Any]) -> None:
    """Require the human-only facts that runtime code cannot observe."""
    for name in ("operator", "scene", "authorization"):
        value = record.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"operator record requires non-empty {name!r}")
    if record.get("e_stop_ready") is not True:
        raise ValueError("operator record requires e_stop_ready=true")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--terminal-log", type=Path, required=True)
    parser.add_argument(
        "--operator-record",
        type=Path,
        required=True,
        help="JSON object recording operator, scene, e-stop, and authorization confirmation",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    receipt_path = args.runtime_receipt.resolve(strict=True)
    log_path = args.terminal_log.resolve(strict=True)
    record_path = args.operator_record.resolve(strict=True)
    receipt = _read_json_object(receipt_path, label="runtime receipt")
    _validate_runtime_receipt(receipt)
    operator_record = _read_json_object(record_path, label="operator record")
    try:
        _validate_operator_record(operator_record)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        log_text = log_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        parser.error(f"cannot read terminal log: {exc}")
    marker_lines = _marker_lines(log_text)
    worker_command_counts = _worker_command_counts(log_text)

    payload = json.dumps(
        {
            "schema_version": 1,
            "sealed_monotonic_ns": time.monotonic_ns(),
            "sealed_utc_ns": time.time_ns(),
            "runtime_receipt": {
                "path": str(receipt_path),
                "sha256": _sha256(receipt_path),
                "execution_mode": receipt["execution_mode"],
                "run_generation": receipt["run_generation"],
                "acknowledged_action_id": receipt["acknowledged_action_id"],
                "timeline_monotonic_ns": receipt["timeline_monotonic_ns"],
            },
            "terminal_log": {
                "path": str(log_path),
                "size_bytes": log_path.stat().st_size,
                "sha256": _sha256(log_path),
                "required_marker_lines": marker_lines,
            },
            "worker_command_counts": worker_command_counts,
            "operator_record": operator_record,
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    destination = write_json_evidence_manifest(
        args.output_dir or receipt_path.parent,
        payload,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
