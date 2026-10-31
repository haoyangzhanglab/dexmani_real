#!/usr/bin/env python3
"""连接 XHand1，顺序执行 HOME → fist → palm → V → OK → HOME，到位后各停留 1 秒。"""

from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import stat
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dexmani_real.config.real import load_config
from dexmani_real.config.xhand import HandConfig
from dexmani_real.robot.commands import OperationCancelled, validate_hand_target
from dexmani_real.robot.model import (
    HAND_DOF,
    XHAND_TACTILE_SENSOR_INDEX_BY_FINGER_ID,
)
from dexmani_real.robot.xhand.driver import XHand
from dexmani_real.robot.xhand.feedback import XHandState
from dexmani_real.utils.log import configure_logging

HARDWARE_WORKER_ARG = "--_xhand-hardware-worker"
FEEDBACK_POLL_INTERVAL_S = 0.01
CONVERGENCE_SAMPLES = 3
ACTION_DWELL_S = 1.0

# XHand1 diagnostic targets in degrees; HOME/palm use the current configuration.
PRESET_QPOS_DEG = {
    "fist": (
        11.85,
        74.58,
        40,
        -3.08,
        106.02,
        109.5,
        109.75,
        107.56,
        107.66,
        109.5,
        109.1,
        109.15,
    ),
    "v": (38.32, 90, 52.08, 6.21, 2.6, 5.0, 2.1, 5.0, 109.5, 109.5, 109.5, 109.23),
    # 简化 OK：拇指、食指轻弯，其余手指张开，远端关节保留配置要求的 5° 下限。
    "ok": (49.0, 43.0, 48.0, 0.0, 48.0, 48.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0),
}


def action_sequence(config: HandConfig) -> list[tuple[str, np.ndarray]]:
    actions = (
        ("HOME", config.home_qpos_deg),
        ("fist", PRESET_QPOS_DEG["fist"]),
        ("palm", config.home_qpos_deg),
        ("V", PRESET_QPOS_DEG["v"]),
        ("OK", PRESET_QPOS_DEG["ok"]),
        ("HOME", config.home_qpos_deg),
    )
    return [
        (
            name,
            validate_hand_target(
                np.deg2rad(values), config.qpos_min_rad, config.qpos_max_rad,
            ),
        )
        for name, values in actions
    ]


class XHandControlExample(XHand):
    """Inspect and exercise the same read/send/stop path used by the runtime."""

    def __init__(self, config: HandConfig, *, cancel_requested=lambda: False) -> None:
        super().__init__(config)
        self.cancel_requested = cancel_requested
        self.motion_pending = False

    def check_cancelled(self) -> None:
        if self.cancel_requested():
            raise OperationCancelled("XHand action sequence interrupted")

    @staticmethod
    def header(title: str) -> None:
        print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")

    def read_sdk_version(self) -> None:
        self.header("SDK versions")
        print(f"  Software SDK: {self.control.get_sdk_version()}")

        error_struct, version = self.control.read_version(self.cfg.device_id, 0)
        print(f"  Hardware SDK: {version}  (error_code={error_struct.error_code})")

    def read_device_info(self) -> None:
        self.header("Device info")
        error_struct, info = self.control.read_device_info(self.cfg.device_id)
        if error_struct.error_code != 0 or info is None:
            print(f"  Device info unavailable (error_code={error_struct.error_code})")
            return
        print(f"  serial_number: {''.join(info.serial_number[:16])}")
        print(f"  hand_id:       {info.hand_id}")
        print(f"  ev_hand:       {info.ev_hand}")

        error_struct, hand_type = self.control.get_hand_type(self.cfg.device_id)
        if error_struct.error_code == 0:
            print(f"  hand_type:     {hand_type}")
        else:
            print(f"  hand_type unavailable (error_code={error_struct.error_code})")

    def read_state(self, finger_id: int = 2) -> None:
        self.header(f"Read state (finger {finger_id})")
        state = self.read_fresh_state()
        print("  joint positions (deg, SDK ids 0..11):", np.rad2deg(state.qpos).round(3).tolist())
        if finger_id in XHAND_TACTILE_SENSOR_INDEX_BY_FINGER_ID:
            index = XHAND_TACTILE_SENSOR_INDEX_BY_FINGER_ID[finger_id]
            force = state.tactile_aggregate
            print("  tactile aggregate (SDK units):", force[index].tolist() if force is not None else "unavailable")
            if state.tactile_dense is not None:
                dense = state.tactile_dense[index]
                print(f"  tactile dense (SDK units): points={len(dense)} max_abs={np.abs(dense).max():.3f}")
            else:
                print("  tactile dense (SDK units): unavailable")

    def read_fresh_state(self, *, action_deadline_ns: int | None = None) -> XHandState:
        """CRC 仅重读，所有尝试共用反馈与动作的固定截止时间。"""
        deadline_ns = time.monotonic_ns() + int(self.cfg.feedback_max_age_s * 1e9)
        if action_deadline_ns is not None:
            deadline_ns = min(deadline_ns, action_deadline_ns)
        while True:
            self.check_cancelled()
            if time.monotonic_ns() >= deadline_ns:
                raise RuntimeError("XHand feedback deadline expired before read")
            state = self.get_state()
            self.check_cancelled()
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                raise RuntimeError("XHand feedback deadline expired during read")
            if state is not None:
                return state
            time.sleep(min(FEEDBACK_POLL_INTERVAL_S, remaining_ns / 1e9))

    def execute_action(self, target: np.ndarray) -> None:
        self.read_fresh_state()
        self.check_cancelled()
        deadline_ns = time.monotonic_ns() + int(self.cfg.home_timeout_s * 1e9)
        self.motion_pending = True  # SDK exceptions do not establish non-delivery.
        self.send_action(target)
        print("  final target submission attempted; checking measured convergence")
        tolerance_rad = np.deg2rad(self.cfg.home_tolerance_deg)
        consecutive = 0
        while time.monotonic_ns() < deadline_ns:
            # Freshness belongs to each newly received sample, independently of previous read/send latency.
            measured = self.read_fresh_state(action_deadline_ns=deadline_ns).qpos
            if time.monotonic_ns() >= deadline_ns:
                break
            error_rad = float(np.max(np.abs(measured - target)))
            consecutive = consecutive + 1 if error_rad <= tolerance_rad else 0
            if consecutive >= CONVERGENCE_SAMPLES:
                print(f"  Measured target reached; max error={np.rad2deg(error_rad):.3f} deg")
                return
            time.sleep(FEEDBACK_POLL_INTERVAL_S)
        raise RuntimeError(f"action did not converge within {self.cfg.home_timeout_s:g}s")

    def stop_action(self) -> None:
        if not self.motion_pending:
            return
        started_ns = time.monotonic_ns()
        measured = None
        try:
            # Cleanup must still read/stop after the operator has cancelled.
            state = self.get_state()
            measured = None if state is None else state.qpos
        finally:
            self.stop(measured, timestamp_ns=started_ns)
            self.motion_pending = False
        print("  Stop command accepted by SDK")


def serial_port_problem(serial_port: str) -> str | None:
    port = Path(serial_port)
    try:
        mode = port.stat().st_mode
    except FileNotFoundError:
        return f"serial port {serial_port} does not exist (or its symlink target disappeared)"
    except OSError as exc:
        return f"cannot stat serial port {serial_port}: {exc}"

    if not stat.S_ISCHR(mode):
        return f"serial port {serial_port} is not a character device"
    if not os.access(port, os.R_OK | os.W_OK):
        return f"serial port {serial_port} is not readable and writable by the current user"
    return None


def run_hardware_session(config: HandConfig, *, finger_id: int, read_only: bool) -> int:
    # Validate every endpoint before connecting, including narrowed experiment limits.
    actions = [] if read_only else action_sequence(config)
    configure_logging()
    interrupted_signal = None
    cleaning_up = False

    def request_interrupt(signum, _):
        nonlocal interrupted_signal
        interrupted_signal = signum
        if not cleaning_up:
            raise OperationCancelled("XHand action sequence interrupted")

    hand = XHandControlExample(config, cancel_requested=lambda: interrupted_signal is not None)
    previous_handlers = {
        signum: signal.signal(signum, request_interrupt)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    result = 0
    try:
        if config.comm_type == "serial" and config.device_name is not None:
            problem = serial_port_problem(config.device_name)
            if problem is not None:
                raise RuntimeError(problem)
        hand.check_cancelled()
        hand.connect(cancel_requested=hand.cancel_requested)
        hand.check_cancelled()
        hand.read_sdk_version()
        hand.read_device_info()
        hand.check_cancelled()
        hand.read_state(finger_id)
        for name, target in actions:
            print(f"\n  -> {name} (degrees): {np.rad2deg(target).round(3).tolist()}")
            hand.execute_action(target)
            time.sleep(ACTION_DWELL_S)
    except OperationCancelled:
        print("\nSequence interrupted; stopping any attempted action before closing")
    except Exception as exc:
        print(f"\nXHand diagnostic failed: {exc}", file=sys.stderr)
        result = 2
    finally:
        cleaning_up = True
        try:
            hand.stop_action()
        except Exception as exc:
            print(f"  Stop failed; hardware stop is unconfirmed: {exc}", file=sys.stderr)
            result = 2
        try:
            hand.disconnect()
        except Exception as exc:
            print(f"  Device close failed: {exc}", file=sys.stderr)
            result = 2
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
    return result or (128 + interrupted_signal if interrupted_signal is not None else 0)


def run_isolated_hardware_session(argv: list[str]) -> int:
    command = [sys.executable, str(Path(__file__).resolve()), *argv, HARDWARE_WORKER_ARG]
    worker = subprocess.Popen(command)

    def forward_interrupt(signum, _):
        try:
            worker.send_signal(signum)
        except ProcessLookupError:
            pass

    previous_handlers = {
        signum: signal.signal(signum, forward_interrupt)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        returncode = worker.wait()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if returncode == -signal.SIGABRT:
        print(
            "\nXHand SDK worker aborted while its native communication thread was running.\n"
            "The launcher remained alive and no core file was written. Hardware stop is\n"
            "unconfirmed. Check hand power, wiring, device permissions and port ownership.",
            file=sys.stderr,
        )
        return 1
    if returncode < 0:
        signal_number = -returncode
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = f"signal {signal_number}"
        print(
            f"\nXHand SDK worker terminated by {signal_name}; hardware stop is unconfirmed.",
            file=sys.stderr,
        )
        return 1
    return returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, help="Trusted Python experiment file; uses CONFIG.hand"
    )
    parser.add_argument(
        "--finger-id", type=int, default=5, help="Joint id inspected for tactile health"
    )
    parser.add_argument("--read-only", action="store_true", help="Read diagnostics once, then exit")
    parser.add_argument(
        "--print-config", action="store_true", help="Print hand settings without hardware"
    )
    parser.add_argument(
        HARDWARE_WORKER_ARG,
        dest="xhand_hardware_worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config).hand
        if not args.print_config:
            config.validate()
        if not 0 <= args.finger_id < HAND_DOF:
            raise ValueError(f"finger-id must be in 0..{HAND_DOF - 1}")
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.print_config:
        print(json.dumps({"hand": asdict(config)}, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    if args.xhand_hardware_worker:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        return run_hardware_session(config, finger_id=args.finger_id, read_only=args.read_only)
    return run_isolated_hardware_session(argv)


if __name__ == "__main__":
    raise SystemExit(main())
