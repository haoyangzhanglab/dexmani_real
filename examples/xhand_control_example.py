#!/usr/bin/env python3
"""连接 XHand1，顺序执行 HOME → fist → palm → V → OK → HOME，到位后各停留 1 秒。

python examples/xhand_control_example.py
python examples/xhand_control_example.py --config local.yaml --read-only
python examples/xhand_control_example.py --print-config

参数：--config 指定手部配置；--read-only 仍连接设备但不发动作，--print-config 不连接设备。
读写与正式运行共用驱动；普通发送遇到 CRC 后继续读取关节反馈，不额外重发动作。
"""

from __future__ import annotations

import argparse
import os
import signal
import stat
import subprocess
import sys
import time
from concurrent.futures import CancelledError
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import yaml

from dexmani_real.config.experiment import load_experiment_config
from dexmani_real.config.hardware import HandParams
from dexmani_real.robot.commands import DispatchStatus
from dexmani_real.robot.drivers.xhand import XHand
from dexmani_real.robot.model import (
    HAND_DOF,
    XHAND_TACTILE_SENSOR_INDEX_BY_FINGER_ID,
)
from dexmani_real.utils.limits import validate_hand_command_bounds
from dexmani_real.utils.log import configure_logging

_HARDWARE_WORKER_ARG = "--_xhand-hardware-worker"
_FEEDBACK_POLL_INTERVAL_S = 0.01
_CONVERGENCE_SAMPLES = 3
_ACTION_DWELL_S = 1.0

# XHand1 example targets in degrees; HOME/palm use the current configuration.
_PRESET_QPOS_DEG = {
    "fist": (
        11.85, 74.58, 40, -3.08, 106.02, 109.5,
        109.75, 107.56, 107.66, 109.5, 109.1, 109.15,
    ),
    "v": (38.32, 90, 52.08, 6.21, 2.6, 5.0, 2.1, 5.0, 109.5, 109.5, 109.5, 109.23),
    "ok": (
        45.88, 41.54, 67.35, 2.22, 80.45, 70.82,
        31.37, 10.39, 13.69, 16.88, 1.39, 10.55,
    ),
}


def _validate_target(qpos_rad, config: HandParams) -> np.ndarray:
    return validate_hand_command_bounds(
        qpos_rad,
        config.qpos_min_rad,
        config.qpos_max_rad,
        config.mechanical_qpos_min_rad,
        config.mechanical_qpos_max_rad,
    )


def _action_sequence(config: HandParams) -> list[tuple[str, np.ndarray]]:
    actions = (
        ("HOME", config.home_qpos_deg),
        ("fist", _PRESET_QPOS_DEG["fist"]),
        ("palm", config.home_qpos_deg),
        ("V", _PRESET_QPOS_DEG["v"]),
        ("OK", _PRESET_QPOS_DEG["ok"]),
        ("HOME", config.home_qpos_deg),
    )
    return [(name, _validate_target(np.deg2rad(values), config)) for name, values in actions]


class XHandControlExample(XHand):
    """Inspect and exercise the same read/send/stop path used by the runtime."""

    def __init__(self, config: HandParams, *, cancel_requested=lambda: False) -> None:
        super().__init__(config)
        self._cancel_requested = cancel_requested
        self._motion_pending = False

    def check_cancelled(self) -> None:
        if self._cancel_requested():
            raise InterruptedError("XHand action sequence interrupted")

    def connect(self) -> int | None:
        if self.cfg.comm_type == "serial" and self.cfg.device_name is not None:
            problem = _serial_port_problem(self.cfg.device_name)
            if problem is not None:
                raise RuntimeError(problem)
        self.check_cancelled()
        deadline_ns = super().connect(cancel_requested=self._cancel_requested)
        self.check_cancelled()
        return deadline_ns

    @staticmethod
    def _header(title: str) -> None:
        print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")

    def read_sdk_version(self) -> None:
        self._header("SDK versions")
        print(f"  Software SDK: {self.device_identity['sdk_version']}")

        error_struct, version = self._control.read_version(self.cfg.device_id, 0)
        print(f"  Hardware SDK: {version}  (error_code={error_struct.error_code})")

    def read_device_info(self) -> None:
        self._header("Device info")
        error_struct, info = self._control.read_device_info(self.cfg.device_id)
        if error_struct.error_code != 0 or info is None:
            print(f"  Device info unavailable (error_code={error_struct.error_code})")
            return
        print(f"  serial_number: {''.join(info.serial_number[:16])}")
        print(f"  hand_id:       {info.hand_id}")
        print(f"  ev_hand:       {info.ev_hand}")

        print(f"  hand_type:     {self.device_identity['hand_type']}")

    def read_state(self, finger_id: int = 2) -> bool:
        self._header(f"Read state (finger {finger_id})")
        self.check_cancelled()
        state = self.get_state()
        self.check_cancelled()
        if state is None:
            print("  Joint feedback unavailable")
            return False
        print("  joint positions (deg, SDK ids 0..11):", np.rad2deg(state.qpos).round(3).tolist())
        print("  joint currents (mA; NaN means unavailable):", state.current_ma.tolist())
        print(
            f"  comm_err={state.commboard_err[finger_id]}  "
            f"joint_err={state.jointboard_err[finger_id]}  tip_err={state.tipboard_err[finger_id]}"
        )
        if finger_id in XHAND_TACTILE_SENSOR_INDEX_BY_FINGER_ID:
            index = XHAND_TACTILE_SENSOR_INDEX_BY_FINGER_ID[finger_id]
            force = state.tactile_aggregate[index]
            print("  calc_pressure:", force.tolist() if np.isfinite(force).all() else "unavailable")
            dense = state.tactile_dense[index]
            if np.isfinite(dense).all():
                print(f"  raw_pressure: points={len(dense)} max_abs={np.abs(dense).max():.3f}")
            else:
                print("  raw_pressure: unavailable")
        return True

    def _read_action_positions(
        self, *, action_deadline_ns: int | None = None
    ) -> np.ndarray:
        """One driver read bounded by feedback freshness and the action deadline."""
        self.check_cancelled()
        started_ns = time.monotonic_ns()
        deadline_ns = started_ns + int(self.cfg.feedback_max_age_s * 1e9)
        if action_deadline_ns is not None:
            deadline_ns = min(deadline_ns, action_deadline_ns)
        if time.monotonic_ns() >= deadline_ns:
            raise RuntimeError("action feedback deadline expired before read")
        state = self.get_state()
        self.check_cancelled()
        if time.monotonic_ns() >= deadline_ns:
            raise RuntimeError("action feedback deadline expired during read")
        if state is None:
            raise RuntimeError("action joint feedback unavailable")
        return state.qpos

    def execute_action(self, qpos_rad) -> None:
        target = _validate_target(qpos_rad, self.cfg)
        self._read_action_positions()
        self.check_cancelled()
        deadline_ns = time.monotonic_ns() + int(self.cfg.home_timeout_s * 1e9)
        self._motion_pending = True  # SDK exceptions do not establish non-delivery.
        status = self.send_action(target)
        print(f"  send_action: {status.name}")
        if status not in (DispatchStatus.ACCEPTED, DispatchStatus.CRC_UNCONFIRMED):
            raise RuntimeError(f"action rejected: {status.name}")
        tolerance_rad = np.deg2rad(self.cfg.home_tolerance_deg)
        consecutive = 0
        while time.monotonic_ns() < deadline_ns:
            # Each live read has its own freshness window. The previous read
            # and send latency do not determine the age of this new sample.
            measured = self._read_action_positions(action_deadline_ns=deadline_ns)
            if time.monotonic_ns() >= deadline_ns:
                break
            error_rad = float(np.max(np.abs(measured - target)))
            consecutive = consecutive + 1 if error_rad <= tolerance_rad else 0
            if consecutive >= _CONVERGENCE_SAMPLES:
                print(f"  Measured target reached; max error={np.rad2deg(error_rad):.3f} deg")
                return
            time.sleep(_FEEDBACK_POLL_INTERVAL_S)
        raise RuntimeError(f"action did not converge within {self.cfg.home_timeout_s:g}s")

    def stop_action(self) -> None:
        if not self._motion_pending:
            return
        started_ns = time.monotonic_ns()
        try:
            # Cleanup must still read/stop after the operator has cancelled.
            state = self.get_state()
            measured = None if state is None else state.qpos
        except Exception as exc:
            print(f"  Stop feedback unavailable ({exc}); requesting vendor passive mode")
            measured, started_ns = None, 0
        self.stop(measured, timestamp_ns=started_ns)
        self._motion_pending = False
        print("  Stop command accepted by SDK")


def _serial_port_problem(serial_port: str) -> str | None:
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


def _run_hardware_session(config: HandParams, *, finger_id: int, read_only: bool) -> int:
    # Validate every endpoint before connecting, including narrowed experiment limits.
    actions = [] if read_only else _action_sequence(config)
    configure_logging()
    interrupted_signal = None
    cleaning_up = False

    def request_interrupt(signum, _frame):
        nonlocal interrupted_signal
        interrupted_signal = signum
        if not cleaning_up:
            raise InterruptedError("XHand action sequence interrupted")

    hand = XHandControlExample(config, cancel_requested=lambda: interrupted_signal is not None)
    previous_handlers = {
        signum: signal.signal(signum, request_interrupt)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    result = 0
    try:
        hand.connect()
        hand.check_cancelled()
        hand.read_sdk_version()
        hand.read_device_info()
        hand.check_cancelled()
        if not hand.read_state(finger_id):
            raise RuntimeError("no valid initial joint state was received")
        for name, target in actions:
            print(f"\n  -> {name} (degrees): {np.rad2deg(target).round(3).tolist()}")
            hand.execute_action(target)
            time.sleep(_ACTION_DWELL_S)
    except (InterruptedError, CancelledError):
        print("\nSequence interrupted; stopping any attempted action before closing")
    except Exception as exc:
        print(f"\nXHand example failed: {exc}", file=sys.stderr)
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


def _disable_worker_core_dump() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError):
        pass


def _run_isolated_hardware_session(argv: list[str]) -> int:
    command = [sys.executable, str(Path(__file__).resolve()), *argv, _HARDWARE_WORKER_ARG]
    worker = subprocess.Popen(command)

    def forward_interrupt(signum, _frame):
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
    parser.add_argument("--config", type=Path, help="Optional experiment YAML; uses its hand section")
    parser.add_argument("--comm", choices=("serial", "ethercat"), help="Override hand transport")
    parser.add_argument("--device", help="Serial path or EtherCAT interface name")
    parser.add_argument("--hand-id", type=int, help="Override hand device id")
    parser.add_argument("--finger-id", type=int, default=5, help="Joint id inspected for tactile health")
    parser.add_argument("--read-only", action="store_true", help="Read diagnostics once, then exit")
    parser.add_argument("--print-config", action="store_true", help="Print hand settings without hardware")
    parser.add_argument(_HARDWARE_WORKER_ARG, action="store_true", help=argparse.SUPPRESS)
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    try:
        config = load_experiment_config(
            yaml_path=args.config,
            cli_overrides={
                "hand.comm_type": args.comm,
                "hand.device_name": args.device,
                "hand.device_id": args.hand_id,
            },
        ).hand
        if config.comm_type == "ethercat" and config.device_name == HandParams.device_name:
            config = replace(config, device_name=None)
        config.validate()
        if not 0 <= args.finger_id < HAND_DOF:
            raise ValueError(f"finger-id must be in 0..{HAND_DOF - 1}")
        if not args.read_only and not args.print_config:
            _action_sequence(config)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    if args.print_config:
        print(yaml.safe_dump({"hand": asdict(config)}, sort_keys=False), end="")
        return 0
    if args._xhand_hardware_worker:
        _disable_worker_core_dump()
        return _run_hardware_session(config, finger_id=args.finger_id, read_only=args.read_only)
    return _run_isolated_hardware_session(argv)


if __name__ == "__main__":
    raise SystemExit(main())
