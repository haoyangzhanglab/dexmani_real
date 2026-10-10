#!/usr/bin/env python3
"""连接 XHand1，顺序执行 HOME → fist → palm → V → OK → HOME，到位后各停留 1 秒。

python examples/xhand_control_example.py
python examples/xhand_control_example.py --config local.yaml --read-only
python examples/xhand_control_example.py --print-config

参数：--config 指定手部配置；--read-only 仍连接设备但不发动作，--print-config 不连接设备。
动作反馈遇到 RS485 CRC 时，在反馈时限内最多重读两次；不重发动作或使用 CRC 帧确认到位。
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import stat
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dexmani_real.config.experiment import config_as_dict, load_experiment_config
from dexmani_real.config.hardware import HandParams
from dexmani_real.robot.drivers.xhand import READ_USABLE_CODES, XHand, XHandSendStatus
from dexmani_real.robot.model import (
    HAND_DOF,
    XHAND_TACTILE_SENSOR_INDEX_BY_FINGER_ID,
)
from dexmani_real.utils.limits import validate_hand_command_bounds
from dexmani_real.utils.log import configure_logging

_RS485_COMBINED_FORCE_ERROR_CODE = 1_501_018
_RS485_DISTRIBUTED_FORCE_ERROR_CODE = 1_501_019
_RS485_TEMPERATURE_ERROR_CODE = 1_501_020
_RS485_TACTILE_STATUS_CODES = frozenset(
    {
        _RS485_COMBINED_FORCE_ERROR_CODE,
        _RS485_DISTRIBUTED_FORCE_ERROR_CODE,
        _RS485_TEMPERATURE_ERROR_CODE,
    }
)
_RS485_TACTILE_STATUS_DETAIL = {
    _RS485_COMBINED_FORCE_ERROR_CODE: (
        "combined force unavailable; force frame invalidated conservatively"
    ),
    _RS485_DISTRIBUTED_FORCE_ERROR_CODE: "distributed force unavailable; combined force retained",
    _RS485_TEMPERATURE_ERROR_CODE: "temperature unavailable; force fields retained",
}
_RS485_CRC_ERROR_CODE = 1_501_070
_RS485_READ_CRC_RETRY_COUNT = 2
_RS485_CRC_RETRY_BACKOFF_S = 0.08
_HARDWARE_WORKER_ARG = "--_xhand-hardware-worker"
_FEEDBACK_POLL_INTERVAL_S = 0.01
_CONVERGENCE_SAMPLES = 3
_STOP_RETRY_TIMEOUT_S = 0.25
_STOP_RETRY_COUNT = 26
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
    """Raw SDK inspection alongside the existing driver's action/stop primitives."""

    def __init__(self, config: HandParams, *, cancel_requested=lambda: False) -> None:
        super().__init__(config)
        self._cancel_requested = cancel_requested
        self._motion_pending = False

    def check_cancelled(self) -> None:
        if self._cancel_requested():
            raise InterruptedError("XHand action sequence interrupted")

    def connect(self) -> None:
        if self.cfg.comm_type == "serial" and self.cfg.device_name is not None:
            problem = _serial_port_problem(self.cfg.device_name)
            if problem is not None:
                raise RuntimeError(problem)
        self.check_cancelled()
        super().connect()
        self.check_cancelled()

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

    def _read_state_response(self, force_update: bool, *, label: str) -> tuple[Any, Any]:
        """Read state and retry only an RS485 CRC on a live transaction."""
        self.check_cancelled()
        error_struct, state = self._control.read_state(self.cfg.device_id, force_update)
        self.check_cancelled()
        code = int(error_struct.error_code)
        if self.cfg.comm_type == "serial" and force_update:
            for retry_index in range(1, _RS485_READ_CRC_RETRY_COUNT + 1):
                if code != _RS485_CRC_ERROR_CODE:
                    break
                print(
                    f"  {label}: CRC ERROR; retrying the live state request "
                    f"({retry_index}/{_RS485_READ_CRC_RETRY_COUNT}) after "
                    f"{_RS485_CRC_RETRY_BACKOFF_S:.2f}s"
                )
                time.sleep(_RS485_CRC_RETRY_BACKOFF_S)
                self.check_cancelled()
                error_struct, state = self._control.read_state(self.cfg.device_id, force_update)
                self.check_cancelled()
                code = int(error_struct.error_code)
        return error_struct, state

    def read_state(self, finger_id: int = 2, force_update: bool = True) -> bool:
        self._header(f"Read state (finger {finger_id})")
        error_struct, state = self._read_state_response(force_update, label="read_state")
        code = int(error_struct.error_code)
        if state is None:
            print(
                "  read_state error: SDK returned no state "
                f"(error_code={code} msg={error_struct.error_message})"
            )
            return False
        tactile_status = self.cfg.comm_type == "serial" and code in _RS485_TACTILE_STATUS_CODES
        crc_status = code == _RS485_CRC_ERROR_CODE
        if code not in READ_USABLE_CODES:
            print(f"  read_state error: {error_struct.error_message} (error_code={code})")
            return False
        try:
            qpos, current, _ = self._parse_joints(state)
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            print(f"  Joint payload unusable: {exc} (error_code={code})")
            return False
        print("  joint positions (deg, SDK ids 0..11):", np.rad2deg(qpos).round(3).tolist())
        print("  joint currents (mA; NaN means unavailable):", current.tolist())
        combined_force_valid = code == 0 or (
            self.cfg.comm_type == "serial"
            and code
            in {
                _RS485_DISTRIBUTED_FORCE_ERROR_CODE,
                _RS485_TEMPERATURE_ERROR_CODE,
            }
        )
        distributed_force_valid = code == 0 or (
            self.cfg.comm_type == "serial" and code == _RS485_TEMPERATURE_ERROR_CODE
        )
        temperature_valid = code == 0 or (
            self.cfg.comm_type == "serial"
            and code in {_RS485_COMBINED_FORCE_ERROR_CODE, _RS485_DISTRIBUTED_FORCE_ERROR_CODE}
        )
        if crc_status:
            print("  read_state: JOINTS USABLE; CRC UNCONFIRMED; SENSOR UNAVAILABLE; continuing")
        elif tactile_status:
            print(
                "  read_state: JOINTS OK; SENSOR PARTIALLY DEGRADED  "
                f"(error_code={code} msg={error_struct.error_message}; "
                f"{_RS485_TACTILE_STATUS_DETAIL[code]})"
            )

        f = next(joint for joint in state.finger_state if int(joint.id) == finger_id)
        print(f"  id={f.id}  temp={f.temperature}  temp&0xFF={f.temperature & 0xFF}")
        print(
            f"  comm_err={f.commboard_err}  joint_err={f.jonitboard_err}  tip_err={f.tipboard_err}"
        )

        if f.id in XHAND_TACTILE_SENSOR_INDEX_BY_FINGER_ID:
            try:
                sensor = state.sensor_data[XHAND_TACTILE_SENSOR_INDEX_BY_FINGER_ID[f.id]]
                if combined_force_valid:
                    calc = sensor.calc_force
                    print(
                        f"  calc_pressure:       fx={calc.fx:.3f} fy={calc.fy:.3f} fz={calc.fz:.3f}"
                    )
                else:
                    print("  calc_pressure:       unavailable")
                if distributed_force_valid:
                    raw_force = list(sensor.raw_force)
                    raw_values = [
                        float(value)
                        for force in raw_force
                        for value in (force.fx, force.fy, force.fz)
                    ]
                    raw_finite = all(math.isfinite(value) for value in raw_values)
                    raw_abs_max = max((abs(value) for value in raw_values), default=0.0)
                    print(
                        "  raw_pressure:        "
                        f"points={len(raw_force)} finite={raw_finite} max_abs={raw_abs_max:.3f}"
                    )
                else:
                    print("  raw_pressure:        unavailable")
                if temperature_valid:
                    print(f"  sensor_temperature: {sensor.calc_temperature}")
                else:
                    print("  sensor_temperature: unavailable")
            except (
                AttributeError,
                IndexError,
                TypeError,
                ValueError,
                OverflowError,
            ) as exc:
                print(f"  sensor payload malformed: {exc}")
        return True

    def _read_action_positions(
        self, *, retry_crc: bool = False, action_deadline: float | None = None
    ) -> tuple[np.ndarray, float, int]:
        """Return trusted joints, read start time, and discarded CRC count.

        Stop uses a single read even during cancellation. Motion reads may retry
        within the freshness/action deadline, without sending another command.
        """
        feedback_started = time.monotonic()
        feedback_deadline = feedback_started + self.cfg.feedback_max_age_s
        deadline_reason = "feedback freshness"
        if action_deadline is not None and action_deadline < feedback_deadline:
            feedback_deadline = action_deadline
            deadline_reason = "action"
        retries = 0
        while True:
            if retry_crc:
                self.check_cancelled()
            started = time.monotonic()
            if started >= feedback_deadline:
                raise RuntimeError(
                    f"action feedback exceeded {deadline_reason} deadline before live state read"
                )
            error, state = self._control.read_state(self.cfg.device_id, True)
            if retry_crc:
                self.check_cancelled()
            code = int(error.error_code)
            finished = time.monotonic()
            if finished >= feedback_deadline:
                raise RuntimeError(
                    f"action feedback exceeded {deadline_reason} deadline "
                    f"(read={finished - started:.3f}s, "
                    f"feedback_window={finished - feedback_started:.3f}s, "
                    f"budget={feedback_deadline - feedback_started:.3f}s, "
                    f"error_code={code})"
                )
            if not (
                retry_crc
                and self.cfg.comm_type == "serial"
                and code == _RS485_CRC_ERROR_CODE
                and retries < _RS485_READ_CRC_RETRY_COUNT
            ):
                break
            retries += 1
            print(
                f"  action feedback: CRC ERROR (error_code={code}); discarding frame, "
                f"retrying live state request ({retries}/{_RS485_READ_CRC_RETRY_COUNT})"
            )
            # The diagnostic's 80 ms backoff can exhaust the motion freshness
            # budget; use the normal feedback poll interval for motion reads.
            remaining = feedback_deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(_FEEDBACK_POLL_INTERVAL_S, remaining))
        # CRC-degraded frames remain inspectable but cannot authorize movement
        # or establish convergence in this diagnostic.
        if state is None or code not in READ_USABLE_CODES or code == _RS485_CRC_ERROR_CODE:
            raise RuntimeError(f"unconfirmed action feedback (error_code={code})")
        qpos, _, boards = self._parse_joints(state)
        if np.any(boards["commboard_err"]) or np.any(boards["jointboard_err"]):
            raise RuntimeError("action feedback reports a communication/joint board fault")
        if time.monotonic() >= feedback_deadline:
            raise RuntimeError(f"action feedback exceeded {deadline_reason} deadline during parsing")
        return qpos, started, retries

    def execute_action(self, qpos_rad) -> None:
        target = _validate_target(qpos_rad, self.cfg)
        self.check_cancelled()
        self._read_action_positions(retry_crc=True)
        self.check_cancelled()
        deadline = time.monotonic() + self.cfg.home_timeout_s
        self._motion_pending = True  # SDK exceptions do not establish non-delivery.
        status = self.send_action(target)
        print(f"  send_action: {status.value}")
        if status is not XHandSendStatus.ACCEPTED:
            raise RuntimeError(f"action {status.value}; no motion command will be replayed")
        tolerance_rad = np.deg2rad(self.cfg.home_tolerance_deg)
        consecutive = 0
        while time.monotonic() < deadline:
            self.check_cancelled()
            # Each live read has its own freshness window. The previous read
            # and send latency do not determine the age of this new sample.
            measured, _, crc_retries = self._read_action_positions(
                retry_crc=True,
                action_deadline=deadline,
            )
            self.check_cancelled()
            if time.monotonic() >= deadline:
                break
            if crc_retries:
                consecutive = 0
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
        try:
            measured, read_started, _ = self._read_action_positions()
        except Exception as exc:
            print(f"  Stop feedback unavailable ({exc}); requesting vendor passive mode")
            measured, read_started = None, 0.0
        deadline = time.monotonic() + _STOP_RETRY_TIMEOUT_S
        for _ in range(_STOP_RETRY_COUNT):
            # Match the runtime stop boundary: hold fresh measured joints;
            # without trusted feedback, use vendor passive mode instead.
            if (
                measured is not None
                and time.monotonic() - read_started <= self.cfg.feedback_max_age_s
            ):
                status = self.hold_current(measured)
                stop_mode = "hold measured joints"
            else:
                status = self.set_passive()
                stop_mode = "passive (motor effort disabled)"
            if status is XHandSendStatus.ACCEPTED:
                self._motion_pending = False
                print(f"  Stop command accepted by SDK: {stop_mode}")
                return
            if status is XHandSendStatus.REJECTED:
                raise RuntimeError("XHand stop command rejected")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            # Only stop commands may be retried; CRC never replays the action.
            time.sleep(min(_FEEDBACK_POLL_INTERVAL_S, remaining))
        raise RuntimeError("XHand stop remains CRC-unconfirmed")


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
    except InterruptedError:
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
        print(yaml.safe_dump({"hand": config_as_dict(config)}, sort_keys=False), end="")
        return 0
    if args._xhand_hardware_worker:
        _disable_worker_core_dump()
        return _run_hardware_session(config, finger_id=args.finger_id, read_only=args.read_only)
    return _run_isolated_hardware_session(argv)


if __name__ == "__main__":
    raise SystemExit(main())
