#!/usr/bin/env python3
"""Usage: ``python examples/xhand_diagnostics.py``.

Reads device identity, joints and tactile health through the native SDK in a
crash-isolated worker. Connects to hardware; sends no motion commands.
Use the supervised teleop/home workflows for motion. No offline --help mode.
"""

from __future__ import annotations

import math
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dexmani_real.config.defaults import hand as hand_defaults
from dexmani_real.robot.model import (
    HAND_DOF,
    XHAND_TACTILE_SENSOR_INDEX_BY_FINGER_ID,
)

_BAUD_RATE_RS485 = hand_defaults.baudrate
_DEFAULT_SERIAL_PORT = hand_defaults.device_name
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
_RS485_POST_OPEN_SETTLE_S = hand_defaults.rs485_post_open_settle_s
_RS485_READ_CRC_RETRY_COUNT = 2
_RS485_CRC_RETRY_BACKOFF_S = 0.08
_HARDWARE_WORKER_ARG = "--_xhand-hardware-worker"



def _joint_payload_problem(state: Any) -> str | None:
    """Validate the joint subset required to tolerate a CRC-degraded read."""
    try:
        joints = list(state.finger_state)
        joint_ids = [int(joint.id) for joint in joints]
        joint_values = [
            float(value) for joint in joints for value in (joint.position, joint.torque)
        ]
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        return str(exc)
    if len(joints) != HAND_DOF or sorted(joint_ids) != list(range(HAND_DOF)):
        return f"expected unique joint ids 0..{HAND_DOF - 1}, got {joint_ids}"
    if not all(math.isfinite(value) for value in joint_values):
        return "joint position/current payload contains non-finite values"
    return None


class XHandDiagnostics:
    """Read-only SDK diagnostics in the child process."""

    def __init__(self, hand_id: int = 0) -> None:
        # Keep the native SDK inside the crash-isolated worker.
        from xhand_controller import xhand_control  # type: ignore[import-untyped]  # isort: skip

        self._sdk = xhand_control
        self._hand_id = hand_id
        self._device = self._sdk.XHandControl()
        self._protocol: str | None = None

    @staticmethod
    def _header(title: str) -> None:
        print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")

    def enumerate_devices(self, protocol: str) -> list[str]:
        self._header(f"Enumerate devices ({protocol})")
        ports = self._device.enumerate_devices(protocol)
        print(f"  ports: {ports}")
        return ports

    def open_device(self, protocol: str, serial_port: str | None = None) -> bool:
        self._header(f"Open device ({protocol})")
        if protocol == "RS485":
            if serial_port is None:
                print("  RS485 serial device is not configured")
                return False
            problem = _serial_port_problem(serial_port)
            if problem is not None:
                print(f"  FAILED: {problem}")
                print("  The native SDK was not asked to open the serial port.")
                return False
            print(f"  serial_port: {serial_port}")
            rsp = self._device.open_serial(serial_port, _BAUD_RATE_RS485)
            ok = rsp.error_code == 0
        elif protocol == "EtherCAT":
            ethercat_ports = self._device.enumerate_devices("EtherCAT")
            if not ethercat_ports:
                print("  No EtherCAT devices found.")
                return False
            rsp = self._device.open_ethercat(ethercat_ports[0])
            ok = rsp.error_code == 0
        else:
            print(f"  Unknown protocol: {protocol}")
            return False

        if ok:
            self._protocol = protocol
            print("  OK")
            if protocol == "RS485":
                print(
                    f"  Waiting {_RS485_POST_OPEN_SETTLE_S:.1f}s for the RS485 "
                    "receive path to settle..."
                )
                time.sleep(_RS485_POST_OPEN_SETTLE_S)
        else:
            err = rsp.error_message if rsp else "no response"
            print(f"  FAILED: {err}")
        return ok

    def read_sdk_version(self) -> None:
        self._header("SDK versions")
        print(f"  Software SDK: {self._device.get_sdk_version()}")

        error_struct, version = self._device.read_version(self._hand_id, 0)
        print(f"  Hardware SDK: {version}  (error_code={error_struct.error_code})")

    def read_device_info(self) -> None:
        self._header("Device info")
        error_struct, info = self._device.read_device_info(self._hand_id)
        print(f"  serial_number: {''.join(info.serial_number[:16])}")
        print(f"  hand_id:       {info.hand_id}")
        print(f"  ev_hand:       {info.ev_hand}")

        error_struct, hand_type = self._device.get_hand_type(self._hand_id)
        print(f"  hand_type:     {hand_type}")

    def read_serial_number(self) -> str:
        self._header("Serial number")
        error_struct, sn = self._device.get_serial_number(self._hand_id)
        print(f"  serial_number: {sn}")
        return sn

    def _read_state_response(
        self, force_update: bool, *, label: str
    ) -> tuple[Any, Any]:
        """Read state and retry only an RS485 CRC on a live transaction."""
        error_struct, state = self._device.read_state(self._hand_id, force_update)
        code = int(error_struct.error_code)
        if self._protocol == "RS485" and force_update:
            for retry_index in range(1, _RS485_READ_CRC_RETRY_COUNT + 1):
                if code != _RS485_CRC_ERROR_CODE:
                    break
                print(
                    f"  {label}: CRC ERROR; retrying the live state request "
                    f"({retry_index}/{_RS485_READ_CRC_RETRY_COUNT}) after "
                    f"{_RS485_CRC_RETRY_BACKOFF_S:.2f}s"
                )
                time.sleep(_RS485_CRC_RETRY_BACKOFF_S)
                error_struct, state = self._device.read_state(
                    self._hand_id, force_update
                )
                code = int(error_struct.error_code)
        return error_struct, state

    def read_state(self, finger_id: int = 2, force_update: bool = True) -> bool:
        # Request a live frame before the first command.
        self._header(f"Read state (finger {finger_id})")
        error_struct, state = self._read_state_response(
            force_update, label="read_state"
        )
        code = int(error_struct.error_code)
        if state is None:
            print(
                "  read_state error: SDK returned no state "
                f"(error_code={code} msg={error_struct.error_message})"
            )
            return False
        tactile_status = (
            self._protocol == "RS485" and code in _RS485_TACTILE_STATUS_CODES
        )
        crc_status = self._protocol == "RS485" and code == _RS485_CRC_ERROR_CODE
        if crc_status:
            joint_problem = _joint_payload_problem(state)
            if joint_problem is not None:
                print(
                    "  read_state CRC payload unusable: "
                    f"{joint_problem} (error_code={code})"
                )
                return False
        if code != 0 and not tactile_status and not crc_status:
            print(
                f"  read_state error: {error_struct.error_message} "
                f"(error_code={code})"
            )
            return False
        combined_force_valid = code == 0 or (
            self._protocol == "RS485"
            and code
            in {
                _RS485_DISTRIBUTED_FORCE_ERROR_CODE,
                _RS485_TEMPERATURE_ERROR_CODE,
            }
        )
        distributed_force_valid = code == 0 or (
            self._protocol == "RS485" and code == _RS485_TEMPERATURE_ERROR_CODE
        )
        temperature_valid = code not in {
            _RS485_TEMPERATURE_ERROR_CODE,
            _RS485_CRC_ERROR_CODE,
        }
        if crc_status:
            print(
                "  read_state: JOINTS USABLE; CRC UNCONFIRMED; SENSOR UNAVAILABLE; "
                "continuing"
            )
        elif tactile_status:
            print(
                "  read_state: JOINTS OK; SENSOR PARTIALLY DEGRADED  "
                f"(error_code={code} msg={error_struct.error_message}; "
                f"{_RS485_TACTILE_STATUS_DETAIL[code]})"
            )

        f = state.finger_state[finger_id]
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

    def close(self) -> None:
        self._header("Close device")
        self._device.close_device()
        print("  Device closed.")


def _serial_port_problem(serial_port: str) -> str | None:
    """Return a preflight error without opening or otherwise touching a TTY."""
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


def _choose_communication(xhand_exam: XHandDiagnostics) -> bool:
    """Prompt user to choose EtherCAT or RS485 and open the device."""
    while True:
        choice = input("Communication method (1=EtherCAT, 2=RS485): ").strip()
        if choice == "1":
            if xhand_exam.open_device("EtherCAT"):
                return True
            return False
        elif choice == "2":
            if xhand_exam.open_device("RS485", _DEFAULT_SERIAL_PORT):
                return True
            return False
        print("Invalid choice -- enter '1' or '2'.")


def _run_hardware_session() -> int:
    """Run one SDK session inside the crash-isolated worker process."""
    xhand_exam = XHandDiagnostics(hand_id=0)

    if not _choose_communication(xhand_exam):
        return 1

    try:
        xhand_exam.read_sdk_version()
        xhand_exam.read_device_info()
        xhand_exam.read_serial_number()
        if not xhand_exam.read_state(finger_id=5, force_update=True):
            print("Aborting: no valid initial joint state was received.")
            return 2

        return 0
    finally:
        xhand_exam.close()


def _disable_worker_core_dump() -> None:
    """Do not leave a large core file when the closed-source SDK aborts."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError):
        pass


def _run_isolated_hardware_session() -> int:
    """Keep a native SDK abort from terminating the user's launcher process."""
    command = [sys.executable, str(Path(__file__).resolve()), _HARDWARE_WORKER_ARG]
    completed = subprocess.run(command, check=False)
    if completed.returncode == -signal.SIGABRT:
        print(
            "\nXHand SDK worker aborted while its native communication thread was running.\n"
            "The launcher remained alive and no core file was written. For RS485, check that\n"
            f"{_DEFAULT_SERIAL_PORT} still exists, the hand is powered, "
            "USB/RS485 wiring is stable,\n"
            "and no other process owns the serial port, then reconnect and retry.",
            file=sys.stderr,
        )
        return 1
    if completed.returncode < 0:
        signal_number = -completed.returncode
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = f"signal {signal_number}"
        print(f"\nXHand SDK worker terminated by {signal_name}.", file=sys.stderr)
        return 1
    return completed.returncode


if __name__ == "__main__":
    if _HARDWARE_WORKER_ARG in sys.argv[1:]:
        _disable_worker_core_dump()
        raise SystemExit(_run_hardware_session())
    raise SystemExit(_run_isolated_hardware_session())
