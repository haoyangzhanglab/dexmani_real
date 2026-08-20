#!/usr/bin/env python3
"""Usage: ``python examples/xhand_control_example.py``.

Run a moving XHand diagnostic through the native SDK in an isolated worker.
"""

from __future__ import annotations

import math
import os
import signal
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HOME_QPOS_DEG = (
    30.0,
    55.33,
    10.0,
    0.17,
    1.08,
    5.0,
    1.25,
    5.0,
    1.33,
    5.0,
    1.33,
    5.0,
)

_BAUD_RATE_RS485 = 3_000_000
_DEFAULT_SERIAL_PORT = "/dev/ttyUSB0"
_HAND_DOF = 12
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
_RS485_POST_OPEN_SETTLE_S = 1.0
_RS485_CRC_RETRY_COUNT = 1
_RS485_READ_CRC_RETRY_COUNT = 2
_RS485_SENSOR_VERIFY_RETRY_COUNT = 2
_RS485_CRC_RETRY_BACKOFF_S = 0.08
_HARDWARE_WORKER_ARG = "--_xhand-hardware-worker"

# Keep the local command envelope aligned with the runtime hand limits.
COMMAND_QPOS_MIN_RAD = (
    0.0,
    -0.698,
    0.17453292519943295,
    -0.174,
    0.0,
    0.08726646259971647,
    0.0,
    0.08726646259971647,
    0.0,
    0.08726646259971647,
    0.0,
    0.08726646259971647,
)
COMMAND_QPOS_MAX_RAD = (
    1.832,
    1.745,
    1.745,
    0.174,
    1.919,
    1.919,
    1.919,
    1.919,
    1.919,
    1.919,
    1.919,
    1.919,
)

_FINGERTIP_IDS = frozenset({2, 5, 7, 9, 11})
_FINGERTIP_TO_SENSOR_IDX = {2: 0, 5: 1, 7: 2, 9: 3, 11: 4}


@dataclass(frozen=True)
class HandCommandParams:
    """Default servo parameters for diagnostic hand commands."""

    mode: int = 3  # 0=powerless, 3=position (default), 5=powerful
    kp: int = 120
    ki: int = 0
    kd: int = 0
    tor_max: int = 380  # mA
    default_position: float = 0.1  # rad


@dataclass(frozen=True)
class PresetActions:
    """Preset joint-angle sets (degrees) per hand variant."""

    fist: tuple[float, ...]
    palm: tuple[float, ...]
    v: tuple[float, ...]
    ok: tuple[float, ...]

    def iter_actions(self):
        yield "fist", self.fist
        yield "palm", self.palm
        yield "v", self.v
        yield "ok", self.ok


PRESET_XHAND1 = PresetActions(
    fist=(
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
    palm=HOME_QPOS_DEG,
    v=(38.32, 90, 52.08, 6.21, 2.6, 5.0, 2.1, 5.0, 109.5, 109.5, 109.5, 109.23),
    ok=(
        45.88,
        41.54,
        67.35,
        2.22,
        80.45,
        70.82,
        31.37,
        10.39,
        13.69,
        16.88,
        1.39,
        10.55,
    ),
)

PRESET_XHAND1_LITE = PresetActions(
    fist=(0, 58, 83, 80, 80, 80, 0, 0, 0, 0, 0, 0),
    palm=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    v=(54, 66, 0, 0, 80, 80, 0, 0, 0, 0, 0, 0),
    ok=(49, 43, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0),
)


class XHandControlExample:
    """Thin wrapper around xhand_controller SDK for diagnostic exercises."""

    def __init__(
        self, hand_id: int = 0, params: HandCommandParams | None = None
    ) -> None:
        # Keep the native SDK inside the crash-isolated worker.
        from xhand_controller import xhand_control  # type: ignore[import-untyped]  # isort: skip

        self._sdk = xhand_control
        self._hand_id = hand_id
        self._params = params or HandCommandParams()
        self._device = self._sdk.XHandControl()
        self._hand_command = self._build_command(self._params.default_position)
        self._protocol: str | None = None

    def _build_command(self, position: float) -> Any:
        """Build a homogeneous hand command with the configured servo params."""
        cmd = self._sdk.HandCommand_t()
        for i in range(_HAND_DOF):
            fc = cmd.finger_command[i]
            fc.id = i
            fc.kp = self._params.kp
            fc.ki = self._params.ki
            fc.kd = self._params.kd
            fc.position = position
            fc.tor_max = self._params.tor_max
            fc.mode = self._params.mode
        return cmd

    def _set_positions(self, qpos_deg: tuple[float, ...] | list[float]) -> None:
        """Write clipped degree inputs as radians into the current command."""
        for i in range(_HAND_DOF):
            rad = qpos_deg[i] * math.pi / 180.0
            lo = COMMAND_QPOS_MIN_RAD[i]
            hi = COMMAND_QPOS_MAX_RAD[i]
            if rad < lo or rad > hi:
                clipped = max(lo, min(hi, rad))
                print(
                    f"  [clip] joint {i}: {rad:.4f} rad -> {clipped:.4f} rad "
                    f"(command envelope [{lo:.4f}, {hi:.4f}])"
                )
                rad = clipped
            self._hand_command.finger_command[i].position = rad

    @staticmethod
    def _header(title: str) -> None:
        print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")

    def enumerate_devices(self, protocol: str) -> list[str]:
        self._header(f"Enumerate devices ({protocol})")
        ports = self._device.enumerate_devices(protocol)
        print(f"  ports: {ports}")
        return ports

    def open_device(
        self, protocol: str, serial_port: str = _DEFAULT_SERIAL_PORT
    ) -> bool:
        self._header(f"Open device ({protocol})")
        if protocol == "RS485":
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
        tactile_status = (
            self._protocol == "RS485" and code in _RS485_TACTILE_STATUS_CODES
        )
        if code != 0 and not tactile_status:
            print(
                f"  read_state error: {error_struct.error_message} "
                f"(error_code={code})"
            )
            return False
        if state is None:
            print(
                "  read_state error: SDK returned no state "
                f"(error_code={code} msg={error_struct.error_message})"
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
        temperature_valid = code != _RS485_TEMPERATURE_ERROR_CODE
        if tactile_status:
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

        if f.id in _FINGERTIP_IDS:
            try:
                sensor = state.sensor_data[_FINGERTIP_TO_SENSOR_IDX[f.id]]
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

    def _verify_sensor_response_after_send(self, finger_id: int = 5) -> bool:
        """Verify a degraded command response without replaying the command."""
        error_struct, state = self._read_state_response(True, label="sensor_refresh")
        code = int(error_struct.error_code)
        for retry_index in range(1, _RS485_SENSOR_VERIFY_RETRY_COUNT + 1):
            if code not in _RS485_TACTILE_STATUS_CODES:
                break
            print(
                "  sensor_refresh: sensor fields still incomplete; retrying the "
                f"read-only request ({retry_index}/"
                f"{_RS485_SENSOR_VERIFY_RETRY_COUNT}) after "
                f"{_RS485_CRC_RETRY_BACKOFF_S:.2f}s"
            )
            time.sleep(_RS485_CRC_RETRY_BACKOFF_S)
            error_struct, state = self._read_state_response(
                True, label="sensor_refresh"
            )
            code = int(error_struct.error_code)

        if code != 0:
            detail = _RS485_TACTILE_STATUS_DETAIL.get(
                code, str(error_struct.error_message)
            )
            status = (
                "STILL PARTIALLY DEGRADED"
                if code in _RS485_TACTILE_STATUS_CODES
                else "FAILED"
            )
            print(
                f"  sensor_refresh: {status}  "
                f"(error_code={code} msg={error_struct.error_message}; {detail})"
            )
            return False
        if state is None:
            print("  sensor_refresh: FAILED  (SDK returned no state)")
            return False

        try:
            finger = state.finger_state[finger_id]
            sensor_index = _FINGERTIP_TO_SENSOR_IDX[int(finger.id)]
            sensor = state.sensor_data[sensor_index]
            calc_values = tuple(
                float(value)
                for value in (
                    sensor.calc_force.fx,
                    sensor.calc_force.fy,
                    sensor.calc_force.fz,
                )
            )
            raw_force = list(sensor.raw_force)
            raw_values = [
                float(value)
                for force in raw_force
                for value in (force.fx, force.fy, force.fz)
            ]
            temperature = float(sensor.calc_temperature)
            if len(raw_force) != 120 or not all(
                math.isfinite(value)
                for value in (*calc_values, *raw_values, temperature)
            ):
                raise ValueError(
                    "expected 120 finite distributed-force points and finite "
                    "combined-force/temperature fields"
                )
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            print(f"  sensor_refresh: FAILED  (malformed sensor payload: {exc})")
            return False

        print(
            "  sensor_refresh: RECOVERED; LIVE SENSOR FRAME OK  "
            f"(finger={int(finger.id)} raw_points={len(raw_force)} "
            f"temperature={temperature:g})"
        )
        return True

    def send_command(self, sleep_s: float = 1.0) -> bool:
        error_struct = self._device.send_command(self._hand_id, self._hand_command)
        code = int(error_struct.error_code)
        if self._protocol == "RS485":
            for retry_index in range(1, _RS485_CRC_RETRY_COUNT + 1):
                if code != _RS485_CRC_ERROR_CODE:
                    break
                print(
                    "  send_command: CRC ERROR; retrying the same absolute target "
                    f"({retry_index}/{_RS485_CRC_RETRY_COUNT}) after "
                    f"{_RS485_CRC_RETRY_BACKOFF_S:.2f}s"
                )
                time.sleep(_RS485_CRC_RETRY_BACKOFF_S)
                error_struct = self._device.send_command(
                    self._hand_id, self._hand_command
                )
                code = int(error_struct.error_code)
        tactile_only = self._protocol == "RS485" and code in _RS485_TACTILE_STATUS_CODES
        if tactile_only:
            print(
                "  send_command: MOTION SENT; SENSOR PARTIALLY DEGRADED  "
                f"(error_code={code} msg={error_struct.error_message}; "
                f"{_RS485_TACTILE_STATUS_DETAIL[code]})"
            )
        else:
            print(
                f"  send_command: {'OK' if code == 0 else 'FAILED'}  "
                f"(error_code={code} msg={error_struct.error_message})"
            )
        time.sleep(sleep_s)
        if tactile_only:
            self._verify_sensor_response_after_send()
        return code == 0 or tactile_only

    def run_preset_actions(self, actions: PresetActions) -> bool:
        """Run preset actions (fist, palm, v, ok) with 1 s dwell each."""
        self._header("Preset actions")
        for name, qpos_deg in actions.iter_actions():
            print(f"  -> {name}")
            self._set_positions(qpos_deg)
            if not self.send_command():
                print(
                    "  Aborting remaining presets after an unresolved command failure."
                )
                return False
        return True

    def go_home(self) -> bool:
        """Return to home position (matches hand.home_qpos_deg)."""
        self._header("Return to home")
        self._set_positions(HOME_QPOS_DEG)
        return self.send_command()

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


def _choose_communication(xhand_exam: XHandControlExample) -> bool:
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


def _select_preset_actions(serial_number: str) -> PresetActions:
    """Select the preset action table based on hand variant."""
    variant_code = serial_number[4] if len(serial_number) > 4 else ""
    if variant_code == "6":
        return PRESET_XHAND1_LITE
    return PRESET_XHAND1  # default (includes variant "3")


def _run_hardware_session() -> int:
    """Run one SDK session inside the crash-isolated worker process."""
    params = HandCommandParams()
    xhand_exam = XHandControlExample(hand_id=0, params=params)

    if not _choose_communication(xhand_exam):
        return 1

    try:
        xhand_exam.read_sdk_version()
        xhand_exam.read_device_info()
        serial_number = xhand_exam.read_serial_number()
        if not xhand_exam.read_state(finger_id=5, force_update=True):
            print("Aborting: no valid initial joint state was received.")
            return 2

        if not xhand_exam.go_home():
            print("Aborting: initial home command failed after the bounded retry.")
            return 2
        actions = _select_preset_actions(serial_number)
        if not xhand_exam.run_preset_actions(actions):
            return 2
        if not xhand_exam.go_home():
            print("Final home command failed after the bounded retry.")
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
