#!/usr/bin/env python3
"""XHand control example -- standalone vendor diagnostic using raw xhand_controller SDK.

Usage::

    conda activate real_robot
    python examples/xhand_control_example.py

Demonstrates enumerate, open, identify, read state, preset actions, and home.
Uses the xhand_controller SDK directly -- no dexmani_real runtime dependencies.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass

from xhand_controller import xhand_control

# Matches dexmani_real.config.defaults.hand.home_qpos_deg -- duplicated here
# so this script stays usable without a full dexmani_real import chain.
HOME_QPOS_DEG = (0.0, 80.66, 33.2, 0.0, 5.11, 5.0, 6.53, 5.0, 6.76, 5.0, 10.13, 5.0)

_BAUD_RATE_RS485 = 3_000_000
_DEFAULT_SERIAL_PORT = "/dev/ttyUSB0"
_HAND_DOF = 12

# Fingertip sensor joint IDs (thumb, index, middle, ring, little).
_FINGERTIP_IDS = frozenset({2, 5, 7, 9, 11})


# ═══════════════════════════════════════════════════════════════════════
# Configuration dataclasses
# ═══════════════════════════════════════════════════════════════════════

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


# XHand 1 (serial digit "3").
PRESET_XHAND1 = PresetActions(
    fist=(11.85, 74.58, 40, -3.08, 106.02, 110, 109.75, 107.56, 107.66, 110, 109.1, 109.15),
    palm=(0, 80.66, 33.2, 0.00, 5.11, 0, 6.53, 0, 6.76, 4.41, 10.13, 0),
    v=(38.32, 90, 52.08, 6.21, 2.6, 0, 2.1, 0, 110, 110, 110, 109.23),
    ok=(45.88, 41.54, 67.35, 2.22, 80.45, 70.82, 31.37, 10.39, 13.69, 16.88, 1.39, 10.55),
)

# XHand 1 Lite (serial digit "6").
PRESET_XHAND1_LITE = PresetActions(
    fist=(0, 58, 83, 80, 80, 80, 0, 0, 0, 0, 0, 0),
    palm=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    v=(54, 66, 0, 0, 80, 80, 0, 0, 0, 0, 0, 0),
    ok=(49, 43, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0),
)


# ═══════════════════════════════════════════════════════════════════════
# XHand control example
# ═══════════════════════════════════════════════════════════════════════

class XHandControlExample:
    """Thin wrapper around xhand_controller SDK for diagnostic exercises."""

    def __init__(self, hand_id: int = 0, params: HandCommandParams | None = None) -> None:
        self._hand_id = hand_id
        self._params = params or HandCommandParams()
        self._device = xhand_control.XHandControl()
        self._hand_command = self._build_command(self._params.default_position)

    # ── Command builders ──

    def _build_command(self, position: float) -> xhand_control.HandCommand_t:
        """Build a homogeneous hand command with the configured servo params."""
        cmd = xhand_control.HandCommand_t()
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
        """Write joint positions (degrees -> radians) into the current command."""
        for i in range(_HAND_DOF):
            self._hand_command.finger_command[i].position = qpos_deg[i] * math.pi / 180.0

    @staticmethod
    def _header(title: str) -> None:
        print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")

    # ── Device enumeration and open ──

    def enumerate_devices(self, protocol: str) -> list[str]:
        self._header(f"Enumerate devices ({protocol})")
        ports = self._device.enumerate_devices(protocol)
        print(f"  ports: {ports}")
        return ports

    def open_device(self, protocol: str, serial_port: str = _DEFAULT_SERIAL_PORT) -> bool:
        self._header(f"Open device ({protocol})")
        if protocol == "RS485":
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
            print("  OK")
        else:
            err = rsp.error_message if rsp else "no response"
            print(f"  FAILED: {err}")
        return ok

    # ── Identity and version ──

    def read_sdk_version(self) -> None:
        self._header("SDK versions")
        print(f"  Software SDK: {self._device.get_sdk_version()}")

        error_struct, version = self._device.read_version(self._hand_id, 0)
        print(f"  Hardware SDK: {version}  (error_code={error_struct.error_code})")

    def read_device_info(self) -> None:
        self._header("Device info")
        error_struct, info = self._device.read_device_info(self._hand_id)
        print(f"  serial_number: {info.serial_number[:16]}")
        print(f"  hand_id:       {info.hand_id}")
        print(f"  ev_hand:       {info.ev_hand}")

        error_struct, hand_type = self._device.get_hand_type(self._hand_id)
        print(f"  hand_type:     {hand_type}")

    def read_serial_number(self) -> str:
        self._header("Serial number")
        error_struct, sn = self._device.get_serial_number(self._hand_id)
        print(f"  serial_number: {sn}")
        return sn

    # ── State ──

    def read_state(self, finger_id: int = 2, force_update: bool = True) -> None:
        self._header(f"Read state (finger {finger_id})")
        error_struct, state = self._device.read_state(self._hand_id, force_update)
        if error_struct.error_code != 0:
            print(f"  read_state error: {error_struct.error_message}")
            return

        f = state.finger_state[finger_id]
        print(f"  id={f.id}  temp={f.temperature}  temp&0xFF={f.temperature & 0xFF}")
        print(f"  comm_err={f.commboard_err}  joint_err={f.jonitboard_err}  tip_err={f.tipboard_err}")

        if f.id in _FINGERTIP_IDS:
            sensor = state.sensor_data[0]
            calc = sensor.calc_force
            print(f"  calc_pressure:     fx={calc.fx:.3f} fy={calc.fy:.3f} fz={calc.fz:.3f}")
            print(f"  sensor_temperature: {sensor.calc_temperature}")

    # ── Motion commands ──

    def send_command(self, sleep_s: float = 1.0) -> None:
        error_struct = self._device.send_command(self._hand_id, self._hand_command)
        ok = error_struct.error_code == 0
        print(f"  send_command: {'OK' if ok else 'FAILED'}  "
              f"(error_code={error_struct.error_code} msg={error_struct.error_message})")
        time.sleep(sleep_s)

    def set_mode(self, mode: int) -> None:
        """Set hand mode (0=powerless, 3=position, 5=powerful)."""
        self._header(f"Set hand mode -> {mode}")
        cmd = xhand_control.HandCommand_t()
        for i in range(_HAND_DOF):
            fc = cmd.finger_command[i]
            fc.id = i
            fc.kp = self._params.kp
            fc.ki = self._params.ki
            fc.kd = self._params.kd
            fc.position = 0.5
            fc.tor_max = self._params.tor_max
            fc.mode = mode
        error_struct = self._device.send_command(self._hand_id, cmd)
        print(f"  {'OK' if error_struct.error_code == 0 else 'FAILED'}  "
              f"(error_code={error_struct.error_code})")
        time.sleep(1.0)

    def run_preset_actions(self, actions: PresetActions) -> None:
        """Run preset actions (fist, palm, v, ok) with 1 s dwell each."""
        self._header("Preset actions")
        for name, qpos_deg in actions.iter_actions():
            print(f"  -> {name}")
            self._set_positions(qpos_deg)
            self.send_command()

    def go_home(self) -> None:
        """Return to home position (matches hand.home_qpos_deg)."""
        self._header("Return to home")
        self._set_positions(HOME_QPOS_DEG)
        self.send_command()

    # ── Cleanup ──

    def close(self) -> None:
        self._header("Close device")
        self._device.close_device()
        print("  Device closed.")


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _choose_communication(xhand_exam: XHandControlExample) -> None:
    """Prompt user to choose EtherCAT or RS485 and open the device."""
    while True:
        choice = input("Communication method (1=EtherCAT, 2=RS485): ").strip()
        if choice == "1":
            if xhand_exam.open_device("EtherCAT"):
                return
            sys.exit(1)
        elif choice == "2":
            xhand_exam.enumerate_devices("RS485")
            if xhand_exam.open_device("RS485", _DEFAULT_SERIAL_PORT):
                return
            sys.exit(1)
        print("Invalid choice -- enter '1' or '2'.")


def _select_preset_actions(serial_number: str) -> PresetActions:
    """Select the preset action table based on hand variant."""
    variant_code = serial_number[4] if len(serial_number) > 4 else ""
    if variant_code == "6":
        return PRESET_XHAND1_LITE
    return PRESET_XHAND1  # default (includes variant "3")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    params = HandCommandParams()
    xhand_exam = XHandControlExample(hand_id=0, params=params)

    _choose_communication(xhand_exam)

    # Identity and diagnostics (read-only, safe).
    xhand_exam.read_sdk_version()
    xhand_exam.read_device_info()
    serial_number = xhand_exam.read_serial_number()
    xhand_exam.read_state(finger_id=5, force_update=True)

    # ── Motion commands -- UNCOMMENT to execute (requires hardware authorization) ──
    #
    # xhand_exam.go_home()
    # actions = _select_preset_actions(serial_number)
    # xhand_exam.run_preset_actions(actions)

    xhand_exam.close()
