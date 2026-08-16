#!/usr/bin/env python3
"""XHand control example -- standalone vendor exercise using raw xhand_controller SDK.

Usage::

    conda activate real_robot
    python examples/xhand_control_example.py              # home + preset motion

Default behaviour: enumerate, open, identify, read/print state, then run home +
preset actions.  The script always moves the hand when it runs -- there is no
read-only mode and no CLI flag gate.  Keep the workspace clear before running.

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

# Production command envelope (rad), duplicated from
# dexmani_real.config.defaults.hand.qpos_min_rad / qpos_max_rad so the
# diagnostic validates/clips presets against the same bounds the production
# policy enforces.  The distal lower bounds are the operator-set anti-clogging
# margins; the upper bounds are the rated mechanical max.  Presets are clipped
# to this envelope (matching the production publish-clip) so the diagnostic
# never sends an out-of-envelope raw SDK command.
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

# Fingertip sensor joint IDs (thumb, index, middle, ring, little).
_FINGERTIP_IDS = frozenset({2, 5, 7, 9, 11})
# Map each fingertip joint ID to its index into state.sensor_data (0-4).
_FINGERTIP_TO_SENSOR_IDX = {2: 0, 5: 1, 7: 2, 9: 3, 11: 4}


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


# XHand 1 (serial digit "3").  Preset angles are kept inside the production
# command envelope above so the diagnostic never emits a [clip] line: distal
# joints sit at the 5° anti-clogging lower bound and full-flexion at 109.5°
# (the rated upper bound is 1.919 rad ≈ 109.95°).
PRESET_XHAND1 = PresetActions(
    fist=(11.85, 74.58, 40, -3.08, 106.02, 109.5, 109.75, 107.56, 107.66, 109.5, 109.1, 109.15),
    palm=(0, 80.66, 33.2, 0.00, 5.11, 5.0, 6.53, 5.0, 6.76, 5.0, 10.13, 5.0),
    v=(38.32, 90, 52.08, 6.21, 2.6, 5.0, 2.1, 5.0, 109.5, 109.5, 109.5, 109.23),
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
        """Write joint positions (deg -> rad) into the current command, clipped
        to the production command envelope so the diagnostic never sends an
        out-of-range raw SDK command."""
        for i in range(_HAND_DOF):
            rad = qpos_deg[i] * math.pi / 180.0
            lo = COMMAND_QPOS_MIN_RAD[i]
            hi = COMMAND_QPOS_MAX_RAD[i]
            if rad < lo or rad > hi:
                clipped = max(lo, min(hi, rad))
                print(
                    f"  [clip] joint {i}: {rad:.4f} rad -> {clipped:.4f} rad "
                    f"(production envelope [{lo:.4f}, {hi:.4f}])"
                )
                rad = clipped
            self._hand_command.finger_command[i].position = rad

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

    # ── State ──

    def read_state(self, finger_id: int = 2, force_update: bool = False) -> None:
        # Default to a cached read.  RS485 read_state(force_update=True) first
        # re-sends the last command (vendored serial_communication.cpp), so a
        # diagnostic state read must never request a fresh read on that bus.
        self._header(f"Read state (finger {finger_id})")
        error_struct, state = self._device.read_state(self._hand_id, force_update)
        if error_struct.error_code != 0:
            print(f"  read_state error: {error_struct.error_message}")
            return

        f = state.finger_state[finger_id]
        print(f"  id={f.id}  temp={f.temperature}  temp&0xFF={f.temperature & 0xFF}")
        print(f"  comm_err={f.commboard_err}  joint_err={f.jonitboard_err}  tip_err={f.tipboard_err}")

        if f.id in _FINGERTIP_IDS:
            sensor = state.sensor_data[_FINGERTIP_TO_SENSOR_IDX[f.id]]
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
            ports = xhand_exam.enumerate_devices("RS485")
            if not ports:
                print("  No RS485 devices found.")
                sys.exit(1)
            if xhand_exam.open_device("RS485", ports[0]):
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

    _choose_communication(xhand_exam)  # opens the device (sys.exit(1) on failure)

    try:
        # Identity and diagnostics.
        xhand_exam.read_sdk_version()
        xhand_exam.read_device_info()
        serial_number = xhand_exam.read_serial_number()
        # Cached read: RS485 force_update=True would re-send the last command.
        xhand_exam.read_state(finger_id=5, force_update=False)

        # Motion: home, then preset actions, then return home.
        xhand_exam.go_home()
        actions = _select_preset_actions(serial_number)
        xhand_exam.run_preset_actions(actions)
        xhand_exam.go_home()
    finally:
        xhand_exam.close()
