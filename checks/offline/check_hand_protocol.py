"""H5: formal dual-protocol support — canonical comm_type + serial read never re-sends.

Covers doc §6.1 item 16 (Phase 4b): ``comm_type``, device name, baudrate, and
device id are resolved from the immutable runtime config (not guessed by the
driver), and the driver's read path honours the transport semantics:

  - ``comm_type`` is validated to the closed set {"ethercat", "serial"} at every
    layer (HandParams / HandProcessConfig / XHandConfig); the former
    rs485/serial/usb/eth/ecat fuzzy aliases are rejected.
  - RS485 ``read_state(force_update=True)`` first re-sends the last command
    (vendored ``serial_communication.cpp``), so a serial state read is coerced
    to ``force_update=False`` and never issues ``send_command``.
  - EtherCAT ignores the parameter (returns the PDO cache), so the driver
    forwards the requested value unchanged.

Runs against a fake SDK ``control`` surface — no hardware.
"""

from __future__ import annotations

import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401  (repo root on sys.path)

from dexmani_real.config import defaults
from dexmani_real.robot import hand_process as hp
from dexmani_real.robot.xhand import XHand, XHandConfig
from dexmani_real.utils.schema import HAND_DOF


# -- minimal fake SDK control surface -----------------------------------------
class _Err:
    def __init__(self, code: int = 0) -> None:
        self.error_code = code
        self.error_message = ""


class _Force:
    def __init__(self) -> None:
        self.fx = self.fy = self.fz = 0.0


class _Finger:
    def __init__(self, idx: int) -> None:
        self.id = idx
        self.position = 0.0
        self.torque = 0.0
        self.commboard_err = 0
        self.jonitboard_err = 0  # SDK misspelling the driver handles
        self.tipboard_err = 0


class _Sensor:
    def __init__(self) -> None:
        self.calc_force = _Force()
        self.raw_force: list[_Force] = []


class _HandState:
    def __init__(self) -> None:
        self.finger_state = [_Finger(i) for i in range(HAND_DOF)]
        self.sensor_data = [_Sensor() for _ in range(5)]


class _Control:
    """Records the exact force_update flag passed to ``read_state`` and counts
    ``send_command`` calls, so the check can assert serial never re-sends."""

    def __init__(self) -> None:
        self.read_calls: list[bool] = []
        self.send_calls: list[tuple] = []

    def read_state(self, device_id: int, force_update: bool):
        self.read_calls.append(bool(force_update))
        return _Err(0), _HandState()

    def send_command(self, device_id: int, hand_command):
        self.send_calls.append((device_id, hand_command))
        return _Err(0)


def _driver(comm_type: str) -> XHand:
    hand = XHand(XHandConfig(comm_type=comm_type))
    hand.control = _Control()
    hand.connected_flag = True
    return hand


def _test_comm_type_canonical() -> None:
    # Canonical values are accepted at the driver layer.
    for good in ("ethercat", "serial"):
        assert XHandConfig(comm_type=good).comm_type == good

    # The fuzzy aliases are gone — the driver no longer guesses.
    for bad in ("EtherCAT", "RS485", "serial ", "usb", "eth", "ecat", "ether-cat", ""):
        try:
            XHandConfig(comm_type=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"XHandConfig(comm_type={bad!r}) must be rejected")


def _test_immutable_runtime_protocol() -> None:
    # The immutable runtime default is the canonical EtherCAT protocol.
    assert defaults.hand.comm_type == "ethercat"
    assert defaults.hand.device_name is None
    assert defaults.hand.baudrate > 0
    assert defaults.hand.device_id >= 0

    # HandParams rejects a fuzzy protocol too.
    try:
        defaults.HandParams(comm_type="EtherCAT")
    except ValueError:
        pass
    else:
        raise AssertionError("HandParams(comm_type='EtherCAT') must be rejected")

    # HandProcessConfig default follows the immutable runtime, and the runtime
    # mapping propagates a serial override to the worker config.
    assert hp.HandProcessConfig().comm_type == "ethercat"
    serial_hand = replace(defaults.hand, comm_type="serial", device_name="/dev/ttyUSB0", baudrate=115200)
    cfg = hp.HandProcessConfig.from_runtime(types.SimpleNamespace(hand=serial_hand))
    assert cfg.comm_type == "serial"
    assert cfg.device_name == "/dev/ttyUSB0"
    assert cfg.baudrate == 115200

    try:
        hp.HandProcessConfig(comm_type="rs485")
    except ValueError:
        pass
    else:
        raise AssertionError("HandProcessConfig(comm_type='rs485') must be rejected")


def _test_serial_read_never_force_updates() -> None:
    hand = _driver("serial")

    # A poll requesting fresh state must be coerced to a cached read ...
    hand.get_state(force_update=True)
    assert hand.control.read_calls == [False], f"serial read must coerce force_update to False, got {hand.control.read_calls}"
    # ... and must never issue a command re-send.
    assert hand.control.send_calls == [], "serial read must never send_command"

    # read_raw_state applies the same coercion.
    hand.control.read_calls.clear()
    hand.read_raw_state(force_update=True)
    assert hand.control.read_calls == [False]
    assert hand.control.send_calls == []


def _test_ethercat_passes_through() -> None:
    hand = _driver("ethercat")

    # EtherCAT ignores the parameter in the SDK, so the driver forwards the
    # requested flag unchanged (and, like serial, never sends on a read).
    hand.get_state(force_update=True)
    assert hand.control.read_calls == [True], f"ethercat must forward force_update, got {hand.control.read_calls}"
    assert hand.control.send_calls == []

    hand.control.read_calls.clear()
    hand.get_state(force_update=False)
    assert hand.control.read_calls == [False]


def _test_source_structural() -> None:
    import dexmani_real.robot.xhand as xhand_mod

    xhand_src = Path(xhand_mod.__file__).read_text()
    # The fuzzy resolver is gone; the coercion helper is the single decision point.
    assert "_resolve_comm_type" not in xhand_src
    assert "def _effective_force_update" in xhand_src
    assert "read_state(self.config.device_id, self._effective_force_update(force_update))" in xhand_src
    # No read path hard-codes force_update=True (the RS485 re-send hazard).
    assert "read_state(device_id, True)" not in xhand_src

    # The worker resolves comm_type from the runtime config and forwards it.
    hp_src = Path(hp.__file__).read_text()
    assert "comm_type=cfg.comm_type" in hp_src
    assert "device_name=cfg.device_name" in hp_src
    assert "baudrate=cfg.baudrate" in hp_src
    assert "device_id=cfg.device_id" in hp_src


def main() -> int:
    _test_comm_type_canonical()
    _test_immutable_runtime_protocol()
    _test_serial_read_never_force_updates()
    _test_ethercat_passes_through()
    _test_source_structural()

    print("check_hand_protocol: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
