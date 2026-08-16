"""H3: XHandSample driver simplification — single failure protocol, immutable sample.

Covers doc §6.1 items 12/13/14 for the Phase 3 driver simplification:

  - item 12: a malformed frame (out-of-range / negative / duplicate / missing
    joint id, or a non-finite position) fails the whole frame — ``_parse_sample``
    raises, and a failed ``get_state`` read raises ``RuntimeError`` (no NaN
    half-state, no None return, no per-caller flag).
  - item 13: a failed ``send_action`` never advances ``last_qpos_cmd``; only an
    SDK success does.
  - item 14: ``XHandSample`` arrays are copied and marked read-only on
    construction, so mutating the SDK's internal arrays (or a previously
    returned sample's arrays) cannot change an already-constructed sample.

Everything runs against a fake SDK ``HandState_t`` surface — no hardware.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401  (repo root on sys.path)

from dexmani_real.config.defaults import hand
from dexmani_real.robot.xhand import XHand, XHandConfig, XHandSample
from dexmani_real.utils.schema import (
    HAND_CONTACT_SHAPE,
    HAND_DOF,
    HAND_JOINT_SHAPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
)


# -- minimal fake SDK HandState_t surface -------------------------------------
class _Force:
    def __init__(self, fx: float = 0.0, fy: float = 0.0, fz: float = 0.0) -> None:
        self.fx = fx
        self.fy = fy
        self.fz = fz


class _Finger:
    def __init__(
        self,
        idx: int,
        position: float = 0.0,
        torque: float = 0.0,
        commboard_err: int = 0,
        jointboard_err: int = 0,
        tipboard_err: int = 0,
    ) -> None:
        self.id = idx
        self.position = position
        self.torque = torque
        self.commboard_err = commboard_err
        self.jonitboard_err = jointboard_err  # SDK misspelling the driver handles
        self.tipboard_err = tipboard_err


class _Sensor:
    def __init__(self) -> None:
        self.calc_force = _Force()
        self.raw_force: list[_Force] = []


class _HandState:
    def __init__(self, fingers: list[_Finger], sensors: list[_Sensor] | None = None) -> None:
        self.finger_state = fingers
        self.sensor_data = sensors if sensors is not None else []


class _Err:
    def __init__(self, code: int) -> None:
        self.error_code = code
        self.error_message = ""


class _FakeControl:
    def __init__(self, code: int) -> None:
        self._code = code

    def send_command(self, device_id, hand_command):
        return _Err(self._code)


class _FingerCmd:
    def __init__(self) -> None:
        self.position = 0.0


class _FakeHandCommand:
    def __init__(self) -> None:
        self.finger_command = [_FingerCmd() for _ in range(HAND_DOF)]


def _fingers(n: int = HAND_DOF, *, start_id: int = 0) -> list[_Finger]:
    return [
        _Finger(
            start_id + i,
            position=float(i) * 0.01,
            torque=float(i) * 0.1,
            commboard_err=i,
            jointboard_err=i + 1,
            tipboard_err=i + 2,
        )
        for i in range(n)
    ]


def _hand() -> XHand:
    return XHand(XHandConfig())


def _test_parse_sample_valid() -> None:
    hand = _hand()
    sample = hand._parse_sample(_HandState(_fingers()))

    assert sample.qpos.shape == HAND_JOINT_SHAPE
    assert sample.current.shape == HAND_JOINT_SHAPE
    assert sample.tactile_force.shape == HAND_TACTILE_FORCE_SHAPE
    assert sample.tactile_sum.shape == HAND_TACTILE_SUM_SHAPE
    assert sample.tactile_contact.shape == HAND_CONTACT_SHAPE

    for i in range(HAND_DOF):
        assert sample.qpos[i] == i * 0.01
        assert sample.current[i] == i * 0.1
        assert sample.commboard_err[i] == i
        assert sample.jointboard_err[i] == i + 1  # parsed from the SDK's "jonitboard_err"
        assert sample.tipboard_err[i] == i + 2


def _test_parse_sample_malformed() -> None:
    hand = _hand()

    # out-of-range id
    _assert_raises(hand, _fingers(HAND_DOF - 1) + [_Finger(HAND_DOF)], RuntimeError)
    # negative id
    _assert_raises(hand, _fingers(HAND_DOF - 1) + [_Finger(-1)], RuntimeError)
    # duplicate id (12 fingers, but id 4 appears twice and id 5 is missing)
    dup = _fingers()
    dup[5].id = 4
    _assert_raises(hand, dup, RuntimeError)
    # missing joint (only 11 enumerated)
    _assert_raises(hand, _fingers(HAND_DOF - 1), RuntimeError)
    # non-finite position
    bad = _fingers()
    bad[3].position = float("nan")
    _assert_raises(hand, bad, ValueError)


def _assert_raises(hand: XHand, fingers: list[_Finger], exc_type: type[Exception]) -> None:
    try:
        hand._parse_sample(_HandState(fingers))
    except exc_type:
        return
    except Exception as exc:  # noqa: BLE001 — surfaced as a clear assertion
        raise AssertionError(f"expected {exc_type.__name__}, got {type(exc).__name__}") from exc
    raise AssertionError(f"expected {exc_type.__name__} for malformed frame")


def _test_get_state_failure_protocol() -> None:
    hand = _hand()  # control is None → read_raw_state returns (None, None)
    try:
        hand.get_state()
    except RuntimeError as exc:
        assert "read failed" in str(exc)
        assert hand.last_error_code is not None
    else:
        raise AssertionError("get_state must raise RuntimeError on a failed read")


def _test_send_failure_does_not_advance() -> None:
    hand = _hand()
    lo = np.asarray(hand.config.qpos_min, dtype=np.float64)
    hi = np.asarray(hand.config.qpos_max, dtype=np.float64)
    target = (lo + hi) / 2.0
    shifted = target + 0.0001

    # A valid endpoint, but no control/hand_command → send fails without advancing.
    hand.last_qpos_cmd = target.copy()
    assert hand.send_action(shifted) is False
    assert np.allclose(hand.last_qpos_cmd, target), "failed send must not advance last_qpos_cmd"

    # A failed SDK send (non-zero code) also leaves last_qpos_cmd untouched.
    hand.control = _FakeControl(code=5)
    hand.hand_command = _FakeHandCommand()
    assert hand.send_action(shifted) is False
    assert np.allclose(hand.last_qpos_cmd, target), "failed SDK send must not advance last_qpos_cmd"

    # Only an SDK success advances the command history.
    hand.control = _FakeControl(code=0)
    assert hand.send_action(shifted) is True
    assert np.allclose(hand.last_qpos_cmd, shifted), "successful send must advance last_qpos_cmd"


def _test_sample_immutability() -> None:
    qpos = np.zeros(HAND_JOINT_SHAPE, dtype=np.float64)
    current = np.zeros(HAND_JOINT_SHAPE, dtype=np.float64)
    tactile_force = np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64)
    tactile_sum = np.zeros(HAND_TACTILE_SUM_SHAPE, dtype=np.float64)
    tactile_contact = np.zeros(HAND_CONTACT_SHAPE, dtype=bool)
    commboard_err = np.zeros(HAND_JOINT_SHAPE, dtype=np.int32)

    sample = XHandSample(
        qpos=qpos,
        current=current,
        tactile_force=tactile_force,
        tactile_sum=tactile_sum,
        tactile_contact=tactile_contact,
        commboard_err=commboard_err,
        jointboard_err=commboard_err,
        tipboard_err=commboard_err,
    )

    # item 14: mutating the source arrays after construction cannot change the sample.
    qpos[0] = 12345.0
    commboard_err[0] = 999
    assert sample.qpos[0] == 0.0
    assert sample.commboard_err[0] == 0

    # The sample's arrays are read-only.
    for name in ("qpos", "current", "tactile_force", "tactile_sum", "commboard_err"):
        assert not getattr(sample, name).flags.writeable, name
    try:
        sample.qpos[0] = 1.0
    except ValueError:
        pass
    else:
        raise AssertionError("XHandSample arrays must be read-only")


def _test_source_structural() -> None:
    import dexmani_real.robot.xhand as xhand_mod

    src = Path(xhand_mod.__file__).read_text()
    # Dead fields and helper protocol removed.
    for gone in ("last_action_code", "last_joint_limit_rejected", "is_valid_qpos_state", "_array12", "_empty_state"):
        assert gone not in src, f"{gone} must be removed"
    assert "nan_array" not in src and "safe_resize" not in src
    # Single failure protocol: get_state returns XHandSample (not a dict/NaN/None).
    assert "def get_state(self, force_update" in src
    assert "-> XHandSample:" in src
    assert "def _parse_sample" in src


def main() -> int:
    _test_parse_sample_valid()
    _test_parse_sample_malformed()
    _test_get_state_failure_protocol()
    _test_send_failure_does_not_advance()
    _test_sample_immutability()
    _test_source_structural()

    print("check_hand_sample: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
