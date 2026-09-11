"""Offline regressions for the XHand driver tactile correctness path.

No hardware and no SDK: a fake ``read_state`` feeds ``XHand.get_state`` and
``XHand.calibrate_tactile``.  Pins the RS485 partial-validity matrix, SDK-native
scale (no ``0.1``), independent aggregate/dense parse failure, and the
candidate-bias + post-bias-verification calibration contract.  Run with:

    python -m unittest discover -s tests -p 'test_xhand_driver_tactile.py'
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from dexmani_real.robot.drivers.xhand import (
    XHand,
    XHandError,
    XHandState,
    _tactile_validity,
)
from dexmani_real.robot.model import HAND_FINGER_COUNT, TACTILE_POINTS_PER_FINGER

_CRC_CODE = 1_501_070
_COMBINED = 1_501_018
_DISTRIBUTED = 1_501_019
_TEMPERATURE = 1_501_020


class _Error:
    def __init__(self, code: int, message: str = ""):
        self.error_code = code
        self.error_message = message


class _Force:
    def __init__(self, fx: float, fy: float, fz: float):
        self.fx = fx
        self.fy = fy
        self.fz = fz


class _Sensor:
    def __init__(self, calc: tuple[float, float, float], raw: tuple[float, float, float]):
        self.calc_force = _Force(*calc)
        self.raw_force = [_Force(*raw) for _ in range(TACTILE_POINTS_PER_FINGER)]


class _Joint:
    def __init__(self, index: int):
        self.id = index
        self.position = 0.0
        self.torque = 0.0
        self.commboard_err = 0
        self.jonitboard_err = 0
        self.tipboard_err = 0


class _State:
    def __init__(self, sensors: list[_Sensor]):
        self.finger_state = [_Joint(i) for i in range(12)]
        self.sensor_data = sensors


def _sensors(calc: tuple[float, float, float], raw: tuple[float, float, float]):
    return [_Sensor(calc, raw) for _ in range(HAND_FINGER_COUNT)]


def _make_hand(comm_type: str, code: int, state) -> XHand:
    hand = XHand(SimpleNamespace(comm_type=comm_type, device_id=0))
    hand._control = SimpleNamespace(read_state=lambda *a, **k: (_Error(code), state))
    hand.connected_flag = True
    return hand


def _state(*, calc=(1.0, 2.0, 3.0), raw=(4.0, 5.0, 6.0)):
    return _State(_sensors(calc, raw))


class _RawSensor:
    """Sensor whose ``raw_force`` list length is controllable (for malformed cases)."""

    def __init__(self, calc: tuple[float, float, float], raw_points: list[_Force]):
        self.calc_force = _Force(*calc)
        self.raw_force = raw_points


class _MalformedState:
    def __init__(self, sensors):
        self.finger_state = [_Joint(i) for i in range(12)]
        self.sensor_data = sensors


class TestValidityMatrix(unittest.TestCase):
    def test_pure_helper_rs485(self) -> None:
        self.assertEqual(_tactile_validity(0, comm_type="serial"), (True, True))
        self.assertEqual(_tactile_validity(_COMBINED, comm_type="serial"), (False, False))
        self.assertEqual(_tactile_validity(_DISTRIBUTED, comm_type="serial"), (True, False))
        self.assertEqual(_tactile_validity(_TEMPERATURE, comm_type="serial"), (True, True))
        self.assertEqual(_tactile_validity(_CRC_CODE, comm_type="serial"), (False, False))

    def test_pure_helper_ethercat(self) -> None:
        self.assertEqual(_tactile_validity(0, comm_type="ethercat"), (True, True))
        self.assertEqual(_tactile_validity(_COMBINED, comm_type="ethercat"), (False, False))

    def test_rs485_matrix(self) -> None:
        cases = {
            0: (True, True),
            _COMBINED: (False, False),
            _DISTRIBUTED: (True, False),
            _TEMPERATURE: (True, True),
            _CRC_CODE: (False, False),
        }
        for code, (aggregate_valid, dense_valid) in cases.items():
            with self.subTest(code=code):
                hand = _make_hand("serial", code, _state())
                out = hand.get_state()
                self.assertIsNotNone(out)
                self.assertEqual(out.tactile_aggregate_valid, aggregate_valid)
                self.assertEqual(out.tactile_dense_valid, dense_valid)
                if aggregate_valid:
                    np.testing.assert_array_equal(out.tactile_aggregate[0], [1.0, 2.0, 3.0])
                else:
                    np.testing.assert_array_equal(out.tactile_aggregate[0], [0.0, 0.0, 0.0])
                if dense_valid:
                    np.testing.assert_array_equal(out.tactile_dense[0, 0], [4.0, 5.0, 6.0])
                else:
                    np.testing.assert_array_equal(out.tactile_dense[0, 0], [0.0, 0.0, 0.0])

    def test_ethercat_nonzero_fails_tactile_closed_joints_usable(self) -> None:
        hand = _make_hand("ethercat", _DISTRIBUTED, _state())
        out = hand.get_state()
        self.assertIsNotNone(out)
        self.assertEqual(out.qpos.shape, (12,))
        self.assertFalse(out.tactile_aggregate_valid)
        self.assertFalse(out.tactile_dense_valid)

    def test_malformed_dense_does_not_erase_valid_aggregate(self) -> None:
        sensors = [
            _RawSensor((1.0, 2.0, 3.0), [_Force(4.0, 5.0, 6.0)] * 119)  # short raw
            for _ in range(HAND_FINGER_COUNT)
        ]
        hand = _make_hand("serial", 0, _MalformedState(sensors))
        out = hand.get_state()
        self.assertIsNotNone(out)
        self.assertTrue(out.tactile_aggregate_valid)
        self.assertFalse(out.tactile_dense_valid)
        np.testing.assert_array_equal(out.tactile_aggregate[0], [1.0, 2.0, 3.0])

    def test_malformed_calc_invalidates_aggregate(self) -> None:
        class _BadCalcSensor:
            calc_force = None  # triggers _force_xyz -> ValueError
            raw_force = [_Force(4.0, 5.0, 6.0) for _ in range(TACTILE_POINTS_PER_FINGER)]

        hand = _make_hand(
            "serial", 0, _MalformedState([_BadCalcSensor() for _ in range(HAND_FINGER_COUNT)])
        )
        out = hand.get_state()
        self.assertIsNotNone(out)
        self.assertFalse(out.tactile_aggregate_valid)
        # Dense parsed independently and stays valid.
        self.assertTrue(out.tactile_dense_valid)

    def test_nonfinite_calc_invalidates_aggregate_only(self) -> None:
        sensors = [
            _RawSensor((np.nan, 2.0, 3.0), [_Force(4.0, 5.0, 6.0)] * TACTILE_POINTS_PER_FINGER)
            for _ in range(HAND_FINGER_COUNT)
        ]
        hand = _make_hand("serial", 0, _MalformedState(sensors))
        out = hand.get_state()
        self.assertIsNotNone(out)
        self.assertFalse(out.tactile_aggregate_valid)
        self.assertTrue(out.tactile_dense_valid)


class TestNativeScale(unittest.TestCase):
    def test_zero_bias_returns_sdk_values_unchanged(self) -> None:
        hand = _make_hand("serial", 0, _state(calc=(10.0, 20.0, 30.0), raw=(4.0, 5.0, 6.0)))
        out = hand.get_state()
        np.testing.assert_array_equal(out.tactile_aggregate[0], [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(out.tactile_dense[0, 0], [4.0, 5.0, 6.0])

    def test_bias_subtracts(self) -> None:
        hand = _make_hand("serial", 0, _state(calc=(10.0, 20.0, 30.0), raw=(4.0, 5.0, 6.0)))
        hand._tactile_bias_aggregate = np.full((5, 3), 1.0)
        hand._tactile_bias_dense = np.full((5, 120, 3), 2.0)
        out = hand.get_state()
        np.testing.assert_array_equal(out.tactile_aggregate[0], [9.0, 19.0, 29.0])
        np.testing.assert_array_equal(out.tactile_dense[0, 0], [2.0, 3.0, 4.0])


def _xstate(tactile_aggregate, *, aggregate_valid=True, dense_valid=True) -> XHandState:
    return XHandState(
        qpos=np.zeros(12),
        current_ma=np.zeros(12),
        tactile_aggregate=np.asarray(tactile_aggregate, dtype=np.float64),
        tactile_dense=np.zeros((5, 120, 3)),
        tactile_aggregate_valid=aggregate_valid,
        tactile_dense_valid=dense_valid,
        commboard_err=np.zeros(12, dtype=np.int32),
        jointboard_err=np.zeros(12, dtype=np.int32),
        tipboard_err=np.zeros(12, dtype=np.int32),
    )


class TestCalibration(unittest.TestCase):
    def _calibrate(self, capture_states, verify_states):
        hand = XHand(SimpleNamespace(comm_type="serial", device_id=0))
        with mock.patch.object(
            hand, "get_state", side_effect=capture_states + verify_states
        ), mock.patch("dexmani_real.robot.drivers.xhand.time.sleep"):
            return hand.calibrate_tactile(), hand

    def test_successful_capture_and_low_residual(self) -> None:
        capture = [_xstate(np.full((5, 3), 1.0)) for _ in range(5)]
        verify = [_xstate(np.full((5, 3), 0.1)) for _ in range(3)]
        ok, hand = self._calibrate(capture, verify)
        self.assertTrue(ok)
        self.assertTrue(hand.tactile_calibrated)
        np.testing.assert_allclose(hand._tactile_bias_aggregate, np.full((5, 3), 1.0))

    def test_residual_above_threshold_fails_and_clears(self) -> None:
        capture = [_xstate(np.full((5, 3), 1.0)) for _ in range(5)]
        verify = [_xstate(np.full((5, 3), 0.1)), _xstate(np.full((5, 3), 0.1)),
                  _xstate(np.array([[3.0, 0.0, 0.0]] * 5))]
        ok, hand = self._calibrate(capture, verify)
        self.assertFalse(ok)
        self.assertFalse(hand.tactile_calibrated)
        self.assertIsNone(hand._tactile_bias_aggregate)
        self.assertIsNone(hand._tactile_bias_dense)

    def test_invalid_aggregate_during_capture_fails(self) -> None:
        capture = [_xstate(np.full((5, 3), 1.0)) for _ in range(4)]
        capture.append(_xstate(np.full((5, 3), 1.0), aggregate_valid=False))
        with self.assertRaises(XHandError):
            self._calibrate(capture, [])

    def test_invalid_dense_during_capture_fails(self) -> None:
        capture = [_xstate(np.full((5, 3), 1.0)) for _ in range(4)]
        capture.append(_xstate(np.full((5, 3), 1.0), dense_valid=False))
        with self.assertRaises(XHandError):
            self._calibrate(capture, [])

    def test_invalid_aggregate_during_verify_fails_and_clears(self) -> None:
        capture = [_xstate(np.full((5, 3), 1.0)) for _ in range(5)]
        verify = [_xstate(np.full((5, 3), 0.1), aggregate_valid=False)]
        ok, hand = self._calibrate(capture, verify)
        self.assertFalse(ok)
        self.assertIsNone(hand._tactile_bias_aggregate)

    def test_invalid_dense_during_verify_fails_and_clears(self) -> None:
        capture = [_xstate(np.full((5, 3), 1.0)) for _ in range(5)]
        verify = [_xstate(np.full((5, 3), 0.1), dense_valid=False)]
        ok, hand = self._calibrate(capture, verify)
        self.assertFalse(ok)
        self.assertIsNone(hand._tactile_bias_aggregate)

    def test_large_finite_dense_does_not_fail_verification(self) -> None:
        # Dense payload magnitude alone is structural-only: a large but finite
        # dense residual must not reject calibration.
        capture = [_xstate(np.full((5, 3), 1.0)) for _ in range(5)]
        verify = [_xstate(np.full((5, 3), 0.1)) for _ in range(3)]
        for state in verify:
            state.tactile_dense = np.full((5, 120, 3), 1000.0)
        ok, hand = self._calibrate(capture, verify)
        self.assertTrue(ok)
        self.assertTrue(hand.tactile_calibrated)


if __name__ == "__main__":
    unittest.main()
