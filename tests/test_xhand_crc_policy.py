"""Offline XHand CRC classification and payload-validation contracts."""

from __future__ import annotations

import importlib
import runpy
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import patch

import numpy as np


def _load_xhand_module():
    fake_package = types.ModuleType("xhand_controller")
    fake_package.xhand_control = SimpleNamespace()  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"xhand_controller": fake_package}):
        return importlib.import_module("dexmani_real.robot.xhand")


class XHandCrcPolicyTest(unittest.TestCase):
    xhand: ClassVar[Any]
    example: ClassVar[dict[str, Any]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.xhand = _load_xhand_module()
        example_path = (
            Path(__file__).resolve().parents[1] / "examples/xhand_control_example.py"
        )
        cls.example = runpy.run_path(
            str(example_path), run_name="xhand_crc_policy_test"
        )

    def test_driver_distinguishes_crc_from_acceptance_and_rejection(self) -> None:
        config = SimpleNamespace(
            device_id=0,
            mechanical_qpos_min_rad=tuple(np.full(12, -1.0)),
            mechanical_qpos_max_rad=tuple(np.full(12, 1.0)),
        )
        hand = self.xhand.XHand(config)
        response = SimpleNamespace(error_code=0, error_message="mock")
        hand._control = SimpleNamespace(send_command=lambda *_args: response)
        hand._command = SimpleNamespace(
            finger_command=[SimpleNamespace(position=0.0) for _ in range(12)]
        )
        hand.connected_flag = True

        cases = (
            (0, self.xhand.XHandSendStatus.ACCEPTED),
            (1_501_035, self.xhand.XHandSendStatus.ACCEPTED),
            (1_501_070, self.xhand.XHandSendStatus.CRC_UNCONFIRMED),
            (999, self.xhand.XHandSendStatus.REJECTED),
        )
        for code, expected in cases:
            with self.subTest(code=code):
                response.error_code = code
                self.assertIs(hand.send_action(np.zeros(12)), expected)

    def test_example_continues_after_unresolved_send_crc(self) -> None:
        response = SimpleNamespace(
            error_code=1_501_070,
            error_message="mock CRC",
        )
        example = object.__new__(self.example["XHandControlExample"])
        example._protocol = "RS485"
        example._hand_id = 0
        example._hand_command = object()
        example._device = SimpleNamespace(send_command=lambda *_args: response)

        with patch("time.sleep"):
            status = example.send_command(sleep_s=0.0)

        self.assertIs(status, self.example["HandCommandStatus"].CRC_UNCONFIRMED)

    def test_crc_read_requires_complete_finite_joint_payload(self) -> None:
        joints = [
            SimpleNamespace(id=index, position=0.1, torque=0.0) for index in range(12)
        ]
        state = SimpleNamespace(finger_state=joints)
        incomplete = SimpleNamespace(finger_state=joints[:-1])
        nonfinite = SimpleNamespace(
            finger_state=[
                *joints[:-1],
                SimpleNamespace(id=11, position=np.nan, torque=0.0),
            ]
        )

        validate = self.example["_joint_payload_problem"]
        self.assertIsNone(validate(state))
        self.assertIsNotNone(validate(incomplete))
        self.assertIsNotNone(validate(nonfinite))


if __name__ == "__main__":
    unittest.main()
