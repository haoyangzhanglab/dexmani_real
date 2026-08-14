"""Offline check: C24 recovery follows the xArm SDK contract, in order.

Deterministic fake-arm check that ``_recover_c24_measured_hold`` calls the
controller in the required order — ``clean_error`` → ``motion_enable(True)`` →
Mode 6 ready (``set_mode(6)``/``set_state(0)``) → fresh ``get_joint_states`` →
one ``set_servo_angle`` measured hold — and returns a (7,) joint vector.

Run from the repo root:
    conda run -n real_robot python checks/offline/check_c24_recovery_order.py
"""

from __future__ import annotations

import numpy as np

from dexmani_real.robot.arm_loop import ArmLoopConfig, _recover_c24_measured_hold


class FakeArm:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.mode = 6

    def get_c24_error_info(self):
        self.calls.append("get_c24_error_info")
        return 0, [3, 1.23]

    def clean_error(self):
        self.calls.append("clean_error")
        return 0

    def clean_warn(self):
        self.calls.append("clean_warn")
        return 0

    def motion_enable(self, value):
        self.calls.append(f"motion_enable({value})")
        return 0

    def set_mode(self, value):
        self.calls.append(f"set_mode({value})")
        self.mode = value
        return 0

    def set_state(self, value):
        self.calls.append(f"set_state({value})")
        return 0

    def get_state(self):
        self.calls.append("get_state")
        return 0, 2

    def get_err_warn_code(self):
        self.calls.append("get_err_warn_code")
        return 0, [0, 0]

    def get_joint_states(self, **kwargs):
        self.calls.append("get_joint_states")
        return 0, [np.zeros(7)]

    def set_servo_angle(self, **kwargs):
        self.calls.append("set_servo_angle")
        return 0


def main() -> int:
    arm = FakeArm()
    q = _recover_c24_measured_hold(arm, ArmLoopConfig())
    assert q.shape == (7,), q.shape

    order = arm.calls
    assert order.index("clean_error") < order.index("motion_enable(True)")
    assert order.index("motion_enable(True)") < order.index("set_mode(6)")
    assert order.index("set_state(0)") < order.index("get_joint_states")
    assert order.index("get_joint_states") < order.index("set_servo_angle")

    print("OK", order)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
