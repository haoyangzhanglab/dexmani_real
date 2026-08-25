"""Offline checks for unclipped keyboard commands and worker tracking limits."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from dexmani_real.teleop.keyboard_session import _publish_keyboard_target


class KeyboardArmLimitsTest(unittest.TestCase):
    def test_keyboard_publishes_full_ik_solution_without_delta_clip(self) -> None:
        ik_qpos = np.deg2rad(np.array([15.0, 7.0, -3.0, 2.0, 0.0, 5.0, -10.0]))
        captured: dict[str, np.ndarray] = {}

        def fake_publish(
            _shared: object, arm_qpos: np.ndarray, **_kwargs: object
        ) -> object:
            captured["arm_qpos"] = arm_qpos.copy()
            return SimpleNamespace(
                succeeded=True,
                candidate=SimpleNamespace(arm_qpos=arm_qpos.copy()),
                reason="",
                gate_code=None,
            )

        planner = SimpleNamespace(
            solve_teleop_ik=lambda *_args: SimpleNamespace(
                success=True,
                qpos=ik_qpos,
                reason="",
            )
        )
        runtime = SimpleNamespace(
            safety=SimpleNamespace(heartbeat_timeouts={"arm": 0.5, "hand": 0.5}),
        )
        with patch(
            "dexmani_real.teleop.keyboard_session.publish_joint_targets",
            side_effect=fake_publish,
        ):
            result = _publish_keyboard_target(
                SimpleNamespace(),
                runtime,
                planner,
                SimpleNamespace(),
                np.zeros(3),
                np.array([1.0, 0.0, 0.0, 0.0]),
                np.zeros(7),
                np.zeros(7),
            )

        self.assertEqual(result.status.value, "published")
        np.testing.assert_array_equal(captured["arm_qpos"], ik_qpos)


if __name__ == "__main__":
    unittest.main()
