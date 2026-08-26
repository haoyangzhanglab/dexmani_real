"""Offline checks for unclipped keyboard commands and worker tracking limits."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from dexmani_real.teleop.keyboard_session import (
    _keyboard_action_was_accepted,
    _KeyboardMotionDiagnostics,
    _KeyboardPublishResult,
    _KeyboardPublishStatus,
    _publish_keyboard_target,
)


class KeyboardArmLimitsTest(unittest.TestCase):
    def test_release_waits_only_for_final_normal_action_acceptance(self) -> None:
        self.assertTrue(_keyboard_action_was_accepted(0, {"last_cmd_seq": 8}))
        self.assertFalse(_keyboard_action_was_accepted(9, {"last_cmd_seq": 8}))
        self.assertTrue(_keyboard_action_was_accepted(9, {"last_cmd_seq": 9}))
        self.assertTrue(_keyboard_action_was_accepted(9, {"last_cmd_seq": 10}))

    def test_keyboard_publishes_full_ik_solution_without_delta_clip(self) -> None:
        ik_qpos = np.deg2rad(np.array([15.0, 7.0, -3.0, 2.0, 0.0, 5.0, -10.0]))
        captured: dict[str, np.ndarray] = {}

        def fake_publish(
            _shared: object, arm_qpos: np.ndarray, **_kwargs: object
        ) -> object:
            captured["arm_qpos"] = arm_qpos.copy()
            return SimpleNamespace(
                succeeded=True,
                candidate=SimpleNamespace(arm_qpos=arm_qpos.copy(), action_id=31),
                ticket=SimpleNamespace(ring_sequence=41),
                reason="",
                gate_code=None,
            )

        planner = SimpleNamespace(
            solve_teleop_ik=lambda *_args: SimpleNamespace(
                success=True,
                qpos=ik_qpos,
                reason="",
                report={
                    "ik_timing_ms": 2.5,
                    "seed": "prev_cmd",
                    "max_qpos_cmd_delta_deg": 15.0,
                },
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
        self.assertEqual(result.action_id, 31)
        self.assertEqual(result.ring_sequence, 41)
        self.assertEqual(result.ik_timing_ms, 2.5)
        self.assertEqual(result.ik_seed, "prev_cmd")
        self.assertEqual(result.max_qpos_delta_to_measured_deg, 15.0)

    def test_keyboard_diagnostics_track_publication_and_observed_sdk_progress(
        self,
    ) -> None:
        loop_stats = SimpleNamespace(
            deadline_overrun_count=2,
            missed_slot_count=3,
        )
        arm_state = {
            "last_cmd_seq": 10,
            "tracking_err": np.deg2rad(1.0),
            "qvel": np.deg2rad(np.full(7, 2.0)),
        }
        diagnostics = _KeyboardMotionDiagnostics()
        diagnostics.begin(frame=100, arm_state=arm_state, loop_stats=loop_stats)
        diagnostics.observe_publication(
            _KeyboardPublishResult(
                _KeyboardPublishStatus.PUBLISHED,
                action_id=13,
                ring_sequence=21,
                ik_timing_ms=4.5,
                ik_seed="prev_cmd",
                max_qpos_delta_to_measured_deg=5.0,
            ),
            command_step_rad=np.deg2rad(3.0),
        )

        diagnostics.observe_feedback(arm_state)
        applied_state = {
            **arm_state,
            "last_cmd_seq": 13,
            "tracking_err": np.deg2rad(4.0),
            "qvel": np.deg2rad(np.full(7, 8.0)),
        }
        diagnostics.observe_feedback(applied_state)
        text = diagnostics.format(frame=102, loop_stats=loop_stats)

        self.assertEqual(diagnostics.published_count, 1)
        self.assertEqual(diagnostics.observed_sdk_updates, 1)
        self.assertEqual(diagnostics.max_observed_sdk_action_gap, 1)
        self.assertEqual(diagnostics.max_pending_frames, 1)
        self.assertEqual(diagnostics.max_command_step_action_id, 13)
        self.assertEqual(diagnostics.max_tracking_error_action_id, 13)
        self.assertIn("latest=13/13", text)
        self.assertIn("cmd_step_max=3.00deg@13(prev_cmd)", text)
        self.assertIn("track_max=4.00deg@13", text)
        self.assertIn("ik=4.5/4.5ms seed=prev_cmd", text)


if __name__ == "__main__":
    unittest.main()
