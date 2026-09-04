"""Offline causal-feedback regressions for teleoperation quiescence."""

from __future__ import annotations

import unittest
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.teleop.config import TeleopConfig
from dexmani_real.teleop.control_grid import (
    TeleopGridResources,
    _read_control_grid_observation,
)
from dexmani_real.teleop.control_state import (
    CommandQuiescence,
    CoordinatorDirective,
    TeleopLoopState,
)


class CommandQuiescenceTest(unittest.TestCase):
    def test_begin_boundary_rejects_stale_feedback(self) -> None:
        quiescence = CommandQuiescence()
        self.assertTrue(quiescence.enter("begin", entered_monotonic_ns=100))

        self.assertFalse(
            quiescence.feedback_is_newer(
                arm_source_monotonic_ns=101,
                vr_receive_monotonic_ns=100,
                hand_source_monotonic_ns=101,
            )
        )
        self.assertTrue(quiescence.active)

    def test_fresh_feedback_releases_without_audio_state(self) -> None:
        quiescence = CommandQuiescence()
        self.assertTrue(quiescence.enter("begin", entered_monotonic_ns=100))

        self.assertTrue(
            quiescence.feedback_is_newer(
                arm_source_monotonic_ns=101,
                vr_receive_monotonic_ns=101,
                hand_source_monotonic_ns=101,
            )
        )

        self.assertNotIn(
            "audio",
            {field.name for field in fields(TeleopGridResources)},
        )

    def test_control_grid_reanchors_from_fresh_feedback_without_audio(self) -> None:
        quiescence = CommandQuiescence()
        self.assertTrue(quiescence.enter("begin", entered_monotonic_ns=100))
        ctx = TeleopLoopState(
            prev_qpos_cmd=np.zeros(7, dtype=np.float64),
            prev_hand_qpos=np.zeros(12, dtype=np.float64),
            teleop_active=True,
        )
        shared = SimpleNamespace(arm_state_ring=object(), hand_state_ring=object())
        resources = TeleopGridResources(
            control=SimpleNamespace(arm_mapper=object(), recorder=None),
            command_limits=Mock(),
            quiescence=quiescence,
            camera_freshness=Mock(),
            stage_timer=Mock(),
            validation_warn=Mock(),
            arm_feedback_warn=Mock(),
            hand_fk=None,
            handbase_position_eef_m=np.zeros(3, dtype=np.float64),
            handbase_quat_eef_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            hand_ramp_total_frames=1,
        )
        arm_state = {
            "qpos": np.zeros((1, 7), dtype=np.float64),
            "source_monotonic_ns": np.array([101], dtype=np.int64),
        }
        vr_frame = {"local_recv_ns": 101, "ring_sequence": 1}

        def read_structured(ring, **_kwargs):
            return (arm_state, 0, 1) if ring is shared.arm_state_ring else None

        with (
            patch(
                "dexmani_real.teleop.control_grid.read_causal_structured_frame",
                side_effect=read_structured,
            ),
            patch(
                "dexmani_real.teleop.control_grid.read_vr_frame_causal",
                return_value=vr_frame,
            ),
            patch(
                "dexmani_real.teleop.control_grid.read_hand_tactile_causal",
                return_value=None,
            ),
            patch(
                "dexmani_real.teleop.control_grid.arm_feedback_issue",
                return_value=None,
            ),
            patch(
                "dexmani_real.teleop.control_grid.hand_feedback_issue",
                return_value=None,
            ),
            patch(
                "dexmani_real.teleop.control_grid.advance_arm_feedback_error_count",
                return_value=(0, False),
            ),
            patch(
                "dexmani_real.teleop.control_grid.complete_reanchor",
                return_value=True,
            ) as complete_reanchor,
            patch("dexmani_real.teleop.control_grid.time.monotonic_ns", return_value=200),
        ):
            first_directive, first_observation = _read_control_grid_observation(
                ctx,
                shared,
                TeleopConfig(),
                resources,
                loop_count=1,
                observation_anchor_monotonic_ns=200,
            )
            second_directive, second_observation = _read_control_grid_observation(
                ctx,
                shared,
                TeleopConfig(),
                resources,
                loop_count=2,
                observation_anchor_monotonic_ns=200,
            )

        self.assertIs(first_directive, CoordinatorDirective.CONTINUE)
        self.assertIsNone(first_observation)
        self.assertFalse(quiescence.active)
        complete_reanchor.assert_called_once()
        self.assertIs(second_directive, CoordinatorDirective.NORMAL)
        self.assertIsNotNone(second_observation)


if __name__ == "__main__":
    unittest.main()
