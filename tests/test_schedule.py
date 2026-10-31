import unittest

import numpy as np

from dexmani_real.deployment.schedule import (
    ActionChunk, Prediction, control_slot, handoff_plan, plan_rejection, prefetch_handoff,
)
from dexmani_real.robot.commands import RobotCommand


class ScheduleTests(unittest.TestCase):
    def test_due_slot_skips_time_without_replaying_commands(self):
        self.assertEqual(control_slot(
            now_ns=9, next_step_ns=10, last_slot=2, dt_ns=10, synchronous=True,
        ), (None, 10, 0))
        for mode, slot in (("sync", 3), ("async", 5), ("rtc", 5)):
            with self.subTest(mode=mode):
                self.assertEqual(control_slot(
                    now_ns=39, next_step_ns=10, last_slot=2, dt_ns=10,
                    synchronous=mode == "sync",
                ), (slot, 40, 2))

    def test_bootstrap_keeps_each_mode_observation_anchor(self):
        for mode, anchor in (("sync", 2), ("async", 2), ("rtc", 5)):
            with self.subTest(mode=mode):
                query = Prediction(7, 2, None, 456, np.arange(8 * 21).reshape(8, 21))
                plan, issue = handoff_plan(
                    query, run_id=7, slot=5, mode=mode, action_steps=3, authorized=True,
                )
                self.assertIsNone(issue)
                self.assertEqual((plan.anchor_slot, plan.segment_end), (anchor, 8))
                np.testing.assert_array_equal(plan.actions, query.result)

    def test_async_late_handoff_uses_valid_prediction_tail(self):
        query = Prediction(7, 0, 4, 456, np.zeros((8, 21)))
        plan, issue = handoff_plan(
            query, run_id=7, slot=5, mode="async", action_steps=4, authorized=True,
        )
        self.assertIsNone(issue)
        self.assertEqual((plan.anchor_slot, plan.segment_end), (0, 8))
        self.assertIsNone(plan_rejection(plan, 7, "async"))
        self.assertEqual(plan_rejection(plan, 8, "async"), "plan_exhausted_without_handoff")

    def test_rtc_handoff_requires_exact_slot_and_ready_result(self):
        query = Prediction(7, 0, 4, 456)
        for slot, issue in ((3, None), (4, "handoff_result_not_ready"), (5, "handoff_slot_missed")):
            with self.subTest(slot=slot):
                self.assertEqual(handoff_plan(
                    query, run_id=7, slot=slot, mode="rtc", action_steps=4, authorized=True,
                ), (None, issue))

    def test_old_run_and_revoked_handoff_never_become_plan(self):
        query = Prediction(7, 0, 4, 456, np.zeros((8, 21)))
        self.assertEqual(handoff_plan(
            query, run_id=8, slot=4, mode="async", action_steps=4, authorized=True,
        ), (None, None))
        self.assertEqual(handoff_plan(
            query, run_id=7, slot=4, mode="async", action_steps=4, authorized=False,
        ), (None, "handoff_authority_revoked"))

    def test_prefetch_schedule_preserves_frozen_commands(self):
        command = RobotCommand(7, np.zeros(7), np.zeros(12))
        plan = ActionChunk(np.zeros((8, 21)), 0, 4, {2: command})
        self.assertEqual(prefetch_handoff(
            plan, slot=3, mode="async", prefetch_steps=2, query_pending=False,
        ), (4, None))
        self.assertEqual(prefetch_handoff(
            plan, slot=3, mode="rtc", prefetch_steps=2, query_pending=False,
        ), (None, None))
        self.assertEqual(prefetch_handoff(
            plan, slot=2, mode="rtc", prefetch_steps=2, query_pending=True,
        ), (None, "prefetch_unavailable"))
        self.assertEqual(prefetch_handoff(
            plan, slot=2, mode="rtc", prefetch_steps=2, query_pending=False,
        ), (4, None))
        self.assertIs(plan.frozen[2], command)
        self.assertFalse(command.hand_qpos.flags.writeable)


if __name__ == "__main__":
    unittest.main()
