import unittest

import numpy as np

from dexmani_real.deployment.schedule import (
    ActionPlan, PolicyQuery, control_slot, decision_expires_at_ns, handoff_plan,
    plan_rejection, prefetch_handoff, rtc_prefix_request, run_budget,
)


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.query = PolicyQuery(2, 7, 10, 1000, 12, np.arange(8 * 19).reshape(8, 19))
        self.options = dict(run_id=7, mode="async", action_steps=4, now_ns=100, authorized=True)

    def test_early_result_waits_for_handoff(self):
        self.assertEqual(handoff_plan(self.query, slot=11, **self.options), (None, None))

    def test_late_result_keeps_query_anchor(self):
        plan, issue = handoff_plan(self.query, slot=14, **self.options)
        self.assertIsNone(issue)
        self.assertEqual((plan.anchor_slot, 14 - plan.anchor_slot, plan.segment_end), (10, 4, 18))

    def test_pending_async_keeps_old_tail_but_rtc_rejects(self):
        self.query.result = None
        self.assertEqual(handoff_plan(self.query, slot=14, **self.options), (None, None))
        _, issue = handoff_plan(self.query, slot=12, **dict(self.options, mode="rtc"))
        self.assertEqual(issue.detail, "handoff_result_not_ready")
        plan = ActionPlan(np.zeros((8, 19)), 10, 14, 1000, {}, 1)
        self.assertIsNone(plan_rejection(plan, 17, "async", 999))
        self.assertEqual(plan_rejection(plan, 18, "async", 999).detail, "plan_exhausted_without_handoff")

    def test_old_run_and_revoked_authority_never_handoff(self):
        self.assertEqual(handoff_plan(self.query, slot=14, **dict(self.options, run_id=8)), (None, None))
        self.assertEqual(handoff_plan(self.query, slot=14, **dict(self.options, authorized=False))[1].detail,
                         "handoff_authority_revoked")

    def test_deadline_cannot_be_renewed_by_handoff(self):
        plan, _ = handoff_plan(self.query, slot=14, **self.options)
        self.assertEqual(plan.expires_at_ns, 1000)
        self.assertEqual(plan_rejection(plan, 14, "async", 1000).detail, "dispatch_deadline_expired")
        self.assertEqual(handoff_plan(self.query, slot=14, **dict(self.options, now_ns=1000))[1].detail,
                         "handoff_decision_expired")
        self.assertEqual(decision_expires_at_ns({"joint": (100, 120)}, 150, 100), 201)
        self.assertEqual(decision_expires_at_ns({"joint": (100, 120)}, 200, 100), 201)
        self.assertIsNone(decision_expires_at_ns({"joint": (100, 120)}, 201, 100))
        self.assertIsNone(decision_expires_at_ns({"joint": (200,)}, 150, 100))

    def test_sync_resume_uses_actual_clock_and_missed_slots_do_not_advance(self):
        args = dict(next_step_ns=200, last_slot=3, dt_ns=100)
        self.assertEqual(control_slot(now_ns=150, synchronous=True, resume_ready=False, **args), (None, 200, 0))
        self.assertEqual(control_slot(now_ns=150, synchronous=True, resume_ready=True, **args), (4, 250, 0))
        self.assertEqual(control_slot(now_ns=450, synchronous=False, resume_ready=False, **args), (4, 200, 2))

    def test_prefetch_and_rtc_prefix_are_owned(self):
        plan = ActionPlan(np.arange(8 * 19).reshape(8, 19), 10, 16, 1000, {}, 1)
        options = dict(mode="rtc", prefetch_steps=2, query_pending=False, model_pending=False, history_ready=True)
        self.assertEqual(prefetch_handoff(plan, slot=13, **options), (None, None))
        self.assertEqual(prefetch_handoff(plan, slot=14, **options), (16, None))
        self.assertEqual(prefetch_handoff(plan, slot=14, **dict(options, model_pending=True))[1].detail,
                         "prefetch_unavailable")
        prefix, slots = rtc_prefix_request(plan, 14, 2)
        self.assertEqual(list(slots), [14, 15])
        np.testing.assert_array_equal(prefix, plan.actions[4:])
        prefix[0, 0] = -1
        self.assertNotEqual(plan.actions[4, 0], -1)

    def test_wait_budget_wins_tie(self):
        self.assertEqual(run_budget(run_id=1, started_ns=100, max_running_s=1,
                                    wait_started_ns=100, max_wait_s=1), (1_000_000_100, "wait"))
