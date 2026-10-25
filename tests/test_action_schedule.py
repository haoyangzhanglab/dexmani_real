"""纯调度的纳秒端点、交接和预取契约；没有墙钟等待或设备发送。"""

import unittest

import numpy as np

from dexmani_real.deployment.rollout.action_schedule import (
    ActionPlan,
    PolicyQuery,
    control_slot,
    decision_expires_at_ns,
    handoff_plan,
    plan_rejection,
    prefetch_handoff,
    rtc_prefix_request,
    run_budget,
)


class ActionScheduleTests(unittest.TestCase):
    def query(self, *, result=True, handoff=2, run_id=7):
        return PolicyQuery(1, run_id, 0, 200, handoff, np.arange(95).reshape(5, 19) if result else None)

    def handoff(self, query, *, mode='async', slot=2, now=100, authorized=True):
        return handoff_plan(query, run_id=7, slot=slot, mode=mode, action_steps=3,
                            now_ns=now, authorized=authorized)

    def test_source_age_has_inclusive_endpoint(self):
        self.assertEqual(decision_expires_at_ns([100, 105], 110, 10), 111)
        for stamps, now in (([100], 111), ([], 110), ([0], 110), ([111], 110)):
            self.assertIsNone(decision_expires_at_ns(stamps, now, 10))

    def test_sync_bootstrap_waits_then_anchors_at_consumption(self):
        query = self.query(result=False, handoff=None)
        self.assertEqual(self.handoff(query, mode='sync', slot=4), (None, None))
        query.result = np.zeros((5, 19))
        plan, error = self.handoff(query, mode='sync', slot=4)
        self.assertIsNone(error)
        self.assertEqual((plan.anchor_slot, plan.segment_end), (4, 7))

    def test_identity_and_authority_rejections(self):
        self.assertEqual(self.handoff(self.query(run_id=6)), (None, None))
        plan, error = self.handoff(self.query(), authorized=False)
        self.assertIsNone(plan)
        self.assertEqual(error.detail, 'handoff_authority_revoked')

    def test_async_can_handoff_late_but_not_after_expiry_or_exhaustion(self):
        plan, error = self.handoff(self.query(), slot=3)
        self.assertIsNone(error)
        self.assertEqual((plan.anchor_slot, plan.segment_end), (0, 5))
        self.assertEqual(self.handoff(self.query(result=False), slot=3), (None, None))
        for kwargs, reason in ((dict(slot=5), 'handoff_prediction_exhausted'),
                               (dict(now=200), 'handoff_decision_expired')):
            self.assertEqual(self.handoff(self.query(), **kwargs)[1].detail, reason)

    def test_rtc_requires_exact_handoff_and_ready_result(self):
        self.assertEqual(self.handoff(self.query(), mode='rtc', slot=1), (None, None))
        self.assertEqual(self.handoff(self.query(), mode='rtc', slot=3)[1].detail, 'handoff_slot_missed')
        self.assertEqual(self.handoff(self.query(result=False), mode='rtc')[1].detail, 'handoff_result_not_ready')

    def test_async_old_tail_and_rtc_segment_end_differ(self):
        plan = ActionPlan(np.zeros((5, 19)), 0, 3, 200, {})
        self.assertIsNone(plan_rejection(plan, 3, 'async', 199))
        self.assertEqual(plan_rejection(plan, 5, 'async', 199).detail, 'plan_exhausted_without_handoff')
        self.assertEqual(plan_rejection(plan, 3, 'async', 200).detail, 'submission_deadline_expired')
        self.assertEqual(plan_rejection(plan, 3, 'rtc', 199).detail, 'plan_exhausted_without_handoff')

    def test_prefetch_slot_busy_worker_and_owned_rtc_prefix(self):
        plan = ActionPlan(np.arange(95).reshape(5, 19), 0, 4, 200, {})
        args = dict(prefetch_steps=2, query_pending=False, model_pending=False, history_ready=True)
        self.assertEqual(prefetch_handoff(plan, slot=3, mode='async', **args), (4, None))
        self.assertEqual(prefetch_handoff(plan, slot=3, mode='rtc', **args), (None, None))
        self.assertEqual(prefetch_handoff(plan, slot=2, mode='rtc', **args), (4, None))
        args['model_pending'] = True
        self.assertEqual(prefetch_handoff(plan, slot=2, mode='rtc', **args)[1].detail, 'prefetch_unavailable')
        self.assertEqual(prefetch_handoff(plan, slot=2, mode='async', **args), (None, None))
        prefix, slots = rtc_prefix_request(plan, 2, 2)
        self.assertEqual(list(slots), [2, 3])
        np.testing.assert_array_equal(prefix, plan.actions[2:])
        prefix[:] = -1
        self.assertTrue(np.all(plan.actions >= 0))

    def test_missed_slots_are_reported_without_catchup(self):
        args = dict(next_step_ns=100, last_slot=0, dt_ns=10, synchronous=False, resume_ready=False)
        self.assertEqual(control_slot(now_ns=99, **args), (None, 100, 0))
        self.assertEqual(control_slot(now_ns=100, **args), (1, 110, 0))
        self.assertEqual(control_slot(now_ns=130, **args), (1, 100, 3))
        args.update(synchronous=True, resume_ready=True)
        self.assertEqual(control_slot(now_ns=95, **args), (1, 105, 0))

    def test_wait_deadline_wins_tie(self):
        args = dict(run_id=7, started_ns=100, max_running_s=1, wait_started_ns=100, max_wait_s=1)
        self.assertEqual(run_budget(**args), (1_000_000_100, 'wait'))
        args['run_id'] = None
        self.assertIsNone(run_budget(**args))
