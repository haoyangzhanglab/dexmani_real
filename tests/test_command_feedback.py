"""Offline tests for the command-feedback temporal transaction.

These exercise ``control.publication`` against a fake clock and synthetic ring
frames built on the real ``ARM_STATE_DTYPE``/``HAND_STATE_DTYPE``. They never
open hardware or an SDK, and they do not import the deployment runtime (so they
run without scipy/pinocchio/dexmani_policy).

The core regression documents the pre-read ``now`` race: a frame committed after
a hypothetical old ``now`` but before ``read_latest()`` returned was previously
misclassified ``FUTURE_TIMESTAMP``. The new ordering contract
``0 < source <= ring_commit <= validation_now`` (and age within ``max_age``)
governs admission instead.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from dexmani_real.ipc.schema import ARM_STATE_DTYPE, HAND_STATE_DTYPE
from dexmani_real.utils.feedback import (
    FeedbackIssueCode,
    diagnose_feedback_timestamp_order,
)

import dexmani_real.control.publication as pub

_MS = 1_000_000


class _FakeClock:
    def __init__(self, ns: int) -> None:
        self.ns = ns

    def monotonic_ns(self) -> int:
        return self.ns

    def monotonic(self) -> float:
        return self.ns / 1e9


class _Ring:
    """Fake ring returning one verified ``(data, commit_ns, sequence)`` frame."""

    def __init__(self, record: np.ndarray, commit_ns: int) -> None:
        self._frame = (record, commit_ns, 1)

    def read_latest(self):
        return self._frame


class _AdvancingRing:
    """Ring whose read advances the shared clock before returning its frame.

    Models a producer committing a frame *during* the read itself: the clock
    jumps forward past the frame's source timestamp. This is exactly the
    pre-read-``now`` race the old implementation misclassified as
    ``FUTURE_TIMESTAMP``.
    """

    def __init__(
        self, clock: "_FakeClock", record: np.ndarray, commit_ns: int, advance_to_ns: int
    ) -> None:
        self._clock = clock
        self._frame = (record, commit_ns, 1)
        self._advance_to_ns = advance_to_ns

    def read_latest(self):
        self._clock.ns = self._advance_to_ns
        return self._frame


def _arm_record(source_ns: int) -> np.ndarray:
    record = np.zeros(1, dtype=ARM_STATE_DTYPE)
    record["connected"] = 1
    record["error_code"] = 0
    record["state_valid"] = 1
    record["source_monotonic_ns"] = source_ns
    record["last_cmd_accepted_monotonic_ns"] = source_ns
    return record


def _hand_record(source_ns: int) -> np.ndarray:
    record = np.zeros(1, dtype=HAND_STATE_DTYPE)
    record["connected"] = 1
    record["state_valid"] = 1
    record["source_monotonic_ns"] = source_ns
    record["accepted_target_monotonic_ns"] = source_ns
    return record


def _shared(arm_source_ns: int, arm_commit_ns: int, hand_source_ns: int, hand_commit_ns: int):
    return SimpleNamespace(
        arm_state_ring=_Ring(_arm_record(arm_source_ns), arm_commit_ns),
        hand_state_ring=_Ring(_hand_record(hand_source_ns), hand_commit_ns),
    )


class TimestampOrderDiagnoseTest(unittest.TestCase):
    """Direct branch coverage of the provenance/freshness predicate."""

    def _issue(self, **kwargs):
        base = dict(
            source_monotonic_ns=100,
            ring_commit_monotonic_ns=110,
            validation_now_ns=120,
            max_age_s=0.1,
            modality="arm",
        )
        base.update(kwargs)
        return diagnose_feedback_timestamp_order(**base)

    def test_healthy(self):
        self.assertIsNone(self._issue())

    def test_missing_source_timestamp(self):
        issue = self._issue(source_monotonic_ns=0)
        self.assertEqual(issue.code, FeedbackIssueCode.MISSING_TIMESTAMP)

    def test_source_after_ring_commit(self):
        issue = self._issue(source_monotonic_ns=120, ring_commit_monotonic_ns=110)
        self.assertEqual(issue.code, FeedbackIssueCode.TIMESTAMP_ORDER)

    def test_ring_commit_after_validation_now(self):
        issue = self._issue(
            source_monotonic_ns=100,
            ring_commit_monotonic_ns=130,
            validation_now_ns=120,
        )
        self.assertEqual(issue.code, FeedbackIssueCode.TIMESTAMP_ORDER)

    def test_stale(self):
        issue = self._issue(
            source_monotonic_ns=100,
            ring_commit_monotonic_ns=105,
            validation_now_ns=100 + 200_000_000,
            max_age_s=0.1,
        )
        self.assertEqual(issue.code, FeedbackIssueCode.STALE)


class ReadCommandFeedbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = _FakeClock(1_000 * _MS)
        patcher = mock.patch.object(pub, "time", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_legal_concurrent_frame_is_not_future(self):
        # The frame commits during the read itself: the clock starts at
        # T0=990ms, ``read_latest()`` advances it to 1000ms, and the returned
        # frame's source (995ms) is AFTER the hypothetical pre-read now (990ms)
        # but BEFORE the post-read now (1000ms). The old implementation
        # captured ``now`` before reading the ring and would misclassify this
        # legal concurrent frame as FUTURE_TIMESTAMP; the post-read ordering
        # accepts it and satisfies ``0 < source <= ring_commit <= validation_now``.
        self.clock.ns = 990 * _MS
        shared = SimpleNamespace(
            arm_state_ring=_AdvancingRing(
                self.clock, _arm_record(995 * _MS), 998 * _MS, advance_to_ns=1_000 * _MS
            ),
            hand_state_ring=_AdvancingRing(
                self.clock, _hand_record(996 * _MS), 998 * _MS, advance_to_ns=1_000 * _MS
            ),
        )
        snapshot, reason, issue = pub.read_command_feedback(
            shared, require_hand=True, arm_max_age_s=0.15, hand_max_age_s=0.15
        )
        self.assertIsNotNone(snapshot)
        self.assertIsNone(issue)
        self.assertEqual(snapshot.arm_source_monotonic_ns, 995 * _MS)
        self.assertEqual(snapshot.arm_ring_commit_monotonic_ns, 998 * _MS)
        self.assertEqual(snapshot.hand_source_monotonic_ns, 996 * _MS)
        self.assertEqual(snapshot.hand_ring_commit_monotonic_ns, 998 * _MS)
        self.assertEqual(snapshot.validation_now_ns, 1_000 * _MS)
        # Explicit ordering invariant per modality: no future source, no
        # out-of-order commit.
        for source, commit in (
            (snapshot.arm_source_monotonic_ns, snapshot.arm_ring_commit_monotonic_ns),
            (snapshot.hand_source_monotonic_ns, snapshot.hand_ring_commit_monotonic_ns),
        ):
            self.assertGreater(source, 0)
            self.assertLessEqual(source, commit)
            self.assertLessEqual(commit, snapshot.validation_now_ns)

    def test_source_after_ring_commit_is_fatal(self):
        shared = _shared(arm_source_ns=1_000 * _MS, arm_commit_ns=999 * _MS,
                         hand_source_ns=1_000 * _MS, hand_commit_ns=999 * _MS)
        snapshot, reason, issue = pub.read_command_feedback(
            shared, require_hand=False, arm_max_age_s=0.15, hand_max_age_s=0.15
        )
        self.assertIsNone(snapshot)
        self.assertEqual(issue.code, FeedbackIssueCode.TIMESTAMP_ORDER)

    def test_ring_commit_after_validation_now_is_fatal(self):
        shared = _shared(arm_source_ns=1_000 * _MS, arm_commit_ns=1_005 * _MS,
                         hand_source_ns=1_000 * _MS, hand_commit_ns=1_005 * _MS)
        snapshot, reason, issue = pub.read_command_feedback(
            shared, require_hand=False, arm_max_age_s=0.15, hand_max_age_s=0.15
        )
        self.assertIsNone(snapshot)
        self.assertEqual(issue.code, FeedbackIssueCode.TIMESTAMP_ORDER)

    def test_stale_source_is_recoverable(self):
        shared = _shared(arm_source_ns=800 * _MS, arm_commit_ns=810 * _MS,
                         hand_source_ns=800 * _MS, hand_commit_ns=810 * _MS)
        snapshot, reason, issue = pub.read_command_feedback(
            shared, require_hand=False, arm_max_age_s=0.1, hand_max_age_s=0.1
        )
        self.assertIsNone(snapshot)
        self.assertEqual(issue.code, FeedbackIssueCode.STALE)

    def test_require_hand_healthy_snapshot(self):
        shared = _shared(arm_source_ns=995 * _MS, arm_commit_ns=998 * _MS,
                         hand_source_ns=996 * _MS, hand_commit_ns=998 * _MS)
        snapshot, reason, issue = pub.read_command_feedback(
            shared, require_hand=True, arm_max_age_s=0.15, hand_max_age_s=0.15
        )
        self.assertIsNotNone(snapshot)
        self.assertIsNone(issue)
        self.assertIsNotNone(snapshot.hand_qpos)
        self.assertEqual(snapshot.hand_source_monotonic_ns, 996 * _MS)
        self.assertEqual(snapshot.hand_ring_commit_monotonic_ns, 998 * _MS)

    def test_standalone_read_hand_feedback_unchanged(self):
        shared = _shared(arm_source_ns=995 * _MS, arm_commit_ns=998 * _MS,
                         hand_source_ns=996 * _MS, hand_commit_ns=998 * _MS)
        hand_feedback, reason, issue = pub.read_hand_feedback(
            shared, max_age_s=0.15
        )
        self.assertIsNotNone(hand_feedback)
        self.assertIsNone(issue)
        self.assertEqual(hand_feedback.source_monotonic_ns, 996 * _MS)
        self.assertEqual(hand_feedback.ring_commit_monotonic_ns, 998 * _MS)


class FixedArmIdentityTest(unittest.TestCase):
    def test_axis_identity_is_the_hardware_model_not_an_experiment_setting(self):
        from types import SimpleNamespace
        from dexmani_real.robot.drivers.xarm7 import XArm7

        # No constructor, connection, discovery or vendor SDK is used.
        arm = XArm7.__new__(XArm7)
        for axes in (6, 7, 8):
            with self.subTest(axes=axes):
                arm._api = SimpleNamespace(axis=axes)
                if axes == 7:
                    arm._wait_for_axis_report(on_poll=None)
                else:
                    with self.assertRaises(RuntimeError):
                        arm._wait_for_axis_report(on_poll=None)


if __name__ == "__main__":
    unittest.main()
