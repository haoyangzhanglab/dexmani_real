from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from dexmani_real.control.publication import (
    CommandPublishStatus,
    _arm_feedback_snapshot,
)
from dexmani_real.ipc.schema import ARM_STATE_DTYPE
from dexmani_real.utils.feedback import validate_arm_feedback


class _Ring:
    def __init__(self, frame: np.ndarray | None) -> None:
        self.frame = frame

    def read_latest(self):
        return None if self.frame is None else (self.frame, 0, 1)


class _Shared:
    def __init__(self, frame: np.ndarray | None) -> None:
        self.arm_state_ring = _Ring(frame)


def arm_frame(*, now_ns: int, error_code: int = 0, age_s: float = 0.01) -> np.ndarray:
    frame = np.zeros(1, dtype=ARM_STATE_DTYPE)
    frame["connected"][0] = 1
    frame["state_valid"][0] = 1
    frame["error_code"][0] = error_code
    frame["source_monotonic_ns"][0] = now_ns - int(age_s * 1e9)
    frame["qpos"][0] = np.linspace(0.0, 0.6, 7)
    frame["qvel"][0] = 0.0
    frame["last_cmd_seq"][0] = 7
    return frame


class ArmFeedbackHealthTest(unittest.TestCase):
    def test_canonical_validator_rejects_controller_error(self) -> None:
        issue = validate_arm_feedback(
            connected=True,
            error_code=31,
            state_valid=True,
            source_monotonic_ns=1,
            now_monotonic_ns=2,
            max_age_s=0.5,
            qpos=np.zeros(7),
            qvel=np.zeros(7),
        )
        self.assertEqual(issue, "arm controller error C31")

    def test_publication_snapshot_rejects_stale_and_faulted_feedback(self) -> None:
        now_ns = 2_000_000_000
        cases = (
            (arm_frame(now_ns=now_ns, error_code=31), "controller error C31"),
            (arm_frame(now_ns=now_ns, age_s=1.0), "stale"),
        )
        with patch(
            "dexmani_real.control.publication.time.monotonic_ns",
            return_value=now_ns,
        ):
            for frame, detail in cases:
                with self.subTest(detail=detail):
                    snapshot, rejection = _arm_feedback_snapshot(
                        _Shared(frame),
                        None,
                        arm_feedback_max_age_s=0.1,
                    )
                    self.assertIsNone(snapshot)
                    assert rejection is not None
                    self.assertEqual(
                        rejection.status,
                        CommandPublishStatus.ARM_FEEDBACK_UNHEALTHY,
                    )
                    self.assertIn(detail, rejection.detail)

    def test_publication_snapshot_accepts_fresh_healthy_feedback(self) -> None:
        now_ns = 2_000_000_000
        with patch(
            "dexmani_real.control.publication.time.monotonic_ns",
            return_value=now_ns,
        ):
            snapshot, rejection = _arm_feedback_snapshot(
                _Shared(arm_frame(now_ns=now_ns)),
                None,
                arm_feedback_max_age_s=0.1,
            )
        self.assertIsNone(rejection)
        assert snapshot is not None
        self.assertEqual(snapshot.last_cmd_seq, 7)
        np.testing.assert_allclose(snapshot.qpos, np.linspace(0.0, 0.6, 7))


if __name__ == "__main__":
    unittest.main()
