"""Offline regressions for the shared EEF/fingertip history geometry helpers.

No hardware and no GUI.  These tests pin the canonical EEF history contract
(``[T,9]`` position+rot6d from aligned qpos), numerical parity of the moved
fingertip history helper with the legacy per-frame loop, and the invariant
that producing ``eef_pose`` and ``fingertip_points`` together runs exactly
one Arm FK per timestep.  Run with:

    python -m unittest discover -s tests -p 'test_eef_fingertip_geometry.py'
"""

from __future__ import annotations

import unittest

import numpy as np

from dexmani_real.planning.kinematics.arm_fk import (
    EEF_POSE_ALGORITHM_ID,
    EEF_POSE_COMPONENTS,
    EEF_POSE_DERIVATION,
    EEF_POSE_FRAME,
    compute_eef_pose_history_xarm_base,
    make_arm_fk,
)
from dexmani_real.planning.kinematics.fingertip import (
    compute_fingertip_history_xarm_base,
    compute_fingertip_points_xarm_base,
)
from dexmani_real.planning.kinematics.pose import validate_canonical_rot6d

_MOUNT_P = np.array([0.0, 0.0, 0.1], dtype=np.float64)
_MOUNT_Q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


class _CountingFakeArmFK:
    """Deterministic canonical-pose fake that counts Arm FK invocations."""

    def __init__(self) -> None:
        self.calls = 0

    def compute(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != (7,) or not np.all(np.isfinite(qpos)):
            raise ValueError("fake arm FK received invalid qpos")
        self.calls += 1
        eef_pos = np.array(
            [float(qpos[0]), float(qpos[1]), float(qpos[2])], dtype=np.float64
        )
        rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        return eef_pos, rot6d


class _FakeHandFK:
    """Deterministic fingertip fake in hand-base coordinates."""

    def is_ready(self) -> bool:
        return True

    def compute_tip_positions_in_handbase(self, hand_qpos: np.ndarray) -> np.ndarray:
        hand = np.asarray(hand_qpos, dtype=np.float64)
        base = np.arange(15, dtype=np.float64).reshape(5, 3) * 0.01
        return base + float(hand[0]) * 0.001


def _sample_qpos_history(count: int, *, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    arm = rng.uniform(-0.5, 0.5, size=(count, 7))
    hand = rng.uniform(-0.3, 0.3, size=(count, 12))
    return arm, hand


class TestEefPoseHistory(unittest.TestCase):
    def test_output_shape_dtype_and_canonical_rot6d(self) -> None:
        arm, _ = _sample_qpos_history(5)
        poses = compute_eef_pose_history_xarm_base(arm)
        self.assertEqual(poses.shape, (5, 9))
        self.assertEqual(poses.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(poses)))
        for row in range(poses.shape[0]):
            validate_canonical_rot6d(
                poses[row, 3:], label=f"test row {row} eef rot6d"
            )

    def test_matches_per_step_canonical_fk(self) -> None:
        arm, _ = _sample_qpos_history(3, seed=11)
        poses = compute_eef_pose_history_xarm_base(arm)
        arm_fk = make_arm_fk()
        for index in range(len(arm)):
            eef_pos, eef_rot6d = arm_fk.compute(arm[index])
            expected = np.concatenate([eef_pos, eef_rot6d])
            np.testing.assert_allclose(poses[index], expected, rtol=0.0, atol=0.0)

    def test_rejects_invalid_input(self) -> None:
        with self.assertRaises(ValueError):
            compute_eef_pose_history_xarm_base(np.zeros((4, 6)))
        with self.assertRaises(ValueError):
            compute_eef_pose_history_xarm_base(np.zeros((4,)))
        bad = np.zeros((4, 7))
        bad[2, 3] = np.nan
        with self.assertRaises(ValueError):
            compute_eef_pose_history_xarm_base(bad)

    def test_semantic_identity_constants(self) -> None:
        self.assertEqual(EEF_POSE_FRAME, "xarm_base")
        self.assertEqual(EEF_POSE_COMPONENTS, "position_m(3)+rot6d(6)")
        self.assertEqual(EEF_POSE_DERIVATION, "canonical_arm_fk_from_aligned_qpos")
        self.assertEqual(EEF_POSE_ALGORITHM_ID, "xarm7_custom_eef_pinocchio_fk_v1")


class TestFingertipHistory(unittest.TestCase):
    def test_parity_with_legacy_per_frame_loop(self) -> None:
        arm, hand = _sample_qpos_history(4, seed=3)
        arm_fk = _CountingFakeArmFK()
        hand_fk = _FakeHandFK()
        shared = compute_fingertip_history_xarm_base(
            arm,
            hand,
            hand_fk=hand_fk,
            handbase_position_eef_m=_MOUNT_P,
            handbase_quat_eef_wxyz=_MOUNT_Q,
            arm_fk=arm_fk,
        )
        legacy = np.asarray(
            [
                compute_fingertip_points_xarm_base(
                    arm[index],
                    hand[index],
                    arm_fk=arm_fk,
                    hand_fk=hand_fk,
                    handbase_position_eef_m=_MOUNT_P,
                    handbase_quat_eef_wxyz=_MOUNT_Q,
                )
                for index in range(len(arm))
            ],
            dtype=np.float32,
        )
        self.assertEqual(shared.shape, (4, 5, 3))
        self.assertEqual(shared.dtype, np.float32)
        np.testing.assert_array_equal(shared, legacy)

    def test_precomputed_eef_history_skips_arm_fk(self) -> None:
        arm, hand = _sample_qpos_history(4, seed=5)
        arm_fk = _CountingFakeArmFK()
        hand_fk = _FakeHandFK()
        eef_history = compute_eef_pose_history_xarm_base(arm, arm_fk=arm_fk)
        self.assertEqual(arm_fk.calls, 4)
        from_history = compute_fingertip_history_xarm_base(
            arm,
            hand,
            hand_fk=hand_fk,
            handbase_position_eef_m=_MOUNT_P,
            handbase_quat_eef_wxyz=_MOUNT_Q,
            eef_pose_history=eef_history,
        )
        self.assertEqual(arm_fk.calls, 4, "precomputed EEF must not re-run Arm FK")
        direct = compute_fingertip_history_xarm_base(
            arm,
            hand,
            hand_fk=hand_fk,
            handbase_position_eef_m=_MOUNT_P,
            handbase_quat_eef_wxyz=_MOUNT_Q,
            arm_fk=_CountingFakeArmFK(),
        )
        np.testing.assert_array_equal(from_history, direct)

    def test_one_arm_fk_per_timestep_for_eef_and_fingertip(self) -> None:
        arm, hand = _sample_qpos_history(6, seed=13)
        arm_fk = _CountingFakeArmFK()
        hand_fk = _FakeHandFK()
        eef_history = compute_eef_pose_history_xarm_base(arm, arm_fk=arm_fk)
        compute_fingertip_history_xarm_base(
            arm,
            hand,
            hand_fk=hand_fk,
            handbase_position_eef_m=_MOUNT_P,
            handbase_quat_eef_wxyz=_MOUNT_Q,
            eef_pose_history=eef_history,
        )
        self.assertEqual(arm_fk.calls, len(arm))

    def test_requires_arm_fk_without_eef_history(self) -> None:
        arm, hand = _sample_qpos_history(2)
        with self.assertRaises(ValueError):
            compute_fingertip_history_xarm_base(
                arm,
                hand,
                hand_fk=_FakeHandFK(),
                handbase_position_eef_m=_MOUNT_P,
                handbase_quat_eef_wxyz=_MOUNT_Q,
            )

    def test_rejects_invalid_inputs(self) -> None:
        arm, hand = _sample_qpos_history(3)
        hand_fk = _FakeHandFK()
        with self.assertRaises(ValueError):
            compute_fingertip_history_xarm_base(
                arm,
                np.zeros((2, 12)),
                hand_fk=hand_fk,
                handbase_position_eef_m=_MOUNT_P,
                handbase_quat_eef_wxyz=_MOUNT_Q,
                arm_fk=_CountingFakeArmFK(),
            )
        bad_history = np.zeros((3, 9))
        bad_history[1, 0] = np.nan
        with self.assertRaises(ValueError):
            compute_fingertip_history_xarm_base(
                arm,
                hand,
                hand_fk=hand_fk,
                handbase_position_eef_m=_MOUNT_P,
                handbase_quat_eef_wxyz=_MOUNT_Q,
                eef_pose_history=bad_history,
            )
        with self.assertRaises(ValueError):
            compute_fingertip_history_xarm_base(
                arm,
                hand,
                hand_fk=hand_fk,
                handbase_position_eef_m=_MOUNT_P,
                handbase_quat_eef_wxyz=_MOUNT_Q,
                eef_pose_history=np.zeros((3, 8)),
            )


if __name__ == "__main__":
    unittest.main()
