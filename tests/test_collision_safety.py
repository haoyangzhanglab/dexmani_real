from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np

from dexmani_real.config.defaults import arm, hand
from dexmani_real.planning.collision_model import CollisionModel
from dexmani_real.planning.constants import HAND_SDK_TO_URDF_IDX
from dexmani_real.planning.ik_candidates import IKCandidateManager
from dexmani_real.policy.vr_teleop_policy import _sanitize_hand_command
from dexmani_real.robot.arm_loop import _latch_collision_fault, _require_sdk_ok


class CollisionModelSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = CollisionModel(hand_dof=True)
        cls.arm_home = np.asarray(arm.home_qpos, dtype=np.float64)
        cls.hand_home = np.deg2rad(np.asarray(hand.home_qpos_deg, dtype=np.float64))

    def test_unset_hand_uses_home_in_urdf_order(self) -> None:
        self.model._hand_qpos = None
        full = self.model._to_full_qpos(self.arm_home)
        expected = self.hand_home[list(HAND_SDK_TO_URDF_IDX)]
        np.testing.assert_allclose(full[7:], expected)

    def test_invalid_collision_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            self.model.check_self_collision(np.full(7, np.nan))
        for step_size in (0.0, -0.1, float("nan")):
            with self.assertRaises(ValueError):
                self.model.check_segment_collision_free(self.arm_home, self.arm_home, step_size)

    def test_single_waypoint_collision_is_reported(self) -> None:
        self.model.set_hand_qpos(self.hand_home)
        manager = object.__new__(IKCandidateManager)
        manager._cm = self.model
        manager.dof = 7
        colliding = np.deg2rad(np.array([-200.0, -71.4, 272.7, 35.7, -32.9, 110.8, 149.3]))
        self.assertTrue(self.model.check_self_collision(colliding))
        report = manager.check_path_collisions(colliding.reshape(1, 7))
        self.assertTrue(report["path_self_collision"])
        self.assertEqual(report["collision_waypoint_index"], 0)

    def test_static_home_transition_is_free(self) -> None:
        self.assertTrue(
            self.model.check_transition_collision_free(self.arm_home, self.arm_home, self.hand_home, self.hand_home)
        )

    def test_minimum_hand_frame_height_is_finite_and_orientation_aware(self) -> None:
        self.model.set_hand_qpos(self.hand_home)
        home_z = self.model.minimum_hand_frame_z(self.arm_home)
        tilted = self.arm_home.copy()
        tilted[5] += 0.5
        tilted_z = self.model.minimum_hand_frame_z(tilted)
        self.assertTrue(np.isfinite(home_z))
        self.assertTrue(np.isfinite(tilted_z))
        self.assertNotAlmostEqual(home_z, tilted_z, places=5)

    def test_active_collision_pair_categories_match_documentation(self) -> None:
        frame_names = [
            self.model._model.frames[geometry.parentFrame].name
            for geometry in self.model._collision_model.geometryObjects
        ]
        counts = {"arm_arm": 0, "arm_hand": 0, "hand_hand": 0}
        for pair in self.model._collision_model.collisionPairs:
            first_hand = frame_names[pair.first].startswith("right_hand_")
            second_hand = frame_names[pair.second].startswith("right_hand_")
            category = (
                "hand_hand" if first_hand and second_hand else "arm_hand" if first_hand != second_hand else "arm_arm"
            )
            counts[category] += 1
        self.assertEqual(counts, {"arm_arm": 17, "arm_hand": 238, "hand_hand": 0})


class CommandSafetyTests(unittest.TestCase):
    def test_hand_command_matches_worker_clipping(self) -> None:
        lower = np.asarray(hand.qpos_min_rad)
        upper = np.asarray(hand.qpos_max_rad)
        previous = np.zeros(12)
        command = upper + 1.0
        actual = _sanitize_hand_command(command, previous, lower, upper, None)
        np.testing.assert_allclose(actual, upper)

        stepped = _sanitize_hand_command(command, previous, lower, upper, 0.1)
        np.testing.assert_allclose(stepped, np.clip(upper, -0.1, 0.1))

    def test_hand_command_rejects_nan(self) -> None:
        with self.assertRaises(ValueError):
            _sanitize_hand_command(
                np.full(12, np.nan),
                np.zeros(12),
                np.asarray(hand.qpos_min_rad),
                np.asarray(hand.qpos_max_rad),
                None,
            )

    def test_sdk_return_codes_are_checked(self) -> None:
        _require_sdk_ok("ok", 0)
        with self.assertRaises(RuntimeError):
            _require_sdk_ok("failed", 1)

    def test_c31_latches_error_for_main_owned_fault_transition(self) -> None:
        shared = Mock()
        shared.error_state.value = False
        arm_api = Mock()
        arm_api.get_c31_error_info.return_value = (0, [3, 1.0, 2.0])
        _latch_collision_fault(shared, arm_api, 31)
        self.assertTrue(shared.error_state.value)


if __name__ == "__main__":
    unittest.main()
