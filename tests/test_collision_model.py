"""Unit tests for CollisionModel — self-collision, env collision, obstacle lifecycle."""
from __future__ import annotations

import numpy as np
import pytest


class TestCollisionModel7DOF:
    """Tests for 7-DOF (arm-only) CollisionModel."""

    @pytest.fixture(scope="class")
    @classmethod
    def cm(cls):
        from dexmani_real.planning.collision_model import CollisionModel

        return CollisionModel(hand_dof=False)

    # ── Self-collision ──

    def test_home_pose_no_self_collision(self, cm):
        """Home pose (all zeros) should be self-collision-free."""
        qpos = np.zeros(7, dtype=np.float64)
        assert not cm.check_self_collision(qpos), "Home pose should not have self-collision"

    def test_self_collision_details_home(self, cm):
        """CollisionInfo for home pose should report no collision."""
        qpos = np.zeros(7, dtype=np.float64)
        info = cm.check_self_collision_details(qpos)
        assert not info, f"Home pose should be collision-free, got: {info.summary}"

    # ── Environment collision ──

    def test_env_collision_fast_no_obstacle(self, cm):
        """With no obstacles registered, env collision should always be False."""
        qpos = np.zeros(7, dtype=np.float64)
        assert not cm.check_env_collision_fast(qpos)

    def test_env_collision_no_obstacle(self, cm):
        """Full env check with no obstacles should return False."""
        qpos = np.zeros(7, dtype=np.float64)
        assert not cm.check_env_collision(qpos)

    def test_env_collision_fast_with_table_home(self, cm):
        """Home pose with table below robot base: arm is above table, should be safe."""
        # Robot base is at z=0, lowest geometry at ~-0.13m.
        # Table at z=-0.5 places it safely below the robot.
        cm.add_table(table_height=-0.5)
        try:
            qpos = np.zeros(7, dtype=np.float64)
            result = cm.check_env_collision_fast(qpos)
            # Home pose should be safe (arm is well above table)
            assert not result, f"Home pose should be above table, got collision={result}"
        finally:
            cm.clear_obstacles()

    def test_env_collision_full_with_table_home(self, cm):
        """Full two-tier env check at home pose with table below robot."""
        cm.add_table(table_height=-0.5)
        try:
            qpos = np.zeros(7, dtype=np.float64)
            result = cm.check_env_collision(qpos)
            assert not result, f"Full env check should pass at home pose, got collision={result}"
        finally:
            cm.clear_obstacles()

    # ── Teleop combined check ──

    def test_teleop_collision_home_no_obstacle(self, cm):
        """Combined self+env check at home pose, no obstacles."""
        qpos = np.zeros(7, dtype=np.float64)
        has_self, has_env = cm.check_teleop_collision(qpos)
        assert not has_self, "Home pose should not self-collide"
        assert not has_env, "No obstacles → no env collision"

    # ── NaN / Inf guards ──

    def test_nan_qpos_self_collision(self, cm):
        """NaN qpos should not crash the system."""
        qpos = np.full(7, np.nan, dtype=np.float64)
        # Should either return bool or raise ValueError (shape validation)
        try:
            result = cm.check_self_collision(qpos)
            assert isinstance(result, bool)
        except ValueError:
            pass  # shape validation error is acceptable

    def test_nan_qpos_env_fast(self, cm):
        """NaN qpos should not crash env_collision_fast."""
        cm.add_table(table_height=-0.5)
        try:
            qpos = np.full(7, np.nan, dtype=np.float64)
            try:
                result = cm.check_env_collision_fast(qpos)
                assert isinstance(result, bool)
            except ValueError:
                pass  # shape error is acceptable
        finally:
            cm.clear_obstacles()

    # ── Obstacle lifecycle ──

    def test_add_remove_obstacle(self, cm):
        """Add and remove a box obstacle, verify state transitions."""
        assert len(cm._obstacle_names) == 0
        cm.add_box_obstacle("test_box", (0.1, 0.1, 0.1), (0.5, 0.0, 0.0))
        assert "test_box" in cm._obstacle_names
        removed = cm.remove_obstacle("test_box")
        assert removed
        assert "test_box" not in cm._obstacle_names

    def test_remove_nonexistent_obstacle(self, cm):
        """Removing a non-existent obstacle should return False."""
        assert not cm.remove_obstacle("does_not_exist")

    def test_add_duplicate_obstacle_raises(self, cm):
        """Adding an obstacle with duplicate name should raise ValueError."""
        cm.add_box_obstacle("dup_test", (0.1, 0.1, 0.1), (0.5, 0.0, 0.0))
        try:
            with pytest.raises(ValueError):
                cm.add_box_obstacle("dup_test", (0.2, 0.2, 0.2), (0.3, 0.0, 0.0))
        finally:
            cm.clear_obstacles()

    def test_clear_obstacles(self, cm):
        """Clear multiple obstacles."""
        cm.add_box_obstacle("box1", (0.1, 0.1, 0.1), (0.5, 0.0, 0.0))
        cm.add_box_obstacle("box2", (0.2, 0.2, 0.2), (0.3, 0.0, 0.0))
        assert len(cm._obstacle_names) == 2
        count = cm.clear_obstacles()
        assert count == 2
        assert len(cm._obstacle_names) == 0

    # ── Self-collision (triggered) ──

    # Known collision pose: joint2=-2.0 joint3=-2.0 causes link2 ↔ link4 contact.
    _COLLISION_POSE = np.array([0.0, -2.0, -2.0, -1.39, 0.0, 0.0, 0.0], dtype=np.float64)

    def test_self_collision_triggered(self, cm):
        """A known collision pose should be detected."""
        assert cm.check_self_collision(self._COLLISION_POSE), "Known collision pose should trigger self-collision"

    def test_self_collision_details_triggered(self, cm):
        """CollisionInfo from a known collision pose should report contacts."""
        info = cm.check_self_collision_details(self._COLLISION_POSE)
        assert bool(info), "Known collision pose should produce CollisionInfo"
        assert info.num_contacts >= 1, f"Expected ≥1 contacts, got {info.num_contacts}"

    # ── Segment collision checking ──

    def test_segment_collision_free_safe(self, cm):
        """A short safe segment should pass (no collision along the path)."""
        q1 = np.zeros(7, dtype=np.float64)
        q2 = np.full(7, 0.1, dtype=np.float64)
        assert cm.check_segment_collision_free(q1, q2)

    def test_segment_collision_free_crosses_collision(self, cm):
        """A segment that crosses through a collision pose should be flagged."""
        q1 = np.zeros(7, dtype=np.float64)
        q2 = self._COLLISION_POSE
        assert not cm.check_segment_collision_free(q1, q2), "Segment crossing collision pose should fail"

    def test_segment_collision_free_single_step(self, cm):
        """A zero-length segment (same q) should pass trivially."""
        q = np.zeros(7, dtype=np.float64)
        assert cm.check_segment_collision_free(q, q)

    def test_segment_collision_free_custom_step(self, cm):
        """Step size affects detection — coarser steps may miss collisions."""
        q1 = np.zeros(7, dtype=np.float64)
        q2 = self._COLLISION_POSE
        # With a very coarse step (1.0 rad), only the endpoints are checked.
        # Both endpoints may be safe while intermediate states collide,
        # so the coarse check can produce a false negative.
        result_coarse = cm.check_segment_collision_free(q1, q2, step_size=5.0)
        # With fine step (0.02 rad), collision is always caught.
        result_fine = cm.check_segment_collision_free(q1, q2, step_size=0.02)
        assert not result_fine, "Fine step should detect collision"
        # Coarse may or may not detect — we don't assert either way,
        # just verify the API works with custom step.

    # ── Environment collision (triggered) ──

    def test_env_collision_fast_triggered(self, cm):
        """Table close enough to home pose should trigger Tier 1."""
        cm.add_table(table_height=-0.05)
        try:
            q = np.zeros(7, dtype=np.float64)
            assert cm.check_env_collision_fast(q), "Table at z=-0.05 should trigger Tier 1 at home pose"
        finally:
            cm.clear_obstacles()

    def test_env_collision_full_triggered(self, cm):
        """Full two-tier check should also detect table collision."""
        cm.add_table(table_height=-0.05)
        try:
            q = np.zeros(7, dtype=np.float64)
            assert cm.check_env_collision(q), "Full env check should detect table collision at home pose"
        finally:
            cm.clear_obstacles()

    # ── Segment environment collision ──

    def test_segment_env_collision_free_no_obstacles(self, cm):
        """With no obstacles registered, any segment should pass env check."""
        q1 = np.zeros(7, dtype=np.float64)
        q2 = np.full(7, 0.5, dtype=np.float64)
        assert cm.check_segment_env_collision_free(q1, q2)

    def test_segment_env_collision_free_with_table(self, cm):
        """A safe segment (arm stays high) should pass even with a table present."""
        cm.add_table(table_height=-0.5)
        try:
            q1 = np.zeros(7, dtype=np.float64)
            q2 = np.full(7, 0.1, dtype=np.float64)
            # Both poses keep the arm above the table → Tier 1 passes.
            assert cm.check_segment_env_collision_free(q1, q2), "Safe segment should pass env check"
        finally:
            cm.clear_obstacles()

    # ── Teleop combined check with env trigger ──

    def test_teleop_collision_with_env_trigger(self, cm):
        """Combined check should report env=True when table is close."""
        cm.add_table(table_height=-0.05)
        try:
            q = np.zeros(7, dtype=np.float64)
            has_self, has_env = cm.check_teleop_collision(q)
            assert not has_self, "Home pose should not self-collide"
            assert has_env, "Table at z=-0.05 should trigger env collision at home pose"
        finally:
            cm.clear_obstacles()

    # ── pad_arm_for_fk ──

    def test_pad_arm_for_fk_7dof(self, cm):
        """In 7-DOF mode, pad_arm_for_fk should return arm qpos as-is."""
        qpos = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float64)
        result = cm.pad_arm_for_fk(qpos)
        assert result.shape == (7,)
        np.testing.assert_array_equal(result, qpos)

    # ── Properties ──

    def test_nq(self, cm):
        assert cm.nq == 7

    def test_hand_dof_false(self, cm):
        assert not cm.hand_dof


class TestCollisionModel19DOF:
    """Tests for 19-DOF (arm + hand) CollisionModel."""

    @pytest.fixture(scope="class")
    @classmethod
    def cm(cls):
        from dexmani_real.planning.collision_model import CollisionModel

        cm = CollisionModel(hand_dof=True)
        # Initialize hand qpos to home (open) pose
        cm.set_hand_qpos(np.zeros(12, dtype=np.float64))
        return cm

    def test_nq_19(self, cm):
        assert cm.nq == 19

    def test_hand_dof_true(self, cm):
        assert cm.hand_dof

    def test_full_qpos_self_collision_home(self, cm):
        """Full 19-DOF home pose should be self-collision-free."""
        qpos = np.zeros(19, dtype=np.float64)
        assert not cm.check_self_collision(qpos)

    def test_arm_only_auto_expand(self, cm):
        """7-DOF arm qpos should auto-expand to 19-DOF with hand_qpos buffer."""
        qpos = np.zeros(7, dtype=np.float64)
        assert not cm.check_self_collision(qpos)

    def test_teleop_collision_19dof_home(self, cm):
        """Combined check at 19-DOF home pose."""
        qpos = np.zeros(7, dtype=np.float64)
        has_self, has_env = cm.check_teleop_collision(qpos)
        assert not has_self
        assert not has_env

    def test_pad_arm_for_fk_19dof(self, cm):
        """In 19-DOF mode, pad should return (19,) array with hand zeros."""
        qpos = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float64)
        result = cm.pad_arm_for_fk(qpos)
        assert result.shape == (19,)
        np.testing.assert_array_equal(result[:7], qpos)
        np.testing.assert_array_equal(result[7:], np.zeros(12, dtype=np.float64))

    def test_set_hand_qpos_rejects_nan(self, cm):
        """set_hand_qpos should reject NaN values."""
        bad_qpos = np.full(12, np.nan, dtype=np.float64)
        with pytest.raises(ValueError):
            cm.set_hand_qpos(bad_qpos)

    def test_set_hand_qpos_rejects_wrong_shape(self, cm):
        """set_hand_qpos should reject wrong shape."""
        with pytest.raises(ValueError):
            cm.set_hand_qpos(np.zeros(7, dtype=np.float64))

    def test_env_collision_fast_with_table_19dof(self, cm):
        """19-DOF env collision fast check with table below robot."""
        cm.add_table(table_height=-0.5)
        try:
            qpos = np.zeros(7, dtype=np.float64)
            result = cm.check_env_collision_fast(qpos)
            assert not result, f"Home pose should be above table, got collision={result}"
        finally:
            cm.clear_obstacles()

    # ── Segment collision (19-DOF) ──

    def test_segment_collision_free_19dof_safe(self, cm):
        """A safe segment should pass in 19-DOF mode."""
        q1 = np.zeros(7, dtype=np.float64)
        q2 = np.full(7, 0.1, dtype=np.float64)
        assert cm.check_segment_collision_free(q1, q2), "Safe segment should pass in 19-DOF mode"

    def test_segment_collision_free_19dof_auto_expand(self, cm):
        """7-DOF qpos should auto-expand to 19-DOF for segment checks."""
        q1 = np.zeros(7, dtype=np.float64)
        q2 = np.full(7, 0.2, dtype=np.float64)
        assert cm.check_segment_collision_free(q1, q2), "Auto-expanded segment should pass"

    def test_segment_env_collision_free_19dof_no_obstacles(self, cm):
        """Env segment check with no obstacles should always pass."""
        q1 = np.zeros(7, dtype=np.float64)
        q2 = np.full(7, 0.5, dtype=np.float64)
        assert cm.check_segment_env_collision_free(q1, q2), "No obstacles → always safe"

    # ── G1: Cross-finger collision pair ──

    def test_cross_finger_pair_exists(self, cm):
        """Verify thumb_tip ↔ index_tip collision pair is active in 19-DOF model."""
        thumb_idx = None
        index_idx = None
        for i in range(cm._collision_model.ngeoms):
            name = cm._collision_model.geometryObjects[i].name
            if name == "right_hand_thumb_rota_tip_0":
                thumb_idx = i
            elif name == "right_hand_index_rota_tip_0":
                index_idx = i
        assert thumb_idx is not None, "thumb_tip geometry not found"
        assert index_idx is not None, "index_tip geometry not found"

        # Check if pair exists
        found = False
        for cp in cm._collision_model.collisionPairs:
            if {cp.first, cp.second} == {thumb_idx, index_idx}:
                found = True
                break
        assert found, f"thumb_tip({thumb_idx}) ↔ index_tip({index_idx}) pair not found in collision pairs"
