"""Unit tests for TeleopPipeline — stateless action computation.

Uses VRFrameSimulator for deterministic VR input (sinusoidal wrist trajectory).

Usage:
    python -m pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

# Minimal test environment — uses the real pipeline with mock components
from dexmani_real.teleop.core.pipeline import TeleopPipeline
from dexmani_real.planning.types import Pose, IKResult, TeleopProfile

# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


class MockArmMapper:
    """Minimal mock: returns identity-mapped EEF pose from VR wrist."""

    def __init__(self, offset: np.ndarray | None = None) -> None:
        self._offset = offset if offset is not None else np.zeros(3)
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def map(self, wrist_pos: np.ndarray, wrist_quat: np.ndarray) -> dict | None:
        return {
            "pos": wrist_pos + self._offset,
            "quat_wxyz": wrist_quat,
        }

    def reset(self, **kwargs) -> None:
        pass


class MockRetargeter:
    """Returns a constant hand pose."""

    def __init__(self, default_qpos: np.ndarray | None = None) -> None:
        self.default_qpos = default_qpos if default_qpos is not None else np.zeros(12, dtype=np.float64)
        self.reload_count = 0

    def retarget(self, landmarks: np.ndarray) -> np.ndarray | None:
        if landmarks is None or landmarks.shape != (21, 3):
            return None
        return self.default_qpos.copy()

    def load_retargeter(self) -> None:
        self.reload_count += 1


class MockPlanner:
    """Minimal mock planner that accepts any IK and returns the seed."""

    def __init__(self) -> None:
        self.ik_call_count = 0
        self.teleop_profile = TeleopProfile()

    def solve_teleop_ik(self, target_pose: Pose, current_qpos: np.ndarray, prev_cmd: np.ndarray) -> IKResult:
        self.ik_call_count += 1
        # Return the seed as the solution (identity IK)
        return IKResult(
            success=True,
            qpos=current_qpos.copy(),
            reason="mock",
        )

    def compute_eef_pose_world(self, qpos: np.ndarray) -> Pose:
        # Simple FK: return identity orientation + position from qpos[:3]
        p = qpos[:3].copy() if len(qpos) >= 3 else np.zeros(3)
        q = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion
        return Pose(p=p, q=q)

    @property
    def desk_safety(self):
        return None


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def make_vr_frame(
    wrist_pos: np.ndarray | None = None,
    wrist_quat: np.ndarray | None = None,
    landmarks: np.ndarray | None = None,
) -> dict:
    """Create a minimal VR frame dict."""
    return {
        "wrist_pos": (wrist_pos if wrist_pos is not None else np.array([0.4, 0.0, 0.3], dtype=np.float64)),
        "wrist_quat_wxyz": (wrist_quat if wrist_quat is not None else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)),
        "landmarks": (landmarks if landmarks is not None else np.random.randn(21, 3).astype(np.float64) * 0.01),
    }


# ═══════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════


class TestTeleopPipelineBasic:
    """Basic compute_action flow tests."""

    def test_full_pipeline_success(self) -> None:
        """Smoke test: full pipeline with mock components should succeed."""
        mapper = MockArmMapper()
        retargeter = MockRetargeter()
        planner = MockPlanner()
        pipeline = TeleopPipeline(mapper, retargeter, planner)

        vr = make_vr_frame()
        arm_qpos = np.zeros(7, dtype=np.float64)
        hand_qpos = np.zeros(12, dtype=np.float64)
        prev_arm = np.zeros(7, dtype=np.float64)
        prev_hand = np.zeros(12, dtype=np.float64)

        action, status = pipeline.compute_action(
            vr,
            arm_qpos,
            hand_qpos,
            prev_arm,
            prev_hand,
        )

        assert status["ik_ok"] is True
        assert status["retarget_ok"] is True
        assert action.arm_qpos_cmd.shape == (7,)
        assert action.hand_qpos_cmd.shape == (12,)
        assert planner.ik_call_count == 1

    def test_vr_frame_none_landmarks(self) -> None:
        """None landmarks should cause retarget failure but not crash."""
        mapper = MockArmMapper()
        retargeter = MockRetargeter()
        planner = MockPlanner()
        pipeline = TeleopPipeline(mapper, retargeter, planner)

        # Explicitly set landmarks=None (bypass make_vr_frame default)
        vr = make_vr_frame()
        vr["landmarks"] = None
        arm_qpos = np.zeros(7, dtype=np.float64)
        hand_qpos = np.zeros(12, dtype=np.float64)
        prev_arm = np.zeros(7, dtype=np.float64)
        prev_hand = np.zeros(12, dtype=np.float64)

        action, status = pipeline.compute_action(
            vr,
            arm_qpos,
            hand_qpos,
            prev_arm,
            prev_hand,
        )

        # IK should still succeed (mock), retarget should fail
        assert status["ik_ok"] is True
        assert status["retarget_ok"] is False
        # Hand should fall back to prev command
        np.testing.assert_array_equal(action.hand_qpos_cmd, prev_hand)

    def test_retarget_returns_none(self) -> None:
        """If retargeter returns None, retarget_ok should be False."""
        mapper = MockArmMapper()

        class FailingRetargeter(MockRetargeter):
            def retarget(self, landmarks):
                return None

        retargeter = FailingRetargeter()
        planner = MockPlanner()
        pipeline = TeleopPipeline(mapper, retargeter, planner)

        vr = make_vr_frame()
        arm_qpos = np.zeros(7, dtype=np.float64)
        hand_qpos = np.ones(12, dtype=np.float64)
        prev_arm = np.zeros(7, dtype=np.float64)
        prev_hand = np.ones(12, dtype=np.float64)

        action, status = pipeline.compute_action(
            vr,
            arm_qpos,
            hand_qpos,
            prev_arm,
            prev_hand,
        )

        assert status["retarget_ok"] is False
        # Should hold previous hand command
        np.testing.assert_array_equal(action.hand_qpos_cmd, prev_hand)


class TestJumpClamp:
    """Joint jump clamp tests."""

    def test_no_jump(self) -> None:
        """Small deltas should pass jump clamp unchanged."""
        pipeline = TeleopPipeline(
            MockArmMapper(),
            MockRetargeter(),
            MockPlanner(),
            arm_jump_limit_rad=np.deg2rad(5.0),
            hand_jump_limit_rad=np.deg2rad(10.0),
        )

        arm_cmd = np.array([0.01] * 7)
        hand_cmd = np.array([0.01] * 12)
        prev_arm = np.zeros(7)
        prev_hand = np.zeros(12)

        arm_out, hand_out, jump_ok = pipeline.apply_jump_clamp(
            arm_cmd,
            hand_cmd,
            prev_arm,
            prev_hand,
        )

        assert jump_ok is True
        np.testing.assert_array_almost_equal(arm_out, arm_cmd)
        np.testing.assert_array_almost_equal(hand_out, hand_cmd)

    def test_arm_jump_clamped(self) -> None:
        """Large arm delta should be clamped."""
        pipeline = TeleopPipeline(
            MockArmMapper(),
            MockRetargeter(),
            MockPlanner(),
            arm_jump_limit_rad=np.deg2rad(5.0),
        )

        arm_cmd = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # ~11.5°
        hand_cmd = np.zeros(12)
        prev_arm = np.zeros(7)
        prev_hand = np.zeros(12)

        arm_out, hand_out, jump_ok = pipeline.apply_jump_clamp(
            arm_cmd,
            hand_cmd,
            prev_arm,
            prev_hand,
        )

        assert jump_ok is False
        # Joint 0 should be clamped to limit
        limit = np.deg2rad(5.0)
        assert abs(arm_out[0]) <= limit + 1e-10
        # Other joints unchanged (already 0)
        np.testing.assert_array_equal(arm_out[1:], np.zeros(6))

    def test_hand_jump_clamped(self) -> None:
        """Large hand delta should be clamped."""
        pipeline = TeleopPipeline(
            MockArmMapper(),
            MockRetargeter(),
            MockPlanner(),
            hand_jump_limit_rad=np.deg2rad(10.0),
        )

        arm_cmd = np.zeros(7)
        hand_cmd = np.array([0.3] * 12)  # ~17.2°
        prev_arm = np.zeros(7)
        prev_hand = np.zeros(12)

        arm_out, hand_out, jump_ok = pipeline.apply_jump_clamp(
            arm_cmd,
            hand_cmd,
            prev_arm,
            prev_hand,
        )

        assert jump_ok is False
        # All joints should be clamped to limit
        limit = np.deg2rad(10.0)
        assert np.all(np.abs(hand_out) <= limit + 1e-10)


class TestSoftDeceleration:
    """Soft deceleration (VR loss) tests."""

    def test_hold_current_position(self) -> None:
        """Should return copies of current position."""
        arm = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        hand = np.arange(12, dtype=np.float64)

        arm_out, hand_out = TeleopPipeline.soft_deceleration(arm, hand)

        np.testing.assert_array_equal(arm_out, arm)
        np.testing.assert_array_equal(hand_out, hand)
        # Should be copies, not same objects
        assert arm_out is not arm
        assert hand_out is not hand


class TestWorkspaceCheck:
    """Workspace boundary clamp + re-IK tests."""

    def test_in_workspace_no_clamp(self) -> None:
        """When EEF is in workspace, no re-IK should be triggered."""
        mapper = MockArmMapper()
        retargeter = MockRetargeter()
        planner = MockPlanner()
        pipeline = TeleopPipeline(mapper, retargeter, planner)

        def always_in_workspace(pos: np.ndarray) -> bool:
            return True

        vr = make_vr_frame()
        arm_qpos = np.arange(7, dtype=np.float64) * 0.1
        hand_qpos = np.zeros(12, dtype=np.float64)
        prev_arm = np.arange(7, dtype=np.float64) * 0.1
        prev_hand = np.zeros(12, dtype=np.float64)

        action, status = pipeline.compute_action(
            vr,
            arm_qpos,
            hand_qpos,
            prev_arm,
            prev_hand,
            check_workspace=always_in_workspace,
        )

        assert status["ik_ok"] is True
        # Should have exactly 1 IK call (no re-IK)
        assert planner.ik_call_count == 1

    def test_out_of_workspace_triggers_re_ik(self) -> None:
        """When EEF is out of workspace, clamp + re-IK should fire."""
        mapper = MockArmMapper()
        retargeter = MockRetargeter()
        planner = MockPlanner()
        pipeline = TeleopPipeline(mapper, retargeter, planner)

        def always_out_of_workspace(pos: np.ndarray) -> bool:
            return False

        def clamp_to_origin(pos: np.ndarray) -> np.ndarray:
            return np.zeros(3, dtype=np.float64)

        vr = make_vr_frame()
        arm_qpos = np.arange(7, dtype=np.float64) * 0.1
        hand_qpos = np.zeros(12, dtype=np.float64)
        prev_arm = np.arange(7, dtype=np.float64) * 0.1
        prev_hand = np.zeros(12, dtype=np.float64)

        action, status = pipeline.compute_action(
            vr,
            arm_qpos,
            hand_qpos,
            prev_arm,
            prev_hand,
            check_workspace=always_out_of_workspace,
            clamp_workspace_pos=clamp_to_origin,
        )

        # Should have 2 IK calls: one primary, one re-IK
        assert planner.ik_call_count == 2
        assert status["ik_ok"] is True


class TestEMASmoothing:
    """EMA smoothing tests for arm commands."""

    def test_ema_with_alpha_0_5(self) -> None:
        """EMA alpha=0.5 should average equally between new and old."""
        mapper = MockArmMapper()
        retargeter = MockRetargeter()

        class CountingPlanner(MockPlanner):
            def __init__(self) -> None:
                super().__init__()
                self._call = 0

            def solve_teleop_ik(self, target_pose, current_qpos, prev_cmd):
                self.ik_call_count += 1
                # Return alternating solutions
                val = 0.1 if self._call % 2 == 0 else 0.2
                self._call += 1
                return IKResult(
                    success=True,
                    qpos=np.full(7, val, dtype=np.float64),
                    reason="mock",
                )

        planner = CountingPlanner()
        pipeline = TeleopPipeline(mapper, retargeter, planner, ema_alpha_arm=0.5)

        vr = make_vr_frame()
        arm_qpos = np.zeros(7, dtype=np.float64)
        prev_arm = np.zeros(7, dtype=np.float64)
        last_arm = np.full(7, 0.15, dtype=np.float64)  # previous EMA state

        arm_cmd, ik_ok, _ = pipeline.compute_arm_command(
            vr,
            arm_qpos,
            prev_arm,
            last_arm_cmd=last_arm,
        )

        assert ik_ok is True
        # EMA = 0.5 * 0.1 + (1-0.5) * 0.15 = 0.05 + 0.075 = 0.125
        expected = np.full(7, 0.125, dtype=np.float64)
        np.testing.assert_array_almost_equal(arm_cmd, expected)
