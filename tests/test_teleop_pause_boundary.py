"""Offline causal-feedback regressions for teleoperation pause boundaries."""

from __future__ import annotations

import unittest
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.control.action import ActionCandidate
from dexmani_real.control.publication import PreparedCommand, PublishResult
from dexmani_real.ipc.causal import vr_frame_is_fresh
from dexmani_real.ipc.schema import HAND_TACTILE_DTYPE
from dexmani_real.teleop.control_loop.action_proposal import compute_target_eef_pose
from dexmani_real.teleop.config import TeleopConfig
from dexmani_real.teleop.control_loop.grid import (
    TeleopActionComputation,
    TeleopGridObservation,
    TeleopGridResources,
    _publish_ik_failure_hold,
    _publish_solved_action,
    _read_control_grid_observation,
    feedback_is_newer_than_pause,
    run_control_grid_tick,
)
from dexmani_real.teleop.episode_samples import FRAME_IK_FAIL, record_held
from dexmani_real.teleop.control_loop.hand_control import smoothstep_hand_ramp
from dexmani_real.teleop.loop import _begin_feedback_issue


class TeleopPauseBoundaryTest(unittest.TestCase):
    @staticmethod
    def _resources() -> TeleopGridResources:
        return TeleopGridResources(
            planner=Mock(),
            safety_gate=Mock(),
            recorder=None,
            command_limits=Mock(),
            camera_freshness=Mock(),
            stage_timer=Mock(),
            validation_warn=Mock(),
            arm_feedback_warn=Mock(),
            hand_fk=None,
            handbase_position_eef_m=np.zeros(3, dtype=np.float64),
            handbase_quat_eef_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            hand_ramp_total_frames=1,
            max_observation_skew_s=0.1,
        )

    def test_resume_requires_all_enabled_feedback_after_pause(self) -> None:
        self.assertFalse(
            feedback_is_newer_than_pause(
                100,
                arm_source_monotonic_ns=101,
                vr_receive_monotonic_ns=100,
                hand_source_monotonic_ns=101,
            )
        )
        self.assertFalse(
            feedback_is_newer_than_pause(
                100,
                arm_source_monotonic_ns=101,
                vr_receive_monotonic_ns=101,
                hand_source_monotonic_ns=100,
            )
        )
        self.assertTrue(
            feedback_is_newer_than_pause(
                100,
                arm_source_monotonic_ns=101,
                vr_receive_monotonic_ns=101,
                hand_source_monotonic_ns=101,
            )
        )

    def test_vr_freshness_uses_sdk_receive_time_not_processing_time(self) -> None:
        self.assertFalse(
            vr_frame_is_fresh(
                {"recv_ts_ns": 100},
                now_monotonic_ns=600_000_101,
                max_age_s=0.5,
            )
        )

    def test_begin_recording_requires_fresh_calibrated_tactile(self) -> None:
        cfg = TeleopConfig()
        now_ns = 1_000_000_000
        vr_frame = {"recv_ts_ns": now_ns - 1_000_000}
        tactile = np.zeros(1, dtype=HAND_TACTILE_DTYPE)
        tactile["source_monotonic_ns"][0] = now_ns - 1_000_000
        tactile["fresh"][0] = 1
        with patch("dexmani_real.teleop.loop.hand_feedback_issue", return_value=None):
            self.assertEqual(
                _begin_feedback_issue(
                    cfg,
                    vr_frame,
                    Mock(),
                    tactile,
                    recording_enabled=True,
                    now_monotonic_ns=now_ns,
                ),
                "tactile feedback is not calibrated",
            )
            tactile["calibrated"][0] = 1
            self.assertIsNone(
                _begin_feedback_issue(
                    cfg,
                    vr_frame,
                    Mock(),
                    tactile,
                    recording_enabled=True,
                    now_monotonic_ns=now_ns,
                )
            )

    def test_held_sample_uses_tick_generation_snapshot(self) -> None:
        recorder = Mock()
        with patch(
            "dexmani_real.teleop.episode_samples._build_robot_state",
            return_value=Mock(),
        ):
            record_held(
                recorder,
                None,
                np.zeros(7),
                np.zeros(12),
                {
                    "wrist_pos": np.zeros(3),
                    "wrist_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
                    "landmarks": np.zeros((21, 3)),
                },
                None,
                control_run_generation=7,
                max_observation_skew_s=0.1,
            )
        self.assertEqual(
            recorder.add_frame.call_args.kwargs["control_run_generation"], 7
        )

    def test_hand_ramp_still_reaches_target_on_last_step(self) -> None:
        start = np.zeros(12)
        target = np.ones(12)
        first = smoothstep_hand_ramp(start, target, 0, 4)
        last = smoothstep_hand_ramp(start, target, 3, 4)
        self.assertTrue(np.all(first > start))
        self.assertTrue(np.all(first < target))
        np.testing.assert_array_equal(last, target)

    def test_workspace_clamps_after_ema_without_changing_raw_target(self) -> None:
        proposal = compute_target_eef_pose(
            np.array([2.0, -2.0, 0.5]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            previous_position_world_m=None,
            previous_quat_world_wxyz=None,
            workspace_bounds_world_m=np.array([[0.0, 1.0], [-1.0, 1.0], [0.0, 1.0]]),
            ema_alpha_position=0.5,
            ema_alpha_rotation=0.5,
        )
        np.testing.assert_array_equal(proposal.position_world_m, [1.0, -1.0, 0.5])
        np.testing.assert_array_equal(proposal.raw_position_world_m, [2.0, -2.0, 0.5])

    def test_grid_reanchors_once_then_allows_observation(self) -> None:
        controller = SimpleNamespace(
            hand_enabled=False,
            prev_qpos_cmd=np.zeros(7, dtype=np.float64),
            ema_prev_pos=None,
            ema_prev_quat=None,
            reset_reference=Mock(return_value=True),
        )
        shared = SimpleNamespace(arm_state_ring=object(), hand_state_ring=object())
        resources = self._resources()
        arm_state = {
            "qpos": np.zeros((1, 7), dtype=np.float64),
            "source_monotonic_ns": np.array([101], dtype=np.int64),
        }
        vr_frame = {"recv_ts_ns": 101, "ring_sequence": 1}

        def read_structured(ring, **_kwargs):
            return (arm_state, 0, 1) if ring is shared.arm_state_ring else None

        with (
            patch(
                "dexmani_real.teleop.control_loop.grid.read_causal_structured_frame",
                side_effect=read_structured,
            ),
            patch(
                "dexmani_real.teleop.control_loop.grid.read_vr_frame_causal",
                return_value=vr_frame,
            ),
            patch(
                "dexmani_real.teleop.control_loop.grid.read_hand_tactile_causal",
                return_value=None,
            ),
            patch(
                "dexmani_real.teleop.control_loop.grid.arm_feedback_issue",
                return_value=None,
            ),
            patch(
                "dexmani_real.teleop.control_loop.grid.hand_feedback_issue",
                return_value=None,
            ),
            patch(
                "dexmani_real.teleop.control_loop.grid.advance_arm_feedback_error_count",
                return_value=(0, False),
            ),
            patch(
                "dexmani_real.teleop.control_loop.grid.time.monotonic_ns", return_value=200
            ),
        ):
            first_result, first_observation = _read_control_grid_observation(
                controller,
                shared,
                TeleopConfig(),
                resources,
                teleop_active=True,
                recording_active=False,
                pause_since_ns=100,
                pause_reason="begin",
                arm_feedback_error_count=0,
                hand_disconnected_at_s=None,
                loop_count=1,
                observation_anchor_monotonic_ns=200,
                control_run_generation=3,
            )
            second_result, second_observation = _read_control_grid_observation(
                controller,
                shared,
                TeleopConfig(),
                resources,
                teleop_active=True,
                recording_active=False,
                pause_since_ns=0,
                pause_reason=None,
                arm_feedback_error_count=0,
                hand_disconnected_at_s=None,
                loop_count=2,
                observation_anchor_monotonic_ns=200,
                control_run_generation=3,
            )

        self.assertTrue(first_result.pause_released)
        self.assertIsNone(first_observation)
        controller.reset_reference.assert_called_once()
        self.assertFalse(second_result.pause_released)
        self.assertIsNotNone(second_observation)

    def test_grid_resources_do_not_own_session_state(self) -> None:
        resource_fields = {field.name for field in fields(TeleopGridResources)}
        self.assertNotIn("quiescence", resource_fields)
        self.assertNotIn("audio", resource_fields)

    def test_vr_stale_requests_pause_without_computing_or_publishing(self) -> None:
        controller = SimpleNamespace(
            hand_enabled=False,
            prev_qpos_cmd=np.zeros(7, dtype=np.float64),
            ema_prev_pos=None,
            ema_prev_quat=None,
            compute=Mock(),
        )
        shared = SimpleNamespace(
            arm_state_ring=object(),
            hand_state_ring=object(),
            run_generation=SimpleNamespace(value=3),
        )
        arm_state = {
            "qpos": np.zeros((1, 7), dtype=np.float64),
            "source_monotonic_ns": np.array([101], dtype=np.int64),
        }
        with (
            patch(
                "dexmani_real.teleop.control_loop.grid.read_causal_structured_frame",
                side_effect=lambda ring, **_kwargs: (
                    (arm_state, 0, 1) if ring is shared.arm_state_ring else None
                ),
            ),
            patch(
                "dexmani_real.teleop.control_loop.grid.read_vr_frame_causal",
                return_value=None,
            ),
            patch(
                "dexmani_real.teleop.control_loop.grid.read_hand_tactile_causal",
                return_value=None,
            ),
            patch(
                "dexmani_real.teleop.control_loop.grid.arm_feedback_issue",
                return_value=None,
            ),
            patch(
                "dexmani_real.teleop.control_loop.grid.hand_feedback_issue",
                return_value=None,
            ),
            patch(
                "dexmani_real.teleop.control_loop.grid._prepare_and_publish_joint_command"
            ) as publish,
        ):
            result = run_control_grid_tick(
                controller,
                shared,
                TeleopConfig(),
                self._resources(),
                teleop_active=True,
                recording_active=False,
                pause_since_ns=0,
                pause_reason=None,
                arm_feedback_error_count=0,
                hand_disconnected_at_s=None,
                loop_count=1,
                observation_anchor_monotonic_ns=200,
            )

        self.assertEqual(result.pause_reason, "vr_stale")
        controller.compute.assert_not_called()
        publish.assert_not_called()

    def test_arm_stale_requests_pause_without_publishing(self) -> None:
        controller = SimpleNamespace(
            hand_enabled=False,
            prev_qpos_cmd=np.zeros(7, dtype=np.float64),
            compute=Mock(),
        )
        shared = SimpleNamespace(
            arm_state_ring=object(),
            hand_state_ring=object(),
            run_generation=SimpleNamespace(value=3),
        )
        with (
            patch(
                "dexmani_real.teleop.control_loop.grid.read_causal_structured_frame",
                return_value=None,
            ),
            patch(
                "dexmani_real.teleop.control_loop.grid.arm_feedback_issue",
                return_value="arm feedback stale",
            ),
            patch(
                "dexmani_real.teleop.control_loop.grid._prepare_and_publish_joint_command"
            ) as publish,
        ):
            result = run_control_grid_tick(
                controller,
                shared,
                TeleopConfig(),
                self._resources(),
                teleop_active=True,
                recording_active=False,
                pause_since_ns=0,
                pause_reason=None,
                arm_feedback_error_count=0,
                hand_disconnected_at_s=None,
                loop_count=1,
                observation_anchor_monotonic_ns=200,
            )

        self.assertEqual(result.pause_reason, "arm_feedback")
        controller.compute.assert_not_called()
        publish.assert_not_called()

    def test_hand_stale_requests_pause_without_publishing(self) -> None:
        controller = SimpleNamespace(
            hand_enabled=True,
            prev_qpos_cmd=np.zeros(7, dtype=np.float64),
            ema_prev_pos=None,
            ema_prev_quat=None,
            compute=Mock(),
        )
        shared = SimpleNamespace(
            arm_state_ring=object(),
            hand_state_ring=object(),
            run_generation=SimpleNamespace(value=3),
        )
        arm_state = {
            "qpos": np.zeros((1, 7), dtype=np.float64),
            "source_monotonic_ns": np.array([101], dtype=np.int64),
        }
        hand_state = {
            "qpos": np.zeros((1, 12), dtype=np.float64),
            "source_monotonic_ns": np.array([101], dtype=np.int64),
        }

        def read_structured(ring, **_kwargs):
            state = arm_state if ring is shared.arm_state_ring else hand_state
            return state, 0, 1

        with (
            patch(
                "dexmani_real.teleop.control_loop.grid.read_causal_structured_frame",
                side_effect=read_structured,
            ),
            patch(
                "dexmani_real.teleop.control_loop.grid.read_vr_frame_causal",
                return_value={"recv_ts_ns": 101, "ring_sequence": 1},
            ),
            patch(
                "dexmani_real.teleop.control_loop.grid.read_hand_tactile_causal",
                return_value=None,
            ),
            patch(
                "dexmani_real.teleop.control_loop.grid.arm_feedback_issue",
                return_value=None,
            ),
            patch(
                "dexmani_real.teleop.control_loop.grid.hand_feedback_issue",
                return_value="hand feedback stale",
            ),
            patch(
                "dexmani_real.teleop.control_loop.grid.time.monotonic_ns", return_value=150
            ),
            patch("dexmani_real.teleop.control_loop.grid.time.monotonic", return_value=1.0),
            patch(
                "dexmani_real.teleop.control_loop.grid._prepare_and_publish_joint_command"
            ) as publish,
        ):
            result = run_control_grid_tick(
                controller,
                shared,
                TeleopConfig(),
                self._resources(),
                teleop_active=True,
                recording_active=False,
                pause_since_ns=0,
                pause_reason=None,
                arm_feedback_error_count=0,
                hand_disconnected_at_s=None,
                loop_count=1,
                observation_anchor_monotonic_ns=200,
            )

        self.assertEqual(result.pause_reason, "hand_feedback")
        controller.compute.assert_not_called()
        publish.assert_not_called()

    def test_recording_receives_exact_post_shaping_published_candidate(self) -> None:
        arm_published = np.linspace(0.1, 0.7, 7)
        hand_published = np.linspace(0.01, 0.12, 12)
        candidate = ActionCandidate(
            observation_id=1,
            run_generation=2,
            created_monotonic_ns=10,
            scheduled_target_monotonic_ns=9,
            target_monotonic_ns=11,
            valid_until_monotonic_ns=12,
            action_id=3,
            arm_qpos=arm_published,
            hand_qpos=hand_published,
        )
        controller = SimpleNamespace(
            hand_enabled=True,
            prev_qpos_cmd=np.zeros(7),
            prev_hand_qpos=np.zeros(12),
            ema_prev_pos=None,
            ema_prev_quat=None,
            consecutive_ik_hold_frames=0,
            ik_hold_started_s=0.0,
            last_target_eef_pos=np.full(3, np.nan),
            last_target_eef_rot6d=np.full(6, np.nan),
        )
        planner = Mock()
        resources = self._resources()
        object.__setattr__(resources, "planner", planner)
        limits = SimpleNamespace(
            arm_joint_lower_rad=np.full(7, -10.0),
            arm_joint_upper_rad=np.full(7, 10.0),
            teleop_arm_max_delta_rad_per_tick=np.ones(7),
            hand_mechanical_lower_rad=np.full(12, -10.0),
            hand_mechanical_upper_rad=np.full(12, 10.0),
        )
        object.__setattr__(resources, "command_limits", limits)
        observation = TeleopGridObservation(
            arm_state=np.zeros(1, dtype=[("tracking_err", "f8")]),
            arm_ring_sequence=4,
            arm_qpos_rad=np.zeros(7),
            vr_frame={"ring_sequence": 1, "recv_ts_ns": 8},
            camera_frame=None,
            hand_state=None,
            hand_ring_sequence=5,
            hand_tactile=None,
            anchor_monotonic_ns=8,
            control_run_generation=2,
            policy_observation_signals=None,
        )
        computation = TeleopActionComputation(
            target_position_world_m=np.array([0.3, 0.0, 0.4]),
            target_quat_world_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            raw_target_position_world_m=np.array([0.3, 0.0, 0.4]),
            raw_target_quat_world_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            position_before_workspace_clamp_world_m=np.array([0.3, 0.0, 0.4]),
            hand_qpos_rad=np.zeros(12),
            raw_hand_qpos_rad=np.zeros(12),
            hand_retarget_succeeded=True,
            hand_retarget_time_ms=1.0,
            ik_qpos_rad=np.zeros(7),
            ik_failure_reason="",
            ik_solve_time_ms=2.0,
            policy_map_time_ms=3.0,
            policy_compute_started_s=0.0,
        )
        arm_proposal = SimpleNamespace(
            qpos_rad=np.zeros(7),
            raw_qpos_rad=np.full(7, 0.5),
            validation_issue=None,
        )
        with (
            patch(
                "dexmani_real.teleop.control_loop.grid.compute_arm_joint_proposal",
                return_value=arm_proposal,
            ),
            patch(
                "dexmani_real.teleop.control_loop.grid._prepare_and_publish_joint_command",
                return_value=(
                    PreparedCommand(candidate=candidate),
                    PublishResult(published=True),
                ),
            ),
            patch("dexmani_real.teleop.control_loop.grid.record_frame") as record_frame,
        ):
            keep_running = _publish_solved_action(
                controller,
                SimpleNamespace(),
                TeleopConfig(),
                resources,
                observation,
                computation,
                recording_active=True,
            )

        self.assertTrue(keep_running)
        args, kwargs = record_frame.call_args
        np.testing.assert_array_equal(args[3], arm_published)
        np.testing.assert_array_equal(args[4], hand_published)
        self.assertIs(kwargs["action_candidate"], candidate)
        self.assertEqual(kwargs["control_run_generation"], 2)
        np.testing.assert_array_equal(
            kwargs["action_arm_joint_raw"], arm_proposal.raw_qpos_rad
        )

    def test_ik_failure_keeps_bounded_hold_and_safe_hand_motion(self) -> None:
        candidate = ActionCandidate(
            observation_id=1,
            run_generation=2,
            created_monotonic_ns=10,
            scheduled_target_monotonic_ns=9,
            target_monotonic_ns=11,
            valid_until_monotonic_ns=12,
            action_id=3,
            arm_qpos=np.zeros(7),
            hand_qpos=np.ones(12) * 0.1,
            is_hold=True,
        )
        controller = SimpleNamespace(
            hand_enabled=True,
            prev_qpos_cmd=np.zeros(7),
            prev_hand_qpos=np.zeros(12),
            consecutive_ik_hold_frames=0,
            ik_hold_started_s=0.0,
        )
        computation = SimpleNamespace(
            ik_failure_reason="infeasible",
            hand_qpos_rad=np.ones(12) * 0.1,
            ik_solve_time_ms=2.0,
            position_before_workspace_clamp_world_m=np.zeros(3),
            raw_target_position_world_m=np.zeros(3),
            raw_target_quat_world_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            raw_hand_qpos_rad=np.ones(12) * 0.1,
            policy_map_time_ms=1.0,
            hand_retarget_time_ms=1.0,
            policy_compute_started_s=0.0,
            hand_retarget_succeeded=True,
        )
        observation = SimpleNamespace(
            arm_state=np.zeros(1, dtype=[("tracking_err", "f8")]),
            vr_frame={"ring_sequence": 1, "recv_ts_ns": 8},
        )
        resources = self._resources()
        resources.command_limits.hand_mechanical_lower_rad = np.full(12, -1.0)
        resources.command_limits.hand_mechanical_upper_rad = np.full(12, 1.0)
        with (
            patch(
                "dexmani_real.teleop.control_loop.grid._prepare_and_publish_joint_command",
                return_value=(
                    PreparedCommand(candidate=candidate),
                    PublishResult(published=True),
                ),
            ) as publish,
            patch("dexmani_real.teleop.control_loop.grid._record_grid_hold") as record_hold,
        ):
            keep_running = _publish_ik_failure_hold(
                controller,
                SimpleNamespace(error_state=SimpleNamespace(value=False)),
                TeleopConfig(),
                resources,
                observation,
                computation,
                recording_active=True,
            )

        self.assertTrue(keep_running)
        self.assertTrue(publish.call_args.kwargs["is_hold"])
        np.testing.assert_array_equal(
            publish.call_args.args[2], computation.hand_qpos_rad
        )
        self.assertIs(record_hold.call_args.kwargs["action_candidate"], candidate)
        self.assertEqual(record_hold.call_args.kwargs["frame_status"], FRAME_IK_FAIL)


if __name__ == "__main__":
    unittest.main()
