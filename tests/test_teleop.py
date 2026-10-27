import threading
import time
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import nlopt
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from dexmani_real.config import load_experiment_config
from dexmani_real.control.action_mapping import ActionResult
from dexmani_real.data.recorder import RecordingError
from dexmani_real.geometry.kinematics import SDK_TO_URDF_IDX
from dexmani_real.geometry.pose import Pose, quat_wxyz_to_rotmat
from dexmani_real.robot.commands import OperationCancelled, RobotError
from dexmani_real.runtime.keyboard import Keyboard
from dexmani_real.runtime.observation import ObservationNeeds
from dexmani_real.runtime.state import RunEndReason, SessionState
from dexmani_real.teleop import audio, loop, session
from dexmani_real.teleop.config import DexPilotConfig, TAGConfig, TeleopConfig
from dexmani_real.teleop.retargeting.dexpilot import DexPilotRetargeter
from dexmani_real.teleop.retargeting.landmarks import estimate_palm_frame, prepare_landmarks
from dexmani_real.teleop.retargeting.tag import TAGRetargeter
from dexmani_real.teleop.vr import VRController, ema_smooth_pose


def hand_landmarks():
    points = np.zeros((21, 3))
    for finger, (x, y) in enumerate(((-.04, .025), (-.03, .07), (0., .08), (.025, .075), (.05, .06))):
        for joint in range(4):
            points[1 + 4 * finger + joint] = (x, y + .025 * joint, 0.)
    return points


def observation(stamp=None):
    stamp = time.monotonic_ns() if stamp is None else stamp
    return SimpleNamespace(
        arm={"qpos": np.zeros((1, 7)), "timestamp_ns": np.array([stamp])},
        hand={
            "qpos": np.zeros((1, 12)), "timestamp_ns": np.array([stamp]),
            "tactile_aggregate_timestamp_ns": np.array([stamp]),
            "tactile_dense_timestamp_ns": np.array([stamp]),
        },
        vr={
            "wrist_pos": np.array([.2, -.1, .4]),
            "wrist_quat_wxyz": np.array([1., 0., 0., 0.]),
            "landmarks": hand_landmarks(), "ring_sequence": 1, "recv_ts_ns": stamp,
        },
        camera={"timestamp_ns": stamp}, point_cloud=None,
        observation_timestamp_ns=stamp,
    )


def vr_controller(hand=None):
    mapper = Mock()
    mapper.constraints.target_fk.forward.return_value = (
        np.array([.4, .1, .3]), np.array([1., 0., 0., 0., 1., 0.]),
    )
    mapper.map.side_effect = lambda intent, current, previous: ActionResult(
        np.arange(7, dtype=float), intent.hand, eef_pose=intent.arm,
    )
    rotation = Rotation.from_euler("xyz", [.3, -.2, .4]).as_matrix()
    return VRController(mapper, hand, TeleopConfig(), rotation)


class VRTests(unittest.TestCase):
    def test_anchor_mapping_matches_relative_transform(self):
        controller = vr_controller()
        row = observation()
        row.vr["wrist_quat_wxyz"] = Rotation.from_euler("xyz", [.5, .2, -.6]).as_quat()[[3, 0, 1, 2]]
        controller.reset(row)
        c = controller.rotation
        initial_rotation = quat_wxyz_to_rotmat(row.vr["wrist_quat_wxyz"])
        rng = np.random.default_rng(7)
        for _ in range(50):
            position = rng.normal(size=3)
            rotation = Rotation.random(random_state=rng)
            mapped = controller.map_wrist(position, rotation.as_quat()[[3, 0, 1, 2]])
            np.testing.assert_allclose(
                mapped.p, [.4, .1, .3] + .4 * (c @ (position - row.vr["wrist_pos"])), atol=1e-12,
            )
            np.testing.assert_allclose(
                quat_wxyz_to_rotmat(mapped.q), c @ rotation.as_matrix() @ initial_rotation.T @ c.T,
                atol=1e-12,
            )

    def test_ema_matches_slerp_including_sign_and_pi(self):
        previous = Pose(np.ones(3), Rotation.from_euler("xyz", [.1, .2, .3]).as_quat()[[3, 0, 1, 2]])
        for quaternion in (previous.q, -previous.q, np.array([0., 1., 0., 0.])):
            target = Pose(np.zeros(3), quaternion)
            rotations = Rotation.from_quat(np.stack((previous.q, target.q))[:, [1, 2, 3, 0]])
            for alpha in (0., .3, 1.):
                actual = ema_smooth_pose(target, previous, alpha, alpha)
                np.testing.assert_allclose(actual.p, 1 - alpha)
                np.testing.assert_allclose(
                    quat_wxyz_to_rotmat(actual.q), Slerp([0, 1], rotations)(alpha).as_matrix(),
                    atol=1e-12,
                )

    def test_reset_failure_clears_old_anchor(self):
        hand = Mock()
        controller = vr_controller(hand)
        row = observation()
        controller.reset(row)

        def reset(initial_qpos=None):
            if initial_qpos is not None:
                raise ValueError("bad seed")

        hand.reset.side_effect = reset
        with self.assertRaisesRegex(ValueError, "bad seed"):
            controller.reset(row)
        self.assertIsNone(controller.previous_arm_command)
        with self.assertRaisesRegex(RuntimeError, "anchor"):
            controller.map_wrist(row.vr["wrist_pos"], row.vr["wrist_quat_wxyz"])

    def test_hand_frame_reuse_and_reference_commit(self):
        hand = Mock()
        hand.retarget.return_value = np.arange(12, dtype=float)
        controller = vr_controller(hand)
        row = observation()
        controller.reset(row)
        hand.reset.assert_called_once()
        np.testing.assert_array_equal(hand.reset.call_args.kwargs["initial_qpos"], row.hand["qpos"][0])
        original = controller.smoothed_eef_pose.copy()
        row.vr["wrist_pos"] += .1
        first = controller.compute_target(row)
        second = controller.compute_target(row)
        hand.retarget.assert_called_once()
        np.testing.assert_array_equal(controller.previous_arm_command, np.zeros(7))
        np.testing.assert_array_equal(controller.smoothed_eef_pose.p, original.p)
        np.testing.assert_array_equal(first.eef_pose.p, second.eef_pose.p)
        controller.commit(first)
        first.arm_qpos[:] = -1
        first.eef_pose.p[:] = -1
        np.testing.assert_array_equal(controller.previous_arm_command, np.arange(7))
        self.assertTrue(np.all(controller.smoothed_eef_pose.p > 0))
        row.vr["ring_sequence"] += 1
        controller.compute_target(row)
        self.assertEqual(hand.retarget.call_count, 2)

    def test_rejected_hand_frame_is_not_retried(self):
        hand = Mock()
        hand.retarget.return_value = None
        controller = vr_controller(hand)
        row = observation()
        controller.reset(row)
        self.assertIsNone(controller.compute_target(row))
        self.assertIsNone(controller.compute_target(row))
        hand.retarget.assert_called_once()
        controller.action_mapper.map.assert_not_called()


class RetargetingTests(unittest.TestCase):
    def test_palm_and_pinky_geometry(self):
        points = hand_landmarks()
        original = points.copy()
        palm = estimate_palm_frame(points)
        np.testing.assert_allclose(palm.T @ palm, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(palm), 1.)
        base = prepare_landmarks(points, pinky_scale=1, pinky_palm_scale=1)
        scaled = prepare_landmarks(points, pinky_scale=1.3, pinky_palm_scale=1.25)
        np.testing.assert_allclose(scaled[:17], base[:17])
        np.testing.assert_allclose(scaled[17] - scaled[0], 1.25 * (base[17] - base[0]))
        np.testing.assert_allclose(np.diff(scaled[17:], axis=0), 1.3 * np.diff(base[17:], axis=0))
        np.testing.assert_array_equal(points, original)

    def test_degenerate_geometry_has_no_target(self):
        for points in (np.zeros((21, 3)), np.column_stack((np.arange(21) * .01, np.zeros((21, 2))))):
            with self.assertLogs("dexmani_real.teleop.retargeting.landmarks", level="WARNING"):
                self.assertIsNone(prepare_landmarks(points, pinky_scale=1, pinky_palm_scale=1))

    def test_tag_gradients_match_finite_differences(self):
        tag = TAGRetargeter(TAGConfig())
        qpos = (tag.lower + tag.upper) / 2
        tag.target_fingertips = np.arange(15).reshape(5, 3) * .01
        tag.previous_solution = qpos + .02
        tag.position_solution = qpos - .03
        tag.pinch_factors = np.array([.7, 0., .3, 1.])
        for objective in (tag.position_objective, tag.pinch_objective):
            gradient = np.zeros(12)
            objective(qpos, gradient)
            numerical = np.empty(12)
            for index in range(12):
                offset = np.eye(12)[index] * 1e-6
                numerical[index] = (
                    objective(qpos + offset, np.empty(0)) - objective(qpos - offset, np.empty(0))
                ) / 2e-6
            np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-7)

    def test_tag_joint_order_and_owned_output(self):
        tag = TAGRetargeter(TAGConfig())
        model_qpos = tag.lower + .4 * (tag.upper - tag.lower)
        sdk_qpos = model_qpos[np.argsort(SDK_TO_URDF_IDX)]
        tag.reset(sdk_qpos)
        np.testing.assert_array_equal(tag.previous_solution, model_qpos)
        with patch.object(tag, "solve", return_value=model_qpos):
            actual = tag.retarget(hand_landmarks())
        np.testing.assert_array_equal(actual, sdk_qpos)
        self.assertFalse(np.shares_memory(actual, model_qpos))

    def test_tag_solver_failures_do_not_fall_back(self):
        for stage in ("position_optimizer", "pinch_optimizer"):
            tag = TAGRetargeter(replace(TAGConfig(), pinch_skip_threshold=0.))
            initial = tag.previous_solution.copy()
            tag.position_optimizer = Mock()
            tag.position_optimizer.optimize.return_value = initial.copy()
            tag.pinch_optimizer = Mock()
            failure = nlopt.RoundoffLimited("roundoff")
            getattr(tag, stage).optimize.side_effect = failure
            with self.assertRaises(nlopt.RoundoffLimited) as caught:
                tag.solve(np.zeros((5, 3)))
            self.assertIs(caught.exception, failure)
            np.testing.assert_array_equal(tag.previous_solution, initial)

    def test_dexpilot_matches_upstream_sequence_and_filter(self):
        from dex_retargeting.optimizer_utils import LPFilter
        from dex_retargeting.seq_retarget import SeqRetargeting

        config = DexPilotConfig()
        actual = DexPilotRetargeter(config)
        reference = SeqRetargeting(
            DexPilotRetargeter(config).optimizer, lp_filter=LPFilter(config.low_pass_alpha),
        )
        pairs = reference.optimizer.target_link_human_indices
        previous_output = None
        for index in range(10):
            points = hand_landmarks()
            points[4::4, 2] = -.002 * index
            mano = prepare_landmarks(
                points, pinky_scale=config.pinky_scale, pinky_palm_scale=config.pinky_palm_scale,
            )
            expected = reference.retarget(mano[pairs[1]] - mano[pairs[0]])
            expected = expected[reference.optimizer.idx_pin2target]
            output = actual.retarget(points)
            np.testing.assert_allclose(output, expected, atol=1e-7, rtol=0)
            if previous_output is not None:
                self.assertFalse(np.shares_memory(output, previous_output))
            previous_output = output

    def test_dexpilot_error_propagates_and_reset_clears_history(self):
        retargeter = DexPilotRetargeter(DexPilotConfig())
        retargeter.retarget(hand_landmarks())
        retargeter.optimizer.projected[:] = True
        retargeter.reset(np.zeros(12))
        self.assertFalse(retargeter.optimizer.projected.any())
        self.assertFalse(retargeter.filter.is_init)
        failure = nlopt.RoundoffLimited("dexpilot failed")
        with patch.object(retargeter.optimizer.opt, "optimize", side_effect=failure):
            with self.assertRaises(nlopt.RoundoffLimited) as caught:
                retargeter.retarget(hand_landmarks())
        self.assertIs(caught.exception, failure)
        np.testing.assert_array_equal(retargeter.previous_solution, np.zeros(12))

    def test_invalid_seeds_and_solver_outputs_raise(self):
        for retargeter in (TAGRetargeter(TAGConfig()), DexPilotRetargeter(DexPilotConfig())):
            for seed in (np.zeros(11), np.zeros(13), np.full(12, np.nan)):
                with self.assertRaises(ValueError):
                    retargeter.reset(seed)
            optimizer = (
                retargeter.position_optimizer if isinstance(retargeter, TAGRetargeter)
                else retargeter.optimizer.opt
            )
            with patch.object(optimizer, "optimize", return_value=np.full(12, np.nan)):
                with self.assertRaisesRegex(ValueError, "solver"):
                    retargeter.retarget(hand_landmarks())


def active_loop(recording=True):
    state = SessionState()
    state.arm()
    state.stop_signal.clear()
    state.begin_motion()
    robot = Mock(stop_required=True)
    robot.stop.side_effect = lambda: setattr(robot, "stop_required", False)
    runtime = load_experiment_config()
    instance = loop.TeleopLoop(
        state, runtime, robot, task_label="test", control_needs=ObservationNeeds(vr=True),
        record_needs=ObservationNeeds(camera="rgbd", required_tactile=("aggregate", "dense")),
        sensors=Mock(), controller=Mock(), home_planner=Mock(), vr_process=Mock(),
        recording_dir=Path("unused") if recording else None, keyboard=Mock(), audio=Mock(),
    )
    instance.phase = loop.TeleopPhase.ACTIVE
    instance.run_id = state.run_id
    instance.capture_started_s = time.monotonic()
    instance.controller.compute_target.return_value = ActionResult(
        np.zeros(7), np.zeros(12), eef_pose=Pose.identity(),
    )
    if recording:
        instance.recorder = Mock(start_attempted=True, is_recording=True, published_path=None)
        instance.recorder.finish.return_value = Path("unused/episode")
    return instance


class LoopTests(unittest.TestCase):
    def test_preparation_uses_the_same_termination_reasons_as_active_control(self):
        for reason, end_reason in (
            ("stop", RunEndReason.OPERATOR), ("quit", RunEndReason.QUIT),
            ("max_record_duration", RunEndReason.TIMEOUT),
        ):
            with self.subTest(reason=reason):
                instance = active_loop()
                recorder = instance.recorder
                instance.runtime_state.revoke_motion()
                instance.run_id = instance.runtime_state.run_id
                instance.phase = loop.TeleopPhase.PREPARING
                instance.end_episode(True, reason, run_end_reason=end_reason)
                self.assertEqual(recorder.finish.call_args.kwargs["reason"], end_reason.name.lower())
                self.assertEqual(recorder.finish.call_args.kwargs["technical_status"], "normal")

    def test_discard_during_computation_revokes_before_submission_and_finalization(self):
        for outcome in ("target", "rejected", "solver_error"):
            with self.subTest(outcome=outcome):
                instance = active_loop()
                recorder = instance.recorder
                keys = Keyboard(discard_callback=instance.request_discard)
                keys.active, keys.listener = True, Mock()
                self.addCleanup(keys.stop)
                instance.keyboard = keys
                result = instance.controller.compute_target.return_value

                def compute(row):
                    keys.handle_key("d", True)
                    self.assertFalse(instance.runtime_state.can_move(run_id=instance.run_id))
                    if outcome == "solver_error":
                        raise RuntimeError("solver failed")
                    return result if outcome == "target" else None

                instance.controller.compute_target.side_effect = compute
                with self.assertLogs(loop.logger, level="WARNING"):
                    if outcome == "solver_error":
                        with self.assertRaisesRegex(RuntimeError, "solver failed"):
                            instance.execute_control_step(observation(), 10**30, 10**30)
                    else:
                        instance.execute_control_step(observation(), 10**30, 10**30)
                instance.robot.send_action.assert_not_called()
                instance.controller.commit.assert_not_called()
                self.assertFalse(recorder.finish.call_args.kwargs["save"])
                self.assertEqual(recorder.finish.call_args.kwargs["reason"], "operator")
                instance.handle_commands(keys.poll(0), instance.runtime_state.run_id)
                recorder.finish.assert_called_once()

    def test_discard_during_prepare_keeps_the_discard_request(self):
        instance = active_loop()
        recorder, instance.recorder = instance.recorder, None
        instance.phase = loop.TeleopPhase.IDLE
        instance.runtime_state.revoke_motion()
        instance.admit_observation = Mock(return_value=([], 10**30))
        keys = Keyboard(discard_callback=instance.request_discard)
        keys.active, keys.listener = True, Mock()
        self.addCleanup(keys.stop)
        instance.keyboard = keys

        def prepare(*args, **kwargs):
            instance.robot.stop_required = True
            keys.handle_key("d", True)
            raise OperationCancelled("prepare cancelled")

        instance.robot.prepare_streaming.side_effect = prepare
        with (
            patch.object(loop, "read_observation", return_value=observation()),
            patch.object(loop, "snapshot_recording_metadata", return_value={}),
            patch.object(loop, "EpisodeRecorder", return_value=recorder),
            self.assertLogs(loop.logger, level="WARNING"),
        ):
            instance.start_episode(instance.runtime_state.run_id)
        self.assertFalse(recorder.finish.call_args.kwargs["save"])
        instance.robot.send_action.assert_not_called()
        self.assertEqual(instance.phase, loop.TeleopPhase.IDLE)

    def test_stop_revokes_before_finishing_and_only_finishes_once(self):
        instance = active_loop()
        recorder = instance.recorder
        calls = []

        def stop():
            self.assertFalse(instance.runtime_state.can_move(run_id=instance.run_id))
            instance.robot.stop_required = False
            calls.append("stop")

        instance.robot.stop.side_effect = stop
        recorder.finish.side_effect = lambda **kwargs: calls.append("finish") or Path("episode")
        instance.handle_operator_command("s")
        instance.close()
        self.assertEqual(calls, ["stop", "finish"])
        self.assertEqual(recorder.finish.call_args.kwargs["reason"], "operator")
        self.assertEqual(recorder.finish.call_args.kwargs["technical_status"], "normal")
        self.assertEqual(instance.phase, loop.TeleopPhase.IDLE)

    def test_discard_survives_stop_and_writer_failure(self):
        instance = active_loop()
        recorder = instance.recorder
        stop_error = RobotError("stop failed")
        instance.robot.stop.side_effect = stop_error
        recorder.finish.side_effect = RecordingError("writer pending")
        with self.assertLogs(loop.logger, level="ERROR"):
            with self.assertRaises(RobotError) as caught:
                instance.handle_operator_command("d")
        self.assertIs(caught.exception, stop_error)
        self.assertIs(instance.recorder, recorder)
        self.assertNotEqual(instance.phase, loop.TeleopPhase.IDLE)
        self.assertFalse(recorder.finish.call_args.kwargs["save"])
        self.assertEqual(recorder.finish.call_args.kwargs["reason"], "operator")
        self.assertEqual(recorder.finish.call_args.kwargs["technical_status"], "failed")
        instance.controller.clear.assert_called_once()
        instance.robot.stop.side_effect = None
        recorder.finish.side_effect = None
        instance.close()
        self.assertFalse(recorder.finish.call_args.kwargs["save"])
        self.assertEqual(recorder.finish.call_args.kwargs["reason"], "operator")

    def test_solver_failure_is_recorded_before_cleanup_and_propagates(self):
        instance = active_loop()
        recorder = instance.recorder
        failure = nlopt.RoundoffLimited("solver failed")
        instance.controller.compute_target.side_effect = failure
        with self.assertLogs(loop.logger, level="WARNING"):
            with self.assertRaises(nlopt.RoundoffLimited) as caught:
                deadline = time.monotonic_ns() + 10**9
                instance.execute_control_step(observation(), deadline, deadline)
        self.assertIs(caught.exception, failure)
        instance.robot.send_action.assert_not_called()
        instance.controller.commit.assert_not_called()
        saved = recorder.finish.call_args.kwargs
        self.assertEqual(saved["reason"], "policy_failure")
        self.assertEqual(saved["technical_status"], "failed")
        self.assertTrue(any(item.get("stage") == "control" for item in saved["details"]))

    def test_solver_error_is_not_masked_by_stop_failure(self):
        instance = active_loop()
        solver_error = RuntimeError("solver failed")
        stop_error = RobotError("stop failed")
        instance.controller.compute_target.side_effect = solver_error
        instance.robot.stop.side_effect = stop_error
        with self.assertRaises(RuntimeError) as caught:
            instance.execute_control_step(observation(), 10**30, 10**30)
        self.assertIs(caught.exception, solver_error)
        self.assertIs(caught.exception.__cause__, stop_error)
        instance.controller.commit.assert_not_called()

    def test_solver_timeout_is_not_treated_as_submission_cancellation(self):
        instance = active_loop()
        failure = TimeoutError("solver timed out")
        instance.controller.compute_target.side_effect = failure
        with self.assertLogs(loop.logger, level="WARNING"):
            with self.assertRaises(TimeoutError) as caught:
                instance.execute_control_step(observation(), 10**30, 10**30)
        self.assertIs(caught.exception, failure)
        self.assertTrue(instance.runtime_state.error_state)
        instance.robot.send_action.assert_not_called()

    def test_keyboard_interrupt_latches_estop_and_records_failure(self):
        instance = active_loop()
        recorder = instance.recorder
        instance.controller.compute_target.side_effect = KeyboardInterrupt()
        with self.assertLogs(loop.logger, level="WARNING"):
            with self.assertRaises(KeyboardInterrupt):
                instance.execute_control_step(observation(), 10**30, 10**30)
        self.assertTrue(instance.runtime_state.emergency_signal.is_set())
        self.assertEqual(recorder.finish.call_args.kwargs["reason"], "estop")
        self.assertEqual(recorder.finish.call_args.kwargs["technical_status"], "failed")
        instance.robot.send_action.assert_not_called()

    def test_rejection_cleanup_error_is_not_retried_as_solver_failure(self):
        instance = active_loop()
        recorder = instance.recorder
        instance.controller.compute_target.return_value = None
        recorder.finish.side_effect = RecordingError("writer failed")
        with self.assertLogs(loop.logger, level="ERROR"):
            with self.assertRaises(RecordingError):
                instance.execute_control_step(observation(), 10**30, 10**30)
        recorder.finish.assert_called_once()
        instance.robot.send_action.assert_not_called()

    def test_stop_during_compute_or_send_does_not_commit(self):
        for stage in ("compute", "send"):
            instance = active_loop()
            result = instance.controller.compute_target.return_value
            if stage == "compute":
                instance.controller.compute_target.side_effect = lambda row: (
                    instance.runtime_state.request_stop() or result
                )
            else:
                instance.robot.send_action.side_effect = (
                    lambda *args, **kwargs: instance.runtime_state.request_stop()
                )
            with self.assertLogs(loop.logger, level="WARNING"):
                instance.execute_control_step(observation(), 10**30, 10**30)
            instance.controller.commit.assert_not_called()
            if stage == "compute":
                instance.robot.send_action.assert_not_called()

    def test_expired_submission_and_late_return_do_not_commit(self):
        for times in ((200,), (100, 200)):
            instance = active_loop()
            recorder = instance.recorder
            with patch.object(loop.time, "monotonic_ns", side_effect=times):
                with self.assertLogs(loop.logger, level="WARNING"):
                    instance.execute_control_step(observation(90), 150, 150)
            instance.controller.commit.assert_not_called()
            self.assertEqual(instance.robot.send_action.call_count, len(times) - 1)
            self.assertEqual(recorder.append.call_count, 1)
            self.assertEqual(recorder.append.call_args.args[1] is None, len(times) == 1)

    def test_successful_step_commits_before_recording(self):
        instance = active_loop()
        calls = []
        instance.robot.send_action.side_effect = lambda *args, **kwargs: calls.append("send")
        instance.controller.commit.side_effect = lambda result: calls.append("commit")
        instance.recorder.append.side_effect = lambda *args: calls.append("record")
        instance.execute_control_step(observation(), 10**30, 10**30)
        self.assertEqual(calls, ["send", "commit", "record"])

    def test_preparation_needs_new_feedback_and_keeps_one_run_id(self):
        instance = active_loop(recording=False)
        instance.runtime_state.revoke_motion()
        instance.run_id = instance.runtime_state.run_id
        instance.phase = loop.TeleopPhase.PREPARING
        instance.prepared_after_ns = 100
        instance.runtime_state.stop_signal.clear()
        epoch = instance.run_id
        instance.try_begin(observation(100))
        instance.controller.reset.assert_not_called()
        instance.try_begin(observation(101))
        self.assertEqual(instance.phase, loop.TeleopPhase.ACTIVE)
        self.assertEqual(instance.run_id, epoch)
        self.assertTrue(instance.runtime_state.can_move(run_id=epoch))

    def test_stop_while_waiting_for_preparation_does_not_start_recorder(self):
        instance = active_loop()
        instance.recorder = None
        instance.phase = loop.TeleopPhase.IDLE
        instance.runtime_state.revoke_motion()
        instance.keyboard.poll.return_value = ["s"]

        def read(*args, **kwargs):
            self.assertTrue(kwargs["cancel_requested"]())
            raise OperationCancelled()

        with (
            patch.object(loop, "read_observation", side_effect=read),
            patch.object(loop, "EpisodeRecorder") as recorder,
        ):
            instance.start_episode(instance.runtime_state.run_id)
        recorder.assert_not_called()
        instance.robot.prepare_streaming.assert_not_called()
        self.assertEqual(instance.phase, loop.TeleopPhase.IDLE)

    def test_old_begin_cannot_cross_stop(self):
        instance = active_loop(recording=False)
        instance.phase = loop.TeleopPhase.IDLE
        old_epoch = instance.runtime_state.run_id
        instance.runtime_state.request_stop()
        instance.start_episode(old_epoch)
        instance.robot.prepare_streaming.assert_not_called()
        self.assertEqual(instance.phase, loop.TeleopPhase.IDLE)

    def test_cancellation_during_prepare_never_begins_motion(self):
        instance = active_loop(recording=False)
        instance.phase = loop.TeleopPhase.IDLE
        instance.runtime_state.revoke_motion()
        instance.admit_observation = Mock(return_value=([], 10**30))

        def prepare(*args, **kwargs):
            instance.robot.stop_required = True
            instance.runtime_state.request_stop()
            raise OperationCancelled("prepare cancelled")

        instance.robot.prepare_streaming.side_effect = prepare
        with patch.object(loop, "read_observation", return_value=observation()):
            with self.assertLogs(loop.logger, level="WARNING"):
                instance.start_episode(instance.runtime_state.run_id)
        self.assertEqual(instance.phase, loop.TeleopPhase.IDLE)
        instance.controller.reset.assert_not_called()
        instance.robot.send_action.assert_not_called()

    def test_missed_tick_stops_without_catchup(self):
        instance = active_loop()
        instance.tick_origin_ns = time.monotonic_ns() - 3 * instance.control_dt_ns
        with patch.object(loop, "read_observation") as read:
            with self.assertLogs(loop.logger, level="WARNING"):
                instance.run_control_tick()
        read.assert_not_called()
        instance.robot.send_action.assert_not_called()
        self.assertEqual(instance.phase, loop.TeleopPhase.IDLE)

    def test_late_observation_is_recorded_after_stop(self):
        instance = active_loop()
        instance.tick_origin_ns = 100
        instance.control_dt_ns = 10
        row = observation(120)
        instance.admit_observation = Mock(return_value=([], 1000))
        calls = []
        instance.robot.stop.side_effect = lambda: calls.append("stop")
        instance.recorder.append.side_effect = lambda *args: calls.append("record")
        with (
            patch.object(loop.time, "monotonic_ns", return_value=100),
            patch.object(loop, "read_observation", return_value=row),
        ):
            with self.assertLogs(loop.logger, level="WARNING"):
                instance.run_control_tick()
        self.assertEqual(calls, ["stop", "record"])
        instance.robot.send_action.assert_not_called()

    def test_recording_limit_does_not_limit_unrecorded_teleop(self):
        for recording in (True, False):
            instance = active_loop(recording)
            instance.capture_started_s -= 100
            instance.admit_observation = Mock(return_value=([], 10**30))
            instance.execute_control_step = Mock()
            with patch.object(loop, "read_observation", return_value=observation()):
                instance.run_control_tick()
            self.assertEqual(instance.execute_control_step.call_count, int(not recording))

    def test_control_and_recording_source_failures_remain_distinct(self):
        instance = active_loop()
        row = observation()
        row.vr["recv_ts_ns"] = 1
        self.assertIn("vr_stale", instance.admit_observation(row)[0])
        row = observation()
        row.camera = None
        with self.assertRaisesRegex(RecordingError, "recording sources"):
            instance.admit_observation(row)


class SessionTests(unittest.TestCase):
    def test_cleanup_runs_after_startup_capture_and_audio_failures(self):
        for failing_stage in (None, "keyboard", "control", "capture", "audio"):
            with self.subTest(stage=failing_stage), ExitStack() as stack:
                state = SessionState()
                robot, keys, sensors, player, control_loop = (Mock() for _ in range(5))
                sensors.shutdown.return_value = True
                player.wait_until_idle.return_value = True
                control_loop.run.side_effect = state.request_quit
                operations = {
                    "keyboard": keys.start, "control": control_loop.run,
                    "capture": control_loop.close,
                    "audio": player.wait_until_idle,
                }
                if failing_stage is not None:
                    operations[failing_stage].side_effect = OSError(failing_stage)
                    stack.enter_context(self.assertLogs(session.logger, level="ERROR"))
                stack.enter_context(patch.multiple(
                    session, configure_logging=Mock(), validate_robot_config=Mock(),
                    validate_control_rate=Mock(), resolve_table_plane=Mock(),
                    load_vr_transform=Mock(return_value=SimpleNamespace(transform=np.eye(3))),
                    ActionMapper=Mock(), VRController=Mock(), HomePlanner=Mock(),
                    SensorChannels=Mock(), SessionState=Mock(return_value=state),
                    SensorWorkers=Mock(return_value=sensors), Robot=Mock(return_value=robot),
                    Keyboard=Mock(return_value=keys), AudioFeedback=Mock(return_value=player),
                    TeleopLoop=Mock(return_value=control_loop),
                ))
                runtime = load_experiment_config(cli_overrides={
                    "collection.enabled": False, "hand.enabled": False,
                })
                result = session.run_teleop(runtime, allow_no_hand=True)
                self.assertEqual(result, int(failing_stage is not None))
                self.assertEqual(keys.discard_callback, control_loop.request_discard)
                control_loop.close.assert_called_once()
                player.close.assert_called_once()
                robot.close.assert_called_once()
                keys.stop.assert_called_once()
                sensors.shutdown.assert_called_once()


class FakePlayer:
    def __init__(self, done=False):
        self.returncode = None
        self.waiting = threading.Event()
        self.finished = threading.Event()
        self.killed = False
        if done:
            self.finished.set()

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9
        self.finished.set()

    def communicate(self):
        self.waiting.set()
        if not self.finished.wait(1):
            raise TimeoutError("fake player was not released")
        if self.returncode is None:
            self.returncode = 0
        return b"", b""


class AudioTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.context = stack.enter_context
        self.context(patch.object(audio.shutil, "which", return_value="/usr/bin/aplay"))
        self.context(patch.object(Path, "is_file", return_value=True))

    def make_audio(self, factory):
        popen = self.context(patch.object(audio.subprocess, "Popen", side_effect=factory))
        player = audio.AudioFeedback()
        self.addCleanup(player.close)
        return player, popen

    def test_preemption_clears_queued_prompts(self):
        first, second = FakePlayer(), FakePlayer()
        player, popen = self.make_audio([first, second])
        player.play("begin")
        self.assertTrue(first.waiting.wait(1))
        player.play("home_done", interrupt=False)
        player.play("emergency")
        self.assertTrue(second.waiting.wait(1))
        self.assertTrue(first.killed)
        second.finished.set()
        self.assertTrue(player.wait_until_idle(1))
        self.assertEqual(popen.call_count, 2)
        self.assertEqual(Path(popen.call_args.args[0][-1]).name, audio.EVENT_FILES["emergency"])

    def test_noninterrupting_prompts_play_in_order(self):
        first, second = FakePlayer(), FakePlayer(done=True)
        player, popen = self.make_audio([first, second])
        player.play("home")
        self.assertTrue(first.waiting.wait(1))
        player.play("home_done", interrupt=False)
        self.assertFalse(player.wait_until_idle(.001))
        first.finished.set()
        self.assertTrue(player.wait_until_idle(1))
        self.assertFalse(first.killed)
        self.assertEqual(popen.call_count, 2)

    def test_preemption_during_process_creation(self):
        started, release = threading.Event(), threading.Event()
        first, second = FakePlayer(), FakePlayer(done=True)
        calls = []

        def create(*args, **kwargs):
            calls.append(args)
            if len(calls) == 1:
                started.set()
                self.assertTrue(release.wait(1))
                return first
            return second

        player, _ = self.make_audio(create)
        player.play("begin")
        self.assertTrue(started.wait(1))
        player.play("emergency")
        release.set()
        self.assertTrue(player.wait_until_idle(1))
        self.assertTrue(first.killed)
        self.assertEqual(len(calls), 2)

    def test_close_during_process_creation_reaps_child(self):
        started, release = threading.Event(), threading.Event()
        child = FakePlayer()

        def create(*args, **kwargs):
            started.set()
            self.assertTrue(release.wait(1))
            return child

        player, popen = self.make_audio(create)
        player.play("begin")
        self.assertTrue(started.wait(1))
        closing = threading.Thread(target=player.close)
        closing.start()
        with player.condition:
            self.assertTrue(player.condition.wait_for(lambda: player.closed, timeout=1))
        release.set()
        closing.join(1)
        self.assertFalse(closing.is_alive())
        self.assertFalse(player.worker.is_alive())
        self.assertTrue(child.killed)
        popen.assert_called_once()

    def test_playback_failure_is_reported_and_disables_audio(self):
        player, popen = self.make_audio(OSError("player missing"))
        with self.assertLogs(audio.logger, level="WARNING") as logs:
            player.play("begin")
            self.assertTrue(player.wait_until_idle(1))
        self.assertIn("player missing", "\n".join(logs.output))
        player.play("end")
        popen.assert_called_once()
        self.assertIsNone(player.player)

    def test_missing_player_starts_no_worker(self):
        with (
            patch.object(audio.shutil, "which", return_value=None),
            patch.object(audio.threading, "Thread") as thread,
        ):
            with self.assertLogs(audio.logger, level="WARNING"):
                player = audio.AudioFeedback()
        thread.assert_not_called()
        player.play("begin")
        self.assertTrue(player.wait_until_idle(0))
        player.close()


if __name__ == "__main__":
    unittest.main()
