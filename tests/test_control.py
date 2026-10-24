"""Jog 授权和目标映射；所有发送仅进入 mock。"""

import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.planning.action import ActionIntent, ActionMapper
from dexmani_real.planning.kinematics.ik import IKFailureKind, IKResult
from dexmani_real.planning.kinematics.pose import Pose
from dexmani_real.planning.target import TargetChecker
from dexmani_real.replay.capture import ReplayRecorder
from dexmani_real.replay.evaluation import compute_metrics, save_replay_data
from dexmani_real.replay.replayer import ReplayStatus, TrajectoryReplayer
from dexmani_real.robot.commands import DispatchError, DispatchResult, DispatchStatus, RobotCommand
from dexmani_real.runtime.jog import JogController
from dexmani_real.runtime.operator_input import KeyboardInput
from dexmani_real.runtime.state import RuntimeState
from tests.offline_fixtures import make_row


class JogTests(unittest.TestCase):
    def setUp(self):
        self.reset_jog()
        timer = patch("dexmani_real.runtime.jog.time.monotonic_ns", side_effect=lambda: self.clock[0])
        timer.start()
        self.addCleanup(timer.stop)

    def reset_jog(self):
        self.state = RuntimeState()
        self.state.arm()
        self.keys = KeyboardInput(capture_commands=False)
        self.keys.callbacks_active = True
        self.keys.held_keys.add("w")
        self.solver = SimpleNamespace(
            kin=SimpleNamespace(compute_eef_pose_base=Mock(return_value=Pose([0, 0, 0], [1, 0, 0, 0]))),
            collision_model=SimpleNamespace(set_hand_qpos=Mock()),
            solve=Mock(side_effect=lambda pose, current, previous, **kw: IKResult(
                success=True, qpos=previous + 0.01,
            )),
        )
        self.checker = SimpleNamespace(endpoint_rejection=lambda q: None)
        self.config = SimpleNamespace(delta_pos_m=0.1, delta_rpy_rad=0.1, workspace_command_margin_m=0)
        self.jog = JogController(self.solver, self.checker, self.config, np.array([[-1., 1.]] * 3))
        self.robot = SimpleNamespace(send_action=Mock(return_value=DispatchResult(arm=DispatchStatus.SUBMITTED)))
        self.clock = [100]

    def propose(self):
        return self.jog.propose({"w"}, np.zeros(7), hand_qpos=np.zeros(12))

    def submit(self, pose, target):
        return self.jog.submit(
            pose, target, self.keys, 0, self.state, self.robot, self.state.run_id,
            valid_until_ns=200, execute_until_ns=180,
        )

    def test_reference_advances_only_on_success_and_owns_copies(self):
        pose, _, result = self.propose()
        self.assertIsNone(self.jog.command_qpos)
        self.submit(pose, result.qpos)
        pose.p[:] = 9
        result.qpos[:] = 9
        np.testing.assert_allclose(self.jog.command_pose.p, [0.1, 0, 0])
        np.testing.assert_allclose(self.jog.command_qpos, 0.01)
        following, _, _ = self.jog.propose({"w"}, np.ones(7), hand_qpos=np.ones(12))
        np.testing.assert_allclose(following.p, [0.2, 0, 0])
        np.testing.assert_allclose(self.solver.solve.call_args.args[1], 1)
        np.testing.assert_allclose(self.solver.solve.call_args.args[2], 0.01)
        self.solver.kin.compute_eef_pose_base.assert_called_once()

    def test_input_and_authority_changes_before_reservation(self):
        for change in ("generation", "authority", "deadline"):
            with self.subTest(change=change):
                self.reset_jog()
                pose, _, result = self.propose()
                if change == "generation":
                    self.keys.generation += 1
                elif change == "authority":
                    self.state.request_stop()
                else:
                    self.clock[0] = 180
                if change == "deadline":
                    with self.assertRaises(DispatchError):
                        self.submit(pose, result.qpos)
                else:
                    self.assertIsNone(self.submit(pose, result.qpos))
                self.robot.send_action.assert_not_called()
                self.assertIsNone(self.jog.command_pose)

    def test_input_authority_deadline_and_partial_dispatch_during_send(self):
        for change in ("generation", "authority", "deadline", "partial"):
            with self.subTest(change=change):
                self.reset_jog()
                pose, _, result = self.propose()

                def send(*args, **kwargs):
                    if change == "generation":
                        self.keys.generation += 1
                    elif change == "authority":
                        self.state.request_stop()
                    elif change == "deadline":
                        self.clock[0] = 180
                    return DispatchResult(arm=(
                        DispatchStatus.ACCEPTED if change == "partial" else DispatchStatus.SUBMITTED
                    ))

                self.robot.send_action.side_effect = send
                with self.assertRaises(DispatchError):
                    self.submit(pose, result.qpos)
                self.assertIsNone(self.jog.command_qpos)
                self.robot.send_action.assert_called_once()

    def test_ik_and_workspace_rejection_require_physical_release(self):
        self.solver.solve.side_effect = None
        self.solver.solve.return_value = IKResult(success=False, qpos=None, reason="self_collision")
        self.propose()
        self.assertTrue(self.jog.blocked_until_release)
        self.jog.clear_reference()
        self.jog.release_if_idle({"w", "s"})
        self.assertTrue(self.jog.blocked_until_release)
        self.jog.release_if_idle(set())
        self.assertFalse(self.jog.blocked_until_release)
        self.jog.command_pose = Pose([1, 0, 0], [1, 0, 0, 0])
        self.jog.command_qpos = np.zeros(7)
        self.solver.solve.reset_mock()
        _, _, result = self.propose()
        self.assertIsNone(result)
        self.solver.solve.assert_not_called()
        self.assertTrue(self.jog.blocked_until_release)

    def test_invalid_ik_output_is_technical_failure(self):
        self.solver.solve.side_effect = None
        self.solver.solve.return_value = IKResult(
            success=False, qpos=None, reason="invalid", failure_kind=IKFailureKind.INVALID_OUTPUT,
        )
        with self.assertRaisesRegex(RuntimeError, "technical failure"):
            self.propose()

    def test_session_feedback_hand_and_generation_boundaries(self):
        from dexmani_real.teleop.keyboard_session import KeyboardTeleopSession
        from dexmani_real.calibration.camera.session import CameraCalibrationSession
        from dexmani_real.runtime.observation import feedback_deadline_ns

        for calibration in (False, True):
            for revoke in (False, True):
                with self.subTest(calibration=calibration, revoke=revoke):
                    self.reset_jog()
                    runtime = ExperimentConfig()
                    row = make_row()
                    row.arm["qpos"][:] = 0.2
                    row.hand["qpos"][:] = 0.4
                    row.hand["timestamp_ns"] -= 50_000_000
                    refreshed = []

                    def prepare(epoch):
                        refreshed.append(True)
                        if revoke:
                            self.keys.generation += 1

                    robot = SimpleNamespace(
                        stop_required=False, prepare_streaming=prepare, stop=Mock(),
                        arm=SimpleNamespace(update_health=Mock()),
                        read_state=Mock(return_value=row),
                    )
                    session_type = CameraCalibrationSession if calibration else KeyboardTeleopSession
                    session = session_type.__new__(session_type)
                    session.runtime, session.runtime_state = runtime, self.state
                    session.keys, session.robot = self.keys, robot
                    session.jog = self.jog
                    if calibration:
                        session.solver = self.solver
                    session.jog_config = runtime.keyboard_teleop
                    session.current_qpos = np.zeros(7)
                    session.feedback_timestamp_ns = 1
                    session.frame = 1
                    session.last_boundary_warning_s = session.last_ik_warning_s = 0
                    session.calibration_config = SimpleNamespace(status_interval_frames=20)
                    self.jog.submit = Mock(return_value=DispatchResult(arm=DispatchStatus.SUBMITTED))

                    def observe(*args, **kwargs):
                        self.assertTrue(refreshed)
                        return row

                    with patch("dexmani_real.teleop.keyboard_session.read_observation", side_effect=observe), patch(
                        "dexmani_real.teleop.keyboard_session.upstream_deadline_ns", return_value=123,
                    ):
                        session.motion_tick() if calibration else session.motion_tick(False)
                    if revoke:
                        self.solver.solve.assert_not_called()
                        self.jog.submit.assert_not_called()
                        if calibration:
                            robot.read_state.assert_called_once()
                        continue
                    np.testing.assert_allclose(self.solver.solve.call_args.args[1], 0.2)
                    expected_hand = np.deg2rad(runtime.hand.home_qpos_deg) if calibration else row.hand["qpos"][0]
                    np.testing.assert_allclose(self.solver.collision_model.set_hand_qpos.call_args.args[0], expected_hand)
                    expected_deadline = (
                        int(row.arm["timestamp_ns"][0]) + int(runtime.arm.feedback_max_age_s * 1e9) + 1
                        if calibration else feedback_deadline_ns(row, runtime)
                    )
                    self.assertEqual(self.jog.submit.call_args.kwargs["valid_until_ns"], expected_deadline)
                    robot.arm.update_health.assert_called_once_with(upstream_until_ns=2**63 - 1 if calibration else 123)


class JointTests(unittest.TestCase):
    def setUp(self):
        self.collision = Mock()
        self.collision.check_self_collision.return_value = False
        self.collision.within_joint_limits.return_value = True
        self.collision.check_environment_collision_details.return_value = SimpleNamespace(in_collision=False)
        checker = TargetChecker(
            self.collision, joint_lower_rad=[-0.25] * 7, joint_upper_rad=[0.25] * 7,
            workspace=[[-1, 1]] * 3, workspace_margin_m=0,
            target_fk=SimpleNamespace(compute=lambda q: (np.zeros(3), None)),
        )
        self.mapper = ActionMapper(
            checker, hand_enabled=True, hand_lower_rad=np.zeros(12), hand_upper_rad=np.ones(12),
            home_hand_qpos=np.zeros(12),
        )
        self.arm = np.array([2 * np.pi + 0.1, 1, -1, 0, 0, 0, 0])
        self.hand = np.linspace(-1, 2, 12)
        self.expected = np.array([0.1, 0.25, -0.25, 0, 0, 0, 0])

    def test_candidates_and_map_rejection(self):
        for rejected in (False, True):
            self.collision.check_self_collision.return_value = rejected
            arm, hand, reason = self.mapper.prepare_joint_target(self.arm, self.hand, np.zeros(7))
            np.testing.assert_allclose(arm, self.expected)
            np.testing.assert_allclose(hand, np.clip(self.hand, 0, 1))
            result = self.mapper.map(ActionIntent("joint", self.arm, self.hand), np.zeros(7))
            self.assertEqual(reason, "self_collision" if rejected else None)
            if rejected:
                self.assertIsNone(result.arm_qpos)
            else:
                np.testing.assert_array_equal(result.arm_qpos, arm)
            np.testing.assert_array_equal(result.hand_qpos, hand)

    def test_replay_records_rejected_candidate_without_sending(self):
        for rejected in (False, True):
            with self.subTest(rejected=rejected):
                self.collision.check_self_collision.return_value = rejected
                row = make_row(100)
                robot = SimpleNamespace(
                    arm=SimpleNamespace(update_health=Mock()),
                    send_action=Mock(return_value=DispatchResult(
                        arm=DispatchStatus.SUBMITTED, hand=DispatchStatus.ACCEPTED,
                    )),
                )
                replay = TrajectoryReplayer(RuntimeState(), ExperimentConfig(), None, robot=robot)
                replay.action_mapper, replay.epoch = self.mapper, 1
                replay.previous_arm = np.zeros(7)
                replay.capture = ReplayRecorder(1)
                replay.wait_replay = Mock(return_value=None)
                trajectory = SimpleNamespace(num_frames=1, action_arm_joint=self.arm[None], action_hand_joint=self.hand[None])
                with patch("dexmani_real.replay.replayer.time.monotonic_ns", return_value=100), patch(
                    "dexmani_real.replay.replayer.read_observation", return_value=row,
                ), patch("dexmani_real.replay.replayer.upstream_deadline_ns", return_value=1000):
                    status, _ = replay.replay_rows(trajectory, 1000)
                data = replay.capture.to_dict()
                np.testing.assert_allclose(data["arm_candidate"][0], self.expected)
                np.testing.assert_allclose(data["hand_candidate"][0], np.clip(self.hand, 0, 1))
                self.assertNotIn("arm_cmd", data)
                self.assertNotIn("hand_cmd", data)
                if rejected:
                    self.assertEqual(status, ReplayStatus.REJECTED)
                    robot.send_action.assert_not_called()
                    self.assertFalse(data["dispatch_status"].any())
                    np.testing.assert_array_equal(replay.previous_arm, 0)
                else:
                    self.assertEqual(status, ReplayStatus.COMPLETED)
                    robot.send_action.assert_called_once()
                    np.testing.assert_allclose(replay.previous_arm, self.expected)

    def test_factory_runtime_and_joint_without_mplib(self):
        runtime = ExperimentConfig()
        with patch.dict(sys.modules, {"mplib": None}), patch(
            "dexmani_real.planning.collision.CollisionModel", return_value=self.collision,
        ) as collision, patch("dexmani_real.planning.target.make_arm_fk", return_value=Mock()), patch(
            "dexmani_real.planning.action.make_online_ik_solver", side_effect=AssertionError("joint constructed IK"),
        ):
            mapper = ActionMapper.for_mode(runtime, "joint")
        self.assertIsNone(mapper.solver)
        collision.assert_called_once_with(static_boxes=runtime.environment.static_boxes)
        np.testing.assert_allclose(mapper.target_checker.limits[:, 0], runtime.arm.joint_limit_lower)
        np.testing.assert_allclose(mapper.home_hand_qpos, np.deg2rad(runtime.hand.home_qpos_deg))
        np.testing.assert_allclose(mapper.hand_lower, runtime.hand.qpos_min_rad)
        np.testing.assert_allclose(mapper.target_checker.workspace, runtime.environment.workspace.as_array())

    def test_replay_metrics_use_dispatch_and_persist_only_current_fields(self):
        capture = ReplayRecorder(3)
        for index, (target, status) in enumerate((
            (10.0, DispatchStatus.NOT_CALLED),
            (0.1, DispatchStatus.UNKNOWN),
            (0.3, DispatchStatus.SUBMITTED),
        )):
            capture.record(
                index, make_row(), RobotCommand(1, np.full(7, target)),
                DispatchResult(arm=status),
            )
        data = capture.to_dict()
        poses = np.tile([0, 0, 0, 1, 0, 0, 0, 1, 0], (3, 1))
        trajectory = SimpleNamespace(
            fps=10, arm_qpos=np.zeros((3, 7)), hand_qpos=np.zeros((3, 12)),
            arm_ee=poses, episode_path="offline", task_label="offline",
        )
        derived = dict(data, eef_pos=poses[:, :3], eef_rot6d=poses[:, 3:])
        metrics = compute_metrics(trajectory, derived)
        self.assertEqual(
            metrics["pre_submit_arm_target_gap"]["mean_deg"], round(np.rad2deg(0.2), 2),
        )
        derived["dispatch_status"] = np.zeros((3, 2), dtype=np.uint8)
        self.assertNotIn("pre_submit_arm_target_gap", compute_metrics(trajectory, derived))
        with tempfile.TemporaryDirectory() as directory:
            saved = save_replay_data(data, directory)
            with np.load(saved, allow_pickle=False) as loaded:
                self.assertNotIn("arm_cmd", loaded.files)
                self.assertNotIn("hand_cmd", loaded.files)
                np.testing.assert_array_equal(loaded["arm_candidate"], data["arm_candidate"])
                np.testing.assert_array_equal(loaded["dispatch_status"], data["dispatch_status"])

    def test_eef_and_frozen_command_semantics(self):
        solver = SimpleNamespace(
            collision_model=self.collision,
            solve=Mock(return_value=IKResult(success=True, qpos=np.zeros(7))),
        )
        self.mapper.solver = solver
        pose = Pose([2, 0, 0], [1, 0, 0, 0])
        result = self.mapper.map(ActionIntent("eef", pose, np.zeros(12)), np.zeros(7))
        np.testing.assert_array_equal(pose.p, [2, 0, 0])
        np.testing.assert_array_equal(result.eef_pose.p, [1, 0, 0])
        command = RobotCommand(1, np.ones(7), np.zeros(12))
        self.assertEqual(self.mapper.frozen_rejection(command, np.zeros(7), None, "joint"), "frozen_equivalent_branch_changed")
        with self.assertRaises(ValueError):
            self.mapper.prepare_joint_target(np.full(7, np.nan), self.hand, np.zeros(7))
