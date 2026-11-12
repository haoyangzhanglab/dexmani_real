"""HOME 离线契约；这里的阈值仅为合成测试数据，不是设备验收值。"""

import multiprocessing as mp
import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.control import StaticCollisionBox
from dexmani_real.config.real import RealConfig
from dexmani_real.control.actions import ActionMapper
from dexmani_real.control.collision import CollisionModel
from dexmani_real.robot.commands import (
    FeedbackUnavailable,
    HomeTimeout,
    OperationCancelled,
)
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE, HomeRequest
from dexmani_real.robot.xarm7.trajectory import (
    PositionLimitError,
    calculate_home_segment,
    validate_extrema,
)
from dexmani_real.robot.xarm7.worker import ArmController
from dexmani_real.robot.xhand.driver import COMMUNICATION_CRC_ERROR_CODE, XHand
from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE
from dexmani_real.runtime.home import (
    HomeConfigurationError,
    HomePathRejected,
    _home_hand,
    home_robot,
    plan_home,
)
from dexmani_real.runtime.state import SessionState


def config():
    cfg = RealConfig()
    arm = replace(
        cfg.arm,
        homing=replace(
            cfg.arm.homing,
            max_jerk_deg_per_s3=1000.0,
            collision_margin_m=0.005,
            tracking_error_rad=0.1,
            hand_hold_tolerance_rad=0.02,
            max_lateness_s=0.005,
        ),
    )
    hand = replace(
        cfg.hand, home_timeout_s=2.0, home_max_step_deg=1.0, home_tor_max=100
    )
    return replace(
        cfg,
        arm=arm,
        hand=hand,
        environment=replace(cfg.environment, home_model_confirmed=True),
    )


class Clock:
    def __init__(self):
        self.ns = 10**10

    def now(self):
        return self.ns

    def sleep(self, seconds):
        self.ns += round(seconds * 1e9)


class HomeContracts(unittest.TestCase):
    def setUp(self):
        self.cfg = config()
        self.q = np.asarray(self.cfg.arm.home_qpos) + 0.05
        self.hand_q = np.deg2rad(self.cfg.hand.home_qpos_deg)

    def test_direct_and_no_via_default(self):
        model = Mock(arm_radii=np.ones(7))
        model.rejection.return_value = None
        with patch(
            "dexmani_real.runtime.home.calculate_home_segment",
            wraps=calculate_home_segment,
        ) as calc:
            samples = plan_home(
                self.cfg.arm, self.q, np.zeros(7), self.hand_q, model, lambda: None
            )
        self.assertEqual(calc.call_count, 1)
        np.testing.assert_array_equal(samples[0], self.q)
        np.testing.assert_array_equal(samples[-1], self.cfg.arm.home_qpos)
        checked = [call.args[0] for call in model.rejection.call_args_list]
        self.assertTrue(
            all(any(np.array_equal(q, c) for c in checked) for q in samples)
        )
        model.rejection.return_value = "collision"
        with self.assertRaises(HomePathRejected):
            plan_home(
                self.cfg.arm, self.q, np.zeros(7), self.hand_q, model, lambda: None
            )

    def test_via_both_segments_and_model_error(self):
        via = tuple(self.q + 0.05)
        cfg = replace(self.cfg.arm, homing=replace(self.cfg.arm.homing, via_qpos=via))
        model = Mock(arm_radii=np.zeros(7))
        for bad_segment in (None, 1, 2):
            segments = []

            def calculate(*args, segments=segments):
                segments.append(args[-1])
                return calculate_home_segment(*args)

            # Direct 是第 1 段，Via 是第 2/3 段。
            def rejection(q, hand, segments=segments, bad_segment=bad_segment):
                if len(segments) == 1 or (
                    bad_segment is not None and len(segments) == bad_segment + 1
                ):
                    return "blocked"
                return None

            model.rejection.side_effect = rejection
            with patch(
                "dexmani_real.runtime.home.calculate_home_segment",
                side_effect=calculate,
            ):
                if bad_segment is None:
                    samples = plan_home(
                        cfg, self.q, np.zeros(7), self.hand_q, model, lambda: None
                    )
                    self.assertTrue(any(np.array_equal(q, via) for q in samples))
                    self.assertEqual(len(segments), 3)
                else:
                    with self.assertRaises(HomePathRejected):
                        plan_home(
                            cfg, self.q, np.zeros(7), self.hand_q, model, lambda: None
                        )
        model.rejection.side_effect = RuntimeError("model broken")
        with patch(
            "dexmani_real.runtime.home.calculate_home_segment",
            wraps=calculate_home_segment,
        ) as calc:
            with self.assertRaisesRegex(RuntimeError, "model broken"):
                plan_home(cfg, self.q, np.zeros(7), self.hand_q, model, lambda: None)
            self.assertEqual(calc.call_count, 1)

    def test_trajectory_limits_and_extrema_regression(self):
        trajectory = calculate_home_segment(
            self.cfg.arm, self.q, np.zeros(7), self.cfg.arm.home_qpos
        )
        q, v, a = trajectory.at_time(trajectory.duration)
        np.testing.assert_allclose(q, self.cfg.arm.home_qpos, atol=1e-12)
        np.testing.assert_allclose(v, 0, atol=1e-12)
        np.testing.assert_allclose(a, 0, atol=1e-12)
        for t in np.linspace(0, trajectory.duration, 200):
            q, v, a = trajectory.at_time(t)
            self.assertLessEqual(
                max(np.abs(v)), self.cfg.arm.homing.max_speed_rad_per_s + 1e-10
            )
            self.assertLessEqual(
                max(np.abs(a)), self.cfg.arm.homing.max_acceleration_rad_per_s2 + 1e-10
            )
        from ruckig import InputParameter, Ruckig, Trajectory

        inp = InputParameter(1)
        inp.current_position = [0.2]
        inp.current_velocity = [-0.6]
        inp.target_position = [0.5]
        inp.max_velocity = [1.0]
        inp.max_acceleration = [1.0]
        inp.max_jerk = [10.0]
        tr = Trajectory(1)
        Ruckig(1, 0.01).calculate(inp, tr)
        with self.assertRaises(PositionLimitError):
            validate_extrema(tr, [0.2], [0.0], [1.0])

    def test_geometry_margin_and_environment(self):
        model = CollisionModel(margin_m=0.005)
        self.assertTrue(
            all(
                r.security_margin == 0.005
                for r in model.geometry_data.collisionRequests
            )
        )
        obstacle = StaticCollisionBox(center_xyz_m=(0, 0, 0), size_xyz_m=(4, 4, 4))
        model = CollisionModel(static_boxes=(obstacle,), margin_m=0.005)
        self.assertIn(
            "environment_collision",
            model.rejection(np.asarray(self.cfg.arm.home_qpos), self.hand_q),
        )

    def test_normal_mapping_never_constructs_collision_model(self):
        with patch(
            "dexmani_real.control.collision.CollisionModel",
            side_effect=AssertionError("normal collision"),
        ):
            mapper = ActionMapper.for_mode(
                "eef",
                arm=self.cfg.arm,
                hand=self.cfg.hand,
                motion=self.cfg.motion,
                environment=self.cfg.environment,
            )
            measured = mapper.solver.measured_reference(self.cfg.arm.home_qpos)
            result = mapper.solver.solve(
                mapper.solver.compute_eef_pose_base(measured),
                measured,
                measured,
                measured,
                h=0.01,
            )
            self.assertFalse(hasattr(mapper.solver.constraints, "collision_model"))
            self.assertIsNotNone(result)

    def test_hand_steps_new_samples_and_convergence(self):
        clock = Clock()
        robot = Mock(hand_config=self.cfg.hand)
        q = self.hand_q - np.deg2rad(3)
        frame = np.zeros(1, dtype=HAND_STATE_DTYPE)
        writes = []
        fresh_after = clock.ns

        def service():
            nonlocal fresh_after
            if clock.ns >= fresh_after:
                frame["timestamp_ns"] = clock.ns
                frame["qpos"] = q
                fresh_after = clock.ns + 33_333_333

        def send(target, **kw):
            self.assertLessEqual(np.max(np.abs(target - q)), np.deg2rad(1) + 1e-12)
            if writes:
                self.assertGreater(int(frame["timestamp_ns"][0]), writes[-1][0])
            writes.append((clock.ns, target.copy()))
            q[:] = target

        robot.service_feedback.side_effect = service
        robot.send_hand_home.side_effect = send
        robot.hand_feedback.snapshot.return_value = frame
        service()
        with (
            patch("dexmani_real.runtime.home.time.monotonic_ns", clock.now),
            patch("dexmani_real.runtime.home.time.sleep", clock.sleep),
        ):
            _home_hand(robot, lambda: None)
        self.assertGreaterEqual(len(writes), 3)

    def test_hand_repeated_sample_times_out_and_stale_sample_fails(self):
        clock = Clock()
        robot = Mock(hand_config=replace(self.cfg.hand, home_timeout_s=0.05))
        frame = np.zeros(1, dtype=HAND_STATE_DTYPE)
        frame["qpos"] = self.hand_q
        frame["timestamp_ns"] = clock.ns
        robot.hand_feedback.snapshot.return_value = frame
        with (
            patch("dexmani_real.runtime.home.time.monotonic_ns", clock.now),
            patch("dexmani_real.runtime.home.time.sleep", clock.sleep),
            self.assertRaises(HomeTimeout),
        ):
            _home_hand(robot, lambda: None)
        self.assertEqual(robot.send_hand_home.call_count, 1)
        frame["timestamp_ns"] = 0
        with self.assertRaises(FeedbackUnavailable):
            _home_hand(robot, lambda: None)

    def test_output_caps_restore_only_on_normal_command(self):
        hand = XHand(self.cfg.hand)
        hand.connected_flag = True
        hand.control = Mock()
        hand.control.send_command.return_value = NS(error_code=0, error_message="")
        hand.command = NS(finger_command=[NS(tor_max=t) for t in self.cfg.hand.tor_max])
        hand._send_position(self.hand_q, tor_max=np.minimum(self.cfg.hand.tor_max, 100))
        self.assertEqual([j.tor_max for j in hand.command.finger_command], [100] * 12)
        hand._send_position(self.hand_q)  # 停止保持不提高输出。
        self.assertEqual([j.tor_max for j in hand.command.finger_command], [100] * 12)
        hand._send_action(self.hand_q)
        self.assertEqual(
            tuple(j.tor_max for j in hand.command.finger_command), self.cfg.hand.tor_max
        )

    def test_owner_sends_identical_samples_and_stops_on_deviation(self):
        clock = Clock()
        samples = np.array(
            [
                self.q,
                (self.q + np.asarray(self.cfg.arm.home_qpos)) / 2,
                self.cfg.arm.home_qpos,
            ]
        )
        for failure in (None, "start", "tracking", "late", "hand"):
            with self.subTest(failure=failure):
                clock.ns = 10**10
                io = NS(
                    stop=Mock(),
                    home_hand_stamp=mp.get_context("spawn").Value("q", clock.ns),
                )
                io.stop.wait.side_effect = clock.sleep
                driver = Mock()
                owner = ArmController(self.cfg.arm, io, None, driver)
                owner.status["qpos"] = samples[0] + (0.1 if failure == "start" else 0)
                owner.wait_settled = Mock()

                def read(owner=owner, failure=failure):
                    index = min(2, max(0, (clock.ns - 10**10) // 10_000_000))
                    owner.status["timestamp_ns"] = clock.ns
                    owner.status["qpos"] = samples[index] + (
                        1 if failure == "tracking" else 0
                    )

                owner.read_feedback = read
                if failure == "late":
                    driver.servo.side_effect = lambda q: clock.sleep(0.02)
                if failure == "hand":
                    io.home_hand_stamp.value = 0
                message = HomeRequest(clock.ns + 10**9, samples, 0.2)
                with patch(
                    "dexmani_real.robot.xarm7.worker.time.monotonic_ns", clock.now
                ):
                    if failure:
                        with self.assertRaises(OperationCancelled):
                            owner.home(
                                message,
                                check_abort=lambda: None,
                                deadline_ns=message.deadline_ns,
                            )
                    else:
                        owner.home(
                            message,
                            check_abort=lambda: None,
                            deadline_ns=message.deadline_ns,
                        )
                        np.testing.assert_array_equal(
                            [c.args[0] for c in driver.servo.call_args_list], samples
                        )
                if failure in ("start", "tracking", "hand"):
                    driver.servo.assert_not_called()

    def test_configuration_rejection_stops_without_motion(self):
        robot = Mock(arm_config=RealConfig().arm, hand_config=self.cfg.hand)
        state = SessionState(
            stop_signal=threading.Event(), emergency_signal=threading.Event()
        )
        with self.assertRaises(HomeConfigurationError):
            home_robot(
                state,
                robot,
                environment=self.cfg.environment,
                cancel_epoch=state.cancel_epoch,
                cancel_requested=lambda: False,
            )
        robot.arm.home.assert_not_called()
        robot.send_hand_home.assert_not_called()
        self.assertEqual(robot.stop.call_count, 2)

    def test_combined_order_rejection_skip_and_execution_failure(self):
        for scenario in (
            "success",
            "rejected",
            "execution_error",
            "already_home",
            "cancel",
        ):
            with self.subTest(scenario=scenario):
                clock = Clock()
                state = SessionState(
                    stop_signal=threading.Event(), emergency_signal=threading.Event()
                )
                robot = Mock(arm_config=self.cfg.arm, hand_config=self.cfg.hand)
                arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
                hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
                arm["qpos"] = (
                    self.cfg.arm.home_qpos if scenario == "already_home" else self.q
                )
                hand["qpos"] = self.hand_q

                def feedback(arm=arm, hand=hand, clock=clock):
                    arm["timestamp_ns"] = hand["timestamp_ns"] = clock.ns

                robot.service_feedback.side_effect = feedback
                robot.arm.snapshot.return_value = arm
                robot.hand_feedback.snapshot.return_value = hand
                calls = []

                def execute(*args, calls=calls, scenario=scenario, arm=arm, **kwargs):
                    calls.append("arm")
                    if scenario == "execution_error":
                        raise RuntimeError("SDK error")
                    arm["qpos"] = self.cfg.arm.home_qpos

                robot.arm.home.side_effect = execute
                model = Mock(arm_radii=np.ones(7))
                model.rejection.return_value = None
                error = HomePathRejected("blocked") if scenario == "rejected" else None
                with (
                    patch("dexmani_real.runtime.home.time.monotonic_ns", clock.now),
                    patch("dexmani_real.runtime.home.time.sleep", clock.sleep),
                    patch(
                        "dexmani_real.runtime.home.CollisionModel", return_value=model
                    ),
                    patch(
                        "dexmani_real.runtime.home.plan_home",
                        side_effect=error,
                        return_value=np.array([self.q, self.cfg.arm.home_qpos]),
                    ) as plan,
                    patch(
                        "dexmani_real.runtime.home._home_hand",
                        side_effect=lambda *a, calls=calls: calls.append("hand"),
                    ),
                ):

                    def run(state=state, robot=robot, scenario=scenario):
                        home_robot(
                            state,
                            robot,
                            environment=self.cfg.environment,
                            cancel_epoch=state.cancel_epoch,
                            cancel_requested=lambda: scenario == "cancel",
                        )

                    if scenario in ("success", "already_home"):
                        run()
                        self.assertEqual(
                            calls,
                            ["hand"] if scenario == "already_home" else ["arm", "hand"],
                        )
                    else:
                        with self.assertRaises((HomePathRejected, RuntimeError)):
                            run()
                        self.assertNotIn("hand", calls)
                    self.assertLessEqual(plan.call_count, 1)
                self.assertEqual(robot.stop.call_count, 2)

    def test_real_hand_submission_crc_is_fatal(self):
        from dexmani_real.robot.system import Robot
        from dexmani_real.robot.xhand.driver import XHandError

        state = SessionState(
            stop_signal=threading.Event(), emergency_signal=threading.Event()
        )
        robot = Robot(self.cfg.arm, self.cfg.hand, runtime_state=state)
        robot.connected = True
        robot.hand = Mock(is_connected=True)
        robot.hand._send_position.return_value = COMMUNICATION_CRC_ERROR_CODE
        robot._check_hand_feedback = Mock()
        with self.assertRaises(XHandError):
            robot.send_hand_home(
                self.hand_q, deadline_ns=10**30, check_abort=lambda: None
            )
        self.assertEqual(robot.hand._send_position.call_count, 1)
        self.assertTrue(robot.hand_stop_required)

    def test_via_arrival_velocity_and_acceleration_zero(self):
        via = self.q + 0.1
        first = calculate_home_segment(self.cfg.arm, self.q, np.zeros(7), via)
        second = calculate_home_segment(
            self.cfg.arm, via, np.zeros(7), self.cfg.arm.home_qpos
        )
        for tr, t in ((first, first.duration), (second, 0)):
            q, v, a = tr.at_time(t)
            np.testing.assert_allclose(q, via, atol=1e-12)
            np.testing.assert_allclose(v, 0, atol=1e-12)
            np.testing.assert_allclose(a, 0, atol=1e-12)

    def test_subdivision_checks_between_execution_samples(self):
        checked = []
        model = Mock(arm_radii=np.ones(7) * 10)
        model.rejection.side_effect = lambda q, h: checked.append(q.copy())
        points = plan_home(
            self.cfg.arm, self.q, np.zeros(7), self.hand_q, model, lambda: None
        )
        self.assertGreater(len(checked), len(points))
        midpoint = next(
            q for q in checked if not any(np.array_equal(q, p) for p in points)
        )
        model.rejection.side_effect = lambda q, h: (
            "thin obstacle" if np.array_equal(q, midpoint) else None
        )
        with self.assertRaises(HomePathRejected):
            plan_home(
                self.cfg.arm, self.q, np.zeros(7), self.hand_q, model, lambda: None
            )

    def test_policy_and_keyboard_share_home_and_do_not_resume_on_rejection(self):
        from dexmani_real.deployment.session import PolicySession
        from dexmani_real.teleop.keyboard import KeyboardTeleop

        for cls, module in (
            (PolicySession, "dexmani_real.deployment.session"),
            (KeyboardTeleop, "dexmani_real.teleop.keyboard"),
        ):
            session = cls.__new__(cls)
            session.runtime = self.cfg
            session.runtime_state = SessionState(
                stop_signal=threading.Event(), emergency_signal=threading.Event()
            )
            session.robot = Mock()
            session.episode = None
            session.prepare_jog = Mock()
            session.period_ns = 10_000_000
            for rejected in (False, True):
                session.prepare_jog.reset_mock()
                with patch(
                    module + ".home_robot",
                    side_effect=HomePathRejected("blocked") if rejected else None,
                ) as home:
                    session.run_home(session.runtime_state.cancel_epoch)
                home.assert_called_once()
                if rejected:
                    session.prepare_jog.assert_not_called()

    def test_owner_operation_stops_on_execution_cancellation(self):
        from dexmani_real.robot.xarm7.protocol import ArmChannels

        ctx = mp.get_context("spawn")
        io = ArmChannels(ctx, stop_signal=ctx.Event(), emergency_signal=ctx.Event())
        pipe, driver = Mock(), Mock()
        pipe.poll.return_value = False
        owner = ArmController(self.cfg.arm, io, pipe, driver)
        owner.home = Mock(side_effect=HomeTimeout("late"))
        owner.operation(
            HomeRequest(10**30, np.array([self.q, self.cfg.arm.home_qpos]), 0.2)
        )
        driver.stop.assert_called_once()
        self.assertTrue(io.stopped.is_set())
        self.assertTrue(io.stop.is_set())
        self.assertIsInstance(pipe.send.call_args.args[0], HomeTimeout)

    def test_settle_cleanup_services_feedback_after_cancellation(self):
        state = SessionState(
            stop_signal=threading.Event(), emergency_signal=threading.Event()
        )
        robot = Mock(arm_config=self.cfg.arm, hand_config=self.cfg.hand)
        clock = Clock()
        frame = np.zeros(1, dtype=HAND_STATE_DTYPE)
        frame["timestamp_ns"] = clock.ns
        frame["qpos"] = self.hand_q
        robot.hand_feedback.snapshot.return_value = frame

        def settle(*args, service, **kwargs):
            state.request_stop()
            before = robot.service_feedback.call_count
            service()  # 取消后的收尾必须能继续服务反馈。
            self.assertEqual(robot.service_feedback.call_count, before + 1)
            raise OperationCancelled("operator cancelled")

        robot.arm.settle.side_effect = settle
        with (
            patch("dexmani_real.runtime.home.time.monotonic_ns", clock.now),
            self.assertRaises(OperationCancelled),
        ):
            home_robot(
                state,
                robot,
                environment=self.cfg.environment,
                cancel_epoch=state.cancel_epoch,
                cancel_requested=lambda: False,
            )
        robot.arm.home.assert_not_called()
        self.assertEqual(robot.stop.call_count, 2)

    def test_passive_hand_stop_cannot_admit_arm_home(self):
        state = SessionState(
            stop_signal=threading.Event(), emergency_signal=threading.Event()
        )
        robot = Mock(arm_config=self.cfg.arm, hand_config=self.cfg.hand)
        robot.stop_hand.return_value = False
        clock = Clock()
        frame = np.zeros(1, dtype=HAND_STATE_DTYPE)
        frame["timestamp_ns"] = clock.ns
        frame["qpos"] = self.hand_q
        robot.hand_feedback.snapshot.return_value = frame
        with (
            patch("dexmani_real.runtime.home.time.monotonic_ns", clock.now),
            self.assertRaisesRegex(FeedbackUnavailable, "position hold"),
        ):
            home_robot(
                state,
                robot,
                environment=self.cfg.environment,
                cancel_epoch=state.cancel_epoch,
                cancel_requested=lambda: False,
            )
        robot.arm.settle.assert_not_called()
        robot.arm.home.assert_not_called()
        robot.send_hand_home.assert_not_called()

    def test_home_velocity_and_duration_rejected_before_sampling(self):
        cfg = replace(
            self.cfg.arm, homing=replace(self.cfg.arm.homing, max_speed_deg_s=0.01)
        )
        with self.assertRaisesRegex(ValueError, "initial velocity"):
            calculate_home_segment(cfg, self.q, np.ones(7) * 0.02, cfg.home_qpos)
        model = Mock()
        with self.assertRaises(HomeTimeout):
            plan_home(cfg, self.q, np.zeros(7), self.hand_q, model, lambda: None)
        model.rejection.assert_not_called()

    def test_wait_settled_preserves_home_timeout_and_feedback_category(self):
        clock = Clock()
        io = NS(stop=Mock())
        io.stop.wait.side_effect = clock.sleep
        owner = ArmController(self.cfg.arm, io, None, None)
        owner.read_feedback = Mock()

        def expired():
            raise HomeTimeout("budget exhausted")

        with patch("dexmani_real.robot.xarm7.worker.time.monotonic_ns", clock.now):
            with self.assertRaises(HomeTimeout):
                owner.wait_settled(expired, stage="HOME", deadline_ns=clock.ns)
            with self.assertRaises(FeedbackUnavailable):
                owner.wait_settled(
                    lambda: None, stage="HOME", deadline_ns=clock.ns + 10**9
                )

    def test_xhand_stop_reports_hold_or_passive(self):
        hand = XHand(self.cfg.hand)
        hand.connected_flag = True
        hand.command = object()
        hand._send_position = Mock(return_value=0)
        hand.set_passive = Mock(return_value=0)
        clock = Clock()
        with patch("dexmani_real.robot.xhand.driver.time.monotonic_ns", clock.now):
            self.assertTrue(hand.stop(self.hand_q, timestamp_ns=clock.ns))
            self.assertFalse(hand.stop(self.hand_q, timestamp_ns=0))
        hand._send_position.assert_called_once()
        hand.set_passive.assert_called_once()

    def test_owner_lock_contention_keeps_original_feedback_expiry(self):
        for max_age in (0.2, 0.005):
            with self.subTest(max_age=max_age):
                clock = Clock()
                lock = Mock()
                lock.acquire.side_effect = [True] + [False] * 30
                shared = NS(value=clock.ns)
                stamp = Mock()
                stamp.get_lock.return_value = lock
                stamp.get_obj.return_value = shared
                io = NS(stop=Mock(), home_hand_stamp=stamp)
                io.stop.wait.side_effect = clock.sleep
                driver = Mock()
                owner = ArmController(self.cfg.arm, io, None, driver)
                owner.status["qpos"] = self.q
                owner.status["timestamp_ns"] = clock.ns
                owner.wait_settled = Mock()
                owner.read_feedback = Mock()
                samples = np.array([self.q, self.cfg.arm.home_qpos])
                message = HomeRequest(clock.ns + 10**9, samples, max_age)
                with patch(
                    "dexmani_real.robot.xarm7.worker.time.monotonic_ns", clock.now
                ):
                    if max_age == 0.2:
                        owner.home(
                            message,
                            check_abort=lambda: None,
                            deadline_ns=message.deadline_ns,
                        )
                        self.assertEqual(driver.servo.call_count, 2)
                    else:
                        with self.assertRaises(FeedbackUnavailable):
                            owner.home(
                                message,
                                check_abort=lambda: None,
                                deadline_ns=message.deadline_ns,
                            )
                        self.assertEqual(driver.servo.call_count, 1)

    def test_diagnostic_rejects_invalid_tolerance_before_device_creation(self):
        from scripts.tools.control_xhand import run_hardware_session

        with patch("scripts.tools.control_xhand.XHandDiagnostic") as device:
            for tolerance in (0.0, -1.0, float("nan")):
                with self.assertRaisesRegex(ValueError, "tolerance"):
                    run_hardware_session(
                        replace(self.cfg.hand, home_tolerance_deg=tolerance),
                        finger_id=0,
                        read_only=False,
                    )
            device.assert_not_called()
