"""HOME 离线契约；这里的阈值仅为合成测试数据，不是设备验收值。"""

import multiprocessing as mp
import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import DEFAULT, Mock, patch

import numpy as np

from dexmani_real.config.control import StaticCollisionBox
from dexmani_real.config.real import RealConfig
from dexmani_real.control.actions import ActionMapper
from dexmani_real.control.collision import CollisionModel
from dexmani_real.robot.commands import (
    FeedbackUnavailable,
    HomeTimeout,
    OperationCancelled,
    RobotCommand,
    RobotError,
)
from dexmani_real.robot.system import Robot
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE, HomeRequest
from dexmani_real.robot.xarm7.trajectory import (
    PositionLimitError,
    calculate_home_segment,
    validate_extrema,
)
from dexmani_real.robot.xarm7.worker import ArmController
from dexmani_real.robot.xhand.driver import (
    COMMUNICATION_CRC_ERROR_CODE, OVERCURRENT_WARNING_CODE, XHand, XHandError,
)
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

    def test_direct_extrema_rejection_checks_both_via_segments(self):
        via = tuple(self.q + 0.05)
        cfg = replace(self.cfg.arm, homing=replace(self.cfg.arm.homing, via_qpos=via))
        model = Mock(arm_radii=np.zeros(7))
        model.rejection.return_value = None
        with patch(
            "dexmani_real.runtime.home.validate_extrema", wraps=validate_extrema,
            side_effect=[PositionLimitError("direct extremum"), DEFAULT, DEFAULT],
        ) as extrema:
            samples = plan_home(cfg, self.q, np.zeros(7), self.hand_q, model, lambda: None)
        self.assertEqual(extrema.call_count, 3)
        self.assertTrue(any(np.array_equal(q, via) for q in samples))
        checked = [call.args[0] for call in model.rejection.call_args_list]
        self.assertTrue(all(any(np.array_equal(q, c) for c in checked) for q in samples))

    def test_ruckig_errors_do_not_try_via(self):
        from ruckig import Result, RuckigError

        via = tuple(self.q + 0.05)
        cfg = replace(self.cfg.arm, homing=replace(self.cfg.arm.homing, via_qpos=via))
        model = Mock(arm_radii=np.zeros(7))
        model.rejection.return_value = None
        for result in (
            Result.ErrorPositionalLimits, Result.Error, Result.ErrorInvalidInput,
            Result.ErrorExecutionTimeCalculation, Result.ErrorSynchronizationCalculation,
            RuckigError("calculation failed"), RuntimeError("program error"),
        ):
            with self.subTest(result=result):
                calculator = Mock()
                calculator.calculate.side_effect = [result]
                with patch("dexmani_real.robot.xarm7.trajectory.Ruckig", return_value=calculator):
                    error = type(result) if isinstance(result, Exception) else RuntimeError
                    with self.assertRaises(error):
                        plan_home(cfg, self.q, np.zeros(7), self.hand_q, model, lambda: None)
                    self.assertEqual(calculator.calculate.call_count, 1)

    def test_shared_home_inputs_rejected_before_candidates(self):
        cfg = replace(self.cfg.arm, homing=replace(self.cfg.arm.homing, via_qpos=tuple(self.q + 0.05)))
        bad_configs = (
            replace(cfg, home_qpos=(float("nan"),) * 7),
            replace(cfg, home_qpos=(100.0,) * 7),
            replace(cfg, homing=replace(cfg.homing, via_qpos=(float("nan"),) * 7)),
            replace(cfg, joint_limit_lower=cfg.joint_limit_upper),
            replace(cfg, joint_limit_upper=(float("nan"),) * 7),
            replace(cfg, servo_hz=0),
            replace(cfg, homing=replace(cfg.homing, max_speed_deg_s=0)),
            replace(cfg, homing=replace(cfg.homing, max_acceleration_deg_per_s2=0)),
            replace(cfg, homing=replace(cfg.homing, max_jerk_deg_per_s3=0)),
        )
        inputs = [(bad, self.q, np.zeros(7), HomeConfigurationError) for bad in bad_configs]
        inputs += [
            (cfg, q, v, ValueError) for q, v in (
                (np.full(7, np.nan), np.zeros(7)), (np.zeros(6), np.zeros(7)),
                (np.full(7, 100.0), np.zeros(7)), (self.q, np.full(7, np.nan)),
                (self.q, np.zeros(6)), (self.q, np.full(7, cfg.homing.max_speed_rad_per_s + 0.01)),
            )
        ]
        for bad, q, v, error in inputs:
            with self.subTest(cfg=bad, q=q, v=v):
                model = Mock()
                with patch("dexmani_real.runtime.home.calculate_home_segment") as calculate:
                    with self.assertRaises(error):
                        plan_home(bad, q, v, self.hand_q, model, lambda: None)
                    calculate.assert_not_called()
                model.rejection.assert_not_called()

    def test_planning_cancel_and_timeout_do_not_try_via(self):
        cfg = replace(self.cfg.arm, homing=replace(self.cfg.arm.homing, via_qpos=tuple(self.q + 0.05)))
        for error in (OperationCancelled("cancelled"), HomeTimeout("budget exhausted")):
            with self.subTest(error=error):
                check = Mock(side_effect=[None, None, error])
                with patch(
                    "dexmani_real.runtime.home.calculate_home_segment", wraps=calculate_home_segment
                ) as calculate:
                    with self.assertRaises(type(error)):
                        plan_home(cfg, self.q, np.zeros(7), self.hand_q, Mock(), check)
                    self.assertEqual(calculate.call_count, 1)

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
                robot.hand.feedback_overcurrent = (clock.ns, False, ())
                fresh_after = clock.ns + 33_333_333

        def send(target, **kw):
            self.assertLessEqual(np.max(np.abs(target - q)), np.deg2rad(1) + 1e-12)
            if writes:
                self.assertGreater(int(frame["timestamp_ns"][0]), writes[-1][0])
            writes.append((clock.ns, target.copy()))
            q[:] = target
            return False

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
        robot.hand.feedback_overcurrent = (clock.ns, False, ())
        robot.send_hand_home.return_value = False
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

    def make_hand_sdk(self):
        hand = XHand(self.cfg.hand)
        hand.connected_flag = True
        hand.control = Mock()
        hand.command = NS(finger_command=[NS() for _ in range(12)])
        raw = NS(finger_state=[NS(id=i, position=q, commboard_err=0,
                                 jonitboard_err=0, tipboard_err=0)
                               for i, q in enumerate(self.hand_q)])
        hand.control.read_state.return_value = (NS(error_code=0, error_message=""), raw)
        hand.control.send_command.return_value = NS(error_code=0, error_message="")
        hand.warn_limited = Mock()
        return hand, raw

    def test_hand_home_overcurrent_cycles(self):
        # 每个字符是一轮采样机会：R 全局读告警、B 关节板告警、W 写告警、
        # X 读写同时告警、C 读 CRC、D 不产生新反馈、N 正常。
        for events, failed, count in (
            ("RNNNN", False, 3), ("WNNNN", False, 3),
            ("RRNRRNNNN", False, 7), ("RRR", True, 2),
            ("WWW", True, 3), ("XXNNNN", False, 4),
            ("RWR", True, 2), ("WRW", True, 3),
            ("RDDNNNN", False, 3), ("RRCR", True, 2),
            ("BBB", True, 2), ("NNWWW", True, 5),
        ):
            with self.subTest(events=events):
                clock = Clock()
                hand, raw = self.make_hand_sdk()
                state = SessionState(stop_signal=threading.Event(), emergency_signal=threading.Event())
                robot = Robot(self.cfg.arm, self.cfg.hand, runtime_state=state)
                robot.connected, robot.hand = True, hand
                sequence = iter(events)
                writes = []
                event = "N"

                def service():
                    nonlocal event
                    event = next(sequence, "N")
                    if event == "D":
                        return
                    code = (COMMUNICATION_CRC_ERROR_CODE if event == "C" else
                            OVERCURRENT_WARNING_CODE if event in "RX" else 0)
                    raw.finger_state[2].jonitboard_err = 110 if event == "B" else 0
                    hand.control.read_state.return_value = (NS(error_code=code, error_message="read"), raw)
                    robot.read_hand()

                def send(*args):
                    writes.append([joint.position for joint in hand.command.finger_command])
                    return NS(error_code=OVERCURRENT_WARNING_CODE if event in "WX" else 0,
                              error_message="send")

                robot.service_feedback = service
                hand.control.send_command.side_effect = send
                with patch("time.monotonic_ns", clock.now), patch("time.sleep", clock.sleep):
                    robot.read_hand()
                    if failed:
                        with self.assertLogs("dexmani_real.runtime.home", level="ERROR") as logs:
                            with self.assertRaises(XHandError) as caught:
                                _home_hand(robot, lambda: None)
                        self.assertEqual(caught.exception.code, OVERCURRENT_WARNING_CODE)
                        self.assertIn("consecutive=3", str(caught.exception))
                        self.assertIn("source=", str(caught.exception))
                        if events == "BBB":
                            self.assertIn("joints=[2]", str(caught.exception))
                        self.assertEqual(len(logs.records), 1)
                    else:
                        _home_hand(robot, lambda: None)
                self.assertEqual(len(writes), count)
                np.testing.assert_array_equal(writes, np.tile(self.hand_q, (count, 1)))

    def test_hand_warning_metadata_only_changes_with_valid_feedback(self):
        clock = Clock()
        hand, raw = self.make_hand_sdk()
        raw.finger_state[4].jonitboard_err = 110
        with patch("time.monotonic_ns", clock.now):
            sample = hand.get_state()
            expected = (sample.timestamp_ns, True, (4,))
            self.assertEqual(hand.feedback_overcurrent, expected)
            with self.assertRaises(AttributeError):
                hand.feedback_overcurrent = expected
            clock.sleep(0.01)
            hand.control.read_state.return_value = (NS(error_code=COMMUNICATION_CRC_ERROR_CODE), None)
            self.assertIsNone(hand.get_state())
            self.assertEqual(hand.feedback_overcurrent, expected)
            for code, payload in ((999, raw), (0, NS(finger_state=[]))):
                hand.control.read_state.return_value = (NS(error_code=code, error_message="fault"), payload)
                with self.assertRaises(XHandError):
                    hand.get_state()
                self.assertEqual(hand.feedback_overcurrent, expected)
            raw.finger_state[4].jonitboard_err = 0
            for field in ("commboard_err", "jonitboard_err"):
                setattr(raw.finger_state[4], field, 99)
                hand.control.read_state.return_value = (NS(error_code=0, error_message=""), raw)
                with self.assertRaises(XHandError):
                    hand.get_state()
                self.assertEqual(hand.feedback_overcurrent, expected)
                setattr(raw.finger_state[4], field, 0)
            sample = hand.get_state()
            self.assertEqual(hand.feedback_overcurrent, (sample.timestamp_ns, False, ()))

    def test_hand_home_metadata_must_match_sample(self):
        clock = Clock()
        robot = Mock(hand_config=self.cfg.hand)
        frame = np.zeros(1, dtype=HAND_STATE_DTYPE)
        frame["timestamp_ns"] = clock.ns
        robot.hand_feedback.snapshot.return_value = frame
        robot.hand.feedback_overcurrent = (clock.ns - 1, False, ())
        with patch("time.monotonic_ns", clock.now), self.assertRaises(FeedbackUnavailable):
            _home_hand(robot, lambda: None)
        robot.send_hand_home.assert_not_called()

    def test_home_write_warning_return_and_normal_control_semantics(self):
        clock = Clock()
        hand, raw = self.make_hand_sdk()
        state = SessionState(stop_signal=threading.Event(), emergency_signal=threading.Event())
        robot = Robot(self.cfg.arm, self.cfg.hand, runtime_state=state)
        robot.connected, robot.hand = True, hand
        with patch("time.monotonic_ns", clock.now):
            robot.read_hand()
            for code in (OVERCURRENT_WARNING_CODE, 0, 999):
                hand.control.send_command.return_value = NS(error_code=code, error_message="fault")
                if code == 999:
                    with self.assertRaises(XHandError):
                        robot.send_hand_home(self.hand_q, deadline_ns=clock.ns + 10**9, check_abort=lambda: None)
                else:
                    self.assertIs(robot.send_hand_home(self.hand_q, deadline_ns=clock.ns + 10**9,
                                                      check_abort=lambda: None), code == OVERCURRENT_WARNING_CODE)
            hand.control.read_state.return_value = (NS(error_code=OVERCURRENT_WARNING_CODE, error_message=""), raw)
            robot.read_hand()
            hand.control.send_command.return_value = NS(error_code=OVERCURRENT_WARNING_CODE, error_message="")
            # Teleop / Policy / Jog 的共享正常发送边界不使用 HOME 消抖。
            for _ in range(4):
                robot.send_action(RobotCommand(hand_qpos=self.hand_q), expires_ns=clock.ns + 10**9)
            self.assertFalse(state.stop_signal.is_set())

    def test_overcurrent_combined_home_stops_and_preserves_cleanup_failure(self):
        for stop_error in (None, RobotError("stop failed")):
            with self.subTest(stop_error=stop_error):
                clock = Clock()
                state = SessionState(stop_signal=threading.Event(), emergency_signal=threading.Event())
                robot = Mock(arm_config=self.cfg.arm, hand_config=self.cfg.hand)
                robot.home_collision_model = None
                robot.stop.side_effect = [None, stop_error]
                robot.send_hand_home.return_value = True
                arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
                arm["qpos"] = self.cfg.arm.home_qpos
                hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
                hand["qpos"] = self.hand_q
                robot.arm.snapshot.return_value = arm
                robot.hand_feedback.snapshot.return_value = hand

                def service():
                    arm["timestamp_ns"] = hand["timestamp_ns"] = clock.ns
                    robot.hand.feedback_overcurrent = (clock.ns, False, ())

                robot.service_feedback.side_effect = service
                with (patch("time.monotonic_ns", clock.now), patch("time.sleep", clock.sleep),
                      patch("dexmani_real.runtime.home.CollisionModel") as model,
                      self.assertLogs("dexmani_real.runtime.home", level="ERROR")):
                    model.return_value.rejection.return_value = None
                    with self.assertRaises(type(stop_error) if stop_error else XHandError) as caught:
                        home_robot(state, robot, environment=self.cfg.environment,
                                   cancel_epoch=state.cancel_epoch, cancel_requested=lambda: False)
                self.assertEqual(robot.stop.call_count, 2)
                self.assertEqual(robot.send_hand_home.call_count, 3)
                if stop_error:
                    self.assertIs(caught.exception, stop_error)
                    self.assertIsInstance(caught.exception.__context__, XHandError)

    def test_owner_start_requires_both_tolerances(self):
        # 二进制精确值保证等于阈值的用例不受浮点相减舍入影响。
        for standstill, tracking, offset, accepted in (
            (0.125, 0.25, 0.0625, True), (0.25, 0.125, 0.0625, True),
            (0.25, 0.125, 0.1875, False), (0.125, 0.25, 0.1875, False),
            (0.125, 0.25, 0.125, True), (0.25, 0.125, 0.125, True),
        ):
            with self.subTest(standstill=standstill, tracking=tracking, offset=offset):
                clock = Clock()
                cfg = replace(self.cfg.arm, standstill_position_tolerance_rad=standstill,
                              homing=replace(self.cfg.arm.homing, tracking_error_rad=tracking))
                samples = np.tile(cfg.home_qpos, (6, 1))
                samples[0, 0] = 0
                io = NS(stop=Mock(), home_hand_stamp=mp.get_context("spawn").Value("q", clock.ns))
                io.stop.wait.side_effect = clock.sleep
                driver = Mock()
                owner = ArmController(cfg, io, None, driver)
                owner.status["qpos"] = samples[0]
                # 模拟 prepare / settle 后实际位置的变化。
                def settled(*args, **kwargs):
                    owner.status["qpos"][0] = offset
                owner.wait_settled = Mock(side_effect=settled)
                def feedback():
                    self.assertEqual(driver.servo.call_count, 4)
                    owner.status["qpos"] = samples[3]
                    owner.status["timestamp_ns"] = clock.ns
                owner.read_feedback = Mock(side_effect=feedback)
                message = HomeRequest(clock.ns + 10**9, samples, 0.2)
                with patch("time.monotonic_ns", clock.now):
                    if accepted:
                        owner.home(message, check_abort=lambda: None, deadline_ns=message.deadline_ns)
                        self.assertEqual(driver.servo.call_count, len(samples))
                        owner.read_feedback.assert_called_once()
                    else:
                        with self.assertRaisesRegex(OperationCancelled, "collision-checked trajectory"):
                            owner.home(message, check_abort=lambda: None, deadline_ns=message.deadline_ns)
                        driver.servo.assert_not_called()
                        owner.read_feedback.assert_not_called()

    def test_owner_rejects_invalid_references_before_motion(self):
        valid = np.array([self.q, self.q, self.cfg.arm.home_qpos])
        cases = {
            "scalar": np.array(0.0),
            "one_dimension": valid[0],
            "three_dimensions": valid[:, None, :],
            "six_joints": valid[:, :6],
            "eight_joints": np.zeros((3, 8)),
            "empty": valid[:0],
            "one_point": valid[-1:],
        }
        for name, joint, value in (
            ("nan", 0, np.nan),
            ("inf", 3, np.inf),
            ("negative_inf", 6, -np.inf),
            ("below_lower", 0, np.nextafter(self.cfg.arm.joint_limit_lower[0], -np.inf)),
            ("above_upper", 6, np.nextafter(self.cfg.arm.joint_limit_upper[6], np.inf)),
        ):
            samples = valid.copy()
            samples[1, joint] = value
            cases[name] = samples
        cases["end_not_home"] = valid.copy()
        cases["end_not_home"][-1, 0] = np.nextafter(valid[-1, 0], np.inf)

        for name, samples in cases.items():
            with self.subTest(case=name):
                driver = Mock()
                owner = ArmController(self.cfg.arm, None, None, driver)
                owner.wait_settled = Mock()
                check_abort = Mock()
                message = HomeRequest(10**10, samples, 0.2)
                with self.assertRaises(ValueError):
                    owner.home(message, check_abort=check_abort, deadline_ns=message.deadline_ns)
                driver.prepare_motion.assert_not_called()
                driver.servo.assert_not_called()
                owner.wait_settled.assert_not_called()
                check_abort.assert_not_called()

    def test_owner_sends_identical_samples_and_stops_on_deviation(self):
        clock = Clock()
        samples = np.linspace(self.q, self.cfg.arm.home_qpos, 6)
        original = samples.copy()
        cfg = replace(
            self.cfg.arm,
            joint_limit_lower=tuple(samples.min(axis=0)),
            joint_limit_upper=tuple(samples.max(axis=0)),
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
                owner = ArmController(cfg, io, None, driver)
                owner.status["qpos"] = samples[0] + (0.1 if failure == "start" else 0)
                owner.wait_settled = Mock()

                def read(owner=owner, failure=failure):
                    index = min(5, max(0, (clock.ns - 10**10) // 10_000_000 - 1))
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
                            [c.args[0] for c in driver.servo.call_args_list], original
                        )
                        np.testing.assert_array_equal(samples, original)
                        for call in driver.servo.call_args_list:
                            self.assertEqual(call.args[0].dtype, samples.dtype)
                            self.assertTrue(np.shares_memory(call.args[0], samples))
                if failure in ("start", "hand"):
                    driver.servo.assert_not_called()
                elif failure == "tracking":
                    np.testing.assert_array_equal(
                        [call.args[0] for call in driver.servo.call_args_list], samples[:4]
                    )

    def test_tracking_at_feedback_boundary_uses_last_successful_submission(self):
        cfg = replace(self.cfg.arm, homing=replace(self.cfg.arm.homing, tracking_error_rad=0.01))
        samples = np.tile(cfg.home_qpos, (6, 1))
        samples[:, 0] += np.arange(-5, 1) * 0.02
        for sdk_failure in (False, True):
            with self.subTest(sdk_failure=sdk_failure):
                clock = Clock()
                start = clock.ns
                io = NS(stop=Mock(), home_hand_stamp=mp.get_context("spawn").Value("q", start))
                io.stop.wait.side_effect = clock.sleep
                driver = Mock()
                owner = ArmController(cfg, io, None, driver)
                owner.status["qpos"] = samples[0]
                owner.wait_settled = Mock()
                sent = []

                def servo(point, sdk_failure=sdk_failure, clock=clock, start=start, sent=sent):
                    if sdk_failure and clock.ns == start + 40_000_000:
                        raise RobotError("servo failed")
                    sent.append(point.copy())

                def feedback(owner=owner, clock=clock, sent=sent):
                    # 100 Hz 发送、25 Hz 查询：40 ms 反馈仍对应 samples[3]。
                    self.assertEqual(clock.ns, start + 40_000_000)
                    self.assertEqual(len(sent), 4)
                    owner.status["timestamp_ns"] = clock.ns
                    owner.status["qpos"] = sent[-1]

                driver.servo.side_effect = servo
                owner.read_feedback = Mock(side_effect=feedback)
                message = HomeRequest(start + 10**9, samples, 0.2)
                with patch("dexmani_real.robot.xarm7.worker.time.monotonic_ns", clock.now):
                    if sdk_failure:
                        with self.assertRaisesRegex(RobotError, "servo failed"):
                            owner.home(message, check_abort=lambda: None, deadline_ns=message.deadline_ns)
                        # 失败不推进已提交序列，也不再检查后续参考或执行终点停稳。
                        np.testing.assert_array_equal(sent, samples[:4])
                        self.assertEqual(driver.servo.call_count, 5)
                        owner.wait_settled.assert_called_once()
                    else:
                        owner.home(message, check_abort=lambda: None, deadline_ns=message.deadline_ns)
                        np.testing.assert_array_equal(sent, samples)
                    self.assertEqual(owner.read_feedback.call_count, 1)

    def test_configuration_rejection_stops_without_motion(self):
        for cfg in (
            RealConfig().arm,
            replace(self.cfg.arm, servo_hz=0),
            replace(self.cfg.arm, joint_limit_upper=(float("nan"),) * 7),
        ):
            robot = Mock(arm_config=cfg, hand_config=self.cfg.hand)
            state = SessionState(
                stop_signal=threading.Event(), emergency_signal=threading.Event()
            )
            with self.subTest(cfg=cfg), self.assertRaises(HomeConfigurationError):
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
            self.assertEqual(robot.stop.call_count, 2)

    def test_combined_order_rejection_skip_and_execution_failure(self):
        for scenario in (
            "success",
            "rejected",
            "execution_error",
            "stop_error",
            "already_home",
            "cancel",
        ):
            with self.subTest(scenario=scenario):
                clock = Clock()
                state = SessionState(
                    stop_signal=threading.Event(), emergency_signal=threading.Event()
                )
                cfg = replace(self.cfg.arm, homing=replace(self.cfg.arm.homing, via_qpos=tuple(self.q + 0.05)))
                robot = Mock(arm_config=cfg, hand_config=self.cfg.hand)
                robot.home_collision_model = None
                if scenario == "stop_error":
                    robot.stop.side_effect = [None, RobotError("stop failed")]
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
                error = HomePathRejected("blocked") if scenario in ("rejected", "stop_error") else None
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
                        error_type = RobotError if scenario == "stop_error" else (HomePathRejected, RuntimeError)
                        with self.assertRaises(error_type):
                            run()
                        self.assertNotIn("hand", calls)
                    self.assertLessEqual(plan.call_count, 1)
                self.assertEqual(robot.stop.call_count, 2)

    def test_real_hand_submission_crc_is_fatal(self):
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

    def test_home_collision_model_is_lazy_and_session_local(self):
        clock = Clock()
        state = SessionState(stop_signal=threading.Event(), emergency_signal=threading.Event())
        arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
        hand = np.zeros(1, dtype=HAND_STATE_DTYPE)

        def feedback():
            arm["timestamp_ns"] = hand["timestamp_ns"] = clock.ns

        def execute(*args, **kwargs):
            arm["qpos"] = self.cfg.arm.home_qpos

        models = [Mock(arm_radii=np.ones(7)), Mock(arm_radii=np.ones(7))]
        for model in models:
            model.rejection.return_value = None
        with (patch("time.monotonic_ns", clock.now), patch("time.sleep", clock.sleep),
              patch("dexmani_real.runtime.home.CollisionModel", side_effect=models) as construct,
              patch("dexmani_real.runtime.home.plan_home", wraps=plan_home) as plan,
              patch("dexmani_real.runtime.home._home_hand")):
            for session in range(2):
                robot = Robot(self.cfg.arm, self.cfg.hand, runtime_state=state)
                self.assertIsNone(robot.home_collision_model)
                robot.arm = Mock()
                robot.arm.snapshot.return_value = arm
                robot.arm.home.side_effect = execute
                robot.hand_feedback.snapshot = Mock(return_value=hand)
                robot.stop = Mock()
                robot.stop_hand = Mock(return_value=True)
                robot.service_feedback = feedback
                for offset in (0.05, 0.08):
                    arm["qpos"] = np.asarray(self.cfg.arm.home_qpos) + offset
                    hand["qpos"] = self.hand_q + offset
                    q, hand_q = arm["qpos"][0].copy(), hand["qpos"][0].copy()
                    model = models[session]
                    model.rejection.reset_mock()
                    home_robot(state, robot, environment=self.cfg.environment,
                               cancel_epoch=state.cancel_epoch, cancel_requested=lambda: False)
                    self.assertEqual(construct.call_count, session + 1)
                    self.assertIs(robot.home_collision_model, model)
                    np.testing.assert_array_equal(plan.call_args.args[1], q)
                    np.testing.assert_array_equal(plan.call_args.args[3], hand_q)
                    samples = robot.arm.home.call_args.kwargs["samples"]
                    np.testing.assert_array_equal(samples[0], q)
                    np.testing.assert_array_equal(samples[-1], self.cfg.arm.home_qpos)
                    self.assertGreaterEqual(model.rejection.call_count, len(samples))
                    for call, point in zip(model.rejection.call_args_list, samples):
                        np.testing.assert_array_equal(call.args[0], point)
                        np.testing.assert_array_equal(call.args[1], hand_q)
                self.assertEqual(robot.arm.home.call_count, 2)
            self.assertEqual(plan.call_count, 4)

    def test_home_collision_construction_failure_keeps_reference_empty(self):
        clock = Clock()
        state = SessionState(stop_signal=threading.Event(), emergency_signal=threading.Event())
        robot = Robot(self.cfg.arm, self.cfg.hand, runtime_state=state)
        robot.arm = Mock()
        robot.stop = Mock()
        robot.stop_hand = Mock(return_value=True)
        arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
        hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
        arm["qpos"] = self.cfg.arm.home_qpos
        hand["qpos"] = self.hand_q
        robot.arm.snapshot.return_value = arm
        robot.hand_feedback.snapshot = Mock(return_value=hand)
        def feedback():
            arm["timestamp_ns"] = hand["timestamp_ns"] = clock.ns
        robot.service_feedback = feedback
        with (patch("time.monotonic_ns", clock.now), patch("time.sleep", clock.sleep),
              patch("dexmani_real.runtime.home.CollisionModel", side_effect=RuntimeError("model failed"))):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                home_robot(state, robot, environment=self.cfg.environment,
                           cancel_epoch=state.cancel_epoch, cancel_requested=lambda: False)
        self.assertIsNone(robot.home_collision_model)
        self.assertEqual(robot.stop.call_count, 2)
        robot.arm.home.assert_not_called()

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

    def test_policy_stays_idle_and_keyboard_prepares_hold_after_path_rejection(self):
        from dexmani_real.deployment.session import PolicySession
        from dexmani_real.teleop.keyboard import KeyboardTeleop

        for cls, module in (
            (PolicySession, "dexmani_real.deployment.session"),
            (KeyboardTeleop, "dexmani_real.teleop.keyboard"),
        ):
            for error in (None, HomePathRejected("blocked"), HomeConfigurationError("config")):
                with self.subTest(cls=cls, error=error):
                    session = cls.__new__(cls)
                    session.runtime = self.cfg
                    state = session.runtime_state = SessionState(
                        stop_signal=threading.Event(), emergency_signal=threading.Event()
                    )
                    session.robot = Mock()
                    session.robot.prepare_streaming.side_effect = lambda epoch, state=state: state.stop_signal.clear()
                    session.episode = None
                    session.jog = Mock()
                    session.jog_config = self.cfg.keyboard_teleop
                    session.period_ns = 10_000_000
                    session.tick_due_ns = 0
                    session.prepare_jog = Mock(
                        wraps=session.prepare_jog if cls is KeyboardTeleop else None
                    )

                    def home_result(*args, state=state, error=error, **kwargs):
                        state.stop_signal.set()  # 模拟 home_robot 的停止收尾。
                        if error is not None:
                            raise error

                    with (
                        patch(module + ".home_robot", side_effect=home_result) as home,
                        patch(module + ".time.monotonic_ns", return_value=10**10),
                    ):
                        session.run_home(state.cancel_epoch)
                    home.assert_called_once()
                    session.jog.step.assert_not_called()
                    self.assertIsNone(session.episode)
                    if cls is KeyboardTeleop and not isinstance(error, HomeConfigurationError):
                        session.prepare_jog.assert_called_once_with(state.cancel_epoch)
                        session.jog.clear_reference.assert_called_once()
                        session.robot.prepare_streaming.assert_called_once_with(state.cancel_epoch)
                        session.robot.send_action.assert_called_once()
                        command = session.robot.send_action.call_args.args[0]
                        self.assertTrue(command.arm_hold)
                        self.assertIsNone(command.arm_qpos)
                        self.assertIsNone(command.hand_qpos)
                        self.assertFalse(state.stop_signal.is_set())
                        self.assertEqual(session.tick_due_ns, 10**10 + session.period_ns)
                    else:
                        session.prepare_jog.assert_not_called()
                        session.robot.prepare_streaming.assert_not_called()
                        session.robot.send_action.assert_not_called()
                        self.assertTrue(state.stop_signal.is_set())

    def test_keyboard_rejection_does_not_override_cancel_estop_or_quit(self):
        from dexmani_real.teleop.keyboard import KeyboardTeleop

        for request in ("request_stop", "request_emergency_stop", "request_quit"):
            with self.subTest(request=request):
                session = KeyboardTeleop(self.cfg)
                state = session.runtime_state = SessionState(
                    stop_signal=threading.Event(), emergency_signal=threading.Event()
                )
                session.robot = Mock()
                session.prepare_jog = Mock()

                def rejected(*args, state=state, request=request, **kwargs):
                    getattr(state, request)()
                    raise HomePathRejected("blocked")

                with patch("dexmani_real.teleop.keyboard.home_robot", side_effect=rejected):
                    session.run_home(state.cancel_epoch)
                session.prepare_jog.assert_not_called()
                self.assertTrue(state.stop_signal.is_set())

    def test_keyboard_execution_errors_do_not_prepare_jog(self):
        from dexmani_real.teleop.keyboard import KeyboardTeleop

        for error in (
            OperationCancelled("cancelled"), HomeTimeout("timed out"),
            FeedbackUnavailable("stale"), RobotError("SDK failed"), RobotError("stop failed"),
            XHandError("HOME", OVERCURRENT_WARNING_CODE, "persistent overcurrent"),
        ):
            with self.subTest(error=error):
                session = KeyboardTeleop(self.cfg)
                session.runtime_state = SessionState(
                    stop_signal=threading.Event(), emergency_signal=threading.Event()
                )
                session.robot = Mock()
                session.prepare_jog = Mock()
                with (
                    patch("dexmani_real.teleop.keyboard.home_robot", side_effect=error),
                    self.assertRaises(type(error)),
                ):
                    session.run_home(session.runtime_state.cancel_epoch)
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
            self.cfg.arm, homing=replace(self.cfg.arm.homing, max_speed_deg_s=0.01,
                                        via_qpos=tuple(self.q + 0.05))
        )
        with self.assertRaisesRegex(ValueError, "initial velocity"):
            plan_home(cfg, self.q, np.ones(7) * 0.02, self.hand_q, Mock(), lambda: None)
        model = Mock()
        with patch("dexmani_real.runtime.home.calculate_home_segment", wraps=calculate_home_segment) as calculate:
            with self.assertRaises(HomeTimeout):
                plan_home(cfg, self.q, np.zeros(7), self.hand_q, model, lambda: None)
            self.assertEqual(calculate.call_count, 1)
        model.rejection.assert_not_called()

    def test_start_and_target_settling_require_continuous_position_stability(self):
        # 每 40 ms 反馈；0.30 s dwell 最早在 0.32 s 确认。
        for target in (None, np.asarray(self.cfg.arm.home_qpos)):
            for name, offsets, speeds, finish_s in (
                ("stationary", [0], [0], 0.32),
                ("inside_tolerance", [-0.75], [0], 0.32),
                ("anchor_drift", [-0.75] * 4 + [0.75], [0], 0.48),
                ("velocity_reset", [0], [0] * 4 + [2, 0], 0.52),
                ("oscillation", [-0.75, 0.75] * 8, [0], None),
            ):
                with self.subTest(target=target, name=name):
                    clock = Clock()
                    start = clock.ns
                    io = NS(stop=Mock())
                    io.stop.wait.side_effect = clock.sleep
                    owner = ArmController(self.cfg.arm, io, None, None)

                    def feedback(clock=clock, start=start, owner=owner, offsets=offsets, speeds=speeds):
                        index = (clock.ns - start) // 40_000_000
                        owner.status["timestamp_ns"] = clock.ns
                        owner.status["qpos"] = self.cfg.arm.home_qpos
                        owner.status["qpos"][0] += (
                            offsets[min(index, len(offsets) - 1)]
                            * self.cfg.arm.standstill_position_tolerance_rad
                        )
                        owner.status["qvel"] = (
                            speeds[min(index, len(speeds) - 1)]
                            * self.cfg.arm.standstill_velocity_tolerance_rad_s
                        )

                    owner.read_feedback = feedback
                    with patch("dexmani_real.robot.xarm7.worker.time.monotonic_ns", clock.now):
                        if finish_s is None:
                            with self.assertRaisesRegex(TimeoutError, "dwell not confirmed"):
                                owner.wait_settled(
                                    lambda: None, stage="HOME", deadline_ns=start + 600_000_000, target=target
                                )
                        else:
                            owner.wait_settled(
                                lambda: None, stage="HOME", deadline_ns=start + 600_000_000, target=target
                            )
                            self.assertEqual(clock.ns - start, round(finish_s * 1e9))

    def test_target_departure_resets_settling_even_within_anchor_tolerance(self):
        clock = Clock()
        start = clock.ns
        io = NS(stop=Mock())
        io.stop.wait.side_effect = clock.sleep
        owner = ArmController(self.cfg.arm, io, None, None)

        def feedback():
            owner.status["timestamp_ns"] = clock.ns
            owner.status["qpos"] = self.cfg.arm.home_qpos
            # 相对 anchor 只变化 0.5 倍容差，但 160 ms 时已离开目标容差。
            offset = 1.25 if clock.ns - start == 160_000_000 else 0.75
            owner.status["qpos"][0] += offset * self.cfg.arm.standstill_position_tolerance_rad

        owner.read_feedback = feedback
        with patch("dexmani_real.robot.xarm7.worker.time.monotonic_ns", clock.now):
            owner.wait_settled(lambda: None, stage="HOME", deadline_ns=start + 10**9,
                               target=np.asarray(self.cfg.arm.home_qpos))
        self.assertEqual(clock.ns - start, 520_000_000)

    def test_wait_settled_preserves_home_timeout_and_feedback_category(self):
        clock = Clock()
        io = NS(stop=Mock())
        io.stop.wait.side_effect = clock.sleep
        owner = ArmController(self.cfg.arm, io, None, None)
        owner.read_feedback = Mock()
        owner.status["timestamp_ns"] = clock.ns - round(self.cfg.arm.feedback_max_age_s * 1e9) - 1

        def expired():
            raise HomeTimeout("budget exhausted")

        with patch("dexmani_real.robot.xarm7.worker.time.monotonic_ns", clock.now):
            for target in (None, np.asarray(self.cfg.arm.home_qpos)):
                with self.assertRaises(HomeTimeout):
                    owner.wait_settled(expired, stage="HOME", deadline_ns=clock.ns, target=target)
                with self.assertRaises(FeedbackUnavailable):
                    owner.wait_settled(
                        lambda: None, stage="HOME", deadline_ns=clock.ns + 10**9, target=target
                    )
                with self.assertRaisesRegex(OperationCancelled, "operator cancelled"):
                    owner.wait_settled(
                        Mock(side_effect=OperationCancelled("operator cancelled")),
                        stage="HOME", deadline_ns=clock.ns + 10**9, target=target,
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
