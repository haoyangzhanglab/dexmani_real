"""不连接设备，验证腕姿保护在提交和录制之前生效。"""

import math
import unittest
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.teleop import TeleopConfig
from dexmani_real.control.actions import ActionResult
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE
from dexmani_real.runtime.observation import Observation
from dexmani_real.runtime.state import SessionState
from dexmani_real.teleop.loop import TeleopLoop


_START_NS = 1_000_000_000
_PERIOD_NS = 62_500_000


def observation(tick, *, wrist_m=0.0, angle_deg=0.0, sign=1, wrist_ns=None, recv_ns=None):
    control_ns = _START_NS + tick * _PERIOD_NS
    wrist_ns = control_ns - 1_000_000 if wrist_ns is None else wrist_ns
    half_angle = math.radians(angle_deg) / 2
    arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
    arm["timestamp_ns"] = control_ns
    hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
    hand["timestamp_ns"] = control_ns
    return Observation(
        arm=arm,
        hand=hand,
        camera=None,
        vr={
            "connection_epoch": 0,
            "sequence_id": tick + 1,
            "recv_ts_ns": wrist_ns if recv_ns is None else recv_ns,
            "wrist_recv_ts_ns": wrist_ns,
            "wrist_pos": np.array([wrist_m, 0.0, 0.0]),
            "wrist_quat_wxyz": sign * np.array([math.cos(half_angle), math.sin(half_angle), 0.0, 0.0]),
        },
        point_cloud=None,
        observation_timestamp_ns=control_ns,
        control_timestamp_ns=control_ns,
    )


class WristPoseGuardTest(unittest.TestCase):
    def make_loop(self):
        state = SessionState(stop_signal=Event(), emergency_signal=Event())
        runtime = SimpleNamespace(
            arm=SimpleNamespace(feedback_max_age_s=0.5),
            hand=SimpleNamespace(enabled=True, feedback_max_age_s=0.5),
            teleop=TeleopConfig(),
        )
        robot = Mock(hand_feedback_pending=False)
        robot.prepare_streaming.side_effect = lambda *args, **kwargs: state.stop_signal.clear()
        robot.send_action.return_value = _START_NS
        controller = Mock(anchor=None)
        controller.reset.side_effect = lambda row: setattr(controller, "anchor", row)
        controller.clear.side_effect = lambda: setattr(controller, "anchor", None)
        controller.compute_target.return_value = (
            ActionResult(np.zeros(7), np.zeros(12)), np.zeros(7), np.zeros(3)
        )
        keyboard = Mock()
        keyboard.poll.return_value = (), False
        keyboard.accept_operation.return_value = state.cancel_epoch
        return TeleopLoop(
            state,
            runtime,
            robot,
            control_dt_ns=_PERIOD_NS,
            task_label="wrist_guard",
            sensors=SimpleNamespace(channels=SimpleNamespace(vr_connection_epoch=SimpleNamespace(value=0))),
            controller=controller,
            recording_dir=None,
            max_duration_s=None,
            keyboard=keyboard,
            audio=Mock(),
            timing=Mock(),
        )

    def start(self, loop):
        first = observation(0)
        with (
            patch("dexmani_real.teleop.loop.read_observation", return_value=first),
            patch("dexmani_real.teleop.loop.time.monotonic_ns", return_value=first.control_timestamp_ns),
        ):
            self.assertTrue(loop.start_episode(loop.runtime_state.cancel_epoch))
        recorder = Mock(error=None)
        recorder.append.return_value = None
        recorder.finish.return_value = SimpleNamespace(reason="cancelled", save=True)
        recorder.result.return_value = None
        loop.recorder = recorder

    def tick(self, loop, row):
        with (
            patch("dexmani_real.teleop.loop.read_observation", return_value=row),
            patch("dexmani_real.teleop.loop.time.monotonic_ns", return_value=row.control_timestamp_ns),
        ):
            loop.run_control_tick()

    def start_motion(self, loop):
        self.start(loop)
        self.tick(loop, observation(0))
        self.tick(loop, observation(1))
        loop.robot.send_action.assert_called_once()

    def assert_jump_stops(self, loop):
        self.assertTrue(loop.runtime_state.stop_signal.is_set())
        self.assertEqual(loop.pending_termination_reason, "cancelled")
        self.assertIn({"stage": "termination", "reason": "vr_pose_jump"}, loop.termination_details)
        self.assertTrue(any(event.args[0] == "vr_pose_jump" for event in loop.timing.add.call_args_list))

    def test_start_waits_for_new_wrist_receipt_and_allows_static_pose(self):
        loop = self.make_loop()
        self.start(loop)
        wrist_ns = _START_NS - 1_000_000
        self.tick(loop, observation(0, wrist_ns=wrist_ns, recv_ns=wrist_ns - 1))
        # 组帧序号和组合接收时间都更新，腕包仍是同一个。
        self.tick(loop, observation(1, wrist_ns=wrist_ns))
        loop.controller.reset.assert_not_called()
        loop.controller.compute_target.assert_not_called()
        loop.robot.send_action.assert_not_called()
        loop.recorder.append.assert_not_called()

        ready = observation(2)
        self.tick(loop, ready)
        self.tick(loop, observation(3, wrist_ns=ready.vr["wrist_recv_ts_ns"]))
        self.assertIs(loop.controller.reset.call_args.args[0], ready)
        self.assertEqual(loop.robot.send_action.call_count, 2)
        self.assertEqual(loop.recorder.append.call_count, 2)
        self.assertFalse(loop.runtime_state.stop_signal.is_set())

    def test_rotation_uses_shortest_arc_between_quaternions(self):
        loop = self.make_loop()
        self.start(loop)
        self.tick(loop, observation(0, angle_deg=179.0))
        self.tick(loop, observation(1, angle_deg=179.0, sign=-1))
        self.tick(loop, observation(2, angle_deg=-179.0))
        self.assertEqual(loop.robot.send_action.call_count, 2)
        self.assertFalse(loop.runtime_state.stop_signal.is_set())

    def test_recorded_jumps_stop_before_solving_sending_or_recording(self):
        for time_s, wrist_m, angle_deg in (
            (10.50, 0.1232, 34.1),
            (12.75, 0.2118, 73.1),
            (13.125, 0.0318, 88.1),
            (13.69, 0.3317, 28.5),
            (20.75, 0.2062, 53.2),
        ):
            with self.subTest(time_s=time_s):
                loop = self.make_loop()
                self.start_motion(loop)
                self.tick(loop, observation(2, wrist_m=wrist_m, angle_deg=angle_deg))
                self.assert_jump_stops(loop)
                loop.controller.compute_target.assert_called_once()
                loop.robot.send_action.assert_called_once()
                loop.recorder.append.assert_called_once()

    def test_jump_on_first_new_wrist_sample_does_not_set_motion_anchor(self):
        loop = self.make_loop()
        self.start(loop)
        self.tick(loop, observation(0))
        self.tick(loop, observation(1, wrist_m=0.25))
        self.assert_jump_stops(loop)
        loop.controller.reset.assert_not_called()
        loop.robot.send_action.assert_not_called()

    def test_missed_tick_caps_motion_budget(self):
        loop = self.make_loop()
        self.start_motion(loop)
        # 150 mm 低于两拍的位移预算，高于 1.5 拍的预算。
        self.tick(loop, observation(3, wrist_m=0.15))
        self.assert_jump_stops(loop)
        loop.recorder.mark_gap.assert_called_once()
        loop.robot.send_action.assert_called_once()

    def test_same_receive_timestamp_does_not_bypass_jump_check(self):
        loop = self.make_loop()
        self.start_motion(loop)
        # 同一个 TCP 块中的不同腕数据可共享接收时间。
        wrist_ns = observation(1).vr["wrist_recv_ts_ns"]
        self.tick(loop, observation(2, wrist_m=0.25, wrist_ns=wrist_ns))
        self.assert_jump_stops(loop)
        loop.robot.send_action.assert_called_once()

    def test_restart_requires_b_and_builds_a_new_baseline(self):
        loop = self.make_loop()
        self.start_motion(loop)
        self.tick(loop, observation(2, wrist_m=0.25))
        self.assert_jump_stops(loop)
        loop.robot.stop.reset_mock()
        loop.end_episode()
        loop.robot.stop.assert_called_once()
        loop.controller.clear.assert_called_once()
        self.tick(loop, observation(3))
        loop.robot.send_action.assert_called_once()

        baseline = observation(5, wrist_m=0.60)
        ready = observation(6, wrist_m=0.60)
        control_rows = iter([baseline, ready])
        current = baseline

        def advance(_timeout):
            nonlocal current
            row = next(control_rows, None)
            if row is None:
                return False
            current = row
            return True

        with (
            patch.object(loop, "service_episode", side_effect=advance),
            patch("dexmani_real.teleop.loop.read_observation", side_effect=lambda *args, **kwargs: current),
            patch("dexmani_real.teleop.loop.time.monotonic_ns", side_effect=lambda: current.control_timestamp_ns),
        ):
            self.assertTrue(loop.handle_operator_command("b"))
        self.assertIs(loop.controller.reset.call_args.args[0], ready)
        self.assertEqual(loop.robot.send_action.call_count, 2)
