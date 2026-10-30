"""启动标定只使用 VR/时钟/设备替身；不验证 Ruckig 或实际设备启停。"""

import importlib
import sys
import types
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_hand_retargeting import make_retargeter

from dexmani_real.config.real import RealConfig

# run.py 的机械臂进程在导入时依赖 Ruckig，本组不调用机械臂轨迹代码。
# 手部数值后端在 test_hand_retargeting 中使用真实 Pinocchio/NLopt。
ruckig_stub = types.ModuleType("ruckig")
for name in ("ControlInterface", "InputParameter", "OutputParameter", "Result", "Ruckig", "Synchronization"):
    setattr(ruckig_stub, name, object)
with patch.dict(sys.modules, {"ruckig": ruckig_stub}):
    teleop_run = importlib.import_module("dexmani_real.teleop.run")


class Clock:
    def __init__(self):
        self.seconds = 1.0

    def monotonic(self):
        return self.seconds

    def monotonic_ns(self):
        return round(self.seconds * 1e9)

    def sleep(self, duration):
        self.seconds += duration


class CalibrationStartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, _, cls.points = make_retargeter()

    def loop(self):
        runtime = RealConfig()
        runtime = replace(
            runtime, hand_retargeting=replace(
                runtime.hand_retargeting, calibration_samples=2, calibration_timeout_s=0.03,
            ),
        )
        loop = SimpleNamespace(
            runtime=runtime, controller=SimpleNamespace(hand_retargeter=Mock()),
            keyboard=Mock(), runtime_state=Mock(), robot=Mock(),
            sensors=Mock(), poll_blocking_commands=Mock(return_value=False),
        )
        loop.keyboard.check_health.return_value = True
        loop.sensors.check.return_value = True
        return loop

    def frame(self, ring, source):
        return dict(ring_sequence=ring, recv_ts_ns=source, landmarks=self.points.copy())

    def test_duplicate_and_backward_source_stamps_do_not_count(self):
        loop, clock = self.loop(), Clock()
        frames = [
            self.frame(1, 1_000_000_000),
            self.frame(1, 1_000_000_000),  # 同一共享内存帧。
            self.frame(2, 1_000_000_000),  # 新 ring 编号，旧来源时间。
            self.frame(3, 999_000_000),    # 来源倒退。
            self.frame(4, 1_020_000_000),
        ]
        with patch.object(teleop_run, "time", clock), patch.object(
            teleop_run, "read_vr_frame", side_effect=frames,
        ) as read:
            self.assertTrue(teleop_run.calibrate_hand(loop))
        self.assertEqual(read.call_count, 5)
        self.assertEqual(len(loop.controller.hand_retargeter.calibrate.call_args.args[0]), 2)

    def test_repeated_cached_frame_times_out_instead_of_calibrating(self):
        loop, clock = self.loop(), Clock()
        with patch.object(teleop_run, "time", clock), patch.object(
            teleop_run, "read_vr_frame", return_value=self.frame(1, 1_000_000_000),
        ):
            with self.assertRaisesRegex(RuntimeError, "1/2"):
                teleop_run.calibrate_hand(loop)
        loop.controller.hand_retargeter.calibrate.assert_not_called()

    def test_stale_or_future_frames_are_not_used(self):
        loop, clock = self.loop(), Clock()
        frames = [self.frame(1, 1), self.frame(2, 5_000_000_000)]
        with patch.object(teleop_run, "time", clock), patch.object(
            teleop_run, "read_vr_frame", side_effect=lambda _: frames.pop(0) if frames else None,
        ):
            with self.assertRaisesRegex(RuntimeError, "0/2"):
                teleop_run.calibrate_hand(loop)

    def test_cancel_and_worker_failure_do_not_publish_calibration(self):
        loop = self.loop()
        loop.poll_blocking_commands.return_value = True
        self.assertFalse(teleop_run.calibrate_hand(loop))
        loop.runtime_state.request_quit.assert_called_once()
        loop.controller.hand_retargeter.calibrate.assert_not_called()
        loop = self.loop()
        loop.sensors.check.return_value = False
        with self.assertRaisesRegex(RuntimeError, "VR worker"):
            teleop_run.calibrate_hand(loop)
        loop.controller.hand_retargeter.calibrate.assert_not_called()

    def test_disabled_hand_does_not_read_vr_for_calibration(self):
        loop = self.loop()
        loop.controller.hand_retargeter = None
        with patch.object(teleop_run, "read_vr_frame") as read:
            self.assertTrue(teleop_run.calibrate_hand(loop))
        read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
