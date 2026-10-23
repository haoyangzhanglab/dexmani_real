import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from fakes import FakeClock
from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.deployment.inference import InferenceWorker
from dexmani_real.robot.commands import DispatchError, DispatchStatus, RobotCommand
from dexmani_real.robot.robot import DexManiRobot
from dexmani_real.runtime.operator_input import KeyboardInput, OperatorCommand
from dexmani_real.runtime.state import RuntimeState, SafetyState


class AuthorityTests(unittest.TestCase):
    def setUp(self):
        self.state = RuntimeState()
        self.state.arm()

    def test_stop_cancels_preparation_and_cannot_acknowledge_newer_stop(self):
        epoch = self.state.run_id
        self.state.request_stop()
        stopped = self.state.run_id
        self.assertFalse(self.state.command_may_cross_sdk(run_id=epoch, required_safety_state=SafetyState.ARMED))
        self.assertIsNone(self.state.begin_motion())
        self.state.request_stop()
        self.state.acknowledge_stop(stopped)
        self.assertTrue(self.state.stop_request)

    def test_estop_cancels_old_running_epoch(self):
        epoch, _ = self.state.begin_motion()
        self.state.request_emergency_stop()
        self.assertFalse(self.state.command_may_cross_sdk(run_id=epoch))
        self.assertIsNone(self.state.begin_motion())

    def robot(self):
        robot = DexManiRobot(self.state, ExperimentConfig())
        robot.owner = threading.get_ident()
        robot.connected = True
        robot.arm = Mock(is_connected=True)
        robot.arm.submit.return_value = (self.state.run_id, 1)
        robot.hand = Mock(is_connected=True)
        robot.hand.send_action.return_value = DispatchStatus.ACCEPTED
        return robot

    def test_partial_dispatch_preserves_arm_submission_and_unknown_hand(self):
        epoch, _ = self.state.begin_motion()
        robot = self.robot()
        robot.hand.send_action.side_effect = RuntimeError("hand failed")
        clock = FakeClock()
        with patch("time.monotonic_ns", clock.monotonic_ns), self.assertRaises(DispatchError) as caught:
            robot.send_action(RobotCommand(epoch, np.zeros(7), np.zeros(12)),
                              valid_until_ns=clock.now + 100, execute_until_ns=clock.now + 200)
        self.assertEqual(caught.exception.result.arm, DispatchStatus.SUBMITTED)
        self.assertEqual(caught.exception.result.hand, DispatchStatus.UNKNOWN)
        self.assertEqual(caught.exception.result.command_id, (epoch, 1))
        robot.arm.request_stop.assert_called_once()

    def test_revocation_between_devices_blocks_hand(self):
        epoch, _ = self.state.begin_motion()
        robot = self.robot()
        def submitted(*args, **kwargs):
            self.state.request_stop()
            return epoch, 1
        robot.arm.submit.side_effect = submitted
        with self.assertRaises(DispatchError) as caught:
            robot.send_action(RobotCommand(epoch, np.zeros(7), np.zeros(12)),
                              valid_until_ns=2**63 - 1, execute_until_ns=2**63 - 1)
        self.assertEqual(caught.exception.result.arm, DispatchStatus.SUBMITTED)
        self.assertEqual(caught.exception.result.hand, DispatchStatus.NOT_CALLED)
        robot.hand.send_action.assert_not_called()

    def test_unconfirmed_stop_retains_fault_and_still_stops_other_device(self):
        self.state.begin_motion()
        robot = self.robot()
        robot.motion_active = True
        robot.arm.wait_stopped.side_effect = RuntimeError("stop unconfirmed")
        with self.assertRaises(RuntimeError):
            robot.stop()
        robot.hand.stop.assert_called_once()
        self.assertTrue(robot.stop_required)
        self.assertEqual(self.state.safety_state, SafetyState.FAULT)

    def test_blocked_model_does_not_block_callback_and_close_is_pending(self):
        entered, release = threading.Event(), threading.Event()
        class Model:
            def reset_episode(self):
                entered.set()
                if not release.wait(2):
                    raise RuntimeError("test did not release model")
            def close(self):
                pass
        worker = InferenceWorker(Model)
        worker.model = Model()
        epoch, _ = self.state.begin_motion()
        keys = KeyboardInput(stop_callback=self.state.request_stop)
        keys.callbacks_active = True
        try:
            worker.submit("reset_episode", query_context={"preparing_epoch": epoch})
            self.assertTrue(entered.wait(1))
            self.assertTrue(keys.dispatch_immediate_callback(OperatorCommand.STOP))
            self.assertFalse(self.state.command_may_cross_sdk(run_id=epoch))
            self.assertFalse(worker.future.done())
            with self.assertRaises(RuntimeError):
                worker.submit("reset_episode")
            worker.close()
            self.assertEqual(worker.close_result(0)["status"], "pending")
        finally:
            release.set()
            worker.close()
            closed = worker.close_result(2)
        self.assertEqual(closed["status"], "completed")
        operation, result = worker.poll()
        self.assertEqual((operation, result.query_context), ("reset_episode", {"preparing_epoch": epoch}))
