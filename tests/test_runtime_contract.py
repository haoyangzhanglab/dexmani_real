"""运行状态、输入、进程及 HOME 的离线契约；不创建真实设备连接。"""

import importlib
import multiprocessing as mp
import pickle
import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config import ExperimentConfig
from dexmani_real.geometry.pose import Pose
from dexmani_real.robot.commands import OperationCancelled, RobotCommand, RobotError
from dexmani_real.robot.feedback import HAND_STATE_DTYPE
from dexmani_real.robot.system import DexManiRobot
from dexmani_real.runtime.motion.home import execute_arm_home, home_hand
from dexmani_real.runtime.motion.jog import JogController
from dexmani_real.runtime.operator_input import KeyboardInput
from dexmani_real.runtime.sensor_processes import SensorProcesses
from dexmani_real.runtime.shutdown import shutdown_local_runtime
from dexmani_real.runtime.state import RunEndReason, RuntimeState, SafetyState


class TrackingLock:
    def __init__(self):
        self.lock = threading.RLock()
        self.depth = 0

    def __enter__(self):
        self.lock.acquire()
        self.depth += 1
        return self

    def __exit__(self, *args):
        self.depth -= 1
        self.lock.release()


def armed_state():
    state = RuntimeState(motion_lock=TrackingLock())
    state.arm()
    # 替身 owner 已完成停止协调，允许下一轮软件准入。
    state.stop_signal.clear()
    return state


def resolve_worker_targets(queue):
    # spawn 子进程只导入、pickle 实际 target，不运行任何设备 worker。
    names = (
        ('dexmani_real.sensor.camera.worker', 'run_camera_worker'),
        ('dexmani_real.sensor.vr.worker', 'run_vr_worker'),
        ('dexmani_real.perception.pointcloud.worker', 'run_pointcloud_worker'),
    )
    for module, name in names:
        target = getattr(importlib.import_module(module), name)
        assert pickle.loads(pickle.dumps(target)) is target
    queue.put(names)


class RuntimeStateTests(unittest.TestCase):
    def test_old_identity_and_first_termination_survive_cleanup(self):
        state = armed_state()
        run_id, _ = state.begin_motion()
        self.assertTrue(state.command_may_cross_sdk(run_id=run_id))
        state.request_stop()
        self.assertFalse(state.command_may_cross_sdk(run_id=run_id))
        self.assertEqual(state.run_ended_reason, RunEndReason.OPERATOR)
        self.assertFalse(state.revoke_motion(expected_run_id=run_id))
        state.end_session()
        self.assertEqual(state.run_ended_id, run_id)
        self.assertEqual(state.run_ended_reason, RunEndReason.OPERATOR)
        self.assertTrue(state.stop_signal.is_set())

    def test_quit_and_idempotent_emergency_latch(self):
        for request, reason in (('request_quit', RunEndReason.QUIT), ('request_emergency_stop', RunEndReason.ESTOP)):
            state = armed_state()
            state.begin_motion()
            getattr(state, request)()
            self.assertFalse(state.requests_allow_motion())
            self.assertEqual(state.run_ended_reason, reason)
            if request == 'request_emergency_stop':
                identity = state.run_id
                state.request_emergency_stop()
                self.assertEqual(state.run_id, identity)
                self.assertTrue(state.emergency_signal.is_set())


class JogSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.state = armed_state()
        self.keys = KeyboardInput()
        self.keys.callbacks_active = True  # 不调用 start()，无全局键盘监听。
        self.epoch = self.state.run_id
        self.jog = JogController(None, None, None, None)
        self.pose = Pose.identity()
        self.target = np.zeros(7)

    def submit(self, robot):
        with patch('dexmani_real.runtime.motion.jog.time.monotonic_ns', return_value=100):
            return self.jog.submit(self.pose, self.target, self.keys, 0, self.state, robot,
                                   self.epoch, valid_until_ns=200, execute_until_ns=300)

    def test_success_commits_reference_after_io_outside_both_locks(self):
        def send(command, **kwargs):
            self.assertIsNone(self.jog.command_qpos)
            self.assertEqual(self.state.motion_lock.depth, 0)
            self.assertTrue(self.keys.condition.acquire(blocking=False))
            self.keys.condition.release()
            self.assertEqual(command.input_generation, 0)
            self.assertEqual(kwargs, dict(valid_until_ns=200, execute_until_ns=300))
        self.assertTrue(self.submit(SimpleNamespace(send_action=send)))
        np.testing.assert_array_equal(self.jog.command_qpos, self.target)
        self.assertIsNot(self.jog.command_qpos, self.target)

    def test_changed_generation_prevents_submission(self):
        self.keys.jog_changed('w', True)
        robot = Mock()
        self.assertFalse(self.submit(robot))
        robot.send_action.assert_not_called()
        self.assertIsNone(self.jog.command_qpos)

    def test_generation_change_during_submission_does_not_commit_reference(self):
        robot = SimpleNamespace(send_action=lambda *a, **k: self.keys.jog_changed('w', True))
        with self.assertRaises(OperationCancelled):
            self.submit(robot)
        self.assertIsNone(self.jog.command_qpos)
        self.assertGreater(self.state.run_id, self.epoch)

    def test_device_error_does_not_commit_reference(self):
        robot = Mock()
        robot.send_action.side_effect = RobotError('offline rejection')
        with self.assertRaises(RobotError):
            self.submit(robot)
        self.assertIsNone(self.jog.command_qpos)
        self.assertEqual(self.state.run_ended_reason, RunEndReason.HARDWARE_FAULT)


class ProcessAndShutdownTests(unittest.TestCase):
    def test_start_failure_does_not_take_ownership(self):
        sensors, process = SensorProcesses(), Mock()
        process.start.side_effect = RuntimeError('start failed')
        with self.assertRaisesRegex(RuntimeError, 'start failed'):
            sensors.start(process, ready=Mock(), may_start=lambda: True)
        self.assertEqual(sensors.started, [])

    def test_ready_timeout_retains_started_owner(self):
        sensors, process, ready = SensorProcesses(), Mock(), Mock()
        process.name = 'offline-sensor'
        process.is_alive.return_value = True
        ready.wait.return_value = False
        with patch('dexmani_real.runtime.sensor_processes.time.monotonic', side_effect=[0, 0, .1, 2]):
            with self.assertRaisesRegex(RuntimeError, 'failed to become ready'):
                sensors.start(process, ready=ready, timeout_s=1, may_start=lambda: True)
        self.assertEqual(sensors.started, [(process, ready)])

    def test_sensor_escalation_is_reported_as_unclean(self):
        sensors, process, ready = SensorProcesses(), Mock(), Mock()
        process.name, process.exitcode = 'offline-sensor', -9
        process.is_alive.side_effect = [True, True, False]
        sensors.started.append((process, ready))
        with self.assertLogs('dexmani_real.runtime.sensor_processes', level='WARNING'):
            self.assertFalse(sensors.shutdown(graceful_timeout_s=0))
        self.assertEqual([call[0] for call in process.method_calls],
                         ['join', 'is_alive', 'terminate', 'join', 'is_alive', 'kill', 'join', 'is_alive'])
        self.assertEqual(sensors.started, [])
        ready.clear.assert_called_once()

    def test_unresolved_sensor_keeps_handle_and_error(self):
        sensors, process, ready = SensorProcesses(), Mock(), Mock()
        process.name, process.exitcode = 'offline-sensor', None
        process.is_alive.return_value = True
        sensors.started.append((process, ready))
        with self.assertLogs('dexmani_real.runtime.sensor_processes', level='ERROR'):
            with self.assertRaisesRegex(RuntimeError, 'shutdown unresolved'):
                sensors.shutdown(graceful_timeout_s=0)
        self.assertEqual(sensors.started, [(process, ready)])

    def test_cleanup_order_and_failures_do_not_become_success(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                state, events = armed_state(), []
                run_id, _ = state.begin_motion()
                def close_robot():
                    self.assertFalse(state.command_may_cross_sdk(run_id=run_id))
                    self.assertEqual(state.motion_lock.depth, 0)
                    events.append('robot')
                    if fail:
                        raise RuntimeError('close failed')
                def close_sensors(**kwargs):
                    self.assertFalse(state.running)
                    events.append('sensors')
                    return True
                robot = SimpleNamespace(close=close_robot)
                keyboard = SimpleNamespace(stop=lambda: events.append('keyboard'))
                sensors = SimpleNamespace(shutdown=close_sensors)
                if fail:
                    with self.assertLogs('dexmani_real.runtime.shutdown', level='ERROR'):
                        clean = shutdown_local_runtime(state, robot, sensors=sensors, keyboard=keyboard)
                else:
                    clean = shutdown_local_runtime(state, robot, sensors=sensors, keyboard=keyboard)
                self.assertEqual(clean, not fail)
                self.assertEqual(events, ['robot', 'keyboard', 'sensors'])
                self.assertEqual(state.safety_state, SafetyState.FAULT if fail else SafetyState.DISARMED)

    def test_spawn_can_resolve_actual_worker_targets(self):
        context = mp.get_context('spawn')
        queue = context.Queue()
        process = context.Process(target=resolve_worker_targets, args=(queue,))
        try:
            process.start()
            process.join(timeout=15)
            self.assertEqual(process.exitcode, 0)
            self.assertEqual(len(queue.get(timeout=1)), 3)
        finally:
            if process.is_alive():
                process.terminate()  # 仅本测试自己的导入子进程。
                process.join(timeout=2)
            queue.close()
            queue.join_thread()
            if process.exitcode is not None:
                process.close()


class HomeExecutionTests(unittest.TestCase):
    def test_cancelled_arm_home_does_not_reach_device(self):
        robot = Mock()
        config = ExperimentConfig().arm
        result = execute_arm_home(armed_state(), config.home_qpos, robot=robot, planner=Mock(),
                                  config=config, service=lambda: None, cancel_requested=lambda: True)
        self.assertFalse(result.ok)
        self.assertTrue(result.interrupted)
        robot.read_state.assert_not_called()
        robot.home_arm.assert_not_called()

    def test_hand_home_requires_three_distinct_post_submission_samples(self):
        config = replace(ExperimentConfig().hand, home_timeout_s=.08)
        for repeated in (False, True):
            with self.subTest(repeated=repeated):
                clock, samples = [1_000_000_000], []
                target = np.deg2rad(config.home_qpos_deg)
                def read_state():
                    clock[0] += 1_000_000
                    frame = np.zeros(1, dtype=HAND_STATE_DTYPE)
                    frame['qpos'] = target
                    frame['timestamp_ns'] = samples[0] if repeated and samples else clock[0]
                    samples.append(int(frame['timestamp_ns'][0]))
                    return SimpleNamespace(hand=frame)
                def sleep(seconds):
                    clock[0] += int(seconds * 1e9)
                robot = Mock(read_state=Mock(side_effect=read_state))
                with patch('dexmani_real.runtime.motion.home.time.monotonic_ns', side_effect=lambda: clock[0]), \
                     patch('dexmani_real.runtime.motion.home.time.sleep', side_effect=sleep):
                    result = home_hand(armed_state(), config, robot=robot)
                robot.send_hand_home.assert_called_once()
                self.assertEqual(result.ok, not repeated)
                if repeated:
                    self.assertIn('timed out', result.reason)
                    robot.stop.assert_called_once()
                else:
                    self.assertEqual(len(samples), 3)
                    robot.stop.assert_not_called()


class RobotBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.state = armed_state()
        self.state.begin_motion()
        config = ExperimentConfig()
        self.robot = DexManiRobot(
            config.arm, config.hand, run_signal=self.state.run_signal,
            stop_signal=self.state.stop_signal, emergency_signal=self.state.emergency_signal,
            signal_lock=self.state.motion_lock,
        )
        # 仅注入软件提交边界，绝不调用 connect() 或任何设备 factory。
        self.robot.owner = threading.get_ident()
        self.robot.connected = True
        self.robot.arm = Mock(spec=['is_connected', 'check_error', 'submit', 'request_stop', 'wait_stopped'])
        self.robot.hand = Mock(spec=['is_connected', 'send_action', 'end_send_interval', 'stop'])
        self.robot.arm.is_connected = self.robot.hand.is_connected = True
        self.command = RobotCommand(self.state.run_id, np.asarray(config.arm.home_qpos),
                                    np.deg2rad(config.hand.home_qpos_deg))

    def test_submission_preserves_distinct_deadlines(self):
        with patch('dexmani_real.robot.system.time.monotonic_ns', return_value=100):
            self.robot.send_action(self.command, valid_until_ns=200, execute_until_ns=300)
        self.robot.arm.submit.assert_called_once_with(self.command, submit_until_ns=200, execute_until_ns=300)
        self.robot.hand.send_action.assert_called_once()
        self.assertTrue(self.robot.motion_active)

    def test_revocation_after_arm_submission_blocks_hand_send(self):
        self.robot.arm.submit.side_effect = lambda *a, **kw: self.state.request_stop()
        with patch('dexmani_real.robot.system.time.monotonic_ns', return_value=100):
            with self.assertRaises(OperationCancelled):
                self.robot.send_action(self.command, valid_until_ns=200, execute_until_ns=300)
        self.robot.arm.submit.assert_called_once()
        self.robot.hand.send_action.assert_not_called()
        self.assertTrue(self.robot.stop_signal.is_set())

    def test_other_thread_cannot_cross_device_boundary(self):
        errors = []
        def send():
            try:
                self.robot.send_action(self.command, valid_until_ns=200, execute_until_ns=300)
            except RuntimeError as exc:
                errors.append(str(exc))
        worker = threading.Thread(target=send)
        worker.start()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, ['robot I/O belongs to its connecting thread'])
        self.robot.arm.submit.assert_not_called()
        self.robot.hand.send_action.assert_not_called()

    def test_accepted_stop_request_does_not_hide_unconfirmed_stop(self):
        self.robot.motion_active = True
        self.robot.arm.wait_stopped.side_effect = TimeoutError('stop unconfirmed')
        with self.assertRaisesRegex(RobotError, 'stop unconfirmed'):
            self.robot.stop()
        self.robot.arm.request_stop.assert_called_once()
        self.assertTrue(self.robot.motion_active)
        self.assertIsInstance(self.robot.stop_failure, TimeoutError)
