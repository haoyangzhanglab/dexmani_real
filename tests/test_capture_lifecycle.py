"""Teleop/Replay 的停止失败、首次原因及发布边界；设备均为替身。"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from raw_fixture import raw_row, start_raw_recorder
from test_runtime_contract import armed_state

from dexmani_real.config import ExperimentConfig
from dexmani_real.data.raw.reader import EpisodeReader, RawDataError
from dexmani_real.data.recorder import RecordingError
from dexmani_real.replay.replayer import ReplayStatus, TrajectoryReplayer
from dexmani_real.robot.commands import RobotCommand, RobotError
from dexmani_real.runtime.observation import ObservationNeeds
from dexmani_real.runtime.operator_input import OperatorCommand
from dexmani_real.runtime.state import RunEndReason
from dexmani_real.teleop.runner import TeleopPhase, TeleopRunner


class TeleopCaptureTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def runner(self):
        state = armed_state()
        runner = TeleopRunner(
            state, ExperimentConfig(), Mock(stop_required=True), task_label='offline',
            control_needs=ObservationNeeds(), record_needs=ObservationNeeds(),
            sensors=Mock(), controller=Mock(), home_planner=Mock(), vr_process=Mock(),
            recording_dir=self.root, keyboard=Mock(healthy=True), audio=Mock(),
        )
        runner.phase = TeleopPhase.ACTIVE
        runner.control_run_id, _ = state.begin_motion()
        runner.recorder = Mock(start_attempted=True, is_recording=True, published_path=None)
        runner.recorder.save_episode.return_value = self.root / 'episode'
        return runner

    def test_operator_quit_home_and_duration_preserve_reason_and_stop_status(self):
        cases = ((OperatorCommand.STOP, 'operator'), (OperatorCommand.QUIT, 'quit'),
                 (OperatorCommand.HOME, 'operator'), ('duration', 'timeout'))
        for command, reason in cases:
            for stop_fails in (False, True):
                with self.subTest(command=command, stop_fails=stop_fails):
                    runner = self.runner()
                    recorder = runner.recorder
                    failure = RobotError('stop unresolved')
                    events = []

                    def stop():
                        events.append('stop')
                        self.assertFalse(runner.runtime_state.command_may_cross_sdk(run_id=runner.control_run_id))
                        if stop_fails and len(events) == 1:
                            raise failure

                    runner.robot.stop.side_effect = stop
                    recorder.save_episode.side_effect = lambda **kw: events.append('save') or self.root / 'episode'
                    with (
                        patch('dexmani_real.teleop.runner.home_robot') as home,
                        patch.object(runner, 'poll_blocking_commands', return_value=False),
                        patch('dexmani_real.teleop.runner.time.monotonic',
                              return_value=runner.runtime.collection.max_duration_s + 1),
                    ):
                        home.return_value = SimpleNamespace(ok=True)

                        def terminate():
                            if command == 'duration':
                                runner.run_control_tick()
                            else:
                                runner.handle_operator_command(command, expected_run_id=runner.runtime_state.run_id)

                        if stop_fails:
                            with self.assertRaises(RobotError) as raised:
                                terminate()
                            self.assertIs(raised.exception, failure)
                            self.assertEqual(runner.phase, TeleopPhase.ACTIVE)
                            home.assert_not_called()
                            runner.close()
                        else:
                            terminate()
                    recorder.save_episode.assert_called_once()
                    saved = recorder.save_episode.call_args.kwargs
                    self.assertEqual(saved['reason'], reason)
                    self.assertEqual(saved['technical_status'], 'failed' if stop_fails else 'normal')
                    self.assertEqual(events, ['stop', 'stop', 'save'] if stop_fails else ['stop', 'save'])
                    if stop_fails:
                        self.assertEqual(saved['details'][-1]['stage'], 'operator_stop')
                        self.assertEqual(saved['details'][-1]['message'], 'stop unresolved')
                    runner.robot.send_action.assert_not_called()

    def test_shutdown_errors_mark_pending_operator_capture_failed(self):
        for stage in ('shutdown_stop', 'reference_cleanup'):
            with self.subTest(stage=stage):
                runner = self.runner()
                recorder = start_raw_recorder(self.root / stage, collection_source='teleop')
                self.addCleanup(recorder.close)
                runner.recorder = recorder
                recorder.add_frame(raw_row(), RobotCommand(runner.control_run_id, np.zeros(7), np.zeros(12)))
                runner.stop_control('operator')
                failure = RobotError(stage)
                operation = runner.robot.stop if stage == 'shutdown_stop' else runner.controller.clear_reference
                operation.side_effect = failure
                with self.assertLogs('dexmani_real.teleop.runner', level='ERROR'), self.assertRaises(RobotError) as raised:
                    runner.close()
                self.assertIs(raised.exception, failure)
                with EpisodeReader(recorder.published_path) as reader:
                    self.assertEqual(reader.meta['termination_reason'], 'operator')
                    self.assertEqual(reader.meta['technical_status'], 'failed')
                    with self.assertRaisesRegex(RawDataError, 'technical failure'):
                        reader.validate_completion()
                self.assertEqual(recorder.finish_request.details[-1]['stage'], stage)

    def test_run_error_is_technical_even_when_first_reason_already_exists(self):
        for stop_error in (False, True):
            with self.subTest(stop_error=stop_error):
                runner = self.runner()
                recorder = runner.recorder
                failure = RobotError('runtime stop unresolved') if stop_error else RuntimeError('control failed')
                if stop_error:
                    runner.initialize = Mock(return_value=True)
                    runner.keyboard.poll.return_value = [OperatorCommand.STOP]
                    runner.robot.stop.side_effect = [failure, None]
                else:
                    runner.stop_control('operator')
                    runner.initialize = Mock(side_effect=failure)
                with self.assertLogs('dexmani_real.teleop.runner', level='ERROR'), self.assertRaises(type(failure)) as raised:
                    runner.run()
                self.assertIs(raised.exception, failure)
                runner.close()
                saved = recorder.save_episode.call_args.kwargs
                self.assertEqual(saved['reason'], 'operator')
                self.assertEqual(saved['technical_status'], 'failed')
                self.assertEqual(saved['details'][-1]['stage'], 'runtime')

    def test_discard_stays_unpublished_after_stop_and_discard_cleanup_errors(self):
        for stop_fails, discard_fails in ((False, False), (True, False), (True, True)):
            with self.subTest(stop_fails=stop_fails, discard_fails=discard_fails):
                runner = self.runner()
                recorder = start_raw_recorder(self.root / f'{stop_fails}_{discard_fails}', collection_source='teleop')
                self.addCleanup(recorder.close)
                runner.recorder = recorder
                recorder.add_frame(raw_row(), RobotCommand(runner.control_run_id, np.zeros(7), np.zeros(12)))
                destination = recorder.episode_dir
                failure = RobotError('discard stop unresolved')
                runner.robot.stop.side_effect = [failure, None] if stop_fails else None
                discard_episode = recorder.discard_episode

                def discard(**kwargs):
                    discard_episode(**kwargs)
                    if discard_fails:
                        raise RecordingError('discard cleanup failed')

                with patch.object(recorder, 'discard_episode', side_effect=discard) as discarded:
                    if stop_fails:
                        with self.assertRaises(RobotError) as raised:
                            runner.handle_operator_command(OperatorCommand.DISCARD)
                        self.assertIs(raised.exception, failure)
                        if discard_fails:
                            self.assertIsInstance(raised.exception.__cause__, RecordingError)
                    else:
                        runner.handle_operator_command(OperatorCommand.DISCARD)
                    runner.close()
                    discarded.assert_called_once()
                self.assertFalse(recorder.finish_request.save)
                self.assertIsNone(recorder.published_path)
                self.assertFalse(destination.exists())
                self.assertFalse(recorder.writer_pending)

    def test_cleanup_failure_after_publication_does_not_create_another_capture(self):
        for stage in ('stop', 'reference'):
            with self.subTest(stage=stage):
                runner = self.runner()
                recorder = runner.recorder
                runner.handle_operator_command(OperatorCommand.STOP)
                operation = runner.robot.stop if stage == 'stop' else runner.controller.clear_reference
                operation.side_effect = RobotError('session cleanup failed')
                with self.assertLogs('dexmani_real.teleop.runner', level='ERROR'), self.assertRaisesRegex(RobotError, 'session cleanup'):
                    runner.close()
                self.assertIsNone(runner.recorder)
                recorder.save_episode.assert_called_once()
                self.assertEqual(recorder.save_episode.call_args.kwargs['technical_status'], 'normal')
                self.assertFalse(runner.capture_failed)

    def test_published_recorder_handle_is_not_marked_as_an_unpublished_capture(self):
        runner = self.runner()
        runner.recorder.published_path = self.root / 'published'
        runner.robot.stop.side_effect = RobotError('session stop unresolved')
        with self.assertRaises(RobotError):
            runner.stop_motion('shutdown_stop')
        self.assertFalse(runner.capture_failed)
        runner.recorder.save_episode.assert_not_called()


class ReplayCaptureTests(unittest.TestCase):
    def test_stop_outcome_preserves_first_reason_and_captured_prefix(self):
        cases = ((ReplayStatus.COMPLETED, ''), (ReplayStatus.USER_QUIT, 'operator quit'),
                 (ReplayStatus.REJECTED, 'control rejected'))
        for initial_status, reason in cases:
            for stop_fails in (False, True):
                with self.subTest(initial_status=initial_status, stop_fails=stop_fails):
                    state = armed_state()
                    robot = Mock()
                    replayer = TrajectoryReplayer(state, ExperimentConfig(), Mock(), robot=robot)
                    row = raw_row()

                    def admit(*args):
                        replayer.epoch, _ = state.begin_motion()
                        return ReplayStatus.COMPLETED, '', row

                    def replay_rows(*args):
                        replayer.capture.record(0, row, RobotCommand(replayer.epoch, np.zeros(7), np.zeros(12)))
                        return initial_status, reason

                    def stop():
                        self.assertFalse(state.command_may_cross_sdk(run_id=replayer.epoch))
                        if stop_fails:
                            raise RobotError('stop unresolved')

                    replayer.admit_replay, replayer.replay_rows = admit, replay_rows
                    robot.stop.side_effect = stop
                    with patch('dexmani_real.replay.replayer.ActionMapper.for_mode'):
                        if stop_fails:
                            with self.assertLogs('dexmani_real.replay.replayer', level='ERROR'):
                                outcome = replayer.run(SimpleNamespace(num_frames=1, fps=10), hand_start_duration_s=0)
                        else:
                            outcome = replayer.run(SimpleNamespace(num_frames=1, fps=10), hand_start_duration_s=0)
                    data = outcome.replay_data
                    failed = stop_fails or initial_status is ReplayStatus.REJECTED
                    self.assertEqual(outcome.status, ReplayStatus.FAULT if stop_fails else initial_status)
                    self.assertEqual(data['termination_reason'].item(), reason or initial_status.value)
                    self.assertEqual(data['technical_status'].item(), 'failed' if failed else 'normal')
                    self.assertEqual(data['cleanup_errors'].item(), 'stop failed: stop unresolved' if stop_fails else '')
                    self.assertEqual(replayer.failure_details, ['stop failed: stop unresolved'] if stop_fails else [])
                    self.assertEqual(state.run_ended_reason, RunEndReason.QUIT if initial_status is ReplayStatus.USER_QUIT
                                     else RunEndReason.EXECUTOR_BOUNDARY)
                    np.testing.assert_array_equal(data['arm_qpos'], row.arm['qpos'])
                    np.testing.assert_array_equal(data['arm_observation_ns'], row.arm['timestamp_ns'])
                    self.assertEqual(data['preparation_submission_count'].item(), 0)
                    robot.stop.assert_called_once()
                    robot.send_action.assert_not_called()

    def test_submission_failure_preserves_failed_prefix_through_normal_stop(self):
        replayer = TrajectoryReplayer(armed_state(), ExperimentConfig(), Mock(), robot=Mock())
        failure = RobotError('submission failed')
        replayer.robot.send_action.side_effect = failure
        row = raw_row()

        def admit(*args):
            replayer.epoch, _ = replayer.runtime_state.begin_motion()
            return ReplayStatus.COMPLETED, '', row

        def replay_rows(*args):
            replayer.submit_row(0, row, np.zeros(7), np.zeros(12), 2**63 - 1)

        replayer.admit_replay, replayer.replay_rows = admit, replay_rows
        with patch('dexmani_real.replay.replayer.ActionMapper.for_mode'), self.assertLogs(
            'dexmani_real.replay.replayer', level='ERROR',
        ):
            outcome = replayer.run(SimpleNamespace(num_frames=1, fps=10), hand_start_duration_s=0)
        self.assertEqual(outcome.status, ReplayStatus.FAULT)
        self.assertEqual(outcome.replay_data['termination_reason'].item(), 'submission failed')
        self.assertEqual(outcome.replay_data['technical_status'].item(), 'failed')
        self.assertEqual(len(outcome.replay_data['arm_qpos']), 1)
        self.assertEqual(outcome.replay_data['cleanup_errors'].item(), '')
        replayer.robot.send_action.assert_called_once()
        replayer.robot.stop.assert_called_once()
