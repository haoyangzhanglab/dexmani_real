import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.control.actions import ActionResult
from dexmani_real.robot.commands import RobotCommand, RobotError
from dexmani_real.teleop.loop import TeleopLoop


@pytest.fixture
def loop(monkeypatch):
    clock = [1000000000]
    monkeypatch.setattr('dexmani_real.teleop.loop.time.monotonic_ns', lambda: clock[0])
    loop = TeleopLoop.__new__(TeleopLoop)
    loop.cancel_epoch = 1
    loop.runtime_state = SimpleNamespace(cancel_epoch=1, requests_allow_motion=lambda: True, stop_signal=threading.Event())
    loop.runtime = SimpleNamespace(teleop=SimpleNamespace(target_max_age_s=.5))
    loop.controller = Mock(previous_eef_target='old_pose')
    loop.controller.compute_target.return_value = ActionResult(None, np.ones(12), 'geometry')
    loop.robot = Mock()
    loop.recorder = Mock()
    loop.recorder.append.return_value = None
    loop.admit_observation = lambda row: []
    loop.audio = Mock()
    loop.last_command = RobotCommand(np.zeros(7), np.zeros(12))
    loop.target_deadline_ns = clock[0] + 100000000
    loop.hold_reason = None
    loop.request_episode_end = Mock()
    loop.pending_termination_reason = None
    loop.poll_blocking_commands = Mock(return_value=False)
    return loop, clock


def test_hold_is_paired_and_does_not_renew(loop):
    obj, clock = loop
    previous = obj.last_command
    deadline = obj.target_deadline_ns
    obj.execute_control_step('row')
    obj.robot.send_action.assert_called_once_with(previous, expires_ns=deadline)
    obj.controller.commit.assert_called_once_with(previous, eef_pose='old_pose', submit_end_ns=clock[0])
    obj.recorder.append.assert_called_once_with('row', previous)
    obj.request_episode_end.assert_not_called()
    assert obj.target_deadline_ns == deadline
    clock[0] = deadline
    obj.execute_control_step('row2')
    assert obj.robot.send_action.call_count == 1
    assert obj.recorder.append.call_count == 1
    obj.request_episode_end.assert_called_with('target_expired')


def test_first_wait_and_recovery(loop):
    obj, clock = loop
    obj.last_command = None
    deadline = obj.target_deadline_ns
    obj.execute_control_step('row')
    obj.execute_control_step('row2')
    assert obj.controller.reset.call_count == 2
    obj.recorder.append.assert_not_called()
    obj.robot.send_action.assert_not_called()
    assert obj.target_deadline_ns == deadline
    obj.controller.compute_target.return_value = ActionResult(np.ones(7), np.ones(12), eef_pose='new_pose')
    obj.execute_control_step('row3')
    assert obj.last_command is not None
    assert obj.target_deadline_ns == clock[0] + 500000000
    obj.controller.commit.assert_called_once()


def test_partial_submission_is_recorded_without_commit(loop):
    obj, _ = loop
    obj.robot.send_action.side_effect = RobotError('hand failed after arm submission')
    with pytest.raises(RobotError):
        obj.execute_control_step('row')
    obj.controller.commit.assert_not_called()
    obj.recorder.append.assert_called_once_with('row', obj.last_command)
    assert obj.runtime_state.stop_signal.is_set()


@pytest.mark.parametrize('cancel,stale', [(True, False), (False, True)])
def test_hold_does_not_override_cancel_or_stale(loop, cancel, stale):
    obj, _ = loop
    if cancel:
        obj.runtime_state.cancel_epoch = 2
    if stale:
        obj.admit_observation = lambda row: ['vr_stale']
    obj.execute_control_step('row')
    obj.robot.send_action.assert_not_called()
    obj.controller.commit.assert_not_called()
    obj.recorder.append.assert_not_called()
    obj.request_episode_end.assert_called_once()


def test_recovery_commits_new_pair_after_hold(loop):
    obj, clock = loop
    obj.execute_control_step('hold')
    clock[0] += 60000000
    obj.controller.compute_target.return_value = ActionResult(np.ones(7), np.ones(12), eef_pose='new_pose')
    obj.execute_control_step('recovered')
    assert obj.target_deadline_ns == clock[0] + 500000000
    np.testing.assert_array_equal(obj.last_command.arm_qpos, np.ones(7))
    np.testing.assert_array_equal(obj.last_command.hand_qpos, np.ones(12))
    assert obj.controller.commit.call_args.kwargs['submit_end_ns'] == clock[0]
    assert obj.controller.commit.call_args.kwargs['eef_pose'] == 'new_pose'


def test_recording_failure_does_not_become_mapping_hold(loop):
    obj, _ = loop
    obj.recorder.check_error.side_effect = OSError('writer failure')
    obj.record_error = Mock()
    with pytest.raises(OSError, match='writer failure'):
        obj.execute_control_step('row')
    obj.robot.send_action.assert_not_called()
    obj.controller.commit.assert_not_called()


def test_first_wait_expires_even_without_vr_snapshot(loop, monkeypatch):
    from dexmani_real.runtime.observation import ObservationNeeds

    obj, clock = loop
    obj.last_command = None
    obj.recorder = None
    obj.sensors = SimpleNamespace(channels=None)
    obj.next_tick_ns = None
    obj.control_dt_ns = 62500000
    obj.control_needs = obj.capture_needs = ObservationNeeds(vr=True)
    read = Mock(return_value=SimpleNamespace(vr=None))
    monkeypatch.setattr('dexmani_real.teleop.loop.read_observation', read)
    monkeypatch.setattr('dexmani_real.teleop.loop.check_device_feedback', lambda *args: [])
    obj.run_control_tick()
    obj.robot.send_action.assert_not_called()
    clock[0] = obj.target_deadline_ns
    obj.run_control_tick()
    obj.request_episode_end.assert_called_once_with('target_expired')
    assert read.call_count == 1


def test_deadline_crossed_during_compute_cannot_be_renewed(loop):
    obj, clock = loop
    def compute(*args, **kwargs):
        clock[0] = obj.target_deadline_ns
        return ActionResult(np.ones(7), np.ones(12), eef_pose='late_pose')
    obj.controller.compute_target.side_effect = compute
    obj.execute_control_step('late')
    obj.robot.send_action.assert_not_called()
    obj.controller.commit.assert_not_called()
    obj.recorder.append.assert_not_called()
    obj.request_episode_end.assert_called_once_with('target_expired')


def test_submission_cancel_records_attempt_without_advancing_reference(loop):
    from dexmani_real.robot.commands import OperationCancelled

    obj, _ = loop
    previous = obj.last_command
    def cancel(*args, **kwargs):
        obj.runtime_state.cancel_epoch += 1
        raise OperationCancelled('cancelled between arm and hand')
    obj.robot.send_action.side_effect = cancel
    obj.execute_control_step('cancelled')
    obj.controller.commit.assert_not_called()
    assert obj.last_command is previous
    obj.recorder.append.assert_called_once_with('cancelled', previous)
    obj.request_episode_end.assert_called_once()


def test_programming_error_is_not_an_ordinary_hold(loop):
    obj, _ = loop
    obj.controller.compute_target.side_effect = ValueError('invalid backend output')
    with pytest.raises(ValueError, match='invalid backend output'):
        obj.execute_control_step('row')
    obj.robot.send_action.assert_not_called()
    obj.recorder.append.assert_not_called()
    assert obj.runtime_state.stop_signal.is_set()


def test_unsubmitted_start_reference_is_rebuilt_without_accumulated_dt(loop):
    from dexmani_real.config.teleop import TeleopConfig
    from dexmani_real.geometry.pose import Pose
    from dexmani_real.teleop.vr import VRController

    obj, clock = loop
    reference = SimpleNamespace(qpos=None, timestamp_ns=None)
    hand = Mock()
    def reset(initial_qpos, *, timestamp_ns):
        reference.qpos = initial_qpos.copy()
        reference.timestamp_ns = timestamp_ns
    hand.reset.side_effect = reset
    hand.retarget.side_effect = [None, None, np.ones(12)*.2]
    hand.rejection_reason = 'geometry'
    mapper = Mock()
    mapper.solver.constraints.workspace = np.array([[0, 1], [-1, 1], [0, 1]])
    mapper.solver.compute_eef_pose_base.return_value = Pose([.4, 0, .3], [1, 0, 0, 0])
    mapper.map_eef.side_effect = lambda pose, h, *args: ActionResult(np.zeros(7), h, eef_pose=pose)
    obj.controller = VRController(mapper, hand, TeleopConfig(), np.eye(3))
    obj.last_command = None
    deadline = obj.target_deadline_ns
    for i in range(3):
        clock[0] += 10000000
        row = SimpleNamespace(
            arm={'qpos': [np.zeros(7)]}, hand={'qpos': [np.ones(12)*(.1*i)]},
            vr={'wrist_pos': np.array([.01*i, 0, 0]), 'wrist_quat_wxyz': np.array([1., 0, 0, 0]), 'landmarks': None},
            observation_timestamp_ns=clock[0]-1000000,
        )
        obj.execute_control_step(row)
        assert reference.timestamp_ns == row.observation_timestamp_ns
        assert hand.retarget.call_args.kwargs['timestamp_ns'] - reference.timestamp_ns == 1000000
        np.testing.assert_array_equal(reference.qpos, row.hand['qpos'][0])
        if i < 2:
            assert obj.target_deadline_ns == deadline
    assert obj.last_command is not None
    assert obj.robot.send_action.call_count == 1
    assert hand.commit.call_count == 1


@pytest.mark.parametrize('failure', [OSError('finish failure'), KeyboardInterrupt()])
def test_recording_finish_failure_releases_episode_state(loop, failure):
    from dexmani_real.teleop.loop import TeleopPhase

    obj, _ = loop
    obj.phase = TeleopPhase.ACTIVE
    obj.runtime_state.quit_requested = obj.runtime_state.estop_request = False
    obj.runtime_state.request_emergency_stop = Mock()
    obj.termination_details = []
    obj.request_episode_end = TeleopLoop.request_episode_end.__get__(obj)
    obj.finish_recording = Mock(side_effect=failure)
    with pytest.raises(type(failure)):
        obj.end_episode()
    obj.robot.stop.assert_called_once()
    obj.controller.clear.assert_called_once()
    assert obj.phase is TeleopPhase.IDLE
    assert obj.cancel_epoch is None
    assert obj.pending_termination_reason == 'failed'
    assert obj.runtime_state.stop_signal.is_set()
    if isinstance(failure, KeyboardInterrupt):
        obj.runtime_state.request_emergency_stop.assert_called_once()
