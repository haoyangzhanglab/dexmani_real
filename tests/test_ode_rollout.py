from collections import deque
from concurrent.futures import Future
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.config.real import RealConfig
from dexmani_real.control.actions import ActionMapper
from dexmani_real.data.export import accepted_joint_targets
from dexmani_real.deployment import rollout as deployment
from dexmani_real.deployment.inference import PolicyWorker
from dexmani_real.deployment.schedule import ActionChunk
from dexmani_real.robot.commands import HandCommandUnconfirmed


class Worker:
    def __init__(self):
        self.future = None
        self.calls = []

    poll = PolicyWorker.poll

    def predict(self, rows, **kwargs):
        self.calls.append((rows, kwargs))
        self.future = Future()

    def finish(self, target):
        self.future.set_result(np.full((1, 19), target, dtype=np.float32))


class Recorder:
    def __init__(self):
        self.rows = []
        self.starts = [0]
        self.new_segment = False

    def mark_gap(self):
        self.new_segment = True

    def check_error(self):
        pass

    def append(self, row, command, *, action_accepted, quality_issue):
        if self.new_segment:
            self.starts.append(len(self.rows))
            self.new_segment = False
        attempted = np.full(19, np.nan) if command is None else np.r_[command.arm_qpos, command.hand_qpos]
        self.rows.append((row, attempted, action_accepted))


@pytest.fixture
def episode(monkeypatch):
    clock = SimpleNamespace(now=1_000_000_000, row=None)
    monkeypatch.setattr(deployment.time, "monotonic_ns", lambda: clock.now)
    monkeypatch.setattr(deployment, "check_device_feedback", lambda *args: [])
    monkeypatch.setattr(deployment, "check_observation", lambda *args: [])
    state = SimpleNamespace(cancel_epoch=0, requests_allow_motion=lambda: True, stop_signal=Event())
    runtime = SimpleNamespace(execution=SimpleNamespace(control_dt_s=0.05, mode="sync", action_steps=1))
    info = SimpleNamespace(n_obs_steps=2, action_mode="joint", requires_prev_action=True)
    observations = SimpleNamespace(
        needs=SimpleNamespace(hoi_joint_devices=(), hoi=False, point_cloud=False),
        project=lambda row: row,
    )
    worker, recorder, robot = Worker(), Recorder(), Mock()
    robot.hand_feedback_pending = False
    robot.feedback_next_ns = 0
    action_mapper = SimpleNamespace(
        map_joint=lambda arm, hand: SimpleNamespace(arm_qpos=arm, hand_qpos=hand, rejection_reason=None),
    )
    rollout = deployment.Rollout(
        state, runtime, info, episode_cancel_epoch=0, robot=robot, observations=observations, sensors=None,
        action_mapper=action_mapper, model=worker, max_duration_s=None, index=0, output_dir=None,
        record_raw=True, effective_parameters={},
    )
    rollout.started_ns = clock.now
    rollout.read_observations = lambda **kwargs: (clock.row, clock.row)
    rollout.recording_issue = lambda row: None
    rollout.recorder = recorder
    control_step = rollout.step

    def step(slot, joint_state):
        clock.now = rollout.started_ns + slot * rollout.dt_ns
        clock.row = np.full(19, joint_state, dtype=np.float32)
        control_step()

    return rollout, worker, recorder, robot, step


def test_ode_queries_and_offline_targets_after_skipped_slots(episode):
    rollout, worker, recorder, robot, step = episode
    step(0, 1)
    worker.finish(10)
    step(1, 2)
    step(2, 3)
    np.testing.assert_array_equal(worker.calls[-1][1]["prev_action"], np.full(19, 10))

    step(4, 4)
    step(5, 5)
    assert rollout.result.skipped_slots == 1
    assert rollout.prev_action is None and not rollout.history
    assert robot.send_action.call_count == 1 and len(recorder.rows) == 3

    worker.finish(999)
    step(6, 6)
    rows, kwargs = worker.calls[-1]
    np.testing.assert_array_equal(rows, np.full((2, 19), 6))
    assert kwargs["reset_episode"] and kwargs["prev_action"] is None
    assert robot.send_action.call_count == 1

    worker.finish(20)
    step(7, 7)
    step(8, 8)
    np.testing.assert_array_equal(worker.calls[-1][1]["prev_action"], np.full(19, 20))
    worker.finish(30)
    robot.send_action.side_effect = OSError("submission failed")
    with pytest.raises(OSError):
        step(9, 9)
    np.testing.assert_array_equal(rollout.prev_action, np.full(19, 20))
    states, attempts, status = map(np.stack, zip(*recorder.rows))
    assert recorder.starts == [0, 3]
    assert status.tolist() == [False, True, False, False, True, False, False]
    raw = {
        "action_send_accepted": status,
        "arm_qpos": states[:, :7],
        "hand_qpos": states[:, 7:],
        "action_arm_joint_target": attempts[:, :7],
        "action_hand_joint_target": attempts[:, 7:],
    }
    targets = accepted_joint_targets(raw, [*recorder.starts, len(states)], {})
    np.testing.assert_array_equal(targets, np.repeat(np.array([1, 10, 10, 6, 20, 20, 20])[:, None], 19, axis=1))


def test_ode_discards_unexecuted_chunk_on_skip(episode):
    rollout, worker, _, robot, step = episode
    rollout.last_slot = 0
    rollout.history = deque([np.zeros(19)], maxlen=2)
    rollout.first_prediction = False
    rollout.prev_action = np.full(19, 10)
    rollout.plan = ActionChunk(np.full((2, 19), 99), 0)
    rollout.cursor = 1

    step(2, 3)
    assert rollout.plan is None and rollout.cursor == 0
    robot.send_action.assert_not_called()
    rows, kwargs = worker.calls[-1]
    np.testing.assert_array_equal(rows, np.full((2, 19), 3))
    assert kwargs["reset_episode"] and kwargs["prev_action"] is None


def test_policy_run_resumes_after_write_crc_without_advancing_ode_history(episode):
    rollout, worker, recorder, robot, step = episode
    rollout.prepare = Mock()
    robot.send_action.side_effect = [None, HandCommandUnconfirmed(2, persistent=False), None]

    def tick():
        step(0, 1)
        worker.finish(10)
        step(1, 2)
        step(2, 3)
        np.testing.assert_array_equal(worker.calls[-1][1]["prev_action"], np.full(19, 10))
        worker.finish(99)
        step(3, 4)
        assert not rollout.finished and not rollout.runtime_state.stop_signal.is_set()
        assert rollout.prev_action is None and not rollout.history and rollout.cursor == 0
        assert not recorder.rows[-1][2]
        np.testing.assert_array_equal(recorder.rows[-1][1], np.full(19, 99))

        step(4, 5)
        rows, kwargs = worker.calls[-1]
        assert kwargs["reset_episode"] and kwargs["prev_action"] is None
        np.testing.assert_array_equal(rows, np.full((2, 19), 5))
        worker.finish(20)
        step(5, 6)
        np.testing.assert_array_equal(rollout.prev_action, np.full(19, 20))
        assert recorder.starts == [0, 4]
        rollout.stop("test complete")

    rollout.step = tick
    result = rollout.run(poll_operator=Mock())
    assert result.termination_details == "test complete" and not result.errors
    assert rollout.model is worker
    assert robot.send_action.call_count == 3


def test_rollout_ends_only_episode_on_persistent_crc(episode):
    rollout, worker, recorder, robot, step = episode
    robot.send_action.side_effect = HandCommandUnconfirmed(5, persistent=True)
    step(0, 1)
    worker.finish(10)
    step(1, 2)
    assert rollout.finished and rollout.runtime_state.stop_signal.is_set()
    assert "5 consecutive write CRC" in rollout.result.termination_details
    assert not recorder.rows[-1][2]


def test_out_of_limits_policy_action_ends_episode_without_exiting_session(episode):
    rollout, worker, _, robot, step = episode
    cfg = RealConfig()
    rollout.action_mapper = ActionMapper.for_mode(
        "joint", arm=cfg.arm, hand=cfg.hand, motion=cfg.motion, environment=cfg.environment,
    )
    rollout.prepare = Mock()

    def tick():
        step(0, 1)
        worker.finish(10)
        step(1, 2)

    rollout.step = tick
    result = rollout.run(poll_operator=Mock())

    assert rollout.finished and rollout.model is worker
    assert "right_hand_thumb_bend_joint" in result.termination_details
    robot.send_action.assert_not_called()
    robot.stop.assert_called_once()


def test_rollout_waits_for_good_feedback_before_using_next_observation(episode):
    rollout, worker, recorder, robot, step = episode
    step(0, 1)
    worker.finish(99)
    robot.hand_feedback_pending = True
    step(1, 2)
    assert worker.future is None and rollout.prev_action is None and not rollout.history
    assert not rollout.finished and not rollout.runtime_state.stop_signal.is_set()
    robot.send_action.assert_not_called()
    assert len(recorder.rows) == 1 and recorder.new_segment

    robot.hand_feedback_pending = False
    step(2, 3)
    rows, kwargs = worker.calls[-1]
    assert kwargs["reset_episode"]
    np.testing.assert_array_equal(rows, np.full((2, 19), 3))
