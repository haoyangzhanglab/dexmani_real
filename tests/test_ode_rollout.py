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
from dexmani_real.robot.commands import COMMAND_LEASE_NS, HandCommandUnconfirmed


class Worker:
    def __init__(self):
        self.future = None
        self.calls = []

    poll = PolicyWorker.poll

    def predict(self, rows, **kwargs):
        assert self.future is None
        self.calls.append((rows, kwargs))
        self.future = Future()
        self.future.set_running_or_notify_cancel()

    def finish(self, target, *, steps=1):
        self.future.set_result(np.full((steps, 19), target, dtype=np.float32))


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
        self.rows.append((row.qpos, attempted, action_accepted))


@pytest.fixture
def episode(monkeypatch):
    clock = SimpleNamespace(now=1_000_000_000, row=None)
    monkeypatch.setattr(deployment.time, "monotonic_ns", lambda: clock.now)
    monkeypatch.setattr(deployment, "check_device_feedback", lambda *args: [])
    monkeypatch.setattr(deployment, "check_observation", lambda *args: [])
    state = SimpleNamespace(cancel_epoch=0, requests_allow_motion=lambda: True, stop_signal=Event())
    runtime = SimpleNamespace(execution=SimpleNamespace(control_dt_s=0.05, mode="sync", action_steps=1))
    info = SimpleNamespace(n_obs_steps=2, action_mode="joint", requires_prev_action=True, future_steps=1)
    observations = SimpleNamespace(
        needs=SimpleNamespace(hoi_joint_devices=(), hoi=False, point_cloud=False),
        project=lambda row: row.qpos,
    )
    worker, recorder, robot = Worker(), Recorder(), Mock()
    robot.hand_feedback_pending = False
    robot.feedback_next_ns = 0

    def map_joint(arm, hand):
        clock.now += clock.map_delay_ns
        return SimpleNamespace(arm_qpos=arm, hand_qpos=hand, rejection_reason=None)

    action_mapper = SimpleNamespace(map_joint=map_joint)
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

    def step(slot, joint_state, *, phase_ns=0, map_delay_ns=0):
        clock.now = rollout.started_ns + slot * rollout.dt_ns + phase_ns
        clock.map_delay_ns = map_delay_ns
        clock.row = SimpleNamespace(
            qpos=np.full(19, joint_state, dtype=np.float32), control_timestamp_ns=clock.now,
        )
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
    rollout.execution.action_steps = rollout.policy_info.future_steps = 2
    step(0, 1)
    worker.finish(99, steps=2)
    step(1, 2)
    assert rollout.cursor == 1
    robot.send_action.reset_mock()

    step(3, 3)
    assert rollout.plan is None and rollout.cursor == 0
    robot.send_action.assert_not_called()
    rows, kwargs = worker.calls[-1]
    np.testing.assert_array_equal(rows, np.full((2, 19), 3))
    assert kwargs["reset_episode"] and kwargs["prev_action"] is None


@pytest.mark.parametrize("reason", [
    "XHand position delivery unconfirmed after 2 consecutive write CRC replies",
    "XHand SDK success reply arrived at or after the hand submission deadline",
])
def test_policy_run_stops_on_unconfirmed_hand_submission(episode, reason):
    rollout, worker, recorder, robot, step = episode
    rollout.prepare = Mock()
    robot.send_action.side_effect = [None, HandCommandUnconfirmed(reason)]
    stop = rollout.stop

    def stop_at_failed_target(detail):
        np.testing.assert_array_equal(rollout.prev_action, np.full(19, 10))
        assert rollout.cursor == 0
        stop(detail)

    rollout.stop = stop_at_failed_target

    def tick():
        step(0, 1)
        worker.finish(10)
        step(1, 2)
        step(2, 3)
        np.testing.assert_array_equal(worker.calls[-1][1]["prev_action"], np.full(19, 10))
        worker.finish(99)
        step(3, 4)

    rollout.step = tick
    result = rollout.run(poll_operator=Mock())
    assert result.termination_details == reason and not result.errors
    assert rollout.finished and rollout.runtime_state.stop_signal.is_set()
    assert rollout.prev_action is None and not rollout.history
    assert recorder.starts == [0] and not recorder.rows[-1][2]
    np.testing.assert_array_equal(recorder.rows[-1][1], np.full(19, 99))
    assert robot.send_action.call_count == 2
    robot.stop.assert_called_once()


@pytest.mark.parametrize("mode", ["async", "rtc"])
def test_prefetch_rollout_stops_and_preserves_inflight_prediction_error(episode, mode):
    rollout, worker, recorder, robot, step = episode
    rollout.execution.mode = mode
    rollout.execution.action_steps = 2
    rollout.execution.prefetch_steps = 1
    rollout.execution.rtc_guidance_cap = 10.0
    rollout.policy_info.requires_prev_action = False
    rollout.policy_info.future_steps = 4
    reason = "XHand position delivery unconfirmed after 2 consecutive write CRC replies"
    robot.send_action.side_effect = [None, HandCommandUnconfirmed(reason)]

    step(0, 1)
    worker.finish(10, steps=4)
    step(1, 2)
    step(2, 3)
    old_future = worker.future
    assert old_future.running()
    assert rollout.reset_pending and rollout.plan is None and rollout.query is None
    assert rollout.finished and rollout.runtime_state.stop_signal.is_set()
    assert rollout.result.termination_details == reason
    assert not recorder.rows[-1][2]
    np.testing.assert_array_equal(recorder.rows[-1][1], np.full(19, 10))
    assert [kwargs["reset_episode"] for _, kwargs in worker.calls] == [True, False]

    step(3, 4)
    assert len(worker.calls) == 2 and robot.send_action.call_count == 2
    assert recorder.starts == [0] and len(recorder.rows) == 3
    assert worker.future is old_future
    old_future.set_exception(OSError("inflight prediction failed"))
    with pytest.raises(OSError, match="inflight prediction failed"):
        rollout.poll_model()


@pytest.mark.parametrize(("mode", "end_slot"), [("sync", 3), ("async", 2), ("rtc", 3)])
def test_late_prediction_ends_episode_even_with_fresh_rows(episode, mode, end_slot):
    rollout, worker, recorder, robot, step = episode
    rollout.execution.mode = mode
    rollout.policy_info.requires_prev_action = False
    rollout.policy_info.future_steps = 2
    step(0, 1, phase_ns=7)
    assert rollout.query.slot == 0
    assert rollout.query.observation_ns == rollout.started_ns + 7
    assert rollout.query.deadline_ns == rollout.started_ns + end_slot * rollout.dt_ns
    for slot in range(1, end_slot):
        step(slot, slot + 1)
    worker.finish(99, steps=2)
    step(end_slot, end_slot + 1)

    assert rollout.result.termination_details == "prediction_expired"
    assert len(worker.calls) == 1
    robot.send_action.assert_not_called()
    assert np.isnan(recorder.rows[-1][1]).all() and not recorder.rows[-1][2]


def test_expired_prediction_still_propagates_inference_failure(episode):
    rollout, worker, _, robot, step = episode
    rollout.policy_info.requires_prev_action = False
    step(0, 1)
    worker.future.set_exception(OSError("prediction failed"))
    with pytest.raises(OSError, match="prediction failed"):
        step(2, 2)
    robot.send_action.assert_not_called()


def test_mapping_cannot_submit_at_prediction_deadline(episode):
    rollout, worker, recorder, robot, step = episode
    step(0, 1)
    worker.finish(10)
    step(1, 2, map_delay_ns=rollout.dt_ns)

    assert rollout.result.termination_details == "prediction_expired"
    robot.send_action.assert_not_called()
    assert np.isnan(recorder.rows[-1][1]).all() and not recorder.rows[-1][2]


def test_mapping_at_cutoff_keeps_ready_continuation_for_next_tick(episode, monkeypatch):
    rollout, worker, recorder, robot, step = episode
    rollout.execution.mode = "async"
    rollout.execution.action_steps = 2
    rollout.execution.prefetch_steps = 1
    rollout.policy_info.requires_prev_action = False
    rollout.policy_info.future_steps = 4
    step(0, 1)
    worker.finish(10, steps=4)
    step(1, 2)
    map_command = rollout.map_command

    def complete_during_mapping(action, row):
        command = map_command(action, row)
        worker.finish(20, steps=4)
        return command

    with monkeypatch.context() as context:
        context.setattr(rollout, "map_command", complete_during_mapping)
        step(2, 3, map_delay_ns=2 * rollout.dt_ns)

    assert not rollout.finished and rollout.plan.prediction.slot == 2
    robot.send_action.assert_called_once()
    assert np.isnan(recorder.rows[-1][1]).all() and not recorder.rows[-1][2]
    step(4, 4)
    np.testing.assert_array_equal(recorder.rows[-1][1], np.full(19, 20))
    assert recorder.rows[-1][2]


@pytest.mark.parametrize(("future_steps", "action_steps"), [(1, 1), (4, 2)])
def test_sync_submits_configured_steps_with_original_device_lease(episode, future_steps, action_steps):
    rollout, worker, recorder, robot, step = episode
    rollout.policy_info.future_steps = future_steps
    rollout.execution.action_steps = action_steps
    step(0, 1)
    worker.finish(10, steps=future_steps)
    for slot in range(1, action_steps + 1):
        step(slot, slot + 1)

    assert [call.kwargs["expires_ns"] for call in robot.send_action.call_args_list] == [
        rollout.started_ns + slot * rollout.dt_ns + COMMAND_LEASE_NS
        for slot in range(1, action_steps + 1)
    ]
    assert rollout.plan is None and not rollout.finished
    assert [accepted for _, _, accepted in recorder.rows] == [False, *([True] * action_steps)]


@pytest.mark.parametrize(("mode", "first_actions", "handoff_action"), [
    ("async", [11, 12], 22), ("rtc", [10, 11], 21),
])
def test_prefetch_keeps_original_window_and_mode_specific_progress(
    episode, mode, first_actions, handoff_action,
):
    rollout, worker, recorder, robot, step = episode
    rollout.execution.mode = mode
    rollout.execution.action_steps = 2
    rollout.execution.prefetch_steps = 1
    rollout.execution.rtc_guidance_cap = 10.0
    rollout.policy_info.requires_prev_action = False
    rollout.policy_info.future_steps = 4

    step(0, 1)
    worker.future.set_result(np.repeat(np.arange(10, 14)[:, None], 19, axis=1))
    step(1, 2)
    step(2, 3)
    query = rollout.query
    assert query.slot == 2
    step(3, 4)
    assert robot.send_action.call_count == 2 and not rollout.finished
    worker.future.set_result(np.repeat(np.arange(20, 24)[:, None], 19, axis=1))
    step(4, 5)

    assert rollout.plan.prediction is query
    assert query.deadline_ns == rollout.started_ns + 6 * rollout.dt_ns
    assert [attempt[0] for _, attempt, accepted in recorder.rows if accepted] == [
        *first_actions, handoff_action,
    ]
    step(6, 7)
    assert rollout.result.termination_details == "prediction_expired"
    assert robot.send_action.call_count == 3
    assert np.isnan(recorder.rows[-1][1]).all() and not recorder.rows[-1][2]


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
