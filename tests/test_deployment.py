import json
from concurrent.futures import Future
from dataclasses import replace
from itertools import count
from types import SimpleNamespace

import numpy as np
import pytest

from dexmani_real.control.actions import ActionResult
from dexmani_real.data.recorder import RecordingError
from dexmani_real.deployment.config import ExecutionConfig
from dexmani_real.deployment.inference import PolicyWorker, PredictionResult
from dexmani_real.deployment.rollout import Rollout
from dexmani_real.deployment.schedule import (
    ActionChunk,
    Prediction,
    control_slot,
    handoff_plan,
    plan_rejection,
)
from dexmani_real.deployment.session import PolicySession
from dexmani_real.robot.commands import RobotError
from dexmani_real.runtime.observation import ObservationNeeds
from dexmani_real.runtime.preflight import validate_control_rate, validate_robot_config


def make_rollout(env, clock, *, mode="sync", action_steps=1, dt=0.0625, delay_ms=20,
                 hand_history_steps=None, control_tactile=(), record_tactile=()):
    runtime = replace(env.runtime, execution=ExecutionConfig(
        mode=mode, action_steps=action_steps, prefetch_steps=1, rtc_guidance_cap=1,
    ))
    info = SimpleNamespace(n_obs_steps=1, hand_history_steps=hand_history_steps,
                           control_dt_s=dt, action_mode="joint", task_name="test")
    projected, predictions = [], []

    def project(row):
        projected.append(row)
        return SimpleNamespace(source_timestamp_ns=int(row.hand["timestamp_ns"][0]),
                               values={"hand_qpos": row.hand["qpos"][0]})

    class Model:
        task = None

        @property
        def has_pending_task(self):
            return self.task is not None

        def reset(self):
            self.task = ("reset", clock.now, None)

        def predict(self, rows, **kwargs):
            actions = np.tile(np.r_[runtime.arm.home_qpos,
                                   np.deg2rad(runtime.hand.home_qpos_deg)], (12, 1))
            actions[:, 0] += len(predictions) * 0.01 + np.arange(12) * 0.001
            actions[:, 7] += len(predictions) * 0.01 + np.arange(12) * 0.001
            predictions.append((clock.now, rows, kwargs, actions))
            self.task = ("predict", clock.now + round(delay_ms * 1e6),
                         PredictionResult(actions, round(delay_ms * 1e6)))

        def poll(self):
            if self.task is None or clock.now < self.task[1]:
                return None
            operation, _, result = self.task
            self.task = None
            return operation, result

    mapper = SimpleNamespace(
        map_joint=lambda arm, hand, current: ActionResult(arm, hand),
        control_from_command=lambda command, mode: np.r_[command.arm_qpos, command.hand_qpos],
        frozen_rejection=lambda *args: None,
    )
    control_needs = ObservationNeeds(required_tactile=control_tactile)
    record_needs = ObservationNeeds(required_tactile=record_tactile)
    rollout = Rollout(env.state, runtime, info, preparing_epoch=env.state.run_id,
                      robot=env.robot, observations=SimpleNamespace(needs=control_needs, project=project),
                      record_needs=record_needs, needs=control_needs.union(record_needs),
                      sensors=SimpleNamespace(channels=None), action_mapper=mapper, model=Model(),
                      max_running_s=None, capture_ids=count(1), index=1, output_dir=None,
                      record_raw=False)
    rollout.step()  # 首次反馈与模型 reset。
    rollout.step()  # 准入和第一次采样。
    return rollout, projected, predictions


def tick(env, rollout, clock, ms):
    clock.at(ms)
    env.robot.service_feedback()
    rollout.step()


@pytest.mark.parametrize("control_tactile,record_tactile", [
    (("dense",), ()), ((), ("aggregate", "dense")),
])
def test_rollout_preparation_waits_for_policy_and_recording_tactile(
    robot_fixture, clock, monkeypatch, control_tactile, record_tactile,
):
    env = robot_fixture
    original = env.robot.hand.get_state

    def read():
        sample = original()
        if clock.now < clock.origin + 70_000_000:
            sample.tactile_dense = None
        return sample

    monkeypatch.setattr(env.robot.hand, "get_state", read)
    rollout, projected, _ = make_rollout(env, clock, control_tactile=control_tactile,
                                        record_tactile=record_tactile)
    assert not rollout.finished and len(projected) == 1
    assert rollout.started_ns >= clock.origin + 70_000_000


def test_async_skips_time_slots_without_reanchoring():
    dt = 62_500_000
    slot, due, skipped = control_slot(now_ns=2 * dt, next_step_ns=dt, last_slot=0,
                                      dt_ns=dt, synchronous=False)
    assert (slot, due, skipped) == (2, 3 * dt, 1)
    actions = np.arange(12).reshape(6, 2)
    for handoff in (None, 1):
        plan, issue = handoff_plan(Prediction(1, 0, 10 * dt, handoff, actions),
                                   run_id=1, slot=slot, mode="async", action_steps=2,
                                   now_ns=2 * dt, authorized=True)
        assert issue is None
        np.testing.assert_array_equal(plan.actions[slot - plan.anchor_slot], actions[2])
        assert plan_rejection(plan, 6, "async", 3 * dt) == "plan_exhausted_without_handoff"
        assert plan_rejection(plan, 2, "async", 10 * dt) == "submission_deadline_expired"
    assert handoff_plan(Prediction(1, 0, 10 * dt, 1, actions), run_id=2, slot=2,
                        mode="async", action_steps=2, now_ns=2 * dt, authorized=True) == (None, None)


def test_async_actual_rollout_skips_one_slot(robot_fixture, clock):
    env = robot_fixture
    rollout, _, predictions = make_rollout(env, clock, mode="async", action_steps=3)
    tick(env, rollout, clock, 62.5)
    tick(env, rollout, clock, 187.5)  # 不执行 slot 2。
    assert not rollout.finished
    assert rollout.result.metrics["skipped_slots"] == 1
    assert len(env.sent) == 2
    np.testing.assert_array_equal(env.sent[-1][1].arm_qpos, predictions[0][3][3, :7])


def test_sync_cadence_and_single_observation_per_step(robot_fixture, clock):
    env = robot_fixture
    rollout, projected, predictions = make_rollout(env, clock, hand_history_steps=2)
    for ms in np.arange(0, 210.5, 0.5):
        tick(env, rollout, clock, ms)
    assert not rollout.finished
    np.testing.assert_array_equal([(t - clock.origin) / 1e6 for t, _, _ in env.sent],
                                  [20, 82.5, 145, 207.5])
    assert len(projected) == len(predictions) == len(env.sent) == 4
    np.testing.assert_array_equal(predictions[1][2]["action_context"]["hand_history"][-1],
                                  env.sent[0][1].hand_qpos)
    assert predictions[2][2]["action_context"]["hand_history"][-1, 0] != env.robot.read_state().hand["qpos"][0, 0]


@pytest.mark.parametrize("delay_ms, expected", [(20, [20, 62.5, 125, 207.5]),
                                               (80, [80, 125, 187.5, 330])])
def test_sync_chunks_and_overrun(robot_fixture, clock, delay_ms, expected):
    env = robot_fixture
    rollout, projected, predictions = make_rollout(env, clock, action_steps=3, delay_ms=delay_ms)
    for ms in np.arange(0, expected[-1] + 0.5, 0.5):
        tick(env, rollout, clock, ms)
    assert not rollout.finished
    np.testing.assert_array_equal([(t - clock.origin) / 1e6 for t, _, _ in env.sent], expected)
    assert len(projected) == 4 and len(predictions) == 2
    for index in range(3):
        np.testing.assert_array_equal(env.sent[index][1].arm_qpos, predictions[0][3][index, :7])
    assert env.sent[0][2]["execute_until_ns"] == predictions[0][1][-1].source_timestamp_ns + 1_000_000_001


def test_reset_and_cancel_during_prediction(robot_fixture, clock):
    env = robot_fixture
    old, _, _ = make_rollout(env, clock)
    env.state.request_stop()
    tick(env, old, clock, 10)
    assert old.finished and not env.sent
    tick(env, old, clock, 20)
    assert not old.model.has_pending_task
    new, projected, _ = make_rollout(env, clock)
    tick(env, new, clock, 40)
    assert not new.finished and len(env.sent) == 1 and len(projected) == 1


def test_five_hz_and_recording_failure_still_stops(robot_fixture, clock, monkeypatch):
    env = robot_fixture
    validate_robot_config(env.runtime)
    validate_control_rate(env.runtime, 5)
    rollout, _, _ = make_rollout(env, clock, dt=0.2)
    for ms in np.arange(0, 440, 5):
        tick(env, rollout, clock, ms)
    assert len(env.sent) == 3 and not rollout.finished
    assert env.robot.hand.reads > 10

    class Recorder:
        def check_error(self):
            raise RecordingError("writer broke")

    rollout.recorder = Recorder()
    with pytest.raises(RecordingError, match="writer broke"):
        tick(env, rollout, clock, 445)
    assert rollout.result.termination_reason == "recording_failure"
    assert env.state.stop_signal.is_set()


def test_rtc_frozen_prefix_and_missed_slot(robot_fixture, clock):
    env = robot_fixture
    rollout, _, predictions = make_rollout(env, clock, mode="rtc", action_steps=2,
                                          hand_history_steps=2)
    tick(env, rollout, clock, 62.5)
    tick(env, rollout, clock, 125)
    prefix = predictions[1][2]["rtc_prefix"]
    np.testing.assert_array_equal(prefix[0], np.r_[env.sent[-1][1].arm_qpos, env.sent[-1][1].hand_qpos])
    np.testing.assert_array_equal(predictions[1][2]["action_context"]["hand_history"][-1],
                                  env.sent[0][1].hand_qpos)
    tick(env, rollout, clock, 250)
    assert rollout.finished and "missed_slot" in rollout.result.termination_details
    assert len(env.sent) == 2


def test_rtc_late_handoff_and_frozen_recheck():
    query = Prediction(1, 0, 1000, 2, np.zeros((5, 19)))
    assert handoff_plan(query, run_id=1, slot=3, mode="rtc", action_steps=2,
                        now_ns=10, authorized=True)[1] == "handoff_slot_missed"
    plan = ActionChunk(np.zeros((5, 19)), 0, 2, 1000, {})
    assert plan_rejection(plan, 2, "rtc", 10) == "plan_exhausted_without_handoff"


def test_prediction_timing_includes_builder(clock):
    worker = PolicyWorker.__new__(PolicyWorker)

    def build(rows):
        clock.sleep(0.01)
        return rows

    def predict(rows):
        clock.sleep(0.02)
        return np.zeros((1, 19))

    worker.build_observation = build
    worker.model = SimpleNamespace(predict=predict)
    assert worker._predict(()).duration_ns == 30_000_000


def test_inference_outlives_feedback_age_without_refreshing_decision(robot_fixture, clock):
    env = robot_fixture
    rollout, projected, predictions = make_rollout(env, clock, delay_ms=250, dt=0.2)
    for ms in range(0, 251, 5):
        tick(env, rollout, clock, ms)
    assert not rollout.finished and len(env.sent) == len(projected) == 1
    assert env.sent[0][2]["execute_until_ns"] == predictions[0][1][0].source_timestamp_ns + 1_000_000_001
    assert env.robot.hand.reads >= 7


def test_recording_requires_both_tactile_channels(robot_fixture, clock, monkeypatch):
    env = robot_fixture
    rollout, _, _ = make_rollout(env, clock)
    rollout.record_needs = ObservationNeeds(required_tactile=("aggregate", "dense"))
    rollout.recorder = SimpleNamespace()
    row = rollout.read_observation()[0]
    broken = row.hand.copy()
    broken["tactile_dense_timestamp_ns"] = 0
    row = replace(row, hand=broken)
    monkeypatch.setattr("dexmani_real.deployment.rollout.read_observation", lambda *args, **kwargs: row)
    with pytest.raises(RecordingError, match="tactile_dense"):
        rollout.read_observation()


def test_frozen_command_is_rechecked_against_new_state(robot_fixture, clock):
    env = robot_fixture
    rollout, _, _ = make_rollout(env, clock, mode="rtc", action_steps=2)
    tick(env, rollout, clock, 62.5)
    tick(env, rollout, clock, 125)
    frozen = rollout.plan.frozen[2]
    current = env.robot.read_state()
    seen = []

    def reject(command, qpos, mode):
        seen.append((command, qpos))
        return "frozen_equivalent_branch_changed"

    rollout.action_mapper.frozen_rejection = reject
    command, issue = rollout.command_for_slot(current, 2)
    assert command is None and "frozen_equivalent_branch_changed" in issue
    assert seen[0][0] is frozen
    np.testing.assert_array_equal(seen[0][1], current.arm["qpos"][0])


@pytest.mark.parametrize("broken_source", ["camera", "aggregate", "dense"])
def test_recording_source_failure_during_sync_wait_stops(robot_fixture, clock, monkeypatch, broken_source):
    env = robot_fixture
    rollout, projected, _ = make_rollout(env, clock, delay_ms=250)
    original_row = rollout.sync_row
    appended = []
    rollout.record_needs = ObservationNeeds(camera_fields=("rgb", "depth"),
                                            required_tactile=("aggregate", "dense"))
    rollout.recorder = SimpleNamespace(check_error=lambda: None,
                                       append=lambda row, command: appended.append(row))
    monkeypatch.setattr("dexmani_real.deployment.rollout.read_camera_frame",
                        lambda *args, **kwargs: None if broken_source == "camera" else {"timestamp_ns": clock.now})
    clock.at(62.5)
    env.robot.service_feedback()
    if broken_source != "camera":
        frame = env.robot.hand_feedback.latest.copy()
        frame[f"tactile_{broken_source}_timestamp_ns"] = 0
        env.robot.hand_feedback.latest = frame
    with pytest.raises(RecordingError):
        rollout.step()
    assert rollout.finished and rollout.result.termination_reason == "recording_failure"
    assert not env.sent and len(projected) == 1 and appended == [original_row]


def test_session_services_feedback_and_cancel_while_inference_pending(robot_fixture, clock):
    env = robot_fixture
    rollout, projected, _ = make_rollout(env, clock, dt=0.2, delay_ms=250)
    session = PolicySession.__new__(PolicySession)
    session.runtime_state, session.robot = env.state, env.robot
    session.episode, session.model = rollout, rollout.model
    session.episodes, session.errors = [], []
    session.start_epoch = session.num_episodes = None
    session.keyboard = SimpleNamespace(discard_events=lambda *args: None)
    session.save_results = lambda: None

    def poll_operator(**kwargs):
        if clock.now - clock.origin >= 150_000_000:
            env.state.request_quit()

    session.poll_operator = poll_operator
    session.run_control_loop()
    assert env.robot.hand.reads >= 5 and not env.sent and len(projected) == 1
    assert rollout.finished and rollout.model.has_pending_task
    assert len(session.episodes) == 1  # 尚未完成的推理仍由该轮持有。
    assert session.episode is rollout


def test_session_stops_when_feedback_really_ceases(robot_fixture, clock):
    env = robot_fixture
    rollout, _, _ = make_rollout(env, clock, dt=0.2)
    session = PolicySession.__new__(PolicySession)
    session.runtime_state, session.robot = env.state, env.robot
    session.episode, session.model = rollout, rollout.model
    session.poll_operator = lambda **kwargs: None
    session.advance_episode = rollout.step
    env.robot.hand.crc = True
    with pytest.raises(RobotError, match="unavailable/stale"):
        session.run_control_loop()
    assert env.state.stop_signal.is_set()
    assert clock.now - clock.origin < 200_000_000


def test_retired_model_failure_is_recorded_once_without_replacing_stop(robot_fixture, clock, tmp_path):
    env = robot_fixture
    rollout, _, _ = make_rollout(env, clock)
    worker = PolicyWorker.__new__(PolicyWorker)
    worker.future, worker.operation = Future(), "predict"
    future = worker.future
    rollout.model = worker
    session = PolicySession.__new__(PolicySession)
    session.runtime_state, session.robot = env.state, env.robot
    session.episode, session.model = rollout, worker
    session.episodes, session.errors = [], []
    session.start_epoch = session.num_episodes = None
    session.keyboard = SimpleNamespace(discard_events=lambda *args: None)
    session.summary, session.output_dir = {}, tmp_path
    env.state.request_stop()
    session.advance_episode()
    assert session.episode is rollout and rollout.result.termination_reason == "operator"
    assert rollout.result.errors == []
    original = ValueError("late prediction failed")
    future.set_exception(original)
    session.advance_episode()
    session.advance_episode()
    session.save_results()
    assert session.episode is None and env.state.error_state
    assert rollout.result.termination_reason == "operator"
    assert rollout.result.errors == [dict(stage="retired_model", exception_type="ValueError",
                                         message="late prediction failed")]
    saved = json.loads((tmp_path / "session_result.json").read_text())
    assert saved["episodes"][0]["errors"] == rollout.result.errors and not session.errors


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_final_frame_failure_still_finishes_recording(robot_fixture, clock, cleanup_fails):
    env = robot_fixture
    rollout, _, _ = make_rollout(env, clock)
    original = RecordingError("last row failed")
    calls = []

    def append(row):
        calls.append("append")
        raise original

    def finish(**kwargs):
        calls.append("finish")
        assert kwargs["technical_status"] == "failed"
        assert kwargs["details"][-1]["message"] == "last row failed"
        if cleanup_fails:
            raise OSError("writer close failed")

    rollout.recorder = SimpleNamespace(start_attempted=True, append=append, finish=finish,
                                       result=lambda: {"status": "failed"})
    env.state.request_stop()
    rollout.stop("operator stop during inference")
    assert rollout.result.termination_reason == "operator"
    with pytest.raises(RecordingError) as failure:
        rollout.finish()
    assert failure.value is original
    assert calls == ["append", "finish"] and rollout.sync_row is None
    assert rollout.result.raw == {"status": "failed"}
    assert len(rollout.result.errors) == (2 if cleanup_fails else 1)
