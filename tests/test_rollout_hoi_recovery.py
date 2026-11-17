"""在线 HOI 恢复与录制收尾；不加载模型或连接设备。"""

from concurrent.futures import Future
from copy import deepcopy
from dataclasses import replace
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.config.deployment import ExecutionConfig
from dexmani_real.config.real import RealConfig
from dexmani_real.data.raw import episode_recording_metadata
from dexmani_real.data.recorder import EpisodeRecorder, RecordingError
from dexmani_real.deployment import rollout as rollout_module
from dexmani_real.deployment.inference import PolicyWorker
from dexmani_real.deployment.observation import ObservationBuilder
from dexmani_real.deployment.rollout import Rollout
from dexmani_real.deployment.schedule import ActionChunk, Prediction
from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
from dexmani_real.robot.commands import (
    COMMAND_LEASE_NS,
    FeedbackUnavailable,
    OperationCancelled,
    RobotError,
)
from dexmani_real.runtime.observation import (
    RAW_RECORDING_NEEDS,
    ObservationNeeds,
    is_recoverable_hoi_failure,
)
from dexmani_real.runtime.state import RunEndReason, SessionState


class FakeRing:
    def __init__(self, data):
        self.data = data
        self.latest_sequence = 1
        self.writes = []
        self.reads = []

    def read(self, *, fields=None):
        self.reads.append(fields)
        if self.data is None:
            return None
        data = dict(self.data) if fields is None else {key: self.data[key] for key in fields}
        return data, self.latest_sequence

    def write(self, **data):
        self.data = data
        self.writes.append(data)


class FakePolicyWorker:
    poll = PolicyWorker.poll

    def __init__(self):
        self.future = None
        self.calls = []
        self.reset_count = 0
        self.model = SimpleNamespace(reset_episode=self.reset, predict=lambda obs, **kw: obs)
        self.build_observation = lambda rows: rows

    def reset(self):
        self.reset_count += 1

    def predict(self, rows, **kwargs):
        assert self.future is None
        self.calls.append((rows, kwargs))
        self.future = Future()

    def complete(self):
        # 使用真正的 reset/predict 调用路径，但不启动推理线程。
        rows, kwargs = self.calls[-1]
        PolicyWorker._predict(self, rows, **kwargs)
        self.future.set_result(np.full((100, 19), 0.2))

    def warmup(self, *args, **kwargs):
        pytest.fail("recovery must not warm up")


class FakeRecorder:
    def __init__(self):
        self.gaps = 0
        self.rows = []

    def check_error(self):
        pass

    def mark_gap(self):
        self.gaps += 1

    def append(self, row, command, **kwargs):
        self.rows.append((row, command))


class FakeRobot:
    def __init__(self, clock, state):
        self.clock, self.state = clock, state
        self.sent = []
        self.stops = self.services = 0
        self.feedback_next_ns = 0
        self.error = self.send_error = None
        self.arm_age = self.hand_age = 0
        self.missing_hand = False
        self.aligned = True
        self.hand_qpos = np.full(12, 0.3)

    def enable_joint_history(self, *args, **kwargs):
        pass

    def check(self):
        if self.error:
            raise self.error

    def service_feedback(self):
        self.check()
        self.services += 1
        self.feedback_next_ns = self.clock.now + 10_000_000

    def read_state(self):
        def frame(size, age):
            value = np.zeros(1, dtype=[
                ("timestamp_ns", "i8"), ("qpos", "f8", (size,)),
                ("tactile_aggregate_timestamp_ns", "i8"),
                ("tactile_dense_timestamp_ns", "i8"),
            ])
            value["timestamp_ns"] = self.clock.now - age
            value["tactile_aggregate_timestamp_ns"] = self.clock.now
            value["tactile_dense_timestamp_ns"] = self.clock.now
            return value

        arm, hand = frame(7, self.arm_age), frame(12, self.hand_age)
        hand["qpos"][0] = self.hand_qpos
        return SimpleNamespace(arm=arm, hand=None if self.missing_hand else hand)

    def aligned_joints(self, *args):
        return {"arm_qpos": np.zeros(7), "hand_qpos": self.hand_qpos.copy()} if self.aligned else None

    def send_action(self, command, *, expires_ns):
        if self.send_error:
            raise self.send_error
        self.sent.append((self.clock.now, command, expires_ns))

    def stop(self):
        self.stops += 1
        self.state.stop_signal.set()


@pytest.fixture
def record_raw():
    return True


@pytest.fixture
def env(monkeypatch, record_raw):
    clock = SimpleNamespace(now=10_000_000_000)

    def sleep(seconds):
        clock.now += round(seconds * 1e9)

    monkeypatch.setattr(rollout_module.time, "monotonic_ns", lambda: clock.now)
    monkeypatch.setattr(rollout_module.time, "sleep", sleep)
    state = SessionState(stop_signal=Event(), emergency_signal=Event())
    robot = FakeRobot(clock, state)
    channels = SimpleNamespace(
        hoi_request=FakeRing({"episode_id": 7, "after_ns": 1, "after_sequence": 0}),
        hoi_ring=FakeRing(None), camera_ring=FakeRing(None),
    )
    runtime = RealConfig(execution=ExecutionConfig(control_dt_s=0.05, action_steps=8))
    info = SimpleNamespace(
        observation_fields=("object_pointcloud", "joint_state"),
        n_obs_steps=2, action_mode="joint", requires_prev_action=True,
    )
    observations = ObservationBuilder(info, runtime)
    record_needs = RAW_RECORDING_NEEDS if record_raw else ObservationNeeds()
    worker, recorder = FakePolicyWorker(), FakeRecorder()
    worker.build_observation = observations.build
    rollout = Rollout(
        state, runtime, info,
        episode_cancel_epoch=state.cancel_epoch, robot=robot, sensors=SimpleNamespace(channels=channels),
        observations=observations,
        read_needs=replace(
            observations.needs.union(record_needs),
            camera_fields=observations.needs.camera_fields,
        ),
        record_needs=record_needs,
        action_mapper=SimpleNamespace(map_joint=lambda arm, hand: SimpleNamespace(
            arm_qpos=arm, hand_qpos=hand, rejection_reason=None)),
        model=worker, max_duration_s=60, index=7, output_dir=None, record_raw=record_raw,
        effective_parameters={},
    )
    rollout.recorder = recorder if record_raw else None
    rollout.started_ns = clock.now

    def publish(valid=True):
        channels.camera_ring.data = {
            "timestamp_ns": clock.now, "rgb": np.zeros((2, 2, 3), np.uint8),
            "depth": np.ones((2, 2), np.uint16),
        }
        channels.hoi_ring.data = {
            "episode_id": 7, "timestamp_ns": clock.now, "source_camera_sequence": 1,
            "valid": valid, "object_pointcloud": np.full((4, 3), clock.now, dtype=float),
        }

    def advance(seconds=0.05, valid=True):
        clock.now += round(seconds * 1e9)
        publish(valid)

    publish()
    return SimpleNamespace(
        rollout=rollout, robot=robot, worker=worker, recorder=recorder,
        clock=clock, state=state, channels=channels, publish=publish, advance=advance,
    )


def enter_hold(env):
    env.publish(False)
    env.rollout.step()
    assert env.rollout.state == "HOI_HOLD"


def ready_to_send(env):
    enter_hold(env)
    env.advance()
    env.rollout.step()
    env.advance()
    env.rollout.step()
    env.worker.complete()
    env.advance()


@pytest.mark.parametrize("issues,expected", [
    ([], False), (["hoi_unavailable"], True), (["hoi_pose_invalid", "hoi_stale"], True),
    (["hoi_joint_alignment_unavailable"], True),
    (["hoi_stale", "arm_feedback_stale"], False),
    (["hand_feedback_unavailable"], False), (["pointcloud_stale"], False),
])
def test_classification(issues, expected):
    assert is_recoverable_hoi_failure(issues) is expected


def test_initial_registration_waits(env):
    env.publish(False)
    polls = 0

    def poll():
        nonlocal polls
        polls += 1
        if polls == 4:
            env.publish()

    assert env.rollout.start_observation(poll, initial_hoi=True) is not None
    assert polls == 4
    assert env.channels.hoi_request.data["episode_id"] == 7
    assert not env.channels.hoi_request.writes


@pytest.mark.parametrize("writer_fails", [False, True])
def test_preparation_records_parameters_and_starts_budget(env, monkeypatch, tmp_path, writer_fails):
    rollout = env.rollout
    rollout.started_ns = 0
    rollout.output_dir = tmp_path
    rollout.policy_info.task_name = "test"
    rollout.camera_snapshot = CameraSnapshot(
        "camera-123", CameraIntrinsics(16, 16, 20., 20., 8., 8., "none", (0.,) * 5), .001, np.eye(4),
    )
    rollout.effective_parameters = {
        "policy": {"task_name": "test"},
        "actual_processing": {"object_pointcloud": {"max_joint_gap_s": .1}},
    }
    original = deepcopy(rollout.effective_parameters)
    env.robot.hand = SimpleNamespace(tactile_reset_metadata={
        "hand_tactile_reset_time_ns": 123456789, "hand_tactile_reset_result": {2: 0},
    })

    def poll():
        env.channels.camera_ring.latest_sequence += 1
        # 准备耗时超过 60 秒，也不能提前消耗正式运行预算。
        env.advance(61)
        env.channels.hoi_ring.data["source_camera_sequence"] = env.channels.camera_ring.latest_sequence
        assert rollout.started_ns == 0
        assert not rollout.duration_expired()

    env.worker.warmed_up = False
    env.worker.warmup = lambda *a, **kw: poll()
    env.robot.prepare_streaming = Mock(side_effect=lambda *a, **kw: env.state.stop_signal.clear())
    failure = RecordingError("writer failed")
    recorder = Mock(start=Mock(side_effect=failure if writer_fails else None))
    constructor = Mock(return_value=recorder)
    monkeypatch.setattr(rollout_module, "EpisodeRecorder", constructor)

    if writer_fails:
        with pytest.raises(RecordingError) as caught:
            rollout.prepare(poll)
        assert caught.value is failure
        env.robot.prepare_streaming.assert_not_called()
    else:
        rollout.prepare(poll)
        env.robot.prepare_streaming.assert_called_once()
        assert rollout.started_ns == env.clock.now
        assert not rollout.duration_expired()
    assert env.worker.warmed_up
    constructor.assert_called_once_with(tmp_path, control_hz=20.)
    recorder.start.assert_called_once_with(
        task_label="test", episode_name="episode_007", collection_source="policy_rollout",
        metadata=episode_recording_metadata(
            rollout.camera_snapshot, original, tactile_reset=env.robot.hand.tactile_reset_metadata,
        ),
    )
    assert rollout.effective_parameters == original


def test_hold_clears_context_and_refreshes_measured_hand(env):
    rollout = env.rollout
    rollout.history.append({"old": np.ones(1)})
    rollout.plan, rollout.query = ActionChunk(np.ones((8, 19)), 0), Prediction(0, None)
    rollout.cursor, rollout.prev_action, rollout.ik_reference = 3, np.ones(19), object()
    rollout.first_prediction = False
    enter_hold(env)
    assert not rollout.history
    assert rollout.plan is rollout.query is None
    assert rollout.prev_action is rollout.ik_reference is None
    assert rollout.cursor == 0 and rollout.first_prediction
    assert not env.state.stop_signal.is_set() and env.robot.stops == 0
    assert env.channels.hoi_request.data["episode_id"] == 7
    env.advance(0.99, False)
    rollout.step()
    assert len(env.robot.sent) == 1
    env.robot.hand_qpos[:] = 0.4
    env.advance(0.01, False)
    rollout.step()
    assert len(env.robot.sent) == 2
    for stamp, cmd, expiry in env.robot.sent:
        assert cmd.arm_hold and cmd.arm_qpos is None
        assert expiry == stamp + COMMAND_LEASE_NS
    np.testing.assert_array_equal(env.robot.sent[-1][1].hand_qpos, np.full(12, 0.4))
    assert env.recorder.gaps == 1 and not env.recorder.rows


def test_hand_disabled(env):
    env.rollout.runtime = replace(env.rollout.runtime, hand=replace(env.rollout.runtime.hand, enabled=False))
    env.robot.missing_hand = True
    env.rollout.record_needs = ObservationNeeds(camera_fields=("rgb", "depth"))
    enter_hold(env)
    assert env.robot.sent[0][1].hand_qpos is None


@pytest.mark.parametrize("mode", ["sync", "async", "rtc"])
@pytest.mark.parametrize("record_raw", [True, False], ids=["raw", "no_raw"])
def test_complete_recovery_and_exact_action_boundary(env, mode, record_raw):
    env.rollout.execution = replace(env.rollout.execution, mode=mode)
    enter_hold(env)
    env.advance()
    env.rollout.step()
    assert env.rollout.state == "RESUMING" and not env.rollout.history
    env.advance()
    env.rollout.step()
    rows, kwargs = env.worker.calls[0]
    assert kwargs["reset_episode"] and kwargs["prev_action"] is None
    assert len(rows) == 2 and all(row["object_pointcloud"][0, 0] == env.clock.now for row in rows)
    assert not env.recorder.rows and env.rollout.state == "RESUMING"
    env.advance(1)
    env.rollout.step()
    assert env.rollout.state == "RESUMING"
    assert not env.recorder.rows
    assert all(cmd.arm_hold for _, cmd, _ in env.robot.sent)
    env.worker.complete()
    assert env.worker.reset_count == 1
    env.advance()
    env.rollout.step()
    assert env.rollout.state == "RUNNING"
    assert not env.robot.sent[-1][1].arm_hold
    if record_raw:
        assert len(env.recorder.rows) == 1 and env.recorder.rows[0][1] is not None
        assert env.recorder.gaps == 1
    else:
        assert not env.recorder.rows and not env.recorder.gaps
    assert env.channels.hoi_request.data["episode_id"] == 7


def test_old_future_blocks_recovery_and_discards_result(env):
    env.worker.future = old = Future()
    enter_hold(env)
    env.advance()
    env.rollout.step()
    assert env.rollout.state == "HOI_HOLD" and not env.worker.calls
    old.set_result(np.full((8, 19), 99))
    env.rollout.step()
    assert env.rollout.state == "RESUMING"
    assert env.worker.future is None and env.rollout.plan is env.rollout.query is None


def test_old_future_exception_propagates(env):
    env.worker.future = old = Future()
    enter_hold(env)
    old.set_exception(ValueError("inference broke"))
    with pytest.raises(ValueError, match="inference broke"):
        env.rollout.step()


def test_repeated_occlusion_keeps_gap_and_recovers_after_five_seconds(env):
    enter_hold(env)
    env.advance()
    env.rollout.step()
    env.advance()
    env.rollout.step()
    old = env.worker.future
    env.advance(0.1, False)
    env.rollout.step()
    assert env.rollout.state == "HOI_HOLD"
    old.set_result(np.ones((20, 19)))
    env.advance()
    env.rollout.step()
    assert env.rollout.state == "RESUMING" and not env.rollout.history
    assert env.recorder.gaps == 1
    env.advance(6)
    env.rollout.step()
    assert not env.rollout.finished and env.rollout.state == "RESUMING"
    assert not env.recorder.rows
    env.worker.complete()
    env.advance()
    env.rollout.step()
    assert env.rollout.state == "RUNNING" and not env.robot.sent[-1][1].arm_hold
    assert env.recorder.gaps == 1


@pytest.mark.parametrize("unbounded", [False, True])
def test_hold_uses_total_episode_budget(env, unbounded):
    if unbounded:
        env.rollout.max_duration_ns = None
    enter_hold(env)
    env.advance(61, False)
    env.rollout.step()
    if unbounded:
        assert not env.rollout.finished and env.rollout.state == "HOI_HOLD"
        assert all(command.arm_hold for _, command, _ in env.robot.sent)
    else:
        assert env.rollout.result.termination_details == "duration_timeout"
        assert env.rollout.result.termination_reason == "duration_complete"
    assert not env.recorder.rows


@pytest.mark.parametrize("operator_stop", [False, True])
def test_duration_boundary_and_finished_episode(env, operator_stop):
    rollout = env.rollout
    env.clock.now = rollout.started_ns + rollout.max_duration_ns - 1
    assert not rollout.duration_expired()
    assert not rollout.finished
    if operator_stop:
        rollout.stop("operator input", run_end_reason=RunEndReason.OPERATOR)
    env.clock.now += 1
    assert rollout.duration_expired() is (not operator_stop)
    assert rollout.result.termination_reason == ("operator" if operator_stop else "duration_complete")
    assert env.state.stop_signal.is_set()
    result = deepcopy(rollout.result)
    requests = len(env.channels.hoi_request.writes)
    env.advance(60)
    assert not rollout.duration_expired()
    assert rollout.result == result
    assert len(env.channels.hoi_request.writes) == requests


def test_invalid_hoi_skips_alignment_but_checks_required_recording(env, monkeypatch):
    aligned = Mock(wraps=env.robot.aligned_joints)
    monkeypatch.setattr(env.robot, "aligned_joints", aligned)
    enter_hold(env)
    aligned.assert_not_called()
    env.channels.camera_ring.data = None
    env.rollout.step()
    aligned.assert_not_called()
    assert env.rollout.finished


@pytest.mark.parametrize("failure", ["arm", "hand", "missing", "send"])
def test_hold_device_errors_propagate(env, failure):
    if failure == "arm":
        env.robot.arm_age = 10_000_000_000
    elif failure == "hand":
        env.robot.hand_age = 10_000_000_000
    elif failure == "missing":
        env.robot.missing_hand = True
    else:
        env.robot.send_error = RobotError("send failed")
    env.publish(False)
    with pytest.raises((FeedbackUnavailable, RobotError)):
        env.rollout.step()


@pytest.mark.parametrize("issue", ["episode", "stale", "alignment", "unavailable"])
def test_recovery_requires_all_hoi_conditions(env, issue):
    enter_hold(env)
    env.advance()
    if issue == "episode":
        env.channels.hoi_ring.data["episode_id"] = 8
    elif issue == "stale":
        env.channels.hoi_ring.data["timestamp_ns"] -= 1_000_000_000
    elif issue == "alignment":
        env.robot.aligned = False
    else:
        env.channels.hoi_ring.data = None
    env.rollout.step()
    assert env.rollout.state == "HOI_HOLD" and not env.worker.calls


def test_unrelated_modality_failure_not_hidden(env):
    env.rollout.observations.needs = replace(env.rollout.observations.needs, point_cloud=True)
    env.publish(False)
    env.rollout.step()
    assert env.rollout.finished
    assert "pointcloud_unavailable" in env.rollout.result.termination_details


def test_final_check_prevents_action_and_recording(env):
    env.rollout.plan = ActionChunk(np.ones((8, 19)), 0)
    original = env.rollout.map_command

    def mapping(action, row):
        result = original(action, row)
        env.publish(False)
        return result

    env.rollout.map_command = mapping
    env.rollout.step()
    assert env.rollout.state == "HOI_HOLD" and not env.recorder.rows
    assert all(cmd.arm_hold for _, cmd, _ in env.robot.sent)


def test_history_validation_enters_hold_without_recording(env, monkeypatch):
    monkeypatch.setattr(rollout_module, "check_hoi_status", lambda *a, **kw: [])
    env.publish(False)
    env.rollout.step()
    assert env.rollout.state == "HOI_HOLD" and not env.recorder.rows


def test_scheduler_skips_to_current_slot(env):
    enter_hold(env)
    env.advance(0.525, False)
    env.rollout.step()
    assert env.rollout.last_slot == 10 and env.rollout.result.skipped_slots == 9
    assert env.rollout.next_step_ns > env.clock.now
    env.advance(0.025)
    env.rollout.step()
    env.rollout.execution = replace(env.rollout.execution, mode="async")
    env.advance()
    env.rollout.step()
    assert env.rollout.query.slot == 12
    assert len(env.worker.calls) == 1


@pytest.mark.parametrize("cause", ["operator", "estop", "duration", "worker"])
def test_run_cleanup_and_precedence(env, cause):
    enter_hold(env)
    env.rollout.prepare = lambda poll: None

    def poll(**kwargs):
        if cause == "worker":
            env.robot.error = RobotError("worker exited")
        else:
            env.advance(5)
        if cause == "operator":
            env.rollout.stop("operator input", run_end_reason=RunEndReason.OPERATOR)
        elif cause == "estop":
            env.state.request_emergency_stop()
            env.rollout.stop("operator input", run_end_reason=RunEndReason.ESTOP)
        elif cause == "duration":
            env.clock.now = env.rollout.started_ns + env.rollout.max_duration_ns

    if cause == "worker":
        with pytest.raises(RobotError, match="worker exited"):
            env.rollout.run(poll_operator=poll)
    else:
        env.rollout.run(poll_operator=poll)
    assert env.robot.stops == 1 and env.state.stop_signal.is_set()
    assert env.channels.hoi_request.data["episode_id"] == 0
    expected = {
        "operator": "operator", "estop": "estop",
        "duration": "duration_complete", "worker": "failed",
    }
    assert env.rollout.result.termination_reason == expected[cause]


@pytest.mark.parametrize("arm_published", [False, True])
@pytest.mark.parametrize("record_raw", [True, False], ids=["raw", "no_raw"])
def test_failed_resumed_submission_records_attempt_without_running(env, arm_published, record_raw):
    ready_to_send(env)
    error = RobotError("submission failed")
    published_arm = []

    def send(command, **kwargs):
        if arm_published:
            # 模拟 arm 已发布、随后 hand 失败；发布不代表设备执行。
            published_arm.append(command.arm_qpos)
        raise error

    env.robot.send_action = send
    with pytest.raises(RobotError, match="submission failed") as caught:
        env.rollout.step()
    assert caught.value is error
    assert env.rollout.state == "RESUMING"
    assert env.rollout.cursor == 0
    assert env.rollout.prev_action is None
    if arm_published:
        assert len(published_arm) == 1
        np.testing.assert_array_equal(published_arm[0], np.full(7, 0.2))
    if record_raw:
        assert len(env.recorder.rows) == 1
        _, command = env.recorder.rows[0]
        np.testing.assert_array_equal(command.arm_qpos, np.full(7, 0.2))
        np.testing.assert_array_equal(command.hand_qpos, np.full(12, 0.2))
        assert not command.arm_hold
    else:
        assert not env.recorder.rows and not env.recorder.gaps


@pytest.mark.parametrize("resuming", [False, True])
def test_recording_failure_preserves_submission_error(env, resuming):
    if resuming:
        ready_to_send(env)
    else:
        env.rollout.plan = ActionChunk(np.full((8, 19), 0.2), 0)
    env.robot.send_error = error = RobotError("submission failed")

    def append(row, command, **kwargs):
        assert command is not None
        raise RecordingError("append failed")

    env.recorder.append = append
    env.rollout.prepare = lambda poll: None
    with pytest.raises(RobotError, match="submission failed") as caught:
        env.rollout.run(poll_operator=lambda **kwargs: None)
    assert caught.value is error
    assert env.rollout.state == ("RESUMING" if resuming else "RUNNING")
    assert env.rollout.result.termination_reason == "failed"
    assert env.rollout.result.errors == [
        {"stage": "recording_append", "exception_type": "RecordingError", "message": "append failed"},
        {"stage": "control", "exception_type": "RobotError", "message": "submission failed"},
    ]
    assert env.robot.stops == 1 and env.state.stop_signal.is_set()


def test_running_without_submission_still_records(env):
    env.rollout.step()
    assert env.rollout.state == "RUNNING" and not env.robot.sent
    assert len(env.recorder.rows) == 1 and env.recorder.rows[0][1] is None


@pytest.mark.parametrize("failure", ["cancelled", "rejected", "exception", "timeout"])
def test_recovery_before_submission_does_not_record(env, failure):
    ready_to_send(env)
    original = env.rollout.action_mapper.map_joint

    def mapping(arm, hand):
        if failure == "cancelled":
            env.state.request_stop()
            return original(arm, hand)
        if failure == "timeout":
            env.advance(60)
            return original(arm, hand)
        if failure == "rejected":
            return SimpleNamespace(arm_qpos=None, hand_qpos=None, rejection_reason="unreachable")
        raise ValueError("mapping failed")

    env.rollout.action_mapper.map_joint = mapping
    if failure in {"rejected", "timeout"}:
        env.rollout.step()
        expected = "action_rejected: unreachable" if failure == "rejected" else "duration_timeout"
        assert env.rollout.result.termination_details == expected
    else:
        error = OperationCancelled if failure == "cancelled" else ValueError
        with pytest.raises(error):
            env.rollout.step()
    assert env.rollout.state == "RESUMING"
    assert env.rollout.cursor == 0
    assert all(cmd.arm_hold for _, cmd, _ in env.robot.sent)
    assert not env.recorder.rows


@pytest.mark.parametrize("kind", ["camera", "tactile"])
def test_raw_requirements_checked_while_paused(env, kind):
    if kind == "tactile":
        original = env.robot.read_state

        def read_state():
            state = original()
            state.hand["tactile_aggregate_timestamp_ns"] = 0
            return state

        env.robot.read_state = read_state
    env.publish(False)
    if kind == "camera":
        env.channels.camera_ring.data = None
    env.rollout.step()
    assert env.rollout.finished and not env.recorder.rows


def test_hold_reads_image_metadata_then_recovers_from_fresh_full_frame(env):
    rollout = env.rollout
    rollout.policy_info.observation_fields += ("rgb",)
    rollout.observations = ObservationBuilder(rollout.policy_info, rollout.runtime)
    rollout.observations.configure_camera(CameraSnapshot(
        "test", CameraIntrinsics(2, 2, 2., 2., 1., 1., "none", (0.,) * 5), .001, None,
    ))
    env.worker.build_observation = rollout.observations.build
    rollout.read_needs = replace(rollout.read_needs, camera_fields=("rgb",))
    enter_hold(env)
    for _ in range(5):
        env.advance(.001, False)
        rollout.step()
    # 暂停逐轮检查来源，但不从任一个 Ring 请求像素载荷。
    assert all("rgb" not in fields and "depth" not in fields
               for ring in (env.channels.camera_ring, env.channels.hoi_ring) for fields in ring.reads)
    assert not rollout.history and not env.worker.calls and not env.recorder.rows

    env.publish()
    env.channels.hoi_ring.data["rgb"] = np.full((2, 2, 3), 3, np.uint8)
    rollout.step()  # 未到下个槽也能立即登记恢复，元数据观测不进入历史。
    assert rollout.state == "RESUMING" and not rollout.history and not env.worker.calls
    assert rollout.last_slot == 0

    env.advance(.05)
    env.channels.hoi_ring.data.update(
        rgb=np.full((2, 2, 3), 7, np.uint8), source_camera_sequence=7,
    )
    env.channels.camera_ring.data["rgb"][:] = 9
    env.channels.camera_ring.latest_sequence = 8
    rollout.step()
    assert len(env.worker.calls) == 1 and rollout.last_slot == 1
    rows, _ = env.worker.calls[0]
    assert all(np.all(row["rgb"] == 7) for row in rows)
    assert rollout.observations.last_rgb_sequence == 7
    assert "rgb" in env.channels.hoi_ring.reads[-1]
    assert "rgb" in env.channels.camera_ring.reads[-1] and "depth" in env.channels.camera_ring.reads[-1]
    assert not env.recorder.rows


@pytest.mark.parametrize("source", ["camera", "arm"])
def test_hold_rechecks_source_health_between_control_slots(env, source):
    enter_hold(env)
    env.clock.now += 1_000_000
    if source == "camera":
        env.channels.camera_ring.data["timestamp_ns"] = 0
        env.rollout.step()
        assert env.rollout.finished
        assert env.rollout.result.termination_details == "camera_source_stale"
    else:
        env.robot.arm_age = 1_000_000_000
        with pytest.raises(FeedbackUnavailable, match="arm_feedback_stale"):
            env.rollout.step()
    assert env.rollout.last_slot == 0 and not env.recorder.rows


def test_recorder_io_failure_uses_existing_stop_path(env):
    enter_hold(env)
    env.rollout.prepare = lambda poll: None

    def fail():
        raise RecordingError("disk failed")

    env.recorder.check_error = fail
    env.rollout.run(poll_operator=lambda **kw: None)
    assert env.rollout.result.termination_details == "recording_failed"
    assert env.robot.stops == 1


def test_initial_registration_timeout_and_cancellation(env):
    env.publish(False)

    def timeout_poll():
        env.advance(120, False)

    assert env.rollout.start_observation(timeout_poll, initial_hoi=True) is None
    assert env.rollout.preparation_issue == "model_input_hoi_pose_invalid"
    env.state.request_stop()
    with pytest.raises(OperationCancelled):
        env.rollout.start_observation(lambda: None, initial_hoi=True)


def test_second_recovery_interval_marks_new_gap(env):
    ready_to_send(env)
    env.rollout.step()
    assert env.rollout.state == "RUNNING"
    env.advance(0.05, False)
    env.rollout.step()
    assert env.recorder.gaps == 2
    assert env.rollout.state == "HOI_HOLD"


@pytest.mark.parametrize("record_raw", [True, False], ids=["raw", "no_raw"])
def test_slow_submission_does_not_extend_episode_budget(env, record_raw):
    ready_to_send(env)
    original = env.robot.send_action

    def send(command, **kwargs):
        original(command, **kwargs)
        env.advance(60)

    env.robot.send_action = send
    env.rollout.prepare = lambda poll: None
    env.rollout.run(poll_operator=lambda **kwargs: None)
    assert env.rollout.result.termination_details == "duration_timeout"
    assert env.rollout.result.termination_reason == "duration_complete"
    assert env.rollout.state == "RESUMING"
    assert env.rollout.cursor == 0
    assert env.rollout.prev_action is None
    assert env.robot.stops == 1 and env.state.stop_signal.is_set()
    assert env.channels.hoi_request.data["episode_id"] == 0
    if record_raw:
        assert len(env.recorder.rows) == 1
        assert env.recorder.rows[0][1] is env.robot.sent[-1][1]
    else:
        assert not env.recorder.rows and not env.recorder.gaps


def test_run_recovers_without_restarting_lifecycle(env):
    env.rollout.prepare = lambda poll: None
    calls = 0

    def poll(**kwargs):
        nonlocal calls
        calls += 1
        assert calls < 10
        if any(not command.arm_hold for _, command, _ in env.robot.sent):
            assert env.rollout.state == "RUNNING"
            assert env.channels.hoi_request.data["episode_id"] == 7
            assert not env.state.stop_signal.is_set() and env.robot.stops == 0
            env.rollout.stop("operator input", run_end_reason=RunEndReason.OPERATOR)
            return
        env.advance(0.05, valid=calls > 1)
        if env.worker.future is not None:
            env.worker.complete()

    env.rollout.run(poll_operator=poll)
    assert env.robot.services >= 4
    assert env.worker.reset_count == 1 and len(env.worker.calls) == 1
    assert env.recorder.gaps == 1 and len(env.recorder.rows) == 1
    assert env.robot.stops == 1


@pytest.mark.parametrize("save", [False, True])
def test_recorder_finish_keeps_first_intent_and_publishes_only_after_exit(tmp_path, save):
    recorder = EpisodeRecorder(tmp_path, control_hz=20)
    staging = tmp_path / ".tmp_episode_test"
    staging.mkdir()
    (staging / "data.h5").write_bytes(b"test writer output")
    recorder.staging_path = staging
    recorder.writer = SimpleNamespace(frame_count=2)
    released = Event()
    recorder.thread = Thread(target=released.wait)
    recorder.thread.start()
    try:
        assert not recorder.finish(save=save, reason="operator", wait=False)
        assert not recorder.finish(save=not save, reason="quit", wait=False)
        assert recorder.result()["status"] == "pending"
        assert staging.exists() and not (tmp_path / "episode_test").exists()
    finally:
        released.set()
        recorder.thread.join(timeout=2)
    assert recorder.finish(save=not save, reason="quit")
    result = recorder.result()
    assert result["status"] == ("published" if save else "not_published")
    assert not staging.exists()
    assert (tmp_path / "episode_test").exists() == save
    # 最终结果可重复读取；重复收尾不会改变首个处置意图。
    assert recorder.finish(save=not save, reason="quit")
    assert recorder.result() == result


def test_recorder_finish_timeout_keeps_live_owner_and_staging(tmp_path, monkeypatch):
    from dexmani_real.data import recorder as recorder_module

    clock = SimpleNamespace(now=10.)
    monkeypatch.setattr(recorder_module.time, "monotonic", lambda: clock.now)
    recorder = EpisodeRecorder(tmp_path, control_hz=20)
    recorder.staging_path = tmp_path / ".tmp_pending"
    recorder.staging_path.mkdir()
    recorder.thread = Mock(is_alive=Mock(return_value=True))
    assert not recorder.finish(save=True, reason="operator", wait=False)
    clock.now += recorder_module.RECORDER_STOP_TIMEOUT_S + 1
    with pytest.raises(TimeoutError, match="writer did not exit"):
        recorder.finish(save=True, reason="quit", wait=False)
    assert recorder.result()["status"] == "pending"
    assert recorder.staging_path.exists() and recorder.thread.is_alive()


@pytest.mark.parametrize("primary_type,close_type,reported_type", [
    (OSError, OSError, RecordingError),
    (ValueError, OSError, ValueError),
    (OSError, RuntimeError, RuntimeError),
    (OSError, KeyboardInterrupt, KeyboardInterrupt),
])
def test_writer_and_close_errors_keep_causes_and_staging(
    tmp_path, monkeypatch, caplog, primary_type, close_type, reported_type,
):
    from dexmani_real.data import recorder as recorder_module

    recorder = EpisodeRecorder(tmp_path, control_hz=20)
    recorder.staging_path = tmp_path / ".tmp_failed"
    recorder.staging_path.mkdir()
    writer = Mock(frame_count=0)
    writer.open.side_effect = primary_type("first write failure")
    writer.close.side_effect = close_type("later close failure")
    monkeypatch.setattr(recorder_module, "RawWriter", Mock(return_value=writer))
    recorder.write_episode({})
    writer.close.assert_called_once()
    assert "first write failure" in caplog.text and "later close failure" in caplog.text
    if reported_type is RecordingError:
        assert recorder.finish(save=True, reason="failed")
        assert isinstance(recorder.error.__cause__, OSError)
    else:
        with pytest.raises(reported_type):
            recorder.finish(save=True, reason="failed")
    assert recorder.result()["status"] == "failed"
    assert recorder.staging_path.exists() and recorder.published_path is None


@pytest.mark.parametrize("outcome", ["success", "failure", "timeout"])
def test_real_recorder_preserves_raw_source_and_segment_boundary(env, tmp_path, outcome):
    recorder = EpisodeRecorder(tmp_path, control_hz=20)
    # 只验证生产者队列契约，不启动写盘线程。
    recorder.is_recording = True
    env.rollout.recorder = recorder
    env.rollout.plan = ActionChunk(np.full((8, 19), 0.1), 0)
    env.rollout.step()
    _, command, segment_start = recorder.queue.get_nowait()
    assert command is not None and not segment_start

    ready_to_send(env)
    assert recorder.queue.empty() and recorder.pending_segment_start
    assert recorder.recording_truncated_reason is None
    env.channels.camera_ring.latest_sequence = 2
    observed_ns = env.clock.now
    env.channels.camera_ring.data["timestamp_ns"] = observed_ns - 10_000_000
    attempted = []
    original = env.robot.send_action

    def send(command, **kwargs):
        attempted.append(command)
        if outcome == "failure":
            raise RobotError("submission failed")
        original(command, **kwargs)
        if outcome == "timeout":
            env.advance(60)

    env.robot.send_action = send
    if outcome == "failure":
        with pytest.raises(RobotError, match="submission failed"):
            env.rollout.step()
    else:
        env.rollout.step()
    raw, command, segment_start = recorder.queue.get_nowait()
    assert segment_start
    assert len(attempted) == 1 and command is attempted[0]
    assert raw.hoi is None and raw.camera["ring_sequence"] == 2
    assert raw.camera["timestamp_ns"] == observed_ns - 10_000_000
    assert raw.observation_timestamp_ns == raw.control_timestamp_ns == observed_ns
    assert recorder.queue.empty() and not recorder.pending_segment_start
    assert recorder.recording_truncated_reason is None
    if outcome == "timeout":
        assert env.rollout.result.termination_details == "duration_timeout"

    if outcome == "success":
        env.advance()
        env.rollout.step()
        _, command, segment_start = recorder.queue.get_nowait()
        assert command is not None and not segment_start
        assert recorder.queue.empty()


def test_recording_queue_full_keeps_termination_reason_after_gap(env, tmp_path):
    recorder = EpisodeRecorder(tmp_path, control_hz=20)
    recorder.is_recording = True
    _, raw = env.rollout.read_observations()
    for _ in range(recorder.queue.maxsize):
        assert recorder.append(raw) is None
    recorder.mark_gap()
    assert recorder.append(raw) == "recording_queue_full"
    assert not recorder.is_recording
    recorder.mark_gap()
    assert recorder.result()["recording_truncated_reason"] == "recording_queue_full"


def test_published_recording_gap_is_a_segment_not_truncation(tmp_path):
    import logging

    from dexmani_real.data.export import ExportConfig, export_zarr
    from dexmani_real.data.raw import RawReader, camera_metadata
    from dexmani_real.data.recorder import format_recording_result
    from dexmani_real.robot.commands import RobotCommand
    from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
    from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE
    from dexmani_real.runtime.observation import Observation

    camera = CameraSnapshot(None, CameraIntrinsics(16, 16, 20, 20, 8, 8, "none", (0.,) * 5), .001, None)
    recorder = EpisodeRecorder(tmp_path, control_hz=20)
    recorder.start(task_label="test", collection_source="policy_rollout", metadata=camera_metadata(camera))
    try:
        for index in range(2):
            if index:
                recorder.mark_gap()
            stamp = 1_000_000_000 + index * 500_000_000
            arm, hand = np.zeros(1, ARM_STATE_DTYPE), np.zeros(1, HAND_STATE_DTYPE)
            arm["timestamp_ns"] = hand["timestamp_ns"] = stamp
            hand["tactile_aggregate_timestamp_ns"] = hand["tactile_dense_timestamp_ns"] = stamp
            row = Observation(
                arm=arm, hand=hand, vr=None, point_cloud=None,
                observation_timestamp_ns=stamp, control_timestamp_ns=stamp,
                camera={
                    "timestamp_ns": stamp, "ring_sequence": index + 1,
                    "rgb": np.full((16, 16, 3), 100, np.uint8), "depth": np.full((16, 16), 500, np.uint16),
                },
            )
            assert recorder.append(row, RobotCommand(arm_qpos=np.zeros(7), hand_qpos=np.zeros(12))) is None
    finally:
        recorder.finish(save=True, reason="operator")
    result = recorder.result()
    assert result["status"] == "published" and result["recording_truncated_reason"] is None
    level, description = format_recording_result(result)
    assert level == logging.INFO and "保存成功" in description
    with RawReader(result["path"]) as reader:
        np.testing.assert_array_equal(reader.segment_starts, [0, 1])
        assert reader.meta["termination_reason"] == "operator"
        assert "recording_truncated_reason" not in reader.meta
        assert len(list(reader.iter_rgb_frames())) == 2
    exported = export_zarr(result["path"], tmp_path / "segmented.zarr", fields=("joint_state",), processing=ExportConfig())
    assert exported["source_count"] == 1 and exported["episode_count"] == 2


def test_eef_resume_uses_measured_reference(env):
    env.rollout.policy_info.action_mode = "eef"
    env.rollout.ik_reference = object()
    references = []

    def map_eef(pose, hand, reference):
        references.append(reference)
        return SimpleNamespace(arm_qpos=np.full(7, 0.2), hand_qpos=hand, rejection_reason=None)

    env.rollout.action_mapper = SimpleNamespace(
        solver=SimpleNamespace(
            limits=np.tile([-10., 10.], (7, 1)),
            measured_reference=lambda measured: measured.copy(),
        ),
        map_eef=map_eef,
    )
    enter_hold(env)
    env.advance()
    env.rollout.step()
    env.advance()
    env.rollout.step()
    action = np.r_[np.zeros(3), [1., 0., 0., 0., 1., 0.], np.full(12, 0.3)]
    env.worker.future.set_result(np.tile(action, (8, 1)))
    env.advance()
    env.rollout.step()
    measured, command, anchor, period = references[0]
    np.testing.assert_array_equal(command, measured)
    np.testing.assert_array_equal(anchor, measured)
    assert period == env.rollout.execution.control_dt_s
    assert env.rollout.state == "RUNNING"
    np.testing.assert_array_equal(env.rollout.ik_reference.anchor, measured)


@pytest.mark.parametrize("failure", ["none", "ordinary", "interrupt", "exit", "timeout"])
def test_session_shutdown_keeps_order_and_error_precedence(env, monkeypatch, failure):
    from dexmani_real.deployment import session as session_module
    from dexmani_real.deployment.inference import ModelShutdownTimeout
    from dexmani_real.deployment.session import PolicySession

    session = PolicySession.__new__(PolicySession)
    session.runtime_state = env.state
    session.robot, session.sensors, session.keyboard = env.robot, env.channels, object()
    session.episode, session.episodes = env.rollout, [env.rollout.result]
    session.errors = []
    session.summary = {"termination_reason": None}
    order = []
    errors = {
        "ordinary": [ValueError("device"), OSError("recording"), ValueError("model"), OSError("result")],
        "interrupt": [None, KeyboardInterrupt(), None, OSError("result")],
        "exit": [SystemExit(7), SystemExit(8), SystemExit(9), SystemExit(10)],
    }.get(failure, [None] * 4)

    def stage(name, index):
        order.append(name)
        if errors[index] is not None:
            raise errors[index]
        return True

    monkeypatch.setattr(session_module, "close_runtime", lambda *a, **kw: stage("devices", 0))
    monkeypatch.setattr(env.rollout, "finish_recording", lambda: stage("recording", 1))
    closed = Future()
    if failure != "timeout":
        if errors[2] is None:
            closed.set_result(None)
        else:
            closed.set_exception(errors[2])

    def close_model():
        order.append("model")
        return closed

    session.model = SimpleNamespace(close=close_model)
    session.save_results = lambda: stage("results", 3)
    monkeypatch.setattr(session_module, "MODEL_SHUTDOWN_TIMEOUT_S", 0)
    if failure == "exit":
        with pytest.raises(SystemExit) as caught:
            session.close()
        assert caught.value is errors[0]
    elif failure == "timeout":
        with pytest.raises(ModelShutdownTimeout):
            session.close()
        assert not closed.done()
    else:
        assert session.close() == (failure != "none")
    assert order == ["devices", "recording", "model", "results"]
    assert session.summary["ended_utc"]
    if failure == "interrupt":
        assert env.state.estop_request and session.summary["termination_reason"] == "estop"
    elif failure == "none":
        assert session.summary["termination_reason"] == "quit"
    else:
        assert session.summary["termination_reason"] == "failed"
    if failure in {"ordinary", "exit"}:
        assert [error["stage"] for error in session.errors] == ["shutdown", "results_write"]
        assert [error["stage"] for error in env.rollout.result.errors] == ["recording_finalize", "model_close"]


@pytest.mark.parametrize("pending", ["recorder", "model", None], ids=["recorder", "model", "complete"])
def test_episode_finalization_releases_only_completed_resources(env, monkeypatch, tmp_path, pending):
    from dexmani_real.deployment import session as session_module

    monkeypatch.setattr(session_module, "SessionState", lambda: env.state)
    session = session_module.PolicySession(env.rollout.runtime, "policy.pt", device="cpu")
    session.robot = Mock()
    session.keyboard = Mock()
    session.episode, session.episodes = env.rollout, [env.rollout.result]
    env.rollout.stop("operator input", run_end_reason=RunEndReason.OPERATOR)
    recorder = env.rollout.recorder = EpisodeRecorder(tmp_path, control_hz=20)
    # 只替换线程存活信号；请求、pending 判定、结果和收尾均走真实 Recorder。
    recorder.thread = Mock(is_alive=Mock(return_value=pending == "recorder"))
    worker = session.model = env.rollout.model = PolicyWorker(lambda: None, build_observation=lambda rows: rows)
    model = worker.model = Mock()
    task = worker.future = Future()
    if pending != "model":
        task.set_result(None)

    def poll_operator(*args, **kwargs):
        # 请求录制收尾后，在尚有资源 pending 时模拟退出。
        if recorder.finish_request is not None and pending is not None:
            env.state.request_quit()
        return (), False

    session.keyboard.poll.side_effect = poll_operator
    save = Mock(wraps=session.save_results)
    monkeypatch.setattr(session, "save_results", save)
    try:
        session.finish_episode()
        assert recorder.finish_request.reason == "operator"
        if pending is None:
            assert session.episode is None and worker.future is None
            save.assert_called_once()
        else:
            assert session.episode is env.rollout
            save.assert_not_called()
        if pending == "recorder":
            assert env.rollout.recorder is recorder
            assert recorder.result()["status"] == "pending"
        else:
            assert env.rollout.recorder is None
        if pending == "model":
            assert worker.future is task and not task.done()

        # 后台工作结束后，由 Session 完成资源回收。
        recorder.thread.is_alive.return_value = False
        if not task.done():
            task.set_result(None)
        assert session.close() == 0
        assert env.rollout.recorder is None and worker.future is None
        assert env.rollout.result.raw["status"] == "not_published"
        assert worker.close_future.done()
        model.close.assert_called_once()
        session.robot.close.assert_called_once()
        assert save.call_count == (2 if pending is None else 1)
    finally:
        if not task.done():
            task.set_result(None)
        worker.close().result()
