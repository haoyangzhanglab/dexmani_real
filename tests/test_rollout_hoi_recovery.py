"""在线 HOI 恢复的控制契约；不加载模型或连接设备。"""

from concurrent.futures import Future
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from dexmani_real.config.deployment import ExecutionConfig
from dexmani_real.config.real import RealConfig
from dexmani_real.data.recorder import EpisodeRecorder, RecordingError
from dexmani_real.deployment import rollout as rollout_module
from dexmani_real.deployment.inference import PolicyWorker
from dexmani_real.deployment.observation import ObservationBuilder
from dexmani_real.deployment.rollout import Rollout
from dexmani_real.deployment.schedule import ActionChunk, Prediction
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

    def read(self, *, fields=None):
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
def env(monkeypatch):
    clock = SimpleNamespace(now=10_000_000_000)

    def sleep(seconds):
        clock.now += round(seconds * 1e9)

    monkeypatch.setattr(rollout_module.time, "monotonic_ns", lambda: clock.now)
    monkeypatch.setattr(rollout_module.time, "sleep", sleep)
    state = SessionState(stop_signal=Event(), emergency_signal=Event())
    robot = FakeRobot(clock, state)
    channels = SimpleNamespace(
        hoi_request=FakeRing(dict(episode_id=7, after_ns=1, after_sequence=0)),
        hoi_ring=FakeRing(None), camera_ring=FakeRing(None),
    )
    runtime = RealConfig(execution=ExecutionConfig(control_dt_s=0.05, action_steps=8))
    info = SimpleNamespace(
        observation_fields=("object_pointcloud", "joint_state"),
        n_obs_steps=2, action_mode="joint", requires_prev_action=True,
    )
    observations = ObservationBuilder(info, runtime)
    worker, recorder = FakePolicyWorker(), FakeRecorder()
    worker.build_observation = observations.build
    rollout = Rollout(
        state, runtime, info,
        cancel_epoch=state.cancel_epoch, robot=robot, sensors=SimpleNamespace(channels=channels),
        observations=observations,
        read_needs=replace(
            observations.needs.union(RAW_RECORDING_NEEDS),
            camera_fields=observations.needs.camera_fields,
        ),
        record_needs=RAW_RECORDING_NEEDS,
        action_mapper=SimpleNamespace(map_joint=lambda arm, hand: SimpleNamespace(
            arm_qpos=arm, hand_qpos=hand, rejection_reason=None)),
        model=worker, max_running_s=60, index=7, output_dir=None, record_raw=True,
    )
    rollout.recorder = recorder
    rollout.started_ns = rollout.next_step_ns = clock.now
    rollout.run_deadline_ns = clock.now + 60_000_000_000

    def publish(valid=True):
        channels.camera_ring.data = dict(
            timestamp_ns=clock.now, rgb=np.zeros((2, 2, 3), np.uint8),
            depth=np.ones((2, 2), np.uint16),
        )
        channels.hoi_ring.data = dict(
            episode_id=7, timestamp_ns=clock.now, source_camera_sequence=1,
            valid=valid, object_pointcloud=np.full((4, 3), clock.now, dtype=float),
        )

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


def test_hold_clears_context_and_refreshes_measured_hand(env):
    rollout = env.rollout
    rollout.history.append({"old": np.ones(1)})
    rollout.plan, rollout.query = ActionChunk(np.ones((8, 19)), 0), Prediction(0, None)
    rollout.cursor, rollout.prev_action, rollout.ik_reference = 3, np.ones(19), object()
    rollout.first_prediction, rollout.has_submitted = False, True
    enter_hold(env)
    assert not rollout.history
    assert rollout.plan is rollout.query is None
    assert rollout.prev_action is rollout.ik_reference is None
    assert rollout.cursor == 0 and rollout.first_prediction and not rollout.has_submitted
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
def test_complete_recovery_and_exact_action_boundary(env, mode):
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
    assert all(cmd.arm_hold for _, cmd, _ in env.robot.sent)
    env.worker.complete()
    assert env.worker.reset_count == 1
    env.advance()
    env.rollout.step()
    assert env.rollout.state == "RUNNING" and env.rollout.hoi_recovery_started_ns is None
    assert env.rollout.has_submitted and not env.robot.sent[-1][1].arm_hold
    assert len(env.recorder.rows) == 1 and env.recorder.rows[0][1] is not None
    assert env.recorder.gaps == 1
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


def test_repeated_occlusion_keeps_timeout_and_gap(env):
    enter_hold(env)
    started = env.rollout.hoi_recovery_started_ns
    env.advance()
    env.rollout.step()
    env.advance()
    env.rollout.step()
    old = env.worker.future
    env.advance(0.1, False)
    env.rollout.step()
    assert env.rollout.state == "HOI_HOLD" and env.rollout.hoi_recovery_started_ns == started
    old.set_result(np.ones((20, 19)))
    env.advance()
    env.rollout.step()
    assert env.rollout.state == "RESUMING" and not env.rollout.history
    assert env.recorder.gaps == 1
    env.advance(5)
    env.rollout.step()
    assert env.rollout.result.termination_details == "hoi_recovery_timeout"
    assert env.channels.hoi_request.data["episode_id"] == 0
    assert env.state.stop_signal.is_set()


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


@pytest.mark.parametrize("cause", ["timeout", "operator", "estop", "duration", "worker"])
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
            env.rollout.run_deadline_ns = env.clock.now

    if cause == "worker":
        with pytest.raises(RobotError, match="worker exited"):
            env.rollout.run(poll_operator=poll)
    else:
        env.rollout.run(poll_operator=poll)
    assert env.robot.stops == 1 and env.state.stop_signal.is_set()
    assert env.channels.hoi_request.data["episode_id"] == 0
    expected = dict(timeout="cancelled", operator="operator", estop="estop",
                    duration="duration_complete", worker="failed")
    assert env.rollout.result.termination_reason == expected[cause]



def test_failed_resumed_submission_never_runs_or_records(env):
    ready_to_send(env)
    env.robot.send_error = RobotError("submission failed")
    with pytest.raises(RobotError, match="submission failed"):
        env.rollout.step()
    assert env.rollout.state == "RESUMING"
    assert not env.rollout.has_submitted and not env.recorder.rows


def test_timeout_during_mapping_prevents_motion(env):
    ready_to_send(env)
    original = env.rollout.map_command

    def mapping(action, row):
        result = original(action, row)
        env.advance(5)
        return result

    env.rollout.map_command = mapping
    env.rollout.step()
    assert env.rollout.result.termination_details == "hoi_recovery_timeout"
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
    assert env.rollout.hoi_recovery_started_ns == env.clock.now


def test_slow_submission_does_not_extend_recovery(env):
    ready_to_send(env)
    original = env.robot.send_action

    def send(command, **kwargs):
        original(command, **kwargs)
        env.advance(5)

    env.robot.send_action = send
    env.rollout.step()
    assert env.rollout.result.termination_details == "hoi_recovery_timeout"
    assert env.rollout.state == "RESUMING" and not env.recorder.rows


def test_run_recovers_without_restarting_lifecycle(env):
    env.rollout.prepare = lambda poll: None
    calls = 0

    def poll(**kwargs):
        nonlocal calls
        calls += 1
        assert calls < 10
        if env.rollout.has_submitted:
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


def test_real_recorder_preserves_raw_source_and_segment_boundary(env, tmp_path):
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
    env.channels.camera_ring.latest_sequence = 2
    env.channels.camera_ring.data["timestamp_ns"] = env.clock.now - 10_000_000
    env.rollout.step()
    raw, command, segment_start = recorder.queue.get_nowait()
    assert segment_start and command is not None
    assert raw.hoi is None and raw.camera["ring_sequence"] == 2
    assert raw.camera["timestamp_ns"] == env.clock.now - 10_000_000
    assert not recorder.pending_segment_start

    env.advance()
    env.rollout.step()
    _, command, segment_start = recorder.queue.get_nowait()
    assert command is not None and not segment_start


def test_recovery_without_raw_recording(env):
    env.rollout.recorder = None
    env.rollout.record_raw = False
    env.rollout.record_needs = ObservationNeeds()
    env.rollout.read_needs = env.rollout.observations.needs
    ready_to_send(env)
    env.rollout.step()
    assert env.rollout.state == "RUNNING"
    assert not env.recorder.rows and not env.recorder.gaps


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
