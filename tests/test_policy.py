"""调用真实 rollout/worker/session，替换模型、设备与键盘 I/O。"""

import json
import threading
import time
from dataclasses import replace
from itertools import count
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.config.real import RealConfig
from dexmani_real.control.actions import ActionMapper
from dexmani_real.data.recorder import EpisodeRecorder, RecordingError
from dexmani_real.deployment.inference import PolicyWorker
from dexmani_real.deployment.rollout import Rollout
from dexmani_real.deployment.schedule import Prediction
from dexmani_real.deployment.session import PolicySession
from dexmani_real.robot.commands import OperationCancelled
from dexmani_real.runtime.keyboard import Keyboard, KeyEvent
from dexmani_real.runtime.observation import ObservationNeeds
from dexmani_real.runtime.state import RunEndReason, SessionState
from dexmani_real.teleop.audio import AudioFeedback
from dexmani_real.utils.log import error_detail


class Audio:
    finish = AudioFeedback.finish

    def __init__(self, events=None):
        self.events = [] if events is None else events

    def play(self, event, **kwargs):
        self.events.append(event)

    def wait_until_idle(self, **kwargs):
        self.events.append("audio_wait")
        return True

    def close(self):
        self.events.append("audio_close")


class Model:
    def __init__(self):
        self.has_pending_task = False
        self.result = None
        self.calls = []

    def predict(self, rows, **kwargs):
        self.calls.append((rows, kwargs))
        self.has_pending_task = True

    def poll(self):
        if self.result is None:
            return None
        result, self.result = self.result, None
        self.has_pending_task = False
        return "predict", result

    def discard_pending(self, timeout_s):
        self.has_pending_task = False


def make_rollout(tmp_path, mode="sync"):
    runtime = RealConfig()
    runtime = replace(runtime, execution=replace(
        runtime.execution, mode=mode, action_steps=3, prefetch_steps=1, rtc_guidance_cap=1.0,
    ))
    state = SessionState()
    epoch = state.arm()
    state.stop_signal.clear()
    state.begin_motion()
    sent = []
    robot = SimpleNamespace(
        send_action=lambda command, **kwargs: sent.append(command),
        stop=lambda: None, check=lambda: None, service_feedback=lambda: None,
    )
    mapper = ActionMapper(
        SimpleNamespace(limits=np.tile([-np.pi, np.pi], (7, 1)),
                        rejection=lambda *args, **kwargs: None),
        hand_enabled=True, hand_lower_rad=np.zeros(12), hand_upper_rad=np.ones(12),
        home_hand_qpos=np.zeros(12),
    )
    info = SimpleNamespace(control_dt_s=0.04, n_obs_steps=1, future_steps=6,
                           control_action_dim=19, action_mode="joint")
    model, audio = Model(), Audio()
    episode = Rollout(
        state, runtime, info, epoch=epoch, robot=robot,
        observations=SimpleNamespace(project=lambda row: row), needs=ObservationNeeds(),
        sensors=SimpleNamespace(channels=None), action_mapper=mapper, model=model,
        max_running_s=None, capture_ids=count(1), index=1, output_dir=tmp_path,
        record_raw=False, audio=audio,
    )
    row = SimpleNamespace(arm={"qpos": np.zeros((1, 7))})
    episode.read_observation = lambda: (row, None)
    episode.started_ns = episode.next_step_ns = 1_000_000_000
    return episode, sent


@pytest.mark.parametrize("mode,first", [("sync", 0.01), ("async", 0.02), ("rtc", 0.01)])
def test_model_handoff_uses_actual_scheduler_and_mapper(tmp_path, monkeypatch, mode, first):
    episode, sent = make_rollout(tmp_path, mode)
    clock = [episode.started_ns]
    monkeypatch.setattr(time, "monotonic_ns", lambda: clock[0])
    episode.step()
    assert not sent and not episode.audio.events
    assert episode.model.calls[0][1] == {"reset_episode": True}
    actions = np.repeat(np.arange(1, 7)[:, None] * 0.01, 19, axis=1)
    episode.model.result = actions
    clock[0] += episode.dt_ns
    episode.step()
    assert sent[0].arm_qpos[0] == pytest.approx(first)
    assert episode.audio.events == ["begin"]
    if mode == "sync":
        assert episode.last_slot == 0
    if mode == "rtc":
        for _ in range(2):
            clock[0] += episode.dt_ns
            episode.step()
        np.testing.assert_array_equal(episode.model.calls[-1][1]["rtc_prefix"], actions[2:])
        assert episode.model.calls[-1][1]["delay_steps"] == 1
        submitted = len(sent)
        clock[0] += 2 * episode.dt_ns
        episode.step()
        assert len(sent) == submitted
        assert "missed_slot" in episode.result.termination_details
    elif mode == "async":
        clock[0] += 8 * episode.dt_ns
        episode.step()
        assert len(sent) == 1
        assert "exhausted" in episode.result.termination_details


def test_sync_wait_does_not_consume_action_cursor(tmp_path, monkeypatch):
    episode, sent = make_rollout(tmp_path)
    clock = [episode.started_ns]
    monkeypatch.setattr(time, "monotonic_ns", lambda: clock[0])
    episode.step()
    clock[0] += episode.dt_ns
    episode.step()
    assert episode.last_slot == -1 and not sent and len(episode.history) == 1


@pytest.mark.parametrize("actions", [np.zeros((5, 19)), np.full((6, 19), np.nan)])
def test_invalid_model_output_rejected_at_poll_boundary(tmp_path, actions):
    episode, _ = make_rollout(tmp_path)
    episode.query = Prediction(0, None)
    episode.model.result = actions
    with pytest.raises(ValueError, match="finite shape"):
        episode.poll_model()


def test_begin_only_after_successful_submission(tmp_path, monkeypatch):
    episode, _ = make_rollout(tmp_path)
    monkeypatch.setattr(time, "monotonic_ns", lambda: episode.started_ns)
    episode.query = Prediction(0, None, np.zeros((6, 19)))
    def fail(*args, **kwargs):
        raise TimeoutError("SDK timeout")
    episode.robot.send_action = fail
    with pytest.raises(TimeoutError):
        episode.step()
    assert "begin" not in episode.audio.events


def test_worker_pending_blocks_reuse_and_consumes_late_error():
    started, release = threading.Event(), threading.Event()
    failure = RuntimeError("late model failure")
    class NativeModel:
        def predict(self, observation):
            started.set()
            assert release.wait(2)
            raise failure
        def close(self):
            pass
    worker = PolicyWorker(NativeModel, build_observation=lambda rows: rows)
    worker.load()
    worker.future.result(timeout=2)
    worker.poll()
    try:
        worker.predict(())
        assert started.wait(2)
        with pytest.raises(TimeoutError, match="pending"):
            worker.discard_pending(0)
        assert worker.has_pending_task
        with pytest.raises(RuntimeError, match="unreclaimed"):
            worker.predict(())
        release.set()
        worker.future.exception(timeout=2)
        with pytest.raises(RuntimeError) as caught:
            worker.discard_pending(1)
        assert caught.value is failure
        assert not worker.has_pending_task
    finally:
        release.set()
        worker.close()
        assert worker.wait_closed(2) == ("completed", [])


def make_session(tmp_path):
    session = PolicySession(
        RealConfig(), SimpleNamespace(), output_dir=tmp_path, run_config={}, device="cpu",
        seed=0, max_running_s=None, num_episodes=1, record_raw=False,
        camera_calibration_path=None, observations=SimpleNamespace(needs=ObservationNeeds()),
    )
    session.runtime_state = SessionState()
    session.runtime_state.request_quit()
    events = []
    session.robot = SimpleNamespace(arm=None, hand=None, close=lambda: events.append("robot_close"))
    session.keyboard = SimpleNamespace(stop=lambda: events.append("keyboard_stop"))
    session.sensors = SimpleNamespace(started=[], shutdown=lambda: events.append("sensor_shutdown"))
    session.model = SimpleNamespace(close=lambda: events.append("model_close"),
                                    wait_closed=lambda timeout: ("completed", []), has_pending_task=False)
    session.audio = Audio(events)
    session.save_results = lambda **kwargs: events.append("results")
    return session, events


@pytest.mark.parametrize("failure_stage", [None, "session", "audio", "results", "robot"])
def test_session_closes_resources_before_audio_and_final_result(tmp_path, failure_stage):
    session, events = make_session(tmp_path)
    def fail():
        events.append({"audio": "audio_close", "robot": "robot_close"}.get(failure_stage, failure_stage))
        raise OSError(failure_stage)
    if failure_stage == "audio":
        session.audio.close = fail
    elif failure_stage == "results":
        session.save_results = fail
    elif failure_stage == "robot":
        session.robot.close = fail
    failure = OSError("primary failure") if failure_stage == "session" else None
    assert session.close(failure=failure) == int(failure_stage is not None)
    assert events.index("sensor_shutdown") < events.index("audio_wait")
    assert events.index("model_close") < events.index("audio_wait")
    assert events.index("audio_close") < events.index("results")
    assert events.count("results") == events.count("audio_close") == 1
    if failure_stage in ("robot", "session"):
        assert "emergency" in events and "end" not in events
    else:
        assert "end" in events and "emergency" not in events
    assert session.summary["outcome"] == ("fault" if failure_stage else "finished")
    if failure_stage == "session":
        assert session.errors[0] == error_detail("session", failure)


@pytest.mark.parametrize("keys", [("d", "q"), ("q", "d")])
def test_policy_intents_and_pending_work_block_preparation(tmp_path, keys):
    session, _ = make_session(tmp_path)
    session.runtime_state = SessionState()
    epoch = session.runtime_state.arm()
    session.episode = SimpleNamespace(discard_requested=False, recorder=None)
    for key in keys:
        {"d": session.request_discard, "q": session.request_quit}[key]()
    assert session.episode.discard_requested and session.runtime_state.quit_requested
    session.episode = None
    session.runtime_state = SessionState()
    epoch = session.runtime_state.arm()
    session.model.has_pending_task = True
    assert not session.request_start(epoch)
    assert not session.run_home(epoch)
    # C/S in the same input batch suppress B without adding preparation state.
    session.model.has_pending_task = False
    session.handle_command_batch([KeyEvent("c"), KeyEvent("b", epoch)])
    assert session.start_epoch is None
    assert session.request_start(epoch)
    assert not session.request_start(epoch)


def test_policy_discard_is_latched_before_first_stop_signal(tmp_path, monkeypatch):
    episode, _ = make_rollout(tmp_path)
    session, _ = make_session(tmp_path)
    session.runtime_state, session.episode = episode.runtime_state, episode
    episode.result.termination_reason = "operator"
    saves = []

    def finish(**kwargs):
        saves.append(kwargs["save"])
        return "episode_001" if kwargs["save"] else None

    episode.recorder = SimpleNamespace(start_attempted=True, finish_request=None, finish=finish,
                                       result=lambda: {"status": "finished"})
    stopped, release, finishing, finished = (threading.Event() for _ in range(4))
    original_set = session.runtime_state.stop_signal.set

    def stop_at_boundary():
        original_set()
        if not stopped.is_set():
            stopped.set()
            assert release.wait(2)

    def finalize():
        finishing.set()
        episode.finish_recording(deadline=time.monotonic() + 2)
        finished.set()

    monkeypatch.setattr(session.runtime_state.stop_signal, "set", stop_at_boundary)
    callback = threading.Thread(target=session.request_discard)
    control = threading.Thread(target=finalize)
    callback.start()
    try:
        assert stopped.wait(2)
        control.start()
        assert finishing.wait(2)
        acquired = session.runtime_state.motion_lock.acquire(blocking=False)
        if acquired:
            session.runtime_state.motion_lock.release()
        assert not acquired
    finally:
        release.set()
        callback.join(2)
        if control.ident is not None:
            control.join(2)
    assert not callback.is_alive() and not control.is_alive()
    assert finished.is_set() and saves == [False]
    assert episode.audio.events == ["discard"]


def test_policy_discard_without_recording_stops(tmp_path):
    session, _ = make_session(tmp_path)
    session.runtime_state = SessionState()
    session.runtime_state.arm()
    session.runtime_state.stop_signal.clear()
    session.request_discard()
    assert session.runtime_state.stop_signal.is_set()
    assert session.episode is None and session.start_epoch is None


def test_error_detail_does_not_serialize_cause_tracebacks():
    failure = RuntimeError("primary")
    failure.__cause__ = ValueError("cause")
    assert error_detail("run", failure) == dict(
        stage="run", exception_type="RuntimeError", message="primary",
    )


def test_rollout_keeps_primary_and_reports_model_and_writer_failures(tmp_path):
    episode, _ = make_rollout(tmp_path)
    primary = ValueError("control failed")
    model_failure, writer_failure = RuntimeError("model failed"), OSError("writer failed")
    events = []
    def prepare(poll):
        raise primary
    def retire(timeout):
        events.append("model")
        raise model_failure
    def finish(**kwargs):
        events.append("writer")
        raise writer_failure
    episode.preparation_valid = lambda: True
    episode.prepare = prepare
    episode.model.discard_pending = retire
    episode.recorder = SimpleNamespace(start_attempted=True, finish=finish,
                                       result=lambda: {"status": "failed"})
    keyboard = SimpleNamespace(discard_events=lambda *args: events.append("keys"))
    with pytest.raises(ValueError) as caught:
        episode.run(keyboard=keyboard, poll_operator=lambda: None)
    assert caught.value is primary
    assert events == ["model", "writer", "keys"]
    assert [item["stage"] for item in episode.result.errors] == [
        "control", "retired_model", "recorder_finalize",
    ]
    assert "emergency" in episode.audio.events


@pytest.mark.parametrize("save,published,expected", [
    (True, "episode_001", ["save"]), (True, None, []), (False, None, ["discard"]),
])
def test_recording_audio_follows_frozen_finish_result(tmp_path, save, published, expected):
    episode, _ = make_rollout(tmp_path)
    session, _ = make_session(tmp_path)
    session.runtime_state, session.episode = episode.runtime_state, episode
    episode.result.termination_reason = "quit"
    episode.discard_requested = not save
    def finish(**kwargs):
        assert kwargs["save"] == save
        assert episode.audio.events == []
        # 保存决定已读取，recorder 尚未设置 finish_request 的窗口。
        session.request_discard()
        return published
    episode.recorder = SimpleNamespace(start_attempted=True, finish_request=None, finish=finish,
                                       result=lambda: {"path": published})
    episode.finish_recording(deadline=time.monotonic() + 1)
    assert episode.audio.events == expected


def test_local_submission_rejection_is_not_policy_failure(tmp_path, monkeypatch):
    episode, _ = make_rollout(tmp_path)
    monkeypatch.setattr(time, "monotonic_ns", lambda: episode.started_ns)
    episode.preparation_valid = lambda: True
    episode.prepare = lambda poll: None
    episode.query = Prediction(0, None, np.zeros((6, 19)))
    def reject(*args, **kwargs):
        raise OperationCancelled("current hand feedback unavailable/stale at send")
    episode.robot.send_action = reject
    keyboard = SimpleNamespace(discard_events=lambda *args: None)
    result = episode.run(keyboard=keyboard, poll_operator=lambda: None)
    assert result.errors == []
    assert result.termination_reason == "executor_boundary"
    assert "current hand feedback" in result.termination_details


def test_persistent_recorder_error_reported_once(tmp_path, caplog):
    episode, _ = make_rollout(tmp_path)
    episode.recorder = EpisodeRecorder(tmp_path, control_hz=25)
    episode.recorder.start_attempted = True
    primary = RecordingError("queue full")
    primary.__cause__ = OSError("file failed")
    episode.recorder.store_error(primary)
    episode.preparation_valid = lambda: True
    episode.prepare = lambda poll: episode.recorder.check_error()
    keyboard = SimpleNamespace(discard_events=lambda *args: None)
    with pytest.raises(RecordingError) as caught:
        episode.run(keyboard=keyboard, poll_operator=lambda: None)
    assert caught.value is primary
    assert [item["stage"] for item in episode.result.errors] == ["control"]
    assert episode.result.raw["status"] == "failed"
    assert "episode recording finalization failed" not in caplog.text


def test_preparation_key_delayed_until_after_finish_stays_revoked(tmp_path):
    episode, _ = make_rollout(tmp_path)
    captured, release = threading.Event(), threading.Event()
    def get_run_id():
        epoch = episode.runtime_state.run_id
        captured.set()
        assert release.wait(2)
        return epoch
    keyboard = Keyboard(get_run_id=get_run_id)
    keyboard.active = True
    key_thread = threading.Thread(target=keyboard.handle_key, args=("b", True))
    def finish(**kwargs):
        key_thread.start()
        assert captured.wait(2)
    episode.preparation_valid = lambda: True
    episode.prepare = lambda poll: episode.stop("operator", run_end_reason=RunEndReason.OPERATOR)
    episode.recorder = SimpleNamespace(start_attempted=True, finish=finish,
                                       result=lambda: {"status": "not_published"})
    try:
        episode.run(keyboard=keyboard, poll_operator=lambda: None)
    finally:
        release.set()
        key_thread.join(2)
    with keyboard.condition:
        events = list(keyboard.events)
    assert len(events) == 1
    session, _ = make_session(tmp_path)
    session.runtime_state = episode.runtime_state
    assert not session.request_start(events[0].run_id)
    keyboard.stop()


@pytest.mark.parametrize("model_pending", [False, True])
@pytest.mark.parametrize("failure", [None, OSError("primary failure")])
def test_audio_close_failure_is_in_single_final_result(tmp_path, model_pending, failure):
    session, _ = make_session(tmp_path)
    session.save_results = Mock(wraps=PolicySession.save_results.__get__(session))
    if model_pending:
        session.model.wait_closed = lambda timeout: ("pending", [])
    def fail_close():
        raise OSError("audio worker remains pending")
    session.audio.close = Mock(side_effect=fail_close)
    assert session.close(failure=failure) == 1
    session.audio.close.assert_called_once_with()
    session.save_results.assert_called_once_with()
    result = json.loads((tmp_path / "session_result.json").read_text())
    assert result["outcome"] == "fault"
    assert result["shutdown"] == ("pending" if model_pending else "failed")
    expected = [error_detail("session", failure)] if failure is not None else []
    expected.append(dict(stage="audio_close", exception_type="OSError",
                         message="audio worker remains pending"))
    assert result["errors"] == expected
    assert result["reason"] == ("primary failure" if failure else "audio worker remains pending")


@pytest.mark.parametrize("failure", [OSError("final JSON failed"), KeyboardInterrupt()])
def test_final_result_publication_failure_is_reported_after_audio(tmp_path, monkeypatch, caplog, failure):
    session, events = make_session(tmp_path)
    session.save_results = PolicySession.save_results.__get__(session)
    def fail_write(*args, **kwargs):
        events.append("results")
        raise failure
    write = Mock(side_effect=fail_write)
    monkeypatch.setattr("dexmani_real.deployment.session.write_json", write)
    assert session.close() == 1
    assert "end" in events and "emergency" not in events
    assert events.count("audio_close") == 1
    assert events.index("audio_close") < events.index("results")
    write.assert_called_once()
    assert session.runtime_state.estop_request == isinstance(failure, KeyboardInterrupt)
    assert session.errors[-1]["stage"] == "session_save"
    assert session.summary["outcome"] == "fault"
    assert not (tmp_path / "session_result.json").exists()
    assert any(record.exc_info and record.exc_info[1] is failure for record in caplog.records)


def test_audio_shutdown_interrupt_marks_emergency_before_final_result(tmp_path):
    session, _ = make_session(tmp_path)
    session.audio.close = Mock(side_effect=KeyboardInterrupt())
    session.save_results = Mock(wraps=PolicySession.save_results.__get__(session))
    assert session.close() == 1
    assert session.runtime_state.estop_request
    session.audio.close.assert_called_once_with()
    session.save_results.assert_called_once_with()
    result = json.loads((tmp_path / "session_result.json").read_text())
    assert result["outcome"] == "fault" and result["shutdown"] == "failed"
    assert result["reason"] == "KeyboardInterrupt"
    assert result["errors"] == [dict(stage="audio_close", exception_type="KeyboardInterrupt", message="")]


def test_missing_player_still_closes_audio_before_final_result(tmp_path, monkeypatch, caplog):
    session, _ = make_session(tmp_path)
    monkeypatch.setattr("dexmani_real.teleop.audio.shutil.which", lambda name: None)
    session.audio = AudioFeedback()
    session.audio.close = Mock(wraps=session.audio.close)
    session.save_results = Mock(wraps=PolicySession.save_results.__get__(session))
    assert session.close() == 0
    assert session.audio.worker is None and session.audio.closed
    session.audio.close.assert_called_once_with()
    session.save_results.assert_called_once_with()
    assert "No audio player" in caplog.text
    result = json.loads((tmp_path / "session_result.json").read_text())
    assert result["outcome"] == "finished" and not result["errors"]


@pytest.mark.parametrize("operation", ["startup", "home", "tactile"])
@pytest.mark.parametrize("key", ["b", "h", "t"])
def test_busy_operation_revokes_delayed_preparation_key(tmp_path, monkeypatch, operation, key):
    session, _ = make_session(tmp_path)
    session.runtime_state = SessionState()
    captured, release = threading.Event(), threading.Event()
    def get_run_id():
        epoch = session.runtime_state.run_id
        captured.set()
        assert release.wait(2)
        return epoch
    session.keyboard = Keyboard(get_run_id=get_run_id)
    session.keyboard.active = True
    key_thread = threading.Thread(target=session.keyboard.handle_key, args=(key, True))
    def busy(*args, **kwargs):
        key_thread.start()
        assert captured.wait(2)
    try:
        if operation == "startup":
            session.robot.connect = session.robot.check = session.robot.service_feedback = lambda: None
            session.camera_snapshot = None
            session.observations.camera_static = False
            session.observations.configure_camera = lambda camera: None
            session.warmup_policy = busy
            monkeypatch.setattr("dexmani_real.deployment.session.initialize_hand", lambda *a, **k: None)
            session.start_devices()
        else:
            epoch = session.runtime_state.arm()
            if operation == "home":
                monkeypatch.setattr("dexmani_real.deployment.session.home_robot", busy)
                assert session.run_home(epoch)
            else:
                session.robot.reset_tactile_sensors = busy
                session.handle_command_batch([KeyEvent("t", epoch)], allow_tactile_reset=True)
    finally:
        release.set()
        if key_thread.ident is not None:
            key_thread.join(2)
    with session.keyboard.condition:
        events = list(session.keyboard.events)
    assert len(events) == 1
    assert not session.request_start(events[0].run_id)
    session.keyboard.stop()


@pytest.mark.parametrize('point_cloud', [False, True])
@pytest.mark.parametrize('camera_fields,stored_fields', [
    ((), ()), ((), ('rgb', 'depth')), (('rgb',), ('rgb',)),
    (('rgb',), ('rgb', 'depth')), (('depth',), ('depth',)),
    (('rgb', 'depth'), ('rgb', 'depth')),
])
def test_observation_snapshot_ownership(point_cloud, camera_fields, stored_fields):
    import multiprocessing as mp

    from dexmani_real.deployment.observation import ObservationBuilder
    from dexmani_real.ipc.sensors import SensorChannels
    from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
    from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE
    from dexmani_real.runtime.observation import ObservationNeeds, read_observation

    runtime = RealConfig()
    camera = replace(runtime.camera, width=4, height=3)
    channels = SensorChannels(mp.get_context('spawn'), camera=camera,
                              pointcloud_num_points=5, pointcloud_camera_fields=stored_fields)
    rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    depth = np.arange(12, dtype=np.uint16).reshape(3, 4)
    cloud = np.arange(30, dtype=np.float32).reshape(5, 6)
    payload = {'timestamp_ns': 10, 'source_camera_sequence': 7, 'color_frame_number': 17,
               'rgb': rgb, 'depth': depth, 'point_cloud': cloud}
    payload = {name: payload[name] for name in channels.pointcloud_ring.field_names}
    channels.pointcloud_ring.write(**payload)
    channels.camera_ring.write(timestamp_ns=20, color_frame_number=18, rgb=rgb + 1, depth=depth + 1)
    arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
    hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
    hand['tactile_aggregate'] = np.arange(15).reshape(5, 3)
    robot = SimpleNamespace(read_state=lambda: SimpleNamespace(arm=arm, hand=hand))
    needs = ObservationNeeds(camera_fields=camera_fields, point_cloud=point_cloud)
    row = read_observation(channels, robot, needs=needs)
    fields = (*camera_fields, *(('point_cloud',) if point_cloud else ()),
              'finger_tactile_force', 'joint_state')
    builder = ObservationBuilder(SimpleNamespace(observation_fields=fields), runtime)
    projected = builder.project(row)
    expected = {name: value.copy() for name, value in projected.values.items()}
    for name in (*camera_fields, *(('point_cloud',) if point_cloud else ())):
        source = row.point_cloud if name == 'point_cloud' else row.camera[name]
        assert projected.values[name] is source  # 投影不做第二次大数组复制。
        assert source.flags.owndata  # 不经 structured payload 持有未请求的字段。
    assert not np.shares_memory(projected.values['finger_tactile_force'], hand)
    assert not np.shares_memory(projected.values['hand_qpos'], hand)
    assert not np.shares_memory(projected.values['arm_qpos'], arm)
    if camera_fields:
        assert set(row.camera) == {'timestamp_ns', 'ring_sequence', 'color_frame_number', *camera_fields}
        assert row.camera['timestamp_ns'] == (10 if point_cloud else 20)
        assert row.camera['ring_sequence'] == (7 if point_cloud else 1)
        assert row.camera['color_frame_number'] == (17 if point_cloud else 18)
        for name in camera_fields:
            np.testing.assert_array_equal(row.camera[name], {'rgb': rgb, 'depth': depth}[name] + (not point_cloud))
    assert row.pointcloud_timestamp_ns == (10 if point_cloud else 0)
    if point_cloud:
        np.testing.assert_array_equal(row.point_cloud, cloud)
    # 覆盖包括原槽；模型输入即使被消费者修改，也不应影响录制/历史快照。
    for _ in range(channels.pointcloud_ring.capacity):
        channels.pointcloud_ring.write(**{name: value + 2 for name, value in payload.items()})
        channels.camera_ring.write(timestamp_ns=30, color_frame_number=19, rgb=rgb + 2, depth=depth + 2)
    built = builder.build((projected, projected))
    for name in fields:
        if name != 'joint_state':
            assert built[name].dtype == expected[name].dtype
            np.testing.assert_array_equal(built[name], np.stack([expected[name]] * 2))
        built[name].fill(0)
    for name, value in expected.items():
        np.testing.assert_array_equal(projected.values[name], value)


def test_ring_rejects_incomplete_or_overwritten_read():
    import multiprocessing as mp

    from dexmani_real.ipc.ring import SharedRing

    ring = SharedRing(mp.get_context('spawn'), np.dtype([('a', 'i8'), ('b', 'i8')]), 1)
    assert ring.read() is None
    ring.write(a=1, b=2)
    ring.slots[0]['sequence'] = 1
    assert ring.read(fields=('a',)) is None
    ring.write(a=3, b=4)

    def fields():
        yield 'a'
        ring.write(a=5, b=6)
        yield 'b'

    assert ring.read(fields=fields()) is None
