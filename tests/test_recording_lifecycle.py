"""录制线程与遥操作收尾的确定性回归。"""

import threading
from unittest.mock import Mock

import pytest

from dexmani_real.data.recorder import EpisodeRecorder, RecordingError
from dexmani_real.runtime.lifecycle import close_runtime
from dexmani_real.runtime.state import SessionState
from dexmani_real.teleop.loop import TeleopLoop, TeleopPhase


@pytest.fixture
def capture(tmp_path, monkeypatch):
    entered, release, closing, closed = (threading.Event() for _ in range(4))

    class Writer:
        def __init__(self, *args, **kwargs):
            self.frame_count = 0
            self.frames = []
            self.finish = Mock()
            self.flush = Mock()
            self.failure = None

        def open(self, metadata):
            pass

        def append_frame(self, row, command):
            entered.set()
            assert release.wait(3)
            if self.failure == "write":
                raise OSError("write failed")
            self.frames.append(row)
            self.frame_count += 1

        def close(self):
            closing.set()
            assert closed.wait(3)
            if self.failure == "close":
                raise OSError("close failed")

    monkeypatch.setattr("dexmani_real.data.recorder.RawWriter", Writer)
    recorder = EpisodeRecorder(tmp_path, control_hz=10)
    recorder.start(task_label="test", collection_source="teleop", metadata={})
    recorder.queue.put((0, None))
    assert entered.wait(3)
    for i in range(1, 5):
        recorder.queue.put((i, None))
    try:
        yield recorder, release, closing, closed
    finally:
        recorder.request_finish(save=False, reason="cleanup")
        release.set()
        closed.set()
        recorder.thread.join(3)
        assert not recorder.thread.is_alive()


@pytest.mark.parametrize("save", [True, False])
def test_save_and_discard(capture, save):
    rec, release, closing, closed = capture
    request = rec.request_finish(save=save, reason="operator")
    assert rec.request_finish(save=not save, reason="quit") is request
    staging = rec.staging_path
    assert not rec.poll_finish()
    release.set()
    assert closing.wait(3)
    assert staging.exists() and not rec.poll_finish()
    assert rec.writer.frames == (list(range(5)) if save else [0])
    assert rec.writer.finish.call_count == int(save)
    rec.writer.flush.assert_not_called()
    closed.set()
    rec.thread.join(3)
    assert rec.poll_finish() and rec.queue.empty()
    assert not staging.exists()
    assert (rec.published_path is not None) == save


@pytest.mark.parametrize(
    "failure,save",
    [
        (failure, save)
        for failure in ("write", "close", "timeout", "publish")
        for save in (True, False)
        if failure != "publish" or save
    ],
)
def test_failure_retains_staging(capture, monkeypatch, failure, save):
    rec, release, _, closed = capture
    rec.writer.failure = failure
    request = rec.request_finish(save=save, reason="operator")
    if failure == "timeout":
        monkeypatch.setattr(
            "dexmani_real.data.recorder.time.monotonic", lambda: request.deadline
        )
    else:
        release.set()
        closed.set()
        rec.thread.join(3)
    if failure == "publish":
        monkeypatch.setattr(
            "dexmani_real.data.recorder.publish",
            Mock(side_effect=OSError("publish failed")),
        )
    loop = make_loop(rec)
    with pytest.raises(RecordingError):
        loop.finish_recording(wait=False)
    assert loop.recorder is rec
    assert rec.staging_path.exists() and rec.published_path is None


def make_loop(rec):
    loop = TeleopLoop.__new__(TeleopLoop)
    loop.recorder = rec
    loop.runtime_state = SessionState()
    loop.phase = TeleopPhase.ACTIVE
    loop.cancel_epoch = 1
    loop.discard_requested = False
    loop.pending_termination_reason = "operator"
    loop.termination_details = []
    loop.robot = Mock()
    loop.controller = Mock()
    loop.keyboard = Mock()
    loop.audio = Mock()
    return loop


@pytest.mark.parametrize("event", ["q", "esc"])
def test_exit_hands_pending_writer_to_final_close(capture, monkeypatch, event):
    rec, release, _, closed = capture
    loop = make_loop(rec)

    def poll(*args, **kwargs):
        if rec.finish_request is None:
            return [], True
        loop.handle_commands(("b", "h", "t", "s", "c", "d"))
        assert not loop.discard_requested
        if event == "q":
            loop.runtime_state.request_quit()
        return [event], True

    loop.keyboard.poll.side_effect = poll
    monkeypatch.setattr("dexmani_real.teleop.loop.time.sleep", lambda _: None)
    loop.end_episode()
    request = rec.finish_request
    assert request.save and request.reason == "operator"
    assert loop.recorder is rec and rec.writer_pending
    loop.keyboard.accept_operation.assert_not_called()
    loop.keyboard.allow_start.assert_not_called()
    loop.robot.stop.assert_called_once()
    loop.controller.clear.assert_called_once()
    loop.robot.service_feedback.assert_called_once()

    def close_robot():
        assert loop.recorder is rec and rec.writer_pending

    loop.robot.close.side_effect = close_robot
    sensors = Mock()

    def shutdown(state):
        release.set()
        closed.set()
        rec.thread.join(3)

    sensors.shutdown.side_effect = shutdown
    close_runtime(
        loop.runtime_state, loop.robot, keyboard=loop.keyboard, sensors=sensors
    )
    loop.keyboard.poll.side_effect = AssertionError("keyboard already closed")
    loop.robot.service_feedback.side_effect = AssertionError("robot already closed")
    assert loop.finish_recording(wait=True)
    assert rec.finish_request is request and loop.recorder is None
    plays = loop.audio.play.call_count
    assert loop.finish_recording(wait=True)
    assert loop.audio.play.call_count == plays


def test_feedback_failure_keeps_owner(capture):
    rec, *_ = capture
    loop = make_loop(rec)
    loop.keyboard.poll.return_value = ([], True)
    loop.robot.service_feedback.side_effect = OSError("feedback failed")
    with pytest.raises(OSError, match="feedback failed"):
        loop.end_episode()
    assert loop.recorder is rec
    assert loop.pending_termination_reason == "failed"


def test_full_queue_saves_accepted_prefix(capture):
    from dexmani_real.runtime.observation import Observation

    rec, release, _, closed = capture
    for i in range(5, 17):
        rec.queue.put_nowait((i, None))
    row = Observation(None, None, None, None, None, 1)
    assert rec.append(row) == "recording_queue_full"
    rec.request_finish(save=True, reason="operator")
    release.set()
    closed.set()
    rec.thread.join(3)
    assert rec.poll_finish()
    assert rec.writer.frames == list(range(17))
    assert (
        rec.writer.finish.call_args.kwargs["recording_truncated_reason"]
        == "recording_queue_full"
    )


@pytest.mark.parametrize("save", [True, False])
def test_finish_arriving_after_empty_queue_observation(capture, monkeypatch, save):
    from queue import Empty

    rec, release, closing, closed = capture
    original_get = rec.queue.get
    injected = False

    def get(*args, **kwargs):
        nonlocal injected
        try:
            return original_get(*args, **kwargs)
        except Empty:
            if not injected:
                # 重现 get 已观察到空队列、生产者随后入队并结束的交错。
                injected = True
                rec.queue.put_nowait((5, None))
                rec.request_finish(save=save, reason="operator")
            raise

    monkeypatch.setattr(rec.queue, "get", get)
    release.set()
    assert closing.wait(3)
    assert rec.writer.frames == list(range(6 if save else 5))
    closed.set()
    rec.thread.join(3)
    assert rec.poll_finish()
