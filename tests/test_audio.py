"""语音触发遵循实际提交/发布结果，播放器外部 I/O 使用 fake。"""

import time
from dataclasses import replace
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch

import pytest
from test_interaction import Recorder, begin, make_loop, observation

from dexmani_real.data.recorder import EpisodeRecorder, RecordingError
from dexmani_real.robot.commands import OperationCancelled
from dexmani_real.runtime.keyboard import KeyEvent
from dexmani_real.runtime.state import RunEndReason
from dexmani_real.teleop.audio import AudioFeedback
from dexmani_real.teleop.loop import TeleopPhase
from dexmani_real.teleop.run import run_teleop


def events(loop):
    return [call.args[0] for call in loop.audio.play.call_args_list]


def test_begin_only_after_first_successful_submission():
    loop = make_loop()
    begin(loop)
    assert events(loop) == []

    def submit(*args, **kwargs):
        assert events(loop) == []

    loop.robot.send_action.side_effect = submit
    loop.execute_control_step(observation(), time.monotonic_ns() + 10**9)
    loop.controller.commit.assert_called_once()
    assert events(loop) == ['begin']
    loop.robot.send_action.side_effect = None
    loop.execute_control_step(observation(), time.monotonic_ns() + 10**9)
    assert events(loop) == ['begin']


def test_cancelled_submission_does_not_announce_begin_or_commit():
    loop = make_loop()
    begin(loop)
    loop.robot.send_action.side_effect = OperationCancelled('expired')
    loop.execute_control_step(observation(), time.monotonic_ns() + 10**9)
    loop.controller.commit.assert_not_called()
    assert events(loop) == []


@pytest.mark.parametrize('frames,discard,expected', [(1, False, ['save']), (0, False, []),
                                                    (1, True, ['discard'])])
def test_audio_follows_actual_recording_result(frames, discard, expected):
    loop = make_loop()
    begin(loop)

    def finish():
        assert events(loop) == []
        loop.robot.stop.assert_called_once()

    recorder = loop.recorder = Recorder(finish)
    recorder.submitted_frames = frames
    loop.discard_requested = discard
    loop.request_episode_end('operator', run_end_reason=RunEndReason.OPERATOR, failed=False)
    loop.end_episode()
    assert events(loop) == expected


@pytest.mark.parametrize('discard', [False, True])
def test_emergency_during_finish_suppresses_success_audio(discard):
    loop = make_loop()
    begin(loop)
    loop.discard_requested = discard
    recorder = loop.recorder = Recorder(loop.runtime_state.request_emergency_stop)
    loop.request_episode_end('operator', run_end_reason=RunEndReason.OPERATOR, failed=False)
    loop.end_episode()
    assert recorder.calls[0]['save'] is not discard
    assert events(loop) == ['emergency']


@pytest.mark.parametrize('failure', [None, OperationCancelled('cancelled')])
def test_home_done_requires_success(failure):
    loop = make_loop()
    with patch('dexmani_real.teleop.loop.home_robot', side_effect=failure):
        loop.handle_operator_command(KeyEvent('h', loop.runtime_state.run_id))
    assert events(loop) == (['home', 'home_done'] if failure is None else ['home'])


def test_missing_player_warns_and_does_not_create_worker(caplog):
    with patch('dexmani_real.teleop.audio.shutil.which', return_value=None):
        audio = AudioFeedback()
    assert 'No audio player' in caplog.text
    audio.play('begin')
    assert audio.worker is None
    audio.finish(True)
    assert audio.closed


@pytest.mark.parametrize('primary', [None, RuntimeError('tail wait failed'), KeyboardInterrupt()])
def test_audio_finish_preserves_primary_and_reports_close_failure(primary, caplog):
    cleanup = RuntimeError('audio worker join failed')
    with patch('dexmani_real.teleop.audio.shutil.which', return_value=None):
        audio = AudioFeedback()
    audio.worker = Mock()
    audio.worker.join.side_effect = cleanup
    expected = primary if primary is not None else cleanup
    with patch.object(audio, 'wait_until_idle', side_effect=primary, return_value=True):
        # 调用者已有异常不代表 finish 自身失败；单独的 close 错误仍须传播。
        try:
            raise ValueError('caller failure')
        except ValueError:
            with pytest.raises(BaseException) as caught:
                audio.finish(False)
    assert caught.value is expected
    assert audio.closed
    audio.worker.join.assert_called_once_with(timeout=1.0)
    cleanup_logs = [record for record in caplog.records
                    if record.exc_info and record.exc_info[1] is cleanup]
    assert len(cleanup_logs) == int(primary is not None)
    assert all(record.levelname == 'WARNING' for record in cleanup_logs)


@pytest.mark.parametrize('interrupt', [KeyboardInterrupt(), SystemExit(7)])
def test_audio_close_interrupt_is_not_hidden_by_playback_error(interrupt):
    with patch('dexmani_real.teleop.audio.shutil.which', return_value=None):
        audio = AudioFeedback()
    audio.wait_until_idle = Mock(side_effect=OSError('audio wait failed'))
    audio.close = Mock(side_effect=interrupt)
    with pytest.raises(type(interrupt)) as caught:
        audio.finish(True)
    assert caught.value is interrupt
    audio.close.assert_called_once_with()


def test_emergency_preempts_running_audio_without_waiting():
    started, killed = Event(), Event()
    proc = Mock()
    proc.poll.side_effect = lambda: None if not killed.is_set() else -9
    proc.kill.side_effect = killed.set
    proc.returncode = -9

    def communicate():
        started.set()
        assert killed.wait(2)
        return b'', b''

    proc.communicate.side_effect = communicate
    emergency = Mock(returncode=0)
    emergency.poll.return_value = 0
    emergency.communicate.return_value = (b'', b'')
    with patch('dexmani_real.teleop.audio.shutil.which', return_value='/bin/aplay'), \
         patch.object(Path, 'is_file', return_value=True), \
         patch('dexmani_real.teleop.audio.subprocess.Popen', side_effect=[proc, emergency]) as popen:
        audio = AudioFeedback()
        audio.play('begin', interrupt=False)
        assert started.wait(2)
        audio.play('emergency')
        assert killed.is_set()
        assert audio.wait_until_idle(2)
        assert popen.call_count == 2
        assert audio.player == 'aplay'
        audio.close()
    assert audio.worker is not None and not audio.worker.is_alive()


@pytest.mark.parametrize('failure', [
    None, 'play', 'wait', 'timeout', 'close', 'recording', 'recording_close', 'interrupt', 'exit',
])
def test_teleop_releases_devices_before_waiting_for_tail_audio(tmp_path, failure, caplog):
    from dexmani_real.config.real import RealConfig

    defaults = RealConfig()
    runtime = replace(defaults, hand=replace(defaults.hand, enabled=False))
    order = []
    audio = Mock()
    audio.finish = AudioFeedback.finish.__get__(audio)
    def wait_audio(*, timeout_s):
        assert timeout_s == 5.0
        order.append('audio_wait')
        if failure == 'wait':
            raise RuntimeError('audio wait failed')
        if failure == 'interrupt':
            raise KeyboardInterrupt
        if failure == 'exit':
            raise SystemExit(7)
        return failure != 'timeout'
    audio.wait_until_idle.side_effect = wait_audio
    def close_audio():
        order.append('audio_close')
        if failure in ('close', 'recording_close'):
            raise RuntimeError('audio worker pending')

    if failure == 'play':
        audio.play.side_effect = RuntimeError('audio request failed')
    audio.close.side_effect = close_audio

    def initialize(loop, *args, **kwargs):
        if failure in ('recording', 'recording_close'):
            loop.phase = TeleopPhase.ACTIVE
            loop.run_id = loop.runtime_state.run_id
            loop.recorder = EpisodeRecorder(tmp_path, control_hz=30)
            loop.recorder.start_attempted = True
            loop.recorder.error = OSError('writer failed')
            loop.run_episode(loop.run_id)
        loop.runtime_state.request_quit()
        return False

    with patch('dexmani_real.teleop.run.validate_robot_config'), \
         patch('dexmani_real.teleop.run.ActionMapper'), \
         patch('dexmani_real.teleop.run.load_vr_transform'), \
         patch('dexmani_real.teleop.run.VRController'), \
         patch('dexmani_real.teleop.run.SensorChannels'), \
         patch('dexmani_real.teleop.run.SensorWorkers'), \
         patch('dexmani_real.teleop.run.Robot'), \
         patch('dexmani_real.teleop.run.Keyboard'), \
         patch('dexmani_real.teleop.run.AudioFeedback', return_value=audio), \
         patch('dexmani_real.teleop.run.initialize_teleop', side_effect=initialize), \
         patch('dexmani_real.teleop.run.close_runtime', side_effect=lambda *a, **k: order.append('devices')):
        kwargs = {
            "record": False, "episodes_dir": str(tmp_path), "max_duration_s": 1,
            "runtime_config_json": None,
        }
        if failure in ('interrupt', 'exit'):
            with pytest.raises(KeyboardInterrupt if failure == 'interrupt' else SystemExit):
                run_teleop(runtime, **kwargs)
        else:
            code = run_teleop(runtime, **kwargs)
            assert code == int(failure in ('recording', 'recording_close'))
    if failure in ('play', 'wait', 'close', 'recording_close'):
        logs = [record for record in caplog.records if record.message == 'teleop audio cleanup failed']
        assert len(logs) == 1 and logs[0].levelname == 'WARNING' and logs[0].exc_info
    if failure == 'timeout':
        assert '退出提示音未在 5 秒内播放完毕' in caplog.text
    if failure in ('recording', 'recording_close'):
        assert sum(record.exc_info is not None and isinstance(record.exc_info[1], RecordingError)
                   for record in caplog.records) == 1
        assert 'teleop capture cleanup failed' not in caplog.text
    assert order == (['devices', 'audio_close'] if failure == 'play' else
                     ['devices', 'audio_wait', 'audio_close'])
    audio.close.assert_called_once()
    audio.play.assert_called_with('emergency' if failure in ('recording', 'recording_close') else 'end',
                                  interrupt=failure in ('recording', 'recording_close'))


def test_emergency_during_popen_creation_cancels_old_generation():
    creating, release, killed = Event(), Event(), Event()
    first = Mock(returncode=-9)
    first.poll.side_effect = lambda: -9 if killed.is_set() else None
    first.kill.side_effect = killed.set
    first.communicate.return_value = (b'', b'')
    emergency = Mock(returncode=0)
    emergency.poll.return_value = 0
    emergency.communicate.return_value = (b'', b'')
    calls = []

    def popen(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            creating.set()
            assert release.wait(2)
            return first
        return emergency

    with patch('dexmani_real.teleop.audio.shutil.which', return_value='/bin/aplay'), \
         patch.object(Path, 'is_file', return_value=True), \
         patch('dexmani_real.teleop.audio.subprocess.Popen', side_effect=popen):
        audio = AudioFeedback()
        try:
            audio.play('begin')
            assert creating.wait(2)
            audio.play('emergency')
            release.set()
            assert audio.wait_until_idle(2)
            assert killed.is_set()
            assert len(calls) == 2
            assert audio.player == 'aplay'
        finally:
            release.set()
            audio.close()
