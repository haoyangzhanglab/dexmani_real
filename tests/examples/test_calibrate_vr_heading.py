from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from examples import calibrate_vr_heading as calibration


class _FakeValue:
    def __init__(self, value: bool) -> None:
        self.value = value


class _FakeRing:
    def __init__(self) -> None:
        self.sequence = 0

    def read_latest(self):
        self.sequence += 1
        return (
            {
                "head_pos": np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
                "head_quat_wxyz": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
                "wrist_quat_wxyz": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
            },
            1,
            self.sequence,
        )


class _FakeChannels:
    def __init__(self) -> None:
        self.is_running = _FakeValue(True)
        self.vr_ring = _FakeRing()

    def wait_ready(self, _name: str, _timeout_s: float) -> bool:
        return True

    def close(self) -> bool:
        return True


class _FakeProcess:
    pid = 123


def test_unclean_shutdown_refuses_calibration_publish(monkeypatch) -> None:
    cfg = SimpleNamespace(
        duration_s=1.0,
        port=8000,
        vr_ready_timeout_s=1.0,
        tracking_data_timeout_s=1.0,
        settle_s=0.0,
        min_frames=5,
        outlier_sigma=3.0,
        excellent_std_deg=2.0,
        good_std_deg=5.0,
    )
    shared = _FakeChannels()
    monotonic_values = iter(
        [0.0, 0.10, 0.11, 0.20, 0.21, 0.30, 0.31, 0.40, 0.41, 0.50, 0.51, 2.0]
    )
    published = False

    monkeypatch.setattr(calibration, "HeadingCalibrationConfig", lambda: cfg)
    monkeypatch.setattr(calibration.RuntimeChannels, "create", lambda **_kwargs: shared)
    monkeypatch.setattr(calibration, "build_processes", lambda *_args: [_FakeProcess()])
    monkeypatch.setattr(calibration, "start_processes", lambda _processes: None)
    monkeypatch.setattr(calibration, "_wait_for_vr_tracking", lambda *_args: True)
    monkeypatch.setattr(calibration.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(calibration.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        calibration,
        "shutdown_processes_verified",
        lambda *_args, **_kwargs: SimpleNamespace(
            clean=False, exits=(("vr-calib", -15, "terminate"),)
        ),
    )

    def _record_publish(*_args, **_kwargs) -> None:
        nonlocal published
        published = True

    monkeypatch.setattr(calibration, "atomic_json_dump", _record_publish)

    assert calibration.main([]) == 1
    assert not published


def test_audio_timeout_is_reported_without_claiming_success(
    monkeypatch, capsys
) -> None:
    events: list[object] = []

    class _FakeAudioFeedback:
        def play(self, event: str) -> None:
            events.append(("play", event))

        def wait_until_idle(self, *, timeout_s: float) -> bool:
            events.append(("wait", timeout_s))
            return False

        def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(calibration, "AudioFeedback", _FakeAudioFeedback)

    calibration._play_completion_audio()

    assert events == [
        ("play", "calibrated"),
        ("wait", calibration._AUDIO_IDLE_TIMEOUT_S),
        "close",
    ]
    output = capsys.readouterr().out
    assert "timed out" in output
    assert "played" not in output
