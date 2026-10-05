import json

import h5py
import numpy as np
import pytest

from dexmani_real.recording.frame import EpisodeFrame
from dexmani_real.recording.recorder import AsyncEpisodeRecorder
from dexmani_real.recording.storage.schema import DATASET_SPECS
from dexmani_real.sensor.camera.geometry import CameraIntrinsics, RGBDGeometry


def start_recording(tmp_path):
    camera = CameraIntrinsics(16, 16, 10.0, 10.0, 8.0, 8.0, "none", (0.0,) * 5)
    geometry = RGBDGeometry(camera, camera, np.eye(4))
    recorder = AsyncEpisodeRecorder(
        tmp_path, control_hz=10, rgb_shape=(16, 16, 3), execution_path="worker_grid_rtc_v1"
    )
    recorder.start_episode(
        task_label="synthetic",
        collection_source="policy_rollout",
        camera_geometry=geometry,
        camera_T_xarm_base_from_color=None,
        depth_scale=0.001,
        handbase_position_eef_m=np.zeros(3),
        handbase_quat_eef_wxyz=np.array([1.0, 0, 0, 0]),
    )
    data = {
        name: np.zeros(spec.tail_shape, dtype=spec.dtype) for name, spec in DATASET_SPECS.items()
    }
    data["hand_contact"][:] = np.nan
    data["hand_tactile_force"][:] = np.nan
    data["dispatch_status"][:] = 1
    recorder.add_frame(
        EpisodeFrame(data, np.zeros((16, 16, 3), np.uint8), np.ones((16, 16), np.uint16))
    )
    return recorder


def test_actual_raw_writer_preserves_mode_trace_and_missing_auxiliary(tmp_path):
    recorder = start_recording(tmp_path)
    recorder.policy_trace = {
        "execute": True,
        "events": [{"event": "end", "reason": "operator", "detail": "after_predict"}],
    }
    saved = recorder.save_episode(reason="operator")
    recorder.close()
    from dexmani_real.recording.storage.reader import EpisodeReader

    with EpisodeReader(saved) as reader:
        assert reader.num_frames == 1
        assert reader.read_camera_frame("rgb", 0).shape == (16, 16, 3)
    with h5py.File(saved / "data.h5") as raw:
        meta = raw["meta"]
        assert meta.attrs["execution_path"] == "worker_grid_rtc_v1"
        assert meta.attrs["termination_reason"] == "operator"
        assert json.loads(meta["policy_trace"][()])["events"][0]["detail"] == "after_predict"
    with h5py.File(saved / "data.h5") as raw:
        assert np.isnan(raw["hand_contact"][:]).all()
        np.testing.assert_array_equal(raw["dispatch_status"][:], [[1, 1]])


@pytest.mark.parametrize("stop_fails", [False, True])
def test_teleop_operator_stop_evidence_saved_by_native_writer(tmp_path, stop_fails):
    from types import SimpleNamespace as NS

    from test_review_remediation import shared_state

    from dexmani_real.runtime.safety import RunEndReason
    from dexmani_real.teleop.runner import TeleopRunner

    runner = TeleopRunner.__new__(TeleopRunner)
    runner.shared = shared_state()
    runner.control_run_id = 1
    runner.pending_termination_reason = None
    runner.termination_details = []
    runner.controller = None
    runner.recorder = start_recording(tmp_path)

    def stop():
        if stop_fails:
            raise TimeoutError("STOP_TIMEOUT")

    runner.robot = NS(stop=stop)
    if stop_fails:
        with pytest.raises(TimeoutError):
            runner._pause_control("operator", run_end_reason=RunEndReason.OPERATOR)
    else:
        runner._pause_control("operator", run_end_reason=RunEndReason.OPERATOR)
    details = runner.termination_details
    runner._finish_capture(True, "shutdown", announce=False)
    assert runner.termination_details == []
    details.append(dict(stage="later", exception_type="RuntimeError", message="later"))
    saved = next(tmp_path.glob("episode_*"))
    with h5py.File(saved / "data.h5") as raw:
        assert raw["meta"].attrs["termination_reason"] == "operator"
        assert len(raw["dispatch_status"]) == 1
        if stop_fails:
            assert json.loads(raw["meta/termination_details"][()]) == [
                dict(stage="pause_stop", exception_type="TimeoutError", message="STOP_TIMEOUT")
            ]
        else:
            assert "termination_details" not in raw["meta"]
    # A new writer capture starts without previous optional details.
    runner.recorder = start_recording(tmp_path / "next")
    saved = runner.recorder.save_episode()
    with h5py.File(saved / "data.h5") as raw:
        assert "termination_details" not in raw["meta"]


def test_writer_failure_preserves_staging_with_session_error(tmp_path, monkeypatch):
    from dexmani_real.recording import recorder as module

    recorder = start_recording(tmp_path)

    def fail(*a, **kw):
        raise OSError("disk failure")

    monkeypatch.setattr(module.EpisodeDataWriter, "update_meta", fail)
    with pytest.raises(module.RecordingError, match="disk failure"):
        recorder.save_episode(
            reason="operator",
            details=[
                dict(stage="pause_stop", exception_type="TimeoutError", message="STOP_TIMEOUT")
            ],
        )
    staging = list(tmp_path.glob(".tmp_*"))
    assert staging
    with h5py.File(staging[0] / "data.h5") as partial:
        assert len(partial["dispatch_status"]) == 1
    assert not list(tmp_path.glob("episode_*"))
    with pytest.raises(module.RecordingError):
        recorder.close()


@pytest.mark.parametrize("operator,reason", [("STOP", "operator"), ("QUIT", "quit")])
def test_teleop_run_finally_saves_stop_failure(tmp_path, operator, reason):
    from types import SimpleNamespace as NS

    from test_review_remediation import shared_state

    from dexmani_real.runtime.operator_input import OperatorCommand
    from dexmani_real.teleop.runner import TeleopRunner

    runner = TeleopRunner.__new__(TeleopRunner)
    runner.shared = shared_state()
    runner.control_run_id = 1
    runner.pending_termination_reason = None
    runner.termination_details = []
    runner.controller = None
    runner.recorder = start_recording(tmp_path)
    runner.active, runner.paused, runner.quit_pending = True, False, False
    runner.next_tick = 0
    runner._initialize = lambda: True

    def stop():
        raise TimeoutError("STOP_TIMEOUT")

    runner.robot = NS(
        stop=stop,
        check=lambda: None,
        check_services=None,
        _motion_active=False,
        _hand_stop_pending=False,
    )
    runner.keyboard = NS(
        start=lambda: None,
        stop=lambda: None,
        healthy=True,
        poll=lambda **kw: [getattr(OperatorCommand, operator)],
    )
    runner.audio = NS(play=lambda _: None, close=lambda: None, wait_until_idle=lambda **kw: True)
    with pytest.raises(TimeoutError, match="STOP_TIMEOUT"):
        runner.run()
    saved = next(tmp_path.glob("episode_*"))
    with h5py.File(saved / "data.h5") as raw:
        assert raw["meta"].attrs["termination_reason"] == reason
        assert len(json.loads(raw["meta/termination_details"][()])) == 1
        assert len(raw["dispatch_status"]) == 1


def test_writer_receives_frozen_termination_and_policy_trace(tmp_path, monkeypatch):
    from dexmani_real.recording import recorder as module

    recorder = start_recording(tmp_path)
    details = [dict(stage="pause_stop", exception_type="TimeoutError", message="STOP_TIMEOUT")]
    trace = {"events": [{"event": "end", "reason": "operator"}]}
    recorder.policy_trace = trace
    put = recorder._queue.put_nowait

    def submit(item):
        if item is module._STOP:
            # Simulate caller-owned state changing before the writer consumes STOP.
            details[0]["message"] = "mutated"
            trace["events"].clear()
        return put(item)

    monkeypatch.setattr(recorder._queue, "put_nowait", submit)
    saved = recorder.save_episode(reason="operator", details=details)
    with h5py.File(saved / "data.h5") as raw:
        assert json.loads(raw["meta/termination_details"][()])[0]["message"] == "STOP_TIMEOUT"
        assert len(json.loads(raw["meta/policy_trace"][()])["events"]) == 1
