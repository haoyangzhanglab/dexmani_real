import json
import threading
import time

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


def test_actual_raw_writer_preserves_mode_and_missing_auxiliary(tmp_path):
    recorder = start_recording(tmp_path)
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
        assert "policy_trace" not in meta
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
    runner.controller = NS(clear_reference=lambda: None)
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
    runner.controller = NS(clear_reference=lambda: None)
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
        stop_required=False,
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


def test_writer_receives_frozen_termination(tmp_path, monkeypatch):
    from dexmani_real.recording import recorder as module

    recorder = start_recording(tmp_path)
    details = [dict(stage="pause_stop", exception_type="TimeoutError", message="STOP_TIMEOUT")]
    put = recorder._queue.put_nowait

    def submit(item):
        if item is module._STOP:
            # Simulate caller-owned state changing before the writer consumes STOP.
            details[0]["message"] = "mutated"
        return put(item)

    monkeypatch.setattr(recorder._queue, "put_nowait", submit)
    saved = recorder.save_episode(reason="operator", details=details)
    with h5py.File(saved / "data.h5") as raw:
        assert raw["meta"].attrs["termination_reason"] == "operator"
        assert json.loads(raw["meta/termination_details"][()])[0]["message"] == "STOP_TIMEOUT"
        assert "policy_trace" not in raw["meta"]


@pytest.mark.parametrize("failure", ["details_type", "details_nan", "copy", "interrupt"])
def test_final_metadata_failure_drains_native_writer(tmp_path, monkeypatch, failure):
    import av

    from dexmani_real.recording import recorder as module

    entered, release = threading.Event(), threading.Event()
    write_frame = module.VideoEncoder.write_frame
    close_threads = []

    def track_close(close):
        def wrapped(resource):
            close_threads.append(threading.get_ident())
            return close(resource)

        return wrapped

    for resource in (module.VideoEncoder, module.EpisodeDataWriter):
        monkeypatch.setattr(resource, "close", track_close(resource.close))

    def blocked_write(video, frame):
        entered.set()
        assert release.wait(3), "test did not release writer"
        return write_frame(video, frame)

    monkeypatch.setattr(module.VideoEncoder, "write_frame", blocked_write)
    recorder = start_recording(tmp_path)
    copies = []
    error = (
        KeyboardInterrupt("snapshot cancelled")
        if failure == "interrupt"
        else RuntimeError("snapshot failed")
    )

    class Uncopyable:
        def __deepcopy__(self, memo):
            copies.append(1)
            raise error

    try:
        assert entered.wait(3)
        for i in (1, 2):
            data = {
                name: np.full(spec.tail_shape, i, dtype=spec.dtype)
                for name, spec in DATASET_SPECS.items()
            }
            recorder.add_frame(
                EpisodeFrame(
                    data, np.zeros((16, 16, 3), np.uint8), np.full((16, 16), i + 1, np.uint16)
                )
            )
        assert recorder._queue.qsize() == 2
        details = [dict(stage="pause_stop", exception_type="TimeoutError", message="STOP_TIMEOUT")]
        if failure.startswith("details_"):
            details[0]["message"] = np.float32(1) if failure == "details_type" else float("nan")
        else:
            details.append(Uncopyable())
        put = recorder._queue.put_nowait

        def submit(item):
            result = put(item)
            if item is module._STOP:
                release.set()
            return result

        monkeypatch.setattr(recorder._queue, "put_nowait", submit)
        expected = KeyboardInterrupt if failure == "interrupt" else module.RecordingError
        with pytest.raises(expected) as raised:
            recorder.save_episode(reason="operator", details=details)
        cause = raised.value if failure == "interrupt" else raised.value.__cause__
        if failure in ("copy", "interrupt"):
            assert cause is error
            assert copies == [1]
        else:
            assert isinstance(cause, TypeError if failure.endswith("type") else ValueError)
        assert not recorder._thread.is_alive()
        assert recorder.resources_released
        assert close_threads and set(close_threads) == {recorder._thread.ident}
        assert not recorder.episode_path.exists()
        assert recorder._temp_dir.is_dir()
        with h5py.File(recorder._temp_dir / "data.h5") as raw:
            assert raw["meta"].attrs["termination_reason"] == "operator"
            np.testing.assert_array_equal(raw["observation_timestamp_ns"][:], [0, 1, 2])
            np.testing.assert_array_equal(raw["dispatch_status"][:], [[1, 1], [1, 1], [2, 2]])
            np.testing.assert_array_equal(raw["depth"][:, 0, 0], [1, 2, 3])
        with av.open(str(recorder._temp_dir / "rgb.mp4")) as video:
            assert len(list(video.decode(video=0))) == 3
        for _ in range(2):
            start = time.monotonic()
            with pytest.raises(module.RecordingError) as closed:
                recorder.close()
            assert closed.value.__cause__ is cause
            assert time.monotonic() - start < 1
        assert copies == ([1] if failure in ("copy", "interrupt") else [])
        assert recorder._reason == "operator"
    finally:
        # Release the writer even if an assertion fails.
        release.set()
        if recorder._thread is not None and recorder._thread.is_alive():
            recorder._store_error(RuntimeError("test cleanup"))
            recorder._thread.join(timeout=3)
        assert recorder.resources_released


def test_trace_only_in_attempt_and_submission_indices(tmp_path):
    from dexmani_real.recording.results import SessionResults, finalize_policy_attempt

    recorder = start_recording(tmp_path)
    # Slot numbers intentionally differ from submitted Raw row indices.
    results = SessionResults(tmp_path, "policy")
    attempt = results.prepare(recording=True)
    results.entered(8)
    trace = [
        dict(event="record_submitted", slot=17, raw_row_index=0),
        dict(event="dispatch", slot=17, arm=1, hand=4),
    ]
    finalize_policy_attempt(
        results,
        recorder,
        recording_started=True,
        reason="operator",
        detail="stopped",
        events=trace,
        query_arrays={"query_1": np.full((3, 19), np.nan)},
    )
    saved = json.loads((tmp_path / "attempts" / f"{attempt}.json").read_text())
    trace.clear()
    assert saved["recording_status"] == "published" and saved["row_count"] == 1
    assert saved["trace"][0]["raw_row_index"] == 0 and saved["trace"][0]["slot"] == 17
    with h5py.File(saved["raw_path"] + "/data.h5") as raw:
        assert "policy_trace" not in raw["meta"]
    with np.load(saved["query_sidecar"], allow_pickle=False) as arrays:
        assert np.isnan(arrays["query_1"]).all()


def test_sidecar_json_failure_preserves_published_raw(tmp_path, monkeypatch):
    from dexmani_real.recording.results import SessionResults, finalize_policy_attempt

    recorder = start_recording(tmp_path)
    results = SessionResults(tmp_path, "policy")
    attempt = results.prepare(recording=True)

    def fail(*a, **kw):
        raise OSError("attempt JSON failed")

    monkeypatch.setattr(results, "finish_attempt", fail)
    with pytest.raises(OSError, match="attempt JSON failed"):
        finalize_policy_attempt(
            results,
            recorder,
            recording_started=True,
            reason="operator",
            detail="first reason",
            events=[],
            query_arrays={},
        )
    assert recorder.episode_path.exists() and recorder.written_frames == 1
    incomplete = json.loads((tmp_path / "attempts" / f"{attempt}.json").read_text())
    assert incomplete["state"] == "incomplete"
    assert (tmp_path / "attempts" / f"{attempt}.npz").exists()


def test_direct_runner_rejects_recording_without_attempt_owner(tmp_path):
    from dexmani_real.deployment.runner import PolicyRunner

    recorder = AsyncEpisodeRecorder(tmp_path)
    with pytest.raises(ValueError, match="requires recorder, recording_config and results"):
        PolicyRunner(
            None,
            None,
            None,
            robot=None,
            realizer=None,
            recorder=recorder,
            model_runtime=None,
            execution_config=None,
            kinematics=None,
            execute=True,
            max_running_s=None,
        )
