"""调用真实 Raw/Canonical 与首帧选择链路；设备和感知模型使用替身。"""

import json
from dataclasses import replace
from itertools import count
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import h5py
import numpy as np
import pytest
import zarr

from dexmani_real.config.perception import PointCloudConfig
from dexmani_real.config.real import RealConfig
from dexmani_real.data import recorder as recorder_module
from dexmani_real.data.export import (
    ExportConfig,
    canonical_array_specs,
    export_zarr,
    validate_export_episode,
)
from dexmani_real.data.raw import (
    RawReader,
    RawWriter,
    _VideoWriter,
    snapshot_recording_metadata,
)
from dexmani_real.data.recorder import EpisodeRecorder, RecordingError
from dexmani_real.deployment.inference import MODEL_SHUTDOWN_TIMEOUT_S, PolicyWorker
from dexmani_real.deployment.rollout import Rollout
from dexmani_real.deployment.schedule import Prediction
from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE
from dexmani_real.runtime.observation import Observation
from dexmani_real.runtime.state import RunEndReason, SessionState
from dexmani_real.utils import io
from dexmani_real.utils.log import error_detail


@pytest.fixture
def capture():
    camera = CameraSnapshot(
        "synthetic", CameraIntrinsics(32, 32, 200., 200., 15.5, 15.5, "none", (0.,) * 5),
        .001, np.eye(4),
    )
    metadata = snapshot_recording_metadata(
        camera, tactile_reset_metadata={}, tactile_hold_max_age_s=.5,
    )

    def row(index):
        stamp = 1_000_000_000 + index * 100_000_000
        arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
        hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
        arm["timestamp_ns"] = stamp
        for name in ("timestamp_ns", "tactile_aggregate_timestamp_ns", "tactile_dense_timestamp_ns"):
            hand[name] = stamp
        hand["tactile_aggregate"] = np.arange(15).reshape(5, 3)
        hand["tactile_dense"] = np.arange(1800).reshape(5, 120, 3)
        return Observation(
            arm=arm, hand=hand,
            camera={"rgb": np.full((32, 32, 3), 80, np.uint8),
                    "depth": np.full((32, 32), 500, np.uint16),
                    "timestamp_ns": stamp, "ring_sequence": index // 2},
            vr={"wrist_pos": np.zeros(3), "wrist_quat_wxyz": np.array([1., 0., 0., 0.]),
                "landmarks": np.zeros((21, 3)), "recv_ts_ns": stamp, "sequence_id": index},
            point_cloud=None, observation_timestamp_ns=stamp + 1_000_000,
            control_timestamp_ns=stamp,
        )

    return metadata, row, RobotCommand(1, np.zeros(7), np.zeros(12))


def record(path, capture, *, frames=3, save=True, status="normal"):
    metadata, row, command = capture
    recorder = EpisodeRecorder(path, control_hz=10)
    recorder.start(task_label="fixture", collection_source="teleop", metadata=metadata)
    for index in range(frames):
        recorder.append(row(index), command)
    result = recorder.finish(save=save, reason="test", technical_status=status)
    return recorder, result


@pytest.fixture(params=["mask.npy", "mask.png"])
def initial_selection(tmp_path, capture, monkeypatch, request):
    monkeypatch.syspath_prepend(str(
        Path(__file__).resolve().parents[1] / "third_party/cordvip_geometry/src"
    ))
    from cordvip_geometry.io import write_confirmed_selection
    from cordvip_geometry.raw import iter_raw_rows
    from cordvip_geometry.rgbd import save_frame

    metadata, _, _ = capture
    metadata["camera_color_distortion_model"] = "brown_conrady"
    _, episode = record(tmp_path / "raw", capture, frames=1)
    with RawReader(episode) as reader:
        [(frame, _)] = iter_raw_rows(reader, reader.read_camera())
    mask = np.indices(frame.depth_m.shape).sum(axis=0) % 2 == 0
    directory = tmp_path / "selection"
    directory.mkdir()
    save_frame(directory / "source_rgbd.npz", frame)
    write_confirmed_selection(directory, mask, {
        "selection": "gui_confirmed", "selected_instance": None,
        "mask_grid": "original_rgb_grid; not_automatically_rectified",
    }, mask_file=request.param)
    return episode, directory, frame, mask


def test_initial_selection_reuses_mask_after_raw_move(tmp_path, initial_selection):
    from cordvip_geometry.cli.scene import choose_masks, load_scene
    from cordvip_geometry.geometry import RectificationMaps
    from cordvip_geometry.io import load_initial_selection
    from cordvip_geometry.raw import iter_raw_rows
    from cordvip_geometry.rgbd import load_frame
    from cordvip_geometry.tracking import SceneTracker

    episode, directory, original, mask = initial_selection
    np.testing.assert_array_equal(load_initial_selection(directory, original), mask)
    moved = episode.rename(tmp_path / "moved_episode")
    with RawReader(moved) as reader:
        [(frame, _)] = iter_raw_rows(reader, reader.read_camera())
    assert frame.source_id != original.source_id
    np.testing.assert_array_equal(load_initial_selection(directory, frame), mask)
    assert load_frame(directory / "source_rgbd.npz").source_id == original.source_id

    output = tmp_path / "tracking"
    output.mkdir()
    segmenter = Mock()
    segmenter.initialize.side_effect = lambda rgb, masks: masks
    masks = choose_masks(
        {"object": {"id": "object", "initial_selection": directory}},
        frame, SimpleNamespace(), output, segmenter,
        RectificationMaps(frame.depth_m.shape, frame.K, frame.distortion_coeffs),
    )
    np.testing.assert_array_equal(masks["object"], mask)
    segmenter.predict.assert_not_called()
    scene = load_scene(output / "scene.json", headless=True)
    tracker = SceneTracker(
        segmenter, {}, foundationpose_root=tmp_path, debug_dir=tmp_path,
        initial_selections={name: item["initial_selection"] for name, item in scene.items()},
    )
    result = tracker.step(frame)
    assert result["operation"] == "initialize"
    np.testing.assert_array_equal(result["objects"]["object"]["cloud"].semantic_mask, mask)


@pytest.mark.parametrize("key,change", [
    ("rgb", lambda value: value ^ np.uint8(1)),
    ("depth_m", lambda value: np.nextafter(value, np.float32(np.inf))),
    ("K", lambda value: value + np.diag([1e-12, 0., 0.])),
    ("distortion_model", lambda value: "none"),
    ("distortion_coeffs", lambda value: (1e-12, *value[1:])),
    ("timestamp_ns", lambda value: value + 1),
    ("clock", lambda value: "host_monotonic_observation"),
    ("row_index", lambda value: value + 1),
])
def test_initial_selection_rejects_changed_binding(initial_selection, key, change):
    from cordvip_geometry.io import load_initial_selection

    _, directory, frame, _ = initial_selection
    changed = replace(frame, **{key: change(getattr(frame, key))})
    with pytest.raises(ValueError, match=f"initial-selection {key} differs"):
        load_initial_selection(directory, changed)


@pytest.mark.parametrize("mask_file", ["../outside.npy", "linked.npy"])
def test_initial_selection_rejects_mask_outside_directory(tmp_path, initial_selection, mask_file):
    from cordvip_geometry.io import load_initial_selection

    _, directory, frame, mask = initial_selection
    outside = tmp_path / "outside.npy"
    np.save(outside, mask, allow_pickle=False)
    if mask_file == "linked.npy":
        (directory / mask_file).symlink_to(outside)
    metadata_path = directory / "selection.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["mask_file"] = mask_file
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="mask must remain inside its directory"):
        load_initial_selection(directory, frame)


@pytest.mark.parametrize("frames,save,status", [(3, True, "normal"), (3, True, "failed"),
                                                   (0, True, "normal"), (3, False, "normal")])
def test_raw_publication_and_video(tmp_path, capture, frames, save, status):
    recorder, result = record(tmp_path, capture, frames=frames, save=save, status=status)
    assert not recorder.writer_pending
    assert recorder.written_frames == frames
    if frames and save:
        assert result == recorder.published_path
        with RawReader(result) as reader:
            assert reader.num_frames == frames
            assert reader.meta["technical_status"] == status
            assert not ({"hand_tactile_processing", "hand_tactile_axes", "vr_hand_side",
                         "vr_coordinate_axes", "vr_position_units", "vr_landmark_order",
                         "vr_timestamp_source"} & set(reader.meta))
            assert not ({"clock", "missing_value", "camera_timestamp_source"}
                        & set(reader.data_file["timing"].attrs))
            assert reader.meta["timestamp_source"] == "host_monotonic_ns"
            np.testing.assert_array_equal(reader["depth"][0], 500)
            assert len(list(reader.iter_rgb_frames())) == frames
    else:
        assert result is None
        assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("case,reason,row_index", [
    ("normal", None, None),
    ("jitter", None, None),
    ("gap", "control_interval_gap_or_drift", 1),
    ("nan", "nonfinite:action_arm_joint_target", 2),
    ("inf", "nonfinite:hand_tactile_force", 2),
    ("failed", "technical failure", None),
    ("tactile", "stale_source:tactile_dense_ns", 2),
    ("readout_between", None, None),
    ("readout_limit", "observation_crossed_control_period", 2),
    ("source_future", "invalid_source_clock:arm_ns", 2),
    ("source_missing", "invalid_source_clock:hand_ns", 2),
    ("source_regressed", "source_clock_regressed:camera_ns", 2),
    ("sequence_regressed", "camera_sequence_regressed", 2),
])
def test_training_admits_only_complete_episode(tmp_path, capture, case, reason, row_index):
    _, path = record(tmp_path / "raw", capture, status="failed" if case == "failed" else "normal")
    with h5py.File(path / "data.h5", "r+") as data:
        if case == "nan":
            data["action_arm_joint_target"][2] = np.full(7, np.nan)
        elif case == "inf":
            data["hand_tactile_force"][2] = np.full((5, 120, 3), np.inf)
        else:
            rows = data["timing/rows"][:]
            if case == "gap":
                rows["control_ns"][1:] += 100_000_000
                rows["observation_ns"][1:] += 100_000_000
            elif case == "jitter":
                # dt/4 边界仍接纳，来源值不随导出重采样。
                rows["control_ns"][1] += 25_000_000
                rows["observation_ns"][1] += 25_000_000
            elif case.startswith("readout"):
                readout_ns = 30_000_000 if case == "readout_between" else 100_000_000
                rows["observation_ns"][2] = rows["control_ns"][2] + readout_ns
            elif case == "tactile":
                rows["tactile_dense_ns"] = rows["control_ns"][0]
                data["meta"].attrs["hand_tactile_hold_max_age_s"] = .15
            elif case == "source_future":
                rows["arm_ns"][2] = rows["observation_ns"][2] + 1
            elif case == "source_missing":
                rows["hand_ns"][2] = -1
            elif case == "source_regressed":
                rows["camera_ns"][2] = rows["camera_ns"][0]
            elif case == "sequence_regressed":
                rows["camera_sequence"] = [0, 1, 0]
            data["timing/rows"][:] = rows
    before = {file.name: file.read_bytes() for file in path.iterdir()}
    output = tmp_path / "canonical.zarr"
    processing = ExportConfig(pointcloud=PointCloudConfig(
        remove_table=False, num_points=32, workspace=(-1., -1., 0., 1., 1., 1.),
    ))
    if reason is not None:
        with pytest.raises(ValueError, match=reason) as caught:
            export_zarr(path, output, processing=processing)
        assert str(path) in str(caught.value)
        if row_index is not None:
            assert f"row {row_index}" in str(caught.value)
        assert not output.exists()
        assert not list(tmp_path.glob(".canonical.zarr.tmp-*"))
    else:
        result = export_zarr(path, output, processing=processing)
        root = zarr.open_group(str(output), mode="r")
        assert result["total_frames"] == 3
        np.testing.assert_array_equal(root["meta/episode_ends"][:], [3])
        assert all(array.shape[0] == 3 for _, array in root["data"].arrays())
        quality = root.attrs["episode_quality"][0]
        assert set(quality) == {"max_control_grid_deviation_ns", "max_readout_ns"}
        if case == "readout_between":
            assert quality["max_readout_ns"] == 30_000_000
        if case == "jitter":
            assert quality["max_control_grid_deviation_ns"] == 25_000_000
    assert before == {file.name: file.read_bytes() for file in path.iterdir()}


def test_canonical_export_real_fk_and_thirteen_fields(tmp_path, capture):
    _, path = record(tmp_path / "raw", capture)
    output = tmp_path / "canonical.zarr"
    result = export_zarr(path, output, processing=ExportConfig(pointcloud=PointCloudConfig(
        remove_table=False, num_points=32, workspace=(-1., -1., 0., 1., 1., 1.),
    )))
    assert result["total_frames"] == 3
    root = zarr.open_group(str(output), mode="r")
    specs = canonical_array_specs(32, 32, 32, 1024)
    assert len(specs) == 13
    assert set(root["data"].array_keys()) == set(specs)
    for name, (shape, dtype) in specs.items():
        array = root["data"][name]
        assert array.shape == (3, *shape)
        assert array.dtype == dtype
        np.testing.assert_array_equal(array[0], array[1])
    np.testing.assert_array_equal(root["data/finger_tactile_force"][0], np.arange(15).reshape(5, 3))
    np.testing.assert_array_equal(root["data/taxel_tactile_force"][0], np.arange(1800).reshape(5, 120, 3))
    assert root.attrs["source_id"]
    assert root.attrs["dt"] == .1
    assert "timing_admission" not in root.attrs


@pytest.mark.parametrize("failure", ["failed", "rgb_short", "rgb_extra", "empty_depth_tail"])
def test_batch_export_rejects_whole_episode_without_publishing_or_changing_raw(tmp_path, capture, failure):
    source = tmp_path / "raw"
    _, good = record(source, capture)
    _, bad = record(source, capture, status="failed" if failure == "failed" else "normal")
    if failure.startswith("rgb"):
        video = _VideoWriter(bad / "rgb.mp4", fps=10, width=32, height=32)
        video.open()
        try:
            for _ in range(2 if failure == "rgb_short" else 4):
                video.write_frame(np.full((32, 32, 3), 80, np.uint8))
        finally:
            video.close()
    elif failure == "empty_depth_tail":
        with h5py.File(bad / "data.h5", "r+") as data:
            data["depth"][2] = np.zeros((32, 32), np.uint16)
    before = {file: file.read_bytes() for path in (good, bad) for file in path.iterdir()}
    output = tmp_path / "canonical.zarr"
    processing = ExportConfig(pointcloud=PointCloudConfig(
        remove_table=False, num_points=32, workspace=(-1., -1., 0., 1., 1., 1.),
    ))
    with pytest.raises(ValueError) as caught:
        export_zarr(source, output, processing=processing)
    assert str(bad) in str(caught.value)
    if failure != "failed":
        assert f"row {3 if failure == 'rgb_extra' else 2}" in str(caught.value)
    assert not output.exists()
    # 失败 staging 继续保留用于诊断，绝不作为可见 Canonical 发布。
    assert len(list(tmp_path.glob(".canonical.zarr.tmp-*"))) == 1
    result = export_zarr(source, output, processing=processing, exclude=(bad.name,))
    assert result["episode_count"] == 1 and result["total_frames"] == 3
    assert result["excluded_episodes"] == [bad.name]
    assert zarr.open_group(str(output), mode="r").attrs["episode_ids"] == [good.name]
    assert before == {file: file.read_bytes() for path in (good, bad) for file in path.iterdir()}


@pytest.mark.parametrize("field", ["task_label", "technical_status"])
def test_export_missing_metadata_reports_episode_without_row_number(tmp_path, capture, field):
    _, path = record(tmp_path / "raw", capture)
    with h5py.File(path / "data.h5", "r+") as data:
        del data["meta"].attrs[field]
    output = tmp_path / "canonical.zarr"
    with pytest.raises(ValueError) as caught:
        export_zarr(path, output, processing=ExportConfig())
    assert str(path) in str(caught.value) and field in str(caught.value)
    assert "at row" not in str(caught.value)
    assert isinstance(caught.value.__cause__, KeyError)
    assert not output.exists() and not list(tmp_path.glob(".canonical.zarr.tmp-*"))


@pytest.mark.parametrize("completed", [False, True])
def test_export_progress_failure_preserves_exception_and_publication(tmp_path, capture, completed):
    _, path = record(tmp_path / "raw", capture)
    output = tmp_path / "canonical.zarr"
    failure = ValueError("progress callback failed")
    def progress(done, total):
        if (done == total) == completed:
            raise failure
    processing = ExportConfig(pointcloud=PointCloudConfig(
        remove_table=False, num_points=32, workspace=(-1., -1., 0., 1., 1., 1.),
    ))
    with pytest.raises(ValueError) as caught:
        export_zarr(path, output, processing=processing, progress_callback=progress)
    assert caught.value is failure
    assert output.exists() == completed
    assert not list(tmp_path.glob(".canonical.zarr.tmp-*"))
    if completed:
        root = zarr.open_group(str(output), mode="r")
        np.testing.assert_array_equal(root["meta/episode_ends"][:], [3])
        assert all(array.shape[0] == 3 for _, array in root["data"].arrays())


@pytest.mark.parametrize("source", ["policy_rollout", "other"])
def test_export_rejects_non_teleop_raw(tmp_path, capture, source):
    _, path = record(tmp_path / "raw", capture)
    with h5py.File(path / "data.h5", "r+") as data:
        data["meta"].attrs["collection_source"] = source
    with RawReader(path) as reader, pytest.raises(ValueError, match=f"got collection_source={source!r}"):
        validate_export_episode(reader)
    output = tmp_path / "canonical.zarr"
    with pytest.raises(ValueError, match=f"got collection_source={source!r}"):
        export_zarr(path, output, processing=ExportConfig())
    assert not output.exists()
    assert not list(tmp_path.glob(".canonical.zarr.tmp-*"))


@pytest.mark.parametrize("recording", [False, True])
def test_soft_timeout_summary_keeps_raw_nanoseconds(tmp_path, capture, caplog, recording):
    from test_interaction import begin, make_loop

    loop = make_loop()
    begin(loop)
    if recording:
        metadata, row, command = capture
        recorder = loop.recorder = EpisodeRecorder(tmp_path, control_hz=10)
        recorder.start(task_label="fixture", collection_source="teleop", metadata=metadata)
        recorder.append(row(0), command)
    caplog.set_level("INFO", logger="dexmani_real.teleop.loop")
    loop.record_soft_timeout("observation", deadline_ns=1_000_000_000, observed_ns=1_100_000_000)
    loop.record_soft_timeout("compute", deadline_ns=2_000_000_000, observed_ns=2_012_345_000)
    before_end = list(caplog.records)
    last = loop.last_soft_timeout.copy()
    loop.request_episode_end("operator", run_end_reason=RunEndReason.OPERATOR, failed=False)
    loop.end_episode()
    assert not any(record.getMessage().startswith("Teleop soft timeouts:") for record in before_end)
    summaries = [record for record in caplog.records
                 if record.getMessage().startswith("Teleop soft timeouts:")]
    assert len(summaries) == 1
    assert summaries[0].levelname == "INFO"
    assert summaries[0].getMessage().endswith("last_lateness_ms=12.345")
    expected = {"stage": "soft_timeouts", "count": 2, "last": last}
    details = [item for item in loop.termination_details if item["stage"] == "soft_timeouts"]
    assert details == ([expected] if recording else [])
    assert set(last) == {"stage", "tick", "deadline_ns", "observed_ns"}
    if recording:
        with RawReader(recorder.published_path) as reader:
            persisted = json.loads(reader.data_file["meta/termination_details"][()])
        assert next(item for item in persisted if item["stage"] == "soft_timeouts") == expected
    assert loop.soft_timeout_count == 0
    assert loop.last_soft_timeout is None
    loop.robot.stop.assert_called_once()
    assert [call.args[0] for call in loop.audio.play.call_args_list] == (["save"] if recording else [])


def test_export_checks_full_raw_field_shape(tmp_path, capture):
    _, path = record(tmp_path, capture)
    with h5py.File(path / "data.h5", "r+") as data:
        del data["action_hand_joint_target"]
        data.create_dataset("action_hand_joint_target", data=np.zeros((3, 11)))
    with RawReader(path) as reader, pytest.raises(ValueError, match="incompatible shape/dtype"):
        validate_export_episode(reader)


@pytest.mark.parametrize("name,value", [("hand_tactile_units", "N"),
                                        ("hand_tactile_finger_order", "pinky,ring,middle,index,thumb"),
                                        ("hand_tactile_hold_max_age_s", float("nan"))])
def test_tactile_numeric_semantics_still_required(tmp_path, capture, name, value):
    _, path = record(tmp_path, capture)
    with h5py.File(path / "data.h5", "r+") as data:
        data["meta"].attrs[name] = value
    with RawReader(path) as reader, pytest.raises(ValueError):
        reader.validate_tactile_metadata()


def test_raw_close_attempts_all_resources_and_raises_first(caplog):
    writer = RawWriter("unused", control_hz=10)
    primary, secondary = OSError("video failed"), OSError("hdf5 failed")
    writer.video = Mock(close=Mock(side_effect=primary))
    writer.data_file = Mock(close=Mock(side_effect=secondary))
    with pytest.raises(OSError) as caught:
        writer.close()
    assert caught.value is primary
    writer.data_file.close.assert_called_once()
    assert "hdf5_close also failed" in caplog.text
    assert not hasattr(primary, "cleanup_errors")


def test_reader_context_preserves_primary_and_reports_close(tmp_path, capture, monkeypatch, caplog):
    _, path = record(tmp_path, capture)
    reader = RawReader(path)
    real_close = reader.close

    def failed_close():
        real_close()
        raise OSError("injected reader close failure")

    monkeypatch.setattr(reader, "close", failed_close)
    primary = ValueError("input failure")
    with pytest.raises(ValueError) as caught:
        with reader:
            raise primary
    assert caught.value is primary
    assert "injected reader close failure" in caplog.text


def test_recording_first_cause_and_secondary_failure_are_flat(caplog):
    recorder = EpisodeRecorder("unused", control_hz=10)
    primary, secondary = OSError("write failure"), OSError("close failure")
    recorder.store_error(primary)
    recorder.store_error(secondary, stage="writer_close")
    with pytest.raises(RecordingError) as caught:
        recorder.check_error()
    assert caught.value.__cause__ is primary
    assert error_detail("recording", caught.value)["message"] == (
        "Recording failed: OSError: write failure"
    )
    with pytest.raises(RecordingError) as repeated:
        recorder.check_error()
    assert repeated.value is caught.value
    assert repeated.value.__cause__ is primary
    assert not hasattr(primary, "cleanup_errors")
    assert "writer_close also failed" in caplog.text


def test_exited_writer_can_publish_after_supervisor_wakes_late(tmp_path, capture, monkeypatch):
    metadata, row, command = capture
    recorder = EpisodeRecorder(tmp_path, control_hz=10)
    recorder.start(task_label="fixture", collection_source="teleop", metadata=metadata)
    recorder.append(row(0), command)
    now = [0.0]
    join = recorder.thread.join

    def delayed_join(timeout):
        join(timeout)
        assert not recorder.writer_pending
        now[0] = recorder_module.RECORDER_STOP_TIMEOUT_S + 1

    monkeypatch.setattr(recorder.thread, "join", delayed_join)
    monkeypatch.setattr(recorder_module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    path = recorder.finish(save=True, reason="test", technical_status="normal")
    assert recorder.result()["status"] == "published"
    with RawReader(path) as reader:
        assert reader.num_frames == recorder.written_frames == 1
        reader.validate_completion()
        np.testing.assert_array_equal(reader["action_arm_joint_target"][0], command.arm_qpos)
        assert len(list(reader.iter_rgb_frames())) == 1


@pytest.mark.parametrize("writer_failure", [False, True])
@pytest.mark.parametrize("model_failure", ["pending", "error"])
def test_rollout_writer_finishes_independently_of_model(
    tmp_path, capture, monkeypatch, writer_failure, model_failure,
):
    metadata, row, command = capture
    entered, release = Event(), Event()
    events = []
    late_error = RuntimeError("late prediction failure")

    def predict(rows):
        entered.set()
        assert release.wait(5)
        if model_failure == "error":
            raise late_error
        return np.zeros((6, 19))

    worker = PolicyWorker(
        lambda: SimpleNamespace(predict=predict, close=lambda: events.append("model_close")),
        build_observation=lambda rows: rows,
    )
    state = SessionState()
    epoch = state.arm()
    state.stop_signal.clear()
    state.begin_motion()

    def stop():
        assert state.stop_signal.is_set()
        events.append("stop")

    episode = Rollout(
        state, RealConfig(), SimpleNamespace(control_dt_s=.1, n_obs_steps=1),
        epoch=epoch, robot=SimpleNamespace(stop=stop), observations=None, needs=None,
        sensors=None, action_mapper=None, model=worker, max_running_s=None,
        capture_ids=count(1), index=1, output_dir=tmp_path, record_raw=True, audio=Mock(),
    )
    episode.started_ns = 1
    episode.query = Prediction(0, None)
    recorder = episode.recorder
    writer_close = RawWriter.close

    def close_writer(writer):
        assert events == ["stop"] and worker.has_pending_task
        assert worker.future.running()
        writer_close(writer)
        events.append("writer_close")
        if writer_failure:
            raise OSError("writer close failed")

    monkeypatch.setattr(RawWriter, "close", close_writer)
    now = [0.0]
    monkeypatch.setattr(recorder_module, "time", SimpleNamespace(monotonic=lambda: now[0]))

    def prepare(poll):
        recorder.start(task_label="fixture", collection_source="policy_rollout",
                       metadata=metadata, episode_name="episode_001")
        recorder.append(row(0), command)
        join = recorder.thread.join

        def finish_join(timeout):
            assert timeout == recorder_module.RECORDER_STOP_TIMEOUT_S
            join(timeout=2)
            assert not recorder.writer_pending
            # 模拟 writer 消耗完整预算；模型仍应得到自己的完整等待预算。
            now[0] += recorder_module.RECORDER_STOP_TIMEOUT_S

        monkeypatch.setattr(recorder.thread, "join", finish_join)
        episode.stop("operator", run_end_reason=RunEndReason.OPERATOR)

    discard_pending = worker.discard_pending

    def retire(timeout):
        assert timeout == MODEL_SHUTDOWN_TIMEOUT_S
        assert events == ["stop", "writer_close"]
        assert recorder.result()["status"] == ("failed" if writer_failure else "published")
        events.append("model_retire")
        if model_failure == "error":
            release.set()
            worker.future.exception(timeout=2)
        discard_pending(0)  # 离线测试不实际等待完整的模型超时。

    monkeypatch.setattr(worker, "discard_pending", retire)
    episode.preparation_valid = lambda: True
    episode.prepare = prepare
    keyboard = SimpleNamespace(discard_events=lambda *args: events.append("keys"))
    try:
        worker.load()
        worker.future.result(timeout=2)
        worker.poll()
        worker.predict(())
        assert entered.wait(2)
        expected = RecordingError if writer_failure else TimeoutError if model_failure == "pending" else RuntimeError
        with pytest.raises(expected) as caught:
            episode.run(keyboard=keyboard, poll_operator=lambda: None)
        assert events == ["stop", "writer_close", "model_retire", "keys"]
        assert [error["stage"] for error in episode.result.errors] == (
            ["recorder_finalize", "retired_model"] if writer_failure else ["retired_model"]
        )
        assert not episode.result.normal_termination
        assert worker.has_pending_task == (model_failure == "pending")
        assert episode.query is None
        if worker.has_pending_task:
            with pytest.raises(RuntimeError, match="unreclaimed"):
                worker.predict(())
        if writer_failure:
            assert caught.value is recorder.error
            assert episode.result.raw["status"] == "failed" and recorder.published_path is None
        else:
            assert episode.result.raw["status"] == "published" and recorder.error is None
            if model_failure == "error":
                assert caught.value is late_error
            with RawReader(recorder.published_path) as reader:
                assert reader.num_frames == 1
                reader.validate_completion()
                assert reader.meta["technical_status"] == "normal"
                np.testing.assert_array_equal(reader["action_arm_joint_target"][0], command.arm_qpos)
    finally:
        release.set()
        worker.close(state)
        assert worker.wait_closed(2) == ("completed", [])


@pytest.mark.parametrize("failure", ["writer", "queue"])
def test_finish_timeout_stays_failed_after_writer_exits(tmp_path, capture, monkeypatch, failure):
    entered, release = Event(), Event()
    append = RawWriter.append_frame

    def blocked_append(writer, *args):
        entered.set()
        assert release.wait(5)
        return append(writer, *args)

    monkeypatch.setattr(RawWriter, "append_frame", blocked_append)
    monkeypatch.setattr(recorder_module, "RECORDER_STOP_TIMEOUT_S", 0)
    metadata, row, command = capture
    recorder = EpisodeRecorder(tmp_path, control_hz=10)
    recorder.start(task_label="fixture", collection_source="teleop", metadata=metadata)
    try:
        recorder.append(row(0), command)
        assert entered.wait(5)
        if failure == "queue":
            for index in range(1, 17):
                recorder.append(row(index), command)
        message = "finish queue timed out" if failure == "queue" else "writer did not exit"
        with pytest.raises(RecordingError, match=message) as caught:
            recorder.finish(save=True, reason="test", technical_status="normal")
        assert recorder.writer_pending
    finally:
        release.set()
        recorder.thread.join(5)
    assert not recorder.writer_pending
    assert recorder.published_path is None and recorder.staging_path.is_dir()
    with pytest.raises(RecordingError) as repeated:
        recorder.finish(save=True, reason="test", technical_status="normal")
    assert repeated.value is caught.value


def test_interrupted_writer_wait_stays_pending_without_restarting_finish(tmp_path, capture, monkeypatch):
    from test_policy import make_rollout, make_session

    from dexmani_real.deployment.session import PolicySession

    entered, release = Event(), Event()
    append = RawWriter.append_frame
    def blocked_append(writer, *args):
        entered.set()
        assert release.wait(5)
        return append(writer, *args)
    monkeypatch.setattr(RawWriter, "append_frame", blocked_append)
    metadata, row, command = capture
    recorder = EpisodeRecorder(tmp_path / "raw", control_hz=10)
    recorder.start(task_label="fixture", collection_source="policy_rollout", metadata=metadata)
    recorder.append(row(0), command)
    join = recorder.thread.join
    interrupt = KeyboardInterrupt()
    interrupted_join = Mock(side_effect=interrupt)
    monkeypatch.setattr(recorder.thread, "join", interrupted_join)
    try:
        assert entered.wait(2)
        episode, _ = make_rollout(tmp_path)
        episode.recorder = recorder
        episode.preparation_valid = lambda: True
        episode.prepare = lambda poll: episode.stop("operator", run_end_reason=RunEndReason.OPERATOR)
        with pytest.raises(KeyboardInterrupt) as caught:
            episode.run(keyboard=Mock(), poll_operator=lambda: None)
        assert caught.value is interrupt
        state = episode.runtime_state
        assert state.estop_request and state.emergency_signal.is_set() and state.stop_signal.is_set()
        assert episode.writer_pending and episode.result.raw["status"] == "pending"
        assert recorder.finish_request is not None and recorder.published_path is None
        session, events = make_session(tmp_path)
        session.runtime_state, session.episode, session.episodes = state, episode, [episode.result]
        session.save_results = PolicySession.save_results.__get__(session)
        assert session.close(failure=interrupt) == 1
        result = json.loads((tmp_path / "session_result.json").read_text())
        assert result["shutdown"] == "pending" and result["outcome"] == "fault"
        assert result["reason"] == "KeyboardInterrupt" and result["emergency_stop_requested"]
        assert result["episodes"][0]["raw"]["status"] == "pending"
        assert "model_close" in events and "audio_close" in events
        interrupted_join.assert_called_once()
    finally:
        release.set()
        join(2)
    assert not recorder.writer_pending
    with pytest.raises(RecordingError) as repeated:
        recorder.finish(save=True, reason="retry", technical_status="normal")
    assert repeated.value.__cause__ is interrupt
    interrupted_join.assert_called_once()
    assert recorder.published_path is None and recorder.staging_path.is_dir()


@pytest.mark.parametrize("overwrite", [False, True])
def test_atomic_write_preserves_publish_error_when_cleanup_fails(
    tmp_path, monkeypatch, caplog, overwrite,
):
    primary = OSError("publication failed")
    cleanup = OSError("temporary cleanup failed")
    boundary = "replace" if overwrite else "publish"
    owner = io.os if overwrite else io
    with monkeypatch.context() as patch:
        patch.setattr(owner, boundary, Mock(side_effect=primary))
        patch.setattr(io.Path, "unlink", Mock(side_effect=cleanup))
        with pytest.raises(OSError) as caught:
            io.write_json({"result": "failed"}, tmp_path / "result.json", overwrite=overwrite)
    assert caught.value is primary
    cleanup_logs = [entry for entry in caplog.records if entry.exc_info and entry.exc_info[1] is cleanup]
    assert len(cleanup_logs) == 1
    assert "Traceback" in caplog.text


@pytest.mark.parametrize("overwrite", [False, True])
def test_successful_atomic_write_consumes_temporary_without_cleanup(tmp_path, monkeypatch, overwrite):
    target = tmp_path / "result.json"
    if overwrite:
        target.write_text("previous result")
    with monkeypatch.context() as patch:
        cleanup = Mock(side_effect=AssertionError("successful publication needs no unlink"))
        patch.setattr(io.Path, "unlink", cleanup)
        assert io.write_json({"result": "saved"}, target, overwrite=overwrite) == target
        cleanup.assert_not_called()
    assert target.read_text() == '{\n  "result": "saved"\n}'
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_publication_refuses_existing_target_and_removes_temporary(tmp_path):
    target = tmp_path / "result.json"
    target.write_text("original result")
    with pytest.raises(FileExistsError):
        io.write_json({"result": "replacement"}, target, overwrite=False)
    assert target.read_text() == "original result"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("failure", ["flush", "count"])
def test_writer_failure_does_not_publish(tmp_path, capture, monkeypatch, failure):
    original_finish = RawWriter.finish

    def fail_finish(self, reason, details, *, expected_frames, technical_status):
        if failure == "flush":
            raise OSError("injected flush failure")
        return original_finish(self, reason, details, expected_frames=expected_frames + 1,
                               technical_status=technical_status)

    if failure == "flush":
        monkeypatch.setattr(RawWriter, "flush", Mock(side_effect=OSError("injected flush failure")))
    else:
        monkeypatch.setattr(RawWriter, "finish", fail_finish)
    with pytest.raises(RecordingError):
        record(tmp_path, capture)
    assert all(path.name.startswith(".tmp_") for path in tmp_path.iterdir())
    assert list(tmp_path.glob(".tmp_*/data.h5"))


def test_queue_full_keeps_staging_and_original_cause(tmp_path, capture, monkeypatch):
    entered, release = Event(), Event()
    original_append = RawWriter.append_frame

    def blocked_append(self, *args):
        entered.set()
        assert release.wait(5)
        return original_append(self, *args)

    monkeypatch.setattr(RawWriter, "append_frame", blocked_append)
    metadata, row, command = capture
    recorder = EpisodeRecorder(tmp_path, control_hz=10)
    recorder.start(task_label="fixture", collection_source="teleop", metadata=metadata)
    try:
        recorder.append(row(0), command)
        assert entered.wait(5)
        for index in range(1, 17):
            recorder.append(row(index), command)
        with pytest.raises(RecordingError, match="queue full"):
            recorder.append(row(17), command)
    finally:
        release.set()
        with pytest.raises(RecordingError, match="queue full"):
            recorder.finish(save=True, reason="test")
    assert recorder.published_path is None
    assert recorder.staging_path.is_dir()
    assert not recorder.writer_pending


@pytest.mark.parametrize("write_fails", [False, True])
def test_writer_close_failure_keeps_staging_and_original_cause(
    tmp_path, capture, monkeypatch, caplog, write_fails,
):
    primary, cleanup = OSError("injected write failure"), OSError("injected close failure")
    original_close = RawWriter.close

    def failed_close(writer):
        original_close(writer)
        raise cleanup

    monkeypatch.setattr(RawWriter, "close", failed_close)
    if write_fails:
        monkeypatch.setattr(RawWriter, "append_frame", Mock(side_effect=primary))
    metadata, row, command = capture
    recorder = EpisodeRecorder(tmp_path, control_hz=10)
    recorder.start(task_label="fixture", collection_source="teleop", metadata=metadata)
    recorder.append(row(0), command)
    with pytest.raises(RecordingError) as caught:
        recorder.finish(save=True, reason="test", technical_status="normal")
    assert caught.value.__cause__ is (primary if write_fails else cleanup)
    assert not recorder.writer_pending and recorder.published_path is None
    assert recorder.staging_path.is_dir()
    assert not recorder.writer.data_file.id.valid
    assert recorder.writer.video.container is None
    if write_fails:
        assert any(record.exc_info and record.exc_info[1] is cleanup for record in caplog.records)


@pytest.mark.parametrize('label', [None, 17, '', '   ', '\u3000', *('demo' + chr(c) for c in (*range(32), 127))])
def test_task_label_rejected_before_recording_side_effects(tmp_path, label):
    from dexmani_real.config.real import RealConfig
    from dexmani_real.data.raw import validate_task_label
    from dexmani_real.deployment.run import session_directory
    from dexmani_real.teleop.run import run_teleop

    with pytest.raises(ValueError, match='task_label'):
        validate_task_label(label)
    destination = tmp_path / 'new'
    recorder = EpisodeRecorder(destination, control_hz=10)
    with pytest.raises(RecordingError) as caught:
        recorder.start(task_label=label, collection_source='teleop', metadata={})
    assert isinstance(caught.value.__cause__, ValueError)
    assert not destination.exists() and recorder.thread is None
    info = SimpleNamespace(task_name=label, policy_name='policy', experiment_dir=tmp_path / 'experiment')
    with pytest.raises(ValueError, match='task_label'):
        session_directory(destination, info)
    assert not destination.exists()
    with pytest.raises(ValueError, match='task_label'):
        run_teleop(RealConfig(), record=True, episodes_dir=str(destination),
                   max_duration_s=1, runtime_config_json=None, task_name=label)
    assert not destination.exists()


@pytest.mark.parametrize('label', ['抓取 toy', ' pick/place ', '.natural language', 'a\\b'])
def test_natural_language_labels_round_trip_without_directory_rules(tmp_path, capture, label):
    from dexmani_real.config.real import RealConfig
    from dexmani_real.teleop.run import run_teleop

    metadata, row, command = capture
    recorder = EpisodeRecorder(tmp_path / 'raw', control_hz=10)
    recorder.start(task_label=label, collection_source='teleop', metadata=metadata)
    for index in range(3):
        recorder.append(row(index), command)
    path = recorder.finish(save=True, reason='test', technical_status='normal')
    output = tmp_path / 'export.zarr'
    export_zarr(path, output, processing=ExportConfig(pointcloud=PointCloudConfig(
        remove_table=False, num_points=32, workspace=(-1., -1., 0., 1., 1., 1.),
    )))
    assert zarr.open_group(str(output), mode='r').attrs['task_name'] == label
    if label != '抓取 toy':
        with pytest.raises(ValueError, match='task_name'):
            run_teleop(RealConfig(), record=True, episodes_dir=str(tmp_path),
                       max_duration_s=1, runtime_config_json=None, task_name=label)


@pytest.mark.parametrize('label', [17, '   ', 'demo\x01', 'demo\x1f', 'demo\x7f'])
def test_export_rejects_invalid_stored_task_label(tmp_path, capture, label):
    _, path = record(tmp_path / 'raw', capture)
    # 模拟历史 Raw 的外部输入；只修改测试临时产物。
    with h5py.File(path / 'data.h5', 'r+') as data:
        data['meta'].attrs['task_label'] = label
    with RawReader(path) as reader, pytest.raises(ValueError, match='task_label'):
        validate_export_episode(reader)
    output = tmp_path / 'invalid.zarr'
    with pytest.raises(ValueError, match='task_label'):
        export_zarr(path, output, processing=ExportConfig())
    assert not output.exists()


def test_model_history_build_preserves_recording_input(tmp_path, capture):
    from dexmani_real.config.real import RealConfig
    from dexmani_real.deployment.observation import ObservationBuilder

    metadata, make_row, command = capture
    row = make_row(0)
    fields = ('rgb', 'depth', 'finger_tactile_force', 'taxel_tactile_force', 'joint_state')
    builder = ObservationBuilder(SimpleNamespace(observation_fields=fields), RealConfig())
    history = builder.project(row)
    recorder = EpisodeRecorder(tmp_path, control_hz=10)
    recorder.start(task_label='model and recording', collection_source='teleop', metadata=metadata)
    recorder.append(row, command)
    for value in builder.build((history,)).values():
        value.fill(0)
    path = recorder.finish(save=True, reason='test', technical_status='normal')
    np.testing.assert_array_equal(row.camera['rgb'], 80)
    with RawReader(path) as reader:
        for recorded, expected in (
            ('depth', row.camera['depth']),
            ('hand_contact', row.hand['tactile_aggregate'][0]),
            ('hand_tactile_force', row.hand['tactile_dense'][0]),
            ('arm_qpos', row.arm['qpos'][0]),
        ):
            np.testing.assert_array_equal(reader[recorded][0], expected)
        assert reader['timing/rows'][0]['camera_ns'] == row.camera['timestamp_ns']
