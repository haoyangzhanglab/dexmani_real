"""离线手部预览只消费所选 Raw 前缀，按记录时钟推进独立参考。"""

import json
from dataclasses import asdict, replace

import h5py
import numpy as np
import pytest

from dexmani_real.config.teleop import HandRetargetingConfig
from dexmani_real.config.xhand import HandConfig
from dexmani_real.data.raw import TIMING_ROW_DTYPE
from dexmani_real.geometry.kinematics import HandFK
from dexmani_real.robot.model import XHAND_SDK_JOINT_NAMES
from dexmani_real.teleop.retargeting.landmarks import hand_lengths, prepare_landmarks
from scripts import preview_hand_retargeting as preview


@pytest.fixture
def episode(tmp_path):
    path = tmp_path / "episode_001"
    path.mkdir()
    landmarks = np.zeros((21, 3))
    distance = np.array((0, .025, .045, .063))
    for finger, (side, forward) in enumerate((
        (.04, .02), (.025, .055), (0, .065), (-.025, .055), (-.045, .045),
    )):
        landmarks[1 + 4 * finger:5 + 4 * finger, 1] = side
        landmarks[1 + 4 * finger:5 + 4 * finger, 2] = forward + distance
    landmarks[1:5, 1] += distance * .7
    landmarks[1:5, 0] = distance * .3
    lengths = hand_lengths(prepare_landmarks(landmarks))
    base_hand = HandConfig()
    hand = replace(
        base_hand, qpos_min_rad=tuple(np.asarray(base_hand.qpos_min_rad) + .01),
        qpos_max_rad=tuple(np.asarray(base_hand.qpos_max_rad) - .01),
    )
    config = replace(HandRetargetingConfig(), smoothing_tau_s=.123)
    parameters = {
        "hand_retargeting": asdict(config),
        "hand": {"qpos_min_rad": hand.qpos_min_rad, "qpos_max_rad": hand.qpos_max_rad},
        "hand_retargeting_calibration": {
            "bone_lengths": lengths[:15].reshape(5, 3).tolist(), "palm_width": lengths[15],
        },
    }
    times = np.array([
        1_000_000_000, 1_062_500_008, 1_110_000_000, 2_000_000_000, 2_060_000_000, 2_100_000_000,
    ])
    rows = np.zeros(6, dtype=TIMING_ROW_DTYPE)
    rows["observation_ns"] = times
    rows["control_ns"] = times - 1_000_000
    qpos = np.tile(np.mean([hand.qpos_min_rad, hand.qpos_max_rad], axis=0), (6, 1))
    qpos[3] += .02
    with h5py.File(path / "data.h5", "w") as data:
        data.create_group("meta").attrs.update(
            effective_parameters=json.dumps(parameters), joint_units="rad",
            hand_joint_names=json.dumps(XHAND_SDK_JOINT_NAMES),
        )
        data.create_dataset("timing/rows", data=rows)
        data.create_dataset("timing/segment_starts", data=np.array([0, 3], dtype=np.int64))
        data.create_dataset("hand_qpos", data=qpos)
        data.create_dataset("vr_landmarks", data=np.tile(landmarks, (6, 1, 1)), dtype=np.float32)
    return path


def edit_parameters(episode, update):
    with h5py.File(episode / "data.h5", "r+") as data:
        parameters = json.loads(data["meta"].attrs["effective_parameters"])
        update(parameters)
        data["meta"].attrs["effective_parameters"] = json.dumps(parameters)


@pytest.fixture
def fake_solver(monkeypatch):
    events = []
    instances = []

    class Retargeter:
        def __init__(self, config, hand):
            self.config, self.hand = config, hand
            self.reference_qpos = self.reference_timestamp_ns = None
            self.rejection_reason = None
            instances.append(self)

        def reset(self, qpos, *, timestamp_ns):
            events.append(("reset", timestamp_ns, qpos.copy()))
            self.reference_qpos, self.reference_timestamp_ns = qpos.copy(), timestamp_ns

        def retarget(self, landmarks, *, timestamp_ns):
            events.append((
                "retarget", timestamp_ns, self.reference_timestamp_ns, self.reference_qpos.copy(),
            ))
            if timestamp_ns == 1_110_000_000:
                self.rejection_reason = "synthetic geometry rejection"
                return None
            self.rejection_reason = None
            return self.reference_qpos + .005

        def commit(self, qpos, *, timestamp_ns):
            events.append(("commit", timestamp_ns, qpos.copy()))
            self.reference_qpos, self.reference_timestamp_ns = qpos.copy(), timestamp_ns

    monkeypatch.setattr(preview, "HandRetargeter", Retargeter)
    return events, instances


def test_recorded_parameters_restore_without_unrelated_assets(episode, fake_solver, monkeypatch):
    monkeypatch.chdir(episode.parent)
    _, instances = fake_solver
    result = preview.preview_episode(episode.name, view=False)
    retargeter = instances[0]
    with h5py.File(episode / "data.h5", "r") as data:
        recorded = json.loads(data["meta"].attrs["effective_parameters"])
    assert retargeter.config.smoothing_tau_s == .123
    np.testing.assert_array_equal(retargeter.hand.qpos_min_rad, recorded["hand"]["qpos_min_rad"])
    np.testing.assert_array_equal(retargeter.hand.qpos_max_rad, recorded["hand"]["qpos_max_rad"])
    calibration = recorded["hand_retargeting_calibration"]
    np.testing.assert_array_equal(retargeter.calibration.bone_lengths, calibration["bone_lengths"])
    assert retargeter.calibration.palm_width == calibration["palm_width"]
    assert result["measured_qpos"].shape == (6, 12)
    assert set(episode.iterdir()) == {episode / "data.h5"}


def test_config_replaces_only_retargeting(episode, fake_solver, tmp_path):
    config_path = tmp_path / "hand.py"
    config_path.write_text(
        "from dexmani_real.config.real import RealConfig\n"
        "from dexmani_real.config.teleop import HandRetargetingConfig\n"
        "from dexmani_real.config.xhand import HandConfig\n"
        "CONFIG = RealConfig(hand_retargeting=HandRetargetingConfig(shape_weight=3.0), "
        "hand=HandConfig(qpos_min_rad=(999.,)*12))\n",
        encoding="utf-8",
    )
    # 被显式替换的记录参数不再消费；标定与限位仍来自 Raw。
    edit_parameters(episode, lambda p: p.pop("hand_retargeting"))
    with h5py.File(episode / "data.h5", "r") as data:
        recorded = json.loads(data["meta"].attrs["effective_parameters"])
    preview.preview_episode(episode, config_path=config_path, view=False)
    retargeter = fake_solver[1][0]
    assert retargeter.config.shape_weight == 3.0
    np.testing.assert_array_equal(retargeter.hand.qpos_min_rad, recorded["hand"]["qpos_min_rad"])
    np.testing.assert_array_equal(retargeter.hand.qpos_max_rad, recorded["hand"]["qpos_max_rad"])
    calibration = recorded["hand_retargeting_calibration"]
    np.testing.assert_array_equal(retargeter.calibration.bone_lengths, calibration["bone_lengths"])
    assert retargeter.calibration.palm_width == calibration["palm_width"]


@pytest.mark.parametrize("missing", ["hand_retargeting_calibration", "hand", "hand_retargeting"])
def test_missing_required_recorded_fields_fail(episode, missing):
    edit_parameters(episode, lambda p: p.pop(missing))
    with pytest.raises(KeyError, match=missing):
        preview.preview_episode(episode, view=False)


def test_recorded_config_does_not_fill_current_defaults(episode):
    edit_parameters(episode, lambda p: p["hand_retargeting"].pop("shape_weight"))
    with pytest.raises(ValueError, match="missing parameters: shape_weight"):
        preview.preview_episode(episode, view=False)


@pytest.mark.parametrize("field,value", [
    ("bone_lengths", [[.01] * 3] * 4),
    ("bone_lengths", [[0] * 3] * 5),
    ("palm_width", float("nan")),
])
def test_invalid_recorded_calibration_fails(episode, field, value):
    edit_parameters(episode, lambda p: p["hand_retargeting_calibration"].update({field: value}))
    with pytest.raises(ValueError, match=field):
        preview.preview_episode(episode, view=False)


def test_recorded_observation_clock_and_segment_resets(episode, fake_solver):
    events, _ = fake_solver
    result = preview.preview_episode(episode, view=False)
    assert [(e[0], e[1]) for e in events] == [
        ("reset", 1_000_000_000), ("retarget", 1_062_500_008), ("commit", 1_062_500_008),
        ("retarget", 1_110_000_000), ("reset", 2_000_000_000),
        ("retarget", 2_060_000_000), ("commit", 2_060_000_000),
        ("retarget", 2_100_000_000), ("commit", 2_100_000_000),
    ]
    assert events[5][2] == 2_000_000_000
    np.testing.assert_array_equal(events[5][3], result["measured_qpos"][3])
    assert np.isnan(result["candidate_qpos"][[0, 2, 3]]).all()
    assert result["rejection_reasons"][2] == "synthetic geometry rejection"


def test_rejection_preserves_reference_position_and_time(episode, fake_solver):
    with h5py.File(episode / "data.h5", "r+") as data:
        del data["timing/segment_starts"]
        data.create_dataset("timing/segment_starts", data=np.array([0], dtype=np.int64))
    events, _ = fake_solver
    result = preview.preview_episode(episode, max_frames=4, view=False)
    after_rejection = [event for event in events if event[0] == "retarget"][-1]
    assert after_rejection[2] == 1_062_500_008
    np.testing.assert_array_equal(after_rejection[3], result["reference_qpos"][1])
    np.testing.assert_array_equal(result["reference_qpos"][2], result["reference_qpos"][1])
    assert np.isnan(result["candidate_qpos"][2]).all()


def test_max_frames_reads_only_selected_prefix(episode, fake_solver, monkeypatch):
    with h5py.File(episode / "data.h5", "r+") as data:
        data["hand_qpos"][3:] = np.nan
        data["vr_landmarks"][3:] = np.nan
        rows = data["timing/rows"][:]
        rows["observation_ns"][3:] = 0
        data["timing/rows"][:] = rows
    reads = []
    original_getitem = h5py.Dataset.__getitem__

    def record_read(dataset, key):
        reads.append((dataset.name, key))
        return original_getitem(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", record_read)
    result = preview.preview_episode(episode, max_frames=3, view=False)
    assert len(result["observation_ns"]) == 3
    for name in ("/hand_qpos", "/timing/rows"):
        assert [(path, key) for path, key in reads if path == name] == [(name, slice(None, 3))]
    assert [(path, key) for path, key in reads if path == "/vr_landmarks"] == [
        ("/vr_landmarks", slice(0, 3)),
    ]
    assert {path for path, _ in reads} == {
        "/hand_qpos", "/vr_landmarks", "/timing/rows", "/timing/segment_starts",
    }


@pytest.mark.parametrize("block_rows", [2, 4])
def test_landmark_blocks_preserve_segments_and_rejected_references(
    episode, fake_solver, monkeypatch, block_rows,
):
    expected = preview.preview_episode(episode, view=False)
    monkeypatch.setattr(preview, "LANDMARK_BLOCK_ROWS", block_rows)
    reads = []
    getitem = h5py.Dataset.__getitem__

    def read(dataset, key):
        if dataset.name == "/vr_landmarks":
            reads.append(key)
        return getitem(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", read)
    actual = preview.preview_episode(episode, view=False)
    assert all(0 < key.stop - key.start <= block_rows for key in reads)
    assert [row for key in reads for row in range(key.start, key.stop)] == list(range(6))
    for name in ("observation_ns", "measured_qpos", "candidate_qpos", "reference_qpos"):
        np.testing.assert_array_equal(actual[name], expected[name])
    assert actual["segment_starts"] == expected["segment_starts"] == {0, 3}
    assert actual["rejection_reasons"] == expected["rejection_reasons"]
    assert len(actual["solve_times_s"]) == len(expected["solve_times_s"]) == 4


def test_no_view_cli_does_not_initialize_viewer(episode, fake_solver, monkeypatch, capsys):
    def fail(*args, **kwargs):
        pytest.fail("no-view initialized viewer")

    monkeypatch.setattr(preview, "show_preview", fail)
    assert preview.main([str(episode), "--max-frames", "3", "--no-view"]) == 0
    output = capsys.readouterr().out
    assert "Frames: 3; rejected: 1" in output
    assert "Rejected row 2: synthetic geometry rejection" in output
    assert "2 calls" in output


def test_reference_only_prefix_has_no_solve_statistics(episode, fake_solver, capsys):
    assert preview.main([str(episode), "--max-frames", "1", "--no-view"]) == 0
    assert "Solve time: no samples" in capsys.readouterr().out
    assert len(fake_solver[0]) == 1


def test_invalid_selected_input_and_solver_errors_propagate(episode, fake_solver, monkeypatch):
    with h5py.File(episode / "data.h5", "r+") as data:
        data["vr_landmarks"][1] = np.nan
    with pytest.raises(ValueError, match="vr_landmarks.*row 1"):
        preview.preview_episode(episode, view=False)
    # 未使用的段首 landmarks 不参与求解，不扩大字段内容检查。
    with h5py.File(episode / "data.h5", "r+") as data:
        data["vr_landmarks"][0] = np.nan
        data["vr_landmarks"][1] = 0

    def fail(*args, **kwargs):
        raise RuntimeError("solver programming error")

    monkeypatch.setattr(preview.HandRetargeter, "retarget", fail)
    with pytest.raises(RuntimeError, match="solver programming error"):
        preview.preview_episode(episode, view=False)


def test_rerun_logging_with_viewer_spawn_disabled(episode, fake_solver, monkeypatch, tmp_path):
    import rerun as rr
    from rerun.dataframe import load_recording

    result = preview.preview_episode(episode, max_frames=3, view=False)
    original_log = rr.log
    logs = []

    def capture_log(path, *args, **kwargs):
        logs.append((path, args))
        original_log(path, *args, **kwargs)

    monkeypatch.setattr(rr, "init", lambda *args, **kwargs: None)
    monkeypatch.setattr(rr, "log", capture_log)
    path = tmp_path / "preview.rrd"
    with rr.new_recording("hand-preview-test", spawn=False) as recording:
        recording.save(path)
        preview.show_preview(result, HandFK(), episode.name)
        recording.disconnect()
    saved = load_recording(path)
    for entity in ("/qpos/measured/0", "/hand/measured/fingertips"):
        axes = saved.view(index="step", contents=entity).select("step", "time").read_all()
        assert axes["step"].to_pylist() == [0, 1, 2]
        assert axes["time"].cast("int64").to_pylist() == [0, 62_500_008, 110_000_000]
    # 候选与保留参考互斥显示；同时运行真实 blueprint、archetypes 和列发送。
    candidates = [args[0] for path, args in logs if path == "hand/candidate/fingertips"]
    assert [type(value) for value in candidates] == [rr.Clear, rr.Points3D, rr.Clear]
    references = [args[0] for path, args in logs if path == "hand/reference/fingertips"]
    assert [type(value) for value in references] == [rr.Points3D, rr.Clear, rr.Points3D]
    assert any(path == "events" and isinstance(args[0], rr.TextLog) for path, args in logs)


def test_real_solver_on_small_recorded_sample(episode):
    result = preview.preview_episode(episode, max_frames=3, view=False)
    assert result["rejection_reasons"] == [None, None, None]
    candidates = result["candidate_qpos"][1:]
    assert np.isfinite(candidates).all()
    with h5py.File(episode / "data.h5", "r") as data:
        hand = json.loads(data["meta"].attrs["effective_parameters"])["hand"]
    assert np.all(candidates >= hand["qpos_min_rad"])
    assert np.all(candidates <= hand["qpos_max_rad"])
    times = result["observation_ns"]
    speeds = np.abs(np.diff(result["reference_qpos"], axis=0)) / (np.diff(times)[:, None] * 1e-9)
    assert np.all(speeds <= HandRetargetingConfig().max_joint_speed_rad_s + 1e-9)
