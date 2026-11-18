"""Raw 固定字段导出、原生分段与跨仓数据消费。"""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import h5py
import numpy as np
import pytest
import zarr

from dexmani_real.data.export import ExportConfig, export_zarr
from dexmani_real.data.raw import (
    ARM_ACTION_SEMANTICS,
    TIMING_ROW_DTYPE,
    RawDataError,
    RawReader,
)
from dexmani_real.geometry.kinematics import ArmFK, eef_poses
from dexmani_real.geometry.pose import encode_eef_pose
from dexmani_real.robot.model import XARM7_JOINT_NAMES, XHAND_SDK_JOINT_NAMES


@pytest.fixture
def raw_episodes(tmp_path):
    source = tmp_path / "unrelated_directory"
    states, actions = [], []
    for index, (frames, starts) in enumerate(((4, [0]), (3, [0]))):
        episode = source / f"episode_{index}"
        episode.mkdir(parents=True)
        state = np.arange(frames * 19).reshape(frames, 19) / 100 + index
        action = state + 0.1
        states.append(state)
        actions.append(action)
        with h5py.File(episode / "data.h5", "w") as data:
            data.create_group("meta").attrs.update(
                task_label="recorded_task",
                control_hz=20.0,
                joint_units="rad",
                arm_action_semantics=ARM_ACTION_SEMANTICS,
                arm_joint_names=json.dumps(XARM7_JOINT_NAMES),
                hand_joint_names=json.dumps(XHAND_SDK_JOINT_NAMES),
            )
            rows = np.zeros(frames, dtype=TIMING_ROW_DTYPE)
            for name in rows.dtype.names:
                rows[name] = 1_000_000_000 + np.arange(frames) * 50_000_000
            rows["camera_sequence"] = np.arange(1, frames + 1)
            data.create_dataset("timing/rows", data=rows)
            data.create_dataset(
                "timing/segment_starts", data=np.array(starts, np.int64)
            )
            for name, values in (
                ("arm_qpos", state[:, :7]),
                ("hand_qpos", state[:, 7:]),
                ("action_arm_joint_target", action[:, :7]),
                ("action_hand_joint_target", action[:, 7:]),
            ):
                data.create_dataset(name, data=values)
    return source, np.concatenate(states), np.concatenate(actions)


def test_export_accepts_float32_arm_qpos(tmp_path, raw_episodes):
    source, states, _ = raw_episodes
    with h5py.File(source / "episode_0/data.h5", "r+") as data:
        joints = data["arm_qpos"][:].astype(np.float32)
        del data["arm_qpos"]
        data["arm_qpos"] = joints

    result = export_zarr(
        source, tmp_path / "joints.zarr", processing=ExportConfig(), fields=("joint_state",),
    )
    root = zarr.open_group(result["path"], mode="r")
    assert root.attrs["source_episodes"] == ["episode_0", "episode_1"]
    assert result["total_frames"] == 7
    np.testing.assert_array_equal(root["meta/episode_ends"][:], [4, 7])
    np.testing.assert_array_equal(root["data/joint_state"][:], states.astype(np.float32))
    root.store.close()


def test_raw_reader_observation_prefix(raw_episodes, monkeypatch):
    stop = 3
    source, *_ = raw_episodes
    episode = source / "episode_0"
    with h5py.File(episode / "data.h5", "r+") as data:
        rows = data["timing/rows"][:]
        rows["observation_ns"] = 2**60 + np.array([0, 1, 3, 7], dtype=np.int64)
        data["timing/rows"][:] = rows
    original = h5py.Dataset.__getitem__
    reads = []

    def read(dataset, key, **kwargs):
        if dataset.name == "/timing/rows":
            assert key == slice(None, stop)
            assert kwargs["new_dtype"].names == ("observation_ns",)
            reads.append(key)
        return original(dataset, key, **kwargs)

    with RawReader(episode) as reader:
        np.testing.assert_array_equal(reader.read_timing(stop=stop), rows[:stop])
        monkeypatch.setattr(h5py.Dataset, "__getitem__", read)
        timestamps = reader.read_observation_stamps(stop=stop)
    assert reads == [slice(None, stop)]
    assert timestamps.ndim == 1 and timestamps.dtype == np.int64
    np.testing.assert_array_equal(timestamps, rows["observation_ns"][:stop])


@pytest.mark.parametrize("method", ["read_timing", "read_observation_stamps"])
@pytest.mark.parametrize("stop", [None, 3])
@pytest.mark.parametrize("offsets,bad_row", [
    ([0, 10, 10, 30], 2),
    ([0, 10, 9, 30], 2),
    ([0, 0, 0, 0], 1),
    ([30, 20, 10, 0], 1),
])
def test_raw_reader_rejects_invalid_observation_clock(raw_episodes, method, stop, offsets, bad_row):
    source, *_ = raw_episodes
    episode = source / "episode_0"
    with h5py.File(episode / "data.h5", "r+") as data:
        rows = data["timing/rows"][:]
        # 保持 observation 晚于 control，单独触发 observation 单调性检查。
        rows["observation_ns"] = 2_000_000_000 + np.array(offsets, dtype=np.int64)
        data["timing/rows"][:] = rows
    with RawReader(episode) as reader, pytest.raises(ValueError) as error:
        getattr(reader, method)(stop=stop)
    assert str(error.value) == (
        f"Raw {episode.resolve()}: observation_clock_not_increasing at row {bad_row}"
    )


@pytest.mark.parametrize("stamps,stop", [
    ([100, 110, 120, 120], 3),
    ([100, 110, 120, 119], 3),
    ([2**63 - 2, 2**63 - 1, -(2**63), -(2**63) + 1], 2),
])
def test_raw_reader_observation_clock_checks_only_selected_prefix(raw_episodes, stamps, stop):
    source, *_ = raw_episodes
    episode = source / "episode_0"
    with h5py.File(episode / "data.h5", "r+") as data:
        rows = data["timing/rows"][:]
        rows["observation_ns"] = stamps
        data["timing/rows"][:] = rows
    with RawReader(episode) as reader:
        np.testing.assert_array_equal(reader.read_observation_stamps(stop=stop), stamps[:stop])
        with pytest.raises(ValueError, match=f"observation_clock_not_increasing at row {stop}"):
            reader.read_observation_stamps()


@pytest.mark.parametrize("control,observation,detail,bad_row", [
    ([0, -1, 2, 3], [4, 4, 4, 4], "invalid_control_or_observation_clock", 1),
    ([0, 1, 2, 3], [4, 4, 4, 2], "invalid_control_or_observation_clock", 3),
    ([0, 1, 1, 3], [4, 4, 4, 4], "control_clock_not_increasing", 2),
    ([0, 2, 1, 3], [4, 4, 4, 4], "control_clock_not_increasing", 2),
])
def test_raw_reader_timing_validation_order(raw_episodes, control, observation, detail, bad_row):
    source, *_ = raw_episodes
    episode = source / "episode_0"
    with h5py.File(episode / "data.h5", "r+") as data:
        rows = data["timing/rows"][:]
        rows["control_ns"], rows["observation_ns"] = control, observation
        data["timing/rows"][:] = rows
    with RawReader(episode) as reader, pytest.raises(ValueError) as error:
        reader.read_timing()
    assert str(error.value) == f"Raw {episode.resolve()}: {detail} at row {bad_row}"


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [((4, 2, 3), np.uint16), ((4, 2, 2), np.float32)],
)
def test_raw_reader_rejects_invalid_depth_grid(raw_episodes, shape, dtype):
    source, _, _ = raw_episodes
    episode = source / "episode_0"
    with h5py.File(episode / "data.h5", "r+") as data:
        data["meta"].attrs.update(camera_color_width=2, camera_color_height=2)
        data.create_dataset("depth", data=np.ones(shape, dtype=dtype))
    with RawReader(episode) as reader, pytest.raises(RawDataError, match="depth: expected shape.*dtype uint16"):
        reader["depth"]


@pytest.mark.parametrize("width", [2, 3])
def test_raw_reader_decodes_rgb_on_recorded_grid(raw_episodes, width):
    source, _, _ = raw_episodes
    episode = source / "episode_0"
    (episode / "rgb.mp4").touch()
    with h5py.File(episode / "data.h5", "r+") as data:
        data["meta"].attrs.update(camera_color_width=2, camera_color_height=2)
    image = np.full((2, width, 3), 128, np.uint8)

    def decode_frame(*, format):
        assert format == "rgb24"
        return image

    with RawReader(episode) as reader:
        reader.video_container = SimpleNamespace(
            seek=lambda offset: None,
            decode=lambda **kwargs: [SimpleNamespace(to_ndarray=decode_frame)],
            close=lambda: None,
        )
        if width == 2:
            np.testing.assert_array_equal(list(reader.iter_rgb_frames()), [image])
        else:
            with pytest.raises(ValueError, match="RGB grid invalid at row 0"):
                list(reader.iter_rgb_frames())


@pytest.mark.parametrize("action_key", ["action", "action_ee", None])
def test_export_metadata_and_arrays_survive_renaming(
    tmp_path, raw_episodes, action_key
):
    source, states, actions = raw_episodes
    fields = ("joint_state", action_key) if action_key else ("joint_state",)
    expected = {"joint_state": states.astype(np.float32)}
    if action_key == "action":
        expected[action_key] = actions.astype(np.float32)
    elif action_key == "action_ee":
        poses = eef_poses(actions[:, :7], arm_fk=ArmFK())
        expected[action_key] = np.concatenate(
            (encode_eef_pose(poses), actions[:, 7:]),
            axis=1,
            dtype=np.float32,
        )
    for input_name, output_name in (
        ("original_input", "original_output.zarr"),
        ("renamed_input", "renamed_output.zarr"),
    ):
        source = source.rename(tmp_path / input_name)
        output = tmp_path / output_name
        result = export_zarr(source, output, processing=ExportConfig(), fields=fields)
        root = zarr.open_group(str(output), mode="r")
        assert root.attrs["task_name"] == "recorded_task"
        assert root.attrs["action_space"] == "joint"
        assert root.attrs["dt"] == 0.05
        assert result["episode_count"] == 2
        assert result["total_frames"] == 7
        assert set(root["data"]) == set(fields)
        np.testing.assert_array_equal(root["meta/episode_ends"][:], [4, 7])
        for key, values in expected.items():
            array = root["data"][key]
            assert array.dtype == np.float32
            np.testing.assert_array_equal(array[:], values)
        root.store.close()


@pytest.mark.parametrize(
    ("episode_name", "task_label"),
    [
        ("episode_1", "another_task"),
        ("episode_0", None),
        ("episode_1", None),
    ],
)
def test_invalid_tasks_skip_only_bad_raw(
    tmp_path, raw_episodes, episode_name, task_label
):
    source, _, _ = raw_episodes
    with h5py.File(source / episode_name / "data.h5", "r+") as data:
        if task_label is None:
            del data["meta"].attrs["task_label"]
        else:
            data["meta"].attrs["task_label"] = task_label
    output = tmp_path / "invalid.zarr"
    result = export_zarr(source, output, processing=ExportConfig(), fields=("joint_state",))
    assert result["episode_count"] == 1
    assert result["rejected_episodes"][0]["episode"] == episode_name
    assert "task_label" in result["rejected_episodes"][0]["reason"]
    assert not list(tmp_path.glob(".invalid.zarr.tmp-*"))


@pytest.fixture
def hoi_episodes(raw_episodes):
    from dexmani_real.data.hoi import write_hoi

    source, states, actions = raw_episodes
    template = {
        "points_local": np.zeros((1024, 3), np.float32),
        "offsets": np.array([0, 1024]),
        "object_names": np.array(["object"]),
    }
    for episode in sorted(source.iterdir()):
        with h5py.File(episode / "data.h5", "r+") as f:
            f["meta"].attrs["camera_T_xarm_base_from_color"] = np.eye(4).ravel()
            count = len(f["timing/rows"])
        write_hoi(
            episode / "hoi.h5",
            template,
            np.tile(np.eye(4), (count, 1, 1, 1)),
            np.eye(4),
            quality=[[{"iou": 1.0}]] * count,
            tracking_config={},
            object_valid=np.ones((count, 1), bool),
        )
    return source, states, actions


@pytest.mark.parametrize("with_object", [False, True])
def test_hoi_unalignable_boundaries_reject_whole_raw(tmp_path, hoi_episodes, with_object):
    source, _, _ = hoi_episodes
    for episode in source.iterdir():
        with h5py.File(episode / "data.h5", "r+") as f:
            rows = f["timing/rows"][:]
            rows["camera_ns"] -= 10_000_000
            f["timing/rows"][:] = rows
    output = tmp_path / "aligned.zarr"
    fields = ("joint_state", "object_pointcloud") if with_object else ("joint_state",)
    if with_object:
        with pytest.raises(ValueError, match="all Raw episodes rejected"):
            export_zarr(source, output, processing=ExportConfig(), fields=fields)
        assert not output.exists()
        output.mkdir()
    else:
        result = export_zarr(source, output, processing=ExportConfig(), fields=fields)
        assert result["source_count"] == 2
    assert not list(tmp_path.glob(".aligned.zarr.tmp-*"))
    sentinel = output / "keep.txt"
    sentinel.write_text("existing output")
    with pytest.raises(FileExistsError):
        export_zarr(source, output, processing=ExportConfig(), fields=fields)
    assert sentinel.read_text() == "existing output"


@pytest.mark.parametrize("with_hoi", [False, True])
@pytest.mark.parametrize("invalid", ["missing", "shape", "semantics"])
def test_export_skips_invalid_actions(tmp_path, hoi_episodes, with_hoi, invalid):
    source, _, _ = hoi_episodes
    # 第二条损坏 Raw 不应影响此前写入的完整 episode。
    with h5py.File(source / "episode_1" / "data.h5", "r+") as data:
        if invalid == "semantics":
            data["meta"].attrs["arm_action_semantics"] = "delta_joint"
        else:
            del data["action_arm_joint_target"]
            if invalid == "shape":
                data.create_dataset("action_arm_joint_target", data=np.zeros((3, 6)))
    output = tmp_path / "invalid_actions.zarr"
    fields = ("action", "object_pointcloud") if with_hoi else ("action",)
    message = "arm_action_semantics" if invalid == "semantics" else "action_arm_joint_target"
    result = export_zarr(source, output, processing=ExportConfig(), fields=fields)
    assert result["episode_count"] == 1
    assert message in result["rejected_episodes"][0]["reason"]
    assert not list(tmp_path.glob(".invalid_actions.zarr.tmp-*"))


@pytest.mark.parametrize("with_object", [False, True])
def test_hoi_eef_export_does_not_require_hand_history(tmp_path, hoi_episodes, with_object):
    source, states, _ = hoi_episodes
    for episode in source.iterdir():
        with h5py.File(episode / "data.h5", "r+") as data:
            del data["hand_qpos"]
            del data["meta"].attrs["hand_joint_names"]
            rows = data["timing/rows"][:]
            rows["hand_ns"] = 0
            data["timing/rows"][:] = rows
    output = tmp_path / "arm_only.zarr"
    fields = ("eef_pose", "object_pointcloud") if with_object else ("eef_pose",)
    export_zarr(source, output, processing=ExportConfig(), fields=fields)
    root = zarr.open_group(str(output), mode="r")
    expected = encode_eef_pose(eef_poses(states[:, :7], arm_fk=ArmFK()))
    np.testing.assert_allclose(root["data/eef_pose"][:], expected, atol=1e-6)
    root.store.close()


@pytest.mark.parametrize("key", ["points_local", "offsets", "object_names"])
def test_hoi_export_allows_legal_templates_and_records_shared_geometry(tmp_path, hoi_episodes, key):
    from dexmani_real.data.hoi import write_hoi

    source, _, _ = hoi_episodes
    template = {
        "points_local": np.zeros((1024, 3), np.float32),
        "offsets": np.array([0, 1024]),
        "object_names": np.array(["object"]),
    }
    if key == "object_names":
        template[key] = np.array(["renamed_object"])
    elif key == "offsets":
        template[key] = np.array([0, 512, 1024])
        template["object_names"] = np.array(["object", "another_object"])
    else:
        template[key][0, 0] = 1
    objects = len(template["object_names"])
    write_hoi(source / "episode_1/hoi.h5", template, np.tile(np.eye(4), (3, objects, 1, 1)),
              np.eye(4), quality=[[{}] * objects] * 3, object_valid=np.ones((3, objects), bool),
              tracking_config={}, overwrite=True)
    output = tmp_path / "different_templates.zarr"
    result = export_zarr(source, output, processing=ExportConfig(), fields=("object_pointcloud",))
    assert result["episode_count"] == result["source_count"] == 2
    assert not result["rejected_episodes"]
    root = zarr.open_group(str(output), mode="r")
    digest = root.attrs["data_processing"]["object_pointcloud"]["geometry_sha256"]
    assert (digest is not None) == (key == "object_names")
    assert not list(tmp_path.glob(".different_templates.zarr.tmp-*"))


def test_hoi_diagnostics_preserve_internal_gap(raw_episodes):
    from dexmani_real.data.hoi import align_hoi_episode
    from dexmani_real.data.raw import RawReader

    source, _, _ = raw_episodes
    episode = source / "episode_1"
    with h5py.File(episode / "data.h5", "r+") as f:
        rows = f["timing/rows"][:]
        rows["camera_ns"][1] -= 10_000_000
        f["timing/rows"][:] = rows
    with RawReader(episode) as reader:
        boundaries = np.r_[reader.segment_starts, reader.num_frames]
        aligned = align_hoi_episode(reader, reader.read_timing(), boundaries, 0.01)
        for values in aligned.values():
            assert len(values) == reader.num_frames
            assert np.isnan(values[1]).all()
            assert np.isfinite(values[[0, 2]]).all()


@pytest.mark.parametrize(
    "fields", [("object_pointcloud",), ("object_pointcloud", "joint_state", "action")]
)
def test_hoi_loss_rejects_whole_raw(tmp_path, hoi_episodes, fields):
    source, _, _ = hoi_episodes
    with h5py.File(source / "episode_1/hoi.h5", "r+") as data:
        data["object_valid"][1, 0] = False
        data["object_pointcloud"][1] = np.nan
    output = tmp_path / "filtered.zarr"
    export_zarr(source, output, processing=ExportConfig(), fields=fields)
    root = zarr.open_group(str(output), mode="r")
    np.testing.assert_array_equal(root["meta/episode_ends"][:], [4])
    assert np.isfinite(root["data/object_pointcloud"][:]).all()
    assert root.attrs["source_episodes"] == ["episode_0"]
    root.store.close()


def test_export_rejects_missing_hoi_validity(tmp_path, hoi_episodes):
    source, _, _ = hoi_episodes
    with h5py.File(source / "episode_0/hoi.h5", "r+") as data:
        del data["object_valid"]
    output = tmp_path / "invalid.zarr"
    result = export_zarr(source, output, processing=ExportConfig(), fields=("object_pointcloud",))
    assert result["episode_count"] == 1
    assert "object_valid" in result["rejected_episodes"][0]["reason"]


def test_export_rejects_episode_with_no_valid_hoi_rows(tmp_path, hoi_episodes):
    source, _, _ = hoi_episodes
    with h5py.File(source / "episode_0/hoi.h5", "r+") as data:
        data["object_valid"][:] = False
    output = tmp_path / "empty.zarr"
    result = export_zarr(source, output, processing=ExportConfig(), fields=("object_pointcloud",))
    assert result["episode_count"] == 1
    assert "invalid rows" in result["rejected_episodes"][0]["reason"]


@pytest.mark.parametrize("rgb_only", [False, True])
def test_pinhole_export_matches_online_and_keeps_raw_cloud(tmp_path, raw_episodes, monkeypatch, rgb_only):
    import cv2

    from dexmani_real.config.perception import PointCloudConfig
    from dexmani_real.config.real import RealConfig
    from dexmani_real.data.raw import _VideoWriter, camera_metadata
    from dexmani_real.deployment.observation import ObservationBuilder
    from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
    from dexmani_real.perception.pointcloud import PointCloudProcessor

    source, _, _ = raw_episodes
    monkeypatch.setattr("dexmani_real.data.export.MAX_BATCH_FRAMES", 2)
    episode = source / "episode_0"
    color = CameraIntrinsics(64, 48, 100., 100., 32., 24., "brown_conrady", (.8, 0, .001, 0, 0))
    camera = CameraSnapshot(None, color, None if rgb_only else .001, None if rgb_only else np.eye(4))
    y, x = np.indices((48, 64))
    image = np.stack((x * 4, y * 5, (x + y) * 2), axis=-1).astype(np.uint8)
    depth = np.where(x < 32, 600, 1200).astype(np.uint16)
    depth[18:28, 26:36] = 0
    with h5py.File(episode / "data.h5", "r+") as f:
        f["meta"].attrs.update(camera_metadata(camera))
        if not rgb_only:
            f.create_dataset("depth", data=np.repeat(depth[None], 4, axis=0))
    video = _VideoWriter(episode / "rgb.mp4", fps=20, width=64, height=48)
    video.open()
    try:
        for i in range(4):
            video.write_frame(image, 1_000_000_000 + i * 50_000_000)
    finally:
        video.close()
    with RawReader(episode) as reader:
        images = list(reader.iter_rgb_frames())
        snapshot = reader.read_camera()
    before = [frame.copy() for frame in images]
    init = Mock(wraps=cv2.initUndistortRectifyMap)
    remap = Mock(wraps=cv2.remap)
    monkeypatch.setattr(cv2, "initUndistortRectifyMap", init)
    monkeypatch.setattr(cv2, "remap", remap)
    config = ExportConfig(pointcloud=PointCloudConfig(
        num_points=32, remove_table=False, workspace=(-2, -2, -2, 2, 2, 2),
    ))
    fields = ("rgb", "action") if rgb_only else (
        "rgb", "depth", "point_cloud", "camera_intrinsic", "camera_extrinsic", "action",
    )
    output = tmp_path / "images.zarr"
    export_zarr(episode, output, processing=config, fields=fields)
    assert init.call_count == 1
    assert remap.call_count == (4 if rgb_only else 8)
    root = zarr.open_group(str(output), mode="r")
    assert root.attrs["image_geometry"] == "pinhole"
    builder = ObservationBuilder(SimpleNamespace(observation_fields=("rgb",)), RealConfig())
    builder.configure_camera(snapshot)
    history = []
    for i, frame in enumerate(images):
        observed = builder.project(SimpleNamespace(camera={"rgb": frame, "ring_sequence": i + 1}))
        history.append(observed)
        np.testing.assert_array_equal(root["data/rgb"][i], observed["rgb"])
        np.testing.assert_array_equal(frame, before[i])
    from dexmani_policy.common.inference import prepare_observations
    from dexmani_policy.datasets.demonstration import DemonstrationDataset

    resize = (24, 32) if rgb_only else None
    modalities = ["rgb"] if rgb_only else ["rgb", "point_cloud", "camera_intrinsic", "camera_extrinsic"]
    dataset = DemonstrationDataset(
        str(output), sensor_modalities=modalities, obs_horizon=2, horizon=2,
        rgb_preprocess_size=resize,
    )
    online = prepare_observations(
        builder.build(history[:2]), normalization_dims={"rgb": 3}, n_obs_steps=2,
        rgb_preprocessing={"resize_hw": resize, "center_crop_hw": None}, device="cpu",
    )
    np.testing.assert_array_equal(dataset[0]["obs"]["rgb"].numpy(), online["rgb"][0].numpy())

    # 足够远的越界点全部填零，边界相邻点仍遵循双线性权重。
    maps = builder.rgb_maps
    outside = (maps[0] < -1) | (maps[0] > 64) | (maps[1] < -1) | (maps[1] > 48)
    assert outside.any()
    assert not root["data/rgb"][0][outside].any()
    if not rgb_only:
        from scripts import visualize_episode

        # 只替换显示输出；实际矫正、Raw 读取及返回给点云的深度均走真实实现。
        display = Mock()
        monkeypatch.setattr(visualize_episode, "log_rgbd", display)
        with RawReader(episode) as reader:
            viewer = SimpleNamespace(depth=reader["depth"], camera_model=snapshot, image_maps=maps)
            raw_depth = visualize_episode.EpisodeVisualizer.log_camera(viewer, 0, images[0])
        np.testing.assert_array_equal(display.call_args.args[0], root["data/rgb"][0])
        np.testing.assert_array_equal(display.call_args.args[1], root["data/depth"][0])
        np.testing.assert_array_equal(raw_depth, depth)
        np.testing.assert_array_equal(images[0], before[0])
        processor = PointCloudProcessor(
            depth_scale_m=.001, intrinsics=color, T_xarm_base_from_color=np.eye(4),
            table_plane_abcd=None, config=config.pointcloud,
        )
        for i, frame in enumerate(images):
            np.testing.assert_array_equal(root["data/point_cloud"][i], processor.process(depth, frame))
        expected_depth = cv2.remap(
            depth, *maps, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        np.testing.assert_array_equal(root["data/depth"][0], expected_depth)
        assert root["data/depth"].dtype == np.uint16
        assert set(np.unique(expected_depth)) == {0, 600, 1200}
        assert not expected_depth[outside].any()
        assert root.attrs["depth_scale_m_per_unit"] == .001
        with h5py.File(episode / "data.h5", "r") as f:
            np.testing.assert_array_equal(f["depth"][0], depth)
        previous_calls = init.call_count, remap.call_count
        cloud_output = tmp_path / "cloud.zarr"
        export_zarr(episode, cloud_output, processing=config, fields=("point_cloud",))
        assert (init.call_count, remap.call_count) == previous_calls
        cloud_root = zarr.open_group(str(cloud_output), mode="r")
        assert "image_geometry" not in cloud_root.attrs
        np.testing.assert_array_equal(cloud_root["data/point_cloud"][:], root["data/point_cloud"][:])
        cloud_root.store.close()
        depth_output = tmp_path / "depth.zarr"
        export_zarr(episode, depth_output, processing=config, fields=("depth",))
        depth_root = zarr.open_group(str(depth_output), mode="r")
        np.testing.assert_array_equal(depth_root["data/depth"][:], root["data/depth"][:])
        assert depth_root.attrs["image_geometry"] == "pinhole"
        depth_root.store.close()
    root.store.close()


@pytest.mark.parametrize("missing", [
    "camera_color_intrinsics", "camera_color_distortion_model", "camera_color_distortion_coeffs",
])
def test_rgb_metadata_requires_actual_geometry(missing):
    from dexmani_real.data.raw import camera_metadata, decode_camera_metadata
    from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot

    camera = CameraSnapshot(None, CameraIntrinsics(4, 3, 4., 4., 2., 1., "none", (0.,) * 5), None, None)
    meta = camera_metadata(camera)
    del meta[missing]
    with pytest.raises(KeyError, match=missing):
        decode_camera_metadata(meta)


@pytest.mark.parametrize("bad_index", [0, 1, 2])
def test_bad_raw_rolls_back_all_fields(tmp_path, raw_episodes, monkeypatch, bad_index):
    import shutil

    source, states, actions = raw_episodes
    shutil.copytree(source / "episode_1", source / "episode_2")
    with h5py.File(source / f"episode_{bad_index}/data.h5", "r+") as data:
        del data["timing/segment_starts"]
        data.create_dataset("timing/segment_starts", data=[0, 2])
        data["action_hand_joint_target"][-1, 0] = np.nan
    monkeypatch.setattr("dexmani_real.data.export.MAX_BATCH_FRAMES", 1)
    output = tmp_path / "clean.zarr"
    result = export_zarr(
        source, output, processing=ExportConfig(), fields=("joint_state", "action")
    )
    assert result["episode_count"] == 2
    root = zarr.open_group(str(output), mode="r")
    expected_states = np.concatenate([part for i, part in enumerate((states[:4], states[4:], states[4:])) if i != bad_index])
    expected_actions = np.concatenate([part for i, part in enumerate((actions[:4], actions[4:], actions[4:])) if i != bad_index])
    np.testing.assert_array_equal(root["data/joint_state"][:], expected_states.astype("f4"))
    np.testing.assert_array_equal(root["data/action"][:], expected_actions.astype("f4"))
    assert root["meta/episode_ends"].shape == (2,)
    assert root.attrs["source_episodes"] == [f"episode_{i}" for i in range(3) if i != bad_index]
    assert result["source_count"] == 2


def test_native_segments_and_truncated_prefix_are_accepted(tmp_path, raw_episodes):
    source, states, _ = raw_episodes
    with h5py.File(source / "episode_0/data.h5", "r+") as data:
        del data["timing/segment_starts"]
        data.create_dataset("timing/segment_starts", data=[0, 2])
        data["meta"].attrs["recording_truncated_reason"] = "recording_queue_full"
    output = tmp_path / "segments.zarr"
    result = export_zarr(source, output, fields=("joint_state",), processing=ExportConfig())
    root = zarr.open_group(str(output), mode="r")
    assert result["source_count"] == 2 and result["episode_count"] == 3
    assert root.attrs["source_episodes"] == ["episode_0", "episode_0", "episode_1"]
    assert root.attrs["source_row_ranges"] == [[0, 2], [2, 4], [0, 3]]
    np.testing.assert_array_equal(root["meta/episode_ends"][:], [2, 4, 7])
    np.testing.assert_array_equal(root["data/joint_state"][:], states.astype("f4"))


@pytest.mark.parametrize("stage", ["read", "write"])
@pytest.mark.parametrize("failure", [
    OSError("disk full"), KeyboardInterrupt(), RuntimeError("bug"),
    TypeError("bug"), IndexError("bug"),
])
def test_export_failures_propagate(tmp_path, raw_episodes, monkeypatch, stage, failure):
    def fail(*args, **kwargs):
        raise failure

    if stage == "read":
        monkeypatch.setattr(RawReader, "read_timing", fail)
    else:
        monkeypatch.setattr("dexmani_real.data.export.append_rows", fail)
    output = tmp_path / "failed.zarr"
    with pytest.raises(type(failure)):
        export_zarr(raw_episodes[0], output, processing=ExportConfig(), fields=("joint_state",))
    assert not output.exists()
    assert not list(tmp_path.glob(".failed.zarr.tmp-*"))


@pytest.fixture
def complete_episodes(raw_episodes):
    from dexmani_real.data.raw import (
        TACTILE_FINGER_ORDER,
        _VideoWriter,
        camera_metadata,
    )
    from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot

    source, _, _ = raw_episodes
    camera = CameraSnapshot(
        None,
        CameraIntrinsics(32, 32, 500.0, 500.0, 16.0, 16.0, "none", (0.0,) * 5),
        0.001,
        np.eye(4),
    )
    for episode in source.iterdir():
        with h5py.File(episode / "data.h5", "r+") as data:
            n = len(data["timing/rows"])
            data["meta"].attrs.update(camera_metadata(camera))
            data["meta"].attrs.update(
                hand_tactile_finger_order=TACTILE_FINGER_ORDER, hand_tactile_scale_divisor=255
            )
            data.create_dataset("hand_contact", data=np.zeros((n, 5, 3), "f4"))
            data.create_dataset("hand_tactile_force", data=np.zeros((n, 5, 120, 3), "f4"))
            data.create_dataset("depth", data=np.full((n, 32, 32), 600, "u2"))
        writer = _VideoWriter(episode / "rgb.mp4", fps=20, width=32, height=32)
        writer.open()
        for i in range(n):
            writer.write_frame(np.full((32, 32, 3), 100, "u1"), 1_000_000_000 + i * 50_000_000)
        writer.close()
    return source


@pytest.mark.parametrize("mode", ["none", "all", "mixed"])
@pytest.mark.parametrize("select_object", [False, True])
def test_fixed_modalities_produce_one_requested_store(tmp_path, complete_episodes, mode, select_object):
    from dexmani_real.config.perception import PointCloudConfig
    from dexmani_real.data.export import FIELD_INPUTS
    from dexmani_real.data.hoi import write_hoi

    source = complete_episodes
    for i, episode in enumerate(sorted(source.iterdir())):
        if mode == "none" or mode == "mixed" and i == 1:
            continue
        with RawReader(episode) as reader:
            n = reader.num_frames
        write_hoi(
            episode / "hoi.h5",
            {
                "points_local": np.zeros((1024, 3), "f4"),
                "offsets": np.array([0, 1024]),
                "object_names": np.array(["object"]),
            },
            np.tile(np.eye(4), (n, 1, 1, 1)),
            np.eye(4),
            quality=[[{"iou": 1.0}]] * n,
            tracking_config={},
            object_valid=np.ones((n, 1), bool),
        )
    output = tmp_path / "task.zarr"
    config = ExportConfig(
        pointcloud=PointCloudConfig(remove_table=False, workspace=(-1.0, -1.0, 0.0, 1.0, 1.0, 2.0))
    )
    fields = tuple(name for name in FIELD_INPUTS if select_object or name != "object_pointcloud")
    if select_object and mode == "none":
        with pytest.raises(ValueError, match="all Raw episodes rejected"):
            export_zarr(source, output, processing=config, fields=fields)
        assert not output.exists()
        return
    result = export_zarr(source, output, processing=config, fields=fields)
    expected_count = 1 if select_object and mode == "mixed" else 2
    assert result["total_frames"] == (4 if expected_count == 1 else 7)
    assert result["episode_count"] == result["source_count"] == expected_count
    assert result["path"] == str(output)
    assert set(result) == {"path", "source_count", "episode_count", "total_frames", "rejected_episodes", "excluded_episodes"}
    root = zarr.open_group(str(output), mode="r")
    assert set(root["data"]) == set(fields)
    assert all(a.shape[0] == result["total_frames"] for a in root["data"].values())
    assert len(root["meta/episode_ends"]) == expected_count
    assert not (tmp_path / "task_modalities").exists()


@pytest.mark.parametrize("fault", ["missing", "extra", "decode"])
@pytest.mark.parametrize("bad_index", [0, 1, 2])
def test_video_row_errors_roll_back_and_continue(tmp_path, complete_episodes, monkeypatch, fault, bad_index):
    import shutil

    from dexmani_real.data.raw import _VideoWriter

    source = complete_episodes
    shutil.copytree(source / "episode_0", source / "episode_2")
    counts = [4, 3, 4]
    bad = source / f"episode_{bad_index}"
    if fault != "decode":
        video = _VideoWriter(bad / "rgb.mp4", fps=20, width=32, height=32)
        video.open()
        try:
            for row in range(counts[bad_index] + (-1 if fault == "missing" else 1)):
                video.write_frame(np.full((32, 32, 3), 100, "u1"), 1_000_000_000 + row * 50_000_000)
        finally:
            video.close()
    original = RawReader.iter_rgb_frames
    reads = []

    def frames(reader):
        reads.append(reader.path.name)
        yield from original(reader)
        if fault == "decode" and reader.path == bad:
            raise RawDataError("corrupt video tail")

    monkeypatch.setattr(RawReader, "iter_rgb_frames", frames)
    monkeypatch.setattr("dexmani_real.data.export.MAX_BATCH_FRAMES", 1)
    result = export_zarr(
        source, tmp_path / "video.zarr", processing=ExportConfig(), fields=("joint_state", "rgb")
    )
    expected_frames = sum(counts) - counts[bad_index]
    assert result["total_frames"] == expected_frames
    assert reads == [f"episode_{i}" for i in range(3)]
    root = zarr.open_group(result["path"], mode="r")
    assert root.attrs["source_episodes"] == [f"episode_{i}" for i in range(3) if i != bad_index]
    assert root["data/rgb"].shape[0] == root["data/joint_state"].shape[0] == expected_frames
    assert [item["episode"] for item in result["rejected_episodes"]] == [bad.name]


def test_processing_survives_contact_and_checkpoint(tmp_path, complete_episodes, hoi_episodes):
    from dexmani_policy.common.checkpoint import (
        Checkpoint,
        load_checkpoint,
        save_checkpoint,
    )
    from dexmani_policy.datasets.demonstration import DemonstrationDataset
    from dexmani_policy.datasets.episode_store import EpisodeStore
    from dexmani_policy.datasets.preprocess.contact_map import generate
    from dexmani_policy.datasets.pretrain.cordvip import CordViPPretrainDataset
    from dexmani_policy.training.common.data import record_data_processing
    from dexmani_policy.training.pretrain.hand_prior.vqvae import prepare_hand_data
    from omegaconf import OmegaConf

    from dexmani_real.config.real import RealConfig
    from dexmani_real.deployment.preflight import collect_data_processing

    output = tmp_path / "source.zarr"
    with h5py.File(complete_episodes / "episode_0/data.h5", "r+") as data:
        del data["timing/segment_starts"]
        data.create_dataset("timing/segment_starts", data=[0, 2])
    export_zarr(
        complete_episodes,
        output,
        processing=ExportConfig(),
        fields=("hand_pointcloud", "object_pointcloud", "joint_state", "action"),
    )
    derived = tmp_path / "contact.zarr"
    generate(output, derived)
    source = EpisodeStore(output)
    result = EpisodeStore(derived)
    assert result.attrs == source.attrs
    demo = DemonstrationDataset(str(output), sensor_modalities=["joint_state"],
                                obs_horizon=2, horizon=2, seed=7, val_episodes=1, max_train_episodes=1)
    pretrain = CordViPPretrainDataset(derived, contact_loss_weight=1.0, seed=7,
                                    val_episodes=1, max_train_episodes=1, horizon=2, obs_horizon=2)
    for dataset in (demo, pretrain):
        assert dataset.train_mask[0] == dataset.train_mask[1]
        assert dataset.val_mask[0] == dataset.val_mask[1]
        assert not (dataset.train_mask & dataset.val_mask).any()
        assert dataset.episode_store.count_groups(dataset.train_mask) == 1
        for start, stop, *_ in dataset.sampler.indices:
            assert not start < 2 < stop
            assert not start < 4 < stop
    hand_cfg = OmegaConf.create({"dataset": {
        "zarr_path": str(output), "seed": 7, "val_episodes": 1, "max_train_episodes": 1,
    }})
    train_hand, val_hand, normalizer = prepare_hand_data(hand_cfg)
    train_hand = normalizer["hand"].unnormalize(train_hand).numpy()
    val_hand = normalizer["hand"].unnormalize(val_hand).numpy()
    assert sorted((len(train_hand), len(val_hand))) == [3, 4]
    assert (train_hand[:, 0] < 1).all() != (val_hand[:, 0] < 1).all()
    cfg = OmegaConf.create({})
    record_data_processing(cfg, pretrain)
    checkpoint = Checkpoint(
        global_step=0, model_state={}, ema_model_state=None, optimizer_state={}, scheduler_state={},
        effective_config=OmegaConf.to_container(cfg, resolve=True),
    )
    restored = load_checkpoint(save_checkpoint(tmp_path / "model.pt", checkpoint))
    assert restored.effective_config["data_processing"] == source.attrs["data_processing"]
    runtime = RealConfig()
    info = SimpleNamespace(
        observation_fields=("hand_pointcloud", "object_pointcloud"),
        data_processing=restored.effective_config["data_processing"],
    )
    with h5py.File(complete_episodes / "episode_0/hoi.h5") as annotations:
        template = {key: annotations[f"template/{key}"][:] for key in ("points_local", "offsets")}
    assert collect_data_processing(info, runtime, template=template) == {
        "hand_pointcloud": {"num_points": runtime.hand_pointcloud.num_points},
        "object_pointcloud": {
            "geometry_sha256": info.data_processing["object_pointcloud"]["geometry_sha256"],
            "max_joint_gap_s": runtime.execution.hoi_max_joint_gap_s,
        },
    }
    assert "geometry_sha256" in info.data_processing["object_pointcloud"]
    assert "geometry_sha256" not in info.data_processing["hand_pointcloud"]


@pytest.mark.parametrize("invalid", [
    "missing_cloud", "invalid_flag", "nonfinite", "scalar_names", "names_group",
])
def test_unselected_hoi_is_not_opened(tmp_path, hoi_episodes, invalid, monkeypatch):
    source, _, _ = hoi_episodes
    with h5py.File(source / "episode_0/hoi.h5", "r+") as data:
        if invalid == "missing_cloud":
            del data["object_pointcloud"]
        elif invalid == "invalid_flag":
            data["object_valid"][0] = False
        elif invalid == "nonfinite":
            data["object_pointcloud"][-1, 0, 0] = np.nan
        else:
            del data["template/object_names"]
            if invalid == "scalar_names":
                data.create_dataset("template/object_names", data="object")
            else:
                data.create_group("template/object_names")
    def unexpected(reader):
        pytest.fail("unselected HOI was opened")

    monkeypatch.setattr(RawReader, "hoi_file", property(unexpected))
    result = export_zarr(
        source, tmp_path / "valid.zarr", processing=ExportConfig(), fields=("joint_state",)
    )
    assert result["source_count"] == 2 and result["total_frames"] == 7
    assert not result["rejected_episodes"]


@pytest.mark.parametrize("with_object", [False, True])
def test_complete_interpolation_preserves_every_row(tmp_path, hoi_episodes, with_object, monkeypatch):
    from dexmani_real.utils.timing import interpolate_joints

    interpolate = Mock(wraps=interpolate_joints)
    monkeypatch.setattr("dexmani_real.data.hoi.interpolate_joints", interpolate)
    source, states, actions = hoi_episodes
    raw_states = states.copy()
    expected = states.copy()
    offset = 0
    for episode in sorted(source.iterdir()):
        with h5py.File(episode / "data.h5", "r+") as data:
            rows = data["timing/rows"][:]
            n = len(rows)
            if offset == 0:
                # 两组重复反馈；首尾相机精确命中，中间一帧跨组插值。
                for device, columns in (("arm", slice(0, 7)), ("hand", slice(7, 19))):
                    rows[f"{device}_ns"] = rows[f"{device}_ns"][[0, 0, 2, 2]]
                    data[f"{device}_qpos"][:] = states[[0, 0, 2, 2], columns]
                rows["camera_ns"][-1] = rows["camera_ns"][-2]
                raw_states[:n] = expected[:n] = states[[0, 0, 2, 2]]
                expected[1] = (states[0] + states[2]) / 2
            else:
                rows["camera_ns"][1:-1] -= 10_000_000
                expected[offset + 1 : offset + n - 1] = (
                    0.2 * states[offset : offset + n - 2] + 0.8 * states[offset + 1 : offset + n - 1]
                )
            data["timing/rows"][:] = rows
            offset += n
    fields = ("joint_state", "action", "object_pointcloud") if with_object else ("joint_state", "action")
    output = tmp_path / "aligned.zarr"
    export_zarr(source, output, processing=ExportConfig(), fields=fields)
    root = zarr.open_group(str(output), mode="r")
    assert interpolate.call_count == (4 if with_object else 0)  # 仅所选 HOI 联合观测需要对齐。
    expected_states = expected if with_object else raw_states
    np.testing.assert_allclose(root["data/joint_state"][:], expected_states, atol=1e-6)
    np.testing.assert_allclose(root["data/action"][:], actions, atol=1e-6)
    np.testing.assert_array_equal(root["meta/episode_ends"][:], [4, 7])


@pytest.mark.parametrize("with_object", [False, True])
@pytest.mark.parametrize("fault", ["zero", "before_history", "gap"])
def test_hoi_timing_is_checked_only_when_selected(tmp_path, hoi_episodes, with_object, fault):
    source, states, _ = hoi_episodes
    with h5py.File(source / "episode_0/data.h5", "r+") as data:
        rows = data["timing/rows"][:]
        if fault == "zero":
            rows["camera_ns"][0] = 0
        elif fault == "before_history":
            rows["camera_ns"][0] -= 10_000_000
        else:
            rows["camera_ns"][1] -= 10_000_000
        data["timing/rows"][:] = rows
    output = tmp_path / "valid.zarr"
    fields = ("joint_state", "object_pointcloud") if with_object else ("joint_state",)
    result = export_zarr(source, output, fields=fields, processing=ExportConfig(
        hoi_max_joint_gap_s=0.01 if fault == "gap" else 0.10,
    ))
    if with_object:
        assert result["episode_count"] == 1 and result["total_frames"] == 3
        assert result["rejected_episodes"][0]["episode"] == "episode_0"
        reason = result["rejected_episodes"][0]["reason"]
        assert ("camera_ns" if fault == "zero" else "unalignable") in reason
    else:
        assert result["episode_count"] == 2 and result["total_frames"] == 7
        assert not result["rejected_episodes"]
    root = zarr.open_group(str(output), mode="r")
    assert set(root["data"]) == set(fields)
    assert root.attrs["source_episodes"] == (["episode_1"] if with_object else ["episode_0", "episode_1"])
    np.testing.assert_array_equal(root["meta/episode_ends"][:], [3] if with_object else [4, 7])
    np.testing.assert_array_equal(root["data/joint_state"][:], (states[4:] if with_object else states).astype("f4"))
    assert all(array.shape[0] == result["total_frames"] for array in root["data"].values())


@pytest.mark.parametrize("bad_index", [0, 1])
@pytest.mark.parametrize("fault,reason", [
    ("scalar_timing", "timing/rows"),
    ("group_timing", "timing/rows"),
    ("matrix_timing", "timing/rows"),
    ("integer_names", "arm_joint_names"),
    ("joint_shape", "arm_qpos: expected shape"),
    ("integer_joints", "floating dtype"),
    ("audio_only", "video stream"),
])
def test_malformed_raw_is_skipped(tmp_path, complete_episodes, bad_index, fault, reason, caplog):
    import shutil

    import av

    from dexmani_real.config.perception import PointCloudConfig

    source = complete_episodes
    shutil.copytree(source / "episode_1", source / "episode_2")
    config = ExportConfig(pointcloud=PointCloudConfig(
        remove_table=False, workspace=(-1., -1., 0., 1., 1., 2.),
    ))
    reference = tmp_path / "reference.zarr"
    fields = ("joint_state", "rgb", "depth", "point_cloud", "action")
    export_zarr(source, reference, processing=config, fields=fields)
    bad = source / f"episode_{bad_index}"
    if fault == "audio_only":
        with av.open(str(bad / "rgb.mp4"), "w", format="mp4") as container:
            stream = container.add_stream("aac", rate=48000)
            stream.layout = "mono"
            frame = av.AudioFrame.from_ndarray(
                np.zeros((1, 2048), np.float32), format="fltp", layout="mono",
            )
            frame.sample_rate = 48000
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)
        with av.open(str(bad / "rgb.mp4")) as container:
            assert len(container.streams.audio) == 1
            assert not container.streams.video
    else:
        with h5py.File(bad / "data.h5", "r+") as data:
            if fault == "integer_names":
                data["meta"].attrs["arm_joint_names"] = 123
            elif fault in {"joint_shape", "integer_joints"}:
                joints = data["arm_qpos"][:]
                del data["arm_qpos"]
                data["arm_qpos"] = joints[:, :6] if fault == "joint_shape" else joints.astype(np.int32)
            else:
                rows = data["timing/rows"][:]
                del data["timing/rows"]
                if fault == "group_timing":
                    data.create_group("timing/rows")
                else:
                    values = rows[0] if fault == "scalar_timing" else rows[:, None]
                    data.create_dataset("timing/rows", data=values)
    reader = RawReader.__new__(RawReader)
    with pytest.raises(RawDataError, match=reason):
        reader.__init__(bad)
        with reader:
            if fault == "audio_only":
                list(reader.iter_rgb_frames())
            else:
                reader["arm_qpos"]
    assert reader.data_file is None and reader.video_container is None
    output = tmp_path / "clean.zarr"
    result = export_zarr(source, output, processing=config, fields=fields)
    assert len(result["rejected_episodes"]) == 1
    assert result["rejected_episodes"][0]["episode"] == bad.name
    assert reason in result["rejected_episodes"][0]["reason"]
    assert bad.name in caplog.text and reason in caplog.text
    root = zarr.open_group(str(output), mode="r")
    baseline = zarr.open_group(str(reference), mode="r")
    keep = np.r_[4:10] if bad_index == 0 else np.r_[0:4, 7:10]
    assert result["episode_count"] == 2 and result["total_frames"] == len(keep)
    assert root.attrs["source_episodes"] == [f"episode_{i}" for i in range(3) if i != bad_index]
    np.testing.assert_array_equal(root["meta/episode_ends"][:], [3, 6] if bad_index == 0 else [4, 7])
    assert set(root["data"]) == set(baseline["data"])
    for name in root["data"]:
        np.testing.assert_array_equal(root["data"][name][:], baseline["data"][name][:][keep])
    assert not list(tmp_path.glob(".clean.zarr.tmp-*"))


@pytest.mark.parametrize("fields", [("point_cloud",), ("point_cloud", "rgb")])
def test_unsupported_distortion_rejects_only_that_raw(tmp_path, complete_episodes, fields):
    import shutil

    from dexmani_real.config.perception import PointCloudConfig

    source = complete_episodes
    shutil.copytree(source / "episode_1", source / "episode_2")
    config = ExportConfig(pointcloud=PointCloudConfig(
        remove_table=False, workspace=(-1., -1., 0., 1., 1., 2.),
    ))
    reference = tmp_path / "reference.zarr"
    export_zarr(source, reference, processing=config, fields=fields)
    bad = source / "episode_1"
    with h5py.File(bad / "data.h5", "r+") as data:
        data["meta"].attrs["camera_color_distortion_model"] = "unsupported_model"

    output = tmp_path / "clean.zarr"
    result = export_zarr(source, output, processing=config, fields=fields)
    assert result["path"] == str(output) and output.is_dir()
    assert result["source_count"] == result["episode_count"] == 2
    assert result["total_frames"] == 7
    assert result["rejected_episodes"] == [{
        "episode": bad.name,
        "reason": "unsupported camera distortion model 'unsupported_model'",
    }]
    root = zarr.open_group(str(output), mode="r")
    baseline = zarr.open_group(str(reference), mode="r")
    assert root.attrs["source_episodes"] == ["episode_0", "episode_2"]
    assert root.attrs["source_row_ranges"] == [[0, 4], [0, 3]]
    np.testing.assert_array_equal(root["meta/episode_ends"][:], [4, 7])
    assert set(root["data"]) == set(fields)
    for name in fields:
        assert len(root["data"][name]) == 7
        np.testing.assert_array_equal(root["data"][name][:], baseline["data"][name][:][np.r_[0:4, 7:10]])


@pytest.mark.parametrize("names,valid", [
    (json.dumps(XARM7_JOINT_NAMES), True),
    (np.bytes_(json.dumps(XARM7_JOINT_NAMES)), True),
    ("[invalid json", False), (np.bytes_(b"\xff"), False),
])
def test_joint_names_json_boundary(raw_episodes, names, valid):
    episode = raw_episodes[0] / "episode_0"
    with h5py.File(episode / "data.h5", "r+") as data:
        data["meta"].attrs["arm_joint_names"] = names
    with RawReader(episode) as reader:
        if valid:
            reader["arm_qpos"]
        else:
            with pytest.raises(RawDataError, match="arm_joint_names: invalid joint names format"):
                reader["arm_qpos"]
    assert reader.data_file is None


@pytest.mark.parametrize("bad_index", [0, 1, 2])
@pytest.mark.parametrize("key,value", [
    ("camera_payload_mode", np.array(["rgb", "rgb"], dtype=h5py.string_dtype())),
    ("camera_color_distortion_coeffs", 0.0),
    ("camera_color_distortion_model", np.array(["none"], dtype=h5py.string_dtype())),
])
def test_bad_camera_metadata_rejects_only_that_raw(tmp_path, complete_episodes, bad_index, key, value):
    import shutil

    source = complete_episodes
    shutil.copytree(source / "episode_1", source / "episode_2")
    bad = source / f"episode_{bad_index}"
    with h5py.File(bad / "data.h5", "r+") as data:
        data["meta"].attrs[key] = value
    output = tmp_path / "clean.zarr"
    result = export_zarr(source, output, processing=ExportConfig(), fields=("joint_state", "rgb", "depth"))
    assert [item["episode"] for item in result["rejected_episodes"]] == [bad.name]
    assert result["episode_count"] == 2
    root = zarr.open_group(str(output), mode="r")
    assert root.attrs["source_episodes"] == [f"episode_{i}" for i in range(3) if i != bad_index]
    ends = [3, 6] if bad_index == 0 else [4, 7]
    np.testing.assert_array_equal(root["meta/episode_ends"][:], ends)
    assert all(len(values) == ends[-1] for values in root["data"].values())
    assert not list(tmp_path.glob(".clean.zarr.tmp-*"))


@pytest.mark.parametrize("fields", [("rgb", "depth"), ("joint_state",)])
def test_export_cli_matches_api(tmp_path, complete_episodes, monkeypatch, fields):
    from dexmani_real.config.perception import PointCloudConfig
    from dexmani_real.utils import log
    from scripts import export_zarr as cli

    monkeypatch.setattr(log, "configure_logging", lambda: None)
    config = ExportConfig(pointcloud=PointCloudConfig(
        remove_table=False, workspace=(-1., -1., 0., 1., 1., 2.),
    ))
    monkeypatch.setattr(cli, "load_processing_config", lambda path, fields: config)
    output = tmp_path / "cli.zarr"
    args = [str(complete_episodes), "--output", str(output), "--fields", *fields]
    assert cli.main(args) == 0
    reference = tmp_path / "api.zarr"
    export_zarr(complete_episodes, reference, processing=config, fields=fields)
    actual = zarr.open_group(str(output), mode="r")
    expected = zarr.open_group(str(reference), mode="r")
    assert dict(actual.attrs) == dict(expected.attrs)
    assert set(actual["data"]) == set(expected["data"])
    np.testing.assert_array_equal(actual["meta/episode_ends"][:], expected["meta/episode_ends"][:])
    for name in actual["data"]:
        np.testing.assert_array_equal(actual["data"][name][:], expected["data"][name][:])


@pytest.mark.parametrize("input_form", ["task_cwd", "episode_cwd", "home"])
def test_export_cli_default_output_uses_resolved_input(raw_episodes, tmp_path, monkeypatch, input_form):
    from dexmani_real.utils import log
    from scripts import export_zarr as cli

    source, _, _ = raw_episodes
    monkeypatch.setattr(log, "configure_logging", lambda: None)
    monkeypatch.setenv("HOME", str(tmp_path))
    working_dir = {
        "task_cwd": source, "episode_cwd": source / "episode_0", "home": tmp_path,
    }[input_form]
    monkeypatch.chdir(working_dir)
    requested = f"~/{source.name}" if input_form == "home" else "."
    assert cli.main([requested, "--fields", "joint_state"]) == 0
    output = working_dir / "datasets" / f"{source.name}.zarr"
    root = zarr.open_group(str(output), mode="r")
    assert root.attrs["task_name"] == "recorded_task"
    expected = ["episode_0"] if input_form == "episode_cwd" else ["episode_0", "episode_1"]
    assert root.attrs["source_episodes"] == expected
    root.store.close()


def test_export_requires_explicit_fields(tmp_path, raw_episodes):
    from scripts.export_zarr import build_parser

    with pytest.raises(TypeError, match="fields"):
        export_zarr(raw_episodes[0], tmp_path / "missing.zarr", processing=ExportConfig())
    with pytest.raises(SystemExit):
        build_parser().parse_args([str(raw_episodes[0])])
    assert not (tmp_path / "missing.zarr").exists()


@pytest.mark.parametrize("key", ["control_hz", "depth_scale"])
@pytest.mark.parametrize("value", ["bad", [20.0], [[20.0]], np.nan, np.inf, True])
@pytest.mark.parametrize("bad_index", [0, 1, 2])
def test_scalar_metadata_errors_are_local_to_raw(tmp_path, complete_episodes, key, value, bad_index):
    import shutil

    source = complete_episodes
    shutil.copytree(source / "episode_1", source / "episode_2")
    bad = source / f"episode_{bad_index}"
    with h5py.File(bad / "data.h5", "r+") as data:
        data["meta"].attrs[key] = value
    result = export_zarr(source, tmp_path / "valid.zarr", processing=ExportConfig(), fields=("depth",))
    assert result["source_count"] == 2
    assert [item["episode"] for item in result["rejected_episodes"]] == [bad.name]
    assert key in result["rejected_episodes"][0]["reason"]


@pytest.mark.parametrize("depth_scale", [None, "bad", [np.nan]])
def test_rgb_ignores_unused_camera_metadata(tmp_path, complete_episodes, depth_scale):
    for episode in complete_episodes.iterdir():
        with h5py.File(episode / "data.h5", "r+") as data:
            del data["meta"].attrs["depth_scale"]
            del data["meta"].attrs["camera_payload_mode"]
            data["meta"].attrs["camera_serial"] = [1, 2]
            if depth_scale is not None:
                data["meta"].attrs["depth_scale"] = depth_scale
    result = export_zarr(complete_episodes, tmp_path / "rgb.zarr", processing=ExportConfig(), fields=("rgb",))
    assert result["source_count"] == 2 and not result["rejected_episodes"]


@pytest.mark.parametrize("index_dtype", [np.int64, np.uint64])
def test_joint_alignment_never_crosses_native_gap(tmp_path, hoi_episodes, index_dtype):
    source, _, _ = hoi_episodes
    with h5py.File(source / "episode_0/data.h5", "r+") as data:
        del data["timing/segment_starts"]
        data.create_dataset("timing/segment_starts", data=np.array([0, 2], dtype=index_dtype))
        timing = data["timing/rows"][:]
        timing["camera_ns"][2] -= 10_000_000
        data["timing/rows"][:] = timing
    result = export_zarr(source, tmp_path / "aligned.zarr", processing=ExportConfig(),
                         fields=("joint_state", "object_pointcloud"))
    assert result["source_count"] == result["episode_count"] == 1
    assert "row 2 in segment 1" in result["rejected_episodes"][0]["reason"]


def test_joint_alignment_reads_one_native_segment_at_a_time(tmp_path, hoi_episodes, monkeypatch):
    source, states, _ = hoi_episodes
    episode = source / "episode_0"
    with h5py.File(episode / "data.h5", "r+") as data:
        del data["timing/segment_starts"]
        data.create_dataset("timing/segment_starts", data=[0, 2])
    reads = {"/arm_qpos": [], "/hand_qpos": []}
    getitem = h5py.Dataset.__getitem__

    def read(dataset, key, *args, **kwargs):
        if dataset.name in reads:
            reads[dataset.name].append(key)
        return getitem(dataset, key, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", read)
    result = export_zarr(episode, tmp_path / "segments.zarr", processing=ExportConfig(),
                         fields=("joint_state", "object_pointcloud"))
    assert not result["rejected_episodes"]
    assert result["source_count"] == 1 and result["episode_count"] == 2
    assert reads == {name: [slice(0, 2), slice(2, 4)] for name in reads}
    root = zarr.open_group(result["path"], mode="r")
    np.testing.assert_array_equal(root["data/joint_state"][:], states[:4].astype("f4"))
    np.testing.assert_array_equal(root["meta/episode_ends"][:], [2, 4])


def test_rejected_first_source_does_not_set_baselines(tmp_path, complete_episodes, hoi_episodes, monkeypatch):
    from dexmani_real.geometry.hoi import geometry_digest

    source = complete_episodes
    with h5py.File(source / "episode_0/data.h5", "r+") as data:
        data["meta"].attrs.update(task_label="rejected_task", control_hz=10.0, depth_scale=0.002)
        del data["timing/segment_starts"]
        data.create_dataset("timing/segment_starts", data=[0, 2])
    with h5py.File(source / "episode_0/hoi.h5", "r+") as data:
        data["template/points_local"][0, 0] = 1
        data["object_pointcloud"][-1] = np.nan
    monkeypatch.setattr("dexmani_real.data.export.MAX_BATCH_FRAMES", 1)
    result = export_zarr(source, tmp_path / "valid.zarr", processing=ExportConfig(),
                         fields=("joint_state", "rgb", "depth", "object_pointcloud"))
    root = zarr.open_group(result["path"], mode="r")
    assert result["source_count"] == result["episode_count"] == 1
    assert root.attrs["source_episodes"] == ["episode_1"]
    assert root.attrs["source_row_ranges"] == [[0, 3]]
    assert root.attrs["task_name"] == "recorded_task"
    assert root.attrs["dt"] == 0.05 and root.attrs["depth_scale_m_per_unit"] == 0.001
    assert root.attrs["data_processing"]["object_pointcloud"]["geometry_sha256"] == geometry_digest(
        np.zeros((1024, 3), "f4"), [0, 1024],
    )
    np.testing.assert_array_equal(root["meta/episode_ends"][:], [3])
    assert all(len(array) == 3 for array in root["data"].values())


@pytest.mark.parametrize("fault", ["missing", "group"])
def test_invalid_segments_reject_only_that_raw(tmp_path, raw_episodes, fault):
    source, states, _ = raw_episodes
    with h5py.File(source / "episode_0/data.h5", "r+") as data:
        del data["timing/segment_starts"]
        if fault == "group":
            data.create_group("timing/segment_starts")
    result = export_zarr(source, tmp_path / "segments.zarr", processing=ExportConfig(), fields=("joint_state",))
    assert result["source_count"] == 1
    message = "segment_starts must be a dataset" if fault == "group" else "segment_starts"
    assert message in result["rejected_episodes"][0]["reason"]
    root = zarr.open_group(result["path"], mode="r")
    np.testing.assert_array_equal(root["data/joint_state"][:], states[4:].astype("f4"))


@pytest.mark.parametrize("starts,gap", [([0], .1), ([0], .01), ([0, 1], .1)])
@pytest.mark.parametrize("stop", [1, 2, 3])
@pytest.mark.parametrize("delay_ns", [20_000_000, 100_000_000])
def test_hoi_prefix_matches_full_alignment(raw_episodes, starts, gap, stop, delay_ns):
    from dexmani_real.data.hoi import align_hoi_episode

    source, *_ = raw_episodes
    episode = source / "episode_0"
    with h5py.File(episode / "data.h5", "r+") as data:
        rows = data["timing/rows"][:]
        rows["arm_ns"] -= delay_ns
        rows["hand_ns"] -= delay_ns + 10_000_000
        data["timing/rows"][:] = rows
        del data["timing/segment_starts"]
        data.create_dataset("timing/segment_starts", data=starts)
    with RawReader(episode) as reader:
        timing = reader.read_timing()
        boundaries = np.r_[reader.segment_starts, reader.num_frames]
        full = align_hoi_episode(reader, timing, boundaries, gap)
        prefix = align_hoi_episode(reader, timing[:stop], boundaries, gap)
        for key in full:
            np.testing.assert_allclose(prefix[key], full[key][:stop], equal_nan=True)
        if starts == [0] and (gap == .1 or delay_ns == 100_000_000):
            assert np.isfinite(prefix["arm_qpos"][0]).all()
        else:
            assert np.isnan(prefix["arm_qpos"][0]).all()


@pytest.mark.parametrize("fault", ["missing", "dtype", "rows", "objects", "group"])
def test_hoi_object_valid_input_boundary(hoi_episodes, fault):
    from dexmani_real.data.hoi import read_hoi_validity

    source, *_ = hoi_episodes
    episode = source / "episode_0"
    with h5py.File(episode / "hoi.h5", "r+") as data:
        assert "valid" not in data
        del data["object_valid"]
        if fault == "group":
            data.create_group("object_valid")
        elif fault != "missing":
            shape = (3, 1) if fault == "rows" else (4, 2) if fault == "objects" else (4, 1)
            data.create_dataset("object_valid", data=np.ones(shape, np.uint8 if fault == "dtype" else bool))
    with RawReader(episode) as reader, pytest.raises((KeyError, RawDataError), match="object_valid"):
        read_hoi_validity(reader, stop=1)


def test_prefix_reads_needed_right_neighbor_not_bad_tail(raw_episodes, monkeypatch):
    from dexmani_real.data.hoi import align_hoi_episode

    source, *_ = raw_episodes
    episode = source / "episode_0"
    with h5py.File(episode / "data.h5", "r+") as data:
        rows = data["timing/rows"][:]
        for device in ("arm", "hand"):
            rows[f"{device}_ns"] -= 20_000_000
            rows[f"{device}_ns"][1] = rows[f"{device}_ns"][0]
            data[f"{device}_qpos"][1] = data[f"{device}_qpos"][0]
        data["timing/rows"][:] = rows
    with RawReader(episode) as reader:
        boundaries = np.r_[reader.segment_starts, reader.num_frames]
        expected = align_hoi_episode(reader, reader.read_timing(), boundaries, .1)
    with h5py.File(episode / "data.h5", "r+") as data:
        data["arm_qpos"][3] = np.nan
        data["hand_qpos"][3] = np.nan
        rows["arm_ns"][3] = rows["camera_ns"][3] = 0
        data["timing/rows"][:] = rows
    original = h5py.Dataset.__getitem__
    reads = []

    def read(dataset, key, **kwargs):
        if dataset.name in {"/arm_qpos", "/hand_qpos"}:
            assert key == slice(0, 1) or key == [1, 2]
            reads.append((dataset.name, key))
        if dataset.name == "/timing/rows":
            assert key in (slice(None, 1), 1, 2)
        return original(dataset, key, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", read)
    with RawReader(episode) as reader:
        boundaries = np.r_[reader.segment_starts, reader.num_frames]
        prefix = align_hoi_episode(reader, reader.read_timing(stop=1), boundaries, .1)
        for key in prefix:
            np.testing.assert_allclose(prefix[key], expected[key][:1], equal_nan=True)
    assert reads == [(name, key) for name in ("/arm_qpos", "/hand_qpos")
                     for key in (slice(0, 1), [1, 2])]


@pytest.mark.parametrize("case", ["missing", "interpolate", "exact"])
def test_prefix_skips_unneeded_stale_joint_values(raw_episodes, monkeypatch, case):
    from dexmani_real.data.hoi import align_hoi_episode

    source, *_ = raw_episodes
    episode = source / "episode_0"
    with h5py.File(episode / "data.h5", "r+") as data:
        rows = data["timing/rows"][:]
        for device in ("arm", "hand"):
            rows[f"{device}_ns"] = 980_000_000
            if case != "missing":
                rows[f"{device}_ns"][3] = 1_000_000_000 if case == "exact" else 1_080_000_000
            data[f"{device}_qpos"][:] = 0
        data["timing/rows"][:] = rows
    with RawReader(episode) as reader:
        boundaries = np.r_[reader.segment_starts, reader.num_frames]
        expected = align_hoi_episode(reader, reader.read_timing(), boundaries, .1)
    with h5py.File(episode / "data.h5", "r+") as data:
        for device in ("arm", "hand"):
            data[f"{device}_qpos"][1] = np.nan
            if case == "exact":
                data[f"{device}_qpos"][2] = np.nan
    original = h5py.Dataset.__getitem__

    def read(dataset, key, **kwargs):
        if dataset.name in {"/arm_qpos", "/hand_qpos"}:
            assert key == slice(0, 1) or (case == "interpolate" and key == [2, 3]) or (case == "exact" and key == [3])
        return original(dataset, key, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", read)
    with RawReader(episode) as reader:
        boundaries = np.r_[reader.segment_starts, reader.num_frames]
        actual = align_hoi_episode(reader, reader.read_timing(stop=1), boundaries, .1)
        for key in actual:
            np.testing.assert_allclose(actual[key], expected[key][:1], equal_nan=True)


def test_raw_alias_selected_only_once(tmp_path, raw_episodes):
    source, _, _ = raw_episodes
    alias = source / "episode_alias"
    alias.symlink_to(source / "episode_0", target_is_directory=True)
    output = tmp_path / "export/data.zarr"
    kwargs = {"processing": ExportConfig(), "fields": ("joint_state",)}
    with pytest.raises(ValueError, match="episode_0.*episode_alias"):
        export_zarr(source, output, **kwargs)
    assert not output.parent.exists()

    result = export_zarr(source, output, exclude=["episode_0"], **kwargs)
    root = zarr.open_group(result["path"], mode="r")
    assert root.attrs["source_episodes"] == ["episode_1", "episode_alias"]
    assert root.attrs["excluded_episodes"] == ["episode_0"]
    result = export_zarr(alias, tmp_path / "single.zarr", **kwargs)
    assert zarr.open_group(result["path"], mode="r").attrs["source_episodes"] == ["episode_alias"]


@pytest.mark.parametrize(("field", "value", "reason"), [
    ("base_from_camera", None, "base_from_camera"),
    ("camera_T_xarm_base_from_color", None, "camera_T_xarm_base_from_color"),
    ("base_from_camera", np.eye(4) + np.eye(4, k=3) * .20, "HOI base_from_camera conflicts"),
    ("base_from_camera", np.full((4, 4), b"x"), "HOI base_from_camera must contain real numeric coefficients"),
    ("camera_T_xarm_base_from_color", np.eye(4) + 1j, "must contain real numeric coefficients"),
])
def test_hoi_extrinsic_rejects_only_selected_raw(
    tmp_path, complete_episodes, hoi_episodes, field, value, reason,
):
    episode = complete_episodes / "episode_1"
    is_raw = field == "camera_T_xarm_base_from_color"
    with h5py.File(episode / ("data.h5" if is_raw else "hoi.h5"), "r+") as data:
        target = data["meta"].attrs if is_raw else data
        del target[field]
        if value is not None:
            target[field] = value
    result = export_zarr(complete_episodes, tmp_path / "hoi.zarr", processing=ExportConfig(),
                         fields=("object_pointcloud",))
    root = zarr.open_group(result["path"], mode="r")
    assert root.attrs["source_episodes"] == ["episode_0"]
    assert root.attrs["rejected_episodes"] == result["rejected_episodes"]
    rejected, = result["rejected_episodes"]
    assert rejected["episode"] == "episode_1" and reason in rejected["reason"]
    result = export_zarr(complete_episodes, tmp_path / "joints.zarr", processing=ExportConfig(),
                         fields=("joint_state",))
    assert result["source_count"] == 2


def test_pointcloud_export_records_normalized_table_plane(tmp_path, complete_episodes):
    from dexmani_real.config.perception import PointCloudConfig

    config = PointCloudConfig(workspace=(-1, -1, 0, 1, 1, 1))
    for height in (.05, .10):
        result = export_zarr(complete_episodes, tmp_path / f"{height}.zarr", fields=("point_cloud",),
                             processing=ExportConfig(pointcloud=config, table_plane_abcd=(0, 0, 2, -2 * height)))
        attrs = zarr.open_group(result["path"], mode="r").attrs
        assert attrs["data_processing"]["point_cloud"]["table_plane_abcd"] == [0, 0, 1, -height]
        assert "table_plane_abcd" not in attrs
    with pytest.raises(ValueError, match="upward"):
        export_zarr(complete_episodes, tmp_path / "invalid.zarr", fields=("point_cloud",),
                    processing=ExportConfig(pointcloud=config, table_plane_abcd=(0, 0, -1, .05)))


@pytest.mark.parametrize("field,value,reason", [
    ("action_arm_joint_target", np.nan, "finite targets"),
    ("action_hand_joint_target", 20.0, "absolute limits"),
])
def test_replay_reports_absolute_target_row_in_selected_segment(raw_episodes, field, value, reason):
    from dexmani_real.config.real import RealConfig
    from dexmani_real.replay.trajectory import load_trajectory

    episode = raw_episodes[0] / "episode_0"
    runtime = RealConfig()
    with h5py.File(episode / "data.h5", "r+") as data:
        del data["timing/segment_starts"]
        data["timing/segment_starts"] = [0, 2]
        data["action_arm_joint_target"][:] = runtime.arm.home_qpos
        data["action_hand_joint_target"][:] = np.deg2rad(runtime.hand.home_qpos_deg)
        data[field][3, 0] = value
    with pytest.raises(ValueError, match=rf"{field}.*{reason} at row 3"):
        load_trajectory(episode, runtime, segment=1)
