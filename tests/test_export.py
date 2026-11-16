"""Raw 整条导出、模态分组及下游元信息契约。"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import h5py
import numpy as np
import pytest
import zarr

from dexmani_real.data.export import ExportConfig, export_zarr
from dexmani_real.data.raw import ARM_ACTION_SEMANTICS, TIMING_ROW_DTYPE, RawReader
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
    with RawReader(episode) as reader:
        with pytest.raises(ValueError, match="depth must be uint16"):
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
    template = dict(
        points_local=np.zeros((1024, 3), np.float32),
        offsets=np.array([0, 1024]),
        object_names=np.array(["object"]),
    )
    for episode in sorted(source.iterdir()):
        with h5py.File(episode / "data.h5", "r+") as f:
            count = len(f["timing/rows"])
        write_hoi(
            episode / "data.h5",
            template,
            np.tile(np.eye(4), (count, 1, 1, 1)),
            np.eye(4),
            quality=[[{"iou": 1.0}]] * count,
            tracking_config={},
            object_valid=np.ones((count, 1), bool),
        )
    return source, states, actions


def test_hoi_unalignable_boundaries_reject_whole_raw(tmp_path, hoi_episodes):
    source, states, actions = hoi_episodes
    for episode in source.iterdir():
        with h5py.File(episode / "data.h5", "r+") as f:
            rows = f["timing/rows"][:]
            rows["camera_ns"] -= 10_000_000
            f["timing/rows"][:] = rows
    output = tmp_path / "aligned.zarr"
    with pytest.raises(ValueError, match="all Raw episodes rejected"):
        export_zarr(source, output, processing=ExportConfig(), fields=("joint_state", "object_pointcloud"))
    assert not output.exists()


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


def test_hoi_eef_export_does_not_require_hand_history(tmp_path, hoi_episodes):
    source, states, _ = hoi_episodes
    for episode in source.iterdir():
        with h5py.File(episode / "data.h5", "r+") as data:
            del data["hand_qpos"]
            del data["meta"].attrs["hand_joint_names"]
            rows = data["timing/rows"][:]
            rows["hand_ns"] = 0
            data["timing/rows"][:] = rows
    output = tmp_path / "arm_only.zarr"
    export_zarr(source, output, processing=ExportConfig(), fields=("eef_pose", "object_pointcloud"))
    root = zarr.open_group(str(output), mode="r")
    expected = encode_eef_pose(eef_poses(states[:, :7], arm_fk=ArmFK()))
    np.testing.assert_allclose(root["data/eef_pose"][:], expected, atol=1e-6)
    root.store.close()


@pytest.mark.parametrize("key", ["points_local", "offsets", "object_names"])
def test_hoi_export_rejects_different_templates(tmp_path, hoi_episodes, key):
    source, _, _ = hoi_episodes
    with h5py.File(source / "episode_1" / "data.h5", "r+") as data:
        array = data[f"hoi/template/{key}"]
        array[0] = "another_object" if key == "object_names" else 1
    output = tmp_path / "different_templates.zarr"
    result = export_zarr(source, output, processing=ExportConfig(), fields=("object_pointcloud",))
    assert result["episode_count"] == 1
    assert result["rejected_episodes"][0]["episode"] == "episode_1"
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
        aligned = align_hoi_episode(reader, 0.01)
        for values in aligned.values():
            assert len(values) == reader.num_frames
            assert np.isnan(values[1]).all()
            assert np.isfinite(values[[0, 2]]).all()


@pytest.mark.parametrize(
    "fields", [("object_pointcloud",), ("object_pointcloud", "joint_state", "action")]
)
def test_hoi_loss_rejects_whole_raw(tmp_path, hoi_episodes, fields):
    source, _, _ = hoi_episodes
    with h5py.File(source / "episode_1/data.h5", "r+") as data:
        data["hoi/object_valid"][1, 0] = False
        data["hoi/valid"][1] = False
        data["hoi/object_pointcloud"][1] = np.nan
    output = tmp_path / "filtered.zarr"
    export_zarr(source, output, processing=ExportConfig(), fields=fields)
    root = zarr.open_group(str(output), mode="r")
    np.testing.assert_array_equal(root["meta/episode_ends"][:], [4])
    assert np.isfinite(root["data/object_pointcloud"][:]).all()
    assert root.attrs["source_episodes"] == ["episode_0"]
    root.store.close()


@pytest.mark.parametrize("missing", ["valid", "object_valid"])
def test_export_rejects_missing_hoi_validity(tmp_path, hoi_episodes, missing):
    source, _, _ = hoi_episodes
    with h5py.File(source / "episode_0/data.h5", "r+") as data:
        del data[f"hoi/{missing}"]
    output = tmp_path / "invalid.zarr"
    result = export_zarr(source, output, processing=ExportConfig(), fields=("object_pointcloud",))
    assert result["episode_count"] == 1
    assert missing in result["rejected_episodes"][0]["reason"]


def test_export_rejects_episode_with_no_valid_hoi_rows(tmp_path, hoi_episodes):
    source, _, _ = hoi_episodes
    with h5py.File(source / "episode_0/data.h5", "r+") as data:
        data["hoi/valid"][:] = False
        data["hoi/object_valid"][:] = False
    output = tmp_path / "empty.zarr"
    result = export_zarr(source, output, processing=ExportConfig(), fields=("object_pointcloud",))
    assert result["episode_count"] == 1
    assert "invalid rows" in result["rejected_episodes"][0]["reason"]


@pytest.mark.parametrize("rgb_only", [False, True])
def test_pinhole_export_matches_online_and_keeps_raw_cloud(tmp_path, raw_episodes, monkeypatch, rgb_only):
    import cv2
    from dexmani_real.config.real import RealConfig
    from dexmani_real.config.perception import PointCloudConfig
    from dexmani_real.data.raw import camera_metadata, _VideoWriter
    from dexmani_real.deployment.observation import ObservationBuilder
    from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
    from dexmani_real.perception.pointcloud import PointCloudProcessor

    source, _, _ = raw_episodes
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
            viewer = SimpleNamespace(reader=reader, camera_model=snapshot, image_maps=maps)
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


@pytest.mark.parametrize("fault", ["gap", "truncated", "late_nan"])
@pytest.mark.parametrize("bad_index", [0, 1])
def test_bad_raw_rolls_back_all_fields(tmp_path, raw_episodes, monkeypatch, fault, bad_index):
    import shutil

    source, states, actions = raw_episodes
    shutil.copytree(source / "episode_1", source / "episode_2")
    with h5py.File(source / f"episode_{bad_index}/data.h5", "r+") as data:
        if fault == "gap":
            del data["timing/segment_starts"]
            data.create_dataset("timing/segment_starts", data=[0, 2])
        elif fault == "truncated":
            data["meta"].attrs["recording_truncated_reason"] = "queue_full"
        else:
            data["action_hand_joint_target"][-1, 0] = np.nan
    monkeypatch.setattr("dexmani_real.data.export.MAX_BATCH_FRAMES", 1)
    output = tmp_path / "clean.zarr"
    result = export_zarr(
        source, output, processing=ExportConfig(), fields=("joint_state", "action")
    )
    assert result["episode_count"] == 2
    root = zarr.open_group(str(output), mode="r")
    expected_states = np.concatenate((states[4:] if bad_index == 0 else states[:4], states[4:]))
    expected_actions = np.concatenate((actions[4:] if bad_index == 0 else actions[:4], actions[4:]))
    np.testing.assert_array_equal(root["data/joint_state"][:], expected_states.astype("f4"))
    np.testing.assert_array_equal(root["data/action"][:], expected_actions.astype("f4"))
    assert root["meta/episode_ends"].shape == (2,)


@pytest.mark.parametrize(
    "failure", [OSError("disk full"), KeyboardInterrupt(), RuntimeError("bug")]
)
def test_output_failures_propagate(tmp_path, raw_episodes, monkeypatch, failure):
    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr("dexmani_real.data.export.append_rows", fail)
    output = tmp_path / "failed.zarr"
    with pytest.raises(type(failure)):
        export_zarr(raw_episodes[0], output, processing=ExportConfig(), fields=("joint_state",))
    assert not output.exists()
    assert not list(tmp_path.glob(".failed.zarr.tmp-*"))


@pytest.fixture
def complete_episodes(raw_episodes):
    from dexmani_real.data.raw import camera_metadata, _VideoWriter, TACTILE_FINGER_ORDER
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
def test_default_complete_modalities(tmp_path, complete_episodes, mode):
    from dexmani_real.data.export import DEFAULT_FIELDS
    from dexmani_real.config.perception import PointCloudConfig
    from dexmani_real.data.hoi import write_hoi

    source = complete_episodes
    for i, episode in enumerate(sorted(source.iterdir())):
        if mode == "none" or mode == "mixed" and i == 1:
            continue
        with RawReader(episode) as reader:
            n = reader.num_frames
        write_hoi(
            episode / "data.h5",
            dict(
                points_local=np.zeros((1024, 3), "f4"),
                offsets=np.array([0, 1024]),
                object_names=np.array(["object"]),
            ),
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
    result = export_zarr(source, output, processing=config)
    assert result["total_frames"] == 7 and result["episode_count"] == 2
    assert len(result["outputs"]) == (2 if mode == "mixed" else 1)
    for product in result["outputs"]:
        root = zarr.open_group(product["path"], mode="r")
        expected = set(DEFAULT_FIELDS)
        if mode == "all" or mode == "mixed" and product["episodes"] == ["episode_0"]:
            expected.add("object_pointcloud")
        assert set(root["data"]) == expected
        assert all(a.shape[0] == product["total_frames"] for a in root["data"].values())
        assert len(root["meta/episode_ends"]) == product["episode_count"]
    if mode == "mixed":
        assert not output.exists()
        assert {Path(p["path"]).name for p in result["outputs"]} == {
            "with_object.zarr",
            "without_object.zarr",
        }


def test_late_video_error_rolls_back_and_continues(tmp_path, complete_episodes, monkeypatch):
    import shutil
    from dexmani_real.data.raw import RawDataError

    source = complete_episodes
    shutil.copytree(source / "episode_0", source / "episode_2")
    original = RawReader.iter_rgb_frames

    def frames(reader):
        yield from original(reader)
        if reader.path.name == "episode_1":
            raise RawDataError("corrupt video tail")

    monkeypatch.setattr(RawReader, "iter_rgb_frames", frames)
    monkeypatch.setattr("dexmani_real.data.export.MAX_BATCH_FRAMES", 1)
    result = export_zarr(
        source, tmp_path / "video.zarr", processing=ExportConfig(), fields=("joint_state", "rgb")
    )
    assert result["total_frames"] == 8
    root = zarr.open_group(result["outputs"][0]["path"], mode="r")
    assert root.attrs["source_episodes"] == ["episode_0", "episode_2"]
    assert root["data/rgb"].shape[0] == root["data/joint_state"].shape[0] == 8
    np.testing.assert_array_equal(root["data/rgb"][:4], root["data/rgb"][4:])


def test_processing_survives_contact_and_checkpoint(tmp_path, complete_episodes, hoi_episodes):
    from omegaconf import OmegaConf
    from dexmani_policy.datasets.preprocess.contact_map import generate
    from dexmani_policy.datasets.episode_store import EpisodeStore
    from dexmani_policy.training.common.data import record_data_processing
    from dexmani_policy.common.checkpoint import Checkpoint, save_checkpoint, load_checkpoint
    from dexmani_real.config.real import RealConfig
    from dexmani_real.deployment.preflight import check_data_processing
    from dexmani_real.geometry.kinematics import HandFK
    from dexmani_real.perception.hand_cloud import HandPointCloud

    output = tmp_path / "source.zarr"
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
    cfg = OmegaConf.create({})
    record_data_processing(cfg, result)
    checkpoint = Checkpoint(0, {}, None, {}, {}, OmegaConf.to_container(cfg, resolve=True))
    restored = load_checkpoint(save_checkpoint(tmp_path / "model.pt", checkpoint))
    assert restored.effective_config["data_processing"] == source.attrs["data_processing"]
    runtime = RealConfig()
    info = SimpleNamespace(
        observation_fields=("hand_pointcloud", "object_pointcloud"),
        data_processing=restored.effective_config["data_processing"],
    )
    observations = SimpleNamespace(
        hand_cloud=HandPointCloud(HandFK(), runtime.hand_pointcloud.num_points)
    )
    with h5py.File(complete_episodes / "episode_0/data.h5", "r") as raw:
        saved = raw["hoi/template"]
        template = dict(
            points_local=saved["points_local"][:],
            offsets=saved["offsets"][:],
            object_names=saved["object_names"].asstr()[:],
        )
    assert check_data_processing(info, runtime, observations, (None, template)) == info.data_processing
    template["points_local"][0, 0] += 0.01
    with pytest.raises(ValueError, match="data_processing mismatch: object_pointcloud"):
        check_data_processing(info, runtime, observations, (None, template))


@pytest.mark.parametrize("invalid", ["missing_cloud", "invalid_flag", "nonfinite"])
def test_bad_optional_hoi_cannot_be_omitted(tmp_path, hoi_episodes, invalid):
    source, _, _ = hoi_episodes
    with h5py.File(source / "episode_0/data.h5", "r+") as data:
        if invalid == "missing_cloud":
            del data["hoi/object_pointcloud"]
        elif invalid == "invalid_flag":
            data["hoi/valid"][0] = False
            data["hoi/object_valid"][0] = False
        else:
            data["hoi/object_pointcloud"][-1, 0, 0] = np.nan
    result = export_zarr(
        source, tmp_path / "valid.zarr", processing=ExportConfig(), fields=("joint_state",)
    )
    assert result["outputs"][0]["episodes"] == ["episode_1"]
    assert result["rejected_episodes"][0]["episode"] == "episode_0"


def test_complete_interpolation_preserves_every_row(tmp_path, hoi_episodes):
    source, states, actions = hoi_episodes
    expected = states.copy()
    offset = 0
    for episode in sorted(source.iterdir()):
        with h5py.File(episode / "data.h5", "r+") as data:
            rows = data["timing/rows"][:]
            rows["camera_ns"][1:-1] -= 10_000_000
            data["timing/rows"][:] = rows
            n = len(rows)
            expected[offset + 1 : offset + n - 1] = (
                0.2 * states[offset : offset + n - 2] + 0.8 * states[offset + 1 : offset + n - 1]
            )
            offset += n
    output = tmp_path / "aligned.zarr"
    export_zarr(
        source,
        output,
        processing=ExportConfig(),
        fields=("joint_state", "action", "object_pointcloud"),
    )
    root = zarr.open_group(str(output), mode="r")
    np.testing.assert_allclose(root["data/joint_state"][:], expected, atol=1e-6)
    np.testing.assert_allclose(root["data/action"][:], actions, atol=1e-6)
    np.testing.assert_array_equal(root["meta/episode_ends"][:], [4, 7])
