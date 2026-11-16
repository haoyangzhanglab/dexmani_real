"""Raw 导出元数据及连续区间的消费者契约。"""

import json
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
    for index, (frames, starts) in enumerate(((4, [0, 2]), (3, [0]))):
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
        assert result["raw_episode_count"] == 2
        assert result["episode_count"] == 3
        assert result["total_frames"] == 7
        assert set(root["data"]) == set(fields)
        np.testing.assert_array_equal(root["meta/episode_ends"][:], [2, 4, 7])
        for key, values in expected.items():
            array = root["data"][key]
            assert array.dtype == np.float32
            np.testing.assert_array_equal(array[:], values)
        root.store.close()


@pytest.mark.parametrize(
    ("episode_name", "task_label", "error"),
    [
        ("episode_1", "another_task", "task_label mismatch"),
        ("episode_0", None, "task_label"),
        ("episode_1", None, "task_label"),
    ],
)
def test_invalid_tasks_leave_no_output(
    tmp_path, raw_episodes, episode_name, task_label, error
):
    source, _, _ = raw_episodes
    with h5py.File(source / episode_name / "data.h5", "r+") as data:
        if task_label is None:
            del data["meta"].attrs["task_label"]
        else:
            data["meta"].attrs["task_label"] = task_label
    output = tmp_path / "invalid.zarr"
    with pytest.raises(ValueError, match=f"{episode_name}.*{error}"):
        export_zarr(source, output, processing=ExportConfig(), fields=("joint_state",))
    assert not output.exists()
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


def test_hoi_export_trims_all_fields_and_preserves_segments(tmp_path, hoi_episodes):
    source, states, actions = hoi_episodes
    for episode in source.iterdir():
        with h5py.File(episode / "data.h5", "r+") as f:
            rows = f["timing/rows"][:]
            rows["camera_ns"] -= 10_000_000
            f["timing/rows"][:] = rows
    output = tmp_path / "aligned.zarr"
    export_zarr(
        source,
        output,
        processing=ExportConfig(),
        fields=("joint_state", "action", "object_pointcloud"),
    )
    root = zarr.open_group(str(output), mode="r")
    np.testing.assert_array_equal(root["meta/episode_ends"][:], [1, 2, 4])
    kept = np.array([1, 3, 5, 6])
    np.testing.assert_allclose(root["data/action"][:], actions[kept], atol=1e-6)
    np.testing.assert_allclose(
        root["data/joint_state"][:], 0.2 * states[kept - 1] + 0.8 * states[kept], atol=1e-6
    )
    assert root.attrs["hoi_alignment"][0]["kept_ranges"] == [[1, 2], [3, 4]]
    root.store.close()


@pytest.mark.parametrize("with_hoi", [False, True])
@pytest.mark.parametrize("invalid", ["missing", "shape", "semantics"])
def test_export_rejects_invalid_actions_without_publishing(tmp_path, hoi_episodes, with_hoi, invalid):
    source, _, _ = hoi_episodes
    # 第一段正常写入后，第二段仍须由实际读取边界拒绝坏输入并清理临时产物。
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
    with pytest.raises(ValueError, match=message):
        export_zarr(source, output, processing=ExportConfig(), fields=fields)
    assert not output.exists()
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
    with pytest.raises(ValueError, match="HOI templates differ"):
        export_zarr(source, output, processing=ExportConfig(), fields=("object_pointcloud",))
    assert not output.exists()
    assert not list(tmp_path.glob(".different_templates.zarr.tmp-*"))


def test_hoi_trim_precedes_fk_and_keeps_images_in_step(tmp_path, hoi_episodes, monkeypatch):
    source, states, actions = hoi_episodes
    for episode in source.iterdir():
        (episode / "rgb.mp4").touch()
        with h5py.File(episode / "data.h5", "r+") as data:
            data["meta"].attrs.update(
                camera_color_width=2, camera_color_height=2, camera_payload_mode="rgb",
                camera_color_intrinsics=np.eye(3).ravel(),
                camera_color_distortion_model="none", camera_color_distortion_coeffs=np.zeros(5),
            )
            rows = data["timing/rows"][:]
            rows["camera_ns"] -= 10_000_000
            data["timing/rows"][:] = rows
            # 裁掉的首帧仍提供插值历史，但其动作不应送入 FK。
            for index in data["timing/segment_starts"][:]:
                data["action_arm_joint_target"][index] = np.nan

    def images(reader):
        for index in range(reader.num_frames):
            yield np.full((2, 2, 3), index, np.uint8)

    forward = ArmFK.forward

    def checked_forward(self, joints):
        assert np.isfinite(joints).all()
        return forward(self, joints)

    monkeypatch.setattr(RawReader, "iter_rgb_frames", images)
    monkeypatch.setattr(ArmFK, "forward", checked_forward)
    monkeypatch.setattr("dexmani_real.data.export.MAX_BATCH_FRAMES", 1)
    output = tmp_path / "trimmed_images.zarr"
    export_zarr(
        source,
        output,
        processing=ExportConfig(),
        fields=("eef_pose", "action_ee", "rgb", "object_pointcloud"),
    )
    root = zarr.open_group(str(output), mode="r")
    kept = np.array([1, 3, 5, 6])
    expected_joints = 0.2 * states[kept - 1, :7] + 0.8 * states[kept, :7]
    expected_poses = encode_eef_pose(eef_poses(expected_joints, arm_fk=ArmFK()))
    expected_actions = np.concatenate(
        (encode_eef_pose(eef_poses(actions[kept, :7], arm_fk=ArmFK())), actions[kept, 7:]), axis=1
    )
    np.testing.assert_allclose(root["data/eef_pose"][:], expected_poses, atol=1e-6)
    np.testing.assert_allclose(root["data/action_ee"][:], expected_actions, atol=1e-6)
    np.testing.assert_array_equal(root["data/rgb"][:, 0, 0, 0], [1, 3, 1, 2])
    np.testing.assert_array_equal(root["meta/episode_ends"][:], [1, 2, 4])
    root.store.close()


def test_hoi_internal_gap_is_not_trimmed(raw_episodes):
    from dexmani_real.data.hoi import align_hoi_episode
    from dexmani_real.data.raw import RawReader

    source, _, _ = raw_episodes
    episode = source / "episode_1"
    with h5py.File(episode / "data.h5", "r+") as f:
        rows = f["timing/rows"][:]
        rows["camera_ns"][1] -= 10_000_000
        f["timing/rows"][:] = rows
    with RawReader(episode) as reader:
        with pytest.raises(ValueError, match="internal joint gap"):
            align_hoi_episode(reader, 0.01)


@pytest.mark.parametrize(
    "fields", [("object_pointcloud",), ("object_pointcloud", "joint_state", "action")]
)
def test_hoi_loss_splits_training_windows(tmp_path, hoi_episodes, fields):
    source, _, _ = hoi_episodes
    with h5py.File(source / "episode_1/data.h5", "r+") as data:
        data["hoi/object_valid"][1, 0] = False
        data["hoi/valid"][1] = False
        data["hoi/object_pointcloud"][
            1
        ] = np.nan  # 该行必须在写入 canonical 数组前被过滤。
    output = tmp_path / "filtered.zarr"
    export_zarr(source, output, processing=ExportConfig(), fields=fields)
    root = zarr.open_group(str(output), mode="r")
    np.testing.assert_array_equal(root["meta/episode_ends"][:], [2, 4, 5, 6])
    assert np.isfinite(root["data/object_pointcloud"][:]).all()
    assert root.attrs["hoi_alignment"][1]["kept_ranges"] == [[0, 1], [2, 3]]
    root.store.close()


@pytest.mark.parametrize("missing", ["valid", "object_valid"])
def test_export_rejects_missing_hoi_validity(tmp_path, hoi_episodes, missing):
    source, _, _ = hoi_episodes
    with h5py.File(source / "episode_0/data.h5", "r+") as data:
        del data[f"hoi/{missing}"]
    output = tmp_path / "invalid.zarr"
    with pytest.raises(ValueError, match=missing):
        export_zarr(
            source, output, processing=ExportConfig(), fields=("object_pointcloud",)
        )
    assert not output.exists()


def test_export_rejects_episode_with_no_valid_hoi_rows(tmp_path, hoi_episodes):
    source, _, _ = hoi_episodes
    with h5py.File(source / "episode_0/data.h5", "r+") as data:
        data["hoi/valid"][:] = False
        data["hoi/object_valid"][:] = False
    output = tmp_path / "empty.zarr"
    with pytest.raises(ValueError, match="No valid HOI rows"):
        export_zarr(
            source, output, processing=ExportConfig(), fields=("object_pointcloud",)
        )
    assert not output.exists()


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
