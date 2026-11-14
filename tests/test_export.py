"""Raw 导出元数据及连续区间的消费者契约。"""

import json

import h5py
import numpy as np
import pytest
import zarr

from dexmani_real.data.export import ExportConfig, export_zarr
from dexmani_real.data.raw import ARM_ACTION_SEMANTICS, TIMING_ROW_DTYPE
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
    from dexmani_real.data.raw import RawReader

    source, states, actions = hoi_episodes
    for episode in source.iterdir():
        (episode / "rgb.mp4").touch()
        with h5py.File(episode / "data.h5", "r+") as data:
            data["meta"].attrs.update(camera_color_width=2, camera_color_height=2)
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
