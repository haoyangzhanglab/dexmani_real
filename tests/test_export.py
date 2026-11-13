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
