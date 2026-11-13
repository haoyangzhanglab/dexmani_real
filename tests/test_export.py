"""Raw → Zarr 的任务、动作及固定 HOI 模板契约，不连接设备。"""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import zarr

from dexmani_real.data.export import ExportConfig, export_zarr
from dexmani_real.data.hoi import write_hoi
from dexmani_real.data.raw import ARM_ACTION_SEMANTICS, TIMING_ROW_DTYPE, RawReader
from dexmani_real.robot.model import XARM7_JOINT_NAMES, XHAND_SDK_JOINT_NAMES


@pytest.fixture
def template():
    return {
        "object_names": np.array(["苹果", "bowl"]),
        "offsets": np.array([0, 400, 1024]),
        "points_local": np.random.default_rng(0).uniform(-0.05, 0.05, (1024, 3)).astype(np.float32),
    }


def make_episode(path, *, task="pick_place", template=None):
    path.mkdir(parents=True)
    frames = 4
    timing = np.zeros(frames, dtype=TIMING_ROW_DTYPE)
    control = 1_000_000_000 + np.arange(frames) * 50_000_000
    for name in timing.dtype.names:
        timing[name] = control
    timing["observation_ns"] += 1_000_000
    timing["camera_sequence"] = np.arange(1, frames + 1)
    with h5py.File(path / "data.h5", "w") as data:
        data.create_group("meta").attrs.update(
            task_label=task, control_hz=20.0, joint_units="rad",
            arm_joint_names=json.dumps(XARM7_JOINT_NAMES),
            hand_joint_names=json.dumps(XHAND_SDK_JOINT_NAMES),
            arm_action_semantics=ARM_ACTION_SEMANTICS,
        )
        data.create_dataset("timing/rows", data=timing)
        data.create_dataset("timing/segment_starts", data=[0, 2])
        for name, width, offset in (
            ("arm_qpos", 7, 0.0), ("hand_qpos", 12, 0.1),
            ("action_arm_joint_target", 7, 0.2), ("action_hand_joint_target", 12, 0.3),
        ):
            data.create_dataset(name, data=np.arange(frames * width).reshape(frames, width) * 0.001 + offset)
    if template is not None:
        poses = np.tile(np.eye(4), (frames, len(template["object_names"]), 1, 1))
        poses[:, :, 0, 3] = np.arange(frames)[:, None] * 0.01
        write_hoi(path / "data.h5", template, poses, np.eye(4))
    return path


def test_export_uses_recorded_task_and_preserves_actions_and_segments(tmp_path):
    source = tmp_path / "renamed_directory"
    first = make_episode(source / "episode_0")
    make_episode(source / "episode_1")
    output = tmp_path / "arbitrary_output_name.zarr"
    result = export_zarr(source, output, processing=ExportConfig(), fields=("joint_state", "action"))
    root = zarr.open_group(str(output), mode="r")
    assert root.attrs["task_name"] == "pick_place"
    assert root.attrs["action_space"] == "joint"
    assert root.attrs["dt"] == 0.05
    assert result["raw_episode_count"] == 2 and result["episode_count"] == 4
    np.testing.assert_array_equal(root["meta/episode_ends"][:], [2, 4, 6, 8])
    with RawReader(first) as reader:
        expected = np.concatenate((reader["action_arm_joint_target"][:], reader["action_hand_joint_target"][:]), axis=1)
    np.testing.assert_array_equal(root["data/action"][:4], expected.astype(np.float32))
    assert root["data/joint_state"].shape == (8, 19)
    assert root["data/action"].dtype == np.float32


@pytest.mark.parametrize("task", ["different_task", "", "   "])
def test_export_rejects_mixed_or_invalid_tasks_without_publishing(tmp_path, task):
    source = tmp_path / "raw"
    make_episode(source / "episode_0")
    make_episode(source / "episode_1", task=task)
    output = tmp_path / "bad.zarr"
    with pytest.raises(ValueError, match="task_label"):
        export_zarr(source, output, processing=ExportConfig(), fields=("action",))
    assert not output.exists()
    assert not list(tmp_path.glob(".bad.zarr.tmp-*"))


def test_observation_only_export_does_not_claim_an_action_space(tmp_path):
    episode = make_episode(tmp_path / "episode")
    output = tmp_path / "obs.zarr"
    export_zarr(episode, output, processing=ExportConfig(), fields=("joint_state",))
    assert "action_space" not in zarr.open_group(str(output), mode="r").attrs


@pytest.mark.parametrize("fields,space", [(("action_ee",), "eef"), (("action", "action_ee"), "joint")])
def test_action_space_describes_exported_actions(tmp_path, fields, space):
    pytest.importorskip("pinocchio")
    episode = make_episode(tmp_path / "episode")
    output = tmp_path / "actions.zarr"
    export_zarr(episode, output, processing=ExportConfig(), fields=fields)
    root = zarr.open_group(str(output), mode="r")
    assert root.attrs["action_space"] == space
    assert root["data/action_ee"].shape == (4, 21)
    with RawReader(episode) as reader:
        np.testing.assert_array_equal(root["data/action_ee"][:, 9:], reader["action_hand_joint_target"][:].astype(np.float32))


def test_hoi_template_roundtrip_and_single_snapshot_per_store(tmp_path, template):
    source = tmp_path / "raw"
    episode = make_episode(source / "episode_0", template=template)
    make_episode(source / "episode_1", template=template)
    with RawReader(episode) as reader:
        saved = reader.read_hoi_template()
        for key in template:
            np.testing.assert_array_equal(saved[key], template[key])
        expected = reader["object_pointcloud"][:]
    output = tmp_path / "hoi.zarr"
    export_zarr(source, output, processing=ExportConfig(), fields=("object_pointcloud", "action"))
    root = zarr.open_group(str(output), mode="r")
    assert root.attrs["object_names"] == template["object_names"].tolist()
    assert root.attrs["object_offsets"] == [0, 400, 1024]
    np.testing.assert_array_equal(root["meta/object_points_local"][:], template["points_local"])
    np.testing.assert_array_equal(root["data/object_pointcloud"][:4], expected)


@pytest.mark.parametrize("change", ["object_names", "offsets", "points_local", "missing"])
def test_export_rejects_mixed_hoi_templates(tmp_path, template, change):
    source = tmp_path / "raw"
    make_episode(source / "episode_0", template=template)
    other = {key: value.copy() for key, value in template.items()}
    if change == "object_names":
        other[change] = other[change][::-1]
    elif change == "offsets":
        other[change][1] += 1
    elif change == "points_local":
        # 对象名和点数完全相同，但物体内的点序发生变化。
        other[change][[0, 1]] = other[change][[1, 0]]
    episode = make_episode(source / "episode_1", template=other)
    if change == "missing":
        with h5py.File(episode / "data.h5", "r+") as data:
            del data["hoi/template"]
    output = tmp_path / "bad.zarr"
    with pytest.raises(ValueError, match="identical HOI templates"):
        export_zarr(source, output, processing=ExportConfig(), fields=("object_pointcloud",))
    assert not output.exists()
    assert not list(tmp_path.glob(".bad.zarr.tmp-*"))


def test_legacy_hoi_export_warns_without_inventing_metadata(tmp_path, template, caplog):
    episode = make_episode(tmp_path / "episode", template=template)
    with h5py.File(episode / "data.h5", "r+") as data:
        del data["hoi/template"]
    output = tmp_path / "legacy.zarr"
    export_zarr(episode, output, processing=ExportConfig(), fields=("object_pointcloud",))
    root = zarr.open_group(str(output), mode="r")
    assert "HOI 缺少模板快照" in caplog.text
    assert "object_names" not in root.attrs and "object_offsets" not in root.attrs
    assert "object_points_local" not in root["meta"]


def test_invalid_hoi_snapshot_is_not_treated_as_legacy(tmp_path, template):
    episode = make_episode(tmp_path / "episode", template=template)
    with h5py.File(episode / "data.h5", "r+") as data:
        data["hoi/template/offsets"][1] = 0
    with pytest.raises(ValueError, match="非法 HOI 点分段"):
        export_zarr(episode, tmp_path / "bad.zarr", processing=ExportConfig(), fields=("object_pointcloud",))


def test_policy_episode_store_reads_real_export(tmp_path, template):
    # 相邻安装 dexmani_policy 时验证真实读取器，不替换或修改策略实现。
    module = pytest.importorskip("dexmani_policy.datasets.episode_store")
    episode = make_episode(tmp_path / "episode", template=template)
    output = tmp_path / "policy.zarr"
    fields = ("joint_state", "action", "object_pointcloud")
    export_zarr(episode, output, processing=ExportConfig(), fields=fields)
    store = module.EpisodeStore(output, keys=fields)
    assert store.attrs["task_name"] == "pick_place"
    assert store.attrs["action_space"] == "joint"
    assert store.n_episodes == 2
    assert store.read("action", slice(0, 2)).shape == (2, 19)
    assert store.read("object_pointcloud", slice(0, 2)).shape == (2, 1024, 3)


def test_cordvip_export_matches_live_observation_geometry(tmp_path, template):
    pytest.importorskip("pinocchio")
    from dexmani_real.config.real import RealConfig
    from dexmani_real.deployment.observation import ObservationBuilder

    episode = make_episode(tmp_path / "episode", template=template)
    output = tmp_path / "cordvip.zarr"
    fields = ("joint_state", "hand_pointcloud", "object_pointcloud")
    export_zarr(episode, output, processing=ExportConfig(), fields=(*fields, "action"))
    builder = ObservationBuilder(SimpleNamespace(observation_fields=fields), RealConfig())
    with RawReader(episode) as reader:
        rows = [builder.project(SimpleNamespace(
            arm={"qpos": reader["arm_qpos"][index:index + 1]},
            hand={"qpos": reader["hand_qpos"][index:index + 1]},
            hoi={"object_pointcloud": reader["object_pointcloud"][index]},
        )) for index in range(reader.num_frames)]
    live = builder.build(rows)
    root = zarr.open_group(str(output), mode="r")
    for name in fields:
        np.testing.assert_array_equal(root[f"data/{name}"][:], live[name])
    assert root["data/hand_pointcloud"].shape == (4, 1024, 3)
