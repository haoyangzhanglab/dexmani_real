"""当前生产者到分组训练、恢复及 CPU 推理的跨仓数值链路。"""

import json
from contextlib import closing
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr

from dexmani_real.config.real import RealConfig
from dexmani_real.data.export import ExportConfig, export_zarr
from dexmani_real.data.hoi import write_hoi
from dexmani_real.data.raw import (
    RawReader,
    RawWriter,
    episode_recording_metadata,
    robot_recording_parameters,
)
from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE


@pytest.fixture(autouse=True)
def isolated_hydra():
    # SAM2 的模块初始化持有全局 Hydra；训练入口应在独立配置上下文中测试。
    from hydra.core.global_hydra import GlobalHydra

    state = GlobalHydra.instance()
    previous = state.hydra
    state.clear()
    try:
        yield
    finally:
        state.hydra = previous


def test_current_raw_to_grouped_policy_resume(tmp_path):
    from dexmani_policy.deployment import load_policy
    from dexmani_policy.test_tools.smoke_test import run_smoke
    from dexmani_policy.training.common.config import load_config
    from dexmani_policy.training.common.data import build_dataset

    source = tmp_path / "raw"
    camera = CameraSnapshot(
        None, CameraIntrinsics(32, 32, 500, 500, 16, 16, "none", (0.,) * 5),
        .001, np.eye(4),
    )
    parameters = {
        **robot_recording_parameters(RealConfig()),
        "policy": {"task_name": "pick_apple_messy"},
        "max_duration_s": 60.,
        "actual_processing": {},
    }
    metadata = episode_recording_metadata(camera, parameters, tactile_reset={})
    template = {
        "points_local": np.random.default_rng(4).uniform(-.01, .01, (1024, 3)).astype("f4"),
        "offsets": np.array([0, 1024]), "object_names": np.array(["object"]),
    }
    for capture in range(3):
        episode = source / f"episode_{capture}"
        episode.mkdir(parents=True)
        writer = RawWriter(episode, control_hz=20)
        writer.open(dict(metadata, collection_source="policy_rollout", task_label="pick_apple_messy"))
        try:
            for index in range(8):
                stamp = 1_000_000_000 + index * 50_000_000 + (500_000_000 if index >= 4 else 0)
                arm, hand = np.zeros(1, ARM_STATE_DTYPE), np.zeros(1, HAND_STATE_DTYPE)
                arm["timestamp_ns"] = hand["timestamp_ns"] = stamp
                hand["tactile_aggregate_timestamp_ns"] = hand["tactile_dense_timestamp_ns"] = stamp
                arm["qpos"] = (index + capture) * .01
                hand["qpos"] = (index + capture) * .02
                row = SimpleNamespace(
                    arm=arm, hand=hand, control_timestamp_ns=stamp, observation_timestamp_ns=stamp,
                    camera={
                        "timestamp_ns": stamp, "ring_sequence": index + 1,
                        "rgb": np.full((32, 32, 3), 90 + index, np.uint8),
                        "depth": np.full((32, 32), 600, np.uint16),
                    },
                )
                command = SimpleNamespace(arm_qpos=arm["qpos"][0] + .01, hand_qpos=hand["qpos"][0] + .01)
                writer.append_frame(row, command, segment_start=index == 4)
            writer.finish("operator")
        finally:
            writer.close()
        with RawReader(episode) as reader:
            assert reader.meta["effective_parameters"] == metadata["effective_parameters"]
            recorded = json.loads(reader.meta["effective_parameters"])
            for key, value in recorded["camera"].items():
                np.testing.assert_equal(reader.meta[key], value)
            assert recorded["tactile_reset"] == {}
            assert reader.meta["collection_source"] == "policy_rollout"
            assert reader.meta["termination_reason"] == "operator"
        poses = np.tile(np.eye(4), (8, 1, 1, 1))
        poses[:, 0, 2, 3] = .5 + np.arange(8) * .001
        write_hoi(episode / "hoi.h5", template, poses, np.eye(4),
                  quality=[[{}]] * 8, object_valid=np.ones((8, 1), bool), tracking_config={})
    output = tmp_path / "training.zarr"
    result = export_zarr(source, output, processing=ExportConfig(),
                         fields=("rgb", "joint_state", "action", "hand_pointcloud", "object_pointcloud"))
    assert (result["source_count"], result["episode_count"], result["total_frames"]) == (3, 6, 24)
    cfg = load_config("cordvip", [
        f"zarr_path={output}", "training.device=cpu", "horizon=4", "n_obs_steps=2", "n_action_steps=2",
        "agent.down_dims=[16,32]", "agent.diffusion_step_embed_dim=16", "agent.num_inference_steps=2",
        "training.use_compile=false", "training.use_bfloat16=false", "training.lr_warmup_steps=0",
        "training.loop.total_train_steps=2", "training.loop.log_interval_steps=1",
        "dataloader.batch_size=2", "dataloader.num_workers=0", "dataloader.pin_memory=false",
        "dataset.val_episodes=1",
    ])
    dataset, _ = build_dataset(cfg)
    groups = np.asarray(dataset.episode_store.episode_groups)
    assert not set(groups[dataset.train_mask]) & set(groups[dataset.val_mask])
    assert dataset.episode_store.count_groups(dataset.train_mask) == 2
    for index in range(len(dataset)):
        rows = dataset.sampler.sample_source_rows(index)
        assert rows[0] // 4 == rows[-1] // 4
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        run_smoke(cfg, tmp_path / "run", max_updates=1, resume_updates=1)
        data = zarr.open_group(str(output), mode="r")
        with closing(load_policy(tmp_path / "run", device="cpu")) as runtime:
            observation = {key: data[f"data/{key}"][:2] for key in runtime.info.observation_fields}
            prediction = runtime.predict(observation)
            assert prediction.shape == (3, 19) and np.isfinite(prediction).all()
    finally:
        torch.set_num_threads(threads)
