from __future__ import annotations

import multiprocessing as mp
import time
import uuid
from pathlib import Path

import h5py
import numpy as np

from dexmani_real.recording.episode_reader import EpisodeReader, ValidityState
from dexmani_real.recording.io_process import RecorderClient, RecorderIOConfig, recorder_io_loop
from dexmani_real.robot.types import RobotAction, RobotState
from dexmani_real.shm.shared_storage import SharedStorage, SharedStorageConfig


def _state(timestamp: float) -> RobotState:
    return RobotState(
        arm_qpos=np.zeros(7),
        arm_qvel=np.zeros(7),
        arm_tau=np.zeros(7),
        eef_pos=np.zeros(3),
        eef_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        eef_rot6d=np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        hand_qpos=np.zeros(12),
        hand_tactile_sum=np.zeros((5, 3)),
        hand_tactile_force=np.zeros((5, 120, 3)),
        hand_tactile_contact=np.zeros(5, dtype=bool),
        hand_tipboard_err=np.zeros(12, dtype=np.int32),
        hand_commboard_err=np.zeros(12, dtype=np.int32),
        hand_jointboard_err=np.zeros(12, dtype=np.int32),
        hand_qpos_stale=False,
        fingertip_pos=np.zeros((5, 3)),
        arm_connected=True,
        hand_connected=True,
        timestamp=timestamp,
        hand_current=np.zeros(12),
    )


def test_recorder_io_round_trip_over_bounded_shared_ring(tmp_path: Path) -> None:
    ctx = mp.get_context("spawn")
    prefix = f"recio_{uuid.uuid4().hex}"
    shared = SharedStorage.create(
        prefix,
        config=SharedStorageConfig(
            camera_rgb_shape=(8, 8, 3),
            camera_depth_shape=(8, 8),
            camera_pc_shape=(2, 6),
            camera_ring_maxlen=2,
            record_sample_ring_maxlen=2,
        ),
        mp_context=ctx,
    )
    process = ctx.Process(
        target=recorder_io_loop,
        args=(
            shared,
            RecorderIOConfig(
                data_dir=str(tmp_path),
                max_frames=4,
                control_hz=10.0,
                min_frames=1,
                resolved_config_json="{}",
                resolved_config_sha256="a" * 64,
            ),
        ),
        daemon=False,
    )
    process.start()
    try:
        assert shared.recorder_ready.wait(timeout=5.0)
        client = RecorderClient(shared)
        assert client.start_episode(camera_metadata={"camera_serial": "offline-test"})
        now_ns = time.monotonic_ns()
        source_ns = now_ns - 3_000_000
        receive_ns = now_ns - 2_000_000
        publish_ns = now_ns - 1_000_000
        signals = {
            "observation_id": 1,
            "observation_anchor_monotonic_ns": now_ns,
            "arm_source_sequence": 1,
            "hand_source_sequence": 1,
            "vr_source_sequence": 1,
            "camera_source_sequence": 1,
            "arm_source_monotonic_ns": source_ns,
            "hand_source_monotonic_ns": source_ns,
            "vr_source_monotonic_ns": source_ns,
            "camera_source_monotonic_ns": source_ns,
            "arm_publish_monotonic_ns": publish_ns,
            "hand_publish_monotonic_ns": publish_ns,
            "vr_publish_monotonic_ns": publish_ns,
            "camera_publish_monotonic_ns": publish_ns,
            "observation_source_receive_monotonic_ns": np.full(4, receive_ns, dtype=np.uint64),
            "observation_source_age_s": np.zeros(4),
            "observation_source_skew_s": np.zeros(4),
            "observation_history_valid_mask": np.ones((4, 1), dtype=bool),
            "observation_valid": True,
            "tactile_fresh": True,
            "tactile_source_monotonic_ns": source_ns,
            "action_id": 1,
            "action_chunk_id": 1,
            "action_created_monotonic_ns": now_ns - 100_000,
            "action_target_monotonic_ns": now_ns + 10_000_000,
            "action_valid_until_monotonic_ns": now_ns + 20_000_000,
            "action_queued": True,
            "action_committed": True,
            "ik_ok": True,
            "retarget_ok": True,
        }
        assert client.add_frame(
            _state(time.perf_counter()),
            RobotAction(np.zeros(7), np.zeros(12)),
            {
                "wrist_pos": np.zeros(3),
                "wrist_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
                "landmarks": np.zeros((21, 3)),
            },
            camera_frame={
                "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
                "depth": np.zeros((8, 8), dtype=np.uint16),
                "pointcloud": np.zeros((2, 6), dtype=np.float32),
                "camera_fresh": True,
                "pointcloud_valid": False,
                "camera_health": 0,
            },
            signals=signals,
            arm_qpos_sent=np.zeros(7),
        )
        client.stop_episode(success=True)
        assert client.join_stop(timeout=10.0)
        result = client.poll_stop()
        # join_stop consumes only the wait state; poll_stop may already have
        # been harvested, so inspect the published directory directly.
        episode_paths = sorted(tmp_path.glob("episode_*"))
        assert result.error is None
        assert len(episode_paths) == 1
        with EpisodeReader(episode_paths[0]) as reader:
            assert reader.validity is ValidityState.VALID
            assert reader.h5f["observation_id"][:].tolist() == [1]

        with h5py.File(episode_paths[0] / "data.h5", "r+") as data_file:
            original_tactile_source = int(data_file["tactile_source_monotonic_ns"][0])
            data_file["tactile_source_monotonic_ns"][0] = data_file["observation_anchor_monotonic_ns"][0] + 1
        with EpisodeReader(episode_paths[0]) as reader:
            assert reader.validity is ValidityState.INVALID
        with h5py.File(episode_paths[0] / "data.h5", "r+") as data_file:
            data_file["tactile_source_monotonic_ns"][0] = original_tactile_source
            data_file["arm_publish_monotonic_ns"][0] = data_file["observation_anchor_monotonic_ns"][0] + 1
        with EpisodeReader(episode_paths[0]) as reader:
            assert reader.validity is ValidityState.INVALID
    finally:
        shared.is_running.value = False
        process.join(timeout=10.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        assert process.exitcode == 0
        shared.close()


def test_recorder_io_aborts_active_episode_and_exits_on_runtime_shutdown(tmp_path: Path) -> None:
    ctx = mp.get_context("spawn")
    prefix = f"recio_shutdown_{uuid.uuid4().hex}"
    shared = SharedStorage.create(
        prefix,
        config=SharedStorageConfig(
            camera_rgb_shape=(8, 8, 3),
            camera_depth_shape=(8, 8),
            camera_pc_shape=(2, 6),
            camera_ring_maxlen=2,
            record_sample_ring_maxlen=2,
        ),
        mp_context=ctx,
    )
    process = ctx.Process(
        target=recorder_io_loop,
        args=(
            shared,
            RecorderIOConfig(
                data_dir=str(tmp_path),
                max_frames=4,
                control_hz=10.0,
                min_frames=1,
                resolved_config_json="{}",
                resolved_config_sha256="b" * 64,
            ),
        ),
        daemon=False,
    )
    process.start()
    try:
        assert shared.recorder_ready.wait(timeout=5.0)
        client = RecorderClient(shared)
        assert client.start_episode(camera_metadata={"camera_serial": "offline-test"})

        shared.is_running.value = False
        process.join(timeout=10.0)

        assert process.exitcode == 0
        manifests = sorted(tmp_path.glob("*.aborted.json"))
        assert len(manifests) == 1
        assert '"reason":"runtime_shutdown"' in manifests[0].read_text(encoding="utf-8")
        assert not list(tmp_path.glob(".tmp_episode_*"))
        assert not [path for path in tmp_path.glob("episode_*") if path.is_dir()]
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        shared.close()
