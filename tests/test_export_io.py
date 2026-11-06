"""真实 HDF5、PyAV 与 Zarr 的小规模一致性验证。"""

import numpy as np
import pytest
import zarr

from dexmani_real.data import export
from dexmani_real.data.raw import (
    TACTILE_FINGER_ORDER,
    TACTILE_UNITS,
    RawReader,
    RawWriter,
)
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE
from dexmani_real.runtime.observation import Observation
from dexmani_real.utils.io import publish


def test_raw_video_and_selected_export(tmp_path, monkeypatch):
    pytest.importorskip("av")
    pytest.importorskip("h5py")
    monkeypatch.setattr(export, "MAX_BATCH_FRAMES", 7)
    fields = (
        "joint_state",
        "action",
        "finger_tactile_force",
        "taxel_tactile_force",
        "rgb",
        "depth",
    )
    expected = {name: [] for name in fields}
    for episode, count in enumerate((35, 3)):
        staging = tmp_path / f".tmp_episode_{episode}"
        staging.mkdir()
        writer = RawWriter(staging, control_hz=10)
        try:
            writer.open(
                {
                    "task_label": "smoke",
                    "collection_source": "policy_rollout",
                    "camera_color_height": 48,
                    "camera_color_width": 64,
                    "depth_scale": 0.001,
                    "hand_tactile_units": TACTILE_UNITS,
                    "hand_tactile_finger_order": TACTILE_FINGER_ORDER,
                }
            )
            for i in range(count):
                stamp = 1_000_000_000 + i * 100_000_000
                arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
                hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
                arm["qpos"] = i * 0.001
                hand["qpos"] = i * 0.002
                arm["timestamp_ns"] = stamp
                for name in (
                    "timestamp_ns",
                    "tactile_aggregate_timestamp_ns",
                    "tactile_dense_timestamp_ns",
                ):
                    hand[name] = stamp
                row = Observation(
                    arm,
                    hand,
                    {
                        "rgb": np.full((48, 64, 3), i, np.uint8),
                        "depth": np.full((48, 64), 500 + i, np.uint16),
                        "timestamp_ns": stamp,
                        "ring_sequence": i,
                    },
                    None,
                    None,
                    stamp + 1000,
                    control_timestamp_ns=stamp,
                )
                writer.append_frame(
                    row, RobotCommand(np.full(7, i * 0.003), np.full(12, i * 0.004))
                )
            writer.finish("operator", [])
        finally:
            writer.close()
        path = publish(staging, tmp_path / f"episode_{episode}")
        with RawReader(path) as reader:
            assert reader.num_frames == count
            images = list(reader.iter_rgb_frames())
            assert len(images) == count
            expected["rgb"].extend(images)
            for name, source in [
                ("depth", "depth"),
                ("finger_tactile_force", "hand_contact"),
                ("taxel_tactile_force", "hand_tactile_force"),
            ]:
                expected[name].extend(reader[source][:])
            expected["joint_state"].extend(
                np.concatenate(
                    (reader["arm_qpos"][:], reader["hand_qpos"][:]), axis=1
                ).astype(np.float32)
            )
            expected["action"].extend(
                np.concatenate(
                    (
                        reader["action_arm_joint_target"][:],
                        reader["action_hand_joint_target"][:],
                    ),
                    axis=1,
                ).astype(np.float32)
            )
    output = tmp_path / "canonical.zarr"
    result = export.export_zarr(
        tmp_path, output, processing=export.ExportConfig(), fields=fields
    )
    assert result["total_frames"] == 38 and result["episode_count"] == 2
    root = zarr.open(str(output), mode="r")
    assert set(root["data"].keys()) == set(fields)
    np.testing.assert_array_equal(root["meta/episode_ends"][:], [35, 38])
    for name, values in expected.items():
        actual = root["data"][name][:]
        desired = np.asarray(values)
        assert actual.dtype == desired.dtype
        np.testing.assert_array_equal(actual, desired)
    root.store.close()


def test_previous_block_released_before_next_read(monkeypatch):
    import weakref
    from pathlib import Path

    monkeypatch.setattr(export, "MAX_BATCH_FRAMES", 1)
    refs = {}

    class Depth:
        def __getitem__(self, rows):
            assert all(ref() is None for ref in refs.values())
            return np.ones((rows.stop - rows.start, 2, 2), dtype=np.uint16)

    class Reader:
        path = Path("synthetic")
        num_frames = 3

        def __getitem__(self, name):
            assert name == "depth"
            return Depth()

        def iter_rgb_frames(self):
            for _ in range(self.num_frames):
                yield np.zeros((2, 2, 3), dtype=np.uint8)

    class Sink:
        def __init__(self, name, shape):
            self.name, self.shape = name, (0, *shape)

        def resize(self, shape):
            self.shape = shape

        def __setitem__(self, rows, values):
            refs[self.name] = weakref.ref(values)

    arrays = {"depth": Sink("depth", (2, 2)), "rgb": Sink("rgb", (2, 2, 3))}
    blocks = export.iter_canonical_blocks(
        Reader(),
        export.ExportConfig(),
        fields=("depth", "rgb"),
        arm_fk=None,
        hand_fk=None,
        hand_cloud=None,
        camera={"shape": (2, 2), "parameters": {}},
    )
    export.append_episode(arrays, 0, blocks, 3)
    assert all(ref() is None for ref in refs.values())


@pytest.mark.parametrize("case", ["valid", "j2", "j4", "narrow", "roundoff", "outside_roundoff"])
def test_replay_respects_device_and_operational_limits(tmp_path, case):
    import json
    from dataclasses import replace

    import h5py

    from dexmani_real.config.real import RealConfig
    from dexmani_real.replay.trajectory import load_trajectory
    from dexmani_real.robot.model import XARM7_JOINT_NAMES, XHAND_SDK_JOINT_NAMES

    runtime = RealConfig()
    target = np.array(runtime.arm.home_qpos)
    target[1] = 2.0
    if case == "j2":
        target[1] = 2.08
    elif case == "j4":
        target[3] = -0.15
    elif case == "narrow":
        upper = list(runtime.arm.joint_limit_upper)
        upper[1] = 1.0
        runtime = replace(runtime, arm=replace(runtime.arm, joint_limit_upper=tuple(upper)))
    elif case in {"roundoff", "outside_roundoff"}:
        target[1] = runtime.arm.joint_limit_upper[1] + (0.5e-6 if case == "roundoff" else 2e-6)
    with h5py.File(tmp_path / "data.h5", "w") as data:
        meta = data.create_group("meta").attrs
        meta["joint_units"] = "rad"
        meta["control_hz"] = 10
        meta["arm_joint_names"] = json.dumps(XARM7_JOINT_NAMES)
        meta["hand_joint_names"] = json.dumps(XHAND_SDK_JOINT_NAMES)
        data["action_arm_joint_target"] = target[None]
        data["action_hand_joint_target"] = np.deg2rad(runtime.hand.home_qpos_deg)[None]
    if case in {"j2", "j4", "narrow", "outside_roundoff"}:
        with pytest.raises(ValueError, match="configured absolute limits"):
            load_trajectory(tmp_path, runtime)
    else:
        trajectory = load_trajectory(tmp_path, runtime)
        expected = target.copy()
        if case == "roundoff":
            expected[1] = runtime.arm.joint_limit_upper[1]
        np.testing.assert_array_equal(trajectory.action_arm_joint[0], expected)
