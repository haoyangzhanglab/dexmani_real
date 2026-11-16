"""手部点云 FK 变换的批量实现回归。"""

from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

from dexmani_real.robot.model import XARM7_XHAND_RIGHT_URDF_PATH


def write_hand_template(directory, points, offsets):
    np.savez(
        directory / f"hand_{len(points)}.npz",
        points=points,
        part_names=np.array(["link_a", "link_b"]),
        part_offsets=offsets,
    )


class FakeHandFK:
    fingertip_frame_ids = (0, 1, 2, 3, 4)

    def __init__(self):
        self.link_ids = {"link_a": 5, "link_b": 6}
        self.model = SimpleNamespace(
            nframes=7,
            getFrameId=lambda name: self.link_ids[name],
        )

    @staticmethod
    def _placement(qpos, frame_id):
        angle = 0.01 * frame_id + 0.03 * qpos[frame_id % len(qpos)]
        cosine, sine = np.cos(angle), np.sin(angle)
        transform = np.eye(4)
        transform[:3, :3] = (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        )
        transform[:3, 3] = (
            0.01 * frame_id,
            0.02 * qpos[(frame_id + 1) % len(qpos)],
            0.003 * frame_id,
        )
        return transform

    def forward(self, qpos, frame_ids):
        return np.stack([self._placement(qpos, frame_id) for frame_id in frame_ids])


@pytest.mark.parametrize("offset_dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("batch_size", [1, 4, 32])
def test_hand_geometry_batch_transform_preserves_link_segments(
    tmp_path,
    monkeypatch,
    offset_dtype,
    batch_size,
):
    from dexmani_real.perception import hand_cloud as module

    rng = np.random.default_rng(42)
    local_points = rng.normal(0, 0.02, (6, 3)).astype(np.float32)
    offsets = np.array([0, 2, 6], dtype=offset_dtype)
    write_hand_template(tmp_path, local_points, offsets)
    monkeypatch.setattr(module, "HAND_TEMPLATE_DIR", tmp_path)

    hand_fk = FakeHandFK()
    hand_cloud = module.HandPointCloud(hand_fk, 6)
    arm_from_eef = np.eye(4)
    arm_from_eef[:3, :3] = (
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    arm_from_eef[:3, 3] = (0.03, -0.02, 0.01)
    arm_fk = SimpleNamespace(eef_from_handbase=arm_from_eef)

    eef_transforms = np.repeat(np.eye(4)[None], batch_size, axis=0)
    eef_transforms[:, :3, 3] = rng.normal(0, 0.1, (batch_size, 3))
    hand_qpos = rng.uniform(-0.5, 0.5, (batch_size, 12))

    actual = module.hand_geometry(
        eef_transforms,
        hand_qpos,
        arm_fk=arm_fk,
        hand_fk=hand_fk,
        hand_cloud=hand_cloud,
    )

    expected_tips = np.empty((len(eef_transforms), 5, 3), dtype=np.float32)
    expected_cloud = np.empty((len(eef_transforms), 6, 3), dtype=np.float32)
    for frame, (eef, qpos) in enumerate(zip(eef_transforms, hand_qpos)):
        base_from_hand = eef @ arm_from_eef
        frame_ids = [*hand_fk.fingertip_frame_ids, *hand_cloud.frame_ids]
        placements = hand_fk.forward(qpos, frame_ids)
        expected_tips[frame] = (
            placements[:5, :3, 3] @ base_from_hand[:3, :3].T + base_from_hand[:3, 3]
        )
        expected_cloud[frame] = local_points
        for link_index, (start, end) in enumerate(pairwise(offsets)):
            transform = base_from_hand @ placements[5 + link_index]
            expected_cloud[frame, start:end] = (
                local_points[start:end] @ transform[:3, :3].T + transform[:3, 3]
            )

    np.testing.assert_allclose(actual["fingertip_xyz"], expected_tips, atol=1e-7)
    np.testing.assert_allclose(actual["hand_pointcloud"], expected_cloud, atol=1e-7)
    assert actual["hand_pointcloud"].dtype == np.float32
    np.testing.assert_array_equal(hand_cloud.local_points, local_points)


def test_hand_cloud_batch_transform_without_fingertips(tmp_path, monkeypatch):
    from dexmani_real.perception import hand_cloud as module

    write_hand_template(
        tmp_path,
        np.arange(12, dtype=np.float32).reshape(4, 3),
        np.array([0, 1, 4]),
    )
    monkeypatch.setattr(module, "HAND_TEMPLATE_DIR", tmp_path)
    hand_fk = FakeHandFK()
    hand_cloud = module.HandPointCloud(hand_fk, 4)
    arm_fk = SimpleNamespace(eef_from_handbase=np.eye(4))
    eef = np.repeat(np.eye(4)[None], 2, axis=0)
    qpos = np.zeros((2, 12))

    result = module.hand_geometry(
        eef,
        qpos,
        arm_fk=arm_fk,
        hand_fk=hand_fk,
        hand_cloud=hand_cloud,
        include_fingertips=False,
    )
    assert set(result) == {"hand_pointcloud"}
    assert result["hand_pointcloud"].shape == (2, 4, 3)


@pytest.mark.parametrize(
    "eef_shape,qpos_shape",
    [
        ((2, 4, 4), (1, 12)),
        ((1, 4, 4), (2, 12)),
        ((0, 4, 4), (1, 12)),
        ((4, 4), (1, 12)),
        ((1, 3, 4), (1, 12)),
        ((1, 4, 4), (12,)),
        ((1, 4, 4), (1, 11)),
    ],
)
def test_hand_geometry_rejects_invalid_batches_before_fk(eef_shape, qpos_shape):
    from dexmani_real.perception.hand_cloud import hand_geometry

    with pytest.raises(ValueError, match="eef_transforms|hand_qpos"):
        hand_geometry(
            np.zeros(eef_shape),
            np.zeros(qpos_shape),
            arm_fk=None,
            hand_fk=None,
        )


@pytest.mark.parametrize("empty", [False, True])
def test_hand_geometry_without_outputs_does_not_run_fk(empty):
    from dexmani_real.perception.hand_cloud import hand_geometry

    count = 0 if empty else 2
    assert (
        hand_geometry(
            np.empty((count, 4, 4)),
            np.empty((count, 12)),
            arm_fk=None,
            hand_fk=None,
            include_fingertips=False,
        )
        == {}
    )


def test_hand_geometry_empty_batch_preserves_requested_shapes(tmp_path, monkeypatch):
    from dexmani_real.perception import hand_cloud as module

    write_hand_template(tmp_path, np.zeros((4, 3), np.float32), np.array([0, 1, 4]))
    monkeypatch.setattr(module, "HAND_TEMPLATE_DIR", tmp_path)
    fk = FakeHandFK()
    cloud = module.HandPointCloud(fk, 4)
    result = module.hand_geometry(
        np.empty((0, 4, 4)),
        np.empty((0, 12)),
        arm_fk=SimpleNamespace(eef_from_handbase=np.eye(4)),
        hand_fk=fk,
        hand_cloud=cloud,
    )
    assert result["fingertip_xyz"].shape == (0, 5, 3)
    assert result["hand_pointcloud"].shape == (0, 4, 3)
    assert all(value.dtype == np.float32 for value in result.values())


def test_split_fk_matches_complete_urdf_in_sdk_order():
    pin = pytest.importorskip("pinocchio")
    from dexmani_real.geometry.kinematics import ArmFK, HandFK
    from dexmani_real.perception.hand_cloud import HandPointCloud
    from dexmani_real.robot.model import (
        XARM7_JOINT_NAMES,
        XHAND_SDK_JOINT_NAMES,
    )

    arm, hand = ArmFK(), HandFK()
    cloud = HandPointCloud(hand, 1024)
    frame_ids = (*hand.fingertip_frame_ids, *cloud.frame_ids)
    full = pin.buildModelFromUrdf(str(XARM7_XHAND_RIGHT_URDF_PATH))
    data = full.createData()
    names = [hand.model.frames[fid].name for fid in frame_ids]
    rng = np.random.default_rng(17)
    for _ in range(8):
        arm_qpos = rng.uniform(-0.25, 0.25, 7)
        hand_qpos = rng.uniform(0, 0.5, 12)
        qpos = np.zeros(full.nq)
        for name, value in zip(
            (*XARM7_JOINT_NAMES, *XHAND_SDK_JOINT_NAMES),
            (*arm_qpos, *hand_qpos),
        ):
            qpos[full.joints[full.getJointId(name)].idx_q] = value
        pin.forwardKinematics(full, data, qpos)
        expected = np.stack(
            [
                pin.updateFramePlacement(full, data, full.getFrameId(name)).homogeneous.copy()
                for name in names
            ]
        )
        actual = arm.forward(arm_qpos) @ arm.eef_from_handbase @ hand.forward(hand_qpos, frame_ids)
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_generated_hand_template_uses_visual_origin_and_mesh_scale(tmp_path, monkeypatch):
    import trimesh
    from scipy.spatial.transform import Rotation
    from scripts.generate_hand_template import generate_hand_template
    from dexmani_real.perception import hand_cloud as module

    trimesh.creation.box(extents=(0.02, 0.03, 0.04)).export(tmp_path / "visual.obj")
    urdf = tmp_path / "hand.urdf"
    urdf.write_text("""<robot name="test"><link name="right_hand_link"><visual>
      <origin xyz="0.1 -0.05 0.02" rpy="0.2 0.3 0.4"/>
      <geometry><mesh filename="visual.obj" scale="1 2 3"/></geometry>
    </visual></link></robot>""")
    path = generate_hand_template(128, urdf_path=urdf, output_dir=tmp_path)
    with np.load(path, allow_pickle=False) as saved:
        assert set(saved.files) == {"points", "part_names", "part_offsets"}
        np.testing.assert_array_equal(saved["part_names"], ["right_hand_link"])
        np.testing.assert_array_equal(saved["part_offsets"], [0, 128])
        xyz = (saved["points"] - [0.1, -0.05, 0.02]) @ Rotation.from_euler(
            "xyz",
            [0.2, 0.3, 0.4],
        ).as_matrix()
        np.testing.assert_allclose(np.max(np.abs(xyz) / [0.01, 0.03, 0.06], axis=1), 1, atol=1e-6)
    monkeypatch.setattr(module, "HAND_TEMPLATE_DIR", tmp_path)
    fk = FakeHandFK()
    fk.model.getFrameId = lambda name: 5
    module.HandPointCloud(fk, 128)
