"""手部点云 FK 变换的批量实现回归。"""

from types import SimpleNamespace

import numpy as np


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


def test_hand_geometry_batch_transform_preserves_link_segments(tmp_path, monkeypatch):
    from dexmani_real.perception import hand_cloud as module

    rng = np.random.default_rng(42)
    local_points = rng.normal(0, 0.02, (6, 3)).astype(np.float32)
    offsets = np.array([0, 2, 6], dtype=np.int64)
    np.savez(
        tmp_path / "hand_6.npz",
        points=local_points,
        part_names=np.array(["link_a", "link_b"], dtype="U"),
        part_offsets=offsets,
    )
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

    eef_transforms = np.repeat(np.eye(4)[None], 4, axis=0)
    eef_transforms[:, :3, 3] = rng.normal(0, 0.1, (4, 3))
    hand_qpos = rng.uniform(-0.5, 0.5, (4, 12))

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
            placements[:5, :3, 3] @ base_from_hand[:3, :3].T
            + base_from_hand[:3, 3]
        )
        expected_cloud[frame] = local_points
        for link_index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            transform = base_from_hand @ placements[5 + link_index]
            expected_cloud[frame, start:end] = (
                local_points[start:end] @ transform[:3, :3].T + transform[:3, 3]
            )

    np.testing.assert_allclose(actual["fingertip_xyz"], expected_tips, atol=1e-7)
    np.testing.assert_allclose(actual["hand_pointcloud"], expected_cloud, atol=1e-7)
    assert actual["hand_pointcloud"].dtype == np.float32
    np.testing.assert_array_equal(hand_cloud.local_points, local_points)

    frame_ids = [*hand_fk.fingertip_frame_ids, *hand_cloud.frame_ids]
    placements = hand_fk.forward(hand_qpos[0], frame_ids)
    single = hand_cloud.transform(placements[5:], eef_transforms[0] @ arm_from_eef)
    np.testing.assert_allclose(single, expected_cloud[0], atol=1e-7)


def test_hand_cloud_batch_transform_without_fingertips(tmp_path, monkeypatch):
    from dexmani_real.perception import hand_cloud as module

    np.savez(
        tmp_path / "hand_4.npz",
        points=np.arange(12, dtype=np.float32).reshape(4, 3),
        part_names=np.array(["link_a", "link_b"], dtype="U"),
        part_offsets=np.array([0, 1, 4], dtype=np.int64),
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
