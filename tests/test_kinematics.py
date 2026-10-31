"""内部 SE(3) 数组归属、组合 URDF 数值与外部表示边界。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pinocchio as pin

from dexmani_real.config.real import RealConfig
from dexmani_real.deployment.observation import (
    ObservationBuilder, PolicyObservationRow, policy_dependencies, policy_needs,
)
from dexmani_real.geometry.kinematics import ArmFK, HandFK, eef_poses
from dexmani_real.geometry.pose import encode_eef_pose, rot6d_to_rotmat
from dexmani_real.perception.hand_cloud import HAND_TEMPLATE_DIR, HandPointCloud, hand_geometry
from dexmani_real.robot.model import (
    XARM7_JOINT_NAMES, XARM7_XHAND_RIGHT_URDF_PATH,
    XHAND_FINGERTIP_LINK_NAMES, XHAND_SDK_JOINT_NAMES,
)


class KinematicsTests(unittest.TestCase):
    def test_fk_results_own_their_arrays(self):
        arm, hand = ArmFK(), HandFK()
        arm_qpos = np.array([.1, -.2, .3, .8, -.4, .7, -.1])
        hand_qpos = np.array([.5, .2, .4, .05, .4, .6, .5, .4, .3, .8, .4, .7])
        arm_pose = arm.forward(arm_qpos)
        hand_poses = hand.forward(hand_qpos, hand.fingertip_frame_ids)
        first_arm, first_hand = arm_pose.copy(), hand_poses.copy()
        arm.forward(arm_qpos + .02)
        hand.forward(hand_qpos + .02, hand.fingertip_frame_ids)
        np.testing.assert_array_equal(arm_pose, first_arm)
        np.testing.assert_array_equal(hand_poses, first_hand)
        self.assertEqual(arm_pose.shape, (4, 4))
        self.assertEqual(arm_pose.dtype, np.float64)
        history = eef_poses(np.stack([arm_qpos, arm_qpos + .02]), arm_fk=arm)
        self.assertEqual(history.shape, (2, 4, 4))
        snapshot = history.copy()
        arm.forward(arm_qpos + .04)
        np.testing.assert_array_equal(history, snapshot)

    def test_external_eef9_keeps_column_order(self):
        pose = np.array([[0., -1., 0., .4], [1., 0., 0., -.2],
                         [0., 0., 1., .3], [0., 0., 0., 1.]])
        expected = np.array([.4, -.2, .3, 0., 1., 0., -1., 0., 0.])
        np.testing.assert_array_equal(encode_eef_pose(pose), expected)
        np.testing.assert_array_equal(encode_eef_pose(np.stack([pose, pose])),
                                      np.stack([expected, expected]))
        for invalid in (np.zeros(6), [1., 0., 0., 2., 0., 0.],
                        [1., 0., 0., 0., float("nan"), 0.]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                rot6d_to_rotmat(invalid)

    def test_hand_geometry_matches_independent_full_urdf(self):
        arm, hand = ArmFK(), HandFK()
        cloud = HandPointCloud(hand, 1024)
        arm_qpos = np.array([[.1, -.2, .3, .8, -.4, .7, -.1],
                             [.2, -.1, .2, 1., -.3, .5, .1]])
        hand_qpos = np.array([[.5, .2, .4, .05, .4, .6, .5, .4, .3, .8, .4, .7],
                              [.6, .3, .5, -.05, .5, .7, .6, .5, .4, .7, .5, .6]])
        result = hand_geometry(eef_poses(arm_qpos, arm_fk=arm), hand_qpos,
                               arm_fk=arm, hand_fk=hand, hand_cloud=cloud)
        model = pin.buildModelFromUrdf(str(XARM7_XHAND_RIGHT_URDF_PATH))
        data = model.createData()
        with np.load(HAND_TEMPLATE_DIR / "hand_1024.npz") as template:
            names = template["part_names"].tolist()
        for index, (arm_row, hand_row) in enumerate(zip(arm_qpos, hand_qpos)):
            configuration = np.empty(model.nq)
            for name, value in zip((*XARM7_JOINT_NAMES, *XHAND_SDK_JOINT_NAMES),
                                   np.concatenate((arm_row, hand_row))):
                configuration[model.joints[model.getJointId(name)].idx_q] = value
            pin.forwardKinematics(model, data, configuration)
            pin.updateFramePlacements(model, data)
            expected_tips = np.stack([
                data.oMf[model.getFrameId(name)].translation for name in XHAND_FINGERTIP_LINK_NAMES
            ]).astype(np.float32)
            expected_cloud = cloud.local_points.copy()
            for name, part in zip(names, cloud.part_slices):
                transform = data.oMf[model.getFrameId(name)].homogeneous
                expected_cloud[part, :3] = (
                    cloud.local_points[part, :3] @ transform[:3, :3].T + transform[:3, 3]
                )
            np.testing.assert_allclose(result["fingertip_xyz"][index], expected_tips, atol=1e-7, rtol=0)
            np.testing.assert_allclose(result["hand_pointcloud"][index], expected_cloud, atol=1e-7, rtol=0)

    def test_policy_boundary_keeps_eef9_and_reuses_window_features(self):
        fields = ("eef_pose", "fingertip_xyz", "hand_pointcloud")
        dependencies = policy_dependencies(fields)
        builder = ObservationBuilder(
            SimpleNamespace(observation_fields=fields, fingertip_link_names=XHAND_FINGERTIP_LINK_NAMES),
            RealConfig(), dependencies=dependencies, needs=policy_needs(dependencies),
        )
        rows = [PolicyObservationRow(dict(arm_qpos=np.array([.1, -.2, .3, .8, -.4, .7, -.1]) + i * .01,
                                         hand_qpos=np.full(12, .4)), i + 1) for i in range(3)]
        with patch.object(builder.arm_fk, "forward", wraps=builder.arm_fk.forward) as forward:
            first = builder.build(rows[:2])
            second = builder.build(rows[1:])
            self.assertEqual(forward.call_count, 3)
        self.assertEqual(first["eef_pose"].shape, (2, 9))
        self.assertEqual(first["fingertip_xyz"].shape, (2, 5, 3))
        self.assertEqual(first["hand_pointcloud"].shape, (2, 1024, 6))
        for name in fields:
            self.assertEqual(first[name].dtype, np.float32)
            np.testing.assert_array_equal(first[name][1], second[name][0])


if __name__ == "__main__":
    unittest.main()
