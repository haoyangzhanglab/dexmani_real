import unittest
from dataclasses import replace

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from dexmani_real.geometry.camera import (
    CameraIntrinsics,
    camera_parameters_cv,
    deproject_depth,
    depth_rays,
    undistorted_depth_coordinates,
)
from dexmani_real.geometry.kinematics import ArmFK, HandFK, eef_pose_sequence
from dexmani_real.geometry.plane import fit_table_plane
from dexmani_real.geometry.pose import (
    Pose,
    compute_pose_error,
    quat_multiply,
    quat_wxyz_to_rotmat,
    rot6d_to_quat_wxyz,
    rot6d_to_rotmat,
    validate_rigid_transform,
    validate_unit_quaternion_wxyz,
    xyzw_to_wxyz,
)
from dexmani_real.perception.config import HandCloudConfig
from dexmani_real.perception.hand import HandSurface, hand_geometry_sequence
from dexmani_real.robot.model import XHAND_RIGHT_URDF_PATH, XHAND_SDK_JOINT_NAMES


class PoseTests(unittest.TestCase):
    def test_rotation_conventions_and_quaternion_sign(self):
        first, second = Rotation.from_euler("xyz", [[0.2, -0.7, 1.1], [-0.4, 0.5, 0.8]])
        q1, q2 = (xyzw_to_wxyz(r.as_quat()) for r in (first, second))
        np.testing.assert_allclose(
            quat_wxyz_to_rotmat(quat_multiply(q1, q2)), (first * second).as_matrix()
        )
        matrix = first.as_matrix()
        rot6d = np.concatenate((matrix[:, 0], matrix[:, 1]))
        np.testing.assert_allclose(rot6d_to_rotmat(rot6d), matrix, atol=1e-14)
        np.testing.assert_allclose(
            quat_wxyz_to_rotmat(rot6d_to_quat_wxyz(rot6d)), matrix, atol=1e-14
        )
        position, angle = compute_pose_error(Pose([1, 0, 0], q1), Pose([0, 0, 0], -q1))
        self.assertEqual(position, 1)
        self.assertAlmostEqual(angle, 0, places=7)

    def test_invalid_rotations_remain_visible(self):
        for basis in (np.zeros(6), [1, 0, 0, 2, 0, 0], [1, 0, 0, 0, np.nan, 0]):
            with self.subTest(basis=basis), self.assertRaises(ValueError):
                rot6d_to_rotmat(basis)
        with self.assertRaises(ValueError):
            Pose([0, 0, 0], np.zeros(4))
        with self.assertRaises(ValueError):
            validate_unit_quaternion_wxyz([2, 0, 0, 0])
        reflection = np.diag([-1, 1, 1, 1])
        with self.assertRaises(ValueError):
            validate_rigid_transform(reflection)
        pose = Pose.identity()
        pose.q[0] = np.nan
        self.assertFalse(np.isfinite(compute_pose_error(Pose.identity(), pose)).all())


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.camera = CameraIntrinsics(5, 3, 10.0, 12.0, 2.0, 1.0, "none", (0, 0, 0, 0, 0))

    def test_color_grid_and_inverse_extrinsics(self):
        depth = np.full((3, 5), 0.75, dtype=np.float32)
        points, rows, columns = deproject_depth(depth, depth > 0, self.camera)
        expected = np.column_stack(((columns - 2) / 10, (rows - 1) / 12, np.ones(15))) * 0.75
        np.testing.assert_allclose(points, expected, atol=1e-7)
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("xyz", [0.4, -0.2, 0.7]).as_matrix()
        transform[:3, 3] = [0.3, -0.1, 0.5]
        inverse = camera_parameters_cv(self.camera, transform)["camera_extrinsic"].reshape(3, 4)
        np.testing.assert_allclose(inverse, np.linalg.inv(transform)[:3], atol=1e-7)
        self.assertIs(depth_rays(self.camera), depth_rays(self.camera))

    def test_brown_conrady_inverts_forward_distortion(self):
        camera = replace(
            self.camera,
            distortion_model="brown_conrady",
            distortion_coeffs=(0.1, -0.03, 0.002, -0.001, 0.01),
        )
        x, y = np.array([-0.3, 0, 0.4]), np.array([0.2, 0, -0.35])
        k1, k2, p1, p2, k3 = camera.distortion_coeffs
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        actual = undistorted_depth_coordinates(
            yd * camera.fy + camera.ppy, xd * camera.fx + camera.ppx, camera
        )
        np.testing.assert_allclose(actual, [x, y], atol=1e-7)
        with self.assertRaises(ValueError):
            depth_rays(replace(camera, distortion_model="ftheta"))


class PlaneTests(unittest.TestCase):
    def test_tilted_table_with_outliers(self):
        rng = np.random.default_rng(12)
        xy = rng.uniform(-0.4, 0.4, (2500, 2))
        z = 0.1 + 0.08 * xy[:, 0] - 0.04 * xy[:, 1] + rng.normal(0, 0.0005, len(xy))
        points = np.vstack((np.column_stack((xy, z)), rng.uniform(-0.4, 0.4, (800, 3))))
        fit = fit_table_plane(points)
        expected = np.array([-0.08, 0.04, 1, -0.1]) / np.linalg.norm([-0.08, 0.04, 1])
        np.testing.assert_allclose(fit.plane_abcd, expected, atol=0.0004)
        self.assertGreater(fit.inlier_ratio, 0.7)
        self.assertLess(fit.rms_residual_m, 0.001)
        self.assertEqual(fit, fit_table_plane(points))

    def test_insufficient_degenerate_and_vertical_points(self):
        line = np.column_stack((np.linspace(0, 1, 1200), np.zeros((1200, 2))))
        rng = np.random.default_rng(0)
        wall = np.column_stack((np.zeros(2000), rng.uniform(-1, 1, (2000, 2))))
        for points in (line[:20], line, wall):
            with self.subTest(size=len(points)), self.assertRaises(ValueError):
                fit_table_plane(points)


class KinematicsTests(unittest.TestCase):
    def test_joint_name_mapping_and_hand_mount(self):
        hand_fk = HandFK()
        model = pin.buildModelFromUrdf(str(XHAND_RIGHT_URDF_PATH))
        data = model.createData()
        q = np.linspace(0.05, 0.6, 12)
        reference_q = np.zeros(model.nq)
        for name, value in zip(XHAND_SDK_JOINT_NAMES, q):
            reference_q[model.joints[model.getJointId(name)].idx_q] = value
        pin.framesForwardKinematics(model, data, reference_q)
        reference = np.stack(
            [
                data.oMf[model.getFrameId(hand_fk.model.frames[fid].name)].homogeneous
                for fid in hand_fk.fingertip_frame_ids
            ]
        )
        actual = hand_fk.forward(q, hand_fk.fingertip_frame_ids)
        np.testing.assert_allclose(actual, reference, atol=1e-14)
        hand_fk.forward(np.zeros(12), hand_fk.fingertip_frame_ids)
        np.testing.assert_array_equal(actual, reference)

        rotation = Rotation.from_euler("xyz", [0.2, -0.3, 0.7]).as_matrix()
        mount_rotation = Rotation.from_euler("xyz", [-0.1, 0.2, 0.3]).as_matrix()
        position, mount = np.array([0.3, 0.1, 0.4]), np.array([-0.015, 0.01, 0.02])
        pose = np.r_[position, rotation[:, 0], rotation[:, 1]][None]
        tips = hand_geometry_sequence(
            pose,
            q[None],
            hand_fk=hand_fk,
            handbase_position_eef_m=mount,
            handbase_rotation_eef=mount_rotation,
        )["fingertip_xyz"][0]
        expected = (reference[:, :3, 3] @ mount_rotation.T + mount) @ rotation.T + position
        np.testing.assert_allclose(tips, expected, atol=3e-8)

    def test_arm_batch_and_surface_samples_own_their_results(self):
        arm_fk = ArmFK()
        q = np.array([[0, -0.5, 0.2, 1, 0, 0.5, 0], [0.1, -0.2, 0.3, 0.5, 0.1, 0.8, -0.3]])
        poses = eef_pose_sequence(q, arm_fk=arm_fk)
        for index, joints in enumerate(q):
            data = arm_fk.model.createData()
            pin.framesForwardKinematics(arm_fk.model, data, joints)
            reference = data.oMf[arm_fk.eef_frame_id]
            np.testing.assert_allclose(poses[index, :3], reference.translation)
            np.testing.assert_allclose(
                rot6d_to_rotmat(poses[index, 3:]), reference.rotation, atol=1e-14
            )

        hand_fk = HandFK()
        surface = HandSurface(HandCloudConfig(), hand_fk)
        duplicate = HandSurface(HandCloudConfig(), hand_fk)
        np.testing.assert_array_equal(surface.points, duplicate.points)
        other = HandSurface(HandCloudConfig(seed=1), hand_fk)
        self.assertFalse(np.array_equal(surface.points, other.points))
        placements = hand_fk.forward(np.zeros(12), surface.frame_ids)
        cloud = surface.transform(placements, np.eye(4))
        for i, link_id in enumerate(surface.link_ids):
            expected = placements[link_id] @ np.r_[surface.points[i], 1]
            np.testing.assert_allclose(cloud[i, :3], expected[:3], atol=2e-8)
        np.testing.assert_array_equal(cloud[:, 3:], surface.colors)
        self.assertEqual(cloud.shape, (1024, 6))
        self.assertEqual(cloud.dtype, np.float32)
        self.assertTrue(((cloud[:, 3:] >= 0) & (cloud[:, 3:] <= 1)).all())


if __name__ == "__main__":
    unittest.main()
