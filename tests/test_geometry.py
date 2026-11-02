"""确定性几何回归；Brown 误差只代表合成样例。"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from dexmani_real.geometry.camera import (
    CameraIntrinsics,
    camera_parameters_cv,
    deproject_depth,
)
from dexmani_real.geometry.kinematics import ArmFK, HandFK, eef_poses
from dexmani_real.geometry.pose import encode_eef_pose
from dexmani_real.perception.hand_cloud import hand_geometry
from dexmani_real.perception.pointcloud import build_raw_point_cloud, sample_indices


def test_fixed_fk_tip_and_eef_values():
    arm, hand = ArmFK(), HandFK()
    poses = eef_poses(np.zeros((1, 7)), arm_fk=arm)
    np.testing.assert_allclose(poses[0, :3, 3], [.208256, -.00169232616, .0836639875], atol=1e-10)
    features = hand_geometry(poses, np.zeros((1, 12)), arm_fk=arm, hand_fk=hand)
    tips_local = hand.forward(np.zeros(12), hand.fingertip_frame_ids)[:, :3, 3]
    np.testing.assert_allclose(tips_local, [
        [.02211149, .15683926, .02872], [.0065, .0265, .20574829],
        [.0065, .004, .206248], [.0065, -.016, .20324829], [.0065, -.036, .20024829],
    ], atol=1e-8)
    base_from_hand = poses[0] @ arm.eef_from_handbase
    expected = tips_local @ base_from_hand[:3, :3].T + base_from_hand[:3, 3]
    np.testing.assert_allclose(features["fingertip_xyz"][0], expected, atol=1e-8)
    encoded = encode_eef_pose(poses)
    assert encoded.shape == (1, 9)
    np.testing.assert_array_equal(encoded[0, :3], poses[0, :3, 3])
    # FK 结果不持有会被下一次调用覆写的 placement。
    saved = poses.copy()
    arm.forward(np.full(7, .1))
    np.testing.assert_array_equal(poses, saved)


def test_same_packet_rgb_depth_units_and_extrinsics():
    intrinsics = CameraIntrinsics(2, 2, 2., 2., 0., 0., "none", (0.,) * 5)
    rgb = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], np.uint8)
    depth = np.array([[500, 0], [1000, 1500]], np.uint16)
    transform = np.eye(4)
    transform[:3, 3] = [.1, .2, .3]
    cloud = build_raw_point_cloud(depth_raw=depth, color=rgb, depth_scale_m=.001,
                                  intrinsics=intrinsics, T_xarm_base_from_color=transform)
    assert cloud.dtype == np.float32
    np.testing.assert_allclose(cloud[:, :3], [[.1, .2, .8], [.1, .7, 1.3], [.85, .95, 1.8]], atol=1e-7)
    np.testing.assert_array_equal(cloud[:, 3:], [[1, 0, 0], [0, 0, 1], [1, 1, 1]])
    extrinsic = camera_parameters_cv(intrinsics, transform)["camera_extrinsic"].reshape(3, 4)
    np.testing.assert_allclose(extrinsic[:, 3], [-.1, -.2, -.3], atol=1e-8)
    np.testing.assert_array_equal(sample_indices(cloud, 5), [0, 1, 2, 0, 1])


def test_brown_deprojection_against_real_policy_projection():
    projection_path = Path(__file__).resolve().parents[2] / "dexmani_policy/dexmani_policy/agents/obs_encoder/pointcloud/projection.py"
    if not projection_path.is_file():
        pytest.skip("配套 Policy 未提供，无法调用其真实投影消费者")
    torch = pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("policy_projection", projection_path)
    projection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(projection)
    width, height = 640, 480
    coefficients = (.1433306634426117, -.4740954637527466,
                    .0006410915520973504, -.0002661988546606153, .4223690330982208)
    valid = np.zeros((height, width), bool)
    valid[[0, 0, 240, 479, 479], [0, 639, 320, 0, 639]] = True
    errors = []
    for values in ((0.,) * 5, coefficients):
        intrinsics = CameraIntrinsics(width, height, 600.1278686523438, 599.9641723632812,
                                      327.4478454589844, 240.62841796875,
                                      "distortion.brown_conrady", values)
        points, rows, columns = deproject_depth(np.ones((height, width), np.float32), valid, intrinsics)
        result = projection.project_points(
            torch.from_numpy(points[None]), torch.from_numpy(intrinsics.matrix()[None]),
            torch.eye(4)[None, :3], torch.eye(3)[None], (height, width),
        )
        pixels = (result.grid_xy[0].numpy() + 1) * np.array([width, height]) / 2 - .5
        errors.append(np.linalg.norm(pixels - np.stack((columns, rows), axis=1), axis=1))
    assert errors[0].max() < .0001
    np.testing.assert_allclose(errors[1], [2.662721, 2.556681, .0002242, 3.304423, 3.192311], atol=.0001)
