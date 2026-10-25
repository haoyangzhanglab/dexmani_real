"""固定输入的表示、真实 FK 后端和共享点云数值检查；不访问设备。"""

import multiprocessing as mp
import unittest
from unittest.mock import patch

import numpy as np

from dexmani_real.control.online_ik import XArm7Kinematics
from dexmani_real.geometry.camera import (
    CameraIntrinsics,
    CameraSnapshot,
    camera_parameters_cv,
)
from dexmani_real.geometry.kinematics.arm_fk import (
    ArmFK,
    compute_eef_pose_history_xarm_base,
)
from dexmani_real.geometry.kinematics.hand_fk import SDK_TO_URDF_IDX, HandKinematics
from dexmani_real.geometry.plane import validate_table_plane
from dexmani_real.geometry.pose import (
    Pose,
    quat_wxyz_to_rotmat,
    rot6d_to_rotmat,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from dexmani_real.geometry.transforms import validate_rigid_transform
from dexmani_real.ipc.channels import SensorChannels
from dexmani_real.ipc.schema import CAMERA_FRAME_HEADER_DTYPE
from dexmani_real.perception.hand_config import HandPointCloudConfig
from dexmani_real.perception.hand_geometry import (
    compute_hand_geometry_history_xarm_base,
)
from dexmani_real.perception.hand_surface import HandPointCloudTemplate
from dexmani_real.perception.pointcloud.config import (
    PointCloudConfig,
    resolve_saved_pointcloud_recipe,
)
from dexmani_real.perception.pointcloud.processing import build_point_cloud
from dexmani_real.perception.pointcloud.worker import run_pointcloud_worker
from dexmani_real.robot.model import (
    XHAND_RIGHT_URDF_PATH,
    XHAND_SDK_JOINT_NAMES,
    XHAND_URDF_JOINT_NAMES,
)
from dexmani_real.robot.xarm7.config import ArmParams
from dexmani_real.robot.xhand.config import HandParams
from dexmani_real.sensor.camera.config import CameraParams


def cloud_inputs():
    transform = np.eye(4)
    transform[0, 3] = 0.4
    return dict(
        depth_raw=np.full((64, 64), 600, dtype=np.uint16),
        color=np.broadcast_to(np.array([100, 150, 200], dtype=np.uint8), (64, 64, 3)).copy(),
        depth_scale_m=0.001,
        intrinsics=CameraIntrinsics(64, 64, 400, 400, 31.5, 31.5, 'none', (0, 0, 0, 0, 0)),
        T_xarm_base_from_color=transform,
        table_plane_abcd=(0, 0, 1, 0),
        config=PointCloudConfig(num_points=128),
    )


def numerical_samples():
    """迁移前后在同一环境精确比较的小型样本，不调用设备或加载策略权重。"""
    arm_home = np.asarray(ArmParams().home_qpos)
    arm = np.stack((arm_home, arm_home + [0.01, -0.02, 0.01, 0.02, -0.01, 0.01, -0.02]))
    hand_home = np.deg2rad(HandParams().home_qpos_deg)
    hand = np.stack((hand_home, hand_home + np.linspace(0, 0.01, 12)))
    poses = compute_eef_pose_history_xarm_base(arm, arm_fk=ArmFK())
    hand_fk = HandKinematics(str(XHAND_RIGHT_URDF_PATH))
    template = HandPointCloudTemplate(HandPointCloudConfig(seed=0), hand_fk)
    features = compute_hand_geometry_history_xarm_base(
        poses, hand, hand_fk=hand_fk, handbase_position_eef_m=(0.01, 0.02, 0.03),
        handbase_rotation_eef=np.eye(3), hand_cloud=template,
    )
    kin = XArm7Kinematics()
    mplib_poses = [kin.compute_eef_pose_base(q) for q in arm]
    return dict(
        arm_poses=poses, hand_template=template.points,
        hand_placements=hand_fk.compute_frame_placements(hand_home, hand_fk.fingertip_frame_ids),
        mplib_poses=np.stack([np.concatenate((pose.p, pose.q)) for pose in mplib_poses]),
        mplib_jacobians=np.stack([kin.compute_eef_jacobian_base(q) for q in arm]),
        pointcloud=build_point_cloud(**cloud_inputs()), **features,
    )


class RepresentationTests(unittest.TestCase):
    def test_wxyz_and_column_ordered_rot6d(self):
        q = np.array([0.5, 0.5, 0.5, 0.5])
        np.testing.assert_array_equal(xyzw_to_wxyz(wxyz_to_xyzw(q)), q)
        rotation = quat_wxyz_to_rotmat(q)
        expected = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        np.testing.assert_allclose(rotation, expected, atol=1e-14, rtol=0)
        np.testing.assert_allclose(
            rot6d_to_rotmat(np.concatenate((rotation[:, 0], rotation[:, 1]))),
            expected, atol=1e-14, rtol=0,
        )

    def test_invalid_geometry_is_rejected(self):
        for q in (np.zeros(4), [np.nan, 0, 0, 0]):
            with self.subTest(q=q), self.assertRaises(ValueError):
                Pose([0, 0, 0], q)
        for value in (np.zeros(6), [1, 0, 0, 2, 0, 0], [np.nan] * 6):
            with self.subTest(value=value), self.assertRaises(ValueError):
                rot6d_to_rotmat(value)
        invalid = np.eye(4)
        invalid[3, 0] = 1
        with self.assertRaises(ValueError):
            validate_rigid_transform(invalid)
        for plane in ((0, 0, 0, 1), (0, 0, -1, 0), (0, 0, 1, np.nan)):
            with self.subTest(plane=plane), self.assertRaises(ValueError):
                validate_table_plane(plane)
        self.assertEqual(validate_table_plane((0, 0, 2, -0.4)), (0, 0, 2, -0.4))

    def test_camera_direction_dtype_and_owned_snapshot(self):
        inputs = cloud_inputs()
        transform = inputs['T_xarm_base_from_color']
        snapshot = CameraSnapshot('offline', inputs['intrinsics'], 0.001, transform)
        self.assertFalse(snapshot.T_xarm_base_from_color.flags.writeable)
        transform[0, 3] = 0.7
        self.assertEqual(snapshot.T_xarm_base_from_color[0, 3], 0.4)
        matrices = camera_parameters_cv(snapshot.color, snapshot.T_xarm_base_from_color)
        self.assertEqual(matrices['camera_intrinsic'].shape, (9,))
        self.assertEqual(matrices['camera_extrinsic'].dtype, np.float32)
        np.testing.assert_array_equal(
            matrices['camera_extrinsic'].reshape(3, 4),
            np.array([[1, 0, 0, -0.4], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32),
        )


class PerceptionTests(unittest.TestCase):
    def test_worker_publishes_source_clocks_and_clears_readiness(self):
        inputs = cloud_inputs()
        camera = CameraSnapshot('offline', inputs['intrinsics'], inputs['depth_scale_m'],
                                inputs['T_xarm_base_from_color'])
        channels = SensorChannels(
            mp.get_context('spawn'), camera=CameraParams(width=64, height=64),
            pointcloud_num_points=inputs['config'].num_points,
        )
        header = np.zeros((), dtype=CAMERA_FRAME_HEADER_DTYPE)
        header['timestamp_ns'], header['received_ns'] = 100, 200
        sequence = channels.camera_ring.write(
            header=header, rgb=inputs['color'], depth=inputs['depth_raw'],
        )
        _, published_ns, _ = channels.camera_ring.read_latest()
        set_ready = channels.pointcloud_ready.set

        def stop_after_publication():
            set_ready()
            channels.is_running.value = False

        with (
            patch('dexmani_real.perception.pointcloud.worker.configure_logging'),
            patch('dexmani_real.perception.pointcloud.worker.time.sleep',
                  side_effect=AssertionError('worker did not publish the available frame')),
            patch.object(channels.pointcloud_ready, 'set', side_effect=stop_after_publication) as ready,
        ):
            run_pointcloud_worker(channels, inputs['config'], camera, inputs['table_plane_abcd'])
        ready.assert_called_once_with()
        self.assertFalse(channels.pointcloud_ready.is_set())
        data, _, cloud_sequence = channels.pointcloud_ring.read_latest()
        self.assertEqual(cloud_sequence, 1)
        self.assertEqual(data['source_camera_sequence'][0], sequence)
        self.assertEqual(data['timestamp_ns'][0], 100)
        self.assertEqual(data['camera_received_ns'][0], 200)
        self.assertEqual(data['camera_published_ns'][0], published_ns)
        self.assertGreaterEqual(data['processing_started_ns'][0], published_ns)
        np.testing.assert_array_equal(data['point_cloud'][0], build_point_cloud(**inputs))

    def test_empty_valid_and_repeated_depth(self):
        kwargs = cloud_inputs()
        cloud = build_point_cloud(**kwargs)
        self.assertEqual(cloud.shape, (128, 6))
        self.assertEqual(cloud.dtype, np.float32)
        np.testing.assert_array_equal(cloud, build_point_cloud(**kwargs))
        np.testing.assert_allclose(cloud[:, 2], 0.6, atol=1e-7, rtol=0)
        self.assertTrue(np.all((cloud[:, 3:] >= 0) & (cloud[:, 3:] <= 1)))
        self.assertTrue(np.all((cloud[:, 0] > 0.35) & (cloud[:, 0] < 0.45)))
        kwargs['depth_raw'] = np.zeros((64, 64), dtype=np.uint16)
        self.assertIsNone(build_point_cloud(**kwargs))

    def test_saved_recipe_is_not_replaced_with_defaults(self):
        from dataclasses import asdict

        recipe = dict(asdict(PointCloudConfig(num_points=37)), sampling_method='quickfps')
        self.assertEqual(resolve_saved_pointcloud_recipe(recipe).num_points, 37)
        del recipe['edge_jump_m']
        with self.assertRaisesRegex(ValueError, 'missing'):
            resolve_saved_pointcloud_recipe(recipe)

    def test_real_backends_and_hand_geometry(self):
        samples = numerical_samples()
        self.assertEqual(samples['arm_poses'].shape, (2, 9))
        self.assertEqual(samples['mplib_jacobians'].shape, (2, 6, 7))
        self.assertEqual(samples['fingertip_xyz'].shape, (2, 5, 3))
        self.assertEqual(samples['hand_pointcloud'].shape, (2, 1024, 6))
        self.assertEqual(samples['hand_pointcloud'].dtype, np.float32)
        self.assertTrue(all(np.isfinite(value).all() for value in samples.values()))
        self.assertEqual(tuple(XHAND_SDK_JOINT_NAMES[i] for i in SDK_TO_URDF_IDX), XHAND_URDF_JOINT_NAMES)
        # 两个后端仍独立；同一 URDF 位置的浮点误差以米计，仅此跨后端断言使用容差。
        np.testing.assert_allclose(samples['arm_poses'][:, :3], samples['mplib_poses'][:, :3], atol=1e-12, rtol=0)

    def test_policy_and_canonical_produce_identical_representations(self):
        from dexmani_real.data.canonical.processing import iter_canonical_blocks
        from dexmani_real.data.canonical.schema import ProcessingConfig
        from dexmani_real.data.raw.camera_geometry import RawEpisodeCameraModel
        from dexmani_real.data.raw.schema import DATASET_SPECS
        from dexmani_real.deployment.policy_observation import (
            PolicyObservationRow,
            build_observation_kinematics,
            build_policy_observation,
            policy_dependencies,
        )

        inputs = cloud_inputs()
        config = ProcessingConfig(pointcloud=inputs['config'], table_plane_abcd=(0, 0, 1, 0))
        raw = {name: np.zeros((2, *shape), dtype=dtype) for name, (shape, dtype) in DATASET_SPECS.items()}
        raw['arm_qpos'][:] = ArmParams().home_qpos
        raw['hand_qpos'][:] = np.deg2rad(HandParams().home_qpos_deg)
        raw['arm_qpos'][1, 0] += 0.01
        raw['action_arm_joint_target'][:] = raw['arm_qpos'] + 0.02
        raw['action_hand_joint_target'][:] = raw['hand_qpos'] + 0.01
        raw['depth'] = np.stack([inputs['depth_raw']] * 2)

        class AdmittedReader(dict):
            # 本替身仅提供已准入的数组，不声称验证了 HDF5/视频文件读取。
            num_frames = 2

            def iter_rgb_frames(self):
                return iter([inputs['color']] * 2)

        hand_fk = HandKinematics(str(XHAND_RIGHT_URDF_PATH))
        parameters = camera_parameters_cv(inputs['intrinsics'], inputs['T_xarm_base_from_color'])
        offline = next(iter_canonical_blocks(
            AdmittedReader(raw), config, arm_fk=ArmFK(), hand_fk=hand_fk,
            hand_cloud=HandPointCloudTemplate(config.hand_pointcloud, hand_fk), chunk_frames=2,
            camera=RawEpisodeCameraModel(inputs['intrinsics'], inputs['depth_scale_m']),
            base_from_color=inputs['T_xarm_base_from_color'], camera_parameters=parameters,
            handbase_rotation_eef=quat_wxyz_to_rotmat(config.handbase_quat_eef_wxyz),
        ))
        fields = tuple(name for name in offline if name not in ('action', 'action_ee'))
        dependencies = policy_dependencies(fields)
        kinematics = build_observation_kinematics(
            dependencies, fingertip_link_names=config.fingertip_link_names,
            hand=HandParams(), hand_pointcloud=config.hand_pointcloud,
        )
        self.assertIsNot(kinematics.hand_fk, hand_fk)
        cloud = build_point_cloud(**inputs)
        rows = [PolicyObservationRow(i + 100, (80 + i, 90 + i), dict(
            arm_qpos=raw['arm_qpos'][i], hand_qpos=raw['hand_qpos'][i],
            finger_tactile_force=raw['hand_contact'][i], taxel_tactile_force=raw['hand_tactile_force'][i],
            point_cloud=cloud, rgb=inputs['color'], depth=inputs['depth_raw'],
        )) for i in range(2)]
        online = build_policy_observation(rows, fields, dependencies, kinematics=kinematics, camera_parameters=parameters)
        repeated = build_policy_observation(rows, fields, dependencies, kinematics=kinematics, camera_parameters=parameters)
        for name in fields:
            with self.subTest(field=name):
                self.assertEqual(online[name].dtype, offline[name].dtype)
                np.testing.assert_array_equal(online[name], offline[name])
                np.testing.assert_array_equal(repeated[name], online[name])
        self.assertEqual(rows[0].source_times, (80, 90))
        self.assertFalse(np.array_equal(offline['joint_state'], offline['action']))

