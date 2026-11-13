"""当前 PolicyInfo 与真机观测配置的契约检查，不连接设备。"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from dexmani_policy.deployment.metadata import PolicyInfo
from dexmani_real.config.deployment import ExecutionConfig
from dexmani_real.config.perception import PointCloudConfig
from dexmani_real.config.real import RealConfig
from dexmani_real.deployment.observation import ObservationBuilder
from dexmani_real.deployment.preflight import resolve_deployment_config
from dexmani_real.geometry.camera import CameraIntrinsics
from dexmani_real.robot.model import XHAND_FINGERTIP_LINK_NAMES
from dexmani_real.runtime.observation import RAW_RECORDING_NEEDS


@pytest.fixture
def info():
    return PolicyInfo(
        checkpoint_path='test.pt', weights='raw', inference_steps=10, task_name='test',
        observation_fields=('point_cloud', 'joint_state'), n_obs_steps=2,
        n_action_steps=8, future_steps=15, control_action_dim=19, action_mode='joint',
        rgb_preprocessing={}, supports_rtc=True, requires_prev_action=False, control_dt_s=.05,
    )


def test_pointcloud_uses_experiment_config(info):
    runtime = RealConfig(pointcloud=PointCloudConfig(num_points=17, remove_table=False))
    resolved = resolve_deployment_config(info, runtime)
    assert resolved.pointcloud is runtime.pointcloud
    assert resolved.execution.control_dt_s == info.control_dt_s
    assert resolved.execution.action_steps == info.n_action_steps
    explicit = replace(runtime, execution=ExecutionConfig(control_dt_s=.025, action_steps=4))
    assert resolve_deployment_config(info, explicit).execution == explicit.execution
    with pytest.raises(ValueError, match='num_points'):
        resolve_deployment_config(info, replace(runtime, pointcloud=PointCloudConfig(num_points=0)))


@pytest.mark.parametrize('dt', [0., float('nan'), float('inf'), 1e-10])
def test_control_period_validation(info, dt):
    with pytest.raises(ValueError, match='control_dt_s'):
        resolve_deployment_config(info, RealConfig(execution=ExecutionConfig(control_dt_s=dt)))


def test_ode_execution_contract(info):
    info = replace(info, requires_prev_action=True)
    for execution in (ExecutionConfig(mode='async'), ExecutionConfig(action_steps=4)):
        with pytest.raises(ValueError, match='ODE requires sync'):
            resolve_deployment_config(info, RealConfig(execution=execution))
    assert resolve_deployment_config(info, RealConfig()).execution.action_steps == 8


def test_fingertips_use_shared_robot_order(info):
    builder = ObservationBuilder(replace(info, observation_fields=('joint_state', 'fingertip_xyz')), RealConfig())
    fk = builder.hand_fk
    assert tuple(fk.model.frames[i].name for i in fk.fingertip_frame_ids) == XHAND_FINGERTIP_LINK_NAMES
    row = SimpleNamespace(arm={'qpos': np.zeros((1, 7))}, hand={'qpos': np.zeros((1, 12))})
    arrays = builder.build([builder.project(row)] * info.n_obs_steps)
    assert arrays['joint_state'].shape == (2, 19)
    assert arrays['fingertip_xyz'].shape == (2, 5, 3)
    assert np.isfinite(arrays['fingertip_xyz']).all()


def test_rgb_cloud_and_camera_parameters(info):
    fields = ('rgb', 'point_cloud', 'camera_intrinsic', 'camera_extrinsic')
    builder = ObservationBuilder(replace(info, observation_fields=fields), RealConfig())
    color = CameraIntrinsics(4, 3, 2., 2., 2., 1., 'none', (0.,) * 5)
    builder.configure_camera(SimpleNamespace(color=color, T_xarm_base_from_color=np.eye(4)))
    row = SimpleNamespace(camera={'rgb': np.zeros((3, 4, 3), np.uint8)}, point_cloud=np.zeros((17, 6), np.float32))
    arrays = builder.build([builder.project(row)] * info.n_obs_steps)
    assert builder.needs.camera_fields == ('rgb',)
    assert builder.needs.point_cloud
    assert arrays['rgb'].shape == (2, 3, 4, 3)
    assert arrays['point_cloud'].shape == (2, 17, 6)
    np.testing.assert_array_equal(arrays['camera_intrinsic'], np.tile(color.matrix().ravel(), (2, 1)))
    np.testing.assert_array_equal(arrays['camera_extrinsic'], np.tile(np.eye(3, 4).ravel(), (2, 1)))


def test_depth_is_recording_input_only(info):
    with pytest.raises(ValueError, match='no live producer.*depth'):
        ObservationBuilder(replace(info, observation_fields=('depth',)), RealConfig())
    assert 'depth' in RAW_RECORDING_NEEDS.camera_fields
