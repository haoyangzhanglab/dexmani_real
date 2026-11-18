"""当前 PolicyInfo 与真机观测配置的契约检查，不连接设备。"""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

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
from dexmani_real.runtime.observation import ObservationNeeds


@pytest.fixture
def info():
    return PolicyInfo(
        checkpoint_path='test.pt', weights='raw', inference_steps=10, task_name='test',
        observation_fields=('point_cloud', 'joint_state'), n_obs_steps=2,
        n_action_steps=8, future_steps=15, control_action_dim=19, action_mode='joint',
        data_processing={}, rgb_preprocessing={}, supports_rtc=True, requires_prev_action=False, control_dt_s=.05,
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


def test_deployment_rejects_mixed_pointcloud_sources(info):
    info = replace(info, observation_fields=(*info.observation_fields, 'object_pointcloud'))
    with pytest.raises(ValueError, match='不支持混合输入'):
        resolve_deployment_config(info, RealConfig())


def test_deployment_requires_hand_feedback(info):
    runtime = RealConfig()
    runtime = replace(runtime, hand=replace(runtime.hand, enabled=False))
    with pytest.raises(ValueError, match='requires hand.enabled=true'):
        resolve_deployment_config(info, runtime)


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
    row = SimpleNamespace(camera={'rgb': np.zeros((3, 4, 3), np.uint8), 'ring_sequence': 1}, point_cloud=np.zeros((17, 6), np.float32))
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


def test_hoi_projection_uses_aligned_joints_without_changing_feedback():
    info = SimpleNamespace(observation_fields=("joint_state", "object_pointcloud"))
    builder = ObservationBuilder(info, RealConfig())
    arm = np.zeros(1, dtype=[("qpos", "f8", (7,))])
    hand = np.zeros(1, dtype=[("qpos", "f8", (12,))])
    row = SimpleNamespace(
        arm=arm,
        hand=hand,
        hoi={"object_pointcloud": np.zeros((1024, 3), np.float32)},
        aligned_joints={"arm_qpos": np.full(7, 0.25), "hand_qpos": np.full(12, 0.5)},
    )
    result = builder.build([builder.project(row)])
    np.testing.assert_allclose(result["joint_state"][0], np.r_[np.full(7, 0.25), np.full(12, 0.5)])
    assert not arm["qpos"].any() and not hand["qpos"].any()


def test_hoi_eef_projection_requires_only_arm_history(info):
    builder = ObservationBuilder(
        replace(info, observation_fields=("eef_pose", "object_pointcloud")), RealConfig()
    )
    assert builder.needs.hoi_joint_devices == ("arm",)
    row = SimpleNamespace(
        hoi={"object_pointcloud": np.zeros((1024, 3), np.float32)},
        aligned_joints={"arm_qpos": np.zeros(7)},
    )
    arrays = builder.build([builder.project(row)])
    assert arrays["eef_pose"].shape == (1, 9)
    assert np.isfinite(arrays["eef_pose"]).all()


@pytest.mark.parametrize("with_hoi", [False, True])
def test_joint_history_is_shared_by_state_and_geometry(info, with_hoi):
    from dexmani_real.geometry.pose import encode_eef_pose

    fields = ("joint_state", "eef_pose", "fingertip_xyz", "finger_tactile_force")
    if with_hoi:
        fields += ("object_pointcloud",)
    builder = ObservationBuilder(replace(info, observation_fields=fields), RealConfig())
    history, expected_state, expected_eef, expected_tips = [], [], [], []
    for step in range(3):
        arm = np.full(7, .02 * step)
        hand = np.full(12, .1 * step)
        # HOI 的历史关节与当前实测值不同，状态和几何必须使用同一来源。
        row = SimpleNamespace(
            arm={"qpos": arm[None].copy()},
            hand={"qpos": hand[None].copy(), "tactile_aggregate": np.full((1, 5, 3), step, np.float32)},
            aligned_joints={"arm_qpos": arm + .03, "hand_qpos": hand + .04},
            hoi={"object_pointcloud": np.full((1024, 3), step, np.float32)},
        )
        history.append(builder.project(row))
        if with_hoi:
            arm, hand = row.aligned_joints["arm_qpos"], row.aligned_joints["hand_qpos"]
        pose = builder.arm_fk.forward(arm)
        tips = builder.hand_fk.forward(hand, builder.hand_fk.fingertip_frame_ids)
        base_from_hand = pose @ builder.arm_fk.eef_from_handbase
        expected_state.append(np.r_[arm, hand])
        expected_eef.append(encode_eef_pose(pose))
        expected_tips.append((base_from_hand @ tips)[:, :3, 3])
        # 历史拥有低维快照，不受之后的反馈或插值缓存更新影响。
        row.arm["qpos"][:] = 99
        row.hand["qpos"][:] = 99
        row.hand["tactile_aggregate"][:] = 99
        for qpos in row.aligned_joints.values():
            qpos[:] = 99
    arrays = builder.build(history)
    assert tuple(arrays) == fields
    for field, expected in (
        ("joint_state", expected_state), ("eef_pose", expected_eef),
        ("fingertip_xyz", expected_tips),
    ):
        assert arrays[field].dtype == np.float32
        np.testing.assert_allclose(arrays[field], np.asarray(expected, np.float32), atol=1e-7)
    np.testing.assert_array_equal(arrays["finger_tactile_force"][:, 0, 0], [0, 1, 2])
    if with_hoi:
        np.testing.assert_array_equal(arrays["object_pointcloud"][:, 0, 0], [0, 1, 2])


def test_rgb_only_rectifies_once_per_source_frame(info, monkeypatch):
    import cv2

    from dexmani_real.geometry.camera import CameraSnapshot

    color = CameraIntrinsics(64, 48, 55., 55., 32., 24., 'brown_conrady', (.2, 0, .001, 0, 0))
    camera = CameraSnapshot(None, color, None, None)
    builder = ObservationBuilder(replace(info, observation_fields=('rgb',)), RealConfig())
    init = Mock(wraps=cv2.initUndistortRectifyMap)
    remap = Mock(wraps=cv2.remap)
    monkeypatch.setattr(cv2, "initUndistortRectifyMap", init)
    monkeypatch.setattr(cv2, "remap", remap)
    builder.configure_camera(camera)
    image = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    before = image.copy()
    row = SimpleNamespace(camera={'rgb': image, 'ring_sequence': 4, 'timestamp_ns': 123})
    first = builder.project(row)
    assert builder.project(row)['rgb'] is first['rgb']
    builder.build([first, first])
    assert (init.call_count, remap.call_count) == (1, 1)
    row.camera['ring_sequence'] = 5
    second = builder.project(row)
    assert second['rgb'] is not first['rgb']
    assert (init.call_count, remap.call_count) == (1, 2)
    builder.configure_camera(camera)
    builder.project(row)
    assert (init.call_count, remap.call_count) == (2, 3)
    np.testing.assert_array_equal(image, before)
    assert row.camera['timestamp_ns'] == 123
    assert builder.camera_parameters == {}


@pytest.mark.parametrize("fault", ["none", "cancel", "failure"])
def test_hoi_warmup_commits_only_successful_uncancelled_result(fault):
    from concurrent.futures import Future
    from threading import Event

    from dexmani_real.deployment.inference import PolicyWorker
    from dexmani_real.deployment.rollout import Rollout
    from dexmani_real.robot.commands import OperationCancelled
    from dexmani_real.runtime.state import SessionState

    state = SessionState(stop_signal=Event(), emergency_signal=Event())
    worker = PolicyWorker(lambda: None, build_observation=lambda rows: rows)
    calls = []

    def warmup(rows, **kwargs):
        calls.append(kwargs)
        worker.future = Future()
        if fault == "failure":
            worker.future.set_exception(ValueError("warmup failed"))
        else:
            worker.future.set_result(None)
        if fault == "cancel":
            state.request_stop()

    worker.warmup = warmup
    runtime = RealConfig(execution=ExecutionConfig(control_dt_s=0.05, action_steps=2, mode="rtc"))
    observations = SimpleNamespace(
        needs=ObservationNeeds(hoi=True), project=lambda row: {}
    )
    robot = SimpleNamespace(
        stop=lambda: None,
        check=lambda: None,
        service_feedback=lambda: None,
        prepare_streaming=lambda *a, **kw: state.stop_signal.clear(),
    )
    sensors = SimpleNamespace(
        channels=SimpleNamespace(hoi_request=Mock(), camera_ring=SimpleNamespace(latest_sequence=1))
    )
    try:
        for index in range(2):
            rollout = Rollout(
                state,
                runtime,
                SimpleNamespace(n_obs_steps=2),
                episode_cancel_epoch=state.cancel_epoch,
                robot=robot,
                observations=observations,
                sensors=sensors,
                action_mapper=None,
                model=worker,
                max_duration_s=1,
                index=index,
                output_dir=None,
                record_raw=False,
                effective_parameters={},
            )
            rollout.start_observation = lambda *a, **kw: object()
            if fault != "none":
                with pytest.raises(OperationCancelled if fault == "cancel" else ValueError):
                    rollout.prepare(lambda: None)
                assert not worker.warmed_up
                break
            rollout.prepare(lambda: None)
            assert worker.warmed_up
        assert len(calls) == 1
        assert "rtc_delay" in calls[0] and "rtc_guidance_cap" in calls[0]
    finally:
        worker.close().result()


def test_processing_records_differences_without_rejecting(info, caplog):
    from dexmani_real.deployment.preflight import collect_data_processing
    from dexmani_real.perception.pointcloud import processing_metadata
    from dexmani_real.utils.table_plane import normalize_table_plane

    runtime = RealConfig()
    caplog.set_level("INFO", logger="dexmani_real.deployment.preflight")
    coefficients = (0.1, 0.2, 1.0, -0.4)
    plane = normalize_table_plane(coefficients)
    saved = {"point_cloud": processing_metadata(runtime.pointcloud, plane)}
    info = replace(info, data_processing=saved)
    assert collect_data_processing(info, runtime, table_plane_abcd=plane) == saved

    scaled_plane = normalize_table_plane(tuple(3 * value for value in coefficients))
    actual = collect_data_processing(info, runtime, table_plane_abcd=scaled_plane)
    assert actual["point_cloud"]["table_plane_abcd"] == list(scaled_plane)
    assert "处理参数差异" not in caplog.text

    changed = replace(runtime, pointcloud=replace(runtime.pointcloud, num_points=17))
    assert collect_data_processing(info, changed, table_plane_abcd=plane) == {
        "point_cloud": processing_metadata(changed.pointcloud, plane),
    }
    assert "处理参数差异 point_cloud.config" in caplog.text

    caplog.clear()
    offset_plane = (*plane[:3], plane[3] - 0.01)
    actual = collect_data_processing(info, runtime, table_plane_abcd=offset_plane)
    assert actual["point_cloud"]["table_plane_abcd"] == list(offset_plane)
    assert "差异 point_cloud.table_plane_abcd" in caplog.text


@pytest.mark.parametrize("fail", [False, True])
def test_rtc_warmup_preserves_rng_and_episode_reset(fail):
    import torch
    from dexmani_policy.deployment.runtime import PolicyRuntime

    from dexmani_real.deployment.inference import PolicyWorker

    model = PolicyRuntime.__new__(PolicyRuntime)
    model.generator = torch.Generator().manual_seed(7)
    calls = []

    def predict(observation, **kwargs):
        calls.append(kwargs)
        value = torch.rand(3, generator=model.generator)
        if fail and len(calls) == 2:
            raise ValueError("rtc failure")
        return value

    model.predict = predict
    original = model.generator.get_state().clone()
    if fail:
        with pytest.raises(ValueError, match="rtc failure"):
            model.warmup({}, rtc_delay=1, rtc_guidance_cap=10.0)
    else:
        model.warmup({}, rtc_delay=1, rtc_guidance_cap=10.0)
    assert len(calls) == 2 and "rtc_prefix" in calls[1]
    assert torch.equal(original, model.generator.get_state())
    worker = PolicyWorker(lambda: model, build_observation=lambda rows: {})
    worker.model = model
    try:
        worker.predict([], reset_episode=True)
        first = worker.future.result()
        worker.poll()
        worker.predict([], reset_episode=True)
        second = worker.future.result()
        worker.poll()
        assert torch.equal(first, second)
    finally:
        worker.close().result()
