"""真实 Ring 与离线反馈验证模型／录制的来源契约，不启动 Worker。"""

import contextlib
import io
import itertools
import sys
import threading
import types
import unittest
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.real import RealConfig
from dexmani_real.deployment.observation import ObservationBuilder
from dexmani_real.deployment.rollout import Rollout
from dexmani_real.deployment.session import PolicySession
from dexmani_real.robot.system import RobotState
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE
from dexmani_real.runtime.observation import (
    Observation,
    ObservationNeeds,
    check_observation,
)
from dexmani_real.runtime.state import SessionState
from dexmani_real.teleop.loop import TeleopLoop


class ObservationContracts(unittest.TestCase):
    def setUp(self):
        runtime = RealConfig()
        self.runtime = replace(runtime, camera=replace(runtime.camera, width=16, height=16),
                               execution=replace(runtime.execution, control_dt_s=0.1))
        self.record = ObservationNeeds(camera_fields=('rgb', 'depth'),
                                       required_tactile=('aggregate', 'dense'))

    def test_union_order_and_immutability(self):
        for bits in itertools.product((False, True), repeat=7):
            needs = ObservationNeeds(
                camera_fields=tuple(n for n, b in zip(('depth', 'rgb'), bits) if b),
                hoi=bits[2], point_cloud=bits[3], vr=bits[4],
                required_tactile=tuple(n for n, b in zip(('dense', 'aggregate'), bits[5:]) if b),
            )
            for other in (ObservationNeeds(), self.record):
                combined = needs.union(other)
                self.assertEqual(combined, other.union(needs))
                self.assertEqual(combined.union(combined), combined)
                self.assertEqual(combined.camera_fields, tuple(n for n in ('rgb', 'depth')
                                 if n in needs.camera_fields or n in other.camera_fields))
                self.assertEqual(combined.required_tactile, tuple(n for n in ('aggregate', 'dense')
                                 if n in needs.required_tactile or n in other.required_tactile))
                self.assertEqual((combined.hoi, combined.point_cloud, combined.vr), bits[2:5])
        with self.assertRaises(FrozenInstanceError):
            self.record.vr = True

    def test_teleop_recording_requirements(self):
        for recording in (False, True):
            loop = TeleopLoop(None, self.runtime, None, task_label='test', sensors=None,
                              controller=None, recording_dir=Path('/tmp') if recording else None,
                              max_duration_s=1, keyboard=None, audio=None)
            expected = ObservationNeeds(vr=True).union(self.record if recording else ObservationNeeds())
            self.assertEqual(loop.capture_needs, expected)

    def test_builder_and_missing_sources(self):
        for fields, expected in (
            (('joint_state',), ObservationNeeds()),
            (('rgb', 'depth', 'finger_tactile_force', 'taxel_tactile_force'), self.record),
            (('point_cloud', 'rgb'), ObservationNeeds(point_cloud=True, camera_fields=('rgb',))),
            (('object_pointcloud',), ObservationNeeds(hoi=True)),
            (('camera_intrinsic',), ObservationNeeds()),
        ):
            builder = ObservationBuilder(NS(observation_fields=fields), self.runtime)
            self.assertEqual(builder.needs, expected)
            self.assertEqual(builder.camera_static, 'camera_intrinsic' in fields)
        with self.assertRaisesRegex(ValueError, 'no live producer'):
            ObservationBuilder(NS(observation_fields=('unknown',)), self.runtime)
        row = Observation(np.zeros(1, dtype=ARM_STATE_DTYPE), None, None, None, None, 10**10)
        needs = ObservationNeeds(hoi=True, point_cloud=True, vr=True)
        self.assertEqual(check_observation(row, self.runtime, needs, now_ns=10**10),
                         ['pointcloud_unavailable', 'hoi_unavailable', 'vr_unavailable'])
        self.assertEqual(check_observation(row, self.runtime, self.record, now_ns=10**10), [
            'hand_tactile_aggregate_timestamp_ns_unavailable_or_stale',
            'hand_tactile_dense_timestamp_ns_unavailable_or_stale',
            'camera_source_unavailable_or_unmatched',
        ])

    def test_session_sensor_allocation_and_rollout_sources(self):
        # 只替换外部加载和启动边界，保留真实需求推导、Ring 和读取代码。
        module = types.ModuleType('dexmani_policy.deployment')
        module.load_policy = lambda *a, **k: self.fail('unexpected model load')
        manifest = {'task_name': 'test', 'object_names': ['object'], 'template_sha256': 'a' * 64}
        scenarios = [('joint_state',), ('camera_intrinsic',)]
        for derived in ((), ('point_cloud',), ('object_pointcloud',)):
            scenarios.extend(derived + images for images in ((), ('rgb',), ('depth',), ('rgb', 'depth')))
        for fields, recording in itertools.product(scenarios, (False, True)):
            with self.subTest(fields=fields, recording=recording), contextlib.ExitStack() as stack:
                info = NS(observation_fields=fields, action_mode='joint', control_action_dim=19,
                          control_dt_s=0.1, n_action_steps=4, future_steps=8, n_obs_steps=2,
                          supports_rtc=False, requires_prev_action=False,
                          pointcloud_config=asdict(self.runtime.pointcloud),
                          depth_config={'shape': (16, 16), 'depth_scale_m': 0.001},
                          checkpoint_path=Path('offline.pt'), weights='ema', inference_steps=1,
                          task_name='test', hoi_pcd=manifest)
                state = SessionState(stop_signal=threading.Event(), emergency_signal=threading.Event())
                stack.enter_context(patch.dict(sys.modules, {'dexmani_policy.deployment': module}))
                stack.enter_context(patch('dexmani_real.deployment.session.SessionState', return_value=state))
                stack.enter_context(patch('dexmani_real.deployment.session.Keyboard.start'))
                stack.enter_context(patch('dexmani_real.deployment.session.Keyboard.check_health', return_value=True))
                stack.enter_context(patch('dexmani_real.deployment.session.PolicyWorker'))
                stack.enter_context(patch('dexmani_real.deployment.session.Robot'))
                stack.enter_context(patch('dexmani_real.deployment.session.session_directory', return_value=Path('/tmp/not-created')))
                stack.enter_context(patch.object(PolicySession, 'wait_model', return_value=info))
                stack.enter_context(patch('dexmani_real.geometry.hoi.load_task_template', return_value=({}, manifest)))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                session = PolicySession(self.runtime, 'offline.pt', weights=None, inference_steps=None,
                                        output_root=None, device='cpu', seed=0, max_running_s=1,
                                        num_episodes=1, record_raw=recording, camera_calibration_path=None,
                                        enable_hoi_pcd='object_pointcloud' in fields)
                session.prepare_resources()
                channels = session.sensors.channels
                needs = session.observations.needs
                self.assertEqual(channels is not None, recording or needs.camera_worker or session.observations.camera_static)
                now = 10**10
                if channels:
                    self.assertEqual('depth' in channels.camera_ring.field_names,
                                     recording or needs.hoi or needs.point_cloud or 'depth' in fields)
                    camera = {'timestamp_ns': now, 'rgb': np.full((16, 16, 3), 7, np.uint8)}
                    if 'depth' in channels.camera_ring.field_names:
                        camera['depth'] = np.full((16, 16), 9, np.uint16)
                    channels.camera_ring.write(**camera)
                    for ring in (channels.hoi_ring, channels.pointcloud_ring):
                        if ring is not None:
                            self.assertEqual(ring.field_names & {'rgb', 'depth'}, set(fields) & {'rgb', 'depth'})
                    images = {n: np.full((16, 16, 3) if n == 'rgb' else (16, 16), 3,
                                        np.uint8 if n == 'rgb' else np.uint16) for n in needs.camera_fields}
                    if needs.hoi:
                        channels.hoi_request.write(episode_id=1, after_ns=0, after_sequence=0)
                        channels.hoi_ring.write(episode_id=1, timestamp_ns=now-1000, source_camera_sequence=42,
                                                valid=True, object_pointcloud=np.zeros((1024, 3)), **images)
                    if needs.point_cloud:
                        channels.pointcloud_ring.write(timestamp_ns=now-1000, source_camera_sequence=42,
                                                       point_cloud=np.zeros((self.runtime.pointcloud.num_points, 6)), **images)
                feedback = RobotState(np.zeros(1, dtype=ARM_STATE_DTYPE), np.zeros(1, dtype=HAND_STATE_DTYPE))
                rollout = Rollout(state, session.runtime, info, cancel_epoch=1,
                                  robot=NS(read_state=Mock(return_value=feedback)), observations=session.observations,
                                  read_needs=session.read_needs, record_needs=session.record_needs,
                                  sensors=session.sensors,
                                  action_mapper=None, model=None, max_running_s=None, index=1,
                                  output_dir=None, record_raw=recording)
                model_row, raw = rollout.read_observations()
                if needs.hoi or needs.point_cloud:
                    if needs.camera_fields:
                        self.assertEqual(model_row.camera['timestamp_ns'], now-1000)
                        self.assertEqual(model_row.camera['ring_sequence'], 42)
                        for name in needs.camera_fields:
                            self.assertTrue(np.all(model_row.camera[name] == 3))
                    else:
                        self.assertIsNone(model_row.camera)
                if recording:
                    self.assertEqual(raw.camera['timestamp_ns'], now)
                    self.assertEqual(raw.camera['ring_sequence'], 1)
                    self.assertEqual(set(raw.camera), {'rgb', 'depth', 'timestamp_ns', 'ring_sequence'})
                    self.assertIs(raw.arm, model_row.arm)
                    self.assertIs(raw.hand, model_row.hand)
                if not recording or not (needs.hoi or needs.point_cloud):
                    self.assertIs(raw, model_row)

    def test_hoi_and_raw_quality_have_independent_budgets(self):
        now = 10**10
        hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
        for name in ('timestamp_ns', 'tactile_aggregate_timestamp_ns', 'tactile_dense_timestamp_ns'):
            hand[name] = now
        row = Observation(np.zeros(1, dtype=ARM_STATE_DTYPE), hand,
                          {'timestamp_ns': now-200_000_000}, None, None, now,
                          hoi={'timestamp_ns': now-200_000_000, 'valid': True})
        model_needs = ObservationNeeds(hoi=True, camera_fields=('rgb',))
        self.assertEqual(check_observation(row, self.runtime, model_needs, now_ns=now), [])
        self.assertEqual(check_observation(row, self.runtime, self.record, now_ns=now), ['camera_source_stale'])
        row = replace(row, camera={'timestamp_ns': now}, hoi={'timestamp_ns': now-300_000_000, 'valid': False})
        self.assertEqual(check_observation(row, self.runtime, self.record, now_ns=now), [])
        self.assertEqual(check_observation(row, self.runtime, model_needs, now_ns=now), ['hoi_pose_invalid', 'hoi_stale'])
