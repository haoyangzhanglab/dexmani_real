"""真实 Ring 与离线反馈验证模型／录制的来源契约，不启动 Worker。"""

import contextlib
import io
import itertools
import sys
import threading
import types
import unittest
from collections import deque
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

    def test_vr_receive_time_survives_queueing_and_assembly(self):
        from hand_tracking_sdk import parse_line

        from dexmani_real.sensor.vr import run_vr

        head = 'Head pose:, 0, 0, 0, 0, 0, 0, 1\n'
        wrist = 'Right wrist:, 0, 0, 0, 0, 0, 0, 1\n'
        landmarks = 'Right landmarks:, ' + ', '.join('0' for _ in range(63)) + '\n'
        for mode in ('tcp_server', 'tcp_client', 'udp'):
            for case in ('fresh', 'queued', 'older_wrist'):
                with self.subTest(mode=mode, case=case):
                    received_ns = 10**10
                    clock = NS(now=received_ns)
                    chunks = deque([(head + wrist).encode(), landmarks.encode()] if case == 'older_wrist'
                                   else [(head + wrist + landmarks).encode()])
                    sock, server = Mock(), Mock()
                    shared = NS(is_running=NS(value=True), vr_ring=Mock())
                    shared.vr_ring.write.side_effect = lambda shared=shared, **kw: setattr(shared.is_running, 'value', False)

                    def receive(size, case=case, chunks=chunks, clock=clock):
                        if case == 'older_wrist' and len(chunks) == 1:
                            clock.now += 10**9
                        return chunks.popleft()

                    def parse(line, case=case, clock=clock):
                        packet = parse_line(line)
                        if case == 'queued' and line.startswith('Head'):
                            # 腕部与关键点已经在同次接收中排队，延迟后才轮到它们解析。
                            clock.now += 10**9
                        return packet

                    sock.recv.side_effect = receive
                    sock.recvfrom.side_effect = lambda size, receive=receive: (receive(size), ('fake', 8000))
                    server.accept.side_effect = [(sock, ('fake', 8000)), BlockingIOError()]
                    with patch('hand_tracking_sdk.transport.socket.socket',
                               return_value=server if mode == 'tcp_server' else sock), \
                            patch('hand_tracking_sdk.transport.select', side_effect=[
                                ([server], [], []), ([sock], [], []), ([sock], [], []),
                            ]), \
                            patch('hand_tracking_sdk.transport.monotonic_ns', side_effect=lambda clock=clock: clock.now), \
                            patch('hand_tracking_sdk.frame.monotonic_ns', side_effect=lambda clock=clock: clock.now), \
                            patch('hand_tracking_sdk.parse_line', side_effect=parse), \
                            patch('dexmani_real.sensor.vr.configure_logging'):
                        run_vr(shared, replace(self.runtime.vr, transport=mode))

                    shared.vr_ring.write.assert_called_once()
                    frame = shared.vr_ring.write.call_args.kwargs
                    self.assertEqual(frame['recv_ts_ns'], received_ns)
                    self.assertEqual(clock.now - frame['recv_ts_ns'], 0 if case == 'fresh' else 10**9)
                    row = Observation(np.zeros(1, dtype=ARM_STATE_DTYPE), None, None, frame, None, clock.now)
                    self.assertEqual(self.runtime.teleop.vr_stale_threshold_s, 0.5)
                    self.assertEqual(check_observation(row, self.runtime, ObservationNeeds(vr=True),
                                                      now_ns=clock.now), [] if case == 'fresh' else ['vr_stale'])

    def test_hand_calibration_accepts_same_time_updates_once_per_publication(self):
        from dexmani_real.teleop.run import calibrate_hand

        now_ns = 10**10
        first = {'recv_ts_ns': now_ns, 'landmarks': np.array([1.0])}
        second = {'recv_ts_ns': now_ns, 'landmarks': np.array([2.0])}
        stale = dict(first, recv_ts_ns=now_ns - 10**9)
        ring = Mock()
        ring.read.side_effect = [(stale, 1), (first, 2), (first, 2), (second, 3)]
        retargeter = Mock()
        loop = NS(
            controller=NS(hand_retargeter=retargeter),
            runtime=replace(self.runtime, hand_retargeting=replace(
                self.runtime.hand_retargeting, calibration_samples=2)),
            runtime_state=NS(cancel_epoch=0),
            robot=Mock(),
            sensors=NS(channels=NS(vr_ring=ring)),
            poll_blocking_commands=Mock(return_value=False),
        )
        with patch('dexmani_real.teleop.run.time.monotonic', return_value=10.0), \
                patch('dexmani_real.teleop.run.time.monotonic_ns', return_value=now_ns), \
                patch('dexmani_real.teleop.run.time.sleep'), \
                patch('dexmani_real.teleop.retargeting.landmarks.prepare_landmarks',
                      side_effect=lambda points: points), \
                patch('dexmani_real.teleop.retargeting.landmarks.hand_lengths',
                      side_effect=lambda points: points) as lengths, \
                contextlib.redirect_stdout(io.StringIO()):
            calibrate_hand(loop, 0)
        self.assertEqual(lengths.call_count, 2)
        retargeter.calibrate.assert_called_once()
        np.testing.assert_array_equal(retargeter.calibrate.call_args.args[0], [[1.0], [2.0]])

    def test_joint_hoi_source_skew(self):
        from dexmani_real.runtime.observation import check_device_feedback
        now = 10**10
        runtime = replace(self.runtime, execution=replace(self.runtime.execution, hoi_max_skew_s=.05))
        arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
        hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
        arm['timestamp_ns'] = now - 10_000_000
        hand['timestamp_ns'] = now - 20_000_000
        row = Observation(arm, hand, None, None, None, now,
                          hoi={'valid': True, 'timestamp_ns': now - 200_000_000})
        needs = ObservationNeeds(hoi=True, hoi_joint_input=True)
        self.assertEqual(check_observation(row, self.runtime, needs, now_ns=now), [])
        self.assertEqual(check_device_feedback(row, runtime, now_ns=now), [])
        self.assertEqual(check_observation(row, runtime, needs, now_ns=now), ['hoi_source_skew'])
        row = replace(row, hoi=dict(row.hoi, timestamp_ns=now-60_000_001))
        self.assertEqual(check_observation(row, runtime, needs, now_ns=now), ['hoi_source_skew'])
        row = replace(row, hoi=dict(row.hoi, timestamp_ns=now-60_000_000))
        self.assertEqual(check_observation(row, runtime, needs, now_ns=now), [])
        self.assertEqual(check_observation(row, runtime, ObservationNeeds(), now_ns=now), [])

    def test_hoi_worker_failure_latch_and_inflight_epoch(self):
        from dexmani_real.perception.hoi_worker import run_hoi_worker
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'third_party/cordvip_geometry/src'))
        from cordvip_geometry.tracking import TrackingFailure
        shared = NS(is_running=NS(value=True), hoi_ready=NS(value=False),
                    hoi_request=Mock(), camera_ring=Mock(), hoi_ring=Mock(field_names=set()))
        command = {'episode_id': 1, 'after_sequence': 0, 'after_ns': 0}
        shared.hoi_request.read.side_effect = lambda: (command.copy(), 1)
        shared.camera_ring.read.side_effect = [({'timestamp_ns': i}, i) for i in range(1, 5)]
        tracker = Mock()
        calls = []
        def step(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise TrackingFailure('object: depth lost')
            if len(calls) == 3:
                command['episode_id'] = 3
            return {'object': np.eye(4)}
        tracker.step.side_effect = step
        def sleep(_):
            self.assertEqual(len(calls), 2)
            command['episode_id'] = 2
        def publish(**values):
            if values['episode_id'] == 3:
                shared.is_running.value = False
        shared.hoi_ring.write.side_effect = publish
        template = {'object_names': ['object'], 'offsets': [0, 1024],
                    'points_local': np.ones((1024, 3), np.float32)}
        with patch('cordvip_geometry.tracking.SceneTracker.open') as opened, \
                patch('cordvip_geometry.rgbd.RGBDFrame.from_camera'), \
                patch('dexmani_real.geometry.hoi.task_objects', return_value=[{'name': 'object', 'prompt': 'object'}]), \
                patch('dexmani_real.perception.hoi_worker.signal.signal'), \
                patch('dexmani_real.perception.hoi_worker.time.sleep', side_effect=sleep):
            opened.return_value.__enter__.return_value = tracker
            run_hoi_worker(shared, NS(T_xarm_base_from_color=np.eye(4)), Path('/tmp'), template)
        published = [call.kwargs for call in shared.hoi_ring.write.call_args_list]
        self.assertEqual([(v['episode_id'], v['valid']) for v in published], [(1, True), (1, False), (3, True)])
        self.assertEqual(len(calls), 4)
        self.assertEqual(tracker.reset.call_count, 3)

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
                          checkpoint_path='offline.pt', weights='ema', inference_steps=1,
                          task_name='test')
                state = SessionState(stop_signal=threading.Event(), emergency_signal=threading.Event())
                stack.enter_context(patch.dict(sys.modules, {'dexmani_policy.deployment': module}))
                stack.enter_context(patch('dexmani_real.deployment.session.SessionState', return_value=state))
                stack.enter_context(patch('dexmani_real.deployment.session.Keyboard.start'))
                stack.enter_context(patch('dexmani_real.deployment.session.Keyboard.check_health', return_value=True))
                stack.enter_context(patch('dexmani_real.deployment.session.PolicyWorker'))
                stack.enter_context(patch('dexmani_real.deployment.session.Robot'))
                stack.enter_context(patch('dexmani_real.deployment.session.session_directory', return_value=Path('/tmp/not-created')))
                stack.enter_context(patch.object(PolicySession, 'wait_model', return_value=info))
                loader = stack.enter_context(patch('dexmani_real.geometry.hoi.load_task_template', return_value={}))
                output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                session = PolicySession(self.runtime, 'offline.pt', weights=None, inference_steps=None,
                                        output_root=None, device='cpu', seed=0, max_running_s=1,
                                        num_episodes=1, record_raw=recording, camera_calibration_path=None)
                session.prepare_resources()
                self.assertEqual(loader.call_count, int('object_pointcloud' in fields))
                self.assertIn('action_mode=joint', output.getvalue())
                self.assertIn('action_steps=4 (Checkpoint)', output.getvalue())
                self.assertIn('control_dt_s=0.1 (RealConfig)', output.getvalue())
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

                # 执行实际启动分支，但 Process 和所有设备边界均使用替身。
                session.process_context = Mock()
                session.process_context.Process.side_effect = lambda **kw: NS(**kw)
                if session.sensors.channels and 'object_pointcloud' in fields:
                    session.sensors.channels.hoi_ready.value = True
                with patch.object(session, 'check_startup'), patch.object(session, 'save_run_config'), \
                        patch.object(session, 'warmup_policy'), patch.object(session.observations, 'configure_camera'), \
                        patch('dexmani_real.deployment.session.snapshot_camera_parameters'), \
                        patch('dexmani_real.deployment.session.load_table_plane'), \
                        patch.object(session.sensors, 'shutdown'), patch.object(session.sensors, 'start') as start:
                    session.start_devices()
                started = [call.args[0].name for call in start.call_args_list]
                self.assertEqual('hoi' in started, 'object_pointcloud' in fields)

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

    def test_hoi_request_boundaries_and_metadata_only_reads(self):
        import multiprocessing as mp

        from dexmani_real.ipc.sensors import SensorChannels
        from dexmani_real.runtime.observation import check_hoi_status, read_observation
        now = 10**10
        cases = [
            ({}, {}, []),
            ({'episode_id': 2}, {}, ['hoi_unavailable']),
            ({'episode_id': 0}, {'episode_id': 0}, ['hoi_unavailable']),
            ({'after_ns': now}, {}, ['hoi_unavailable']),
            ({'after_ns': now+1}, {}, ['hoi_unavailable']),
            ({'after_sequence': 5}, {}, ['hoi_unavailable']),
            ({'after_sequence': 6}, {}, ['hoi_unavailable']),
            ({}, {'valid': False}, ['hoi_pose_invalid']),
            ({}, {'timestamp_ns': now-1_000_000_000}, ['hoi_stale']),
            ({}, {'timestamp_ns': now+1}, ['hoi_stale']),
            ({}, {'valid': False, 'timestamp_ns': now-1_000_000_000}, ['hoi_pose_invalid', 'hoi_stale']),
        ]
        for request_changes, result_changes, expected in cases:
            with self.subTest(request=request_changes, result=result_changes):
                channels = SensorChannels(mp.get_context('spawn'), camera=self.runtime.camera, hoi=True)
                request = {'episode_id': 1, 'after_ns': 1, 'after_sequence': 1}
                request.update(request_changes)
                result = {'episode_id': 1, 'timestamp_ns': now, 'source_camera_sequence': 5,
                              'valid': True, 'object_pointcloud': np.zeros((1024, 3), np.float32)}
                result.update(result_changes)
                channels.hoi_request.write(**request)
                channels.hoi_ring.write(**result)
                with patch.object(channels.hoi_ring, 'read', wraps=channels.hoi_ring.read) as read:
                    self.assertEqual(check_hoi_status(channels, self.runtime, now_ns=now), expected)
                    if request["episode_id"] == 0:
                        read.assert_not_called()
                    else:
                        read.assert_called_once()
                        self.assertEqual(set(read.call_args.kwargs['fields']),
                                         {'episode_id', 'timestamp_ns', 'source_camera_sequence', 'valid'})
                robot = Mock(read_state=Mock(return_value=RobotState(np.zeros(1, dtype=ARM_STATE_DTYPE), None)))
                needs = ObservationNeeds(hoi=True)
                row = read_observation(channels, robot, needs=needs)
                self.assertEqual(check_observation(row, self.runtime, needs, now_ns=now), expected)
        channels = SensorChannels(mp.get_context('spawn'), camera=self.runtime.camera, hoi=True)
        self.assertEqual(check_hoi_status(channels, self.runtime, now_ns=now), ['hoi_unavailable'])
        channels.hoi_ring.write(episode_id=1, timestamp_ns=now, source_camera_sequence=5,
                                valid=True, object_pointcloud=np.zeros((1024, 3)))
        self.assertEqual(check_hoi_status(channels, self.runtime, now_ns=now), ['hoi_unavailable'])
        channels.hoi_request.write(episode_id=1, after_ns=1, after_sequence=1)
        self.assertEqual(check_hoi_status(channels, self.runtime, now_ns=now), [])
        channels.hoi_request.write(episode_id=2, after_ns=now, after_sequence=5)
        self.assertEqual(check_hoi_status(channels, self.runtime, now_ns=now), ['hoi_unavailable'])

    def test_active_hoi_request_without_result_is_unavailable(self):
        import multiprocessing as mp

        from dexmani_real.ipc.sensors import SensorChannels
        from dexmani_real.runtime.observation import check_hoi_status

        channels = SensorChannels(mp.get_context('spawn'), camera=self.runtime.camera, hoi=True)
        channels.hoi_request.write(episode_id=1, after_ns=1, after_sequence=1)
        self.assertEqual(check_hoi_status(channels, self.runtime, now_ns=10**10), ['hoi_unavailable'])
