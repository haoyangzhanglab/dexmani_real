"""部署解析与用户输出契约；不加载实际 Checkpoint。"""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.deployment import ExecutionConfig
from dexmani_real.config.real import RealConfig
from dexmani_real.deployment.preflight import (
    resolve_deployment_config,
    validate_execution_config,
)
from dexmani_real.deployment.session import PolicySession
from dexmani_real.runtime.state import SessionState


def policy_info(**overrides):
    return NS(**({'action_mode': 'joint', 'control_action_dim': 19, 'control_dt_s': 0.1,
                     'n_action_steps': 4, 'future_steps': 8, 'supports_rtc': True, 'requires_prev_action': False,
                     'observation_fields': ('joint_state',), 'pointcloud_config': None,
                     'fingertip_link_names': None, 'depth_config': None,
                     'task_name': 'test'} | overrides))


class DeploymentContracts(unittest.TestCase):
    def test_template_order_and_coordinates(self):
        from dexmani_real.geometry.hoi import load_task_template, transform_task_points
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            (task / 'hoi').mkdir()
            (task / 'task.json').write_text(json.dumps({'objects': [{'name': 'b'}, {'name': 'a'}]}))
            points = np.tile(np.array([1, 2, 3], np.float32), (1024, 1))
            np.savez(task / 'hoi/template.npz', points_local=points,
                     offsets=[0, 512, 1024], object_names=['a', 'b'])
            template = load_task_template(task)
            poses = np.tile(np.eye(4), (2, 1, 1))
            poses[0, :3, 3] = [1, 0, 0]
            poses[1, :3, 3] = [0, 2, 0]
            base = np.eye(4)
            base[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
            cloud = transform_task_points(template, poses, base)
            np.testing.assert_array_equal(cloud[0], [-2, 2, 3])
            np.testing.assert_array_equal(cloud[-1], [-4, 1, 3])
            self.assertEqual(template['object_names'].tolist(), ['a', 'b'])

    def test_tracking_depth_gate_and_no_recovery(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'third_party/cordvip_geometry/src'))
        from cordvip_geometry.tracking import SceneTracker
        from cordvip_geometry.rgbd import RGBDFrame
        segmenter = Mock()
        masks = {'object': np.ones((40, 40), bool)}
        segmenter.initialize.return_value = masks
        segmenter.track.return_value = masks
        frame = RGBDFrame(np.zeros((40, 40, 3), np.uint8), np.ones((40, 40), np.float32),
                          np.array([[100., 0, 20], [0, 100., 20], [0, 0, 1]]), 'none', (0, 0, 0, 0, 0))
        with patch('cordvip_geometry.tracking.PoseTracker') as pose_type, \
                patch('cordvip_geometry.tracking.ModelDepthRenderer') as renderer_type, \
                patch('cordvip_geometry.tracking.prepare_tracking_frame', return_value={}):
            pose = pose_type.return_value
            pose.register.return_value = pose.track.return_value = np.eye(4)
            renderer = renderer_type.return_value
            def depths(count):
                values = np.full(400, 1.1, np.float32)
                values[:count] = 1
                return values
            renderer.render.side_effect = [depths(60), depths(59)]
            tracker = SceneTracker(segmenter, {'object': Mock()}, foundationpose_root='.', debug_dir='.')
            self.assertTrue(tracker.step(frame, initial_masks=masks)['objects']['object']['pose_valid'])
            for _ in range(2):
                result = tracker.step(frame)['objects']['object']
                self.assertFalse(result['pose_valid'])
                self.assertIsNone(result['pose'])
            pose.register.assert_called_once()
            pose.track.assert_called_once()

    def test_hand_xyz_and_joint_mapping(self):
        from dexmani_real.geometry.kinematics import ArmFK, HandFK
        from dexmani_real.perception.hand_cloud import HandPointCloud, hand_geometry
        from dexmani_real.deployment.observation import ObservationBuilder
        arm, hand = ArmFK(), HandFK()
        cloud = HandPointCloud(hand, 1024)
        np.testing.assert_array_equal(hand.sdk_to_urdf, [3, 4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2])
        arm_q, hand_q = np.arange(7)*.1, np.arange(12)*.03
        offline = hand_geometry(arm.forward(arm_q)[None], hand_q[None], arm_fk=arm,
                                hand_fk=hand, hand_cloud=cloud)['hand_pointcloud']
        builder = ObservationBuilder(policy_info(observation_fields=('hand_pointcloud',)), RealConfig())
        online = builder.build([{'arm_qpos': arm_q, 'hand_qpos': hand_q}])['hand_pointcloud']
        np.testing.assert_array_equal(online, offline)
        self.assertEqual(online.shape, (1, 1024, 3))
        np.testing.assert_allclose(online[0, :3], [
            [.2200222164, .01861092262, .02495918050],
            [.2782891691, .06748141348, .03761848062],
            [.2365280837, .04420054704, .06319667399]], atol=1e-7)

    def test_reconstruction_selection_outputs(self):
        from PIL import Image
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'third_party/cordvip_geometry/src'))
        from cordvip_geometry.cli.selection import choose, save_selection
        rgb = np.full((30, 40, 3), 127, np.uint8)
        mask = np.zeros((30, 40), bool)
        mask[5:20, 10:30] = True
        candidate = dict(mask=mask, box_xyxy=np.array([10, 5, 30, 20]), label='object', score=.9, mask_score=.8)
        self.assertEqual(choose(rgb, [candidate], None, headless=True), 0)
        with self.assertRaisesRegex(ValueError, '多个候选'):
            choose(rgb, [candidate, candidate], None, headless=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_selection(output, rgb, [candidate], 0)
            metadata = json.loads((output / 'selection.json').read_text())
            with Image.open(output / metadata['mask_file']) as image:
                np.testing.assert_array_equal(np.asarray(image), mask.astype(np.uint8) * 255)
            with Image.open(output / metadata['foreground_file']) as image:
                rgba = np.asarray(image)
                np.testing.assert_array_equal(rgba[:, :, :3], rgb)
                np.testing.assert_array_equal(rgba[:, :, 3], mask.astype(np.uint8) * 255)

    def test_modes_and_ode(self):
        base = ExecutionConfig(control_dt_s=0.1, action_steps=4)
        for mode in ('sync', 'async', 'rtc'):
            validate_execution_config(policy_info(), replace(base, mode=mode))
        for changes in ({'control_dt_s': 0}, {'control_dt_s': float('nan')}, {'action_steps': 0},
                        {'action_steps': 9}, {'mode': 'async', 'prefetch_steps': 0},
                        {'mode': 'rtc', 'prefetch_steps': 5}, {'mode': 'rtc', 'rtc_guidance_cap': -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_execution_config(policy_info(), replace(base, **changes))
        for mode, steps in (('async', 4), ('rtc', 4), ('sync', 3)):
            with self.assertRaisesRegex(ValueError, 'ODE requires sync'):
                validate_execution_config(policy_info(requires_prev_action=True), replace(base, mode=mode, action_steps=steps))
        validate_execution_config(policy_info(requires_prev_action=True), base)

    def test_runtime_precedence_and_missing_recipes(self):
        runtime = RealConfig()
        resolved = resolve_deployment_config(policy_info(), runtime)
        self.assertEqual((resolved.execution.control_dt_s, resolved.execution.action_steps), (0.1, 4))
        override = replace(runtime, execution=replace(runtime.execution, control_dt_s=0.05, action_steps=2))
        self.assertEqual(resolve_deployment_config(policy_info(), override).execution, override.execution)
        for fields, error in ((('point_cloud',), 'pointcloud_config'),
                              (('fingertip_xyz',), 'fingertip_link_names'), (('depth',), 'saved shape')):
            with self.assertRaisesRegex(ValueError, error):
                resolve_deployment_config(policy_info(observation_fields=fields), runtime)
        with self.assertRaisesRegex(ValueError, 'no control period'):
            resolve_deployment_config(policy_info(control_dt_s=None), runtime)

    def test_allocate_task_samples_mesh_and_reuses_template(self):
        from dexmani_real.geometry.hoi import load_task_template
        from scripts.cordvip.allocate_points import allocate_task

        with tempfile.TemporaryDirectory(dir='/tmp') as directory:
            task = Path(directory) / 'test'
            asset = task / 'object'
            asset.mkdir(parents=True)
            (asset / 'mesh.obj').write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
            (asset / 'asset.json').write_text(json.dumps({
                'mesh_file': 'mesh.obj',
                'units': 'm', 'frame': 'object',
            }))
            config = task / 'task.json'
            config.write_text(json.dumps({'objects': [{'name': 'object', 'prompt': 'red object'}]}))
            output = allocate_task(task)
            template = load_task_template(task)
            self.assertEqual(template['object_names'].tolist(), ['object'])
            self.assertEqual(template['offsets'].tolist(), [0, 1024])
            points = template['points_local']
            self.assertTrue((points[:, :2] >= 0).all())
            self.assertTrue((points[:, :2].sum(axis=1) <= 1).all())
            self.assertGreater(float(points[:, 0].std()), 0)
            np.testing.assert_array_equal(points[:, 2], 0)
            frozen = {path: path.read_bytes() for path in output.iterdir()}
            self.assertEqual(allocate_task(task), output)
            config.unlink()
            self.assertEqual(allocate_task(task), output)
            self.assertEqual({path: path.read_bytes() for path in frozen}, frozen)

    def test_geometry_precedence_defaults_and_errors(self):
        runtime = RealConfig()
        local = replace(runtime, pointcloud=replace(runtime.pointcloud, num_points=32),
                        hand_pointcloud=replace(runtime.hand_pointcloud, num_points=64))
        self.assertEqual(resolve_deployment_config(policy_info(), local),
                         replace(local, execution=replace(local.execution, control_dt_s=0.1, action_steps=4)))
        resolved = resolve_deployment_config(policy_info(observation_fields=('point_cloud',), pointcloud_config={}), local)
        self.assertEqual(resolved.pointcloud, runtime.pointcloud)
        resolved = resolve_deployment_config(policy_info(observation_fields=('hand_pointcloud',)), local)
        self.assertEqual(resolved.hand_pointcloud, local.hand_pointcloud)
        for recipe, error in (({'num_points': 0}, ValueError), ({'unknown': 1}, TypeError)):
            with self.assertRaisesRegex(error, 'Checkpoint pointcloud_config'):
                resolve_deployment_config(policy_info(observation_fields=('point_cloud',), pointcloud_config=recipe), local)
        for dt in (0, float('nan')):
            with self.assertRaises(ValueError):
                resolve_deployment_config(policy_info(), replace(runtime, execution=replace(runtime.execution, control_dt_s=dt)))
        explicit = replace(runtime.execution, control_dt_s=0.05, action_steps=2, mode='rtc',
                           prefetch_steps=2, rtc_guidance_cap=3, hoi_max_age_s=0.4)
        self.assertEqual(resolve_deployment_config(policy_info(control_dt_s=None),
                         replace(runtime, execution=explicit)).execution, explicit)

    def test_depth_contract_diagnostics(self):
        from dexmani_real.deployment.observation import ObservationBuilder
        runtime = RealConfig()
        info = policy_info(observation_fields=('depth',),
                           depth_config={'shape': (480, 640), 'depth_scale_m': 0.001})
        resolve_deployment_config(info, runtime)
        with self.assertRaisesRegex(ValueError, r'480, 640.*240, 640.*camera.height.*camera.width'):
            resolve_deployment_config(info, replace(runtime, camera=replace(runtime.camera, height=240)))
        builder = ObservationBuilder(info, runtime)
        builder.configure_camera(NS(depth_scale_m=0.001))
        with self.assertRaisesRegex(ValueError, '0.002.*0.001'):
            builder.configure_camera(NS(depth_scale_m=0.002))
        for scale in (0, -1, float('nan')):
            with self.assertRaisesRegex(ValueError, 'depth_scale_m'):
                resolve_deployment_config(policy_info(observation_fields=('depth',),
                    depth_config={'shape': (480, 640), 'depth_scale_m': scale}), runtime)

    def test_fingertip_order_from_checkpoint(self):
        from dexmani_real.deployment.observation import ObservationBuilder
        links = ('thumb_tip', 'index_tip', 'middle_tip', 'ring_tip')
        info = policy_info(observation_fields=('fingertip_xyz',), fingertip_link_names=links)
        runtime = resolve_deployment_config(info, RealConfig())
        with patch('dexmani_real.deployment.observation.ArmFK'), \
                patch('dexmani_real.deployment.observation.HandFK') as hand_fk:
            ObservationBuilder(info, runtime)
        hand_fk.assert_called_once_with(links)

    def test_cli_print_config_contract(self):
        root = Path(__file__).resolve().parents[1]
        runtime = asdict(RealConfig())
        sections = ('execution', 'arm', 'hand', 'motion', 'camera', 'pointcloud', 'hand_pointcloud', 'environment')
        for script, options, expected in (
            ('run_policy.py', [], {name: runtime[name] for name in sections}),
            ('collect_teleop.py', [], runtime | {'collection': {
                'enabled': True, 'episodes_dir': 'episodes', 'max_duration_s': 60.0,
            }}),
            ('collect_teleop.py', ['--no-record', '--episodes-dir', 'offline', '--max-duration', '12'],
             runtime | {'collection': {'enabled': False, 'episodes_dir': 'offline', 'max_duration_s': 12.0}}),
        ):
            with self.subTest(script=script, options=options):
                result = subprocess.run([sys.executable, '-B', str(root/'scripts'/script), '--print-config', *options],
                                        cwd=root, capture_output=True, text=True, check=True)
                actual = json.loads(result.stdout)
                self.assertEqual(tuple(actual), tuple(expected))
                # JSON 将配置元组编码为数组；逐字段比较可直接定位变化。
                self.assertEqual(actual, json.loads(json.dumps(expected)))
                self.assertEqual(result.stdout, json.dumps(expected, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

    def test_archive_sections_and_early_failure_cleanup(self):
        runtime = RealConfig()
        state = SessionState(stop_signal=threading.Event(), emergency_signal=threading.Event())
        with patch('dexmani_real.deployment.session.SessionState', return_value=state):
            session = PolicySession(runtime, 'offline.pt', weights=None, inference_steps=None,
                                    output_root=None, device='cpu', seed=0, max_running_s=1,
                                    num_episodes=1, record_raw=False, camera_calibration_path=None)
        from dexmani_policy.deployment import PolicyInfo
        session.info = PolicyInfo(
            checkpoint_path='offline.pt', weights='raw', inference_steps=2,
            task_name='test', observation_fields=('joint_state',), n_obs_steps=2,
            n_action_steps=4, future_steps=8, control_action_dim=19, action_mode='joint',
            rgb_preprocessing={}, supports_rtc=True, requires_prev_action=False,
            control_dt_s=0.1, depth_config=None, pointcloud_config=None,
            fingertip_link_names=None,
        )
        session.output_dir = Path('/tmp/not-created')
        with patch('dexmani_real.deployment.session.write_yaml') as write:
            session.save_run_config()
        payload = write.call_args.args[0]
        self.assertEqual(payload['policy'], asdict(session.info))
        self.assertEqual(payload['policy']['checkpoint_path'], 'offline.pt')
        sections = ('execution', 'arm', 'hand', 'motion', 'camera', 'pointcloud', 'hand_pointcloud', 'environment')
        self.assertEqual(tuple(payload['runtime']), sections)
        self.assertEqual(payload['runtime'], {name: asdict(getattr(runtime, name)) for name in sections})
        self.assertEqual(tuple(payload), ('policy', 'runtime', 'camera_calibration_path', 'effective_camera',
                                         'table_plane_abcd', 'table_plane_path', 'seed', 'device',
                                         'num_episodes', 'max_duration_s', 'record_raw'))
        session.output_dir = None
        del session.info
        with patch.object(session, 'prepare_resources', side_effect=ValueError('model load failed')), \
                patch('dexmani_real.deployment.session.close_runtime') as close, \
                patch('dexmani_real.deployment.session.configure_logging'), \
                self.assertLogs('dexmani_real.deployment.session', level='ERROR'), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(session.run(), 1)
        close.assert_called_once()
        self.assertEqual(session.summary['termination_reason'], 'failed')
