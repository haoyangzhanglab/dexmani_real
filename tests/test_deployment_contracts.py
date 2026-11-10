"""部署解析与用户输出契约；不加载实际 Checkpoint。"""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import threading
import types
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
    resolve_hoi_inputs,
    validate_execution_config,
)
from dexmani_real.deployment.session import PolicySession
from dexmani_real.runtime.state import SessionState


def policy_info(**overrides):
    return NS(**({'action_mode': 'joint', 'control_action_dim': 19, 'control_dt_s': 0.1,
                     'n_action_steps': 4, 'future_steps': 8, 'supports_rtc': True, 'requires_prev_action': False,
                     'observation_fields': ('joint_state',), 'pointcloud_config': None,
                     'hand_pointcloud_config': None, 'fingertip_link_names': None, 'depth_config': None,
                     'hoi_pcd': None, 'task_name': 'test'} | overrides))


class DeploymentContracts(unittest.TestCase):
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
        for fields, error in ((('point_cloud',), 'pointcloud_config'), (('hand_pointcloud',), 'hand_pointcloud_config'),
                              (('fingertip_xyz',), 'fingertip_link_names'), (('depth',), 'saved shape')):
            with self.assertRaisesRegex(ValueError, error):
                resolve_deployment_config(policy_info(observation_fields=fields), runtime)
        with self.assertRaisesRegex(ValueError, 'no control period'):
            resolve_deployment_config(policy_info(control_dt_s=None), runtime)

    def test_hoi_auto_detection_and_template_identity(self):
        manifest = {'task_name': 'test', 'object_names': ['object'], 'template_sha256': 'a' * 64}
        template = object()
        with patch('dexmani_real.geometry.hoi.load_task_template', return_value=(template, manifest)) as loader:
            for metadata in (None, manifest):
                self.assertIsNone(resolve_hoi_inputs(policy_info(hoi_pcd=metadata)))
            loader.assert_not_called()
            for fields, metadata, error in (
                (('object_pointcloud', 'point_cloud'), manifest, '不支持混合'),
                (('object_pointcloud',), None, '缺少'),
                (('object_pointcloud',), dict(manifest, task_name='other'), '任务名不匹配'),
                (('object_pointcloud',), dict(manifest, task_name='../test'), '任务名不匹配'),
                (('object_pointcloud',), dict(manifest, task_name='/test'), '任务名不匹配'),
            ):
                with self.subTest(metadata=metadata), self.assertRaisesRegex(ValueError, error):
                    resolve_hoi_inputs(policy_info(observation_fields=fields, hoi_pcd=metadata))
            loader.assert_not_called()
            task, loaded = resolve_hoi_inputs(policy_info(observation_fields=('object_pointcloud',), hoi_pcd=manifest))
            self.assertIs(loaded, template)
            self.assertEqual(task.name, 'test')
            with self.assertRaisesRegex(ValueError, '哈希不匹配'):
                resolve_hoi_inputs(policy_info(observation_fields=('object_pointcloud',),
                                               hoi_pcd=dict(manifest, template_sha256='b' * 64)))

    def test_real_template_digest_and_contents(self):
        from dexmani_real.geometry.hoi import load_task_template, template_digest
        with tempfile.TemporaryDirectory(dir='/tmp') as directory:
            task = Path(directory) / 'test'
            (task / 'hoi').mkdir(parents=True)
            (task / 'object').mkdir()
            with self.assertRaises(FileNotFoundError):
                load_task_template(task)
            mesh = task / 'object/mesh.obj'
            mesh.write_text('v 0 0 0\n')
            (task / 'object/asset.json').write_text(json.dumps({'mesh_file': 'mesh.obj'}))
            template = task / 'hoi/template.npz'
            np.savez(template, points_local=np.zeros((1024, 3), np.float32), offsets=np.array([0, 1024]))
            manifest = {'task_name': 'test', 'object_names': ['object'],
                        'template_sha256': template_digest(template, ['object'], [mesh])}
            (task / 'hoi/manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(load_task_template(task)[1], manifest)
            for path in (mesh, template):
                original = path.read_bytes()
                path.write_bytes(original + b'changed')
                with self.assertRaisesRegex(ValueError, '模板或 Mesh 已改变'):
                    load_task_template(task)
                path.write_bytes(original)
            np.savez(template, points_local=np.zeros((1023, 3), np.float32), offsets=np.array([0, 1024]))
            manifest['template_sha256'] = template_digest(template, ['object'], [mesh])
            (task / 'hoi/manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, '非法 HOI 模板点'):
                load_task_template(task)

    def test_geometry_precedence_defaults_and_errors(self):
        runtime = RealConfig()
        local = replace(runtime, pointcloud=replace(runtime.pointcloud, num_points=32),
                        hand_pointcloud=replace(runtime.hand_pointcloud, num_points=64))
        self.assertEqual(resolve_deployment_config(policy_info(), local),
                         replace(local, execution=replace(local.execution, control_dt_s=0.1, action_steps=4)))
        for field, metadata, attribute in (('point_cloud', 'pointcloud_config', 'pointcloud'),
                                           ('hand_pointcloud', 'hand_pointcloud_config', 'hand_pointcloud')):
            resolved = resolve_deployment_config(policy_info(observation_fields=(field,), **{metadata: {}}), local)
            self.assertEqual(getattr(resolved, attribute), getattr(runtime, attribute))
            for recipe, error in (({'num_points': 0}, ValueError), ({'unknown': 1}, TypeError)):
                with self.assertRaisesRegex(error, 'Checkpoint ' + metadata):
                    resolve_deployment_config(policy_info(observation_fields=(field,), **{metadata: recipe}), local)
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

    def test_hoi_errors_precede_device_connection(self):
        module = types.ModuleType('dexmani_policy.deployment')
        module.load_policy = Mock()
        manifest = {'task_name': 'test', 'object_names': ['object'], 'template_sha256': 'a' * 64}
        for metadata, fields in ((None, ('object_pointcloud',)),
                (dict(manifest, task_name='wrong'), ('object_pointcloud',)),
                (dict(manifest, template_sha256='b' * 64), ('object_pointcloud',)),
                (manifest, ('object_pointcloud', 'point_cloud'))):
            with self.subTest(metadata=metadata), contextlib.ExitStack() as stack:
                stack.enter_context(patch.dict(sys.modules, {'dexmani_policy.deployment': module}))
                stack.enter_context(patch('dexmani_real.deployment.session.PolicyWorker'))
                robot = stack.enter_context(patch('dexmani_real.deployment.session.Robot'))
                stack.enter_context(patch('dexmani_real.geometry.hoi.load_task_template', return_value=({}, manifest)))
                state = SessionState(stop_signal=threading.Event(), emergency_signal=threading.Event())
                stack.enter_context(patch('dexmani_real.deployment.session.SessionState', return_value=state))
                session = PolicySession(RealConfig(), 'offline.pt', weights=None, inference_steps=None,
                    output_root=None, device='cpu', seed=0, max_running_s=1, num_episodes=1,
                    record_raw=False, camera_calibration_path=None)
                session.keyboard = Mock()
                stack.enter_context(patch.object(session, 'wait_model', return_value=policy_info(
                    observation_fields=fields, hoi_pcd=metadata)))
                with self.assertRaises(ValueError):
                    session.prepare_resources()
                robot.assert_not_called()
                self.assertEqual(session.sensors.processes, [])

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
        session.info = NS(to_dict=lambda: {'checkpoint': 'offline'})
        session.output_dir = Path('/tmp/not-created')
        with patch('dexmani_real.deployment.session.write_yaml') as write:
            session.save_run_config()
        payload = write.call_args.args[0]
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
