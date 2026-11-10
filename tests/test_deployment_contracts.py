"""部署解析与用户输出契约；不加载实际 Checkpoint。"""

import contextlib
import io
import json
import subprocess
import sys
import threading
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

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

    def test_hoi_opt_in_and_template_identity(self):
        self.assertIsNone(resolve_hoi_inputs(policy_info(), False))
        for fields, enabled in ((('object_pointcloud',), False), (('joint_state',), True)):
            with self.assertRaisesRegex(ValueError, '必须同时启用'):
                resolve_hoi_inputs(policy_info(observation_fields=fields), enabled)
        with self.assertRaisesRegex(ValueError, '不支持混合'):
            resolve_hoi_inputs(policy_info(observation_fields=('object_pointcloud', 'point_cloud')), True)
        manifest = {'task_name': 'test', 'object_names': ['object'], 'template_sha256': 'a' * 64}
        template = object()
        with patch('dexmani_real.geometry.hoi.load_task_template', return_value=(template, manifest)):
            task, loaded = resolve_hoi_inputs(policy_info(observation_fields=('object_pointcloud',), hoi_pcd=manifest), True)
            self.assertIs(loaded, template)
            self.assertEqual(task.name, 'test')
            with self.assertRaisesRegex(ValueError, '哈希不匹配'):
                resolve_hoi_inputs(policy_info(observation_fields=('object_pointcloud',), hoi_pcd=dict(manifest, template_sha256='b' * 64)), True)

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
