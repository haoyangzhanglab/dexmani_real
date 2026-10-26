"""配置及标定路径契约；CLI 在拒绝后端导入和标定读取的子进程中运行。"""

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import yaml

from dexmani_real import PACKAGE_DIR
from dexmani_real.calibration.paths import (
    CAMERAS_PATH,
    TABLE_PLANE_PATH,
    VR_TRANSFORM_PATH,
)
from dexmani_real.calibration.table_plane import (
    resolve_table_plane,
    resolve_table_plane_path,
)
from dexmani_real.config import ExperimentConfig, load_experiment_config
from dexmani_real.control.config import TableCollisionConfig


class ConfigurationTests(unittest.TestCase):
    def test_cli_yaml_default_precedence_and_safety_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'experiment.yaml'
            path.write_text('execution:\n  action_steps: 3\nsafety:\n  camera_ready_timeout_s: 17\n')
            config = load_experiment_config(yaml_path=path, cli_overrides={'execution.action_steps': 5})
        self.assertEqual(config.execution.action_steps, 5)
        self.assertEqual(config.safety.camera_ready_timeout_s, 17)
        self.assertEqual(config.safety.vr_ready_timeout_s, ExperimentConfig().safety.vr_ready_timeout_s)

    def test_unknown_fields_and_quoted_switches_are_rejected(self):
        for changes in ({'safety.unknown': 1}, {'unknown.value': 1}, {'hand.enabled': 'false'}):
            with self.subTest(changes=changes), self.assertRaises(TypeError):
                load_experiment_config(cli_overrides=changes)

    def test_print_config_does_not_import_backends_or_read_calibration(self):
        root = PACKAGE_DIR.parent
        program = '''
import importlib.abc, pathlib, runpy, sys
blocked = {'pinocchio', 'mplib', 'trimesh', 'cv2', 'dexmani_policy', 'pyrealsense2', 'xarm', 'pynput'}
class ForbidBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('print-config imported backend: ' + fullname)
sys.meta_path.insert(0, ForbidBackends())
def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        name = str(args[0])
        if name.endswith('.json') and ('calibration' in name or '/state/' in name):
            raise AssertionError('print-config read calibration: ' + name)
sys.addaudithook(audit)
script = sys.argv[1]
sys.argv = [script, '--print-config']
runpy.run_path(script, run_name='__main__')
'''
        for name in ('run_policy', 'collect_teleop'):
            with self.subTest(script=name):
                result = subprocess.run([sys.executable, '-c', program, str(root / 'scripts' / (name + '.py'))],
                                        cwd=root, text=True, capture_output=True, timeout=20, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                values = yaml.safe_load(result.stdout)
                self.assertIn('safety', values)
                self.assertEqual(values['environment']['table']['plane_path'],
                                 'dexmani_real/calibration/parameters/table_plane.json')

    def test_keyboard_help_preserves_existing_cli(self):
        result = subprocess.run([sys.executable, str(PACKAGE_DIR.parent / 'scripts/keyboard_teleop.py'), '--help'],
                                text=True, capture_output=True, timeout=20, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--control-hz', result.stdout)
        self.assertNotIn('--print-config', result.stdout)

    def test_joint_target_checks_do_not_import_online_ik(self):
        program = '''
import sys
from dexmani_real.control.target_validation import TargetChecker
assert 'dexmani_real.control.online_ik' not in sys.modules
assert not {'pinocchio', 'mplib'} & sys.modules.keys()
'''
        result = subprocess.run(
            [sys.executable, '-c', program], cwd=PACKAGE_DIR.parent,
            text=True, capture_output=True, timeout=20, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class CalibrationPathsTests(unittest.TestCase):
    def test_defaults_exist_in_new_resource_directory(self):
        for path in (CAMERAS_PATH, TABLE_PLANE_PATH, VR_TRANSFORM_PATH):
            self.assertEqual(path.parent, PACKAGE_DIR / 'calibration' / 'parameters')
            with path.open() as stream:
                self.assertIsInstance(json.load(stream), dict)
        self.assertEqual(resolve_table_plane_path(TableCollisionConfig()), TABLE_PLANE_PATH)

    def test_explicit_absolute_and_relative_paths_keep_their_meaning(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'explicit.json'
            path.write_text(json.dumps(dict(a=0, b=0, c=2, d=-.4)))
            config = replace(TableCollisionConfig(), plane_path=str(path))
            self.assertEqual(resolve_table_plane_path(config), path)
            self.assertEqual(resolve_table_plane(config), (0, 0, 2, -.4))
        relative = replace(TableCollisionConfig(), plane_path='custom/table.json')
        self.assertEqual(resolve_table_plane_path(relative), PACKAGE_DIR.parent / 'custom/table.json')

    def test_missing_explicit_path_does_not_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(TableCollisionConfig(), plane_path=str(Path(directory) / 'missing.json'))
            with self.assertRaises(FileNotFoundError):
                resolve_table_plane(config)
            self.assertIsNone(resolve_table_plane(config, required=False))
