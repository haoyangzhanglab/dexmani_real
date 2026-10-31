import contextlib
import io
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from dexmani_real.config.real import RealConfig, load_config
from scripts import collect_teleop, keyboard_teleop, run_policy
from scripts.tools import control_xhand


class ConfigTests(unittest.TestCase):
    def write_config(self, directory, body):
        path = Path(directory) / "experiment.py"
        path.write_text(body, encoding="utf-8")
        return path

    def test_example_and_default_values_match_existing_experiment_defaults(self):
        default = load_config()
        self.assertEqual(asdict(load_config("experiments/default.py")), asdict(default))
        self.assertEqual(default.arm.home_qpos,
                         (0.0, .041888, .001745, .417134, -3.138102, 1.195551, 0.0))
        self.assertEqual((default.arm.servo_hz, default.arm.feedback_hz), (100, 25))
        self.assertEqual((default.arm.max_joint_velocity_deg_per_s,
                          default.arm.max_joint_acceleration_deg_per_s2,
                          default.arm.max_joint_jerk_deg_per_s3), (150, 900, 12000))
        self.assertEqual((default.teleop.control_hz, default.teleop.ema_alpha_pos,
                          default.teleop.ema_alpha_rot, default.teleop.vr_pos_scale), (16, .95, .85, .7))
        self.assertEqual((default.pointcloud.num_points, default.hand_pointcloud.num_points), (1024, 1024))
        self.assertEqual(default.pointcloud.table_margin_m, .01)
        self.assertEqual(default.pointcloud.workspace, (.15, -.5, 0, .85, .5, .8))
        self.assertEqual((default.execution.mode, default.execution.action_steps,
                          default.execution.prefetch_steps, default.execution.rtc_guidance_cap),
                         ("sync", None, None, None))

    def test_python_file_directly_constructs_nested_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(directory, """
from dataclasses import replace
from dexmani_real.config.real import RealConfig
from dexmani_real.config.control import StaticCollisionBox
base = RealConfig()
CONFIG = replace(base,
    teleop=replace(base.teleop, control_hz=20),
    environment=replace(base.environment, static_boxes=(StaticCollisionBox(name='cup'),)),
)
""")
            config = load_config(path)
        self.assertEqual(config.teleop.control_hz, 20)
        self.assertEqual(config.environment.static_boxes[0].name, "cup")
        self.assertEqual(config.hand, RealConfig().hand)

    def test_missing_or_wrong_export_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for body in ("x = 1", "CONFIG = {}", "CONFIG = None"):
                with self.subTest(body=body):
                    with self.assertRaisesRegex(TypeError, "CONFIG: RealConfig"):
                        load_config(self.write_config(directory, body))

    def test_yaml_and_python_failures_have_no_fallback(self):
        with self.assertRaisesRegex(ValueError, "Python .py"):
            load_config("old_experiment.yaml")
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(directory, "raise RuntimeError('broken experiment')")
            with self.assertRaisesRegex(RuntimeError, "broken experiment"):
                load_config(path)

    def test_nonfinite_dormant_values_cannot_break_archived_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(directory, """
from dataclasses import replace
from dexmani_real.config.real import RealConfig
base = RealConfig()
CONFIG = replace(base, pointcloud=replace(base.pointcloud, remove_table=False, table_margin_m=float('nan')))
""")
            with self.assertRaisesRegex(ValueError, "JSON compliant"):
                load_config(path)

    def test_print_config_has_no_policy_or_hardware_imports(self):
        blocked = {
            "dexmani_real.teleop.run": None,
            "dexmani_real.deployment.run": None,
            "dexmani_policy": None,
            "pyrealsense2": None,
            "xhand_controller": None,
        }
        for main in (collect_teleop.main, run_policy.main, control_xhand.main):
            with self.subTest(main=main.__module__), patch.dict(sys.modules, blocked):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(main(["--config", "experiments/default.py", "--print-config"]), 0)
                payload = json.loads(output.getvalue())
                self.assertEqual(payload["hand"]["enabled"], True)

    def test_no_hand_is_confirmation_and_cannot_disable_default_device(self):
        for main in (collect_teleop.main, keyboard_teleop.main):
            with self.subTest(main=main.__module__), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(["--no-hand"])
                self.assertEqual(error.exception.code, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(directory, """
from dataclasses import replace
from dexmani_real.config.real import RealConfig
base = RealConfig()
CONFIG = replace(base, hand=replace(base.hand, enabled=False), collection=replace(base.collection, enabled=False))
""")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(collect_teleop.main([
                    "--config", str(path), "--no-hand", "--print-config",
                ]), 0)
            self.assertFalse(json.loads(output.getvalue())["hand"]["enabled"])


if __name__ == "__main__":
    unittest.main()
