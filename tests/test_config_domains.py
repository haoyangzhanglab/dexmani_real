"""只验证启用路径的业务域，存档配置仍须完整、可序列化。"""

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dexmani_real.config.calibration_data import TABLE_PLANE_PATH, TABLE_PLANE_RELATIVE_PATH
from dexmani_real.config.control import EnvironmentConfig, TableCollisionConfig
from dexmani_real.config.perception import PointCloudConfig
from dexmani_real.config.real import load_config
from dexmani_real.config.sensor import CameraConfig
from dexmani_real.sensor.realsense import RealSenseCamera
from dexmani_real.utils.table_plane import load_table_plane, resolve_table_plane_path


class ConfigurationDomainTests(unittest.TestCase):
    def test_disabled_table_does_not_validate_home_box(self):
        table = TableCollisionConfig(enabled=False, size_xy_m=(-1., 0.), thickness_m=-1.,
                                     soft_clearance_m=-1., allowed_contact_links=("",))
        environment = EnvironmentConfig(table=table)
        environment.validate()
        with self.assertRaises(ValueError):
            replace(environment, table=replace(table, enabled=True)).validate()
        json.dumps(asdict(environment), allow_nan=False)

    def test_table_margin_only_checked_when_cropping(self):
        config = PointCloudConfig(remove_table=False, table_margin_m=-1.)
        self.assertEqual(config.table_margin_m, -1.)
        with self.assertRaisesRegex(ValueError, "table_margin_m"):
            replace(config, remove_table=True)

    def test_plane_path_is_shared_and_keeps_original_coefficients(self):
        environment = EnvironmentConfig()
        self.assertEqual(environment.table_plane_path, TABLE_PLANE_RELATIVE_PATH)
        self.assertEqual(resolve_table_plane_path(environment.table_plane_path), TABLE_PLANE_PATH)
        original = json.loads(TABLE_PLANE_PATH.read_text())
        self.assertEqual(load_table_plane(environment.table_plane_path),
                         tuple(float(original[name]) for name in ("a", "b", "c", "d")))
        with self.assertRaisesRegex(RuntimeError, "table_plane_path"):
            load_table_plane(None)

    def test_inactive_nan_still_rejected_by_full_config_snapshot(self):
        configurations = (
            "replace(RealConfig(), camera=CameraConfig(l515_visual_preset=float('nan')))",
            "replace(RealConfig(), pointcloud=PointCloudConfig(remove_table=False, table_margin_m=float('nan')))",
            "replace(RealConfig(), environment=EnvironmentConfig(table=TableCollisionConfig(enabled=False, thickness_m=float('nan'))))",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.py"
            for expression in configurations:
                with self.subTest(expression=expression):
                    path.write_text(
                        "from dataclasses import replace\n"
                        "from dexmani_real.config.real import RealConfig\n"
                        "from dexmani_real.config.sensor import CameraConfig\n"
                        "from dexmani_real.config.perception import PointCloudConfig\n"
                        "from dexmani_real.config.control import EnvironmentConfig, TableCollisionConfig\n"
                        f"CONFIG = {expression}\n"
                    )
                    with self.assertRaisesRegex(ValueError, "JSON compliant"):
                        load_config(path)


class CameraDomainTests(unittest.TestCase):
    def fake_sdk(self):
        device = Mock()
        device.get_info.return_value = "camera-serial"
        depth = device.first_depth_sensor.return_value
        depth.supports.return_value = False
        depth.get_depth_scale.return_value = .001
        device.first_color_sensor.return_value.supports.return_value = False
        profile = Mock()
        profile.get_device.return_value = device
        profile.get_stream.return_value.as_video_stream_profile.return_value.get_intrinsics.return_value = SimpleNamespace(
            width=640, height=480, fx=600., fy=600., ppx=320., ppy=240., model="none", coeffs=[0.] * 5,
        )
        sdk = SimpleNamespace(
            context=Mock(), config=Mock(), pipeline=Mock(), align=Mock(),
            camera_info=SimpleNamespace(serial_number="serial"),
            stream=SimpleNamespace(depth="depth", color="color"),
            format=SimpleNamespace(z16="z16", bgr8="bgr8"),
            option=SimpleNamespace(visual_preset="preset", emitter_enabled="emitter",
                                   auto_exposure_priority="exposure", confidence_threshold="confidence"),
        )
        sdk.context.return_value.query_devices.return_value = [device]
        sdk.pipeline.return_value.start.return_value = profile
        return sdk, depth

    def test_d435_ignores_unused_l515_ranges(self):
        config = CameraConfig(l515_visual_preset=-100, l515_confidence_threshold=100)
        config.validate()
        sdk, depth = self.fake_sdk()
        camera = RealSenseCamera(config)
        camera.read = Mock()
        with patch.dict("sys.modules", {"pyrealsense2": sdk}), patch.object(
            camera, "device_info", return_value={"product_line": "D400", "name": "D435"},
        ):
            try:
                camera.connect()
                depth.set_option.assert_not_called()
            finally:
                camera.disconnect()
        sdk.pipeline.return_value.stop.assert_called_once()

    def test_l515_checks_its_ranges_before_setting_options(self):
        for config in (CameraConfig(l515_visual_preset=-1), CameraConfig(l515_confidence_threshold=4)):
            with self.subTest(config=config):
                config.validate()
                sdk, depth = self.fake_sdk()
                camera = RealSenseCamera(config)
                with patch.dict("sys.modules", {"pyrealsense2": sdk}), patch.object(
                    camera, "device_info", return_value={"product_line": "L500", "name": "L515"},
                ):
                    with self.assertRaisesRegex(ValueError, "l515_"):
                        camera.connect()
                depth.set_option.assert_not_called()
                sdk.pipeline.return_value.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
