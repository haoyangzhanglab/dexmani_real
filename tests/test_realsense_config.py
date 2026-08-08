from __future__ import annotations

from unittest.mock import Mock, patch

from dexmani_real.sensor.realsense import L515DepthConfig, RealSense, RealSenseConfig, rs


def test_l515_depth_offset_is_verified_without_setter() -> None:
    depth_config = L515DepthConfig()
    camera = object.__new__(RealSense)
    camera.config = RealSenseConfig(l515_depth_config=depth_config)

    sensor = Mock()
    sensor.supports.return_value = True
    sensor.get_option.side_effect = lambda option: (
        float(depth_config.receiver_gain) if option == rs.option.receiver_gain else float(depth_config.depth_offset)
    )
    device = Mock()
    device.first_depth_sensor.return_value = sensor
    camera.profile = Mock()
    camera.profile.get_device.return_value = device

    with patch("dexmani_real.sensor.realsense.time.sleep"):
        camera._apply_l515_depth_config()

    written_options = [call.args[0] for call in sensor.set_option.call_args_list]
    assert rs.option.depth_offset not in written_options
    sensor.get_option.assert_any_call(rs.option.depth_offset)
