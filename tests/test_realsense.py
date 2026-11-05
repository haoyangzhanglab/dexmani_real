import multiprocessing as mp
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.config.sensor import CameraConfig
from dexmani_real.geometry.camera import CameraIntrinsics
from dexmani_real.ipc.sensors import (
    SensorChannels,
    camera_dtype,
    read_camera_frame,
    snapshot_camera_parameters,
)
from dexmani_real.sensor import realsense

CONFIG = CameraConfig(width=32, height=32)
INTRINSICS = CameraIntrinsics(32, 32, 100., 100., 16., 16., "none", (0.,) * 5)


def frameset(color=1, depth=None):
    result = SimpleNamespace(
        get_color_frame=Mock(return_value=False if color is None else SimpleNamespace(
            get_frame_number=lambda: color,
            get_data=lambda: np.full((32, 32, 3), [1, 2, 3], np.uint8))),
        get_depth_frame=Mock(return_value=False if depth is None else SimpleNamespace(
            get_frame_number=lambda: depth,
            get_data=lambda: np.full((32, 32), 500, np.uint16))),
        keep=Mock(),
    )
    result.as_frameset = lambda: result
    return result


@pytest.fixture
def sdk(monkeypatch):
    device = Mock()
    device.get_info.side_effect = {"serial": "camera-A", "name": "L515", "line": "L500"}.__getitem__
    device.first_depth_sensor.return_value.get_depth_scale.return_value = .001
    profile = Mock()
    profile.get_device.return_value = device
    profile.get_stream.return_value.as_video_stream_profile.return_value.get_intrinsics.return_value = (
        SimpleNamespace(width=32, height=32, fx=100., fy=100., ppx=16., ppy=16.,
                        model="none", coeffs=[0.] * 5)
    )
    sdk = SimpleNamespace(
        context=Mock(return_value=SimpleNamespace(query_devices=lambda: [device])),
        pipeline=Mock(), config=Mock(), align=Mock(),
        stream=SimpleNamespace(color="color", depth="depth"),
        format=SimpleNamespace(bgr8="bgr8", z16="z16"),
        camera_info=SimpleNamespace(serial_number="serial", name="name", product_line="line"),
        option=SimpleNamespace(visual_preset="preset", confidence_threshold="confidence",
                               auto_exposure_priority="exposure"),
    )

    def start(config, callback):
        enabled = any(call.args[0] == "depth" for call in config.enable_stream.call_args_list)
        callback(frameset(depth=1 if enabled else None))
        return profile

    sdk.pipeline.return_value.start.side_effect = start
    sdk.pipeline.return_value.start.return_value = profile
    sdk.align.return_value.process.side_effect = lambda frames: frames
    monkeypatch.setitem(sys.modules, "pyrealsense2", sdk)
    monkeypatch.setattr(realsense, "WARMUP_FRAMES", 1)
    monkeypatch.setattr(realsense.time, "sleep", Mock())
    return sdk, device


@pytest.mark.parametrize("enabled", [False, True])
def test_startup_and_warmup_only_use_enabled_streams(sdk, enabled):
    sdk, device = sdk
    config = CONFIG if enabled else replace(CONFIG, l515_visual_preset=-1, l515_confidence_threshold=-1)
    camera = realsense.RealSenseCamera(config, depth_enabled=enabled)
    try:
        camera.connect()
        streams = [call.args[0] for call in sdk.config.return_value.enable_stream.call_args_list]
        assert streams == (["color", "depth"] if enabled else ["color"])
        assert device.first_depth_sensor.call_count == int(enabled)
        assert sdk.align.call_count == int(enabled)
        assert (camera.depth_scale_m is not None) == enabled
        assert len(camera.last_identity) == 1 + enabled
        assert camera.callback_error is None
        assert camera.context is sdk.context.return_value
        sdk.pipeline.assert_called_once_with(sdk.context.return_value)
    finally:
        camera.disconnect()
    sdk.pipeline.return_value.stop.assert_called_once()
    assert camera.context is None


@pytest.mark.parametrize("enabled,color,depth,missing", [
    (False, None, None, "RGB"), (True, None, 1, "RGB"), (True, 1, None, "depth"),
])
def test_missing_enabled_channel_propagates_and_closes(sdk, enabled, color, depth, missing):
    sdk, _ = sdk
    profile = sdk.pipeline.return_value.start.return_value

    def start(config, callback):
        callback(frameset(color, depth))
        return profile

    sdk.pipeline.return_value.start.side_effect = start
    camera = realsense.RealSenseCamera(CONFIG, depth_enabled=enabled)
    with pytest.raises(RuntimeError, match=f"missing {missing}"):
        camera.connect()
    assert camera.pipeline is None
    sdk.pipeline.return_value.stop.assert_called_once()


@pytest.mark.parametrize("scale", [0., -1., np.nan, np.inf])
def test_enabled_depth_requires_valid_scale(sdk, scale):
    sdk, device = sdk
    device.first_depth_sensor.return_value.get_depth_scale.return_value = scale
    camera = realsense.RealSenseCamera(CONFIG)
    with pytest.raises(ValueError, match="depth scale"):
        camera.connect()
    assert camera.pipeline is None


def test_alignment_failure_remains_an_error(sdk):
    sdk, _ = sdk
    sdk.align.return_value.process.side_effect = lambda frames: frameset(1, None)
    camera = realsense.RealSenseCamera(CONFIG)
    with pytest.raises(RuntimeError, match="Failed to align"):
        camera.connect()
    sdk.pipeline.return_value.stop.assert_called_once()


def test_explicit_live_serial_must_match(sdk):
    camera = realsense.RealSenseCamera(replace(CONFIG, serial="camera-B"), depth_enabled=False)
    with pytest.raises(RuntimeError, match="serial=camera-B"):
        camera.connect()
    sdk[0].pipeline.assert_not_called()


def test_rgb_duplicate_does_not_renew_time_and_stall_is_reported(monkeypatch):
    clock = Mock(return_value=100)
    monkeypatch.setattr(realsense.time, "monotonic_ns", clock)
    camera = realsense.RealSenseCamera(CONFIG, depth_enabled=False)
    camera.color_intrinsics = INTRINSICS
    monkeypatch.setattr(camera.condition, "wait_for", lambda *args, **kwargs: True)
    source = frameset()
    camera.receive_frame(source)
    frame = camera.read()
    assert frame.timestamp_ns == 100
    assert frame.depth is None and frame.depth_frame_number is None
    np.testing.assert_array_equal(frame.rgb[0, 0], [3, 2, 1])
    assert frame.rgb.flags.owndata and frame.rgb.flags.c_contiguous
    source.get_depth_frame.assert_not_called()
    clock.return_value = 200
    camera.receive_frame(source)
    assert camera.read() is None and camera.received_ns == [100]
    clock.return_value = 100 + realsense.SOURCE_STALL_TIMEOUT_NS
    camera.receive_frame(source)
    with pytest.raises(TimeoutError, match="stopped producing"):
        camera.read()


@pytest.mark.parametrize("stalled", ["RGB", "depth"])
def test_rgbd_keeps_per_channel_first_receive_times(monkeypatch, stalled):
    clock = Mock(return_value=100)
    monkeypatch.setattr(realsense.time, "monotonic_ns", clock)
    camera = realsense.RealSenseCamera(CONFIG)
    camera.color_intrinsics = INTRINSICS
    camera.aligner = SimpleNamespace(process=lambda frames: frames)
    camera.receive_frame(frameset(1, 1))
    first = camera.read()
    clock.return_value = 200
    camera.receive_frame(frameset(1, 2) if stalled == "RGB" else frameset(2, 1))
    second = camera.read()
    assert first.timestamp_ns == second.timestamp_ns == 100
    assert camera.received_ns == ([100, 200] if stalled == "RGB" else [200, 100])
    assert second.depth.dtype == np.uint16 and second.depth.flags.owndata
    clock.return_value = 100 + realsense.SOURCE_STALL_TIMEOUT_NS
    camera.receive_frame(frameset(1, 3) if stalled == "RGB" else frameset(3, 1))
    with pytest.raises(TimeoutError, match=f"new {stalled}"):
        camera.read()


@pytest.mark.parametrize("color,depth", [(0, 1), (1, 0), (-1, 2)])
def test_frame_identity_regression_remains_an_error(color, depth):
    camera = realsense.RealSenseCamera(CONFIG)
    camera.receive_frame(frameset(1, 1))
    camera.receive_frame(frameset(color, depth))
    with pytest.raises(RuntimeError, match="identity"):
        camera.read()


@pytest.mark.parametrize("enabled", [False, True])
def test_worker_ring_and_static_snapshot_use_same_payload(monkeypatch, enabled):
    channels = SensorChannels(mp.get_context("spawn"), camera=CONFIG, depth_enabled=enabled)
    camera = Mock(active_serial="camera-A", color_intrinsics=INTRINSICS,
                  depth_scale_m=.001 if enabled else None)

    def read():
        channels.is_running.value = False
        return realsense.CameraFrame(
            np.zeros((32, 32, 3), np.uint8),
            np.full((32, 32), 500, np.uint16) if enabled else None,
            4 if enabled else None, 5, 100,
        )

    camera.read.side_effect = read
    constructor = Mock(return_value=camera)
    monkeypatch.setattr(realsense, "RealSenseCamera", constructor)
    realsense.run_camera(channels, CONFIG)
    constructor.assert_called_once_with(CONFIG, depth_enabled=enabled)
    result = read_camera_frame(channels)
    assert set(result) == {"rgb", "timestamp_ns", "ring_sequence"} | ({"depth"} if enabled else set())
    assert result["timestamp_ns"] == 100 and result["ring_sequence"] == 1
    snapshot = snapshot_camera_parameters(channels, None, include_extrinsics=False)
    assert snapshot.serial == "camera-A"
    assert snapshot.depth_scale_m == (.001 if enabled else None)
    assert (channels.camera_depth_scale is not None) == enabled
    with pytest.raises(ValueError, match="fields"):
        channels.camera_ring.write(timestamp_ns=100, rgb=result["rgb"], unused=0)
    assert camera_dtype(32, 32).itemsize - camera_dtype(32, 32, depth_enabled=False).itemsize == 32 * 32 * 2
    camera.disconnect.assert_called_once()
