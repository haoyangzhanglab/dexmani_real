import importlib.util
import subprocess
import sys
import threading
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.sensor import vr
from dexmani_real.sensor.config import CameraConfig, VRConfig


class RealSenseTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "sensor_driver_test",
            Path(__file__).resolve().parents[1] / "dexmani_real/sensor/realsense.py",
        )
        self.driver = importlib.util.module_from_spec(spec)
        self.sdk = SimpleNamespace(
            camera_info=SimpleNamespace(
                name="name",
                serial_number="serial",
                firmware_version="firmware",
                product_line="product_line",
            ),
            option=SimpleNamespace(
                emitter_enabled="emitter",
                visual_preset="preset",
                confidence_threshold="confidence",
                auto_exposure_priority="priority",
            ),
            stream=SimpleNamespace(color="color", depth="depth"),
            format=SimpleNamespace(z16="z16", bgr8="bgr8"),
            context=Mock(),
            config=Mock(),
            pipeline=Mock(),
            frame_queue=Mock(),
            align=Mock(),
        )
        with patch.dict(sys.modules, {spec.name: self.driver, "pyrealsense2": self.sdk}):
            spec.loader.exec_module(self.driver)
        self.camera = self.driver.RealSenseCamera(CameraConfig(width=3, height=2))
        self.camera.frame_queue = Mock()
        self.camera.aligner = Mock()
        self.camera.aligner.process.side_effect = lambda frames: frames
        self.camera.received_ns = [100, 100]

    def frame(self, depth_number, color_number):
        depth = Mock()
        depth.get_frame_number.return_value = depth_number
        depth.get_data.return_value = np.full((2, 3), depth_number, dtype=np.uint16)
        color = Mock()
        color.get_frame_number.return_value = color_number
        color.get_data.return_value = np.full((2, 3, 3), [1, 2, 3], dtype=np.uint8)
        frames = Mock()
        frames.get_depth_frame.return_value = depth
        frames.get_color_frame.return_value = color
        queued = Mock()
        queued.as_frameset.return_value = frames
        return queued

    def test_duplicate_frames_are_skipped_before_alignment_and_owned_copies(self):
        queued = [self.frame(*identity) for identity in ((1, 10), (1, 10), (2, 10), (2, 11))]
        self.camera.frame_queue.try_wait_for_frame.side_effect = [(True, frame) for frame in queued]
        with patch.object(self.driver.time, "monotonic_ns", side_effect=[100, 200, 300, 400]):
            frames = [self.camera.read() for _ in queued]
        self.assertEqual(
            [None if frame is None else frame.timestamp_ns for frame in frames],
            [100, None, 100, 300],
        )
        self.assertEqual(self.camera.aligner.process.call_count, 3)
        frameset = queued[0].as_frameset()
        frameset.get_color_frame().get_data()[:] = 0
        frameset.get_depth_frame().get_data()[:] = 0
        np.testing.assert_array_equal(frames[0].rgb, np.full((2, 3, 3), [3, 2, 1]))
        np.testing.assert_array_equal(frames[0].depth, np.ones((2, 3)))
        self.assertTrue(frames[0].rgb.flags.owndata)
        self.assertTrue(frames[0].depth.flags.owndata)

    def test_empty_read_and_persistent_stall_are_distinct(self):
        self.camera.frame_queue.try_wait_for_frame.return_value = (False, None)
        with patch.object(self.driver.time, "monotonic_ns", side_effect=[1_000_000, 3_000_000_000]):
            self.assertIsNone(self.camera.read())
            with self.assertRaisesRegex(TimeoutError, "stopped producing"):
                self.camera.read()
        self.camera.aligner.process.assert_not_called()

    def test_frame_regression_and_single_channel_stall_are_rejected(self):
        self.camera.last_identity = (2, 10)
        self.camera.frame_queue.try_wait_for_frame.return_value = (True, self.frame(1, 10))
        with patch.object(self.driver.time, "monotonic_ns", return_value=200):
            with self.assertRaisesRegex(RuntimeError, "backward"):
                self.camera.read()
        self.camera.frame_queue.try_wait_for_frame.return_value = (True, self.frame(2, 11))
        with patch.object(self.driver.time, "monotonic_ns", return_value=3_000_000_000):
            with self.assertRaisesRegex(TimeoutError, "depth"):
                self.camera.read()
        self.camera.aligner.process.assert_not_called()

    def test_sdk_failure_propagates_without_retry(self):
        error = OSError("USB disconnected")
        self.camera.frame_queue.try_wait_for_frame.side_effect = error
        with self.assertRaises(OSError) as caught:
            self.camera.read()
        self.assertIs(caught.exception, error)
        self.camera.frame_queue.try_wait_for_frame.assert_called_once()

    def connect_device(self, product_line):
        info = dict(name="RealSense", serial="test", firmware="test", product_line=product_line)
        device = Mock()
        device.supports.return_value = True
        device.get_info.side_effect = info.__getitem__
        device.first_depth_sensor().get_depth_scale.return_value = 0.001
        self.sdk.context().query_devices.return_value = [device]
        profile = self.sdk.pipeline().start.return_value
        profile.get_device.return_value = device
        profile.get_stream().as_video_stream_profile().get_intrinsics.return_value = (
            SimpleNamespace(
                width=3,
                height=2,
                fx=3.0,
                fy=3.0,
                ppx=1.0,
                ppy=1.0,
                model="none",
                coeffs=(0, 0, 0, 0, 0),
            )
        )
        return device

    def test_both_camera_models_apply_settings_and_close(self):
        for model in ("D400", "L500"):
            with self.subTest(model=model):
                device = self.connect_device(model)
                with (
                    patch.object(self.driver, "WARMUP_FRAMES", 0),
                    patch.object(self.driver.time, "sleep"),
                ):
                    self.camera.connect()
                self.assertEqual(self.camera.depth_scale_m, 0.001)
                self.assertEqual(self.camera.active_serial, "test")
                device.first_depth_sensor().set_option.assert_any_call(
                    "preset", 5 if model == "L500" else 3.0
                )
                self.camera.disconnect()
                self.assertIsNone(self.camera.pipeline)

    def test_startup_failure_releases_pipeline_and_keeps_original_error(self):
        device = self.connect_device("D400")
        error = OSError("option failed")
        device.first_depth_sensor().set_option.side_effect = error
        self.sdk.pipeline().stop.side_effect = RuntimeError("stop failed")
        with self.assertLogs(self.driver.logger, level="ERROR"):
            with self.assertRaises(OSError) as caught:
                self.camera.connect()
        self.assertIs(caught.exception, error)
        self.sdk.pipeline().stop.assert_called_once()
        self.assertIsNotNone(self.camera.pipeline)

    def test_device_ambiguity_does_not_start_a_pipeline(self):
        self.sdk.context().query_devices.return_value = [Mock(), Mock()]
        with self.assertRaisesRegex(RuntimeError, "configure a serial"):
            self.camera.connect()
        self.sdk.pipeline.assert_not_called()

    def test_config_and_workers_import_without_hardware_sdks(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.modules['pyrealsense2'] = None; sys.modules['hand_tracking_sdk'] = None; from dexmani_real.config import load_experiment_config; from dexmani_real.sensor.camera import run_camera_worker; from dexmani_real.sensor.vr import run_vr_worker; load_experiment_config()",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class HeadFrame(SimpleNamespace):
    pass


class HandFrame(SimpleNamespace):
    pass


class VRTests(unittest.TestCase):
    def test_invalid_pose_is_rejected_and_source_clocks_are_preserved(self):
        channels = SimpleNamespace(
            is_running=SimpleNamespace(value=True),
            vr_ready=threading.Event(),
            vr_ring=Mock(),
        )
        pose = dict(x=1.0, y=2.0, z=3.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0)
        hand = dict(
            wrist=SimpleNamespace(**pose),
            landmarks=SimpleNamespace(points=[(0, 0, 0)] * 21),
            side="right",
            wrist_recv_ts_ns=100,
            landmarks_recv_ts_ns=120,
            sequence_id=0,
        )
        invalid = dict(hand, wrist=SimpleNamespace(**dict(pose, x=np.nan)))
        events = (
            event
            for event in [
                HeadFrame(head=SimpleNamespace(**pose), recv_ts_ns=80),
                HandFrame(**invalid),
                HandFrame(**hand),
            ]
        )
        sdk = SimpleNamespace(
            HandFrame=HandFrame,
            HeadFrame=HeadFrame,
            HandSide=SimpleNamespace(RIGHT="right"),
            unity_left_to_flu_position=lambda *values: values,
            unity_left_to_flu_rotation=lambda *values: values,
        )
        with (
            patch.dict(sys.modules, {"hand_tracking_sdk": sdk}),
            patch.object(vr, "iter_vr_events", return_value=events),
            patch.object(vr, "configure_logging"),
        ):
            with self.assertLogs(vr.logger, level="WARNING"):
                vr.run_vr_worker(channels, VRConfig())
        channels.vr_ring.write.assert_called_once()
        fields = channels.vr_ring.write.call_args.kwargs
        self.assertEqual(
            (fields["recv_ts_ns"], fields["head_recv_ts_ns"], fields["sequence_id"]), (100, 80, 0)
        )
        np.testing.assert_array_equal(fields["wrist_quat_wxyz"], [1, 0, 0, 0])
        self.assertFalse(channels.vr_ready.is_set())

    def test_transports_allow_idle_waits_bad_packets_and_shutdown(self):
        class Mode(Enum):
            UDP = "udp"
            TCP_SERVER = "tcp_server"
            TCP_CLIENT = "tcp_client"

        class ParseError(ValueError):
            pass

        class Timeout(Exception):
            pass

        class Disconnected(Exception):
            pass

        for mode in Mode:
            with self.subTest(mode=mode):
                channels = SimpleNamespace(is_running=SimpleNamespace(value=True))
                receiver = Mock()
                receiver.__enter__ = Mock(return_value=receiver)
                receiver.__exit__ = Mock()

                def stop():
                    channels.is_running.value = False
                    return "late packet"

                reads = iter([Timeout(), "bad", "good"])

                def read():
                    value = next(reads, None)
                    if value is None:
                        return stop()
                    if isinstance(value, Exception):
                        raise value
                    return value

                receiver.recv_line.side_effect = read
                event = object()
                sdk = SimpleNamespace(
                    TransportMode=Mode,
                    ParseError=ParseError,
                    TransportTimeoutError=Timeout,
                    TransportDisconnectedError=Disconnected,
                    HandFrameAssembler=Mock(
                        return_value=SimpleNamespace(push_packet=lambda packet: event)
                    ),
                    parse_line=Mock(side_effect=[ParseError("bad packet"), object()]),
                    TCPClientConfig=Mock(),
                    TCPServerConfig=Mock(),
                    UDPReceiverConfig=Mock(),
                    TCPClientLineReceiver=Mock(return_value=receiver),
                    TCPServerLineReceiver=Mock(return_value=receiver),
                    UDPLineReceiver=Mock(return_value=receiver),
                )
                with patch.dict(sys.modules, {"hand_tracking_sdk": sdk}):
                    self.assertEqual(
                        list(vr.iter_vr_events(channels, VRConfig(transport=mode.value))), [event]
                    )
                receiver.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
