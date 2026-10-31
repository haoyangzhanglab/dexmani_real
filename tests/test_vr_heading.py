import contextlib
import io
import types
import unittest
from dataclasses import dataclass
from enum import Enum
from unittest.mock import patch

import numpy as np

from dexmani_real.config.sensor import VRConfig
from dexmani_real.sensor.vr import iter_vr_events
from dexmani_real.utils.vr_heading import collect_headings, estimate_heading, read_heading
from scripts.calibrate_vr_heading import main, wait_for_vr_tracking


@dataclass
class HeadFrame:
    head: object
    recv_ts_ns: int


@dataclass
class HandFrame:
    wrist: object
    sequence_id: int
    side: str = "right"


def pose(yaw_deg=0):
    theta = np.deg2rad(yaw_deg) / 2
    return types.SimpleNamespace(qx=0, qy=0, qz=np.sin(theta), qw=np.cos(theta))


def sdk_module():
    return types.SimpleNamespace(
        HeadFrame=HeadFrame,
        HandFrame=HandFrame,
        HandSide=types.SimpleNamespace(RIGHT="right", LEFT="left"),
        unity_left_to_flu_rotation=lambda x, y, z, w: (x, y, z, w),
    )


class VRHeadingTests(unittest.TestCase):
    def setUp(self):
        self.sdk = sdk_module()
        self.sdk_patch = patch.dict("sys.modules", hand_tracking_sdk=self.sdk)
        self.sdk_patch.start()
        self.addCleanup(self.sdk_patch.stop)

    def test_head_only_tracking_does_not_require_hand_or_position(self):
        frame = HeadFrame(pose(), 100)
        self.assertEqual(read_heading(frame, "head")[0], 100)
        self.assertIsNone(read_heading(frame, "wrist"))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(wait_for_vr_tracking(iter([None, frame]), "head", 1))

    def test_wrist_keeps_zero_sequence_and_ignores_other_events(self):
        self.assertEqual(read_heading(HandFrame(pose(), 0), "wrist")[0], 0)
        self.assertIsNone(read_heading(HandFrame(pose(), 2, "left"), "wrist"))
        self.assertIsNone(read_heading(HandFrame(pose(), 2), "head"))
        self.assertIsNone(read_heading(HeadFrame(pose(), 0), "head"))

    def test_bad_quaternion_is_rejected_at_input(self):
        bad = pose()
        bad.qw = np.nan
        with self.assertLogs("dexmani_real.utils.vr_heading", level="WARNING"):
            self.assertIsNone(read_heading(HeadFrame(bad, 7), "head"))

    def test_collection_deduplicates_source_frames(self):
        events = iter([
            HeadFrame(pose(25), 3), HeadFrame(pose(25), 3),
            HeadFrame(pose(25), 4), None,
        ])
        # 每帧各读两次时钟；最后一次读取越过采样期限。
        with patch("dexmani_real.utils.vr_heading.time.monotonic",
                   side_effect=[0, .1, .2, .3, .4, .5, .6, 2]), \
                contextlib.redirect_stdout(io.StringIO()):
            forwards, duplicates = collect_headings(events, "head", 1)
        self.assertEqual(forwards.shape, (2, 3))
        self.assertEqual(duplicates, 1)
        np.testing.assert_array_equal(forwards[0], forwards[1])

    def test_collection_stream_end_is_failure(self):
        with self.assertRaisesRegex(RuntimeError, "stopped during collection"):
            collect_headings(iter(()), "head", 1)

    def test_heading_math_is_unchanged(self):
        angle = np.deg2rad(37)
        directions = np.tile([np.cos(angle), np.sin(angle), 0], (35, 1))
        calibration = estimate_heading(directions, "head")
        self.assertAlmostEqual(calibration.theta_deg, 37, places=12)
        self.assertEqual(calibration.frames, 35)
        np.testing.assert_allclose(
            calibration.transform @ directions[0], [1, 0, 0], rtol=0, atol=2e-16,
        )

    def test_receiver_timeout_returns_control_and_closes(self):
        class Timeout(Exception):
            pass

        class Receiver:
            closed = False

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.closed = True

            def recv_line(self):
                raise Timeout

        receiver = Receiver()
        class TransportMode(Enum):
            UDP = "udp"
            TCP_SERVER = "tcp_server"
            TCP_CLIENT = "tcp_client"

        self.sdk.TransportMode = TransportMode
        for name in ("TCPClientConfig", "TCPServerConfig", "UDPReceiverConfig"):
            setattr(self.sdk, name, lambda *args, **kwargs: None)
        for name in ("TCPClientLineReceiver", "TCPServerLineReceiver", "UDPLineReceiver"):
            setattr(self.sdk, name, lambda *args: receiver)
        self.sdk.HandFrameAssembler = lambda **kwargs: None
        self.sdk.TransportTimeoutError = Timeout
        self.sdk.TransportDisconnectedError = type("Disconnected", (Exception,), {})
        self.sdk.ParseError = type("ParseError", (Exception,), {})
        self.sdk.parse_line = lambda line: line
        events = iter_vr_events(VRConfig(), running=lambda: True, include_head_frames=True)
        self.assertIsNone(next(events))
        self.assertFalse(receiver.closed)
        events.close()
        self.assertTrue(receiver.closed)

    def test_calibration_closes_direct_stream_on_interrupt(self):
        closed = []

        def events():
            try:
                yield HeadFrame(pose(), 3)
                raise KeyboardInterrupt
            finally:
                closed.append(True)

        with patch("scripts.calibrate_vr_heading.iter_vr_events", return_value=events()), \
                patch("dexmani_real.utils.log.configure_logging"), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                main(["--duration", "1"])
        self.assertEqual(closed, [True])


if __name__ == "__main__":
    unittest.main()
