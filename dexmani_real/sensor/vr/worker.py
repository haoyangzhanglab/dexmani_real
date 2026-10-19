"""HTS receiver with cancellable reads and right-hand observation publication."""

import logging
import time
from contextlib import closing

import numpy as np

from dexmani_real.sensor.vr.config import VRParams
from dexmani_real.planning.kinematics.pose import xyzw_to_wxyz
from dexmani_real.utils.geometry import normalize_quat_wxyz
from dexmani_real.utils.log import configure_logging

logger = logging.getLogger(__name__)


def finite_vector(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return array


def uint64(value):
    if not 0 <= value < 2**64:
        raise ValueError("VR timestamp/sequence is outside uint64 range")
    return np.uint64(value)


class VRWorker:
    """Own HTS reception and the cached head state used by hand observations."""

    def __init__(self, shared, config: VRParams):
        self.shared = shared
        self.config = config

    def run(self):
        configure_logging()
        from hand_tracking_sdk import (
            HandFrame,
            HandSide,
            HeadFrame,
            unity_left_to_flu_position,
            unity_left_to_flu_rotation,
        )

        self.head_pos = np.full(3, np.nan)
        self.head_quat_wxyz = np.full(4, np.nan)
        self.head_recv_ts_ns = 0
        try:
            with closing(self.iter_events()) as events:
                for event in events:
                    if not self.shared.is_running.value:
                        break
                    if isinstance(event, HeadFrame):
                        try:
                            position = finite_vector(
                                unity_left_to_flu_position(
                                    event.head.x, event.head.y, event.head.z
                                ),
                                (3,),
                                "head_pos",
                            )
                            rotation = normalize_quat_wxyz(
                                xyzw_to_wxyz(
                                    unity_left_to_flu_rotation(
                                        event.head.qx, event.head.qy, event.head.qz, event.head.qw
                                    )
                                ),
                                name="head_quat_wxyz",
                            )
                            stamp = uint64(event.recv_ts_ns)
                            if stamp == 0:
                                raise ValueError("head receive timestamp must be positive")
                        except ValueError:
                            logger.warning("Invalid VR head pose rejected", exc_info=True)
                            continue
                        self.head_pos, self.head_quat_wxyz = position, rotation
                        self.head_recv_ts_ns = stamp
                        continue
                    if not isinstance(event, HandFrame) or event.side != HandSide.RIGHT:
                        continue
                    try:
                        wrist = event.wrist
                        wrist_pos = finite_vector(
                            unity_left_to_flu_position(wrist.x, wrist.y, wrist.z), (3,), "wrist_pos"
                        )
                        wrist_quat = normalize_quat_wxyz(
                            xyzw_to_wxyz(
                                unity_left_to_flu_rotation(wrist.qx, wrist.qy, wrist.qz, wrist.qw)
                            ),
                            name="wrist_quat_wxyz",
                        )
                        landmarks = finite_vector(
                            [unity_left_to_flu_position(*p) for p in event.landmarks.points],
                            (21, 3),
                            "landmarks",
                        )
                        # Either HTS component can advance; the older receive time establishes freshness of both.
                        recv_ts_ns = min(
                            uint64(event.wrist_recv_ts_ns), uint64(event.landmarks_recv_ts_ns)
                        )
                        sequence_id = uint64(event.sequence_id)
                    except ValueError:
                        logger.warning("Invalid VR hand frame rejected", exc_info=True)
                        continue
                    # IPC errors are worker failures, not malformed sensor payloads.
                    self.shared.vr_ring.write(
                        wrist_pos=wrist_pos,
                        wrist_quat_wxyz=wrist_quat,
                        landmarks=landmarks,
                        head_pos=self.head_pos,
                        head_quat_wxyz=self.head_quat_wxyz,
                        head_recv_ts_ns=np.uint64(self.head_recv_ts_ns),
                        recv_ts_ns=recv_ts_ns,
                        sequence_id=sequence_id,
                    )
                    if not self.shared.vr_ready.is_set():
                        self.shared.vr_ready.set()
                        logger.info("VR ready: first valid right-hand frame published")
        finally:
            self.shared.vr_ready.clear()
        logger.info("VR receiver exited")

    def iter_events(self):
        from hand_tracking_sdk import (
            HandFrameAssembler,
            ParseError,
            TCPClientConfig,
            TCPClientLineReceiver,
            TCPServerConfig,
            TCPServerLineReceiver,
            TransportDisconnectedError,
            TransportMode,
            TransportTimeoutError,
            UDPLineReceiver,
            UDPReceiverConfig,
            parse_line,
        )

        mode = TransportMode(self.config.transport)
        if mode == TransportMode.UDP:
            receiver = UDPLineReceiver(
                UDPReceiverConfig(self.config.host, self.config.port, timeout_s=1.0)
            )
        elif mode == TransportMode.TCP_SERVER:
            receiver = TCPServerLineReceiver(
                TCPServerConfig(
                    self.config.host, self.config.port, accept_timeout_s=1.0, read_timeout_s=1.0
                )
            )
        else:
            receiver = TCPClientLineReceiver(
                TCPClientConfig(
                    self.config.host, self.config.port, connect_timeout_s=1.0, read_timeout_s=1.0
                )
            )
        # Filter hands at publication so heading calibration still receives head packets.
        assembler = HandFrameAssembler(include_wall_time=False, include_head_frames=True)
        if not self.shared.is_running.value:
            return
        with receiver:
            logger.info("VR receiver opened: %s port=%d", mode.value, self.config.port)
            # recv_line exposes timeouts so an idle/disconnected stream cannot hide shutdown.
            while self.shared.is_running.value:
                try:
                    line = receiver.recv_line()
                except TransportTimeoutError:
                    continue
                except TransportDisconnectedError:
                    if mode == TransportMode.TCP_CLIENT:
                        receiver.close()
                        if self.shared.is_running.value:
                            time.sleep(0.25)
                        if self.shared.is_running.value:
                            receiver.open()
                    continue
                if not self.shared.is_running.value:
                    break
                try:
                    packet = parse_line(line)
                except ParseError:
                    continue
                event = assembler.push_packet(packet)
                if event is not None:
                    yield event


def run_vr_worker(shared, config: VRParams) -> None:
    VRWorker(shared, config).run()
