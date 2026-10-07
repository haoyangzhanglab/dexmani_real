"""HTS receiver with cancellable reads and right-hand observation publication."""

import time
from contextlib import closing

import numpy as np

from dexmani_real.config.hardware import VRParams
from dexmani_real.ipc.schema import VR_FRAME_DTYPE
from dexmani_real.planning.kinematics.pose import xyzw_to_wxyz
from dexmani_real.utils.geometry import normalize_quat_wxyz
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


def _finite_vector(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return array.copy()


def _uint64(value):
    if not 0 <= value < 2**64:
        raise ValueError("VR timestamp/sequence is outside uint64 range")
    return np.uint64(value)


def _iter_events(shared, config):
    from hand_tracking_sdk import (
        HandFilter,
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

    mode, hand_filter = TransportMode(config.transport), HandFilter(config.hand_side)
    if mode == TransportMode.UDP:
        receiver = UDPLineReceiver(UDPReceiverConfig(config.host, config.port, timeout_s=1.0))
    elif mode == TransportMode.TCP_SERVER:
        receiver = TCPServerLineReceiver(
            TCPServerConfig(config.host, config.port, accept_timeout_s=1.0, read_timeout_s=1.0)
        )
    else:
        receiver = TCPClientLineReceiver(
            TCPClientConfig(config.host, config.port, connect_timeout_s=1.0, read_timeout_s=1.0)
        )
    assembler = HandFrameAssembler(include_wall_time=True, include_head_frames=True)
    if not shared.is_running.value:
        return
    with receiver:
        logger.info("VR receiver opened: %s port=%d", mode.value, config.port)
        # SDK iter_events()/iter_lines() swallow timeouts internally; use recv_line()
        # so an idle or disconnected stream cannot hide the shared shutdown request.
        while shared.is_running.value:
            try:
                line = receiver.recv_line()
            except TransportTimeoutError:
                continue
            except TransportDisconnectedError:
                if mode == TransportMode.TCP_CLIENT:
                    receiver.close()
                    # Preserve the SDK's existing reconnect delay and failure propagation.
                    if shared.is_running.value:
                        time.sleep(0.25)
                    if shared.is_running.value:
                        receiver.open()
                continue
            if not shared.is_running.value:
                break
            try:
                packet = parse_line(line)
            except ParseError:
                continue
            if hand_filter != HandFilter.BOTH and packet.side.value.lower() != hand_filter.value:
                continue
            event = assembler.push_packet(packet)
            if event is not None:
                yield event


def run_vr_worker(shared, config: VRParams) -> None:
    config.validate()
    if shared.vr_ring is None:
        raise ValueError("VR worker requires an allocated VR ring")
    from hand_tracking_sdk import (
        HandFrame,
        HandSide,
        HeadFrame,
        unity_left_to_flu_position,
        unity_left_to_flu_rotation,
    )

    head_pos = np.full(3, np.nan)
    head_quat_wxyz = np.full(4, np.nan)
    head_sequence_id = head_recv_ts_ns = 0
    try:
        with closing(_iter_events(shared, config)) as events:
            for event in events:
                if not shared.is_running.value:
                    break
                if isinstance(event, HeadFrame):
                    try:
                        position = _finite_vector(
                            unity_left_to_flu_position(event.head.x, event.head.y, event.head.z),
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
                        sequence, stamp = _uint64(event.sequence_id), _uint64(event.recv_ts_ns)
                        if stamp == 0:
                            raise ValueError("head receive timestamp must be positive")
                    except ValueError:
                        logger.warning("Invalid VR head pose rejected", exc_info=True)
                        continue
                    head_pos, head_quat_wxyz = position, rotation
                    head_sequence_id, head_recv_ts_ns = sequence, stamp
                    continue
                if not isinstance(event, HandFrame) or event.side != HandSide.RIGHT:
                    continue
                try:
                    wrist = event.wrist
                    wrist_pos = _finite_vector(
                        unity_left_to_flu_position(wrist.x, wrist.y, wrist.z), (3,), "wrist_pos"
                    )
                    wrist_quat = normalize_quat_wxyz(
                        xyzw_to_wxyz(
                            unity_left_to_flu_rotation(wrist.qx, wrist.qy, wrist.qz, wrist.qw)
                        ),
                        name="wrist_quat_wxyz",
                    )
                    landmarks = _finite_vector(
                        [unity_left_to_flu_position(*p) for p in event.landmarks.points],
                        (21, 3),
                        "landmarks",
                    )
                    frame = np.zeros(1, dtype=VR_FRAME_DTYPE)
                    frame["wrist_pos"][0] = wrist_pos
                    frame["wrist_quat_wxyz"][0] = wrist_quat
                    frame["landmarks"][0] = landmarks
                    frame["head_pos"][0] = head_pos
                    frame["head_quat_wxyz"][0] = head_quat_wxyz
                    frame["head_sequence_id"][0] = head_sequence_id
                    frame["head_recv_ts_ns"][0] = head_recv_ts_ns
                    # HTS emits when either component advances; its recv_ts_ns is
                    # the newer timestamp and cannot establish freshness of both.
                    frame["recv_ts_ns"][0] = min(
                        _uint64(event.wrist_recv_ts_ns), _uint64(event.landmarks_recv_ts_ns)
                    )
                    frame["source_ts_ns"][0] = _uint64(event.source_ts_ns or 0)
                    frame["sequence_id"][0] = _uint64(event.sequence_id)
                    frame["source_frame_seq"][0] = _uint64(event.source_frame_seq or 0)
                    frame["side"][0] = 0
                except ValueError:
                    logger.warning("Invalid VR hand frame rejected", exc_info=True)
                    continue
                # IPC errors are worker failures, not malformed sensor payloads.
                shared.vr_ring.write(frame)
                if not shared.vr_ready.is_set():
                    shared.vr_ready.set()
                    logger.info("VR ready: first valid right-hand frame published")
    finally:
        shared.vr_ready.clear()
    logger.info("VR receiver exited")
