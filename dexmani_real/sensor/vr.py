import logging
import time
from contextlib import closing

import numpy as np

from dexmani_real.config.sensor import VRConfig
from dexmani_real.geometry.pose import normalize_quat_wxyz, xyzw_to_wxyz
from dexmani_real.utils.log import configure_logging

logger = logging.getLogger(__name__)


def finite_vector(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return array


def _pose_to_flu(pose):
    from hand_tracking_sdk import unity_left_to_flu_position

    position = finite_vector(unity_left_to_flu_position(pose.x, pose.y, pose.z), (3,), "position")
    return position, _rotation_to_flu(pose)


def _rotation_to_flu(pose):
    from hand_tracking_sdk import unity_left_to_flu_rotation

    return normalize_quat_wxyz(
        xyzw_to_wxyz(unity_left_to_flu_rotation(pose.qx, pose.qy, pose.qz, pose.qw)),
        name="quaternion",
    )


def _landmarks_to_flu(points):
    from hand_tracking_sdk import unity_left_to_flu_position

    return finite_vector(
        [unity_left_to_flu_position(*point) for point in points], (21, 3), "landmarks"
    )


def run_vr(shared, config: VRConfig) -> None:
    """接收 HTS 右手事件并发布控制观测。"""
    configure_logging()
    from hand_tracking_sdk import HandFrame, HandSide

    try:
        with closing(iter_vr_events(config, running=lambda: shared.is_running.value)) as events:
            for event in events:
                if not shared.is_running.value:
                    break
                if not isinstance(event, HandFrame) or event.side != HandSide.RIGHT:
                    continue
                try:
                    wrist_pos, wrist_quat = _pose_to_flu(event.wrist)
                    landmarks = _landmarks_to_flu(event.landmarks.points)
                except ValueError:
                    logger.warning("Invalid VR hand frame rejected", exc_info=True)
                    continue
                # IPC 写入失败属于 worker 故障，不能当作坏包跳过。
                shared.vr_ring.write(
                    wrist_pos=wrist_pos,
                    wrist_quat_wxyz=wrist_quat,
                    landmarks=landmarks,
                    recv_ts_ns=min(event.wrist_recv_ts_ns, event.landmarks_recv_ts_ns),
                    sequence_id=event.sequence_id,
                )
                if not shared.vr_ready.is_set():
                    shared.vr_ready.set()
                    logger.info("VR ready: first valid right-hand frame published")
    finally:
        shared.vr_ready.clear()
    logger.info("VR receiver exited")


def iter_vr_events(config: VRConfig, *, running, include_head_frames=False):
    """每次有限读取后返回事件或 None，让调用方检查自己的期限与退出条件。"""
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

    mode = TransportMode(config.transport)
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
    assembler = HandFrameAssembler(
        include_wall_time=False, include_head_frames=include_head_frames,
    )
    if not running():
        return
    with receiver:
        logger.info("VR receiver opened: %s port=%d", mode.value, config.port)
        # 使用有期限的单次读取，空闲连接也能响应退出。
        while running():
            try:
                line = receiver.recv_line()
            except TransportTimeoutError:
                yield None
                continue
            except TransportDisconnectedError:
                if mode == TransportMode.TCP_CLIENT:
                    receiver.close()
                    if running():
                        time.sleep(0.25)
                    if running():
                        receiver.open()
                yield None
                continue
            if not running():
                break
            try:
                packet = parse_line(line)
            except ParseError:
                yield None
                continue
            event = assembler.push_packet(packet)
            yield event
