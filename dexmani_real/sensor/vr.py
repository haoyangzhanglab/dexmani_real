from __future__ import annotations

import logging
import time
from collections.abc import Generator
from contextlib import closing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from dexmani_real.config.sensor import VRConfig
from dexmani_real.geometry.pose import normalize_quat_wxyz, xyzw_to_wxyz
from dexmani_real.utils.log import configure_logging

if TYPE_CHECKING:
    from hand_tracking_sdk import (
        HandLandmarks,
        HeadPose,
        TransportDisconnectedError,
        WristPose,
    )

logger = logging.getLogger(__name__)


@dataclass
class VRReceiveStatus:
    lines_received: int = 0
    parse_errors: int = 0
    transport_message: str | None = None


@dataclass(frozen=True)
class VRSample:
    pose: WristPose | HeadPose
    pose_recv_ts_ns: int
    sequence_id: int
    landmarks: HandLandmarks | None = None
    landmarks_recv_ts_ns: int = 0
    landmarks_sequence_id: int = 0


def _finite_vector(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return array


def _pose_to_flu(pose):
    from hand_tracking_sdk import unity_left_to_flu_position

    position = _finite_vector(unity_left_to_flu_position(pose.x, pose.y, pose.z), (3,), "position")
    return position, rotation_to_flu(pose)


def rotation_to_flu(pose):
    from hand_tracking_sdk import unity_left_to_flu_rotation

    # SDK 基变换会把零四元数变成单位姿态，必须先检查原始输入。
    qw, qx, qy, qz = normalize_quat_wxyz((pose.qw, pose.qx, pose.qy, pose.qz))
    return xyzw_to_wxyz(unity_left_to_flu_rotation(qx, qy, qz, qw))


def _landmarks_to_flu(points):
    from hand_tracking_sdk import unity_left_to_flu_position

    return _finite_vector(
        [unity_left_to_flu_position(*point) for point in points], (21, 3), "landmarks"
    )


def run_vr(shared, config: VRConfig, *, hand_enabled: bool) -> None:
    """接收 HTS 右手事件并发布控制观测。"""
    configure_logging()
    from hand_tracking_sdk import TransportDisconnectedError

    source: Literal["hand", "wrist"] = "hand" if hand_enabled else "wrist"
    empty_landmarks = np.zeros((21, 3))
    with closing(iter_vr_events(config, source=source, running=lambda: shared.is_running.value)) as events:
        for event in events:
            if not shared.is_running.value:
                break
            if isinstance(event, TransportDisconnectedError):
                shared.vr_connection_epoch.value += 1
                logger.warning("VR transport disconnected: %s", event)
                continue
            if event is None:
                continue
            try:
                wrist_pos, wrist_quat = _pose_to_flu(event.pose)
                landmarks = empty_landmarks if event.landmarks is None else _landmarks_to_flu(event.landmarks.points)
            except ValueError as exc:
                logger.warning("Invalid VR hand frame rejected: %s", exc)
                continue
            recv_ts_ns = event.pose_recv_ts_ns
            if event.landmarks is not None:
                recv_ts_ns = min(recv_ts_ns, event.landmarks_recv_ts_ns)
            # IPC 写入失败属于 worker 故障，不能当作坏包跳过。
            shared.vr_ring.write(
                wrist_pos=wrist_pos,
                wrist_quat_wxyz=wrist_quat,
                landmarks=landmarks,
                recv_ts_ns=recv_ts_ns,
                wrist_recv_ts_ns=event.pose_recv_ts_ns,
                landmarks_recv_ts_ns=event.landmarks_recv_ts_ns,
                sequence_id=event.sequence_id,
                landmarks_sequence_id=event.landmarks_sequence_id,
                connection_epoch=shared.vr_connection_epoch.value,
            )
    logger.info("VR receiver exited")


def iter_vr_events(
    config: VRConfig, *, source: Literal["hand", "wrist", "head"], running,
    status: VRReceiveStatus | None = None,
) -> Generator[VRSample | TransportDisconnectedError | None, None, None]:
    """每次有限读取后返回采样、断开或 None，保留数据首次接收时刻。"""
    from hand_tracking_sdk import (
        HandFrameAssembler,
        HandSide,
        HeadPosePacket,
        LandmarksPacket,
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
        WristPacket,
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
    assembler = HandFrameAssembler(include_wall_time=False) if source == "hand" else None
    sequence_id = landmarks_sequence_id = 0
    if not running():
        return
    with receiver:
        logger.info("VR receiver opened: %s %s:%d", mode.value, config.host, config.port)
        # 使用有期限的单次读取，空闲连接也能响应退出。
        while running():
            try:
                line = receiver.recv_line()
                recv_ts_ns = receiver.last_recv_ts_ns
            except TransportTimeoutError as exc:
                if status is not None:
                    status.transport_message = str(exc)
                yield None
                continue
            except TransportDisconnectedError as exc:
                if status is not None:
                    status.transport_message = str(exc)
                if assembler is not None:
                    assembler.reset()
                # 先通知调用方撤销旧连接，再尝试客户端重连。
                yield exc
                if mode == TransportMode.TCP_CLIENT:
                    receiver.close()
                    if running():
                        time.sleep(0.25)
                    if running():
                        receiver.open()
                continue
            if not running():
                break
            if status is not None:
                status.lines_received += 1
                status.transport_message = None
            try:
                packet = parse_line(line)
            except ParseError:
                if status is not None:
                    status.parse_errors += 1
                yield None
                continue
            if source == "head":
                if isinstance(packet, HeadPosePacket):
                    sequence_id += 1
                    yield VRSample(packet.data, recv_ts_ns, sequence_id)
                else:
                    yield None
            elif packet.side != HandSide.RIGHT:
                yield None
            elif assembler is None:
                if isinstance(packet, WristPacket):
                    sequence_id += 1
                    yield VRSample(packet.data, recv_ts_ns, sequence_id)
                else:
                    yield None
            else:
                # 同一个 TCP 块内的不同关键点包可以具有相同接收时间。
                if isinstance(packet, LandmarksPacket):
                    landmarks_sequence_id += 1
                frame = assembler.push_packet(packet, recv_ts_ns=recv_ts_ns)
                if frame is None:
                    yield None
                    continue
                sequence_id += 1
                yield VRSample(
                    frame.wrist, frame.wrist_recv_ts_ns, sequence_id, frame.landmarks,
                    frame.landmarks_recv_ts_ns, landmarks_sequence_id,
                )
