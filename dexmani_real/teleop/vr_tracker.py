import av
import time
import math
import asyncio
import threading
import numpy as np
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

from hand_tracking_sdk import (
    ClientStats,
    HandFilter,
    HandFrame,
    HTSClient,
    HTSClientConfig,
    StreamOutput,
    TransportMode,
)
from hand_tracking_sdk.video.sources import VideoFormat
from hand_tracking_sdk.video.webrtc_sender import VideoWebRTCSender
from hand_tracking_sdk.video.service import VideoService, VideoServiceConfig
from hand_tracking_sdk.convert import BASIS_UNITY_LEFT_TO_FLU, basis_transform_position, basis_transform_rotation

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # (qw, qx, qy, qz)
IDENTITY_QUAT: Quat = (1.0, 0.0, 0.0, 0.0)
FLU_BASIS = BASIS_UNITY_LEFT_TO_FLU

HAND_JOINT_NAMES = (
    "Wrist", "ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip",
    "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip",
    "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip",
    "RingProximal", "RingIntermediate", "RingDistal", "RingTip",
    "LittleProximal", "LittleIntermediate", "LittleDistal", "LittleTip",
)

FLU_TO_OPERATOR_RH = ((0, 0, -1), (0, 1, 0), (1, 0, 0))
FLU_TO_OPERATOR_LH = ((0, 0, -1), (0, -1, 0), (1, 0, 0))


@dataclass(frozen=True, slots=True)
class HandData:

    side: str
    sequence_id: int
    wrist_pos: Vec3
    wrist_quat: Quat
    landmarks: tuple[Vec3, ...]
    source_ts_ns: int | None
    recv_ts_ns: int
    source_frame_seq: int | None

    def to_dict(self):
        return {
            "side": self.side,
            "sequence_id": self.sequence_id,
            "wrist_pos": list(self.wrist_pos),
            "wrist_quat": list(self.wrist_quat),
            "landmarks": [list(p) for p in self.landmarks],
            "source_ts_ns": self.source_ts_ns,
            "recv_ts_ns": self.recv_ts_ns,
            "source_frame_seq": self.source_frame_seq,
        }

    def wrist_relative_landmarks(self) -> tuple[Vec3, ...]:
        return wrist_relative_landmarks(self.landmarks, self.wrist_pos)

    def landmarks_for_retargeting(self):
        return landmarks_for_retargeting(self.landmarks, self.wrist_pos, self.side)


@dataclass(frozen=True, slots=True)
class RobotTarget:
    position: Vec3
    orientation: Quat
    delta_pos: Vec3
    delta_quat: Quat
    grip: float | None = None


def vec_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def vec_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def vec_scale(v: Vec3, s: float) -> Vec3:
    return (v[0] * s, v[1] * s, v[2] * s)

def vec_norm(v: Vec3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

def quat_mul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )

def quat_conjugate(q: Quat) -> Quat:
    return (q[0], -q[1], -q[2], -q[3])

def quat_inverse(q: Quat) -> Quat:
    return quat_conjugate(q)

def quat_dot(a: Quat, b: Quat) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

def quat_norm(q: Quat) -> float:
    return math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])

def quat_normalize(q: Quat, fallback: Quat = IDENTITY_QUAT) -> Quat:
    n = quat_norm(q)
    if n < 1e-12:
        return fallback
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)

def quat_match_hemisphere(q: Quat, reference: Quat) -> Quat:
    return (-q[0], -q[1], -q[2], -q[3]) if quat_dot(q, reference) < 0.0 else q

def quat_angle(q: Quat) -> float:
    qw, qx, qy, qz = quat_normalize(q)
    return 2.0 * math.atan2(vec_norm((qx, qy, qz)), abs(qw))

def quat_slerp_from_identity(q: Quat, t: float) -> Quat:
    qw, qx, qy, qz = quat_normalize(q)
    if qw < 0.0:
        qw, qx, qy, qz = -qw, -qx, -qy, -qz
    qw = max(-1.0, min(1.0, qw))
    angle = 2.0 * math.acos(qw)
    if angle < 1e-6:
        return IDENTITY_QUAT
    sin_half = math.sqrt(max(0.0, 1.0 - qw * qw))
    if sin_half < 1e-6:
        return IDENTITY_QUAT
    half_scaled = 0.5 * angle * t
    scale = math.sin(half_scaled) / sin_half
    return quat_normalize((math.cos(half_scaled), qx * scale, qy * scale, qz * scale))

def quat_rotate_vector(q: Quat, v: Vec3) -> Vec3:
    rotated = quat_mul(quat_mul(quat_normalize(q), (0.0, v[0], v[1], v[2])), quat_inverse(quat_normalize(q)))
    return (rotated[1], rotated[2], rotated[3])

def change_basis_rotation(delta_quat: Quat, target_from_source_quat: Quat) -> Quat:
    t_from_s = quat_normalize(target_from_source_quat)
    return quat_normalize(quat_mul(quat_mul(t_from_s, delta_quat), quat_inverse(t_from_s)))

def unity_lh_to_flu_position(x: float, y: float, z: float) -> Vec3:
    return basis_transform_position((x, y, z), FLU_BASIS)

def unity_lh_to_flu_quaternion(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float, float]:
    return basis_transform_rotation(qx, qy, qz, qw, FLU_BASIS)

def wrist_relative_landmarks(landmarks: tuple[Vec3, ...], wrist_pos: Vec3) -> tuple[Vec3, ...]:
    return tuple(vec_sub(p, wrist_pos) for p in landmarks)



def landmarks_for_retargeting(landmarks: tuple[Vec3, ...], wrist_pos: Vec3, side: str = "right"):

    basis = FLU_TO_OPERATOR_RH if side.lower() == "right" else FLU_TO_OPERATOR_LH
    relative = wrist_relative_landmarks(landmarks, wrist_pos)
    result = np.zeros((21, 3), dtype=np.float64)
    for i, (px, py, pz) in enumerate(relative):
        result[i, 0] = basis[0][0] * px + basis[0][1] * py + basis[0][2] * pz
        result[i, 1] = basis[1][0] * px + basis[1][1] * py + basis[1][2] * pz
        result[i, 2] = basis[2][0] * px + basis[2][1] * py + basis[2][2] * pz
    return result


def hand_frame_to_data(frame: HandFrame) -> HandData:
    pos = unity_lh_to_flu_position(frame.wrist.x, frame.wrist.y, frame.wrist.z)
    quat_xyzw = unity_lh_to_flu_quaternion(frame.wrist.qx, frame.wrist.qy, frame.wrist.qz, frame.wrist.qw)
    quat_wxyz = quat_normalize((quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]))
    landmarks_flu = tuple(unity_lh_to_flu_position(p[0], p[1], p[2]) for p in frame.landmarks.points)
    return HandData(
        side=frame.side.value.lower(),
        sequence_id=frame.sequence_id,
        wrist_pos=pos,
        wrist_quat=quat_wxyz,
        landmarks=landmarks_flu,
        source_ts_ns=frame.source_ts_ns,
        recv_ts_ns=frame.recv_ts_ns,
        source_frame_seq=frame.source_frame_seq,
    )


class FrameQueueSource:
    """Async video source backed by a small frame queue."""
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.queue: deque[bytes] = deque(maxlen=120)
        self.stopped = False

    def get_format(self):
        return VideoFormat(width=self.width, height=self.height, fps=self.fps)

    async def start(self):
        return None

    async def stop(self):
        self.stopped = True

    def push_frame(self, rgb_array):
        self.queue.append(rgb_array.tobytes())

    async def next_frame(self):
        while not self.queue and not self.stopped:
            await asyncio.sleep(0.002)

        if self.stopped and not self.queue:
            black = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return av.VideoFrame.from_ndarray(black, format="rgb24")

        rgb = np.frombuffer(self.queue.popleft(), dtype=np.uint8).reshape(self.height, self.width, 3)
        return av.VideoFrame.from_ndarray(rgb, format="rgb24")


class EWMAFilter:
    """Simple EWMA for Euclidean vectors."""
    def __init__(self, alpha: float, dim: int):
        self.alpha = alpha
        self.dim = dim
        self.value: list[float] | None = None

    def update(self, raw: tuple[float, ...]) -> tuple[float, ...]:
        if self.value is None:
            self.value = list(raw)
            return tuple(self.value)
        for i in range(self.dim):
            self.value[i] = self.alpha * raw[i] + (1.0 - self.alpha) * self.value[i]
        return tuple(self.value)

    def reset(self):
        self.value = None


class QuaternionEWMAFilter:
    """Quaternion smoothing with hemisphere alignment + normalize."""
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value: Quat | None = None

    def update(self, raw: Quat) -> Quat:
        raw_q = quat_normalize(raw)
        if self.value is None:
            self.value = raw_q
            return self.value
        aligned = quat_match_hemisphere(raw_q, self.value)
        blended = (
            self.alpha * aligned[0] + (1.0 - self.alpha) * self.value[0],
            self.alpha * aligned[1] + (1.0 - self.alpha) * self.value[1],
            self.alpha * aligned[2] + (1.0 - self.alpha) * self.value[2],
            self.alpha * aligned[3] + (1.0 - self.alpha) * self.value[3],
        )
        self.value = quat_normalize(blended, fallback=self.value)
        return self.value

    def reset(self):
        self.value = None


class QuestHandReceiver:
    """Quest hand receiver with FLU hand output and world-aligned robot targets.

    `stream()` always yields hand data in receiver FLU.
    `compute_robot_target()` always returns targets in the configured world frame.

    When world == FLU, use identity alignment:
        world x = forward
        world y = left
        world z = up
    """
    def __init__(
        self,
        *,
        port: int = 8000,
        host: str = "0.0.0.0",
        hand_filter: str = "both",
        timeout_s: float = 1.0,
        smooth_alpha: float = 1.0,
        max_delta_pos: float = 0.0,
        max_delta_rot: float = 0.0,
        workspace_limits: tuple[float, float, float, float, float, float] | None = None,
    ):
        filter_map = {"both": HandFilter.BOTH, "left": HandFilter.LEFT, "right": HandFilter.RIGHT}
        filt = filter_map.get(hand_filter.lower())
        if filt is None:
            raise ValueError(f"hand_filter must be one of {list(filter_map.keys())}, got {hand_filter!r}")

        self.client = HTSClient(
            HTSClientConfig(
                transport_mode=TransportMode.TCP_SERVER,
                host=host,
                port=port,
                timeout_s=timeout_s,
                output=StreamOutput.FRAMES,
                hand_filter=filt,
                error_policy="tolerant",
            )
        )

        self.pos_filter: EWMAFilter | None = None
        self.quat_filter: QuaternionEWMAFilter | None = None
        if 0.0 < smooth_alpha < 1.0:
            self.pos_filter = EWMAFilter(alpha=smooth_alpha, dim=3)
            self.quat_filter = QuaternionEWMAFilter(alpha=smooth_alpha)

        self.max_delta_pos = max_delta_pos
        self.max_delta_rot = max_delta_rot
        self.workspace_limits = workspace_limits

        self.ref_hand_pos: Vec3 | None = None
        self.ref_hand_quat: Quat | None = None
        self.robot_home_pos: Vec3 | None = None
        self.robot_home_quat: Quat | None = None
        self.world_from_flu_quat: Quat = IDENTITY_QUAT

        self.last_output_pos: Vec3 | None = None
        self.last_output_quat: Quat | None = None

        self.video_service = None
        self.video_loop: asyncio.AbstractEventLoop | None = None
        self.video_thread: threading.Thread | None = None
        self.frame_source: FrameQueueSource | None = None

    def __enter__(self) -> "QuestHandReceiver":
        return self

    def __exit__(self, *exc: object):
        self.stop()

    def stop(self):
        self.stop_video_service()

    def stream(self) -> Iterator[HandData]:
        for event in self.client.iter_events():
            if isinstance(event, HandFrame):
                yield hand_frame_to_data(event)

    def set_world_alignment(self, world_from_flu_quat: Quat):
        self.world_from_flu_quat = quat_normalize(world_from_flu_quat)

    def set_world_alignment_yaw(self, yaw_rad: float):
        half = 0.5 * yaw_rad
        self.set_world_alignment((math.cos(half), 0.0, 0.0, math.sin(half)))

    def set_robot_reference(self, robot_home_pos: Vec3, robot_home_quat: Quat):
        self.robot_home_pos = robot_home_pos
        self.robot_home_quat = quat_normalize(robot_home_quat)
        self.ref_hand_pos = None
        self.ref_hand_quat = None
        self._reset_tracking_state()

    def capture_hand_reference(self, data: HandData):
        self.ref_hand_pos = data.wrist_pos
        self.ref_hand_quat = quat_normalize(data.wrist_quat)
        self._reset_tracking_state()

    def calibrate(self, data: HandData, robot_home_pos: Vec3, robot_home_quat: Quat):
        self.set_robot_reference(robot_home_pos, robot_home_quat)
        self.capture_hand_reference(data)

    def calibrate_world(
        self,
        data: HandData,
        robot_home_pos: Vec3,
        robot_home_quat: Quat,
        world_from_flu_quat: Quat | None = None,
    ):
        if world_from_flu_quat is not None:
            self.set_world_alignment(world_from_flu_quat)
        self.calibrate(data, robot_home_pos, robot_home_quat)

    def set_workspace_limits(self, limits: tuple[float, float, float, float, float, float] | None = None):
        self.workspace_limits = limits

    def _reset_tracking_state(self):
        self.last_output_pos = None
        self.last_output_quat = None
        if self.pos_filter is not None:
            self.pos_filter.reset()
        if self.quat_filter is not None:
            self.quat_filter.reset()

    def _smoothed_hand_pose(self, data: HandData) -> tuple[Vec3, Quat]:
        pos = data.wrist_pos if self.pos_filter is None else self.pos_filter.update(data.wrist_pos)
        quat = quat_normalize(data.wrist_quat) if self.quat_filter is None else self.quat_filter.update(data.wrist_quat)
        return pos, quat

    def _world_delta_from_hand(self, pos: Vec3, quat: Quat) -> tuple[Vec3, Quat]:
        assert self.ref_hand_pos is not None and self.ref_hand_quat is not None

        dp_flu = vec_sub(pos, self.ref_hand_pos)
        dp_world = quat_rotate_vector(self.world_from_flu_quat, dp_flu)

        dq_flu = quat_normalize(quat_mul(quat_inverse(self.ref_hand_quat), quat))
        dq_world = change_basis_rotation(dq_flu, self.world_from_flu_quat)
        return dp_world, dq_world

    def _limit_position_step(self, robot_pos: Vec3) -> Vec3:
        if self.last_output_pos is None or self.max_delta_pos <= 0.0:
            return robot_pos
        step = vec_sub(robot_pos, self.last_output_pos)
        dist = vec_norm(step)
        if dist <= self.max_delta_pos or dist < 1e-12:
            return robot_pos
        return vec_add(self.last_output_pos, vec_scale(step, self.max_delta_pos / dist))

    def _limit_rotation_step(self, robot_quat: Quat) -> Quat:
        if self.last_output_quat is None or self.max_delta_rot <= 0.0:
            return robot_quat
        delta_q = quat_normalize(quat_mul(quat_inverse(self.last_output_quat), robot_quat))
        delta_rot = quat_angle(delta_q)
        if delta_rot <= self.max_delta_rot or delta_rot < 1e-12:
            return robot_quat
        limited_delta = quat_slerp_from_identity(delta_q, self.max_delta_rot / delta_rot)
        return quat_normalize(quat_mul(self.last_output_quat, limited_delta))

    def _clamp_workspace(self, robot_pos: Vec3) -> Vec3:
        if self.workspace_limits is None or self.robot_home_pos is None:
            return robot_pos
        min_x, max_x, min_y, max_y, min_z, max_z = self.workspace_limits
        home = self.robot_home_pos
        delta = vec_sub(robot_pos, home)
        clamped = (
            max(min_x, min(max_x, delta[0])),
            max(min_y, min(max_y, delta[1])),
            max(min_z, min(max_z, delta[2])),
        )
        return vec_add(home, clamped)

    def compute_robot_target(self, data: HandData) -> RobotTarget | None:
        if None in (self.robot_home_pos, self.robot_home_quat, self.ref_hand_pos, self.ref_hand_quat):
            return None

        assert self.robot_home_pos is not None and self.robot_home_quat is not None

        hand_pos, hand_quat = self._smoothed_hand_pose(data)
        delta_pos_world, delta_quat_world = self._world_delta_from_hand(hand_pos, hand_quat)

        robot_pos = vec_add(self.robot_home_pos, delta_pos_world)
        robot_quat = quat_normalize(quat_mul(self.robot_home_quat, delta_quat_world))

        robot_pos = self._limit_position_step(robot_pos)
        robot_quat = self._limit_rotation_step(robot_quat)
        robot_pos = self._clamp_workspace(robot_pos)

        delta_pos_world = vec_sub(robot_pos, self.robot_home_pos)
        delta_quat_world = quat_normalize(quat_mul(quat_inverse(self.robot_home_quat), robot_quat))

        self.last_output_pos = robot_pos
        self.last_output_quat = robot_quat

        return RobotTarget(
            position=robot_pos,
            orientation=robot_quat,
            delta_pos=delta_pos_world,
            delta_quat=delta_quat_world,
        )

    def start_video_service(
        self,
        *,
        video_port: int = 8765,
        preset: str = "720p",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ):
        if self.video_service is not None:
            return
        self.frame_source = FrameQueueSource(width=width, height=height, fps=fps)

        def sender_factory(source, session_id: str, fps_val: int):
            _ = (source, session_id, fps_val)
            return VideoWebRTCSender(source=self.frame_source)

        self.video_service = VideoService(
            VideoServiceConfig(
                signaling_host="0.0.0.0",
                signaling_port=video_port,
                source="test",
                preset=preset,
                verbose=False,
            ),
            sender_factory=sender_factory,
        )

        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.video_loop = loop
            loop.run_until_complete(self.video_service.start())
            loop.run_forever()

        self.video_thread = threading.Thread(target=run_loop, daemon=True)
        self.video_thread.start()
        while self.video_loop is None:
            time.sleep(0.05)

    def stop_video_service(self):
        if self.video_service is None:
            return
        if self.video_loop is not None:
            fut = asyncio.run_coroutine_threadsafe(self.video_service.stop(), self.video_loop)
            fut.result(timeout=5)
            self.video_loop.call_soon_threadsafe(self.video_loop.stop)
        if self.video_thread is not None and self.video_thread.is_alive():
            self.video_thread.join(timeout=5)
        self.video_service = None
        self.video_loop = None
        self.video_thread = None
        self.frame_source = None

    def push_video_frame(self, rgb_array):
        if self.frame_source is not None:
            self.frame_source.push_frame(rgb_array)

    def stats(self) -> ClientStats:
        return self.client.get_stats()

    def reset_stats(self):
        self.client.reset_stats()