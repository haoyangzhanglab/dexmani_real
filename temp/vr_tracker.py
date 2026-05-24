"""Quest hand-tracking runtime for robot teleoperation.

Design goals
------------
- Keep VR tracking, video return, hand retargeting, IK, and robot drivers
  separated.
- Run VR tracking in a background latest-state service.
- Return plain Python / NumPy data for downstream workflows.
- Use wxyz quaternions internally and for EEF targets.

Coordinate split
----------------
- Arm / EEF branch:
    wrist pose -> LeFranX-style differential SE(3) intent -> EEF target.
- Hand branch:
    21 landmarks -> wrist-relative skeleton / finger curl -> external hand
    retargeting.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from hand_tracking_sdk import (
    ClientStats,
    ErrorPolicy,
    HandFilter,
    HandFrame,
    HTSClient,
    HTSClientConfig,
    StreamOutput,
    TransportMode,
)
from hand_tracking_sdk.convert import (
    BASIS_UNITY_LEFT_TO_FLU,
    basis_transform_position,
    basis_transform_rotation,
)

from vr_utils import (
    IDENTITY_QUAT,
    Quat,
    Vec3,
    change_basis_rotation,
    clamp_relative_orientation,
    hand_data_to_finger_curl_vector,
    hand_data_to_retargeting_joints,
    matrix_to_quat_wxyz,
    quat_angle,
    quat_inverse,
    quat_match_hemisphere,
    quat_mul,
    quat_normalize,
    quat_rotate_vector,
    quat_slerp_from_identity,
    quat_wxyz_to_matrix,
    quat_wxyz_to_xyzw,
    vec_add,
    vec_norm,
    vec_scale,
    vec_sub,
)

FLU_BASIS = BASIS_UNITY_LEFT_TO_FLU


def load_video_backend():
    """Import optional video dependencies only when video return is used."""
    try:
        import av
        from hand_tracking_sdk.video.service import VideoService, VideoServiceConfig
        from hand_tracking_sdk.video.sources import VideoFormat
        from hand_tracking_sdk.video.webrtc_sender import VideoWebRTCSender
    except BaseException as exc:
        raise ImportError("Video return requires hand-tracking-sdk[video] and PyAV") from exc
    return av, VideoService, VideoServiceConfig, VideoFormat, VideoWebRTCSender


def unity_lh_to_flu_position(x: float, y: float, z: float) -> Vec3:
    return basis_transform_position((x, y, z), FLU_BASIS)


def unity_lh_to_flu_quaternion(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float, float]:
    return basis_transform_rotation(qx, qy, qz, qw, FLU_BASIS)


def normalize_side(side: Any) -> str:
    value = getattr(side, "value", side)
    return str(value).lower()


def validate_limits(limits: tuple[float, float, float, float, float, float] | None, *, name: str):
    if limits is None:
        return
    if len(limits) != 6:
        raise ValueError(f"{name} must have 6 values")
    for low, high in zip(limits[0::2], limits[1::2], strict=True):
        if low > high:
            raise ValueError(f"each {name} min must be <= max")


@dataclass(frozen=True, slots=True)
class VRTrackerConfig:
    """Configuration for Quest hand tracking and EEF target mapping."""

    host: str = "0.0.0.0"
    port: int = 8000
    hand_filter: str = "both"
    timeout_s: float = 1.0
    smooth_alpha: float = 1.0
    max_frame_age_s: float = 0.0
    max_linear_speed: float = 0.0
    max_angular_speed: float = 0.0
    position_scale: float = 1.0
    rotation_scale: float = 1.0
    workspace_limits: tuple[float, float, float, float, float, float] | None = None
    orientation_limits: tuple[float, float, float, float, float, float] | None = None
    max_orientation_angle: float = 0.0

    def __post_init__(self):
        hand_filter = self.hand_filter.lower()
        if hand_filter not in {"both", "left", "right"}:
            raise ValueError(f"hand_filter must be 'both', 'left', or 'right', got {self.hand_filter!r}")
        if not self.host:
            raise ValueError("host must be non-empty")
        if not (0 <= self.port <= 65535):
            raise ValueError("port must be in [0, 65535]")
        if self.timeout_s <= 0.0:
            raise ValueError("timeout_s must be > 0.0")
        if not (0.0 < self.smooth_alpha <= 1.0):
            raise ValueError("smooth_alpha must be in (0, 1]")

        nonnegative = {
            "max_frame_age_s": self.max_frame_age_s,
            "max_linear_speed": self.max_linear_speed,
            "max_angular_speed": self.max_angular_speed,
            "position_scale": self.position_scale,
            "rotation_scale": self.rotation_scale,
            "max_orientation_angle": self.max_orientation_angle,
        }
        for name, value in nonnegative.items():
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0.0")

        validate_limits(self.workspace_limits, name="workspace_limits")
        validate_limits(self.orientation_limits, name="orientation_limits")

    @property
    def normalized_hand_filter(self) -> str:
        return self.hand_filter.lower()


@dataclass(frozen=True, slots=True)
class VRTrackerStats:
    running: bool
    frames_received: int
    frames_dropped: int
    stale_frames: int
    update_hz: float
    last_frame_age_ms: float | None
    last_update_ns: int | None
    latest_sides: tuple[str, ...]
    last_error: str | None


@dataclass(frozen=True, slots=True)
class HandData:
    """VR/SDK hand data in FLU coordinates.

    This is the raw data layer for downstream workflows. It intentionally does
    not know about dex-retargeting, IK, or robot drivers.

    Quaternion convention: wxyz.
    Landmark convention: absolute FLU positions converted from SDK Unity-LH data.
    """

    side: str
    sequence_id: int
    frame_id: str | None
    wrist_pos: Vec3
    wrist_quat: Quat
    landmarks: tuple[Vec3, ...]
    recv_ts_ns: int
    recv_time_unix_ns: int | None
    source_ts_ns: int | None
    source_frame_seq: int | None

    def landmarks_np(self) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(self.landmarks, dtype=np.float64))

    def is_finite(self) -> bool:
        values = list(self.wrist_pos) + list(self.wrist_quat)
        for point in self.landmarks:
            values.extend(point)
        return all(math.isfinite(v) for v in values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "sequence_id": self.sequence_id,
            "frame_id": self.frame_id,
            "wrist_pos": list(self.wrist_pos),
            "wrist_quat": list(self.wrist_quat),
            "landmarks": [list(point) for point in self.landmarks],
            "recv_ts_ns": self.recv_ts_ns,
            "recv_time_unix_ns": self.recv_time_unix_ns,
            "source_ts_ns": self.source_ts_ns,
            "source_frame_seq": self.source_frame_seq,
        }


@dataclass(frozen=True, slots=True)
class EefTarget:
    """World-frame EEF target for external IK solvers."""

    side: str
    position_world: Vec3
    quat_wxyz_world: Quat
    delta_pos_world: Vec3
    delta_quat_world: Quat
    sequence_id: int
    recv_ts_ns: int
    recv_time_unix_ns: int | None
    source_ts_ns: int | None

    @property
    def quat_xyzw_world(self) -> tuple[float, float, float, float]:
        return quat_wxyz_to_xyzw(self.quat_wxyz_world)

    def pos_quat_wxyz(self) -> tuple[Vec3, Quat]:
        return self.position_world, self.quat_wxyz_world

    def pose_matrix_world(self) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = quat_wxyz_to_matrix(self.quat_wxyz_world)
        pose[:3, 3] = np.asarray(self.position_world, dtype=np.float64)
        return pose


@dataclass(frozen=True, slots=True)
class VRState:
    """Latest state cached by VRTrackerService for one hand side."""

    data: HandData
    retargeting_joints: np.ndarray
    finger_curl: np.ndarray
    eef_target: EefTarget | None
    timestamp_ns: int
    is_fresh: bool
    valid: bool


class EWMAFilter:
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
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value: Quat | None = None

    def update(self, raw: Quat) -> Quat:
        raw_q = quat_normalize(raw)
        if self.value is None:
            self.value = raw_q
            return raw_q
        aligned = quat_match_hemisphere(raw_q, self.value)
        mixed = (
            self.alpha * aligned[0] + (1.0 - self.alpha) * self.value[0],
            self.alpha * aligned[1] + (1.0 - self.alpha) * self.value[1],
            self.alpha * aligned[2] + (1.0 - self.alpha) * self.value[2],
            self.alpha * aligned[3] + (1.0 - self.alpha) * self.value[3],
        )
        self.value = quat_normalize(mixed, fallback=self.value)
        return self.value

    def reset(self):
        self.value = None


@dataclass(slots=True)
class SideState:
    pos_filter: EWMAFilter | None = None
    quat_filter: QuaternionEWMAFilter | None = None
    ref_hand_pos: Vec3 | None = None
    ref_hand_quat: Quat | None = None
    home_pos_world: Vec3 | None = None
    home_quat_world: Quat | None = None
    last_pos_world: Vec3 | None = None
    last_quat_world: Quat | None = None
    last_time_ns: int | None = None

    @property
    def calibrated(self) -> bool:
        return None not in (self.ref_hand_pos, self.ref_hand_quat, self.home_pos_world, self.home_quat_world)

    def reset_tracking(self):
        self.last_pos_world = None
        self.last_quat_world = None
        self.last_time_ns = None
        if self.pos_filter is not None:
            self.pos_filter.reset()
        if self.quat_filter is not None:
            self.quat_filter.reset()


def hand_frame_to_data(frame: HandFrame) -> HandData:
    """Convert SDK HandFrame to HandData in FLU with wxyz wrist quaternion."""
    wrist_pos = unity_lh_to_flu_position(frame.wrist.x, frame.wrist.y, frame.wrist.z)
    wrist_q_xyzw = unity_lh_to_flu_quaternion(frame.wrist.qx, frame.wrist.qy, frame.wrist.qz, frame.wrist.qw)
    wrist_q_wxyz = quat_normalize((wrist_q_xyzw[3], wrist_q_xyzw[0], wrist_q_xyzw[1], wrist_q_xyzw[2]))
    landmarks = tuple(unity_lh_to_flu_position(*point) for point in frame.landmarks.points)
    return HandData(
        side=normalize_side(frame.side),
        sequence_id=frame.sequence_id,
        frame_id=getattr(frame, "frame_id", None),
        wrist_pos=wrist_pos,
        wrist_quat=wrist_q_wxyz,
        landmarks=landmarks,
        recv_ts_ns=frame.recv_ts_ns,
        recv_time_unix_ns=getattr(frame, "recv_time_unix_ns", None),
        source_ts_ns=getattr(frame, "source_ts_ns", None),
        source_frame_seq=getattr(frame, "source_frame_seq", None),
    )


class QuestHandReceiver:
    """Synchronous SDK receiver plus LeFranX-style EEF target mapper."""

    def __init__(self, config: VRTrackerConfig | None = None):
        self.config = config if config is not None else VRTrackerConfig()
        self.smooth_alpha = self.config.smooth_alpha
        self.max_frame_age_s = self.config.max_frame_age_s
        self.max_linear_speed = self.config.max_linear_speed
        self.max_angular_speed = self.config.max_angular_speed
        self.position_scale = self.config.position_scale
        self.rotation_scale = self.config.rotation_scale
        self.workspace_limits = self.config.workspace_limits
        self.orientation_limits = self.config.orientation_limits
        self.max_orientation_angle = self.config.max_orientation_angle
        self.hand_filter = self.config.normalized_hand_filter

        filter_map = {"both": HandFilter.BOTH, "left": HandFilter.LEFT, "right": HandFilter.RIGHT}
        self.client = HTSClient(
            HTSClientConfig(
                transport_mode=TransportMode.TCP_SERVER,
                host=self.config.host,
                port=self.config.port,
                timeout_s=self.config.timeout_s,
                output=StreamOutput.FRAMES,
                hand_filter=filter_map[self.hand_filter],
                error_policy=ErrorPolicy.TOLERANT,
            )
        )
        self.tracked_sides = ("left", "right") if self.hand_filter == "both" else (self.hand_filter,)
        self.states = {side: self.new_side_state() for side in self.tracked_sides}
        self.world_from_flu: Quat = IDENTITY_QUAT
        self.state_lock = threading.RLock()
        self.stopped = False

    def __enter__(self) -> QuestHandReceiver:
        return self

    def __exit__(self, *exc: object):
        self.stop()

    def new_side_state(self) -> SideState:
        if 0.0 < self.smooth_alpha < 1.0:
            return SideState(EWMAFilter(self.smooth_alpha, 3), QuaternionEWMAFilter(self.smooth_alpha))
        return SideState()

    def side_state(self, side: str) -> SideState:
        side_key = side.lower()
        if side_key not in self.states:
            raise ValueError(f"side {side_key!r} is not tracked; tracked_sides={self.tracked_sides}")
        return self.states[side_key]

    def stream(self) -> Iterator[HandData]:
        self.stopped = False
        for event in self.client.iter_events():
            if self.stopped:
                break
            if isinstance(event, HandFrame):
                yield hand_frame_to_data(event)

    def stop(self):
        self.stopped = True
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def stats(self) -> ClientStats:
        return self.client.get_stats()

    def reset_stats(self):
        self.client.reset_stats()

    def set_world_alignment(self, world_from_flu: Quat):
        with self.state_lock:
            self.world_from_flu = quat_normalize(world_from_flu)

    def set_world_alignment_yaw(self, yaw_rad: float):
        from vr_utils import quat_from_rpy

        self.set_world_alignment(quat_from_rpy(0.0, 0.0, yaw_rad))

    def set_world_alignment_matrix(self, world_from_flu_matrix: np.ndarray):
        self.set_world_alignment(matrix_to_quat_wxyz(world_from_flu_matrix))

    def set_workspace_limits(self, limits: tuple[float, float, float, float, float, float] | None):
        validate_limits(limits, name="workspace_limits")
        with self.state_lock:
            self.workspace_limits = limits

    def set_orientation_limits(self, limits: tuple[float, float, float, float, float, float] | None):
        validate_limits(limits, name="orientation_limits")
        with self.state_lock:
            self.orientation_limits = limits

    def set_max_orientation_angle(self, angle_rad: float):
        if angle_rad < 0.0:
            raise ValueError("max orientation angle must be >= 0")
        with self.state_lock:
            self.max_orientation_angle = angle_rad

    def calibrate(self, data: HandData, home_pos_world: Vec3, home_quat_world: Quat):
        with self.state_lock:
            state = self.side_state(data.side)
            state.ref_hand_pos = data.wrist_pos
            state.ref_hand_quat = quat_normalize(data.wrist_quat)
            state.home_pos_world = home_pos_world
            state.home_quat_world = quat_normalize(home_quat_world)
            state.last_pos_world = home_pos_world
            state.last_quat_world = state.home_quat_world
            state.last_time_ns = data.recv_ts_ns
            if state.pos_filter is not None:
                state.pos_filter.reset()
            if state.quat_filter is not None:
                state.quat_filter.reset()

    def clear_calibration(self, side: str | None = None):
        with self.state_lock:
            sides = self.tracked_sides if side is None else (side.lower(),)
            for side_key in sides:
                state = self.side_state(side_key)
                state.ref_hand_pos = None
                state.ref_hand_quat = None
                state.home_pos_world = None
                state.home_quat_world = None
                state.reset_tracking()

    def is_fresh(self, data: HandData) -> bool:
        if self.max_frame_age_s <= 0.0:
            return True
        return (time.monotonic_ns() - data.recv_ts_ns) * 1e-9 <= self.max_frame_age_s

    def data_time_ns(self, data: HandData) -> int:
        return data.recv_ts_ns

    def delta_time_s(self, data: HandData, state: SideState) -> float | None:
        if state.last_time_ns is None:
            return None
        dt_ns = self.data_time_ns(data) - state.last_time_ns
        return None if dt_ns <= 0 else dt_ns * 1e-9

    def smoothed_pose(self, data: HandData, state: SideState) -> tuple[Vec3, Quat]:
        pos = data.wrist_pos if state.pos_filter is None else state.pos_filter.update(data.wrist_pos)
        quat = data.wrist_quat if state.quat_filter is None else state.quat_filter.update(data.wrist_quat)
        return pos, quat_normalize(quat)

    def world_delta(self, pos: Vec3, quat: Quat, state: SideState) -> tuple[Vec3, Quat]:
        assert state.ref_hand_pos is not None and state.ref_hand_quat is not None
        delta_pos_flu = vec_sub(pos, state.ref_hand_pos)
        delta_pos_world = quat_rotate_vector(self.world_from_flu, delta_pos_flu)
        delta_quat_flu = quat_normalize(quat_mul(quat, quat_inverse(state.ref_hand_quat)))
        delta_quat_world = change_basis_rotation(delta_quat_flu, self.world_from_flu)
        return delta_pos_world, delta_quat_world

    def limit_position(self, pos: Vec3, state: SideState, dt_s: float | None) -> Vec3:
        if state.last_pos_world is None or self.max_linear_speed <= 0.0 or dt_s is None:
            return pos
        step = vec_sub(pos, state.last_pos_world)
        distance = vec_norm(step)
        max_step = self.max_linear_speed * dt_s
        if distance <= max_step or distance < 1e-12:
            return pos
        return vec_add(state.last_pos_world, vec_scale(step, max_step / distance))

    def limit_rotation(self, quat: Quat, state: SideState, dt_s: float | None) -> Quat:
        if state.last_quat_world is None or self.max_angular_speed <= 0.0 or dt_s is None:
            return quat
        delta = quat_normalize(quat_mul(quat, quat_inverse(state.last_quat_world)))
        angle = quat_angle(delta)
        max_step = self.max_angular_speed * dt_s
        if angle <= max_step or angle < 1e-12:
            return quat
        limited_delta = quat_slerp_from_identity(delta, max_step / angle)
        return quat_normalize(quat_mul(limited_delta, state.last_quat_world))

    def clamp_workspace(self, pos: Vec3, home_pos: Vec3) -> Vec3:
        if self.workspace_limits is None:
            return pos
        min_x, max_x, min_y, max_y, min_z, max_z = self.workspace_limits
        delta = vec_sub(pos, home_pos)
        clamped = (
            max(min_x, min(max_x, delta[0])),
            max(min_y, min(max_y, delta[1])),
            max(min_z, min(max_z, delta[2])),
        )
        return vec_add(home_pos, clamped)

    def compute_eef_target(self, data: HandData) -> EefTarget | None:
        with self.state_lock:
            state = self.side_state(data.side)
            if not state.calibrated or not data.is_finite() or not self.is_fresh(data):
                return None

            hand_pos, hand_quat = self.smoothed_pose(data, state)
            delta_pos, delta_quat = self.world_delta(hand_pos, hand_quat, state)
            delta_pos = vec_scale(delta_pos, self.position_scale)
            delta_quat = quat_slerp_from_identity(delta_quat, self.rotation_scale)

            home_pos = state.home_pos_world
            home_quat = state.home_quat_world
            assert home_pos is not None and home_quat is not None

            pos = vec_add(home_pos, delta_pos)
            quat = quat_normalize(quat_mul(delta_quat, home_quat))
            dt = self.delta_time_s(data, state)
            pos = self.limit_position(pos, state, dt)
            quat = self.limit_rotation(quat, state, dt)
            pos = self.clamp_workspace(pos, home_pos)
            quat = clamp_relative_orientation(quat, home_quat, self.orientation_limits, self.max_orientation_angle)

            state.last_pos_world = pos
            state.last_quat_world = quat
            state.last_time_ns = self.data_time_ns(data)

            return EefTarget(
                side=data.side,
                position_world=pos,
                quat_wxyz_world=quat,
                delta_pos_world=vec_sub(pos, home_pos),
                delta_quat_world=quat_normalize(quat_mul(quat, quat_inverse(home_quat))),
                sequence_id=data.sequence_id,
                recv_ts_ns=data.recv_ts_ns,
                recv_time_unix_ns=data.recv_time_unix_ns,
                source_ts_ns=data.source_ts_ns,
            )


class VRTrackerService:
    """Background latest-state service for external robot workflows."""

    def __init__(self, receiver: QuestHandReceiver):
        self.receiver = receiver
        self.lock = threading.RLock()
        self.latest: dict[str, VRState] = {}
        self.thread: threading.Thread | None = None
        self.running = False
        self.last_error: BaseException | None = None
        self.frames_received = 0
        self.frames_dropped = 0
        self.stale_frames = 0
        self.start_time_ns: int | None = None
        self.last_update_ns: int | None = None

    def __enter__(self) -> VRTrackerService:
        self.start()
        return self

    def __exit__(self, *exc: object):
        self.stop()

    def start(self):
        if self.running:
            return
        self.running = True
        self.last_error = None
        self.frames_received = 0
        self.frames_dropped = 0
        self.stale_frames = 0
        self.start_time_ns = time.monotonic_ns()
        self.last_update_ns = None
        self.thread = threading.Thread(target=self.run_loop, name="VRTrackerService", daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.receiver.stop()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None

    def run_loop(self):
        try:
            for data in self.receiver.stream():
                if not self.running:
                    break
                now_ns = time.monotonic_ns()
                try:
                    is_fresh = self.receiver.is_fresh(data)
                    eef_target = self.receiver.compute_eef_target(data) if is_fresh else None
                    state = VRState(
                        data=data,
                        retargeting_joints=hand_data_to_retargeting_joints(data),
                        finger_curl=hand_data_to_finger_curl_vector(data),
                        eef_target=eef_target,
                        timestamp_ns=now_ns,
                        is_fresh=is_fresh,
                        valid=is_fresh and eef_target is not None,
                    )
                except BaseException as exc:
                    with self.lock:
                        self.frames_dropped += 1
                        self.last_error = exc
                    continue

                with self.lock:
                    self.frames_received += 1
                    if not is_fresh:
                        self.stale_frames += 1
                    self.last_update_ns = now_ns
                    self.last_error = None
                    self.latest[data.side] = state
        except BaseException as exc:
            with self.lock:
                self.last_error = exc
            self.running = False

    def get_latest(self, side: str = "right", *, copy_arrays: bool = True) -> VRState | None:
        with self.lock:
            state = self.latest.get(side.lower())
            if state is None:
                return None
            if not copy_arrays:
                return state
            return VRState(
                data=state.data,
                retargeting_joints=state.retargeting_joints.copy(),
                finger_curl=state.finger_curl.copy(),
                eef_target=state.eef_target,
                timestamp_ns=state.timestamp_ns,
                is_fresh=state.is_fresh,
                valid=state.valid,
            )

    def get_latest_all(self, *, copy_arrays: bool = True) -> dict[str, VRState]:
        with self.lock:
            sides = tuple(self.latest.keys())
        result: dict[str, VRState] = {}
        for side in sides:
            state = self.get_latest(side, copy_arrays=copy_arrays)
            if state is not None:
                result[side] = state
        return result

    def get_latest_data(self, side: str = "right") -> HandData | None:
        state = self.get_latest(side, copy_arrays=False)
        return None if state is None else state.data

    def get_latest_retargeting_joints(self, side: str = "right", *, copy: bool = True) -> np.ndarray | None:
        state = self.get_latest(side, copy_arrays=copy)
        return None if state is None else state.retargeting_joints

    def get_latest_finger_curl_vector(self, side: str = "right", *, copy: bool = True) -> np.ndarray | None:
        state = self.get_latest(side, copy_arrays=copy)
        return None if state is None else state.finger_curl

    def get_latest_eef_target(self, side: str = "right") -> EefTarget | None:
        state = self.get_latest(side, copy_arrays=False)
        return None if state is None else state.eef_target

    def calibrate_from_latest(self, side: str, home_pos_world: Vec3, home_quat_world: Quat) -> bool:
        data = self.get_latest_data(side)
        if data is None:
            return False
        self.receiver.calibrate(data, home_pos_world, home_quat_world)
        return True

    def raise_if_failed(self):
        with self.lock:
            err = self.last_error
        if err is not None:
            raise RuntimeError("VRTrackerService background loop failed") from err

    def stats(self) -> VRTrackerStats:
        now_ns = time.monotonic_ns()
        with self.lock:
            elapsed_s = 0.0 if self.start_time_ns is None else max(0.0, (now_ns - self.start_time_ns) * 1e-9)
            update_hz = self.frames_received / elapsed_s if elapsed_s > 1e-9 else 0.0
            latest_sides = tuple(sorted(self.latest.keys()))
            newest_recv_time = None
            for state in self.latest.values():
                recv_time = state.data.recv_ts_ns
                if newest_recv_time is None or recv_time > newest_recv_time:
                    newest_recv_time = recv_time
            last_frame_age_ms = None if newest_recv_time is None else max(0.0, (now_ns - newest_recv_time) * 1e-6)
            err = None if self.last_error is None else f"{type(self.last_error).__name__}: {self.last_error}"
            return VRTrackerStats(
                running=self.running,
                frames_received=self.frames_received,
                frames_dropped=self.frames_dropped,
                stale_frames=self.stale_frames,
                update_hz=update_hz,
                last_frame_age_ms=last_frame_age_ms,
                last_update_ns=self.last_update_ns,
                latest_sides=latest_sides,
                last_error=err,
            )

    def sdk_stats(self) -> ClientStats:
        return self.receiver.stats()


class LatestFrameVideoSource:
    """Thread-safe latest-frame source consumed by SDK WebRTC sender."""

    def __init__(self, *, width: int = 1280, height: int = 720, fps: int = 30):
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("width, height, and fps must be positive")
        self.av, _, _, self.VideoFormat, _ = load_video_backend()
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_period_s = 1.0 / float(fps)
        self.lock = threading.Lock()
        self.latest_frame: bytes | None = None
        self.last_frame: bytes | None = None
        self.stopped = False

    def get_format(self):
        return self.VideoFormat(width=self.width, height=self.height, fps=self.fps)

    def reset_for_start(self):
        with self.lock:
            self.latest_frame = None
            self.last_frame = None
            self.stopped = False

    async def start(self):
        self.reset_for_start()
        return None

    async def stop(self):
        with self.lock:
            self.stopped = True

    def submit_frame(self, rgb: np.ndarray):
        frame = np.asarray(rgb)
        expected_shape = (self.height, self.width, 3)
        if frame.shape != expected_shape:
            raise ValueError(f"video frame must have shape {expected_shape}, got {frame.shape}")
        if frame.dtype != np.uint8:
            raise ValueError(f"video frame must have dtype uint8, got {frame.dtype}")
        frame_bytes = np.ascontiguousarray(frame).tobytes()
        with self.lock:
            if not self.stopped:
                self.latest_frame = frame_bytes

    async def next_frame(self):
        await asyncio.sleep(self.frame_period_s)
        with self.lock:
            if self.latest_frame is not None:
                self.last_frame = self.latest_frame
                self.latest_frame = None
            frame_bytes = self.last_frame
            stopped = self.stopped

        if frame_bytes is None or stopped:
            rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            rgb = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(self.height, self.width, 3)
        return self.av.VideoFrame.from_ndarray(rgb, format="rgb24")


class VideoReturnService:
    """Independent WebRTC video return to Quest."""

    def __init__(
        self,
        *,
        signaling_host: str = "0.0.0.0",
        signaling_port: int = 8765,
        preset: str = "720p",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        verbose: bool = False,
    ):
        _, self.VideoService, self.VideoServiceConfig, _, self.VideoWebRTCSender = load_video_backend()
        self.source = LatestFrameVideoSource(width=width, height=height, fps=fps)
        self.config = self.VideoServiceConfig(
            signaling_host=signaling_host,
            signaling_port=signaling_port,
            source="test",
            preset=preset,
            verbose=verbose,
        )
        self.service: Any | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None

    def start(self, *, timeout_s: float = 5.0):
        if self.service is not None:
            return

        self.source.reset_for_start()

        def sender_factory(source, session_id: str, fps_val: int):
            _ = (source, session_id, fps_val)
            return self.VideoWebRTCSender(source=self.source)

        self.service = self.VideoService(self.config, sender_factory=sender_factory)
        started = threading.Event()
        errors: list[BaseException] = []

        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.loop = loop
            try:
                loop.run_until_complete(self.service.start())
            except BaseException as exc:
                errors.append(exc)
                started.set()
                loop.close()
                return
            started.set()
            try:
                loop.run_forever()
            finally:
                loop.close()

        self.thread = threading.Thread(target=run_loop, name="VideoReturnService", daemon=True)
        self.thread.start()
        if not started.wait(timeout=timeout_s):
            try:
                self.stop(timeout_s=1.0)
            finally:
                raise TimeoutError(f"video return service did not start within {timeout_s:.1f}s")
        if errors:
            self.service = None
            self.loop = None
            self.thread = None
            raise RuntimeError("video return service failed to start") from errors[0]

    def stop(self, *, timeout_s: float = 5.0):
        if self.service is None:
            return
        with self.source.lock:
            self.source.stopped = True
        if self.loop is not None and self.loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self.service.stop(), self.loop)
            future.result(timeout=timeout_s)
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=timeout_s)
        self.service = None
        self.loop = None
        self.thread = None

    def submit_frame(self, rgb: np.ndarray):
        self.source.submit_frame(rgb)

    @property
    def is_running(self) -> bool:
        return self.service is not None and self.thread is not None and self.thread.is_alive()
