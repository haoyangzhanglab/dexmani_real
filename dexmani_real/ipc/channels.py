"""RuntimeChannels — centralized data plane for cross-process communication.

A single class owns all rings, queues, events, and flags. Processes exchange data
through it — no direct references, no RPC, no business logic.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dexmani_real.config.defaults import camera
from dexmani_real.ipc.camera_ring import CameraRingBuffer
from dexmani_real.ipc.ring import SharedMemoryRingBuffer
from dexmani_real.ipc.schema import (
    ARM_STATE_DTYPE,
    COUPLED_COMMAND_DTYPE,
    HAND_STATE_DTYPE,
    HAND_TACTILE_DTYPE,
    PREDICTION_DTYPE,
    RECORD_CONTROL_DTYPE,
    RECORD_STATUS_DTYPE,
    SUPPORTED_POINT_CLOUD_COUNTS,
    VR_FRAME_DTYPE,
    make_pointcloud_frame_dtype,
    make_record_sample_dtype,
)
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

PREDICTION_RING_MAXLEN = 1
# ``runtime.safety.SafetyState`` owns the enum; IPC carries this stable wire value
# without importing the runtime state machine back into the data plane.
DISARMED_SAFETY_STATE_WIRE_VALUE = 0


@dataclass
class RuntimeChannelsConfig:
    """Centralized configuration for RuntimeChannels variable sizes and camera defaults.

    Variable ring ``maxlen`` values and camera resolution defaults are gathered
    here so they have a single source of truth rather than being scattered across
    entry points. Fixed wire details remain module constants.

    Usage::

        cfg = RuntimeChannelsConfig()
        shared = RuntimeChannels.create(config=cfg)
    """

    camera_ring_maxlen: int = field(default_factory=lambda: camera.ring_maxlen)
    vr_ring_maxlen: int = 8
    arm_state_ring_maxlen: int = 8
    hand_state_ring_maxlen: int = 8
    hand_tactile_ring_maxlen: int = 8
    coupled_cmd_ring_maxlen: int = 8
    record_control_ring_maxlen: int = 1
    record_sample_ring_maxlen: int = 4
    record_status_ring_maxlen: int = 1
    pointcloud_num_points: int = 1024
    camera_requested: bool = False
    pointcloud_requested: bool = False
    # Point-cloud ring capacity must cover the observation horizon so the
    # per-step point-cloud history is a real window, not a broadcast.
    pointcloud_ring_maxlen: int = 8

    camera_rgb_shape: tuple[int, int, int] = field(
        default_factory=lambda: camera.rgb_shape
    )
    camera_depth_shape: tuple[int, int] = field(
        default_factory=lambda: camera.depth_shape
    )

    arm_home_q_maxsize: int = 2

    def __post_init__(self) -> None:
        capacities = (
            self.camera_ring_maxlen,
            self.vr_ring_maxlen,
            self.arm_state_ring_maxlen,
            self.hand_state_ring_maxlen,
            self.hand_tactile_ring_maxlen,
            self.coupled_cmd_ring_maxlen,
            self.record_control_ring_maxlen,
            self.record_sample_ring_maxlen,
            self.record_status_ring_maxlen,
            self.pointcloud_ring_maxlen,
            self.arm_home_q_maxsize,
        )
        if any(int(value) <= 0 for value in capacities):
            raise ValueError("RuntimeChannels ring/queue capacities must be positive")
        if (
            isinstance(self.pointcloud_num_points, bool)
            or int(self.pointcloud_num_points) not in SUPPORTED_POINT_CLOUD_COUNTS
        ):
            raise ValueError(
                "RuntimeChannels pointcloud_num_points must be one of "
                f"{sorted(SUPPORTED_POINT_CLOUD_COUNTS)}"
            )
        if not isinstance(self.camera_requested, bool):
            raise TypeError("RuntimeChannels camera_requested must be boolean")
        if not isinstance(self.pointcloud_requested, bool):
            raise TypeError("RuntimeChannels pointcloud_requested must be boolean")
        if self.pointcloud_requested and not self.camera_requested:
            raise ValueError("pointcloud_requested requires camera_requested")

    @classmethod
    def from_runtime(
        cls,
        runtime: object,
        *,
        pointcloud_num_points: int = 1024,
        camera_requested: bool = False,
        pointcloud_requested: bool = False,
    ) -> "RuntimeChannelsConfig":
        cam = getattr(runtime, "camera")
        return cls(
            camera_ring_maxlen=int(cam.ring_maxlen),
            camera_rgb_shape=(int(cam.height), int(cam.width), 3),
            camera_depth_shape=(int(cam.height), int(cam.width)),
            pointcloud_num_points=int(pointcloud_num_points),
            camera_requested=camera_requested,
            pointcloud_requested=pointcloud_requested,
        )


_RING_RESOURCE_NAMES = (
    "camera_ring",
    "vr_ring",
    "arm_state_ring",
    "hand_state_ring",
    "hand_tactile_ring",
    "coupled_cmd_ring",
    "record_control_ring",
    "record_sample_ring",
    "record_status_ring",
    "prediction_ring",
    "pointcloud_ring",
)
_QUEUE_RESOURCE_NAMES = ("arm_home_q",)

# Heartbeat slots use a fixed process-stable order.
HEARTBEAT_FIELDS: tuple[str, ...] = (
    "arm",
    "hand",
    "policy",
    "recorder",
    "vr",
    "camera",
    "pointcloud",
    "inference",
)
HEARTBEAT_INDEX: dict[str, int] = {
    name: index for index, name in enumerate(HEARTBEAT_FIELDS)
}

# Readiness slots use a fixed, process-stable order and atomic 0/1 flags.
READY_FIELDS: tuple[str, ...] = (
    "arm",
    "hand",
    "camera",
    "pointcloud",
    "vr",
    "policy",
    "recorder",
    "inference",
)
READY_INDEX: dict[str, int] = {name: index for index, name in enumerate(READY_FIELDS)}


def new_frame(dtype: np.dtype) -> np.ndarray:
    """Allocate a zero-initialized 1-element structured array for ring writes."""
    return np.zeros(1, dtype=dtype)


@dataclass
class RuntimeChannels:
    """Central data plane — all cross-process state in one place.

    Created by Main before spawning child processes. Each process receives a
    reference and reads/writes its designated rings/queues/flags.
    """

    camera_ring: CameraRingBuffer  # camera -> policy
    vr_ring: SharedMemoryRingBuffer  # vr -> policy
    arm_state_ring: SharedMemoryRingBuffer  # arm -> policy
    hand_state_ring: SharedMemoryRingBuffer  # hand -> policy
    hand_tactile_ring: SharedMemoryRingBuffer  # hand -> policy (sparse)
    coupled_cmd_ring: SharedMemoryRingBuffer  # serialized control -> arm/hand endpoint
    record_control_ring: SharedMemoryRingBuffer  # policy -> RecorderIO episode boundary
    record_sample_ring: SharedMemoryRingBuffer  # policy -> RecorderIO fixed payload
    record_status_ring: SharedMemoryRingBuffer  # RecorderIO -> controller/main
    prediction_ring: SharedMemoryRingBuffer  # inference -> policy executor, single latest
    pointcloud_ring: SharedMemoryRingBuffer  # pointcloud worker -> inference

    arm_home_q: mp.Queue  # requester -> arm HOME (waypoints, final_qpos, generation)
    arm_command_seq: (
        Any  # all actuator-action producers -> globally unique monotonic IDs
    )
    run_generation: Any  # controller advances it to invalidate old policy proposals
    run_started_monotonic_ns: Any  # start of the current RUNNING observation epoch
    recorder_consumed_sequence: Any

    is_running: Any  # Main -> all
    is_recording: Any  # policy -> arm/hand/camera
    error_state: Any  # arm/hand -> all (sticky latch)
    estop_request: Any  # policy -> arm/hand
    quit_requested: Any  # policy -> Main
    camera_requested: (
        Any  # Main -> camera; keep native RGB-D payload publication active
    )
    pointcloud_requested: Any  # Main -> pointcloud worker
    start_request: Any  # Main -> policy executor: B (start a new policy run)
    # Main/operator -> policy executor: true only after this process completed the
    # authorized hand-home + collision-checked arm-home sequence.
    physical_home_completed: Any
    # Main/operator -> policy executor: explicit S request.
    stop_request: Any
    inference_request: Any  # policy executor -> inference, sync one-prediction trigger

    safety_state: Any  # SafetyState enum (0-3), Main + policy write
    # Serializes the motion permit and coupled-command ring writer. It is never
    # held across hardware SDK calls.
    motion_lock: Any

    heartbeats: Any  # fixed-order array of per-subsystem heartbeat timestamps (s)

    ready_flags: Any  # fixed-order array of per-subsystem readiness flags (0/1)

    arm_device_identity: Any  # worker-reported canonical identity JSON
    hand_device_identity: Any  # worker-reported canonical identity JSON
    camera_depth_scale: Any  # depth scale (mm to meters)
    camera_serial: Any  # serial number string
    camera_firmware: Any  # firmware version string
    camera_sdk_version: Any  # pyrealsense2/librealsense version string
    camera_profile: Any  # active stream/profile and L515 setting JSON
    camera_geometry: Any  # static native RGB-D geometry JSON
    _closed: bool = field(init=False, repr=False, default=False)

    @classmethod
    def create(
        cls,
        prefix: str = "dexmani",
        *,
        config: RuntimeChannelsConfig | None = None,
        camera_rgb_shape: tuple[int, int, int] | None = None,
        camera_depth_shape: tuple[int, int] | None = None,
        mp_context: Any | None = None,
    ) -> "RuntimeChannels":
        """Create all rings, queues, flags, events, and heartbeats.

        Call once from Main before spawning child processes.
        """
        cfg = config or RuntimeChannelsConfig()
        ctx = mp_context or mp.get_context("spawn")

        _rgb_shape = camera_rgb_shape or cfg.camera_rgb_shape
        _depth_shape = camera_depth_shape or cfg.camera_depth_shape

        storage = cls.__new__(cls)
        storage._closed = False
        try:
            cls._allocate_resources(storage, prefix, cfg, ctx, _rgb_shape, _depth_shape)
        except BaseException as allocation_error:
            try:
                cleanup_succeeded = storage.close()
            except BaseException:
                logger.critical(
                    "RuntimeChannels allocation rollback raised", exc_info=True
                )
                raise RuntimeError(
                    "RuntimeChannels allocation failed and rollback raised"
                ) from allocation_error
            if not cleanup_succeeded:
                raise RuntimeError(
                    "RuntimeChannels allocation failed and rollback was incomplete"
                ) from allocation_error
            raise

        logger.info("RuntimeChannels created (prefix=%s)", prefix)
        return storage

    @staticmethod
    def _allocate_resources(
        storage: "RuntimeChannels",
        prefix: str,
        cfg: RuntimeChannelsConfig,
        ctx: Any,
        rgb_shape: tuple[int, int, int],
        depth_shape: tuple[int, int],
    ) -> None:
        storage.camera_ring = CameraRingBuffer(
            name=f"{prefix}_camera",
            rgb_shape=rgb_shape,
            depth_shape=depth_shape,
            maxlen=cfg.camera_ring_maxlen,
            create=True,
        )
        storage.vr_ring = SharedMemoryRingBuffer(
            f"{prefix}_vr",
            dtype=vr_frame_dtype(),
            maxlen=cfg.vr_ring_maxlen,
            create=True,
        )
        storage.arm_state_ring = SharedMemoryRingBuffer(
            f"{prefix}_arm_state",
            dtype=ARM_STATE_DTYPE,
            maxlen=cfg.arm_state_ring_maxlen,
            create=True,
        )
        storage.hand_state_ring = SharedMemoryRingBuffer(
            f"{prefix}_hand_state",
            dtype=HAND_STATE_DTYPE,
            maxlen=cfg.hand_state_ring_maxlen,
            create=True,
        )
        storage.hand_tactile_ring = SharedMemoryRingBuffer(
            f"{prefix}_hand_tactile",
            dtype=HAND_TACTILE_DTYPE,
            maxlen=cfg.hand_tactile_ring_maxlen,
            create=True,
        )
        storage.coupled_cmd_ring = SharedMemoryRingBuffer(
            f"{prefix}_coupled_cmd",
            dtype=COUPLED_COMMAND_DTYPE,
            maxlen=cfg.coupled_cmd_ring_maxlen,
            create=True,
        )
        storage.record_control_ring = SharedMemoryRingBuffer(
            f"{prefix}_record_control",
            dtype=RECORD_CONTROL_DTYPE,
            maxlen=cfg.record_control_ring_maxlen,
            create=True,
        )
        storage.record_sample_ring = SharedMemoryRingBuffer(
            f"{prefix}_record_sample",
            dtype=make_record_sample_dtype(rgb_shape, depth_shape),
            maxlen=cfg.record_sample_ring_maxlen,
            create=True,
        )
        storage.record_status_ring = SharedMemoryRingBuffer(
            f"{prefix}_record_status",
            dtype=RECORD_STATUS_DTYPE,
            maxlen=cfg.record_status_ring_maxlen,
            create=True,
        )
        storage.prediction_ring = SharedMemoryRingBuffer(
            f"{prefix}_prediction",
            dtype=PREDICTION_DTYPE,
            maxlen=PREDICTION_RING_MAXLEN,
            create=True,
        )
        storage.pointcloud_ring = SharedMemoryRingBuffer(
            f"{prefix}_pointcloud",
            dtype=make_pointcloud_frame_dtype(cfg.pointcloud_num_points),
            maxlen=cfg.pointcloud_ring_maxlen,
            create=True,
        )

        storage.arm_home_q = ctx.Queue(maxsize=cfg.arm_home_q_maxsize)
        storage.arm_command_seq = ctx.Value("Q", 0)
        storage.run_generation = ctx.Value("Q", 1)
        storage.run_started_monotonic_ns = ctx.Value("Q", 0)
        storage.recorder_consumed_sequence = ctx.Value("Q", 0)

        storage.is_running = ctx.Value("b", True)
        storage.is_recording = ctx.Value("b", False)
        storage.error_state = ctx.Value("b", False)
        storage.estop_request = ctx.Value("b", False)
        storage.quit_requested = ctx.Value("b", False)
        storage.camera_requested = ctx.Value("b", cfg.camera_requested)
        storage.pointcloud_requested = ctx.Value("b", cfg.pointcloud_requested)
        storage.start_request = ctx.Value("b", False)
        storage.physical_home_completed = ctx.Value("b", False)
        storage.stop_request = ctx.Value("b", False)
        storage.inference_request = ctx.Event()

        storage.safety_state = ctx.Value("i", DISARMED_SAFETY_STATE_WIRE_VALUE)
        storage.motion_lock = ctx.RLock()

        storage.heartbeats = ctx.Array("d", [0.0] * len(HEARTBEAT_FIELDS))

        storage.ready_flags = ctx.Array("b", len(READY_FIELDS))

        storage.arm_device_identity = ctx.Array("c", b"\x00" * 1024)
        storage.hand_device_identity = ctx.Array("c", b"\x00" * 1024)
        storage.camera_depth_scale = ctx.Value("d", 0.0)
        storage.camera_serial = ctx.Array("c", b"\x00" * 32)
        storage.camera_firmware = ctx.Array("c", b"\x00" * 64)
        storage.camera_sdk_version = ctx.Array("c", b"\x00" * 64)
        storage.camera_profile = ctx.Array("c", b"\x00" * 2048)
        storage.camera_geometry = ctx.Array("c", b"\x00" * 2048)

    def close(self) -> bool:
        """Release all shared memory primitives.

        ``unlink()`` destroys the POSIX shared-memory segment, preventing
        Python's resource tracker "leaked shared_memory objects" warning.
        Cleanup is best-effort across every resource. Underlying close calls are
        idempotent and an already-unlinked shared-memory segment raises
        ``FileNotFoundError``, so a retry can simply repeat the full sequence.

        Returns:
            Whether every owned resource was closed and unlinked successfully.
        """
        if bool(getattr(self, "_closed", False)):
            return True

        errors: list[str] = []

        def _attempt(
            operation: str, callback: Any, *, missing_ok: bool = False
        ) -> bool:
            try:
                callback()
            except FileNotFoundError:
                if not missing_ok:
                    errors.append(operation)
                    logger.warning(
                        "RuntimeChannels close: %s failed", operation, exc_info=True
                    )
                    return False
            except Exception:
                errors.append(operation)
                logger.warning(
                    "RuntimeChannels close: %s failed", operation, exc_info=True
                )
                return False
            return True

        for ring_name in _RING_RESOURCE_NAMES:
            ring = getattr(self, ring_name, None)
            if ring is None:
                continue
            _attempt(f"{ring_name}.close", ring.close)
            _attempt(f"{ring_name}.unlink", ring.unlink, missing_ok=True)

        for queue_name in _QUEUE_RESOURCE_NAMES:
            queue = getattr(self, queue_name, None)
            if queue is None:
                continue
            if _attempt(f"{queue_name}.close", queue.close):
                _attempt(f"{queue_name}.join_thread", queue.join_thread)

        self._closed = not errors
        if self._closed:
            logger.info("RuntimeChannels closed cleanly")
        else:
            logger.error("RuntimeChannels close incomplete: %s", ", ".join(errors))
        return self._closed

    def set_heartbeat(self, name: str, value_s: float) -> None:
        """Record a fresh heartbeat timestamp (s) for *name* (a HEARTBEAT_FIELDS key)."""
        self.heartbeats[HEARTBEAT_INDEX[name]] = float(value_s)

    def get_heartbeat(self, name: str) -> float:
        """Return the last recorded heartbeat timestamp (s) for *name*, or 0.0."""
        return float(self.heartbeats[HEARTBEAT_INDEX[name]])

    def set_ready(self, name: str) -> None:
        """Mark *name* (a READY_FIELDS key) ready."""
        self.ready_flags[READY_INDEX[name]] = 1

    def is_ready(self, name: str) -> bool:
        """Return True when *name* is ready."""
        return bool(self.ready_flags[READY_INDEX[name]])


def vr_frame_dtype() -> np.dtype:
    """Return the wire dtype published by ``sensor.vr_worker``."""
    return VR_FRAME_DTYPE


def read_arm_state(shared: "RuntimeChannels") -> "np.ndarray | None":
    """Read latest arm state from ring. Returns raw structured array or None."""
    result = shared.arm_state_ring.read_latest()
    if result is None:
        return None
    data, _ts_ns, _seq = result
    return data


def read_hand_state(shared: "RuntimeChannels") -> "np.ndarray | None":
    """Read latest hand state from ring. Returns raw structured array or None."""
    result = shared.hand_state_ring.read_latest()
    if result is None:
        return None
    data, _ts_ns, _seq = result
    return data


def read_arm_state_dict(shared: "RuntimeChannels") -> "dict | None":
    """Read latest arm state from ring. Return dict of numpy arrays or None.

    Fields: qpos(7), qvel(7), tau(7), error_code, connected, tracking_err,
            last_cmd_seq, last_cmd_is_hold, source/publish timestamps,
            state_valid.
    Callers must validate fields they depend on (e.g. ``np.all(np.isfinite(d["qpos"]))``).
    The EEF pose is not published; derive it from ``qpos`` via
    ``planning.arm_fk.make_arm_fk()`` when needed.
    """
    data = read_arm_state(shared)
    if data is None:
        return None
    return {
        "qpos": np.asarray(data["qpos"][0], dtype=np.float64),
        "qvel": np.asarray(data["qvel"][0], dtype=np.float64),
        "tau": np.asarray(data["tau"][0], dtype=np.float64),
        "error_code": int(data["error_code"][0]),
        "connected": bool(data["connected"][0]),
        "tracking_err": float(data["tracking_err"][0]),
        "last_cmd_seq": int(data["last_cmd_seq"][0]),
        "last_cmd_is_hold": bool(data["last_cmd_is_hold"][0]),
        "source_monotonic_ns": int(data["source_monotonic_ns"][0]),
        "publish_monotonic_ns": int(data["publish_monotonic_ns"][0]),
        "state_valid": bool(data["state_valid"][0]),
    }


def read_hand_state_dict(shared: "RuntimeChannels") -> "dict | None":
    """Read latest hand state from ring. Return dict of numpy arrays or None.

    Fields include qpos/current/tactile data, freshness validity, board
    telemetry, and the last hand action ID accepted by the worker/SDK.
    """
    data = read_hand_state(shared)
    if data is None:
        return None
    return {
        "qpos": np.asarray(data["qpos"][0], dtype=np.float64),
        "current": np.asarray(data["current"][0], dtype=np.float64),
        "tactile_sum": np.asarray(data["tactile_sum"][0], dtype=np.float64),
        "tactile_sum_valid": bool(data["tactile_sum_valid"][0]),
        "tactile_contact": np.asarray(data["tactile_contact"][0], dtype=bool),
        "connected": bool(data["connected"][0]),
        "qpos_stale": bool(data["qpos_stale"][0]),
        "accepted_target_action_id": int(data["accepted_target_action_id"][0]),
        "last_sdk_setpoint_accepted_monotonic_ns": int(
            data["last_sdk_setpoint_accepted_monotonic_ns"][0]
        ),
        "source_monotonic_ns": int(data["source_monotonic_ns"][0]),
        "publish_monotonic_ns": int(data["publish_monotonic_ns"][0]),
        "state_valid": bool(data["state_valid"][0]),
    }
