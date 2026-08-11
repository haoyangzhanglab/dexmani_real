"""SharedStorage — centralized data plane for cross-process communication.

A single class owns all rings, queues, events, and flags. Processes exchange data
through it — no direct references, no RPC, no business logic.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dexmani_real.config.defaults import arm, camera, hand, policy
from dexmani_real.ipc.schema import (
    ACK_DTYPE,
    ARM_STATE_DTYPE,
    COMMIT_DTYPE,
    COMPONENT_METRICS_DTYPE,
    COMPONENT_STATUS_DTYPE,
    HAND_COMMAND_DTYPE,
    HAND_JOINT_SHAPE,
    HAND_STATE_DTYPE,
    HAND_TACTILE_DTYPE,
    INFERENCE_CANDIDATE_DTYPE,
    RECORD_CONTROL_DTYPE,
    RECORD_STATUS_DTYPE,
    VR_FRAME_DTYPE,
    make_record_sample_dtype,
)
from dexmani_real.robot.safety import SafetyState
from dexmani_real.runtime.status import ComponentPhase, ExitReason, FaultCode
from dexmani_real.shm.ring_buffer import CameraRingBuffer
from dexmani_real.shm.robot_ring import SeqlockRingBuffer
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


# SharedStorageConfig — centralized tuning constants


@dataclass
class SharedStorageConfig:
    """Centralized configuration for SharedStorage ring sizes, camera defaults,
    and workspace bounds.

    All ring ``maxlen`` values, camera resolution defaults, and workspace
    boundary constants are gathered here so they have a single source of truth
    rather than being scattered across entry points.

    Usage::

        cfg = SharedStorageConfig()
        shared = SharedStorage.create(config=cfg)
    """

    camera_ring_maxlen: int = field(default_factory=lambda: camera.ring_maxlen)
    vr_ring_maxlen: int = 8
    arm_state_ring_maxlen: int = 8
    hand_state_ring_maxlen: int = 8
    hand_tactile_ring_maxlen: int = 8
    hand_cmd_ring_maxlen: int = 8
    action_commit_ring_maxlen: int = 8
    action_ack_ring_maxlen: int = 16
    component_status_ring_maxlen: int = 32
    component_metrics_ring_maxlen: int = 8
    record_control_ring_maxlen: int = 8
    record_sample_ring_maxlen: int = 4
    record_status_ring_maxlen: int = 16
    inference_candidate_ring_maxlen: int = 16

    control_hz: float = field(default_factory=lambda: policy.control_hz)
    arm_loop_hz: float = field(default_factory=lambda: arm.loop_hz)
    hand_loop_hz: float = field(default_factory=lambda: hand.loop_hz)
    hand_home_qpos_rad: tuple[float, ...] = field(
        default_factory=lambda: tuple(float(value) for value in np.deg2rad(hand.home_qpos_deg))
    )

    camera_rgb_shape: tuple[int, int, int] = field(default_factory=lambda: camera.rgb_shape)
    camera_depth_shape: tuple[int, int] = field(default_factory=lambda: camera.depth_shape)
    camera_pc_shape: tuple[int, int] = field(default_factory=lambda: camera.pointcloud_shape)

    arm_action_q_maxsize: int = 2

    workspace_bounds: "np.ndarray" = field(default_factory=lambda: policy.workspace.as_array())

    def __post_init__(self) -> None:
        capacities = (
            self.camera_ring_maxlen,
            self.vr_ring_maxlen,
            self.arm_state_ring_maxlen,
            self.hand_state_ring_maxlen,
            self.hand_tactile_ring_maxlen,
            self.hand_cmd_ring_maxlen,
            self.action_commit_ring_maxlen,
            self.action_ack_ring_maxlen,
            self.component_status_ring_maxlen,
            self.component_metrics_ring_maxlen,
            self.record_control_ring_maxlen,
            self.record_sample_ring_maxlen,
            self.record_status_ring_maxlen,
            self.inference_candidate_ring_maxlen,
            self.arm_action_q_maxsize,
        )
        if any(int(value) <= 0 for value in capacities):
            raise ValueError("SharedStorage ring/queue capacities must be positive")
        if min(self.control_hz, self.arm_loop_hz, self.hand_loop_hz) <= 0:
            raise ValueError("SharedStorage action rates must be positive")
        bounds = np.asarray(self.workspace_bounds, dtype=np.float64)
        if bounds.shape != (3, 2) or not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError("SharedStorage workspace_bounds must be finite shape (3, 2)")
        hand_home = np.asarray(self.hand_home_qpos_rad, dtype=np.float64)
        if hand_home.shape != HAND_JOINT_SHAPE or not np.all(np.isfinite(hand_home)):
            raise ValueError("SharedStorage hand_home_qpos_rad must be finite shape (12,)")

    @classmethod
    def from_runtime(cls, runtime: object) -> "SharedStorageConfig":
        cam = getattr(runtime, "camera")
        pol = getattr(runtime, "policy")
        arm_cfg = getattr(runtime, "arm")
        hand_cfg = getattr(runtime, "hand")
        bounds = np.array(
            [
                [pol.workspace.x_min, pol.workspace.x_max],
                [pol.workspace.y_min, pol.workspace.y_max],
                [pol.workspace.z_min, pol.workspace.z_max],
            ],
            dtype=np.float64,
        )
        return cls(
            camera_ring_maxlen=int(cam.ring_maxlen),
            camera_rgb_shape=(int(cam.height), int(cam.width), 3),
            camera_depth_shape=(int(cam.height), int(cam.width)),
            camera_pc_shape=(int(cam.pointcloud_num_points), 6),
            control_hz=float(pol.control_hz),
            arm_loop_hz=float(arm_cfg.loop_hz),
            hand_loop_hz=float(hand_cfg.loop_hz),
            hand_home_qpos_rad=tuple(float(value) for value in np.deg2rad(hand_cfg.home_qpos_deg)),
            workspace_bounds=bounds,
        )


HOME_SENTINEL = "__HOME__"


@dataclass(frozen=True)
class HomeRequest:
    request_id: int
    waypoints: np.ndarray
    final_qpos: np.ndarray
    execution_timeout_s: float


@dataclass(frozen=True)
class HomeResult:
    request_id: int
    success: bool
    reason: str
    final_qpos: np.ndarray
    completed_at_s: float


# Backward-compatible alias retained for external callers; the canonical
# command schema is defined in dexmani_real.ipc.schema.
HAND_CMD_DTYPE = HAND_COMMAND_DTYPE

_RING_RESOURCE_NAMES = (
    "camera_ring",
    "vr_ring",
    "arm_state_ring",
    "hand_state_ring",
    "hand_tactile_ring",
    "hand_cmd_ring",
    "action_commit_ring",
    "arm_ack_ring",
    "hand_ack_ring",
    "component_status_ring",
    "arm_metrics_ring",
    "hand_metrics_ring",
    "camera_metrics_ring",
    "policy_metrics_ring",
    "record_control_ring",
    "record_sample_ring",
    "record_status_ring",
    "inference_candidate_ring",
)
_QUEUE_RESOURCE_NAMES = ("arm_action_q", "arm_home_result_q")
_ALLOCATION_ROLLBACK_ATTEMPTS = 2


def new_frame(dtype: np.dtype) -> np.ndarray:
    """Allocate a zero-initialized 1-element structured array for ring writes."""
    return np.zeros(1, dtype=dtype)


@dataclass
class SharedStorage:
    """Central data plane — all cross-process state in one place.

    Created by Main before spawning child processes. Each process receives a
    reference and reads/writes its designated rings/queues/flags.
    """

    camera_ring: CameraRingBuffer  # camera -> policy
    vr_ring: SeqlockRingBuffer  # vr -> policy
    arm_state_ring: SeqlockRingBuffer  # arm -> policy
    hand_state_ring: SeqlockRingBuffer  # hand -> policy
    hand_tactile_ring: SeqlockRingBuffer  # hand -> policy (sparse)
    hand_cmd_ring: SeqlockRingBuffer  # policy -> hand
    action_commit_ring: SeqlockRingBuffer  # coordinator -> arm + hand
    arm_ack_ring: SeqlockRingBuffer  # arm -> coordinator
    hand_ack_ring: SeqlockRingBuffer  # hand -> coordinator
    component_status_ring: SeqlockRingBuffer  # workers -> supervisor diagnostics
    arm_metrics_ring: SeqlockRingBuffer  # arm -> supervisor, single writer
    hand_metrics_ring: SeqlockRingBuffer  # hand -> supervisor, single writer
    camera_metrics_ring: SeqlockRingBuffer  # camera -> supervisor, single writer
    policy_metrics_ring: SeqlockRingBuffer  # policy -> supervisor, single writer
    record_control_ring: SeqlockRingBuffer  # policy -> RecorderIO episode boundary
    record_sample_ring: SeqlockRingBuffer  # policy -> RecorderIO fixed payload
    record_status_ring: SeqlockRingBuffer  # RecorderIO -> policy/main
    inference_candidate_ring: SeqlockRingBuffer  # Inference -> coordinator only

    arm_action_q: mp.Queue  # policy -> arm, maxsize=2
    arm_home_result_q: mp.Queue  # arm -> requester; request_id correlates replies
    arm_command_seq: Any  # all arm-action producers -> globally unique monotonic IDs
    session_generation: Any
    policy_epoch: Any
    recorder_consumed_sequence: Any
    action_control_hz: float
    action_lead_time_s: float
    action_validity_s: float
    hand_home_qpos_rad: tuple[float, ...]

    is_running: Any  # Main -> all
    is_recording: Any  # policy -> arm/hand/camera
    error_state: Any  # arm/hand -> all (sticky latch)
    estop_request: Any  # policy -> arm/hand
    quit_requested: Any  # policy -> Main

    safety_state: Any  # SafetyState enum (0-3), Main + policy write

    arm_heartbeat_s: Any
    hand_heartbeat_s: Any
    policy_heartbeat_s: Any
    recorder_heartbeat_s: Any
    inference_heartbeat_s: Any
    vr_heartbeat_s: Any
    camera_heartbeat_s: Any

    arm_ready: Any  # -> Main
    hand_ready: Any  # -> Main
    camera_ready: Any  # -> Main
    vr_ready: Any  # -> Main
    policy_ready: Any  # -> Main, only after policy/backend warmup
    inference_ready: Any  # optional capability -> Main
    recorder_ready: Any  # optional capability -> Main
    component_status_lock: Any  # serializes the multi-producer diagnostic ring

    arm_device_identity: Any  # worker-reported canonical identity JSON
    hand_device_identity: Any  # worker-reported canonical identity JSON
    camera_depth_scale: Any  # depth scale (mm to meters)
    camera_K: Any  # 3x3 intrinsics, row-major
    camera_serial: Any  # serial number string
    camera_firmware: Any  # firmware version string
    camera_sdk_version: Any  # pyrealsense2/librealsense version string
    camera_profile: Any  # actual color/depth profile JSON
    camera_observation_required: Any  # learned policy requests camera payload publication
    _closed: bool = field(init=False, repr=False, default=False)
    _close_completed_operations: set[str] = field(init=False, repr=False, default_factory=set)

    @classmethod
    def create(
        cls,
        prefix: str = "dexmani",
        *,
        config: SharedStorageConfig | None = None,
        camera_rgb_shape: tuple[int, int, int] | None = None,
        camera_depth_shape: tuple[int, int] | None = None,
        mp_context: Any | None = None,
    ) -> "SharedStorage":
        """Create all rings, queues, flags, events, and heartbeats.

        Call once from Main before spawning child processes.
        """
        cfg = config or SharedStorageConfig()
        ctx = mp_context or mp.get_context("spawn")

        _rgb_shape = camera_rgb_shape or cfg.camera_rgb_shape
        _depth_shape = camera_depth_shape or cfg.camera_depth_shape

        storage = cls.__new__(cls)
        storage._closed = False
        storage._close_completed_operations = set()
        try:
            cls._allocate_resources(storage, prefix, cfg, ctx, _rgb_shape, _depth_shape)
        except BaseException as allocation_error:
            cleanup_succeeded = False
            for _ in range(_ALLOCATION_ROLLBACK_ATTEMPTS):
                try:
                    cleanup_succeeded = storage.close()
                except BaseException:
                    logger.critical("SharedStorage allocation rollback raised", exc_info=True)
                    raise RuntimeError("SharedStorage allocation failed and rollback raised") from allocation_error
                if cleanup_succeeded:
                    break
            if not cleanup_succeeded:
                raise RuntimeError("SharedStorage allocation failed and rollback was incomplete") from allocation_error
            raise

        logger.info("SharedStorage created (prefix=%s)", prefix)
        return storage

    @staticmethod
    def _allocate_resources(
        storage: "SharedStorage",
        prefix: str,
        cfg: SharedStorageConfig,
        ctx: Any,
        rgb_shape: tuple[int, int, int],
        depth_shape: tuple[int, int],
    ) -> None:
        storage.camera_ring = CameraRingBuffer(
            name=f"{prefix}_camera",
            rgb_shape=rgb_shape,
            depth_shape=depth_shape,
            pc_shape=cfg.camera_pc_shape,
            maxlen=cfg.camera_ring_maxlen,
            create=True,
        )
        storage.vr_ring = SeqlockRingBuffer.create_or_replace(
            f"{prefix}_vr",
            dtype=vr_frame_dtype(),
            maxlen=cfg.vr_ring_maxlen,
        )
        storage.arm_state_ring = SeqlockRingBuffer.create_or_replace(
            f"{prefix}_arm_state",
            dtype=ARM_STATE_DTYPE,
            maxlen=cfg.arm_state_ring_maxlen,
        )
        storage.hand_state_ring = SeqlockRingBuffer.create_or_replace(
            f"{prefix}_hand_state",
            dtype=HAND_STATE_DTYPE,
            maxlen=cfg.hand_state_ring_maxlen,
        )
        storage.hand_tactile_ring = SeqlockRingBuffer.create_or_replace(
            f"{prefix}_hand_tactile",
            dtype=HAND_TACTILE_DTYPE,
            maxlen=cfg.hand_tactile_ring_maxlen,
        )
        storage.hand_cmd_ring = SeqlockRingBuffer.create_or_replace(
            f"{prefix}_hand_cmd",
            dtype=HAND_CMD_DTYPE,
            maxlen=cfg.hand_cmd_ring_maxlen,
        )
        storage.action_commit_ring = SeqlockRingBuffer.create_or_replace(
            f"{prefix}_action_commit", dtype=COMMIT_DTYPE, maxlen=cfg.action_commit_ring_maxlen
        )
        storage.arm_ack_ring = SeqlockRingBuffer.create_or_replace(
            f"{prefix}_arm_ack", dtype=ACK_DTYPE, maxlen=cfg.action_ack_ring_maxlen
        )
        storage.hand_ack_ring = SeqlockRingBuffer.create_or_replace(
            f"{prefix}_hand_ack", dtype=ACK_DTYPE, maxlen=cfg.action_ack_ring_maxlen
        )
        storage.component_status_ring = SeqlockRingBuffer.create_or_replace(
            f"{prefix}_component_status",
            dtype=COMPONENT_STATUS_DTYPE,
            maxlen=cfg.component_status_ring_maxlen,
        )
        for component in ("arm", "hand", "camera", "policy"):
            setattr(
                storage,
                f"{component}_metrics_ring",
                SeqlockRingBuffer.create_or_replace(
                    f"{prefix}_{component}_metrics",
                    dtype=COMPONENT_METRICS_DTYPE,
                    maxlen=cfg.component_metrics_ring_maxlen,
                ),
            )
        storage.record_control_ring = SeqlockRingBuffer.create_or_replace(
            f"{prefix}_record_control", dtype=RECORD_CONTROL_DTYPE, maxlen=cfg.record_control_ring_maxlen
        )
        storage.record_sample_ring = SeqlockRingBuffer.create_or_replace(
            f"{prefix}_record_sample",
            dtype=make_record_sample_dtype(rgb_shape, depth_shape, cfg.camera_pc_shape),
            maxlen=cfg.record_sample_ring_maxlen,
        )
        storage.record_status_ring = SeqlockRingBuffer.create_or_replace(
            f"{prefix}_record_status", dtype=RECORD_STATUS_DTYPE, maxlen=cfg.record_status_ring_maxlen
        )
        storage.inference_candidate_ring = SeqlockRingBuffer.create_or_replace(
            f"{prefix}_inference_candidate",
            dtype=INFERENCE_CANDIDATE_DTYPE,
            maxlen=cfg.inference_candidate_ring_maxlen,
        )

        storage.arm_action_q = ctx.Queue(maxsize=cfg.arm_action_q_maxsize)
        storage.arm_home_result_q = ctx.Queue(maxsize=cfg.arm_action_q_maxsize)
        storage.arm_command_seq = ctx.Value("Q", 0)
        storage.session_generation = ctx.Value("Q", time.monotonic_ns())
        storage.policy_epoch = ctx.Value("Q", 1)
        storage.recorder_consumed_sequence = ctx.Value("Q", 0)
        storage.action_control_hz = float(cfg.control_hz)
        storage.action_lead_time_s = 2.0 / min(float(cfg.arm_loop_hz), float(cfg.hand_loop_hz))
        storage.action_validity_s = 1.0 / float(cfg.control_hz)
        storage.hand_home_qpos_rad = tuple(float(value) for value in cfg.hand_home_qpos_rad)

        storage.is_running = ctx.Value("b", True)
        storage.is_recording = ctx.Value("b", False)
        storage.error_state = ctx.Value("b", False)
        storage.estop_request = ctx.Value("b", False)
        storage.quit_requested = ctx.Value("b", False)

        storage.safety_state = ctx.Value("i", int(SafetyState.DISARMED))

        storage.arm_heartbeat_s = ctx.Value("d", 0.0)
        storage.hand_heartbeat_s = ctx.Value("d", 0.0)
        storage.policy_heartbeat_s = ctx.Value("d", 0.0)
        storage.recorder_heartbeat_s = ctx.Value("d", 0.0)
        storage.inference_heartbeat_s = ctx.Value("d", 0.0)
        storage.vr_heartbeat_s = ctx.Value("d", 0.0)
        storage.camera_heartbeat_s = ctx.Value("d", 0.0)

        storage.arm_ready = ctx.Event()
        storage.hand_ready = ctx.Event()
        storage.camera_ready = ctx.Event()
        storage.vr_ready = ctx.Event()
        storage.policy_ready = ctx.Event()
        storage.inference_ready = ctx.Event()
        storage.recorder_ready = ctx.Event()
        storage.component_status_lock = ctx.Lock()

        storage.arm_device_identity = ctx.Array("c", b"\x00" * 1024)
        storage.hand_device_identity = ctx.Array("c", b"\x00" * 1024)
        storage.camera_depth_scale = ctx.Value("d", 0.0)
        storage.camera_K = ctx.Array("d", [0.0] * 9)
        storage.camera_serial = ctx.Array("c", b"\x00" * 32)
        storage.camera_firmware = ctx.Array("c", b"\x00" * 64)
        storage.camera_sdk_version = ctx.Array("c", b"\x00" * 64)
        storage.camera_profile = ctx.Array("c", b"\x00" * 2048)
        storage.camera_observation_required = ctx.Value("b", False)

    def close(self) -> bool:
        """Release all shared memory primitives.

        ``unlink()`` destroys the POSIX shared-memory segment, preventing
        Python's resource tracker "leaked shared_memory objects" warning.
        Cleanup is best-effort across every resource.  A failed operation is
        retryable, while operations that already succeeded are not repeated.

        Returns:
            Whether every owned resource was closed and unlinked successfully.
        """
        if bool(getattr(self, "_closed", False)):
            return True

        completed: set[str] = getattr(self, "_close_completed_operations", set())
        if not isinstance(completed, set):
            completed = set()
        self._close_completed_operations = completed
        expected: set[str] = set()
        _close_errors: list[str] = []

        def _attempt(operation: str, callback: Any, *, missing_ok: bool = False) -> bool:
            expected.add(operation)
            if operation in completed:
                return True
            try:
                callback()
            except FileNotFoundError:
                if not missing_ok:
                    _close_errors.append(operation)
                    logger.warning("SharedStorage close: %s failed", operation, exc_info=True)
                    return False
            except Exception:
                _close_errors.append(operation)
                logger.warning("SharedStorage close: %s failed", operation, exc_info=True)
                return False
            completed.add(operation)
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
            queue_close = f"{queue_name}.close"
            queue_join = f"{queue_name}.join_thread"
            expected.add(queue_join)
            if _attempt(queue_close, queue.close):
                _attempt(queue_join, queue.join_thread)

        self._closed = not _close_errors and expected.issubset(completed)
        if self._closed:
            logger.info("SharedStorage closed cleanly")
        else:
            logger.error("SharedStorage close incomplete: %s", ", ".join(_close_errors))
        return self._closed


def vr_frame_dtype() -> np.dtype:
    """VR frame dtype — mirrors vr_receiver_process frame output."""
    return VR_FRAME_DTYPE


# Shared ring read/write helpers


def read_arm_state(shared: "SharedStorage") -> "np.ndarray | None":
    """Read latest arm state from ring. Returns raw structured array or None."""
    result = shared.arm_state_ring.read_latest()
    if result is None:
        return None
    data, _ts_ns, _seq = result
    return data


def read_hand_state(shared: "SharedStorage") -> "np.ndarray | None":
    """Read latest hand state from ring. Returns raw structured array or None."""
    result = shared.hand_state_ring.read_latest()
    if result is None:
        return None
    data, _ts_ns, _seq = result
    return data


def publish_component_status(
    shared: Any,
    component: str,
    phase: ComponentPhase,
    *,
    fault_code: FaultCode = FaultCode.NONE,
    exit_reason: ExitReason = ExitReason.NONE,
    detail: str = "",
) -> None:
    """Publish one structured component health transition when available.

    Minimal offline mocks and older external façades may omit the diagnostic
    ring; status reporting is observability-only and must not mask the worker's
    primary safety behavior in that compatibility case.
    """
    ring = getattr(shared, "component_status_ring", None)
    if ring is None:
        return
    frame = new_frame(COMPONENT_STATUS_DTYPE)
    frame["component"][0] = component.encode("utf-8")[:24]
    frame["phase"][0] = int(phase)
    frame["fault_code"][0] = int(fault_code)
    frame["exit_reason"][0] = int(exit_reason)
    generation = getattr(shared, "session_generation", None)
    frame["generation"][0] = int(generation.value) if generation is not None else 0
    frame["updated_monotonic_ns"][0] = time.monotonic_ns()
    frame["detail"][0] = detail.encode("utf-8")[:160]
    lock = getattr(shared, "component_status_lock", None)
    if lock is None:
        ring.write(frame)
    else:
        with lock:
            ring.write(frame)


def publish_component_metrics(
    shared: Any,
    component: str,
    rate_manager: Any,
    *,
    interval_s: float = 1.0,
    now_s: float | None = None,
) -> bool:
    """Best-effort, at-most-once-per-interval publication of loop metrics.

    Metrics are observability-only: every exception is contained here so a
    diagnostics failure cannot block a worker or replace its original fault.
    """
    try:
        now = time.monotonic() if now_s is None else float(now_s)
        marker_name = "_dexmani_metrics_last_publish_s"
        last = float(getattr(rate_manager, marker_name, float("-inf")))
        if now - last < interval_s:
            return False
        setattr(rate_manager, marker_name, now)
        ring = getattr(shared, f"{component}_metrics_ring", None)
        if ring is None:
            return False
        stats = rate_manager.stats
        frame = new_frame(COMPONENT_METRICS_DTYPE)
        frame["component"][0] = component.encode("utf-8")[:16]
        for field_name in (
            "target_period_s",
            "loop_count",
            "last_work_duration_s",
            "max_work_duration_s",
            "deadline_overrun_count",
            "missed_slot_count",
            "long_block_reanchor_count",
            "elapsed_s",
            "actual_hz",
        ):
            frame[field_name][0] = getattr(stats, field_name)
        frame["updated_monotonic_ns"][0] = time.monotonic_ns()
        ring.write(frame)
        return True
    except Exception:
        logger.warning("component=%s metrics publication failed", component, exc_info=True)
        return False


def read_component_metrics(shared: Any, component: str) -> dict[str, float | int | str] | None:
    ring = getattr(shared, f"{component}_metrics_ring", None)
    if ring is None:
        return None
    result = ring.read_latest()
    if result is None:
        return None
    frame, _timestamp_ns, _sequence = result
    return {
        "component": bytes(frame["component"][0]).rstrip(b"\x00").decode("utf-8", errors="replace"),
        "target_period_s": float(frame["target_period_s"][0]),
        "loop_count": int(frame["loop_count"][0]),
        "last_work_duration_s": float(frame["last_work_duration_s"][0]),
        "max_work_duration_s": float(frame["max_work_duration_s"][0]),
        "deadline_overrun_count": int(frame["deadline_overrun_count"][0]),
        "missed_slot_count": int(frame["missed_slot_count"][0]),
        "long_block_reanchor_count": int(frame["long_block_reanchor_count"][0]),
        "elapsed_s": float(frame["elapsed_s"][0]),
        "actual_hz": float(frame["actual_hz"][0]),
    }


def format_component_metrics_summary(shared: Any) -> str:
    parts: list[str] = []
    for component in ("arm", "hand", "camera", "policy"):
        metrics = read_component_metrics(shared, component)
        if metrics is None:
            continue
        parts.append(
            f"{component}={metrics['actual_hz']:.1f}Hz/"
            f"max={1000.0 * float(metrics['max_work_duration_s']):.1f}ms/"
            f"over={metrics['deadline_overrun_count']}/miss={metrics['missed_slot_count']}/"
            f"reanchor={metrics['long_block_reanchor_count']}"
        )
    return ", ".join(parts) if parts else "unavailable"


def read_arm_state_dict(shared: "SharedStorage") -> "dict | None":
    """Read latest arm state from ring. Return dict of numpy arrays or None.

    Fields: qpos(7), qvel(7), tau(7), eef_pos(3), eef_rot6d(6),
            error_code, connected, tracking_err, and last-command timing.
    Callers must validate fields they depend on (e.g. ``np.all(np.isfinite(d["qpos"]))``).
    """
    data = read_arm_state(shared)
    if data is None:
        return None
    return {
        "qpos": np.asarray(data["qpos"][0], dtype=np.float64),
        "qvel": np.asarray(data["qvel"][0], dtype=np.float64),
        "tau": np.asarray(data["tau"][0], dtype=np.float64),
        "eef_pos": np.asarray(data["eef_pos"][0], dtype=np.float64),
        "eef_rot6d": np.asarray(data["eef_rot6d"][0], dtype=np.float64),
        "error_code": int(data["error_code"][0]),
        "connected": bool(data["connected"][0]),
        "tracking_err": float(data["tracking_err"][0]),
        "last_cmd_seq": int(data["last_cmd_seq"][0]),
        "last_cmd_created_s": float(data["last_cmd_created_s"][0]),
        "last_cmd_received_s": float(data["last_cmd_received_s"][0]),
        "last_cmd_applied_s": float(data["last_cmd_applied_s"][0]),
        "last_cmd_queue_latency_s": float(data["last_cmd_queue_latency_s"][0]),
        "last_cmd_apply_latency_s": float(data["last_cmd_apply_latency_s"][0]),
        "last_cmd_sdk_duration_s": float(data["last_cmd_sdk_duration_s"][0]),
        "last_cmd_is_hold": bool(data["last_cmd_is_hold"][0]),
        "source_monotonic_ns": int(data["source_monotonic_ns"][0]),
        "publish_monotonic_ns": int(data["publish_monotonic_ns"][0]),
        "state_valid": bool(data["state_valid"][0]),
    }


def read_hand_state_dict(shared: "SharedStorage") -> "dict | None":
    """Read latest hand state from ring. Return dict of numpy arrays or None.

    Fields: qpos(12), current(12), tactile_sum(5,3), tactile_contact(5),
            connected, error_state, qpos_stale.
    """
    data = read_hand_state(shared)
    if data is None:
        return None
    return {
        "qpos": np.asarray(data["qpos"][0], dtype=np.float64),
        "current": np.asarray(data["current"][0], dtype=np.float64),
        "tactile_sum": np.asarray(data["tactile_sum"][0], dtype=np.float64),
        "tactile_contact": np.asarray(data["tactile_contact"][0], dtype=bool),
        "connected": bool(data["connected"][0]),
        "error_state": bool(data["error_state"][0]),
        "qpos_stale": bool(data["qpos_stale"][0]),
        "source_monotonic_ns": int(data["source_monotonic_ns"][0]),
        "publish_monotonic_ns": int(data["publish_monotonic_ns"][0]),
        "state_valid": bool(data["state_valid"][0]),
        "send_healthy": bool(data["send_healthy"][0]),
        "read_healthy": bool(data["read_healthy"][0]),
    }
