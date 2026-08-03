"""SharedStorage — centralized data plane for cross-process communication.

A single class owns all rings, queues, events, and flags. Processes exchange data
through it — no direct references, no RPC, no business logic.

Ref: ManiUniCon SharedStorage pattern.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dexmani_real.config.defaults import camera, policy
from dexmani_real.robot.safety import SafetyState
from dexmani_real.shm.ring_buffer import CameraRingBuffer
from dexmani_real.shm.robot_ring import SeqlockRingBuffer
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SharedStorageConfig — centralized tuning constants
# ═══════════════════════════════════════════════════════════════════════════════


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

    # ── Ring capacities ──
    camera_ring_maxlen: int = 5
    vr_ring_maxlen: int = 8
    arm_state_ring_maxlen: int = 3
    hand_state_ring_maxlen: int = 3
    hand_tactile_ring_maxlen: int = 8
    hand_cmd_ring_maxlen: int = 8

    # ── Camera defaults — sourced from camera singleton ──
    camera_rgb_shape: tuple[int, int, int] = field(default_factory=lambda: camera.rgb_shape)
    camera_depth_shape: tuple[int, int] = field(default_factory=lambda: camera.depth_shape)

    # ── Queue sizes ──
    arm_action_q_maxsize: int = 2

    # ── Workspace bounds (arm base frame, meters) — sourced from policy singleton ──
    workspace_bounds: "np.ndarray" = field(default_factory=lambda: policy.workspace.as_array())


# ── Home sentinel for arm_action_q ──
# Policy puts this sentinel to request homing; arm_loop detects and executes.
# Using a string sentinel (not None — None means "no action / hold").
HOME_SENTINEL = "__HOME__"

# ── Compact dtypes for SeqlockRingBuffer structured arrays ──

ARM_STATE_DTYPE = np.dtype(
    [
        ("qpos", "<f8", (7,)),
        ("qvel", "<f8", (7,)),
        ("tau", "<f8", (7,)),
        ("eef_pos", "<f8", (3,)),
        ("eef_rot6d", "<f8", (6,)),
        ("error_code", "<i4"),
        ("connected", "<u1"),
        ("mode", "<i4"),
        ("tracking_err", "<f8"),
        ("timestamp", "<f8"),
    ]
)  # 265 bytes

HAND_STATE_DTYPE = np.dtype(
    [
        ("qpos", "<f8", (12,)),
        ("current", "<f8", (12,)),
        ("tactile_sum", "<f8", (5, 3)),
        ("tactile_contact", "<u1", (5,)),
        ("error_state", "<u1"),
        ("connected", "<u1"),
        ("qpos_stale", "<u1"),
        ("timestamp", "<f8"),
    ]
)  # 328 bytes (no tactile_force — that's in hand_tactile_ring)

HAND_CMD_DTYPE = np.dtype(
    [
        ("qpos_cmd", "<f8", (12,)),
    ]
)  # 96 bytes

HAND_TACTILE_DTYPE = np.dtype(
    [
        ("tactile_force", "<f8", (5, 120, 3)),
    ]
)  # 14,400 bytes


def new_frame(dtype: np.dtype) -> np.ndarray:
    """Allocate a zero-initialized 1-element structured array for ring writes."""
    return np.zeros(1, dtype=dtype)


@dataclass
class SharedStorage:
    """Central data plane — all cross-process state in one place.

    Created by Main before spawning child processes. Each process receives a
    reference and reads/writes its designated rings/queues/flags.

    Usage::

        shared = SharedStorage.create(prefix="dexmani",
                                       camera_rgb_shape=(480, 848, 3),
                                       camera_depth_shape=(480, 848))
        # ... spawn processes with shared ...
        shared.close()
    """

    # ---- Rings: continuous streams, read-latest ----
    camera_ring: CameraRingBuffer  # camera_loop  -> policy_loop
    vr_ring: SeqlockRingBuffer  # VRProcess      -> PolicyProcess
    arm_state_ring: SeqlockRingBuffer  # arm_loop     -> PolicyProcess
    hand_state_ring: SeqlockRingBuffer  # hand_loop    -> PolicyProcess
    hand_tactile_ring: SeqlockRingBuffer  # hand_loop    -> PolicyProcess (sparse)
    hand_cmd_ring: SeqlockRingBuffer  # PolicyProcess  -> hand_loop (latest-wins)

    # ---- Queue: ordered actions (arm only — Mode 6 needs ordering) ----
    arm_action_q: mp.Queue  # PolicyProcess -> arm_loop, maxsize=2

    # ---- Flags ----
    is_running: Any  # mp.Value('b') — Main -> all processes (sole writer)
    is_recording: Any  # mp.Value('b') — PolicyProcess -> Arm/Hand/Camera
    error_state: Any  # mp.Value('b') — Arm/Hand -> all (sticky latch, set-only)
    estop_request: Any  # mp.Value('b') — PolicyProcess -> Arm/Hand

    # ---- Safety state machine (ManiUniCon P0) ----
    safety_state: Any  # mp.Value('i') — SafetyState enum (0-3), Main + Policy write

    # ---- Process heartbeats (each process writes its own, Main monitors) ----
    arm_heartbeat_s: Any  # mp.Value('d') — arm_loop writes time.monotonic()
    hand_heartbeat_s: Any  # mp.Value('d') — hand_loop writes time.monotonic()
    policy_heartbeat_s: Any  # mp.Value('d') — policy_loop writes time.monotonic()
    vr_heartbeat_s: Any  # mp.Value('d') — vr_loop writes time.monotonic()
    camera_heartbeat_s: Any  # mp.Value('d') — camera_loop writes time.monotonic()

    # ---- Events ----
    arm_ready: Any  # mp.Event — arm_loop -> Main
    hand_ready: Any  # mp.Event — hand_loop -> Main
    camera_ready: Any  # mp.Event — camera_loop -> Main
    vr_ready: Any  # mp.Event — vr_loop -> Main

    # ---- Diagnostics ----

    # ---- Camera metadata (set by camera_loop, read by policy_loop) ----
    camera_depth_scale: Any  # mp.Value('d') — depth scale (mm to meters)
    camera_K: Any  # mp.Array('d', 9) — 3x3 intrinsics matrix (row-major)
    camera_serial: Any  # mp.Array('c', 32) — camera serial number string

    # ---- Recording metadata (Policy writes) ----
    record_dir: str | None = None
    record_dt: float = 1.0 / 16.0

    @classmethod
    def create(
        cls,
        prefix: str = "dexmani",
        *,
        config: SharedStorageConfig | None = None,
        camera_rgb_shape: tuple[int, int, int] | None = None,
        camera_depth_shape: tuple[int, int] | None = None,
    ) -> "SharedStorage":
        """Create all shared memory primitives.

        Call once from Main before spawning child processes.
        Each child calls the per-ring attach constructor (create=False).

        Args:
            prefix: Name prefix for POSIX shared memory segments.
            config: Optional :class:`SharedStorageConfig` to set ring
                capacities, camera defaults, and workspace bounds.
            camera_rgb_shape: Override RGB frame shape (H, W, C). Takes
                precedence over *config* values when both are provided.
            camera_depth_shape: Override depth frame shape (H, W). Same
                precedence rule as *camera_rgb_shape*.
        """
        cfg = config or SharedStorageConfig()

        _rgb_shape = camera_rgb_shape or cfg.camera_rgb_shape
        _depth_shape = camera_depth_shape or cfg.camera_depth_shape

        storage = cls.__new__(cls)

        # ---- Rings (create=True — Main owns creation) ----
        storage.camera_ring = CameraRingBuffer(
            name=f"{prefix}_camera",
            rgb_shape=_rgb_shape,
            depth_shape=_depth_shape,
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

        # ---- Queue (bounded — provides backpressure) ----
        storage.arm_action_q = mp.Queue(maxsize=cfg.arm_action_q_maxsize)

        # ---- Flags ----
        storage.is_running = mp.Value("b", True)
        storage.is_recording = mp.Value("b", False)
        storage.error_state = mp.Value("b", False)
        storage.estop_request = mp.Value("b", False)

        # ---- Safety state machine ----
        storage.safety_state = mp.Value("i", int(SafetyState.DISARMED))

        # ---- Heartbeats ----
        storage.arm_heartbeat_s = mp.Value("d", 0.0)
        storage.hand_heartbeat_s = mp.Value("d", 0.0)
        storage.policy_heartbeat_s = mp.Value("d", 0.0)
        storage.vr_heartbeat_s = mp.Value("d", 0.0)
        storage.camera_heartbeat_s = mp.Value("d", 0.0)

        # ---- Events ----
        storage.arm_ready = mp.Event()
        storage.hand_ready = mp.Event()
        storage.camera_ready = mp.Event()
        storage.vr_ready = mp.Event()

        # ---- Camera metadata (written by camera_loop) ----
        storage.camera_depth_scale = mp.Value("d", 0.0)
        storage.camera_K = mp.Array("d", [0.0] * 9)
        storage.camera_serial = mp.Array("c", b"\x00" * 32)

        logger.info("SharedStorage created (prefix=%s)", prefix)
        return storage

    def close(self) -> None:
        """Release all shared memory and multiprocessing primitives.

        Calls ``close()`` (release fd) then ``unlink()`` (destroy the POSIX
        shared-memory segment) on every ring.  ``unlink()`` is required to avoid
        the "leaked shared_memory objects" warning from Python's resource tracker;
        *close* alone only releases the local file descriptor — the kernel
        segment (and the tracker registration) survives until ``unlink()`` is
        called by the creating process.
        """
        for ring in (
            self.camera_ring,
            self.vr_ring,
            self.arm_state_ring,
            self.hand_state_ring,
            self.hand_tactile_ring,
            self.hand_cmd_ring,
        ):
            try:
                ring.close()  # type: ignore[attr-defined]
                ring.unlink()  # type: ignore[attr-defined]
            except Exception:
                pass

        try:
            self.arm_action_q.close()
            self.arm_action_q.join_thread()
        except Exception:
            pass

        logger.info("SharedStorage closed")


def vr_frame_dtype() -> np.dtype:
    """VR frame dtype — mirrors vr_receiver_process frame output."""
    return np.dtype(
        [
            ("wrist_pos", "<f8", (3,)),
            ("wrist_quat_wxyz", "<f8", (4,)),
            ("landmarks", "<f8", (21, 3)),
            ("head_pos", "<f8", (3,)),
            ("head_quat_wxyz", "<f8", (4,)),
            ("recv_ts_ns", "<u8"),
            ("source_ts_ns", "<u8"),
            ("sequence_id", "<u8"),
            ("source_frame_seq", "<u8"),
            ("local_recv_ns", "<u8"),
            ("side", "<i4"),
        ],
        align=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Shared ring read/write helpers (single source of truth for all entry points)
# ═══════════════════════════════════════════════════════════════════════════════


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


def write_hand_cmd(shared: "SharedStorage", qpos: "np.ndarray") -> None:
    """Write hand position command to ring (latest-wins)."""
    frame = new_frame(HAND_CMD_DTYPE)
    frame["qpos_cmd"][0] = np.asarray(qpos, dtype=np.float64).ravel()[:12]
    shared.hand_cmd_ring.write(frame)
