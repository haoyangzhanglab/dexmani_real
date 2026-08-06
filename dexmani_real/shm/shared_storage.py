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

from dexmani_real.config.defaults import camera, policy
from dexmani_real.robot.safety import SafetyState
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

    camera_ring_maxlen: int = 5
    vr_ring_maxlen: int = 8
    arm_state_ring_maxlen: int = 8
    hand_state_ring_maxlen: int = 8
    hand_tactile_ring_maxlen: int = 8
    hand_cmd_ring_maxlen: int = 8

    camera_rgb_shape: tuple[int, int, int] = field(default_factory=lambda: camera.rgb_shape)
    camera_depth_shape: tuple[int, int] = field(default_factory=lambda: camera.depth_shape)
    camera_pc_shape: tuple[int, int] = (2048, 6)

    arm_action_q_maxsize: int = 2

    workspace_bounds: "np.ndarray" = field(default_factory=lambda: policy.workspace.as_array())


HOME_SENTINEL = "__HOME__"

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
    """

    camera_ring: CameraRingBuffer  # camera -> policy
    vr_ring: SeqlockRingBuffer  # vr -> policy
    arm_state_ring: SeqlockRingBuffer  # arm -> policy
    hand_state_ring: SeqlockRingBuffer  # hand -> policy
    hand_tactile_ring: SeqlockRingBuffer  # hand -> policy (sparse)
    hand_cmd_ring: SeqlockRingBuffer  # policy -> hand

    arm_action_q: mp.Queue  # policy -> arm, maxsize=2

    is_running: Any  # Main -> all
    is_recording: Any  # policy -> arm/hand/camera
    error_state: Any  # arm/hand -> all (sticky latch)
    estop_request: Any  # policy -> arm/hand
    quit_requested: Any  # policy -> Main

    safety_state: Any  # SafetyState enum (0-3), Main + policy write

    arm_heartbeat_s: Any
    hand_heartbeat_s: Any
    policy_heartbeat_s: Any
    vr_heartbeat_s: Any
    camera_heartbeat_s: Any

    arm_ready: Any  # -> Main
    hand_ready: Any  # -> Main
    camera_ready: Any  # -> Main
    vr_ready: Any  # -> Main

    camera_depth_scale: Any  # depth scale (mm to meters)
    camera_K: Any  # 3x3 intrinsics, row-major
    camera_serial: Any  # serial number string

    @classmethod
    def create(
        cls,
        prefix: str = "dexmani",
        *,
        config: SharedStorageConfig | None = None,
        camera_rgb_shape: tuple[int, int, int] | None = None,
        camera_depth_shape: tuple[int, int] | None = None,
    ) -> "SharedStorage":
        """Create all rings, queues, flags, events, and heartbeats.

        Call once from Main before spawning child processes.
        """
        cfg = config or SharedStorageConfig()

        _rgb_shape = camera_rgb_shape or cfg.camera_rgb_shape
        _depth_shape = camera_depth_shape or cfg.camera_depth_shape

        storage = cls.__new__(cls)

        storage.camera_ring = CameraRingBuffer(
            name=f"{prefix}_camera",
            rgb_shape=_rgb_shape,
            depth_shape=_depth_shape,
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

        storage.arm_action_q = mp.Queue(maxsize=cfg.arm_action_q_maxsize)

        storage.is_running = mp.Value("b", True)
        storage.is_recording = mp.Value("b", False)
        storage.error_state = mp.Value("b", False)
        storage.estop_request = mp.Value("b", False)
        storage.quit_requested = mp.Value("b", False)

        storage.safety_state = mp.Value("i", int(SafetyState.DISARMED))

        storage.arm_heartbeat_s = mp.Value("d", 0.0)
        storage.hand_heartbeat_s = mp.Value("d", 0.0)
        storage.policy_heartbeat_s = mp.Value("d", 0.0)
        storage.vr_heartbeat_s = mp.Value("d", 0.0)
        storage.camera_heartbeat_s = mp.Value("d", 0.0)

        storage.arm_ready = mp.Event()
        storage.hand_ready = mp.Event()
        storage.camera_ready = mp.Event()
        storage.vr_ready = mp.Event()

        storage.camera_depth_scale = mp.Value("d", 0.0)
        storage.camera_K = mp.Array("d", [0.0] * 9)
        storage.camera_serial = mp.Array("c", b"\x00" * 32)

        logger.info("SharedStorage created (prefix=%s)", prefix)
        return storage

    def close(self) -> None:
        """Release all shared memory primitives.

        ``unlink()`` destroys the POSIX shared-memory segment, preventing
        Python's resource tracker "leaked shared_memory objects" warning.
        """
        _close_errors: list[str] = []

        for ring_name, ring in (
            ("camera_ring", self.camera_ring),
            ("vr_ring", self.vr_ring),
            ("arm_state_ring", self.arm_state_ring),
            ("hand_state_ring", self.hand_state_ring),
            ("hand_tactile_ring", self.hand_tactile_ring),
            ("hand_cmd_ring", self.hand_cmd_ring),
        ):
            try:
                ring.close()  # type: ignore[attr-defined]
            except Exception:
                _close_errors.append(f"{ring_name}.close() failed")

            try:
                ring.unlink()  # type: ignore[attr-defined]
            except FileNotFoundError:
                pass  # already unlinked by another process — expected
            except Exception:
                _close_errors.append(f"{ring_name}.unlink() failed")

        try:
            self.arm_action_q.close()
            self.arm_action_q.join_thread()
        except Exception:
            _close_errors.append("arm_action_q cleanup failed")

        if _close_errors:
            logger.warning("SharedStorage close: %d error(s): %s", len(_close_errors), "; ".join(_close_errors))
        else:
            logger.info("SharedStorage closed cleanly")


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


def write_hand_cmd(shared: "SharedStorage", qpos: "np.ndarray") -> None:
    """Write hand position command to ring (latest-wins)."""
    frame = new_frame(HAND_CMD_DTYPE)
    frame["qpos_cmd"][0] = np.asarray(qpos, dtype=np.float64).ravel()[:12]
    shared.hand_cmd_ring.write(frame)


def hand_home_converge(
    shared: SharedStorage,
    home_qpos: np.ndarray,
    *,
    timeout_s: float = 5.0,
    tol_deg: float = 5.0,
    heartbeat: bool = False,
    check_is_running: bool = True,
    verbose: bool = True,
) -> "tuple[bool, np.ndarray | None]":
    """Poll hand_state_ring until hand converges to home_qpos.

    Returns ``(reached, final_qpos_or_none)``.
    """
    tol = np.deg2rad(tol_deg)
    deadline = time.monotonic() + timeout_s
    first = True

    while time.monotonic() < deadline:
        if check_is_running and not shared.is_running.value:
            break
        if heartbeat:
            shared.policy_heartbeat_s.value = time.monotonic()
        write_hand_cmd(shared, home_qpos)
        hs = read_hand_state(shared)
        if hs is not None:
            current = np.asarray(hs["qpos"][0], dtype=np.float64)
            if np.all(np.isfinite(current)):
                err = float(np.max(np.abs(current - home_qpos)))
                if err < tol:
                    if verbose:
                        print("  hand: home reached", flush=True)
                    return True, current.copy()
                if verbose and first:
                    print(f"  hand: homing... (max_err={np.rad2deg(err):.0f}°)", flush=True)
                    first = False
        time.sleep(0.05)

    if verbose:
        print(f"  hand: home settle timeout after {timeout_s:.0f}s — proceeding", flush=True)
    return False, None


def read_arm_state_k(shared: "SharedStorage", k: int) -> "list[np.ndarray]":
    """Read up to *k* most recent arm state frames (oldest-first), may be shorter than *k*."""
    frames = shared.arm_state_ring.get_last_k(k)
    return [data for data, _ts, _seq in frames]


def read_hand_state_k(shared: "SharedStorage", k: int) -> "list[np.ndarray]":
    """Read up to *k* most recent hand state frames (oldest-first), may be shorter than *k*."""
    frames = shared.hand_state_ring.get_last_k(k)
    return [data for data, _ts, _seq in frames]
