"""SharedStorage — centralized data plane for cross-process communication.

A single class owns all rings, queues, events, and flags. Processes exchange data
through it — no direct references, no RPC, no business logic.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
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
        ("commboard_err", "<i4", (12,)),
        ("jointboard_err", "<i4", (12,)),
        ("tipboard_err", "<i4", (12,)),
        ("timestamp", "<f8"),
    ]
)  # 472 bytes (no tactile_force — that's in hand_tactile_ring)

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


# ═══════════════════════════════════════════════ Shared entry-point helpers


def read_arm_state_dict(shared: "SharedStorage") -> "dict | None":
    """Read latest arm state from ring. Return dict of numpy arrays or None.

    Fields: qpos(7), qvel(7), tau(7), eef_pos(3), eef_rot6d(6),
            error_code, connected, tracking_err.
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
    }


def shutdown_processes(shared: "SharedStorage", procs: "list[mp.Process]") -> None:
    """Graceful shutdown: is_running=False, join(5s), terminate stragglers, close shared.

    Safe to call with an empty list (e.g. dry-run).
    """
    shared.is_running.value = False
    _status: list[str] = []
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
            p.join(timeout=1)
            _status.append(f"{p.name}=term")
        else:
            _status.append(f"{p.name}=ok")
    shared.close()
    if _status:
        print(f"  shutdown: {'  '.join(_status)}")


def wait_for_arm_home(
    shared: "SharedStorage",
    home_qpos: "np.ndarray",
    *,
    timeout_s: float = 20.0,
    tol_rad: float = 0.03,
    heartbeat: bool = False,
    verbose: bool = True,
) -> bool:
    """Poll arm_state_ring until qpos converges to *home_qpos* or timeout.

    Does NOT use ``wrap_nearest_equivalent`` — ``arm_loop._planned_homing``
    (triggered by ``HOME_SENTINEL``) already handles joint-band wrapping and
    finishes with ``set_servo_angle(home_qpos)``, so joints converge to the
    canonical home position.

    If *heartbeat* is True, ticks ``policy_heartbeat_s`` on every poll so the
    supervisor does not FAULT during homing (required inside policy_loop).

    Returns True if home is reached, False on timeout.
    """
    _deadline = time.monotonic() + timeout_s
    while time.monotonic() < _deadline:
        if heartbeat:
            shared.policy_heartbeat_s.value = time.monotonic()
        _as = read_arm_state(shared)
        if _as is not None:
            _q = np.asarray(_as["qpos"][0], dtype=np.float64)
            if np.all(np.isfinite(_q)):
                if float(np.max(np.abs(_q - home_qpos))) < tol_rad:
                    if verbose:
                        print("  arm: home reached", flush=True)
                    return True
        time.sleep(0.1)
    if verbose:
        print(f"  arm: home settle timeout ({timeout_s:.0f}s)", flush=True)
    return False


def send_arm_home(
    shared: "SharedStorage",
    home_qpos: "np.ndarray",
    *,
    planner: "Any | None" = None,
    table_z_surface_m: float = 0.0,
    current_qpos: "np.ndarray | None" = None,
    queue_timeout: float = 2.0,
    converge_timeout_s: float = 15.0,
    heartbeat: bool = True,
    verbose: bool = True,
) -> bool:
    """Send arm to home via collision-safe path and wait for convergence.

    Encapsulates the full home sequence used by all entry points:
    1. Read current qpos (from *current_qpos* or ``arm_state_ring``).
    2. Plan collision-safe path via ``plan_joint_home_path``.
    3. Queue ``(HOME_SENTINEL, waypoints)`` to ``arm_action_q``.
    4. Wait for convergence via ``wait_for_arm_home``.

    If *planner* is None (post-exit callers), uses equivalent-joint wrapping
    without collision checking.

    Returns True if home reached, False on timeout or error.
    """
    from dexmani_real.planning.path_utils import plan_joint_home_path

    # ── Step 1: resolve current qpos ──
    if current_qpos is None:
        _as = read_arm_state(shared)
        current_qpos = home_qpos.copy()
        if _as is not None and np.all(np.isfinite(_as["qpos"][0])):
            current_qpos = np.asarray(_as["qpos"][0], dtype=np.float64)

    # ── Step 2: plan collision-safe path ──
    try:
        _waypoints = plan_joint_home_path(
            current_qpos, home_qpos, planner, table_z_surface_m=table_z_surface_m
        )
    except Exception:
        if verbose:
            print("  arm: home path planning failed — falling back to direct home", flush=True)
        _waypoints = None

    # ── Step 3: queue HOME_SENTINEL ──
    try:
        shared.arm_action_q.put((HOME_SENTINEL, _waypoints), timeout=queue_timeout)
    except Exception:
        if verbose:
            print("  arm_action_q put failed — arm may have already exited", flush=True)
        return False

    # ── Step 4: wait for convergence ──
    return wait_for_arm_home(
        shared, home_qpos, timeout_s=converge_timeout_s,
        tol_rad=np.deg2rad(2.0), heartbeat=heartbeat, verbose=verbose,
    )


def run_supervisor(
    shared: "SharedStorage",
    procs: "list",
    proc_names: "list[str]",
    heartbeat_fields: "dict[str, Any]",
    *,
    status_interval_s: float = 30.0,
) -> "tuple[str, bool]":
    """Run the standard 10 Hz supervisor loop — monitors process health, error state, and heartbeats.

    Returns ``(exit_reason, normal_exit)``.  *exit_reason* describes why the
    supervisor stopped; *normal_exit* is True for user-requested clean exits
    (Q key or KeyboardInterrupt), False for faults.

    The caller should have already transitioned to ARMED before calling this
    and must handle shutdown + DISARMED transition after it returns.
    """
    import time as _time

    from dexmani_real.config.defaults import safety
    from dexmani_real.robot.safety import SafetyState, transition

    _start_time = _time.monotonic()
    _last_status_s = _start_time
    _exit_reason = "unknown"
    normal_exit = False

    try:
        while True:
            # 0. Normal exit — Policy set is_running=False (Q key).
            if not shared.is_running.value:
                normal_exit = True
                _exit_reason = "is_running=False (Q key)"
                break

            # 1. Process aliveness.
            for _p, _name in zip(procs, proc_names):
                if not _p.is_alive():
                    _exit_reason = f"process={_name} died"
                    transition(shared, SafetyState.FAULT)
                    break
            if shared.safety_state.value == int(SafetyState.FAULT):
                if shared.error_state.value:
                    _exit_reason = "error_state set (subprocess)"
                elif shared.estop_request.value:
                    _exit_reason = "e-stop (subprocess)"
                else:
                    _exit_reason = "FAULT set by subprocess"
                break

            # 2. Error state (sticky latch from arm/hand).
            if shared.error_state.value:
                _exit_reason = "error_state set"
                transition(shared, SafetyState.FAULT)
                break

            # 3. Heartbeat timeouts.
            _now = _time.monotonic()
            for _name in proc_names:
                _last_hb = float(heartbeat_fields[_name].value)
                _age_s = _now - _last_hb if _last_hb > 0 else float("inf")
                _timeout_s = float(safety.heartbeat_timeouts[_name])
                if _age_s > _timeout_s:
                    _exit_reason = f"heartbeat={_name} timeout={_age_s:.1f}s>{_timeout_s:.1f}s"
                    transition(shared, SafetyState.FAULT)
                    break
            if shared.safety_state.value == int(SafetyState.FAULT):
                break

            # 4. Periodic status print.
            if _now - _last_status_s >= status_interval_s:
                _runtime_m = (_now - _start_time) / 60.0
                _safety = shared.safety_state.value
                _hb_ages = ", ".join(
                    f"{n}={_now - float(heartbeat_fields[n].value):.1f}s" for n in proc_names
                )
                print(
                    f"  [supervisor]  runtime={_runtime_m:.1f}min  safety={_safety}  hb_age=({_hb_ages})",
                    flush=True,
                )
                _last_status_s = _now

            _time.sleep(0.1)  # 10 Hz

    except KeyboardInterrupt:
        _exit_reason = "KeyboardInterrupt"
        normal_exit = True
        shared.is_running.value = False

    return _exit_reason, normal_exit


def wait_subsystem_ready(
    shared: "SharedStorage",
    ready_checks: "list[tuple[str, Any, float]]",
    procs: "list[mp.Process]",
) -> bool:
    """Wait for each ``(name, event, timeout_s)`` ready event to be set.

    Checks ``error_state`` and process liveness on every poll tick.
    Returns True if all subsystems are ready, False if any fail.

    The caller is responsible for printing pre-wait user messages
    (e.g. "put on Quest headset") before calling this function.
    """
    for name, ev, timeout in ready_checks:
        _deadline = time.monotonic() + timeout
        _ok = False
        _logged = False
        while time.monotonic() < _deadline:
            if ev.is_set():
                _ok = True
                break
            if shared.error_state.value:
                logger.error("subsystem=%s init failed: error_state set", name)
                _logged = True
                break
            if not all(p.is_alive() for p in procs):
                _dead_names = [p.name for p in procs if not p.is_alive()]
                logger.error(
                    "subsystem=%s init failed: process(es) %s exited prematurely",
                    name,
                    _dead_names,
                )
                _logged = True
                break
            time.sleep(0.2)
        if not _ok and not _logged:
            logger.error("subsystem=%s ready_timeout=%ds", name, timeout)
        if not _ok:
            return False
    return True


def print_health_summary(shared: "SharedStorage") -> None:
    """Print a pre-flight health summary from ring data (arm, hand, VR, camera)."""
    print("\n── Health Check ──")

    # Arm
    arm_result = shared.arm_state_ring.read_latest()
    if arm_result is not None:
        arm_data, _, _ = arm_result
        arm_connected = bool(arm_data["connected"][0])
        arm_error = int(arm_data["error_code"][0])
        arm_qpos = np.asarray(arm_data["qpos"][0], dtype=np.float64)
        arm_qpos_ok = int(np.all(np.isfinite(arm_qpos)))
        arm_ok = arm_connected and arm_error == 0 and bool(arm_qpos_ok)
        print(
            f"  arm   {'OK' if arm_ok else 'FAIL':>4s}  connected={int(arm_connected)}  "
            f"error={arm_error}  qpos_ok={arm_qpos_ok}"
        )
    else:
        print("  arm   ----  (no data yet)")

    # Hand
    hand_result = shared.hand_state_ring.read_latest()
    if hand_result is not None:
        hand_data, _, _ = hand_result
        hand_connected = bool(hand_data["connected"][0])
        hand_error = bool(hand_data["error_state"][0])
        hand_qpos_stale = bool(hand_data["qpos_stale"][0])
        hand_qpos = np.asarray(hand_data["qpos"][0], dtype=np.float64)
        hand_qpos_ok = int(np.all(np.isfinite(hand_qpos)))
        hand_ok = hand_connected and not hand_error and bool(hand_qpos_ok)
        stale_note = " stale=1" if hand_qpos_stale else ""
        print(
            f"  hand  {'OK' if hand_ok else 'FAIL':>4s}  connected={int(hand_connected)}  "
            f"error={int(hand_error)}  qpos_ok={hand_qpos_ok}{stale_note}"
        )
    else:
        print("  hand  ----  (no data yet)")

    # VR
    vr_result = shared.vr_ring.read_latest()
    if vr_result is not None:
        vr_data, _, _ = vr_result
        vr_age_s = (
            (time.monotonic_ns() - int(vr_data["local_recv_ns"][0])) / 1e9 if vr_data["local_recv_ns"][0] > 0 else -1
        )
        print(f"  vr     OK   age={vr_age_s:.1f}s  seq={int(vr_data['sequence_id'][0])}")
    else:
        print("  vr    ----  (no data yet)")

    # Camera
    cam_serial_bytes = shared.camera_serial.value.rstrip(b"\x00")
    if cam_serial_bytes:
        print(f"  cam    OK   serial={cam_serial_bytes.decode()}")
    elif shared.camera_heartbeat_s.value > 0:
        print("  cam    OK   serial=unknown")
    else:
        print("  cam   ----  (no data yet)")

    print("──")
    sys.stdout.flush()
