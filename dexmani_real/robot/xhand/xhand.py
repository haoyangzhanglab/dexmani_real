"""XHand 12-DOF robot hand hardware driver via xhand_controller SDK."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dexmani_real.log import get_logger
from dexmani_real.robot._connection_state import ConnectionStateMixin
from dexmani_real.utils.array_utils import nan_array
from dexmani_real.utils.rate_limiter import RateLimiter
from dexmani_real.utils.serialization import from_dict_helper
from xhand_controller import xhand_control as xh

logger = get_logger(__name__)


JOINT_NAMES = [
    "thumb_abduction",
    "thumb_joint1",
    "thumb_joint2",
    "index_abduction",
    "index_joint1",
    "index_joint2",
    "middle_joint1",
    "middle_joint2",
    "ring_joint1",
    "ring_joint2",
    "little_joint1",
    "little_joint2",
]

SENSOR_IDS = [0x11, 0x12, 0x13, 0x14, 0x15]
SENSOR_NAMES = ["thumb", "index", "middle", "ring", "little"]

# Known non-critical sensor error patterns (ref: LeFranX xhand.py:230-241).
# These are hardware-level warnings that do not affect joint position reading
# or motion control — filtering them prevents spurious error_state triggers
# during normal teleoperation.
_KNOWN_SENSOR_ERROR_PATTERNS = [
    "sensor fails to read the combined force",
    "sensor fails to read the distributed force",
    "sensor fails to read temperature",
    "communication data crc error",
    "this hardware version does not support force control mode",
]

# ── F2: Predefined grasp presets (ref: DexUMI constants.py:18-21) ──
# Each preset is a dict with "qpos" (12,) array in radians and "description".
GRASP_PRESETS: dict[str, dict] = {
    "home": {
        "qpos": np.deg2rad(np.array([
            0.0, 45.0, 0.0,   # thumb
            0.0, 0.0, 0.0,    # index
            0.0, 0.0,          # middle
            0.0, 0.0,          # ring
            0.0, 0.0,          # little
        ], dtype=np.float64)),
        "description": "Open hand, all fingers extended",
    },
    "open": {
        "qpos": np.deg2rad(np.array([
            0.0, 45.0, 0.0,   # thumb
            0.0, 0.0, 5.0,    # index
            0.0, 5.0,          # middle
            0.0, 5.0,          # ring
            0.0, 5.0,          # little
        ], dtype=np.float64)),
        "description": "Wide-open hand",
    },
    "fist": {
        "qpos": np.deg2rad(np.array([
            50.0, 90.0, 90.0,   # thumb
            10.0, 110.0, 110.0, # index
            110.0, 110.0,        # middle
            110.0, 110.0,        # ring
            110.0, 110.0,        # little
        ], dtype=np.float64)),
        "description": "Full fist closure",
    },
    "pinch": {
        "qpos": np.deg2rad(np.array([
            30.0, 60.0, 45.0,   # thumb
            5.0, 50.0, 60.0,    # index
            0.0, 5.0,            # middle (open)
            0.0, 5.0,            # ring (open)
            0.0, 5.0,            # little (open)
        ], dtype=np.float64)),
        "description": "Thumb-index pinch grip",
    },
    "tripod": {
        "qpos": np.deg2rad(np.array([
            35.0, 65.0, 60.0,    # thumb
            5.0, 50.0, 60.0,     # index
            50.0, 60.0,           # middle
            0.0, 5.0,             # ring (open)
            0.0, 5.0,             # little (open)
        ], dtype=np.float64)),
        "description": "Thumb-index-middle tripod grip",
    },
}


@dataclass
class XHandConfig:
    comm_type: str = "RS485"
    device_name: str | None = None
    baudrate: int = 3_000_000
    device_id: int = 0

    # Connection retry (RS485 may need several attempts after cold start)
    open_serial_retries: int = 5
    open_serial_retry_delay_s: float = 2.0

    dt: float = 1.0 / 83.0
    num_joints: int = 12

    # Important:
    # True  -> force SDK to refresh state from hardware.
    # False -> may return SDK cached state. After open_serial(), cache may be all zeros.
    force_update_state: bool = True

    # Connect-time state initialization.
    # Even if force_update_state is manually set to False for runtime speed,
    # connect() should still force refresh several frames to avoid zero-cache initialization.
    init_state_read_attempts: int = 10
    init_state_read_interval: float = 0.02

    home_qpos: np.ndarray = field(default_factory=lambda: np.deg2rad(np.array([
        0.0, 45.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0,
        0.0, 0.0,
        0.0, 0.0,
    ], dtype=np.float64)))

    # Distal joints (index/middle/ring/little joint2) min=5°
    # to prevent mechanical clogging. ref: LeFranX xhand_config.py L50-51.
    qpos_min: np.ndarray = field(
        default_factory=lambda: np.deg2rad(np.array([
            0.0, -40.0, 0.0,
            -10.0, 0.0, 5.0,
            0.0, 5.0,
            0.0, 5.0,
            0.0, 5.0,
        ], dtype=np.float64))
    )

    qpos_max: np.ndarray = field(
        default_factory=lambda: np.deg2rad(np.array([
            105.0, 90.0, 90.0,
            10.0, 110.0, 110.0,
            110.0, 110.0,
            110.0, 110.0,
            110.0, 110.0,
        ], dtype=np.float64))
    )

    max_qvel: np.ndarray = field(default_factory=lambda: np.deg2rad(np.ones(12) * 180.0))

    kp: int = 100
    ki: int = 0
    kd: int = 1
    # Per-joint gain overrides (ref: DexUMI hand_api_cls.py:317-319).
    # When set (shape (12,)), individual joint gains replace the scalar kp/ki/kd.
    # Distal joints (especially little finger joint 11) benefit from higher gains
    # to compensate for longer linkage and higher mechanical load.
    kp_per_joint: np.ndarray | None = None  # (12,) per-joint kp overrides
    ki_per_joint: np.ndarray | None = None  # (12,) per-joint ki overrides
    kd_per_joint: np.ndarray | None = None  # (12,) per-joint kd overrides
    tor_max: int = 400  # ref: LeFranX xhand_config.py:25, DexUMI hand_api_cls.py:289 — 400mA torque-current limit
    mode: int = 3

    use_delta_limit: bool = False
    clip_joint_limit: bool = True

    tactile_scale: float = 0.1

    # Known non-critical sensor error filtering (ref: LeFranX xhand.py:230-241).
    # When True, errors matching known patterns (sensor read failures, CRC errors,
    # unsupported-force-mode warnings) are logged but do NOT trigger error_state.
    filter_known_sensor_errors: bool = True

    # ── E1: Background state reader (ref: DexUMI hand_api_cls.py:228-275) ──
    # When True, a daemon thread continuously reads hardware state at state_reader_hz
    # and caches it. get_state() returns the cached value, decoupling read timing
    # from consumption and guaranteeing consistent sample rate.
    use_background_state_reader: bool = False
    state_reader_hz: float = 100.0

    # ── E2: EMA smoothing (ref: LeFranX xhand_vr_teleoperator.py:306-308) ──
    # 0.0 = disabled (default, backward compatible). 0.3 = LeFranX recommended.
    # Exponential Moving Average filters high-frequency jitter from position commands,
    # producing smoother finger motion.
    ema_alpha: float = 0.0

    # ── F1: Tactile contact detection ──
    # L2 norm threshold (Newtons) on per-finger combined force for contact detection.
    tactile_contact_threshold: float = 0.5

    # ── F3: Trajectory interpolation ──
    # Minimum duration for trajectory execution; actual duration is max(min_duration, computed).
    traj_min_duration_s: float = 0.5

    @classmethod
    def from_dict(cls, d: dict) -> "XHandConfig":
        """Reconstruct from a serialized dict."""
        kw = from_dict_helper(cls, d)
        return cls(**kw)


class XHand(ConnectionStateMixin):
    def __init__(self, config: XHandConfig):
        super().__init__()
        self.config = config
        self.control = None
        self.device_name: str | None = None
        self.hand_command = None

        self.last_qpos_cmd: np.ndarray | None = None
        self.last_cmd_time: float | None = None
        self.last_error_code: int | None = None
        self.last_joint_limit_clipped = False
        self.last_delta_limited = False

        self.last_commboard_err = np.zeros(12, dtype=np.int32)
        self.last_jointboard_err = np.zeros(12, dtype=np.int32)
        self.last_tipboard_err = np.zeros(12, dtype=np.int32)
        self.last_hand_ids: list[int] = []
        self.last_sensor_error_filtered: str | None = None  # D1: last known-non-critical error
        self.cached_comm_type = self._resolve_comm_type()

        # ── E1: Background state reader ──
        self._state_thread: threading.Thread | None = None
        self._state_stop: threading.Event = threading.Event()
        self._state_lock: threading.Lock = threading.Lock()
        self._latest_state: dict[str, Any] | None = None

        # ── E2: EMA filtering ──
        self._ema_qpos: np.ndarray | None = None

    # ── Connect lifecycle ──

    def connect(self) -> bool:
        """Connect to XHand hardware.

        Orchestrates device enumeration, retry-based port opening, and
        initial state initialization. Returns True on success.
        """
        comm_type = self.cached_comm_type

        if not self._retry_open_device(comm_type):
            return False

        try:
            self.last_hand_ids = list(self.control.list_hands_id())
        except (OSError, RuntimeError):
            self.last_hand_ids = []

        self.connected_flag = True
        self.error_state = False
        self.hand_command = self.make_command(self._array12(self.config.home_qpos))

        self._init_hand_state()

        self.last_cmd_time = time.time()

        # E1: Start background state reader if configured
        if self.config.use_background_state_reader:
            self._start_state_reader()

        return True

    def _retry_open_device(self, comm_type: str) -> bool:
        """Enumerate devices and open port with configurable retries.

        RS485 may need several attempts after cold start (C++ SDK retries
        internally, but may still fail intermittently).
        """
        self.control = xh.XHandControl()
        if self.config.device_name is None:
            devices = self.control.enumerate_devices(comm_type)
            if devices is None or len(devices) == 0:
                self.error_state = True
                self.last_error_code = -2
                self.last_error_message = f"no XHand device found for {comm_type}"
                self._diagnose_connection_failure()
                return False
            self.device_name = devices[0]
        else:
            self.device_name = self.config.device_name

        retries = max(1, int(self.config.open_serial_retries))
        delay = max(0.0, float(self.config.open_serial_retry_delay_s))

        for attempt in range(1, retries + 1):
            if comm_type == "RS485":
                err = self.control.open_serial(self.device_name, int(self.config.baudrate))
            elif comm_type == "EtherCAT":
                err = self.control.open_ethercat(self.device_name)
            else:
                self.error_state = True
                self.last_error_code = -3
                self.last_error_message = f"unsupported comm_type: {self.config.comm_type}"
                return False

            if self.error_ok(err):
                return True

            self._record_error(err)
            if attempt < retries:
                logger.warning(
                    "XHand connect attempt %s/%s failed: %s, retrying in %.1fs...",
                    attempt, retries, self.last_error_message, delay,
                )
                # Close and recreate control for clean retry
                try:
                    self.control.close_device()
                except (OSError, RuntimeError):
                    pass
                self.control = xh.XHandControl()
                time.sleep(delay)

        # All retries exhausted
        self.error_state = True
        logger.error(
            "XHand connect failed after %s attempts: %s",
            retries, self.last_error_message,
        )
        self._diagnose_connection_failure()
        return False

    def _init_hand_state(self) -> None:
        """Force-refresh hardware state and read initial qpos.

        Do not use SDK cache here — after open_serial() the cache
        may be all zeros. Falls back to home_qpos if no valid state
        is obtained.
        """
        valid_state: dict[str, Any] | None = None
        attempts = max(1, int(self.config.init_state_read_attempts))
        interval = max(0.0, float(self.config.init_state_read_interval))

        for _ in range(attempts):
            state = self.get_state(full=True, force_update=True)
            if self.is_valid_qpos_state(state):
                valid_state = state
            if interval > 0:
                time.sleep(interval)

        if valid_state is not None:
            self.last_qpos_cmd = valid_state["qpos"].copy()
            logger.info("Initial qpos from hand state: %s", self.last_qpos_cmd)
        else:
            self.last_qpos_cmd = self._array12(self.config.home_qpos)
            logger.info("Using home_qpos as initial qpos: %s", self.last_qpos_cmd)

    def disconnect(self):
        self._stop_state_reader()  # E1
        if self.control is not None:
            self.control.close_device()
        self.connected_flag = False

    def reconnect(self) -> bool:
        """Close existing connection and re-connect.

        Returns True if reconnection succeeded.  Callers (e.g. RobotInterface)
        can use this at runtime to recover from a transient hand disconnect.
        """
        if self.control is not None:
            try:
                self.control.close_device()
            except (OSError, RuntimeError):
                pass
        self.connected_flag = False
        self.error_state = False
        time.sleep(1.0)
        return self.connect()

    def _diagnose_connection_failure(self) -> None:
        """Print diagnostic info for a failed RS485/EtherCAT connection.

        Checks /dev/ttyUSB* existence, permissions, and whether another
        process is holding the device.
        """
        comm_type = self.cached_comm_type
        device = self.device_name or "<auto>"
        logger.warning("[XHand diagnostics] comm_type=%s, device=%s", comm_type, device)

        # List available ttyUSB devices
        if comm_type == "RS485":
            import glob
            tty_devices = sorted(glob.glob("/dev/ttyUSB*"))
            if not tty_devices:
                logger.warning("  No /dev/ttyUSB* devices found. Check USB cable and power.")
            else:
                logger.warning("  Available tty devices: %s", tty_devices)
                for tty in tty_devices:
                    try:
                        st = os.stat(tty)
                        import stat
                        perms = stat.filemode(st.st_mode)
                        logger.warning("    %s: %s owner=%s group=%s", tty, perms, st.st_uid, st.st_gid)
                    except OSError as e:
                        logger.warning("    %s: stat failed — %s", tty, e)

            # Check if device is held by another process (lsof)
            target = self.device_name if self.device_name else (
                tty_devices[0] if tty_devices else None
            )
            if target is not None:
                import subprocess
                try:
                    result = subprocess.run(
                        ["lsof", target],
                        capture_output=True, text=True, timeout=5,
                    )
                    if result.stdout.strip():
                        logger.warning("  lsof %s:", target)
                        for line in result.stdout.strip().splitlines():
                            logger.warning("    %s", line)
                    else:
                        logger.warning("  lsof %s: not held by any process", target)
                except FileNotFoundError:
                    pass  # lsof not installed
                except subprocess.TimeoutExpired:
                    logger.warning("  lsof %s: timed out", target)
                except (subprocess.SubprocessError, OSError) as e:
                    logger.warning("  lsof %s: error — %s", target, e)

            # Suggestions
            logger.warning("  Troubleshooting:")
            logger.warning("    1. Check XHand power supply is on")
            logger.warning("    2. Check USB cable is firmly connected")
            logger.warning("    3. sudo chmod 666 /dev/ttyUSB* (or add user to dialout group)")
            logger.warning("    4. Verify no other process is holding the device (lsof /dev/ttyUSB*)")
            logger.warning("    5. Try power-cycling the XHand controller")

    def is_connected(self) -> bool:
        return self.control is not None and self.connected_flag and not self.error_state

    def is_error(self) -> bool:
        if self.control is None:
            return True
        if not self.connected_flag:
            return True
        if self.error_state:
            return True
        if self.last_error_code not in [None, 0]:
            return True
        if np.any(self.last_commboard_err != 0):
            return True
        if np.any(self.last_jointboard_err != 0):
            return True
        if np.any(self.last_tipboard_err != 0):
            return True
        return False

    def clear_error(self) -> bool:
        self.error_state = False
        self.last_error_code = None
        self.last_error_message = ""
        self.last_commboard_err[:] = 0
        self.last_jointboard_err[:] = 0
        self.last_tipboard_err[:] = 0
        return self.control is not None and self.connected_flag

    def stop(self) -> bool:
        if self.control is None or not self.connected_flag:
            return False
        command = self.make_command(
            self._array12(self.config.home_qpos),
            mode=0,
            tor_max=0,
            kp=0,
            ki=0,
            kd=0,
        )
        err = self.control.send_command(self.config.device_id, command)
        self.last_action_code = self.error_code(err)
        self.error_state = True
        if not self.error_ok(err):
            self._record_error(err)
            return False
        return True

    def reset(self, qpos: np.ndarray | None = None) -> bool:
        target = self._array12(self.config.home_qpos if qpos is None else qpos)
        return self.send_action(target)

    def get_state(
        self,
        full: bool = False,
        force_update: bool | None = None,
    ) -> dict[str, Any]:
        # ── E1: Return cached state when background reader is active ──
        if self.config.use_background_state_reader and self._state_thread is not None:
            with self._state_lock:
                cached = self._latest_state
            if cached is not None and not force_update:
                if not full:
                    return {
                        "qpos": cached.get("qpos", nan_array(12)).copy(),
                        "current": cached.get("current", nan_array(12)).copy(),
                        "timestamp": cached.get("timestamp", time.time()),
                    }
                state = {k: v.copy() if hasattr(v, "copy") else v for k, v in cached.items()}
                # F1: Add tactile contact detection to full state
                if full:
                    state["tactile_contact"] = self.detect_contact()
                return state

        if force_update is None:
            force_update = self.config.force_update_state

        err, hand_state = self.read_raw_state(force_update=force_update)

        if not self.error_ok(err) or hand_state is None:
            self._record_error(err)
            state = {
                "qpos": nan_array(12),
                "current": nan_array(12),
                "timestamp": time.time(),
            }
            if full:
                state.update(self._empty_state())
            return state

        state = self.parse_state(hand_state, full=full)
        self.last_error_code = 0
        self.last_error_message = ""
        return state

    def send_action(self, action: np.ndarray) -> bool:
        if self.control is None or self.hand_command is None:
            return False

        target_qpos = self._array12(action)
        qpos_after_limit = self._limit_joint_range(target_qpos)
        qpos_cmd = self._limit_joint_step(qpos_after_limit)

        # ── E2: EMA smoothing (ref: LeFranX xhand_vr_teleoperator.py:306-308) ──
        if self.config.ema_alpha > 0 and self._ema_qpos is not None:
            qpos_cmd = (
                (1.0 - self.config.ema_alpha) * self._ema_qpos
                + self.config.ema_alpha * qpos_cmd
            )
        self._ema_qpos = qpos_cmd.copy()

        self.write_command_positions(qpos_cmd)
        err = self.control.send_command(self.config.device_id, self.hand_command)
        self.last_action_code = self.error_code(err)

        if self.error_ok(err):
            self.last_qpos_cmd = qpos_cmd.copy()
            self.last_cmd_time = time.time()
            return True

        self._record_error(err)
        return False

    def reset_sensor(self, sensor_id: int | None = None) -> bool:
        if self.control is None:
            return False
        sensor_ids = SENSOR_IDS if sensor_id is None else [int(sensor_id)]
        ok = True
        for sid in sensor_ids:
            err = self.control.reset_sensor(self.config.device_id, sid)
            if not self.error_ok(err):
                self._record_error(err)
                ok = False
        return ok

    # ------------------------------------------------------------------
    # E1: Background state reader (ref: DexUMI hand_api_cls.py:228-275)
    # ------------------------------------------------------------------

    def _start_state_reader(self) -> None:
        """Start the daemon thread that reads hardware state at state_reader_hz."""
        if self._state_thread is not None and self._state_thread.is_alive():
            return
        self._state_stop.clear()
        self._state_thread = threading.Thread(
            target=self._read_state_loop,
            name="xhand_state_reader",
            daemon=True,
        )
        self._state_thread.start()
        logger.info(
            "XHand state reader started at %.0f Hz", self.config.state_reader_hz,
        )

    def _stop_state_reader(self) -> None:
        """Signal the state reader thread to stop and wait for exit."""
        if self._state_thread is None or not self._state_thread.is_alive():
            return
        self._state_stop.set()
        self._state_thread.join(timeout=2.0)
        if self._state_thread.is_alive():
            logger.warning("XHand state reader thread did not exit within timeout")
        else:
            logger.info("XHand state reader stopped")

    def _read_state_loop(self) -> None:
        """Daemon loop: read hardware state at state_reader_hz, cache latest.

        Uses RateLimiter (E3) for precise frame timing instead of plain sleep.
        """
        rate_limiter = RateLimiter(self.config.state_reader_hz)
        while not self._state_stop.is_set():
            rate_limiter.wait()

            # Read raw state from hardware (force_update=True to bypass SDK cache)
            err, hand_state = self.read_raw_state(force_update=True)
            if not self.error_ok(err) or hand_state is None:
                self._record_error(err)
                continue

            state = self.parse_state(hand_state, full=True)
            with self._state_lock:
                self._latest_state = state

    # ------------------------------------------------------------------
    # F1: Tactile contact detection (ref: DexUMI eval_xhand.py:40-57)
    # ------------------------------------------------------------------

    def detect_contact(self, threshold: float | None = None) -> np.ndarray:
        """Detect per-finger contact from tactile force.

        Uses the L2 norm of the combined force vector (fx, fy, fz) on each
        fingertip sensor, compared against tactile_contact_threshold.

        Args:
            threshold: Override for tactile_contact_threshold (Newtons).

        Returns:
            bool array of shape (5,), True where L2-norm > threshold.
        """
        thresh = threshold if threshold is not None else self.config.tactile_contact_threshold
        state = self.get_state(full=True)
        force_sum = np.asarray(state.get("tactile_force_sum", np.zeros((5, 3))))
        norm = np.linalg.norm(force_sum, axis=1)  # (5,) L2 per finger
        return norm > thresh

    def get_finger_contacts(self) -> dict[str, bool]:
        """Get per-finger contact status as a named dict.

        Returns:
            Dict mapping sensor name → contact boolean.
        """
        contacts = self.detect_contact()
        return {SENSOR_NAMES[i]: bool(contacts[i]) for i in range(5)}

    # ------------------------------------------------------------------
    # F2: Predefined grasp presets (ref: DexUMI constants.py:18-21)
    # ------------------------------------------------------------------

    def move_to_preset(self, name: str, duration_s: float = 1.0) -> bool:
        """Move hand to a predefined grasp preset.

        Args:
            name: Preset name — one of "home", "open", "fist", "pinch", "tripod".
            duration_s: Time to execute the motion.

        Returns:
            True if preset was found and command sent.
        """
        name = name.lower().strip()
        if name not in GRASP_PRESETS:
            logger.warning("Unknown preset '%s'. Available: %s", name, self.list_presets())
            return False

        target = GRASP_PRESETS[name]["qpos"].copy()
        return self.send_trajectory(target.reshape(1, 12), duration_s)

    @staticmethod
    def list_presets() -> list[str]:
        """Return available preset names."""
        return [
            f"{k}: {GRASP_PRESETS[k]['description']}"
            for k in GRASP_PRESETS
        ]

    # ------------------------------------------------------------------
    # F3: Trajectory interpolation with velocity constraints
    #     (ref: DexUMI motor_trajectory_interpolator.py:1-217)
    # ------------------------------------------------------------------

    def send_trajectory(self, waypoints: np.ndarray, duration_s: float) -> bool:
        """Execute a joint-space trajectory with linear interpolation.

        Waypoints are interpolated at the control rate (1/dt). If the
        computed speed exceeds max_qvel, duration is automatically extended.

        Args:
            waypoints: (N, 12) array of joint positions in radians.
            duration_s: Desired total duration; clamped to traj_min_duration_s.

        Returns:
            True if all waypoints were reached.
        """
        waypoints = np.asarray(waypoints, dtype=np.float64)
        if waypoints.ndim == 1:
            waypoints = waypoints.reshape(1, 12)

        duration_s = max(duration_s, self.config.traj_min_duration_s)
        n_waypoints = waypoints.shape[0]

        if n_waypoints == 1:
            # Single waypoint — send directly
            return self.send_action(waypoints[0])

        # Compute required velocity and extend duration if needed
        segment_dist = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        total_dist = np.sum(segment_dist)
        required_vel = total_dist / duration_s
        max_allowed_vel = np.min(self.config.max_qvel)
        if required_vel > max_allowed_vel:
            duration_s = total_dist / max_allowed_vel
            logger.info(
                "Trajectory speed limited: extended duration to %.2fs (max_qvel=%.2f)",
                duration_s, max_allowed_vel,
            )

        # Linear interpolation
        n_steps = max(2, int(duration_s / self.config.dt))
        t = np.linspace(0, 1, n_steps)
        segment_indices = np.linspace(0, n_waypoints - 1, n_steps)

        ok = True
        for i in range(n_steps):
            idx_float = segment_indices[i]
            idx_lo = int(np.floor(idx_float))
            idx_hi = min(idx_lo + 1, n_waypoints - 1)
            frac = idx_float - idx_lo
            interp_qpos = (1 - frac) * waypoints[idx_lo] + frac * waypoints[idx_hi]
            if not self.send_action(interp_qpos):
                ok = False
                break
            if i < n_steps - 1:
                time.sleep(self.config.dt)

        return ok

    def make_command(
        self,
        qpos: np.ndarray,
        mode: int | None = None,
        tor_max: int | None = None,
        kp: int | None = None,
        ki: int | None = None,
        kd: int | None = None,
    ):
        command = xh.HandCommand_t()
        mode = self.config.mode if mode is None else mode
        tor_max = self.config.tor_max if tor_max is None else tor_max
        kp = self.config.kp if kp is None else kp
        ki = self.config.ki if ki is None else ki
        kd = self.config.kd if kd is None else kd

        for i in range(12):
            cmd = command.finger_command[i]
            cmd.id = i
            # Per-joint gain overrides (D2) — when provided, individual joint
            # gains replace the scalar defaults (ref: DexUMI hand_api_cls.py:317-319).
            cmd.kp = int(self.config.kp_per_joint[i]) if self.config.kp_per_joint is not None else int(kp)
            cmd.ki = int(self.config.ki_per_joint[i]) if self.config.ki_per_joint is not None else int(ki)
            cmd.kd = int(self.config.kd_per_joint[i]) if self.config.kd_per_joint is not None else int(kd)
            cmd.position = float(qpos[i])
            cmd.tor_max = int(tor_max)
            cmd.mode = int(mode)
            cmd.res0 = 0
            cmd.res1 = 0
            cmd.res2 = 0
            cmd.res3 = 0
        return command

    def write_command_positions(self, qpos: np.ndarray):
        for i in range(12):
            self.hand_command.finger_command[i].position = float(qpos[i])

    def read_raw_state(self, force_update: bool = False):
        if self.control is None or not self.connected_flag:
            return None, None
        result = self.control.read_state(self.config.device_id, force_update)
        return self._unpack_result(result)

    def parse_state(self, hand_state, full: bool = False) -> dict[str, Any]:
        qpos = nan_array(12)
        current = nan_array(12)
        finger_ids = np.full(12, -1, dtype=np.int32)
        sensor_ids = np.full(12, -1, dtype=np.int32)
        raw_position = nan_array(12)
        temperature = nan_array(12)
        commboard_err = np.zeros(12, dtype=np.int32)
        jointboard_err = np.zeros(12, dtype=np.int32)
        tipboard_err = np.zeros(12, dtype=np.int32)

        finger_state = getattr(hand_state, "finger_state", [])
        for item in finger_state:
            idx = int(getattr(item, "id", -1))
            if idx < 0 or idx >= 12:
                continue

            finger_ids[idx] = idx
            sensor_ids[idx] = int(getattr(item, "sensor_id", -1))
            qpos[idx] = float(getattr(item, "position", np.nan))
            current[idx] = float(getattr(item, "torque", np.nan))
            raw_position[idx] = float(getattr(item, "raw_position", np.nan))
            temperature[idx] = float(getattr(item, "temperature", np.nan))
            commboard_err[idx] = int(getattr(item, "commboard_err", 0))
            # SDK v1.1.8: field is misspelled "jonitboard_err" (not "jointboard_err").
            # Use fallback chain to be compatible with both current and future SDK versions.
            jointboard_err[idx] = int(getattr(item, "jonitboard_err", getattr(item, "jointboard_err", 0)))
            tipboard_err[idx] = int(getattr(item, "tipboard_err", 0))

        self.last_commboard_err = commboard_err.copy()
        self.last_jointboard_err = jointboard_err.copy()
        self.last_tipboard_err = tipboard_err.copy()

        state = {
            "qpos": qpos,
            "current": current,
            "timestamp": time.time(),
        }

        if full:
            state.update(
                {
                    "finger_ids": finger_ids,
                    "sensor_ids": sensor_ids,
                    "raw_position": raw_position,
                    "temperature": temperature,
                    "commboard_err": commboard_err,
                    "jointboard_err": jointboard_err,
                    "tipboard_err": tipboard_err,
                    "tactile_force": self.parse_tactile(hand_state, scaled=True),
                    "tactile_force_raw": self.parse_tactile(hand_state, scaled=False),
                    "tactile_force_sum": self.parse_tactile_sum(hand_state, scaled=True),
                    "tactile_force_sum_raw": self.parse_tactile_sum(hand_state, scaled=False),
                    "tactile_temperature": self.parse_tactile_temperature(hand_state),
                    "connected_flag": self.connected_flag,
                    "error_state": self.error_state,
                    "last_action_code": self.last_action_code,
                    "last_error_code": self.last_error_code,
                    "last_error_message": self.last_error_message,
                    "last_joint_limit_clipped": self.last_joint_limit_clipped,
                    "last_delta_limited": self.last_delta_limited,
                    "last_hand_ids": self.last_hand_ids,
                    "comm_type": self.cached_comm_type,
                    "device_name": self.device_name,
                    "joint_names": JOINT_NAMES,
                    "sensor_names": SENSOR_NAMES,
                    "tactile_contact": self.detect_contact(),  # F1
                }
            )
        return state

    def _iter_sensors(self, hand_state):
        sensor_data = getattr(hand_state, "sensor_data", None)
        if sensor_data is None:
            sensor_data = getattr(hand_state, "sensor_data", [])
        return enumerate(list(sensor_data)[:5])

    def parse_tactile(self, hand_state, scaled: bool = True) -> np.ndarray:
        tactile = np.zeros((5, 120, 3), dtype=np.float64)
        for i, sensor in self._iter_sensors(hand_state):
            raw_force = getattr(sensor, "raw_force", [])
            for j, force in enumerate(list(raw_force)[:120]):
                tactile[i, j, 0] = float(getattr(force, "fx", 0.0))
                tactile[i, j, 1] = float(getattr(force, "fy", 0.0))
                tactile[i, j, 2] = float(getattr(force, "fz", 0.0))

        if scaled:
            tactile *= self.config.tactile_scale
        return tactile

    def parse_tactile_sum(self, hand_state, scaled: bool = True) -> np.ndarray:
        force_sum = np.zeros((5, 3), dtype=np.float64)
        for i, sensor in self._iter_sensors(hand_state):
            calc_force = getattr(sensor, "calc_force", None)
            if calc_force is None:
                continue
            force_sum[i, 0] = float(getattr(calc_force, "fx", 0.0))
            force_sum[i, 1] = float(getattr(calc_force, "fy", 0.0))
            force_sum[i, 2] = float(getattr(calc_force, "fz", 0.0))

        if scaled:
            force_sum *= self.config.tactile_scale
        return force_sum

    def parse_tactile_temperature(self, hand_state) -> np.ndarray:
        temperature = nan_array((5, 20))
        for i, sensor in self._iter_sensors(hand_state):
            temp = np.asarray(getattr(sensor, "temperature", []), dtype=np.float64).reshape(-1)
            if temp.size > 0:
                temperature[i, : min(20, temp.size)] = temp[:20]
        return temperature

    def _limit_joint_range(self, qpos: np.ndarray) -> np.ndarray:
        if not self.config.clip_joint_limit:
            self.last_joint_limit_clipped = False
            return qpos

        clipped = np.clip(qpos, self.config.qpos_min, self.config.qpos_max)
        self.last_joint_limit_clipped = not np.allclose(qpos, clipped)
        return clipped

    def _limit_joint_step(self, target_qpos: np.ndarray) -> np.ndarray:
        if not self.config.use_delta_limit:
            self.last_delta_limited = False
            return target_qpos

        now = time.time()

        if self.last_qpos_cmd is None:
            state = self.get_state(force_update=True)
            self.last_qpos_cmd = state["qpos"].copy()
            if not np.all(np.isfinite(self.last_qpos_cmd)):
                self.last_qpos_cmd = self._array12(self.config.home_qpos)

        if self.last_cmd_time is None:
            self.last_cmd_time = now

        dt = max(now - self.last_cmd_time, self.config.dt)
        max_step = self.config.max_qvel * dt
        raw_step = target_qpos - self.last_qpos_cmd
        step = np.clip(raw_step, -max_step, max_step)
        self.last_delta_limited = not np.allclose(raw_step, step)
        return self.last_qpos_cmd + step

    def is_valid_qpos_state(self, state: dict[str, Any]) -> bool:
        qpos = state.get("qpos", None)
        if qpos is None:
            return False

        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos.size != 12:
            return False

        if not np.all(np.isfinite(qpos)):
            return False

        finger_ids = state.get("finger_ids", None)
        if finger_ids is not None:
            finger_ids = np.asarray(finger_ids).reshape(-1)
            if finger_ids.size >= 12 and np.sum(finger_ids[:12] >= 0) < 12:
                return False

        return True

    def _array12(self, value) -> np.ndarray:
        if value is None:
            return nan_array(12)

        arr = np.asarray(value, dtype=np.float64).reshape(-1)

        if arr.size >= 12:
            return arr[:12]

        out = nan_array(12)
        out[: arr.size] = arr
        return out

    def _empty_state(self) -> dict[str, Any]:
        return {
            "finger_ids": np.full(12, -1, dtype=np.int32),
            "sensor_ids": np.full(12, -1, dtype=np.int32),
            "raw_position": nan_array(12),
            "temperature": nan_array(12),
            "commboard_err": np.zeros(12, dtype=np.int32),
            "jointboard_err": np.zeros(12, dtype=np.int32),
            "tipboard_err": np.zeros(12, dtype=np.int32),
            "tactile_force": np.zeros((5, 120, 3), dtype=np.float64),
            "tactile_force_raw": np.zeros((5, 120, 3), dtype=np.float64),
            "tactile_force_sum": np.zeros((5, 3), dtype=np.float64),
            "tactile_force_sum_raw": np.zeros((5, 3), dtype=np.float64),
            "tactile_temperature": nan_array((5, 20)),
            "connected_flag": self.connected_flag,
            "error_state": self.error_state,
            "last_action_code": self.last_action_code,
            "last_error_code": self.last_error_code,
            "last_error_message": self.last_error_message,
            "last_joint_limit_clipped": self.last_joint_limit_clipped,
            "last_delta_limited": self.last_delta_limited,
            "last_hand_ids": self.last_hand_ids,
            "comm_type": self.cached_comm_type,
            "device_name": self.device_name,
            "joint_names": JOINT_NAMES,
            "sensor_names": SENSOR_NAMES,
        }

    def _resolve_comm_type(self) -> str:
        name = str(self.config.comm_type).strip().lower()
        if name in ["rs485", "serial", "usb"]:
            return "RS485"
        if name in ["ethercat", "ethernet", "eth", "ecat"]:
            return "EtherCAT"
        return self.config.comm_type

    def _unpack_result(self, result):
        if isinstance(result, (tuple, list)):
            if len(result) >= 2:
                return result[0], result[1]

        if isinstance(result, dict):
            items = list(result.items())
            if len(items) > 0:
                return items[0][0], items[0][1]

        return None, None

    def error_code(self, err) -> int | None:
        if err is None:
            return None
        return int(getattr(err, "error_code", -1))

    def error_ok(self, err) -> bool:
        return err is not None and self.error_code(err) == 0

    def _record_error(self, err):
        if err is None:
            self.last_error_code = -1
            self.last_error_message = "empty error object"
            return

        code = self.error_code(err)
        msg = str(getattr(err, "error_message", ""))
        self.last_error_code = code
        self.last_error_message = msg

        # D1: Filter known non-critical sensor errors (ref: LeFranX xhand.py:230-241).
        # These hardware-level warnings do not affect position reading or motion
        # control — log them but don't trigger error_state.
        if self.config.filter_known_sensor_errors and code != 0 and msg:
            msg_lower = msg.lower()
            for pattern in _KNOWN_SENSOR_ERROR_PATTERNS:
                if pattern in msg_lower:
                    self.last_sensor_error_filtered = msg
                    return  # do NOT set error_state for known non-critical errors

        self.error_state = True


