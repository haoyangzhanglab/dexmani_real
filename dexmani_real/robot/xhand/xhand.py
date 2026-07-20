"""XHand 12-DOF robot hand hardware driver via xhand_controller SDK."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from xhand_controller import xhand_control as xhc

    _SDK_AVAILABLE = True
except ImportError:
    xhc = None  # type: ignore[assignment]
    _SDK_AVAILABLE = False

from dexmani_real.robot._connection_state import ConnectionStateMixin
from dexmani_real.robot.xhand.motor_trajectory_interpolator import MotorTrajectoryInterpolator
from dexmani_real.utils.array_utils import nan_array, safe_resize
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.serialization import FromDictMixin

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

# XHand SDK error codes (xhand_controller)
ERR_CRC = 1501070  # Communication data CRC error (RS485 transient)
ERR_BOOT_CMD = 1501036  # Error running CMD during boot, non-existent CMD (hand re-initializing)

# Recovery delays per error type (seconds).
# - CRC errors are transient and clear immediately — short delay suffices.
# - Boot CMD errors mean the hand controller is re-initializing after a
#   communication fault — needs longer for firmware to complete boot.
_RECOVERY_DELAY: dict[int, float] = {
    ERR_CRC: 0.05,  # 50 ms — transient, retry quickly
    ERR_BOOT_CMD: 0.5,  # 500 ms — hand needs time to finish boot sequence
}


@dataclass
class XHandConfig(FromDictMixin):
    comm_type: str = "EtherCAT"
    device_name: str | None = None
    baudrate: int = 3_000_000
    device_id: int = 0

    # Connection retry (RS485 may need several attempts after cold start)
    open_serial_retries: int = 3  # ref: LeFranX (no retry, but RS485 needs a few attempts)
    open_serial_retry_delay_s: float = 2.0

    dt: float = 1.0 / 30.0  # 30 Hz (ref: LeFranX, DexUMI)

    # Important:
    # True  -> force SDK to refresh state from hardware.
    # False -> may return SDK cached state. After open_serial(), cache may be all zeros.
    force_update_state: bool = True

    # Connect-time state initialization.
    # Even if force_update_state is manually set to False for runtime speed,
    # connect() should still force refresh several frames to avoid zero-cache initialization.
    init_state_read_attempts: int = 3
    init_state_read_interval: float = 0.02

    home_qpos: np.ndarray = field(
        default_factory=lambda: np.deg2rad(
            np.array(
                [
                    0.0,
                    45.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            )
        )
    )

    qpos_min: np.ndarray = field(
        default_factory=lambda: np.deg2rad(
            np.array(
                [
                    0.0,
                    -40.0,
                    10.0,  # thumb_j2:   prevent mechanical clogging (ref: LeFranX)
                    -10.0,
                    0.0,
                    5.0,  # index_j2:  prevent mechanical clogging (ref: LeFranX)
                    0.0,
                    5.0,  # middle_j2: prevent mechanical clogging (ref: LeFranX)
                    0.0,
                    5.0,  # ring_j2:   prevent mechanical clogging (ref: LeFranX)
                    0.0,
                    5.0,  # little_j2: prevent mechanical clogging (ref: LeFranX)
                ],
                dtype=np.float64,
            )
        )
    )

    qpos_max: np.ndarray = field(
        default_factory=lambda: np.array(
            # XHand joint limits from URDF (xhand_right.urdf), in radians.
            # Using exact URDF values avoids floating-point rounding from deg2rad.
            [
                1.832,  # J0  thumb_abd
                1.745,  # J1  thumb_j1 (-40° ~ 100°)
                1.745,  # J2  thumb_j2 (0° ~ 100°)
                0.174,  # J3  index_abd
                1.919,  # J4  index_j1
                1.919,  # J5  index_j2
                1.919,  # J6  middle_j1
                1.919,  # J7  middle_j2
                1.919,  # J8  ring_j1
                1.919,  # J9  ring_j2
                1.919,  # J10 little_j1
                1.919,  # J11 little_j2
            ],
            dtype=np.float64,
        ),
    )

    max_qvel: np.ndarray = field(
        default_factory=lambda: np.deg2rad(np.ones(12) * 180.0),
        metadata={
            "help": "Per-joint max velocity (rad/s). Used with MotorTrajectoryInterpolator.drive_to_waypoint as max_speed."
        },
    )

    kp: int = 80  # ref: LeFranX xhand_config.py:22
    ki: int = 0  # ref: LeFranX xhand_config.py:23
    kd: int = 0  # ref: LeFranX xhand_config.py:24
    # Per-joint gain overrides (ref: DexUMI hand_api_cls.py:317-319).
    # When set (shape (12,)), individual joint gains replace the scalar kp/ki/kd.
    # Distal joints (especially little finger joint 11) benefit from higher gains
    # to compensate for longer linkage and higher mechanical load.
    kp_per_joint: np.ndarray | None = None  # (12,) per-joint kp overrides
    ki_per_joint: np.ndarray | None = None  # (12,) per-joint ki overrides
    kd_per_joint: np.ndarray | None = None  # (12,) per-joint kd overrides
    tor_max: int = (
        300  # max 320mA; ref: LeFranX xhand_config.py:25 (400), DexUMI hand_api_cls.py:289 (400). Reduced to 300 for safety margin per XHand spec.
    )
    mode: int = 3

    clip_joint_limit: bool = True
    # Minimum per-joint deviation (rad) before CLIP flag is set in status output.
    # Prevents logspam from sub-degree retargeting imprecision (≈0.57° = 0.01 rad).
    # Actual np.clip is always enforced regardless of this threshold.
    clip_report_tolerance: float = 0.01

    # Known non-critical sensor error filtering (ref: LeFranX xhand.py:230-241).
    # When True, errors matching known patterns (sensor read failures, CRC errors,
    # unsupported-force-mode warnings) are logged but do NOT trigger error_state.
    filter_known_sensor_errors: bool = True

    # ── E2: EMA smoothing (ref: LeFranX xhand_vr_teleoperator.py:306-308) ──
    # 0.0 = disabled (default, backward compatible). 0.3 = LeFranX recommended.
    # Exponential Moving Average filters high-frequency jitter from position commands,
    # producing smoother finger motion.
    # NOTE: dex_retargetingʼs LPFilter(alpha=0.6) already applies EMA post-optimization.
    # Enable this only if additional hardware-level smoothing is needed.
    ema_alpha: float = 0.0

    # ── E3: Per-step delta jump limit ──
    # Hard-clips the per-frame change in each joint command (rad). Complements
    # dex_retargetingʼs LPFilter — EMA attenuates noise, this safety-gates outliers.
    # Scalar: same limit for all 12 joints.  (12,) array: per-joint limits.
    # 0.0 = disabled. Recommended: 0.3 rad (~17°/step, ~850°/s at 50 Hz).
    max_delta_rad: float | np.ndarray = 0.3

    # ── F1: Tactile contact detection ──
    # L2 norm threshold (Newtons) on per-finger combined force for contact detection.
    tactile_contact_threshold: float = 10.0  # raw sensor units (ref: DexUMI eval_xhand.py:72 binary_cutoff=[10,10,10])


class XHand(ConnectionStateMixin):
    def __init__(self, config: XHandConfig):
        super().__init__()
        self.config = config
        self.control: Any = None
        self.device_name: str | None = None
        self.hand_command: Any = None

        self.last_qpos_cmd: np.ndarray | None = None
        self.last_cmd_time: float | None = None
        self.last_error_code: int | None = None
        self.last_joint_limit_clipped = False

        # Error recovery: track consecutive send failures for circuit breaker
        self._consecutive_send_errors: int = 0

        self._stub_mode = False  # True when xhand_controller SDK unavailable (ref: LeFranX)
        self.last_hand_ids: list[int] = []
        self.cached_comm_type = self._resolve_comm_type()

        # ── E2: EMA filtering ──
        self._ema_qpos: np.ndarray | None = None

    # ── Connect lifecycle ──

    def connect(self) -> bool:
        """Connect to XHand hardware.

        Orchestrates device enumeration, retry-based port opening, and
        initial state initialization. Returns True on success.

        Falls back to stub mode when xhand_controller SDK is unavailable
        (ref: LeFranX xhand.py:158-163).
        """
        if self.connected_flag:
            return True  # re-entry guard: already connected

        if not _SDK_AVAILABLE:
            logger.warning("XHand SDK unavailable — entering stub mode (ref: LeFranX)")
            self._stub_mode = True
            self.connected_flag = True
            self.last_qpos_cmd = self._array12(self.config.home_qpos)
            return True

        comm_type = self.cached_comm_type

        if not self._retry_open_device(comm_type):
            return False

        try:
            self.last_hand_ids = list(self.control.list_hands_id())
        except (OSError, RuntimeError):
            self.last_hand_ids = []

        self.connected_flag = True
        self.error_state = False
        self._consecutive_send_errors = 0
        self.hand_command = self.make_command(self._array12(self.config.home_qpos))

        self._init_hand_state()

        self.last_cmd_time = time.time()

        return True

    def _retry_open_device(self, comm_type: str) -> bool:
        """Enumerate devices and open port with configurable retries.

        RS485 may need several attempts after cold start (C++ SDK retries
        internally, but may still fail intermittently).
        """
        self.control = xhc.XHandControl()
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
                    attempt,
                    retries,
                    self.last_error_message,
                    delay,
                )
                # Close and recreate control for clean retry
                try:
                    self.control.close_device()
                except (OSError, RuntimeError):
                    pass
                self.control = xhc.XHandControl()
                time.sleep(delay)

        # All retries exhausted
        self.error_state = True
        logger.error(
            "XHand connect failed after %s attempts: %s",
            retries,
            self.last_error_message,
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
        if self._stub_mode:
            self.connected_flag = False
            return
        if self.control is not None:
            self.control.close_device()
        self.connected_flag = False

    def _diagnose_connection_failure(self) -> None:
        logger.warning("XHand connection failed — check power, USB cable, and /dev/ttyUSB* permissions")

    def _stub_state(self, full: bool = False) -> dict[str, Any]:
        """Return zero state for stub mode (ref: LeFranX xhand.py:219-223)."""
        state: dict[str, Any] = {
            "qpos": np.zeros(12, dtype=np.float64),
            "current": np.zeros(12, dtype=np.float64),
            "timestamp": time.time(),
            "tactile_force": np.zeros((5, 120, 3), dtype=np.float64),
            "tactile_force_sum": np.zeros((5, 3), dtype=np.float64),
            "tactile_contact": np.zeros(5, dtype=bool),
        }
        if full:
            state.update(self._empty_state())
        return state

    def is_connected(self) -> bool:
        return self.control is not None and self.connected_flag and not self.error_state

    def is_error(self) -> bool:
        return self.control is None or not self.connected_flag or self.error_state

    def clear_error(self) -> bool:
        self.error_state = False
        self.last_error_code = None
        self.last_error_message = ""
        return self.control is not None and self.connected_flag

    @property
    def consecutive_send_errors(self) -> int:
        """Number of consecutive send_action() failures (circuit breaker counter)."""
        return self._consecutive_send_errors

    @staticmethod
    def get_recovery_delay(error_code: int | None = None) -> float:
        """Recommended recovery delay (seconds) for a send error code.

        Different errors need different recovery times:
        - ERR_CRC (1501070): transient RS485 corruption → 50ms
        - ERR_BOOT_CMD (1501036): hand controller re-initializing → 500ms
        - Unknown: conservative 100ms

        Callers should sleep this duration before retrying send_action().
        """
        if error_code is not None and error_code in _RECOVERY_DELAY:
            return _RECOVERY_DELAY[error_code]
        return 0.1  # conservative default for unknown errors

    def reset_connection(self) -> bool:
        """Full hardware reconnect after persistent send errors.

        Disconnects, waits 1s for hardware to stabilize, then reconnects.
        Resets consecutive error counter on success.

        Returns:
            True if reconnection succeeded.
        """
        logger.warning(
            "XHand: resetting connection after %d consecutive send errors (last code=%d)",
            self._consecutive_send_errors,
            self.last_error_code,
        )
        try:
            self.disconnect()
        except Exception:
            pass
        time.sleep(1.0)
        ok = self.connect()
        if ok:
            logger.info("XHand: reconnection succeeded")
        else:
            logger.error("XHand: reconnection failed")
        return ok

    def stop(self) -> bool:
        if self._stub_mode:
            self.error_state = True
            return True
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
        if self._stub_mode:
            return self._stub_state(full)

        if force_update is None:
            force_update = self.config.force_update_state

        err, hand_state = self.read_raw_state(force_update=force_update)

        if not self.error_ok(err) or hand_state is None:
            self._record_error(err)
            state = {
                "qpos": nan_array(12),
                "current": nan_array(12),
                "timestamp": time.time(),
                "tactile_force": np.zeros((5, 120, 3), dtype=np.float64),
                "tactile_force_sum": np.zeros((5, 3), dtype=np.float64),
                "tactile_contact": np.zeros(5, dtype=bool),
            }
            if full:
                state.update(self._empty_state())
            return state

        state = self.parse_state(hand_state, full=full)
        self.last_error_code = 0
        self.last_error_message = ""
        return state

    def send_action(self, action: np.ndarray) -> bool:
        if self._stub_mode:
            # Track the (joint-limited) request so last_qpos_cmd follows the
            # action stream instead of freezing at home_qpos — recorded actions
            # would otherwise be silently replaced by a constant.
            self.last_qpos_cmd = self._limit_joint_range(self._array12(action))
            return True

        if self.control is None or self.hand_command is None:
            return False

        target_qpos = self._array12(action)
        qpos_cmd = self._limit_joint_range(target_qpos)

        # ── E3: Delta jump limit ──
        # Hard safety gate: per-step change never exceeds max_delta_rad on any
        # joint, regardless of EMA state.  Complements dex_retargetingʼs LPFilter.
        # Supports per-joint limits: pass a (12,) ndarray for joint-specific caps.
        limit = np.broadcast_to(np.asarray(self.config.max_delta_rad), (12,))
        if np.any(limit > 0) and self.last_qpos_cmd is not None:
            delta = qpos_cmd - self.last_qpos_cmd
            delta = np.clip(delta, -limit, limit)
            qpos_cmd = self.last_qpos_cmd + delta

        # ── E2: EMA smoothing (ref: LeFranX xhand_vr_teleoperator.py:306-308) ──
        if self.config.ema_alpha > 0 and self._ema_qpos is not None:
            qpos_cmd = (1.0 - self.config.ema_alpha) * self._ema_qpos + self.config.ema_alpha * qpos_cmd
        self._ema_qpos = qpos_cmd.copy()

        self.write_command_positions(qpos_cmd)
        err = self.control.send_command(self.config.device_id, self.hand_command)
        self.last_action_code = self.error_code(err)

        if self.error_ok(err):
            self.last_qpos_cmd = qpos_cmd.copy()
            self.last_cmd_time = time.time()
            self._consecutive_send_errors = 0
            return True

        self._record_error(err)
        self._consecutive_send_errors += 1
        return False

    # F1: Tactile contact detection (ref: DexUMI eval_xhand.py:40-57)

    def detect_contact(self, threshold: float | None = None, force_sum: np.ndarray | None = None) -> np.ndarray:
        """Detect per-finger contact from tactile force.

        Uses the L2 norm of the combined force vector (fx, fy, fz) on each
        fingertip sensor, compared against tactile_contact_threshold.

        Args:
            threshold: Override for tactile_contact_threshold (Newtons).
            force_sum: Pre-parsed (5,3) force_sum array. When provided,
                       skips get_state() call (used inside parse_state to
                       avoid recursion).

        Returns:
            bool array of shape (5,), True where L2-norm > threshold.
        """
        thresh = threshold if threshold is not None else self.config.tactile_contact_threshold
        if force_sum is None:
            state = self.get_state(full=True)
            force_sum = np.asarray(state.get("tactile_force_sum", np.zeros((5, 3))))
        norm = np.linalg.norm(force_sum, axis=1)  # (5,) L2 per finger
        return norm > thresh

    # F3: Trajectory interpolation (ref: DexUMI MotorTrajectoryInterpolator)

    def send_trajectory(
        self,
        waypoints: np.ndarray,
        duration_s: float,
        max_speed: float | None = None,
        abort_event: Any | None = None,
    ) -> bool:
        """Execute a joint-space trajectory with scipy-based linear interpolation.

        Uses MotorTrajectoryInterpolator (ref: DexUMI) for smooth interp1d-based
        interpolation. When max_speed is provided, speed-limited waypoint driving
        is used — duration is automatically extended if the required speed exceeds
        the limit.

        Args:
            waypoints: (N, 12) array of joint positions in radians.
            duration_s: Desired total duration in seconds.
            max_speed: Optional scalar speed limit (L2 norm). When provided,
                       uses MotorTrajectoryInterpolator.drive_to_waypoint which
                       auto-extends duration to respect the speed limit.
                       Default: config.max_qvel.min().
            abort_event: Optional object with an ``is_set()`` method (e.g.
                       threading.Event / mp.Event). Checked between steps; when
                       set, the trajectory aborts at the next step boundary and
                       returns False — lets the hand control child preempt a
                       long trajectory on e-stop (plan §4.8). None (default)
                       preserves the original run-to-completion behavior.

        Returns:
            True if all waypoints were reached.
        """
        waypoints = np.asarray(waypoints, dtype=np.float64)
        if waypoints.ndim == 1:
            waypoints = waypoints.reshape(1, 12)

        n_waypoints = waypoints.shape[0]
        if n_waypoints == 1:
            return self.send_action(waypoints[0])

        duration_s = max(duration_s, 0.0)

        # Resolve max_speed
        speed_limit = max_speed if max_speed is not None else float(np.min(self.config.max_qvel))

        if np.isfinite(speed_limit) and speed_limit > 0:
            # Speed-limited: build interpolator from start, drive to final waypoint
            interp = MotorTrajectoryInterpolator(times=np.array([0.0]), values=waypoints[0:1]).drive_to_waypoint(
                value=waypoints[-1],
                time=duration_s,
                curr_time=0.0,
                max_speed=speed_limit,
            )
        else:
            # No speed limit: interpolate all waypoints directly
            times = np.linspace(0.0, duration_s, n_waypoints)
            interp = MotorTrajectoryInterpolator(times, waypoints)

        # Execute at control rate
        start_t = float(interp.times[0])
        end_t = float(interp.times[-1])
        exec_duration = end_t - start_t
        n_steps = max(2, int(exec_duration / self.config.dt))
        t_exec = np.linspace(start_t, end_t, n_steps)

        ok = True
        for i in range(n_steps):
            if abort_event is not None and abort_event.is_set():
                logger.info("XHand.send_trajectory: aborted at step %d/%d (abort event set).", i, n_steps)
                ok = False
                break
            interp_qpos = interp(t_exec[i])
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
        command = xhc.HandCommand_t()
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
            # SDK misspelling: "jonitboard_err" for "jointboard_err".
            jointboard_err[idx] = int(getattr(item, "jonitboard_err", getattr(item, "jointboard_err", 0)))
            tipboard_err[idx] = int(getattr(item, "tipboard_err", 0))

        tactile_force_sum = self.parse_tactile_sum(hand_state)
        state = {
            "qpos": qpos,
            "current": current,
            "timestamp": time.time(),
            # Tactile data in default mode (ref: DexUMI eval_xhand.py:40-57).
            # (5,120,3) raw force array + (5,3) combined force per finger.
            "tactile_force": self.parse_tactile(hand_state),
            "tactile_force_sum": tactile_force_sum,
            "tactile_contact": self.detect_contact(force_sum=tactile_force_sum),
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
                    "tactile_temperature": self.parse_tactile_temperature(hand_state),
                    "connected_flag": self.connected_flag,
                    "error_state": self.error_state,
                    "last_action_code": self.last_action_code,
                    "last_error_code": self.last_error_code,
                    "last_error_message": self.last_error_message,
                    "last_joint_limit_clipped": self.last_joint_limit_clipped,
                    "last_hand_ids": self.last_hand_ids,
                    "comm_type": self.cached_comm_type,
                    "device_name": self.device_name,
                    "joint_names": JOINT_NAMES,
                    "sensor_names": SENSOR_NAMES,
                }
            )
        return state

    def _iter_sensors(self, hand_state):
        sensor_data = getattr(hand_state, "sensor_data", None)
        if sensor_data is None:
            sensor_data = getattr(hand_state, "sensor_data", [])
        return enumerate(list(sensor_data)[:5])

    def parse_tactile(self, hand_state) -> np.ndarray:
        """Parse raw tactile force array (5 fingers × 120 points × 3 axes).

        Returns raw sensor values without scaling (ref: DexUMI, skill-teleop).
        """
        tactile = np.zeros((5, 120, 3), dtype=np.float64)
        for i, sensor in self._iter_sensors(hand_state):
            raw_force = getattr(sensor, "raw_force", [])
            for j, force in enumerate(list(raw_force)[:120]):
                tactile[i, j, 0] = float(getattr(force, "fx", 0.0))
                tactile[i, j, 1] = float(getattr(force, "fy", 0.0))
                tactile[i, j, 2] = float(getattr(force, "fz", 0.0))
        return tactile

    def parse_tactile_sum(self, hand_state) -> np.ndarray:
        """Parse combined force per finger (5 × 3 axes).

        Returns raw sensor values without scaling (ref: DexUMI, skill-teleop).
        """
        force_sum = np.zeros((5, 3), dtype=np.float64)
        for i, sensor in self._iter_sensors(hand_state):
            calc_force = getattr(sensor, "calc_force", None)
            if calc_force is None:
                continue
            force_sum[i, 0] = float(getattr(calc_force, "fx", 0.0))
            force_sum[i, 1] = float(getattr(calc_force, "fy", 0.0))
            force_sum[i, 2] = float(getattr(calc_force, "fz", 0.0))
        return force_sum

    def parse_tactile_temperature(self, hand_state) -> np.ndarray:
        temperature = nan_array((5, 20))
        for i, sensor in self._iter_sensors(hand_state):
            temp = np.asarray(getattr(sensor, "temperature", []), dtype=np.float64).reshape(-1)
            if temp.size > 0:
                temperature[i, : min(20, temp.size)] = temp[:20]
        return temperature

    def _limit_joint_range(self, qpos: np.ndarray) -> np.ndarray:
        # XHand variant: same np.clip logic as XArm7._limit_joint_range (xarm7.py:855)
        # but with different clipping targets (hand finger ranges vs arm joint ranges).
        # clip_report_tolerance suppresses false CLIP flags from sub-degree retargeting noise.
        if not self.config.clip_joint_limit:
            self.last_joint_limit_clipped = False
            return qpos

        clipped = np.clip(qpos, self.config.qpos_min, self.config.qpos_max)
        max_deviation = float(np.max(np.abs(qpos - clipped)))
        self.last_joint_limit_clipped = max_deviation > self.config.clip_report_tolerance
        return clipped

    def is_valid_qpos_state(self, state: dict[str, Any]) -> bool | np.bool_:
        qpos = state.get("qpos", None)
        if qpos is None:
            return False
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        return qpos.size == 12 and np.all(np.isfinite(qpos))

    def _array12(self, value) -> np.ndarray:
        return safe_resize(value, 12)

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
            "tactile_force_sum": np.zeros((5, 3), dtype=np.float64),
            "tactile_temperature": nan_array((5, 20)),
            "connected_flag": self.connected_flag,
            "error_state": self.error_state,
            "last_action_code": self.last_action_code,
            "last_error_code": self.last_error_code,
            "last_error_message": self.last_error_message,
            "last_joint_limit_clipped": self.last_joint_limit_clipped,
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
                    logger.debug("XHand filtered known sensor error (code=%d): %s", code, msg)
                    return  # do NOT set error_state for known non-critical errors

        self.error_state = True
