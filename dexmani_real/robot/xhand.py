"""XHand 12-DOF robot hand hardware driver via xhand_controller SDK."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dexmani_real.config.defaults import hand
from dexmani_real.ipc.schema import (
    HAND_CONTACT_SHAPE,
    HAND_DOF,
    HAND_FINGER_COUNT,
    HAND_JOINT_SHAPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
    TACTILE_POINTS_PER_FINGER,
)

try:
    from xhand_controller import xhand_control as xhc

    _SDK_AVAILABLE = True
except ImportError:
    xhc = None  # type: ignore[assignment]
    _SDK_AVAILABLE = False


from dexmani_real.utils.array_utils import nan_array, safe_resize
from dexmani_real.utils.log import ThrottledWarner, get_logger
from dexmani_real.utils.serialization import from_dict_helper

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


@dataclass
class XHandConfig:
    comm_type: str = "EtherCAT"
    device_name: str | None = None
    baudrate: int = 3_000_000
    device_id: int = 0
    ethercat_slave_position: int = -1
    simulation_backend: bool = False

    # Connection retry.  The C++ SDK retries some internal steps, but SDO
    # configuration writes (e.g. "write sdo failed 1,0,13") are surfaced as
    # hard failures from open_ethercat() without SDK-level retry.  A fresh
    # XHandControl + re-open_ethercat() resolves most transient SDO glitches,
    # so we retry at the Python level (matching the arm's 3-attempt pattern).
    open_serial_retries: int = 3
    # EtherCAT retries (separate from RS485).  Each failed open_ethercat() call
    # transitions the slave to OP even on error — the SDK's internal ec_init →
    # PRE_OP → SAFE_OP → OP sequence runs to completion regardless of PDO/SDO
    # outcome.  Repeated retries multiply slave-state corruption without recovery
    # value when the failure is persistent (CoE dictionary lock).  One retry
    # after the stale-OP wait covers transient SDO glitches; beyond that the
    # failure requires a power cycle.
    open_ethercat_retries: int = 2
    open_serial_retry_delay_s: float = 2.0

    # Important:
    # True  -> force SDK to refresh state from hardware.
    # False -> may return SDK cached state. After open_serial(), cache may be all zeros.
    force_update_state: bool = True

    # Connect-time state initialization.
    # Even if force_update_state is manually set to False for runtime speed,
    # connect() should still force refresh several frames to avoid zero-cache initialization.
    init_state_read_attempts: int = 3
    init_state_read_interval: float = 0.02

    home_qpos: np.ndarray = field(default_factory=lambda: np.deg2rad(np.asarray(hand.home_qpos_deg, dtype=np.float64)))

    qpos_min: np.ndarray = field(default_factory=lambda: np.asarray(hand.qpos_min_rad, dtype=np.float64))

    qpos_max: np.ndarray = field(default_factory=lambda: np.asarray(hand.qpos_max_rad, dtype=np.float64))

    # Feedback-only tolerance for measured encoder positions outside the strict
    # command limits. It classifies diagnostics; it never widens or clips the
    # feedback stream and is independent of clip_report_tolerance below.
    feedback_bound_tolerance_rad: float = field(default_factory=lambda: hand.feedback_bound_tolerance_rad)

    kp: int = 100
    ki: int = 0
    kd: int = 0
    # Per-joint overrides replace the scalar gains when configured.
    # When set (shape (12,)), individual joint gains replace the scalar kp/ki/kd.
    # Distal joints (especially little finger joint 11) benefit from higher gains
    # to compensate for longer linkage and higher mechanical load.
    kp_per_joint: np.ndarray | None = None  # (12,) per-joint kp overrides
    ki_per_joint: np.ndarray | None = None  # (12,) per-joint ki overrides
    kd_per_joint: np.ndarray | None = None  # (12,) per-joint kd overrides
    tor_max: int = 300  # max 300mA (per-joint default)
    # Per-joint tor_max overrides. When set (shape (12,)), individual joint
    # current limits replace the scalar tor_max.  Index abduction (J3) benefits
    # from higher current to compensate for sideways loading.
    tor_max_per_joint: np.ndarray | None = None  # (12,) per-joint tor_max overrides
    mode: int = 3

    clip_joint_limit: bool = True
    # Command-side reporting threshold: minimum requested-target deviation (rad)
    # before the CLIP status flag is set. np.clip is always enforced. This does
    # not classify encoder feedback; feedback_bound_tolerance_rad does that.
    clip_report_tolerance: float = 0.01

    # ── Per-step delta clamp (off by default) ──
    # Mode 3 PID has no firmware trajectory planning (unlike arm Mode 6).
    # A large instantaneous target jump is forwarded directly to the motor PID.
    # When enabled, this clamp limits the per-step joint delta as a safety
    # backstop. It is disabled because the shared action gate already bounds
    # target changes and the firmware current limit is the hardware backstop.
    #   - scalar: applied to all 12 joints
    #   - (12,) array: per-joint limits
    #   - None: disabled (default)
    max_delta_rad: float | np.ndarray | None = field(default_factory=lambda: hand.max_delta_rad)

    # ── F1: Tactile contact detection ──
    # L2 threshold in the SDK's scaled tactile units.  The vendor/firmware
    # provenance for a physical SI conversion has not been validated on this
    # installation, so recordings deliberately label the unit unknown.
    tactile_contact_threshold: float = 1.0

    def __post_init__(self) -> None:
        if self.ethercat_slave_position < -1:
            raise ValueError("ethercat_slave_position must be -1 (unknown) or non-negative")
        if not np.isfinite(self.feedback_bound_tolerance_rad) or self.feedback_bound_tolerance_rad < 0:
            raise ValueError("feedback_bound_tolerance_rad must be finite and >= 0")
        if not np.isfinite(self.tactile_contact_threshold) or self.tactile_contact_threshold < 0:
            raise ValueError("tactile_contact_threshold must be finite and non-negative")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "XHandConfig":
        """Reconstruct from a serialized dict."""
        return cls(**from_dict_helper(cls, d))  # type: ignore[arg-type]


class XHand:
    def __init__(self, config: XHandConfig):
        self.connected_flag: bool = False
        self.error_state: bool = False
        self.last_error_message: str = ""
        self.last_action_code: int | None = None
        self.config = config
        self.control: Any = None
        self.device_name: str | None = None
        self.hand_command: Any = None

        self.last_qpos_cmd: np.ndarray | None = None
        self.last_cmd_time: float | None = None
        self.last_error_code: int | None = None
        self.last_joint_limit_clipped = False

        # Measured-feedback diagnostics are deliberately separate from command
        # clipping. Raw finite qpos remains unchanged when published/recorded.
        self._feedback_bounds_warn = ThrottledWarner(interval_s=5.0, logger=logger)
        self._feedback_bound_checks = 0
        self._feedback_bound_outside = 0
        self._feedback_bound_over_tolerance = 0
        self._feedback_bound_max_violation_rad = 0.0
        self._feedback_bound_joint_outside_counts = np.zeros(HAND_JOINT_SHAPE, dtype=np.int64)
        self._feedback_bound_joint_over_tolerance_counts = np.zeros(HAND_JOINT_SHAPE, dtype=np.int64)

        # Error recovery: track consecutive send failures for circuit breaker
        self._consecutive_send_errors: int = 0

        # reset_sensor() can leave an offset; subtract a fresh no-contact mean.
        self._tactile_bias_ft: np.ndarray | None = None  # (5, 3)   — calc_force bias
        self._tactile_bias_raw: np.ndarray | None = None  # (5, 120, 3) — raw_force bias
        self.tactile_calibrated: bool = False

        self._stub_mode = False
        self.last_hand_ids: list[int] = []
        self.cached_comm_type = self._resolve_comm_type()
        self.device_identity: dict[str, str] = {
            "backend": "hardware" if _SDK_AVAILABLE else "unavailable",
            "hand_type": "unavailable",
            "sdk_version": "unavailable",
            "serial_number": "unavailable",
        }

    # ── Connect lifecycle ──

    def connect(self) -> bool:
        """Connect to XHand hardware.

        Orchestrates device enumeration, retry-based port opening, and
        initial state initialization. Returns True on success.

        The hardware backend fails closed when the vendor SDK is unavailable.
        A following simulation backend must be requested explicitly.
        """
        if self.connected_flag:
            return True  # re-entry guard: already connected

        if not _SDK_AVAILABLE:
            if not self.config.simulation_backend:
                self.error_state = True
                self.last_error_message = "XHand SDK unavailable and simulation_backend=False"
                logger.error(self.last_error_message)
                return False
            logger.warning("XHand SDK unavailable — explicit following simulation backend enabled")
            self._stub_mode = True
            self.connected_flag = True
            self.last_qpos_cmd = self._array12(self.config.home_qpos)
            self.device_identity = {
                "backend": "simulation",
                "hand_type": "right",
                "sdk_version": "simulation",
                "serial_number": "simulation",
            }
            return True

        comm_type = self.cached_comm_type

        if not self._retry_open_device(comm_type):
            return False

        self.connected_flag = True
        try:
            self.last_hand_ids = list(self.control.list_hands_id())
            if self.config.device_id not in self.last_hand_ids:
                raise RuntimeError(
                    f"configured device_id={self.config.device_id} not found in enumerated hands {self.last_hand_ids}"
                )
            self.error_state = False
            self._consecutive_send_errors = 0
            self._verify_device()
            self.hand_command = self.make_command(self._array12(self.config.home_qpos))
            if not self._init_hand_state():
                raise RuntimeError("initial XHand state is unavailable or invalid")
        except Exception:
            logger.error("XHand post-open initialization failed", exc_info=True)
            self.error_state = True
            try:
                self.disconnect()
            except Exception:
                logger.error("XHand post-open cleanup failed", exc_info=True)
            return False

        self.last_cmd_time = time.time()

        return True

    def _retry_open_device(self, comm_type: str) -> bool:
        """Enumerate devices and open port with configurable retries.

        RS485 may need several attempts after cold start (C++ SDK retries
        internally, but may still fail intermittently).

        Enumeration and open use SEPARATE XHandControl instances.  The
        enumeration pass opens raw sockets on every interface to scan for
        slaves; those sockets must be fully released before open_ethercat
        creates its own socket on the target interface.  Reusing the same
        control instance can leave duplicate raw sockets on the same
        interface, causing SDO responses to route to the stale socket while
        the open path waits on the new one — producing "write sdo failed"
        errors observed in forked child processes.

        This two-phase pattern isolates the vendor SDK's raw-socket discovery
        lifetime from the long-lived control connection.  Keep it inside the
        spawned hand worker: moving SDK ownership to Main would violate the
        process boundary and make clean shutdown harder to prove.
        """
        # Device discovery uses a temporary control that is closed afterward.
        if self.config.device_name is None:
            temp_control = xhc.XHandControl()
            try:
                devices = temp_control.enumerate_devices(comm_type)
            finally:
                try:
                    temp_control.close_device()
                except (OSError, RuntimeError):
                    logger.warning("temporary XHand discovery control did not close cleanly", exc_info=True)
            if devices is None or len(devices) == 0:
                self.error_state = True
                self.last_error_code = -2
                self.last_error_message = f"no XHand device found for {comm_type}"
                self._diagnose_connection_failure()
                return False
            self.device_name = devices[0]
        else:
            self.device_name = self.config.device_name

        # Open on a fresh control; never reuse the discovery control.
        # RS485 may need several attempts after cold start; EtherCAT uses a
        # lower retry cap because each failed open_ethercat() transitions the
        # slave to OP regardless of PDO/SDO outcome, and repeated transitions
        # compound state corruption without recovery value.
        retries = max(
            1, int(self.config.open_ethercat_retries if comm_type == "EtherCAT" else self.config.open_serial_retries)
        )
        delay = max(0.0, float(self.config.open_serial_retry_delay_s))

        for attempt in range(1, retries + 1):
            self.control = xhc.XHandControl()
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
                if attempt > 1:
                    logger.warning(
                        "XHand connect succeeded on attempt %d/%d "
                        "(retries indicate SDO/communication glitch) — "
                        "adding 1.0s post-recovery stabilisation delay.",
                        attempt,
                        retries,
                    )
                    time.sleep(1.0)
                return True

            self._record_error(err)
            # Close the failed control before retry
            try:
                self.control.close_device()
            except (OSError, RuntimeError):
                logger.warning("failed XHand control did not close cleanly", exc_info=True)
            if attempt < retries:
                # On EtherCAT, the first failure may be a stale-slave-state
                # condition (previous session exited without clean disconnect,
                # leaving the slave in OP).  A longer wait gives the slave's
                # SM-watchdog time to expire so the retry sees a clean INIT
                # slave.  On later attempts the standard retry delay is
                # sufficient — if the slave was in OP, the first retry's wait
                # already handled it; if not, it's a different error class.
                _retry_delay = delay
                if comm_type == "EtherCAT" and attempt == 1:
                    _retry_delay = max(delay, self._STALE_OP_RECOVERY_WAIT_S)
                    logger.warning(
                        "XHand connect attempt %d/%d failed: %s — "
                        "waiting %.1fs for potential stale-slave recovery before retry...",
                        attempt,
                        retries,
                        self.last_error_message,
                        _retry_delay,
                    )
                else:
                    logger.warning(
                        "XHand connect attempt %s/%s failed: %s, retrying in %.1fs...",
                        attempt,
                        retries,
                        self.last_error_message,
                        _retry_delay,
                    )
                time.sleep(_retry_delay)

        # All retries exhausted
        self.error_state = True
        logger.error(
            "XHand connect failed after %s attempts: %s",
            retries,
            self.last_error_message,
        )
        self._diagnose_connection_failure()
        return False

    def _init_hand_state(self) -> bool:
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
            return False

        # ── Tactile sensor initialisation ──
        # Check the raw startup load before reset_sensor() can redefine the
        # loaded state as zero. Missing/malformed tactile feedback also fails
        # calibration closed while leaving non-tactile hand operation usable.
        startup_force = valid_state.get("tactile_force_sum")
        if self._tactile_load_present(startup_force):
            self._tactile_bias_ft = np.zeros(HAND_TACTILE_SUM_SHAPE, dtype=np.float64)
            self._tactile_bias_raw = np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64)
            self.tactile_calibrated = False
            logger.error("Tactile calibration refused: contact/load or invalid tactile data detected at startup")
            return True
        # Reset all five fingertip sensors after power-on.  The SDK documents
        # that some sensors may need an explicit reset before they begin
        # reporting data (sensor IDs 17–21: thumb, index, middle, ring, little).
        # Failures are logged but do not block operation — a dead sensor is
        # less harmful than a refused connection.
        self._reset_tactile_sensors()
        return True

    def _tactile_load_present(self, force_sum: Any) -> bool:
        """Fail-closed startup-load check in the SDK's unverified units."""
        value = np.asarray(force_sum, dtype=np.float64)
        if value.shape != HAND_TACTILE_SUM_SHAPE or not np.all(np.isfinite(value)):
            return True
        return bool(np.any(np.linalg.norm(value, axis=1) > self.config.tactile_contact_threshold))

    def _reset_tactile_sensors(self) -> None:
        """Reset sensors 17--21 and estimate a no-contact software bias.

        Sensors with a residual offset are retried selectively. Individual
        reset failures are non-fatal, but calibration stays false when the
        no-contact quality gate fails.

        The C++ SDK (libxhand_control.so) prints "Unknow Cmd!" to stdout for
        each ``reset_sensor()`` call when the hand firmware does not recognise
        the command.  The error codes are handled in Python regardless.
        """
        if self._stub_mode:
            self.tactile_calibrated = True
            return
        device_id = self.config.device_id

        MAX_OUTER_ITERS = 5
        VERIFY_THRESHOLD = 2.0
        FINGER_LABELS = ["thumb", "index", "middle", "ring", "pinky"]

        bad_indices: list[int] = list(range(HAND_FINGER_COUNT))
        _last_mags: np.ndarray | None = None  # cached for else-clause diagnostics

        for outer in range(MAX_OUTER_ITERS):
            # ── Reset only sensors that still have residual offset ──
            for idx in bad_indices:
                sensor_id = 17 + idx
                for attempt in range(3):
                    try:
                        err = self.control.reset_sensor(device_id, sensor_id)
                        if self.error_ok(err):
                            break
                    except Exception:
                        logger.warning(
                            "Tactile sensor %d reset attempt raised",
                            sensor_id,
                            exc_info=True,
                        )
                    time.sleep(0.2)
                else:
                    logger.warning(
                        "Tactile sensor %d (%s) reset failed after 3 attempts",
                        sensor_id,
                        FINGER_LABELS[idx],
                    )

            # ── Verify: read fresh state (before bias) and check force magnitudes ──
            err, hand_state = self._unpack_result(self.control.read_state(device_id, True))  # force_update=True
            if not self.error_ok(err):
                logger.warning(
                    "Tactile verify read failed (iter %d/%d, code=%s) — retrying all sensors",
                    outer + 1,
                    MAX_OUTER_ITERS,
                    self.error_code(err),
                )
                continue  # retry all sensors next iteration

            force_sum = self.parse_tactile_sum(hand_state)  # raw — bias is None during init
            mags = np.linalg.norm(force_sum, axis=1)  # (5,) — |F| per finger
            _last_mags = mags

            new_bad = [i for i in bad_indices if mags[i] > VERIFY_THRESHOLD]
            if not new_bad:
                break  # all previously-bad sensors now within threshold

            if outer < MAX_OUTER_ITERS - 1:
                _labels = [FINGER_LABELS[i] for i in new_bad]
                _max_mag = max(float(mags[i]) for i in new_bad)
                logger.warning(
                    "Tactile verify-retry %d/%d: %s still > %.1f scaled units (max %.2f)",
                    outer + 1,
                    MAX_OUTER_ITERS,
                    ", ".join(_labels),
                    VERIFY_THRESHOLD,
                    _max_mag,
                )
            bad_indices = new_bad
        else:
            _labels = [FINGER_LABELS[i] for i in bad_indices]
            _max_mag = max(float(_last_mags[i]) for i in bad_indices) if _last_mags is not None else float("nan")
            logger.warning(
                "Tactile bias non-zero after %d retries: %s (max %.2f scaled units) — "
                "software bias will compensate for residual offset",
                MAX_OUTER_ITERS,
                ", ".join(_labels),
                _max_mag,
            )

        # reset_sensor() can leave an offset, so estimate a bias from fresh reads.
        try:
            self._compute_tactile_bias(device_id)
        except Exception:
            logger.warning(
                "Tactile bias computation failed — bias will be zero (no correction).",
                exc_info=True,
            )
            self._tactile_bias_ft = np.zeros(HAND_TACTILE_SUM_SHAPE, dtype=np.float64)
            self._tactile_bias_raw = np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64)
            self.tactile_calibrated = False

    def _compute_tactile_bias(self, device_id: int, n_samples: int = 5) -> None:
        """Average *n_samples* fresh tactile readings as the no-contact bias.

        Stores ``self._tactile_bias_ft`` (5,3) and ``self._tactile_bias_raw``
        (5,120,3).  Called once after ``reset_sensor()`` at connect time.

        The hand must NOT be in contact with any object during bias capture.
        If ``read_state`` returns an error on any sample, bias is set to zeros
        (no correction) and a warning is logged.
        """
        if self._stub_mode or self.control is None:
            self._tactile_bias_ft = np.zeros(HAND_TACTILE_SUM_SHAPE, dtype=np.float64)
            self._tactile_bias_raw = np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64)
            self.tactile_calibrated = bool(self._stub_mode)
            return

        ft_samples: list[np.ndarray] = []
        raw_samples: list[np.ndarray] = []

        for _ in range(n_samples):
            err, hand_state = self._unpack_result(self.control.read_state(device_id, True))  # force_update=True
            if not self.error_ok(err):
                logger.warning(
                    "Tactile bias sample read failed (code=%s) — "
                    "bias will be zero (no correction).  "
                    "Ensure the hand is not in contact with any object during init.",
                    self.error_code(err),
                )
                self._tactile_bias_ft = np.zeros(HAND_TACTILE_SUM_SHAPE, dtype=np.float64)
                self._tactile_bias_raw = np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64)
                return

            ft_samples.append(self.parse_tactile_sum(hand_state))
            raw_samples.append(self.parse_tactile(hand_state))

        # A loaded/contacting hand is not a valid zero reference. Refuse to
        # absorb the external load into the software bias.
        pre_bias_magnitude = np.linalg.norm(np.stack(ft_samples, axis=0), axis=2)
        if float(np.max(pre_bias_magnitude)) > self.config.tactile_contact_threshold:
            self._tactile_bias_ft = np.zeros(HAND_TACTILE_SUM_SHAPE, dtype=np.float64)
            self._tactile_bias_raw = np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64)
            self.tactile_calibrated = False
            logger.error("Tactile bias calibration refused: contact/load detected during startup")
            return

        self._tactile_bias_ft = np.nanmean(np.stack(ft_samples, axis=0), axis=0)
        self._tactile_bias_raw = np.nanmean(np.stack(raw_samples, axis=0), axis=0)
        self.tactile_calibrated = True

        # Log which fingers have significant bias
        ft_mag = np.linalg.norm(self._tactile_bias_ft, axis=1)
        for i, mag in enumerate(ft_mag):
            if mag > 0.5:  # scaled units; lower than the reset verification threshold
                logger.info(
                    "Tactile sensor %d bias: |F|=%.2f scaled units (%.2f, %.2f, %.2f)",
                    i + 1,
                    mag,
                    self._tactile_bias_ft[i, 0],
                    self._tactile_bias_ft[i, 1],
                    self._tactile_bias_ft[i, 2],
                )

        max_bias = float(np.max(np.abs(self._tactile_bias_raw)))
        logger.info(
            "Tactile bias computed from %d samples — max bias %.2f scaled units",
            n_samples,
            max_bias,
        )

    def _verify_device(self) -> None:
        """Log hardware identity for diagnostics (non-fatal — never blocks connect).

        Calls SDK introspection methods: SDK version, hand type (left/right),
        and serial number.  Failures are logged at WARNING level but do not
        set error_state or prevent operation.
        """
        device_id = self.config.device_id

        # SDK version
        try:
            ver = self.control.get_sdk_version()
            self.device_identity["sdk_version"] = str(ver)
            logger.info("XHand SDK version: %s", ver)
        except Exception:
            logger.warning("XHand SDK version: unavailable", exc_info=True)

        # Hand type (left / right)
        try:
            err, hand_type = self.control.get_hand_type(device_id)
            if self.error_ok(err):
                self.device_identity["hand_type"] = str(hand_type)
                logger.info("XHand hand type: %s (device_id=%d)", hand_type, device_id)
            else:
                code = self.error_code(err)
                msg = str(getattr(err, "error_message", ""))
                logger.warning("XHand get_hand_type failed: code=%s msg=%s", code, msg)
        except Exception:
            logger.warning("XHand get_hand_type: unavailable", exc_info=True)

        # Serial number
        try:
            err, serial = self.control.get_serial_number(device_id)
            if self.error_ok(err):
                self.device_identity["serial_number"] = str(serial)
                logger.info("XHand serial: %s", serial)
            else:
                code = self.error_code(err)
                msg = str(getattr(err, "error_message", ""))
                logger.warning("XHand get_serial_number failed: code=%s msg=%s", code, msg)
        except Exception:
            logger.warning("XHand get_serial_number: unavailable", exc_info=True)

    # ── EtherCAT slave state management ──

    # AL state constants (EtherCAT standard / SOEM convention).
    _EC_STATE_INIT = 1
    _EC_STATE_PRE_OP = 2
    _EC_STATE_SAFE_OP = 4
    _EC_STATE_OP = 8

    # Post-disconnect watchdog wait (seconds).  After close_device() the slave
    # firmware has no more master frames; its internal SM-watchdog must expire
    # before it auto-transitions to SAFE_OP+Error and then INIT.  Typical
    # EtherCAT watchdogs are 100–1000 ms; 2.0 s gives a comfortable margin.
    _POST_DISCONNECT_WATCHDOG_WAIT_S = 2.0

    # Extra wait on the first EtherCAT connect retry when the slave may still be
    # in OP from a previous session whose disconnect path was skipped (kill -9,
    # power blip, SDK crash, etc.).  Combine with the standard retry delay.
    _STALE_OP_RECOVERY_WAIT_S = 3.0

    def _request_slave_init(self) -> bool:
        """Request the EtherCAT slave to transition to INIT via the SDK.

        Uses ``set_firmware_state(device_id, slave_pos, state, timeout_us)``
        when available.  On success the slave releases its Operational state so
        the next ``open_ethercat()`` can reconfigure PDOs from scratch.

        Returns True if the SDK acknowledged the transition request.
        """
        if self._stub_mode or self.control is None:
            return False
        if self.cached_comm_type != "EtherCAT":
            return False  # RS485 has no slave state machine
        if self.config.ethercat_slave_position < 0:
            logger.warning(
                "XHand: EtherCAT slave position is unknown; skipping explicit INIT request and using close/watchdog"
            )
            return False
        try:
            err, _prev_state = self.control.set_firmware_state(
                self.config.device_id,
                self.config.ethercat_slave_position,
                self._EC_STATE_INIT,
                500_000,  # timeout (µs) for the AL state transition
            )
            if self.error_ok(err):
                logger.info("XHand: EtherCAT slave transitioned to INIT before disconnect.")
                time.sleep(0.2)  # let the AL state transition propagate
                return True
            code = self.error_code(err)
            msg = str(getattr(err, "error_message", ""))
            logger.debug(
                "XHand: set_firmware_state(INIT) returned code=%s msg=%r — "
                "falling back to post-disconnect watchdog wait.",
                code,
                msg,
            )
        except Exception:
            logger.debug(
                "XHand: set_firmware_state() unavailable — " "falling back to post-disconnect watchdog wait.",
                exc_info=True,
            )
        return False

    def disconnect(self):
        """Release the hardware connection.

        Two-stage cleanup so the EtherCAT slave is left in a state that
        permits reconnection without a power cycle:

        1.  Request INIT via the SDK's ``set_firmware_state`` (best-effort).
        2.  Call ``close_device()``.
        3.  Wait for the slave's internal SM-watchdog to expire so it
            auto-transitions out of OP before the next session starts.
        """
        if self._stub_mode:
            self.connected_flag = False
            return
        # Only perform EtherCAT cleanup when we actually connected successfully.
        # After a failed connect(), self.control may still reference a
        # close_device()'d handle from _retry_open_device — calling
        # _request_slave_init() or close_device() on it again is at best a no-op
        # and at worst triggers undefined behaviour in the C++ SDK.
        if self.control is not None and self.connected_flag:
            self._request_slave_init()
            self.control.close_device()
            time.sleep(self._POST_DISCONNECT_WATCHDOG_WAIT_S)
        self.connected_flag = False

    def _diagnose_connection_failure(self) -> None:
        if self.cached_comm_type == "EtherCAT":
            if self.last_error_code == -2:
                logger.warning(
                    "No XHand EtherCAT slave was enumerated — check 24V power, "
                    "EtherCAT cable, and eno1 link/carrier. If the SDK printed "
                    "'ec_init ... succeeded', raw-socket permission is already "
                    "working; CAP_NET_RAW is relevant only when ec_init/socket "
                    "creation reports a permission failure."
                )
            else:
                logger.warning(
                    "XHand EtherCAT open failed — check power, cable, eno1 link, "
                    "and the SDK error above. CAP_NET_RAW is required only when "
                    "ec_init/socket creation reports a permission failure."
                )
            if self.last_error_code != -2:
                # SDO write failures during open_ethercat (e.g. "write sdo
                # failed 1,0,13") indicate that a previous unclean exit may
                # have left the slave state inconsistent. No-slave discovery
                # (-2) occurs before SDO traffic, so this advice would be noise.
                logger.error(
                    "If SDO errors appeared above: the previous session may not "
                    "have called disconnect() cleanly (e.g. kill -9 or SDK crash). "
                    "Power-cycle the XHand (disconnect + reconnect 24V power, "
                    "wait ≥5 s), then retry."
                )
        else:
            logger.warning("XHand connection failed — check power, USB cable, and /dev/ttyUSB* permissions")

    def _stub_state(self, full: bool = False) -> dict[str, Any]:
        """Return following-simulation state with zero current/tactile data."""
        state: dict[str, Any] = {
            "qpos": np.array(
                self.last_qpos_cmd if self.last_qpos_cmd is not None else self.config.home_qpos,
                dtype=np.float64,
                copy=True,
            ),
            "current": np.zeros(HAND_JOINT_SHAPE, dtype=np.float64),
            "timestamp": time.time(),
            "tactile_force": np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64),
            "tactile_force_sum": np.zeros(HAND_TACTILE_SUM_SHAPE, dtype=np.float64),
            "tactile_contact": np.zeros(HAND_CONTACT_SHAPE, dtype=bool),
            "tipboard_err": np.zeros(HAND_JOINT_SHAPE, dtype=np.int32),
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
            state: dict[str, Any] = {
                "qpos": nan_array(HAND_JOINT_SHAPE),
                "current": nan_array(HAND_JOINT_SHAPE),
                "timestamp": time.time(),
            }
            if full:
                state.update(self._empty_state())
            return state

        state = self.parse_state(hand_state, full=full)
        self._record_feedback_bound_check(state["qpos"])
        self.last_error_code = 0
        self.last_error_message = ""
        # Bridge board-status (Layer 2) into safety gate: per-joint hardware
        # board error registers gate commands on hardware-level faults.
        # Unlike send/read errors (tracked via _record_error +
        # _consecutive_send_errors watchdog), board errors are transient —
        # auto-clear when hardware status returns to normal (no manual
        # clear_error() needed after an RS485 glitch).
        self.error_state = bool(
            np.any(np.asarray(state["commboard_err"], dtype=np.int32))
            or np.any(np.asarray(state["jointboard_err"], dtype=np.int32))
            or np.any(np.asarray(state["tipboard_err"], dtype=np.int32))
        )
        return state

    def send_action(self, action: np.ndarray) -> bool:
        target_qpos = np.asarray(action, dtype=np.float64)
        if target_qpos.shape != HAND_JOINT_SHAPE or not np.all(np.isfinite(target_qpos)):
            self.last_error_message = "XHand.send_action rejected invalid shape or NaN/Inf"
            logger.warning(self.last_error_message)
            return False
        if self._stub_mode:
            # Track the (joint-limited) request so last_qpos_cmd follows the
            # action stream instead of freezing at home_qpos — recorded actions
            # would otherwise be silently replaced by a constant.
            self.last_qpos_cmd = self._limit_joint_range(target_qpos)
            return True

        if self.control is None or self.hand_command is None:
            if self.hand_command is None:
                self.error_state = True
                self.last_error_message = "hand_command is None (not initialized)"
            return False

        qpos_cmd = self._limit_joint_range(target_qpos)

        # ── Per-step delta clamp ──
        # Mode 3 PID has no firmware trajectory planning.  Large instantaneous
        # jumps (e.g. home_qpos → first VR pose) are forwarded directly to the
        # motor PID at full torque.  This clamp is a safety backstop — normal
        # teleop should rarely trigger it.
        if self.config.max_delta_rad is not None and self.last_qpos_cmd is not None:
            limit = np.broadcast_to(self.config.max_delta_rad, HAND_JOINT_SHAPE)
            delta = qpos_cmd - self.last_qpos_cmd
            clipped_delta = np.clip(delta, -limit, limit)
            qpos_cmd = self.last_qpos_cmd + clipped_delta

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

    def detect_contact(self, threshold: float | None = None, force_sum: np.ndarray | None = None) -> np.ndarray:
        """Detect per-finger contact from tactile force.

        Uses the L2 norm of the combined force vector (fx, fy, fz) on each
        fingertip sensor, compared against tactile_contact_threshold.

        Args:
            threshold: Override in the same unknown scaled units as force_sum.
            force_sum: Pre-parsed (5,3) force_sum array. When provided,
                       skips get_state() call (used inside parse_state to
                       avoid recursion).

        Returns:
            bool array of shape (5,), True where L2-norm > threshold.
        """
        thresh = threshold if threshold is not None else self.config.tactile_contact_threshold
        if force_sum is None:
            state = self.get_state(full=False)
            force_sum = np.asarray(state.get("tactile_force_sum", np.zeros(HAND_TACTILE_SUM_SHAPE)))
        norm = np.linalg.norm(force_sum, axis=1)  # (5,) L2 per finger
        return norm > thresh

    def make_command(
        self,
        qpos: np.ndarray,
        mode: int | None = None,
        tor_max: int | None = None,
        kp: int | None = None,
        ki: int | None = None,
        kd: int | None = None,
    ) -> Any:
        command = xhc.HandCommand_t()
        mode = self.config.mode if mode is None else mode
        tor_max = self.config.tor_max if tor_max is None else tor_max
        kp = self.config.kp if kp is None else kp
        ki = self.config.ki if ki is None else ki
        kd = self.config.kd if kd is None else kd

        for i in range(HAND_DOF):
            cmd = command.finger_command[i]
            cmd.id = i
            # Per-joint gains replace scalar defaults when configured.
            cmd.kp = int(self.config.kp_per_joint[i]) if self.config.kp_per_joint is not None else int(kp)
            cmd.ki = int(self.config.ki_per_joint[i]) if self.config.ki_per_joint is not None else int(ki)
            cmd.kd = int(self.config.kd_per_joint[i]) if self.config.kd_per_joint is not None else int(kd)
            cmd.position = float(qpos[i])
            cmd.tor_max = (
                int(self.config.tor_max_per_joint[i]) if self.config.tor_max_per_joint is not None else int(tor_max)
            )
            cmd.mode = int(mode)
            cmd.res0 = 0
            cmd.res1 = 0
            cmd.res2 = 0
            cmd.res3 = 0
        return command

    def write_command_positions(self, qpos: np.ndarray) -> None:
        for i in range(HAND_DOF):
            self.hand_command.finger_command[i].position = float(qpos[i])

    def read_raw_state(self, force_update: bool = False):
        if self.control is None or not self.connected_flag:
            return None, None
        result = self.control.read_state(self.config.device_id, force_update)
        return self._unpack_result(result)

    def parse_state(self, hand_state, full: bool = False) -> dict[str, Any]:
        qpos = nan_array(HAND_JOINT_SHAPE)
        current = nan_array(HAND_JOINT_SHAPE)
        finger_ids = np.full(HAND_JOINT_SHAPE, -1, dtype=np.int32)
        sensor_ids = np.full(HAND_JOINT_SHAPE, -1, dtype=np.int32)
        raw_position = nan_array(HAND_JOINT_SHAPE)
        finger_temperatures = np.zeros(HAND_JOINT_SHAPE, dtype=np.float64)
        commboard_err = np.zeros(HAND_JOINT_SHAPE, dtype=np.int32)
        jointboard_err = np.zeros(HAND_JOINT_SHAPE, dtype=np.int32)
        tipboard_err = np.zeros(HAND_JOINT_SHAPE, dtype=np.int32)

        finger_state = getattr(hand_state, "finger_state", [])
        for item in finger_state:
            idx = int(getattr(item, "id", -1))
            if idx < 0 or idx >= HAND_DOF:
                continue

            finger_ids[idx] = idx
            sensor_ids[idx] = int(getattr(item, "sensor_id", -1))
            qpos[idx] = float(getattr(item, "position", np.nan))
            current[idx] = float(getattr(item, "torque", np.nan))
            raw_position[idx] = float(getattr(item, "raw_position", np.nan))
            finger_temperatures[idx] = float(getattr(item, "temperature", 0.0))
            commboard_err[idx] = int(getattr(item, "commboard_err", 0))
            # SDK misspelling: "jonitboard_err" for "jointboard_err".
            jointboard_err[idx] = int(getattr(item, "jonitboard_err", getattr(item, "jointboard_err", 0)))
            tipboard_err[idx] = int(getattr(item, "tipboard_err", 0))

        tactile_force_sum = self.parse_tactile_sum(hand_state)
        state = {
            "qpos": qpos,
            "current": current,
            "timestamp": time.time(),
            # (5,120,3) raw force array + (5,3) combined force per finger.
            "tactile_force": self.parse_tactile(hand_state),
            "tactile_force_sum": tactile_force_sum,
            "tactile_contact": self.detect_contact(force_sum=tactile_force_sum),
            "commboard_err": commboard_err,
            "jointboard_err": jointboard_err,
            "tipboard_err": tipboard_err,
        }

        # Sensor-level diagnostics are populated only for full-state callers.
        sensor_temperatures = np.zeros(HAND_CONTACT_SHAPE, dtype=np.float64)
        if full:
            sensor_data = getattr(hand_state, "sensor_data", None)
            if sensor_data:
                for j, sd in enumerate(sensor_data):
                    if j >= HAND_FINGER_COUNT:
                        break
                    sensor_temperatures[j] = float(getattr(sd, "calc_temperature", 0.0))

        if full:
            state.update(
                {
                    "finger_ids": finger_ids,
                    "sensor_ids": sensor_ids,
                    "raw_position": raw_position,
                    "finger_temperatures": finger_temperatures,
                    "sensor_temperatures": sensor_temperatures,
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

    _MAX_SENSORS: int = 5  # thumb, index, middle, ring, little

    def _iter_sensors(self, hand_state):
        """Iterate sensor data entries by positional index (0-4 → thumb..little)."""
        sensor_data = getattr(hand_state, "sensor_data", None)
        if not sensor_data:
            return
        for i, sensor in enumerate(sensor_data):
            if i >= self._MAX_SENSORS:
                break
            yield i, sensor

    def parse_tactile(self, hand_state) -> np.ndarray:
        """Parse tactile force array (5 fingers × 120 points × 3 axes).

        Returns SDK readings scaled by 0.1 for compatibility with the deployed
        thresholds.  This scale is not claimed to be a verified SI conversion.

        A measured software bias is subtracted when available.
        """
        tactile = np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64)
        for i, sensor in self._iter_sensors(hand_state):
            raw_force = getattr(sensor, "raw_force", None)
            if raw_force is None:
                continue
            for j, force in enumerate(raw_force):
                if j >= TACTILE_POINTS_PER_FINGER:
                    break
                tactile[i, j, 0] = float(getattr(force, "fx", 0.0)) * 0.1
                tactile[i, j, 1] = float(getattr(force, "fy", 0.0)) * 0.1
                tactile[i, j, 2] = float(getattr(force, "fz", 0.0)) * 0.1
        if self._tactile_bias_raw is not None:
            tactile -= self._tactile_bias_raw
        return tactile

    def parse_tactile_sum(self, hand_state) -> np.ndarray:
        """Parse combined force per finger (5 × 3 axes).

        Returns SDK readings scaled by 0.1 for compatibility with the deployed
        thresholds.  This scale is not claimed to be a verified SI conversion.

        A measured software bias is subtracted when available.
        """
        force_sum = np.zeros(HAND_TACTILE_SUM_SHAPE, dtype=np.float64)
        for i, sensor in self._iter_sensors(hand_state):
            calc_force = getattr(sensor, "calc_force", None)
            if calc_force is None:
                continue
            force_sum[i, 0] = float(getattr(calc_force, "fx", 0.0)) * 0.1
            force_sum[i, 1] = float(getattr(calc_force, "fy", 0.0)) * 0.1
            force_sum[i, 2] = float(getattr(calc_force, "fz", 0.0)) * 0.1
        if self._tactile_bias_ft is not None:
            force_sum -= self._tactile_bias_ft
        return force_sum

    def _limit_joint_range(self, qpos: np.ndarray) -> np.ndarray:
        # Command targets always stay inside strict SDK bounds. The reporting
        # tolerance here is command-side only and is unrelated to measured
        # feedback_bound_tolerance_rad.
        if not self.config.clip_joint_limit:
            self.last_joint_limit_clipped = False
            return qpos

        clipped = np.clip(qpos, self.config.qpos_min, self.config.qpos_max)
        max_deviation = float(np.max(np.abs(qpos - clipped)))
        self.last_joint_limit_clipped = max_deviation > self.config.clip_report_tolerance
        return clipped

    @property
    def feedback_bound_stats(self) -> dict[str, Any]:
        """Return a copy of cumulative finite measured-qpos bound statistics."""
        return {
            "checks": self._feedback_bound_checks,
            "outside_bounds_frames": self._feedback_bound_outside,
            "over_tolerance_frames": self._feedback_bound_over_tolerance,
            "max_violation_rad": self._feedback_bound_max_violation_rad,
            "per_joint_outside_counts": self._feedback_bound_joint_outside_counts.copy(),
            "per_joint_over_tolerance_counts": self._feedback_bound_joint_over_tolerance_counts.copy(),
        }

    def _record_feedback_bound_check(self, qpos: Any) -> None:
        """Count one raw feedback frame without modifying it or command bounds."""
        qpos_array = np.asarray(qpos, dtype=np.float64)
        if qpos_array.shape != HAND_JOINT_SHAPE or not np.all(np.isfinite(qpos_array)):
            return

        violation = np.maximum(
            np.maximum(self.config.qpos_min - qpos_array, 0.0),
            np.maximum(qpos_array - self.config.qpos_max, 0.0),
        )
        outside = violation > 1e-12
        over_tolerance = violation > self.config.feedback_bound_tolerance_rad

        self._feedback_bound_checks += 1
        self._feedback_bound_joint_outside_counts[outside] += 1
        self._feedback_bound_joint_over_tolerance_counts[over_tolerance] += 1
        if np.any(outside):
            self._feedback_bound_outside += 1
            self._feedback_bound_max_violation_rad = max(
                self._feedback_bound_max_violation_rad,
                float(np.max(violation)),
            )
        if np.any(over_tolerance):
            self._feedback_bound_over_tolerance += 1
            joints = np.flatnonzero(over_tolerance)
            self._feedback_bounds_warn(
                "XHand feedback exceeds bound tolerance "
                "(SDK joints=%s, deviation_deg=%s, tolerance=%.3fdeg, over_tolerance=%d/%d)",
                joints.tolist(),
                np.round(np.rad2deg(violation[joints]), 3).tolist(),
                float(np.rad2deg(self.config.feedback_bound_tolerance_rad)),
                self._feedback_bound_over_tolerance,
                self._feedback_bound_checks,
            )

    def is_valid_qpos_state(self, state: dict[str, Any]) -> bool:
        qpos = state.get("qpos", None)
        if qpos is None:
            return False
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        return qpos.size == HAND_DOF and bool(np.all(np.isfinite(qpos)))

    def _array12(self, value) -> np.ndarray:
        return safe_resize(value, HAND_DOF)

    def _empty_state(self) -> dict[str, Any]:
        return {
            "finger_ids": np.full(HAND_JOINT_SHAPE, -1, dtype=np.int32),
            "sensor_ids": np.full(HAND_JOINT_SHAPE, -1, dtype=np.int32),
            "raw_position": nan_array(HAND_JOINT_SHAPE),
            "finger_temperatures": np.zeros(HAND_JOINT_SHAPE, dtype=np.float64),
            "sensor_temperatures": np.zeros(HAND_CONTACT_SHAPE, dtype=np.float64),
            "commboard_err": np.zeros(HAND_JOINT_SHAPE, dtype=np.int32),
            "jointboard_err": np.zeros(HAND_JOINT_SHAPE, dtype=np.int32),
            "tipboard_err": np.zeros(HAND_JOINT_SHAPE, dtype=np.int32),
            "tactile_force": np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64),
            "tactile_force_sum": np.zeros(HAND_TACTILE_SUM_SHAPE, dtype=np.float64),
            "tactile_contact": np.zeros(HAND_CONTACT_SHAPE, dtype=bool),
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
        # The SDK always returns tuple[ErrorStruct, HandState_t].
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            return result[0], result[1]
        return None, None

    def error_code(self, err) -> int | None:
        if err is None:
            return None
        code = getattr(err, "error_code", -1)
        if code is None:
            code = -1
        return int(code)

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

        self.error_state = True
