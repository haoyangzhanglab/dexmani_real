"""XHand 12-DOF robot hand hardware driver via xhand_controller SDK."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dexmani_real.config.defaults import hand
from dexmani_real.utils.limits import validate_hand_limit_nesting
from dexmani_real.utils.schema import (
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


from dexmani_real.utils.log import capture_native_stdout, extract_native_diagnostics, get_logger
from dexmani_real.utils.serialization import from_dict_helper

logger = get_logger(__name__)


@dataclass
class XHandConfig:
    # Canonical transport protocol: "ethercat" (production) or "serial" (RS485
    # diagnostic adapter). Validated to this closed set in __post_init__ so the
    # driver never guesses a protocol from a fuzzy alias.
    comm_type: str = "ethercat"
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

    mechanical_qpos_min: np.ndarray = field(
        default_factory=lambda: np.asarray(hand.mechanical_qpos_min_rad, dtype=np.float64)
    )

    mechanical_qpos_max: np.ndarray = field(
        default_factory=lambda: np.asarray(hand.mechanical_qpos_max_rad, dtype=np.float64)
    )

    # Scalar fallback gains, applied to every joint when the per-joint
    # overrides below are not supplied. The production hand worker supplies
    # per-joint gains resolved from config.defaults.HandParams.
    kp: int = 100
    ki: int = 0
    kd: int = 0
    # Per-joint overrides replace the scalar gains when configured.
    # When set (shape (12,)), individual joint gains replace the scalar
    # kp/ki/kd. The deployed config raises index abduction (J3) kp to 120.
    kp_per_joint: np.ndarray | None = None  # (12,) per-joint kp overrides
    ki_per_joint: np.ndarray | None = None  # (12,) per-joint ki overrides
    kd_per_joint: np.ndarray | None = None  # (12,) per-joint kd overrides
    # Scalar fallback current limit, applied to every joint when the per-joint
    # overrides below are not supplied.
    tor_max: int = 300  # mA
    # Per-joint tor_max overrides. When set (shape (12,)), individual joint
    # current limits replace the scalar tor_max. The deployed config raises
    # index abduction (J3) to 360 mA to handle sideways loading.
    tor_max_per_joint: np.ndarray | None = None  # (12,) per-joint tor_max overrides
    mode: int = 3

    # ── F1: Tactile contact detection ──
    # L2 threshold in the SDK's scaled tactile units.  The vendor/firmware
    # provenance for a physical SI conversion has not been validated on this
    # installation, so recordings deliberately label the unit unknown.
    tactile_contact_threshold: float = 1.0

    def __post_init__(self) -> None:
        if self.comm_type not in ("ethercat", "serial"):
            raise ValueError("XHand comm_type must be 'ethercat' or 'serial'")
        if self.device_name is not None and not isinstance(self.device_name, str):
            raise ValueError("XHand device_name must be a string or null")
        if not isinstance(self.baudrate, int) or self.baudrate <= 0:
            raise ValueError("XHand baudrate must be a positive integer")
        if not isinstance(self.device_id, int) or self.device_id < 0:
            raise ValueError("XHand device_id must be a non-negative integer")
        if self.ethercat_slave_position < -1:
            raise ValueError("ethercat_slave_position must be -1 (unknown) or non-negative")
        command_lower = np.asarray(self.qpos_min, dtype=np.float64)
        command_upper = np.asarray(self.qpos_max, dtype=np.float64)
        mechanical_lower = np.asarray(self.mechanical_qpos_min, dtype=np.float64)
        mechanical_upper = np.asarray(self.mechanical_qpos_max, dtype=np.float64)
        rated_lower = np.asarray(hand.mechanical_qpos_min_rad, dtype=np.float64)
        rated_upper = np.asarray(hand.mechanical_qpos_max_rad, dtype=np.float64)
        home_qpos = np.asarray(self.home_qpos, dtype=np.float64)
        vectors = (command_lower, command_upper, mechanical_lower, mechanical_upper, home_qpos)
        if any(value.shape != HAND_JOINT_SHAPE for value in vectors):
            raise ValueError(f"XHand home and limit arrays must have shape {HAND_JOINT_SHAPE}")
        if not np.all(np.isfinite(np.concatenate(vectors))):
            raise ValueError("XHand home and limit arrays must be finite")
        validate_hand_limit_nesting(
            command_lower,
            command_upper,
            mechanical_lower,
            mechanical_upper,
            rated_lower,
            rated_upper,
            label="XHand",
        )
        if np.any(home_qpos < command_lower - 1e-12) or np.any(home_qpos > command_upper + 1e-12):
            raise ValueError("XHand home_qpos must be inside command limits")
        if not np.isfinite(self.tactile_contact_threshold) or self.tactile_contact_threshold < 0:
            raise ValueError("tactile_contact_threshold must be finite and non-negative")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "XHandConfig":
        """Reconstruct from a serialized dict."""
        return cls(**from_dict_helper(cls, d))  # type: ignore[arg-type]


@dataclass(frozen=True)
class XHandSample:
    """Immutable snapshot of one successful XHand read.

    Field names mirror the shared ``HAND_STATE_DTYPE`` (``tactile_sum`` not
    ``tactile_force_sum``) so the worker's per-tick parse is a straight copy.
    Arrays are validated for shape/finite-ness, copied, and marked read-only on
    construction, so a caller can never alias or mutate the driver's live
    buffers. ``tactile_valid`` is process-local provenance: malformed tactile
    payloads degrade to shape-stable zeros without invalidating complete joint
    feedback. ``error_state`` and liveness are object fields on :class:`XHand`,
    not part of this snapshot — the snapshot is pure feedback.
    """

    qpos: np.ndarray  # (12,) rad
    current: np.ndarray  # (12,) per-joint current
    tactile_force: np.ndarray  # (5, 120, 3) raw force per sensor point
    tactile_sum: np.ndarray  # (5, 3) combined force per finger
    tactile_contact: np.ndarray  # (5,) bool per-finger contact
    tactile_valid: bool  # exact 5 x (120 raw points + one calc force), all finite
    commboard_err: np.ndarray  # (12,) int32 comm board error register
    jointboard_err: np.ndarray  # (12,) int32 joint board error register
    tipboard_err: np.ndarray  # (12,) int32 tip board error register

    _SPEC: tuple[tuple[str, tuple[int, ...], type], ...] = (
        ("qpos", HAND_JOINT_SHAPE, np.float64),
        ("current", HAND_JOINT_SHAPE, np.float64),
        ("tactile_force", HAND_TACTILE_FORCE_SHAPE, np.float64),
        ("tactile_sum", HAND_TACTILE_SUM_SHAPE, np.float64),
        ("tactile_contact", HAND_CONTACT_SHAPE, bool),
        ("commboard_err", HAND_JOINT_SHAPE, np.int32),
        ("jointboard_err", HAND_JOINT_SHAPE, np.int32),
        ("tipboard_err", HAND_JOINT_SHAPE, np.int32),
    )

    def __post_init__(self) -> None:
        if not isinstance(self.tactile_valid, (bool, np.bool_)):
            raise ValueError("XHandSample.tactile_valid must be boolean")
        object.__setattr__(self, "tactile_valid", bool(self.tactile_valid))
        for name, shape, dtype in self._SPEC:
            arr: np.ndarray = np.asarray(getattr(self, name), dtype=dtype)
            if arr.shape != shape:
                raise ValueError(f"XHandSample.{name} must have shape {shape}, got {arr.shape}")
            if name != "tactile_contact" and not np.all(np.isfinite(arr)):
                raise ValueError(f"XHandSample.{name} must be finite")
            arr = arr.copy()
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)


class XHand:
    def __init__(self, config: XHandConfig):
        self.connected_flag: bool = False
        self.error_state: bool = False
        self.last_error_message: str = ""
        self.config = config
        self.control: Any = None
        self.device_name: str | None = None
        self.hand_command: Any = None

        self.last_qpos_cmd: np.ndarray | None = None
        self.last_error_code: int | None = None

        # reset_sensor() can leave an offset; subtract a fresh no-contact mean.
        self._tactile_bias_ft: np.ndarray | None = None  # (5, 3)   — calc_force bias
        self._tactile_bias_raw: np.ndarray | None = None  # (5, 120, 3) — raw_force bias
        self.tactile_calibrated: bool = False
        self._last_tactile_payload_valid: bool | None = None

        self._stub_mode = False
        self.last_hand_ids: list[int] = []
        # Canonical transport protocol (validated in XHandConfig.__post_init__).
        self.cached_comm_type = self.config.comm_type
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
            self.last_qpos_cmd = np.asarray(self.config.home_qpos, dtype=np.float64)
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
            self._verify_device()
            self.hand_command = self.make_command(np.asarray(self.config.home_qpos, dtype=np.float64))
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

        return True

    def _retry_open_device(self, comm_type: str) -> bool:
        """Enumerate devices and open port with configurable retries.

        RS485 may need several attempts after cold start (C++ SDK retries
        internally, but may still fail intermittently).

        Discovery and open share ONE XHandControl per attempt (no throwaway
        discovery control followed by a fresh open control), which avoids
        "write sdo failed" from duplicate raw sockets left behind by a separate
        enumeration pass.  SDK ownership stays inside the spawned hand worker:
        moving it to Main would violate the process boundary and make clean
        shutdown harder to prove.
        """
        # RS485 may need several attempts after cold start; EtherCAT uses a
        # lower retry cap because each failed open_ethercat() transitions the
        # slave to OP regardless of PDO/SDO outcome, and repeated transitions
        # compound state corruption without recovery value.
        retries = max(
            1, int(self.config.open_ethercat_retries if comm_type == "ethercat" else self.config.open_serial_retries)
        )
        delay = max(0.0, float(self.config.open_serial_retry_delay_s))

        # device_name stays None only while the config supplies no name and we
        # have not discovered one this connect.  Discovery runs on the same
        # control that will open (no throwaway discovery control).
        device_name = self.config.device_name
        self.device_name = device_name

        for attempt in range(1, retries + 1):
            self.control = xhc.XHandControl()  # one controller per attempt

            if device_name is None:
                discovery_output = ""
                discovery_capture = None
                try:
                    with capture_native_stdout() as discovery_capture:
                        devices = self.control.enumerate_devices(comm_type)
                    discovery_output = discovery_capture.text
                    discovery_diagnostics = extract_native_diagnostics(discovery_output)
                    if discovery_diagnostics:
                        logger.warning("XHand discovery SDK diagnostics:\n%s", "\n".join(discovery_diagnostics))
                except Exception:
                    discovery_output = discovery_capture.text if discovery_capture is not None else ""
                    logger.error(
                        "XHand %s discovery raised%s",
                        comm_type,
                        f"; vendor output:\n{discovery_output}" if discovery_output else "",
                        exc_info=True,
                    )
                    self._close_control()
                    raise
                if devices is None or len(devices) == 0:
                    self.error_state = True
                    self.last_error_code = -2
                    self.last_error_message = f"no XHand device found for {comm_type}"
                    if discovery_output:
                        logger.warning("XHand discovery vendor output:\n%s", discovery_output)
                    self._close_control()
                    self._diagnose_connection_failure()
                    return False
                device_name = devices[0]
                self.device_name = device_name

            open_capture = None
            try:
                with capture_native_stdout() as open_capture:
                    if comm_type == "serial":
                        err = self.control.open_serial(device_name, int(self.config.baudrate))
                    elif comm_type == "ethercat":
                        err = self.control.open_ethercat(device_name)
                    else:
                        self.error_state = True
                        self.last_error_code = -3
                        self.last_error_message = f"unsupported comm_type: {self.config.comm_type}"
                        self._close_control()
                        return False
            except Exception:
                open_output = open_capture.text if open_capture is not None else ""
                self._close_control()
                logger.error(
                    "XHand open attempt %d/%d raised%s",
                    attempt,
                    retries,
                    f"; vendor output:\n{open_output}" if open_output else "",
                    exc_info=True,
                )
                raise
            open_output = open_capture.text

            if self.error_ok(err):
                if "Operation not permitted" in open_output:
                    logger.warning(
                        "XHand EtherCAT entered OP, but the SDK could not enable real-time thread scheduling; "
                        "operation continues with normal scheduling"
                    )
                open_diagnostics = extract_native_diagnostics(
                    open_output,
                    ignore=("Operation not permitted",),
                )
                if open_diagnostics:
                    logger.warning("XHand SDK initialization diagnostics:\n%s", "\n".join(open_diagnostics))
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
            if open_output:
                logger.warning("XHand open attempt %d/%d vendor output:\n%s", attempt, retries, open_output)
            # Close the failed control before retry
            self._close_control()
            if attempt < retries:
                # On EtherCAT, the first failure may be a stale-slave-state
                # condition (previous session exited without clean disconnect,
                # leaving the slave in OP).  A longer wait gives the slave's
                # SM-watchdog time to expire so the retry sees a clean INIT
                # slave.  On later attempts the standard retry delay is
                # sufficient — if the slave was in OP, the first retry's wait
                # already handled it; if not, it's a different error class.
                _retry_delay = delay
                if comm_type == "ethercat" and attempt == 1:
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

    def _close_control(self) -> None:
        """Best-effort close of the current control and clear the reference.

        Used on failed open attempts and aborted discovery so a dangling
        handle to a close_device()'d control is not left for a later
        disconnect() to trip over.
        """
        if self.control is not None:
            try:
                self.control.close_device()
            except (OSError, RuntimeError):
                logger.warning("failed XHand control did not close cleanly", exc_info=True)
            self.control = None

    def _init_hand_state(self) -> bool:
        """Force-refresh hardware state and read the initial qpos seed.

        Do not use SDK cache here — after open_serial() the cache
        may be all zeros. Returns False when no valid state can be read.
        """
        attempts = max(1, int(self.config.init_state_read_attempts))
        interval = max(0.0, float(self.config.init_state_read_interval))

        sample: XHandSample | None = None
        for _ in range(attempts):
            try:
                sample = self.get_state(force_update=True)
                break
            except (RuntimeError, ValueError):
                sample = None
            if interval > 0:
                time.sleep(interval)

        if sample is None:
            return False

        measured_qpos = sample.qpos
        # This projects only the internal command-history seed. No command
        # is published here, and raw measured feedback stays unchanged.
        self.last_qpos_cmd = np.clip(measured_qpos, self.config.qpos_min, self.config.qpos_max)
        logger.debug("Initial qpos from hand state: %s", measured_qpos)
        return True

    def initialize_tactile(self) -> bool:
        """Reset fingertip sensors and estimate a no-contact software bias.

        Explicit, worker-invoked step (split out of ``connect()``): connecting
        only opens the device, confirms the device ID, and seeds the command
        history.  Any failure here — contact/load at startup, a dead sensor, or
        a read error — degrades to ``calibrated=False`` and returns False
        without blocking joint control.
        """
        # Never reuse a previous calibration across a fresh initialization.
        # An old bias must not hide a load in the startup read.
        self._tactile_bias_ft = None
        self._tactile_bias_raw = None
        self.tactile_calibrated = False

        if self._stub_mode:
            # Following simulation has no physical tactile payload.
            self._tactile_bias_ft = np.zeros(HAND_TACTILE_SUM_SHAPE, dtype=np.float64)
            self._tactile_bias_raw = np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64)
            return False

        try:
            sample = self.get_state(force_update=True)
        except Exception:
            self._tactile_bias_ft = np.zeros(HAND_TACTILE_SUM_SHAPE, dtype=np.float64)
            self._tactile_bias_raw = np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64)
            self.tactile_calibrated = False
            logger.warning("Tactile initialization degraded: startup read failed", exc_info=True)
            return False

        # Check the raw startup load before reset_sensor() can redefine the
        # loaded state as zero. Missing/malformed tactile feedback also fails
        # calibration closed while leaving non-tactile hand operation usable.
        if not sample.tactile_valid or self._tactile_load_present(sample.tactile_sum):
            self._tactile_bias_ft = np.zeros(HAND_TACTILE_SUM_SHAPE, dtype=np.float64)
            self._tactile_bias_raw = np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64)
            self.tactile_calibrated = False
            logger.error("Tactile calibration refused: contact/load or invalid tactile data detected at startup")
            return False

        # Reset all five fingertip sensors after power-on.  The SDK documents
        # that some sensors may need an explicit reset before they begin
        # reporting data (sensor IDs 17–21: thumb, index, middle, ring, little).
        # Failures are logged but do not block operation — a dead sensor is
        # less harmful than a refused connection.
        try:
            self._reset_tactile_sensors()
        except Exception:
            self._tactile_bias_ft = np.zeros(HAND_TACTILE_SUM_SHAPE, dtype=np.float64)
            self._tactile_bias_raw = np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64)
            self.tactile_calibrated = False
            logger.warning("Tactile initialization degraded: reset/verification failed", exc_info=True)
        return self.tactile_calibrated

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

        Some firmware prints ``Unknow Cmd!`` despite returning success for
        ``reset_sensor()``.  Capture that native stdout noise, count it once,
        and rely on the subsequent measured verification and software bias.
        """
        if self._stub_mode:
            self.tactile_calibrated = False
            return
        device_id = self.config.device_id

        MAX_OUTER_ITERS = 5
        VERIFY_THRESHOLD = 2.0
        FINGER_LABELS = ["thumb", "index", "middle", "ring", "pinky"]

        bad_indices: list[int] = list(range(HAND_FINGER_COUNT))
        _last_mags: np.ndarray | None = None  # cached for else-clause diagnostics
        unsupported_reset_count = 0

        for outer in range(MAX_OUTER_ITERS):
            # ── Reset only sensors that still have residual offset ──
            for idx in bad_indices:
                sensor_id = 17 + idx
                for attempt in range(3):
                    reset_capture = None
                    try:
                        with capture_native_stdout() as reset_capture:
                            err = self.control.reset_sensor(device_id, sensor_id)
                        if "Unknow Cmd!" in reset_capture.text:
                            unsupported_reset_count += 1
                        reset_diagnostics = extract_native_diagnostics(
                            reset_capture.text,
                            ignore=("Unknow Cmd!",),
                        )
                        if reset_diagnostics:
                            logger.warning(
                                "XHand tactile-reset SDK diagnostics:\n%s",
                                "\n".join(reset_diagnostics),
                            )
                        if self.error_ok(err):
                            break
                    except Exception:
                        reset_output = reset_capture.text if reset_capture is not None else ""
                        logger.warning(
                            "Tactile sensor %d reset attempt raised%s",
                            sensor_id,
                            f"; vendor output:\n{reset_output}" if reset_output else "",
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
            err, hand_state = self._unpack_result(self.control.read_state(device_id, self._effective_force_update(True)))
            if not self.error_ok(err):
                logger.warning(
                    "Tactile verify read failed (iter %d/%d, code=%s) — retrying all sensors",
                    outer + 1,
                    MAX_OUTER_ITERS,
                    self.error_code(err),
                )
                continue  # retry all sensors next iteration

            try:
                _, force_sum = self._parse_tactile_payload(hand_state)  # raw — bias is None during init
            except Exception:
                logger.warning(
                    "Tactile verify payload malformed (iter %d/%d) — retrying all sensors",
                    outer + 1,
                    MAX_OUTER_ITERS,
                    exc_info=True,
                )
                continue
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

        if unsupported_reset_count:
            logger.info(
                "XHand firmware did not implement %d tactile reset request(s); "
                "measured startup verification and software bias were used",
                unsupported_reset_count,
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
            self.tactile_calibrated = False
            return

        ft_samples: list[np.ndarray] = []
        raw_samples: list[np.ndarray] = []

        for _ in range(n_samples):
            err, hand_state = self._unpack_result(self.control.read_state(device_id, self._effective_force_update(True)))
            if not self.error_ok(err):
                logger.warning(
                    "Tactile bias sample read failed (code=%s) — "
                    "bias will be zero (no correction).  "
                    "Ensure the hand is not in contact with any object during init.",
                    self.error_code(err),
                )
                self._tactile_bias_ft = np.zeros(HAND_TACTILE_SUM_SHAPE, dtype=np.float64)
                self._tactile_bias_raw = np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64)
                self.tactile_calibrated = False
                return

            try:
                raw_force, force_sum = self._parse_tactile_payload(hand_state)
            except Exception:
                logger.warning(
                    "Tactile bias sample payload malformed — bias will be zero (no correction)",
                    exc_info=True,
                )
                self._tactile_bias_ft = np.zeros(HAND_TACTILE_SUM_SHAPE, dtype=np.float64)
                self._tactile_bias_raw = np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64)
                self.tactile_calibrated = False
                return

            ft_samples.append(force_sum)
            raw_samples.append(raw_force)

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
        logger.debug(
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
        except Exception:
            logger.warning("XHand SDK version: unavailable", exc_info=True)

        # Hand type (left / right)
        try:
            err, hand_type = self.control.get_hand_type(device_id)
            if self.error_ok(err):
                self.device_identity["hand_type"] = str(hand_type)
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
            else:
                code = self.error_code(err)
                msg = str(getattr(err, "error_message", ""))
                logger.warning("XHand get_serial_number failed: code=%s msg=%s", code, msg)
        except Exception:
            logger.warning("XHand get_serial_number: unavailable", exc_info=True)

        logger.info(
            "XHand ready: SDK=%s type=%s serial=%s device_id=%d",
            self.device_identity["sdk_version"],
            self.device_identity["hand_type"],
            self.device_identity["serial_number"],
            device_id,
        )

    # ── EtherCAT slave state management ──

    # AL state constants (EtherCAT standard / SOEM convention). Only INIT is
    # used here: disconnect requests the slave transition back to INIT.
    _EC_STATE_INIT = 1

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
        if self.cached_comm_type != "ethercat":
            return False  # serial/RS485 has no slave state machine
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

    def disconnect(self) -> None:
        """Release the hardware connection (idempotent).

        Two-stage cleanup so the EtherCAT slave is left in a state that
        permits reconnection without a power cycle:

        1.  Request INIT via the SDK's ``set_firmware_state`` (best-effort).
        2.  Call ``close_device()``.
        3.  Wait for the slave's internal SM-watchdog to expire so it
            auto-transitions out of OP before the next session starts.

        A never-opened or already-closed control is closed at most once: the
        close clears ``self.control``, so a repeated ``disconnect()`` is a
        no-op rather than a second access to a freed device handle.
        """
        if self._stub_mode:
            self.connected_flag = False
            return
        # Close whenever a control handle exists — including the residual
        # open-raise / unsupported-comm_type paths that leave a never-opened
        # control behind (see _retry_open_device).  The slave INIT request and
        # post-close watchdog wait apply only when we actually connected.
        if self.control is not None:
            if self.connected_flag:
                self._request_slave_init()
            self._close_control()
            if self.connected_flag:
                time.sleep(self._POST_DISCONNECT_WATCHDOG_WAIT_S)
        self.connected_flag = False

    def _diagnose_connection_failure(self) -> None:
        if self.cached_comm_type == "ethercat":
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

    def _stub_sample(self) -> XHandSample:
        """Return following-simulation state with zero current/tactile data."""
        return XHandSample(
            qpos=np.array(
                self.last_qpos_cmd if self.last_qpos_cmd is not None else self.config.home_qpos,
                dtype=np.float64,
                copy=True,
            ),
            current=np.zeros(HAND_JOINT_SHAPE, dtype=np.float64),
            tactile_force=np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64),
            tactile_sum=np.zeros(HAND_TACTILE_SUM_SHAPE, dtype=np.float64),
            tactile_contact=np.zeros(HAND_CONTACT_SHAPE, dtype=bool),
            tactile_valid=False,
            commboard_err=np.zeros(HAND_JOINT_SHAPE, dtype=np.int32),
            jointboard_err=np.zeros(HAND_JOINT_SHAPE, dtype=np.int32),
            tipboard_err=np.zeros(HAND_JOINT_SHAPE, dtype=np.int32),
        )

    def get_state(self, force_update: bool | None = None) -> XHandSample:
        """Read one hardware state and return an immutable ``XHandSample``.

        A failed read raises ``RuntimeError`` (the single failure protocol —
        there is no NaN half-state or None return).  A malformed frame
        (out-of-range/duplicate/missing joint, non-finite position) raises
        ``ValueError`` from parsing.
        """
        if self._stub_mode:
            return self._stub_sample()

        if force_update is None:
            force_update = self.config.force_update_state

        err, hand_state = self.read_raw_state(force_update=force_update)

        if not self.error_ok(err) or hand_state is None:
            self._record_error(err)
            raise RuntimeError(
                f"XHand read failed: code={self.last_error_code} msg={self.last_error_message!r}"
            )

        sample = self._parse_sample(hand_state)
        self.last_error_code = 0
        self.last_error_message = ""
        # Bridge board-status (Layer 2) into safety gate: per-joint hardware
        # board error registers gate commands on hardware-level faults.
        # Unlike send/read errors (tracked via _record_error), board errors are
        # transient and self-clear on the next healthy frame.  A single
        # read/write result is decided by the current SDK return: ``error_state``
        # is recomputed from the board registers on every successful read, so
        # there is no separate "clear" step to invoke.
        self.error_state = bool(
            np.any(sample.commboard_err)
            or np.any(sample.jointboard_err)
            or np.any(sample.tipboard_err)
        )
        return sample

    def send_action(self, action: np.ndarray) -> bool:
        target_qpos = np.asarray(action, dtype=np.float64)
        if target_qpos.shape != HAND_JOINT_SHAPE or not np.all(np.isfinite(target_qpos)):
            self.last_error_message = "XHand.send_action rejected invalid shape or NaN/Inf"
            logger.warning(self.last_error_message)
            return False
        if not self._validate_joint_range(target_qpos):
            return False
        if self._stub_mode:
            # Track the accepted request so following simulation mirrors the
            # same all-or-nothing command contract as hardware.
            self.last_qpos_cmd = target_qpos.copy()
            return True

        if self.control is None or self.hand_command is None:
            if self.hand_command is None:
                self.error_state = True
                self.last_error_message = "hand_command is None (not initialized)"
            return False

        # The policy boundary owns primary command-to-command validation. The
        # driver redundantly checks bounds/delta and forwards the endpoint
        # unchanged.
        qpos_cmd = target_qpos.copy()
        self.write_command_positions(qpos_cmd)
        err = self.control.send_command(self.config.device_id, self.hand_command)

        if self.error_ok(err):
            self.last_qpos_cmd = qpos_cmd.copy()
            return True

        self._record_error(err)
        return False

    def detect_contact(self, threshold: float | None = None, force_sum: np.ndarray | None = None) -> np.ndarray:
        """Detect per-finger contact from tactile force.

        Uses the L2 norm of the combined force vector (fx, fy, fz) on each
        fingertip sensor, compared against tactile_contact_threshold.

        Args:
            threshold: Override in the same unknown scaled units as force_sum.
            force_sum: Pre-parsed (5,3) force_sum array. When provided,
                       skips get_state() call (used inside _parse_sample to
                       avoid recursion).

        Returns:
            bool array of shape (5,), True where L2-norm > threshold.
        """
        thresh = threshold if threshold is not None else self.config.tactile_contact_threshold
        if force_sum is None:
            force_sum = self.get_state().tactile_sum
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
        result = self.control.read_state(self.config.device_id, self._effective_force_update(force_update))
        return self._unpack_result(result)

    def _effective_force_update(self, force_update: bool) -> bool:
        """Coerce the read-state force_update flag for the active transport.

        RS485 ``read_state(force_update=True)`` first re-sends the last command
        (vendored ``serial_communication.cpp``); a state poll must never
        implicitly re-issue a motion command, so serial reads are always cached
        (force_update=False).  EtherCAT ignores the parameter (returns the PDO
        cache), so the requested value is passed through unchanged.
        """
        if self.cached_comm_type == "serial":
            return False
        return bool(force_update)

    def _parse_sample(self, hand_state) -> XHandSample:
        """Parse one raw SDK ``HandState_t`` into an immutable ``XHandSample``.

        A malformed frame — out-of-range, negative, duplicate, or missing joint
        id, or a non-finite position — fails the whole frame (``RuntimeError`` /
        ``ValueError``) rather than silently emitting a partial or NaN half-state.
        """
        qpos = np.full(HAND_JOINT_SHAPE, np.nan, dtype=np.float64)
        current = np.full(HAND_JOINT_SHAPE, np.nan, dtype=np.float64)
        commboard_err = np.zeros(HAND_JOINT_SHAPE, dtype=np.int32)
        jointboard_err = np.zeros(HAND_JOINT_SHAPE, dtype=np.int32)
        tipboard_err = np.zeros(HAND_JOINT_SHAPE, dtype=np.int32)

        finger_state = getattr(hand_state, "finger_state", [])
        seen: set[int] = set()
        for item in finger_state:
            idx = int(getattr(item, "id", -1))
            if idx < 0 or idx >= HAND_DOF:
                raise RuntimeError(f"XHand parse: joint id {idx} out of range [0, {HAND_DOF})")
            if idx in seen:
                raise RuntimeError(f"XHand parse: duplicate joint id {idx}")
            seen.add(idx)

            qpos[idx] = float(getattr(item, "position", np.nan))
            current[idx] = float(getattr(item, "torque", np.nan))
            commboard_err[idx] = int(getattr(item, "commboard_err", 0))
            # SDK misspelling: "jonitboard_err" for "jointboard_err".
            jointboard_err[idx] = int(getattr(item, "jonitboard_err", getattr(item, "jointboard_err", 0)))
            tipboard_err[idx] = int(getattr(item, "tipboard_err", 0))

        # A complete frame must enumerate every joint exactly once.
        if len(seen) != HAND_DOF:
            raise RuntimeError(f"XHand parse: {len(seen)}/{HAND_DOF} joints reported")

        try:
            tactile_force, tactile_sum = self._parse_tactile_payload(hand_state)
        except Exception:
            if self._last_tactile_payload_valid is not False:
                logger.warning(
                    "XHand tactile payload invalid — preserving joint feedback and publishing invalid zeros",
                    exc_info=True,
                )
            tactile_force = np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64)
            tactile_sum = np.zeros(HAND_TACTILE_SUM_SHAPE, dtype=np.float64)
            tactile_contact = np.zeros(HAND_CONTACT_SHAPE, dtype=bool)
            tactile_valid = False
        else:
            if self._last_tactile_payload_valid is False:
                logger.info("XHand tactile payload recovered")
            tactile_contact = self.detect_contact(force_sum=tactile_sum)
            tactile_valid = True
        self._last_tactile_payload_valid = tactile_valid
        return XHandSample(
            qpos=qpos,
            current=current,
            tactile_force=tactile_force,
            tactile_sum=tactile_sum,
            tactile_contact=tactile_contact,
            tactile_valid=tactile_valid,
            commboard_err=commboard_err,
            jointboard_err=jointboard_err,
            tipboard_err=tipboard_err,
        )

    @staticmethod
    def _force_xyz(force: Any, *, label: str) -> np.ndarray:
        """Return one finite SDK force vector or raise a tactile-only error."""
        if force is None:
            raise ValueError(f"{label} is missing")
        try:
            value = np.asarray(
                [float(getattr(force, axis)) for axis in ("fx", "fy", "fz")],
                dtype=np.float64,
            )
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label} must expose numeric fx/fy/fz") from exc
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{label} must contain three finite values")
        return value

    def _parse_tactile_payload(self, hand_state: Any) -> tuple[np.ndarray, np.ndarray]:
        """Strictly parse one complete five-sensor tactile payload."""
        sensor_data = getattr(hand_state, "sensor_data", None)
        if sensor_data is None:
            raise ValueError("sensor_data is missing")
        try:
            sensors = list(sensor_data)
        except (TypeError, ValueError) as exc:
            raise ValueError("sensor_data must be iterable") from exc
        if len(sensors) != HAND_FINGER_COUNT:
            raise ValueError(
                f"sensor_data must contain exactly {HAND_FINGER_COUNT} sensors, got {len(sensors)}"
            )

        tactile = np.empty(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64)
        force_sum = np.empty(HAND_TACTILE_SUM_SHAPE, dtype=np.float64)
        for sensor_index, sensor in enumerate(sensors):
            force_sum[sensor_index] = self._force_xyz(
                getattr(sensor, "calc_force", None),
                label=f"sensor_data[{sensor_index}].calc_force",
            )
            raw_force = getattr(sensor, "raw_force", None)
            if raw_force is None:
                raise ValueError(f"sensor_data[{sensor_index}].raw_force is missing")
            try:
                raw_points = list(raw_force)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"sensor_data[{sensor_index}].raw_force must be iterable"
                ) from exc
            if len(raw_points) != TACTILE_POINTS_PER_FINGER:
                raise ValueError(
                    f"sensor_data[{sensor_index}].raw_force must contain exactly "
                    f"{TACTILE_POINTS_PER_FINGER} points, got {len(raw_points)}"
                )
            for point_index, force in enumerate(raw_points):
                tactile[sensor_index, point_index] = self._force_xyz(
                    force,
                    label=f"sensor_data[{sensor_index}].raw_force[{point_index}]",
                )

        # Preserve the deployed scale without claiming an SI conversion.
        tactile *= 0.1
        force_sum *= 0.1
        if self._tactile_bias_raw is not None:
            tactile -= self._tactile_bias_raw
        if self._tactile_bias_ft is not None:
            force_sum -= self._tactile_bias_ft
        if not np.all(np.isfinite(tactile)) or not np.all(np.isfinite(force_sum)):
            raise ValueError("scaled/bias-corrected tactile payload must remain finite")
        return tactile, force_sum

    def parse_tactile(self, hand_state) -> np.ndarray:
        """Parse tactile force array (5 fingers × 120 points × 3 axes).

        Returns SDK readings scaled by 0.1 for compatibility with the deployed
        thresholds.  This scale is not claimed to be a verified SI conversion.

        A measured software bias is subtracted when available.
        """
        tactile, _ = self._parse_tactile_payload(hand_state)
        return tactile

    def parse_tactile_sum(self, hand_state) -> np.ndarray:
        """Parse combined force per finger (5 × 3 axes).

        Returns SDK readings scaled by 0.1 for compatibility with the deployed
        thresholds.  This scale is not claimed to be a verified SI conversion.

        A measured software bias is subtracted when available.
        """
        _, force_sum = self._parse_tactile_payload(hand_state)
        return force_sum

    def _validate_joint_range(self, qpos: np.ndarray) -> bool:
        """Reject an out-of-range endpoint without changing any joint target."""
        command_low = np.asarray(self.config.qpos_min, dtype=np.float64)
        command_high = np.asarray(self.config.qpos_max, dtype=np.float64)
        mechanical_low = np.asarray(self.config.mechanical_qpos_min, dtype=np.float64)
        mechanical_high = np.asarray(self.config.mechanical_qpos_max, dtype=np.float64)
        command_violation = np.flatnonzero((qpos < command_low - 1e-12) | (qpos > command_high + 1e-12))
        mechanical_violation = np.flatnonzero((qpos < mechanical_low - 1e-12) | (qpos > mechanical_high + 1e-12))
        violating = mechanical_violation if mechanical_violation.size else command_violation
        if violating.size == 0:
            return True
        joint_index = int(violating[0])
        boundary = "mechanical" if mechanical_violation.size else "command"
        lower = mechanical_low if mechanical_violation.size else command_low
        upper = mechanical_high if mechanical_violation.size else command_high
        self.last_error_message = (
            f"XHand.send_action rejected {boundary} joint limit violation: "
            f"joint={joint_index} target={qpos[joint_index]:.6f}rad "
            f"range=[{lower[joint_index]:.6f},{upper[joint_index]:.6f}]rad"
        )
        logger.warning(self.last_error_message)
        return False

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
