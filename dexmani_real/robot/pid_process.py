"""PIDProcess — standalone process owning XArmAPI SDK connection.

Runs a 250Hz target-forwarding loop in a separate process, communicating with the
main (teleop) process via shared memory channels (PIDTargetChannel, PIDStateChannel).

Uses position servo mode (mode 1) with set_servo_angle_j() to avoid mode conflicts
with the main process's XArmAPI connection. The arm's internal servo handles
smoothing, velocity limiting, and PID.

Architecture:
    Main Process (50Hz)                    PID Process (250Hz)
    ──────────────────                    ───────────────────
    PIDTargetChannel.write(target)  ──→   target_ch.read() → set_servo_angle_j (mode 1)
    PIDStateChannel.read()         ←──    state_ch.write(qpos, error)

Ref: ManiUniCon process isolation pattern — SDK connection owned by PID process,
     main process never calls XArmAPI directly.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
import time

import numpy as np

from dexmani_real.log import get_logger

logger = get_logger(__name__)

# ── PIDProcess ──


class PIDProcess(mp.Process):
    """Standalone process that owns the XArmAPI SDK connection.

    Runs a 250Hz loop: reads target qpos from shared memory, forwards to arm
    via set_servo_angle_j() (position servo mode 1). The arm's internal servo
    handles PID, smoothing, and velocity limiting.

    A separate 50Hz thread publishes current arm state back to shared memory
    for the main process to consume.
    """

    def __init__(
        self,
        ip: str = "192.168.1.111",
        dt: float = 1.0 / 250.0,
        target_timeout_s: float = 0.2,
    ) -> None:
        super().__init__(name="pid_process", daemon=True)
        self._ip = ip
        self._dt = float(dt)
        self._target_timeout_s = float(target_timeout_s)

        # Events
        self._stop_event = mp.Event()

    def run(self) -> None:
        """Main loop at 250Hz — runs in the child process.

        Uses position servo mode (mode 1) with set_servo_angle_j() to avoid
        mode conflicts with the main process's XArmAPI connection.
        The arm's internal servo handles smoothing/velocity limiting.
        """
        from dexmani_real.shm.pid_channels import PIDStateChannel, PIDTargetChannel

        # Attach to shared memory channels (created by main process)
        target_ch = PIDTargetChannel(create=False)
        state_ch = PIDStateChannel(create=False)

        # Create own XArmAPI connection (isolated from main process)
        from xarm.wrapper import XArmAPI

        try:
            arm = XArmAPI(self._ip, is_radian=True)
        except (OSError, ConnectionError, RuntimeError) as e:
            logger.error("PIDProcess: XArmAPI init failed: %s", e)
            state_ch.write(np.zeros(7), error_state=True)
            return

        try:
            arm.clean_error()
            arm.clean_warn()
            arm.motion_enable(True)

            # Init sequence: set mode 1 (position servo) — same as main process
            self._init_mode(arm, 1)
            arm.set_collision_sensitivity(1)

            # Read initial position
            code, states = arm.get_joint_states(is_radian=True, num=1)
            if code == 0 and len(states) > 0:
                current_qpos = np.asarray(states[0], dtype=np.float64)
            else:
                current_qpos = np.zeros(7, dtype=np.float64)

            # Start state reader thread (50Hz)
            state_thread = threading.Thread(
                target=self._state_reader, args=(arm, state_ch), daemon=True
            )
            state_thread.start()

            last_target_ts: float = 0.0
            last_valid_qpos: np.ndarray = current_qpos.copy()
            rate_limiter = self._rate_limiter(1.0 / self._dt)

            logger.info("PIDProcess: 250Hz loop started (mode 1, position servo)")

            while not self._stop_event.is_set():
                rate_limiter()

                # 1. Read target from main process
                target, target_ts = target_ch.read()

                # 2. Timeout check: no new target → hold current position
                now = time.perf_counter()
                if target is None or (now - max(target_ts, last_target_ts) > self._target_timeout_s):
                    # Hold position: re-send current position to arm
                    try:
                        code, states = arm.get_joint_states(is_radian=True, num=1)
                        if code == 0 and len(states) > 0:
                            hold_qpos = np.asarray(states[0], dtype=np.float64)
                            if np.all(np.isfinite(hold_qpos)) and hold_qpos.shape[0] >= 7:
                                arm.set_servo_angle_j(angles=hold_qpos[:7].tolist(), is_radian=True)
                    except (RuntimeError, OSError):
                        pass
                    if target is not None:
                        last_target_ts = target_ts
                    continue
                last_target_ts = target_ts

                # NaN guard
                if not np.all(np.isfinite(target)):
                    try:
                        arm.set_servo_angle_j(angles=last_valid_qpos.tolist(), is_radian=True)
                    except (RuntimeError, OSError):
                        pass
                    continue

                last_valid_qpos = target[:7].copy()

                # 3. Read current position for state channel
                try:
                    code, states = arm.get_joint_states(is_radian=True, num=1)
                except (RuntimeError, OSError) as e:
                    logger.error("PIDProcess: get_joint_states failed: %s", e)
                    state_ch.write(current_qpos, error_state=True)
                    continue

                if code != 0:
                    logger.error("PIDProcess: arm error code=%d", code)
                    state_ch.write(current_qpos, error_state=True)
                    continue

                if len(states) > 0:
                    current_qpos = np.asarray(states[0], dtype=np.float64)
                if current_qpos.shape[0] < 7:
                    continue

                # 4. Send target position to arm (arm's internal servo does PID/smoothing)
                try:
                    code = arm.set_servo_angle_j(angles=target[:7].tolist(), is_radian=True)
                except (RuntimeError, OSError) as e:
                    logger.error("PIDProcess: set_servo_angle_j failed: %s", e)
                    state_ch.write(current_qpos, error_state=True)
                    continue

                if code != 0:
                    logger.error("PIDProcess: set_servo_angle_j code=%d", code)
                    state_ch.write(current_qpos, error_state=True)
                    continue

            # Cleanup
            state_thread.join(timeout=1.0)
            arm.disconnect()

        except Exception:
            logger.exception("PIDProcess: fatal error in main loop")
            try:
                state_ch.write(np.zeros(7), error_state=True)
            except Exception:
                pass

    def stop(self) -> None:
        """Signal the PID process to stop and wait for it to exit."""
        self._stop_event.set()
        self.join(timeout=3.0)
        if self.is_alive():
            logger.warning("PIDProcess did not exit within timeout, terminating")
            self.terminate()
            self.join(timeout=1.0)

    # ── Internal helpers ──

    @staticmethod
    def _rate_limiter(hz: float):
        """Return a simple sleep-based rate limiter closure."""
        period = 1.0 / hz
        last = time.perf_counter()

        def wait() -> None:
            nonlocal last
            now = time.perf_counter()
            elapsed = now - last
            if elapsed < period:
                time.sleep(period - elapsed)
            last = time.perf_counter()

        return wait

    def _state_reader(self, arm, state_ch) -> None:
        """50Hz thread: read arm state → publish to shared memory."""
        period = 0.02  # 50 Hz
        while not self._stop_event.is_set():
            try:
                code, states = arm.get_joint_states(is_radian=True, num=1)
                if code == 0:
                    qpos = np.asarray(states[0], dtype=np.float64)
                    if qpos.shape[0] >= 7 and np.all(np.isfinite(qpos)):
                        state_ch.write(qpos[:7], error_state=False)
                    else:
                        state_ch.write(np.zeros(7), error_state=True)
                else:
                    state_ch.write(np.zeros(7), error_state=True)
            except (RuntimeError, OSError):
                state_ch.write(np.zeros(7), error_state=True)
            time.sleep(period)

    def _init_mode(self, arm, mode: int) -> None:
        """Transition arm to target control mode via idle intermediate state."""
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(0.05)
        arm.set_mode(mode)
        arm.set_state(0)
        time.sleep(0.05)
        arm.set_state(0)
