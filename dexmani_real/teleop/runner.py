"""Operator-supervised teleop with one current observation per real control step."""

import time
from pathlib import Path

import numpy as np

from dexmani_real.calibration import VR_TRANSFORM_PATH
from dexmani_real.recording.recorder import (
    AsyncEpisodeRecorder,
    RecordingError,
    snapshot_recording_metadata,
)
from dexmani_real.robot.action import ActionRealizer
from dexmani_real.robot.arm_homing import build_policy_home_planner, home_policy_robot
from dexmani_real.robot.hand_homing import home_hand
from dexmani_real.robot.robot import DispatchError
from dexmani_real.runtime.observation import read_observation
from dexmani_real.runtime.operator_input import KeyboardInput, OperatorCommand
from dexmani_real.runtime.safety import RunEndReason, begin_motion, revoke_motion
from dexmani_real.teleop.audio_feedback import AudioFeedback
from dexmani_real.teleop.config import TeleopConfig
from dexmani_real.teleop.control.controller import TeleopController, execute_control_step
from dexmani_real.teleop.control.vr_mapping import VRWristMapper
from dexmani_real.teleop.retargeting.retargeter import DexPilotHandRetargeter, TAGHandRetargeter
from dexmani_real.teleop.vr_transform import load_vr_transform
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)
_DEBUG_FAILURE_LIMIT = 10
_END_AUDIO_GRACE_S = 3.0


def _build_hand_retargeter(config: TeleopConfig):
    """Build the configured hand retargeter without owning runtime state."""
    if not config.runtime.policy.hand_enabled:
        return None
    if config.runtime.policy.hand_retargeting_type == "tag":
        return TAGHandRetargeter(
            fingertip_link_names=config.runtime.hand.fingertip_link_names,
            tag_config=config.runtime.tag_retargeting,
            urdf_path=config.hand_urdf_path,
        )
    return DexPilotHandRetargeter(
        hand_type="right",
        retargeting_type=config.runtime.policy.hand_retargeting_type,
        dexpilot_config=config.runtime.dexpilot_retargeting,
    )


class TeleopRunner:
    """Own operator, recording and control state for one local teleop session."""

    def __init__(self, shared, config: TeleopConfig, robot, *, start_vr=None):
        self.shared = shared
        self.robot = robot
        self.start_vr = start_vr
        self.config = config
        self.runtime = config.runtime
        self.recorder = (
            AsyncEpisodeRecorder(
                Path(__file__).resolve().parents[2]
                / self.runtime.policy.episodes_dir
                / config.task_label,
                control_hz=self.runtime.teleop.control_hz,
                rgb_shape=(self.runtime.camera.height, self.runtime.camera.width, 3),
            )
            if self.runtime.policy.recording_enabled
            else None
        )
        self.keyboard = KeyboardInput(
            estop_callback=lambda: setattr(shared.estop_request, "value", True),
            stop_callback=lambda: revoke_motion(shared, reason=RunEndReason.OPERATOR),
            quit_callback=lambda: revoke_motion(shared, reason=RunEndReason.QUIT),
        )
        self.audio = AudioFeedback()
        self.controller = None
        self.active = False
        self.paused = False
        self.resume_requested = False
        self.resume_after_ns = 0
        self.quit_pending = False
        self.pending_termination_reason = None
        self.next_tick = 0.0
        self.control_dt = 1 / self.runtime.teleop.control_hz
        self.failures = 0
        self.capture_started_s = 0.0

        self.home_planner = None

    def _pause_control(self, reason, *, run_end_reason=RunEndReason.OPERATOR, stop_motion=True):
        revoke_motion(self.shared, reason=run_end_reason)
        if stop_motion:
            self.robot.stop()
        if self.controller is not None:
            self.controller.clear_reference()
        if (
            self.recorder is not None
            and self.recorder.is_recording
            and self.pending_termination_reason is None
        ):
            self.pending_termination_reason = reason
        self.paused, self.resume_requested = True, False

    def _finish_capture(self, save, reason, *, announce=True) -> None:
        if self.recorder is None or not self.recorder.is_recording:
            self.pending_termination_reason = None
            return
        reason = self.pending_termination_reason or reason
        published = None
        if save:
            published = self.recorder.save_episode(reason=reason)
        else:
            self.recorder.discard_episode(reason=reason)
        self.pending_termination_reason = None
        if announce:
            self.audio.play("save" if published is not None else "discard")

    def _stop_capture(self, reason, *, stop_motion=True):
        self._pause_control(
            reason, run_end_reason=RunEndReason.EXECUTOR_BOUNDARY, stop_motion=stop_motion
        )
        logger.warning("Teleop paused: %s", reason)
        self.audio.play("emergency")
        if self.recorder is not None and self.recorder.is_recording:
            self._finish_capture(True, reason, announce=False)

    def _poll_blocking_commands(self, commands=()):
        self.robot.check()
        aborted = False
        # Other keys are consumed, including the tail of the poll that contained H.
        # Q revokes HOME authority before any blocking capture finalization.
        for cmd in (*commands, *self.keyboard.poll(timeout=0)):
            if cmd in (OperatorCommand.EMERGENCY_STOP, OperatorCommand.QUIT, OperatorCommand.STOP):
                self._handle_operator_command(cmd)
                aborted = True
        return bool(
            aborted
            or self.shared.estop_request.value
            or self.shared.error_state.value
            or self.shared.quit_requested.value
            or not self.shared.is_running.value
        )

    def _start_segment(self):
        if self.resume_requested:
            return
        revoke_motion(self.shared)
        self.robot.stop()
        self.paused = True
        if self.recorder is not None and self.recorder.is_recording:
            self._finish_capture(True, "pause")
        row = read_observation(
            self.shared,
            self.runtime,
            self.robot,
            require_hand=self.runtime.policy.hand_enabled,
            require_camera=self.recorder is not None,
            require_vr=True,
        )
        if row is None:
            print("Start requires fresh robot, VR and recording resources", flush=True)
            return
        if self.recorder is not None:
            self.recorder.start_episode(
                task_label=self.config.task_label,
                **snapshot_recording_metadata(
                    self.shared, self.runtime, collection_source="teleop"
                ),
            )
        self.capture_started_s = time.monotonic()
        self.pending_termination_reason = None
        # Disk finalization/START may block. Never anchor to observations from
        # before that boundary or append across the operator's wall-clock pause.
        self.resume_after_ns = time.monotonic_ns()
        self.resume_requested = True

    def _handle_operator_command(self, cmd, *, home_commands=()) -> bool:
        if cmd is OperatorCommand.EMERGENCY_STOP:
            self.shared.estop_request.value = True
            revoke_motion(self.shared, reason=RunEndReason.ESTOP)
            self.audio.play("emergency")
            return True
        if (
            self.shared.estop_request.value
            or self.shared.error_state.value
            or self.shared.quit_requested.value
            or not self.shared.is_running.value
        ):
            return True
        if cmd is OperatorCommand.QUIT:
            self._pause_control("quit", run_end_reason=RunEndReason.QUIT)
            has_capture = self.recorder is not None and self.recorder.is_recording
            if self.quit_pending or (not self.active and not has_capture):
                if has_capture:
                    self._finish_capture(True, "quit", announce=False)
                self.audio.play("end")
                self.shared.quit_requested.value = True
                return True
            self.quit_pending = True
            self.audio.play("quit")
        elif cmd in (OperatorCommand.STOP, OperatorCommand.DISCARD):
            reason = cmd.value.lower()
            self._pause_control(reason)
            self._finish_capture(cmd is OperatorCommand.STOP, reason)
        elif cmd is OperatorCommand.HOME:
            self._pause_control("home")
            if not self._poll_blocking_commands(home_commands):
                self.audio.play("home")
                self.home_planner = self.home_planner or build_policy_home_planner(self.runtime)
                if home_policy_robot(
                    self.shared,
                    self.runtime,
                    self.home_planner,
                    robot=self.robot,
                    abort_requested=self._poll_blocking_commands,
                ):
                    self.audio.queue("home_done")
            self._poll_blocking_commands()
            return True
        elif cmd is OperatorCommand.PAUSE and (self.active or self.quit_pending):
            self.quit_pending = False
            if self.paused:
                self._start_segment()
            else:
                self._pause_control("pause")
                self.audio.play("pause")
        elif cmd is OperatorCommand.BEGIN and not self.active and not self.quit_pending:
            self._start_segment()
        return False

    def _try_resume(self, row) -> bool:
        if self.resume_requested and all(
            int(stamp) > self.resume_after_ns
            for stamp in (
                row.arm["timestamp_ns"][0],
                row.vr["recv_ts_ns"],
                row.hand["timestamp_ns"][0]
                if row.hand is not None
                else row.observation_timestamp_ns,
            )
        ):
            epoch = int(self.shared.run_id.value)
            prepared = self.controller.reset_reference(row)
            with self.shared.motion_lock:
                authorized = (
                    prepared
                    and epoch == int(self.shared.run_id.value)
                    and begin_motion(self.shared)
                )
            if authorized:
                event = "resume" if self.active else "begin"
                self.active = True
                self.paused = self.resume_requested = False
                self.failures = 0
                self.next_tick = time.monotonic()
                # C may have just saved the previous segment; do not cut off
                # that result announcement when fresh observations arrive.
                self.audio.queue(event)
                return True
            self._stop_capture("resume_unavailable" if self.active else "begin_unavailable")
        return False

    def _execute_control_step(self, row):
        control_ok, result, interrupted = execute_control_step(
            self.controller, self.shared, self.robot, row, self.recorder
        )
        if interrupted:
            logger.info("teleop interrupted: dispatch=%s", result)
            self._stop_capture("motion_revoked", stop_motion=False)
            return
        if self.recorder is not None and not control_ok:
            self._stop_capture("control_failure")
            return
        if (
            self.recorder is not None
            and time.monotonic() - self.capture_started_s
            >= self.runtime.policy.max_record_duration_s
        ):
            self._pause_control("max_record_duration", run_end_reason=RunEndReason.TIMEOUT)
            self._finish_capture(True, "max_record_duration")
            return

        if self.recorder is None:
            self.failures = self.failures + 1 if not control_ok else 0
            if self.failures >= _DEBUG_FAILURE_LIMIT:
                self._stop_capture("control_failure")

    def _wait_for_vr(self) -> bool:
        deadline = time.monotonic() + self.runtime.safety.readiness_timeouts_s["vr"]
        while True:
            if not self.keyboard.healthy:
                self.shared.estop_request.value = True
            if self._poll_blocking_commands():
                return False
            self.robot.service_idle()
            if self.shared.vr_ready.wait(timeout=0.05):
                # Consume startup keystrokes before accepting a new B after readiness.
                return not self._poll_blocking_commands()
            if time.monotonic() >= deadline:
                raise RuntimeError("VR failed to become ready before the startup timeout")

    def _initialize(self) -> bool:
        # Sensor readiness follows XHand connection and the tactile calibration
        # attempt; reset before the slower IK/retargeting initialization.
        home_result = home_hand(
            self.shared,
            self.runtime,
            robot=self.robot,
            abort_requested=self._poll_blocking_commands,
        )
        if not home_result.ok:
            if home_result.interrupted:
                self.shared.quit_requested.value = True
                return False
            if (
                self.shared.quit_requested.value
                and self.shared.is_running.value
                and not self.shared.estop_request.value
                and not self.shared.error_state.value
            ):
                return False
            raise RuntimeError(f"startup hand home failed: {home_result.reason}")
        if self._poll_blocking_commands():
            return False
        realizer = ActionRealizer.for_mode(self.runtime, "eef")
        calibration = load_vr_transform(VR_TRANSFORM_PATH)
        mapping = self.runtime.policy.vr_mapping
        mapper = VRWristMapper(
            pos_scale=mapping.pos_scale,
            rot_scale=mapping.rot_scale,
            vr_to_robot_rot=calibration.transform,
            base_to_world_rot=np.eye(3),
        )
        self.controller = TeleopController(
            realizer, mapper, self.runtime, _build_hand_retargeter(self.config)
        )
        if self._poll_blocking_commands():
            return False
        if self.start_vr is not None:
            self.start_vr()
        if not self._wait_for_vr():
            return False
        print("准备进入遥操作", flush=True)
        return True

    def run(self) -> None:
        failure = None
        try:
            self.keyboard.start()
            check_services = self.robot.check_services
            self.robot.check_services = lambda: (
                (check_services is None or check_services()) and self.keyboard.healthy
            )
            ready = self._initialize()
            while ready and self.shared.is_running.value and not self.shared.quit_requested.value:
                self.robot.check()
                if not self.active or self.paused:
                    self.robot.service_idle()
                if not self.keyboard.healthy:
                    self.shared.estop_request.value = True
                if self.shared.estop_request.value or self.shared.error_state.value:
                    self._stop_capture(
                        "estop" if self.shared.estop_request.value else "runtime_failure"
                    )
                    break
                if self.recorder is not None:
                    self.recorder.check_error()
                timeout = 0.005
                if self.active and not self.paused:
                    timeout = min(timeout, max(0.0, self.next_tick - time.monotonic()))
                commands = self.keyboard.poll(timeout=timeout)
                if any(
                    c in commands
                    for c in (
                        OperatorCommand.STOP,
                        OperatorCommand.QUIT,
                        OperatorCommand.EMERGENCY_STOP,
                    )
                ):
                    commands = [
                        c
                        for c in commands
                        if c not in (OperatorCommand.HOME, OperatorCommand.BEGIN)
                    ]
                for index, cmd in enumerate(commands):
                    if self._handle_operator_command(
                        cmd,
                        home_commands=commands[index + 1 :] if cmd is OperatorCommand.HOME else (),
                    ):
                        break
                if (
                    self.shared.estop_request.value
                    or self.shared.error_state.value
                    or not self.shared.is_running.value
                ):
                    break
                if self.shared.quit_requested.value or (
                    not self.resume_requested and (not self.active or self.paused)
                ):
                    continue
                if not self.paused and time.monotonic() < self.next_tick:
                    continue
                if (
                    self.recorder is not None
                    and self.recorder.is_recording
                    and time.monotonic() - self.capture_started_s
                    >= self.runtime.policy.max_record_duration_s
                ):
                    self._pause_control("max_record_duration", run_end_reason=RunEndReason.TIMEOUT)
                    self._finish_capture(True, "max_record_duration")
                    continue
                self.next_tick = time.monotonic() + self.control_dt
                row = read_observation(
                    self.shared,
                    self.runtime,
                    self.robot,
                    require_hand=self.runtime.policy.hand_enabled,
                    require_camera=self.recorder is not None,
                    require_vr=True,
                )
                if row is None:
                    if self.recorder is not None:
                        from dexmani_real.runtime.observation import (
                            read_camera_frame,
                            sample_is_fresh,
                        )

                        camera = read_camera_frame(self.shared)
                        if camera is None or not sample_is_fresh(
                            camera["timestamp_ns"], self.runtime.camera.max_frame_age_s
                        ):
                            revoke_motion(self.shared, reason=RunEndReason.HARDWARE_FAULT)
                            self._stop_capture("camera_unavailable")
                            raise RuntimeError("required recording camera unavailable")
                    self._stop_capture("observation_unavailable")
                    continue
                if self.paused:
                    self._try_resume(row)
                    continue
                self._execute_control_step(row)
        except KeyboardInterrupt:
            self.shared.estop_request.value = True
            raise
        except Exception as exc:
            failure = exc
            self.shared.error_state.value = True
            self.shared.is_running.value = False
            revoke_motion(
                self.shared,
                reason=RunEndReason.RECORDING_FAILURE
                if isinstance(exc, RecordingError)
                else RunEndReason.HARDWARE_FAULT
                if isinstance(exc, DispatchError)
                else RunEndReason.POLICY_FAILURE,
            )
            logger.exception("teleop failed")
            self.audio.play("emergency")
        finally:
            revoke_motion(self.shared, reason=RunEndReason.RUNTIME_SHUTDOWN)
            try:
                if self.robot._motion_active or self.robot._hand_stop_pending:
                    self.robot.stop()
            except Exception as exc:
                if failure is None:
                    failure = exc
                logger.exception("teleop stop failed")
            try:
                if self.controller is not None:
                    self.controller.clear_reference()
            except Exception as exc:
                if failure is None:
                    failure = exc
                logger.exception("teleop reference cleanup failed")
            try:
                if self.recorder is not None:
                    if self.recorder.is_recording:
                        self._finish_capture(True, "interrupted", announce=False)
            except Exception as exc:
                if failure is None:
                    failure = exc
                logger.exception("teleop recording cleanup failed")
            finally:
                if self.recorder is not None:
                    try:
                        self.recorder.close()
                    except Exception as exc:
                        if failure is None:
                            failure = exc
                        logger.exception("teleop recorder close failed")
            try:
                # Motion is revoked and recording is finalized. Let the exit
                # or emergency cue finish before close() cancels the player.
                if not self.audio.wait_until_idle(timeout_s=_END_AUDIO_GRACE_S):
                    logger.warning("Final audio did not finish within %.1fs", _END_AUDIO_GRACE_S)
            except Exception:
                logger.warning("Final audio unavailable", exc_info=True)
            for close in (self.keyboard.stop, self.audio.close):
                try:
                    close()
                except Exception as exc:
                    if failure is None:
                        failure = exc
                    logger.exception("teleop resource cleanup failed")
        if failure is not None:
            raise failure
