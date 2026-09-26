"""Operator-supervised teleop with one current observation per real control step."""

import signal
import time

import numpy as np

from dexmani_real.calibration import VR_TRANSFORM_PATH
from dexmani_real.planning import XArm7MotionPlanner
from dexmani_real.planning.kinematics.ik import make_online_ik_config
from dexmani_real.recording.client import RecorderClient
from dexmani_real.recording.storage.schema import FRAME_OK
from dexmani_real.robot.arm_homing import build_policy_home_planner, home_policy_robot
from dexmani_real.robot.hand_homing import home_hand
from dexmani_real.runtime.observation import read_observation
from dexmani_real.runtime.operator_input import KeyboardInput, OperatorCommand
from dexmani_real.runtime.safety import begin_motion, revoke_motion
from dexmani_real.teleop.audio_feedback import AudioFeedback
from dexmani_real.teleop.config import TeleopConfig
from dexmani_real.teleop.control.controller import TeleopController, execute_control_step
from dexmani_real.teleop.control.vr_mapping import VRWristMapper
from dexmani_real.teleop.retargeting.retargeter import DexPilotHandRetargeter, TAGHandRetargeter
from dexmani_real.teleop.vr_transform import load_vr_transform
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)
_DEBUG_FAILURE_LIMIT = 10


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
    """Own operator, recording and control state for one teleop worker."""

    def __init__(self, shared, config: TeleopConfig):
        self.shared = shared
        self.config = config
        self.runtime = config.runtime
        self.recorder = RecorderClient(shared) if self.runtime.policy.recording_enabled else None
        self.keyboard = KeyboardInput(
            estop_callback=lambda: setattr(shared.estop_request, "value", True)
        )
        self.audio = AudioFeedback()
        self.controller = None
        self.active = False
        self.paused = False
        self.resume_requested = False
        self.pause_ns = 0
        self.quit_pending = False
        self.quit_deadline = 0.0
        self.next_tick = 0.0
        self.failures = 0
        self.max_rows = round(
            self.runtime.policy.max_record_duration_s * self.runtime.teleop.control_hz
        )

        self.home_planner = None

    def _stop_episode(self, save, reason, abnormal=False):
        revoke_motion(self.shared)
        if self.controller is not None:
            self.controller.clear_reference()
        self.active = self.paused = self.resume_requested = False
        if self.recorder is not None and self.recorder.is_recording:
            if abnormal:
                self.recorder.technical_status = "invalid"
            self.recorder.stop_episode(save=save, reason=reason)

    def _pause(self, manual, mark_episode=True):
        revoke_motion(self.shared)
        self.pause_ns = time.monotonic_ns()
        if self.controller is not None:
            self.controller.clear_reference()
        if mark_episode and self.recorder is not None and self.recorder.is_recording:
            self.recorder.had_pause = True
        self.paused, self.resume_requested = True, not manual
        self.audio.play("pause")

    def _home_abort_requested(self):
        for cmd in self.keyboard.poll(timeout=0):
            if cmd is OperatorCommand.EMERGENCY_STOP:
                self.shared.estop_request.value = True
            elif cmd is OperatorCommand.QUIT:
                self.shared.quit_requested.value = True
            elif cmd in (OperatorCommand.STOP, OperatorCommand.DISCARD):
                return True
        return bool(
            self.shared.estop_request.value
            or self.shared.quit_requested.value
            or not self.shared.is_running.value
        )

    def _begin_episode(self):
        if self.recorder is not None and self.recorder.stop_pending:
            return
        row = read_observation(
            self.shared,
            self.runtime,
            require_hand=self.runtime.policy.hand_enabled,
            require_camera=self.recorder is not None,
            require_vr=True,
        )
        if row is None:
            print("Begin requires fresh robot, VR and recording resources", flush=True)
            return
        if self.recorder is not None and not self.recorder.start_episode(
            task_label=self.config.task_label, operator=self.config.operator
        ):
            return
        # START may block on disk; anchor only after it finishes.
        row = read_observation(
            self.shared,
            self.runtime,
            require_hand=self.runtime.policy.hand_enabled,
            require_camera=self.recorder is not None,
            require_vr=True,
        )
        if row is None or not self.controller.reset_reference(row) or not begin_motion(self.shared):
            self._stop_episode(False, "begin_unavailable")
            return
        self.active = True
        self.failures = 0
        self.next_tick = time.monotonic()
        self.audio.play("begin")

    def _handle_operator_command(self, cmd) -> bool:
        if cmd is OperatorCommand.EMERGENCY_STOP:
            self.shared.estop_request.value = True
            return True
        if cmd is OperatorCommand.QUIT:
            revoke_motion(self.shared)
            if self.recorder is not None and self.recorder.is_recording:
                self._pause(True, mark_episode=False)
                self.quit_pending = True
                self.quit_deadline = time.monotonic() + self.runtime.policy.quit_save_timeout_s
                print("Quit: S save, D discard, H save and home", flush=True)
            else:
                self.shared.quit_requested.value = True
        elif cmd in (OperatorCommand.STOP, OperatorCommand.DISCARD, OperatorCommand.HOME):
            self._stop_episode(cmd is not OperatorCommand.DISCARD, cmd.value.lower())
            self.audio.play("discard" if cmd is OperatorCommand.DISCARD else "end")
            if self.recorder is not None:
                self.recorder.join_stop()
            if cmd is OperatorCommand.HOME and not self.shared.error_state.value:
                self.home_planner = self.home_planner or build_policy_home_planner(self.runtime)
                self.audio.play("home")
                if home_policy_robot(
                    self.shared,
                    self.runtime,
                    self.home_planner,
                    abort_requested=self._home_abort_requested,
                ):
                    self.audio.queue("home_done")
            if self.quit_pending:
                self.shared.quit_requested.value = True
        elif cmd is OperatorCommand.PAUSE and self.active and not self.quit_pending:
            if self.paused:
                self.resume_requested = True
            else:
                self._pause(True)
        elif cmd is OperatorCommand.BEGIN and not self.active and not self.quit_pending:
            self._begin_episode()
        return False

    def _try_resume(self, row) -> bool:
        if self.resume_requested and all(
            int(stamp) > self.pause_ns
            for stamp in (
                row.arm["timestamp_ns"][0],
                row.vr["recv_ts_ns"],
                row.hand["timestamp_ns"][0]
                if row.hand is not None
                else row.observation_timestamp_ns,
            )
        ):
            if self.controller.reset_reference(row) and begin_motion(self.shared):
                self.paused = self.resume_requested = False
                self.failures = 0
                self.next_tick = time.monotonic()
                self.audio.play("resume")
                return True
        return False

    def _execute_control_step(self, row, tick_started):
        submitted_before = self.recorder.frame_count if self.recorder is not None else 0
        status = execute_control_step(self.controller, self.shared, row, self.recorder)
        if (
            self.recorder is not None
            and self.recorder.frame_count > submitted_before
            and self.recorder.frame_count >= self.max_rows
        ):
            self._stop_episode(True, "max_record_duration")
            return
        self.next_tick = tick_started + 1 / self.runtime.teleop.control_hz
        self.failures = self.failures + 1 if status != FRAME_OK else 0
        if self.failures >= _DEBUG_FAILURE_LIMIT:
            self._pause(True)

    def run(self) -> None:
        failure = None
        try:
            planner = XArm7MotionPlanner.create_default(
                online_ik_profile=make_online_ik_config(self.runtime)
            )
            calibration = load_vr_transform(VR_TRANSFORM_PATH)
            mapping = self.runtime.policy.vr_mapping
            mapper = VRWristMapper(
                pos_scale=mapping.pos_scale,
                rot_scale=mapping.rot_scale,
                vr_to_robot_rot=calibration.transform,
                max_delta_rot_rad=mapping.max_delta_rot_rad,
                base_to_world_rot=np.eye(3),
            )
            self.controller = TeleopController(
                planner, mapper, self.runtime, _build_hand_retargeter(self.config)
            )
            self.keyboard.start()
            signal.signal(
                signal.SIGTERM, lambda *_: setattr(self.shared.is_running, "value", False)
            )
            home_result = home_hand(
                self.shared, self.runtime, abort_requested=self._home_abort_requested
            )
            if not home_result.ok:
                raise RuntimeError(f"startup hand home failed: {home_result.reason}")
            self.shared.policy_ready.set()
            print(
                "B begin | C pause/resume | S save | D discard | H home | Q quit | ESC emergency stop",
                flush=True,
            )
            while self.shared.is_running.value and not self.shared.quit_requested.value:
                if not self.keyboard.healthy:
                    self.shared.estop_request.value = True
                if self.shared.estop_request.value or self.shared.error_state.value:
                    self._stop_episode(True, "hardware_failure", abnormal=True)
                    break
                if self.recorder is not None:
                    result = (
                        self.recorder.join_stop()
                        if self.recorder.stop_pending
                        else self.recorder.poll_stop()
                    )
                    if not self.recorder.is_recording and self.active:
                        self._stop_episode(
                            True, result.reason if result is not None else "recording_unavailable"
                        )
                for cmd in self.keyboard.poll(timeout=0.005):
                    if self._handle_operator_command(cmd):
                        break
                if self.quit_pending and time.monotonic() >= self.quit_deadline:
                    self._stop_episode(True, "quit_decision_timeout", abnormal=True)
                    self.shared.quit_requested.value = True
                if not self.active or self.shared.quit_requested.value:
                    continue
                tick_started = time.monotonic()
                if not self.paused and tick_started < self.next_tick:
                    continue
                row = read_observation(
                    self.shared,
                    self.runtime,
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
                            self._stop_episode(True, "camera_unavailable", abnormal=True)
                            raise RuntimeError("required recording camera unavailable")
                    if not self.paused:
                        self._pause(False)
                    continue
                if self.paused:
                    self._try_resume(row)
                    continue
                self._execute_control_step(row, tick_started)
        except Exception as exc:
            failure = exc
            self.shared.is_running.value = False
            revoke_motion(self.shared)
            logger.exception("teleop failed")
        finally:
            revoke_motion(self.shared)
            try:
                if self.recorder is not None:
                    if self.recorder.is_recording:
                        self.recorder.technical_status = "invalid"
                        self.recorder.stop_episode(save=True, reason="interrupted")
                    self.recorder.join_stop()
            except Exception as exc:
                if failure is None:
                    failure = exc
                logger.exception("teleop recording cleanup failed")
            for close in (self.keyboard.stop, self.audio.close):
                try:
                    close()
                except Exception as exc:
                    if failure is None:
                        failure = exc
                    logger.exception("teleop resource cleanup failed")
            self.shared.policy_ready.clear()
        if failure is not None:
            raise failure


def run_teleop_worker(shared, config: TeleopConfig) -> None:
    TeleopRunner(shared, config).run()
