"""Operator-supervised teleop with one current observation per real control step."""
from pathlib import Path
import signal
import time
import numpy as np

from dexmani_real.planning import OnlineIKConfig, XArm7MotionPlanner
from dexmani_real.robot.arm_homing import build_policy_home_planner, home_policy_robot
from dexmani_real.robot.hand_homing import home_hand
from dexmani_real.recording.client import RecorderClient
from dexmani_real.runtime.observation import read_observation
from dexmani_real.runtime.operator_input import KeyboardInput, OperatorCommand
from dexmani_real.runtime.safety import begin_motion, revoke_motion
from dexmani_real.teleop.audio_feedback import AudioFeedback
from dexmani_real.teleop.config import TeleopConfig
from dexmani_real.teleop.control_loop.grid import TeleopController, run_control_grid_tick
from dexmani_real.teleop.control_loop.vr_mapping import VRWristMapper
from dexmani_real.teleop.retargeting.retargeter import TAGHandRetargeter, DexPilotHandRetargeter
from dexmani_real.teleop.vr_transform import load_vr_transform
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)
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


def teleop_loop(shared, config):
    runtime = config.runtime
    recorder = RecorderClient(shared) if runtime.policy.recording_enabled else None
    keyboard = KeyboardInput(estop_callback=lambda: setattr(shared.estop_request, "value", True))
    audio = AudioFeedback()
    controller = None
    active = False
    paused = False
    resume_requested = False
    pause_ns = 0
    quit_pending = False
    quit_deadline = 0.0
    next_tick = 0.0
    failures = 0
    def stop(save, reason, incomplete=False):
        nonlocal active, paused, resume_requested
        revoke_motion(shared)
        if controller is not None:
            controller.clear_reference()
        active = paused = resume_requested = False
        if recorder is not None and recorder.is_recording:
            if incomplete:
                recorder.technical_status = "invalid"
            recorder.stop_episode(save=save, reason=reason, retain_partial=incomplete)
        shared.is_recording.value = False
    def pause(manual, mark_episode=True):
        nonlocal paused, resume_requested, pause_ns
        revoke_motion(shared)
        pause_ns = time.monotonic_ns()
        if controller is not None:
            controller.clear_reference()
        if mark_episode and recorder is not None and recorder.is_recording:
            recorder.had_pause = True
        paused, resume_requested = True, not manual
        audio.play("pause")
    def abort_home():
        for cmd in keyboard.poll(timeout=0):
            if cmd is OperatorCommand.EMERGENCY_STOP:
                shared.estop_request.value = True
            elif cmd is OperatorCommand.QUIT:
                shared.quit_requested.value = True
            elif cmd in (OperatorCommand.STOP, OperatorCommand.DISCARD):
                return True
        return bool(shared.estop_request.value or shared.quit_requested.value or not shared.is_running.value)
    try:
        planner = XArm7MotionPlanner.create_default(teleop_profile=OnlineIKConfig(
            max_pose_error_pos_m=runtime.policy.ik_max_pose_error_pos_m,
            max_pose_error_rot_rad=runtime.policy.ik_max_pose_error_rot_rad,
            nullspace_step_size_deg=runtime.policy.ik_nullspace_step_rate_deg_s/runtime.teleop.control_hz))
        calibration = load_vr_transform(Path(__file__).resolve().parents[2] / config.vr_transform_path)
        mapping = runtime.policy.vr_mapping
        mapper = VRWristMapper(pos_scale=mapping.pos_scale, rot_scale=mapping.rot_scale,
            vr_to_robot_rot=calibration.transform, max_delta_rot_rad=mapping.max_delta_rot_rad,
            base_to_world_rot=np.eye(3))
        controller = TeleopController(planner, mapper, runtime, _build_hand_retargeter(config))
        keyboard.start()
        signal.signal(signal.SIGTERM, lambda *_: setattr(shared.is_running, "value", False))
        home_result = home_hand(shared, runtime, abort_requested=abort_home)
        if not home_result.ok:
            raise RuntimeError(f"startup hand home failed: {home_result.reason}")
        shared.policy_ready.set()
        print("B begin | C pause/resume | S save | D discard | H home | Q quit | ESC emergency stop", flush=True)
        home_planner = None
        while shared.is_running.value and not shared.quit_requested.value:
            if not keyboard.healthy:
                shared.estop_request.value = True
            if shared.estop_request.value or shared.error_state.value:
                stop(True, "hardware_failure", incomplete=True)
                break
            if recorder is not None:
                result = recorder.poll_stop()
                if result.error:
                    shared.workflow_failed.value = True
                if not recorder.is_recording and active:
                    stop(True, result.reason or "recording_complete")
            if shared.workflow_failed.value and active:
                stop(True, "required_recording_resource_failed", incomplete=True)
            for cmd in keyboard.poll(timeout=0.005):
                if cmd is OperatorCommand.EMERGENCY_STOP:
                    shared.estop_request.value = True
                    break
                if cmd is OperatorCommand.QUIT:
                    revoke_motion(shared)
                    if recorder is not None and recorder.is_recording:
                        pause(True, mark_episode=False)
                        quit_pending = True
                        quit_deadline = time.monotonic() + runtime.policy.quit_save_timeout_s
                        print("Quit: S save, D discard, H save and home", flush=True)
                    else:
                        shared.quit_requested.value = True
                elif cmd in (OperatorCommand.STOP, OperatorCommand.DISCARD, OperatorCommand.HOME):
                    stop(cmd is not OperatorCommand.DISCARD, cmd.value.lower())
                    audio.play("discard" if cmd is OperatorCommand.DISCARD else "end")
                    if recorder is not None:
                        result = recorder.join_stop()
                        if result.error:
                            shared.workflow_failed.value = True
                    if cmd is OperatorCommand.HOME and not shared.error_state.value:
                        home_planner = home_planner or build_policy_home_planner(runtime)
                        audio.play("home")
                        if home_policy_robot(shared, runtime, home_planner, abort_requested=abort_home):
                            audio.queue("home_done")
                    if quit_pending:
                        shared.quit_requested.value = True
                elif cmd is OperatorCommand.PAUSE and active and not quit_pending:
                    if paused:
                        resume_requested = True
                    else:
                        pause(True)
                elif cmd is OperatorCommand.BEGIN and not active and not quit_pending:
                    if shared.workflow_failed.value or (recorder is not None and recorder.stop_pending):
                        continue
                    row = read_observation(shared, runtime, require_hand=runtime.policy.hand_enabled,
                                           require_camera=recorder is not None, require_vr=True)
                    if row is None:
                        print("Begin requires fresh robot, VR and recording resources", flush=True)
                        continue
                    if recorder is not None and not recorder.start_episode(task_label=config.task_label, operator=config.operator):
                        continue
                    # START may block on disk; anchor only after it finishes.
                    row = read_observation(shared, runtime, require_hand=runtime.policy.hand_enabled,
                                           require_camera=recorder is not None, require_vr=True)
                    if row is None or not controller.reset_reference(row) or not begin_motion(shared):
                        stop(False, "begin_unavailable")
                        continue
                    active = True
                    shared.is_recording.value = recorder is not None
                    next_tick = time.monotonic()
                    audio.play("begin")
            if quit_pending and time.monotonic() >= quit_deadline:
                stop(True, "quit_decision_timeout", incomplete=True)
                shared.quit_requested.value = True
            if not active or shared.quit_requested.value:
                continue
            if shared.workflow_failed.value:
                continue
            tick_started = time.monotonic()
            if not paused and tick_started < next_tick:
                continue
            row = read_observation(shared, runtime, require_hand=runtime.policy.hand_enabled,
                                   require_camera=recorder is not None, require_vr=True)
            if row is None:
                if recorder is not None:
                    from dexmani_real.runtime.observation import read_camera_frame, sample_is_fresh
                    camera = read_camera_frame(shared)
                    if camera is None or not sample_is_fresh(camera["timestamp_ns"], runtime.camera.max_frame_age_s):
                        shared.workflow_failed.value = True
                        stop(True, "camera_unavailable", incomplete=True)
                        continue
                if not paused:
                    pause(False)
                continue
            if paused:
                if resume_requested and all(int(stamp) > pause_ns for stamp in (
                        row.arm["timestamp_ns"][0], row.vr["recv_ts_ns"],
                        row.hand["timestamp_ns"][0] if row.hand is not None else row.observation_timestamp_ns)):
                    if controller.reset_reference(row) and begin_motion(shared):
                        paused = resume_requested = False
                        failures = 0
                        next_tick = time.monotonic()
                        audio.play("resume")
                continue
            status = run_control_grid_tick(controller, shared, row, recorder)
            next_tick = tick_started + 1/runtime.teleop.control_hz
            failures = failures + 1 if status else 0
            if failures >= runtime.policy.max_consecutive_errors:
                pause(True)
    except Exception:
        shared.workflow_failed.value = True
        revoke_motion(shared)
        logger.exception("teleop failed")
        raise
    finally:
        revoke_motion(shared)
        if recorder is not None:
            if recorder.is_recording:
                recorder.technical_status = "invalid"
                recorder.stop_episode(save=True, reason="interrupted", retain_partial=True)
            recorder.join_stop()
        keyboard.stop()
        audio.close()
