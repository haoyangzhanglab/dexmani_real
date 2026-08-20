"""Operator signals and bounded recording decisions for teleoperation."""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from dexmani_real.planning import XArm7MotionPlanner
from dexmani_real.recording.recorder_client import RecorderClient, RecorderPhase
from dexmani_real.robot.safety import SafetyState, transition
from dexmani_real.shm.causal_reader import read_hand_state_causal, read_vr_frame_causal
from dexmani_real.shm.shared_storage import SharedStorage
from dexmani_real.teleop.audio_feedback import AudioFeedback
from dexmani_real.teleop.camera_freshness import CameraFreshnessTracker
from dexmani_real.teleop.config import TeleopConfig
from dexmani_real.teleop.control_grid import TeleopControlResources
from dexmani_real.teleop.control_state import (
    CommandQuiescence,
    CoordinatorDirective,
    TeleopLoopState,
)
from dexmani_real.teleop.episode_samples import _stop_recording
from dexmani_real.teleop.hand_control import _seed_hand_retargeter
from dexmani_real.teleop.hand_retarget import (
    TAGHandRetargeter,
    XHandRetargeter,
    _tag_config_with_urdf,
)
from dexmani_real.teleop.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.teleop.recording_session import (
    QuitRecordingDecision,
    await_quit_recording_decision,
)
from dexmani_real.teleop.safety import (
    _do_configured_teleop_home,
    _enter_command_quiescence_impl,
    _hand_feedback_issue_impl,
)
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager

logger = get_logger(__name__)


@dataclass(frozen=True)
class TeleopOperatorResources:
    """Resources used only while applying operator control signals."""

    control: TeleopControlResources
    keyboard: KeyboardHandler
    audio: AudioFeedback
    limiter: RateManager
    quiescence: CommandQuiescence
    camera_freshness: CameraFreshnessTracker


def _try_init_hand_retargeter_impl(ctx: TeleopLoopState, cfg: TeleopConfig) -> bool:
    """Lazily initialize ctx.hand_retargeter if not already created."""
    if ctx.hand_retargeter is not None:
        return True
    try:
        if cfg.runtime.policy.hand_retargeting_type == "tag":
            ctx.hand_retargeter = TAGHandRetargeter(
                hand_type="right",
                fingertip_link_names=cfg.runtime.hand.fingertip_link_names,
                tag_config=_tag_config_with_urdf(
                    cfg.runtime.tag_retargeting, cfg.hand_urdf_path
                ),
            )
        else:
            ctx.hand_retargeter = XHandRetargeter(
                hand_type="right",
                retargeting_type=cfg.runtime.policy.hand_retargeting_type,
                dexpilot_config=cfg.runtime.dexpilot_retargeting,
            )
        logger.info(
            "Hand retargeter ready (type=%s)", cfg.runtime.policy.hand_retargeting_type
        )
        return True
    except Exception:
        logger.error("Hand retargeter initialization failed", exc_info=True)
        ctx.hand_retargeter = None
        return False


def _init_and_seed_hand_retargeter_impl(
    ctx: TeleopLoopState, cfg: TeleopConfig, shared: SharedStorage
) -> np.ndarray | None:
    """Lazy-init retargeter and seed NLP warm-start from hardware qpos.

    Returns the seeded qpos (for updating ``ctx.prev_hand_qpos``) or None.
    """
    if not cfg.runtime.policy.hand_enabled:
        return None
    if not _try_init_hand_retargeter_impl(ctx, cfg):
        return None
    hs = read_hand_state_causal(shared)
    qpos = (
        hs["qpos"][0]
        if _hand_feedback_issue_impl(cfg, hs) is None and hs is not None
        else None
    )
    return _seed_hand_retargeter(ctx.hand_retargeter, qpos)


def _transition_or_fault(
    shared: SharedStorage,
    new_state: SafetyState,
    reason: str,
) -> bool:
    """Apply one safety transition and make any rejection a sticky fault."""
    if transition(shared, new_state):
        return True
    logger.error(
        "teleop_loop: safety transition to %s failed during %s",
        new_state.name,
        reason,
    )
    shared.error_state.value = True
    return False


def _enter_operator_quiescence(
    ctx: TeleopLoopState,
    shared: SharedStorage,
    resources: TeleopOperatorResources,
    reason: str,
    *,
    start_new_run: bool = False,
    replace_existing_reason: bool = False,
) -> None:
    _enter_command_quiescence_impl(
        ctx,
        shared,
        resources.quiescence,
        resources.control.arm_mapper,
        reason,
        start_new_run=start_new_run,
        replace_existing_reason=replace_existing_reason,
    )


def _handoff_quiescence_to_home(resources: TeleopOperatorResources) -> None:
    reason, _entered_ns = resources.quiescence.clear()
    if reason is not None:
        logger.info(
            "teleop_loop: homing supersedes %s command quiescence",
            reason,
        )


def _keyboard_estop_requested(keyboard: KeyboardHandler) -> bool:
    return keyboard.estop_latched or not keyboard.healthy


def _apply_quit_signal(
    ctx: TeleopLoopState,
    shared: SharedStorage,
    cfg: TeleopConfig,
    resources: TeleopOperatorResources,
) -> CoordinatorDirective:
    """Enter the bounded post-teleop state after an operator quit request."""
    recorder = resources.control.recorder
    print("\nQ: 退出")
    resources.audio.play("quit")
    _enter_operator_quiescence(
        ctx,
        shared,
        resources,
        "quit",
        replace_existing_reason=True,
    )
    ctx.teleop_active = False
    if not _transition_or_fault(shared, SafetyState.ARMED, "quit"):
        return CoordinatorDirective.BREAK

    if ctx.recording_active:
        resources.audio.queue("quit_save_prompt")
        print(
            "  [S] 保存并退出  [D] 丢弃并退出  [H] 保存并归位 "
            f"({cfg.runtime.policy.quit_save_timeout_s:.0f}s 超时默认丢弃)"
        )
        decision = await_quit_recording_decision(
            shared,
            resources.keyboard,
            timeout_s=cfg.runtime.policy.quit_save_timeout_s,
        )
        save = decision in (
            QuitRecordingDecision.SAVE,
            QuitRecordingDecision.SAVE_AND_HOME,
        )
        if decision is QuitRecordingDecision.ESTOP:
            resources.audio.play("emergency")
        else:
            resources.audio.play("save" if save else "discard")
        _stop_recording(
            recorder,
            ctx.recording_active,
            save=save,
            shared=shared,
        )
        ctx.recording_active = False
        if decision is QuitRecordingDecision.TIMEOUT:
            print("  超时，默认丢弃请求已提交")
        elif decision is QuitRecordingDecision.DISCARD:
            print("  丢弃请求已提交")
        elif save:
            print("  保存请求已提交")

        if decision is QuitRecordingDecision.SAVE_AND_HOME and shared.is_running.value:
            assert ctx.prev_hand_qpos is not None
            resources.audio.play("home")
            ctx.ema_prev_pos = ctx.ema_prev_quat = None
            _handoff_quiescence_to_home(resources)
            ctx.prev_hand_qpos = _do_configured_teleop_home(
                shared,
                cfg,
                hand_available=ctx.hand_available,
                prev_hand_qpos=ctx.prev_hand_qpos,
                planner=resources.control.planner,
                audio=resources.audio,
                estop_requested=lambda: _keyboard_estop_requested(resources.keyboard),
                arm_mapper=resources.control.arm_mapper,
                hand_retargeter=ctx.hand_retargeter,
            )

    ctx.quit_pending = True
    ctx.post_teleop_deadline_s = (
        time.perf_counter() + cfg.runtime.policy.post_teleop_timeout_s
    )
    print(
        f"\n[H] return_home  [Q] quit  ({cfg.runtime.policy.post_teleop_timeout_s:.0f}s timeout)",
        flush=True,
    )
    return CoordinatorDirective.CONTINUE


def _apply_home_signal(
    ctx: TeleopLoopState,
    shared: SharedStorage,
    cfg: TeleopConfig,
    resources: TeleopOperatorResources,
) -> CoordinatorDirective:
    """Stop the current session and execute the configured home operation."""
    print("\nH: return_home")
    resources.audio.play("home")
    _stop_recording(
        resources.control.recorder,
        ctx.recording_active,
        save=True,
        shared=shared,
    )
    ctx.recording_active = False
    ctx.teleop_active = False
    if not _transition_or_fault(shared, SafetyState.ARMED, "home"):
        return CoordinatorDirective.BREAK
    ctx.ema_prev_pos = ctx.ema_prev_quat = None
    _handoff_quiescence_to_home(resources)
    assert ctx.prev_hand_qpos is not None
    ctx.prev_hand_qpos = _do_configured_teleop_home(
        shared,
        cfg,
        hand_available=ctx.hand_available,
        prev_hand_qpos=ctx.prev_hand_qpos,
        planner=resources.control.planner,
        audio=resources.audio,
        estop_requested=lambda: _keyboard_estop_requested(resources.keyboard),
        arm_mapper=resources.control.arm_mapper,
        hand_retargeter=ctx.hand_retargeter,
    )
    resources.keyboard.drain_signal(ControlSignal.HOME)
    resources.limiter.reset()
    return CoordinatorDirective.CONTINUE


def _apply_pause_signal(
    ctx: TeleopLoopState,
    shared: SharedStorage,
    resources: TeleopOperatorResources,
) -> bool:
    """Pause or resume one existing session; return false on a safety fault."""
    if ctx.teleop_active:
        _enter_operator_quiescence(
            ctx,
            shared,
            resources,
            "pause",
            replace_existing_reason=True,
        )
        ctx.teleop_active = False
        if not _transition_or_fault(shared, SafetyState.ARMED, "pause"):
            return False
    else:
        if resources.quiescence.reason != "pause":
            print("\nC: 没有可恢复的暂停 session — 请按 B 开始新的遥操作 session")
            return True
        if shared.safety_state.value != SafetyState.ARMED:
            print(
                f"\nC: safety_state={shared.safety_state.value} — must be ARMED to resume"
            )
            return True
        if not _transition_or_fault(shared, SafetyState.RUNNING, "resume"):
            return False
        ctx.teleop_active = True
    state_str = "恢复" if ctx.teleop_active else "暂停"
    print(f"\nC: {state_str}遥操作")
    resources.audio.play("resume" if ctx.teleop_active else "pause")
    return True


def _apply_begin_signal(
    ctx: TeleopLoopState,
    shared: SharedStorage,
    cfg: TeleopConfig,
    resources: TeleopOperatorResources,
) -> bool:
    """Start a new run and optional recording transaction."""
    if ctx.teleop_active or ctx.recording_active:
        print(
            "\nB: session already active — use C to pause/resume, S to save, or D to discard"
        )
        return True
    if shared.safety_state.value != SafetyState.ARMED:
        print(
            f"\nB: safety_state={shared.safety_state.value} — must be ARMED({SafetyState.ARMED})"
        )
        return True
    if read_vr_frame_causal(shared) is None:
        print("\nB: 无 VR 帧，无法开始遥操作")
        return True
    begin_hand_state = (
        read_hand_state_causal(shared) if cfg.runtime.policy.hand_enabled else None
    )
    begin_hand_issue = _hand_feedback_issue_impl(cfg, begin_hand_state)
    if begin_hand_issue is not None:
        print(f"\nB: hand feedback unhealthy ({begin_hand_issue}) — cannot begin")
        return True

    recorder = resources.control.recorder
    gc.collect()
    if recorder is None:
        ctx.recording_active = False
        shared.is_recording.value = False
        begin_reason = "begin"
        begin_message = "\nB: 遥操作开始（未启用录制 capability）"
    else:
        if not recorder.start_episode(
            task_label=cfg.task_label,
            operator=cfg.operator,
        ):
            print("  ⚠ 无法开始录制")
            return True
        ctx.recording_active = True
        resources.camera_freshness.reset(time.monotonic())
        shared.is_recording.value = True
        begin_reason = "begin recording"
        begin_message = f"\nB: 遥操作+录制开始  episode={recorder.frame_count}"

    resources.keyboard.drain_signal(ControlSignal.BEGIN)
    if not _transition_or_fault(shared, SafetyState.RUNNING, begin_reason):
        _stop_recording(
            recorder,
            ctx.recording_active,
            save=False,
            shared=shared,
            reason="safety_transition_failed",
        )
        ctx.recording_active = False
        return False
    _enter_operator_quiescence(ctx, shared, resources, "begin", start_new_run=True)
    ctx.teleop_active = True
    logger.debug("teleop_loop: RUNNING")
    seeded_qpos = _init_and_seed_hand_retargeter_impl(ctx, cfg, shared)
    if seeded_qpos is not None:
        ctx.prev_hand_qpos = seeded_qpos
    resources.audio.play("begin")
    control_period_s = 1.0 / cfg.runtime.policy.control_hz
    ctx.begin_audio_gate_deadline_s = time.monotonic() + max(
        0.0,
        cfg.runtime.policy.begin_motion_gate_timeout_s - control_period_s,
    )
    ctx.ignore_begin_audio_until_silent = False
    print(begin_message)
    resources.limiter.reset()
    return True


def _apply_operator_controls(
    ctx: TeleopLoopState,
    shared: SharedStorage,
    cfg: TeleopConfig,
    resources: TeleopOperatorResources,
    controls: tuple[ControlSignal, ...],
) -> CoordinatorDirective:
    """Apply queued controls while preserving their original ordering semantics."""
    skip_control_tick = False
    for control in controls:
        if control is ControlSignal.EMERGENCY_STOP:
            print("\nESC: emergency_stop")
            resources.audio.play("emergency")
            shared.estop_request.value = True
            _stop_recording(
                resources.control.recorder,
                ctx.recording_active,
                save=False,
                shared=shared,
            )
            ctx.recording_active = False
            return CoordinatorDirective.BREAK
        if control is ControlSignal.QUIT:
            return _apply_quit_signal(ctx, shared, cfg, resources)
        if control is ControlSignal.HOME:
            return _apply_home_signal(ctx, shared, cfg, resources)
        if control in (ControlSignal.STOP, ControlSignal.DISCARD):
            save_episode = control is ControlSignal.STOP
            stop_reason = "stop" if save_episode else "discard"
            print("\nS: 停止录制" if save_episode else "\nD: 丢弃录制")
            resources.audio.play("save" if save_episode else "discard")
            _enter_operator_quiescence(
                ctx,
                shared,
                resources,
                stop_reason,
                replace_existing_reason=True,
            )
            _stop_recording(
                resources.control.recorder,
                ctx.recording_active,
                save=save_episode,
                shared=shared,
            )
            ctx.recording_active = False
            ctx.teleop_active = False
            if not _transition_or_fault(shared, SafetyState.ARMED, stop_reason):
                return CoordinatorDirective.BREAK
            skip_control_tick = True
        elif control is ControlSignal.PAUSE:
            if not _apply_pause_signal(ctx, shared, resources):
                return CoordinatorDirective.BREAK
            skip_control_tick = True
        elif control is ControlSignal.BEGIN:
            if not _apply_begin_signal(ctx, shared, cfg, resources):
                return CoordinatorDirective.BREAK
            skip_control_tick = True

    if (
        shared.estop_request.value
        or shared.quit_requested.value
        or not shared.is_running.value
        or shared.error_state.value
    ):
        return CoordinatorDirective.BREAK
    if skip_control_tick:
        return CoordinatorDirective.CONTINUE
    return CoordinatorDirective.NORMAL


def _poll_recording_lifecycle(
    ctx: TeleopLoopState,
    shared: SharedStorage,
    recorder: RecorderClient | None,
    audio: AudioFeedback,
    *,
    enter_quiescence: Callable[..., None],
    transition_or_fault: Callable[[SafetyState, str], bool],
) -> bool:
    """Poll asynchronous recorder state and handle writer failure fail-closed."""
    if recorder is not None:
        stop_result = recorder.poll_stop()
        reached_limit = (
            stop_result.phase
            in (
                RecorderPhase.FINALIZING,
                RecorderPhase.COMPLETED,
                RecorderPhase.ERROR,
            )
            and stop_result.reason == "max_frames"
            and (ctx.teleop_active or ctx.recording_active)
        )
        if reached_limit:
            enter_quiescence("max_frames", replace_existing_reason=True)
            ctx.teleop_active = False
            ctx.recording_active = False
            shared.is_recording.value = False
            if not transition_or_fault(SafetyState.ARMED, "maximum recording duration"):
                return False
            print("  已达到最大录制时长：正在自动保存，遥操作进入静默暂停")
            audio.play("pause")
        if stop_result.done:
            ctx.recording_active = False
            shared.is_recording.value = False
            if stop_result.error:
                path_label = f": {stop_result.path}" if stop_result.path else ""
                print(f"  ⚠ 录制终结失败 ({stop_result.error}){path_label}")
            elif stop_result.saved:
                print(
                    f"  录制已保存: {stop_result.path}  ({stop_result.frame_count} 帧)"
                )
                if not stop_result.min_frames_met:
                    print("  ⚠ 已保存，但未达到配置的最短质量时长")
            else:
                print(f"  录制已丢弃 ({stop_result.frame_count} 帧)")
            gc.collect()
            if ctx.quit_after_recording:
                shared.quit_requested.value = True
        elif stop_result.phase is RecorderPhase.FINALIZING and stop_result.error:
            print("  ⚠ 录制终结超过时限；仍在安全回收，本会话将标记为失败")

    writer_error = (
        recorder.camera_writer_error
        if recorder is not None and ctx.recording_active
        else None
    )
    if writer_error is None:
        return True
    logger.error(
        "Camera writer failed — discarding current episode: %s",
        writer_error,
    )
    print(f"  ⚠ 相机写盘失败，当前 episode 已废弃: {writer_error}")
    _stop_recording(
        recorder,
        ctx.recording_active,
        save=False,
        shared=shared,
        reason="camera_writer_error",
    )
    ctx.recording_active = False
    return True


def _advance_post_teleop_state(
    ctx: TeleopLoopState,
    shared: SharedStorage,
    cfg: TeleopConfig,
    kb: KeyboardHandler,
    audio: AudioFeedback,
    planner: XArm7MotionPlanner,
    limiter: RateManager,
    recorder: RecorderClient | None,
    *,
    handoff_quiescence_to_home: Callable[[], None],
    keyboard_estop_requested: Callable[[], bool],
) -> CoordinatorDirective:
    """Keep workers alive after Q for optional home and bounded recorder exit."""
    if not ctx.quit_pending:
        return CoordinatorDirective.NORMAL

    home_handled = False
    for control in kb.poll(timeout=0.1):
        if control is ControlSignal.HOME:
            if home_handled:
                continue
            home_handled = True
            print("  H: return_home")
            audio.play("home")
            handoff_quiescence_to_home()
            assert ctx.prev_hand_qpos is not None
            ctx.prev_hand_qpos = _do_configured_teleop_home(
                shared,
                cfg,
                hand_available=ctx.hand_available,
                prev_hand_qpos=ctx.prev_hand_qpos,
                planner=planner,
                audio=audio,
                estop_requested=keyboard_estop_requested,
            )
            kb.drain_signal(ControlSignal.HOME)
            limiter.reset()
            print("  [Q] quit", flush=True)
        elif control in (ControlSignal.QUIT, ControlSignal.EMERGENCY_STOP):
            if control is ControlSignal.EMERGENCY_STOP:
                shared.estop_request.value = True
            elif recorder is not None and recorder.stop_pending:
                if not ctx.quit_after_recording:
                    ctx.quit_after_recording = True
                    ctx.quit_recording_deadline_s = (
                        time.monotonic() + cfg.runtime.policy.quit_save_timeout_s
                    )
                print("  录制仍在终结；完成后自动退出", flush=True)
            else:
                shared.quit_requested.value = True
                break

    if (
        shared.estop_request.value
        or shared.quit_requested.value
        or not shared.is_running.value
    ):
        return CoordinatorDirective.BREAK

    recording_stop_pending = recorder is not None and recorder.stop_pending
    if (
        ctx.quit_after_recording
        and recording_stop_pending
        and time.monotonic() >= ctx.quit_recording_deadline_s
    ):
        print("  录制终结超时 — 退出并将本会话标记为失败")
        shared.quit_requested.value = True
        return CoordinatorDirective.BREAK

    if time.perf_counter() <= ctx.post_teleop_deadline_s:
        return CoordinatorDirective.CONTINUE
    if recording_stop_pending:
        if not ctx.quit_after_recording:
            ctx.quit_after_recording = True
            ctx.quit_recording_deadline_s = (
                time.monotonic() + cfg.runtime.policy.quit_save_timeout_s
            )
            print("  timeout — 等待录制终结后自动退出", flush=True)
        elif time.monotonic() >= ctx.quit_recording_deadline_s:
            print("  录制终结超时 — 退出并将本会话标记为失败")
            shared.quit_requested.value = True
            return CoordinatorDirective.BREAK
    else:
        print("  timeout — auto exit")
        shared.quit_requested.value = True
        return CoordinatorDirective.BREAK
    return CoordinatorDirective.CONTINUE
