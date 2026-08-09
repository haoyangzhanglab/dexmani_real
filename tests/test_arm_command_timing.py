from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import Mock

import h5py
import numpy as np
import pytest

from dexmani_real.config.defaults import KeyboardTeleopParams
from dexmani_real.policy.action_protocol import ARM_COMMAND_DTYPE
from dexmani_real.recording.episode_recorder import EpisodeRecorder
from dexmani_real.robot.arm_loop import (
    ArmLoopConfig,
    _decode_joint_state_feedback,
    _parse_arm_action_metadata,
    _recover_c24_measured_hold,
    _update_state_read_watchdog,
)
from dexmani_real.robot.types import RobotAction, RobotState
from dexmani_real.shm.shared_storage import ARM_STATE_DTYPE
from dexmani_real.teleop.keyboard import (
    MotionActivityLatch,
    MotionTraceSample,
    ReleaseMotionTracer,
    eef_delta_from_keys,
)
from dexmani_real.utils.retry import RetryCounter


class _Keys:
    def __init__(self, *pressed: str) -> None:
        self.pressed = set(pressed)

    def is_pressed(self, key: str) -> bool:
        return key in self.pressed


def test_arm_worker_metadata_parser_accepts_only_fixed_command_frames() -> None:
    received_s = 123.0
    frame = np.zeros(1, dtype=ARM_COMMAND_DTYPE)
    frame["action_id"][0] = 7
    frame["created_monotonic_ns"][0] = 122_500_000_000
    frame["is_hold"][0] = 1
    assert _parse_arm_action_metadata(frame, received_s) == (7, 122.5, True)
    assert _parse_arm_action_metadata({"command_seq": 7}, received_s) == (0, received_s, False)


def test_arm_worker_metadata_parser_bounds_invalid_fixed_timestamp() -> None:
    received_s = 123.0
    frame = np.zeros(1, dtype=ARM_COMMAND_DTYPE)
    frame["action_id"][0] = 8
    frame["created_monotonic_ns"][0] = 124_000_000_000
    assert _parse_arm_action_metadata(frame, received_s) == (8, received_s, False)


def test_arm_state_dtype_contains_end_to_end_timing_fields() -> None:
    assert ARM_STATE_DTYPE.names[-12:] == (
        "last_cmd_seq",
        "last_cmd_created_s",
        "last_cmd_received_s",
        "last_cmd_applied_s",
        "last_cmd_queue_latency_s",
        "last_cmd_apply_latency_s",
        "last_cmd_sdk_duration_s",
        "last_cmd_is_hold",
        "source_monotonic_ns",
        "publish_monotonic_ns",
        "state_valid",
        "timestamp",
    )


def test_c24_recovery_sends_exactly_one_fresh_measured_hold() -> None:
    arm = Mock()
    arm.clean_error.return_value = 0
    arm.clean_warn.return_value = 0
    arm.set_mode.return_value = 0
    arm.set_state.return_value = 0
    arm.set_servo_angle.return_value = 0
    measured = np.linspace(-0.3, 0.3, 7)
    arm.get_joint_states.return_value = (0, [measured.copy()])

    recovered = _recover_c24_measured_hold(arm, ArmLoopConfig())

    np.testing.assert_allclose(recovered, measured)
    arm.get_joint_states.assert_called_once_with(is_radian=True, num=1)
    arm.set_servo_angle.assert_called_once()
    np.testing.assert_allclose(arm.set_servo_angle.call_args.kwargs["angle"], measured)


def test_c24_recovery_fails_before_sending_an_invalid_measurement() -> None:
    arm = Mock()
    arm.clean_error.return_value = 0
    arm.clean_warn.return_value = 0
    arm.set_mode.return_value = 0
    arm.set_state.return_value = 0
    arm.get_joint_states.return_value = (0, [np.full(7, np.nan)])

    with pytest.raises(RuntimeError, match="measured hold is invalid"):
        _recover_c24_measured_hold(arm, ArmLoopConfig())

    arm.set_servo_angle.assert_not_called()


def test_arm_feedback_boundary_and_watchdog_reject_persistent_bad_reads() -> None:
    qpos, qvel, tau = _decode_joint_state_feedback(
        0,
        [np.zeros(7), np.ones(7), np.full(7, 2.0)],
    )
    np.testing.assert_array_equal(qpos, np.zeros(7))
    np.testing.assert_array_equal(qvel, np.ones(7))
    np.testing.assert_array_equal(tau, np.full(7, 2.0))

    with pytest.raises(RuntimeError, match="SDK code"):
        _decode_joint_state_feedback(1, [])
    with pytest.raises(RuntimeError, match="invalid qpos"):
        _decode_joint_state_feedback(0, [np.zeros(6)])
    with pytest.raises(RuntimeError, match="invalid qvel"):
        _decode_joint_state_feedback(0, [np.zeros(7), np.full(7, np.nan)])

    watchdog = RetryCounter(max_consecutive=2, label="arm_state")
    assert not _update_state_read_watchdog(watchdog, succeeded=False)
    assert _update_state_read_watchdog(watchdog, succeeded=False)
    assert not _update_state_read_watchdog(watchdog, succeeded=True)
    assert watchdog.count == 0


def test_motion_latch_emits_one_release_edge() -> None:
    latch = MotionActivityLatch()

    assert latch.update(False) is False
    assert latch.update(True) is False
    assert latch.update(True) is False
    assert latch.update(False) is True
    assert latch.update(False) is False
    latch.update(True)
    latch.reset()
    assert latch.update(False) is False


def test_keyboard_translation_default_is_eight_mm() -> None:
    cfg = KeyboardTeleopParams()
    dx, drpy = eef_delta_from_keys(_Keys("w"), cfg.delta_pos_m, cfg.delta_rpy_rad)

    np.testing.assert_allclose(dx, [0.008, 0.0, 0.0])
    np.testing.assert_array_equal(drpy, np.zeros(3))


def test_keyboard_config_rejects_nonpositive_step() -> None:
    with pytest.raises(ValueError):
        KeyboardTeleopParams(delta_pos_m=0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"release_trace_pre_frames": 0},
        {"release_trace_post_frames": 0},
        {"release_trace_cooldown_s": -1.0},
    ],
)
def test_keyboard_config_rejects_invalid_release_trace(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        KeyboardTeleopParams(**kwargs)


def _trace_sample(frame: int, x_m: float, *, active: bool, y_m: float = 0.0) -> MotionTraceSample:
    return MotionTraceSample(
        frame=frame,
        timestamp_s=frame / 30.0,
        input_active=active,
        eef_pos_m=np.array([x_m, y_m, 0.2]),
        command_pos_m=np.array([0.10, 0.0, 0.2]),
        qpos_error_rad=0.1,
        qvel_peak_rad_s=0.2,
        state_age_s=0.03,
        queue_latency_s=0.02,
        apply_latency_s=0.021,
    )


def test_release_motion_trace_reports_rollback_lateral_motion_and_reversal() -> None:
    tracer = ReleaseMotionTracer(pre_frames=3, post_frames=4, cooldown_s=0.0)
    direction = np.array([1.0, 0.0, 0.0])
    for frame, x_m in enumerate([0.000, 0.008, 0.016]):
        assert tracer.observe(_trace_sample(frame, x_m, active=True), translation_direction=direction) == []

    start_lines = tracer.observe(
        _trace_sample(3, 0.024, active=False),
        release_edge=True,
        translation_direction=direction,
    )
    assert start_lines == []  # capture is buffered to avoid perturbing the measured control window

    post_positions = [(0.028, 0.0), (0.030, 0.002), (0.029, 0.001), (0.029, 0.001)]
    final_lines: list[str] = []
    for frame, (x_m, y_m) in enumerate(post_positions, start=4):
        final_lines = tracer.observe(
            _trace_sample(frame, x_m, active=False, y_m=y_m),
            translation_direction=direction,
        )

    assert len(final_lines) == 10  # header + 3 pre + release + 4 post + summary
    assert "START" in final_lines[0]
    assert "buffered=1" in final_lines[0]
    assert "PRE-3" in final_lines[1]
    assert " REL " in final_lines[4]
    assert "SUMMARY" in final_lines[-1]
    assert tracer.last_summary is not None
    assert tracer.last_summary["clean"] is True
    assert tracer.last_summary["peak_forward_m"] == pytest.approx(0.006)
    assert tracer.last_summary["rollback_m"] == pytest.approx(0.001)
    assert tracer.last_summary["peak_lateral_m"] == pytest.approx(0.002)
    assert tracer.last_summary["direction_reversals"] == 1


def test_release_motion_trace_marks_reengagement_as_contaminated() -> None:
    tracer = ReleaseMotionTracer(pre_frames=2, post_frames=5, cooldown_s=0.0)
    direction = np.array([1.0, 0.0, 0.0])
    tracer.observe(_trace_sample(0, 0.0, active=True), translation_direction=direction)
    tracer.observe(
        _trace_sample(1, 0.008, active=False),
        release_edge=True,
        translation_direction=direction,
    )

    lines = tracer.observe(_trace_sample(2, 0.012, active=True), translation_direction=direction)

    assert len(lines) == 1  # no per-frame output in a newly active control interval
    assert "reason=reengaged" in lines[-1]
    assert tracer.last_summary is not None
    assert tracer.last_summary["clean"] is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pre_frames": 0},
        {"post_frames": 0},
        {"cooldown_s": -1.0},
        {"velocity_deadband_m_s": -1.0},
    ],
)
def test_release_motion_trace_rejects_invalid_config(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ReleaseMotionTracer(**kwargs)


def test_episode_recorder_persists_arm_command_timing(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(str(tmp_path), max_frames=4, control_hz=16.0, min_frames=1)
    assert recorder.start_episode(task_label="timing-test")
    state = RobotState(
        arm_qpos=np.zeros(7),
        arm_qvel=np.zeros(7),
        arm_tau=np.zeros(7),
        eef_pos=np.zeros(3),
        eef_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        eef_rot6d=np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        hand_qpos=np.zeros(12),
        hand_tactile_sum=np.zeros((5, 3)),
        hand_tactile_force=np.zeros((5, 120, 3)),
        hand_tactile_contact=np.zeros(5, dtype=bool),
        hand_tipboard_err=np.zeros(12, dtype=np.int32),
        hand_commboard_err=np.zeros(12, dtype=np.int32),
        hand_jointboard_err=np.zeros(12, dtype=np.int32),
        hand_qpos_stale=False,
        fingertip_pos=np.zeros((5, 3)),
        arm_connected=True,
        hand_connected=False,
        timestamp=time.perf_counter(),
        arm_last_cmd_seq=42,
        arm_last_cmd_queue_latency_s=0.004,
        arm_last_cmd_apply_latency_s=0.006,
        arm_last_cmd_sdk_duration_s=0.002,
        arm_last_cmd_is_hold=True,
    )
    action = RobotAction(arm_qpos_cmd=np.zeros(7), hand_qpos_cmd=np.zeros(12))
    vr_frame = {
        "wrist_pos": np.zeros(3),
        "wrist_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
        "landmarks": np.zeros((21, 3)),
    }

    assert recorder.add_frame(state, action, vr_frame)
    episode_path = recorder.stop_episode(success=True)
    assert episode_path is not None
    assert recorder.join_stop(timeout=5.0)

    with h5py.File(Path(episode_path) / "data.h5", "r") as h5_file:
        meta = h5_file["meta"].attrs
        assert int(meta["schema_version"]) == 15
        assert float(meta["fps"]) == pytest.approx(16.0)
        assert float(meta["grid_dt_s"]) == pytest.approx(1.0 / 16.0)
        assert float(meta["grid_duration_s"]) == pytest.approx(0.0)
        assert float(meta["duration"]) == pytest.approx(float(meta["wall_duration_s"]))
        assert float(meta["non_sampled_duration_s"]) >= 0.0
        assert bool(h5_file["flag_sample_valid"][0]) is True
        assert int(h5_file["arm_last_cmd_seq"][0]) == 42
        assert float(h5_file["arm_last_cmd_queue_latency_s"][0]) == pytest.approx(0.004)
        assert float(h5_file["arm_last_cmd_apply_latency_s"][0]) == pytest.approx(0.006)
        assert float(h5_file["arm_last_cmd_sdk_duration_s"][0]) == pytest.approx(0.002)
        assert bool(h5_file["arm_last_cmd_is_hold"][0]) is True
        assert h5_file["target_eef_pos_raw"].shape == (1, 3)
        assert h5_file["action_hand_joint_raw"].shape == (1, 12)
        assert np.isnan(h5_file["policy_compute_time_ms"][0])
