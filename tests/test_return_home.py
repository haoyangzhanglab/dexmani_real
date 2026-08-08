from __future__ import annotations

import time
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.planning.path_utils import plan_joint_home_path
from dexmani_real.policy.vr_teleop_policy import _do_teleop_home
from dexmani_real.robot.arm_loop import ArmLoopConfig, _planned_homing
from dexmani_real.robot.safety import SafetyState
from dexmani_real.shm.shared_storage import HomeRequest, HomeResult, _estimate_home_timeout_s, wait_for_arm_home


class _Value:
    def __init__(self, value):
        self.value = value


def _shared() -> SimpleNamespace:
    return SimpleNamespace(
        is_running=_Value(True),
        error_state=_Value(False),
        safety_state=_Value(int(SafetyState.ARMED)),
        arm_heartbeat_s=_Value(0.0),
        policy_heartbeat_s=_Value(0.0),
        arm_home_result_q=Queue(),
    )


class _FeedbackArm:
    def __init__(self, qpos: np.ndarray, *, send_code: int = 0, read_code: int = 0):
        self.qpos = np.asarray(qpos, dtype=np.float64).copy()
        self.send_code = send_code
        self.read_code = read_code
        self.commands: list[tuple[np.ndarray, float, float]] = []

    def get_joint_states(self, *, is_radian: bool, num: int):
        if self.read_code != 0:
            return self.read_code, []
        return 0, [self.qpos.copy()]

    def set_servo_angle(self, *, angle, is_radian, speed, mvacc, wait):
        self.commands.append((np.asarray(angle).copy(), float(speed), float(mvacc)))
        if self.send_code == 0:
            self.qpos = np.asarray(angle, dtype=np.float64).copy()
        return self.send_code


def _request(waypoints: np.ndarray, *, request_id: int = 7) -> HomeRequest:
    return HomeRequest(
        request_id=request_id,
        waypoints=np.asarray(waypoints, dtype=np.float64),
        final_qpos=np.asarray(waypoints[-1], dtype=np.float64),
        execution_timeout_s=1.0,
    )


def test_planned_homing_advances_only_with_feedback_and_returns_ack() -> None:
    q0 = np.zeros(7)
    q1 = np.full(7, 0.01)
    arm = _FeedbackArm(q0)
    cfg = ArmLoopConfig(homing_step_interval_s=0.0, homing_target_timeout_s=0.1)

    result = _planned_homing(arm, _request(np.stack([q0, q1])), cfg, shared=_shared())

    assert result.success
    assert result.request_id == 7
    np.testing.assert_allclose(result.final_qpos, q1)
    assert len(arm.commands) == 2
    assert all(speed == cfg.homing_max_speed_rad_per_s for _, speed, _ in arm.commands)


def test_planned_homing_propagates_sdk_rejection_without_fabricating_home() -> None:
    q0 = np.zeros(7)
    q1 = np.full(7, 0.01)
    arm = _FeedbackArm(q0, send_code=9)

    result = _planned_homing(arm, _request(np.stack([q1])), shared=_shared())

    assert not result.success
    assert "rejected" in result.reason
    np.testing.assert_allclose(result.final_qpos, q0)


def test_planned_homing_propagates_initial_state_failure() -> None:
    result = _planned_homing(
        _FeedbackArm(np.zeros(7), read_code=4),
        _request(np.stack([np.zeros(7)])),
        shared=_shared(),
    )

    assert not result.success
    assert "initial state read failed" in result.reason
    assert np.all(np.isnan(result.final_qpos))


def test_wait_for_arm_home_requires_matching_ack() -> None:
    shared = _shared()
    home = np.zeros(7)
    shared.arm_home_result_q.put(HomeResult(1, True, "stale", home, 1.0))
    shared.arm_home_result_q.put(HomeResult(2, True, "done", home, 2.0))

    assert wait_for_arm_home(shared, home, request_id=2, timeout_s=0.2, verbose=False)


def test_home_timeout_scales_past_fixed_15_seconds_for_full_band() -> None:
    path = np.zeros((361, 7), dtype=np.float64)
    path[:, 6] = np.linspace(0.0, 2.0 * np.pi, len(path))

    assert _estimate_home_timeout_s(path) > 15.0


def test_home_path_fails_closed_when_workspace_segment_is_unsafe() -> None:
    class _Ik:
        @staticmethod
        def nearest_equivalent_qpos(home, current):
            return np.asarray(home, dtype=np.float64)

        @staticmethod
        def compute_qpos_delta(a, b):
            return np.asarray(a) - np.asarray(b)

        @staticmethod
        def check_path_collisions(path):
            return {"path_self_collision": False}

    planner = SimpleNamespace(
        ik_mgr=_Ik(),
        planning_profile=SimpleNamespace(check_self_collision=True),
        is_workspace_segment_safe=lambda start, end: False,
    )
    path = plan_joint_home_path(np.zeros(7), np.full(7, 0.1), planner)

    assert path is not None
    assert path.shape == (0, 7)


def test_policy_home_syncs_hand_and_reloads_fresh_arm_state() -> None:
    shared = _shared()
    shared.error_state = _Value(False)
    hand_home = np.full(12, 0.1)
    arm_state = np.zeros(
        1, dtype=[("qpos", "<f8", (7,)), ("connected", "u1"), ("error_code", "<i4"), ("timestamp", "<f8")]
    )
    arm_state["qpos"][0] = 0.2
    arm_state["connected"][0] = 1
    arm_state["timestamp"][0] = time.monotonic()
    planner = Mock()
    audio = Mock()

    with (
        patch("dexmani_real.policy.vr_teleop_policy.hand_home_converge", return_value=(True, hand_home)),
        patch("dexmani_real.policy.vr_teleop_policy._read_arm_state", return_value=arm_state),
        patch("dexmani_real.policy.vr_teleop_policy.send_arm_home", return_value=True) as send_home,
    ):
        result = _do_teleop_home(
            shared,
            hand_available=True,
            prev_hand_qpos=np.zeros(12),
            planner=planner,
            audio=audio,
            hand_home_qpos=hand_home,
            table_z_surface_m=0.02,
        )

    np.testing.assert_allclose(result, hand_home)
    planner.set_hand_qpos.assert_called_once()
    np.testing.assert_allclose(send_home.call_args.kwargs["current_qpos"], arm_state["qpos"][0])


def test_policy_home_fails_closed_when_hand_is_unavailable() -> None:
    shared = _shared()
    shared.error_state = _Value(False)
    with patch("dexmani_real.policy.vr_teleop_policy.send_arm_home") as send_home:
        result = _do_teleop_home(
            shared,
            hand_available=False,
            prev_hand_qpos=np.zeros(12),
            planner=Mock(),
            audio=Mock(),
            hand_home_qpos=np.zeros(12),
            table_z_surface_m=0.02,
        )
    np.testing.assert_allclose(result, np.zeros(12))
    send_home.assert_not_called()
