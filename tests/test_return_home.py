from __future__ import annotations

import time
from queue import Queue
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.planning.path_utils import plan_joint_home_path
from dexmani_real.planning.types import PathResult
from dexmani_real.policy.action_protocol import COMMIT_DTYPE, AckStatus, make_command_frame
from dexmani_real.policy.runtime import ActionCandidate
from dexmani_real.policy.vr_teleop_policy import _do_teleop_home
from dexmani_real.robot.arm_loop import ArmLoopConfig, _planned_homing
from dexmani_real.robot.hand_process import HandProcessConfig, _update_tracking_stall, hand_loop
from dexmani_real.robot.safety import SafetyState
from dexmani_real.shm.shared_storage import (
    HomeRequest,
    HomeResult,
    _estimate_home_timeout_s,
    send_arm_home,
    wait_for_arm_home,
)


class _Value:
    def __init__(self, value):
        self.value = value


def _shared() -> SimpleNamespace:
    return SimpleNamespace(
        is_running=_Value(True),
        error_state=_Value(False),
        estop_request=_Value(False),
        safety_state=_Value(int(SafetyState.ARMED)),
        arm_heartbeat_s=_Value(0.0),
        policy_heartbeat_s=_Value(0.0),
        arm_home_result_q=Queue(),
        arm_action_q=Queue(maxsize=2),
    )


class _FeedbackArm:
    def __init__(
        self,
        qpos: np.ndarray,
        *,
        send_code: int = 0,
        read_code: int = 0,
        follow_commands: bool = True,
        mode_codes: dict[int, int] | None = None,
        state_code: int = 0,
        error_after_send: int = 0,
    ):
        self.qpos = np.asarray(qpos, dtype=np.float64).copy()
        self.qvel = np.zeros(7, dtype=np.float64)
        self.send_code = send_code
        self.read_code = read_code
        self.follow_commands = follow_commands
        self.error_code = 0
        self.mode = 6
        self.mode_codes = mode_codes or {}
        self.state_code = state_code
        self.error_after_send = error_after_send
        self.mode_calls: list[int] = []
        self.state_calls: list[int] = []
        self.commands: list[tuple[np.ndarray, float, float, float | None]] = []

    def get_joint_states(self, *, is_radian: bool, num: int):
        if self.read_code != 0:
            return self.read_code, []
        return 0, [self.qpos.copy(), self.qvel.copy(), np.zeros(7, dtype=np.float64)]

    def set_mode(self, mode: int):
        self.mode_calls.append(mode)
        code = self.mode_codes.get(mode, 0)
        if code == 0:
            self.mode = mode
        return code

    def set_state(self, state: int):
        self.state_calls.append(state)
        return self.state_code

    def set_servo_angle(self, *, angle, is_radian, speed, mvacc, wait, radius):
        self.commands.append((np.asarray(angle).copy(), float(speed), float(mvacc), radius))
        if self.send_code == 0 and self.follow_commands:
            self.qpos = np.asarray(angle, dtype=np.float64).copy()
        if self.send_code == 0:
            self.error_code = self.error_after_send
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
    cfg = ArmLoopConfig(homing_step_interval_s=1e-6, homing_target_timeout_s=0.5)

    result = _planned_homing(arm, _request(np.stack([q0, q1])), cfg, shared=_shared())

    assert result.success
    assert result.request_id == 7
    np.testing.assert_allclose(result.final_qpos, q1)
    assert len(arm.commands) == 1
    assert all(speed == cfg.homing_max_speed_rad_per_s for _, speed, _, _ in arm.commands)
    assert arm.commands[0][3] is None
    assert arm.mode_calls == [0, 6]
    assert arm.state_calls == [0, 0]
    assert arm.mode == 6


def test_planned_homing_propagates_sdk_rejection_without_fabricating_home() -> None:
    q0 = np.zeros(7)
    q1 = np.full(7, 0.01)
    arm = _FeedbackArm(q0, send_code=9)

    result = _planned_homing(arm, _request(np.stack([q0, q1])), shared=_shared())

    assert not result.success
    assert "rejected" in result.reason
    np.testing.assert_allclose(result.final_qpos, q0)
    assert arm.mode_calls == [0, 6]


def test_planned_homing_fails_if_mode6_cannot_be_restored() -> None:
    q0 = np.zeros(7)
    q1 = np.full(7, 0.01)
    arm = _FeedbackArm(q0, mode_codes={6: 8})

    result = _planned_homing(arm, _request(np.stack([q0, q1])), shared=_shared())

    assert not result.success
    assert "Mode 6 restore failed" in result.reason
    np.testing.assert_allclose(result.final_qpos, q1)


def test_planned_homing_fails_closed_when_mode0_entry_is_rejected() -> None:
    q0 = np.zeros(7)
    q1 = np.full(7, 0.01)
    arm = _FeedbackArm(q0, mode_codes={0: 9})

    result = _planned_homing(arm, _request(np.stack([q0, q1])), shared=_shared())

    assert not result.success
    assert "Mode 0 entry failed" in result.reason
    assert not arm.commands
    assert arm.mode_calls == [0, 6]


def test_planned_homing_stops_instead_of_restoring_ready_after_controller_fault() -> None:
    q0 = np.zeros(7)
    q1 = np.full(7, 0.01)
    arm = _FeedbackArm(q0, error_after_send=31)

    result = _planned_homing(arm, _request(np.stack([q0, q1])), shared=_shared())

    assert not result.success
    assert "controller error C31" in result.reason
    assert arm.mode_calls == [0]
    assert arm.state_calls == [0, 4]


def test_planned_homing_rejects_estop_before_changing_mode() -> None:
    q0 = np.zeros(7)
    q1 = np.full(7, 0.01)
    arm = _FeedbackArm(q0)
    shared = _shared()
    shared.estop_request.value = True

    result = _planned_homing(arm, _request(np.stack([q0, q1])), shared=shared)

    assert not result.success
    assert result.reason == "e-stop requested"
    assert not arm.mode_calls
    assert not arm.commands


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
    path = np.zeros((2, 7), dtype=np.float64)
    path[:, 6] = [0.0, 2.0 * np.pi]

    assert _estimate_home_timeout_s(path) > 15.0


def test_planned_homing_distinguishes_overall_timeout_and_reports_joint_error() -> None:
    q0 = np.zeros(7)
    q1 = q0.copy()
    q1[0] = 0.1
    arm = _FeedbackArm(q0, follow_commands=False)
    shared = _shared()
    request = HomeRequest(8, np.stack([q0, q1]), q1, execution_timeout_s=0.12)
    cfg = ArmLoopConfig(homing_step_interval_s=0.04, homing_target_timeout_s=0.1)
    clock = [0.0]

    def _advance(dt: float) -> None:
        clock[0] += dt

    with (
        patch("dexmani_real.robot.arm_loop.time.monotonic", side_effect=lambda: clock[0]),
        patch("dexmani_real.robot.arm_loop.time.sleep", side_effect=_advance),
    ):
        result = _planned_homing(arm, request, cfg, shared=shared)

    assert not result.success
    assert "overall timeout" in result.reason
    assert "J1 error=" in result.reason


def test_planned_homing_publishes_fresh_feedback_while_waiting() -> None:
    q0 = np.zeros(7)
    q1 = np.full(7, 0.01)
    feedback = Mock()

    result = _planned_homing(
        _FeedbackArm(q0),
        _request(np.stack([q0, q1])),
        ArmLoopConfig(homing_step_interval_s=1e-6),
        shared=_shared(),
        feedback_callback=feedback,
    )

    assert result.success
    assert feedback.call_count >= 2
    np.testing.assert_allclose(feedback.call_args.args[0], q1)


def test_planned_homing_requires_low_velocity_for_full_dwell() -> None:
    q0 = np.zeros(7)
    q1 = np.full(7, 0.01)
    arm = _FeedbackArm(q0)
    arm.qvel[:] = 0.2
    cfg = ArmLoopConfig(homing_target_timeout_s=0.04, homing_step_interval_s=1e-6)

    result = _planned_homing(arm, _request(np.stack([q0, q1])), cfg, shared=_shared())

    assert not result.success
    assert "convergence timeout" in result.reason
    assert "J1 error=0.00deg" in result.reason


def test_single_point_home_path_rejects_high_velocity() -> None:
    home = np.zeros(7)
    arm = _FeedbackArm(home)
    arm.qvel[:] = 0.2

    result = _planned_homing(arm, _request(np.stack([home])), shared=_shared())

    assert not result.success
    assert "stationary canonical home" in result.reason
    assert not arm.commands


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


def test_home_path_densely_validates_but_returns_sparse_firmware_milestones() -> None:
    checked_sizes: list[int] = []

    class _Ik:
        @staticmethod
        def nearest_equivalent_qpos(home, current):
            return np.asarray(home, dtype=np.float64)

        @staticmethod
        def compute_qpos_delta(a, b):
            return np.asarray(a) - np.asarray(b)

        @staticmethod
        def check_path_collisions(path):
            checked_sizes.append(len(path))
            return {"path_self_collision": False}

    planner = SimpleNamespace(
        ik_mgr=_Ik(),
        planning_profile=SimpleNamespace(check_self_collision=True),
        is_workspace_segment_safe=lambda start, end: True,
    )
    path = plan_joint_home_path(np.zeros(7), np.full(7, 0.1), planner)

    assert path is not None
    assert path.shape == (2, 7)
    assert checked_sizes and checked_sizes[0] > len(path)


def test_home_path_tries_distal_first_when_other_linear_candidates_collide() -> None:
    collision_checks = 0

    class _Ik:
        @staticmethod
        def nearest_equivalent_qpos(home, current):
            return np.asarray(home, dtype=np.float64)

        @staticmethod
        def compute_qpos_delta(a, b):
            return np.asarray(a) - np.asarray(b)

        @staticmethod
        def check_path_collisions(path):
            nonlocal collision_checks
            collision_checks += 1
            return {"path_self_collision": collision_checks <= 2}

    planner = SimpleNamespace(
        ik_mgr=_Ik(),
        planning_profile=SimpleNamespace(check_self_collision=True),
        is_workspace_segment_safe=lambda start, end: True,
    )
    report: dict = {}
    path = plan_joint_home_path(np.zeros(7), np.full(7, 0.1), planner, report=report)

    assert path is not None
    assert path.shape == (3, 7)
    np.testing.assert_allclose(path[1, :4], 0.0)
    np.testing.assert_allclose(path[1, 4:], 0.1)
    assert report["selected_candidate"] == "distal_first"
    assert [candidate["reason"] for candidate in report["candidates"][:2]] == [
        "self_collision",
        "self_collision",
    ]


def test_home_path_uses_bounded_rrt_after_all_linear_candidates_fail() -> None:
    collision_checks = 0

    class _Ik:
        @staticmethod
        def nearest_equivalent_qpos(home, current):
            return np.asarray(home, dtype=np.float64)

        @staticmethod
        def compute_qpos_delta(a, b):
            return np.asarray(a) - np.asarray(b)

        @staticmethod
        def check_path_collisions(path):
            nonlocal collision_checks
            collision_checks += 1
            return {"path_self_collision": collision_checks <= 3}

    rrt_path = np.stack([np.zeros(7), np.full(7, 0.04), np.full(7, 0.1)])
    planner = SimpleNamespace(
        ik_mgr=_Ik(),
        planning_profile=SimpleNamespace(check_self_collision=True),
        is_workspace_segment_safe=lambda start, end: True,
        plan_joint_qpos_path=Mock(return_value=PathResult(success=True, qpos_path=rrt_path, source="joint_qpos_rrt")),
    )
    report: dict = {}
    path = plan_joint_home_path(np.zeros(7), np.full(7, 0.1), planner, report=report)

    assert path is not None
    np.testing.assert_allclose(path, rrt_path)
    planner.plan_joint_qpos_path.assert_called_once()
    assert planner.plan_joint_qpos_path.call_args.kwargs["planning_time_s"] == 0.5
    assert report["selected_candidate"] == "joint_qpos_rrt"


def test_send_arm_home_rejects_fault_before_planning_or_queueing() -> None:
    shared = _shared()
    shared.error_state.value = True
    planner = Mock()

    assert not send_arm_home(shared, np.zeros(7), planner=planner, verbose=False)
    planner.ik_mgr.nearest_equivalent_qpos.assert_not_called()
    assert shared.arm_action_q.empty()


def test_hand_startup_failure_sets_fault_latch_without_false_ready() -> None:
    shared = SimpleNamespace(error_state=_Value(False), hand_ready=Event())
    hand_instance = Mock()
    hand_instance.connect.return_value = False

    with patch("dexmani_real.robot.xhand.XHand", return_value=hand_instance):
        hand_loop(shared)

    assert shared.error_state.value
    assert not shared.hand_ready.is_set()
    hand_instance.disconnect.assert_called_once()


def test_optional_hand_startup_failure_does_not_fault_arm_only_entry() -> None:
    shared = SimpleNamespace(error_state=_Value(False), hand_ready=Event())
    hand_instance = Mock()
    hand_instance.connect.return_value = False

    with patch("dexmani_real.robot.xhand.XHand", return_value=hand_instance):
        hand_loop(shared, HandProcessConfig(startup_failure_is_fatal=False))

    assert not shared.error_state.value
    assert not shared.hand_ready.is_set()
    hand_instance.disconnect.assert_called_once()


def test_hand_disarmed_startup_validates_feedback_without_home_motion() -> None:
    shared = SimpleNamespace(
        is_running=_Value(False),
        error_state=_Value(False),
        estop_request=_Value(False),
        safety_state=_Value(int(SafetyState.DISARMED)),
        session_generation=_Value(1),
        policy_epoch=_Value(1),
        hand_heartbeat_s=_Value(0.0),
        hand_ready=Event(),
        hand_state_ring=Mock(),
        hand_ack_ring=Mock(),
        component_status_ring=None,
    )
    hand_instance = Mock()
    hand_instance.connect.return_value = True
    hand_instance.get_state.return_value = {"qpos": np.zeros(12)}
    hand_instance.stop.return_value = True
    hand_instance.feedback_bound_stats = {"checks": 0}

    with patch("dexmani_real.robot.xhand.XHand", return_value=hand_instance):
        hand_loop(shared)

    assert shared.hand_ready.is_set()
    hand_instance.send_action.assert_not_called()
    hand_instance.stop.assert_called_once()
    hand_instance.disconnect.assert_called_once()


def test_hand_runtime_counts_boolean_command_rejection() -> None:
    now_ns = time.monotonic_ns()
    candidate = ActionCandidate(
        observation_id=1,
        session_generation=1,
        policy_epoch=1,
        action_id=1,
        created_monotonic_ns=now_ns,
        target_monotonic_ns=now_ns + 20_000_000,
        valid_until_monotonic_ns=now_ns + 1_000_000_000,
        arm_qpos=None,
        hand_qpos=np.zeros(12),
        chunk_id=1,
    )
    command = make_command_frame(candidate, actuator="hand")
    commit = np.zeros(1, dtype=COMMIT_DTYPE)
    for name in (
        "session_generation",
        "policy_epoch",
        "observation_id",
        "action_id",
        "chunk_id",
        "step_index",
        "created_monotonic_ns",
        "target_monotonic_ns",
        "valid_until_monotonic_ns",
        "is_hold",
    ):
        commit[name][0] = command[name][0]
    commit["committed_monotonic_ns"][0] = now_ns + 1
    shared = SimpleNamespace(
        is_running=_Value(True),
        error_state=_Value(False),
        estop_request=_Value(False),
        safety_state=_Value(int(SafetyState.ARMED)),
        hand_heartbeat_s=_Value(0.0),
        hand_ready=Event(),
        hand_cmd_ring=Mock(),
        hand_state_ring=Mock(),
        hand_tactile_ring=Mock(),
        hand_ack_ring=Mock(),
        action_commit_ring=Mock(),
        policy_epoch=_Value(1),
        session_generation=_Value(1),
    )
    shared.hand_cmd_ring.read_latest.return_value = (command, 0.0, 1)
    shared.action_commit_ring.read_latest.return_value = (commit, 0.0, 1)

    hand_instance = Mock()
    hand_instance.config = SimpleNamespace(home_qpos=None, qpos_min=np.full(12, -1.0), qpos_max=np.full(12, 1.0))
    hand_instance.connect.return_value = True
    hand_instance.send_action.return_value = False
    hand_instance.connected_flag = True
    hand_instance.error_state = False
    hand_instance.last_action_code = 9
    hand_instance.tactile_calibrated = False
    hand_instance.feedback_bound_stats = {"checks": 0}
    state_reads = 0

    def _get_state(*args, **kwargs):
        nonlocal state_reads
        state_reads += 1
        if state_reads >= 3:
            shared.is_running.value = False
        return {"qpos": np.zeros(12)}

    hand_instance.get_state.side_effect = _get_state

    with patch("dexmani_real.robot.xhand.XHand", return_value=hand_instance):
        hand_loop(shared, HandProcessConfig(send_err_watchdog_frames=1))

    hand_instance.send_action.assert_called_once()
    hand_instance.clear_error.assert_called_once()
    assert shared.error_state.value


def test_hand_tracking_stall_distinguishes_settled_feedback_from_no_progress() -> None:
    target = np.ones(12)
    stale_frames, previous_error, active = _update_tracking_stall(
        target.copy(),
        target,
        active=True,
        previous_error_rad=1.0,
        stale_frames=14,
        progress_epsilon_rad=1e-4,
    )
    assert (stale_frames, active) == (0, False)

    qpos = np.zeros(12)
    previous_error = float(np.max(np.abs(qpos - target)))
    stale_frames = 0
    active = True
    for _ in range(15):
        stale_frames, previous_error, active = _update_tracking_stall(
            qpos,
            target,
            active=active,
            previous_error_rad=previous_error,
            stale_frames=stale_frames,
            progress_epsilon_rad=1e-4,
        )
    assert active
    assert stale_frames == 15

    stale_frames, previous_error, active = _update_tracking_stall(
        np.full(12, 0.5),
        target,
        active=True,
        previous_error_rad=1.0,
        stale_frames=7,
        progress_epsilon_rad=1e-4,
    )
    assert (stale_frames, previous_error, active) == (0, 0.5, True)


def test_hand_runtime_prepares_next_chunk_step_in_apply_tick() -> None:
    now_ns = time.monotonic_ns()

    def candidate(action_id: int, target_offset_ns: int, step_index: int) -> ActionCandidate:
        return ActionCandidate(
            observation_id=1,
            session_generation=1,
            policy_epoch=1,
            action_id=action_id,
            created_monotonic_ns=now_ns,
            target_monotonic_ns=now_ns + target_offset_ns,
            valid_until_monotonic_ns=now_ns + 1_000_000_000,
            arm_qpos=None,
            hand_qpos=np.full(12, action_id / 10.0),
            chunk_id=1,
            step_index=step_index,
        )

    commands = [
        make_command_frame(candidate(1, 20_000_000, 0), actuator="hand"),
        make_command_frame(candidate(2, 80_000_000, 1), actuator="hand"),
    ]

    def commit_for(command: np.ndarray) -> np.ndarray:
        commit = np.zeros(1, dtype=COMMIT_DTYPE)
        for name in (
            "session_generation",
            "policy_epoch",
            "observation_id",
            "action_id",
            "chunk_id",
            "step_index",
            "created_monotonic_ns",
            "target_monotonic_ns",
            "valid_until_monotonic_ns",
            "is_hold",
        ):
            commit[name][0] = command[name][0]
        commit["committed_monotonic_ns"][0] = now_ns + 1
        return commit

    commits = [commit_for(command) for command in commands]
    command_reads = 0
    commit_reads = 0

    def read_command():
        nonlocal command_reads
        if command_reads >= len(commands):
            return None
        result = (commands[command_reads], 0.0, command_reads + 1)
        command_reads += 1
        return result

    def read_commit():
        nonlocal commit_reads
        if commit_reads >= len(commits):
            return None
        result = (commits[commit_reads], 0.0, commit_reads + 1)
        commit_reads += 1
        return result

    shared = SimpleNamespace(
        is_running=_Value(True),
        error_state=_Value(False),
        estop_request=_Value(False),
        safety_state=_Value(int(SafetyState.ARMED)),
        hand_heartbeat_s=_Value(0.0),
        hand_ready=Event(),
        hand_cmd_ring=Mock(read_latest=Mock(side_effect=read_command)),
        hand_state_ring=Mock(),
        hand_tactile_ring=Mock(),
        hand_ack_ring=Mock(),
        action_commit_ring=Mock(read_latest=Mock(side_effect=read_commit)),
        policy_epoch=_Value(1),
        session_generation=_Value(1),
    )
    hand_instance = Mock()
    hand_instance.config = SimpleNamespace(home_qpos=None, qpos_min=np.full(12, -1.0), qpos_max=np.full(12, 1.0))
    hand_instance.connect.return_value = True
    hand_instance.send_action.return_value = True
    hand_instance.connected_flag = True
    hand_instance.error_state = False
    hand_instance.last_action_code = 0
    hand_instance.tactile_calibrated = False
    hand_instance.feedback_bound_stats = {"checks": 0}
    hand_instance.stop.return_value = True
    state_reads = 0

    def get_state(*_args, **_kwargs):
        nonlocal state_reads
        state_reads += 1
        if state_reads >= 4:
            shared.is_running.value = False
        return {"qpos": np.zeros(12)}

    hand_instance.get_state.side_effect = get_state

    with patch("dexmani_real.robot.xhand.XHand", return_value=hand_instance):
        hand_loop(shared, HandProcessConfig(loop_hz=20.0))

    assert hand_instance.send_action.call_count == 2
    np.testing.assert_allclose(hand_instance.send_action.call_args_list[0].args[0], np.full(12, 0.1))
    np.testing.assert_allclose(hand_instance.send_action.call_args_list[1].args[0], np.full(12, 0.2))
    prepared_ids = [
        int(call.args[0]["action_id"][0])
        for call in shared.hand_ack_ring.write.call_args_list
        if int(call.args[0]["status"][0]) == int(AckStatus.PREPARED)
    ]
    assert prepared_ids == [1, 2]


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
