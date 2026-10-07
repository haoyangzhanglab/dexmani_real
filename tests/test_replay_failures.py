"""Replay must not turn technical failures into successful operator exits."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
from test_review_remediation import fake_robot

from dexmani_real.replay import replayer
from dexmani_real.robot.robot import DispatchError, DispatchResult, DispatchStatus
from dexmani_real.runtime.safety import RunEndReason, SafetyState, revoke_motion


@pytest.fixture
def replay_case(monkeypatch):
    robot, clock, command = fake_robot(monkeypatch)
    robot.shared.safety_state.value = int(SafetyState.ARMED)
    row = NS(
        arm={"qpos": np.array([command.arm_qpos]), "timestamp_ns": [100]},
        hand={"qpos": np.array([command.hand_qpos]), "timestamp_ns": [100]},
    )
    monkeypatch.setattr(replayer, "read_observation", lambda *a, **kw: row)
    monkeypatch.setattr(replayer, "_wait_replay", lambda *a: None)
    monkeypatch.setattr(
        replayer,
        "make_arm_fk",
        lambda: NS(compute=lambda _: (np.zeros(3), np.array([1, 0, 0, 0, 1, 0]))),
    )
    trajectory = NS(
        num_frames=3,
        fps=30,
        arm_qpos=np.tile(command.arm_qpos, (3, 1)),
        hand_qpos=np.tile(command.hand_qpos, (3, 1)),
        action_arm_joint=np.tile(command.arm_qpos, (3, 1)),
        action_hand_joint=np.tile(command.hand_qpos, (3, 1)),
    )
    stops = []

    def stop():
        stops.append(True)
        # Model Q arriving while dispatch failure is being stopped/finalized.
        revoke_motion(robot.shared, reason=RunEndReason.QUIT)
        robot.shared.quit_requested.value = True

    robot.stop = stop
    return robot, trajectory, stops


def run_case(robot, trajectory):
    return replayer.replay_targets(
        robot.shared, robot.runtime, trajectory, None, robot=robot, hand_start_duration_s=0
    )


@pytest.mark.parametrize("cause", ["deadline_expired", "stop_unconfirmed", None])
def test_quit_after_technical_dispatch_failure_is_not_success(replay_case, cause):
    robot, trajectory, stops = replay_case
    sends = []

    def send(command, **kwargs):
        sends.append(command)
        raise DispatchError(
            "technical rejection",
            DispatchResult(DispatchStatus.ACCEPTED, timestamp_ns=100),
            revoked=True,
            cause=cause,
        )

    robot.send_action = send
    outcome = run_case(robot, trajectory)
    assert outcome.status == replayer.ReplayStatus.REJECTED
    assert not outcome.successful
    assert len(sends) == 1 and stops
    np.testing.assert_array_equal(outcome.replay_data["dispatch_status"], [[1, 0]])
    assert np.isnan(outcome.replay_data["hand_cmd"]).all()


@pytest.mark.parametrize("reason", [RunEndReason.QUIT, RunEndReason.EXECUTOR_BOUNDARY])
def test_only_current_operator_revocation_is_success(replay_case, reason):
    robot, trajectory, stops = replay_case

    def send(command, **kwargs):
        revoke_motion(robot.shared, reason=reason)
        raise DispatchError(
            "authority revoked",
            DispatchResult(DispatchStatus.ACCEPTED, timestamp_ns=100),
            revoked=True,
            cause="authority_revoked",
        )

    robot.send_action = send
    outcome = run_case(robot, trajectory)
    expected = (
        replayer.ReplayStatus.USER_QUIT
        if reason == RunEndReason.QUIT
        else replayer.ReplayStatus.REJECTED
    )
    assert outcome.status == expected
    assert stops
    np.testing.assert_array_equal(outcome.replay_data["dispatch_status"], [[1, 0]])


def test_capture_failure_during_quit_remains_fault_with_saved_prefix(replay_case, monkeypatch):
    robot, trajectory, stops = replay_case
    sends = []

    def send(command, **kwargs):
        sends.append(command)
        result = DispatchResult(DispatchStatus.ACCEPTED, DispatchStatus.ACCEPTED, timestamp_ns=100)
        if len(sends) == 1:
            return result
        revoke_motion(robot.shared, reason=RunEndReason.QUIT)
        raise DispatchError("operator quit", result, revoked=True, cause="authority_revoked")

    native_record = replayer.ReplayRecorder.record

    def record(self, idx, *args, **kwargs):
        if idx == 1:
            raise ValueError("capture conversion failed")
        return native_record(self, idx, *args, **kwargs)

    robot.send_action = send
    monkeypatch.setattr(replayer.ReplayRecorder, "record", record)
    outcome = run_case(robot, trajectory)
    assert outcome.status == replayer.ReplayStatus.FAULT
    assert robot.shared.error_state.value
    assert "capture conversion failed" in outcome.reason
    assert len(sends) == 2 and stops
    np.testing.assert_array_equal(outcome.replay_data["dispatch_status"], [[1, 1]])
