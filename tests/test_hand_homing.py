"""Offline HOME boundary checks using the robot's real dispatch path and fake devices."""

import pytest
from test_review_remediation import fake_robot

from dexmani_real.robot.hand_homing import home_hand
from dexmani_real.robot.robot import DispatchError, DispatchStatus
from dexmani_real.runtime.safety import RunEndReason, SafetyState, revoke_motion
from dexmani_real.teleop.runner import TeleopRunner


@pytest.mark.parametrize("cause", ["deadline_expired", "stop_unconfirmed"])
@pytest.mark.parametrize("late_quit", [False, True])
def test_startup_home_technical_failure_cannot_become_normal_quit(monkeypatch, cause, late_quit):
    robot, clock, _ = fake_robot(monkeypatch)
    shared = robot.shared
    shared.safety_state.value = int(SafetyState.ARMED)
    if cause == "deadline_expired":

        def check_services():
            clock.now += int(robot.runtime.hand.home_timeout_s * 1e9)
            return True

        robot.check_services = check_services
    else:
        robot._hand_stop_pending = True

    stops = []

    def stop():
        # HOME must revoke its epoch before attempting to stop.
        stops.append(shared.run_id.value)
        if late_quit:
            revoke_motion(shared, reason=RunEndReason.QUIT)
            shared.quit_requested.value = True

    robot.stop = stop
    runner = TeleopRunner.__new__(TeleopRunner)
    runner.shared, runner.robot, runner.runtime = shared, robot, robot.runtime
    runner._poll_blocking_commands = lambda: False
    runner.start_vr = lambda: pytest.fail("failed HOME must not start VR")

    with pytest.raises(DispatchError) as caught:
        runner._initialize()

    assert caught.value.cause == cause
    assert caught.value.result.hand == DispatchStatus.NOT_CALLED
    assert clock.calls == []
    assert stops == [3]
    assert shared.quit_requested.value == late_quit


@pytest.mark.parametrize("late_quit", [False, True])
def test_startup_home_convergence_timeout_survives_late_quit(monkeypatch, late_quit):
    robot, clock, _ = fake_robot(monkeypatch)
    shared = robot.shared
    shared.safety_state.value = int(SafetyState.ARMED)
    clock.hand_delay = int(robot.runtime.hand.home_timeout_s * 1e9)
    stops = []

    def stop():
        stops.append(shared.run_id.value)
        if late_quit:
            revoke_motion(shared, reason=RunEndReason.QUIT)
            shared.quit_requested.value = True

    robot.stop = stop
    runner = TeleopRunner.__new__(TeleopRunner)
    runner.shared, runner.robot, runner.runtime = shared, robot, robot.runtime
    runner._poll_blocking_commands = lambda: False
    runner.start_vr = lambda: pytest.fail("failed HOME must not start VR")

    with pytest.raises(RuntimeError, match="hand home convergence timed out"):
        runner._initialize()

    assert clock.calls == ["hand"] and stops == [3]
    assert shared.quit_requested.value == late_quit


@pytest.mark.parametrize("quit_requested", [False, True])
def test_hand_home_operator_revocation_still_returns_interrupted(monkeypatch, quit_requested):
    robot, clock, _ = fake_robot(monkeypatch)
    shared = robot.shared
    shared.safety_state.value = int(SafetyState.ARMED)
    stops = []
    robot.stop = lambda: stops.append(True)

    def check_services():
        revoke_motion(shared, reason=RunEndReason.QUIT if quit_requested else RunEndReason.OPERATOR)
        shared.quit_requested.value = quit_requested
        return True

    robot.check_services = check_services
    result = home_hand(shared, robot.runtime, robot=robot)
    assert not result.ok and result.interrupted
    assert clock.calls == [] and stops == [True]
