from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from dexmani_real.control.constraints import joint_target_rejection
from dexmani_real.robot.commands import OperationCancelled, RobotCommand, RobotError
from dexmani_real.robot.xarm7.process import ArmProcess
from dexmani_real.robot.xarm7.protocol import GOAL_DTYPE, MAX_TIME, LifecycleReply, PrepareRequest
from dexmani_real.robot.xarm7.trajectory import ArmTrajectory
from dexmani_real.robot.xarm7.worker import ArmController
from dexmani_real.robot.xhand.config import HandConfig
from dexmani_real.robot.xhand.driver import COMMUNICATION_CRC_ERROR_CODE as CRC, XHand, XHandError
from dexmani_real.runtime.observation import ObservationNeeds, check_observation


@pytest.mark.parametrize("side,bad", [("arm", np.full(7, np.nan)), ("hand", np.full(12, np.nan)),
                                     ("hand", np.zeros(11)), ("arm", np.full(7, 100)),
                                     ("hand", np.full(12, 100))])
def test_combined_preflight_precedes_both_actuators(robot_fixture, clock, side, bad):
    env = robot_fixture
    arm = np.asarray(env.runtime.arm.home_qpos)
    hand = np.deg2rad(env.runtime.hand.home_qpos_deg)
    command = RobotCommand(env.state.run_id, bad if side == "arm" else arm,
                           bad if side == "hand" else hand)
    with pytest.raises(ValueError):
        env.robot.send_action(command, valid_until_ns=clock.now + 100, execute_until_ns=clock.now + 100)
    assert not env.sent and not env.hand_sent


def test_home_preflight_and_cancel_across_preparation(robot_fixture, clock, monkeypatch):
    env = robot_fixture
    begins = []

    def begin(run_id):
        begins.append(run_id)
        env.state.stop_signal.clear()

    monkeypatch.setattr(env.robot.arm, "begin_operation", begin)
    with pytest.raises(ValueError):
        env.robot.send_hand_home(RobotCommand(env.state.run_id, hand_qpos=np.full(12, np.nan)),
                                  valid_until_ns=clock.now + 100)
    assert not begins and not env.hand_sent
    command = RobotCommand(env.state.run_id, hand_qpos=np.deg2rad(env.runtime.hand.home_qpos_deg))
    env.robot.send_hand_home(command, valid_until_ns=clock.now + 100)
    assert len(begins) == len(env.hand_sent) == 1

    def cancelled_begin(run_id):
        env.state.run_signal.value += 1

    monkeypatch.setattr(env.robot.arm, "begin_operation", cancelled_begin)
    with pytest.raises(OperationCancelled):
        env.robot.send_hand_home(command, valid_until_ns=clock.now + 100)
    assert len(env.hand_sent) == 1


def make_hand(send_codes=(), read_codes=()):
    cfg = HandConfig()
    hand = XHand(cfg)
    hand.connected_flag = True
    hand.command = SimpleNamespace(finger_command=[SimpleNamespace() for _ in range(12)])
    force = SimpleNamespace(fx=1., fy=2., fz=3.)
    raw = SimpleNamespace(
        finger_state=[SimpleNamespace(id=i, position=float(q), torque=None,
                                      commboard_err=0, jonitboard_err=0, tipboard_err=0)
                      for i, q in enumerate(np.deg2rad(cfg.home_qpos_deg))],
        sensor_data=[SimpleNamespace(calc_force=force, raw_force=[force] * 120) for _ in range(5)],
    )
    sends, reads = deque(send_codes), deque(read_codes)
    history = []

    def send(device, command):
        history.append([(joint.mode, getattr(joint, "position", None)) for joint in command.finger_command])
        return SimpleNamespace(error_code=sends.popleft() if sends else 0, error_message="test reply")

    def read(device, refresh):
        return SimpleNamespace(error_code=reads.popleft() if reads else 0, error_message="test reply"), raw

    hand.control = SimpleNamespace(send_command=send, read_state=read)
    return hand, history


def test_torque_none_and_read_crc_payload_discarded():
    hand, _ = make_hand(read_codes=[CRC, 0])
    assert hand.get_state() is None
    sample = hand.get_state()
    np.testing.assert_allclose(sample.qpos, np.deg2rad(hand.cfg.home_qpos_deg))
    assert not hasattr(sample, "current_ma") and not hasattr(sample, "commboard_err")


def test_crc_low_rate_recovery_and_continuous_fault(clock):
    hand, history = make_hand(send_codes=[CRC, 0, CRC, CRC])
    target = np.deg2rad(hand.cfg.home_qpos_deg)
    hand.send_action(target)
    for ms in (40, 80, 120, 160):
        clock.at(ms)
        hand.get_state()
    assert hand.send_crc_started_ns == clock.origin  # 读取成功不是发送 ACK。
    clock.at(200)
    hand.send_action(target)
    assert hand.send_crc_started_ns is None
    clock.at(400)
    hand.send_action(target)
    clock.at(600)
    with pytest.raises(XHandError, match="continuous send CRC"):
        hand.send_action(target)
    assert len(history) == 4
    faulty, _ = make_hand(send_codes=[1_501_035])
    with pytest.raises(XHandError, match="1501035"):
        faulty.send_action(target)


def test_stop_requires_non_crc_response_and_validates_hold(clock):
    hand, history = make_hand(send_codes=[CRC, 0])
    hand.stop(np.full(12, 10.), timestamp_ns=clock.now)
    assert len(history) == 2
    np.testing.assert_allclose([position for _, position in history[-1]], hand.mechanical_upper)
    assert hand.send_crc_started_ns is None
    with pytest.raises(ValueError):
        hand.hold_current(np.full(12, np.nan))
    with pytest.raises(ValueError):
        hand.stop(np.zeros(11), timestamp_ns=clock.now)
    assert len(history) == 2
    failed, history = make_hand(send_codes=[CRC] * 100)
    with pytest.raises(RuntimeError, match="stop timed out"):
        failed.stop()
    assert all(mode == 0 for mode, _ in history[-1])
    assert len(history) < 100


def test_cache_reads_and_crc_never_refresh_source_time(robot_fixture, clock):
    env = robot_fixture
    stamp = int(env.robot.read_state().hand["timestamp_ns"][0])
    reads = env.robot.hand.reads
    clock.at(10)
    assert int(env.robot.read_state().hand["timestamp_ns"][0]) == stamp
    assert env.robot.hand.reads == reads
    env.robot.hand.crc = True
    for ms in (40, 80, 120):
        clock.at(ms)
        frame = env.robot.service_feedback()
        assert int(frame.hand["timestamp_ns"][0]) == stamp
        assert env.health[-1] <= stamp + int(env.runtime.hand.feedback_max_age_s * 1e9)
    clock.at(135)
    with pytest.raises(RobotError, match="unavailable/stale"):
        env.robot.service_feedback()
    assert env.state.stop_signal.is_set()


def test_observation_deadline_matches_feedback_expiry(robot_fixture, clock):
    env = robot_fixture
    row = env.robot.read_state()
    expires = clock.now + int(env.runtime.arm.feedback_max_age_s * 1e9)
    clock.now = expires - 1
    issues, deadline = check_observation(row, env.runtime, ObservationNeeds())
    assert issues == [] and deadline == expires
    clock.now = expires
    issues, deadline = check_observation(row, env.runtime, ObservationNeeds())
    assert "arm_feedback_stale" in issues and deadline is None


def controller_fixture(env, clock):
    state = env.state
    state.stop_signal.clear()
    io = SimpleNamespace(run_id=state.run_signal, stop=state.stop_signal,
                         emergency=state.emergency_signal, closing=SimpleNamespace(is_set=lambda: False),
                         fault=SimpleNamespace(value=b""), health_until=SimpleNamespace(value=clock.now + 1_000_000_000),
                         execution_limit=[MAX_TIME, 0], input_generation=SimpleNamespace(value=3))
    driver = SimpleNamespace(mode=1, servo=lambda q: pytest.fail("unexpected servo side effect"))
    controller = ArmController(env.runtime.arm, io, None, driver)
    controller.status["valid"] = True
    controller.status["prepared_run_id"] = state.run_id
    controller.status["frame"] = env.robot.arm.snapshot()[0]
    goal = np.zeros((), dtype=GOAL_DTYPE)
    goal["run_id"] = state.run_id
    goal["submit_until_ns"] = clock.now + 1_000_000_000
    goal["execute_until_ns"] = clock.now + 60_000_000
    goal["input_generation"] = 3
    goal["sequence"] = 1
    return controller, goal


def test_health_cannot_extend_command_or_input_authority(robot_fixture, clock):
    env = robot_fixture
    controller, goal = controller_fixture(env, clock)
    assert controller.command_issue(goal, clock.now, first=True) is None
    clock.at(200)
    env.robot.service_feedback()
    controller.io.health_until.value = env.health[-1]
    controller.status["frame"] = env.robot.arm.snapshot()[0]
    assert controller.command_issue(goal, clock.now) == "input_decision_expired"
    goal["execute_until_ns"] = clock.now + 1000
    controller.io.input_generation.value += 1
    assert controller.command_issue(goal, clock.now, first=True) == "input_changed"
    env.state.run_signal.value += 1
    assert controller.command_issue(goal, clock.now) == "revoked"


def test_large_valid_target_and_braking_and_tracking_guards(robot_fixture, clock):
    env = robot_fixture
    cfg = env.runtime.arm
    q = np.asarray(cfg.home_qpos).copy()
    target = q.copy()
    target[0] += 0.6
    limits = np.column_stack((cfg.joint_limit_lower, cfg.joint_limit_upper))
    assert joint_target_rejection(target, limits) is None
    invalid = target.copy()
    invalid[0] = limits[0, 1] + 0.01
    assert joint_target_rejection(invalid, limits) == "operational_limits"
    trajectory = ArmTrajectory(cfg, q, np.zeros(7), clock.now)
    trajectory.step(clock.now, target)
    np.testing.assert_allclose(trajectory.step(clock.now + 2_000_000_000), target)
    q[0] = limits[0, 0] + 0.001
    velocity = np.zeros(7)
    velocity[0] = -1.
    with pytest.raises(ValueError, match="braking/position extremum"):
        ArmTrajectory(cfg, q, velocity, clock.now)
    controller, goal = controller_fixture(env, clock)
    controller.io.take_goal = lambda: None
    controller.active = goal
    reference = np.asarray(cfg.home_qpos).copy()
    reference[0] += cfg.tracking_error_rad + 0.01
    controller.reference = SimpleNamespace(step=lambda *args: reference)
    with pytest.raises(RuntimeError, match="tracking_error"):
        controller.tick()


def test_close_interrupt_does_not_skip_other_resources(robot_fixture, monkeypatch):
    env = robot_fixture
    arm, hand = env.robot.arm, env.robot.hand
    original = KeyboardInterrupt("stop interrupted")

    def interrupted(*args, **kwargs):
        raise original

    monkeypatch.setattr(hand, "stop", interrupted)
    with pytest.raises(KeyboardInterrupt) as failure:
        env.robot.close()
    assert failure.value is original
    assert not arm.is_connected and not hand.is_connected
    assert env.robot.stop_required


@pytest.mark.parametrize("expired", [False, True])
def test_combined_send_rechecks_after_arm_submission(robot_fixture, clock, monkeypatch, expired):
    env = robot_fixture
    env.robot.prepare_streaming(env.state.run_id, service=env.robot.service_feedback)
    original_submit = env.robot.arm.submit

    def blocked_submit(command, **kwargs):
        original_submit(command, **kwargs)
        if expired:
            clock.sleep(0.1)
        else:
            env.state.run_signal.value += 1

    monkeypatch.setattr(env.robot.arm, "submit", blocked_submit)
    command = RobotCommand(env.state.run_id, np.asarray(env.runtime.arm.home_qpos),
                           np.deg2rad(env.runtime.hand.home_qpos_deg))
    with pytest.raises(TimeoutError if expired else OperationCancelled):
        env.robot.send_action(command, valid_until_ns=clock.now + 50_000_000,
                              execute_until_ns=clock.now + 500_000_000)
    assert len(env.sent) == 1 and not env.hand_sent
    assert env.state.stop_signal.is_set()


def test_arm_operation_preserves_original_error_and_traceback(robot_fixture, clock, caplog):
    import threading

    env = robot_fixture
    controller, _ = controller_fixture(env, clock)
    replies = []
    controller.pipe = SimpleNamespace(send=replies.append)
    controller.io.stopped = threading.Event()
    controller.io.close_error = SimpleNamespace(value=b"")
    controller.io.take_goal = lambda **kwargs: None
    controller.io.publish_status = lambda status: None
    controller.driver.stop = lambda: None
    original = ValueError("planner malfunction")

    def fail(check_abort):
        raise original

    controller.wait_stationary = fail
    controller.operation(PrepareRequest(1, env.state.run_id))
    assert replies[0].error is original
    assert controller.io.stopped.is_set()
    assert caplog.records[-1].exc_info[1] is original


@pytest.mark.parametrize("operation_failed", [False, True])
def test_blocking_arm_operation_keeps_feedback_service_and_error_cause(robot_fixture, clock, operation_failed):
    env = robot_fixture
    process = ArmProcess.__new__(ArmProcess)
    process.cfg = env.runtime.arm
    process.io = SimpleNamespace(connected=SimpleNamespace(is_set=lambda: True),
                                 close_error=SimpleNamespace(value=b""))
    process.process = SimpleNamespace(is_alive=lambda: True)
    process.operation_unresolved = False
    original = ValueError("bad plan") if operation_failed else None

    def poll(timeout):
        clock.sleep(timeout)
        return clock.now >= clock.origin + 150_000_000

    process.pipe = SimpleNamespace(poll=poll, send=lambda request: None,
                                   recv=lambda: LifecycleReply(1, original))
    process.request_stop = env.robot.arm.request_stop
    args = dict(service=env.robot.service_feedback, abort=lambda: False,
                deadline_ns=clock.now + 1_000_000_000)
    if operation_failed:
        with pytest.raises(RobotError) as failure:
            process.request_operation(PrepareRequest(1, env.state.run_id), **args)
        assert failure.value.__cause__ is original
    else:
        process.request_operation(PrepareRequest(1, env.state.run_id), **args)
    assert env.robot.hand.reads >= 5
    assert int(env.robot.read_state().hand["timestamp_ns"][0]) >= clock.origin + 130_000_000


@pytest.mark.parametrize("case", ["pipe_failure", "owner_failed", "owner_pending"])
def test_arm_close_preserves_pending_owner_and_closes_other_pipe(case):
    process = ArmProcess.__new__(ArmProcess)
    process.request_stop = lambda: None
    process.io = SimpleNamespace(closing=SimpleNamespace(set=lambda: None),
                                 close_error=SimpleNamespace(value=b""))
    process.process = SimpleNamespace(pid=1, join=lambda timeout: None,
                                      is_alive=lambda: case == "owner_pending",
                                      exitcode=1 if case == "owner_failed" else 0)
    closed = []
    original = OSError("pipe close failed")

    def close_parent():
        closed.append("parent")
        if case == "pipe_failure":
            raise original

    process.pipe = SimpleNamespace(close=close_parent)
    process.child_pipe = SimpleNamespace(close=lambda: closed.append("child"))
    with pytest.raises(OSError if case == "pipe_failure" else RuntimeError if case == "owner_failed" else TimeoutError) as failure:
        process.close()
    assert closed == ([] if case == "owner_pending" else ["parent", "child"])
    if case == "pipe_failure":
        assert failure.value is original


def test_tag_stage_two_roundoff_uses_stage_one_only():
    import nlopt

    from dexmani_real.teleop.config import TAGConfig
    from dexmani_real.teleop.retargeting.tag import TAGRetargeter

    retargeter = TAGRetargeter(TAGConfig())
    first = (retargeter.lower + retargeter.upper) / 2
    retargeter.position_optimizer = SimpleNamespace(optimize=lambda seed: first.copy())

    def fail(error):
        def optimize(seed):
            raise error
        return SimpleNamespace(optimize=optimize)

    retargeter.pinch_optimizer = fail(nlopt.RoundoffLimited())
    np.testing.assert_array_equal(retargeter.solve(np.zeros((5, 3))), first)
    np.testing.assert_array_equal(retargeter.previous_solution, first)
    np.testing.assert_array_equal(retargeter.position_solution, first)
    original = ValueError("objective bug")
    retargeter.pinch_optimizer = fail(original)
    with pytest.raises(ValueError) as failure:
        retargeter.solve(np.zeros((5, 3)))
    assert failure.value is original
    retargeter.position_optimizer = fail(nlopt.RoundoffLimited())
    with pytest.raises(nlopt.RoundoffLimited):
        retargeter.solve(np.zeros((5, 3)))
