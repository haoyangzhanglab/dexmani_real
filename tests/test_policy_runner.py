from concurrent.futures import Future
from threading import RLock
from types import SimpleNamespace as NS

import numpy as np
import pytest

from dexmani_real.deployment.config import ExecutionConfig
from dexmani_real.deployment.inference import ModelResult
from dexmani_real.deployment.observation import decision_is_fresh, policy_sources
from dexmani_real.deployment.runner import PolicyRunner
from dexmani_real.planning.kinematics.ik import IKFailureKind, IKResult
from dexmani_real.robot.action import ActionRealization, ActionRealizer
from dexmani_real.robot.robot import (
    DispatchError,
    DispatchInterrupted,
    DispatchResult,
    DispatchStatus,
)
from dexmani_real.runtime.observation import ObservationRow
from dexmani_real.runtime.safety import RunEndReason, SafetyState, request_policy_stop


class Worker:
    def __init__(self, clock):
        self.clock = clock
        self.future = None
        self.submissions = []
        self.reclaimed = []
        self.closed = False

    def submit(self, op, *args, **kwargs):
        assert self.future is None
        self.op = op
        self.future = Future()
        self.submissions.append((op, args, kwargs))

    def complete(self, value=None, error=None):
        self.future.set_result(ModelResult(value, error, self.clock(), self.clock()))

    def poll(self):
        if self.future is None or not self.future.done():
            return None
        result = self.future.result()
        self.future = None
        self.reclaimed.append(result)
        return self.op, result

    def close(self):
        self.closed = True


class Robot:
    stop_required = False

    def __init__(self):
        self.sent = []
        self.stops = 0
        self.result = DispatchResult(DispatchStatus.ACCEPTED, DispatchStatus.ACCEPTED)

    def check(self):
        pass

    def service_idle(self):
        pass

    def send_action(self, command, *, valid_until_ns):
        import time

        assert time.monotonic_ns() < valid_until_ns
        self.sent.append(command)
        return self.result

    def stop(self):
        self.stops += 1


@pytest.fixture
def make_runner(monkeypatch):
    def make(mode="async", a=3, d=2, n=2, h=8, execute=True, wait=1.0):
        clock = NS(now=1_000_000_000)
        monkeypatch.setattr("time.monotonic_ns", lambda: clock.now)

        def row():
            return ObservationRow(
                {"timestamp_ns": np.array([clock.now]), "qpos": np.zeros((1, 7))},
                {"timestamp_ns": np.array([clock.now]), "qpos": np.zeros((1, 12))},
                None,
                None,
                None,
                clock.now,
            )

        shared = NS(
            motion_lock=RLock(),
            **{
                k: NS(value=v)
                for k, v in dict(
                    run_id=1,
                    run_ended_id=0,
                    run_ended_reason=0,
                    is_running=True,
                    error_state=False,
                    estop_request=False,
                    quit_requested=False,
                    start_request=False,
                    stop_request=0,
                    safety_state=int(SafetyState.RUNNING),
                ).items()
            },
        )
        runtime = NS(
            policy=NS(hand_enabled=True),
            arm=NS(
                joint_limit_lower=np.full(7, -3.0),
                joint_limit_upper=np.full(7, 3.0),
                feedback_max_age_s=0.15,
            ),
            hand=NS(
                qpos_min_rad=np.full(12, -2.0),
                qpos_max_rad=np.full(12, 2.0),
                feedback_max_age_s=0.15,
            ),
        )
        info = NS(
            n_obs_steps=n,
            n_action_steps=a,
            horizon=h,
            control_dt_s=0.1,
            observation_fields=("joint_state",),
            action_mode="joint",
        )
        worker = Worker(lambda: clock.now)
        robot = Robot()
        runner = PolicyRunner(
            shared,
            runtime,
            info,
            robot=robot,
            realizer=ActionRealizer(
                runtime,
                collision_model=NS(
                    set_hand_qpos=lambda q: None,
                    check_self_collision=lambda q: False,
                ),
            ),
            model_runtime=worker,
            execution_config=ExecutionConfig(mode, 10.0, wait, 0.03, d, 2.0),
            kinematics=None,
            execute=execute,
            max_running_s=100.0,
            num_episodes=10,
        )
        runner._read_observation = row
        runner.run_id = 1
        runner.recording_started = True  # The fixture starts inside a synthetic active run.
        runner.started_ns = clock.now
        runner.next_step_ns = clock.now
        runner.wait_started_ns = clock.now
        runner.events = []

        def tick(slot, extra=0):
            clock.now = 1_000_000_000 + int(slot * 1e8) + extra
            runner.step()

        runner.tick = tick
        runner.clock = clock
        return runner

    return make


def future(r, bias=0):
    p = r.policy_info.horizon - r.policy_info.n_obs_steps + 1
    return np.repeat((np.arange(p) * 0.01 + bias)[:, None], 19, axis=1)


def bootstrap(r):
    for slot in range(r.policy_info.n_obs_steps):
        r.tick(slot)
    assert r.model.future is not None
    q = r.policy_info.n_obs_steps - 1
    r.clock.now += 10_000_000
    r.model.complete(future(r))
    r.step()
    b = q + 1
    assert not r.robot.sent
    r.tick(b)
    return b


@pytest.mark.parametrize("mode", ["async", "rtc"])
@pytest.mark.parametrize("a,d,h", [(3, 2, 6), (2, 2, 5)])
def test_fixed_handoff_and_prefix(make_runner, mode, a, d, h):
    r = make_runner(mode=mode, a=a, d=d, h=h)
    b = bootstrap(r)
    for k in range(b + 1, b + a - d + 1):
        r.tick(k)
    assert len(r.model.submissions) == 2
    _, args, kwargs = r.model.submissions[-1]
    if mode == "rtc":
        assert kwargs["delay_steps"] == d
        # Prefix begins at the current, not the next, unsent control slot.
        np.testing.assert_allclose(kwargs["rtc_prefix"][0], (a - d) * 0.01)
    r.clock.now += 10_000_000
    r.model.complete(future(r, 0.5))
    r.step()
    for k in range(b + a - d + 1, b + a):
        r.tick(k)
    assert np.max(r.robot.sent[-1].arm_qpos) < 0.5
    r.tick(b + a)
    np.testing.assert_allclose(r.robot.sent[-1].arm_qpos, 0.5 + d * 0.01)
    assert any(e["event"] == "handoff" for e in r.events)


def test_sync_no_prefetch_and_wait(make_runner):
    r = make_runner(mode="sync")
    b = bootstrap(r)
    for k in range(b + 1, b + 3):
        r.tick(k)
    assert len(r.model.submissions) == 1
    count = len(r.robot.sent)
    r.tick(b + 3)
    assert len(r.model.submissions) == 2
    r.tick(b + 4)
    assert len(r.robot.sent) == count


def test_late_future_reclaimed_before_bootstrap(make_runner):
    r = make_runner()
    b = bootstrap(r)
    r.tick(b + 1)
    r.tick(b + 2)
    r.tick(b + 3)
    deadline = r.wait_started_ns
    assert r.plan is None and r.model.future is not None
    count = len(r.model.submissions)
    r.tick(b + 4)
    assert len(r.model.submissions) == count
    r.clock.now += 1_000_000
    r.model.complete(error=RuntimeError("late CUDA error"))
    r.step()
    assert r.model.reclaimed[-1].error is not None
    r.tick(b + 5)
    assert len(r.model.submissions) == count + 1 and r.wait_started_ns == deadline
    assert r.run_id is not None


def test_wait_timeout_not_reset_by_invalid_results(make_runner):
    r = make_runner(wait=0.35)
    r.tick(0)
    r.tick(1)
    r.clock.now += 1_000_000
    r.model.complete(future(r))
    r.execution = ExecutionConfig("async", 0.001, 0.35, 0.03, 2, 2.0)
    r.step()
    r.tick(2)
    r.tick(3)
    with pytest.raises(TimeoutError, match="WAIT"):
        r.tick(4)
    assert r.completed == 1 and r.run_id is None
    assert r.shared.run_ended_reason.value == RunEndReason.TIMEOUT


def test_owner_missed_slot_invalidates_early_result(make_runner):
    r = make_runner()
    b = bootstrap(r)
    r.tick(b + 1)
    r.clock.now += 1_000_000
    r.model.complete(future(r, 0.5))
    r.step()
    count = len(r.robot.sent)
    r.tick(b + 4)
    assert r.plan is None and len(r.robot.sent) == count


@pytest.mark.parametrize("mode", ["async", "rtc"])
def test_execute_false_consumes_without_accepted(make_runner, mode):
    r = make_runner(mode=mode, execute=False)
    b = bootstrap(r)
    assert not r.robot.sent and r.wait_started_ns is None
    r.tick(b + 1)
    r.clock.now += 1_000_000
    r.model.complete(future(r, 0.5))
    r.step()
    r.tick(b + 2)
    r.tick(b + 3)
    dispatch = [e for e in r.events if e["event"] == "dispatch"]
    assert all(e["arm"] == e["hand"] == 0 and e["logical_consume"] for e in dispatch)
    assert any(e["event"] == "handoff" for e in r.events)


@pytest.mark.parametrize("reason", [RunEndReason.OPERATOR, RunEndReason.QUIT])
@pytest.mark.parametrize("where", ["busy", "completed", "stop_error", "recorder_error"])
def test_end_reason_once(make_runner, reason, where):
    r = make_runner()
    r.tick(0)
    r.tick(1)
    request_policy_stop(r.shared, reason=reason)
    if where == "completed":
        r.model.complete(future(r))
    if where == "stop_error":

        def bad_stop():
            raise RuntimeError("stop failed")

        r.robot.stop = bad_stop
    if where == "recorder_error":

        def save(**kwargs):
            assert kwargs["reason"] == reason.name.lower()
            raise RuntimeError("writer failed")

        r.recorder = NS(save_episode=save)
    if where in {"stop_error", "recorder_error"}:
        with pytest.raises(RuntimeError):
            r.step()
    else:
        r.step()
    r._finish_episode("again")
    assert r.completed == 1 and r.shared.run_ended_reason.value == reason
    end = [e for e in r.events if e["event"] == "end"]
    assert len(end) == 1 and end[0]["reason"] == reason.name.lower()


def test_unknown_dispatch_ends_reservation(make_runner):
    r = make_runner()
    r.robot.result = DispatchResult(DispatchStatus.ACCEPTED, DispatchStatus.CRC_UNCONFIRMED)
    with pytest.raises(DispatchError, match="not confirmed") as failure:
        bootstrap(r)
    assert failure.value.result == r.robot.result
    assert r.run_id is None and r.completed == 1 and r.plan is None


@pytest.mark.parametrize("statuses", [(0, 0), (4, 0), (1, 4)])
@pytest.mark.parametrize("stop_fails", [False, True])
def test_dispatch_cancel_preserves_row_and_first_reason(make_runner, statuses, stop_fails):
    r = make_runner()
    order = []
    result = DispatchResult(*(DispatchStatus(value) for value in statuses))

    def send(command, *, valid_until_ns):
        raise DispatchInterrupted(result)

    def stop():
        order.append("stop")
        if stop_fails:
            raise RuntimeError("stop failed")

    def record(row, command=None, dispatch=None):
        if command is not None:
            order.append(dispatch)

    r.robot.send_action = send
    r.robot.stop = stop
    r._record = record
    with pytest.raises(DispatchInterrupted) as caught:
        bootstrap(r)
    assert caught.value.result is result
    assert order == ["stop", result]
    assert r.completed == 1 and r.run_id is None
    assert r.shared.run_ended_reason.value == RunEndReason.ESTOP
    ends = [event for event in r.events if event["event"] == "end"]
    assert len(ends) == 1 and ends[0]["reason"] == "estop"


def test_each_actual_modality_source_and_fk_dependency():
    row = ObservationRow(
        {"timestamp_ns": [900]},
        {"timestamp_ns": [700]},
        {"timestamp_ns": 100},
        None,
        None,
        999,
        200,
        4,
    )
    sources = policy_sources(row, ("joint_state", "eef_pose", "fingertip_points"))
    assert sources["eef_pose"] == (900,)
    assert sources["joint_state"] == (900, 700)
    assert not decision_is_fresh(sources, 1000, 200e-9)
    assert decision_is_fresh(policy_sources(row, ("eef_pose",)), 1000, 200e-9)
    assert policy_sources(row, ("point_cloud",))["point_cloud"] == (200,)


@pytest.mark.parametrize("a,d,h", [(2, 3, 8), (3, 2, 5), (0, 1, 8)])
def test_invalid_scheduling_config(make_runner, a, d, h):
    with pytest.raises(ValueError):
        make_runner(a=a, d=d, h=h)


def test_prepare_is_transactional_and_frozen_dynamic_rejection(make_runner):
    r = make_runner(mode="rtc")
    b = bootstrap(r)
    original = r.realizer.realize
    calls = []

    def realize(intent, current, previous):
        calls.append(previous)
        if len(calls) == 2:
            return ActionRealization(
                None,
                None,
                ik_result=IKResult(
                    success=False,
                    qpos=None,
                    reason="unreachable",
                    failure_kind=IKFailureKind.NO_SOLUTION_FOUND,
                ),
            )
        return original(intent, current, previous)

    r.realizer.realize = realize
    previous = r.previous_arm.copy()
    r.tick(b + 1)
    assert r.plan is None and len(r.model.submissions) == 1
    np.testing.assert_array_equal(r.previous_arm, previous)
    assert len(r.robot.sent) == 1
    r = make_runner(mode="rtc")
    b = bootstrap(r)
    r.realizer.frozen_is_valid = lambda *args: False
    r.realizer.rejection_reason = "self_collision"
    r.tick(b + 1)
    assert r.plan is None and len(r.robot.sent) == 1 and not r.query.valid
    assert any(e["event"] == "invalidate" and e["reason"] == "self_collision" for e in r.events)


def test_epoch_latch_cannot_be_replaced_by_later_revocation(make_runner):
    from dexmani_real.runtime.safety import revoke_motion

    r = make_runner()
    request_policy_stop(r.shared, reason=RunEndReason.QUIT)
    revoke_motion(r.shared, reason=RunEndReason.RUNTIME_SHUTDOWN)
    r._finish_episode("shutdown")
    assert [e for e in r.events if e["event"] == "end"][0]["reason"] == "quit"


def test_old_future_blocks_new_episode_reset(make_runner):
    r = make_runner(execute=False)
    r.tick(0)
    r.tick(1)
    request_policy_stop(r.shared)
    r.step()
    r.shared.start_request.value = True
    r._start_observation = r._read_observation
    r.step()
    assert len(r.model.submissions) == 1
    r.model.complete(error=RuntimeError("old"))
    r.step()
    assert [op for op, _, _ in r.model.submissions] == ["predict", "reset_episode"]
    r.model.complete()
    r.step()
    assert r.run_id is not None and r.completed == 1


def test_decision_age_not_refreshed_by_new_feedback(make_runner):
    r = make_runner(mode="sync", a=1)
    r.execution = ExecutionConfig("sync", 0.15, 1.0, 0.03)
    r.tick(0)
    r.tick(1)
    r.tick(2)
    r.model.complete(future(r))
    r.step()
    count = len(r.robot.sent)
    r.tick(3)
    assert len(r.robot.sent) == count and r.plan is None


def test_missing_history_does_not_refresh_wait(make_runner):
    r = make_runner(wait=0.25)
    r._read_observation = lambda: None
    for i in range(3):
        r.tick(i)
    with pytest.raises(TimeoutError, match="WAIT"):
        r.tick(3)
    assert r.completed == 1 and r.shared.run_ended_reason.value == RunEndReason.TIMEOUT


def test_eef_runner_prefix_uses_real_fk_and_same_joint_command(make_runner):
    from dexmani_real.config.experiment import ExperimentConfig
    from dexmani_real.deployment.runner import Plan
    from dexmani_real.planning.kinematics.arm_fk import compute_eef_pose_history_xarm_base
    from dexmani_real.robot.action import ActionRealizer

    r = make_runner(mode="rtc", n=1)
    cfg = ExperimentConfig()
    q = np.asarray(cfg.arm.home_qpos)
    hand = np.deg2rad(cfg.hand.home_qpos_deg)
    r.runtime = cfg
    r.policy_info.action_mode = "eef"
    r.realizer = ActionRealizer.for_mode(cfg, "eef")
    # Synthetic physical input, actual online IK and public FK; no SDK/device.
    row = ObservationRow(
        {"timestamp_ns": [r.clock.now], "qpos": q[None]},
        {"timestamp_ns": [r.clock.now], "qpos": hand[None]},
        None,
        None,
        None,
        r.clock.now,
    )
    target = np.concatenate((compute_eef_pose_history_xarm_base(q[None])[0], hand))
    actions = np.repeat(target[None], 8, axis=0)
    r.plan = Plan(actions, 0, 3, 1, policy_sources(row, ("joint_state",)), {})
    r.previous_arm = q.copy()
    prefix, commands = r._prepare_prefix(row, 0)
    assert prefix is not None
    for i, command in commands.items():
        expected = np.concatenate(
            (compute_eef_pose_history_xarm_base(command.arm_qpos[None])[0], command.hand_qpos)
        )
        np.testing.assert_array_equal(prefix[i], expected)
    np.testing.assert_array_equal(r.previous_arm, q)
    r.plan.frozen.update(commands)
    r.last_slot = 0
    r._execute_slot(row, 0)
    assert r.robot.sent[0] is commands[0]


def test_slow_observation_read_breaks_history_even_without_full_slot_skip(make_runner):
    r = make_runner()
    read = r._read_observation

    def slow():
        r.clock.now += 50_000_000
        return read()

    r._read_observation = slow
    r.tick(0)
    assert not r.history.rows and not r.model.submissions
    r._read_observation = read
    r.tick(1)
    assert not r.model.submissions
    r.tick(2)
    assert len(r.model.submissions) == 1


@pytest.mark.parametrize("mode", ["async", "rtc"])
@pytest.mark.parametrize("first_reason", [None, RunEndReason.OPERATOR, RunEndReason.QUIT])
def test_eef_prefix_technical_failure_ends_run_without_partial_publication(
    make_runner, mode, first_reason
):
    r = make_runner(mode=mode)
    b = bootstrap(r)
    plan = r.plan
    actions = np.pad(plan.actions, ((0, 0), (0, 2)))
    original_actions = actions.copy()
    plan.actions = actions
    r.policy_info.action_mode = "eef"
    previous = r.previous_arm.copy()
    saved = []
    r.recorder = NS(
        check_error=lambda: None,
        save_episode=lambda **kwargs: saved.append(kwargs),
        close=lambda: None,
    )
    calls = []

    def unexpected_record(*args):
        raise AssertionError("technical IK failure continued execution")

    r._record = unexpected_record

    def realize(intent, current, prior):
        np.testing.assert_array_equal(r.previous_arm, previous)
        calls.append(prior.copy())
        if len(calls) == 2:
            if first_reason is not None:
                request_policy_stop(r.shared, reason=first_reason)
            return ActionRealization(
                None,
                None,
                ik_result=IKResult(
                    success=False,
                    qpos=None,
                    reason="nonfinite solver output",
                    failure_kind=IKFailureKind.INVALID_OUTPUT,
                ),
            )
        return ActionRealization(np.full(7, 0.01), np.zeros(12))

    r.realizer.realize = realize
    r.clock.now = 1_000_000_000 + int((b + 1) * 1e8)
    with pytest.raises(RuntimeError, match="online IK technical failure: nonfinite solver output"):
        r.run()
    assert r.run_id is None and r.completed == 1
    assert r.robot.stops == 1 and len(r.robot.sent) == 1
    assert [op for op, _, _ in r.model.submissions] == ["predict"]
    assert r.model.future is None and r.model.closed
    assert not plan.frozen and r.query is None and r.plan is None
    np.testing.assert_array_equal(plan.actions, original_actions)
    np.testing.assert_array_equal(calls[0], previous)
    assert r.previous_arm is None
    reason = first_reason or RunEndReason.POLICY_FAILURE
    assert saved == [{"reason": reason.name.lower(), "details": []}]
    end = [e for e in r.events if e["event"] == "end"]
    assert len(end) == 1 and end[0]["reason"] == reason.name.lower()
    assert "nonfinite solver output" in end[0]["detail"]


@pytest.mark.parametrize("mode", ["async", "rtc"])
@pytest.mark.parametrize(
    "failure_kind", [IKFailureKind.NO_SOLUTION_FOUND, IKFailureKind.NO_VALID_CANDIDATE]
)
def test_eef_prefix_no_solution_retains_wait_and_bootstrap(make_runner, mode, failure_kind):
    r = make_runner(mode=mode)
    b = bootstrap(r)
    plan = r.plan
    plan.actions = np.pad(plan.actions, ((0, 0), (0, 2)))
    r.policy_info.action_mode = "eef"
    previous = r.previous_arm.copy()
    r.realizer.realize = lambda *args: ActionRealization(
        None,
        None,
        ik_result=IKResult(
            success=False, qpos=None, reason="unreachable", failure_kind=failure_kind
        ),
    )
    r.tick(b + 1)
    deadline = r.wait_started_ns
    assert r.run_id is not None and r.completed == 0 and r.robot.stops == 0
    assert r.plan is None and r.query is None and not plan.frozen
    assert len(r.model.submissions) == 1
    np.testing.assert_array_equal(r.previous_arm, previous)
    r.tick(b + 2)
    assert [op for op, _, _ in r.model.submissions] == ["predict", "predict"]
    assert r.query.handoff_slot is None and r.wait_started_ns == deadline


def test_future_finishes_while_reading_observation(make_runner):
    r = make_runner(mode="async", n=1)
    b = bootstrap(r)
    r.tick(b + 1)  # prefetch for b+3
    r.tick(b + 2)
    count = len(r.robot.sent)
    read = r._read_observation

    def read_and_complete():
        r.clock.now += 5_000_000
        r.model.complete(future(r, 0.5))
        return read()

    r._read_observation = read_and_complete
    r.tick(b + 3)
    assert len(r.robot.sent) == count + 1
    assert not any(e.get("reason") == "handoff_miss" for e in r.events)


@pytest.mark.parametrize("budget", ["wait", "episode"])
@pytest.mark.parametrize("late", [False, True])
def test_dispatch_return_checks_total_budget(make_runner, budget, late):
    r = make_runner(wait=0.205 if budget == "wait" else 1.0)
    if budget == "episode":
        r.max_running_s = 0.205
    rows = []
    r._record = lambda row, command=None, result=None: rows.append((command, result))
    send = r.robot.send_action

    def delayed(command, **kwargs):
        result = send(command, **kwargs)
        r.clock.now += 40_000_000 if late else 4_000_000
        return result

    r.robot.send_action = delayed
    if late and budget == "wait":
        with pytest.raises(TimeoutError, match="WAIT"):
            bootstrap(r)
    else:
        bootstrap(r)
    assert len([entry for entry in rows if entry[0] is not None]) == 1
    assert rows[-1][1] == r.robot.result
    if late:
        assert r.run_id is None
        assert r.shared.run_ended_reason.value == int(RunEndReason.TIMEOUT)
        assert r.robot.stops == 1
        r.tick(3)
        assert len(r.robot.sent) == 1
    else:
        assert r.run_id == 1
        assert r.wait_started_ns is None
        assert not r.robot.stops


@pytest.mark.parametrize("cause", [RunEndReason.OPERATOR, RunEndReason.ESTOP])
def test_late_return_preserves_already_latched_cause(make_runner, cause):
    from dexmani_real.runtime.safety import revoke_motion_if_run_id

    r = make_runner(wait=0.205)

    def send(command, **kwargs):
        r.robot.sent.append(command)
        revoke_motion_if_run_id(r.shared, r.run_id, reason=cause)
        r.clock.now += 40_000_000
        if cause == RunEndReason.ESTOP:
            raise DispatchInterrupted(r.robot.result)
        return r.robot.result

    r.robot.send_action = send
    if cause == RunEndReason.ESTOP:
        with pytest.raises(DispatchInterrupted):
            bootstrap(r)
    else:
        with pytest.raises(TimeoutError, match="WAIT"):
            bootstrap(r)
    assert r.shared.run_ended_reason.value == int(cause)
    assert r.completed == 1


@pytest.mark.parametrize("stop_fails,record_fails", [(True, False), (False, True), (True, True)])
def test_late_return_cleanup_records_once(make_runner, stop_fails, record_fails):
    r = make_runner(wait=0.205)
    rows, saved = [], []

    def send(command, **kwargs):
        r.robot.sent.append(command)
        r.clock.now += 40_000_000
        return r.robot.result

    def stop():
        r.robot.stops += 1
        if stop_fails:
            raise TimeoutError("STOP_TIMEOUT")

    def record(row, command=None, result=None):
        if command is not None:
            rows.append(result)
            if record_fails:
                raise OSError("writer failed")

    r.robot.send_action = send
    r.robot.stop = stop
    r._record = record
    r.recorder = NS(check_error=lambda: None, save_episode=lambda **kw: saved.append(kw))
    with pytest.raises((TimeoutError, OSError)):
        bootstrap(r)
    assert rows == [r.robot.result]
    assert len(saved) == 1 and saved[0]["reason"] == "timeout"
    assert {e["stage"] for e in saved[0]["details"]} == ({"stop"} if stop_fails else set()) | (
        {"recording"} if record_fails else set()
    )
    assert r.completed == 1 and r.run_id is None
    assert len([e for e in r.events if e["event"] == "dispatch"]) == 1


@pytest.mark.parametrize("late_device,expected", [("arm", (1, 0)), ("hand", (1, 1))])
def test_wait_deadline_between_real_sdk_calls(make_runner, monkeypatch, late_device, expected):
    from test_review_remediation import fake_robot

    robot, clock, _ = fake_robot(monkeypatch)
    r = make_runner(wait=0.205)
    robot.shared = r.shared
    calls = []

    def call(name):
        calls.append(name)
        if name == late_device:
            r.clock.now += 40_000_000
        return 0 if name == "arm" else DispatchStatus.ACCEPTED

    robot.arm.servo = lambda q: call("arm")
    robot.hand.send_action = lambda q: call("hand")
    r.robot.send_action = robot.send_action
    rows = []
    r._record = lambda row, command=None, result=None: (
        rows.append(result) if command is not None else None
    )
    with pytest.raises(TimeoutError, match="WAIT"):
        bootstrap(r)
    assert len(rows) == 1
    assert (rows[0].arm, rows[0].hand) == expected
    assert calls == (["arm"] if late_device == "arm" else ["arm", "hand"])
    assert r.shared.run_ended_reason.value == int(RunEndReason.TIMEOUT)
    assert r.completed == 1 and r.run_id is None


@pytest.mark.parametrize("mode", ["async", "rtc"])
@pytest.mark.parametrize("limit", ["feedback", "decision", "duration_tie"])
def test_native_dispatch_uses_original_action_deadline(make_runner, monkeypatch, mode, limit):
    from dataclasses import replace

    from test_review_remediation import fake_robot

    robot, _, _ = fake_robot(monkeypatch)
    r = make_runner(mode=mode, n=1)
    robot.shared = r.shared
    if limit == "feedback":
        r.runtime.arm.feedback_max_age_s = 0.005
    elif limit == "decision":
        r.execution = replace(r.execution, max_decision_age_s=0.105)
    else:
        # Slot endpoint is inclusive, hence the extra ns in the SDK deadline.
        r.max_running_s = 0.130000001

    def arm(q):
        r.clock.now += 40_000_000
        return 0

    robot.arm.servo = arm
    r.robot.send_action = robot.send_action
    with pytest.raises(DispatchError) as caught:
        bootstrap(r)
    assert caught.value.cause == "deadline_expired"
    assert (caught.value.result.arm, caught.value.result.hand) == (1, 0)
    event = next(e for e in r.events if e["event"] == "dispatch")
    assert event["valid_until_ns"] == r.started_ns + (
        130_000_001 if limit == "duration_tie" else 105_000_001
    )
    assert event["cause"] == "deadline_expired"
    assert r.completed == 1 and r.robot.stops == 1


@pytest.mark.parametrize("reason", [RunEndReason.OPERATOR, RunEndReason.QUIT])
@pytest.mark.parametrize("evidence", ["current", "stale", "unexplained", "crc", "stop", "record"])
def test_cancellation_requires_current_cause_and_clean_evidence(make_runner, reason, evidence):
    r = make_runner()
    result = DispatchResult(DispatchStatus.ACCEPTED, DispatchStatus.NOT_CALLED)
    if evidence == "crc":
        result = DispatchResult(DispatchStatus.ACCEPTED, DispatchStatus.CRC_UNCONFIRMED)

    def send(command, **kwargs):
        request_policy_stop(r.shared, reason=reason)
        if evidence == "stale":
            r.shared.run_ended_id.value -= 1
        raise DispatchError(
            "boundary",
            result,
            revoked=True,
            cause=None if evidence == "unexplained" else "authority_revoked",
        )

    def fail(*args, **kwargs):
        raise OSError("cleanup failed")

    r.robot.send_action = send
    if evidence == "stop":
        r.robot.stop = fail
    elif evidence == "record":
        r._record = lambda row, command=None, result=None: fail() if command is not None else None
    if evidence == "current":
        bootstrap(r)
    else:
        with pytest.raises(OSError if evidence in {"stop", "record"} else DispatchError):
            bootstrap(r)
    assert r.completed == 1 and r.run_id is None


def test_sync_crc_tolerance_is_unchanged(make_runner):
    r = make_runner(mode="sync")
    r.robot.result = DispatchResult(DispatchStatus.ACCEPTED, DispatchStatus.CRC_UNCONFIRMED)
    bootstrap(r)
    assert r.run_id == 1 and r.completed == 0 and r.wait_started_ns is None
    assert not r.robot.stops


@pytest.mark.parametrize("cancelled", [False, True])
def test_late_return_writer_finalization_failure_keeps_dispatch_and_end(make_runner, cancelled):
    from dexmani_real.recording.recorder import RecordingError

    r = make_runner(wait=0.205)
    rows = []
    r._record = lambda row, command=None, result=None: (
        rows.append(result) if command is not None else None
    )

    def send(command, **kwargs):
        r.clock.now += 40_000_000
        if cancelled:
            raise DispatchInterrupted(r.robot.result)
        return r.robot.result

    def save(**kwargs):
        raise RecordingError("publish failed; staging retained")

    r.robot.send_action = send
    r.recorder = NS(check_error=lambda: None, save_episode=save)
    with pytest.raises(DispatchInterrupted if cancelled else TimeoutError) as failure:
        bootstrap(r)
    assert isinstance(failure.value.__cause__, RecordingError)
    assert rows == [r.robot.result]
    events = r.events
    assert len([e for e in events if e["event"] == "dispatch"]) == 1
    assert len([e for e in events if e["event"] == "end"]) == 1
    assert r.completed == 1 and r.run_id is None
    assert r.shared.run_ended_reason.value == (
        RunEndReason.ESTOP if cancelled else RunEndReason.TIMEOUT
    )


@pytest.mark.parametrize(
    "stop_fails,writer_fails,sidecar_fails",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, True, False),
        (False, False, True),
        (True, True, True),
    ],
)
def test_attempt_finalization_orders_stop_before_sidecar(
    make_runner, tmp_path, monkeypatch, stop_fails, writer_fails, sidecar_fails
):
    import json

    from dexmani_real.recording.results import SessionResults

    r = make_runner()
    r.results = SessionResults(tmp_path, "policy")
    attempt = r.results.prepare(recording=True)
    r.results.entered(r.run_id)
    order = []

    def stop():
        assert r.run_id is None
        order.append("stop")
        if stop_fails:
            raise RuntimeError("stop failed")

    def save(**kw):
        order.append("raw")
        if writer_fails:
            raise RuntimeError("writer failed")

    r.robot.stop = stop
    r.recorder = NS(save_episode=save, written_frames=0, staging_path=tmp_path / "staging")
    native = np.savez

    def sidecar(*a, **kw):
        assert order == ["stop", "raw"]
        order.append("sidecar")
        if sidecar_fails:
            raise OSError("sidecar failed")
        native(*a, **kw)

    monkeypatch.setattr(np, "savez", sidecar)
    r.query_arrays = {"query_1": np.ones((2, 19), dtype=np.float16)}
    if stop_fails or writer_fails or sidecar_fails:
        message = (
            "stop failed" if stop_fails else "writer failed" if writer_fails else "sidecar failed"
        )
        with pytest.raises((RuntimeError, OSError), match=message):
            r._finish_episode("timeout", run_end_reason=RunEndReason.TIMEOUT)
    else:
        r._finish_episode("timeout", run_end_reason=RunEndReason.TIMEOUT)
    result = json.loads((tmp_path / "attempts" / f"{attempt}.json").read_text())
    assert result["entered_running"] and result["row_count"] == 0
    assert result["recording_status"] == ("failed" if writer_fails else "empty")
    assert result["termination_reason"] == "timeout"
    assert len(result["finalization_errors"]) == sum((stop_fails, writer_fails, sidecar_fails))
    assert r.results.session["artifact_errors"] == result["finalization_errors"]
    if sidecar_fails:
        assert result["query_sidecar"] is None
        assert result["finalization_errors"][-1]["stage"] == "query_sidecar"
    else:
        with np.load(result["query_sidecar"], allow_pickle=False) as arrays:
            assert arrays["query_1"].dtype == np.float16
    r._finish_episode("again")
    assert order == ["stop", "raw", "sidecar"]


@pytest.mark.parametrize("dtype", [np.float16, np.float64])
@pytest.mark.parametrize(
    "value", [np.nan, np.inf, -np.inf, 0.04], ids=["nan", "posinf", "neginf", "finite"]
)
def test_query_evidence_preserves_floating_output(make_runner, tmp_path, dtype, value):
    import json

    from dexmani_real.recording.results import SessionResults

    r = make_runner(mode="sync")
    r.results = SessionResults(tmp_path, "policy")
    attempt = r.results.prepare(recording=False)
    r.results.entered(r.run_id)
    r.tick(0)
    r.tick(1)
    prediction = future(r).astype(dtype)
    prediction[1, 3] = value
    expected = prediction.copy()
    query_key = f"query_{r.query_id}"
    stop = r.robot.stop

    def stop_and_release_prediction():
        assert r.run_id is None and r.shared.safety_state.value != int(SafetyState.RUNNING)
        # A producer-owned buffer can change after receipt; the archive owns its copy.
        prediction.fill(-12)
        stop()

    r.robot.stop = stop_and_release_prediction
    r.model.complete(prediction)
    if np.isfinite(value):
        r.step()
        assert not r.robot.sent
        r.tick(2)
        assert len(r.robot.sent) == 1
        np.testing.assert_array_equal(r.robot.sent[0].arm_qpos, expected[0, :7])
        np.testing.assert_array_equal(r.robot.sent[0].hand_qpos, expected[0, 7:])
        r._finish_episode("operator", run_end_reason=RunEndReason.OPERATOR)
    else:
        with pytest.raises(ValueError, match="finite floating point"):
            r.run()
        assert not r.robot.sent
    record = json.loads((tmp_path / "attempts" / f"{attempt}.json").read_text())
    assert record["termination_reason"] == ("operator" if np.isfinite(value) else "policy_failure")
    assert record["finalization_errors"] == []
    json.dumps(record, allow_nan=False)
    with np.load(record["query_sidecar"], allow_pickle=False) as arrays:
        assert arrays.files == [query_key]
        archived = arrays[query_key]
        assert archived.dtype == dtype and archived.shape == (7, 19)
        np.testing.assert_array_equal(archived, expected)
        assert np.signbit(archived[1, 3]) == np.signbit(expected[1, 3])
    assert r.robot.stops == 1


@pytest.mark.parametrize("value", [np.inf, 0.02], ids=["nonfinite", "finite"])
def test_retired_query_evidence_stays_out_of_next_attempt(make_runner, tmp_path, value):
    import json
    from pathlib import Path

    from dexmani_real.recording.results import SessionResults

    r = make_runner(execute=False)
    r.results = SessionResults(tmp_path, "policy")
    first = r.results.prepare(recording=False)
    r.results.entered(r.run_id)
    r.tick(0)
    r.tick(1)
    r._finish_episode("operator", run_end_reason=RunEndReason.OPERATOR)
    assert r.robot.stops == 1 and not r.model.future.done()
    first_path = tmp_path / "attempts" / f"{first}.json"
    first_bytes = first_path.read_bytes()
    first_sidecar = Path(json.loads(first_bytes)["query_sidecar"])
    sidecar_bytes = first_sidecar.read_bytes()
    r.shared.start_request.value = True
    r._start_observation = r._read_observation
    r.step()
    assert r.results.attempt is None  # Pending old Future still prevents a new reset.
    prediction = future(r)
    prediction[1, 3] = value
    r.model.complete(prediction)
    r.step()
    second = r.results.attempt["attempt_id"]
    r.model.complete()  # New episode reset.
    r.step()
    r._finish_episode("operator", run_end_reason=RunEndReason.OPERATOR)
    assert first_path.read_bytes() == first_bytes and first_sidecar.read_bytes() == sidecar_bytes
    record = json.loads((tmp_path / "attempts" / f"{second}.json").read_text())
    with np.load(record["query_sidecar"], allow_pickle=False) as arrays:
        assert arrays.files == []
    assert r.results.session["retired_queries"][0]["run_id"] != record["run_id"]
    assert not r.robot.sent


@pytest.mark.parametrize("kind", ["shape", "integer", "object"])
def test_query_evidence_rejects_unsupported_arrays(make_runner, tmp_path, kind):
    import json

    from dexmani_real.recording.results import SessionResults

    r = make_runner(mode="sync")
    r.results = SessionResults(tmp_path, "policy")
    attempt = r.results.prepare(recording=False)
    r.results.entered(r.run_id)
    r.tick(0)
    r.tick(1)
    prediction = (
        np.zeros((7, 18))
        if kind == "shape"
        else np.zeros((7, 19), dtype=np.int64 if kind == "integer" else object)
    )
    r.model.complete(prediction)
    with pytest.raises(ValueError, match="finite floating point"):
        r.run()
    assert not r.robot.sent
    record = json.loads((tmp_path / "attempts" / f"{attempt}.json").read_text())
    diagnostic = next(e for e in record["trace"] if e["event"] == "query_result")
    assert diagnostic["shape"] == list(prediction.shape)
    assert diagnostic["dtype"] == str(prediction.dtype)
    with np.load(record["query_sidecar"], allow_pickle=False) as arrays:
        assert arrays.files == []


@pytest.mark.parametrize("cancel", [False, True])
def test_start_result_io_precedes_final_fresh_observation(
    make_runner, tmp_path, monkeypatch, cancel
):
    import json

    from dexmani_real.recording.results import SessionResults

    r = make_runner()
    r.run_id = None
    r.shared.safety_state.value = int(SafetyState.ARMED)
    r.shared.start_request.value = True
    r.results = SessionResults(tmp_path, "policy")
    reads = []

    def observe():
        reads.append(r.clock.now)
        if len(reads) == 2:
            assert r.results.attempt["entered_running"] is None
            assert r.clock.now > reads[0]
        return r._read_observation()

    r._start_observation = observe
    r._begin_episode()
    attempt = r.results.attempt["attempt_id"]
    r.model.complete()
    r._poll_model()
    r.clock.now += 1_000_000_000
    if cancel:
        r.shared.quit_requested.value = True
    r._begin_episode()
    if cancel:
        assert r.completed == 0 and r.run_id is None
        result = json.loads((tmp_path / "attempts" / f"{attempt}.json").read_text())
        assert (
            result["entered_running"] is False and result["termination_reason"] == "start_cancelled"
        )
    else:
        assert len(reads) == 2 and r.run_id is not None
        assert r.results.attempt["entered_running"] is True
        r._finish_episode("operator")


def test_start_result_failure_never_grants_motion(make_runner, tmp_path, monkeypatch):
    import dexmani_real.recording.results as records

    r = make_runner()
    r.run_id = None
    r.shared.safety_state.value = int(SafetyState.ARMED)
    r.shared.start_request.value = True
    r.results = records.SessionResults(tmp_path, "policy")
    r._start_observation = r._read_observation
    monkeypatch.setattr(
        records, "atomic_json_dump", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full"))
    )
    with pytest.raises(OSError):
        r._begin_episode()
    assert r.run_id is None and r.shared.safety_state.value == int(SafetyState.ARMED)
    assert r.model.future is None


def test_joint_rejection_without_ik_result_remains_wait(make_runner):
    r = make_runner(mode="sync")
    bootstrap(r)
    r.realizer.realize = lambda *a: ActionRealization(None, np.zeros(12), rejection_reason="jump")
    before = len(r.robot.sent)
    r.tick(r.last_slot + 1)
    assert len(r.robot.sent) == before and r.run_id is not None and r.plan is None
    assert any(e["event"] == "invalidate" and e["reason"] == "jump" for e in r.events)


@pytest.mark.parametrize("cancel_at", ["reset", "metadata", "recorder_start"])
def test_cancelled_attempt_never_reuses_previous_raw_evidence(
    make_runner, tmp_path, monkeypatch, cancel_at
):
    import json

    from test_policy_recording import start_recording

    from dexmani_real.recording.recorder import RecordingError
    from dexmani_real.recording.results import SessionResults

    r = make_runner()
    r.recorder = start_recording(tmp_path / "raw")
    previous = r.recorder.save_episode(reason="operator")
    previous_data = (previous / "data.h5").read_bytes()
    assert r.recorder.written_frames == 1
    r.recording_config = NS(task_label="synthetic")
    r.run_id = None
    r.completed = 1
    r.shared.safety_state.value = int(SafetyState.ARMED)
    r.shared.start_request.value = True
    r.results = SessionResults(tmp_path / "results", "policy")
    r._start_observation = r._read_observation
    r._begin_episode()
    attempt = r.results.attempt["attempt_id"]
    r.model.complete()
    r._poll_model()
    if cancel_at == "reset":
        r.shared.quit_requested.value = True
        r._begin_episode()
    else:

        def metadata(*args, **kwargs):
            if cancel_at == "metadata":
                raise ValueError("missing metadata")
            return dict(
                collection_source="policy_rollout",
                camera_geometry=None,  # Invalid before any new staging is created.
                camera_T_xarm_base_from_color=None,
                depth_scale=0.001,
                handbase_position_eef_m=np.zeros(3),
                handbase_quat_eef_wxyz=np.array([1.0, 0, 0, 0]),
            )

        monkeypatch.setattr("dexmani_real.deployment.runner.snapshot_recording_metadata", metadata)
        with pytest.raises(ValueError if cancel_at == "metadata" else RecordingError):
            r._begin_episode()
    record = json.loads((tmp_path / "results" / "attempts" / f"{attempt}.json").read_text())
    assert record["entered_running"] is False and record["row_count"] == 0
    assert record["raw_path"] is None and record["staging_path"] is None
    assert record["termination_reason"] == "start_cancelled"
    assert r.completed == 1 and r.run_id is None
    assert (previous / "data.h5").read_bytes() == previous_data


def test_record_event_maps_queue_row_not_slot(make_runner):
    r = make_runner()
    r.last_slot = 17
    r.recorder = NS(add_frame=lambda frame: 0)
    row = r._read_observation()
    from dexmani_real.ipc.schema import ARM_STATE_DTYPE, HAND_STATE_DTYPE
    from dexmani_real.runtime.observation import ObservationRow

    arm, hand = np.zeros(1, dtype=ARM_STATE_DTYPE), np.zeros(1, dtype=HAND_STATE_DTYPE)
    row = ObservationRow(
        arm,
        hand,
        dict(
            timestamp_ns=1,
            color_frame_number=1,
            depth_frame_number=1,
            rgb=np.zeros((4, 4, 3), np.uint8),
            depth=np.ones((4, 4), np.uint16),
        ),
        None,
        None,
        1,
    )
    r._record(row)
    assert r.events[-1]["raw_row_index"] == 0 and r.events[-1]["slot"] == 17
    r.recorder.add_frame = lambda frame: None
    count = len(r.events)
    r._record(row)
    assert len(r.events) == count


@pytest.fixture
def timed_owner(make_runner, tmp_path, monkeypatch):
    from dexmani_real.deployment.runner import Plan
    from dexmani_real.ipc.schema import ARM_STATE_DTYPE, HAND_STATE_DTYPE
    from dexmani_real.recording.results import SessionResults

    r = make_runner(n=1)
    effects = NS(
        read_error=False,
        missing=False,
        record_error=False,
        stop_error=False,
        archive_error=False,
        send_delay_ns=0,
        cancelled=False,
        nonfinite_joint=False,
        frames=[],
    )
    r.results = SessionResults(tmp_path, "policy")
    attempt = r.results.prepare(recording=True)
    r.results.entered(r.run_id)
    native_savez = np.savez
    native_write = r.results._write_attempt

    def sidecar(*args, **kwargs):
        r.clock.now += 50_000_000
        return native_savez(*args, **kwargs)

    def write_attempt(attempt):
        r.clock.now += 70_000_000
        return native_write(attempt)

    monkeypatch.setattr(np, "savez", sidecar)
    monkeypatch.setattr(r.results, "_write_attempt", write_attempt)
    r.plan = Plan(future(r), 0, 3, 1, {"arm": (r.clock.now,), "hand": (r.clock.now,)}, {})

    def read():
        r.clock.now += 10_000_000
        if effects.read_error:
            raise RuntimeError("read primary")
        if effects.missing:
            return None
        arm, hand = np.zeros(1, dtype=ARM_STATE_DTYPE), np.zeros(1, dtype=HAND_STATE_DTYPE)
        arm["timestamp_ns"] = hand["timestamp_ns"] = r.clock.now
        if effects.nonfinite_joint:
            arm["qpos"] = np.nan
        return ObservationRow(
            arm,
            hand,
            dict(
                timestamp_ns=r.clock.now,
                color_frame_number=1,
                depth_frame_number=1,
                rgb=np.zeros((4, 4, 3), np.uint8),
                depth=np.ones((4, 4), np.uint16),
            ),
            None,
            None,
            r.clock.now,
        )

    def add(frame):
        r.clock.now += 4_000_000
        if effects.record_error:
            raise RuntimeError("record primary")
        effects.frames.append(frame)
        return len(effects.frames) - 1

    def save(**kwargs):
        assert r.robot.stops == 1
        assert len([e for e in r.events if e["event"] == "owner_tick"]) == 1
        r.clock.now += 100_000_000
        if effects.archive_error:
            raise OSError("archive secondary")

    def stop():
        assert not r._has_motion_authority()
        if r._active_tick is not None:
            assert not any(e["event"] == "owner_tick" for e in r.events)
        r.robot.stops += 1
        r.clock.now += 2_000_000
        if effects.stop_error:
            raise RuntimeError("stop secondary")

    native_send = r.robot.send_action

    def send(command, **kwargs):
        result = native_send(command, **kwargs)
        r.clock.now += effects.send_delay_ns
        if effects.cancelled:
            raise DispatchInterrupted(result)
        return result

    r._read_observation = read
    r.robot.send_action = send
    r.robot.stop = stop
    r.recorder = NS(
        add_frame=add,
        save_episode=save,
        check_error=lambda: None,
        close=lambda: None,
        written_frames=0,
        staging_path=None,
    )
    r.clock.now += 5_000_000
    return r, effects, tmp_path / "attempts" / f"{attempt}.json"


@pytest.mark.parametrize(
    "case,duration_ms,rows",
    [
        ("normal", 14, 1),
        ("wait", 14, 1),
        ("missing", 10, 0),
        ("missed", 14, 1),
        ("read_error", 10, 0),
        ("realize_error", 10, 0),
        ("record_error", 14, 0),
        ("build_error", 10, 0),
        ("before_dispatch", 12, 0),
        ("partial", 16, 1),
        ("timeout", 26, 1),
        ("unconfirmed", 16, 1),
        ("cancelled", 16, 1),
        ("stop_error", 16, 1),
        ("archive_error", 16, 1),
    ],
)
def test_owner_tick_saved_once_with_consistent_interval(timed_owner, case, duration_ms, rows):
    import json

    from dexmani_real.deployment.timing import summarize_trace

    r, effects, path = timed_owner
    if case == "wait":
        r.plan = None
    elif case == "missed":
        r.last_slot = 0
        r.next_step_ns = r.started_ns + r.dt_ns
        r.clock.now += 2 * r.dt_ns
    elif case == "realize_error":

        def fail(_):
            raise RuntimeError("realize primary")

        r.realizer.collision_model.check_self_collision = fail
    elif case == "build_error":
        r.plan = None
        effects.nonfinite_joint = True
    elif case == "before_dispatch":

        def quit_during_realization(_):
            r.shared.quit_requested.value = True
            return False

        r.realizer.collision_model.check_self_collision = quit_during_realization
    elif hasattr(effects, case):
        setattr(effects, case, True)
    if case == "timeout":
        r.max_running_s = 0.020
        effects.send_delay_ns = 10_000_000
    if case in {"unconfirmed", "stop_error", "archive_error"}:
        r.robot.result = DispatchResult(DispatchStatus.ACCEPTED, DispatchStatus.CRC_UNCONFIRMED)
    elif case == "partial":
        r.robot.result = DispatchResult(DispatchStatus.ACCEPTED, DispatchStatus.NOT_CALLED)
    if case in {"read_error", "realize_error", "record_error", "build_error"}:
        # Cleanup failures must not replace the primary tick exception.
        effects.stop_error = effects.archive_error = True
        with pytest.raises(
            ValueError if case == "build_error" else RuntimeError,
            match="Nonfinite" if case == "build_error" else f"{case.split('_')[0]} primary",
        ):
            r.run()
    elif case in {"partial", "unconfirmed"}:
        with pytest.raises(DispatchError, match="not confirmed"):
            r.step()
    elif case in {"stop_error", "archive_error", "cancelled"}:
        with pytest.raises(DispatchInterrupted if case == "cancelled" else (RuntimeError, OSError)):
            r.step()
    else:
        r.step()
    if r.run_id is not None:
        r._finish_episode("test_end")
    saved = json.loads(path.read_text())
    ticks = [e for e in saved["trace"] if e["event"] == "owner_tick"]
    assert len(ticks) == 1
    tick = ticks[0]
    assert tick["run_id"] == saved["run_id"] == 1
    assert tick["slot"] == (2 if case == "missed" else 0)
    assert tick["scheduled_ns"] == r.started_ns + tick["slot"] * r.dt_ns
    assert tick["started_ns"] == tick["scheduled_ns"] + 5_000_000
    assert tick["lateness_ns"] == 5_000_000
    assert tick["duration_ns"] == duration_ms * 1_000_000
    assert [e for e in r.events if e["event"] == "owner_tick"] == ticks
    assert len(effects.frames) == rows
    assert len([e for e in saved["trace"] if e["event"] == "record_submitted"]) == rows
    summary = summarize_trace(saved["trace"])["seconds"]
    assert summary["owner_tick"]["p50"] == pytest.approx(duration_ms / 1000)
    assert summary["slot_lateness"]["p50"] == pytest.approx(0.005)
    if case in {"timeout", "unconfirmed", "cancelled", "stop_error", "archive_error", "partial"}:
        dispatch = [e for e in saved["trace"] if e["event"] == "dispatch"]
        assert len(dispatch) == 1 and dispatch[0]["arm"] == int(DispatchStatus.ACCEPTED)
        assert dispatch[0]["hand"] == int(r.robot.result.hand)


def test_idle_and_not_due_do_not_create_ticks(make_runner):
    r = make_runner()
    r.next_step_ns = r.clock.now + r.dt_ns
    r.step()
    r.run_id = None
    r.step()
    assert not any(e["event"] == "owner_tick" for e in r.events)


def test_tick_trace_failure_does_not_replace_read_error(timed_owner, caplog):
    r, effects, _ = timed_owner

    class BrokenTickTrace(list):
        def append(self, event):
            if event["event"] == "owner_tick":
                raise RuntimeError("telemetry secondary")
            super().append(event)

    r.events = BrokenTickTrace()
    effects.read_error = True
    with pytest.raises(RuntimeError, match="read primary"):
        r.run()
    assert r.robot.stops == 1 and r._active_tick is None
    assert "owner tick trace failed" in caplog.text


@pytest.mark.parametrize(
    "field,validity,payload",
    [
        ("contact_force", "tactile_aggregate_valid", "tactile_aggregate"),
        ("tactile_force", "tactile_dense_valid", "tactile_dense"),
    ],
)
def test_required_tactile_diagnostic_is_local_and_throttled(
    timed_owner, monkeypatch, caplog, field, validity, payload
):
    from dataclasses import replace

    from dexmani_real.deployment import observation
    from dexmani_real.utils.log import ThrottledWarner, get_logger

    r, _, _ = timed_owner
    r.clock.now = r.started_ns = r.wait_started_ns = 10_000_000_000
    monkeypatch.setattr(
        observation,
        "_warn_unavailable",
        ThrottledWarner(logger=get_logger(observation.__name__)),
    )
    row = r._read_observation()
    row.hand["tactile_aggregate"] = np.nan
    row.hand["tactile_dense"] = np.nan
    r.policy_info.observation_fields = ("joint_state", field)
    r.policy_info.n_obs_steps = 2
    for _ in range(4):
        assert not r._submit([row, row], 0)
    assert r.model.future is None and not r.events
    warnings = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1 and field in warnings[0] and "history" in warnings[0]
    assert "tare" not in warnings[0].lower()
    r.clock.now += 5_000_000_000
    assert not r._submit([row, row], 0)
    assert len([record for record in caplog.records if record.levelname == "WARNING"]) == 2

    caplog.clear()
    row.hand[validity] = True
    row.hand[payload] = 1.0
    result = observation.build_policy_observation([row, row], r.policy_info)
    assert set(result) == {"joint_state", field}
    np.testing.assert_array_equal(result[field], np.ones_like(result[field]))
    assert not caplog.records
    row.hand[payload] = np.nan
    with pytest.raises(ValueError, match=f"Nonfinite policy observation {field}"):
        observation.build_policy_observation([row, row], r.policy_info)

    for rows, message in (([], "no observation rows"), ([replace(row, hand=None)], "hand history")):
        r.clock.now += 5_000_000_000
        caplog.clear()
        assert observation.build_policy_observation(rows, r.policy_info) is None
        assert field in caplog.text and message in caplog.text

    caplog.clear()
    r.policy_info.observation_fields = ("joint_state",)
    row.hand[validity] = False
    r.clock.now = r.started_ns = int(row.arm["timestamp_ns"][0])
    assert r._submit([row, row], 0)
    assert not any(record.levelname == "WARNING" for record in caplog.records)
    assert set(r.model.submissions[-1][1][0]) == {"joint_state"}


@pytest.mark.parametrize("mode", ["async", "rtc"])
@pytest.mark.parametrize("recording", [True, False])
@pytest.mark.parametrize(
    "case",
    [
        "crc",
        "nan",
        "inf",
        "-inf",
        "structure",
        "wait",
        "duration",
        "tie",
        "stop_error",
        "slot",
        "before_sdk",
        "mode_restore",
        "slot_before_duration",
        "duration_partial",
        "wait_partial",
        "tie_partial",
        "operator",
        "quit",
        "crc_operator",
        "crc_duration",
        "crc_cleanup",
        "pending_stop_operator",
        "record_operator",
        "close_operator",
        "rejected_operator",
        "unknown_operator",
        "estop",
        "device_operator",
        "accepted_late",
    ],
)
def test_native_bridge_worker_runner_session_result(tmp_path, monkeypatch, recording, case, mode):
    """Only hardware boundaries are replaced; bridge, worker, owner and archives are real."""
    import json
    import time
    from pathlib import Path

    import torch
    from dexmani_policy.deployment.runtime import LoadedPolicy
    from test_review_remediation import shared_state

    from dexmani_real.config.experiment import ExperimentConfig
    from dexmani_real.deployment import session
    from dexmani_real.deployment.config import RolloutRecordingConfig
    from dexmani_real.ipc.schema import ARM_STATE_DTYPE, HAND_STATE_DTYPE
    from dexmani_real.sensor.camera.geometry import CameraIntrinsics, RGBDGeometry

    runtime = ExperimentConfig()
    shared = shared_state()
    shared.safety_state.value = int(SafetyState.DISARMED)
    shared.start_request = NS(value=False)
    clock = NS(now=1_000_000_000)
    monkeypatch.setattr(time, "monotonic_ns", lambda: clock.now)
    home = np.r_[runtime.arm.home_qpos, np.deg2rad(runtime.hand.home_qpos_deg)]
    state = NS(
        connected=False, frames=[], worker=None, runner=None, polls=0, starts=[], model_threads=[]
    )
    info = NS(
        n_obs_steps=1,
        n_action_steps=1,
        horizon=3,
        control_dt_s=0.1,
        observation_fields=("joint_state",),
        action_mode="joint",
        inference_steps=1,
    )

    class Agent(torch.nn.Module):
        consumed_observation_fields = ("joint_state",)

        def predict_action(self, obs, **kwargs):
            import threading

            state.model_threads.append(threading.get_ident())
            output = torch.tensor(home, dtype=torch.float64).repeat(1, 3, 1)
            if state.connected:
                if case in {"nan", "inf", "-inf"}:
                    output[0, 0, 0] = float(case)
                elif case == "structure":
                    return {"pred_action": [object()]}
            return {"pred_action": output}

    def factory(*args):
        return LoadedPolicy(Agent(), {"dataset": {}}, info, device="cpu", seed=7)

    import threading

    from dexmani_real.robot.robot import DexManiRobot

    robot = DexManiRobot(shared, runtime)
    robot.sent, robot.stops = [], 0
    sdk_calls = []

    def connect():
        state.connected = robot._connected = True
        robot._owner = threading.get_ident()

    robot.connect = connect
    robot.service_idle = lambda: None
    native_send = robot.send_action
    normal = case in {"duration", "duration_partial", "operator", "quit", "accepted_late"}

    def cancel():
        reason = RunEndReason.QUIT if case == "quit" else RunEndReason.OPERATOR
        if case == "estop":
            reason = RunEndReason.ESTOP
            shared.estop_request.value = True
        request_policy_stop(shared, reason=reason)

    def arm(q):
        sdk_calls.append("arm")
        if case in {
            "slot",
            "slot_before_duration",
            "duration_partial",
            "wait_partial",
            "tie_partial",
        }:
            clock.now += 40_000_000
        if case in {
            "operator",
            "quit",
            "stop_error",
            "record_operator",
            "close_operator",
            "estop",
            "device_operator",
        }:
            cancel()
        if case == "device_operator":
            raise OSError("actual SDK failure")
        return 0

    def hand(q):
        sdk_calls.append("hand")
        if case in {"wait", "duration", "tie", "crc_duration", "accepted_late"}:
            clock.now += 40_000_000
        if case in {"crc_operator", "rejected_operator", "unknown_operator", "crc_cleanup"}:
            cancel()
        return (
            DispatchStatus.CRC_UNCONFIRMED
            if case in {"crc", "crc_operator", "crc_duration", "crc_cleanup"}
            else DispatchStatus.REJECTED
            if case == "rejected_operator"
            else DispatchStatus.UNKNOWN
            if case == "unknown_operator"
            else DispatchStatus.ACCEPTED
        )

    def restore():
        sdk_calls.append("mode")
        clock.now += 40_000_000

    robot.arm = NS(is_connected=True, servo=arm, enter_mode6=restore)
    robot.hand = NS(is_connected=True, send_action=hand)

    def dispatch(command, **kwargs):
        robot.sent.append(command)
        if case == "before_sdk":
            clock.now += 40_000_000
        if case == "mode_restore":
            robot._arm_stopped = True
        if case == "pending_stop_operator":
            robot._hand_stop_pending = True
            cancel()
        return native_send(command, **kwargs)

    robot.send_action = dispatch

    def stop():
        robot.stops += 1
        if case in {"stop_error", "crc_cleanup"}:
            raise OSError("actual stop failure")
        robot._motion_active = robot._hand_stop_pending = False

    robot.stop = stop

    def read(self):
        arm, hand = np.zeros(1, ARM_STATE_DTYPE), np.zeros(1, HAND_STATE_DTYPE)
        arm["timestamp_ns"] = hand["timestamp_ns"] = clock.now
        arm["qpos"], hand["qpos"] = home[:7], home[7:]
        camera = dict(
            timestamp_ns=clock.now,
            color_frame_number=1,
            depth_frame_number=1,
            rgb=np.zeros((16, 16, 3), "u1"),
            depth=np.ones((16, 16), "u2"),
        )
        return ObservationRow(arm, hand, camera if recording else None, None, None, clock.now)

    native_runner = session.PolicyRunner

    def runner(*args, **kwargs):
        owner = native_runner(*args, **kwargs)
        state.worker, state.runner = kwargs["model_runtime"], owner
        if case in {"record_operator", "crc_cleanup"}:
            record = owner._record

            def fail_record(row, command=None, result=None):
                record(row, command, result)
                if command is not None:
                    raise OSError("actual recording failure")

            owner._record = fail_record
        return owner

    class Operator:
        home_results = []
        keyboard = NS(start=lambda: None, healthy=True)

        def __init__(self, *a, **kw):
            pass

        def poll(self):
            state.polls += 1
            assert state.polls < 2000, "session failed to terminate"
            if state.runner.completed:
                shared.quit_requested.value = True
            if state.polls == 1:
                shared.start_request.value = True
            if state.worker.future is not None:
                state.worker.future.result(timeout=5)
            if case == "accepted_late" and robot.sent:
                request_policy_stop(shared, reason=RunEndReason.QUIT)
                shared.quit_requested.value = True
            clock.now += 1_000_000

    def shutdown(robot, supervisor, *, model, **kwargs):
        model.close()
        deadline = time.perf_counter() + 5
        while not model.closed:
            assert time.perf_counter() < deadline
            time.sleep(0.001)
        if case in {"close_operator", "crc_cleanup"}:
            raise OSError("actual close failure")
        return model.close_error is None

    intrinsic = CameraIntrinsics(16, 16, 10.0, 10.0, 8.0, 8.0, "none", (0.0,) * 5)
    geometry = RGBDGeometry(intrinsic, intrinsic, np.eye(4))
    monkeypatch.setattr(
        "dexmani_real.deployment.runner.snapshot_recording_metadata",
        lambda *a, **k: dict(
            collection_source="policy_rollout",
            camera_geometry=geometry,
            camera_T_xarm_base_from_color=None,
            depth_scale=0.001,
            handbase_position_eef_m=np.zeros(3),
            handbase_quat_eef_wxyz=np.array([1.0, 0, 0, 0]),
        ),
    )
    from dataclasses import replace

    runtime = replace(runtime, camera=replace(runtime.camera, width=16, height=16))
    monkeypatch.setattr(session.RuntimeChannels, "create", lambda **kw: shared)
    monkeypatch.setattr(
        session,
        "RuntimeSupervisor",
        lambda *a: NS(check=lambda: True, start=lambda sensors: state.starts.extend(sensors)),
    )
    monkeypatch.setattr(session, "DexManiRobot", lambda *a, **kw: robot)
    monkeypatch.setattr(session, "_load_configured_policy", factory)
    monkeypatch.setattr(session, "build_home_planner", lambda *a: object())
    monkeypatch.setattr(session, "PolicyOperator", Operator)
    monkeypatch.setattr(session, "PolicyRunner", runner)
    monkeypatch.setattr(PolicyRunner, "_read_observation", read)
    monkeypatch.setattr(session, "shutdown_local_runtime", shutdown)
    monkeypatch.setattr(
        session.ActionRealizer,
        "for_mode",
        lambda *a: ActionRealizer(
            runtime,
            collision_model=NS(set_hand_qpos=lambda q: None, check_self_collision=lambda q: False),
        ),
    )
    # No Process.start is called; sensor lifecycle is the device boundary substitute.
    output = tmp_path / "session"
    budget = (
        0.105
        if case in {"duration", "tie", "duration_partial", "tie_partial", "crc_duration"}
        else 0.135
        if case == "slot_before_duration"
        else 0.5
    )
    wait = 0.105 if case in {"wait", "tie", "wait_partial", "tie_partial"} else 1.0
    execution = ExecutionConfig(mode, 1.0, wait, 0.03, 1, 0.0)
    code = session.run_policy_deployment(
        runtime,
        NS(info=info),
        True,
        execution_config=execution,
        max_running_s=budget,
        num_episodes=1 if normal else 3,
        recording_config=RolloutRecordingConfig(str(output), "synthetic") if recording else None,
        save_run_config=lambda *a: state.frames.append("config"),
    )
    assert code == (0 if normal else 1)
    assert state.runner.completed == 1 and state.runner.run_id is None
    assert len(set(state.model_threads)) == 1
    assert len(robot.sent) == (0 if case in {"nan", "inf", "-inf", "structure"} else 1)
    assert len(state.starts) == int(recording)
    assert state.frames == (["config"] if recording else [])
    if not recording:
        assert not output.exists()
        return
    result = json.loads((output / "session_result.json").read_text())
    assert result["outcome"] == ("finished" if normal else "fault")
    attempts = list((output / "attempts").glob("*.json"))
    assert len(attempts) == 1
    attempt = json.loads(attempts[0].read_text())
    json.dumps(attempt, allow_nan=False)
    assert len([e for e in attempt["trace"] if e["event"] == "end"]) == 1
    if case in {"nan", "inf", "-inf"}:
        with np.load(attempt["query_sidecar"], allow_pickle=False) as arrays:
            assert len(arrays.files) == 1
            value = arrays[arrays.files[0]]
            assert value.dtype == np.float64 and value.shape == (3, 19)
            assert np.isnan(value[0, 0]) if case == "nan" else value[0, 0] == float(case)
    elif case == "structure":
        with np.load(attempt["query_sidecar"], allow_pickle=False) as arrays:
            assert arrays.files == []
    else:
        dispatches = [e for e in attempt["trace"] if e["event"] == "dispatch"]
        expected = (
            (0, 0)
            if case in {"before_sdk", "mode_restore", "pending_stop_operator"}
            else (4, 0)
            if case == "device_operator"
            else (1, 0)
            if "hand" not in sdk_calls
            else (1, 2)
            if case in {"crc", "crc_operator", "crc_duration", "crc_cleanup"}
            else (1, 3)
            if case == "rejected_operator"
            else (1, 4)
            if case == "unknown_operator"
            else (1, 1)
        )
        assert len(dispatches) == 1
        assert (dispatches[0]["arm"], dispatches[0]["hand"]) == expected
        if case == "pending_stop_operator":
            assert dispatches[0]["cause"] == "stop_unconfirmed"
        if case == "device_operator":
            assert dispatches[0]["cause"] is None
        if case in {
            "slot",
            "before_sdk",
            "mode_restore",
            "slot_before_duration",
            "duration_partial",
            "wait_partial",
            "tie_partial",
        }:
            event = dispatches[0]
            assert event["cause"] == "deadline_expired"
            assert event["valid_until_ns"] == min(
                event["action_deadline_ns"], event["budget_deadline_ns"]
            )
            assert event["start_ns"] + event["duration_ns"] >= event["valid_until_ns"]
            if case == "slot_before_duration":
                assert event["action_deadline_ns"] < event["budget_deadline_ns"]
                assert attempt["termination_reason"] == "executor_boundary"
        import h5py

        with h5py.File(Path(attempt["raw_path"]) / "data.h5") as raw:
            evidence = raw["dispatch_status"][:]
            np.testing.assert_array_equal(
                evidence[np.any(evidence != 0, axis=1)],
                [expected] if any(expected) else np.empty((0, 2)),
            )
    if case in {"wait", "tie", "duration", "wait_partial", "tie_partial", "duration_partial"}:
        assert attempt["termination_reason"] == "timeout"
        assert ("duration" if case.startswith("duration") else "wait") in attempt[
            "termination_details"
        ]
    if "operator" in case or case in {"stop_error", "quit", "estop", "crc_cleanup"}:
        assert attempt["termination_reason"] == (case if case in {"quit", "estop"} else "operator")
    if case in {"stop_error", "record_operator"}:
        assert attempt["finalization_errors"][0]["stage"] == (
            "stop" if case == "stop_error" else "recording"
        )
    if case == "crc_cleanup":
        assert [e["stage"] for e in attempt["finalization_errors"]] == ["stop", "recording"]
        assert {e["stage"] for e in result["artifact_errors"]} == {"stop", "recording", "shutdown"}
        assert "motion authority revoked" in result["reason"]
    assert attempt["recording_status"] == "published"
    assert Path(attempt["raw_path"]).is_dir()


def test_r2_legal_query_phase_inclusive_age_and_bootstrap_grid(make_runner):
    r = make_runner(n=1)
    r.execution = ExecutionConfig("async", 0.37, 1.0, 0.03, 2).validate(r.policy_info)
    r.tick(0, extra=30_000_000)
    r.clock.now += 10_000_000
    r.model.complete(future(r))
    r.step()
    r.tick(1)
    r.tick(2, extra=30_000_000)
    r.clock.now += 10_000_000
    r.model.complete(future(r, 0.05))
    r.step()
    for slot in range(3, 7):
        r.tick(slot)
    assert len(r.robot.sent) == 6
    np.testing.assert_allclose(r.robot.sent[-1].arm_qpos, 0.09)
    assert r.plan.sources["joint_state"][0] == r.started_ns + 230_000_000
    # 600 - 230 = 370 ms: the freshness endpoint is inclusive.
    assert r.clock.now - r.plan.sources["joint_state"][0] == 370_000_000

    other = make_runner(n=1, mode="sync")
    other.tick(0)
    other.clock.now += 100_000_000
    other.model.complete(future(other))
    other.step()
    assert other.query.handoff_slot == 2 and not other.robot.sent
    other.tick(2)
    assert len(other.robot.sent) == 1


@pytest.mark.parametrize("stage", ["observation", "build", "record"])
def test_wait_expiry_during_tick_finishes_before_next_operator_poll(
    make_runner, monkeypatch, stage
):
    from dexmani_real.deployment import runner as module

    r = make_runner(n=1, mode="sync", wait=0.15)
    if stage == "observation":
        read = r._read_observation

        def slow_read():
            r.clock.now += 160_000_000
            return read()

        r._read_observation = slow_read
    elif stage == "build":
        build = module.build_policy_observation

        def slow_build(*args, **kwargs):
            r.clock.now += 160_000_000
            return build(*args, **kwargs)

        monkeypatch.setattr(module, "build_policy_observation", slow_build)
    else:

        def slow_record(*args):
            r.clock.now += 160_000_000

        r._record = slow_record
    with pytest.raises(TimeoutError, match="WAIT"):
        r.tick(0)
    assert r.run_id is None and r.completed == 1
    assert r.shared.run_ended_reason.value == RunEndReason.TIMEOUT
    assert not r.robot.sent
    assert len([e for e in r.events if e["event"] == "end"]) == 1
