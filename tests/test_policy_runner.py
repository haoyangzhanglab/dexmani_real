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
from dexmani_real.robot.robot import DispatchInterrupted, DispatchResult, DispatchStatus
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
    factory = ActionRealizer.for_mode
    # Scheduling oracles use a collision-free synthetic scene; native admission
    # and mount propagation are covered separately with actual model assets.
    monkeypatch.setattr(
        ActionRealizer,
        "for_mode",
        lambda runtime, mode: (
            ActionRealizer(
                runtime,
                collision_model=NS(
                    set_hand_qpos=lambda q: None, check_self_collision=lambda q: False
                ),
            )
            if mode == "joint"
            else factory(runtime, mode)
        ),
    )

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
    bootstrap(r)
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
    for i in range(4):
        r.tick(i)
    assert r.completed == 1 and r.shared.run_ended_reason.value == RunEndReason.TIMEOUT


def test_eef_runner_prefix_uses_real_fk_and_same_joint_command(make_runner):
    from dexmani_real.config.experiment import ExperimentConfig
    from dexmani_real.deployment.runner import Plan
    from dexmani_real.planning.kinematics.arm_fk import compute_eef_pose_history_xarm_base
    from dexmani_real.robot.action import ActionRealizer

    r = make_runner(mode="rtc", n=1)
    from dataclasses import replace

    cfg = ExperimentConfig()
    # Explicit nominal test asset, not the current physical mount.
    cfg = replace(cfg, hand=replace(cfg.hand, T_eef_handbase_pos_xyz=(-0.005, 0, 0)))
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
    assert saved == [{"reason": reason.name.lower()}]
    end = [e for e in r.recorder.policy_trace["events"] if e["event"] == "end"]
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
    assert saved == [dict(reason="timeout")]
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
    bootstrap(r)
    assert len(rows) == 1
    assert (rows[0].arm, rows[0].hand) == expected
    assert calls == (["arm"] if late_device == "arm" else ["arm", "hand"])
    assert r.shared.run_ended_reason.value == int(RunEndReason.TIMEOUT)
    assert r.completed == 1 and r.run_id is None


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
    with pytest.raises(DispatchInterrupted if cancelled else RecordingError):
        bootstrap(r)
    assert rows == [r.robot.result]
    events = r.recorder.policy_trace["events"]
    assert len([e for e in events if e["event"] == "dispatch"]) == 1
    assert len([e for e in events if e["event"] == "end"]) == 1
    assert r.completed == 1 and r.run_id is None
    assert r.shared.run_ended_reason.value == (
        RunEndReason.ESTOP if cancelled else RunEndReason.TIMEOUT
    )


@pytest.mark.parametrize(
    "stop_fails,writer_fails", [(False, False), (True, False), (False, True), (True, True)]
)
def test_attempt_finalization_orders_stop_before_sidecar(
    make_runner, tmp_path, monkeypatch, stop_fails, writer_fails
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
        native(*a, **kw)

    monkeypatch.setattr(np, "savez", sidecar)
    r.query_arrays = {"query_1": np.ones((2, 19), dtype=np.float16)}
    if stop_fails or writer_fails:
        with pytest.raises(RuntimeError, match="stop failed" if stop_fails else "writer failed"):
            r._finish_episode("timeout", run_end_reason=RunEndReason.TIMEOUT)
    else:
        r._finish_episode("timeout", run_end_reason=RunEndReason.TIMEOUT)
    result = json.loads((tmp_path / "attempts" / f"{attempt}.json").read_text())
    assert result["entered_running"] and result["row_count"] == 0
    assert result["recording_status"] == ("failed" if writer_fails else "empty")
    assert result["termination_reason"] == "timeout"
    assert len(result["finalization_errors"]) == int(stop_fails) + int(writer_fails)
    with np.load(result["query_sidecar"]) as arrays:
        assert arrays["query_1"].dtype == np.float16
    r._finish_episode("again")
    assert order == ["stop", "raw", "sidecar"]


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
