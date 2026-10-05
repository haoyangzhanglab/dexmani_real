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
from dexmani_real.robot.action import ActionRealization
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
    _motion_active = False
    _hand_stop_pending = False

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
            model_runtime=worker,
            execution_config=ExecutionConfig(mode, 10.0, wait, 0.03, d, 2.0),
            fingertip_runtime=None,
            execute=execute,
            max_running_s=100.0,
            num_episodes=10,
        )
        runner._read_observation = row
        runner.run_id = 1
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
    r.tick(b + 1)
    assert r.plan is None and len(r.robot.sent) == 1 and not r.query.valid


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
    r._send(row, 0)
    assert r.robot.sent[0] is commands[0]


def test_mode_change_requires_idle_and_serial_configuration(make_runner):
    r = make_runner()
    config = ExecutionConfig("rtc", 10.0, 1.0, 0.03, 2, 2.0)
    with pytest.raises(RuntimeError):
        r.set_execution_mode(config)
    r._finish_episode("test")
    r.runtime.camera = NS(height=16, width=16)
    r.set_execution_mode(config)
    assert r.model.submissions[-1][0] == "configure_execution"
    assert r.model.submissions[-1][2]["warmup"]
    with pytest.raises(RuntimeError):
        r.set_execution_mode(config)


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


@pytest.fixture
def idle_runner(make_runner):
    r = make_runner(mode="sync", execute=False)
    r._finish_episode("test setup")
    r.runtime.camera = NS(height=16, width=16)
    r._start_observation = r._read_observation
    r.recorder = NS(
        execution_path="worker_grid_sync_v1", check_error=lambda: None, close=lambda: None
    )
    return r


@pytest.mark.parametrize(
    "mode,age,wait,durations,message",
    [
        ("async", 10.0, 1.0, (0.01, 2.0), "max_wait_s"),
        ("rtc", 10.0, 1.0, (0.01, 2.0), "max_wait_s"),
        ("sync", 10.0, 1.0, (1.0,), "max_wait_s"),
        ("async", 10.0, 1.0, (0.2,), "prefetch budget"),
        ("rtc", 10.0, 1.0, (0.2,), "prefetch budget"),
        ("sync", 0.5, 1.0, (0.3,), "max_decision_age_s"),
        ("sync", 0.5, 1.0, (0.31,), "max_decision_age_s"),
    ],
)
def test_mode_warmup_budget_failure_blocks_start_and_preserves_official_config(
    idle_runner, mode, age, wait, durations, message
):
    r = idle_runner
    old = r.execution
    config = ExecutionConfig(mode, age, wait, 0.03, 2, 2.0)
    r.set_execution_mode(config)
    assert r.execution is old and r.recorder.execution_path == "worker_grid_sync_v1"
    assert r.pending_execution is config
    r.shared.start_request.value = True
    r.step()
    assert r.preparing_epoch is None and r.model.op == "configure_execution"
    r.model.complete(durations)
    with pytest.raises(ValueError, match=message):
        r.run()
    assert r.execution is old and r.recorder.execution_path == "worker_grid_sync_v1"
    assert r.run_id is None and r.preparing_epoch is None
    assert r.shared.error_state.value and not r.shared.is_running.value
    assert [op for op, _, _ in r.model.submissions] == ["configure_execution"]
    assert r.model.future is None and r.model.closed
    assert r.pending_execution is config
    with pytest.raises(RuntimeError):
        r.set_execution_mode(config)
    assert [op for op, _, _ in r.model.submissions] == ["configure_execution"]


def test_mode_worker_failure_blocks_start_and_preserves_official_config(idle_runner):
    r = idle_runner
    old = r.execution
    r.set_execution_mode(ExecutionConfig("rtc", 10.0, 1.0, 0.03, 2, 2.0))
    r.shared.start_request.value = True
    r.model.complete(error=RuntimeError("warmup failed"))
    with pytest.raises(RuntimeError, match="warmup failed"):
        r.run()
    assert r.execution is old and r.recorder.execution_path == "worker_grid_sync_v1"
    assert r.run_id is None and r.preparing_epoch is None and r.model.closed
    assert [op for op, _, _ in r.model.submissions] == ["configure_execution"]


@pytest.mark.parametrize("mode", ["sync", "async", "rtc"])
def test_legal_mode_warmup_commits_before_reset_and_start(idle_runner, mode, monkeypatch):
    r = idle_runner
    old = r.execution
    config = ExecutionConfig(mode, 10.0, 1.0, 0.03, 2, 2.0)
    r.reset_ready = True
    r.set_execution_mode(config)
    op, _, kwargs = r.model.submissions[-1]
    assert op == "configure_execution" and kwargs["warmup"]
    assert kwargs["rtc_delay"] == (2 if mode == "rtc" else 0)
    r.shared.start_request.value = True
    r.step()
    assert r.execution is old and r.recorder.execution_path == "worker_grid_sync_v1"
    assert not r.reset_ready
    assert r.preparing_epoch is None and len(r.model.submissions) == 1
    r.model.complete((0.01, 0.19))
    r.step()
    assert r.execution is config and r.recorder.execution_path == f"worker_grid_{mode}_v1"
    assert r.pending_execution is None
    assert [op for op, _, _ in r.model.submissions] == ["configure_execution", "reset_episode"]
    assert r.run_id is None and r.preparing_epoch is not None
    r.recording_config = NS(task_label="synthetic")
    r.recorder.start_episode = lambda **kwargs: True
    r.recorder.is_recording = False
    r.recorder.add_frame = lambda frame: None
    monkeypatch.setattr(
        "dexmani_real.deployment.runner.snapshot_recording_metadata", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        "dexmani_real.deployment.runner.build_episode_frame", lambda row, *args: row
    )
    r.model.complete()
    r.step()
    assert r.run_id is not None
    assert r.events[0]["event"] == "begin" and r.events[0]["mode"] == mode
    assert [op for op, _, _ in r.model.submissions] == ["configure_execution", "reset_episode"]


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
