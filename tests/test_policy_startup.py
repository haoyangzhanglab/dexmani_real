from concurrent.futures import Future
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dexmani_real.deployment import rollout as rollout_module
from dexmani_real.deployment import session as deployment
from dexmani_real.deployment.inference import PolicyWorker
from dexmani_real.robot.commands import OperationCancelled
from dexmani_real.runtime.state import SessionState


@pytest.fixture
def startup(monkeypatch):
    policy = deployment.PolicySession(None, None)
    policy.keyboard = Mock()
    policy.model = Mock(future=Future(), warmed_up=False)
    policy.model.poll.side_effect = lambda: PolicyWorker.poll(policy.model)
    clock = SimpleNamespace(now=0.0)

    def sleep(seconds):
        clock.now += seconds

    monkeypatch.setattr(deployment, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep))
    monkeypatch.setattr(deployment, "POLICY_STARTUP_TIMEOUT_S", 0.012)
    return policy, clock


def test_wait_model_returns_completed_load(startup, monkeypatch):
    policy, clock = startup
    info = object()

    def complete(seconds):
        clock.now += seconds
        policy.model.future.set_result(info)

    monkeypatch.setattr(deployment.time, "sleep", complete)

    assert policy.wait_model(stage="policy model load") is info
    assert policy.model.future is None


def test_wait_model_rejects_completion_if_final_feedback_crosses_deadline(startup):
    policy, clock = startup
    policy.model.future.set_result(None)
    policy.robot = Mock()

    def service_feedback():
        if policy.model.future is None:
            clock.now += 1.0

    policy.robot.service_feedback.side_effect = service_feedback

    with pytest.raises(TimeoutError, match="policy warmup timed out"):
        policy.wait_model(stage="policy warmup")


def test_pending_warmup_rejects_startup(startup, monkeypatch, capsys):
    policy, clock = startup
    policy.runtime = SimpleNamespace(execution=SimpleNamespace(mode="sync"))
    policy.info = SimpleNamespace(n_obs_steps=2)
    policy.observations = Mock(needs=SimpleNamespace(hoi=False, point_cloud=False))
    policy.robot = Mock()
    policy.prepare_resources = Mock()
    policy.run_control_loop = Mock()
    closed = Future()
    closed.set_result(None)
    policy.model.close.return_value = closed
    monkeypatch.setattr(deployment, "configure_logging", Mock())
    monkeypatch.setattr(deployment, "read_observation", Mock())
    monkeypatch.setattr(deployment, "check_device_feedback", lambda *args: [])
    monkeypatch.setattr(deployment, "check_observation", lambda *args: [])

    assert policy.run() == 1
    assert clock.now == pytest.approx(0.012)
    assert policy.model.poll.call_count == 3
    assert "policy warmup timed out" in policy.errors[0]["message"]
    assert "就绪" not in capsys.readouterr().out
    assert not policy.model.warmed_up and not policy.model.future.done()
    policy.run_control_loop.assert_not_called()
    policy.robot.service_feedback.assert_called()
    policy.model.close.assert_called_once()


def test_wait_model_checks_cancellation_while_pending(startup, monkeypatch):
    policy, _ = startup
    monkeypatch.setattr(deployment.time, "sleep", lambda _: policy.runtime_state.request_stop())

    with pytest.raises(OperationCancelled, match="startup cancelled"):
        policy.wait_model(stage="policy model load")

    policy.model.poll.assert_called_once()


def test_wait_model_propagates_original_error(startup):
    policy, _ = startup
    error = OSError("bad checkpoint")
    policy.model.future.set_exception(error)

    with pytest.raises(OSError) as exc:
        policy.wait_model(stage="policy model load")

    assert exc.value is error


@pytest.fixture
def hoi_startup(monkeypatch):
    clock = SimpleNamespace(now=0.0, deadline=0.012)

    def sleep(seconds):
        clock.now += seconds

    fake_time = SimpleNamespace(
        monotonic=lambda: clock.now, monotonic_ns=lambda: int(clock.now * 1e9), sleep=Mock(side_effect=sleep),
    )
    monkeypatch.setattr(rollout_module, "time", fake_time)
    monkeypatch.setattr(rollout_module, "POLICY_STARTUP_TIMEOUT_S", clock.deadline)

    model = Mock(future=None, warmed_up=False)

    def warmup(_rows):
        model.future = Future()
        model.future.set_result(None)

    model.warmup.side_effect = warmup
    model.poll.side_effect = lambda: PolicyWorker.poll(model)
    state = SessionState(stop_signal=Event(), emergency_signal=Event())
    rollout = rollout_module.Rollout(
        state, SimpleNamespace(execution=SimpleNamespace(control_dt_s=0.05, mode="sync")),
        SimpleNamespace(n_obs_steps=1), episode_cancel_epoch=state.cancel_epoch, robot=Mock(),
        observations=Mock(needs=SimpleNamespace(hoi=True, hoi_joint_devices=())), sensors=Mock(),
        action_mapper=None, model=model, max_duration_s=None, index=1, output_dir=None, record_raw=False,
        effective_parameters={},
    )
    rollout.start_observation = Mock(return_value=object())
    return rollout, clock


@pytest.mark.parametrize("stage, overrun", [("feedback", 0.001), ("operator", 0.0)])
def test_hoi_warmup_rejects_success_at_or_after_deadline(hoi_startup, stage, overrun):
    rollout, clock = hoi_startup

    def advance_clock():
        clock.now = clock.deadline + overrun

    def poll_operator():
        if stage == "operator" and rollout.model.poll.called and rollout.model.future is None:
            advance_clock()

    if stage == "feedback":
        rollout.robot.service_feedback.side_effect = advance_clock

    with pytest.raises(TimeoutError, match="HOI policy warmup timeout"):
        rollout.prepare(poll_operator)

    assert rollout.model.future is None
    assert not rollout.model.warmed_up
    rollout.robot.prepare_streaming.assert_not_called()


def test_hoi_warmup_registers_success_without_sleeping_after_completion(hoi_startup):
    rollout, clock = hoi_startup
    rollout.robot.service_feedback.side_effect = lambda: setattr(clock, "now", clock.deadline - 0.001)

    rollout.prepare(Mock())

    assert rollout.model.warmed_up
    rollout_module.time.sleep.assert_not_called()
    rollout.robot.prepare_streaming.assert_called_once()
