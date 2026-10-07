"""Explicit tactile preparation with fake driver reads and operator events."""

from concurrent.futures import CancelledError
from types import SimpleNamespace as NS

import numpy as np
import pytest
from test_review_remediation import shared_state

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.robot.drivers.xhand import XHand
from dexmani_real.robot.robot import DexManiRobot, RobotState
from dexmani_real.runtime.operator_input import OperatorCommand
from dexmani_real.runtime.safety import SafetyState


def test_connect_only_reads_joints_no_tare():
    shared = shared_state()
    calls = []
    arm = NS(connect=lambda: calls.append("arm_connect"))
    hand = NS(
        connect=lambda: calls.append("hand_connect"),
        tare_tactile=lambda **kw: pytest.fail("implicit tare"),
    )
    robot = DexManiRobot(
        shared, ExperimentConfig(), arm_factory=lambda cfg: arm, hand_factory=lambda cfg: hand
    )
    robot.read_state = lambda: RobotState(np.zeros(1), np.zeros(1))
    robot.connect()
    assert calls == ["arm_connect", "hand_connect"]


@pytest.mark.parametrize("cancel_at", [1, 2, 3, 4, "verification", "publication"])
def test_tare_cancellation_does_not_publish_candidates(monkeypatch, cancel_at):
    from dexmani_real.robot.drivers import xhand

    monkeypatch.setattr(xhand.time, "sleep", lambda _: None)
    hand = XHand(ExperimentConfig().hand)
    hand._tactile_bias_aggregate = np.ones((5, 3))
    hand._tactile_bias_dense = np.ones((5, 120, 3))
    reads = []

    def read(**kwargs):
        assert hand._tactile_bias_aggregate is hand._tactile_bias_dense is None
        reads.append(1)
        return NS(
            tactile_aggregate_valid=True,
            tactile_dense_valid=True,
            tactile_aggregate=np.full((5, 3), 10.0),
            tactile_dense=np.full((5, 120, 3), 10.0),
        )

    hand._read_state = read
    checks = 0
    if cancel_at == "verification":
        cancel_at = 3 * xhand._TACTILE_BIAS_SAMPLE_COUNT + 1
    if cancel_at == "publication":
        cancel_at = 3 * (xhand._TACTILE_BIAS_SAMPLE_COUNT + xhand._TACTILE_VERIFY_SAMPLE_COUNT) + 1

    def cancel():
        nonlocal checks
        checks += 1
        return checks >= cancel_at

    with pytest.raises(CancelledError):
        hand.tare_tactile(cancel_requested=cancel)
    assert hand._tactile_bias_aggregate is hand._tactile_bias_dense is None


def test_stable_contact_cannot_be_inferred_from_zero_residual(monkeypatch):
    from dexmani_real.robot.drivers import xhand

    monkeypatch.setattr(xhand.time, "sleep", lambda _: None)
    hand = XHand(ExperimentConfig().hand)
    hand._read_state = lambda **kw: NS(
        tactile_aggregate_valid=True,
        tactile_dense_valid=True,
        tactile_aggregate=np.full((5, 3), 100.0),
        tactile_dense=np.full((5, 120, 3), 100.0),
    )
    assert hand.tare_tactile() == (True, True)
    np.testing.assert_array_equal(hand._tactile_bias_aggregate, np.full((5, 3), 100.0))


@pytest.mark.parametrize(
    "active,capture,preparing,armed",
    [
        (True, False, False, True),
        (False, True, False, True),
        (False, False, True, True),
        (False, False, False, False),
        (False, False, False, True),
    ],
)
def test_teleop_tare_idle_admission(active, capture, preparing, armed):
    from dexmani_real.teleop.runner import TeleopRunner

    runner = TeleopRunner.__new__(TeleopRunner)
    runner.shared = shared_state()
    runner.shared.safety_state.value = int(SafetyState.ARMED if armed else SafetyState.RUNNING)
    runner.active, runner.paused, runner.resume_requested = active, False, preparing
    runner.recorder = NS(is_recording=capture)
    calls = []
    runner.robot = NS(tare_tactile=lambda **kw: calls.append("tare"))
    runner.keyboard = NS(drain_signal=lambda signal: None)
    runner._handle_operator_command(OperatorCommand.TARE)
    assert calls == (["tare"] if not (active or capture or preparing or not armed) else [])


def test_policy_preparation_rejects_tare_and_does_not_begin_same_batch():
    from dexmani_real.deployment.operator import PolicyOperator

    shared = shared_state()
    shared.safety_state.value = int(SafetyState.ARMED)
    shared.start_request = NS(value=False)
    calls = []
    operator = PolicyOperator(
        shared,
        ExperimentConfig(),
        None,
        robot=NS(tare_tactile=lambda **kw: calls.append("tare")),
        execute=False,
        idle_for_tare=lambda: False,
    )
    operator._handle_command_batch([OperatorCommand.TARE, OperatorCommand.BEGIN])
    assert calls == [] and not shared.start_request.value
    operator.idle_for_tare = lambda: True
    operator._handle_command_batch([OperatorCommand.TARE])
    assert calls == ["tare"]


def test_policy_tare_preserves_stop_received_during_admission():
    from dexmani_real.deployment.operator import PolicyOperator
    from dexmani_real.runtime.safety import StopRequest

    shared = shared_state()
    shared.safety_state.value = int(SafetyState.ARMED)

    def idle():
        shared.stop_request.value = int(StopRequest.OPERATOR)
        return True

    def tare(*, cancel_requested):
        assert cancel_requested()
        raise CancelledError()

    operator = PolicyOperator(
        shared,
        ExperimentConfig(),
        None,
        robot=NS(tare_tactile=tare, check=lambda: None),
        execute=False,
        idle_for_tare=idle,
    )
    operator._handle_command_batch([OperatorCommand.TARE])
    assert shared.stop_request.value == int(StopRequest.OPERATOR)


def test_owner_tare_stops_before_sampling_and_passes_cancellation():
    import threading

    from dexmani_real.runtime.safety import StopRequest

    shared = shared_state()
    shared.safety_state.value = int(SafetyState.ARMED)
    robot = DexManiRobot(shared, ExperimentConfig())
    robot._owner = threading.get_ident()
    calls = []
    robot.stop = lambda: calls.append("stop")

    def sample(*, cancel_requested):
        calls.append("sample")
        shared.stop_request.value = int(StopRequest.OPERATOR)
        assert cancel_requested()
        raise CancelledError()

    robot.hand = NS(tare_tactile=sample)
    with pytest.raises(CancelledError):
        robot.tare_tactile(cancel_requested=lambda: False)
    assert calls == ["stop", "sample"]
