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
from dexmani_real.runtime.safety import SafetyState, StopRequest


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


@pytest.fixture
def policy_keyboard(monkeypatch):
    from dexmani_real.deployment import operator as module

    shared = shared_state()
    shared.safety_state.value = int(SafetyState.ARMED)
    shared.start_request = NS(value=False)
    calls = []
    effects = NS(home_completed=True, during_home=lambda: None, during_tare=lambda: None)

    def home(*args, abort_requested, **kwargs):
        calls.append("home")
        effects.during_home()
        effects.home_cancelled = abort_requested()
        return effects.home_completed and not effects.home_cancelled

    def tare(*, cancel_requested):
        calls.append("tare")
        effects.during_tare()
        if cancel_requested():
            raise CancelledError()
        return True, True

    monkeypatch.setattr(module, "home_robot", home)
    operator = module.PolicyOperator(
        shared,
        ExperimentConfig(),
        NS(),
        robot=NS(tare_tactile=tare, check=lambda: None),
        execute=True,
        idle_for_tare=lambda: not shared.start_request.value,
    )
    # Use real callbacks, queue and ESC latch without starting a pynput listener.
    keyboard = operator.keyboard
    keyboard._callbacks_active = keyboard._running = True
    keyboard._listener = NS(is_alive=lambda: True)
    on_press, on_release = keyboard._callbacks(NS(Key=NS(esc="esc")))

    def press(*keys):
        for name in keys:
            key = "esc" if name == "esc" else NS(char=name)
            on_press(key)
            on_release(key)

    return NS(operator=operator, shared=shared, calls=calls, effects=effects, press=press)


@pytest.mark.parametrize("completed", [True, False])
def test_policy_home_discards_same_batch_tare_but_accepts_new_tare(policy_keyboard, completed):
    case = policy_keyboard
    case.effects.home_completed = completed
    case.press("h", "t")
    case.operator.poll()
    assert case.calls == ["home"]
    assert case.operator.home_results == [{"outcome": "completed" if completed else "failed"}]
    assert not case.shared.start_request.value
    case.press("t")
    case.operator.poll()
    assert case.calls == ["home", "tare"]
    assert case.shared.run_id.value == 1


def test_policy_home_discards_tare_queued_during_home(policy_keyboard):
    case = policy_keyboard
    case.effects.during_home = lambda: case.press("t")
    case.press("h")
    case.operator.poll()
    case.operator.poll()
    assert case.calls == ["home"]


@pytest.mark.parametrize("keys,expected", [(("h", "b"), ["home"]), (("t", "h"), ["tare"])])
def test_policy_preparation_does_not_chain_home_or_begin(policy_keyboard, keys, expected):
    case = policy_keyboard
    case.press(*keys)
    case.operator.poll()
    assert case.calls == expected
    assert not case.shared.start_request.value
    assert case.shared.run_id.value == 1


def test_policy_disabled_home_does_not_discard_tare(policy_keyboard):
    case = policy_keyboard
    case.operator.planner = None
    case.press("h", "t")
    case.operator.poll()
    assert case.calls == ["tare"]
    assert case.operator.home_results == []


@pytest.mark.parametrize("key", ["s", "q", "esc"])
def test_policy_stop_in_same_batch_prevents_preparation_and_begin(policy_keyboard, key):
    case = policy_keyboard
    case.press("h", "t", "b", key)
    assert case.shared.estop_request.value if key == "esc" else case.shared.stop_request.value
    case.operator.poll()
    assert case.calls == []
    assert not case.shared.start_request.value
    assert case.shared.quit_requested.value == (key == "q")
    assert case.operator.keyboard.estop_latched == (key == "esc")
    assert case.shared.run_id.value == (1 if key == "esc" else 2)


@pytest.mark.parametrize("key", ["s", "q", "esc"])
@pytest.mark.parametrize("preparation", ["home", "tare"])
def test_policy_stop_during_preparation_cancels_and_survives_drain(
    policy_keyboard, key, preparation
):
    case = policy_keyboard
    setattr(case.effects, f"during_{preparation}", lambda: case.press("t", key, "b"))
    case.press("h" if preparation == "home" else "t", "t", "b")
    case.operator.poll()
    assert case.calls == [preparation]
    if preparation == "home":
        assert case.effects.home_cancelled
        assert case.operator.home_results == [{"outcome": "interrupted"}]
    assert not case.shared.start_request.value
    assert case.shared.stop_request.value == (0 if key == "esc" else int(StopRequest.OPERATOR))
    assert case.shared.quit_requested.value == (key == "q")
    assert case.shared.estop_request.value == (key == "esc")
    assert case.operator.keyboard.estop_latched == (key == "esc")
    assert case.shared.run_id.value == (1 if key == "esc" else 2)
    expected = {
        "s": OperatorCommand.STOP,
        "q": OperatorCommand.QUIT,
        "esc": OperatorCommand.EMERGENCY_STOP,
    }
    assert case.operator.keyboard.poll(timeout=0) == [expected[key]]


def test_owner_tare_stops_before_sampling_and_passes_cancellation():
    import threading

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
