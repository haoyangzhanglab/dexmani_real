"""Offline contracts for configured hand-first teleoperation homing."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import numpy as np

from dexmani_real.config.experiment import resolve_experiment_config
from dexmani_real.control.arm_homing import ArmHomeConfig
from dexmani_real.ipc.schema import ARM_STATE_DTYPE
from dexmani_real.teleop.config import TeleopConfig
from dexmani_real.teleop.homing import do_configured_teleop_home


class _Value:
    def __init__(self, value: bool) -> None:
        self.value = value


class _Shared:
    def __init__(self, *, error_state: bool = False) -> None:
        self.error_state = _Value(error_state)


class _Planner:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.hand_qpos: np.ndarray | None = None

    def set_hand_qpos(self, hand_qpos: np.ndarray) -> None:
        self.events.append("planner.set_hand_qpos")
        self.hand_qpos = hand_qpos.copy()


def _healthy_arm_state(qpos: np.ndarray, source_monotonic_ns: int) -> np.ndarray:
    state = np.zeros(1, dtype=ARM_STATE_DTYPE)
    state["qpos"][0] = qpos
    state["connected"][0] = True
    state["error_code"][0] = 0
    state["source_monotonic_ns"][0] = source_monotonic_ns
    state["state_valid"][0] = True
    return state


def test_configured_homing_keeps_hand_sdk_acceptance_before_arm_home():
    runtime = resolve_experiment_config()
    config = TeleopConfig.from_runtime(runtime)
    events: list[str] = []
    planner = _Planner(events)
    audio = SimpleNamespace(queue=lambda cue: events.append(f"audio.{cue}"))
    mapper = SimpleNamespace(clear=lambda: events.append("mapper.clear"))
    retargeter = object()
    previous_hand_qpos = np.full(12, -0.1)
    arm_qpos = np.linspace(-0.2, 0.2, 7)
    source_monotonic_ns = 1_000_000_000
    state = _healthy_arm_state(arm_qpos, source_monotonic_ns)
    hand_call: dict[str, object] = {}
    arm_call: dict[str, object] = {}

    def request_hand_home(shared, hand_home_qpos, **kwargs):
        events.append("hand.sdk_acceptance")
        hand_call["shared"] = shared
        hand_call["qpos"] = hand_home_qpos.copy()
        hand_call["kwargs"] = kwargs
        return True

    def read_arm_state(shared):
        assert shared is hand_call["shared"]
        events.append("arm.feedback")
        return state

    def execute_home(shared, home_qpos, **kwargs):
        events.append("arm.home")
        arm_call["shared"] = shared
        arm_call["qpos"] = home_qpos.copy()
        arm_call["kwargs"] = kwargs
        return SimpleNamespace(succeeded=True)

    def estop_requested() -> bool:
        return False

    with (
        mock.patch(
            "dexmani_real.teleop.homing.reset_hand_retargeter",
            side_effect=lambda _retargeter: events.append("retargeter.reset"),
        ),
        mock.patch(
            "dexmani_real.teleop.homing.publish_hand_home_and_wait_accepted",
            side_effect=request_hand_home,
        ),
        mock.patch(
            "dexmani_real.teleop.homing.read_arm_state_causal",
            side_effect=read_arm_state,
        ),
        mock.patch(
            "dexmani_real.teleop.homing.execute_arm_home",
            side_effect=execute_home,
        ),
        mock.patch(
            "dexmani_real.teleop.homing.time.monotonic_ns",
            return_value=source_monotonic_ns + 1_000_000,
        ),
    ):
        result = do_configured_teleop_home(
            _Shared(),
            config,
            hand_available=True,
            prev_hand_qpos=previous_hand_qpos,
            planner=planner,
            audio=audio,
            estop_requested=estop_requested,
            arm_mapper=mapper,
            hand_retargeter=retargeter,
        )

    expected_hand_home_qpos = np.deg2rad(
        np.asarray(runtime.hand.home_qpos_deg, dtype=np.float64)
    )
    np.testing.assert_array_equal(result, expected_hand_home_qpos)
    np.testing.assert_array_equal(planner.hand_qpos, expected_hand_home_qpos)
    assert events == [
        "mapper.clear",
        "retargeter.reset",
        "hand.sdk_acceptance",
        "planner.set_hand_qpos",
        "arm.feedback",
        "arm.home",
        "audio.home_done",
    ]

    assert hand_call["qpos"] is not None
    np.testing.assert_array_equal(hand_call["qpos"], expected_hand_home_qpos)
    hand_kwargs = hand_call["kwargs"]
    assert isinstance(hand_kwargs, dict)
    np.testing.assert_array_equal(
        hand_kwargs["command_lower_rad"], runtime.hand.qpos_min_rad
    )
    np.testing.assert_array_equal(
        hand_kwargs["command_upper_rad"], runtime.hand.qpos_max_rad
    )
    np.testing.assert_array_equal(
        hand_kwargs["mechanical_lower_rad"], runtime.hand.mechanical_qpos_min_rad
    )
    np.testing.assert_array_equal(
        hand_kwargs["mechanical_upper_rad"], runtime.hand.mechanical_qpos_max_rad
    )
    assert (
        hand_kwargs["hand_feedback_max_age_s"]
        == runtime.safety.heartbeat_timeouts["hand"]
    )
    assert hand_kwargs["timeout_s"] == runtime.hand.home_command_ack_timeout_s
    assert hand_kwargs["heartbeat"] is True
    assert hand_kwargs["abort_requested"] is estop_requested

    np.testing.assert_array_equal(arm_call["qpos"], runtime.arm.home_qpos)
    arm_kwargs = arm_call["kwargs"]
    assert isinstance(arm_kwargs, dict)
    assert arm_kwargs["planner"] is planner
    assert arm_kwargs["config"] == ArmHomeConfig.from_runtime(
        runtime,
        publish_policy_heartbeat=True,
    )
    assert arm_kwargs["table_z_surface_m"] == runtime.arm.table_z_surface_m
    np.testing.assert_array_equal(arm_kwargs["current_qpos"], arm_qpos)
    assert arm_kwargs["estop_requested"] is estop_requested
    assert callable(arm_kwargs["progress"])


def test_configured_homing_does_not_start_arm_home_after_hand_rejection():
    config = TeleopConfig.from_runtime(resolve_experiment_config())
    previous_hand_qpos = np.full(12, -0.1)
    planner = _Planner([])

    def unexpected_audio(_cue: str) -> None:
        raise AssertionError("audio must not announce a rejected home")

    audio = SimpleNamespace(queue=unexpected_audio)

    with (
        mock.patch(
            "dexmani_real.teleop.homing.publish_hand_home_and_wait_accepted",
            return_value=False,
        ) as hand_home,
        mock.patch(
            "dexmani_real.teleop.homing.read_arm_state_causal",
            side_effect=AssertionError("arm feedback must not be read"),
        ),
        mock.patch(
            "dexmani_real.teleop.homing.execute_arm_home",
            side_effect=AssertionError("arm home must not start"),
        ),
    ):
        result = do_configured_teleop_home(
            _Shared(),
            config,
            hand_available=True,
            prev_hand_qpos=previous_hand_qpos,
            planner=planner,
            audio=audio,
            estop_requested=lambda: False,
        )

    assert result is previous_hand_qpos
    hand_home.assert_called_once()
    assert planner.hand_qpos is None


def test_configured_homing_requires_fresh_arm_feedback_after_hand_acceptance():
    runtime = resolve_experiment_config()
    config = TeleopConfig.from_runtime(runtime)
    events: list[str] = []
    planner = _Planner(events)
    audio = SimpleNamespace(queue=lambda cue: events.append(f"audio.{cue}"))
    previous_hand_qpos = np.full(12, -0.1)
    stale_source_ns = 1_000_000_000
    state = _healthy_arm_state(np.zeros(7), stale_source_ns)

    with (
        mock.patch(
            "dexmani_real.teleop.homing.publish_hand_home_and_wait_accepted",
            return_value=True,
        ),
        mock.patch(
            "dexmani_real.teleop.homing.read_arm_state_causal",
            return_value=state,
        ),
        mock.patch(
            "dexmani_real.teleop.homing.execute_arm_home",
            side_effect=AssertionError("stale feedback must cancel arm home"),
        ),
        mock.patch(
            "dexmani_real.teleop.homing.time.monotonic_ns",
            return_value=stale_source_ns
            + int((runtime.arm.homing.state_max_age_s + 0.01) * 1e9),
        ),
    ):
        result = do_configured_teleop_home(
            _Shared(),
            config,
            hand_available=True,
            prev_hand_qpos=previous_hand_qpos,
            planner=planner,
            audio=audio,
            estop_requested=lambda: False,
        )

    np.testing.assert_array_equal(
        result,
        np.deg2rad(np.asarray(runtime.hand.home_qpos_deg, dtype=np.float64)),
    )
    assert planner.hand_qpos is not None
    assert events == ["planner.set_hand_qpos"]


def test_configured_homing_keeps_fixed_hand_fallback_when_runtime_disables_hand():
    runtime = resolve_experiment_config()
    fixed_hand_runtime = replace(
        runtime,
        policy=replace(runtime.policy, hand_enabled=False),
    )
    config = TeleopConfig.from_runtime(fixed_hand_runtime)
    events: list[str] = []
    planner = _Planner(events)
    audio = SimpleNamespace(queue=lambda cue: events.append(f"audio.{cue}"))
    source_monotonic_ns = 1_000_000_000
    state = _healthy_arm_state(np.zeros(7), source_monotonic_ns)

    def read_arm_state(_shared):
        events.append("arm.feedback")
        return state

    def execute_home(_shared, _home_qpos, **_kwargs):
        events.append("arm.home")
        return SimpleNamespace(succeeded=True)

    with (
        mock.patch(
            "dexmani_real.teleop.homing.publish_hand_home_and_wait_accepted",
            side_effect=AssertionError("fixed hand fallback must not publish"),
        ),
        mock.patch(
            "dexmani_real.teleop.homing.read_arm_state_causal",
            side_effect=read_arm_state,
        ),
        mock.patch(
            "dexmani_real.teleop.homing.execute_arm_home",
            side_effect=execute_home,
        ),
        mock.patch(
            "dexmani_real.teleop.homing.time.monotonic_ns",
            return_value=source_monotonic_ns + 1_000_000,
        ),
    ):
        result = do_configured_teleop_home(
            _Shared(error_state=True),
            config,
            hand_available=True,
            prev_hand_qpos=np.full(12, -0.1),
            planner=planner,
            audio=audio,
            estop_requested=lambda: False,
        )

    np.testing.assert_array_equal(
        result,
        np.deg2rad(np.asarray(fixed_hand_runtime.hand.home_qpos_deg, dtype=np.float64)),
    )
    assert events == [
        "planner.set_hand_qpos",
        "arm.feedback",
        "arm.home",
        "audio.home_done",
    ]
