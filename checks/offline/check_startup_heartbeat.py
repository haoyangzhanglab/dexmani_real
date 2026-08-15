"""Regression: startup heartbeat races (Phase C review H1/H2).

Two workers set a heartbeat once at startup and then block before their main
loop refreshes it: the coordinator blocks in its wait-for-ARMED loop, and the
inference worker spins on its no-arm-feedback ``continue`` path. Because
``run_supervisor`` checks heartbeats with no first-poll grace and ``policy``/
``inference`` timeouts are 1.0s/5.0s, a slow arm/inference startup used to FAULT
the whole deployment before any command. These two checks assert each worker's
heartbeat stays fresh (advances monotonically) while it blocks.
"""

from __future__ import annotations

import sys
import threading
import time

import _bootstrap  # noqa: F401  (repo root on sys.path)

from dexmani_real.config.defaults import arm as arm_defaults
from dexmani_real.config.defaults import hand as hand_defaults
from dexmani_real.deployment.config import DeploymentConfig
from dexmani_real.deployment.coordinator import CoordinatorConfig, coordinator_loop
from dexmani_real.deployment.worker import inference_loop
from dexmani_real.robot.safety import SafetyState
from dexmani_real.shm.shared_storage import SharedStorage

_FAKE_BACKEND = "dexmani_real.deployment.fake:FakePolicyBackend"
_FAKE_OBS = "dexmani_real.deployment.fake:FakeObservationAdapter"
_FAKE_ACT = "dexmani_real.deployment.fake:FakeActionAdapter"


def _coordinator_config() -> CoordinatorConfig:
    return CoordinatorConfig(
        deployment=DeploymentConfig(),
        arm_joint_lower_rad=arm_defaults.joint_limit_lower,
        arm_joint_upper_rad=arm_defaults.joint_limit_upper,
        hand_joint_lower_rad=hand_defaults.qpos_min_rad,
        hand_joint_upper_rad=hand_defaults.qpos_max_rad,
        hand_mechanical_lower_rad=hand_defaults.mechanical_qpos_min_rad,
        hand_mechanical_upper_rad=hand_defaults.mechanical_qpos_max_rad,
        hand_max_delta_rad=hand_defaults.max_delta_rad,
        control_hz=16.0,
    )


def _assert_heartbeat_advances(shared: SharedStorage, name: str) -> None:
    """Read the heartbeat twice and assert it is being refreshed, not stuck."""
    first = shared.get_heartbeat(name)
    time.sleep(0.5)
    second = shared.get_heartbeat(name)
    assert second > first, (
        f"{name} heartbeat must advance while blocked (got {first} -> {second})"
    )


def _test_coordinator_armed_wait() -> None:
    shared = SharedStorage.create(prefix="check_hb_coordinator")
    try:
        # Stay DISARMED so the coordinator blocks in its wait-for-ARMED loop.
        assert int(shared.safety_state.value) == int(SafetyState.DISARMED)
        thread = threading.Thread(
            target=coordinator_loop, args=(shared, _coordinator_config()), daemon=True
        )
        thread.start()
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not shared.is_ready("policy"):
                time.sleep(0.01)
            assert shared.is_ready("policy"), "coordinator did not mark ready"
            _assert_heartbeat_advances(shared, "policy")
        finally:
            shared.is_running.value = False
            thread.join(timeout=5.0)
            assert not thread.is_alive(), "coordinator thread failed to exit"
    finally:
        shared.close()


def _test_worker_no_feedback() -> None:
    shared = SharedStorage.create(prefix="check_hb_worker")
    try:
        config = DeploymentConfig(
            backend_target=_FAKE_BACKEND,
            observation_adapter_target=_FAKE_OBS,
            action_adapter_target=_FAKE_ACT,
        )
        thread = threading.Thread(target=inference_loop, args=(shared, config), daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not shared.is_ready("inference"):
                time.sleep(0.01)
            assert shared.is_ready("inference"), "inference worker did not mark ready"
            # No arm feedback: the worker spins on the arm_history-is-None
            # continue path. Its heartbeat must stay fresh regardless.
            _assert_heartbeat_advances(shared, "inference")
        finally:
            shared.is_running.value = False
            thread.join(timeout=5.0)
            assert not thread.is_alive(), "inference worker thread failed to exit"
    finally:
        shared.close()


def main() -> int:
    _test_coordinator_armed_wait()
    _test_worker_no_feedback()
    print("check_startup_heartbeat: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
