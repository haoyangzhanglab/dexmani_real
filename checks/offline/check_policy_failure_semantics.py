"""P9: learned-policy failure semantics (§80).

Locks the coordinator's failure taxonomy:

  - A **SafetyGate rejection** (the model proposed an invalid endpoint) is a
    policy-semantic failure: abort the run (advance generation, ``RUNNING ->
    ARMED``, not FAULT), distinct from transient failures.
  - **Feedback unavailable/unhealthy** and **transport backpressure** are
    transient: drop this tick, no abort; the command-silence watchdog is the
    eventual backstop.

The distinction rides on ``validate_and_send_candidate``'s ``reject_reason_out``
out-param, populated *only* when the gate rejects — so the coordinator can abort
on rejection while still dropping feedback/transport failures.
"""

from __future__ import annotations

import sys
import threading
import time
from queue import Empty

import numpy as np

import _bootstrap  # noqa: F401  (repo root on sys.path)
from _fakes import make_arm_state_frame

from dexmani_real.config.defaults import arm as arm_defaults
from dexmani_real.config.defaults import hand as hand_defaults
from dexmani_real.deployment.config import DeploymentConfig
from dexmani_real.deployment.coordinator import CoordinatorConfig, coordinator_loop
from dexmani_real.policy.safety import (
    SafetyGate,
    build_action_candidate,
    validate_and_send_candidate,
)
from dexmani_real.robot.safety import SafetyState, transition
from dexmani_real.shm.shared_storage import SharedStorage
from dexmani_real.utils.schema import POLICY_PLAN_DTYPE


def _mid(low: tuple[float, ...], high: tuple[float, ...]) -> np.ndarray:
    lo = np.asarray(low, dtype=np.float64)
    hi = np.asarray(high, dtype=np.float64)
    return (lo + hi) / 2.0


def _gate() -> SafetyGate:
    return SafetyGate(
        arm_joint_lower_rad=arm_defaults.joint_limit_lower,
        arm_joint_upper_rad=arm_defaults.joint_limit_upper,
        hand_joint_lower_rad=hand_defaults.qpos_min_rad,
        hand_joint_upper_rad=hand_defaults.qpos_max_rad,
    )


def _coordinator_config() -> CoordinatorConfig:
    return CoordinatorConfig(
        deployment=DeploymentConfig(),
        arm_joint_lower_rad=arm_defaults.joint_limit_lower,
        arm_joint_upper_rad=arm_defaults.joint_limit_upper,
        hand_joint_lower_rad=hand_defaults.qpos_min_rad,
        hand_joint_upper_rad=hand_defaults.qpos_max_rad,
        hand_mechanical_lower_rad=hand_defaults.mechanical_qpos_min_rad,
        hand_mechanical_upper_rad=hand_defaults.mechanical_qpos_max_rad,
        hand_max_delta_rad=None,
        control_hz=100.0,
    )


def _drain_arm_queue(shared: SharedStorage) -> int:
    count = 0
    while True:
        try:
            shared.arm_action_q.get_nowait()
            count += 1
        except Empty:
            return count


def _test_gate_reject_abort(arm_mid: np.ndarray) -> None:
    """A due plan whose endpoint violates the joint limit must abort to ARMED."""
    shared = SharedStorage.create(prefix="check_policy_failure_semantics_abort")
    try:
        assert transition(shared, SafetyState.ARMED)
        shared.arm_state_ring.write(make_arm_state_frame(arm_mid))

        n = 2
        bad_arm = np.asarray(arm_defaults.joint_limit_upper, dtype=np.float64) + 10.0
        plan = np.zeros(1, dtype=POLICY_PLAN_DTYPE)
        now_ns = time.monotonic_ns()
        plan["plan_id"][0] = 1
        plan["run_generation"][0] = 2  # coordinator advances to 2 on RUNNING entry
        plan["observation_id"][0] = 1
        plan["observation_anchor_monotonic_ns"][0] = now_ns - 10_000_000
        plan["inference_started_monotonic_ns"][0] = now_ns - 5_000_000
        plan["inference_finished_monotonic_ns"][0] = now_ns - 1_000_000
        plan["num_steps"][0] = n
        plan["arm_present"][0] = 1
        plan["hand_present"][0] = 0
        plan["arm_qpos"][0, :n] = np.tile(bad_arm, (n, 1))
        plan["target_monotonic_ns"][0, :n] = (
            np.asarray(now_ns - 1_000_000, dtype=np.uint64)
            + np.arange(n, dtype=np.uint64) * np.uint64(1000)
        )
        plan["valid_mask"][0, :n] = 1
        shared.policy_plan_ring.write(plan)

        thread = threading.Thread(
            target=coordinator_loop, args=(shared, _coordinator_config()), daemon=True
        )
        thread.start()
        try:
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if int(shared.safety_state.value) == int(SafetyState.ARMED):
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("gate rejection did not abort to ARMED")
            assert int(shared.run_generation.value) >= 2, "abort must advance generation"
            assert int(shared.safety_state.value) != int(SafetyState.FAULT), "abort is not FAULT"
        finally:
            shared.is_running.value = False
            thread.join(timeout=5.0)
            assert not thread.is_alive(), "coordinator thread failed to exit"
    finally:
        shared.close()


def main() -> int:
    arm_mid = _mid(arm_defaults.joint_limit_lower, arm_defaults.joint_limit_upper)
    gate = _gate()

    shared = SharedStorage.create(prefix="check_policy_failure_semantics")
    try:
        # ── gate rejection populates reject_reason_out (and writes nothing) ──
        shared.arm_state_ring.write(make_arm_state_frame(arm_mid))
        bad_arm = np.asarray(arm_defaults.joint_limit_upper, dtype=np.float64) + 10.0
        candidate = build_action_candidate(shared, bad_arm, None)
        reason: list[str] = []
        assert (
            validate_and_send_candidate(
                shared, candidate, gate=gate, reject_reason_out=reason
            )
            is None
        ), "gate rejection must return None"
        assert reason, "gate rejection must populate reject_reason_out"
        assert reason[0] == "arm joint limit violation", reason
        assert _drain_arm_queue(shared) == 0, "gate rejection must not write transport"
    finally:
        shared.close()

    # ── feedback unavailable -> drop (empty reason) ──
    shared2 = SharedStorage.create(prefix="check_policy_failure_semantics_nofb")
    try:
        candidate = build_action_candidate(shared2, arm_mid, None)
        reason = []
        assert (
            validate_and_send_candidate(
                shared2, candidate, gate=gate, reject_reason_out=reason
            )
            is None
        ), "unavailable feedback must return None"
        assert reason == [], "feedback failure must not populate reject_reason_out"
    finally:
        shared2.close()

    # ── transport backpressure -> drop (empty reason) ──
    shared3 = SharedStorage.create(prefix="check_policy_failure_semantics_full")
    try:
        shared3.arm_state_ring.write(make_arm_state_frame(arm_mid))
        shared3.arm_action_q.put_nowait(np.zeros(1, dtype=np.float64))
        shared3.arm_action_q.put_nowait(np.zeros(1, dtype=np.float64))  # queue now full
        candidate = build_action_candidate(shared3, arm_mid, None)
        reason = []
        assert (
            validate_and_send_candidate(
                shared3, candidate, gate=gate, prepare_timeout_s=0.05, reject_reason_out=reason
            )
            is None
        ), "full arm queue must return None"
        assert reason == [], "transport failure must not populate reject_reason_out"
        _drain_arm_queue(shared3)
    finally:
        shared3.close()

    # ── end-to-end: gate rejection aborts the run to ARMED (not FAULT) ──
    _test_gate_reject_abort(arm_mid)

    print("check_policy_failure_semantics: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
