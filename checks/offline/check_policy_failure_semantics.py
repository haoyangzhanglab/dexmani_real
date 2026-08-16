"""P9: learned-policy failure semantics (§80).

Locks the coordinator's failure taxonomy:

  - A **SafetyGate or coupled-hand preflight rejection** (the model proposed an
    invalid endpoint) is a policy-semantic failure: abort the run (advance
    generation, ``RUNNING -> ARMED``, not FAULT), distinct from transient
    failures.
  - **Feedback unavailable/unhealthy** and **transport backpressure** are
    transient: drop this tick, no abort; the command-silence watchdog is the
    eventual backstop.

The distinction rides on ``validate_and_send_candidate``'s typed result:
``GATE_REJECTED`` and ``HAND_PREFLIGHT_REJECTED`` are policy-semantic, while
feedback and transport statuses remain transient drops.
"""

from __future__ import annotations

import sys
import threading
import time
from queue import Empty

import numpy as np

import _bootstrap  # noqa: F401  (repo root on sys.path)
from _fakes import make_arm_state_frame, make_hand_state_frame

from dexmani_real.config.defaults import arm as arm_defaults
from dexmani_real.config.defaults import hand as hand_defaults
from dexmani_real.deployment.config import DeploymentConfig
from dexmani_real.deployment.coordinator import CoordinatorConfig, coordinator_loop
from dexmani_real.policy.safety import (
    CommandPublishStatus,
    GateRejectCode,
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


def _coordinator_config(
    *,
    hand_enabled: bool = False,
    hand_max_delta_rad: float | None = None,
) -> CoordinatorConfig:
    return CoordinatorConfig(
        deployment=DeploymentConfig(hand_enabled=hand_enabled),
        arm_joint_lower_rad=arm_defaults.joint_limit_lower,
        arm_joint_upper_rad=arm_defaults.joint_limit_upper,
        hand_joint_lower_rad=hand_defaults.qpos_min_rad,
        hand_joint_upper_rad=hand_defaults.qpos_max_rad,
        hand_mechanical_lower_rad=hand_defaults.mechanical_qpos_min_rad,
        hand_mechanical_upper_rad=hand_defaults.mechanical_qpos_max_rad,
        hand_max_delta_rad=hand_max_delta_rad,
        hand_feedback_max_age_s=1.0,
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
                if (
                    int(shared.safety_state.value) == int(SafetyState.ARMED)
                    and int(shared.run_generation.value) >= 3
                ):
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("gate rejection did not abort to ARMED")
            assert int(shared.run_generation.value) >= 3, "entry + abort must advance generation"
            assert int(shared.safety_state.value) != int(SafetyState.FAULT), "abort is not FAULT"
        finally:
            shared.is_running.value = False
            thread.join(timeout=5.0)
            assert not thread.is_alive(), "coordinator thread failed to exit"
    finally:
        shared.close()


def _test_hand_preflight_abort(arm_mid: np.ndarray, hand_mid: np.ndarray) -> None:
    """A coupled delta violation must abort before any arm endpoint is queued."""
    shared = SharedStorage.create(prefix="check_policy_hand_preflight_abort")
    try:
        assert transition(shared, SafetyState.ARMED)
        shared.arm_state_ring.write(make_arm_state_frame(arm_mid))
        shared.hand_state_ring.write(make_hand_state_frame(hand_mid))

        hand_step = hand_mid.copy()
        hand_step[0] += 0.01
        plan = np.zeros(1, dtype=POLICY_PLAN_DTYPE)
        now_ns = time.monotonic_ns()
        plan["plan_id"][0] = 1
        plan["run_generation"][0] = 2
        plan["observation_id"][0] = 1
        plan["observation_anchor_monotonic_ns"][0] = now_ns - 10_000_000
        plan["inference_started_monotonic_ns"][0] = now_ns - 5_000_000
        plan["inference_finished_monotonic_ns"][0] = now_ns - 1_000_000
        plan["num_steps"][0] = 1
        plan["arm_present"][0] = 1
        plan["hand_present"][0] = 1
        plan["arm_qpos"][0, 0] = arm_mid
        plan["hand_qpos"][0, 0] = hand_step
        plan["target_monotonic_ns"][0, 0] = now_ns - 1_000_000
        plan["valid_mask"][0, 0] = 1
        shared.policy_plan_ring.write(plan)

        config = _coordinator_config(hand_enabled=True, hand_max_delta_rad=0.001)
        thread = threading.Thread(
            target=coordinator_loop,
            args=(shared, config),
            daemon=True,
        )
        thread.start()
        try:
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if (
                    int(shared.safety_state.value) == int(SafetyState.ARMED)
                    and int(shared.run_generation.value) >= 3
                ):
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("hand preflight rejection did not abort to ARMED")
            assert _drain_arm_queue(shared) == 0, "hand preflight must run before arm enqueue"
            assert int(shared.safety_state.value) != int(SafetyState.FAULT)
        finally:
            shared.is_running.value = False
            thread.join(timeout=5.0)
            assert not thread.is_alive(), "coordinator thread failed to exit"
    finally:
        shared.close()


def main() -> int:
    arm_mid = _mid(arm_defaults.joint_limit_lower, arm_defaults.joint_limit_upper)
    hand_mid = _mid(hand_defaults.qpos_min_rad, hand_defaults.qpos_max_rad)
    gate = _gate()

    shared = SharedStorage.create(prefix="check_policy_failure_semantics")
    try:
        assert transition(shared, SafetyState.ARMED)
        # ── gate rejection is typed and writes nothing ──
        shared.arm_state_ring.write(make_arm_state_frame(arm_mid))
        bad_arm = np.asarray(arm_defaults.joint_limit_upper, dtype=np.float64) + 10.0
        candidate = build_action_candidate(shared, bad_arm, None)
        assert candidate is not None
        result = validate_and_send_candidate(
            shared, candidate, gate=gate, hand_feedback_max_age_s=1.0
        )
        assert result.status == CommandPublishStatus.GATE_REJECTED, result
        assert result.gate_code == GateRejectCode.ARM_JOINT_LIMIT, result
        assert result.reason == "arm joint limit violation", result
        assert _drain_arm_queue(shared) == 0, "gate rejection must not write transport"
    finally:
        shared.close()

    # ── feedback unavailable -> drop (empty reason) ──
    shared2 = SharedStorage.create(prefix="check_policy_failure_semantics_nofb")
    try:
        assert transition(shared2, SafetyState.ARMED)
        candidate = build_action_candidate(shared2, arm_mid, None)
        assert candidate is not None
        result = validate_and_send_candidate(
            shared2, candidate, gate=gate, hand_feedback_max_age_s=1.0
        )
        assert result.status == CommandPublishStatus.ARM_FEEDBACK_UNAVAILABLE, result
    finally:
        shared2.close()

    # ── transport backpressure -> drop (empty reason) ──
    shared3 = SharedStorage.create(prefix="check_policy_failure_semantics_full")
    try:
        assert transition(shared3, SafetyState.ARMED)
        shared3.arm_state_ring.write(make_arm_state_frame(arm_mid))
        shared3.arm_action_q.put_nowait(np.zeros(1, dtype=np.float64))
        shared3.arm_action_q.put_nowait(np.zeros(1, dtype=np.float64))  # queue now full
        candidate = build_action_candidate(shared3, arm_mid, None)
        assert candidate is not None
        result = validate_and_send_candidate(
            shared3, candidate, gate=gate, hand_feedback_max_age_s=1.0, prepare_timeout_s=0.05
        )
        assert result.status == CommandPublishStatus.ARM_QUEUE_FULL, result
        _drain_arm_queue(shared3)
    finally:
        shared3.close()

    # ── end-to-end: gate rejection aborts the run to ARMED (not FAULT) ──
    _test_gate_reject_abort(arm_mid)
    _test_hand_preflight_abort(arm_mid, hand_mid)

    print("check_policy_failure_semantics: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
