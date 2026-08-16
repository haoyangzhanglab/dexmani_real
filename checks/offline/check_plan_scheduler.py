"""P8: the learned-policy coordinator and endpoint scheduler.

Locks the coordinator's correctness guarantees without hardware:

  - ``_select_due_step`` coalesces overdue steps to the latest due endpoint and
    publishes nothing when no step is due yet (§76/§77), skipping invalid steps.
  - ``_adoptable`` drops stale-generation / stale-observation / expired / malformed
    plans (§75).
  - the first optional hand-delta reference is seeded from healthy measured
    feedback; the shared publication boundary owns the coupled preflight (§74).
  - ``_abort_policy_run`` advances the generation and drops RUNNING -> ARMED
    (policy-semantic failure, not a hardware FAULT) (§80.2/§82).

Two end-to-end runs prove the wiring: a due plan produces exactly one coalesced
endpoint, and a command-to-command silence timeout (after one publish) aborts the
run to ARMED.
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
from dexmani_real.deployment.coordinator import (
    CoordinatorConfig,
    _abort_policy_run,
    _adoptable,
    _seed_hand_reference,
    _select_due_step,
    coordinator_loop,
)
from dexmani_real.robot.safety import SafetyState, transition
from dexmani_real.shm.shared_storage import SharedStorage
from dexmani_real.utils.schema import MAX_POLICY_CHUNK_STEPS, POLICY_PLAN_DTYPE


def _mid(low: tuple[float, ...], high: tuple[float, ...]) -> np.ndarray:
    lo = np.asarray(low, dtype=np.float64)
    hi = np.asarray(high, dtype=np.float64)
    return (lo + hi) / 2.0


def _coordinator_config(
    *,
    hand_max_delta_rad: float | None = None,
    control_hz: float = 16.0,
    **deployment_overrides,
) -> CoordinatorConfig:
    return CoordinatorConfig(
        deployment=DeploymentConfig(**deployment_overrides),
        arm_joint_lower_rad=arm_defaults.joint_limit_lower,
        arm_joint_upper_rad=arm_defaults.joint_limit_upper,
        hand_joint_lower_rad=hand_defaults.qpos_min_rad,
        hand_joint_upper_rad=hand_defaults.qpos_max_rad,
        hand_mechanical_lower_rad=hand_defaults.mechanical_qpos_min_rad,
        hand_mechanical_upper_rad=hand_defaults.mechanical_qpos_max_rad,
        hand_max_delta_rad=(
            hand_defaults.max_delta_rad if hand_max_delta_rad is None else hand_max_delta_rad
        ),
        hand_feedback_max_age_s=1.0,
        control_hz=control_hz,
    )


def _plan_frame(
    *,
    run_generation: int = 1,
    observation_id: int = 1,
    num_steps: int = 4,
    arm_qpos: np.ndarray | None = None,
    hand_qpos: np.ndarray | None = None,
    target_ns: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    hand_present: int = 0,
    anchor_ns: int | None = None,
    started_ns: int | None = None,
    finished_ns: int | None = None,
) -> np.ndarray:
    frame = np.zeros(1, dtype=POLICY_PLAN_DTYPE)
    now_ns = time.monotonic_ns()
    frame["plan_id"][0] = 1
    frame["run_generation"][0] = run_generation
    frame["observation_id"][0] = observation_id
    frame["observation_anchor_monotonic_ns"][0] = (
        now_ns - 10_000_000 if anchor_ns is None else anchor_ns
    )
    frame["inference_started_monotonic_ns"][0] = (
        now_ns - 5_000_000 if started_ns is None else started_ns
    )
    frame["inference_finished_monotonic_ns"][0] = (
        now_ns - 1_000_000 if finished_ns is None else finished_ns
    )
    frame["num_steps"][0] = num_steps
    frame["arm_present"][0] = 1
    frame["hand_present"][0] = hand_present
    if arm_qpos is not None:
        frame["arm_qpos"][0, :num_steps] = arm_qpos
    if hand_qpos is not None:
        frame["hand_qpos"][0, :num_steps] = hand_qpos
    if target_ns is not None:
        frame["target_monotonic_ns"][0, :num_steps] = target_ns
    frame["valid_mask"][0, :num_steps] = 1 if valid_mask is None else valid_mask
    return frame


def _adopt_args(**overrides) -> dict:
    args = dict(
        current_generation=5,
        last_observation_id=0,
        now_ns=time.monotonic_ns(),
        max_plan_age_ns=int(1e9),
        max_observation_age_ns=int(5e8),
    )
    args.update(overrides)
    return args


def _test_end_to_end_publish(arm_mid: np.ndarray) -> None:
    shared = SharedStorage.create(prefix="check_plan_scheduler_e2e")
    try:
        assert transition(shared, SafetyState.ARMED)
        shared.arm_state_ring.write(make_arm_state_frame(arm_mid))

        n = 4
        arm_steps = arm_mid[None, :] + np.arange(n, dtype=np.float64)[:, None] * 0.001
        target = np.asarray(time.monotonic_ns() - 1_000_000, dtype=np.uint64) + (
            np.arange(n, dtype=np.uint64) * np.uint64(1000)
        )
        # The coordinator advances the generation to 2 on RUNNING entry, so the
        # plan it adopts must already carry generation 2 (what the inference
        # worker would produce after observing the advance).
        plan = _plan_frame(
            run_generation=2,
            observation_id=1,
            num_steps=n,
            arm_qpos=arm_steps,
            target_ns=target,
            hand_present=0,
        )
        shared.policy_plan_ring.write(plan)

        thread = threading.Thread(
            target=coordinator_loop, args=(shared, _coordinator_config()), daemon=True
        )
        thread.start()
        try:
            try:
                command = shared.arm_action_q.get(timeout=3.0)
            except Empty:
                raise AssertionError("coordinator published no endpoint") from None
            np.testing.assert_allclose(
                np.asarray(command["qpos_cmd"][0]), arm_steps[3], atol=1e-12
            )
            # Coalescing: every step was due, but only the latest is published.
            try:
                shared.arm_action_q.get(timeout=0.4)
            except Empty:
                pass
            else:
                raise AssertionError("coordinator published >1 endpoint (coalescing violated)")
        finally:
            shared.is_running.value = False
            thread.join(timeout=5.0)
            assert not thread.is_alive(), "coordinator thread failed to exit"
    finally:
        shared.close()


def _test_silence_abort(arm_mid: np.ndarray) -> None:
    shared = SharedStorage.create(prefix="check_plan_scheduler_silence")
    try:
        assert transition(shared, SafetyState.ARMED)
        shared.arm_state_ring.write(make_arm_state_frame(arm_mid))

        # Publish exactly one command (generation 2, single due step), then go
        # silent: the command-to-command silence watchdog must drop to ARMED.
        plan = _plan_frame(
            run_generation=2,
            observation_id=1,
            num_steps=1,
            arm_qpos=arm_mid[None, :],
            target_ns=np.asarray([time.monotonic_ns() - 1_000_000], dtype=np.uint64),
            hand_present=0,
        )
        shared.policy_plan_ring.write(plan)

        cfg = _coordinator_config(max_command_silence_s=0.05, control_hz=100.0)
        thread = threading.Thread(target=coordinator_loop, args=(shared, cfg), daemon=True)
        thread.start()
        try:
            try:
                shared.arm_action_q.get(timeout=3.0)
            except Empty:
                raise AssertionError("coordinator published no endpoint") from None
            # No further plan/commands: the watchdog must drop the run to ARMED.
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if int(shared.safety_state.value) == int(SafetyState.ARMED):
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("silence watchdog did not abort to ARMED")
            assert int(shared.run_generation.value) >= 2, "silence abort must advance generation"
        finally:
            shared.is_running.value = False
            thread.join(timeout=5.0)
            assert not thread.is_alive(), "coordinator thread failed to exit"
    finally:
        shared.close()


def main() -> int:
    arm_mid = _mid(arm_defaults.joint_limit_lower, arm_defaults.joint_limit_upper)
    hand_mid = _mid(hand_defaults.qpos_min_rad, hand_defaults.qpos_max_rad)

    # ── scheduler: coalesce / no-due / valid_mask skip ──
    target = np.array([100, 200, 300, 400], dtype=np.uint64)
    valid = np.ones(4, dtype=np.uint8)
    sel, nxt = _select_due_step(target, valid, 4, 0, 250)
    assert sel == 1 and nxt == 2, "coalesce to latest due (step 1)"
    sel, nxt = _select_due_step(target, valid, 4, 0, 50)
    assert sel is None and nxt == 0, "no due step -> publish nothing"
    sel, nxt = _select_due_step(target, valid, 4, 0, 500)
    assert sel == 3 and nxt == 4, "all due -> latest (step 3)"
    valid2 = np.array([1, 0, 1, 0], dtype=np.uint8)
    sel, nxt = _select_due_step(target, valid2, 4, 0, 250)
    assert sel == 0 and nxt == 1, "skip invalid step, stop at non-due"
    sel, nxt = _select_due_step(target, valid, 4, 2, 500)
    assert sel == 3 and nxt == 4, "respect consumed next_step"

    # ── adoption gate ──
    rec = _plan_frame(run_generation=5, observation_id=3, num_steps=4)[0]
    ok, _ = _adoptable(rec, **_adopt_args())
    assert ok, "fresh matching plan is adoptable"
    assert not _adoptable(rec, **_adopt_args(current_generation=6))[0], "generation mismatch"
    assert not _adoptable(rec, **_adopt_args(last_observation_id=10))[0], "stale observation"
    assert not _adoptable(rec, **_adopt_args(max_plan_age_ns=int(1e3)))[0], "plan expired"
    assert not _adoptable(rec, **_adopt_args(max_observation_age_ns=int(1e3)))[0], "observation expired"

    bad_n = _plan_frame(run_generation=5, num_steps=0)[0]
    assert not _adoptable(bad_n, **_adopt_args())[0], "zero num_steps"
    over = _plan_frame(run_generation=5, num_steps=4)
    over["num_steps"][0] = MAX_POLICY_CHUNK_STEPS + 1
    assert not _adoptable(over[0], **_adopt_args())[0], "over-capacity num_steps"
    bad_mask = _plan_frame(run_generation=5, num_steps=4)
    bad_mask["valid_mask"][0, 2] = 2
    assert not _adoptable(bad_mask[0], **_adopt_args())[0], "non-binary valid_mask"

    # ── abort semantics: advance generation + RUNNING -> ARMED (no FAULT) ──
    shared = SharedStorage.create(prefix="check_plan_scheduler_abort")
    try:
        assert transition(shared, SafetyState.ARMED)
        assert transition(shared, SafetyState.RUNNING)
        gen = int(shared.run_generation.value)
        _abort_policy_run(shared, "test abort")
        assert int(shared.run_generation.value) == gen + 1
        assert int(shared.safety_state.value) == int(SafetyState.ARMED)
        assert int(shared.safety_state.value) != int(SafetyState.FAULT)
    finally:
        shared.close()

    # ── hand-delta reference seeding ──
    shared2 = SharedStorage.create(prefix="check_plan_scheduler_seed")
    try:
        shared2.hand_state_ring.write(make_hand_state_frame(hand_mid))
        seed = _seed_hand_reference(shared2, hand_feedback_max_age_s=1.0)
        assert seed is not None, "valid hand feedback must seed the reference"
        np.testing.assert_allclose(seed, hand_mid)
        shared2.hand_state_ring.write(make_hand_state_frame(hand_mid, send_healthy=0))
        assert (
            _seed_hand_reference(shared2, hand_feedback_max_age_s=1.0) is None
        ), "unhealthy command I/O must not seed"
    finally:
        shared2.close()

    # ── end-to-end wiring ──
    _test_end_to_end_publish(arm_mid)
    _test_silence_abort(arm_mid)

    print("check_plan_scheduler: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
