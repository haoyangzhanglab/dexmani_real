"""P10: policy deployment lifecycle worker-set composition (§83).

Locks the worker-spec builder so the lifecycle spawns exactly the joint-only
workflow and its ready/heartbeat names stay consistent with the supervisor's
``proc_names == heartbeat_names`` invariant:

  - always ``arm``, ``inference``, ``policy``
  - ``hand`` only when ``deployment.hand_enabled``
  - ``ready_name == process name`` and unique for every spec
  - every ready name resolves against the runtime readiness/heartbeat timeouts
"""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401  (repo root on sys.path)

from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.deployment.config import DeploymentConfig
from dexmani_real.deployment.lifecycle import build_policy_worker_specs
from dexmani_real.shm.shared_storage import SharedStorage


def _spec_names(specs) -> tuple[list[str], list[str]]:
    names = [spec.name for spec in specs]
    ready_names = [spec.ready_name for spec in specs]
    return names, ready_names


def _assert_consistent(runtime, specs, *, expect_hand: bool) -> None:
    names, ready_names = _spec_names(specs)
    expected = ["arm", "inference", "policy"] + (["hand"] if expect_hand else [])
    assert sorted(names) == sorted(expected), f"unexpected worker set: {names}"
    assert sorted(ready_names) == sorted(expected), f"unexpected ready set: {ready_names}"
    assert len(set(names)) == len(names), "worker names must be unique"
    assert len(set(ready_names)) == len(ready_names), "ready names must be unique"
    # Every ready name and heartbeat name must resolve in the runtime safety
    # timeouts, or wait_subsystem_ready / run_supervisor would raise.
    for name in ready_names:
        assert name in runtime.safety.readiness_timeouts_s, f"missing readiness timeout for {name}"
    for name in names:
        assert name in runtime.safety.heartbeat_timeouts, f"missing heartbeat timeout for {name}"


def main() -> int:
    runtime = resolve_runtime_config()
    shared = SharedStorage.create(prefix="check_policy_lifecycle")
    try:
        # ── joint-only (default): arm, inference, policy ──
        specs = build_policy_worker_specs(shared, runtime, DeploymentConfig())
        _assert_consistent(runtime, specs, expect_hand=False)

        # ── coupled-hand: adds exactly one hand worker ──
        hand_deployment = DeploymentConfig(hand_enabled=True)
        specs = build_policy_worker_specs(shared, runtime, hand_deployment)
        _assert_consistent(runtime, specs, expect_hand=True)

        # ── every spec targets the plain *_loop(shared, config) shape ──
        for spec in specs:
            assert len(spec.args) == 2, f"worker {spec.name} must take (shared, config)"
            assert spec.args[0] is shared
            assert spec.ready_name is not None
    finally:
        shared.close()

    print("check_policy_lifecycle: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
