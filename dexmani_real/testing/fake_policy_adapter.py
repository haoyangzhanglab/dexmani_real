"""Deterministic inference adapter for offline tests; never deploy to hardware."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from dexmani_real.ipc.schema import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE
from dexmani_real.policy.runtime import ActionCandidate, ObservationSnapshot
from dexmani_real.policy.spec import PolicySpec


@dataclass
class _FakePolicy:
    spec: PolicySpec
    action_id: int = 0


def load_policy(spec: PolicySpec) -> object:
    return _FakePolicy(spec)


def predict(policy: object, observation: ObservationSnapshot) -> ActionCandidate:
    if not isinstance(policy, _FakePolicy):
        raise TypeError("fake adapter received an unexpected policy object")
    policy.action_id += 1
    now_ns = time.monotonic_ns()
    target_ns = max(now_ns, observation.anchor_monotonic_ns) + int(policy.spec.action.dt_s * 1e9)
    return ActionCandidate(
        observation_id=observation.observation_id,
        session_generation=observation.session_generation,
        policy_epoch=0,
        action_id=policy.action_id,
        created_monotonic_ns=now_ns,
        target_monotonic_ns=target_ns,
        valid_until_monotonic_ns=target_ns + int(policy.spec.action.deadline_s * 1e9),
        arm_qpos=np.zeros(ARM_JOINT_SHAPE, dtype=np.float64),
        hand_qpos=(np.zeros(HAND_JOINT_SHAPE, dtype=np.float64) if "hand" in policy.spec.actuators else None),
        is_hold=True,
    )
