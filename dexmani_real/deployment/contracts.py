"""Backend-neutral deployment contracts (execution doc §52–§56).

The three Protocols define the model boundary; ``JointActionChunk`` is the
runtime-canonical joint action. None of these import torch or SharedStorage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from dexmani_real.deployment.observation import ObservationBatch, _freeze
from dexmani_real.utils.schema import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE, MAX_POLICY_CHUNK_STEPS


@dataclass(frozen=True)
class InferenceContext:
    """Timing/identity context passed to ``ActionAdapter.decode``.

    Field shape is design latitude (the execution doc references the type but
    does not enumerate its fields); these six suffice for generation checks and
    metric timing.
    """

    run_generation: int
    observation_id: int
    observation_anchor_monotonic_ns: int
    inference_started_monotonic_ns: int
    inference_finished_monotonic_ns: int
    step_dt_ns: int

    def __post_init__(self) -> None:
        if self.run_generation < 0 or self.observation_id < 0:
            raise ValueError("run_generation and observation_id must be non-negative")
        if min(
            self.observation_anchor_monotonic_ns,
            self.inference_started_monotonic_ns,
            self.inference_finished_monotonic_ns,
        ) <= 0:
            raise ValueError("context timestamps must be positive")
        if self.inference_started_monotonic_ns > self.inference_finished_monotonic_ns:
            raise ValueError("inference finish precedes start")
        if self.step_dt_ns <= 0:
            raise ValueError("step_dt_ns must be positive")


@dataclass(frozen=True)
class JointActionChunk:
    """Runtime-canonical joint action chunk (execution doc §56).

    Fixed representation: joint_position / rad / robot_joint. ``hand_qpos`` is
    None when the policy does not command the hand. All ``N`` steps share the
    leading axis; ``target_monotonic_ns`` holds strictly increasing intended
    execution timestamps; ``valid_mask[i] == 0`` marks a step to skip.
    """

    arm_qpos: np.ndarray  # [N, 7] float64 rad
    hand_qpos: np.ndarray | None  # [N, 12] float64 rad
    target_monotonic_ns: np.ndarray  # [N] uint64
    valid_mask: np.ndarray  # [N] uint8

    def __post_init__(self) -> None:
        arm = _freeze(self.arm_qpos, name="arm_qpos", dtype=np.float64)
        if arm.ndim != 2 or arm.shape[1] != ARM_JOINT_SHAPE[0]:
            raise ValueError(f"arm_qpos must be [N, {ARM_JOINT_SHAPE[0]}], got {arm.shape}")
        n = arm.shape[0]
        if n == 0:
            raise ValueError("JointActionChunk must have at least one step")
        if n > MAX_POLICY_CHUNK_STEPS:
            raise ValueError(
                f"JointActionChunk has {n} steps; transport capacity is {MAX_POLICY_CHUNK_STEPS}"
            )

        hand = self.hand_qpos
        if hand is not None:
            hand = _freeze(hand, name="hand_qpos", dtype=np.float64)
            if hand.ndim != 2 or hand.shape != (n, HAND_JOINT_SHAPE[0]):
                raise ValueError(
                    f"hand_qpos must be [N, {HAND_JOINT_SHAPE[0]}] matching arm N={n}, got {hand.shape}"
                )
            object.__setattr__(self, "hand_qpos", hand)

        target = _freeze(self.target_monotonic_ns, name="target_monotonic_ns", dtype=np.uint64)
        if target.shape != (n,):
            raise ValueError(f"target_monotonic_ns must have shape ({n},), got {target.shape}")
        # Element-wise comparison, not np.diff: uint64 diff underflows on a decrease.
        if n > 1 and not bool(np.all(target[1:] > target[:-1])):
            raise ValueError("target_monotonic_ns must be strictly increasing")

        mask = _freeze(self.valid_mask, name="valid_mask", dtype=np.uint8)
        if mask.shape != (n,):
            raise ValueError(f"valid_mask must have shape ({n},), got {mask.shape}")
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError("valid_mask must be 0 or 1")

        object.__setattr__(self, "arm_qpos", arm)
        object.__setattr__(self, "target_monotonic_ns", target)
        object.__setattr__(self, "valid_mask", mask)


@runtime_checkable
class PolicyBackend(Protocol):
    """Model-side inference; may import torch/checkpoint/CUDA (execution doc §53)."""

    def load(self) -> None: ...

    def reset(self, *, run_generation: int) -> None: ...

    def infer(self, model_input: Any) -> Any: ...

    def close(self) -> None: ...


@runtime_checkable
class ObservationAdapter(Protocol):
    """``ObservationBatch`` -> model-native input."""

    def encode(self, observation: ObservationBatch) -> Any: ...


@runtime_checkable
class ActionAdapter(Protocol):
    """Model-native output -> ``JointActionChunk``."""

    def decode(self, raw_output: Any, *, context: InferenceContext) -> JointActionChunk: ...
