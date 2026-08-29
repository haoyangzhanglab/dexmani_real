"""Backend-neutral deployment contracts.

``PolicyRuntime`` defines the model boundary. ``PolicyPrediction`` is the
untimed model output, ``InferenceContext`` carries per-publish provenance, and
``JointActionChunk`` is the runtime-canonical timed joint/EE action. None of
these import torch or RuntimeChannels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from dexmani_real.deployment.observation import ObservationBatch, freeze_array
from dexmani_real.ipc.schema import (
    ARM_JOINT_SHAPE,
    EE_POS_DIM,
    EE_ROT6D_DIM,
    HAND_JOINT_SHAPE,
    MAX_POLICY_CHUNK_STEPS,
)


@dataclass(frozen=True)
class InferenceContext:
    """Timing/identity provenance carried from inference to plan publication.

    The causal cut, latest physical source time, and logical control-grid step
    are distinct: conflating them hides sensor age and shifts action timing.
    """

    run_generation: int
    observation_id: int
    observation_anchor_monotonic_ns: int
    observation_latest_source_monotonic_ns: int
    observation_logical_step_monotonic_ns: int
    inference_started_monotonic_ns: int
    inference_finished_monotonic_ns: int
    step_dt_ns: int

    def __post_init__(self) -> None:
        if self.run_generation < 0 or self.observation_id < 0:
            raise ValueError("run_generation and observation_id must be non-negative")
        if (
            min(
                self.observation_anchor_monotonic_ns,
                self.observation_latest_source_monotonic_ns,
                self.observation_logical_step_monotonic_ns,
                self.inference_started_monotonic_ns,
                self.inference_finished_monotonic_ns,
            )
            <= 0
        ):
            raise ValueError("context timestamps must be positive")
        if not (
            self.observation_latest_source_monotonic_ns
            <= self.observation_logical_step_monotonic_ns
            <= self.observation_anchor_monotonic_ns
        ):
            raise ValueError(
                "observation time order must be source <= logical step <= causal cut"
            )
        if not (
            self.observation_anchor_monotonic_ns
            <= self.inference_started_monotonic_ns
            <= self.inference_finished_monotonic_ns
        ):
            raise ValueError(
                "inference time order must be causal cut <= start <= finish"
            )
        if self.step_dt_ns <= 0:
            raise ValueError("step_dt_ns must be positive")


@dataclass(frozen=True)
class PolicyPrediction:
    """Untimed, immutable policy output in joint or end-effector space.

    Exactly one of ``arm_qpos`` (joint, ``[N, 7]``) or
    ``ee_pos``/``ee_rot6d`` (EE, ``[N, 3]`` + ``[N, 6]``) is present.
    ``hand_qpos`` is optional.  Timing, validity, run identity, and all shared
    state are deliberately absent: the inference worker owns their assignment.
    """

    arm_qpos: np.ndarray | None
    hand_qpos: np.ndarray | None
    ee_pos: np.ndarray | None = None
    ee_rot6d: np.ndarray | None = None

    @staticmethod
    def _freeze_float64(value: object, *, name: str) -> np.ndarray:
        raw = np.asarray(value)
        if raw.dtype != np.dtype(np.float64):
            raise TypeError(f"{name} must have dtype float64, got {raw.dtype}")
        frozen = freeze_array(raw, name=name, dtype=np.float64)
        if frozen is None:
            raise ValueError(f"{name} must not be None")
        return frozen

    def __post_init__(self) -> None:
        arm = (
            self._freeze_float64(self.arm_qpos, name="arm_qpos")
            if self.arm_qpos is not None
            else None
        )
        ee_pos = (
            self._freeze_float64(self.ee_pos, name="ee_pos")
            if self.ee_pos is not None
            else None
        )
        ee_rot6d = (
            self._freeze_float64(self.ee_rot6d, name="ee_rot6d")
            if self.ee_rot6d is not None
            else None
        )
        has_arm = arm is not None
        has_ee = ee_pos is not None or ee_rot6d is not None
        if not has_arm and not has_ee:
            raise ValueError(
                "PolicyPrediction requires arm_qpos (joint) or ee_pos/ee_rot6d (EE)"
            )
        if has_arm and has_ee:
            raise ValueError("PolicyPrediction cannot mix arm_qpos with EE fields")
        if ee_pos is None and ee_rot6d is not None:
            raise ValueError("PolicyPrediction ee_pos is required with ee_rot6d")
        if ee_pos is not None and ee_rot6d is None:
            raise ValueError("PolicyPrediction ee_rot6d is required with ee_pos")

        if arm is not None:
            if arm.ndim != 2 or arm.shape[1] != ARM_JOINT_SHAPE[0]:
                raise ValueError(
                    f"arm_qpos must be [N, {ARM_JOINT_SHAPE[0]}], got {arm.shape}"
                )
            n = arm.shape[0]
        else:
            assert ee_pos is not None and ee_rot6d is not None
            if ee_pos.ndim != 2 or ee_pos.shape[1] != EE_POS_DIM:
                raise ValueError(
                    f"ee_pos must be [N, {EE_POS_DIM}], got {ee_pos.shape}"
                )
            if ee_rot6d.ndim != 2 or ee_rot6d.shape[1] != EE_ROT6D_DIM:
                raise ValueError(
                    f"ee_rot6d must be [N, {EE_ROT6D_DIM}], got {ee_rot6d.shape}"
                )
            if ee_pos.shape[0] != ee_rot6d.shape[0]:
                raise ValueError("ee_pos and ee_rot6d must have matching N")
            n = ee_pos.shape[0]
        if n <= 0:
            raise ValueError("PolicyPrediction must have at least one step")
        if n > MAX_POLICY_CHUNK_STEPS:
            raise ValueError(
                f"PolicyPrediction has {n} steps; transport capacity is {MAX_POLICY_CHUNK_STEPS}"
            )

        hand = (
            self._freeze_float64(self.hand_qpos, name="hand_qpos")
            if self.hand_qpos is not None
            else None
        )
        if hand is not None and hand.shape != (n, HAND_JOINT_SHAPE[0]):
            raise ValueError(
                f"hand_qpos must be [N, {HAND_JOINT_SHAPE[0]}] matching N={n}, got {hand.shape}"
            )

        object.__setattr__(self, "arm_qpos", arm)
        object.__setattr__(self, "hand_qpos", hand)
        object.__setattr__(self, "ee_pos", ee_pos)
        object.__setattr__(self, "ee_rot6d", ee_rot6d)

    @property
    def is_ee(self) -> bool:
        return self.ee_pos is not None


@dataclass(frozen=True)
class JointActionChunk:
    """Runtime-canonical action chunk (joint or EE arm target).

    Exactly one of ``arm_qpos`` (joint, ``[N,7]`` rad) or ``ee_pos``/``ee_rot6d``
    (EE, ``[N,3]`` pos + ``[N,6]`` rot6d) is present; an EE chunk leaves
    ``arm_qpos`` as None and is resolved to joint space by the coordinator's
    IK dispatch. ``hand_qpos`` is None when the policy
    does not command the hand.  ``target_monotonic_ns`` holds strictly
    increasing intended execution timestamps; ``valid_mask[i] == 0`` marks a
    step to skip.
    """

    arm_qpos: np.ndarray | None  # [N, 7] float64 rad (joint) or None (EE)
    hand_qpos: np.ndarray | None  # [N, 12] float64 rad
    target_monotonic_ns: np.ndarray  # [N] uint64
    valid_mask: np.ndarray  # [N] uint8
    ee_pos: np.ndarray | None = None  # [N, 3] float64 m (EE only)
    ee_rot6d: np.ndarray | None = None  # [N, 6] float64 (EE only)

    def __post_init__(self) -> None:
        arm = (
            freeze_array(self.arm_qpos, name="arm_qpos", dtype=np.float64)
            if self.arm_qpos is not None
            else None
        )
        ee_pos = (
            freeze_array(self.ee_pos, name="ee_pos", dtype=np.float64)
            if self.ee_pos is not None
            else None
        )
        ee_rot6d = (
            freeze_array(self.ee_rot6d, name="ee_rot6d", dtype=np.float64)
            if self.ee_rot6d is not None
            else None
        )
        has_arm = arm is not None
        has_ee = ee_pos is not None or ee_rot6d is not None
        if not has_arm and not has_ee:
            raise ValueError(
                "JointActionChunk requires arm_qpos (joint) or ee_pos/ee_rot6d (EE)"
            )
        if has_arm and has_ee:
            raise ValueError("JointActionChunk cannot mix arm_qpos with EE fields")
        if ee_pos is not None and ee_rot6d is None:
            raise ValueError(
                "JointActionChunk ee_rot6d is required when ee_pos is present"
            )
        if ee_rot6d is not None and ee_pos is None:
            raise ValueError(
                "JointActionChunk ee_pos is required when ee_rot6d is present"
            )

        if arm is not None:
            if arm.ndim != 2 or arm.shape[1] != ARM_JOINT_SHAPE[0]:
                raise ValueError(
                    f"arm_qpos must be [N, {ARM_JOINT_SHAPE[0]}], got {arm.shape}"
                )
            n = arm.shape[0]
        else:
            if ee_pos is None or ee_rot6d is None:
                raise ValueError("EE action requires both ee_pos and ee_rot6d")
            if ee_pos.ndim != 2 or ee_pos.shape[1] != EE_POS_DIM:
                raise ValueError(
                    f"ee_pos must be [N, {EE_POS_DIM}], got {ee_pos.shape}"
                )
            if ee_rot6d.ndim != 2 or ee_rot6d.shape[1] != EE_ROT6D_DIM:
                raise ValueError(
                    f"ee_rot6d must be [N, {EE_ROT6D_DIM}] when ee_pos is present, got "
                    f"{ee_rot6d.shape}"
                )
            if ee_rot6d.shape[0] != ee_pos.shape[0]:
                raise ValueError(
                    f"ee_rot6d N={ee_rot6d.shape[0]} must match ee_pos N={ee_pos.shape[0]}"
                )
            n = ee_pos.shape[0]
        if n == 0:
            raise ValueError("JointActionChunk must have at least one step")
        if n > MAX_POLICY_CHUNK_STEPS:
            raise ValueError(
                f"JointActionChunk has {n} steps; transport capacity is {MAX_POLICY_CHUNK_STEPS}"
            )

        hand = self.hand_qpos
        if hand is not None:
            hand = freeze_array(hand, name="hand_qpos", dtype=np.float64)
            if hand is None:
                raise ValueError("hand_qpos must not be None")
            if hand.ndim != 2 or hand.shape != (n, HAND_JOINT_SHAPE[0]):
                raise ValueError(
                    f"hand_qpos must be [N, {HAND_JOINT_SHAPE[0]}] matching N={n}, got {hand.shape}"
                )
            object.__setattr__(self, "hand_qpos", hand)

        target = freeze_array(
            self.target_monotonic_ns, name="target_monotonic_ns", dtype=np.uint64
        )
        if target is None:
            raise ValueError("target_monotonic_ns must not be None")
        if target.shape != (n,):
            raise ValueError(
                f"target_monotonic_ns must have shape ({n},), got {target.shape}"
            )
        # Element-wise comparison, not np.diff: uint64 diff underflows on a decrease.
        if n > 1 and not bool(np.all(target[1:] > target[:-1])):
            raise ValueError("target_monotonic_ns must be strictly increasing")

        mask = freeze_array(self.valid_mask, name="valid_mask", dtype=np.uint8)
        if mask is None:
            raise ValueError("valid_mask must not be None")
        if mask.shape != (n,):
            raise ValueError(f"valid_mask must have shape ({n},), got {mask.shape}")
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError("valid_mask must be 0 or 1")

        object.__setattr__(self, "arm_qpos", arm)
        object.__setattr__(self, "ee_pos", ee_pos)
        object.__setattr__(self, "ee_rot6d", ee_rot6d)
        object.__setattr__(self, "target_monotonic_ns", target)
        object.__setattr__(self, "valid_mask", mask)

    @property
    def is_ee(self) -> bool:
        return self.ee_pos is not None


@runtime_checkable
class PolicyRuntime(Protocol):
    """Model-side policy boundary: load -> predict -> reset_episode.

    ``predict`` encodes an ``ObservationBatch`` into the model-native input,
    runs inference, and decodes the result into an untimed
    ``PolicyPrediction``.  The
    implementation may import torch/checkpoint/CUDA; it never sees
    RuntimeChannels or a robot command.
    """

    def load(self) -> None: ...

    def reset_episode(self) -> None: ...

    def predict(self, observation: ObservationBatch) -> PolicyPrediction: ...

    def close(self) -> None: ...
