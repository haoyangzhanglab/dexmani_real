"""Backend-neutral deployment contracts.

``PolicyRuntime`` defines the model boundary. ``PolicyPrediction`` is the
untimed model output, ``InferenceContext`` carries per-publish provenance, and
``ActionChunk`` is the runtime-canonical joint/EE action. None of these import
torch or RuntimeChannels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from dexmani_real.deployment.observation import PolicyObservation, freeze_array
from dexmani_real.ipc.schema import (
    ARM_JOINT_SHAPE,
    EE_POS_DIM,
    EE_ROT6D_DIM,
    HAND_JOINT_SHAPE,
    MAX_POLICY_CHUNK_STEPS,
)


def _require_chunk_integer(value: object, *, name: str, positive: bool) -> int:
    """Return one exact integer suitable for ActionChunk identity/timing."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0 or (positive and result == 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    if result > int(np.iinfo(np.uint64).max):
        raise ValueError(f"{name} exceeds uint64")
    return result


@dataclass(frozen=True)
class InferenceContext:
    """Timing/identity provenance carried from inference to chunk publication.

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
class ActionChunk:
    """One immutable latest-wins inference result for coordinator activation.

    The chunk carries model outputs plus causal/inference provenance only.
    Execution targets are derived by the coordinator from the logical step and
    control period; per-step target arrays, validity masks, and scheduler deadlines
    deliberately do not cross this boundary.
    """

    chunk_id: int
    run_generation: int
    observation_id: int
    observation_anchor_monotonic_ns: int
    observation_latest_source_monotonic_ns: int
    observation_logical_step_monotonic_ns: int
    inference_started_monotonic_ns: int
    inference_finished_monotonic_ns: int
    num_steps: int
    arm_present: bool
    ee_present: bool
    hand_present: bool
    arm_qpos: np.ndarray | None
    hand_qpos: np.ndarray | None
    ee_pos: np.ndarray | None = None
    ee_rot6d: np.ndarray | None = None

    def __post_init__(self) -> None:
        integer_fields = (
            ("chunk_id", self.chunk_id, True),
            ("run_generation", self.run_generation, False),
            ("observation_id", self.observation_id, True),
            (
                "observation_anchor_monotonic_ns",
                self.observation_anchor_monotonic_ns,
                True,
            ),
            (
                "observation_latest_source_monotonic_ns",
                self.observation_latest_source_monotonic_ns,
                True,
            ),
            (
                "observation_logical_step_monotonic_ns",
                self.observation_logical_step_monotonic_ns,
                True,
            ),
            (
                "inference_started_monotonic_ns",
                self.inference_started_monotonic_ns,
                True,
            ),
            (
                "inference_finished_monotonic_ns",
                self.inference_finished_monotonic_ns,
                True,
            ),
            ("num_steps", self.num_steps, True),
        )
        normalized: dict[str, int] = {}
        for name, value, positive in integer_fields:
            normalized[name] = _require_chunk_integer(
                value,
                name=name,
                positive=positive,
            )
        if normalized["num_steps"] > MAX_POLICY_CHUNK_STEPS:
            raise ValueError(
                "ActionChunk num_steps exceeds transport capacity "
                f"{MAX_POLICY_CHUNK_STEPS}"
            )
        if not (
            normalized["observation_latest_source_monotonic_ns"]
            <= normalized["observation_logical_step_monotonic_ns"]
            <= normalized["observation_anchor_monotonic_ns"]
            <= normalized["inference_started_monotonic_ns"]
            <= normalized["inference_finished_monotonic_ns"]
        ):
            raise ValueError(
                "ActionChunk timestamps must satisfy source <= logical step <= "
                "causal cut <= inference start <= inference finish"
            )

        for name in ("arm_present", "ee_present", "hand_present"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")
        if self.arm_present == self.ee_present:
            raise ValueError("ActionChunk requires exactly one arm representation")

        arm = freeze_array(self.arm_qpos, name="arm_qpos", dtype=np.float64)
        hand = freeze_array(self.hand_qpos, name="hand_qpos", dtype=np.float64)
        ee_pos = freeze_array(self.ee_pos, name="ee_pos", dtype=np.float64)
        ee_rot6d = freeze_array(self.ee_rot6d, name="ee_rot6d", dtype=np.float64)
        n = normalized["num_steps"]

        if self.arm_present:
            if arm is None or arm.shape != (n, ARM_JOINT_SHAPE[0]):
                raise ValueError(
                    f"arm_qpos must be [{n}, {ARM_JOINT_SHAPE[0]}] when present"
                )
            if ee_pos is not None or ee_rot6d is not None:
                raise ValueError("joint ActionChunk cannot contain EE arrays")
        else:
            if arm is not None:
                raise ValueError("EE ActionChunk cannot contain arm_qpos")
            if ee_pos is None or ee_pos.shape != (n, EE_POS_DIM):
                raise ValueError(f"ee_pos must be [{n}, {EE_POS_DIM}] when present")
            if ee_rot6d is None or ee_rot6d.shape != (n, EE_ROT6D_DIM):
                raise ValueError(f"ee_rot6d must be [{n}, {EE_ROT6D_DIM}] when present")

        if self.hand_present:
            if hand is None or hand.shape != (n, HAND_JOINT_SHAPE[0]):
                raise ValueError(
                    f"hand_qpos must be [{n}, {HAND_JOINT_SHAPE[0]}] when present"
                )
        elif hand is not None:
            raise ValueError("hand_qpos must be None when hand_present is false")

        for name, value in normalized.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "arm_qpos", arm)
        object.__setattr__(self, "hand_qpos", hand)
        object.__setattr__(self, "ee_pos", ee_pos)
        object.__setattr__(self, "ee_rot6d", ee_rot6d)

    @property
    def is_ee(self) -> bool:
        return self.ee_present


class PolicyRuntime(Protocol):
    """Real-side adapter boundary after Policy-owned strict restore.

    ``predict`` passes one validated ``PolicyObservation`` to the model runtime,
    runs inference, and decodes the result into an untimed
    ``PolicyPrediction``.  The
    implementation operates on NumPy at this boundary; it never sees
    RuntimeChannels or a robot command.
    """

    def warmup(self, *, samples: int) -> tuple[float, ...]: ...

    def reset_episode(self) -> None: ...

    def predict(self, observation: PolicyObservation) -> PolicyPrediction: ...

    def close(self) -> None: ...
