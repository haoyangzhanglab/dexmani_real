"""Deterministic, CPU-only, torch-free fake policy.

``FakeObservationAdapter`` -> ``FakePolicyBackend`` -> ``FakeActionAdapter`` is
a complete implementation of the three deployment Protocols that exercises the
full obs -> backend -> chunk -> plan-ring path with no model, no torch, and no
hardware. It is the reference for the real learned-policy integration and the
swap fixture for backend-replacement verification.

Determinism contract: for a fixed observation and run_generation, ``infer``
returns byte-identical arrays across calls and processes. The backend never
reads the clock, never samples randomness, and never touches SharedStorage.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dexmani_real.deployment.contracts import InferenceContext, JointActionChunk
from dexmani_real.deployment.observation import ObservationBatch
from dexmani_real.utils.schema import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE

# Fixed chunk length the fake emits when no explicit horizon is supplied. Kept
# small so scheduler/generation checks stay readable; a real backend derives its
# horizon from its model config, not from the transport capacity.
_FAKE_HORIZON = 8


def _last_valid(
    values: np.ndarray | None,
    valid_mask: np.ndarray | None,
    dof: int,
) -> np.ndarray:
    """Return the last valid frame's values, or a zero vector when none/absent."""
    if values is None:
        return np.zeros(dof, dtype=np.float64)
    if valid_mask is not None:
        idx = np.flatnonzero(np.asarray(valid_mask).astype(bool))
        if idx.size == 0:
            return np.zeros(dof, dtype=np.float64)
        return np.asarray(values[idx[-1]], dtype=np.float64).reshape(dof)
    return np.asarray(values[-1], dtype=np.float64).reshape(dof)


class FakeObservationAdapter:
    """Encode an ``ObservationBatch`` as the latest valid arm/hand joint vector.

    ``encode`` returns ``{"arm_qpos": [7], "hand_qpos": [12] | None}``. Absent
    modality windows become a zero vector rather than crashing, so the backend
    never depends on a sensor being present (absent = valid_mask 0).
    """

    def __init__(self, config: Any = None) -> None:
        self.config = config

    def encode(self, observation: ObservationBatch) -> dict[str, np.ndarray | None]:
        arm = _last_valid(
            observation.arm_history.values if observation.arm_history is not None else None,
            observation.arm_history.valid_mask if observation.arm_history is not None else None,
            ARM_JOINT_SHAPE[0],
        )
        hand: np.ndarray | None = None
        if observation.hand_history is not None:
            hand = _last_valid(
                observation.hand_history.values,
                observation.hand_history.valid_mask,
                HAND_JOINT_SHAPE[0],
            )
        return {"arm_qpos": arm, "hand_qpos": hand}


class FakePolicyBackend:
    """Deterministic hold-with-offset policy: tile the latest arm/hand vector.

    ``infer`` returns ``{"arm_qpos": [N,7], "hand_qpos": [N,12] | None}`` where
    the i-th step is ``input + (i+1)*offset_rad``. ``offset_rad=0`` (the default)
    is a pure hold; a non-zero offset proves step ordering without touching the
    clock. ``reset`` is a no-op because the policy is stateless.
    """

    def __init__(
        self,
        config: Any = None,
        *,
        offset_rad: float = 0.0,
        horizon: int | None = None,
    ) -> None:
        self.config = config
        self.offset_rad = float(offset_rad)
        self.horizon = int(horizon) if horizon is not None else _FAKE_HORIZON
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def reset(self, *, run_generation: int) -> None:
        self._loaded = True

    def infer(self, model_input: Any) -> dict[str, np.ndarray | None]:
        arm = np.asarray(model_input["arm_qpos"], dtype=np.float64).reshape(ARM_JOINT_SHAPE[0])
        n = self.horizon
        offsets = np.arange(1, n + 1, dtype=np.float64) * self.offset_rad
        arm_out = np.empty((n, ARM_JOINT_SHAPE[0]), dtype=np.float64)
        arm_out[:] = arm[None, :] + offsets[:, None]

        hand = model_input.get("hand_qpos")
        hand_out: np.ndarray | None = None
        if hand is not None:
            hand = np.asarray(hand, dtype=np.float64).reshape(HAND_JOINT_SHAPE[0])
            hand_out = np.tile(hand[None, :], (n, 1))
        return {"arm_qpos": arm_out, "hand_qpos": hand_out}

    def close(self) -> None:
        self._loaded = False


class FakeActionAdapter:
    """Decode the fake backend's dict output into a ``JointActionChunk``.

    Target timestamps are ``inference_finished + (i+1)*step_dt_ns`` so they are
    strictly increasing and anchored to the causal cut carried by the context.
    """

    def __init__(self, config: Any = None) -> None:
        self.config = config

    def decode(self, raw_output: Any, *, context: InferenceContext) -> JointActionChunk:
        arm = np.asarray(raw_output["arm_qpos"], dtype=np.float64)
        if arm.ndim != 2 or arm.shape[1] != ARM_JOINT_SHAPE[0]:
            raise ValueError(f"fake backend arm_qpos must be [N, {ARM_JOINT_SHAPE[0]}]")
        n = arm.shape[0]
        if n == 0:
            raise ValueError("fake backend produced an empty chunk")

        hand = raw_output.get("hand_qpos")
        if hand is not None:
            hand = np.asarray(hand, dtype=np.float64)
            if hand.shape != (n, HAND_JOINT_SHAPE[0]):
                raise ValueError(f"fake backend hand_qpos must be [N, {HAND_JOINT_SHAPE[0]}]")

        steps = np.arange(1, n + 1, dtype=np.uint64)
        target = np.asarray(context.inference_finished_monotonic_ns, dtype=np.uint64) + (
            steps * np.uint64(context.step_dt_ns)
        )
        return JointActionChunk(
            arm_qpos=arm,
            hand_qpos=hand,
            target_monotonic_ns=target,
            valid_mask=np.ones(n, dtype=np.uint8),
        )
