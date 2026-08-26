"""Deterministic, CPU-only, torch-free fake PolicyRuntime.

``FakePolicyRuntime`` is a complete implementation of the ``PolicyRuntime``
Protocol that exercises the full obs -> chunk -> plan-ring path with no model,
no torch, and no hardware. It is the reference for the real learned-policy
integration and the swap fixture for runtime-replacement verification.

Determinism contract: for a fixed observation and run_generation, ``predict``
returns byte-identical arrays across calls and processes. The fake never reads
the clock beyond the observation anchor, never samples randomness, and never
touches RuntimeChannels.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dexmani_real.deployment.contracts import InferenceContext, JointActionChunk
from dexmani_real.deployment.observation import ObservationBatch
from dexmani_real.robot_spec import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE

# Fixed fake horizon for scheduler tests; real backends derive it from model config.
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


class FakePolicyRuntime:
    """Deterministic hold-with-offset policy: tile the latest arm/hand vector.

    ``predict`` returns a ``JointActionChunk`` whose i-th step is
    ``input + i*offset_rad``.  ``offset_rad=0`` (the default) is a pure
    hold; a non-zero offset proves step ordering without touching the clock.
    Step zero is the action aligned with the latest logical observation step;
    the inference worker masks it if inference has already made it undeliverable.
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
        action_key = (
            getattr(self.config, "action_key", "action")
            if self.config is not None
            else "action"
        )
        if action_key != "action":
            raise ValueError(
                f"FakePolicyRuntime is joint-only; action_key={action_key!r} "
                "is not supported"
            )
        self._loaded = True

    def reset_episode(self) -> None:
        self._loaded = True

    def predict(
        self, observation: ObservationBatch, *, context: InferenceContext
    ) -> JointActionChunk:
        arm = _last_valid(
            (
                observation.arm_history.values
                if observation.arm_history is not None
                else None
            ),
            (
                observation.arm_history.valid_mask
                if observation.arm_history is not None
                else None
            ),
            ARM_JOINT_SHAPE[0],
        )
        hand: np.ndarray | None = None
        if observation.hand_history is not None:
            hand = _last_valid(
                observation.hand_history.values,
                observation.hand_history.valid_mask,
                HAND_JOINT_SHAPE[0],
            )

        n = self.horizon
        offsets = np.arange(n, dtype=np.float64) * self.offset_rad
        arm_out = np.empty((n, ARM_JOINT_SHAPE[0]), dtype=np.float64)
        arm_out[:] = arm[None, :] + offsets[:, None]

        hand_out: np.ndarray | None = None
        if hand is not None:
            hand_out = np.tile(hand[None, :], (n, 1))

        steps = np.arange(n, dtype=np.uint64)
        target = np.asarray(
            context.observation_logical_step_monotonic_ns, dtype=np.uint64
        ) + steps * np.uint64(context.step_dt_ns)
        return JointActionChunk(
            arm_qpos=arm_out,
            hand_qpos=hand_out,
            target_monotonic_ns=target,
            valid_mask=np.ones(n, dtype=np.uint8),
        )

    def close(self) -> None:
        self._loaded = False
