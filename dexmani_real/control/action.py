"""Internal joint target proposed to the controller safety boundary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActionCandidate:
    """One current command candidate proposed by a control producer.

    The controller assigns a globally monotonic action ID while building the
    candidate; publication confirms it still belongs to the active
    ``run_generation``.

    ``created_monotonic_ns`` is the candidate creation time.
    ``scheduled_target_monotonic_ns`` retains an optional producer-provided
    nominal/reference timestamp for provenance only; when omitted by the
    producer, the builder defaults it to ``target_monotonic_ns``.
    ``target_monotonic_ns`` is the actual worker delivery target assigned when
    building the candidate. ``valid_until_monotonic_ns`` is the hard expiry
    at and after which the command may no longer cross an actuator SDK boundary.
    """

    observation_id: int
    run_generation: int
    created_monotonic_ns: int
    scheduled_target_monotonic_ns: int
    target_monotonic_ns: int
    valid_until_monotonic_ns: int
    action_id: int = 0
    arm_qpos: np.ndarray | None = None
    hand_qpos: np.ndarray | None = None
    is_hold: bool = False
