"""Internal joint target proposed to the controller safety boundary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActionCandidate:
    """One current-tick joint target proposed by a control source.

    The controller assigns a globally monotonic action ID while building the
    candidate; publication confirms it still belongs to the active
    ``run_generation``. ``scheduled_target_monotonic_ns`` preserves the policy
    grid endpoint for provenance. ``target_monotonic_ns`` is the worker delivery
    target chosen at publication, and ``valid_until_monotonic_ns`` is its hard
    expiry. An overdue policy endpoint is never relabeled as a fresh endpoint.
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
