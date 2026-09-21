"""Internal joint target proposed to the controller safety boundary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActionCandidate:
    """One current command candidate proposed by a control producer.

    Publication confirms the candidate still belongs to the active
    ``run_generation`` and commits it once to the ordered command FIFO. The
    candidate is an owner-owned immutable numeric snapshot: a FULL commit
    result retries this exact object without rebuilding it, re-solving IK, or
    re-clipping its targets, and it carries no delivery lease — it stays
    committable until its generation is revoked.
    """

    run_generation: int
    arm_qpos: np.ndarray | None = None
    hand_qpos: np.ndarray | None = None
    is_hold: bool = False
