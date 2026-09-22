"""Raw-episode workflow provenance classification for fixed-dt teleop consumers.

Policy-eval rollouts carry synchronous/irregular execution timing (real
inference-boundary gaps) and must not silently enter the fixed-dt teleop
processing pipeline or the teleop-only physical replay workflow. This module is
the single classifier both consumers share, so the two boundaries never drift.
"""

from __future__ import annotations

from typing import Any

POLICY_EVAL_WORKFLOW = "policy_eval"
TELEOP_WORKFLOW = "teleop"
_PROVENANCE_WORKFLOW_ATTR = "provenance_workflow"


def read_provenance_workflow(meta_attrs: Any) -> str | None:
    """Return the raw episode's provenance workflow, normalized, or ``None``.

    ``meta_attrs`` is the raw ``/meta`` HDF5 attribute mapping. Absent, empty,
    or whitespace-only values normalize to ``None``: the producer declared no
    workflow. Both collection and rollout recorders declare their workflow.
    """
    if meta_attrs is None:
        return None
    value = meta_attrs.get(_PROVENANCE_WORKFLOW_ATTR)
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    value = str(value).strip()
    return value or None


def supports_fixed_dt_teleop(workflow: str | None) -> bool:
    """Only explicitly declared teleop raw is a BC/replay source."""
    return workflow == TELEOP_WORKFLOW
