"""Pure timing calculations for immutable learned-policy action grids."""

from __future__ import annotations

import numpy as np

_UINT64_MAX = int(np.iinfo(np.uint64).max)


def _require_uint64_integer(
    value: object,
    *,
    name: str,
    positive: bool,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0 or (positive and result == 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    if result > _UINT64_MAX:
        raise ValueError(f"{name} exceeds uint64")
    return result


def _validated_target_grid(target_ns: np.ndarray) -> np.ndarray:
    values = np.asarray(target_ns)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("target_ns must be a non-empty 1-D array")
    if values.dtype.kind not in "iu":
        raise TypeError("target_ns must have an integer dtype")
    if values.dtype.kind == "i" and bool(np.any(values <= 0)):
        raise ValueError("target_ns values must be positive")
    unsigned = values.astype(np.uint64, copy=False)
    if bool(np.any(unsigned == 0)):
        raise ValueError("target_ns values must be positive")
    if unsigned.size > 1 and bool(np.any(unsigned[1:] <= unsigned[:-1])):
        raise ValueError("target_ns must be strictly increasing")
    return unsigned


def build_target_grid(
    logical_step_ns: int,
    steps: int,
    step_dt_ns: int,
) -> np.ndarray:
    """Build ``logical_step_ns + i * step_dt_ns`` without shifting the grid."""
    logical = _require_uint64_integer(
        logical_step_ns,
        name="logical_step_ns",
        positive=True,
    )
    count = _require_uint64_integer(steps, name="steps", positive=True)
    step_dt = _require_uint64_integer(
        step_dt_ns,
        name="step_dt_ns",
        positive=True,
    )
    final_offset = (count - 1) * step_dt
    if final_offset > _UINT64_MAX or logical > _UINT64_MAX - final_offset:
        raise ValueError("target grid exceeds uint64")
    targets = logical + np.arange(count, dtype=np.uint64) * np.uint64(step_dt)
    targets.setflags(write=False)
    return targets


def first_deliverable_index(
    target_ns: np.ndarray,
    inference_finished_ns: int,
    command_lead_ns: int,
) -> int:
    """Return the first target strictly after inference finish plus command lead."""
    targets = _validated_target_grid(target_ns)
    finished = _require_uint64_integer(
        inference_finished_ns,
        name="inference_finished_ns",
        positive=True,
    )
    lead = _require_uint64_integer(
        command_lead_ns,
        name="command_lead_ns",
        positive=False,
    )
    if finished > _UINT64_MAX - lead:
        raise ValueError("inference_finished_ns + command_lead_ns exceeds uint64")
    earliest_ns = finished + lead
    return int(np.searchsorted(targets, np.uint64(earliest_ns), side="right"))


def first_valid_index_from_prefix_mask(valid_mask: np.ndarray) -> int:
    """Validate a binary ``0*1*`` transport mask and return its first one."""
    mask = np.asarray(valid_mask)
    if mask.ndim != 1 or mask.size == 0:
        raise ValueError("valid_mask must be a non-empty 1-D array")
    if mask.dtype.kind not in "biu":
        raise TypeError("valid_mask must have a boolean or integer dtype")
    if not bool(np.all((mask == 0) | (mask == 1))):
        raise ValueError("valid_mask must contain only 0 or 1")
    valid_indices = np.flatnonzero(mask == 1)
    if valid_indices.size == 0:
        return int(mask.size)
    first_index = int(valid_indices[0])
    if not bool(np.all(mask[first_index:] == 1)):
        raise ValueError("valid_mask must have expired-prefix topology 0*1*")
    return first_index


def compute_plan_deadline_ns(
    inference_finished_ns: int,
    observation_source_ns: int,
    max_plan_age_ns: int,
    max_source_to_command_age_ns: int,
) -> int:
    """Return the earlier independent plan-age and source-age upper deadline."""
    finished = _require_uint64_integer(
        inference_finished_ns,
        name="inference_finished_ns",
        positive=True,
    )
    source = _require_uint64_integer(
        observation_source_ns,
        name="observation_source_ns",
        positive=True,
    )
    plan_age = _require_uint64_integer(
        max_plan_age_ns,
        name="max_plan_age_ns",
        positive=True,
    )
    source_age = _require_uint64_integer(
        max_source_to_command_age_ns,
        name="max_source_to_command_age_ns",
        positive=True,
    )
    if finished > _UINT64_MAX - plan_age:
        raise ValueError("inference_finished_ns + max_plan_age_ns exceeds uint64")
    if source > _UINT64_MAX - source_age:
        raise ValueError(
            "observation_source_ns + max_source_to_command_age_ns exceeds uint64"
        )
    return min(finished + plan_age, source + source_age)


def usable_target_mask(
    target_ns: np.ndarray,
    first_index: int,
    deadline_ns: int,
) -> np.ndarray:
    """Compose lower and upper timing bounds for diagnostics only.

    This result may have both prefix and suffix zeros. It is not transport state
    and must never be written to ``JointActionChunk.valid_mask``.
    """
    targets = _validated_target_grid(target_ns)
    index = _require_uint64_integer(
        first_index,
        name="first_index",
        positive=False,
    )
    if index > len(targets):
        raise ValueError("first_index must not exceed len(target_ns)")
    deadline = _require_uint64_integer(
        deadline_ns,
        name="deadline_ns",
        positive=True,
    )
    mask = np.zeros(len(targets), dtype=np.uint8)
    mask[index:] = targets[index:] < np.uint64(deadline)
    return mask
