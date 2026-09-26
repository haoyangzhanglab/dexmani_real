"""The stable raw v34 layout for research control rows."""

from dataclasses import dataclass

import numpy as np

EPISODE_SCHEMA_VERSION = 34


@dataclass(frozen=True)
class DatasetSpec:
    tail_shape: tuple[int, ...]
    dtype: np.dtype


def _spec(dtype, tail_shape=()):
    return DatasetSpec(tail_shape, np.dtype(dtype))


DATASET_SPECS = {
    "arm_qpos": _spec(np.float64, (7,)),
    "arm_qvel": _spec(np.float64, (7,)),
    # xArm SDK ``get_joint_states(num=3)`` effort telemetry. The SDK does not
    # establish an SI torque unit, so its native numeric semantics are preserved.
    "arm_effort": _spec(np.float64, (7,)),
    "hand_qpos": _spec(np.float64, (12,)),
    "hand_current": _spec(np.float64, (12,)),
    # XHand SDK-native, bias-corrected aggregate tactile values. Invalid
    # aggregate samples are represented by NaN.
    "hand_contact": _spec(np.float32, (5, 3)),
    # XHand SDK-native, bias-corrected dense tactile values. Invalid dense
    # samples are represented by NaN independently of aggregate tactile state.
    "hand_tactile_force": _spec(np.float32, (5, 120, 3)),
    "action_arm_joint_target": _spec(np.float64, (7,)),
    "action_hand_joint_target": _spec(np.float64, (12,)),
    "frame_valid": _spec(np.bool_),
}


def validate_data_layout(shapes, dtypes, *, frame_count: int) -> tuple[str, ...]:
    """Check required raw arrays, row counts, shapes, and dtypes."""
    errors = []
    if frame_count < 0:
        errors.append("num_frames must be non-negative")
    for name, spec in DATASET_SPECS.items():
        if name not in shapes:
            errors.append(f"missing required data.h5 dataset: {name}")
            continue
        if tuple(shapes[name]) != (frame_count,) + spec.tail_shape:
            errors.append(f"wrong shape for {name}: {shapes[name]}")
        if name not in dtypes or np.dtype(dtypes[name]) != spec.dtype:
            errors.append(f"wrong dtype for {name}: expected {spec.dtype}")
    for name in set(shapes) - DATASET_SPECS.keys():
        errors.append(f"unexpected data.h5 dataset: {name}")
    return tuple(errors)
