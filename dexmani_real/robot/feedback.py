"""NumPy layouts for robot feedback, independent of sensor IPC."""

import numpy as np

from dexmani_real.robot.model import (
    ARM_JOINT_SHAPE,
    HAND_JOINT_SHAPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
)

ARM_STATE_DTYPE = np.dtype(
    [
        ("qpos", "<f8", ARM_JOINT_SHAPE),
        ("qvel", "<f8", ARM_JOINT_SHAPE),
        ("timestamp_ns", "<u8"),
    ]
)
HAND_STATE_DTYPE = np.dtype(
    [
        ("qpos", "<f8", HAND_JOINT_SHAPE),
        ("tactile_aggregate", "<f4", HAND_TACTILE_SUM_SHAPE),
        ("tactile_dense", "<f4", HAND_TACTILE_FORCE_SHAPE),
        # Oldest original valid vector per modality; individual taxel clocks remain local.
        ("tactile_aggregate_timestamp_ns", "<u8"),
        ("tactile_dense_timestamp_ns", "<u8"),
        # Last trustworthy joint read, independent of held tactile values.
        ("timestamp_ns", "<u8"),
    ]
)
