"""Finite operational targets; the Robot also checks physical limits before SDK dispatch."""

import numpy as np

from dexmani_real.planning.paths import wrap_nearest_equivalent


def project_arm_command(target_arm_qpos, reference_arm_qpos, *, joint_lower_rad, joint_upper_rad):
    equivalent = wrap_nearest_equivalent(
        target_arm_qpos, reference_arm_qpos, joint_lower_rad, joint_upper_rad
    )
    return np.clip(equivalent, joint_lower_rad, joint_upper_rad)


def project_hand_command(target_hand_qpos, *, qpos_min_rad, qpos_max_rad):
    target = np.asarray(target_hand_qpos, dtype=np.float64)
    if target.shape != (12,) or not np.isfinite(target).all():
        raise ValueError("hand preparation requires a finite (12,) target")
    return np.clip(target, qpos_min_rad, qpos_max_rad)
