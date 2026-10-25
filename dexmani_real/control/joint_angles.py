"""xArm7 等价角计算与绝对关节目标的限位、跳变约束。"""

import numpy as np

from dexmani_real.robot.model import ARM_JOINT_SHAPE, XARM7_EQUIVALENT_JOINT_MASK

DEFAULT_JUMP_DEG = (30, 30, 30, 35, 40, 40, 40)


def joint_target_rejection(q, current, previous, limits, jump_limit):
    """Absolute hardware target differences, never periodic shortest angles."""
    q, current, previous = (np.asarray(v, dtype=np.float64) for v in (q, current, previous))
    if any(v.shape != (7,) or not np.isfinite(v).all() for v in (q, current, previous)):
        return "invalid_target"
    lo, hi = np.asarray(limits).T
    if np.any(q < lo) or np.any(q > hi):
        return "operational_limits"
    if np.any(np.abs(q - previous) > jump_limit):
        return "jump"
    if np.max(np.abs(q - current)) > np.deg2rad(150):
        return "hw_dist"
    return None


def wrap_nearest_equivalent(
    qpos: np.ndarray,
    reference: np.ndarray,
    joint_limit_lower: tuple[float, ...] | np.ndarray,
    joint_limit_upper: tuple[float, ...] | np.ndarray,
) -> np.ndarray:
    """返回 xArm7 七关节副本，将可等价轴移到限位内最接近参考值的 2π 分支。"""
    result = np.asarray(qpos, dtype=np.float64).copy()
    lo = np.asarray(joint_limit_lower, dtype=np.float64)
    hi = np.asarray(joint_limit_upper, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if (
        result.shape != ARM_JOINT_SHAPE
        or ref.shape != result.shape
        or lo.shape != result.shape
        or hi.shape != result.shape
    ):
        raise ValueError(f"qpos, reference, and joint limits must have shape {ARM_JOINT_SHAPE}")
    if not all(np.all(np.isfinite(values)) for values in (result, ref, lo, hi)):
        raise ValueError("qpos, reference, and joint limits must be finite")
    if np.any(lo > hi):
        raise ValueError("joint lower limits must not exceed upper limits")
    is_equiv = np.asarray(XARM7_EQUIVALENT_JOINT_MASK, dtype=bool)

    period = 2.0 * np.pi
    lo_equiv = lo[is_equiv]
    hi_equiv = hi[is_equiv]
    res_equiv = result[is_equiv]
    ref_equiv = ref[is_equiv]
    k_min = np.ceil((lo_equiv - res_equiv) / period)
    k_max = np.floor((hi_equiv - res_equiv) / period)
    valid = k_min <= k_max
    nearest_k = np.rint((ref_equiv - res_equiv) / period)
    nearest_k = np.minimum(np.maximum(nearest_k, k_min), k_max)
    # Preserve invalid raw values so downstream validation can reject them.
    result[is_equiv] = np.where(valid, res_equiv + nearest_k * period, res_equiv)

    return result


def equivalent_joint_delta(qpos: np.ndarray, reference_qpos: np.ndarray) -> np.ndarray:
    """Signed joint deltas; only the model's full-turn axes wrap to [-π, π)."""
    delta = qpos - reference_qpos
    mask = np.asarray(XARM7_EQUIVALENT_JOINT_MASK, dtype=bool)
    delta[mask] = (delta[mask] + np.pi) % (2.0 * np.pi) - np.pi
    return delta
