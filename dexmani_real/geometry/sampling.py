"""点云的最远点采样与表面模板分段校验。"""

import numpy as np


def validate_point_segments(
    points: np.ndarray, names: np.ndarray, offsets: np.ndarray, *, num_points: int, label: str,
) -> None:
    """校验固定点序模板中按对象或连杆划分的连续分段。"""
    if (
        names.ndim != 1
        or names.dtype.kind != "U"
        or not len(names)
        or len(set(names)) != len(names)
    ):
        raise ValueError(f"{label}: names must be a nonempty 1-D array of unique Unicode strings")
    if (
        points.shape != (num_points, 3)
        or points.dtype != np.float32
        or not np.isfinite(points).all()
    ):
        raise ValueError(f"{label}: points must be finite float32 values of shape ({num_points}, 3)")
    # 直接比较相邻值，避免整数差分溢出。
    if (
        offsets.dtype.kind not in "iu"
        or offsets.shape != (len(names) + 1,)
        or offsets[0] != 0
        or offsets[-1] != num_points
        or np.any(offsets[1:] <= offsets[:-1])
    ):
        raise ValueError(f"{label}: offsets must be strictly increasing integers from 0 to {num_points}")


def farthest_point_indices(xyz: np.ndarray, count: int) -> np.ndarray:
    """从有限的 [N, 3] XYZ 中选出 count 个不同源索引，调用方保证 0 < count < N。"""
    from fpsample import bucket_fps_kdline_sampling

    size = len(xyz)
    indices = bucket_fps_kdline_sampling(
        xyz, count, h=min(5, size.bit_length() - 1), start_idx=0,
    ).astype(np.intp, copy=False)
    used = np.zeros(size, dtype=bool)
    used[indices] = True
    if np.count_nonzero(used) == count:
        return indices

    # 重合点耗尽非零距离时，QuickFPS 可能重复返回一个源索引。
    _, first = np.unique(indices, return_index=True)
    remaining = np.flatnonzero(~used)[:count - first.size]
    return np.concatenate((indices[np.sort(first)], remaining))
