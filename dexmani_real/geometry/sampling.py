"""场景点云与表面模板共用的 CPU 最远点采样。"""

import numpy as np


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
