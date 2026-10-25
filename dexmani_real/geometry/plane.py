"""共享的机器人基座系平面验证，不读取标定或调用拟合。"""

import numpy as np


def validate_table_plane(value, *, min_normal_norm=1e-9):
    """Validate only the consumed upward plane; collision box dimensions are separate."""
    plane = np.asarray(value, dtype=np.float64)
    if plane.shape != (4,) or not np.isfinite(plane).all():
        raise ValueError("table plane must be finite shape (4,)")
    norm = float(np.linalg.norm(plane[:3]))
    if norm <= min_normal_norm or plane[2] / norm <= 0:
        raise ValueError("table plane must have an upward non-zero normal")
    return tuple(float(x) for x in plane)

