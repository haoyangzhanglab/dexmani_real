"""设备与运行资源的启动、关闭超时参数。"""

from __future__ import annotations

from dataclasses import astuple, dataclass

import numpy as np


@dataclass(frozen=True)
class SafetyParams:
    model_startup_timeout_s: float = 300.0
    camera_ready_timeout_s: float = 15.0
    pointcloud_ready_timeout_s: float = 30.0
    vr_ready_timeout_s: float = 120.0
    shutdown_timeout_s: float = 65.0

    def validate(self) -> None:
        if not all(np.isfinite(value) and value > 0 for value in astuple(self)):
            raise ValueError("safety timeouts must be finite and positive")
