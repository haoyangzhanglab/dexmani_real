"""设备与运行资源的启动、关闭超时参数。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeTimeouts:
    model_startup_timeout_s: float = 300.0
    camera_ready_timeout_s: float = 15.0
    pointcloud_ready_timeout_s: float = 30.0
    vr_ready_timeout_s: float = 120.0
    shutdown_timeout_s: float = 65.0
