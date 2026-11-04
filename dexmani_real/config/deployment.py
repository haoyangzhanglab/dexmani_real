"""策略部署的执行参数与默认实验预算。"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_NUM_EPISODES = 20
DEFAULT_MAX_RUNNING_S = 60.0
# 设备 owner 检查目标租期；它不代表物理停止延迟。
COMMAND_LEASE_NS = 2_000_000_000
RUNTIME_SECTIONS = (
    "execution", "arm", "hand", "motion", "camera", "pointcloud", "hand_pointcloud", "environment",
)


@dataclass(frozen=True)
class ExecutionConfig:
    # async/RTC 可继续旧预测尾部；action_steps 是名义窗口，RTC 按提交进度接续。
    action_steps: int | None = None
    mode: str = "sync"
    prefetch_steps: int | None = None
    rtc_guidance_cap: float | None = None
