"""策略部署的执行参数与默认实验预算。"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_NUM_EPISODES = 20
DEFAULT_MAX_RUNNING_S = 60.0
START_ARM_TOLERANCE_DEG = 0.5
# 等待下一动作与部署/回放命令断流共用预算，不代表物理停止延迟。
ACTION_TIMEOUT_NS = 2_000_000_000
RUNTIME_SECTIONS = (
    "execution", "arm", "hand", "motion", "camera", "pointcloud", "hand_pointcloud", "environment",
)


@dataclass(frozen=True)
class ExecutionConfig:
    # async 可继续使用旧预测的有效尾部，action_steps 是名义执行窗口。
    action_steps: int | None = None
    mode: str = "sync"
    prefetch_steps: int | None = None
    rtc_guidance_cap: float | None = None
