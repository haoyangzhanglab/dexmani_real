"""策略部署的执行参数与默认实验预算。"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_NUM_EPISODES = 20
DEFAULT_MAX_DURATION_S = 60.0


@dataclass(frozen=True)
class ExecutionConfig:
    # 未指定时使用 checkpoint 保存的控制周期。
    control_dt_s: float | None = None
    # action_steps 限制每次执行的步数，预测全长决定可提交窗口。
    # None 使用 checkpoint n_action_steps；显式值仍须满足模型约束。
    action_steps: int | None = None
    mode: str = "sync"
    prefetch_steps: int = 1
    rtc_guidance_cap: float = 10.0
    # 回放测量后的暂定来源年龄预算，非真机安全保证。
    hoi_max_age_s: float = 0.25

POLICY_RUNTIME_SECTIONS = (
    "execution", "arm", "hand", "motion", "camera", "pointcloud", "hand_pointcloud",
    "hoi_tracking", "environment",
)
