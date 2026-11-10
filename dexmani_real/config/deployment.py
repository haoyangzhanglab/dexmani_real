"""策略部署的执行参数与默认实验预算。"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_NUM_EPISODES = 20
DEFAULT_MAX_RUNNING_S = 60.0


@dataclass(frozen=True)
class ExecutionConfig:
    # 未指定时使用 checkpoint 保存的周期；两处均缺失时须明确填写。
    control_dt_s: float | None = None
    # async/RTC 可继续旧预测尾部；action_steps 是名义窗口，RTC 按提交进度接续。
    # None 使用 checkpoint n_action_steps；显式值仍须满足模型约束。
    action_steps: int | None = None
    mode: str = "sync"
    prefetch_steps: int = 1
    rtc_guidance_cap: float = 10.0
    # 回放测量后的暂定来源年龄预算，非真机安全保证。
    hoi_max_age_s: float = 0.25
    hoi_max_skew_s: float | None = None

POLICY_RUNTIME_SECTIONS = (
    "execution", "arm", "hand", "motion", "camera", "pointcloud", "hand_pointcloud", "environment",
)
