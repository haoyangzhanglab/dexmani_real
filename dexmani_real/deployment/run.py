"""校验实验预算，将模型加载和设备生命周期交给会话。"""

from __future__ import annotations

import math
from pathlib import Path

from dexmani_real.config.deployment import (
    DEFAULT_MAX_RUNNING_S,
    DEFAULT_NUM_EPISODES,
)
from dexmani_real.deployment.session import PolicySession


def run_policy(
    policy,
    runtime,
    *,
    enable_hoi_pcd=False,
    weights=None,
    inference_steps=None,
    device="cuda:0",
    seed=0,
    num_episodes=DEFAULT_NUM_EPISODES,
    max_running_s=DEFAULT_MAX_RUNNING_S,
    record_raw=False,
    output=None,
    camera_calibration_path=None,
) -> int:
    if max_running_s is not None:
        if isinstance(max_running_s, bool):
            raise TypeError("max_running_s must be a finite positive number or None")
        max_running_s = float(max_running_s)
        if not math.isfinite(max_running_s) or max_running_s <= 0:
            raise ValueError("max_running_s must be finite and positive")
    if num_episodes is not None:
        if type(num_episodes) is not int:
            raise TypeError("num_episodes must be a positive integer or None")
        if num_episodes < 1:
            raise ValueError("num_episodes must be a positive integer")

    if camera_calibration_path is not None:
        camera_calibration_path = Path(camera_calibration_path).expanduser().resolve()
    return PolicySession(
        runtime,
        policy,
        enable_hoi_pcd=enable_hoi_pcd,
        weights=weights,
        inference_steps=inference_steps,
        device=device,
        seed=seed,
        max_running_s=max_running_s,
        num_episodes=num_episodes,
        record_raw=record_raw,
        output_root=output,
        camera_calibration_path=camera_calibration_path,
    ).run()
