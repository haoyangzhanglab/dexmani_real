"""加载策略信息并完成准入，再创建评估产物和设备会话。"""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path

from dexmani_real.config.calibration_data import CAMERAS_PATH
from dexmani_real.config.deployment import (
    DEFAULT_MAX_RUNNING_S,
    DEFAULT_NUM_EPISODES,
    RUNTIME_SECTIONS,
)
from dexmani_real.deployment.observation import policy_dependencies, policy_needs
from dexmani_real.deployment.preflight import resolve_deployment_config
from dexmani_real.deployment.session import PolicySession
from dexmani_real.runtime.observation import ObservationNeeds
from dexmani_real.utils.io import write_yaml
from dexmani_real.utils.table_plane import (
    load_table_plane,
    resolve_table_plane_path,
)


def session_directory(root: Path, info) -> Path:
    parts = (info.policy_name, info.task_name, info.experiment_dir.name)
    if any(
        not isinstance(part, str)
        or not part
        or part in {".", ".."}
        or "\\" in part
        or Path(part).name != part
        for part in parts
    ):
        raise ValueError("policy, task and experiment names must be single directory components")
    root = root.expanduser().resolve()
    base = root / Path(*parts)
    if not base.resolve().is_relative_to(root):
        raise ValueError("session output must remain inside its root")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = base / f"session_{stamp}"
    dedup = 1
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = base / f"session_{stamp}_{dedup:02d}"
            dedup += 1


def run_policy(
    experiment,
    runtime,
    *,
    checkpoint="latest",
    weights=None,
    inference_steps=None,
    device="cuda:0",
    seed=0,
    num_episodes=DEFAULT_NUM_EPISODES,
    max_running_s=DEFAULT_MAX_RUNNING_S,
    record_raw=True,
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

    from dexmani_policy.deployment import inspect_policy

    info = inspect_policy(
        experiment, checkpoint=checkpoint, weights=weights, inference_steps=inference_steps,
    )
    dependencies = policy_dependencies(info.observation_fields)
    control_needs = policy_needs(dependencies)
    record_needs = (
        ObservationNeeds(camera_fields=("rgb", "depth"), required_tactile=("aggregate", "dense"))
        if record_raw else ObservationNeeds()
    )
    needs = control_needs.union(record_needs)
    runtime = resolve_deployment_config(info, runtime, needs)
    table_plane_abcd = (
        load_table_plane(runtime.environment.table)
        if runtime.environment.table.enabled or (needs.point_cloud and runtime.pointcloud.remove_table)
        else None
    )

    if camera_calibration_path is not None:
        camera_calibration_path = Path(camera_calibration_path).expanduser().resolve()
    root = output or Path(__file__).resolve().parents[2] / "rollouts"
    session_dir = session_directory(Path(root), info)
    payload = {
        "policy": info.to_dict(),
        "runtime": {name: asdict(getattr(runtime, name)) for name in RUNTIME_SECTIONS},
        "camera_calibration_path": str(camera_calibration_path or CAMERAS_PATH),
        "effective_camera": None,
        "table_plane_abcd": table_plane_abcd,
        "table_plane_path": (
            str(resolve_table_plane_path(runtime.environment.table))
            if table_plane_abcd is not None else None
        ),
        "seed": seed,
        "device": device,
        "num_episodes": num_episodes,
        "max_duration_s": max_running_s,
        "record_raw": record_raw,
    }
    write_yaml(payload, session_dir / "run_config.yaml")

    print(f"策略：{experiment}；数值表示 recipe 与控制 dt 来自训练 artifact，设备和标定来自 Real")
    duration = f"{max_running_s:g} s" if max_running_s is not None else "无限制"
    print(
        f"评估：{num_episodes if num_episodes is not None else '不限'} 轮正式运行结束后退出 | "
        f"每轮预算 {duration} | 输入：{', '.join(info.observation_fields)}"
    )
    print(f"输出：{session_dir}")
    return PolicySession(
        runtime,
        info,
        device=device,
        seed=seed,
        max_running_s=max_running_s,
        num_episodes=num_episodes,
        record_raw=record_raw,
        output_dir=session_dir,
        camera_calibration_path=camera_calibration_path,
        table_plane_abcd=table_plane_abcd,
        run_config=payload,
        dependencies=dependencies, control_needs=control_needs,
        record_needs=record_needs, needs=needs,
    ).run()
