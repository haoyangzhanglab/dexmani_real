"""Public Policy deployment entry: preflight once before artifacts or device startup."""

from __future__ import annotations

import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

from dexmani_real.config.experiment import resolve_runtime_table
from dexmani_real.deployment.preflight import (
    validate_deployment_config,
    validate_max_running_s,
    validate_num_episodes,
)
from dexmani_real.deployment.session import PolicySession
from dexmani_real.runtime.observation import ObservationNeeds


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
    while candidate.exists():
        candidate = base / f"session_{stamp}_{dedup:02d}"
        dedup += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def git_head(root):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_policy(
    experiment,
    runtime,
    *,
    checkpoint="latest",
    weights=None,
    inference_steps=None,
    device="cuda:0",
    seed=0,
    num_episodes=20,
    max_running_s=60.0,
    record_raw=True,
    record_results=True,
    output=None,
    camera_calibration_path=None,
    action_steps_override=None,
) -> int:
    """Inspect, resolve and run one evaluation session from CLI or Python."""
    from dexmani_policy.deployment import inspect_policy

    info = inspect_policy(
        experiment,
        checkpoint=checkpoint,
        weights=weights,
        inference_steps=inference_steps,
    )
    action_steps = action_steps_override
    if action_steps is None:
        action_steps = runtime.execution.action_steps
    if action_steps is None:
        action_steps = info.n_action_steps
    runtime = replace(runtime, execution=replace(runtime.execution, action_steps=action_steps))
    needs = ObservationNeeds.for_policy(info.observation_fields, record_raw=record_raw)
    cloud_recipe = validate_deployment_config(info, runtime, needs=needs)
    max_running_s = validate_max_running_s(max_running_s)
    num_episodes = validate_num_episodes(num_episodes)
    if cloud_recipe is not None:
        runtime = replace(runtime, pointcloud=cloud_recipe)
    runtime = resolve_runtime_table(runtime, pointcloud=cloud_recipe)

    session_dir = None
    if record_raw or record_results:
        root = output or Path(__file__).resolve().parents[2] / "rollouts"
        session_dir = session_directory(Path(root), info)
        import dexmani_policy
        import yaml

        payload = {
            "experiment": str(experiment),
            "checkpoint": str(checkpoint),
            "policy": info.to_dict(),
            "future_steps": info.horizon - info.n_obs_steps + 1,
            "runtime": asdict(runtime),
            "seed": seed,
            "device": device,
            "num_episodes": num_episodes,
            "max_duration_s": max_running_s,
            "record_raw": record_raw,
            "record_results": record_results,
            "n_action_steps_override": action_steps_override,
            "dexmani_real_head": git_head(Path(__file__).resolve().parents[2]),
            "dexmani_policy_head": git_head(Path(dexmani_policy.__file__).resolve().parents[1]),
        }
        (session_dir / "run_config.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
    elif output is not None:
        print("--no-record --no-results: --output is unused; no session artifacts will be created")

    print(f"策略：{experiment}")
    duration = f"{max_running_s:g} s" if max_running_s is not None else "无限制"
    print(
        f"评估：{num_episodes if num_episodes is not None else '不限'} 轮正式运行结束后退出 | "
        f"每轮预算 {duration} | 输入：{', '.join(info.observation_fields)}"
    )
    print(f"输出：{session_dir}" if session_dir else "输出：Raw 和评估摘要均关闭")
    return PolicySession(
        runtime,
        info,
        needs=needs,
        device=device,
        seed=seed,
        max_running_s=max_running_s,
        num_episodes=num_episodes,
        recording_dir=session_dir if record_raw else None,
        results_dir=session_dir if record_results else None,
        camera_calibration_path=camera_calibration_path,
    ).run()
