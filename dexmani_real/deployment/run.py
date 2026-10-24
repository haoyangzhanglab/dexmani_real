"""Public Policy deployment entry: preflight once before artifacts or device startup."""

from __future__ import annotations

import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

from dexmani_real.calibration import CAMERAS_PATH
from dexmani_real.config.execution import DEFAULT_NUM_EPISODES, DEFAULT_MAX_RUNNING_S
from dexmani_real.config.experiment import resolve_table_plane, resolve_table_plane_path
from dexmani_real.deployment.observation import policy_needs
from dexmani_real.deployment.preflight import (
    validate_deployment_config,
    validate_max_running_s,
    validate_num_episodes,
)
from dexmani_real.deployment.session import PolicySession
from dexmani_real.runtime.observation import ObservationNeeds
from dexmani_real.utils.atomic_io import atomic_yaml_dump


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
    num_episodes=DEFAULT_NUM_EPISODES,
    max_running_s=DEFAULT_MAX_RUNNING_S,
    record_raw=True,
    record_results=True,
    output=None,
    camera_calibration_path=None,
) -> int:
    """Inspect, resolve and run one evaluation session from CLI or Python."""
    from dexmani_policy.deployment import inspect_policy

    info = inspect_policy(
        experiment,
        checkpoint=checkpoint,
        weights=weights,
        inference_steps=inference_steps,
    )
    action_steps = runtime.execution.action_steps
    if action_steps is None:
        action_steps = info.n_action_steps
    runtime = replace(runtime, execution=replace(runtime.execution, action_steps=action_steps))
    control_needs = policy_needs(info.observation_fields)
    record_needs = (
        ObservationNeeds(camera="rgbd", required_tactile=("aggregate", "dense"))
        if record_raw else ObservationNeeds()
    )
    needs = control_needs.union(record_needs)
    cloud_recipe, hand_cloud_recipe = validate_deployment_config(info, runtime, needs=needs)
    max_running_s = validate_max_running_s(max_running_s)
    num_episodes = validate_num_episodes(num_episodes)
    if cloud_recipe is not None or hand_cloud_recipe is not None:
        runtime = replace(
            runtime,
            pointcloud=cloud_recipe if cloud_recipe is not None else runtime.pointcloud,
            hand_pointcloud=(
                hand_cloud_recipe if hand_cloud_recipe is not None else runtime.hand_pointcloud
            ),
        )
    table_plane_abcd = resolve_table_plane(
        runtime.environment.table,
        required=runtime.environment.table.enabled or (cloud_recipe is not None and cloud_recipe.remove_table),
    )

    camera_calibration_path = Path(camera_calibration_path or CAMERAS_PATH).expanduser().resolve()
    session_dir = payload = None
    if record_raw or record_results:
        root = output or Path(__file__).resolve().parents[2] / "rollouts"
        session_dir = session_directory(Path(root), info)
        import dexmani_policy

        payload = {
            "experiment": str(experiment),
            "checkpoint": str(checkpoint),
            "policy": info.to_dict(),
            "future_steps": info.future_steps,
            "runtime": asdict(runtime),
            "camera_calibration_path": str(camera_calibration_path),
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
            "output": str(session_dir),
            "record_raw": record_raw,
            "record_results": record_results,
            "dexmani_real_head": git_head(Path(__file__).resolve().parents[2]),
            "dexmani_policy_head": git_head(Path(dexmani_policy.__file__).resolve().parents[1]),
        }
        atomic_yaml_dump(payload, session_dir / "run_config.yaml")
    elif output is not None:
        print("--no-record --no-results: --output is unused; no session artifacts will be created")

    print(f"策略：{experiment}；数值表示 recipe 与控制 dt 来自训练 artifact，设备和标定来自 Real")
    duration = f"{max_running_s:g} s" if max_running_s is not None else "无限制"
    print(
        f"评估：{num_episodes if num_episodes is not None else '不限'} 轮正式运行结束后退出 | "
        f"每轮预算 {duration} | 输入：{', '.join(info.observation_fields)}"
    )
    print(f"输出：{session_dir}" if session_dir else "输出：Raw 和评估摘要均关闭")
    return PolicySession(
        runtime,
        info,
        control_needs=control_needs, record_needs=record_needs,
        device=device,
        seed=seed,
        max_running_s=max_running_s,
        num_episodes=num_episodes,
        recording_dir=session_dir if record_raw else None,
        results_dir=session_dir if record_results else None,
        camera_calibration_path=camera_calibration_path,
        table_plane_abcd=table_plane_abcd,
        run_config=payload,
    ).run()
