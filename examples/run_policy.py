#!/usr/bin/env python3
"""真机策略评估。

用法（EXPERIMENT 为 policy/task/experiment，不带 experiments/ 前缀）：
    python examples/run_policy.py policy/task/experiment
    python examples/run_policy.py policy/task/experiment --checkpoint 80pct --inference-steps 4
    python examples/run_policy.py policy/task/experiment --config local.yaml --n-action-steps 8
    python examples/run_policy.py --print-config

常用参数：
    --checkpoint 默认 latest；也支持已有百分比 milestone、best 或文件名。
    --inference-steps 设置 NFE，--weights 选择 ema/raw；默认使用实验保存的评估设置。
    --n-action-steps 设置每段执行步数，默认使用策略的 n_action_steps。
    --no-record / --no-results 分别关闭 Raw / 评估摘要；--output 指定输出目录。

执行模式通过 --config YAML 设置，默认 sync；异步示例：
    execution:
      execution_mode: async
      prefetch_steps: 2

RTC 使用 execution_mode: rtc，并显式设置 prefetch_steps 和 rtc_guidance_cap。
当前 RTC 仅支持连续动作 BaseAgent DDIM 策略；ManiFlow 支持 sync/async。
普通运行会连接设备，关闭输出不是离线模式；--print-config 不连接设备。
其他参数见 --help。
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _nonnegative_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def _positive_running_seconds(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one policy-evaluation session with independently optional Raw recording and evaluation summary"
    )
    parser.add_argument(
        "experiment",
        metavar="EXPERIMENT",
        nargs="?",
        help="Saved Policy selector policy/task/experiment; required unless --print-config",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Optional Real hardware/execution YAML (execution_mode, prefetch_steps, "
            "rtc_guidance_cap under execution); Policy uses experiment/config.yaml"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="latest (default), best, milestone percentage (e.g. 80pct), or checkpoint filename",
    )
    parser.add_argument(
        "--num-episodes",
        dest="num_episodes",
        type=_positive_int,
        default=1,
        help=(
            "episodes to run; each truly begun episode counts once at its end, "
            "independently of whether its recording saved"
        ),
    )
    parser.add_argument(
        "--max-duration",
        dest="max_running_s",
        type=_positive_running_seconds,
        default=60.0,
        help="cooperative duration budget from RUNNING admission (default: 60 seconds)",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print declared Real config without loading Policy or calibration",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Session root (default: checkout/rollouts)"
    )
    parser.add_argument("--camera-calibration", type=Path, default=None)
    parser.add_argument(
        "--no-record", action="store_true", help="Disable Raw recording independently of --no-results"
    )
    parser.add_argument(
        "--no-results", action="store_true", help="Disable evaluation summary independently of --no-record"
    )
    execution = parser.add_argument_group("Real execution")
    execution.add_argument(
        "--n-action-steps",
        type=_positive_int,
        default=None,
        help="A_exec: actions per segment (default: saved Policy n_action_steps)",
    )
    research = parser.add_argument_group("Policy inference")
    research.add_argument("--weights", choices=("ema", "raw"), default=None)
    research.add_argument(
        "--inference-steps",
        type=_positive_int,
        default=None,
        help="override saved inference step count (NFE)",
    )
    research.add_argument(
        "--seed", type=_nonnegative_int, default=0, help="fixed inference seed reset each episode"
    )
    research.add_argument("--device", default="cuda:0")
    return parser


def _safe_selector_parts(selector: Any) -> tuple[str, str, str]:
    if not isinstance(selector, str):
        raise ValueError("Policy selector must be a string")
    parts = tuple(selector.split("/"))
    if len(parts) != 3 or any(
        not part or part in {".", ".."} or "/" in part or "\\" in part or Path(part).name != part
        for part in parts
    ):
        raise ValueError("experiment must be a canonical policy/task/experiment selector")
    return parts[0], parts[1], parts[2]


def _session_directory(root: Path, selector: str) -> Path:
    parts = _safe_selector_parts(selector)
    base = root / Path(*parts)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = base / f"session_{stamp}"
    dedup = 1
    while candidate.exists():
        candidate = base / f"session_{stamp}_{dedup:02d}"
        dedup += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _write_run_config(session_dir, *, args, runtime, info, execution):
    import dexmani_policy
    import yaml

    from dexmani_real.config.experiment import config_as_dict

    def head(root):
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    compact_info = config_as_dict(info)
    compact_info["experiment_dir"] = str(info.experiment_dir)
    compact_info["checkpoint_path"] = str(info.checkpoint_path)
    payload = {
        "experiment": args.experiment,
        "checkpoint": args.checkpoint,
        "policy": compact_info,
        "future_steps": info.horizon - info.n_obs_steps + 1,
        "runtime": config_as_dict(runtime),
        "execution": config_as_dict(execution),
        "seed": args.seed,
        "device": args.device,
        "num_episodes": args.num_episodes,
        "max_duration_s": args.max_running_s,
        "record_raw": not args.no_record,
        "record_results": not args.no_results,
        "n_action_steps_override": args.n_action_steps,
        "dexmani_real_head": head(Path(__file__).resolve().parents[1]),
        "dexmani_policy_head": head(Path(dexmani_policy.__file__).resolve().parents[1]),
    }
    (session_dir / "run_config.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.print_config:
        if args.experiment is None:
            parser.error("EXPERIMENT is required unless --print-config is used")
        try:
            _safe_selector_parts(args.experiment)
        except ValueError as exc:
            parser.error(str(exc))

    import yaml

    from dexmani_real.config.experiment import config_as_dict, load_experiment_config

    try:
        runtime = load_experiment_config(yaml_path=args.config)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.error(f"invalid experiment config: {exc}")
    if args.print_config:
        print(yaml.safe_dump(config_as_dict(runtime), allow_unicode=True, sort_keys=False), end="")
        return 0

    try:
        from dexmani_policy.deployment import (
            inspect_policy,
            load_experiment_config,
            resolve_experiment,
        )

        experiment = resolve_experiment(args.experiment)
        saved_config = load_experiment_config(experiment)
        checkpoint = args.checkpoint
        if checkpoint.endswith("pct"):
            from dexmani_policy.evaluation.protocol import resolve_checkpoint_path

            checkpoint, _ = resolve_checkpoint_path(experiment, checkpoint)
        info = inspect_policy(
            experiment,
            config=saved_config,
            checkpoint=checkpoint,
            weights=args.weights,
            inference_steps=args.inference_steps,
        )
    except Exception as exc:
        print(f"[POLICY] experiment inspection failed: {exc}", file=sys.stderr)
        return 1

    try:
        from dexmani_real.deployment.config import (
            PolicyRuntimeConfig,
            RolloutRecordingConfig,
            resolve_execution_config,
        )

        from dataclasses import replace

        runtime = replace(runtime, execution=resolve_execution_config(
            runtime.execution, info, args.n_action_steps
        ))
        runtime.execution.validate(info, max_running_s=args.max_running_s)
        selector = "/".join((info.policy_name, info.task_name, info.experiment_dir.name))
        _safe_selector_parts(selector)
    except Exception as exc:
        print(f"[PREFLIGHT] preflight failed: {exc}", file=sys.stderr)
        return 1

    try:
        policy_config = PolicyRuntimeConfig(saved_config, info, args.device, args.seed)
        session_dir = None
        recording_config = None
        if not (args.no_record and args.no_results):
            root = args.output or Path(__file__).resolve().parents[1] / "rollouts"
            session_dir = _session_directory(root, selector)
            if not args.no_record:
                recording_config = RolloutRecordingConfig(str(session_dir), info.task_name)
        elif args.output is not None:
            print("--no-record --no-results: --output is unused; no session artifacts will be created")
    except Exception as exc:
        print(f"[SESSION] session setup failed: {exc}", file=sys.stderr)
        return 1

    print(f"Experiment: {info.experiment_dir}")
    print(f"Checkpoint: {info.checkpoint_path.name} ({info.weights}); steps={info.inference_steps}")
    print(f"Observations: {', '.join(info.observation_fields)}")
    print(
        f"A_exec: {runtime.execution.action_steps}; "
        f"source: {'CLI' if args.n_action_steps is not None else 'saved Policy default'}"
    )
    print(f"Episodes: {args.num_episodes}; cooperative budget={args.max_running_s:g} s")
    print(f"Session: {session_dir}" if session_dir else "Session artifacts: disabled", flush=True)
    try:
        from dexmani_real.deployment.session import run_policy_deployment

        return run_policy_deployment(
            runtime,
            policy_config,
            True,
            max_running_s=args.max_running_s,
            num_episodes=args.num_episodes,
            recording_config=recording_config,
            results_dir=session_dir if not args.no_results else None,
            camera_calibration_path=args.camera_calibration,
            save_run_config=(
                lambda resolved, effective: _write_run_config(
                    session_dir,
                    args=args,
                    runtime=resolved,
                    info=info,
                    execution=effective,
                )
            )
            if session_dir is not None
            else None,
        )
    except Exception as exc:
        print(f"[LIFECYCLE] lifecycle failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
