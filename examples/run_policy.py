#!/usr/bin/env python3
"""真机策略评估；--print-config 仅打印配置，不连接设备。"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any


def positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def nonnegative_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def nonnegative_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite non-negative number") from exc
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return value


def positive_running_seconds(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one policy-evaluation session with independently optional "
            "Raw recording and evaluation summary"
        )
    )
    parser.add_argument(
        "experiment",
        metavar="EXPERIMENT",
        nargs="?",
        help="Saved Policy selector or absolute experiment directory; required unless --print-config",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Optional Real hardware/execution YAML; execution CLI options take precedence. "
            "Policy uses experiment/config.yaml"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="latest (default), best, milestone percentage (e.g. 80pct), or checkpoint filename",
    )
    parser.add_argument(
        "--num-episodes",
        type=positive_int,
        default=20,
        help=(
            "exit after N episodes (default: 20); press B for each episode or Q to exit early; "
            "each truly begun episode counts once at its end, regardless of recording"
        ),
    )
    parser.add_argument(
        "--max-duration",
        dest="max_running_s",
        type=positive_running_seconds,
        default=60.0,
        help="per-episode cooperative duration budget from RUNNING admission (default: 60 seconds)",
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
        "--no-record",
        action="store_true",
        help="Disable Raw recording independently of --no-results",
    )
    parser.add_argument(
        "--no-results",
        action="store_true",
        help="Disable evaluation summary independently of --no-record",
    )
    execution = parser.add_argument_group("Real execution")
    execution.add_argument(
        "--execution-mode",
        choices=("sync", "async", "rtc"),
        default=None,
        help="Override execution mode (default: YAML or sync)",
    )
    execution.add_argument(
        "--prefetch-steps",
        type=positive_int,
        default=None,
        help="Override prefetch delay d for async/RTC; requires 1 <= d <= A_exec and A_exec+d <= P",
    )
    execution.add_argument(
        "--rtc-guidance-cap",
        type=nonnegative_float,
        default=None,
        help="Override finite nonnegative RTC guidance cap; 0 retains prefix freezing without guidance",
    )
    execution.add_argument(
        "--n-action-steps",
        type=positive_int,
        default=None,
        help=(
            "A_exec: actions per segment "
            "(default: YAML execution.action_steps or saved Policy n_action_steps)"
        ),
    )
    research = parser.add_argument_group("Policy inference")
    research.add_argument("--weights", choices=("ema", "raw"), default=None)
    research.add_argument(
        "--inference-steps",
        type=positive_int,
        default=None,
        help="override saved inference step count (NFE)",
    )
    research.add_argument(
        "--seed", type=nonnegative_int, default=0, help="fixed inference seed reset each episode"
    )
    research.add_argument("--device", default="cuda:0")
    return parser


def safe_selector_parts(selector: Any) -> tuple[str, str, str]:
    if not isinstance(selector, str):
        raise ValueError("Policy selector must be a string")
    parts = tuple(selector.split("/"))
    if len(parts) != 3 or any(
        not part or part in {".", ".."} or "/" in part or "\\" in part or Path(part).name != part
        for part in parts
    ):
        raise ValueError("experiment must be a canonical policy/task/experiment selector")
    return parts[0], parts[1], parts[2]


def session_directory(root: Path, selector: str) -> Path:
    parts = safe_selector_parts(selector)
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


def write_run_config(session_dir, *, args, runtime, info):
    import dexmani_policy
    import yaml

    def head(root):
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    compact_info = info.to_dict()
    payload = {
        "experiment": args.experiment,
        "checkpoint": args.checkpoint,
        "policy": compact_info,
        "future_steps": info.horizon - info.n_obs_steps + 1,
        "runtime": asdict(runtime),
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
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.print_config and args.experiment is None:
        parser.error("EXPERIMENT is required unless --print-config is used")

    import yaml

    from dexmani_real.config.experiment import load_experiment_config

    try:
        runtime = load_experiment_config(
            yaml_path=args.config,
            cli_overrides={
                "execution.execution_mode": args.execution_mode,
                "execution.prefetch_steps": args.prefetch_steps,
                "execution.rtc_guidance_cap": args.rtc_guidance_cap,
                "execution.action_steps": args.n_action_steps,
            },
        )
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.error(f"invalid experiment config: {exc}")
    if args.print_config:
        print(yaml.safe_dump(asdict(runtime), allow_unicode=True, sort_keys=False), end="")
        return 0

    try:
        from dexmani_policy.deployment import (
            inspect_policy,
            resolve_experiment,
        )

        experiment = resolve_experiment(args.experiment)
        checkpoint = args.checkpoint
        if checkpoint.endswith("pct"):
            from dexmani_policy.evaluation.checkpoints import resolve_checkpoint_path

            checkpoint, _ = resolve_checkpoint_path(experiment, checkpoint)
        info = inspect_policy(
            experiment,
            checkpoint=checkpoint,
            weights=args.weights,
            inference_steps=args.inference_steps,
        )
    except Exception as exc:
        print(f"[POLICY] experiment inspection failed: {exc}", file=sys.stderr)
        return 1

    try:
        from dexmani_real.deployment.startup import validate_deployment_config

        action_steps = runtime.execution.action_steps
        if action_steps is None:
            action_steps = info.n_action_steps
        runtime = replace(runtime, execution=replace(runtime.execution, action_steps=action_steps))
        validate_deployment_config(info, runtime, record_raw=not args.no_record)
        selector = "/".join((info.policy_name, info.task_name, info.experiment_dir.name))
    except Exception as exc:
        print(f"[PREFLIGHT] preflight failed: {exc}", file=sys.stderr)
        return 1

    try:
        session_dir = None
        recording_dir = None
        if not (args.no_record and args.no_results):
            root = args.output or Path(__file__).resolve().parents[1] / "rollouts"
            session_dir = session_directory(root, selector)
            if not args.no_record:
                recording_dir = session_dir
        elif args.output is not None:
            print(
                "--no-record --no-results: --output is unused; no session artifacts will be created"
            )
    except Exception as exc:
        print(f"[SESSION] session setup failed: {exc}", file=sys.stderr)
        return 1

    print(f"策略：{args.experiment}")
    print(
        f"评估：{args.num_episodes} 轮后退出 | 每轮预算 {args.max_running_s:g} s | 输入：{', '.join(info.observation_fields)}"
    )
    print(f"输出：{session_dir}" if session_dir else "输出：Raw 和评估摘要均关闭")
    try:
        from dexmani_real.deployment.session import PolicyDeploymentSession

        return PolicyDeploymentSession(
            runtime,
            info,
            device=args.device,
            seed=args.seed,
            max_running_s=args.max_running_s,
            num_episodes=args.num_episodes,
            recording_dir=recording_dir,
            results_dir=session_dir if not args.no_results else None,
            camera_calibration_path=args.camera_calibration,
            save_run_config=(
                (
                    lambda resolved: write_run_config(
                        session_dir,
                        args=args,
                        runtime=resolved,
                        info=info,
                    )
                )
                if session_dir is not None
                else None
            ),
        ).run()
    except Exception as exc:
        print(f"[LIFECYCLE] lifecycle failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
