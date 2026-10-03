#!/usr/bin/env python3
"""Usage: python examples/run_policy.py POLICY/TASK/EXPERIMENT [--config YAML] [--checkpoint best|latest|FILE]
       [--weights ema|raw] [--inference-steps N] [--seed S] [--num-episodes N] [--max-duration SEC] [--device D]
真机评估：H 回零 → 布置场景 → B 开始 → S 停止；保存 rollout 与 run_config.yaml，任务成功由离线评估判定。"""

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
        description="Run one persistent recorded policy-evaluation session"
    )
    parser.add_argument("experiment", metavar="EXPERIMENT")
    parser.add_argument(
        "--config", help="Real hardware/runtime YAML; Policy uses experiment/config.yaml"
    )
    parser.add_argument("--checkpoint", default="best", help="best, latest, or checkpoint filename")
    parser.add_argument("--weights", choices=("ema", "raw"), default=None)
    parser.add_argument(
        "--inference-steps",
        type=_positive_int,
        default=None,
        help="override saved inference steps",
    )
    parser.add_argument(
        "--n-action-steps",
        type=_positive_int,
        default=None,
        help="override deployment chunk length without changing the training snapshot",
    )
    parser.add_argument(
        "--seed", type=_nonnegative_int, default=0, help="fixed inference seed reset each episode"
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
    parser.add_argument("--device", default="cuda:0")
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


def _write_run_config(session_dir, *, args, runtime, info):
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
        "execution_path": "synchronous_direct_sdk_v1",
        "observation_action_pairing": "control_tick_input_and_attempted_targets",
        "experiment": args.experiment,
        "checkpoint": args.checkpoint,
        "policy": compact_info,
        "runtime": config_as_dict(runtime),
        "seed": args.seed,
        "device": args.device,
        "num_episodes": args.num_episodes,
        "max_duration_s": args.max_running_s,
        "n_action_steps_override": args.n_action_steps,
        "dexmani_real_head": head(Path(__file__).resolve().parents[1]),
        "dexmani_policy_head": head(Path(dexmani_policy.__file__).resolve().parents[1]),
    }
    (session_dir / "run_config.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    try:
        from dexmani_policy.deployment import (
            inspect_policy,
            load_experiment_config,
            resolve_experiment,
        )

        experiment = resolve_experiment(args.experiment)
        from dexmani_real.deployment.config import with_action_steps

        saved_config = with_action_steps(load_experiment_config(experiment), args.n_action_steps)
        info = inspect_policy(
            experiment,
            config=saved_config,
            checkpoint=args.checkpoint,
            weights=args.weights,
            inference_steps=args.inference_steps,
        )
    except Exception as exc:
        print(f"[POLICY] experiment inspection failed: {exc}", file=sys.stderr)
        return 1

    try:
        from dexmani_real.config.experiment import resolve_experiment_config
        from dexmani_real.deployment.config import (
            PolicyRuntimeConfig,
            RolloutRecordingConfig,
            validate_num_episodes,
            validate_policy_runtime_compatibility,
        )

        runtime = resolve_experiment_config(yaml_path=args.config)
        validate_policy_runtime_compatibility(info, runtime)
        validate_num_episodes(args.num_episodes)
        selector = "/".join((info.policy_name, info.task_name, info.experiment_dir.name))
        _safe_selector_parts(selector)
    except Exception as exc:
        print(f"[COMPAT] preflight failed: {exc}", file=sys.stderr)
        return 1

    try:
        root = Path(__file__).resolve().parents[1] / "rollouts"
        session_dir = _session_directory(root, selector)
        _write_run_config(session_dir, args=args, runtime=runtime, info=info)
        policy_config = PolicyRuntimeConfig(saved_config, info, args.device, args.seed)
        recording_config = RolloutRecordingConfig(str(session_dir), info.task_name)
    except Exception as exc:
        print(f"[COMPAT] session setup failed: {exc}", file=sys.stderr)
        return 1

    print(f"Experiment: {info.experiment_dir}")
    print(f"Checkpoint: {info.checkpoint_path.name} ({info.weights}); steps={info.inference_steps}")
    print(f"Observations: {', '.join(info.observation_fields)}")
    print(f"Action: {info.action_mode}; {1 / info.control_dt_s:g} Hz; chunk={info.n_action_steps}")
    print(f"Episodes: {args.num_episodes}; cooperative budget={args.max_running_s:g} s")
    print(f"Session: {session_dir}", flush=True)
    try:
        from dexmani_real.deployment.session import run_policy_deployment

        return run_policy_deployment(
            runtime,
            policy_config,
            True,
            max_running_s=args.max_running_s,
            num_episodes=args.num_episodes,
            recording_config=recording_config,
        )
    except Exception as exc:
        print(f"[LIFECYCLE] lifecycle failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
