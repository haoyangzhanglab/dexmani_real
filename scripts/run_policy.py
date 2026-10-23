#!/usr/bin/env python3
"""真机策略评估；--print-config 仅打印配置，不连接设备。"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path


def positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def nonnegative_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return value


def positive_running_seconds(raw: str) -> float:
    value = float(raw)
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
        help=(
            "latest (default), best, milestone percentage (e.g. 80pct), "
            "or a relative/absolute path within the experiment's checkpoints directory"
        ),
    )
    parser.add_argument(
        "--num-episodes",
        type=positive_int,
        default=20,
        help=(
            "exit after N episodes that entered RUNNING have ended (default: 20); "
            "press B for each episode or Q to exit early; "
            "failed/cancelled preparations do not count; independent of Raw recording"
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
            "A_exec: actions per segment (nominal replan length in async; "
            "valid old tails may run longer) "
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.print_config and args.experiment is None:
        parser.error("EXPERIMENT is required unless --print-config is used")

    import yaml

    from dexmani_real.config.experiment import load_experiment_config

    runtime = load_experiment_config(
        yaml_path=args.config,
        cli_overrides={
            "execution.execution_mode": args.execution_mode,
            "execution.prefetch_steps": args.prefetch_steps,
            "execution.rtc_guidance_cap": args.rtc_guidance_cap,
            "execution.action_steps": args.n_action_steps,
        },
    )
    if args.print_config:
        print(yaml.safe_dump(asdict(runtime), allow_unicode=True, sort_keys=False), end="")
        return 0

    from dexmani_real.deployment.run import run_policy

    return run_policy(
        args.experiment,
        runtime,
        checkpoint=args.checkpoint,
        weights=args.weights,
        inference_steps=args.inference_steps,
        device=args.device,
        seed=args.seed,
        num_episodes=args.num_episodes,
        max_running_s=args.max_running_s,
        record_raw=not args.no_record,
        record_results=not args.no_results,
        output=args.output,
        camera_calibration_path=args.camera_calibration,
    )


if __name__ == "__main__":
    raise SystemExit(main())
