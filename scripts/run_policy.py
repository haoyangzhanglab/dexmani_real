#!/usr/bin/env python3
"""真机策略评估；--print-config 仅打印配置，不连接设备。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from dexmani_real.config.deployment import (
    DEFAULT_MAX_RUNNING_S,
    DEFAULT_NUM_EPISODES,
    RUNTIME_SECTIONS,
)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one policy-evaluation session with optional results and full Raw recording"
        )
    )
    parser.add_argument(
        "policy",
        metavar="POLICY",
        nargs="?",
        help="Checkpoint file or directory containing checkpoints/latest.pt; relative to cwd",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Trusted Python experiment file exporting CONFIG: RealConfig. "
            "Policy recipe is restored from its checkpoint"
        ),
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help=(
            f"exit after N episodes that started running have ended (default: {DEFAULT_NUM_EPISODES}); "
            "press B for each episode or Q to exit early; "
            "failed/cancelled preparations do not count; independent of Raw recording"
        ),
    )
    parser.add_argument(
        "--max-duration",
        dest="max_running_s",
        type=float,
        default=DEFAULT_MAX_RUNNING_S,
        help=(
            "per-episode cooperative duration budget from the actual episode start "
            f"(default: {DEFAULT_MAX_RUNNING_S:g} seconds)"
        ),
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print declared Real config without loading Policy or calibration",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional output root; defaults to checkout/rollouts only with --record-raw",
    )
    parser.add_argument("--camera-calibration", type=Path, default=None)
    parser.add_argument(
        "--record-raw",
        action="store_true",
        help="Enable full Raw recording and evaluation results",
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
    if not args.print_config and args.policy is None:
        parser.error("POLICY is required unless --print-config is used")

    from dexmani_real.config.real import load_config

    runtime = load_config(args.config)
    if args.print_config:
        declared = {name: asdict(getattr(runtime, name)) for name in RUNTIME_SECTIONS}
        print(json.dumps(declared, ensure_ascii=False, indent=2, allow_nan=False))
        return 0

    from dexmani_real.deployment.run import run_policy

    return run_policy(
        args.policy,
        runtime,
        weights=args.weights,
        inference_steps=args.inference_steps,
        device=args.device,
        seed=args.seed,
        num_episodes=args.num_episodes,
        max_running_s=args.max_running_s,
        record_raw=args.record_raw,
        output=args.output,
        camera_calibration_path=args.camera_calibration,
    )


if __name__ == "__main__":
    raise SystemExit(main())
