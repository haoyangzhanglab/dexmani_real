"""真机策略评估；--print-config 仅打印配置，不连接设备。"""
# Usage: python scripts/run_policy.py CHECKPOINT
# HOI 需求由 Checkpoint 自动识别。

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from pprint import pprint

from dexmani_real.config.deployment import (
    DEFAULT_MAX_DURATION_S,
    DEFAULT_NUM_EPISODES,
    POLICY_RUNTIME_SECTIONS,
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


def duration_budget(raw: str) -> float | None:
    return None if raw.lower() == "none" else float(raw)


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
        dest="max_duration_s",
        type=duration_budget,
        default=DEFAULT_MAX_DURATION_S,
        help=(
            "per-episode cooperative duration budget from the actual episode start "
            f"(default: {DEFAULT_MAX_DURATION_S:g} seconds); none runs until user stop or failure"
        ),
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print declared RealConfig, not the effective deployment config; no Policy or calibration loading",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output root; defaults to checkout/rollouts with --record-raw; output failures end the session",
    )
    parser.add_argument("--camera-calibration", type=Path, default=None)
    parser.add_argument(
        "--task-dir", type=Path, default=None,
        help="HOI task assets directory; required by HOI policies; relative to cwd, supports ~",
    )
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
        help="override saved inference steps; ODE uses substeps per control period",
    )
    research.add_argument(
        "--seed", type=nonnegative_int, default=0, help="fixed inference seed reset at episode start and after HOI recovery"
    )
    research.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--profile-output", default=None,
        help="Write an opt-in performance JSON after shutdown; refuses overwrite",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.print_config and args.policy is None:
        parser.error("POLICY is required unless --print-config is used")

    from dexmani_real.config.real import load_config

    runtime = load_config(args.config)
    if args.print_config:
        pprint({name: getattr(runtime, name) for name in POLICY_RUNTIME_SECTIONS}, sort_dicts=False)
        return 0

    from dexmani_real.deployment.session import PolicySession
    from dexmani_real.utils.profile import profile_session

    with profile_session(args.profile_output, "policy", runtime, control_hz=None) as profiler:
        exit_code = PolicySession(
            runtime,
            args.policy,
            profiler=profiler,
            weights=args.weights,
            inference_steps=args.inference_steps,
            device=args.device,
            seed=args.seed,
            num_episodes=args.num_episodes,
            max_duration_s=args.max_duration_s,
            record_raw=args.record_raw,
            output_root=args.output,
            camera_calibration_path=args.camera_calibration,
            task_dir=args.task_dir,
        ).run()
        if profiler is not None:
            profiler.exit_code = exit_code
        return exit_code


if __name__ == "__main__":
    from dexmani_real.deployment.inference import ModelShutdownTimeout

    try:
        raise SystemExit(main())
    except ModelShutdownTimeout:
        # 设备、录制与结果已收尾；绕过解释器对卡住的模型线程的无限等待。
        try:
            logging.shutdown()
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(1)
