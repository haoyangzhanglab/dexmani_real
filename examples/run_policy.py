#!/usr/bin/env python3
"""Resolve configs and start hardware-affecting learned-policy deployment.

The deployment can command xArm7/XHand and connect RealSense when required by
the model observation contract.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml

from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.deployment.config import resolve_deployment_config
from dexmani_real.deployment.lifecycle import run_policy_deployment
from dexmani_real.ipc.schema import SUPPORTED_POINT_CLOUD_COUNTS
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a learned-policy deployment on the xArm7 + XHand runtime"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML with runtime overrides (arm/hand/safety/control rates)",
    )
    parser.add_argument(
        "--deployment-config",
        type=str,
        default=None,
        help="YAML with deployment overrides (runtime target, checkpoint, device)",
    )
    parser.add_argument(
        "--runtime", type=str, default=None, help="module:symbol PolicyRuntime target"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="model checkpoint path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="inference device (default from config)",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="Expected training task identity; required by DexMani Policy",
    )
    parser.add_argument(
        "--hand",
        action="store_true",
        help="Enable coupled XHand control (deployment.hand_enabled=true)",
    )
    parser.add_argument(
        "--pointcloud-num-points",
        type=int,
        choices=sorted(SUPPORTED_POINT_CLOUD_COUNTS),
        default=None,
        help="Fixed point-cloud observation size (default from deployment config)",
    )
    parser.add_argument(
        "--action-key",
        type=str,
        choices=("action", "action_ee"),
        default=None,
        help="Action contract (joint 19D or EE 21D); must match the checkpoint",
    )
    parser.add_argument(
        "--observation-fields",
        type=str,
        default=None,
        help="Comma-separated observation contract (deployment.observation_fields)",
    )
    parser.add_argument(
        "--print-config", action="store_true", help="Print resolved configs and exit"
    )
    args = parser.parse_args(argv)

    try:
        runtime = resolve_runtime_config(yaml_path=args.config)
        deployment_resolved = resolve_deployment_config(
            yaml_path=args.deployment_config,
            cli_overrides={
                "runtime_target": args.runtime,
                "checkpoint": args.checkpoint,
                "device": args.device,
                "task_name": args.task_name,
                "hand_enabled": True if args.hand else None,
                "pointcloud_num_points": args.pointcloud_num_points,
                "action_key": args.action_key,
                "observation_fields": args.observation_fields,
            },
        )
    except (
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        yaml.YAMLError,
    ) as exc:
        parser.error(f"invalid deployment config: {exc}")

    if args.print_config:
        print(deployment_resolved.canonical_json)
        print(f"deployment_sha256={deployment_resolved.sha256}")
        print(f"runtime_sha256={runtime.sha256}")
        return 0

    deployment = deployment_resolved.deployment
    try:
        return run_policy_deployment(runtime, deployment)
    except Exception:
        logger.error(
            "policy deployment failed before lifecycle ownership was established",
            exc_info=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
