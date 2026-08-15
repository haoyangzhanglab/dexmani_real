#!/usr/bin/env python3
"""Thin learned-policy deployment CLI (execution doc §84).

Contains only ``argparse -> config override -> resolve -> run lifecycle -> exit
code``. It never carries model code, CUDA code, scheduling, a SafetyGate, or
SharedStorage business logic — those live in ``dexmani_real/deployment`` and the
backend/adapter targets named by the deployment config.

The deployment config names ``backend_target`` / ``observation_adapter_target``
/ ``action_adapter_target`` as ``module:symbol`` factories (resolved lazily in
the inference child). The runtime config supplies the arm/hand safety limits and
control rates shared with every other workflow.

Usage:
    python examples/run_policy.py --deployment-config deploy.yaml [--config runtime.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.deployment.config import resolve_deployment_config
from dexmani_real.deployment.lifecycle import run_policy_deployment
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a learned-policy deployment on the xArm7 + XHand runtime"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="YAML with runtime overrides (arm/hand/safety/control rates)"
    )
    parser.add_argument(
        "--deployment-config",
        type=str,
        default=None,
        help="YAML with deployment overrides (backend/adapter targets, checkpoint, device)",
    )
    parser.add_argument("--backend", type=str, default=None, help="module:symbol backend target")
    parser.add_argument(
        "--observation-adapter", type=str, default=None, help="module:symbol observation adapter target"
    )
    parser.add_argument("--action-adapter", type=str, default=None, help="module:symbol action adapter target")
    parser.add_argument("--checkpoint", type=str, default=None, help="model checkpoint path")
    parser.add_argument("--device", type=str, default=None, help="inference device (default from config)")
    parser.add_argument(
        "--hand",
        action="store_true",
        help="Enable coupled XHand control (deployment.hand_enabled=true)",
    )
    parser.add_argument("--print-config", action="store_true", help="Print resolved configs and exit")
    args = parser.parse_args(argv)

    try:
        runtime = resolve_runtime_config(yaml_path=args.config)
        deployment_resolved = resolve_deployment_config(
            yaml_path=args.deployment_config,
            cli_overrides={
                "backend_target": args.backend,
                "observation_adapter_target": args.observation_adapter,
                "action_adapter_target": args.action_adapter,
                "checkpoint": args.checkpoint,
                "device": args.device,
                "hand_enabled": True if args.hand else None,
            },
        )
    except (KeyError, OSError, TypeError, UnicodeError, ValueError, yaml.YAMLError) as exc:
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
        logger.error("policy deployment failed before lifecycle ownership was established", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
