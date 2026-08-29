#!/usr/bin/env python3
"""Resolve one hash-bound DexMani Policy artifact and run a bounded policy lifecycle.

``shadow`` is the default operational mode. ``execute`` remains the deliberately
one-publication H4 profile. ``task`` is a separate bounded full-episode profile.
Both physical profiles need independent review and hardware authorization; this
entry point never grants that authorization itself.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.deployment.artifact import resolve_policy_artifact
from dexmani_real.deployment.config import (
    H4ExecuteBounds,
    TaskExecuteBounds,
    resolve_policy_runtime_config,
)
from dexmani_real.deployment.run_identity import (
    canonical_run_receipt_json,
    resolve_real_source_identity,
)


def _positive_finite_seconds(raw: str) -> float:
    """Parse one operator-supplied positive duration without lifecycle imports."""
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def _one_positive_endpoint(raw: str) -> int:
    """Parse the deliberately fixed H4 physical-publication bound."""
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be exactly 1 for H4") from exc
    if value != 1:
        raise argparse.ArgumentTypeError("must be exactly 1 for H4")
    return value


def _multiple_positive_endpoints(raw: str) -> int:
    """Parse a bounded task publication count without widening H4."""
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer greater than 1") from exc
    if value <= 1:
        raise argparse.ArgumentTypeError("must be an integer greater than 1")
    return value


def _inference_seed(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be an integer in [0, 2**31 - 1]"
        ) from exc
    if not 0 <= value <= 2**31 - 1:
        raise argparse.ArgumentTypeError("must be an integer in [0, 2**31 - 1]")
    return value


def _sha256_hex(raw: str) -> str:
    """Parse one immutable H4 checkpoint digest."""
    value = raw.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise argparse.ArgumentTypeError("must be a 64-character SHA-256 hex digest")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve, preflight, shadow-validate, H4-test, or run one bounded DexMani Policy task"
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="experiment directory containing config.yaml and checkpoints/",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Real runtime YAML used after a shadow receipt is resolved",
    )
    parser.add_argument(
        "--deployment-config",
        default=None,
        help="Real-owned deployment timing/readiness YAML or artifact expectations",
    )
    parser.add_argument("--device", default=None, help="operator inference device")
    parser.add_argument(
        "--inference-seed",
        type=_inference_seed,
        default=None,
        help="Real-owned deterministic diffusion seed (set explicitly for receipts)",
    )
    parser.add_argument(
        "--hand",
        action="store_true",
        help="acknowledge XHand control required for hand startup/reset and H4 execute",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("shadow", "execute", "task"),
        default="shadow",
        help="shadow validates policy endpoints without actuator publication",
    )
    parser.add_argument(
        "--max-running-seconds",
        type=_positive_finite_seconds,
        default=None,
        help=(
            "optional B-relative shadow-run limit; coordinator must acknowledge "
            "the stop before shutdown"
        ),
    )
    parser.add_argument(
        "--execute-max-published-endpoints",
        type=_one_positive_endpoint,
        default=None,
        help="required H4 execute bound; currently must be exactly 1",
    )
    parser.add_argument(
        "--execute-ack-timeout-seconds",
        type=_positive_finite_seconds,
        default=None,
        help="required H4 execute arm+hand acknowledgement deadline",
    )
    parser.add_argument(
        "--execute-expected-checkpoint-sha256",
        type=_sha256_hex,
        default=None,
        help="required immutable H4 checkpoint SHA-256 from the approved reference",
    )
    parser.add_argument(
        "--task-max-published-endpoints",
        type=_multiple_positive_endpoints,
        default=None,
        help="required bounded full-episode coupled-publication count",
    )
    parser.add_argument(
        "--task-ack-timeout-seconds",
        type=_positive_finite_seconds,
        default=None,
        help="required per-endpoint arm+hand acknowledgement deadline",
    )
    parser.add_argument(
        "--task-expected-checkpoint-sha256",
        type=_sha256_hex,
        default=None,
        help="required immutable task checkpoint SHA-256",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--print-config",
        action="store_true",
        help="print pure resolved receipt and exit",
    )
    modes.add_argument(
        "--preflight-only",
        action="store_true",
        help="hash, single-load, restore, and fake-observation check in a child",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    h4_execute_bounds = None
    task_execute_bounds = None
    execute_args_present = any(
        value is not None
        for value in (
            args.execute_max_published_endpoints,
            args.execute_ack_timeout_seconds,
            args.execute_expected_checkpoint_sha256,
        )
    )
    task_args_present = any(
        value is not None
        for value in (
            args.task_max_published_endpoints,
            args.task_ack_timeout_seconds,
            args.task_expected_checkpoint_sha256,
        )
    )
    if args.execution_mode == "shadow":
        if execute_args_present or task_args_present:
            parser.error("physical execution bounds require a physical execution mode")
        if args.max_running_seconds is not None and (
            args.print_config or args.preflight_only
        ):
            parser.error(
                "--max-running-seconds is only valid for an operational shadow lifecycle"
            )
    elif args.execution_mode == "execute":
        if task_args_present:
            parser.error("--task-* bounds require --execution-mode task")
        if not args.hand:
            parser.error("H4 execute requires explicit --hand acknowledgement")
        if args.max_running_seconds is None:
            parser.error("H4 execute requires --max-running-seconds")
        if args.execute_max_published_endpoints is None:
            parser.error("H4 execute requires --execute-max-published-endpoints 1")
        if args.execute_ack_timeout_seconds is None:
            parser.error("H4 execute requires --execute-ack-timeout-seconds")
        if args.execute_expected_checkpoint_sha256 is None:
            parser.error("H4 execute requires --execute-expected-checkpoint-sha256")
        try:
            h4_execute_bounds = H4ExecuteBounds(
                max_published_endpoints=args.execute_max_published_endpoints,
                acknowledgement_timeout_s=args.execute_ack_timeout_seconds,
                max_running_s=args.max_running_seconds,
            )
        except (TypeError, ValueError) as exc:
            parser.error(f"invalid H4 execute bounds: {exc}")
    else:
        if execute_args_present:
            parser.error("--execute-* bounds require --execution-mode execute")
        if not args.hand:
            parser.error("task execute requires explicit --hand acknowledgement")
        if args.max_running_seconds is None:
            parser.error("task execute requires --max-running-seconds")
        if args.task_max_published_endpoints is None:
            parser.error("task execute requires --task-max-published-endpoints")
        if args.task_ack_timeout_seconds is None:
            parser.error("task execute requires --task-ack-timeout-seconds")
        if args.task_expected_checkpoint_sha256 is None:
            parser.error("task execute requires --task-expected-checkpoint-sha256")
        try:
            task_execute_bounds = TaskExecuteBounds(
                max_published_endpoints=args.task_max_published_endpoints,
                acknowledgement_timeout_s=args.task_ack_timeout_seconds,
                max_running_s=args.max_running_seconds,
            )
        except (TypeError, ValueError) as exc:
            parser.error(f"invalid task execute bounds: {exc}")
    try:
        artifact = resolve_policy_artifact(args.experiment_dir)
        runtime = resolve_runtime_config(yaml_path=args.config)
        projection = resolve_policy_runtime_config(
            artifact=artifact,
            runtime_config=runtime,
            yaml_path=args.deployment_config,
            device=args.device,
            inference_seed=args.inference_seed,
            execution_mode=args.execution_mode,
            hand_acknowledged=args.hand,
            h4_execute_bounds=h4_execute_bounds,
            task_execute_bounds=task_execute_bounds,
        )
    except (
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        yaml.YAMLError,
    ) as exc:
        parser.error(f"invalid deployment receipt/config: {exc}")

    expected_checkpoint_sha256 = (
        args.execute_expected_checkpoint_sha256
        if args.execution_mode == "execute"
        else (
            args.task_expected_checkpoint_sha256
            if args.execution_mode == "task"
            else None
        )
    )
    if expected_checkpoint_sha256 is not None and (
        artifact.checkpoint_sha256_from_index != expected_checkpoint_sha256
    ):
        parser.error(
            "selected checkpoint SHA-256 does not match "
            "the expected physical-execution checkpoint SHA-256"
        )

    real_source = resolve_real_source_identity()
    if args.print_config:
        print(
            canonical_run_receipt_json(
                artifact=artifact,
                projection=projection,
                runtime_sha256=runtime.sha256,
                real_source=real_source,
                preflight_result=None,
            )
        )
        return 0
    if args.preflight_only:
        # Deliberately lazy: print-config must not import torch, Policy, worker,
        # lifecycle, camera, robot, or other hardware-owning modules.
        from dexmani_real.deployment.preflight import run_isolated_preflight

        try:
            result = run_isolated_preflight(projection.runtime)
        except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
            print(f"policy preflight failed: {exc}", file=sys.stderr)
            return 1
        print(
            canonical_run_receipt_json(
                artifact=artifact,
                projection=projection,
                runtime_sha256=runtime.sha256,
                real_source=real_source,
                preflight_result=result,
            )
        )
        return 0
    if args.execution_mode in {"execute", "task"} and (
        real_source.availability != "available" or real_source.dirty != "false"
    ):
        parser.error(
            "physical execute requires a clean, identifiable DexMani Real revision"
        )
    # Deliberately lazy: receipt/preflight modes above must remain isolated
    # from lifecycle, camera, robot, and hardware-owning imports.
    from dexmani_real.deployment.lifecycle import run_policy_deployment

    if args.execution_mode == "shadow":
        return run_policy_deployment(
            runtime,
            projection.runtime,
            max_running_s=args.max_running_seconds,
        )
    return run_policy_deployment(
        runtime,
        projection.runtime,
        max_running_s=None,
        real_source=real_source,
        invocation_argv=tuple(sys.argv if argv is None else [sys.argv[0], *argv]),
    )


if __name__ == "__main__":
    raise SystemExit(main())
