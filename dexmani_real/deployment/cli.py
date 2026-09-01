"""Side-effect-explicit command line for learned-policy deployment."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any


def _positive_seconds(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def _seed(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be an integer in [0, 2**31 - 1]"
        ) from exc
    if not 0 <= value <= 2**31 - 1:
        raise argparse.ArgumentTypeError("must be an integer in [0, 2**31 - 1]")
    return value


def _benchmark_samples(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer in [1, 1000]") from exc
    if not 1 <= value <= 1000:
        raise argparse.ArgumentTypeError("must be an integer in [1, 1000]")
    return value


def _add_artifact_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("experiment_dir", metavar="EXP")
    parser.add_argument("--config", default=None, help="Real runtime YAML")
    parser.add_argument(
        "--deployment-config",
        default=None,
        help="Real-owned deployment timing/readiness YAML",
    )
    parser.add_argument("--device", default="cuda:0", help="inference torch device")


def build_parser() -> argparse.ArgumentParser:
    """Build the stdlib-only parser without importing model or hardware paths."""
    parser = argparse.ArgumentParser(
        description="Inspect, qualify, shadow, or physically run one Policy artifact"
    )
    commands = parser.add_subparsers(dest="command")

    inspect_parser = commands.add_parser(
        "inspect", help="resolve and print an immutable receipt (no Torch/hardware)"
    )
    _add_artifact_options(inspect_parser)

    check_parser = commands.add_parser(
        "check", help="run isolated hardware-free restore and inference qualification"
    )
    _add_artifact_options(check_parser)
    check_parser.add_argument("--seed", type=_seed, required=True)
    check_parser.add_argument(
        "--benchmark-samples", type=_benchmark_samples, default=100
    )

    shadow_parser = commands.add_parser(
        "shadow", help="connect hardware and validate without learned command writes"
    )
    _add_artifact_options(shadow_parser)
    shadow_parser.add_argument("--seed", type=_seed, required=True)
    shadow_parser.add_argument(
        "--hand",
        action="store_true",
        help="acknowledge required XHand startup/reset side effects",
    )
    shadow_parser.add_argument(
        "--max-running-seconds", type=_positive_seconds, required=True
    )

    h4_parser = commands.add_parser(
        "h4", help="run one profile-bounded physical H4 endpoint"
    )
    h4_parser.add_argument("profile", metavar="PROFILE.yaml")

    run_parser = commands.add_parser(
        "run", help="run one bounded multi-endpoint physical profile"
    )
    run_parser.add_argument("profile", metavar="PROFILE.yaml")
    return parser


def _resolve_artifact_projection(
    *,
    experiment_dir: str | Path,
    runtime_config_path: str | Path | None,
    deployment_config_path: str | Path | None,
    device: str,
    seed: int | None,
    execution_mode: str,
    hand_acknowledged: bool,
    h4_execute_bounds: Any = None,
    task_execute_bounds: Any = None,
) -> tuple[Any, Any, Any]:
    from dexmani_real.config.runtime import resolve_runtime_config
    from dexmani_real.deployment.artifact import resolve_policy_artifact
    from dexmani_real.deployment.config import resolve_policy_runtime_config

    artifact = resolve_policy_artifact(experiment_dir)
    runtime = resolve_runtime_config(yaml_path=runtime_config_path)
    projection = resolve_policy_runtime_config(
        artifact=artifact,
        runtime_config=runtime,
        yaml_path=deployment_config_path,
        device=device,
        inference_seed=seed,
        execution_mode=execution_mode,
        hand_acknowledged=hand_acknowledged,
        h4_execute_bounds=h4_execute_bounds,
        task_execute_bounds=task_execute_bounds,
    )
    return artifact, runtime, projection


def _inspect(args: argparse.Namespace) -> int:
    artifact, runtime, projection = _resolve_artifact_projection(
        experiment_dir=args.experiment_dir,
        runtime_config_path=args.config,
        deployment_config_path=args.deployment_config,
        device=args.device,
        seed=None,
        execution_mode="shadow",
        hand_acknowledged=False,
    )
    from dexmani_real.deployment.run_identity import (
        canonical_run_receipt_json,
        resolve_real_source_identity,
    )

    print(
        canonical_run_receipt_json(
            artifact=artifact,
            projection=projection,
            runtime_sha256=runtime.sha256,
            real_source=resolve_real_source_identity(),
            preflight_result=None,
        )
    )
    return 0


def _check(args: argparse.Namespace) -> int:
    artifact, runtime, projection = _resolve_artifact_projection(
        experiment_dir=args.experiment_dir,
        runtime_config_path=args.config,
        deployment_config_path=args.deployment_config,
        device=args.device,
        seed=args.seed,
        execution_mode="shadow",
        hand_acknowledged=False,
    )
    from dexmani_real.deployment.preflight import run_isolated_policy_check
    from dexmani_real.deployment.run_identity import (
        canonical_run_receipt_json,
        resolve_real_source_identity,
    )

    result = run_isolated_policy_check(
        projection.runtime, benchmark_samples=args.benchmark_samples
    )
    print(
        canonical_run_receipt_json(
            artifact=artifact,
            projection=projection,
            runtime_sha256=runtime.sha256,
            real_source=resolve_real_source_identity(),
            preflight_result=result,
            check_result=result,
        )
    )
    return 0


def _shadow(args: argparse.Namespace) -> int:
    artifact, runtime, projection = _resolve_artifact_projection(
        experiment_dir=args.experiment_dir,
        runtime_config_path=args.config,
        deployment_config_path=args.deployment_config,
        device=args.device,
        seed=args.seed,
        execution_mode="shadow",
        hand_acknowledged=args.hand,
    )
    if artifact.allocation_contract.requires_hand and not args.hand:
        raise ValueError("shadow requires explicit --hand acknowledgement")
    from dexmani_real.deployment.run_identity import resolve_real_source_identity

    print(
        "shadow connects hardware; hand startup may reset/home; "
        "no learned coupled command publication is expected.",
        file=sys.stderr,
    )
    from dexmani_real.deployment.lifecycle import run_policy_deployment

    return run_policy_deployment(
        runtime,
        projection.runtime,
        max_running_s=args.max_running_seconds,
        real_source=resolve_real_source_identity(),
    )


def _physical(
    args: argparse.Namespace, *, command: str, invocation_argv: tuple[str, ...]
) -> int:
    from dexmani_real.deployment.config import H4ExecuteBounds, TaskExecuteBounds
    from dexmani_real.deployment.profile import load_physical_run_profile

    profile = load_physical_run_profile(args.profile)
    if command == "h4":
        if profile.max_published_endpoints != 1:
            raise ValueError("h4 profile max_published_endpoints must be exactly 1")
        h4_bounds = H4ExecuteBounds(
            max_published_endpoints=profile.max_published_endpoints,
            acknowledgement_timeout_s=profile.acknowledgement_timeout_seconds,
            max_running_s=profile.max_running_seconds,
        )
        task_bounds = None
        execution_mode = "execute"
    else:
        if profile.max_published_endpoints <= 1:
            raise ValueError(
                "run profile max_published_endpoints must be greater than 1"
            )
        h4_bounds = None
        task_bounds = TaskExecuteBounds(
            max_published_endpoints=profile.max_published_endpoints,
            acknowledgement_timeout_s=profile.acknowledgement_timeout_seconds,
            max_running_s=profile.max_running_seconds,
        )
        execution_mode = "task"
    artifact, runtime, projection = _resolve_artifact_projection(
        experiment_dir=profile.experiment_dir,
        runtime_config_path=profile.runtime_config,
        deployment_config_path=profile.deployment_config,
        device=profile.device,
        seed=profile.seed,
        execution_mode=execution_mode,
        hand_acknowledged=profile.hand_acknowledged,
        h4_execute_bounds=h4_bounds,
        task_execute_bounds=task_bounds,
    )
    if artifact.checkpoint_sha256_from_index != profile.expected_checkpoint_sha256:
        raise ValueError("selected checkpoint SHA-256 does not match physical profile")
    from dexmani_real.deployment.run_identity import resolve_real_source_identity

    real_source = resolve_real_source_identity()
    if real_source.availability != "available" or real_source.dirty != "false":
        raise ValueError(
            "physical execution requires a clean, identifiable DexMani Real revision"
        )
    from dexmani_real.deployment.lifecycle import run_policy_deployment

    return run_policy_deployment(
        runtime,
        projection.runtime,
        max_running_s=None,
        real_source=real_source,
        invocation_argv=invocation_argv,
        projection_sha256=projection.sha256,
    )


def _dispatch(args: argparse.Namespace, *, invocation_argv: tuple[str, ...]) -> int:
    if args.command == "inspect":
        return _inspect(args)
    if args.command == "check":
        return _check(args)
    if args.command == "shadow":
        return _shadow(args)
    if args.command in {"h4", "run"}:
        return _physical(args, command=args.command, invocation_argv=invocation_argv)
    raise RuntimeError(f"unsupported deployment command: {args.command!r}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    invocation_argv = tuple(sys.argv if argv is None else [sys.argv[0], *argv])
    try:
        return _dispatch(args, invocation_argv=invocation_argv)
    except (
        KeyError,
        OSError,
        RuntimeError,
        TimeoutError,
        TypeError,
        UnicodeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
