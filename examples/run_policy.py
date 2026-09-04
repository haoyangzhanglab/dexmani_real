#!/usr/bin/env python3
"""Research-facing entry point for one DexMani Policy experiment.

The command line owns experiment selection and operator intent. Policy owns
checkpoint inspection and restore; Real owns validation and robot lifecycle.
All imports that can reach Policy, Torch, or Real runtime code remain inside
their command handlers so ``list`` stays a filesystem-only Policy operation.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Callable, NoReturn


class _ArgumentParser(argparse.ArgumentParser):
    """Render command-line contract failures with their owner marker."""

    def error(self, message: str) -> NoReturn:
        self.exit(2, f"{self.prog}: error: [CLI] {message}\n")

    def parse_args(
        self, args: list[str] | None = None, namespace: argparse.Namespace | None = None
    ) -> argparse.Namespace:
        parsed = super().parse_args(args, namespace)
        _apply_command_defaults(parsed)
        return parsed


_LIFECYCLE_OPTION_FLAGS = {
    "runtime_config": "--runtime-config",
    "inference_mode": "--inference-mode",
    "max_action_steps": "--max-action-steps",
}


@dataclass(frozen=True)
class _LifecycleInputs:
    """Resolved inputs for one validate-only or physical lifecycle."""

    execute: bool
    runtime: Any
    policy_spec: Any
    worker_config: Any
    deployment_config: Any


def _positive_action_steps(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _add_device_option(parser: argparse.ArgumentParser, *, default: object) -> None:
    """Add the inference-device option for a model-consuming command."""
    parser.add_argument(
        "--device",
        default=default,
        help="Policy inference device for check, shadow, or run (default: cuda:0)",
    )


def _add_lifecycle_options(parser: argparse.ArgumentParser, *, default: object) -> None:
    """Add options consumed only by a Real deployment lifecycle."""
    parser.add_argument(
        "--runtime-config",
        dest="runtime_config",
        default=default,
        help="optional Real runtime YAML for shadow/run",
    )
    parser.add_argument(
        "--inference-mode",
        choices=("sync", "async"),
        default=default,
        help="inference scheduling mode (default: sync)",
    )
    parser.add_argument(
        "--max-action-steps",
        type=_positive_action_steps,
        default=default,
        help="episode action-step limit (default: unlimited)",
    )


def _add_precommand_options(parser: argparse.ArgumentParser) -> None:
    """Accept model/lifecycle options before their owning subcommand.

    ``SUPPRESS`` preserves an explicitly pre-command value against the
    subparser and lets ``main`` reject an option not owned by ``list/check``.
    """
    _add_device_option(parser, default=argparse.SUPPRESS)
    _add_lifecycle_options(parser, default=argparse.SUPPRESS)


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        description="Inspect, check, shadow, or run one DexMani Policy experiment"
    )
    _add_precommand_options(parser)
    subcommands = parser.add_subparsers(dest="command", required=True)

    list_parser = subcommands.add_parser(
        "list", help="list deployable Policy experiment selectors"
    )
    list_parser.add_argument(
        "filter",
        nargs="?",
        default=None,
        help="optional case-insensitive selector substring",
    )

    check_parser = subcommands.add_parser(
        "check", help="strictly restore and smoke-test one Policy experiment"
    )
    check_parser.add_argument("experiment", metavar="EXPERIMENT")
    _add_device_option(check_parser, default=argparse.SUPPRESS)

    for command, help_text in (
        ("shadow", "run with full validation and no actuator publication"),
        ("run", "run with physical coupled arm/hand publication"),
    ):
        command_parser = subcommands.add_parser(command, help=help_text)
        command_parser.add_argument("experiment", metavar="EXPERIMENT")
        _add_device_option(command_parser, default=argparse.SUPPRESS)
        _add_lifecycle_options(command_parser, default=argparse.SUPPRESS)
    return parser


def _list_policy_experiments(filter_value: str | None) -> tuple[str, ...]:
    """Call the sole Policy API permitted by the ``list`` command."""
    from dexmani_policy.deployment import list_experiments

    return list_experiments(filter_value)


def _inspect_policy_experiment(experiment: str) -> Any:
    """Inspect metadata through Policy without constructing a model."""
    from dexmani_policy.deployment import inspect_experiment

    return inspect_experiment(experiment)


def _load_policy_experiment(experiment: str, device: str) -> Any:
    """Strictly restore one model through the Policy-owned runtime."""
    from dexmani_policy.deployment import load_experiment

    return load_experiment(experiment, device=device, seed=0)


def _print_policy_error(message: str) -> None:
    print(f"[POLICY] {message}", file=sys.stderr)


def _exception_detail(exc: Exception) -> str:
    """Keep an owned error readable without hiding its immediate cause."""
    cause = exc.__cause__
    if cause is None or not str(cause):
        return str(exc)
    return f"{exc}: {cause}"


def _print_compatibility_error(message: str) -> None:
    print(f"[COMPAT] {message}", file=sys.stderr)


def _print_lifecycle_error(message: str) -> None:
    print(f"[LIFECYCLE] {message}", file=sys.stderr)


def _print_experiment_summary(info: Any, *, mode: str, device: str) -> None:
    """Print the small operator-facing subset of a Policy experiment contract."""
    spec = info.spec
    fields = tuple(spec.observation_fields)
    point_cloud = "none"
    for field in fields:
        if field.name == "point_cloud":
            point_cloud = " x ".join(str(value) for value in field.shape)
            break
    print("\u2500\u2500 Policy Experiment \u2500\u2500")
    print(f"Mode        : {mode}")
    print(f"Experiment  : {info.selector}")
    print(f"Policy      : {info.policy_name}")
    print(f"Checkpoint  : {info.checkpoint_name}")
    print(f"Device      : {device}")
    print(f"Observation : {' + '.join(field.name for field in fields)}")
    print(f"History     : {spec.n_obs_steps}")
    print(f"Point Cloud : {point_cloud}")
    print(f"Action      : {spec.action_key} ({spec.control_action_dim}D)")
    print(f"Chunk       : {spec.n_action_steps}")
    print(f"Control     : {1.0 / spec.control_dt_s:g} Hz")
    print(
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
    )
    sys.stdout.flush()


def _validated_warmup_durations(raw: Any) -> tuple[float, ...]:
    """Validate the public warmup timings before reporting them."""
    if not isinstance(raw, tuple) or not raw:
        raise RuntimeError("Policy warmup returned no timing samples")
    durations = tuple(float(value) for value in raw)
    if any(not math.isfinite(value) or value < 0.0 for value in durations):
        raise RuntimeError("Policy warmup returned invalid timing samples")
    return durations


def _run_check(args: argparse.Namespace) -> int:
    try:
        info = _inspect_policy_experiment(args.experiment)
    except Exception as exc:
        _print_policy_error(f"experiment inspection failed: {exc}")
        return 1
    _print_experiment_summary(info, mode="CHECK", device=args.device)

    policy = None
    exit_code = 1
    try:
        policy = _load_policy_experiment(args.experiment, args.device)
        print("restore .......... OK")
        # Policy's strict restore performs its normalizer and metadata checks.
        print("normalizer ....... OK")
        # ``LoadedPolicy.warmup`` builds its deterministic synthetic observation,
        # calls ``predict``, and validates the finite [N, D] control output.
        durations = _validated_warmup_durations(policy.warmup(samples=3))
        print("warmup ........... OK")
        print("prediction ....... OK")
        print()
        print(f"inference warmup p50: {statistics.median(durations) * 1000.0:.0f} ms")
        print(f"inference warmup max: {max(durations) * 1000.0:.0f} ms")
        print()
        print("READY")
        exit_code = 0
    except Exception as exc:
        _print_policy_error(
            f"checkpoint restore or smoke test failed: {_exception_detail(exc)}"
        )
    finally:
        if policy is not None:
            try:
                policy.close()
            except Exception as exc:
                _print_policy_error(f"runtime cleanup failed: {exc}")
                exit_code = 1
    return exit_code


def _prepare_lifecycle_inputs(
    args: argparse.Namespace, info: Any, *, execute: bool
) -> _LifecycleInputs:
    """Resolve the PolicySpec/Real projection before lifecycle startup."""
    from dexmani_real.config.runtime import resolve_runtime_config
    from dexmani_real.deployment.config import (
        PolicyDeploymentConfig,
        PolicyWorkerConfig,
        validate_policy_runtime_compatibility,
    )

    if not isinstance(execute, bool):
        raise TypeError("execute must be a boolean")
    runtime = resolve_runtime_config(yaml_path=args.runtime_config)
    deployment_config = PolicyDeploymentConfig(
        inference_mode=args.inference_mode,
        max_action_steps=args.max_action_steps,
    )
    validate_policy_runtime_compatibility(info.spec, runtime)
    worker_config = PolicyWorkerConfig(
        experiment=info.selector,
        device=args.device,
        spec=info.spec,
    )
    return _LifecycleInputs(
        execute=execute,
        runtime=runtime,
        policy_spec=info.spec,
        worker_config=worker_config,
        deployment_config=deployment_config,
    )


def _start_lifecycle(inputs: _LifecycleInputs) -> int:
    """Enter the hardware lifecycle after CLI-owned compatibility checks."""
    from dexmani_real.deployment.lifecycle import run_policy_deployment

    return run_policy_deployment(
        inputs.runtime,
        inputs.policy_spec,
        inputs.worker_config,
        inputs.execute,
        deployment_config=inputs.deployment_config,
    )


def _run_lifecycle(args: argparse.Namespace, *, execute: bool) -> int:
    try:
        info = _inspect_policy_experiment(args.experiment)
    except Exception as exc:
        _print_policy_error(f"experiment inspection failed: {exc}")
        return 1
    _print_experiment_summary(
        info,
        mode="RUN" if execute else "SHADOW",
        device=args.device,
    )
    try:
        inputs = _prepare_lifecycle_inputs(args, info, execute=execute)
    except Exception as exc:
        _print_compatibility_error(f"runtime projection failed: {exc}")
        return 1
    try:
        return _start_lifecycle(inputs)
    except Exception as exc:
        _print_lifecycle_error(f"lifecycle failed: {exc}")
        return 1


def _run_list(args: argparse.Namespace) -> int:
    try:
        experiments = _list_policy_experiments(args.filter)
    except Exception as exc:
        _print_policy_error(f"experiment listing failed: {exc}")
        return 1
    if not experiments:
        print("No deployable Policy experiments found.")
        return 0
    print("Deployable Policy experiments:")
    for experiment in experiments:
        print(experiment)
    return 0


def _validate_command_options(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Reject pre-command options that no selected command consumes."""
    if args.command == "list":
        invalid_destinations = ("device", *_LIFECYCLE_OPTION_FLAGS)
    elif args.command == "check":
        invalid_destinations = tuple(_LIFECYCLE_OPTION_FLAGS)
    else:
        return
    supplied = [
        _LIFECYCLE_OPTION_FLAGS.get(destination, "--device")
        for destination in invalid_destinations
        if hasattr(args, destination)
    ]
    if supplied:
        parser.error(f"{args.command} does not accept {', '.join(supplied)}")


def _apply_command_defaults(args: argparse.Namespace) -> None:
    """Fill defaults only after explicit options from both parser levels survive."""
    if args.command == "list":
        return
    if not hasattr(args, "device"):
        args.device = "cuda:0"
    if args.command in {"shadow", "run"}:
        for destination in _LIFECYCLE_OPTION_FLAGS:
            if not hasattr(args, destination):
                default = "sync" if destination == "inference_mode" else None
                setattr(args, destination, default)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_command_options(parser, args)
    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "list": _run_list,
        "check": _run_check,
        "shadow": lambda parsed: _run_lifecycle(parsed, execute=False),
        "run": lambda parsed: _run_lifecycle(parsed, execute=True),
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
