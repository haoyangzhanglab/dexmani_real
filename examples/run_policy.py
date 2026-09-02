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
    """Render CLI-contract failures with the compatibility owner marker."""

    def error(self, message: str) -> NoReturn:
        self.exit(2, f"{self.prog}: error: [COMPAT] {message}\n")


@dataclass(frozen=True)
class _LifecycleInputs:
    """Resolved inputs for one validate-only or physical lifecycle."""

    execute: bool
    runtime: Any
    policy_spec: Any
    worker_config: Any


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        description="Inspect, check, shadow, or stage one DexMani Policy experiment"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Policy inference device for check or a lifecycle (default: cuda:0)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="optional Real runtime YAML for shadow/run",
    )
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

    for command, help_text in (
        ("check", "strictly restore and smoke-test one Policy experiment"),
        ("shadow", "run with full validation and no actuator publication"),
        ("run", "run with physical coupled arm/hand publication"),
    ):
        command_parser = subcommands.add_parser(command, help=help_text)
        command_parser.add_argument("experiment", metavar="EXPERIMENT")
        # Accept shared advanced options after the subcommand as well as before it.
        # SUPPRESS preserves a value supplied to the top-level parser.
        command_parser.add_argument("--device", default=argparse.SUPPRESS)
        command_parser.add_argument("--config", default=argparse.SUPPRESS)
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


def _print_compatibility_error(message: str) -> None:
    print(f"[COMPAT] {message}", file=sys.stderr)


def _print_experiment_summary(info: Any, *, mode: str, device: str) -> None:
    """Print the small operator-facing subset of a Policy experiment contract."""
    spec = info.spec
    point_cloud = "none"
    if spec.point_cloud_num_points is not None:
        point_cloud = (
            f"{spec.point_cloud_num_points} \u00d7 {spec.point_cloud_feature_dim}"
        )
    print("\u2500\u2500 Policy Experiment \u2500\u2500")
    print(f"Mode        : {mode}")
    print(f"Experiment  : {info.selector}")
    print(f"Policy      : {info.policy_name}")
    print(f"Checkpoint  : {info.checkpoint_name}")
    print(f"Device      : {device}")
    print(f"Observation : {' + '.join(spec.sensor_modalities)}")
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
        _print_policy_error(f"checkpoint restore or smoke test failed: {exc}")
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
        PolicyWorkerConfig,
        validate_policy_runtime_compatibility,
    )

    if not isinstance(execute, bool):
        raise TypeError("execute must be a boolean")
    runtime = resolve_runtime_config(yaml_path=args.config)
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
    )


def _start_lifecycle(inputs: _LifecycleInputs) -> int:
    """Enter the hardware lifecycle after CLI-owned compatibility checks."""
    from dexmani_real.deployment.lifecycle import run_policy_deployment

    return run_policy_deployment(
        inputs.runtime,
        inputs.policy_spec,
        inputs.worker_config,
        inputs.execute,
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
        _print_compatibility_error(f"lifecycle failed: {exc}")
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


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "list": _run_list,
        "check": _run_check,
        "shadow": lambda parsed: _run_lifecycle(parsed, execute=False),
        "run": lambda parsed: _run_lifecycle(parsed, execute=True),
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
