#!/usr/bin/env python3
"""Research-facing entry point for one DexMani Policy experiment.

``list`` reads Policy selectors only; ``check`` restores a checkpoint and warms
inference without starting robot workers. ``shadow`` starts the Real hardware
lifecycle while disabling command publication; ``run`` and ``eval`` can
physically command xArm7/XHand and record one rollout. The command line owns
experiment selection and operator intent. Policy owns checkpoint
inspection and restore; Real owns validation and robot lifecycle. All imports
that can reach Policy, Torch, or Real runtime code remain inside their command
handlers so ``list`` stays a filesystem-only Policy operation.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NoReturn


class _ArgumentParser(argparse.ArgumentParser):
    """Render command-line contract failures with their owner marker."""

    def error(self, message: str) -> NoReturn:
        self.exit(2, f"{self.prog}: error: [CLI] {message}\n")


@dataclass(frozen=True)
class _LifecycleInputs:
    """Resolved inputs for one validate-only or physical lifecycle."""

    execute: bool
    runtime: Any
    policy_spec: Any
    worker_config: Any
    evaluation_config: Any | None = None


def _positive_running_seconds(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def _eval_seed(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def _add_device_option(parser: argparse.ArgumentParser) -> None:
    """Add the inference-device option for a model-consuming command."""
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Policy inference device for check, shadow, run, or eval (default: cuda:0)",
    )


def _add_lifecycle_options(
    parser: argparse.ArgumentParser, *, evaluation: bool = False, physical: bool = True
) -> None:
    parser.add_argument("--eval-seed", type=_eval_seed, default=0, required=evaluation)
    if not physical:
        return
    parser.add_argument(
        "--max-duration",
        dest="max_running_s",
        type=_positive_running_seconds,
        default=None if evaluation else 60.0,
        required=evaluation,
    )
    parser.add_argument("--task-label", default=None)
    parser.add_argument("--operator", default=None, help="optional metadata only")


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        description=(
            "Inspect, check, shadow, run, or formally evaluate one DexMani "
            "Policy experiment"
        )
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

    check_parser = subcommands.add_parser(
        "check", help="strictly restore and smoke-test one Policy experiment"
    )
    check_parser.add_argument("experiment", metavar="EXPERIMENT")
    _add_device_option(check_parser)

    for command, help_text in (
        ("shadow", "run with full validation and no actuator publication"),
        ("run", "run with physical coupled arm/hand publication"),
    ):
        command_parser = subcommands.add_parser(command, help=help_text)
        command_parser.add_argument("experiment", metavar="EXPERIMENT")
        _add_device_option(command_parser)
        _add_lifecycle_options(command_parser, physical=command == "run")

    eval_parser = subcommands.add_parser(
        "eval",
        help="run one formally recorded physical policy evaluation",
    )
    eval_parser.add_argument("experiment", metavar="EXPERIMENT")
    _add_device_option(eval_parser)
    _add_lifecycle_options(eval_parser, evaluation=True)
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
    print(f"Chunk size  : {spec.chunk_size}")
    print(f"Query steps : {spec.n_action_steps}")
    print(
        f"Future tail : {spec.chunk_size - spec.n_action_steps} steps / {(spec.chunk_size - spec.n_action_steps) * spec.control_dt_s * 1000:g} ms"
    )
    print(f"Control     : {1.0 / spec.control_dt_s:g} Hz")
    print(
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
    )
    sys.stdout.flush()


def _print_evaluation_summary(info: Any, inputs: _LifecycleInputs) -> None:
    """Print recorded-rollout facts after preflight has fixed their values."""
    evaluation = inputs.evaluation_config
    if evaluation is None:
        raise ValueError("recorded rollout summary requires a recording contract")
    print("── Recorded Rollout ──")
    print(f"Eval seed             : {inputs.worker_config.seed}")
    print(f"Policy selector       : {info.selector}")
    print(f"Checkpoint            : {info.checkpoint_name}")
    print(f"Device                : {inputs.worker_config.device}")
    print(
        "Observation modalities : "
        + " + ".join(field.name for field in info.spec.observation_fields)
    )
    print(
        f"Action representation : {info.spec.action_key} ({info.spec.control_action_dim}D)"
    )
    print(f"n_action_steps        : {info.spec.n_action_steps}")
    print(f"Control Hz            : {1.0 / info.spec.control_dt_s:g}")
    print(f"Max running seconds   : {evaluation.max_running_s:g}")
    print(f"Rollout output dir    : {evaluation.data_dir}")
    print(f"Task                  : {evaluation.task_label}")
    print(f"Operator              : {evaluation.operator}")
    print("──────────────────────")
    sys.stdout.flush()


def _validated_warmup_durations(raw: Any) -> tuple[float, ...]:
    """Validate the public warmup timings before reporting them."""
    if not isinstance(raw, tuple) or not raw:
        raise RuntimeError("Policy warmup returned no timing samples")
    durations = tuple(float(value) for value in raw)
    if any(not math.isfinite(value) or value < 0.0 for value in durations):
        raise RuntimeError("Policy warmup returned invalid timing samples")
    return durations


def _safe_selector_parts(selector: Any) -> tuple[str, str, str]:
    """Accept only the canonical policy/task/experiment selector shape."""
    if not isinstance(selector, str):
        raise ValueError("Policy selector must be a string")
    parts = tuple(selector.split("/"))
    if len(parts) != 3 or any(
        not part
        or part in {".", ".."}
        or "/" in part
        or "\\" in part
        or Path(part).name != part
        for part in parts
    ):
        raise ValueError(
            "eval requires canonical selector policy/task/experiment"
        )
    return parts[0], parts[1], parts[2]


def _check_action_chunk(policy: Any, spec: Any) -> None:
    """Smoke-test the production NumPy boundary without starting hardware."""
    import numpy as np

    observation = {
        field.name: np.zeros((spec.n_obs_steps, *field.shape), dtype=field.dtype)
        for field in spec.observation_fields
    }
    policy.reset_episode()
    actions = policy.predict_action_chunk(observation)
    if (
        not isinstance(actions, np.ndarray)
        or actions.shape != (spec.chunk_size, spec.control_action_dim)
        or actions.dtype != np.float64
        or not np.all(np.isfinite(actions))
    ):
        raise RuntimeError(
            "Policy action chunk violates finite float64 [chunk_size, D] contract"
        )


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
        # ``LoadedPolicy.warmup`` checks the model's deterministic path; the
        # explicit smoke test below checks the production full-chunk API.
        durations = _validated_warmup_durations(policy.warmup(samples=3))
        print("warmup ........... OK")
        _check_action_chunk(policy, info.spec)
        print("action chunk ..... OK")
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
    args: argparse.Namespace,
    info: Any,
    *,
    execute: bool,
    evaluation: bool = False,
) -> _LifecycleInputs:
    """Resolve CLI-owned inputs before lifecycle compatibility validation."""
    from dexmani_real.config.experiment import resolve_experiment_config
    from dexmani_real.deployment.config import (
        InferenceWorkerConfig,
        RolloutRecordingConfig,
    )

    if not isinstance(execute, bool):
        raise TypeError("execute must be a boolean")
    runtime = resolve_experiment_config()
    worker_config = InferenceWorkerConfig(
        experiment=info.selector,
        device=args.device,
        spec=info.spec,
        seed=args.eval_seed,
        artifact=info.checkpoint_name,
        inference_steps=info.spec.default_inference_steps,
    )
    evaluation_config = None
    if execute:
        import getpass

        mode = "eval" if evaluation else "run"
        output_dir = (
            Path(__file__).resolve().parents[1]
            / "rollouts"
            / Path(*_safe_selector_parts(info.selector))
            / mode
            / f"seed_{args.eval_seed:03d}"
        )
        evaluation_config = RolloutRecordingConfig(
            data_dir=str(output_dir),
            task_label=args.task_label or info.task_name,
            operator=args.operator or getpass.getuser(),
            max_running_s=args.max_running_s,
        )
    return _LifecycleInputs(
        execute=execute,
        runtime=runtime,
        policy_spec=info.spec,
        worker_config=worker_config,
        evaluation_config=evaluation_config,
    )


def _start_lifecycle(inputs: _LifecycleInputs) -> int:
    """Enter the lifecycle that owns Real/Policy compatibility validation."""
    from dexmani_real.deployment.lifecycle import run_policy_deployment

    return run_policy_deployment(
        inputs.runtime,
        inputs.policy_spec,
        inputs.worker_config,
        inputs.execute,
        max_running_s=(
            None
            if inputs.evaluation_config is None
            else inputs.evaluation_config.max_running_s
        ),
        recording_config=inputs.evaluation_config,
    )


def _run_lifecycle(
    args: argparse.Namespace,
    *,
    execute: bool,
    evaluation: bool = False,
) -> int:
    try:
        info = _inspect_policy_experiment(args.experiment)
    except Exception as exc:
        _print_policy_error(f"experiment inspection failed: {exc}")
        return 1
    _print_experiment_summary(
        info,
        mode="EVAL" if evaluation else "RUN" if execute else "SHADOW",
        device=args.device,
    )
    try:
        inputs = _prepare_lifecycle_inputs(
            args,
            info,
            execute=execute,
            evaluation=evaluation,
        )
    except Exception as exc:
        _print_compatibility_error(f"runtime projection failed: {exc}")
        return 1
    if execute:
        _print_evaluation_summary(info, inputs)
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


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "list": _run_list,
        "check": _run_check,
        "shadow": lambda parsed: _run_lifecycle(parsed, execute=False),
        "run": lambda parsed: _run_lifecycle(parsed, execute=True),
        "eval": lambda parsed: _run_lifecycle(
            parsed,
            execute=True,
            evaluation=True,
        ),
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
