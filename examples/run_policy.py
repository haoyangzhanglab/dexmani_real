#!/usr/bin/env python3
"""Research-facing entry point for one DexMani Policy experiment.

``list`` reads Policy selectors only; ``check`` restores a checkpoint and warms
inference without starting robot workers. ``shadow`` starts the Real hardware
lifecycle while disabling command publication; ``run`` can physically command
xArm7/XHand; ``eval`` is the formally recorded physical protocol. The command
line owns experiment selection and operator intent. Policy owns checkpoint
inspection and restore; Real owns validation and robot lifecycle. All imports
that can reach Policy, Torch, or Real runtime code remain inside their command
handlers so ``list`` stays a filesystem-only Policy operation.
"""

from __future__ import annotations

import argparse
import hashlib
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
    deployment_config: Any
    evaluation_config: Any | None = None


def _positive_action_steps(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _positive_running_seconds(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def _add_device_option(parser: argparse.ArgumentParser) -> None:
    """Add the inference-device option for a model-consuming command."""
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Policy inference device for check, shadow, run, or eval (default: cuda:0)",
    )


def _add_lifecycle_options(
    parser: argparse.ArgumentParser,
    *,
    inference_mode_default: str = "sync",
    include_max_action_steps: bool = True,
) -> None:
    """Add options consumed only by a Real deployment lifecycle."""
    parser.add_argument(
        "--runtime-config",
        dest="runtime_config",
        default=None,
        help="optional Real runtime YAML for shadow/run/eval",
    )
    parser.add_argument(
        "--inference-mode",
        choices=("sync", "async"),
        default=inference_mode_default,
        help=f"inference scheduling mode (default: {inference_mode_default})",
    )
    if include_max_action_steps:
        parser.add_argument(
            "--max-action-steps",
            type=_positive_action_steps,
            default=None,
            help="episode action-step limit (default: unlimited)",
        )


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
        _add_lifecycle_options(command_parser)

    eval_parser = subcommands.add_parser(
        "eval",
        help="run one formally recorded physical policy evaluation",
    )
    eval_parser.add_argument("experiment", metavar="EXPERIMENT")
    _add_device_option(eval_parser)
    _add_lifecycle_options(
        eval_parser,
        inference_mode_default="async",
        include_max_action_steps=False,
    )
    eval_parser.add_argument(
        "--max-running-s",
        type=_positive_running_seconds,
        required=True,
        help="required wall-clock trial timeout in seconds",
    )
    eval_parser.add_argument(
        "--task-label",
        required=True,
        help="non-empty task label persisted in the episode metadata",
    )
    eval_parser.add_argument(
        "--operator",
        required=True,
        help="non-empty operator name persisted in the episode metadata",
    )
    eval_parser.add_argument(
        "--output-dir",
        default=None,
        help="isolated eval output directory (default: evaluations/<selector>)",
    )
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


def _print_evaluation_summary(info: Any, inputs: _LifecycleInputs) -> None:
    """Print formal-protocol facts after preflight has fixed their values."""
    evaluation = inputs.evaluation_config
    if evaluation is None:
        raise ValueError("formal evaluation summary requires an evaluation contract")
    checkpoint_sha256 = evaluation.provenance.get("checkpoint_sha256")
    if checkpoint_sha256 is None:
        raise ValueError("formal evaluation provenance lacks checkpoint_sha256")
    print("── Formal Evaluation ──")
    print(f"Policy selector       : {info.selector}")
    print(f"Checkpoint            : {info.checkpoint_name}")
    print(f"Checkpoint SHA-256    : {checkpoint_sha256}")
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
    print(f"Inference mode        : {inputs.deployment_config.inference_mode}")
    print(f"Max running seconds   : {evaluation.max_running_s:g}")
    print(f"Evaluation output dir : {evaluation.data_dir}")
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


def _checkpoint_sha256(checkpoint_path: Any) -> str:
    """Hash the exact checkpoint artifact before any worker can start."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise ValueError(f"checkpoint artifact is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        while chunk := checkpoint_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
            "formal eval requires canonical selector policy/task/experiment"
        )
    return parts[0], parts[1], parts[2]


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_evaluation_output_dir(
    raw_output_dir: str | None,
    *,
    selector: Any,
    episodes_dir: Any,
) -> Path:
    """Resolve a non-overlapping formal-evaluation output namespace."""
    repo_root = Path(__file__).resolve().parents[1]
    policy_name, task_name, experiment_name = _safe_selector_parts(selector)
    if raw_output_dir is None:
        output_dir = (
            repo_root / "evaluations" / policy_name / task_name / experiment_name
        )
    else:
        candidate = Path(raw_output_dir)
        output_dir = candidate if candidate.is_absolute() else repo_root / candidate
    output_dir = output_dir.resolve(strict=False)
    configured_episodes = Path(episodes_dir)
    episode_root = (
        configured_episodes
        if configured_episodes.is_absolute()
        else repo_root / configured_episodes
    ).resolve(strict=False)
    if _is_within(output_dir, episode_root) or _is_within(episode_root, output_dir):
        raise ValueError(
            "formal eval output must not contain or be contained by the demonstration episodes directory"
        )
    if output_dir == output_dir.parent:
        raise ValueError("formal eval output must not be a filesystem root")
    return output_dir


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
        PolicyDeploymentConfig,
    )
    from dexmani_real.deployment.evaluation import PolicyEvaluationConfig

    if not isinstance(execute, bool):
        raise TypeError("execute must be a boolean")
    runtime = resolve_experiment_config(yaml_path=args.runtime_config)
    deployment_config = PolicyDeploymentConfig(
        inference_mode=args.inference_mode,
        max_action_steps=getattr(args, "max_action_steps", None),
    )
    worker_config = InferenceWorkerConfig(
        experiment=info.selector,
        device=args.device,
        spec=info.spec,
    )
    evaluation_config = None
    if evaluation:
        if not execute:
            raise ValueError("formal policy evaluation requires physical execution")
        output_dir = _resolve_evaluation_output_dir(
            args.output_dir,
            selector=info.selector,
            episodes_dir=runtime.policy.episodes_dir,
        )
        checkpoint_sha256 = _checkpoint_sha256(info.checkpoint_path)
        evaluation_config = PolicyEvaluationConfig(
            data_dir=str(output_dir),
            task_label=args.task_label,
            operator=args.operator,
            max_running_s=args.max_running_s,
            provenance={
                "workflow": "policy_eval",
                "policy_selector": info.selector,
                "checkpoint_name": str(info.checkpoint_name),
                "checkpoint_sha256": checkpoint_sha256,
                "inference_mode": args.inference_mode,
                "max_running_s": f"{float(args.max_running_s):.17g}",
            },
        )
    return _LifecycleInputs(
        execute=execute,
        runtime=runtime,
        policy_spec=info.spec,
        worker_config=worker_config,
        deployment_config=deployment_config,
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
        deployment_config=inputs.deployment_config,
        max_running_s=(
            None
            if inputs.evaluation_config is None
            else inputs.evaluation_config.max_running_s
        ),
        evaluation_config=inputs.evaluation_config,
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
    if evaluation:
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
