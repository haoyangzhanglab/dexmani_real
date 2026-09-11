#!/usr/bin/env python3
"""Research-facing entry point for one recorded DexMani Policy evaluation session.

``python examples/run_policy.py EXPERIMENT [--artifact A] [--inference-steps N]
[--seed S] [--num-episodes N] [--max-duration SEC] [--device D]`` runs one
persistent multi-episode physical session (H -> scene setup -> B -> S per
episode). The parent inspects and pins the deployment artifact, resolves the
effective inference steps, creates a session directory and writes
``run_config.yaml``, then hands a thin set of inputs to the deployment
lifecycle. Task success is judged offline from the published raw episodes; this
command records only technical stop reasons.

The command always connects to hardware (the lifecycle starts the inference
child first and waits for its READY before any actuator or camera worker), so
do not run it outside a supervised robot session.
"""

from __future__ import annotations

import argparse
import getpass
import math
import sys
import time
from pathlib import Path
from typing import Any, NoReturn


class _ArgumentParser(argparse.ArgumentParser):
    """Render command-line contract failures with their owner marker."""

    def error(self, message: str) -> NoReturn:
        self.exit(2, f"{self.prog}: error: [CLI] {message}\n")


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _nonnegative_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def _positive_running_seconds(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        description="Run one persistent recorded policy-evaluation session"
    )
    parser.add_argument("experiment", metavar="EXPERIMENT")
    parser.add_argument(
        "--artifact",
        default=None,
        help="deployment artifact filename in experiment/checkpoints/ "
        "(default: deployment_latest.pt)",
    )
    parser.add_argument(
        "--inference-steps",
        type=_positive_int,
        default=None,
        help="override the artifact's default inference steps (default: artifact default)",
    )
    parser.add_argument(
        "--seed", type=_nonnegative_int, default=0, help="per-episode inference seed"
    )
    parser.add_argument(
        "--num-episodes", type=_positive_int, default=1, help="episodes to save"
    )
    parser.add_argument(
        "--max-duration",
        dest="max_running_s",
        type=_positive_running_seconds,
        default=60.0,
        help="per-episode running-seconds budget after B (default: 60)",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


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
        raise ValueError("experiment must be a canonical policy/task/experiment selector")
    return parts[0], parts[1], parts[2]


def _inspect_policy_experiment(experiment: str, *, artifact: str | None) -> Any:
    """Inspect and pin metadata through Policy without constructing a model."""
    from dexmani_policy.deployment import inspect_experiment

    return inspect_experiment(experiment, artifact=artifact)


def _session_directory(root: Path, selector: str) -> Path:
    """Create and return ``rollouts/<policy>/<task>/<experiment>/session_<ts>``."""
    parts = _safe_selector_parts(selector)
    base = root / Path(*parts)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = base / f"session_{stamp}"
    dedup = 1
    while candidate.exists():
        candidate = base / f"session_{stamp}_{dedup:02d}"
        dedup += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _write_run_config(
    session_dir: Path,
    *,
    experiment: str,
    artifact: str,
    inference_steps: int,
    seed: int,
    num_episodes: int,
    max_duration_s: float,
) -> None:
    """Write the resolved experimental conditions before any worker starts."""
    import yaml

    payload = {
        "experiment": experiment,
        "artifact": artifact,
        "inference_steps": inference_steps,
        "seed": seed,
        "num_episodes": num_episodes,
        "max_duration_s": float(max_duration_s),
    }
    (session_dir / "run_config.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )


def _print_summary(
    info: Any,
    *,
    device: str,
    artifact: str,
    inference_steps: int,
    seed: int,
    num_episodes: int,
    max_running_s: float,
    session_dir: Path,
) -> None:
    """Print the small operator-facing resolved session facts."""
    spec = info.spec
    fields = tuple(field.name for field in spec.observation_fields)
    print("── Policy Evaluation Session ──")
    print(f"Experiment     : {info.selector}")
    print(f"Policy         : {info.policy_name}")
    print(f"Artifact       : {artifact}")
    print(f"Inference steps: {inference_steps}")
    print(f"Seed           : {seed}")
    print(f"Episodes       : {num_episodes}")
    print(f"Max duration   : {max_running_s:g} s per episode")
    print(f"Device         : {device}")
    print(f"Observation    : {' + '.join(fields)}")
    print(f"Action         : {spec.action_key} ({spec.control_action_dim}D)")
    print(f"Control        : {1.0 / spec.control_dt_s:g} Hz")
    print(f"Session dir    : {session_dir}")
    print("──────────────────────────────")
    sys.stdout.flush()


def _print_policy_error(message: str) -> None:
    print(f"[POLICY] {message}", file=sys.stderr)


def _print_compatibility_error(message: str) -> None:
    print(f"[COMPAT] {message}", file=sys.stderr)


def _print_lifecycle_error(message: str) -> None:
    print(f"[LIFECYCLE] {message}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    try:
        info = _inspect_policy_experiment(args.experiment, artifact=args.artifact)
    except Exception as exc:
        _print_policy_error(f"experiment inspection failed: {exc}")
        return 1
    inference_steps = (
        args.inference_steps
        if args.inference_steps is not None
        else info.spec.default_inference_steps
    )

    try:
        from dexmani_real.config.experiment import resolve_experiment_config

        runtime = resolve_experiment_config()
    except Exception as exc:
        _print_compatibility_error(f"runtime resolution failed: {exc}")
        return 1

    rollouts_root = Path(__file__).resolve().parents[1] / "rollouts"
    try:
        session_dir = _session_directory(rollouts_root, info.selector)
        _write_run_config(
            session_dir,
            experiment=info.selector,
            artifact=info.checkpoint_name,
            inference_steps=inference_steps,
            seed=args.seed,
            num_episodes=args.num_episodes,
            max_duration_s=args.max_running_s,
        )
    except Exception as exc:
        # Fail before any hardware or model process can start.
        _print_compatibility_error(f"session setup failed: {exc}")
        return 1

    from dexmani_real.deployment.config import (
        InferenceWorkerConfig,
        RolloutRecordingConfig,
    )

    worker_config = InferenceWorkerConfig(
        experiment=info.selector,
        device=args.device,
        spec=info.spec,
        seed=args.seed,
        artifact=info.checkpoint_name,
        inference_steps=inference_steps,
    )
    recording_config = RolloutRecordingConfig(
        data_dir=str(session_dir),
        task_label=info.task_name,
        operator=getpass.getuser(),
        max_running_s=args.max_running_s,
        num_episodes=args.num_episodes,
    )
    _print_summary(
        info,
        device=args.device,
        artifact=info.checkpoint_name,
        inference_steps=inference_steps,
        seed=args.seed,
        num_episodes=args.num_episodes,
        max_running_s=args.max_running_s,
        session_dir=session_dir,
    )

    try:
        from dexmani_real.deployment.lifecycle import run_policy_deployment

        return run_policy_deployment(
            runtime,
            info.spec,
            worker_config,
            True,
            max_running_s=args.max_running_s,
            recording_config=recording_config,
        )
    except Exception as exc:
        _print_lifecycle_error(f"lifecycle failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
