#!/usr/bin/env python3
"""Research-facing entry point for one recorded DexMani Policy evaluation session.

``python examples/run_policy.py EXPERIMENT [--artifact A] [--inference-steps N]
[--seed S] [--num-episodes N] [--max-duration SEC] [--device D]`` runs one
persistent multi-episode physical session (H -> scene setup -> B -> S).
Required recording failure ends the invalid evaluation with a nonzero result.
Before any hardware starts, run_config.yaml records the full resolved runtime,
public policy specification, artifact and session settings. Task success remains
an independent offline judgment.

Inference is synchronous: each query supplies PolicySpec.n_action_steps actions,
dispatched in order at a cadence anchored to actual publication. The next query
starts one control period after the previous publish; the new chunk's first
action publishes immediately after inference, without an extra control period.
Late inference never triggers catch-up dispatch. Fixed-FPS video is for viewing;
use HDF5 timestamps for actual control timing.

The command always connects to hardware (the lifecycle starts the policy
child first and waits for its READY before any actuator or camera worker), so
do not run it outside a supervised robot session.
"""

from __future__ import annotations

import argparse
import getpass
import math
import sys
import subprocess
import time
from pathlib import Path
from typing import Any


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
    parser = argparse.ArgumentParser(
        description="Run one persistent recorded policy-evaluation session"
    )
    parser.add_argument("experiment", metavar="EXPERIMENT")
    parser.add_argument("--config", help="experiment YAML configuration")
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
        "--seed", type=_nonnegative_int, default=0, help="per-session inference seed"
    )
    parser.add_argument(
        "--num-episodes",
        dest="num_episodes",
        type=_positive_int,
        default=1,
        help=(
            "episodes to run; each truly begun episode counts once at its end, "
            "independently of whether its recording saved"
        ),
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
    n_action_steps: int,
    seed: int,
    device: str,
    num_episodes: int,
    max_duration_s: float,
    runtime: Any,
    info: Any,
) -> None:
    """Write the resolved experimental conditions before any worker starts."""
    import yaml

    from dexmani_real.config.experiment import config_as_dict

    root = Path(__file__).resolve().parents[1]
    def git_fact(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    payload = {
        "runtime": config_as_dict(runtime),
        "policy_spec": config_as_dict(info.spec),
        "checkpoint_path": str(info.checkpoint_path),
        "repository_head": git_fact("rev-parse", "HEAD"),
        "repository_status": git_fact("status", "--short"),
        "experiment": experiment,
        "artifact": artifact,
        "inference_steps": inference_steps,
        "n_action_steps": n_action_steps,
        "seed": seed,
        "device": device,
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
    print(f"Episodes to run  : {num_episodes} (saved episodes counted separately)")
    print(f"Max duration   : {max_running_s:g} s per episode")
    print(f"Device         : {device}")
    print(f"Observation    : {' + '.join(fields)}")
    print(f"Action         : {spec.action_key} ({spec.control_action_dim}D)")
    print(f"Control        : {1.0 / spec.control_dt_s:g} Hz")
    print(f"Action chunk   : {spec.n_action_steps} steps; infer when queue is empty")
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

        runtime = resolve_experiment_config(yaml_path=args.config)
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
            n_action_steps=info.spec.n_action_steps,
            seed=args.seed,
            device=args.device,
            num_episodes=args.num_episodes,
            max_duration_s=args.max_running_s,
            runtime=runtime,
            info=info,
        )
    except Exception as exc:
        # Fail before any hardware or model process can start.
        _print_compatibility_error(f"session setup failed: {exc}")
        return 1

    from dexmani_real.deployment.config import (
        PolicyRuntimeConfig,
        RolloutRecordingConfig,
    )

    worker_config = PolicyRuntimeConfig(
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
        from dexmani_real.deployment.session import run_policy_deployment

        result = run_policy_deployment(
            runtime,
            info.spec,
            worker_config,
            True,
            max_running_s=args.max_running_s,
            num_episodes=args.num_episodes,
            recording_config=recording_config,
        )
    except Exception as exc:
        _print_lifecycle_error(f"lifecycle failed: {exc}")
        result = 1

    from dexmani_real.utils.atomic_io import atomic_json_dump
    try:
        atomic_json_dump({
            "technical_status": "valid" if result == 0 else "invalid",
            "termination_reason": "session_finished" if result == 0 else "workflow_failed",
            "task_success": "unknown",
            "exit_code": result,
        }, session_dir / "session_result.json")
    except OSError as exc:
        _print_lifecycle_error(f"could not persist session result: {exc}")
        return 1
    return result


if __name__ == "__main__":
    raise SystemExit(main())
