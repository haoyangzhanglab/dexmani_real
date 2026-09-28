#!/usr/bin/env python3
"""Offline multi-seed predictions of the exact observations saved by --diagnostics."""

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import yaml

from dexmani_real.deployment.config import with_action_steps
from dexmani_real.deployment.diagnostic_analysis import compare_chunks, joint_delta, statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()
    if len(set(args.seeds)) < 2 or min(args.seeds) < 0:
        parser.error("provide at least two distinct nonnegative seeds")
    import torch
    from dexmani_policy.deployment import inspect_policy, load_experiment_config, load_policy

    run = yaml.safe_load((args.session / "run_config.yaml").read_text())
    saved = run["policy"]
    if saved["action_mode"] != "joint":
        parser.error("this joint-continuity probe requires action_mode=joint")
    config = with_action_steps(
        load_experiment_config(saved["experiment_dir"]), saved["n_action_steps"]
    )
    info = inspect_policy(
        saved["experiment_dir"],
        config=config,
        checkpoint=saved["checkpoint_path"],
        weights=saved["weights"],
        inference_steps=saved["inference_steps"],
    )
    with h5py.File(args.session / "diagnostics" / "policy.h5", "r") as f:
        if not f.attrs.get("complete", False):
            parser.error("policy trace is incomplete")
        q = {k: v[:] for k, v in f["queries"].items()}
    observations = [
        {name: q[f"obs_{name}"][i] for name in info.observation_fields}
        for i in range(len(q["chunk_id"]))
    ]
    model = load_policy(config, info, device=args.device, seed=int(run["seed"]))

    def predict(observation, seed):
        model.reset_episode()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return model.predict_with_diagnostics(observation)["pred_action"]

    try:
        model.warmup(samples=5)
        repeat = predict(observations[0], args.seeds[0])
        np.testing.assert_array_equal(repeat, predict(observations[0], args.seeds[0]))
        samples = np.stack(
            [np.stack([predict(obs, seed) for obs in observations]) for seed in args.seeds]
        )
        # Reproduce the original per-episode RNG progression, not a reset at each query.
        model.reset_episode()
        sequential = np.stack(
            [model.predict_with_diagnostics(obs)["pred_action"] for obs in observations]
        )
    finally:
        model.close()
    start = info.n_obs_steps - 1
    # Align periodic branches before measuring dispersion, without changing saved predictions.
    aligned = samples[0] + joint_delta(samples, samples[0])
    spread = np.rad2deg(np.std(aligned[:, :, start, :7], axis=0)).max(axis=1)
    span = np.rad2deg(np.ptp(aligned[:, :, start, :7], axis=0)).max(axis=1)
    fixed_seed = []
    for seed, predictions in zip(args.seeds, samples):
        revisions, boundaries, planned = [], [], []
        for i in range(1, len(predictions)):
            if q["run_id"][i] != q["run_id"][i - 1] or q["chunk_id"][i] != q["chunk_id"][i - 1] + 1:
                continue
            delta = compare_chunks(
                predictions[i - 1],
                predictions[i],
                n_obs_steps=info.n_obs_steps,
                n_action_steps=info.n_action_steps,
            )
            revisions.append(float(np.rad2deg(abs(delta["revision"][:7])).max()))
            boundaries.append(float(np.rad2deg(abs(delta["boundary"][:7])).max()))
            planned.append(float(np.rad2deg(abs(delta["planned_step"][:7])).max()))
        fixed_seed.append(
            {
                "seed": seed,
                "revision_max_joint_deg": statistics(revisions),
                "boundary_max_joint_deg": statistics(boundaries),
                "planned_max_joint_deg": statistics(planned),
            }
        )
    report = {
        "original_sequence_replay_exact": bool(np.array_equal(sequential, q["pred_action"])),
        "original_sequence_replay_max_error_rad": float(
            np.max(np.abs(sequential - q["pred_action"]))
        ),
        "same_input_same_seed_exact_repeat": True,
        "seeds": args.seeds,
        "same_input_first_action_max_joint_std_deg": statistics(spread),
        "same_input_first_action_max_joint_range_deg": statistics(span),
        "fixed_seed_across_recorded_observations": fixed_seed,
        "limitation": "Offline conditional predictions, not a counterfactual closed-loop rollout. Seeds affect flow and any stochastic encoder sampling. Logical overlap does not remove wall-clock drift.",
    }
    out = args.session / "diagnostics" / "analysis"
    out.mkdir(exist_ok=True)
    np.savez_compressed(
        out / "seed_predictions.npz",
        seeds=args.seeds,
        predictions=samples,
        chunk_id=q["chunk_id"],
        run_id=q["run_id"],
    )
    (out / "seed_probe.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
