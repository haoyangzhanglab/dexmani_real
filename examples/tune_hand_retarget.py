#!/usr/bin/env python3
"""Thin CLI for deterministic offline TAG/DexPilot evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

from dexmani_real.config.defaults import dexpilot_retargeting, tag_retargeting
from dexmani_real.teleop.hand_retarget_eval import (
    estimate_home_qpos,
    evaluate_default_backends,
    load_episode_hand_data,
    pareto_front,
    passes_default_gates,
    search_retarget_configs,
)


def _route_project_logs_to_stderr() -> None:
    """Keep stdout machine-readable while preserving evaluator diagnostics."""

    for logger_name, value in logging.root.manager.loggerDict.items():
        if not logger_name.startswith("dexmani_real") or not isinstance(
            value, logging.Logger
        ):
            continue
        for handler in value.handlers:
            if (
                isinstance(handler, logging.StreamHandler)
                and handler.stream is sys.stdout
            ):
                handler.setStream(sys.stderr)


def main() -> int:
    _route_project_logs_to_stderr()
    parser = argparse.ArgumentParser(
        description="Offline TAG/DexPilot retarget evaluation"
    )
    parser.add_argument("episode", type=Path, help="Schema-v16 episode directory")
    parser.add_argument(
        "--search",
        action="store_true",
        help="Run the bounded dual-backend parameter search",
    )
    parser.add_argument(
        "--top", type=int, default=10, help="Maximum ranked candidates to print"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional JSON report path"
    )
    args = parser.parse_args()
    if args.top <= 0:
        parser.error("--top must be positive")

    data = load_episode_hand_data(args.episode)
    metrics = (
        search_retarget_configs(data)
        if args.search
        else evaluate_default_backends(data)
    )
    front = pareto_front(metrics)
    recommended = metrics[0]
    if recommended.backend == "tag":
        home_config = replace(
            tag_retargeting,
            smooth_weight=recommended.parameters["smooth_weight"],
            pinch_start_dist_m=recommended.parameters["pinch_start_dist_m"],
            pinch_full_dist_m=recommended.parameters["pinch_full_dist_m"],
        )
        home_estimate = estimate_home_qpos(data, recommended.backend, home_config)
    else:
        dexpilot_home_config = replace(
            dexpilot_retargeting,
            scaling_factor=recommended.parameters["scaling_factor"],
            low_pass_alpha=recommended.parameters["low_pass_alpha"],
            project_dist_m=recommended.parameters["project_dist_m"],
            escape_dist_m=recommended.parameters["escape_dist_m"],
        )
        home_estimate = estimate_home_qpos(
            data, recommended.backend, dexpilot_home_config
        )
    report = {
        "episode": data.path,
        "control_hz": data.control_hz,
        "frame_count": len(data.landmarks),
        "search": bool(args.search),
        "candidate_count": len(metrics),
        "pareto_count": len(front),
        "recommended": {
            **recommended.to_dict(),
            "passes_gates": passes_default_gates(recommended),
            "home_estimate": home_estimate.to_dict(),
        },
        "ranked": [
            {**item.to_dict(), "passes_gates": passes_default_gates(item)}
            for item in metrics[: args.top]
        ],
        "pareto_front": [
            {**item.to_dict(), "passes_gates": passes_default_gates(item)}
            for item in front[: args.top]
        ],
    }
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
