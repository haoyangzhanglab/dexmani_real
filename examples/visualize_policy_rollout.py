#!/usr/bin/env python3
"""Offline prediction-to-execution debugging of a published rollout episode.

Saved predictions, terminal decisions, and raw state/commands/RGB-D/contact are
exact recorded evidence. Point clouds use current production reconstruction;
EEF/fingertips and joint-policy Cartesian proposals are FK-derived geometry.
This viewer does not reconstruct exact PolicyObservation tensors or load a model.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dexmani_real.deployment.policy_trace import load_trace, trace_path_for_episode
from dexmani_real.recording.storage.reader import EpisodeReader
from dexmani_real.planning.kinematics.arm_fk import compute_eef_pose_history_xarm_base


def action_key(action_dim: int) -> str:
    """Interpret the two current Real action contracts, without loading Policy."""
    if action_dim == 19:
        return "action"
    if action_dim == 21:
        return "action_ee"
    raise ValueError(f"unsupported rollout action dimension: {action_dim}")


def prediction_xyz(actions: np.ndarray) -> np.ndarray:
    """Return xArm-base proposals; joint outputs use the canonical offline FK."""
    if actions.ndim != 3 or not np.all(np.isfinite(actions)):
        raise ValueError("prediction actions must be finite [P,H,A]")
    key = action_key(actions.shape[2])
    if key == "action_ee":
        return actions[:, :, :3].copy()
    if actions.shape[0] == 0:
        return np.empty((*actions.shape[:2], 3), dtype=np.float64)
    poses = compute_eef_pose_history_xarm_base(actions[:, :, :7].reshape(-1, 7))
    return poses[:, :3].reshape(*actions.shape[:2], 3)


def scheduled_targets(trace: dict[str, np.ndarray], control_hz: float) -> np.ndarray:
    """Reconstruct logical targets, distinct from Executor control-slot due times."""
    if not np.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("control_hz must be finite and positive")
    step_dt_ns = int(round((1.0 / control_hz) * 1e9))
    if step_dt_ns <= 0:
        raise ValueError("control_hz produces a non-positive step interval")
    logical = dict(
        zip(
            map(int, trace["prediction_sequence"]),
            map(int, trace["prediction_logical_step_monotonic_ns"]),
        )
    )
    targets = [
        logical[int(sequence)] + int(index) * step_dt_ns
        for sequence, index in zip(
            trace["decision_prediction_sequence"],
            trace["decision_chunk_index"],
            strict=True,
        )
    ]
    if any(value > np.iinfo(np.uint64).max for value in targets):
        raise ValueError("reconstructed scheduled target exceeds uint64")
    return np.asarray(targets, dtype=np.uint64)


def load_rollout(episode: Path) -> tuple[dict[str, np.ndarray], float, np.ndarray]:
    trace = load_trace(trace_path_for_episode(episode))
    action_key(int(trace["action_dim"]))
    with EpisodeReader(episode) as reader:
        hz = reader.timing.rate_hz
        timestamps = np.asarray(reader.h5f["timestamp"][:], dtype=np.float64)
    if timestamps.size == 0 or not np.all(np.isfinite(timestamps)):
        raise ValueError("rollout must contain finite raw timestamps")
    if np.any(timestamps <= 0) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("raw rollout timestamps must be positive and increasing")
    scheduled_targets(trace, hz)
    return trace, hz, timestamps


def print_info(
    episode: Path, trace: dict[str, np.ndarray], hz: float, frames: int
) -> None:
    print(f"Episode: {episode}")
    print(f"Trace: {trace_path_for_episode(episode)}")
    print(f"Raw frames: {frames}; control_hz: {hz:g}")
    print(f"Predictions: {len(trace['prediction_sequence'])}")
    print(f"Decisions: {len(trace['decision_prediction_sequence'])}")
    print(
        f"Actions: {trace['actions'].shape}; action_key: {action_key(int(trace['action_dim']))}"
    )
    print(
        f"Fully stale at ingest: {np.count_nonzero(trace['prediction_first_index'] == -1)}"
    )
    for status, name in ((0, "published"), (2, "IK rejected"), (3, "safety rejected")):
        print(
            f"  {status} {name}: {np.count_nonzero(trace['decision_frame_status'] == status)}"
        )
    print(
        "Exact: saved predictions/timing/decisions and raw state/commands/RGB-D/contact."
    )
    print(
        "Reconstructed: current production point cloud and FK geometry; not exact PolicyObservation."
    )


def visualize(episode: Path, trace: dict[str, np.ndarray], hz: float, args) -> None:
    # GUI dependencies and raw visualizer initialization are absent from --info.
    import rerun as rr
    import rerun.blueprint as rrb
    from visualize_episode import EpisodeVisualizer
    from dexmani_real.config.experiment import resolve_experiment_config

    class RolloutVisualizer(EpisodeVisualizer):
        def _build_blueprint(self):
            joint_views = []
            for group, count in (("arm", 7), ("hand", 12)):
                joint_views.append(
                    rrb.Vertical(
                        name=group.title(),
                        contents=[
                            rrb.TimeSeriesView(
                                origin=f"comparison/{group}/{i}",
                                name=f"{group} joint {i} (rad)",
                            )
                            for i in range(count)
                        ],
                    )
                )
            return rrb.Blueprint(
                rrb.Horizontal(
                    contents=[
                        rrb.Vertical(
                            name="Camera",
                            contents=[
                                rrb.Spatial2DView(
                                    origin="camera/color/rgb", name="Raw RGB"
                                ),
                                rrb.Spatial2DView(
                                    origin="depth/image", name="Raw Depth"
                                ),
                            ],
                        ),
                        rrb.Spatial3DView(
                            origin="/",
                            name="3D Scene — reconstructed cloud / FK",
                            background=[0.12, 0.12, 0.14],
                        ),
                        rrb.Tabs(
                            name="Action/State",
                            active_tab=0,
                            contents=[
                                rrb.TimeSeriesView(
                                    origin="comparison/error_norm",
                                    name="Tracking error norms (rad)",
                                ),
                                *joint_views,
                                rrb.TimeSeriesView(
                                    origin="state/hand_contact_mag", name="Raw contact"
                                ),
                            ],
                        ),
                        rrb.Vertical(
                            name="Policy Timing",
                            contents=[
                                rrb.TimeSeriesView(
                                    origin="policy/timing",
                                    name="Prediction latency / age / skew (ms)",
                                ),
                                rrb.TimeSeriesView(
                                    origin="policy/arrivals",
                                    name="Prediction arrivals / first index",
                                ),
                                rrb.TimeSeriesView(
                                    origin="policy/decisions", name="Terminal decisions"
                                ),
                                rrb.TimeSeriesView(
                                    origin="policy/lateness",
                                    name="Terminal timing (ms)",
                                ),
                            ],
                        ),
                    ]
                )
            )

        def _log_static(self):
            super()._log_static()
            rr.log(
                "policy/timing/scheduler_tick_minus_commit_ms",
                rr.SeriesLine(name="Scheduler tick − ring commit (signed ms)"),
                static=True,
            )
            for group, count in (("arm", 7), ("hand", 12)):
                rr.log(
                    f"comparison/error_norm/{group}",
                    rr.SeriesLine(name=group),
                    static=True,
                )
                for i in range(count):
                    for kind, color in (
                        ("actual", [80, 180, 255]),
                        ("submitted", [255, 170, 60]),
                    ):
                        rr.log(
                            f"comparison/{group}/{i}/{kind}",
                            rr.SeriesLine(name=kind, color=color),
                            static=True,
                        )

        def _log_time_series(self, step_idx):
            super()._log_time_series(step_idx)
            for group, command in (
                ("arm", "action_arm_joint_sent"),
                ("hand", "action_hand_joint"),
            ):
                actual = self._state[f"{group}_qpos"][step_idx]
                submitted = self._state[command][step_idx]
                rr.log(
                    f"comparison/error_norm/{group}",
                    rr.Scalar(float(np.linalg.norm(actual - submitted))),
                )
                for i, (state, target) in enumerate(
                    zip(actual, submitted, strict=True)
                ):
                    rr.log(f"comparison/{group}/{i}/actual", rr.Scalar(float(state)))
                    rr.log(
                        f"comparison/{group}/{i}/submitted", rr.Scalar(float(target))
                    )

        def log_predictions_and_decisions(self):
            xyz = prediction_xyz(trace["actions"])
            rows = {int(seq): i for i, seq in enumerate(trace["prediction_sequence"])}
            targets = scheduled_targets(trace, hz)
            start, end = self._state["timestamp"][[0, -1]]

            def set_event_time(ns):
                seconds = int(ns) / 1e9
                if args.max_frames is not None and not start <= seconds <= end:
                    return False
                rr.set_time_seconds("time", seconds)
                return True

            for i, ingest in enumerate(trace["prediction_ingest_monotonic_ns"]):
                if not set_event_time(ingest):
                    continue
                first = int(trace["prediction_first_index"][i])
                label = "fully stale proposal" if first == -1 else "saved proposal"
                rr.log(
                    "policy/proposal",
                    rr.Points3D(
                        xyz[i], radii=0.007, colors=[80, 200, 255], labels=[label]
                    ),
                )
                for metric in (
                    "inference_latency_ms",
                    "observation_age_ms",
                    "observation_skew_ms",
                ):
                    rr.log(
                        f"policy/timing/{metric}",
                        rr.Scalar(float(trace[f"prediction_{metric}"][i])),
                    )
                # The scheduler tick can precede a concurrently committed chunk;
                # this signed clock difference is not a transport latency.
                tick_minus_commit_ms = (
                    int(ingest) - int(trace["prediction_publish_monotonic_ns"][i])
                ) / 1e6
                rr.log(
                    "policy/timing/scheduler_tick_minus_commit_ms",
                    rr.Scalar(tick_minus_commit_ms),
                )
                rr.log(
                    "policy/arrivals/sequence",
                    rr.Scalar(int(trace["prediction_sequence"][i])),
                )
                rr.log("policy/arrivals/first_index", rr.Scalar(first))

            colors = {0: [60, 255, 100], 2: [255, 100, 180], 3: [255, 70, 50]}
            labels = {
                0: "published policy target",
                2: "IK rejected policy target",
                3: "safety rejected policy target",
            }
            for i, event in enumerate(trace["decision_event_monotonic_ns"]):
                if not set_event_time(event):
                    continue
                row = rows[int(trace["decision_prediction_sequence"][i])]
                index = int(trace["decision_chunk_index"][i])
                status = int(trace["decision_frame_status"][i])
                rr.log(
                    "policy/terminal_target",
                    rr.Points3D(
                        xyz[row, index][None, :],
                        radii=0.018,
                        colors=colors[status],
                        labels=[labels[status]],
                    ),
                )
                rr.log("policy/decisions/chunk_index", rr.Scalar(index))
                rr.log("policy/decisions/frame_status", rr.Scalar(status))
                rr.log(
                    "policy/lateness/schedule_lateness_ms",
                    rr.Scalar(
                        (int(event) - int(trace["decision_due_monotonic_ns"][i])) / 1e6
                    ),
                )
                rr.log(
                    "policy/lateness/logical_target_offset_ms",
                    rr.Scalar((int(event) - int(targets[i])) / 1e6),
                )

    runtime = resolve_experiment_config() if args.point_cloud else None
    cloud = runtime.pointcloud if runtime is not None else None
    if args.pointcloud_num_points is not None:
        cloud = replace(cloud, num_points=args.pointcloud_num_points)
    table = runtime.environment.table if runtime is not None else None
    viz = RolloutVisualizer(
        str(episode),
        max_frames=args.max_frames,
        point_cloud=args.point_cloud,
        pointcloud_config=cloud,
        table_plane_abcd=(
            table.plane_abcd if table is not None and table.enabled else None
        ),
    )
    try:
        # Policy events never inherit a raw row's secondary step timeline.
        viz.log_predictions_and_decisions()
        for step in range(viz.num_steps):
            viz.log_step(step)
    finally:
        viz.close()


def main(argv: list[str] | None = None) -> int:
    from dexmani_real.ipc.schema import SUPPORTED_POINT_CLOUD_COUNTS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "episode", help="Published rollout directory (sibling trace is automatic)."
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Validate and summarize without opening Rerun.",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--point-cloud", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--pointcloud-num-points",
        type=int,
        choices=sorted(SUPPORTED_POINT_CLOUD_COUNTS),
    )
    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if args.pointcloud_num_points is not None and not args.point_cloud:
        parser.error("--pointcloud-num-points requires --point-cloud")
    episode = Path(args.episode).expanduser().resolve()
    try:
        trace, hz, timestamps = load_rollout(episode)
        if args.info:
            print_info(episode, trace, hz, len(timestamps))
        else:
            visualize(episode, trace, hz, args)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"rollout visualization: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
