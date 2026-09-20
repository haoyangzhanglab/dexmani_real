#!/usr/bin/env python3
"""Offline debugging of legacy async rollouts with an existing policy trace.

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
from dexmani_real.robot.model import HAND_FINGER_NAMES
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
    """Reconstruct legacy logical targets separately from control-slot due times."""
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


def print_raw_only_notice(episode: Path) -> int:
    """Explain one trace-less synchronous raw rollout without calling it corrupt.

    The refactored synchronous pipeline publishes raw rollouts and no
    prediction-trace sidecar, so a missing sidecar is a normal rollout shape,
    not damaged data: report the raw basics and point to the raw viewer.
    """
    with EpisodeReader(episode) as reader:
        meta = reader.h5f["meta"].attrs
        print(f"Episode: {episode}")
        print(
            "Trace:   none — this is a synchronous-raw rollout (no legacy "
            "policy-trace sidecar); the raw data is intact, not corrupt"
        )
        print(
            f"Raw frames: {int(meta.get('num_frames', 0))}; "
            f"control_hz: {reader.timing.rate_hz:g}; "
            f"task: {str(meta.get('task_label', ''))}"
        )
        print(
            "This viewer requires a legacy trace sidecar. Inspect the raw "
            "episode with:\n"
            f"  python examples/visualize_episode.py {episode}"
        )
    return 0


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
        def _preload_state(self):
            # Legacy raw rows do not carry the old inference worker's observation
            # verdict. Show sample validity and camera freshness instead.
            self._available = {
                "arm": ["arm_qpos"],
                "hand": ["hand_qpos", "hand_contact"],
                "action": ["action_arm_joint_sent", "action_hand_joint"],
                "flags": [
                    "flag_frame_status",
                    "flag_action_queued",
                    "flag_sample_valid",
                    "flag_camera_fresh",
                    "arm_connected",
                    "hand_connected",
                    "hand_qpos_stale",
                    "hand_contact_valid",
                ],
                "camera": self._available.get("camera", []),
                "meta": ["timestamp"],
            }
            return super()._preload_state()

        def _build_blueprint(self):
            def series(origin, name, **kwargs):
                return rrb.TimeSeriesView(origin=origin, name=name, **kwargs)

            def flags(keys, name):
                return series("flags", name, contents=[f"flags/{key}" for key in keys])

            arm_views = [
                series(
                    f"comparison/arm/{i}", f"Arm joint {i}: actual / submitted (rad)"
                )
                for i in range(7)
            ]
            hand_views = []
            for finger, indices in (
                ("Thumb", range(0, 3)),
                ("Index", range(3, 6)),
                ("Middle", range(6, 8)),
                ("Ring", range(8, 10)),
                ("Pinky", range(10, 12)),
            ):
                hand_views.append(
                    rrb.Vertical(
                        name=finger,
                        contents=[
                            series(f"comparison/hand/{i}", f"{finger} joint {i} (rad)")
                            for i in indices
                        ],
                    )
                )
            return rrb.Blueprint(
                rrb.Vertical(
                    row_shares=[3, 2],
                    contents=[
                        rrb.Horizontal(
                            column_shares=[2, 3, 2],
                            contents=[
                                rrb.Tabs(
                                    name="Camera",
                                    active_tab=0,
                                    contents=[
                                        rrb.Spatial2DView(
                                            origin="camera/color/rgb", name="RGB"
                                        ),
                                        rrb.Spatial2DView(
                                            origin="depth/image", name="Depth"
                                        ),
                                    ],
                                ),
                                rrb.Spatial3DView(
                                    origin="/",
                                    name="Proposals / terminal target / FK / cloud",
                                    background=[0.12, 0.12, 0.14],
                                ),
                                rrb.Vertical(
                                    name="Execution overview",
                                    contents=[
                                        series(
                                            "comparison/error_norm",
                                            "Joint tracking error norms (rad)",
                                        ),
                                        series(
                                            "state/hand_contact_mag",
                                            "Five-finger contact magnitude (SDK-scaled)",
                                        ),
                                        series(
                                            "flags/flag_frame_status",
                                            "Frame: 0 OK / 1 held / 2 IK / 3 safety",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        rrb.Tabs(
                            name="Diagnostics",
                            active_tab=0,
                            contents=[
                                rrb.Grid(
                                    name="Arm joints",
                                    grid_columns=4,
                                    contents=arm_views,
                                ),
                                rrb.Horizontal(name="Hand joints", contents=hand_views),
                                rrb.Grid(
                                    name="Contact",
                                    grid_columns=3,
                                    contents=[
                                        *[
                                            series(
                                                f"contact/{finger}",
                                                f"{finger.title()} Fx / Fy / Fz (SDK-scaled)",
                                            )
                                            for finger in HAND_FINGER_NAMES
                                        ],
                                        flags(
                                            ["hand_contact_valid"],
                                            "Contact valid (not contact detected)",
                                        ),
                                    ],
                                ),
                                rrb.Grid(
                                    name="Status",
                                    grid_columns=2,
                                    contents=[
                                        flags(["flag_action_queued"], "Action queued"),
                                        flags(
                                            [
                                                "flag_sample_valid",
                                                "flag_camera_fresh",
                                            ],
                                            "Sample validity and camera freshness",
                                        ),
                                        flags(
                                            ["arm_connected", "hand_connected"],
                                            "Device connected",
                                        ),
                                        flags(
                                            ["hand_qpos_stale"], "Hand feedback stale"
                                        ),
                                    ],
                                ),
                                rrb.Grid(
                                    name="Policy timing",
                                    grid_columns=2,
                                    contents=[
                                        series(
                                            "policy/timing",
                                            "Prediction latency / age / skew (ms)",
                                        ),
                                        series(
                                            "policy/arrivals",
                                            "Prediction sequence / first usable index (-1: stale)",
                                        ),
                                        series(
                                            "policy/decisions",
                                            "Terminal chunk index / status (0 OK / 2 IK / 3 safety)",
                                        ),
                                        series(
                                            "policy/lateness", "Terminal timing (ms)"
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                auto_views=False,
            )

        def _log_static(self):
            super()._log_static()
            rr.log(
                "policy/timing/scheduler_tick_minus_commit_ms",
                rr.SeriesLine(name="Scheduler tick − ring commit (signed ms)"),
                static=True,
            )
            for finger in HAND_FINGER_NAMES:
                for axis, color in zip(
                    ("Fx", "Fy", "Fz"),
                    ([255, 90, 90], [90, 210, 90], [90, 150, 255]),
                    strict=True,
                ):
                    rr.log(
                        f"contact/{finger}/{axis}",
                        rr.SeriesLine(name=axis, color=color),
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
            for finger, force in zip(
                HAND_FINGER_NAMES,
                self._state["hand_contact"][step_idx],
                strict=True,
            ):
                for axis, value in zip(("Fx", "Fy", "Fz"), force, strict=True):
                    rr.log(f"contact/{finger}/{axis}", rr.Scalar(float(value)))
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
        if not trace_path_for_episode(episode).is_file():
            # No sidecar is a distinct, non-corrupt outcome; a sidecar that
            # exists but cannot be read still fails loudly below.
            return print_raw_only_notice(episode)
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
