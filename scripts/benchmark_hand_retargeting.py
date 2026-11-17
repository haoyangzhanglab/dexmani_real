"""固定 Raw、预算和 CPU 线程，交错测量两个 checkout 的真实手部求解。

计时不含导入、I/O、初始化或诊断；分项测量另行执行，不包装计时中的求解器。
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter_ns


def measure(root, episode, warmup):
    # 同一脚本可测没有此基准入口的旧 checkout；不保存第二份算法。
    sys.path.insert(0, str(root))
    import numpy as np

    from dexmani_real.data.raw import RawReader
    from dexmani_real.teleop.retargeting.landmarks import prepare_landmarks
    from dexmani_real.teleop.retargeting.objective import HandObjective
    from dexmani_real.teleop.retargeting.solver import HandRetargeter
    from scripts.preview_hand_retargeting import read_recorded_parameters

    def percentiles(values):
        return dict(zip(("p50", "p95"), np.percentile(values, [50, 95]).tolist())) if len(values) else None

    with RawReader(episode) as reader:
        config, limits, calibration = read_recorded_parameters(reader)
        points = reader["vr_landmarks"][:]
        measured = reader["hand_qpos"][:]
        times = reader.read_timing()["observation_ns"].copy()
        starts = set(reader.segment_starts)
    solver = HandRetargeter(config, limits)
    solver.calibration = calibration
    if not len(times) or not np.isfinite(measured).all():
        raise ValueError("benchmark requires nonempty finite measured joints")
    if not np.isfinite(points[[i for i in range(len(times)) if i not in starts]]).all():
        raise ValueError("benchmark landmarks contain nonfinite values outside segment starts")
    for _ in range(warmup + 1):
        elapsed, evaluations, rejected = [], [], []
        commands, references, intervals = [], [], []
        for index, timestamp in enumerate(times):
            timestamp = int(timestamp)
            if index in starts:
                solver.reset(measured[index], timestamp_ns=timestamp)
                commands.append(solver.reference_qpos.copy())
                references.append(None)
                intervals.append(None)
                continue
            references.append(solver.reference_qpos.copy())
            intervals.append((timestamp - solver.reference_timestamp_ns) * 1e-9)
            before = perf_counter_ns()
            command = solver.retarget(points[index], timestamp_ns=timestamp)
            elapsed.append((perf_counter_ns() - before) / 1000)
            if command is None:
                rejected.append([index, solver.rejection_reason])
            else:
                evaluations.append(solver.optimizer.get_numevals())
                solver.commit(command, timestamp_ns=timestamp)
            commands.append(solver.reference_qpos.copy())

    # 以下诊断不进入完整 retarget 耗时；同一帧的 warm start 服务各分项。
    components = {key: [] for key in (
        "targets_us", "objective_construction_us", "fk_us", "fk_jacobian_us",
        "objective_gradient_us", "objective_value_us",
    )}
    tips, directions, pinches = [], [], []
    rejected_rows = {row for row, _ in rejected}
    for index, qpos in enumerate(commands):
        if index in starts or index in rejected_rows:
            continue
        def targets():
            return calibration.targets(prepare_landmarks(points[index]), solver.geometry, config,
                                       fk=solver.fk, frame_ids=solver.frame_ids)
        target = targets()
        previous = references[index]
        def objective():
            return HandObjective(solver.fk, solver.frame_ids, target, previous, intervals[index], config)
        loss = objective()
        gradient, empty = np.empty(12), np.empty(0)
        for name, call in (
            ("targets_us", targets), ("objective_construction_us", objective),
            ("fk_us", lambda: solver.fk.forward(previous, solver.frame_ids)),
            ("fk_jacobian_us", lambda: solver.fk.positions_and_jacobians(previous, solver.frame_ids)),
            ("objective_gradient_us", lambda: loss(previous, gradient)),
            ("objective_value_us", lambda: loss(previous, empty)),
        ):
            before = perf_counter_ns()
            call()
            components[name].append((perf_counter_ns() - before) / 1000)
        actual = solver.fk.forward(qpos, solver.frame_ids)[:, :3, 3].reshape(5, 3, 3)
        tips.append(np.sqrt(np.mean(np.sum((actual[:, 2] - target.positions[:, 2]) ** 2, axis=-1))) * 1000)
        segments = np.diff(actual, axis=1)
        unit = segments / np.linalg.norm(segments, axis=-1, keepdims=True)
        directions.append(np.rad2deg(np.arccos(np.clip(np.sum(unit * target.directions, axis=-1), -1, 1))).mean())
        active = target.activation > 0
        if active.any():
            distance = np.linalg.norm(actual[1:, 2] - actual[0, 2], axis=-1)
            pinches.append(np.abs(distance[active] - target.pinch_distances[active]).mean() * 1000)

    revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    sources = ("teleop/retargeting/landmarks.py", "teleop/retargeting/objective.py",
               "teleop/retargeting/solver.py", "geometry/kinematics.py")
    from dataclasses import asdict
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "source_sha256": {name: hashlib.sha256((root / "dexmani_real" / name).read_bytes()).hexdigest() for name in sources},
        "config": asdict(config), "frames": len(times), "calls": len(elapsed),
        "retarget_us": percentiles(elapsed), "evaluations": percentiles(evaluations),
        "evaluations_total": sum(evaluations), "rejected": rejected,
        "components": {name: percentiles(values) for name, values in components.items()},
        "tip_rmse_mm": percentiles(tips), "direction_mean_deg": percentiles(directions),
        "active_pinch_mean_error_mm": percentiles(pinches), "active_pinch_frames": len(pinches),
        "commands": np.asarray(commands).tolist(),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pin", "nlopt")}},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--baseline", type=Path, help="基线 checkout；省略时只测当前代码")
    parser.add_argument("--candidate", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2, help="每个独立进程先完整回放的次数")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1:
        parser.error("repeats and warmup must be positive")
    episode = args.episode.expanduser().resolve()
    if args.worker:
        print(json.dumps(measure(args.candidate.resolve(), episode, args.warmup), allow_nan=False))
        return
    import numpy as np

    roots = {"candidate": args.candidate.expanduser().resolve()}
    if args.baseline:
        roots = {"baseline": args.baseline.expanduser().resolve(), **roots}
    runs = {name: [] for name in roots}
    commands = {}
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       PYTHONDONTWRITEBYTECODE="1")
    for index in range(args.repeats):
        order = list(roots) if index % 2 == 0 else list(reversed(roots))
        for name in order:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), str(episode), "--worker",
                 "--candidate", str(roots[name]), "--warmup", str(args.warmup)],
                env=environment, check=True, stdout=subprocess.PIPE, text=True,
            )
            run = json.loads(result.stdout)
            command = np.asarray(run.pop("commands"))
            if name in commands:
                np.testing.assert_allclose(command, commands[name], atol=1e-7, rtol=0)
            commands[name] = command
            runs[name].append(run)
            print(f"{index + 1}/{args.repeats} {name}: {run['retarget_us']}", file=sys.stderr)
    comparison = None
    if args.baseline:
        for before, after in zip(runs["baseline"], runs["candidate"]):
            if before["config"] != after["config"] or before["rejected"] != after["rejected"]:
                raise ValueError("config or rejection behavior differs")
        np.testing.assert_allclose(commands["baseline"], commands["candidate"], atol=1e-5, rtol=0)
        comparison = {
            "max_command_difference_rad": float(np.max(np.abs(commands["baseline"] - commands["candidate"]))),
            "paired_speedup_percent": {
                quantile: [100 * (1 - after["retarget_us"][quantile] / before["retarget_us"][quantile])
                           for before, after in zip(runs["baseline"], runs["candidate"])]
                for quantile in ("p50", "p95")
            },
        }
    report = {"input_sha256": hashlib.sha256((episode / "data.h5").read_bytes()).hexdigest(),
              "warmup_replays": args.warmup, "threads": 1, "runs": runs, "comparison": comparison}
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.expanduser().write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
