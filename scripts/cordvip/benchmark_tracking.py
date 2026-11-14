"""只读 RGB-D A/B 回放；计时同步 CUDA，不写 Raw 或生成模板。"""

import argparse
import json
import random
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party/cordvip_geometry/src")]

from scripts.cordvip.annotate_episodes import track_episode
from cordvip_geometry import tracking
from dexmani_real.data.raw import RawReader, require_published_raw
from dexmani_real.geometry.hoi import task_objects

SAM_METHODS = ("initialize", "track", "segment_boxes", "correct_masks")


def percentiles(values, *, total=True):
    if not values:
        return None
    summary = dict(
        p50=float(np.percentile(values, 50)),
        p95=float(np.percentile(values, 95)),
    )
    if total:
        summary["total"] = sum(values)
    return summary


def tracking_statistics(records):
    """单段和跨段汇总使用相同的样本与计时口径。"""
    sam_times = [
        elapsed
        for row in records
        for stage, times in row["stages"].items()
        if stage.startswith("sam_")
        for elapsed in times
    ]
    counts = sam_counts(records)
    return dict(
        frames=len(records),
        counts=counts,
        sam_calls=sum(counts.values()),
        sam_ratio=len(sam_times) / len(records),
        video_ratio=(counts["sam_initialize"] + counts["sam_track"]) / len(records),
        sam_ms=percentiles(sam_times),
        fp_ms=percentiles(
            [sum(row["stages"]["fp"]) for row in records if "fp" in row["stages"]]
        ),
        tracking_ms=percentiles(
            [row["total_ms"] for row in records if not row["initial"]]
        ),
        valid_fraction=np.mean([all(row["valid"].values()) for row in records]),
        coverage_rejected_frames=sum(
            any(q["coverage_only_rejection"] for q in row["depth_checks"])
            for row in records
        ),
        coverage_triggered_sam_frames=sum(
            bool(row["stages"].get("sam_track"))
            and any(q["low_coverage"] for q in row["depth_checks"])
            for row in records
        ),
        reference_rejections=sum(len(row["reference_rejected"]) for row in records),
    )


def sam_counts(records):
    return {
        f"sam_{method}": sum(
            len(row["stages"].get(f"sam_{method}", [])) for row in records
        )
        for method in SAM_METHODS
    }


def summarize(output):
    """仅重读逐帧结果，可在不重跑模型的情况下复核统计。"""
    summaries = json.loads((output / "summary.json").read_text())
    comparisons = []
    for interval in (1, 2, 3):
        records: list[dict[str, Any]] = []
        stable: list[dict[str, Any]] = []
        for run in summaries:
            if run["interval"] != interval:
                continue
            stem = f"{run['episode']}_r{run['repeat']}_i"
            rows = json.loads((output / (stem + str(interval) + ".json")).read_text())
            baseline = json.loads((output / (stem + "1.json")).read_text())
            if len(rows) != len(baseline):
                raise ValueError(f"Frame count differs from baseline: {stem}{interval}")
            eligible = np.array(
                [
                    not r["initial"]
                    and all(r["valid"].values())
                    and not r["stages"].get("sam_segment_boxes")
                    for r in baseline
                ]
            )
            transitions = np.diff(np.r_[False, eligible, False].astype(int))
            selected = np.zeros(len(rows), bool)
            for lo, hi in zip(
                np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
            ):
                if hi - lo >= 30:
                    selected[lo:hi] = True
            stable.extend(row for row, keep in zip(rows, selected) if keep)
            records.extend(rows)

        depths = {}
        for source in (0, 1):
            metrics = [
                q
                for r in records
                for n, q in r["quality"].items()
                if r["valid"][n] and q["mask_source"] == source
            ]
            depths[str(source)] = {
                key: percentiles([q[key] for q in metrics], total=False)
                for key in ("depth_pixels", "depth_fraction", "depth_coverage")
            }
        comparisons.append(
            dict(
                interval=interval,
                **tracking_statistics(records),
                stable_frames=len(stable),
                stable_sam_calls=sum(sam_counts(stable).values()),
                depth_quality_by_mask_source=depths,
            )
        )
    (output / "comparison.json").write_text(json.dumps(comparisons, indent=2))
    return comparisons


def main(argv=None):
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="pnp_apple_banana")
    parser.add_argument("--episode", type=Path, action="append")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if Path(args.task).name != args.task or args.task in (".", ".."):
        parser.error("非法任务名")
    if args.repeats < 1:
        parser.error("repeats must be positive")
    task = ROOT / "assets/pointcloud_template/object" / args.task
    objects = task_objects(task)
    names = [obj["name"] for obj in objects]
    prompts = {obj["name"]: obj["prompt"] for obj in objects}
    episodes = args.episode or sorted((ROOT / "episodes" / args.task).glob("episode_*"))
    if not episodes:
        raise ValueError("No episodes")
    if len({episode.name for episode in episodes}) != len(episodes):
        raise ValueError("Episode names must be unique in benchmark output")
    for episode in episodes:
        require_published_raw(episode)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for tracking benchmarks")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        git=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=subprocess.check_output(["git", "diff", "--stat"], text=True),
        gpu=torch.cuda.get_device_name(0),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        driver=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
        episodes=[str(p.resolve()) for p in episodes],
        objects=names,
        repeats=args.repeats,
    )
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2))
    tracker = tracking.SceneTracker(
        {n: task / n / "mesh/mesh_m.obj" for n in names}, workspace=ROOT
    )
    current: dict[str, list[float]] = {}
    latest_masks: dict[str, np.ndarray] = {}
    original_batch = tracking.track_objects
    original_depth_quality = tracking.pose_depth_quality
    depth_checks = []

    def checked_depth(observed, rendered, config):
        metrics, accepted = original_depth_quality(observed, rendered, config)
        low_coverage = metrics["depth_coverage"] < config.min_depth_coverage
        depth_checks.append(
            dict(
                depth_coverage=metrics["depth_coverage"],
                depth_fraction=metrics["depth_fraction"],
                depth_pixels=metrics["depth_pixels"],
                low_coverage=low_coverage,
                coverage_only_rejection=(
                    low_coverage
                    and metrics["depth_pixels"] >= config.min_depth_pixels
                    and metrics["depth_fraction"] >= config.min_depth_fraction
                ),
            )
        )
        return metrics, accepted

    def timed(function, key, masks=False):
        def call(*args, **kwargs):
            torch.cuda.synchronize()
            start = perf_counter()
            try:
                result = function(*args, **kwargs)
                if masks:
                    latest_masks.clear()
                    latest_masks.update(result)
                return result
            finally:
                torch.cuda.synchronize()
                current.setdefault(key, []).append((perf_counter() - start) * 1000)

        return call

    for method in SAM_METHODS:
        setattr(
            tracker.segmenter,
            method,
            timed(
                getattr(tracker.segmenter, method),
                "sam_" + method,
                masks=method != "segment_boxes",
            ),
        )
    tracking.track_objects = timed(original_batch, "fp")
    tracking.pose_depth_quality = checked_depth
    records: list[dict[str, Any]] = []
    references: list[dict[str, np.ndarray]] | None = None

    def record(function, initial):
        def call(frame, *args):
            current.clear()
            depth_checks.clear()
            if initial:
                latest_masks.clear()
            torch.cuda.synchronize()
            start = perf_counter()
            result = None
            error = None
            try:
                result = function(frame, *args)
                return result
            except tracking.TrackingFailure as exc:
                error = str(exc)
                raise
            finally:
                torch.cuda.synchronize()
                elapsed = (perf_counter() - start) * 1000
                row = dict(
                    initial=initial,
                    total_ms=elapsed,
                    stages=dict(current),
                    depth_checks=list(depth_checks),
                    valid={n: False for n in names} if result is None else result.valid,
                    quality={} if result is None else result.quality,
                    error=error,
                    reference_rejected=[],
                )
                if (
                    references is not None
                    and tracker.config.segmentation_interval_frames == 1
                ):
                    stride = tracking.PIXEL_STRIDE
                    references.append(
                        {
                            name: mask[::stride, ::stride].copy()
                            for name, mask in latest_masks.items()
                        }
                    )
                elif references is not None and result is not None:
                    _, depth = tracker.rectifier.apply(frame.rgb, frame.depth_m)
                    stride = tracking.PIXEL_STRIDE
                    sampled_depth = depth[::stride, ::stride]
                    for name, pose in result.poses.items():
                        mask = references[len(records)].get(name)
                        if mask is None:
                            row["reference_rejected"].append(name)
                            continue
                        rendered = (
                            tracker.renderers[name]
                            .render(tracker.directions.reshape(-1, 3), pose)
                            .reshape(sampled_depth.shape)
                        )
                        _, accepted = tracking.pose_quality(
                            mask, sampled_depth, rendered, tracker.config
                        )
                        if not accepted:
                            row["reference_rejected"].append(name)
                records.append(row)

        return call

    for method in ("initialize", "track"):
        setattr(
            tracker, method, record(getattr(tracker, method), method == "initialize")
        )
    summaries = []
    try:
        for interval in (1, 2, 3):
            tracker.config = replace(
                tracker.config, segmentation_interval_frames=interval
            )
            with RawReader(episodes[0]) as reader:
                # 完整预热；正式回放通过 initialize 重置跟踪状态。
                track_episode(reader, tracker, names, prompts)
            records.clear()
        for repeat in range(args.repeats):
            for episode in episodes:
                references = []
                for interval in (1, 2, 3):
                    random.seed(0)
                    np.random.seed(0)
                    torch.manual_seed(0)
                    tracker.config = replace(
                        tracker.config, segmentation_interval_frames=interval
                    )
                    records.clear()
                    with RawReader(episode) as reader:
                        camera = reader.read_camera()
                        resolution = [camera.color.width, camera.color.height]
                        track_episode(reader, tracker, names, prompts)
                    transitions = {n: {"lost": 0, "recovery": 0} for n in names}
                    previous = dict.fromkeys(names, False)
                    initialized = False
                    for row in records:
                        for n in names:
                            now = row["valid"][n]
                            if initialized and previous[n] != now:
                                transitions[n]["recovery" if now else "lost"] += 1
                            previous[n] = now
                        initialized |= all(row["valid"].values())
                    summary = dict(
                        episode=episode.name,
                        repeat=repeat,
                        interval=interval,
                        config=asdict(tracker.config),
                        resolution=resolution,
                        **tracking_statistics(records),
                        transitions=transitions,
                    )
                    summaries.append(summary)
                    stem = f"{episode.name}_r{repeat}_i{interval}"
                    (args.output / (stem + ".json")).write_text(
                        json.dumps(records, indent=2)
                    )
                    (args.output / "summary.json").write_text(
                        json.dumps(summaries, indent=2)
                    )
                    print(json.dumps(summary), flush=True)
        summarize(args.output)
    finally:
        tracking.track_objects = original_batch
        tracking.pose_depth_quality = original_depth_quality


if __name__ == "__main__":
    main()
