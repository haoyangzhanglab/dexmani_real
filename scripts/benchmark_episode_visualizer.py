"""只读、无 GUI 的 EpisodeVisualizer 性能测量；报告与真实数据分开保存。

例：python scripts/benchmark_episode_visualizer.py EPISODE --output /tmp/visualizer-e1
每轮使用新进程；阶段探针和无探针对照交错执行。计时不包含报告生成。
"""

from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from functools import wraps
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
from unittest.mock import patch
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "RAYON_NUM_THREADS",
    "RERUN",
    "RERUN_FLUSH_NUM_BYTES",
    "RERUN_FLUSH_TICK_SECS",
)


def summary(values):
    """线性插值分位数；空样本不伪装成零耗时。"""
    ordered = sorted(values)
    count = len(ordered)
    result = {"sample_count": count, "p99_under_sampled": count < 100}
    if not count:
        return result | dict.fromkeys(("p50", "p95", "p99", "mean", "max", "total"))

    def percentile(q):
        position = (count - 1) * q
        left = int(position)
        right = min(left + 1, count - 1)
        return ordered[left] + (ordered[right] - ordered[left]) * (position - left)

    return result | {
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "mean": statistics.fmean(ordered),
        "max": ordered[-1],
        "total": sum(ordered),
    }


class Probe:
    """仅包裹本进程的调用；父阶段扣除子阶段，避免重复相加。"""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.phase = "init"
        self.stack = []
        self.samples: dict[str, array] = defaultdict(lambda: array("Q"))
        self.self_samples: dict[str, array] = defaultdict(lambda: array("Q"))
        self.reads: Counter[str] = Counter()
        self.failures: Counter[str] = Counter()
        self.patches = ExitStack()

    @contextmanager
    def measure(self, label):
        if not self.enabled:
            yield
            return
        name = f"{self.phase}.{label}"
        entry = [time.perf_counter_ns(), 0]
        self.stack.append(entry)
        try:
            yield
        except BaseException:
            self.failures[name] += 1
            raise
        finally:
            elapsed = time.perf_counter_ns() - entry[0]
            self.stack.pop()
            if self.stack:
                self.stack[-1][1] += elapsed
            self.samples[name].append(elapsed)
            self.self_samples[name].append(elapsed - entry[1])

    def wrap(self, owner, name, label):
        original = getattr(owner, name)

        @wraps(original)
        def measured(*args, **kwargs):
            key = label(*args, **kwargs) if callable(label) else label
            with self.measure(key):
                return original(*args, **kwargs)

        self.patches.enter_context(patch.object(owner, name, measured))

    def install(self, visual, rr):
        import h5py
        from dexmani_real.perception import pointcloud

        for owner, names in (
            (visual.RawReader, ("__init__", "read_timestamps_s", "require_fields", "read_camera")),
            (
                visual.EpisodeVisualizer,
                (
                    "preload_state",
                    "log_recording_data",
                    "log_camera",
                    "log_pointcloud",
                    "log_hand_geometry",
                    "log_hoi_diagnostics",
                ),
            ),
            (visual.ArmFK, ("__init__", "forward")),
            (visual.HandFK, ("__init__", "forward")),
            (visual.HandPointCloud, ("__init__", "transform_batch")),
            (visual.PointCloudProcessor, ("__init__", "process")),
        ):
            for name in names:
                self.wrap(owner, name, f"{owner.__name__}.{name}")
        for name in (
            "read_hoi_validity",
            "align_hoi_episode",
            "init_undistort_maps",
            "hand_geometry",
            "log_rgbd",
            "log_series",
            "log_camera_geometry",
            "episode_blueprint",
        ):
            self.wrap(visual, name, name)
        self.wrap(
            visual, "remap_image", lambda image, *a, **kw: "remap_rgb" if image.ndim == 3 else "remap_depth"
        )
        for name in ("filter_depth", "voxel_downsample", "radius_inliers", "sample_indices"):
            self.wrap(pointcloud, name, f"pointcloud.{name}")
        for name in ("send_columns", "send_blueprint", "set_time_sequence", "set_time_seconds"):
            self.wrap(rr, name, f"sdk.{name}")
        self.wrap(rr.Scalar, "__init__", "sdk.Scalar_construct")
        self.wrap(visual, "log_points", lambda path, *a, **kw: f"log_points.{path}")
        self.wrap(
            rr, "log", lambda path, *a, **kw: "sdk.log_hoi" if path.startswith("hoi/") else "sdk.log_other"
        )

        original_read = h5py.Dataset.__getitem__

        def read(dataset, key, **kwargs):
            path = dataset.name
            self.reads[f"{self.phase}:{path}"] += 1
            kind = "quality" if path.startswith("/quality/") else path.lstrip("/")
            with self.measure(f"hdf5.{kind}"):
                return original_read(dataset, key, **kwargs)

        self.patches.enter_context(patch.object(h5py.Dataset, "__getitem__", read))
        original_items = h5py.Group.items

        def items(group):
            if group.name != "/quality":
                return original_items(group)

            def traverse():
                iterator = iter(original_items(group))
                sentinel = object()
                while True:
                    with self.measure("hoi.quality_group_next"):
                        item = next(iterator, sentinel)
                    if item is sentinel:
                        return
                    yield item

            return traverse()

        self.patches.enter_context(patch.object(h5py.Group, "items", items))

    def results(self):
        return {
            name: {
                "inclusive_ms": summary(v / 1e6 for v in samples),
                "exclusive_ms": summary(v / 1e6 for v in self.self_samples[name]),
                "failed_calls": self.failures[name],
            }
            for name, samples in sorted(self.samples.items())
        }


@contextmanager
def headless(rr, probe):
    """保留全部展示内容，仅将 SDK 输出接到 /dev/null；不测 GUI。"""
    original = rr.init

    def initialize(*args, **kwargs):
        kwargs["spawn"] = False
        with probe.measure("sdk.init_and_null_sink"):
            original(*args, **kwargs)
            if not rr.is_enabled():
                raise RuntimeError("Rerun logging is disabled; refusing no-op timing")
            rr.save(os.devnull)

    with patch.object(rr, "init", initialize):
        yield


def profile_rows(profiler):
    import pstats

    rows = []
    # 原始 stats 保留秒级精度；get_stats_profile 会将秒数舍入到三位小数。
    stats = pstats.Stats(profiler).stats  # type: ignore[attr-defined]
    for (file, line, name), (primitive, calls, own, cumulative, _) in stats.items():
        path = Path(file)
        label = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name
        rows.append(
            dict(
                function=f"{label}:{line}:{name}",
                calls=calls,
                primitive_calls=primitive,
                self_ms=own * 1000,
                cumulative_ms=cumulative * 1000,
            )
        )
    return sorted(rows, key=lambda row: row["cumulative_ms"], reverse=True)[:40]


def run_worker(job):
    started = time.perf_counter_ns()
    import cv2
    import h5py
    import numpy as np
    import rerun as rr
    from scripts import visualize_episode as visual
    from dexmani_real.config.real import load_config
    from dexmani_real.perception.pointcloud import processing_metadata

    imported = time.perf_counter_ns()
    probe = Probe(job["mode"] == "instrumented")
    runtime = load_config(job["config_path"])
    result: dict[str, Any] = dict(status="ok", frames_decoded=0, frames_logged=0, missing_frames=None)
    viewer = None
    profiler = None
    if job["mode"] == "profile":
        import cProfile

        profiler = cProfile.Profile()
    rss_import = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    workload_start = time.perf_counter_ns()
    try:
        if probe.enabled:
            probe.install(visual, rr)
        with headless(rr, probe):
            if profiler is not None:
                profiler.enable()
            begin = time.perf_counter_ns()
            with probe.measure("total"):
                viewer = visual.EpisodeVisualizer(
                    job["episode"],
                    table_plane_path=runtime.environment.table_plane_path,
                    max_frames=job["max_frames"],
                    hoi=job["hoi"],
                    pointcloud_config=runtime.pointcloud if job["point_cloud"] else None,
                    hand_pointcloud_config=runtime.hand_pointcloud,
                    hoi_max_joint_gap_s=runtime.execution.hoi_max_joint_gap_s,
                )
            result["initialization_ms"] = (time.perf_counter_ns() - begin) / 1e6
            result["frames_expected"] = viewer.frame_count
            result["hoi_active"] = viewer.hoi_alignment is not None
            result["point_cloud_active"] = viewer.pointcloud_processor is not None
            if job["hoi"] and not result["hoi_active"]:
                raise ValueError("HOI requested but annotation is absent")
            if job["point_cloud"] and not result["point_cloud_active"]:
                raise ValueError("Camera point cloud requested but extrinsics are absent")
            probe.phase = "frames"
            images = iter(viewer.reader.iter_rgb_frames())
            sentinel = object()
            begin = time.perf_counter_ns()
            for index in range(viewer.frame_count):
                with probe.measure("frame_wall"):
                    with probe.measure("rgb_decode_next"):
                        rgb = next(images, sentinel)
                    if rgb is sentinel:
                        break
                    result["frames_decoded"] += 1
                    with probe.measure("log_step"):
                        viewer.log_step(index, rgb)
                    result["frames_logged"] += 1
            result["frame_loop_ms"] = (time.perf_counter_ns() - begin) / 1e6
            result["missing_frames"] = viewer.frame_count - result["frames_decoded"]
            result["empty_camera_cloud_frames"] = viewer.empty_cloud_count
            if result["missing_frames"]:
                raise ValueError(f"RGB missing at frame {result['frames_decoded']}")
    except Exception as error:
        message = str(error).replace(job["episode"], "<episode>").replace(str(ROOT), "<repo>")
        result.update(status="failed", error=f"{type(error).__name__}: {message}")
    finally:
        probe.phase = "shutdown"
        try:
            try:
                recording = rr.get_global_data_recording()
                if recording is not None:
                    with probe.measure("sdk.flush"):
                        recording.flush(blocking=True)
            finally:
                if viewer is not None:
                    with probe.measure("close"):
                        viewer.close(sys.exc_info()[1])
        except Exception as error:
            result.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}")
        if profiler is not None:
            profiler.disable()
        finished = time.perf_counter_ns()
        rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        probe.patches.close()
    # Linux ru_maxrss 单位 KiB；macOS 为 bytes。统计与 JSON 构造放在峰值采样之后。
    rss_divisor = 1024**2 if sys.platform == "darwin" else 1024
    result.update(
        import_ms=(imported - started) / 1e6,
        workload_ms=(finished - workload_start) / 1e6,
        worker_ms=(finished - started) / 1e6,
        peak_rss_mib=rss_peak / rss_divisor,
        import_peak_rss_mib=rss_import / rss_divisor,
        timing_buffer_mib=sum(
            len(v) * v.itemsize for data in (probe.samples, probe.self_samples) for v in data.values()
        )
        / 1024**2,
        stages=probe.results(),
        hdf5_dataset_getitem_calls=dict(probe.reads),
        pointcloud=processing_metadata(runtime.pointcloud),
        hand_pointcloud=asdict(runtime.hand_pointcloud),
        hoi_max_joint_gap_s=runtime.execution.hoi_max_joint_gap_s,
        versions=dict(
            numpy=np.__version__, opencv=cv2.__version__, h5py=h5py.__version__, rerun=rr.__version__
        ),
        opencv_threads=cv2.getNumThreads(),
    )
    if profiler is not None:
        result["cprofile_top"] = profile_rows(profiler)
    return result


def dataset_info(path, index):
    import h5py

    result = dict(id=f"episode_{index}", path_label=f"…/{path.name}", hoi_exists=(path / "hoi.h5").is_file())
    with h5py.File(path / "data.h5", "r") as data:
        result["frames"] = len(data["timing/rows"])
        result["resolution_wh"] = [
            int(data["meta"].attrs[f"camera_color_{axis}"]) for axis in ("width", "height")
        ]
        result["depth_storage"] = dict(chunks=data["depth"].chunks, compression=data["depth"].compression)
    return result


def collect_environment():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    cpu = platform.processor()
    if Path("/proc/cpuinfo").is_file():
        cpu = next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        )
    return dict(
        commit=git("rev-parse", "HEAD"),
        branch=git("branch", "--show-current"),
        tracked_diff=git("diff", "--name-only", "HEAD"),
        python=platform.python_version(),
        measured_at_utc=datetime.now(timezone.utc).isoformat(),
        os=platform.platform(),
        cpu=cpu,
        cpu_count=os.cpu_count(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        thread_environment={name: os.environ.get(name) for name in THREAD_ENV},
    )


def aggregate(runs):
    groups: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        if run["phase"] != "formal" or run["status"] != "ok":
            continue
        groups.setdefault((run["case"], run["mode"]), []).append(run)
    result = []
    for (case, mode), selected in groups.items():
        stages = {}
        for name in sorted({name for run in selected for name in run["stages"]}):
            values = [run["stages"][name] for run in selected if name in run["stages"]]
            stages[name] = {
                kind: summary(value[kind]["total"] for value in values)
                for kind in ("inclusive_ms", "exclusive_ms")
            }
            stages[name]["call_count"] = sum(value["inclusive_ms"]["sample_count"] for value in values)
        result.append(
            dict(
                case=case,
                mode=mode,
                run_count=len(selected),
                workload_ms=summary(run["workload_ms"] for run in selected),
                initialization_ms=summary(run["initialization_ms"] for run in selected),
                frame_loop_ms=summary(run["frame_loop_ms"] for run in selected),
                peak_rss_mib=summary(run["peak_rss_mib"] for run in selected),
                stages=stages,
            )
        )
    return result


def markdown(report):
    env, protocol = report["environment"], report["protocol"]
    lines = [
        "# EpisodeVisualizer 性能画像",
        "",
        "## 环境与数据",
        "",
        f"- Git：`{env['commit']}`（{env['branch']}）；已跟踪改动：`{env['tracked_diff'] or '无'}`。",
        f"- Python：{env['python']}；OS：{env['os']}；CPU：{env['cpu']}。",
        f"- 线程环境：`{json.dumps(env['thread_environment'])}`。",
        f"- 数据类型：**{report['data_kind']}**；预热 {protocol['warmups']} 次，正式 {protocol['repeats']} 次。",
    ]
    for data in report["datasets"]:
        lines.append(
            f"- {data['id']}：`{data['path_label']}`，{data['frames']} 帧，"
            f"分辨率 {data['resolution_wh']}，HOI={data['hoi_exists']}。"
        )
    completed = next((run for run in report["runs"] if "versions" in run), None)
    if completed is not None:
        lines += [
            f"- 数值库 / SDK：`{json.dumps(completed['versions'])}`；OpenCV 线程数：{completed['opencv_threads']}。",
            f"- PointCloud：`{json.dumps(completed['pointcloud'])}`。",
        ]
    lines += [
        "",
        "## 测量边界",
        "",
        "每轮使用独立进程调用当前 EpisodeVisualizer，只读 Raw；不清理系统缓存。"
        "Rerun 使用 spawn=False 和空设备输出，记录 SDK 构造、提交及结束时的 blocking flush。"
        "Viewer 启动与 GUI 渲染均未测量。",
        "时间单位为 ms（perf_counter_ns）。inclusive 包含子阶段，exclusive 扣除子阶段，二者不能混合累加。"
        "HOI 时间差计算与有限值检查留在原函数耗时内，不再跨函数拼接计时。"
        "cProfile 单独运行，其累计耗时不作为 wall-clock。",
        "视频解码仅计 next(iter_rgb_frames)，包含打开、解码和 RGB 转换；不含后续图像处理。"
        "HDF5 计数是 Dataset.__getitem__ 的逻辑调用，quality Group 计时覆盖迭代器 next。",
        "RSS 为子进程 resource.getrusage(RUSAGE_SELF).ru_maxrss 换算的 MiB，包含导入、原生库、SDK 队列和探针缓冲。"
        "跨运行分位数以每轮总量为样本；逐调用分位数在 JSON 的 stages 中。样本少于 100 时不解读 P99。",
        "",
        "## 配置对比",
        "",
        "| 配置 | 模式 | n | 总耗时 P50 | P95 | P99 | mean | max | RSS P50 / max MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in report["aggregates"]:
        s, rss = group["workload_ms"], group["peak_rss_mib"]
        lines.append(
            f"| {group['case']} | {group['mode']} | {s['sample_count']} | "
            + " | ".join(f"{s[key]:.3f}" for key in ("p50", "p95", "p99", "mean", "max"))
            + f" | {rss['p50']:.1f} / {rss['max']:.1f} |"
        )
    lines += ["", "### 探针对照", "", "仪器化与 plain 的平均总耗时差异，包含系统波动，不作机械校正。", ""]
    for group in report["aggregates"]:
        if group["mode"] != "instrumented":
            continue
        baseline = next(
            (
                value
                for value in report["aggregates"]
                if value["case"] == group["case"] and value["mode"] == "plain"
            ),
            None,
        )
        if baseline is not None:
            overhead = (group["workload_ms"]["mean"] / baseline["workload_ms"]["mean"] - 1) * 100
            lines.append(f"- {group['case']}：{overhead:+.2f}%。")
    lines += [
        "",
        "### 视频解码（逐帧样本，ms）",
        "",
        "| 配置 / 轮 | n | 合计 | mean | P50 | P95 | P99 | max | 空点云帧 | HDF5 读取 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        if run["phase"] != "formal" or run["mode"] != "instrumented" or run["status"] != "ok":
            continue
        s = run["stages"]["frames.rgb_decode_next"]["inclusive_ms"]
        lines.append(
            f"| {run['case']} / {run['repeat'] + 1} | {s['sample_count']} | "
            + " | ".join(f"{s[key]:.3f}" for key in ("total", "mean", "p50", "p95", "p99", "max"))
            + f" | {run['empty_camera_cloud_frames']} | {sum(run['hdf5_dataset_getitem_calls'].values())} |"
        )
    lines += ["", "### 阶段分解", "", "每轮阶段总耗时的分位数，按独占耗时排序。", ""]
    for group in report["aggregates"]:
        if group["mode"] != "instrumented":
            continue
        lines += [
            f"#### {group['case']}",
            "",
            "| 阶段 | n 轮 / 调用数 | inclusive P50 | exclusive P50 | P95 | P99 | mean | max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, value in sorted(
            group["stages"].items(), key=lambda item: item[1]["exclusive_ms"]["mean"], reverse=True
        ):
            s = value["exclusive_ms"]
            lines.append(
                f"| {name} | {s['sample_count']} / {value['call_count']} | "
                f"{value['inclusive_ms']['p50']:.3f} | "
                + " | ".join(f"{s[key]:.3f}" for key in ("p50", "p95", "p99", "mean", "max"))
                + " |"
            )
    lines += [
        "",
        "## 运行明细",
        "",
        "| 顺序 | 配置 | 阶段 / 模式 | 状态 | 解码 / 提交 / 缺帧 | 总 ms | 峰值 MiB |",
        "|---:|---|---|---|---|---:|---:|",
    ]
    for index, run in enumerate(report["runs"]):
        elapsed = f"{run['workload_ms']:.3f}" if "workload_ms" in run else "—"
        rss = f"{run['peak_rss_mib']:.1f}" if "peak_rss_mib" in run else "—"
        lines.append(
            f"| {index} | {run['case']} | {run['phase']} / {run['mode']} | {run['status']} | "
            f"{run.get('frames_decoded')} / {run.get('frames_logged')} / {run.get('missing_frames')} | "
            f"{elapsed} | {rss} |"
        )
    lines.append("")
    for skipped in report["skipped"]:
        lines.append(f"- 未测：{skipped['case']}，{skipped['reason']}。")
    for run in report["runs"]:
        for key in ("error", "cleanup_error"):
            if key in run:
                lines.append(f"- 失败：{run['case']} / {run['mode']}：{run[key]}。")
    lines += [
        "",
        "## 复现",
        "",
        "EPISODE_1、EPISODE_2 等变量对应上述数据集；CONFIG 为可选的同一实验配置。",
        "```bash",
        report["reproduce"],
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+", type=Path)
    parser.add_argument("--max-frames", nargs="+", type=int)
    parser.add_argument("--hoi", choices=("on", "off", "both"), default="both")
    parser.add_argument("--point-cloud", choices=("on", "off", "both"), default="both")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", required=True, type=Path, help="新建报告目录；拒绝覆盖已有目录")
    parser.add_argument("--config", type=Path, help="与可视化 CLI 相同的可信实验配置")
    parser.add_argument("--profile", action="store_true", help="各组合另跑一次 cProfile，不并入正式统计")
    parser.add_argument(
        "--synthetic", action="store_true", help="明确标记 fixture/合成数据，禁止用于真实性能结论"
    )
    args = parser.parse_args(argv)
    if args.warmups < 0 or args.repeats < 1 or any(n <= 0 for n in args.max_frames or []):
        parser.error("warmups >= 0，repeats >= 1，max_frames > 0")
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [path.expanduser().resolve() for path in args.episodes]
    datasets = [dataset_info(path, i + 1) for i, path in enumerate(paths)]
    configs, skipped = [], []
    for path, data in zip(paths, datasets):
        for limit in args.max_frames or [None]:
            for hoi in (True, False) if args.hoi == "both" else (args.hoi == "on",):
                for cloud in (True, False) if args.point_cloud == "both" else (args.point_cloud == "on",):
                    case = f"{data['id']}-cap{limit or 'all'}-hoi{int(hoi)}-pc{int(cloud)}"
                    if hoi and not data["hoi_exists"]:
                        skipped.append(dict(case=case, reason="缺少 hoi.h5"))
                        continue
                    configs.append(
                        dict(
                            case=case,
                            episode=str(path),
                            max_frames=limit,
                            hoi=hoi,
                            point_cloud=cloud,
                            config_path=str(args.config.expanduser().resolve()) if args.config else None,
                        )
                    )
    env = os.environ | {"PYTHONDONTWRITEBYTECODE": "1"}
    report = dict(
        environment=collect_environment(),
        data_kind="synthetic / 仅功能验证" if args.synthetic else "real",
        datasets=datasets,
        runs=[],
        skipped=skipped,
        protocol=dict(warmups=args.warmups, repeats=args.repeats),
    )
    command = "python scripts/benchmark_episode_visualizer.py " + " ".join(
        f'"$EPISODE_{index + 1}"' for index in range(len(paths))
    )
    if args.max_frames:
        command += " --max-frames " + " ".join(map(str, args.max_frames))
    command += f" --hoi {args.hoi} --point-cloud {args.point_cloud} --warmups {args.warmups} --repeats {args.repeats}"
    command += " --output /tmp/visualizer-e1-new"
    command += ' --config "$CONFIG"' if args.config else ""
    command += " --profile" if args.profile else ""
    command += " --synthetic" if args.synthetic else ""
    report["reproduce"] = command
    for phase, count in (("warmup", args.warmups), ("formal", args.repeats), ("profile", int(args.profile))):
        for repeat in range(count):
            ordered = configs[repeat % len(configs) :] + configs[: repeat % len(configs)] if configs else []
            for job in ordered:
                modes = (
                    ("profile",)
                    if phase == "profile"
                    else (("plain", "instrumented") if repeat % 2 else ("instrumented", "plain"))
                )
                for mode in modes:
                    print(f"{phase} {repeat + 1}/{count} {job['case']} {mode}", flush=True)
                    begin = time.perf_counter_ns()
                    process = subprocess.run(
                        [sys.executable, str(Path(__file__).resolve()), "--worker"],
                        input=json.dumps(job | dict(mode=mode)),
                        text=True,
                        capture_output=True,
                        cwd=ROOT,
                        env=env,
                    )
                    wall = (time.perf_counter_ns() - begin) / 1e6
                    if process.returncode:
                        result = dict(
                            status="failed",
                            error=f"worker exited {process.returncode}: "
                            + process.stderr[-2000:]
                            .replace(str(ROOT), "<repo>")
                            .replace(str(Path.home()), "<home>"),
                        )
                    else:
                        result = json.loads(process.stdout)
                    report["runs"].append(
                        result
                        | dict(case=job["case"], phase=phase, repeat=repeat, mode=mode, subprocess_ms=wall)
                    )
    report["aggregates"] = aggregate(report["runs"])
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    (args.output / "report.md").write_text(markdown(report))
    print(f"报告：{args.output / 'report.md'}")
    return int(not report["runs"] or any(run["status"] != "ok" for run in report["runs"]))


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        print(json.dumps(run_worker(json.load(sys.stdin)), allow_nan=False))
    else:
        raise SystemExit(main())
