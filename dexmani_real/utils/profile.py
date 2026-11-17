"""显式启用的实验统计；只存标量，结束后写出独立 JSON。"""

from collections import Counter, defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import threading
import time
from typing import ParamSpec, TypeVar

logger = logging.getLogger(__name__)
P = ParamSpec("P")
T = TypeVar("T")


def call(
    profiler: "Profiler | None", name: str, function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
) -> T:
    """保留返回值和异常；关闭时不读时钟、不创建采样容器。"""
    if profiler is None:
        return function(*args, **kwargs)
    started = time.perf_counter_ns()
    try:
        return function(*args, **kwargs)
    except BaseException:
        profiler.count(f"{name}_failures")
        raise
    finally:
        profiler.sample(name, (time.perf_counter_ns() - started) / 1e6)


class Profiler:
    def __init__(self, kind, *, max_samples=50_000):
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        self.kind = kind
        self.exit_code = None
        self.max_samples = max_samples
        self.lock = threading.Lock()
        self.samples: dict[str, deque[float]] = {}
        self.counts: Counter[str] = Counter()
        self.totals: dict[str, float] = defaultdict(float)
        self.maxima: dict[str, float] = {}
        self.counters: Counter[str] = Counter()
        self.last_tick = {}
        self.last_vr_stamp = None
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.started_ns = time.monotonic_ns()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
        )
        self.environment = dict(
            git_commit=commit.stdout.strip() if commit.returncode == 0 else None,
            python=platform.python_version(),
            platform=platform.platform(),
        )
        self.thresholds = {}
        self.hand_enabled = False

    def configure(self, runtime, *, control_hz, mode=None):
        self.environment["control_hz"] = control_hz
        if mode is not None:
            self.environment["execution_mode"] = mode
        self.hand_enabled = runtime.hand.enabled
        self.thresholds = dict(
            arm=runtime.arm.feedback_max_age_s,
            hand=runtime.hand.feedback_max_age_s,
            camera=runtime.camera.max_frame_age_s,
            raw_camera=runtime.camera.max_frame_age_s,
            pointcloud=runtime.camera.max_frame_age_s,
            hoi=runtime.execution.hoi_max_age_s,
            tactile_aggregate=runtime.hand.feedback_max_age_s,
            tactile_dense=runtime.hand.feedback_max_age_s,
            vr=runtime.teleop.vr_stale_threshold_s,
        )

    def sample(self, name: str, value: float | None):
        with self.lock:
            if name not in self.samples:
                self.samples[name] = deque(maxlen=self.max_samples)
            if value is None:
                return
            if not math.isfinite(value):
                self.counters["nonfinite_samples_rejected"] += 1
                return
            self.samples[name].append(value)
            self.counts[name] += 1
            self.totals[name] += value
            self.maxima[name] = max(self.maxima.get(name, value), value)

    def count(self, name: str, amount: int = 1):
        with self.lock:
            self.counters[name] += amount

    def maximum(self, name: str, value: int):
        with self.lock:
            self.counters[name] = max(self.counters[name], value)

    def start_episode(self):
        self.last_tick.clear()
        self.count("episodes")

    def control_slot(self, now_ns, planned_ns, period_ns, skipped, *, group="control"):
        """来源为原调度器；lateness 相对跳槽之前的到期时刻。"""
        previous = self.last_tick.get(group)
        if previous is not None:
            self.sample(f"{group}_period_ms", (now_ns - previous) / 1e6)
        self.last_tick[group] = now_ns
        self.sample(f"{group}_target_period_ms", period_ns / 1e6)
        self.sample(f"{group}_lateness_ms", max(0, now_ns - planned_ns) / 1e6)
        self.count(f"{group}_ticks")
        self.count("skipped_slots", skipped)

    @contextmanager
    def tick(self, deadline_ns):
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed = time.perf_counter_ns() - started
            ended_ns = time.monotonic_ns()
            self.sample("control_tick_ms", elapsed / 1e6)
            self.count("deadline_misses", int(ended_ns >= deadline_ns))

    def sensor(self, name, stamp, now_ns, *, available=True):
        self.count(f"{name}_observations")
        metric = f"sensor_age_ms.{name}"
        if not available or stamp <= 0 or stamp > now_ns:
            self.count(f"{name}_unavailable")
        if stamp <= 0 or stamp > now_ns:
            self.sample(metric, None)
            if stamp > now_ns:
                self.count(f"{name}_future_timestamp")
            return
        age = now_ns - stamp
        threshold = int(self.thresholds[name] * 1e9)
        self.sample(metric, age / 1e6)
        self.count(f"{name}_stale", int(age >= threshold))
        self.count(f"{name}_near_stale", int(age >= 0.8 * threshold))

    def observe(self, row, needs, *, include_visual_payloads=True):
        """只读已取回的来源时间；年龄参考该快照完成时刻，不再读取 Ring。"""
        now = row.observation_timestamp_ns
        self.sensor("arm", int(row.arm["timestamp_ns"][0]), now)
        if self.hand_enabled:
            self.sensor(
                "hand",
                int(row.hand["timestamp_ns"][0]) if row.hand is not None else 0,
                now,
            )
        for channel in needs.required_tactile:
            stamp = int(row.hand[f"tactile_{channel}_timestamp_ns"][0]) if row.hand is not None else 0
            self.sensor(f"tactile_{channel}", stamp, now)
        if needs.camera_fields:
            self.sensor(
                "camera",
                int(row.camera["timestamp_ns"]) if row.camera is not None else 0,
                now,
            )
        if needs.point_cloud:
            self.sensor(
                "pointcloud",
                row.pointcloud_timestamp_ns,
                now,
                available=not include_visual_payloads or row.point_cloud is not None,
            )
        if needs.hoi:
            self.sensor("hoi", int(row.hoi["timestamp_ns"]) if row.hoi is not None else 0, now)
            self.count("hoi_invalid", int(row.hoi is not None and not row.hoi["valid"]))
        if needs.vr:
            stamp = int(row.vr["recv_ts_ns"]) if row.vr is not None else 0
            self.sensor("vr", stamp, now)
            if 0 < stamp <= now:
                self.count("vr_reused_samples" if stamp == self.last_vr_stamp else "vr_new_samples")
                self.last_vr_stamp = stamp

    def report(self):
        ended_ns = time.monotonic_ns()
        # 只在收尾调用；逐指标短锁复制标量窗口，排序不持锁。
        with self.lock:
            names = list(self.samples)
            counters = dict(self.counters)
        metrics = {}
        for name in names:
            with self.lock:
                values = list(self.samples[name])
                count, total, maximum = (
                    self.counts[name],
                    self.totals[name],
                    self.maxima.get(name),
                )
            values.sort()
            n = len(values)

            def percentile(q):
                if not n:
                    return None
                position = (n - 1) * q
                lo = int(position)
                return values[lo] + (values[min(lo + 1, n - 1)] - values[lo]) * (position - lo)

            metrics[name] = dict(
                count=count,
                sample_count=n,
                dropped_count=count - n,
                p50=percentile(0.5),
                p95=percentile(0.95),
                p99=percentile(0.99),
                p99_under_sampled=n < 100,
                mean=total / count if count else None,
                max=maximum,
                window_mean=statistics.fmean(values) if n else None,
            )
        return dict(
            schema_version=1,
            kind=self.kind,
            environment=dict(self.environment),
            session=dict(
                started_at=self.started_at,
                duration_s=(ended_ns - self.started_ns) / 1e9,
                episodes=counters.get("episodes", 0),
                exit_code=self.exit_code,
            ),
            sampling=dict(
                max_samples_per_metric=self.max_samples,
                quantiles="last_samples",
                count_mean_max="all_samples",
            ),
            sensor_threshold_ms={k: v * 1000 for k, v in self.thresholds.items()},
            metrics=metrics,
            counters=counters,
        )

    def write(self, path):
        payload = self.report()
        with Path(path).open("x") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")


@contextmanager
def profile_session(output, kind, runtime, *, control_hz):
    """CLI 最外层收尾；保存失败只报告，不替换实验异常或退出码。"""
    if output is None:
        yield None
        return
    path = Path(output).expanduser()
    if os.path.lexists(path):
        raise FileExistsError(f"profile output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    profiler = Profiler(kind)
    profiler.configure(runtime, control_hz=control_hz)
    try:
        yield profiler
    except BaseException:
        profiler.count("session_exceptions")
        raise
    finally:
        try:
            profiler.write(path)
        except BaseException as error:
            # 尽力保存也不能覆盖此前的实验失败或中断。
            logger.warning("性能报告保存失败：%s", type(error).__name__)
