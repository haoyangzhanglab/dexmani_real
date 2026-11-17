from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty, Full, Queue

from av.error import FFmpegError

from dexmani_real.data.raw import RawWriter, validate_task_label
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.runtime.observation import Observation
from dexmani_real.utils.io import publish
from dexmani_real.utils.timing import sampling_clock_ns

logger = logging.getLogger(__name__)
RECORDER_START_TIMEOUT_S = 10.0
RECORDER_STOP_TIMEOUT_S = 60.0


def format_recording_result(raw: dict | None) -> tuple[int, str]:
    """展示 Recorder 的观察结果；不判断设备停止或是否允许开始新段。"""
    if raw is None:
        return logging.INFO, "Raw：未启用或尚未开始录制。"
    status = raw["status"]
    level, description = {
        "published": (logging.INFO, "Raw 保存成功，已发布"),
        "failed": (logging.ERROR, "Raw 录制失败，未发布有效 Episode"),
        "not_published": (logging.INFO, "Raw 未发布"),
        "pending": (logging.WARNING, "仍在收尾，尚无最终录制结果"),
    }[status]
    truncated = raw["recording_truncated_reason"]
    if status == "published" and truncated is not None:
        description = "部分数据已保存，录制发生截断"
        level = logging.WARNING
    lines = [f"Raw：{status}（{description}）。",
             f"Recorded rows{' (当前)' if status == 'pending' else ''}: {raw['row_count']}"]
    if raw["path"] is not None:
        lines.append(f"Path: {raw['path']}")
    if truncated is not None:
        lines.append(f"录制截断原因：{truncated}")
    if raw["error"] is not None:
        lines.append(f"录制错误：{raw['error']}")
    staging = raw["staging_path"]
    if staging is not None or status in {"failed", "pending"}:
        lines.append(f"Staging：{staging if staging is not None else 'Recorder 未提供路径'}")
    return level, "\n".join(lines)


class RecordingError(RuntimeError):
    """明确的录制 I/O 失败；不代表控制或设备故障。"""


@dataclass(frozen=True)
class _FinishRequest:
    save: bool
    reason: str
    deadline: float


class EpisodeRecorder:
    """每段独占一个 writer；失败由控制流程终段，线程退出后才处置文件。"""

    def __init__(self, data_dir, *, control_hz, profiler=None):
        self.profiler = profiler
        self._profile_finish_ns = None
        sampling_clock_ns(control_hz)
        self.data_dir = Path(data_dir)
        self.control_hz = float(control_hz)
        self.thread = None
        self.queue: Queue[tuple[Observation, RobotCommand | None, bool]] = Queue(maxsize=16)
        self.ready = threading.Event()
        self.error = None
        self.is_recording = False
        self.recording_truncated_reason = None
        self.pending_segment_start = False
        self.finish_request: _FinishRequest | None = None
        self.staging_path: Path | None = None
        self.published_path: Path | None = None
        self.writer = None

    @property
    def written_frames(self):
        return self.writer.frame_count if self.writer is not None else 0

    @property
    def writer_pending(self):
        return self.thread is not None and self.thread.is_alive()

    def store_error(self, error, *, stage="recording_cleanup"):
        self.is_recording = False
        if isinstance(error, (OSError, FFmpegError)):
            failure = RecordingError(f"Recording failed: {type(error).__name__}: {error}")
            failure.__cause__ = error
            error = failure
        if self.error is None:
            self.error = error
            logger.error("录制失败；staging 保留：%s", self.staging_path,
                         exc_info=(type(error), error, error.__traceback__))
        else:
            logger.error("%s also failed", stage, exc_info=(type(error), error, error.__traceback__))
            # 程序错误或中断不能被更早的 I/O 错误降级。
            if (isinstance(self.error, RecordingError) and not isinstance(error, RecordingError)
                    or isinstance(self.error, Exception) and not isinstance(error, Exception)):
                self.error = error

    def check_error(self):
        if self.error is not None:
            raise self.error

    def mark_gap(self):
        if self.profiler is not None:
            self.profiler.count("recording_gap_requests")
        self.pending_segment_start = True
        logger.info("录制间断；下一行开始新的连续区间。")

    def result(self):
        """仅报告最后可观察的结果，pending 不推断线程之后的结局。"""
        pending = self.writer_pending
        published, staging = self.published_path, self.staging_path
        if published is not None:
            status = "published"
        elif pending:
            status = "pending"
        elif self.error is not None:
            status = "failed"
        else:
            status = "not_published"
        return {
            "status": status,
            "path": str(published) if published is not None else None,
            "staging_path": str(staging) if staging is not None and published is None else None,
            "row_count": self.written_frames,
            "error": str(self.error) if self.error is not None else None,
            "recording_truncated_reason": self.recording_truncated_reason,
        }

    def start(
        self,
        *,
        task_label,
        collection_source,
        metadata,
        episode_name=None,
    ):
        try:
            validate_task_label(task_label)
            metadata = dict(metadata, task_label=task_label, collection_source=collection_source)
            if episode_name is not None and (
                type(episode_name) is not str
                or not episode_name
                or episode_name.startswith(".")
                or "/" in episode_name
                or "\\" in episode_name
            ):
                raise ValueError("episode_name must be a plain non-empty directory name")
            self.data_dir.mkdir(parents=True, exist_ok=True)
            name = episode_name or f"episode_{time.strftime('%Y%m%d_%H%M%S')}"
            suffix = 0
            while True:
                candidate = name if suffix == 0 else f"{name}_{suffix}"
                destination, staging = (
                    self.data_dir / candidate,
                    self.data_dir / f".tmp_{candidate}",
                )
                if not os.path.lexists(destination) and not os.path.lexists(staging):
                    break
                if episode_name is not None:
                    raise FileExistsError(f"explicit episode name already exists: {destination}")
                suffix += 1
            staging.mkdir(exist_ok=False)
            self.staging_path = staging
            self.thread = threading.Thread(
                target=self.write_episode,
                args=(metadata,),
                name="episode-writer",
                daemon=False,
            )
            self.thread.start()
            if not self.ready.wait(RECORDER_START_TIMEOUT_S):
                raise TimeoutError("recording START timed out")
        except BaseException as exc:
            self.store_error(exc)
            if not isinstance(exc, Exception):
                raise
        # writer 已登记的失败只向调用方传播，不作为第二次启动失败登记。
        self.check_error()
        self.is_recording = True

    def append(self, row, command=None, *, quality_issue=None) -> str | None:
        """质量拒绝或队列满停止接纳，返回终段原因；写入错误原样传播。"""
        self.check_error()
        if not self.is_recording:
            if self.profiler is not None:
                self.profiler.count("recorder_append_inactive")
            return None
        if quality_issue is None:
            # 数组已脱离共享缓冲，入队后生产者不再修改。
            snapshot = replace(row, point_cloud=None)
            try:
                self.queue.put_nowait((snapshot, command, self.pending_segment_start))
            except Full:
                if self.profiler is not None:
                    self.profiler.count("recorder_queue_full")
                    self.profiler.maximum("recorder_queue_high_water_observed", self.queue.maxsize)
                quality_issue = "recording_queue_full"
            else:
                if self.profiler is not None:
                    self.profiler.count("recorder_enqueued_frames")
                    self.profiler.maximum("recorder_queue_high_water_observed", self.queue.qsize())
                self.pending_segment_start = False
                return None
        self.is_recording = False
        self.recording_truncated_reason = quality_issue
        if self.profiler is not None:
            self.profiler.count("recording_truncations")
        return quality_issue

    def finish(self, *, save, reason, wait=True) -> _FinishRequest | None:
        """收尾完成后返回首次处置请求，仍在写入时返回 None。成败由 result 报告。"""
        try:
            self.is_recording = False
            # 首次请求固定处置意图和整个等待预算；反复轮询不能延长 deadline。
            if self.finish_request is None:
                if self.profiler is not None:
                    self._profile_finish_ns = time.perf_counter_ns()
                self.finish_request = _FinishRequest(save, reason, time.monotonic() + RECORDER_STOP_TIMEOUT_S)
            request = self.finish_request
            while self.writer_pending:
                remaining = request.deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("recording writer did not exit before the deadline")
                if not wait:
                    return None
                # 中断 join 可能误报活线程已退出，不能据此发布文件。
                time.sleep(min(0.005, remaining))
            # 线程退出后才判定最终结果；close 的程序错误不能被 I/O 失败掩盖。
            if isinstance(self.error, RecordingError):
                return request
            self.check_error()
            if self.staging_path is None:
                return request
            try:
                if request.save and self.written_frames > 0:
                    destination = self.staging_path.with_name(self.staging_path.name.removeprefix(".tmp_"))
                    publish(self.staging_path, destination)
                    self.published_path, self.staging_path = destination, None
                    if self.recording_truncated_reason is not None:
                        logger.warning("部分录制已保存：%s（%d 行），%s。",
                                       destination, self.written_frames, self.recording_truncated_reason)
                    else:
                        logger.info("Raw 已保存：%s（%d 帧）。", destination, self.written_frames)
                else:
                    shutil.rmtree(self.staging_path)
                    self.staging_path = None
                    if request.save and self.written_frames == 0:
                        logger.info("未录到数据，本段未发布。")
            except BaseException as exc:
                self.store_error(exc)
                if not isinstance(self.error, RecordingError):
                    raise
            return request
        finally:
            if self.profiler is not None and not self.writer_pending and self._profile_finish_ns is not None:
                self.profiler.sample("recorder_finish_elapsed_ms",
                                     (time.perf_counter_ns() - self._profile_finish_ns) / 1e6)
                self._profile_finish_ns = None

    def write_episode(self, metadata):
        if self.profiler is not None:
            started = time.perf_counter_ns()
            self.profiler.count("recorder_writers_started")
        staging = self.staging_path
        finish = None
        try:
            self.writer = writer = RawWriter(staging, control_hz=self.control_hz)
            try:
                writer.open(metadata)
                self.ready.set()
                while self.error is None:
                    finish = self.finish_request
                    if finish is not None and not finish.save:
                        break
                    try:
                        item = (
                            self.queue.get(timeout=0.05)
                            if finish is None
                            else self.queue.get_nowait()
                        )
                    except Empty:
                        # 仅取帧前已登记的请求能保证生产者停止；超时后到达的
                        # 请求留到下轮处理，避免漏掉超时后刚入队的最后一帧。
                        if finish is not None:
                            break
                        continue
                    writer.append_frame(*item)
                    if self.profiler is not None and item[2]:
                        self.profiler.count("recording_gaps")
                if finish is None or self.error is not None:
                    writer.flush()
                elif finish.save:
                    writer.finish(
                        finish.reason,
                        recording_truncated_reason=self.recording_truncated_reason,
                    )
            except BaseException as exc:  # noqa: BLE001 — writer 线程的失败由控制线程读取并抛出。
                self.store_error(exc)
                self.ready.set()  # 启动失败无需等待可能仍在阻塞的 close。
            finally:
                writer.close()
        except BaseException as exc:  # noqa: BLE001 — 关闭失败也必须回传，不能发布不完整录制。
            self.store_error(exc, stage="writer_close")
        finally:
            self.ready.set()
            self.drain_frames()
            if self.profiler is not None:
                self.profiler.sample("recorder_writer_total_ms", (time.perf_counter_ns() - started) / 1e6)
                self.profiler.count("recorder_written_frames", self.written_frames)
                self.profiler.count("recorder_writers_completed")

    def drain_frames(self):
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                return
