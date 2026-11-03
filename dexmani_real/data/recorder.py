from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty, Full, Queue

from dexmani_real.data.raw import RawWriter, validate_task_label
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.runtime.observation import Observation
from dexmani_real.utils.io import publish
from dexmani_real.utils.timing import sampling_clock_ns

logger = logging.getLogger(__name__)
RECORDER_START_TIMEOUT_S = 10.0
RECORDER_STOP_TIMEOUT_S = 60.0


class RecordingError(RuntimeError):
    """录制 I/O 失败，由控制线程停止本段并退出会话。"""


@dataclass(frozen=True)
class _FinishRequest:
    reason: str
    details: tuple[dict, ...]
    recording_truncated_reason: str | None


class EpisodeRecorder:
    """每个 episode 独占一个 writer，调用方须先停止运动再阻塞收尾。"""

    def __init__(self, data_dir, *, control_hz):
        sampling_clock_ns(control_hz)
        self.data_dir = Path(data_dir)
        self.control_hz = float(control_hz)
        self.thread = None
        self.queue: Queue[tuple[Observation, RobotCommand | None] | _FinishRequest] = Queue(maxsize=16)
        self.ready = threading.Event()
        self.error = None
        self.is_recording = False
        self.recording_truncated_reason = None
        self.finish_request = None
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
        if self.error is None:
            self.error = error
        else:
            logger.error("%s also failed", stage, exc_info=(type(error), error, error.__traceback__))

    def check_error(self):
        if self.error is not None:
            if not isinstance(self.error, RecordingError):
                failure = RecordingError(f"Recording failed: {type(self.error).__name__}: {self.error}")
                raise failure from self.error
            raise self.error

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
        return dict(
            status=status,
            path=str(published) if published is not None else None,
            staging_path=str(staging) if staging is not None and published is None else None,
            row_count=self.written_frames,
            recording_truncated_reason=self.recording_truncated_reason,
        )

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
            self.check_error()
            self.is_recording = True
        except BaseException as exc:
            self.store_error(exc)
            if not isinstance(exc, Exception):
                raise
            self.check_error()

    def append(self, row, command=None, *, quality_issue=None) -> str | None:
        """质量拒绝返回原因，调用方立即结束本段；I/O 错误继续抛出。"""
        self.check_error()
        if not self.is_recording:
            return None
        if quality_issue is not None:
            self.recording_truncated_reason = quality_issue
            return quality_issue
        # 数组已脱离共享缓冲，入队后生产者不再修改。
        snapshot = replace(row, point_cloud=None)
        try:
            self.queue.put_nowait((snapshot, command))
        except Full:
            self.recording_truncated_reason = "recording_queue_full"
            return self.recording_truncated_reason
        return None

    def finish(self, *, save, reason, details=None):
        """关闭后由监督方发布；首次 finish 建立唯一收尾预算。"""
        if self.finish_request is not None:
            self.check_error()
            return self.published_path
        self.is_recording = False
        try:
            self.finish_request = request = _FinishRequest(
                reason, tuple(dict(item) for item in (details or ())),
                self.recording_truncated_reason,
            )
            # 建立 finish 后的中断也须记为失败，不能让重入误报已完成。
            deadline = time.monotonic() + RECORDER_STOP_TIMEOUT_S
            if self.writer_pending and self.error is None:
                self.queue.put(request, timeout=max(0.0, deadline - time.monotonic()))
            if self.thread is not None and self.thread.ident is not None:
                self.thread.join(timeout=max(0.0, deadline - time.monotonic()))
        except Full:
            self.store_error(TimeoutError("recording finish queue timed out"))
        except BaseException as exc:
            self.store_error(exc)
            if not isinstance(exc, Exception):
                raise
        if self.writer_pending:
            self.store_error(TimeoutError("recording writer did not exit before the deadline"))
        self.check_error()
        if self.staging_path is None:
            return None
        try:
            if save and self.written_frames > 0:
                destination = self.staging_path.with_name(self.staging_path.name.removeprefix(".tmp_"))
                publish(self.staging_path, destination)
                self.published_path, self.staging_path = destination, None
                if self.recording_truncated_reason is not None:
                    logger.warning("录制已截断：%s（%d 行），%s。",
                                   destination, self.written_frames, self.recording_truncated_reason)
                else:
                    logger.info("Raw 已保存：%s（%d 帧）。", destination, self.written_frames)
            else:
                shutil.rmtree(self.staging_path)
                self.staging_path = None
                if self.written_frames == 0:
                    logger.info("未录到数据，本段未发布。")
        except BaseException as exc:
            self.store_error(exc)
            if not isinstance(exc, Exception):
                raise
            self.check_error()
        return self.published_path

    def write_episode(self, metadata):
        staging = self.staging_path
        finish = None
        try:
            self.writer = writer = RawWriter(staging, control_hz=self.control_hz)
            try:
                writer.open(metadata)
                self.ready.set()
                while self.error is None:
                    try:
                        item = self.queue.get(timeout=0.05)
                    except Empty:
                        continue
                    if isinstance(item, _FinishRequest):
                        finish = item
                        break
                    writer.append_frame(*item)
                if finish is not None and self.error is None:
                    writer.finish(
                        finish.reason,
                        finish.details,
                        recording_truncated_reason=finish.recording_truncated_reason,
                    )
                else:
                    writer.flush()
            except BaseException as exc:
                self.store_error(exc)
            finally:
                writer.close()
        except BaseException as exc:
            self.store_error(exc, stage="writer_close")
        finally:
            self.ready.set()
            self.drain_frames()
            if self.error is not None:
                logger.error("Recording failed; staging retained: %s", staging)

    def drain_frames(self):
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                return
