from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty, Full, Queue

import numpy as np

from dexmani_real.data.raw import RawWriter
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.runtime.observation import Observation
from dexmani_real.utils.io import publish

logger = logging.getLogger(__name__)
RECORDER_START_TIMEOUT_S = 10.0
RECORDER_STOP_TIMEOUT_S = 60.0
COLLECTION_SOURCES = frozenset({"teleop", "policy_rollout"})


class RecordingError(RuntimeError):
    """录制失败，由控制线程撤销运动授权。"""


@dataclass(frozen=True)
class _FinishRequest:
    reason: str
    details: tuple[dict, ...]
    technical_status: str


class EpisodeRecorder:
    """每个 episode 独占一个 writer，调用方须先停止运动再阻塞收尾。"""

    def __init__(self, data_dir, *, control_hz):
        if not np.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("recording control_hz must be finite and positive")
        self.data_dir = Path(data_dir)
        self.control_hz = float(control_hz)
        self.thread = None
        self.queue: Queue[tuple[Observation, RobotCommand | None] | _FinishRequest] = Queue(maxsize=16)
        self.ready = threading.Event()
        self.cancelled = threading.Event()
        self.error = None
        self.is_recording = False
        self.start_attempted = False
        self.finish_request = None
        self.closed_counts = None
        self.staging_path: Path | None = None
        self.published_path: Path | None = None
        self.submitted_frames = 0
        self.writer = None

    @property
    def written_frames(self):
        return self.writer.frame_count if self.writer is not None else 0

    @property
    def writer_pending(self):
        return self.thread is not None and self.thread.is_alive()

    def store_error(self, error):
        if self.error is None:
            self.error = error
        self.cancelled.set()

    def check_error(self):
        if self.error is not None:
            if isinstance(self.error, RecordingError):
                raise self.error
            raise RecordingError(f"Recording failed: {self.error}") from self.error

    def result(self):
        """仅报告最后可观察的结果，pending 不推断线程之后的结局。"""
        if not self.start_attempted:
            return None
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
        )

    def start(
        self,
        *,
        task_label,
        collection_source,
        metadata,
        episode_name=None,
    ):
        if self.start_attempted or self.finish_request is not None:
            raise RecordingError("recorder permits only one START attempt")
        self.start_attempted = True
        try:
            if not isinstance(task_label, str) or not task_label.strip():
                raise ValueError("task_label must be a non-empty string")
            if collection_source not in COLLECTION_SOURCES:
                raise ValueError(f"collection_source must be one of {sorted(COLLECTION_SOURCES)}")
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
            if self.thread is not None and self.thread.ident is not None:
                self.thread.join(timeout=RECORDER_STOP_TIMEOUT_S)
            if not isinstance(exc, Exception):
                raise
            self.check_error()

    def append(self, row, command=None):
        self.check_error()
        if not self.is_recording:
            return
        # 数组已脱离共享缓冲，入队后生产者不再修改。
        snapshot = replace(row, point_cloud=None)
        try:
            self.queue.put_nowait((snapshot, command))
        except Full as exc:
            error = RecordingError("recording queue full (capacity=16)")
            self.store_error(error)
            raise error from exc
        self.submitted_frames += 1

    def finish(self, *, save, reason, details=None, technical_status="failed"):
        """调用方已撤权并请求停机；本次决定在 writer 关闭后才生效。"""
        if self.finish_request is not None:
            self.check_error()
            return self.published_path
        was_recording, self.is_recording = self.is_recording, False
        self.finish_request = request = _FinishRequest(
            reason, tuple(dict(item) for item in (details or ())), technical_status,
        )
        deadline = time.monotonic() + RECORDER_STOP_TIMEOUT_S
        try:
            if was_recording and self.writer_pending and not self.cancelled.is_set():
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
            if self.closed_counts != (self.submitted_frames, self.submitted_frames):
                raise RecordingError("recording accepted/written/camera row count mismatch")
            if save:
                destination = self.staging_path.with_name(self.staging_path.name.removeprefix(".tmp_"))
                publish(self.staging_path, destination)
                self.published_path, self.staging_path = destination, None
                logger.info("Raw 已保存：%s（%d 帧）。", destination.name, self.written_frames)
            else:
                shutil.rmtree(self.staging_path)
                self.staging_path = None
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
                while not self.cancelled.is_set():
                    try:
                        item = self.queue.get(timeout=0.05)
                    except Empty:
                        continue
                    if isinstance(item, _FinishRequest):
                        finish = item
                        break
                    writer.append_frame(*item)
                if finish is not None and self.error is None:
                    counts = writer.finish(
                        finish.reason,
                        finish.details,
                        expected_frames=self.submitted_frames,
                        technical_status=finish.technical_status,
                    )
                else:
                    writer.flush()
            except BaseException as exc:
                if self.error is not None and exc is not self.error:
                    logger.exception("Additional recording worker failure")
                self.store_error(exc)
            finally:
                writer.close()
            if finish is not None and self.error is None:
                self.closed_counts = counts
        except BaseException as exc:
            if self.error is not None and exc is not self.error:
                logger.exception("Recording worker close also failed")
            self.store_error(exc)
        finally:
            self.ready.set()
            self.drain_frames()
            if self.error is not None:
                logger.error(
                    "Recording failed; staging retained: %s",
                    staging,
                    exc_info=(type(self.error), self.error, self.error.__traceback__),
                )

    def drain_frames(self):
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                return
