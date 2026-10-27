from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from types import SimpleNamespace

import numpy as np

from dexmani_real.data.raw import RawWriter
from dexmani_real.utils.io import publish

logger = logging.getLogger(__name__)
RECORDER_START_TIMEOUT_S = 10.0
RECORDER_STOP_TIMEOUT_S = 60.0
COLLECTION_SOURCES = frozenset({"teleop", "policy_rollout"})


class RecordingError(RuntimeError):
    """录制失败，由控制线程撤销运动授权。"""


@dataclass(frozen=True)
class _FinishRequest:
    save: bool
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
        self.queue = Queue(maxsize=16)
        self.ready = threading.Event()
        self.cancelled = threading.Event()
        self.publication_lock = threading.Lock()
        self.error = None
        self.is_recording = False
        self.start_attempted = False
        self.finish_request = None
        self.stop_deadline = None
        self.staging_path = self.episode_dir = self.published_path = None
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
            raise RecordingError(f"Recording failed: {self.error}") from self.error

    def result(self, *, failed=False):
        """仅报告最后可观察的结果，pending 不推断线程之后的结局。"""
        if not self.start_attempted:
            return None
        pending = self.writer_pending
        published, staging = self.published_path, self.staging_path
        if published is not None:
            status = "published"
        elif pending:
            status = "pending"
        elif failed or self.error is not None:
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
            self.staging_path, self.episode_dir = staging, destination
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
            self.stop_deadline = time.monotonic() + RECORDER_STOP_TIMEOUT_S
            if self.thread is not None and self.thread.ident is not None:
                self.thread.join(timeout=max(0.0, self.stop_deadline - time.monotonic()))
            if not isinstance(exc, Exception):
                raise
            self.check_error()

    def append(self, row, command=None):
        self.check_error()
        if not self.is_recording:
            return
        # 数组已脱离共享缓冲，入队后生产者不再修改。
        snapshot = SimpleNamespace(
            arm=row.arm,
            hand=row.hand,
            camera=row.camera,
            observation_timestamp_ns=row.observation_timestamp_ns,
            control_timestamp_ns=row.control_timestamp_ns,
        )
        try:
            self.queue.put_nowait((snapshot, command))
        except Full as exc:
            error = RecordingError("recording queue full (capacity=16)")
            self.store_error(error)
            raise error from exc
        self.submitted_frames += 1

    def close(self):
        if self.start_attempted:
            self.finish(save=True, reason="close")

    def cancel_publication(self):
        """提交点之后不可撤销；控制线程不得等待发布锁。"""
        if not self.publication_lock.acquire(blocking=False):
            return False
        try:
            if self.published_path is not None:
                return False
            self.cancelled.set()
            return True
        finally:
            self.publication_lock.release()

    def finish(self, *, save, reason, details=None, technical_status="failed"):
        was_recording, self.is_recording = self.is_recording, False
        first_request = self.finish_request is None
        if first_request:
            self.finish_request = _FinishRequest(
                save,
                reason,
                tuple(dict(item) for item in (details or ())),
                technical_status,
            )
        if self.stop_deadline is None:
            self.stop_deadline = time.monotonic() + RECORDER_STOP_TIMEOUT_S
        request = self.finish_request
        if not request.save:
            self.cancel_publication()

        try:
            if (
                first_request
                and was_recording
                and self.writer_pending
                and not self.cancelled.is_set()
            ):
                self.queue.put(request, timeout=max(0.0, self.stop_deadline - time.monotonic()))
            if self.thread is not None and self.thread.ident is not None:
                self.thread.join(timeout=max(0.0, self.stop_deadline - time.monotonic()))
        except Full:
            self.store_error(TimeoutError("recording finalization pending; finish queue timed out"))
        except BaseException as exc:
            self.cancel_publication()
            self.store_error(exc)
            if not isinstance(exc, Exception):
                raise
            self.check_error()

        if self.writer_pending:
            cancelled = self.cancel_publication()
            error = TimeoutError(
                f"recording finalization pending; publication_cancelled={cancelled}"
            )
            if cancelled:
                self.store_error(error)
            raise RecordingError(str(error)) from error

        self.drain_frames()
        if not request.save and self.staging_path is not None:
            shutil.rmtree(self.staging_path)
            self.staging_path = None
        self.check_error()
        return self.published_path

    def write_episode(self, metadata):
        staging, destination = self.staging_path, self.episode_dir
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
            except BaseException as exc:
                # 先撤销发布，再进入可能阻塞的资源收尾。
                self.store_error(exc)
            try:
                if finish is not None and self.error is None:
                    writer.finish(
                        finish.reason,
                        finish.details,
                        expected_frames=self.submitted_frames,
                        technical_status=finish.technical_status,
                    )
                else:
                    writer.flush()
            except BaseException as exc:
                self.store_error(exc)
            finally:
                writer.close()

            with self.publication_lock:
                if (
                    self.error is None
                    and not self.cancelled.is_set()
                    and finish is not None
                    and finish.save
                ):
                    publish(staging, destination)
                    self.published_path, self.staging_path = destination, None
            if self.published_path is not None:
                logger.info("Raw 已保存：%s（%d 帧）。", destination.name, self.written_frames)
        except BaseException as exc:
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
