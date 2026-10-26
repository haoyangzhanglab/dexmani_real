"""Nonblocking control-row submission and one streaming writer per Raw episode."""

from __future__ import annotations

import logging
import shutil
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from types import SimpleNamespace

import numpy as np

from dexmani_real.data.raw.schema import (
    TACTILE_AXES,
    TACTILE_FINGER_ORDER,
    TACTILE_PROCESSING,
    TACTILE_UNITS,
)
from dexmani_real.data.raw.writer import RawEpisodeWriter
from dexmani_real.utils.atomic_io import (
    atomic_json_dump,
    atomic_publish,
    target_is_occupied,
)

logger = logging.getLogger(__name__)
RECORDER_START_TIMEOUT_S = 10.0
RECORDER_STOP_TIMEOUT_S = 60.0
COLLECTION_SOURCES = frozenset({"teleop", "policy_rollout"})


class RecordingError(RuntimeError):
    """Required recording failed; the control owner must revoke motion."""


@dataclass(frozen=True)
class CaptureFinishRequest:
    save: bool
    reason: str
    details: object
    technical_status: str = "failed"


def snapshot_recording_metadata(camera_snapshot, *, tactile_reset_metadata, tactile_hold_max_age_s):
    """复用初始化相机快照，补入当前 capture 消费的触觉处理语义。"""
    metadata = camera_snapshot.metadata()
    metadata.update(tactile_reset_metadata)
    metadata.update(
        hand_tactile_processing=TACTILE_PROCESSING,
        hand_tactile_units=TACTILE_UNITS,
        hand_tactile_axes=TACTILE_AXES,
        hand_tactile_finger_order=TACTILE_FINGER_ORDER,
        hand_tactile_hold_max_age_s=float(tactile_hold_max_age_s),
    )
    return metadata


def snapshot_recording_result(recorder: AsyncEpisodeRecorder, *, failed: bool = False):
    """只读最后可观察的录制结果；pending 不推断线程之后的发布结局。"""
    if not recorder.start_attempted:
        return None
    pending = recorder.writer_pending
    published, staging = recorder.published_path, recorder.staging_path
    return dict(
        status=(
            "published" if published is not None else "pending" if pending
            else "failed" if failed or recorder.error is not None else "not_published"
        ),
        path=str(published) if published is not None else None,
        staging_path=str(staging) if staging is not None and published is None else None,
        row_count=recorder.written_frames,
    )


class AsyncEpisodeRecorder:
    """单次 START 的 FIFO writer；阻塞收尾前由调用方撤销运动授权。"""

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
        self.io_failed = False
        self.is_recording = False
        self.start_attempted = False
        self.finish_request = None
        self.staging_path = self.episode_dir = self.published_path = None
        self.submitted_frames = 0
        self.writer = None

    @property
    def written_frames(self):
        return self.writer.frame_count if self.writer is not None else 0

    @property
    def writer_pending(self):
        return self.thread is not None and self.thread.is_alive()

    def store_error(self, error, *, io=False):
        if io:
            self.io_failed = True
        if self.error is None:
            self.error = error

    def check_error(self):
        if self.error is not None:
            raise RecordingError(f"Recording failed: {self.error}") from self.error

    def start_episode(
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
            metadata = dict(
                deepcopy(metadata), task_label=task_label, collection_source=collection_source
            )
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
                if not target_is_occupied(destination) and not target_is_occupied(staging):
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
            self.cancelled.set()
            if self.thread is not None and self.thread.ident is not None:
                self.thread.join(timeout=RECORDER_STOP_TIMEOUT_S)
            if not isinstance(exc, Exception):
                raise
            raise RecordingError(f"recording START failed: {exc}") from exc

    def add_frame(self, row, command=None):
        """Snapshot mutable inputs once; live observations already own read-only arrays."""
        def owned(value):
            if isinstance(value, np.ndarray) and value.flags.writeable:
                value = value.copy()
                value.flags.writeable = False
            return value

        try:
            self.check_error()
            if not self.is_recording:
                return
            snapshot = SimpleNamespace(
                arm=owned(row.arm), hand=owned(row.hand),
                camera=(
                    {name: owned(value) for name, value in row.camera.items()}
                    if row.camera is not None else None
                ),
                observation_timestamp_ns=row.observation_timestamp_ns,
                control_timestamp_ns=row.control_timestamp_ns,
            )
            self.queue.put_nowait((snapshot, command))
            self.submitted_frames += 1
        except Full as exc:
            error = RecordingError("recording queue full (capacity=16)")
            self.store_error(error)
            raise error from exc

    def close(self):
        """Join an active writer without inventing a physical stop confirmation."""
        if self.finish_request is None or self.writer_pending:
            self.finish(save=True, reason="close")

    def cancel_publication(self):
        """提交点之后不可撤销；协调线程不得等待发布锁。"""
        if not self.publication_lock.acquire(blocking=False):
            return False
        try:
            if self.published_path is not None:
                return False
            self.cancelled.set()
            return True
        finally:
            self.publication_lock.release()

    def finish(
        self, *, save, reason, details=None, technical_status="failed"
    ):
        was_recording, self.is_recording = self.is_recording, False
        thread = self.thread
        deadline = time.monotonic() + RECORDER_STOP_TIMEOUT_S
        interrupted = pending = None
        try:
            if self.finish_request is None:
                try:
                    self.finish_request = CaptureFinishRequest(
                        save, reason, deepcopy(details), technical_status
                    )
                except BaseException as exc:
                    self.store_error(exc, io=True)
                    self.finish_request = CaptureFinishRequest(save, reason, None)
                    if not isinstance(exc, Exception):
                        interrupted = exc
            if was_recording and self.writer_pending and not self.cancelled.is_set():
                self.queue.put(self.finish_request, timeout=max(0.0, deadline - time.monotonic()))
        except BaseException as exc:
            self.store_error(exc)
            self.cancel_publication()
            if not isinstance(exc, Exception):
                interrupted = exc
        finally:
            if thread is not None and thread.ident is not None:
                try:
                    thread.join(timeout=max(0.0, deadline - time.monotonic()))
                except BaseException as exc:
                    self.store_error(exc)
                    self.cancel_publication()
                    if not isinstance(exc, Exception):
                        interrupted = exc
                    thread.join(timeout=max(0.0, deadline - time.monotonic()))
                if self.writer_pending:
                    cancelled = self.cancel_publication()
                    publication = (
                        "cancelled" if cancelled
                        else "published" if self.published_path is not None else "unresolved"
                    )
                    pending = TimeoutError(
                        f"recording finalization timed out; status=pending; publication={publication}; "
                        "writer retains ownership"
                    )
                    if cancelled:
                        self.store_error(pending)
        if not self.writer_pending:
            # Writer 的末次 drain 之后仍可能有已越过 check_error 的提交。
            self.drain_frames()
        if interrupted is not None:
            raise interrupted
        if pending is not None:
            raise RecordingError(str(pending)) from pending
        if not self.finish_request.save:
            if self.staging_path is not None:
                shutil.rmtree(self.staging_path)
                self.staging_path = None
            return None
        if self.staging_path is not None and self.error is not None:
            failure_path = self.staging_path / "capture_failure.json"
            if not failure_path.exists():
                atomic_json_dump(
                    dict(
                        reason=self.finish_request.reason, technical_status="failed",
                        error=str(self.error),
                        accepted_frames=self.submitted_frames,
                        written_frames=self.written_frames,
                    ),
                    failure_path, overwrite=False,
                )
        if self.io_failed or self.published_path is None:
            self.check_error()
        return self.published_path

    def write_episode(self, metadata):
        staging, destination = self.staging_path, self.episode_dir
        finish = None
        try:
            self.writer = writer = RawEpisodeWriter(staging, control_hz=self.control_hz)
            try:
                writer.open(metadata)
                self.ready.set()
                try:
                    while not self.cancelled.is_set():
                        try:
                            item = self.queue.get(timeout=0.05)
                        except Empty:
                            continue
                        if isinstance(item, CaptureFinishRequest):
                            finish = item
                            break
                        writer.append_frame(*item)
                except BaseException as exc:
                    # Notify control before flushing or closing potentially blocking native I/O.
                    self.store_error(exc, io=True)
                writer.flush()
                reason = finish.reason if finish is not None else "recording_failure"
                details = list(finish.details or []) if finish is not None else []
                if self.error is not None:
                    details.append(
                        dict(
                            stage="recording",
                            exception_type=type(self.error).__name__,
                            message=str(self.error),
                        )
                    )
                writer.write_termination(
                    reason, details,
                    expected_frames=self.submitted_frames if not self.io_failed else None,
                    technical_status=(
                        finish.technical_status
                        if finish is not None and self.error is None else "failed"
                    ),
                )
            finally:
                writer.close()
            with self.publication_lock:
                if (
                    not self.io_failed and not self.cancelled.is_set()
                    and finish is not None and finish.save
                ):
                    # 锁内决策是不可撤销提交点，覆盖 rename 与结果登记。
                    try:
                        atomic_publish(staging, destination)
                    except BaseException as exc:
                        # 先保留真实发布错误，再允许超时取消取得锁。
                        self.store_error(exc, io=True)
                        raise
                    self.published_path, self.staging_path = destination, None
            if self.published_path is not None:
                logger.info("Raw 已保存：%s（%d 帧）。", destination.name, self.written_frames)
                logger.debug(
                    "Episode saved: %s frames=%d reason=%s",
                    destination, self.written_frames, finish.reason,
                )
        except BaseException as exc:
            self.store_error(exc, io=True)
            logger.exception("Recording failed; staging retained: %s", staging)
        finally:
            self.ready.set()
            if self.error is not None and self.published_path is None:
                logger.error(
                    "Recording staging retained: %s frames=%d", staging, self.written_frames
                )
            self.drain_frames()

    def drain_frames(self):
        """Release queued snapshots after the writer stops consuming rows."""
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                return
