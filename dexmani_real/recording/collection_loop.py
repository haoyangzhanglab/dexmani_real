"""CollectionLoop — data collection lifecycle orchestrator (simplified).

Provides hooks for TeleopController: start_episode / record_frame / stop_episode.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from dexmani_real.recording.collection_config import CollectionConfig
from dexmani_real.utils.log import get_logger

if TYPE_CHECKING:
    from dexmani_real.config.camera_calib import CameraCalib
    from dexmani_real.recording.episode_recorder import EpisodeRecorder

logger = get_logger(__name__)


class CollectionLoop:
    """Orchestrates data collection lifecycle around EpisodeRecorder."""

    def __init__(
        self,
        recorder: EpisodeRecorder,
        config: CollectionConfig | None = None,
    ) -> None:
        self.recorder = recorder
        self.config = config or CollectionConfig()
        self._episode_start_time: float | None = None
        self._episode_frame_count: int = 0
        self._last_episode_path: str | None = None
        self._stopped_reason: str = "manual"
        self._classification: str = "success"
        self._held_count: int = 0
        # Effective per-episode metadata (resolved at start_episode) — the
        # sidecar must record what the HDF5 /meta got, not the config defaults.
        self._task: str = self.config.task_label
        self._operator: str = self.config.operator
        self._tags: list[str] = self.config.tags

    def start_episode(
        self,
        task_label: str | None = None,
        operator: str | None = None,
        tags: list[str] | None = None,
        camera_K: np.ndarray | None = None,
        calib: CameraCalib | None = None,
        camera_name: str | None = None,
        depth_scale: float | None = None,
        record_config: dict | None = None,
    ) -> bool:
        if self.is_recording:
            logger.warning("Episode already in progress.")
            return False

        task = task_label or self.config.task_label
        op = operator or self.config.operator
        tags_list = tags or self.config.tags
        self._task = task
        self._operator = op
        self._tags = tags_list

        success = self.recorder.start_episode(
            task_label=task,
            operator=op,
            tags=tags_list,
            camera_K=camera_K,
            calib=calib,
            camera_name=camera_name,
            depth_scale=depth_scale,
            record_config=record_config,
            skip_initial_frames=self.config.skip_initial_frames,
        )

        if success:
            self._episode_start_time = time.perf_counter()
            self._episode_frame_count = 0
            self._held_count = 0
            self._stopped_reason = "manual"
            self._classification = "success"
            logger.info("Episode started: task=%s operator=%s", task, op)

        return success

    def record_frame(
        self,
        state: object,
        action: object,
        vr_frame: dict,
        camera_frame: dict | None = None,
        camera_frames: dict[str, dict] | None = None,
        signals: dict | None = None,
        arm_qpos_sent: np.ndarray | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> bool:
        if not self.is_recording:
            return False

        recorded = self.recorder.add_frame(
            state=state,
            action=action,
            vr_frame=vr_frame,
            camera_frame=camera_frame,
            camera_frames=camera_frames,
            signals=signals,
            arm_qpos_sent=arm_qpos_sent,
            diagnostics=diagnostics,
        )

        if self.recorder.max_frames_reached and self.config.auto_stop_on_max_frames:
            logger.warning("Auto-stopping at max_frames=%d", self.recorder.max_frames)
            self.stop_episode(success=True, reason="max_frames")
            return False

        if recorded:
            self._episode_frame_count += 1
            if signals and signals.get("held"):
                self._held_count += 1
        return recorded

    def stop_episode(
        self,
        success: bool = True,
        reason: str | None = None,
        classification: str | None = None,
    ) -> str | None:
        if not self.is_recording:
            return None

        if reason is not None:
            self._stopped_reason = reason
        if classification is not None:
            self._classification = classification

        path = self.recorder.stop_episode(success=success, reason=self._stopped_reason)
        self._last_episode_path = path

        duration = time.perf_counter() - (self._episode_start_time or 0.0)

        # Block until the HDF5 is fully written + closed — the sidecar and log
        # below must describe a durable file.  Runs on the EpisodeRecorder
        # writer thread (or the caller's thread for direct CollectionLoop users).
        if not self.recorder.join_stop():
            logger.warning("Episode flush still running after join timeout: %s", path)

        if self.config.save_sidecar_json and path is not None:
            self._write_sidecar_json(path, duration)

        logger.info(
            "Episode stopped: frames=%d duration=%.1fs reason=%s path=%s",
            self._episode_frame_count,
            duration,
            self._stopped_reason,
            path,
        )

        return path

    def discard_episode(self) -> bool:
        if self._last_episode_path is None:
            return False
        try:
            h5_path = Path(self._last_episode_path)
            h5_path.unlink(missing_ok=True)
            h5_path.with_suffix(".json").unlink(missing_ok=True)
            # Sidecar video files (.rgb.mp4, etc.) — EpisodeRecorder's
            # _stop_episode_impl_inner already discards them from the temp
            # path when success=False; this is belt-and-suspenders for any
            # code path where the final-path file may still exist.
            from dexmani_real.recording.episode_recorder import EpisodeRecorder

            for suffix in EpisodeRecorder._SIDECAR_SUFFIXES:
                h5_path.with_suffix(suffix).unlink(missing_ok=True)
            logger.info("Episode discarded: %s", self._last_episode_path)
            self._last_episode_path = None
            return True
        except OSError as e:
            logger.warning("Failed to discard episode: %s", e)
            return False

    def _write_sidecar_json(self, h5_path: str, duration_s: float) -> None:
        h5_file = Path(h5_path)
        json_path = h5_file.with_suffix(".json")

        metadata = {
            "frame_count": self._episode_frame_count,
            "duration_s": round(duration_s, 2),
            "task_label": self._task,
            "operator": self._operator,
            "tags": self._tags,
            "classification": self._classification,
            "stopped_reason": self._stopped_reason,
            "min_frames_met": self._episode_frame_count >= self.config.min_frames,
            "held_ratio": round(self._held_count / max(1, self._episode_frame_count), 3),
            "h5_file": h5_file.name,
        }

        try:
            json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("Sidecar JSON written: %s", json_path)
        except OSError as e:
            logger.warning("Failed to write sidecar JSON: %s", e)

    @property
    def is_recording(self) -> bool:
        return self.recorder.is_recording

    @property
    def frame_count(self) -> int:
        return self._episode_frame_count

    def get_episode_summary(self) -> dict:
        duration = (
            time.perf_counter() - self._episode_start_time
            if self._episode_start_time is not None and self.is_recording
            else 0.0
        )
        return {
            "frame_count": self._episode_frame_count,
            "duration_s": round(duration, 2),
            "is_recording": self.is_recording,
            "last_path": self._last_episode_path,
            "min_frames_met": self._episode_frame_count >= self.config.min_frames,
            "held_ratio": round(self._held_count / max(1, self._episode_frame_count), 3),
        }
