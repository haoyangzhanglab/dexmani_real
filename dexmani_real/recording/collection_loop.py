"""CollectionLoop — data collection lifecycle orchestrator.

Provides methods for TeleopController to call at lifecycle key points:
  - start_episode() / should_record() / tick_recording() / stop_episode()
  - get_episode_summary() / discard_episode() / annotate_episode()

Does NOT own the control loop — TeleopController.run() still owns the main loop.
CollectionLoop provides method hooks invoked at the right lifecycle points.

Ref: data collection loop design — Phase 2 (Collection lifecycle).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.log import get_logger
from dexmani_real.recording.collection_config import CollectionConfig
from dexmani_real.recording.quality_flags import ALL_GOOD_MASK

if TYPE_CHECKING:
    from dexmani_real.recording.episode_recorder import EpisodeRecorder

logger = get_logger(__name__)


class CollectionLoop:
    """Orchestrates data collection lifecycle around EpisodeRecorder.

    Usage in TeleopController:
        self.collection = CollectionLoop(recorder, config)
        ...
        # On T→TELEOP transition:
        if should_record:
            self.collection.start_episode()

        # In _tick(), after computing action:
        self.collection.record_frame(state, action, vr_frame, quality_flags,
                                      camera_frame, T_base_eef)

        # On stop:
        summary = self.collection.stop_episode(success=True)
    """

    def __init__(
        self,
        recorder: EpisodeRecorder,
        config: CollectionConfig | None = None,
    ) -> None:
        self.recorder = recorder
        self.config = config or CollectionConfig()

        # Per-episode state
        self._episode_start_time: float | None = None
        self._episode_frame_count: int = 0
        self._episode_quality_ok_count: int = 0
        self._last_episode_path: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def start_episode(
        self,
        task_label: str | None = None,
        operator: str | None = None,
        tags: list[str] | None = None,
        camera_K: np.ndarray | None = None,
    ) -> bool:
        """Begin a new recording episode.

        Returns True if episode started successfully.
        """
        if self.is_recording:
            logger.warning("Episode already in progress, ignoring start_episode.")
            return False

        task = task_label or self.config.task_label
        op = operator or self.config.operator
        tags_list = tags or self.config.tags

        success = self.recorder.start_episode(
            task_label=task,
            operator=op,
            tags=tags_list,
            camera_K=camera_K,
        )

        if success:
            self._episode_start_time = time.perf_counter()
            self._episode_frame_count = 0
            self._episode_quality_ok_count = 0
            logger.info(
                "Episode started: task=%s operator=%s tags=%s",
                task, op, tags_list,
            )

        return success

    def record_frame(
        self,
        state: object,
        action: object,
        vr_frame: dict,
        quality_flags: int,
        camera_frame: dict | None = None,
        T_base_eef: np.ndarray | None = None,
    ) -> bool:
        """Record one frame and track quality statistics.

        Returns True if frame was recorded, False if skipped or episode not active.
        """
        if not self.is_recording:
            return False

        recorded = self.recorder.add_frame(
            state=state,
            action=action,
            vr_frame=vr_frame,
            quality_flags=quality_flags,
            camera_frame=camera_frame,
            T_base_eef=T_base_eef,
        )

        if recorded:
            self._episode_frame_count += 1
            if (quality_flags & ALL_GOOD_MASK) == ALL_GOOD_MASK:
                self._episode_quality_ok_count += 1

        return recorded

    def stop_episode(self, success: bool = True) -> str | None:
        """Stop recording and finalize the episode file.

        Returns the file path of the saved episode, or None if no episode was active.
        """
        if not self.is_recording:
            return None

        path = self.recorder.stop_episode(success=success)
        self._last_episode_path = path

        # Log summary
        duration = time.perf_counter() - (self._episode_start_time or 0.0)
        quality_ratio = (
            self._episode_quality_ok_count / max(self._episode_frame_count, 1)
        )
        logger.info(
            "Episode stopped: frames=%d duration=%.1fs quality=%.1f%% path=%s",
            self._episode_frame_count, duration,
            quality_ratio * 100, path,
        )

        return path

    def discard_episode(self) -> bool:
        """Discard the current (stopped) episode file.

        Returns True if file was deleted.
        """
        if self._last_episode_path is None:
            return False
        try:
            Path(self._last_episode_path).unlink(missing_ok=True)
            logger.info("Episode discarded: %s", self._last_episode_path)
            self._last_episode_path = None
            return True
        except OSError as e:
            logger.warning("Failed to discard episode: %s", e)
            return False

    def annotate_episode(
        self,
        success: bool | None = None,
        notes: str = "",
    ) -> None:
        """Post-recording annotation.

        Currently a no-op. Future: write annotation metadata to a sidecar
        JSON file or update HDF5 attributes.
        """
        if self._last_episode_path is None:
            logger.warning("No episode to annotate.")
            return
        # TODO: write sidecar annotation JSON
        logger.debug(
            "Episode annotation (no-op): success=%s notes=%s",
            success, notes,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return self.recorder.is_recording

    @property
    def frame_count(self) -> int:
        return self._episode_frame_count

    def get_episode_summary(self) -> dict:
        """Return a summary dict for the current/last episode.

        Useful for logging, monitoring, and pydantic validation.
        """
        duration = (
            time.perf_counter() - self._episode_start_time
            if self._episode_start_time is not None and self.is_recording
            else 0.0
        )
        quality_ratio = (
            self._episode_quality_ok_count / max(self._episode_frame_count, 1)
        )
        return {
            "frame_count": self._episode_frame_count,
            "quality_ok_count": self._episode_quality_ok_count,
            "quality_ratio": round(quality_ratio, 4),
            "duration_s": round(duration, 2),
            "is_recording": self.is_recording,
            "last_path": self._last_episode_path,
            "min_frames_met": self._episode_frame_count >= self.config.min_frames,
            "quality_ok": quality_ratio >= self.config.min_quality_ratio,
        }
