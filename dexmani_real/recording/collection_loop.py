"""CollectionLoop — data collection lifecycle orchestrator.

Provides methods for TeleopController to call at lifecycle key points:
  - start_episode() / record_frame() / stop_episode()
  - get_episode_summary() / discard_episode() / annotate_episode()

Does NOT own the control loop — TeleopController.run() still owns the main loop.
CollectionLoop provides method hooks invoked at the right lifecycle points.

Ref: data collection loop design — Phase 2 (Collection lifecycle).
"""

from __future__ import annotations

import json
import time
from datetime import datetime
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
        self._stopped_reason: str = "manual"
        self._low_quality_streak: int = 0   # consecutive low-quality frame counter

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
            self._stopped_reason = "manual"
            self._low_quality_streak = 0
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

            # ── Auto-stop on quality drop ──
            if self.config.auto_stop_on_quality_drop:
                current_ratio = (
                    self._episode_quality_ok_count
                    / max(self._episode_frame_count, 1)
                )
                if current_ratio < self.config.quality_drop_threshold:
                    self._low_quality_streak += 1
                    if self._low_quality_streak >= self.config.quality_drop_streak:
                        logger.warning(
                            "Auto-stopping: %d consecutive frames below quality"
                            " threshold %.0f%% (current ratio=%.1f%%)",
                            self._low_quality_streak,
                            self.config.quality_drop_threshold * 100,
                            current_ratio * 100,
                        )
                        self.stop_episode(success=False, reason="quality_drop")
                else:
                    self._low_quality_streak = 0

        return recorded

    def stop_episode(
        self, success: bool = True, reason: str | None = None,
    ) -> str | None:
        """Stop recording and finalize the episode file.

        Args:
            success: Whether the episode was completed successfully.
            reason: Why the episode stopped (manual, max_frames, quality_drop, error).
                    If None, uses the previously set _stopped_reason.

        Returns the file path of the saved episode, or None if no episode was active.
        """
        if not self.is_recording:
            return None

        if reason is not None:
            self._stopped_reason = reason

        path = self.recorder.stop_episode(success=success)
        self._last_episode_path = path

        # Log summary
        duration = time.perf_counter() - (self._episode_start_time or 0.0)
        quality_ratio = (
            self._episode_quality_ok_count / max(self._episode_frame_count, 1)
        )
        logger.info(
            "Episode stopped: frames=%d duration=%.1fs quality=%.1f%% "
            "reason=%s path=%s",
            self._episode_frame_count, duration,
            quality_ratio * 100, self._stopped_reason, path,
        )

        # ── Write sidecar annotation JSON ──
        if self.config.annotate_on_save and path is not None:
            self._write_sidecar_json(path, success, duration, quality_ratio)

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
        tags: list[str] | None = None,
    ) -> None:
        """Post-recording annotation — write/update sidecard JSON and HDF5 attrs.

        If the sidecar JSON already exists (written by stop_episode), this
        updates it. Otherwise a new file is created.
        """
        if self._last_episode_path is None:
            logger.warning("No episode to annotate.")
            return

        # Update sidecar JSON with additional fields
        sidecar_path = self._sidecar_path(self._last_episode_path)
        if sidecar_path.exists():
            try:
                with open(sidecar_path, "r") as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError):
                meta = {}
        else:
            meta = {}

        if success is not None:
            meta["success"] = success
        if notes:
            meta["notes"] = notes
        if tags is not None:
            meta["tags"] = list(set(meta.get("tags", []) + tags))

        try:
            with open(sidecar_path, "w") as f:
                json.dump(meta, f, indent=2)
            logger.info("Episode annotated: %s", sidecar_path)
        except OSError as e:
            logger.warning("Failed to write annotation JSON: %s", e)

        # Also update the HDF5 metadata using EpisodeAnnotator
        from dexmani_real.recording.episode_annotator import EpisodeAnnotator
        EpisodeAnnotator.annotate(
            self._last_episode_path,
            success=success,
            tags=tags,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sidecar_path(h5_path: str) -> Path:
        """Derive the sidecar JSON path from the HDF5 path."""
        p = Path(h5_path)
        return p.with_suffix(".json")

    def _write_sidecar_json(
        self,
        h5_path: str,
        success: bool,
        duration_s: float,
        quality_ratio: float,
    ) -> None:
        """Write episode metadata sidecar JSON next to the HDF5 file.

        Contains: episode_id, task_label, operator, duration_s, num_frames,
        quality_ratio, success, tags, stopped_reason, timestamp.
        """
        sidecar_path = self._sidecar_path(h5_path)
        meta = {
            "episode_id": Path(h5_path).stem,
            "task_label": self.config.task_label,
            "operator": self.config.operator,
            "tags": self.config.tags,
            "duration_s": round(duration_s, 2),
            "num_frames": self._episode_frame_count,
            "quality_ratio": round(quality_ratio, 4),
            "quality_ok_count": self._episode_quality_ok_count,
            "success": success,
            "stopped_reason": self._stopped_reason,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            with open(sidecar_path, "w") as f:
                json.dump(meta, f, indent=2)
            logger.debug("Sidecar annotation written: %s", sidecar_path)
        except OSError as e:
            logger.warning("Failed to write sidecar annotation: %s", e)

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
