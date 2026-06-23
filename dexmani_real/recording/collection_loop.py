"""CollectionLoop — data collection lifecycle orchestrator.

Provides methods for TeleopController to call at lifecycle key points:
  - start_episode() / record_frame() / stop_episode()
  - get_episode_summary() / discard_episode()

Does NOT own the control loop — TeleopController.run() still owns the main loop.
CollectionLoop provides method hooks invoked at the right lifecycle points.

Ref: data collection loop design — Phase 2 (Collection lifecycle).
     T-Rex data_writer.py move_episode_files() — file classification.
"""

from __future__ import annotations

import json
import shutil
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.log import get_logger
from dexmani_real.recording.collection_config import CollectionConfig

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
        self.collection.record_frame(state, action, vr_frame,
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
        self._last_episode_path: str | None = None
        self._stopped_reason: str = "manual"
        self._classification: str = "success"

        # ── Pre-record ring buffer (Phase 3.1) ──
        # Buffers the last N seconds of frames before the Record key is pressed.
        # On start_episode(), buffered frames are flushed into the HDF5 file.
        # Stores dicts (not full numpy arrays) to keep memory low.
        pre_record_frames = int(self.config.pre_record_duration_s * 50)
        self._pre_record_deque: deque[dict] | None = (
            deque(maxlen=max(pre_record_frames, 1)) if pre_record_frames > 0 else None
        )

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
            self._stopped_reason = "manual"
            self._classification = "success"
            logger.info(
                "Episode started: task=%s operator=%s tags=%s",
                task,
                op,
                tags_list,
            )

        # ── Flush pre-record buffer (Phase 3.1) ──
        if success and self._pre_record_deque is not None and len(self._pre_record_deque) > 0:
            pre_count = len(self._pre_record_deque)
            for frame_data in self._pre_record_deque:
                self.recorder.add_frame(**frame_data)
            logger.info(
                "Pre-record buffer flushed: %d frames (%.1fs)",
                pre_count,
                pre_count / 50.0,
            )
            self._pre_record_deque.clear()

        return success

    def add_pre_frame(
        self,
        state: object,
        action: object,
        vr_frame: dict,
        camera_frame: dict | None = None,
        T_base_eef: np.ndarray | None = None,
        camera_frames: dict[str, dict] | None = None,
    ) -> None:
        """Store a frame in the pre-record ring buffer.

        Called on every tick regardless of recording state.  When
        start_episode() is later called, buffered frames are flushed
        to the HDF5 file before normal recording begins.

        If pre_record_duration_s is 0 (disabled), this is a no-op.
        """
        if self._pre_record_deque is None:
            return
        self._pre_record_deque.append(
            {
                "state": state,
                "action": action,
                "vr_frame": vr_frame,
                "camera_frame": camera_frame,
                "T_base_eef": T_base_eef,
                "camera_frames": camera_frames,
            }
        )

    def record_frame(
        self,
        state: object,
        action: object,
        vr_frame: dict,
        camera_frame: dict | None = None,
        T_base_eef: np.ndarray | None = None,
        camera_frames: dict[str, dict] | None = None,
    ) -> bool:
        """Record one frame.

        Returns True if frame was recorded, False if skipped or episode not active.
        """
        if not self.is_recording:
            return False

        recorded = self.recorder.add_frame(
            state=state,
            action=action,
            vr_frame=vr_frame,
            camera_frame=camera_frame,
            T_base_eef=T_base_eef,
            camera_frames=camera_frames,
        )

        if recorded:
            self._episode_frame_count += 1

        return recorded

    def stop_episode(
        self,
        success: bool = True,
        reason: str | None = None,
        classification: str | None = None,
        ik_success_rate: float | None = None,
        vr_drop_rate: float | None = None,
        ik_miss_count: int | None = None,
        ik_miss_max_consecutive: int = 0,
        camera_frame_rate: float | None = None,
    ) -> str | None:
        """Stop recording and finalize the episode file.

        Args:
            success: Whether the episode was completed successfully.
            reason: Why the episode stopped (manual, max_frames, error).
                    If None, uses the previously set _stopped_reason.
            classification: "success", "failure", or "partial" — controls
                    file routing to success_dir / failure_dir.
            ik_success_rate: Optional IK success rate (0.0–1.0) for metadata.
            vr_drop_rate: Optional VR frame drop rate (0.0–1.0) for metadata.
            ik_miss_count: Total IK miss count (all frames, not just consecutive).
            ik_miss_max_consecutive: Maximum consecutive IK misses during episode.
            camera_frame_rate: Average camera frame rate during episode (Hz).

        Returns the file path of the saved episode, or None if no episode was active.
        """
        if not self.is_recording:
            return None

        if reason is not None:
            self._stopped_reason = reason
        if classification is not None:
            self._classification = classification

        path = self.recorder.stop_episode(success=success)
        self._last_episode_path = path

        duration = time.perf_counter() - (self._episode_start_time or 0.0)

        # ── Sidecar JSON (Phase 1.2) ──
        if self.config.save_sidecar_json and path is not None:
            self._write_sidecar_json(
                path,
                duration,
                ik_success_rate=ik_success_rate,
                vr_drop_rate=vr_drop_rate,
                ik_miss_count=ik_miss_count,
                ik_miss_max_consecutive=ik_miss_max_consecutive,
                camera_frame_rate=camera_frame_rate,
            )

        # ── File classification routing (Phase 1.2) ──
        if path is not None:
            moved_path = self._route_episode_file(path)
            if moved_path is not None:
                path = moved_path
                self._last_episode_path = path

        # Log summary
        logger.info(
            "Episode stopped: frames=%d duration=%.1fs reason=%s classification=%s path=%s",
            self._episode_frame_count,
            duration,
            self._stopped_reason,
            self._classification,
            path,
        )

        return path

    def discard_episode(self) -> bool:
        """Discard the current (stopped) episode file and its sidecar JSON.

        Returns True if file was deleted.
        """
        if self._last_episode_path is None:
            return False
        try:
            h5_path = Path(self._last_episode_path)
            h5_path.unlink(missing_ok=True)
            # Also delete sidecar JSON if present
            json_path = h5_path.with_suffix(".json")
            json_path.unlink(missing_ok=True)
            logger.info("Episode discarded: %s", self._last_episode_path)
            self._last_episode_path = None
            return True
        except OSError as e:
            logger.warning("Failed to discard episode: %s", e)
            return False

    # ------------------------------------------------------------------
    # Sidecar JSON (Phase 1.2)
    # ------------------------------------------------------------------

    def _write_sidecar_json(
        self,
        h5_path: str,
        duration_s: float,
        ik_success_rate: float | None = None,
        vr_drop_rate: float | None = None,
        ik_miss_count: int | None = None,
        ik_miss_max_consecutive: int = 0,
        camera_frame_rate: float | None = None,
    ) -> None:
        """Write episode metadata as JSON sidecar next to the HDF5 file.

        Ref: T-Rex data_writer.py — full metadata recording.
        """
        h5_file = Path(h5_path)
        json_path = h5_file.with_suffix(".json")

        metadata = {
            "frame_count": self._episode_frame_count,
            "duration_s": round(duration_s, 2),
            "task_label": self.config.task_label,
            "operator": self.config.operator,
            "tags": self.config.tags,
            "classification": self._classification,
            "stopped_reason": self._stopped_reason,
            "min_frames_met": self._episode_frame_count >= self.config.min_frames,
            "h5_file": h5_file.name,
        }

        if ik_success_rate is not None:
            metadata["ik_success_rate"] = round(ik_success_rate, 4)
        if vr_drop_rate is not None:
            metadata["vr_drop_rate"] = round(vr_drop_rate, 4)
        if ik_miss_count is not None:
            metadata["ik_miss_count"] = ik_miss_count
        if ik_miss_max_consecutive > 0:
            metadata["ik_miss_max_consecutive"] = ik_miss_max_consecutive
        if camera_frame_rate is not None:
            metadata["camera_frame_rate_hz"] = round(camera_frame_rate, 2)

        # ── Episode score (weighted composite, 0.0–1.0) ──
        # IK success rate (0.0–1.0): weight 0.5 — most critical for data quality
        # VR drop rate    (0.0–1.0): weight 0.3 — framerate matters for training
        # Camera rate     (0.0–1.0): weight 0.2 — normalized to 30 Hz target
        components = []
        weights = []

        if ik_success_rate is not None:
            components.append(ik_success_rate)
            weights.append(0.5)

        if vr_drop_rate is not None:
            # Invert: low drop rate → high score
            components.append(1.0 - vr_drop_rate)
            weights.append(0.3)

        if camera_frame_rate is not None:
            # Normalize: assume 30 Hz is perfect
            cam_score = min(camera_frame_rate / 30.0, 1.0)
            components.append(cam_score)
            weights.append(0.2)

        if components:
            total_w = sum(weights)
            if total_w > 0:
                episode_score = sum(c * w for c, w in zip(components, weights)) / total_w
                metadata["episode_score"] = round(episode_score, 4)

        try:
            json_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Sidecar JSON written: %s", json_path)
        except OSError as e:
            logger.warning("Failed to write sidecar JSON: %s", e)

    # ------------------------------------------------------------------
    # File routing (Phase 1.2)
    # ------------------------------------------------------------------

    def _route_episode_file(self, h5_path: str) -> str | None:
        """Move episode files to success_dir / failure_dir based on classification.

        Ref: T-Rex data_writer.py move_episode_files().
        """
        target_dir = None
        if self._classification in ("success", "partial") and self.config.success_dir is not None:
            target_dir = Path(self.config.success_dir)
        elif self._classification == "failure" and self.config.failure_dir is not None:
            target_dir = Path(self.config.failure_dir)

        if target_dir is None:
            return None  # no routing configured

        target_dir.mkdir(parents=True, exist_ok=True)

        h5_file = Path(h5_path)
        json_file = h5_file.with_suffix(".json")

        try:
            new_h5 = target_dir / h5_file.name
            shutil.move(str(h5_file), str(new_h5))
            logger.info("Episode routed: %s → %s", h5_file.name, target_dir)

            if json_file.exists():
                new_json = target_dir / json_file.name
                shutil.move(str(json_file), str(new_json))

            return str(new_h5)
        except OSError as e:
            logger.warning("Failed to route episode file: %s", e)
            return None

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
        return {
            "frame_count": self._episode_frame_count,
            "duration_s": round(duration, 2),
            "is_recording": self.is_recording,
            "last_path": self._last_episode_path,
            "min_frames_met": self._episode_frame_count >= self.config.min_frames,
        }
