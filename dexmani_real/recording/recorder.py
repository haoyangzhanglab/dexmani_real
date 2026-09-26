"""Serialize Raw v34 control rows and publish completed episodes atomically."""

from __future__ import annotations

__all__ = ["EpisodeRecorder", "EpisodeFinalizationError"]

import shutil
import time
from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np

from dexmani_real.config.hardware import CameraParams
from dexmani_real.recording.frame import EpisodeFrame
from dexmani_real.recording.storage.camera_writer import (
    CameraStreamWriter,
    CameraStreamWriterConfig,
)
from dexmani_real.recording.storage.hdf5_writer import EpisodeDataWriter
from dexmani_real.recording.storage.schema import (
    DATASET_SPECS,
    EPISODE_SCHEMA_VERSION,
)
from dexmani_real.sensor.camera.geometry import RGBDGeometry
from dexmani_real.utils.atomic_io import atomic_publish
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

# Defensive storage ceiling; normal episode budgets belong to the control owner.
HARD_MAX_RECORD_FRAMES: int = 10_000
_RIGID_ATOL = 1e-6
_MOUNT_QUAT_ATOL = 1e-6
_COLLECTION_SOURCES = frozenset({"teleop", "policy_rollout"})


class EpisodeFinalizationError(RuntimeError):
    """An episode transaction failed; callers must also check resource release."""


def _validate_explicit_episode_name(episode_name: str) -> None:
    """Require a plain directory name that stays inside the recorder data dir."""
    if (
        type(episode_name) is not str
        or not episode_name
        or episode_name in {".", ".."}
        or "/" in episode_name
        or "\\" in episode_name
        or episode_name.startswith(".")
    ):
        raise ValueError(
            "episode_name must be a plain non-empty directory name without "
            f"path separators: {episode_name!r}"
        )


def _validate_rigid_transform(value: object, *, label: str) -> np.ndarray:
    try:
        transform = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{label} must be a finite 4x4 transform")
    rotation = transform[:3, :3]
    if (
        not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9, rtol=0.0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=_RIGID_ATOL, rtol=0.0)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=_RIGID_ATOL, rtol=0.0)
    ):
        raise ValueError(f"{label} must be a finite rigid homogeneous transform")
    return transform.copy()


def _validate_hand_mount(
    handbase_position_eef_m: object,
    handbase_quat_eef_wxyz: object,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        position = np.asarray(handbase_position_eef_m, dtype=np.float64)
        quaternion = np.asarray(handbase_quat_eef_wxyz, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("hand mount must be numeric") from exc
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("handbase_position_eef_m must have shape (3,) and finite values")
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("handbase_quat_eef_wxyz must have shape (4,) and finite values")
    if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=_MOUNT_QUAT_ATOL, rtol=0.0):
        raise ValueError("handbase_quat_eef_wxyz must be unit length within 1e-6")
    return position.copy(), quaternion.copy()


class EpisodeRecorder:
    """Own one Raw v34 transaction from START through atomic publication."""

    def __init__(
        self,
        data_dir: str,
        max_frames: int = HARD_MAX_RECORD_FRAMES,
        control_hz: float = 16.0,
        camera_writer_config: CameraStreamWriterConfig | None = None,
    ) -> None:
        if not np.isfinite(control_hz) or control_hz <= 0:
            raise ValueError(f"control_hz must be positive, got {control_hz}")
        if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0:
            raise ValueError("max_frames must be a positive integer")
        self.data_dir = Path(data_dir)
        self.max_frames = max_frames
        self.control_hz = float(control_hz)
        self._camera_writer_config = camera_writer_config or CameraStreamWriterConfig(
            rgb_shape=CameraParams().rgb_shape,
            depth_shape=CameraParams().depth_shape,
            fps=self.control_hz,
        )
        if not np.isclose(self._camera_writer_config.fps, self.control_hz):
            raise ValueError("camera writer fps must match recorder control_hz")

        self._data_writer: EpisodeDataWriter | None = None
        self._camera_writer: CameraStreamWriter | None = None
        self._frame_count = 0
        self._recording = False
        self._episode_dir: str | None = None
        self._temp_dir: str | None = None
        self._pending_meta: dict[str, object] = {}
        self._episode_valid = True
        self._pending_rows: list[dict[str, object]] = []
        self._flush_interval = 32
        self._finishing = False
        self._last_finish_saved = False

    @property
    def last_finish_saved(self) -> bool:
        """Whether the most recent finalization published one raw episode."""
        return self._last_finish_saved

    @property
    def resources_released(self) -> bool:
        """Whether writers and local staging ownership have been released."""
        return self._camera_writer is None and self._data_writer is None and self._temp_dir is None

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def episode_path(self) -> str | None:
        """Reserved final path; raw data is published only after validation."""
        return self._episode_dir

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def episode_valid(self) -> bool:
        """The active episode's monotonic-false lifecycle latch."""
        return self._episode_valid

    def invalidate_episode(self) -> None:
        """Record one non-recoverable lifecycle failure for the active episode."""
        self._episode_valid = False

    def _snapshot_start_metadata(
        self,
        *,
        task_label: str,
        collection_source: str,
        camera_geometry: RGBDGeometry,
        camera_T_xarm_base_from_color: object,
        depth_scale: float,
        handbase_position_eef_m: object,
        handbase_quat_eef_wxyz: object,
    ) -> dict[str, object]:
        if not isinstance(task_label, str) or not task_label.strip():
            raise ValueError("task_label must be a non-empty string")
        if not isinstance(collection_source, str) or collection_source not in _COLLECTION_SOURCES:
            raise ValueError(
                f"collection_source must be one of {sorted(_COLLECTION_SOURCES)}, "
                f"got {collection_source!r}"
            )
        if not isinstance(camera_geometry, RGBDGeometry):
            raise TypeError("camera_geometry must be an RGBDGeometry instance")
        color = camera_geometry.color
        expected_rgb_shape = (color.height, color.width, 3)
        expected_depth_shape = (color.height, color.width)
        if (
            self._camera_writer_config.rgb_shape != expected_rgb_shape
            or self._camera_writer_config.depth_shape != expected_depth_shape
        ):
            raise ValueError(
                "aligned RGB-D geometry must match recorder transport: "
                f"expected rgb={self._camera_writer_config.rgb_shape}, "
                f"depth={self._camera_writer_config.depth_shape}; "
                f"got rgb={expected_rgb_shape}, depth={expected_depth_shape}"
            )
        try:
            depth_scale_value = float(depth_scale)
        except (TypeError, ValueError) as exc:
            raise ValueError("depth_scale must be numeric") from exc
        if not np.isfinite(depth_scale_value) or depth_scale_value <= 0.0:
            raise ValueError("depth_scale must be finite and positive")
        transform = _validate_rigid_transform(
            camera_T_xarm_base_from_color,
            label="camera_T_xarm_base_from_color",
        )
        handbase_position, handbase_quaternion = _validate_hand_mount(
            handbase_position_eef_m,
            handbase_quat_eef_wxyz,
        )
        return {
            "task_label": task_label,
            "collection_source": collection_source,
            "camera_payload_mode": "depth_to_color_aligned_rgbd",
            "camera_color_width": color.width,
            "camera_color_height": color.height,
            "camera_color_intrinsics": color.matrix().reshape(-1).copy(),
            "camera_color_distortion_model": color.distortion_model,
            "camera_color_distortion_coeffs": np.asarray(
                color.distortion_coeffs, dtype=np.float64
            ).copy(),
            "camera_T_xarm_base_from_color": transform.reshape(-1).copy(),
            "depth_scale": depth_scale_value,
            "handbase_position_eef_m": handbase_position,
            "handbase_quat_eef_wxyz": handbase_quaternion,
        }

    def start_episode(
        self,
        task_label: str,
        collection_source: str,
        camera_geometry: RGBDGeometry,
        camera_T_xarm_base_from_color: object,
        depth_scale: float,
        handbase_position_eef_m: object,
        handbase_quat_eef_wxyz: object,
        episode_name: str | None = None,
    ) -> bool:
        """Validate and snapshot static experiment facts before opening staging."""
        if self._finishing or not self.resources_released or self._recording:
            return False
        metadata = self._snapshot_start_metadata(
            task_label=task_label,
            collection_source=collection_source,
            camera_geometry=camera_geometry,
            camera_T_xarm_base_from_color=camera_T_xarm_base_from_color,
            depth_scale=depth_scale,
            handbase_position_eef_m=handbase_position_eef_m,
            handbase_quat_eef_wxyz=handbase_quat_eef_wxyz,
        )
        if episode_name is not None:
            _validate_explicit_episode_name(episode_name)

        self.data_dir.mkdir(parents=True, exist_ok=True)
        if episode_name is None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            episode_dir = self.data_dir / f"episode_{stamp}"
            temp_dir = self.data_dir / f".tmp_episode_{stamp}"
            dedup = 1
            while episode_dir.exists() or temp_dir.exists():
                episode_dir = self.data_dir / f"episode_{stamp}_{dedup}"
                temp_dir = self.data_dir / f".tmp_episode_{stamp}_{dedup}"
                dedup += 1
        else:
            episode_dir = self.data_dir / episode_name
            temp_dir = self.data_dir / f".tmp_{episode_name}"
            if episode_dir.exists() or temp_dir.exists():
                raise FileExistsError(f"explicit episode name already exists: {episode_dir}")

        temp_dir.mkdir(parents=True, exist_ok=False)
        # Construction may fail while the camera writer still owns a sidecar.
        # Keep the staging owner visible until a successful transaction has
        # established normal writer ownership or a caller investigates it.
        self._temp_dir = str(temp_dir)
        camera_writer = CameraStreamWriter(temp_dir, self._camera_writer_config)

        self._episode_dir = str(episode_dir)
        self._pending_meta = metadata
        self._camera_writer = camera_writer
        self._data_writer = None
        self._frame_count = 0
        self._pending_rows.clear()
        self._recording = True
        self._episode_valid = True
        self._last_finish_saved = False
        return True

    def _write_initial_meta(self, meta: h5py.Group) -> None:
        """Write the static Raw v34 snapshot through the writer-owned handle."""
        for name, value in self._pending_meta.items():
            meta.attrs[name] = value
        meta.attrs["control_hz"] = self.control_hz

    def add_frame(self, frame: EpisodeFrame) -> bool:
        """Append one already-constructed control row without resampling."""
        if not self._recording:
            return False
        if self._frame_count >= self.max_frames:
            raise RuntimeError(f"recording exceeded hard frame limit {self.max_frames}")
        if set(frame.data) != set(DATASET_SPECS):
            raise ValueError("episode frame fields do not match the Raw v34 schema")
        if frame.camera_rgb is None or frame.camera_depth is None:
            raise ValueError("recorded row is missing RGB-D")
        writer = self._camera_writer
        if writer is None:
            raise RuntimeError("camera writer missing")

        self._pending_rows.append(dict(frame.data))
        self._frame_count += 1
        writer.write(frame.camera_rgb, frame.camera_depth)
        if len(self._pending_rows) >= self._flush_interval:
            self._flush_buffered()
        return True

    def _ensure_hdf5(self) -> None:
        if self._data_writer is not None:
            return
        if self._temp_dir is None:
            raise RuntimeError("EpisodeRecorder has no staging directory during HDF5 open")
        self._data_writer = EpisodeDataWriter(
            Path(self._temp_dir) / "data.h5",
            write_initial_meta=self._write_initial_meta,
        )

    def _flush_buffered(self) -> None:
        if not self._pending_rows:
            return
        self._ensure_hdf5()
        batch = {
            name: np.asarray([row[name] for row in self._pending_rows], dtype=spec.dtype)
            for name, spec in DATASET_SPECS.items()
        }
        assert self._data_writer is not None
        self._data_writer.append(batch)
        self._pending_rows.clear()

    def finish_episode(self, save: bool = True, reason: str = "") -> str | None:
        """Close, verify, and atomically publish one complete episode when requested."""
        if self._finishing:
            raise RuntimeError("episode finalization is still active")
        if not self._recording:
            return None
        self._last_finish_saved = False
        if save and self._frame_count == 0:
            logger.warning(
                "[RECORD] reason=%s saved_rows=0; discarding empty staging", reason or "manual"
            )
            save = False
        path = self._episode_dir
        self._recording = False
        self._finishing = True
        try:
            self._finish_episode_transaction(save, reason)
        finally:
            self._finishing = False
        return path

    def _finish_episode_transaction(self, save: bool, reason: str) -> None:
        """Finalize one transaction and retain staging only if cleanup fails."""
        failure: Exception | None = None
        try:
            self._finalize_episode_files(save, reason)
        except Exception as exc:
            failure = exc
            logger.error("episode finalization failed", exc_info=True)
            try:
                if self._camera_writer is not None:
                    self._camera_writer.close()
            except Exception:
                logger.warning("camera writer cleanup failed", exc_info=True)
            if self._camera_writer is not None and self._camera_writer.resources_released:
                self._camera_writer = None
            try:
                if self._data_writer is not None:
                    self._data_writer.close()
                    self._data_writer = None
            except Exception:
                logger.warning("HDF5 cleanup failed", exc_info=True)
        finally:
            if (
                self._camera_writer is None
                and self._data_writer is None
                and self._temp_dir is not None
            ):
                try:
                    self._discard_temp_files(self._temp_dir)
                except Exception:
                    logger.warning(
                        "temporary episode cleanup failed; staging remains at %s",
                        self._temp_dir,
                        exc_info=True,
                    )
                else:
                    self._reset_episode_state()
        if failure is not None:
            raise EpisodeFinalizationError(f"{type(failure).__name__}: {failure}") from failure

    def _finalize_episode_files(self, save: bool, reason: str) -> None:
        """Close all writers, validate closed files, then atomically publish."""
        camera_writer = self._camera_writer
        camera_frame_count = 0
        if camera_writer is not None:
            camera_writer.close()
            if not camera_writer.resources_released:
                raise RuntimeError("camera writer resources were not released")
            camera_frame_count = camera_writer.frame_count
            self._camera_writer = None
        if not save:
            if self._data_writer is not None:
                self._data_writer.close()
                self._data_writer = None
            return
        if camera_writer is None:
            raise RuntimeError("camera writer missing at episode stop")
        if camera_frame_count != self._frame_count:
            raise RuntimeError(
                "camera/source row count mismatch: "
                f"camera={camera_frame_count}, source={self._frame_count}"
            )

        self._flush_buffered()
        self._ensure_hdf5()
        assert self._data_writer is not None
        if self._temp_dir is None:
            raise RuntimeError("episode staging directory missing during finalization")

        def write_final_meta(meta: h5py.Group) -> None:
            meta.attrs["schema_version"] = EPISODE_SCHEMA_VERSION
            meta.attrs["num_frames"] = self._frame_count
            meta.attrs["episode_valid"] = bool(self._episode_valid)

        self._data_writer.update_meta(write_final_meta)
        self._data_writer.close()
        self._data_writer = None
        if self._episode_dir is None:
            raise RuntimeError("episode final directory missing during finalization")
        self._validate_temp_episode(Path(self._temp_dir), self._frame_count)
        atomic_publish(self._temp_dir, self._episode_dir)
        self._last_finish_saved = True
        logger.info(
            "Episode saved: %s frames=%d episode_valid=%s reason=%s",
            self._episode_dir,
            self._frame_count,
            self._episode_valid,
            reason or "manual",
        )

    def _reset_episode_state(self) -> None:
        """Release transaction ownership without changing the final validity latch."""
        self._data_writer = None
        self._recording = False
        self._frame_count = 0
        self._episode_dir = None
        self._temp_dir = None
        self._pending_rows.clear()
        self._pending_meta.clear()
        self._camera_writer = None

    @staticmethod
    def _validate_temp_episode(temp_dir: Path, expected_frames: int) -> None:
        """Open closed staging with the v34 structural reader before publication."""
        from dexmani_real.recording.storage.reader import EpisodeReader

        with EpisodeReader(temp_dir) as reader:
            if reader.num_frames != expected_frames:
                raise RuntimeError(
                    "data.h5 frame count metadata mismatch: "
                    f"expected {expected_frames}, got {reader.num_frames}"
                )

    @staticmethod
    def _discard_temp_files(tmp: str) -> None:
        """Remove only this transaction's owned staging after writers are closed."""
        if Path(tmp).exists():
            shutil.rmtree(tmp)
