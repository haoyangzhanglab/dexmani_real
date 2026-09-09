"""Transactional raw-v27 episode serialization from owned ``EpisodeFrame`` rows.

Each controller-emitted source sample becomes exactly one persisted row.
The recorder owns transaction lifecycle, camera sidecar coordination, metadata,
verification, and atomic publication. ``EpisodeDataWriter`` is the sole owner
of the ``data.h5`` handle, datasets, and append offset. Neither component reads
shared memory or controls hardware.
"""

from __future__ import annotations

__all__ = ["EpisodeRecorder", "EpisodeFinalizationError", "normalize_provenance_metadata"]

import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy as np

from dexmani_real.calibration.camera.extrinsics import CameraExtrinsics
from dexmani_real.config.defaults import camera
from dexmani_real.recording.storage.camera_writer import (
    CameraStreamWriter,
    CameraStreamWriterConfig,
)
from dexmani_real.recording.frame import EpisodeFrame, build_episode_frame
from dexmani_real.recording.storage.hdf5_writer import EpisodeDataWriter
from dexmani_real.recording.storage.schema import (
    EPISODE_SCHEMA_VERSION,
    DATASET_SPECS,
    SOURCE_FRAME_DATASET_NAMES,
    FillReason,
    validate_data_layout,
)
from dexmani_real.recording.sample import EpisodeAction, EpisodeState
from dexmani_real.sensor.camera.geometry import RGBDGeometry
from dexmani_real.utils.atomic_io import atomic_json_dump, atomic_publish
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_RECORD_FRAMES: int = 10000
_CAMERA_WRITER_CLOSE_TIMEOUT_S = 60.0
_MAX_PROVENANCE_VALUE_BYTES = 4096


def normalize_provenance_metadata(
    provenance: Mapping[str, object] | None,
) -> dict[str, str]:
    """Validate recorder-owned, scalar episode provenance attributes.

    Provenance is deliberately separate from camera metadata: the recorder
    owns the ``provenance_*`` namespace and callers may not repurpose it to
    override schema or camera fields.
    """
    if provenance is None:
        return {}
    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping of string keys and values")

    normalized: dict[str, str] = {}
    for key, value in provenance.items():
        if (
            not isinstance(key, str)
            or not key
            or not key.isascii()
            or not key.isidentifier()
            or key.startswith("provenance_")
        ):
            raise ValueError(
                "provenance keys must be non-empty ASCII identifiers without "
                "the provenance_ prefix"
            )
        if not isinstance(value, str):
            raise TypeError("provenance values must be strings")
        if len(value.encode("utf-8")) > _MAX_PROVENANCE_VALUE_BYTES:
            raise ValueError(
                "provenance values must be at most "
                f"{_MAX_PROVENANCE_VALUE_BYTES} UTF-8 bytes"
            )
        normalized[key] = value
    return normalized


class EpisodeFinalizationError(RuntimeError):
    """An episode transaction failed; callers must also check resource release."""


class EpisodeRecorder:
    """Coordinate one transactional episode around a dedicated data writer.

    Lifecycle: start_episode() → add_frame() × N → finish_episode()
    """

    def __init__(
        self,
        data_dir: str,
        max_frames: int = DEFAULT_MAX_RECORD_FRAMES,
        control_hz: float = 16.0,
        min_frames: int = 50,
        camera_writer_config: CameraStreamWriterConfig | None = None,
    ) -> None:
        if not np.isfinite(control_hz) or control_hz <= 0:
            raise ValueError(f"control_hz must be positive, got {control_hz}")
        self.data_dir = Path(data_dir)
        self.max_frames = max_frames
        self.control_hz = float(control_hz)
        self.min_frames = int(min_frames)

        self._data_writer: EpisodeDataWriter | None = None
        self._camera_writer: CameraStreamWriter | None = None
        self._camera_writer_config = camera_writer_config or CameraStreamWriterConfig(
            rgb_shape=camera.rgb_shape,
            depth_shape=camera.depth_shape,
            fps=self.control_hz,
            queue_size=camera.writer_queue_size,
        )
        if not np.isclose(self._camera_writer_config.fps, self.control_hz):
            raise ValueError("camera writer fps must match recorder control_hz")
        self._frame_count: int = 0
        self._recording: bool = False
        self._max_frames_reached: bool = False
        self._start_time: float | None = None
        self._episode_dir: str | None = None  # episode_XXX/ directory
        self._temp_dir: str | None = None  # .tmp_episode_XXX/ directory
        self._pending_meta: dict[str, Any] = {}

        self._pending_rows: list[dict[str, Any]] = []
        self._last_timestamp_s: float | None = None
        self._flush_interval = 32

        self._finishing = False

    @property
    def resources_released(self) -> bool:
        """Whether storage owners and temporary transaction files were released."""
        return (
            self._camera_writer is None
            and self._data_writer is None
            and self._temp_dir is None
        )

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
    def max_frames_reached(self) -> bool:
        return self._max_frames_reached

    @property
    def camera_writer_error(self) -> str | None:
        """Latched camera sidecar error requiring episode discard."""
        return self._camera_writer.error if self._camera_writer is not None else None

    def start_episode(
        self,
        task_label: str = "",
        operator: str = "",
        calib: CameraExtrinsics | None = None,
        camera_geometry: RGBDGeometry | None = None,
        camera_name: str | None = None,
        camera_serial: str | None = None,
        depth_scale: float | None = None,
        provenance: Mapping[str, object] | None = None,
    ) -> bool:
        normalized_provenance = normalize_provenance_metadata(provenance)
        if self._finishing or not self.resources_released:
            return False
        if self._recording:
            return False

        self.data_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        ep_dir = self.data_dir / f"episode_{stamp}"
        tmp_dir = self.data_dir / f".tmp_episode_{stamp}"
        dedup = 1
        while ep_dir.exists() or tmp_dir.exists():  # same-second collision → suffix
            ep_dir = self.data_dir / f"episode_{stamp}_{dedup}"
            tmp_dir = self.data_dir / f".tmp_episode_{stamp}_{dedup}"
            dedup += 1

        self._episode_dir = str(ep_dir)
        self._temp_dir = str(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=False)
        self._frame_count = 0
        self._max_frames_reached = False
        self._start_time = time.perf_counter()
        self._recording = True
        self._data_writer = None

        self._pending_meta = {
            "task_label": task_label,
            "operator": operator,
            "calib": calib,
            "camera_geometry": camera_geometry,
            "camera_name": camera_name,
            "camera_serial": camera_serial,
            "depth_scale": depth_scale,
            "provenance": normalized_provenance,
        }

        self._pending_rows.clear()
        self._last_timestamp_s = None
        self._camera_writer = CameraStreamWriter(tmp_dir, self._camera_writer_config)
        return True

    def _write_meta_attrs(self, meta: h5py.Group) -> None:
        """Write initial metadata through the data writer's owned HDF5 handle."""
        p = self._pending_meta
        meta.attrs["task_label"] = p.get("task_label", "")
        meta.attrs["operator"] = p.get("operator", "")
        meta.attrs["control_hz"] = (
            self.control_hz
        )  # nominal grid rate; dt = 1/control_hz
        meta.attrs["fps"] = self.control_hz
        meta.attrs["camera_payload_mode"] = "depth_to_color_aligned_rgbd"
        self._write_camera_meta_attrs(meta)

        provenance = p.get("provenance") or {}
        for key, value in provenance.items():
            meta.attrs[f"provenance_{key}"] = value

    def _write_camera_meta_attrs(self, meta: h5py.Group) -> None:
        """Camera identity/geometry attrs from _pending_meta (None entries skipped).

        Idempotent — finalization re-runs it so values supplied late
        still reach /meta after the initial lazy write.
        """
        p = self._pending_meta
        calib = p.get("calib")
        camera_name = p.get("camera_name")
        camera_serial = p.get("camera_serial")
        if camera_serial is not None:
            meta.attrs["camera_serial"] = str(camera_serial)
        if camera_name is not None:
            meta.attrs["camera_name"] = str(camera_name)
        camera_geometry = p.get("camera_geometry")
        if camera_geometry is not None and not isinstance(
            camera_geometry, RGBDGeometry
        ):
            raise TypeError("camera_geometry must be an RGBDGeometry instance")
        if calib is not None and camera_name is not None:
            # Verify a supplied serial against the named calibration entry.
            calib_meta = calib.to_meta_dict(camera_name, expected_serial=camera_serial)
            meta.attrs["camera_serial"] = calib_meta.get("camera_serial", "")
            meta.attrs["camera_type"] = calib_meta.get("camera_type", "")
            if camera_geometry is None:
                raise RuntimeError("camera calibration requires native RGB-D geometry")
            if "camera_T_world_camera" in calib_meta:
                T_xarm_base_from_color = np.asarray(
                    calib_meta["camera_T_world_camera"], dtype=np.float64
                ).reshape(4, 4)
                meta.attrs["camera_T_xarm_base_from_color"] = (
                    T_xarm_base_from_color.reshape(-1).tolist()
                )
                meta.attrs["camera_T_xarm_base_from_depth"] = (
                    (T_xarm_base_from_color @ camera_geometry.T_color_from_depth)
                    .reshape(-1)
                    .tolist()
                )
            if "camera_T_eef_camera" in calib_meta:
                T_eef_from_color = np.asarray(
                    calib_meta["camera_T_eef_camera"], dtype=np.float64
                ).reshape(4, 4)
                meta.attrs["camera_T_eef_from_depth"] = (
                    (T_eef_from_color @ camera_geometry.T_color_from_depth)
                    .reshape(-1)
                    .tolist()
                )
            meta.attrs["camera_calibration_source_optical_frame"] = (
                "camera_color_optical"
            )

        if camera_geometry is not None:
            meta.attrs["camera_depth_intrinsics"] = (
                camera_geometry.depth.matrix().reshape(-1).tolist()
            )
            meta.attrs["camera_depth_width"] = camera_geometry.depth.width
            meta.attrs["camera_depth_height"] = camera_geometry.depth.height
            meta.attrs["camera_depth_distortion_model"] = (
                camera_geometry.depth.distortion_model
            )
            meta.attrs["camera_depth_distortion_coeffs"] = list(
                camera_geometry.depth.distortion_coeffs
            )
            meta.attrs["camera_color_intrinsics"] = (
                camera_geometry.color.matrix().reshape(-1).tolist()
            )
            meta.attrs["camera_color_width"] = camera_geometry.color.width
            meta.attrs["camera_color_height"] = camera_geometry.color.height
            meta.attrs["camera_color_distortion_model"] = (
                camera_geometry.color.distortion_model
            )
            meta.attrs["camera_color_distortion_coeffs"] = list(
                camera_geometry.color.distortion_coeffs
            )
            meta.attrs["camera_T_color_from_depth"] = (
                camera_geometry.T_color_from_depth.reshape(-1).tolist()
            )
            meta.attrs["camera_geometry_frame_semantics"] = (
                "native_depth_and_native_color_optical_frames"
            )

        # Raw uint16 depth units in meters (L515: 0.00025) — without this,
        # offline consumers cannot convert /depth correctly.
        depth_scale = p.get("depth_scale")
        if depth_scale is not None:
            meta.attrs["depth_scale"] = float(depth_scale)

    def add_frame(
        self,
        state: EpisodeState,
        action: EpisodeAction,
        vr_frame: Mapping[str, object],
        camera_frame: Mapping[str, object] | None = None,
        signals: Mapping[str, object] | None = None,
        arm_qpos_sent: np.ndarray | None = None,
    ) -> bool:
        """Build and add one :class:`EpisodeFrame` from component inputs."""
        if not self._accept_source_frame():
            return False
        frame = build_episode_frame(
            state,
            action,
            vr_frame,
            camera_frame=camera_frame,
            signals=signals,
            arm_qpos_sent=arm_qpos_sent,
        )
        return self._add_episode_frame(frame)

    def add_episode_frame(self, frame: EpisodeFrame) -> bool:
        """Append one owned controller source frame."""
        if not self._accept_source_frame():
            return False
        return self._add_episode_frame(frame)

    def _accept_source_frame(self) -> bool:
        """Admit source frames only during the active episode and below capacity."""
        if not self._recording:
            return False

        if self._frame_count >= self.max_frames:
            logger.warning(
                "Episode reached max_frames=%d, auto-stopping.", self.max_frames
            )
            self._max_frames_reached = True
            return False

        return True

    def _add_episode_frame(self, frame: EpisodeFrame) -> bool:
        """Store one emitted source row without aligning or filling timestamps."""
        ts = float(frame.timestamp_s)
        if not np.isfinite(ts) or (
            self._last_timestamp_s is not None and ts <= self._last_timestamp_s
        ):
            raise ValueError("recording timestamps must be finite and increasing")
        if set(frame.data) != SOURCE_FRAME_DATASET_NAMES:
            raise ValueError("episode source frame fields do not match raw v27")
        row = dict(frame.data)
        row.update(
            timestamp=ts, source_sample_index=self._frame_count,
            fill_reason=FillReason.SOURCE, flag_sample_valid=True,
        )
        self._pending_rows.append(row)
        self._frame_count += 1
        self._last_timestamp_s = ts
        writer = self._camera_writer
        if writer is None or not writer.submit(*self._camera_payload(frame)):
            raise RuntimeError("camera writer failed to accept the source row")
        if len(self._pending_rows) >= self._flush_interval:
            self._flush_buffered()
        if self._frame_count >= self.max_frames:
            self._max_frames_reached = True
            return False
        return True

    def _camera_payload(
        self, frame: EpisodeFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return shape-stable camera arrays, using explicit zero placeholders."""
        cfg = self._camera_writer_config
        rgb = frame.camera_rgb
        depth = frame.camera_depth
        if rgb is None:
            rgb = np.zeros(cfg.rgb_shape, dtype=np.uint8)
        if depth is None:
            depth = np.zeros(cfg.depth_shape, dtype=np.uint16)
        return rgb, depth

    def _ensure_hdf5(self) -> None:
        """Lazily create the ``data.h5`` owner for the active temp directory."""
        if self._data_writer is not None:
            return
        if self._temp_dir is None:
            raise RuntimeError("EpisodeRecorder: _temp_dir is None during HDF5 open")
        self._data_writer = EpisodeDataWriter(
            Path(self._temp_dir) / "data.h5",
            write_initial_meta=self._write_meta_attrs,
        )

    def _flush_buffered(self) -> None:
        """Append pending rows; the batch has no temporal semantics."""
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
        """Synchronously finish one episode and return its reserved final path.

        Discard also returns the reserved path, although no raw episode is
        published there. Failure raises after transaction cleanup. The caller
        must serialize this operation with all other recorder access.
        """
        if self._finishing:
            raise RuntimeError("episode finalization is still active")
        if not self._recording:
            return None
        path = self._episode_dir
        truncated = self._max_frames_reached
        self._recording = False
        self._max_frames_reached = False
        self._finishing = True
        try:
            self._finish_episode_transaction(save, reason, truncated)
        finally:
            self._finishing = False
        return path

    def _finish_episode_transaction(
        self,
        save: bool,
        reason: str,
        truncated: bool,
    ) -> None:
        """Finalize one transaction; retain any resource that failed cleanup."""
        failure = None
        try:
            self._finalize_episode_files(save, reason, truncated)
        except Exception as exc:
            failure = exc
            logger.error("episode finalization failed", exc_info=True)
            try:
                self._write_aborted_manifest(
                    reason=reason, error=f"{type(exc).__name__}: {exc}"
                )
            except Exception:
                logger.error(
                    "failed to publish aborted episode manifest", exc_info=True
                )
            try:
                if self._camera_writer is not None:
                    self._camera_writer.close(timeout=5.0)
            except Exception:
                logger.warning("camera writer cleanup failed", exc_info=True)
            if (
                self._camera_writer is not None
                and self._camera_writer.resources_released
            ):
                self._camera_writer = None
            try:
                if self._data_writer is not None:
                    self._data_writer.close()
                    self._data_writer = None
            except Exception:
                logger.warning("HDF5 cleanup failed", exc_info=True)
        finally:
            # Retain handles and staging while an owner may still access them.
            if self._camera_writer is None and self._data_writer is None:
                try:
                    if self._temp_dir is not None:
                        self._discard_temp_files(self._temp_dir)
                except Exception as exc:
                    failure = exc
                    logger.error("temporary episode cleanup failed", exc_info=True)
                else:
                    self._reset_episode_state()
        if failure is not None:
            raise EpisodeFinalizationError(
                f"{type(failure).__name__}: {failure}"
            ) from failure

    def _finalize_episode_files(
        self,
        save: bool,
        reason: str,
        truncated: bool,
    ) -> None:
        """Close, validate, and publish files before releasing staging ownership."""
        duration = time.perf_counter() - (self._start_time or 0.0)

        writer = self._camera_writer
        if writer is None:
            raise RuntimeError("camera writer missing at episode stop")
        writer.close(timeout=_CAMERA_WRITER_CLOSE_TIMEOUT_S)
        if not writer.resources_released:
            raise RuntimeError("camera writer resources were not released")
        camera_frame_count = writer.frame_count
        self._camera_writer = None
        if camera_frame_count != self._frame_count:
            raise RuntimeError(
                f"camera/source row count mismatch: camera={camera_frame_count}, source={self._frame_count}"
            )

        self._flush_buffered()
        self._ensure_hdf5()

        assert self._data_writer is not None
        data_writer = self._data_writer
        _had_rgb = camera_frame_count > 0
        if self._temp_dir is None:
            raise RuntimeError("episode temp directory missing during finalization")

        def _write_final_meta(meta: h5py.Group) -> None:
            meta.attrs["schema_version"] = EPISODE_SCHEMA_VERSION
            meta.attrs["duration"] = duration
            meta.attrs["wall_duration_s"] = duration
            meta.attrs["num_frames"] = self._frame_count
            meta.attrs["success"] = save
            meta.attrs["fps"] = self.control_hz
            meta.attrs["wall_fps"] = (
                self._frame_count / duration if duration > 0 else self.control_hz
            )
            meta.attrs["min_frames_met"] = self._frame_count >= self.min_frames
            meta.attrs["has_camera"] = _had_rgb
            meta.attrs["has_timestamps"] = "timestamp" in data_writer.datasets
            meta.attrs["camera_stream_frames"] = camera_frame_count
            meta.attrs["camera_writer_error"] = ""
            meta.attrs["truncated"] = bool(truncated)
            meta.attrs["stop_reason"] = reason or (
                "max_frames" if truncated else "manual"
            )
            # Repeat the camera snapshot during finalization before the handle closes.
            self._write_camera_meta_attrs(meta)

        data_writer.update_meta(_write_final_meta)
        data_writer.close()
        self._data_writer = None
        # Atomically rename the temporary directory into its final location.
        _final = self._episode_dir
        _tmp = self._temp_dir
        if _tmp is not None and _final is not None:
            if save:
                self._validate_temp_episode(Path(_tmp), self._frame_count)
                atomic_publish(_tmp, _final)
                logger.info("Episode saved: %s frames=%d", _final, self._frame_count)
            else:
                self._write_aborted_manifest(reason=reason or "discarded", error="")

    def _reset_episode_state(self) -> None:
        """Reset episode state only after all resources and staging are released."""
        self._data_writer = None
        self._recording = False
        self._max_frames_reached = False
        self._frame_count = 0
        self._start_time = None
        self._episode_dir = None
        self._temp_dir = None
        self._pending_rows.clear()
        self._camera_writer = None
        self._last_timestamp_s = None

    # ── Atomic file finalisation ──────────────────────────────────────

    def _write_aborted_manifest(self, *, reason: str, error: str) -> Path:
        """Persist only small failure provenance; never retain partial payloads."""
        episode_name = Path(self._episode_dir or "aborted_episode_unknown").name
        target = self.data_dir / f"{episode_name}.aborted.json"
        suffix = 1
        while target.exists():
            target = self.data_dir / f"{episode_name}.{suffix}.aborted.json"
            suffix += 1
        payload = {
            "episode": episode_name,
            "status": "aborted",
            "reason": reason,
            "error": error,
            "frame_count_before_abort": int(self._frame_count),
            "created_wall_time_ns": time.time_ns(),
        }
        return atomic_json_dump(payload, target, indent=2, ensure_ascii=False)

    @staticmethod
    def _validate_temp_episode(temp_dir: Path, expected_frames: int) -> None:
        """Verify closed files structurally before publication, without RGB decoding."""
        paths = {
            "data": temp_dir / "data.h5",
            "depth": temp_dir / "depth.h5",
            "rgb": temp_dir / "rgb.mp4",
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise RuntimeError(f"episode finalization missing modalities: {missing}")
        with h5py.File(paths["data"], "r") as data_h5:
            meta = data_h5.get("meta")
            if meta is None or int(meta.attrs.get("num_frames", -1)) != expected_frames:
                raise RuntimeError("data.h5 frame count metadata mismatch")
            if int(meta.attrs.get("schema_version", -1)) != EPISODE_SCHEMA_VERSION:
                raise RuntimeError(
                    "new writer produced an unexpected raw schema version"
                )
            datasets = {
                key: dataset
                for key, dataset in data_h5.items()
                if isinstance(dataset, h5py.Dataset)
            }
            dataset_shapes = {
                key: tuple(dataset.shape) for key, dataset in datasets.items()
            }
            dataset_dtypes = {key: dataset.dtype for key, dataset in datasets.items()}
            layout_errors = validate_data_layout(
                dataset_shapes,
                dataset_dtypes,
                frame_count=expected_frames,
            )
            if layout_errors:
                raise RuntimeError(
                    "data.h5 episode layout mismatch: " + "; ".join(layout_errors)
                )
        for key in ("depth",):
            with h5py.File(paths[key], "r") as sidecar:
                if key not in sidecar or int(sidecar[key].shape[0]) != expected_frames:
                    raise RuntimeError(f"{key} sidecar length mismatch")
        if paths["rgb"].stat().st_size == 0:
            raise RuntimeError("RGB sidecar is empty")

    @staticmethod
    def _discard_temp_files(tmp: str) -> None:
        """Remove staging after resource release; expose incomplete cleanup."""
        if Path(tmp).exists():
            shutil.rmtree(tmp)
