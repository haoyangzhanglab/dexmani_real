"""Minimal schema-v29 raw-episode fixture for provenance integration tests.

Builds a real, self-consistent raw episode (``data.h5`` + ``depth.h5`` +
``rgb.mp4``) that :class:`EpisodeReader` accepts as VALID and that both the
processing admission boundary (:func:`dexmani_real.dataset.processing.analyze_episode`)
and the replay loader (:func:`dexmani_real.replay.trajectory.load_trajectory`)
can read through. Provenance is injected via the ``provenance_workflow``
``/meta`` attribute; the rest of the episode is a deterministically valid
teleop-shaped recording.

This helper is deliberately not a ``test_*`` module, so ``unittest`` discovery
never collects it as a test.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from dexmani_real.recording.storage.schema import DATASET_SPECS, EPISODE_SCHEMA_VERSION
from dexmani_real.recording.storage.video import VideoEncoder

# Tiny aligned RGB-D resolution keeps the MP4 sidecar negligible while still
# exercising the real camera-model and decode admission paths.
_CAMERA_WIDTH = 8
_CAMERA_HEIGHT = 8
_DEFAULT_NUM_FRAMES = 3
_CONTROL_HZ = 16.0

# Canonical pinhole intrinsics: last row exactly [0, 0, 1] (required by the
# raw camera-model loader), principal point at the image centre.
_PINHOLE = np.array(
    [[10.0, 0.0, _CAMERA_WIDTH / 2.0], [0.0, 10.0, _CAMERA_HEIGHT / 2.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
_IDENTITY_4X4 = np.eye(4, dtype=np.float64)
_DISTORTION_COEFFS = np.zeros(5, dtype=np.float64)
_ROT6D_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)


def _timestamps(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / _CONTROL_HZ


def _anchors(n: int) -> np.ndarray:
    # Positive, strictly increasing, one microsecond per row.
    return ((np.arange(n, dtype=np.uint64) + 1) * 1_000).astype(np.uint64)


def _dataset_values(name: str, n: int) -> np.ndarray:
    """Return semantically valid values for one schema-v29 dataset row block."""
    anchors = _anchors(n)
    causal_source = anchors - 1  # positive and strictly behind the anchor

    if name == "timestamp":
        return _timestamps(n)
    if name == "source_sample_index":
        return np.arange(n, dtype=np.int64)
    if name == "observation_anchor_monotonic_ns":
        return anchors
    if name in (
        "arm_source_monotonic_ns",
        "hand_source_monotonic_ns",
        "camera_source_monotonic_ns",
    ):
        return causal_source.astype(np.uint64)
    if name == "action_arm_ee":
        rows = np.zeros((n, 9), dtype=np.float64)
        rows[:, 3:9] = _ROT6D_IDENTITY
        return rows
    if name == "vr_wrist_rot6d":
        rows = np.zeros((n, 6), dtype=np.float64)
        rows[:] = _ROT6D_IDENTITY
        return rows
    if name == "head_quat_wxyz":
        rows = np.zeros((n, 4), dtype=np.float64)
        rows[:, 0] = 1.0
        return rows
    if name == "arm_last_cmd_seq":
        return np.arange(n, dtype=np.int64)
    if name in ("camera_depth_frame_number", "camera_color_frame_number"):
        return np.arange(n, dtype=np.uint64)

    spec = DATASET_SPECS[name]
    dtype = spec.dtype
    shape = (n, *spec.tail_shape)
    if dtype == np.dtype(np.bool_):
        # Positive liveness/validity flags are True; hand_qpos_stale is the one
        # negative-polarity defect flag and must be False.
        value = np.ones(shape, dtype=np.bool_)
        if name == "hand_qpos_stale":
            value[...] = False
        return value
    return np.zeros(shape, dtype=dtype)


def build_raw_episode(
    root: str | Path,
    *,
    provenance_workflow: str | None = None,
    num_frames: int = _DEFAULT_NUM_FRAMES,
) -> Path:
    """Write a valid schema-v29 raw episode directory and return its path."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    n = int(num_frames)

    meta_attrs = {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "num_frames": n,
        "min_frames_met": True,
        "control_hz": _CONTROL_HZ,
        "task_label": "provenance_fixture",
        "wall_duration_s": float(n) / _CONTROL_HZ,
        "camera_payload_mode": "depth_to_color_aligned_rgbd",
        "camera_type": "eye_to_hand",
        "camera_depth_intrinsics": _PINHOLE,
        "camera_depth_distortion_coeffs": _DISTORTION_COEFFS,
        "camera_depth_width": _CAMERA_WIDTH,
        "camera_depth_height": _CAMERA_HEIGHT,
        "camera_depth_distortion_model": "none",
        "camera_color_intrinsics": _PINHOLE,
        "camera_color_distortion_coeffs": _DISTORTION_COEFFS,
        "camera_color_width": _CAMERA_WIDTH,
        "camera_color_height": _CAMERA_HEIGHT,
        "camera_color_distortion_model": "none",
        "camera_T_color_from_depth": _IDENTITY_4X4,
        "camera_T_xarm_base_from_color": _IDENTITY_4X4,
        "depth_scale": 0.001,
    }
    if provenance_workflow is not None:
        meta_attrs["provenance_workflow"] = provenance_workflow

    with h5py.File(root / "data.h5", "w") as data:
        meta = data.create_group("meta")
        for key, value in meta_attrs.items():
            meta.attrs[key] = value
        for name, spec in DATASET_SPECS.items():
            data.create_dataset(
                name,
                data=_dataset_values(name, n),
                dtype=spec.dtype,
            )

    with h5py.File(root / "depth.h5", "w") as depth:
        depth.create_dataset(
            "depth",
            data=np.zeros((n, _CAMERA_HEIGHT, _CAMERA_WIDTH), dtype=np.uint16),
        )

    with VideoEncoder(
        root / "rgb.mp4", fps=_CONTROL_HZ, width=_CAMERA_WIDTH, height=_CAMERA_HEIGHT
    ) as encoder:
        for _ in range(n):
            encoder.write_frame(
                np.zeros((_CAMERA_HEIGHT, _CAMERA_WIDTH, 3), dtype=np.uint8)
            )

    return root
