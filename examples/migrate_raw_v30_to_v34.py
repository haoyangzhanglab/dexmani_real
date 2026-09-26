#!/usr/bin/env python3
"""One-off, evidence-bound migration from legacy Raw v30 to Raw v34.

This is deliberately the only v30 reader in the repository.  It never changes
the source collection and it does not teach the normal Raw reader about legacy
layouts.  The evidence JSON is an externally audited input, not data inferred
from a v30 file.  Its intentionally small per-episode form is::

    {
      "episodes": {
        "episode_name": {
          "source_hashes": {
            "data.h5": "<sha256>",
            "depth.h5": "<sha256>",
            "rgb.mp4": "<sha256>"
          },
          "evidence": {
            "action": "audited action-lineage reference",
            "observation_action_causality": "audited causal row-alignment reference",
            "tactile": "audited tactile-lineage reference",
            "effort": "audited arm-effort reference",
            "geometry": "audited RGB-D geometry reference",
            "hand_mount": "audited physical hand-mount reference"
          },
          "collection_source": "teleop",
          "tactile_scale_to_native": 1.0,
          "handbase_position_eef_m": [0.0, 0.0, 0.0],
          "handbase_quat_eef_wxyz": [1.0, 0.0, 0.0, 0.0],
          "episode_valid": false,
          "lifecycle_evidence": "optional audited lifecycle reference"
        }
      }
    }

``tactile_scale_to_native`` may only be 1.0 or 10.0.  The latter is allowed
only because the evidence supplied by the caller ties the batch to the known
historical scale/bias conversion.  Missing global semantic evidence blocks
migration.  Missing lifecycle evidence is different: a structurally and
semantically proven episode may be migrated with ``episode_valid=false``.
``collection_source`` must explicitly be ``teleop`` and
``observation_action_causality`` must cite an audit that each row's observation
is causal and precedes the final target represented by that row.  The geometry
reference must cover aligned RGB-D geometry as well as arm/hand units, joint
ordering, EEF convention, and current FK interpretability.  References are
reviewed evidence; this tool never infers those claims from filenames or hashes.

The report records evidence references and decisions.  It is not persisted in
Raw v34, and this tool never fabricates timestamps, calibration, or action
semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from dexmani_real.recording.storage.reader import EpisodeReader
from dexmani_real.recording.storage.schema import DATASET_SPECS, EPISODE_SCHEMA_VERSION
from dexmani_real.utils.atomic_io import atomic_json_dump, atomic_publish, target_is_occupied

LEGACY_SCHEMA_VERSION = 30
_SOURCE_FILE_NAMES = ("data.h5", "depth.h5", "rgb.mp4")
_SEMANTIC_EVIDENCE_NAMES = (
    "action",
    "observation_action_causality",
    "tactile",
    "effort",
    "geometry",
    "hand_mount",
)
_LEGACY_TO_V34 = {
    "arm_qpos": "arm_qpos",
    "arm_qvel": "arm_qvel",
    "arm_effort": "arm_tau",
    "hand_qpos": "hand_qpos",
    "hand_current": "hand_current",
    "hand_contact": "hand_contact",
    "hand_tactile_force": "hand_tactile_force",
    "action_arm_joint_target": "action_arm_joint_sent",
    "action_hand_joint_target": "action_hand_joint",
}
_LEGACY_REQUIRED_SPECS = {
    "arm_qpos": (np.dtype(np.float64), (7,)),
    "arm_qvel": (np.dtype(np.float64), (7,)),
    "arm_tau": (np.dtype(np.float64), (7,)),
    "hand_qpos": (np.dtype(np.float64), (12,)),
    "hand_current": (np.dtype(np.float64), (12,)),
    "hand_contact": (np.dtype(np.float32), (5, 3)),
    "hand_contact_valid": (np.dtype(np.bool_), ()),
    "hand_tactile_force": (np.dtype(np.float32), (5, 120, 3)),
    "hand_tactile_force_valid": (np.dtype(np.bool_), ()),
    "action_arm_joint_sent": (np.dtype(np.float64), (7,)),
    "action_hand_joint": (np.dtype(np.float64), (12,)),
    "flag_action_queued": (np.dtype(np.bool_), ()),
    "flag_frame_status": (np.dtype(np.uint8), ()),
    "observation_valid": (np.dtype(np.bool_), ()),
    "flag_camera_fresh": (np.dtype(np.bool_), ()),
    "arm_connected": (np.dtype(np.bool_), ()),
    "hand_connected": (np.dtype(np.bool_), ()),
    "hand_qpos_stale": (np.dtype(np.bool_), ()),
}


class MigrationError(ValueError):
    """The source or its audited interpretation cannot form Raw v34."""


@dataclass(frozen=True)
class EpisodePlan:
    """Validated facts needed to write one canonical v34 episode."""

    source: Path
    source_hashes: dict[str, str]
    metadata: dict[str, object]
    frame_valid: np.ndarray
    tactile_scale_to_native: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes(episode: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in _SOURCE_FILE_NAMES:
        path = episode / name
        if not path.is_file():
            raise MigrationError(f"{episode.name}: missing source file {name}")
        hashes[name] = _sha256(path)
    return hashes


def _available_source_hashes(episode: Path) -> tuple[dict[str, str | None], list[str]]:
    """Preserve every available hash when a damaged episode blocks migration."""
    hashes: dict[str, str | None] = {}
    errors: list[str] = []
    for name in _SOURCE_FILE_NAMES:
        path = episode / name
        try:
            if not path.is_file():
                raise MigrationError(f"missing source file {name}")
            hashes[name] = _sha256(path)
        except (MigrationError, OSError) as exc:
            hashes[name] = None
            errors.append(str(exc))
    return hashes, errors


def _as_scalar(attrs: h5py.AttributeManager, name: str) -> object:
    if name not in attrs:
        raise MigrationError(f"missing legacy metadata {name}")
    value = np.asarray(attrs[name])
    if value.shape != ():
        raise MigrationError(f"legacy metadata {name} must be scalar")
    return value.item()


def _as_text(attrs: h5py.AttributeManager, name: str) -> str:
    value = _as_scalar(attrs, name)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value.strip():
        raise MigrationError(f"legacy metadata {name} must be a non-empty string")
    return value


def _as_positive_int(attrs: h5py.AttributeManager, name: str) -> int:
    value = _as_scalar(attrs, name)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise MigrationError(f"legacy metadata {name} must be an integer")
    result = int(value)
    if result <= 0:
        raise MigrationError(f"legacy metadata {name} must be positive")
    return result


def _as_positive_float(attrs: h5py.AttributeManager, name: str) -> float:
    value = _as_scalar(attrs, name)
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise MigrationError(f"legacy metadata {name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise MigrationError(f"legacy metadata {name} must be finite and positive")
    return result


def _as_flat_finite(attrs: h5py.AttributeManager, name: str, size: int) -> np.ndarray:
    if name not in attrs:
        raise MigrationError(f"missing legacy metadata {name}")
    try:
        value = np.asarray(attrs[name], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MigrationError(f"legacy metadata {name} must be numeric") from exc
    if value.shape != (size,) or not np.all(np.isfinite(value)):
        raise MigrationError(f"legacy metadata {name} must have shape ({size},) and finite values")
    return value


def _validate_rigid_transform(value: np.ndarray, *, label: str) -> None:
    transform = value.reshape(4, 4)
    rotation = transform[:3, :3]
    if (
        not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9, rtol=0.0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0.0)
    ):
        raise MigrationError(f"legacy metadata {label} must be a finite rigid transform")


def _optional_lifecycle_bool(attrs: h5py.AttributeManager, name: str) -> bool | None:
    if name not in attrs:
        return None
    value = np.asarray(attrs[name])
    if value.shape != ():
        return None
    item = value.item()
    return bool(item) if isinstance(item, (bool, np.bool_)) else None


def _optional_lifecycle_text(attrs: h5py.AttributeManager, name: str) -> str | None:
    if name not in attrs:
        return None
    value = np.asarray(attrs[name])
    if value.shape != ():
        return None
    item = value.item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return item if isinstance(item, str) else None


def _legacy_lifecycle_facts(
    attrs: h5py.AttributeManager,
    data: h5py.File,
    *,
    rows: int,
    control_hz: float,
    frame_valid: np.ndarray,
) -> tuple[dict[str, object], list[str]]:
    """Record legacy lifecycle facts without treating their absence as success."""
    success = _optional_lifecycle_bool(attrs, "success")
    truncated = _optional_lifecycle_bool(attrs, "truncated")
    stop_reason = _optional_lifecycle_text(attrs, "stop_reason")
    camera_writer_error = _optional_lifecycle_text(attrs, "camera_writer_error")
    facts: dict[str, object] = {
        "success": success,
        "truncated": truncated,
        "stop_reason": stop_reason,
        "camera_writer_error": camera_writer_error,
        "valid_observation_anchor_max_gap_ns": None,
    }
    concerns: list[str] = []
    if success is False:
        concerns.append("legacy success=false")
    if truncated is True:
        concerns.append("legacy truncated=true")
    if camera_writer_error:
        concerns.append("legacy camera_writer_error is nonempty")
    if stop_reason in {
        "max_frames",
        "sample_ring_overflow",
        "camera_writer_error",
        "start_error",
        "recorder_process_shutdown",
    }:
        concerns.append(f"legacy abnormal stop_reason={stop_reason}")
    anchors = data.get("observation_anchor_monotonic_ns")
    if isinstance(anchors, h5py.Dataset):
        if anchors.shape == (rows,) and np.issubdtype(anchors.dtype, np.unsignedinteger):
            anchor_values = np.asarray(anchors[:], dtype=np.uint64)
            anchor_deltas = np.diff(anchor_values.astype(np.int64))
            strictly_increasing = bool(np.all(anchor_deltas > 0))
            facts["observation_anchor_strictly_increasing"] = strictly_increasing
            facts["observation_anchor_has_zero"] = bool(np.any(anchor_values == 0))
            facts["observation_anchor_max_gap_ns"] = (
                int(np.max(anchor_deltas)) if anchor_deltas.size else 0
            )
            if not strictly_increasing:
                concerns.append("legacy observation anchor is not strictly increasing")
            if facts["observation_anchor_has_zero"]:
                concerns.append("legacy observation anchor contains zero")
            if anchor_deltas.size and np.max(anchor_deltas) > 2_000_000_000.0 / control_hz:
                concerns.append("legacy observation anchor gap exceeds 2/control_hz")
            # Native continuity advances only on usable control rows. Failed
            # rows must not hide a longer gap between successful demonstrations.
            valid_deltas = np.diff(anchor_values[frame_valid].astype(np.int64))
            facts["valid_observation_anchor_max_gap_ns"] = (
                int(np.max(valid_deltas)) if valid_deltas.size else 0
            )
            if valid_deltas.size and np.max(valid_deltas) > 2_000_000_000.0 / control_hz:
                concerns.append("legacy valid observation anchor gap exceeds 2/control_hz")
        else:
            facts["observation_anchor_strictly_increasing"] = None
            facts["observation_anchor_has_zero"] = None
            facts["observation_anchor_max_gap_ns"] = None
    else:
        facts["observation_anchor_strictly_increasing"] = None
        facts["observation_anchor_has_zero"] = None
        facts["observation_anchor_max_gap_ns"] = None
    return facts, concerns


def _legacy_metadata(attrs: h5py.AttributeManager) -> dict[str, object]:
    schema_version = _as_positive_int(attrs, "schema_version")
    if schema_version != LEGACY_SCHEMA_VERSION:
        raise MigrationError(
            f"unsupported legacy schema v{schema_version}; expected v{LEGACY_SCHEMA_VERSION}"
        )
    if _as_text(attrs, "camera_payload_mode") != "depth_to_color_aligned_rgbd":
        raise MigrationError("legacy camera payload is not depth_to_color_aligned_rgbd")
    if _as_text(attrs, "camera_type") != "eye_to_hand":
        raise MigrationError("legacy camera_type is not eye_to_hand")

    width = _as_positive_int(attrs, "camera_color_width")
    height = _as_positive_int(attrs, "camera_color_height")
    intrinsics = _as_flat_finite(attrs, "camera_color_intrinsics", 9)
    if not np.allclose(intrinsics.reshape(3, 3)[2], (0.0, 0.0, 1.0), atol=1e-9, rtol=0.0):
        raise MigrationError("legacy camera_color_intrinsics is not a canonical pinhole matrix")
    if intrinsics[0] <= 0.0 or intrinsics[4] <= 0.0:
        raise MigrationError("legacy camera_color_intrinsics must have positive focal lengths")
    transform = _as_flat_finite(attrs, "camera_T_xarm_base_from_color", 16)
    _validate_rigid_transform(transform, label="camera_T_xarm_base_from_color")

    return {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "task_label": _as_text(attrs, "task_label"),
        "collection_source": "teleop",
        "control_hz": _as_positive_float(attrs, "control_hz"),
        "num_frames": _as_positive_int(attrs, "num_frames"),
        "camera_payload_mode": "depth_to_color_aligned_rgbd",
        "camera_color_width": width,
        "camera_color_height": height,
        "camera_color_intrinsics": intrinsics,
        "camera_color_distortion_model": _as_text(attrs, "camera_color_distortion_model"),
        "camera_color_distortion_coeffs": _as_flat_finite(
            attrs, "camera_color_distortion_coeffs", 5
        ),
        "camera_T_xarm_base_from_color": transform,
        "depth_scale": _as_positive_float(attrs, "depth_scale"),
    }


def _validate_legacy_datasets(data: h5py.File, *, rows: int) -> None:
    for name, (dtype, tail_shape) in _LEGACY_REQUIRED_SPECS.items():
        if name not in data or not isinstance(data[name], h5py.Dataset):
            raise MigrationError(f"missing legacy dataset {name}")
        dataset = data[name]
        if dataset.shape != (rows,) + tail_shape:
            raise MigrationError(
                f"legacy dataset {name} has shape {dataset.shape}, expected {(rows,) + tail_shape}"
            )
        if np.dtype(dataset.dtype) != dtype:
            raise MigrationError(
                f"legacy dataset {name} has dtype {dataset.dtype}, expected {dtype}"
            )


def _validate_legacy_depth(episode: Path, *, rows: int, height: int, width: int) -> None:
    with h5py.File(episode / "depth.h5", "r") as depth_file:
        if set(depth_file.keys()) != {"depth"}:
            raise MigrationError("legacy depth.h5 must contain exactly the depth dataset")
        depth = depth_file["depth"]
        if not isinstance(depth, h5py.Dataset):
            raise MigrationError("legacy depth entry is not a dataset")
        if depth.shape != (rows, height, width):
            raise MigrationError(
                f"legacy depth shape {depth.shape}, expected {(rows, height, width)}"
            )
        if np.dtype(depth.dtype) != np.dtype(np.uint16):
            raise MigrationError(f"legacy depth dtype {depth.dtype}, expected uint16")


def _validate_hash_evidence(entry: object, source_hashes: Mapping[str, str]) -> list[str]:
    if not isinstance(entry, Mapping):
        return ["missing per-episode evidence entry"]
    value = entry.get("source_hashes")
    if not isinstance(value, Mapping):
        return ["evidence.source_hashes must be an object"]
    errors: list[str] = []
    for name, actual in source_hashes.items():
        expected = value.get(name)
        if not isinstance(expected, str) or expected.lower() != actual:
            errors.append(f"evidence source hash mismatch for {name}")
    return errors


def _nonempty_reference(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _evidence_plan(
    entry: object,
    source_hashes: Mapping[str, str],
    lifecycle_concerns: list[str],
) -> tuple[dict[str, object] | None, list[str], list[str]]:
    """Return semantic evidence, fatal semantic errors, and lifecycle notes."""
    hash_errors = _validate_hash_evidence(entry, source_hashes)
    if not isinstance(entry, Mapping):
        return None, hash_errors, ["lifecycle evidence unavailable; episode_valid=false"]
    if entry.get("collection_source") != "teleop":
        hash_errors.append("evidence collection_source must explicitly be teleop")
    evidence = entry.get("evidence")
    errors = list(hash_errors)
    if not isinstance(evidence, Mapping):
        errors.append("evidence.evidence must be an object")
    else:
        for name in _SEMANTIC_EVIDENCE_NAMES:
            if not _nonempty_reference(evidence.get(name)):
                errors.append(f"missing audited {name} evidence reference")

    scale_value = entry.get("tactile_scale_to_native")
    if isinstance(scale_value, bool) or not isinstance(scale_value, (int, float)):
        errors.append("tactile_scale_to_native must be 1.0 or 10.0")
    else:
        scale = float(scale_value)
        if scale not in (1.0, 10.0):
            errors.append("tactile_scale_to_native must be 1.0 or 10.0")

    try:
        mount_position = np.asarray(entry["handbase_position_eef_m"], dtype=np.float64)
        mount_quaternion = np.asarray(entry["handbase_quat_eef_wxyz"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        errors.append("evidence must provide numeric hand mount position and quaternion")
        mount_position = mount_quaternion = np.array([], dtype=np.float64)
    if mount_position.shape != (3,) or not np.all(np.isfinite(mount_position)):
        errors.append("handbase_position_eef_m must have shape (3,) and finite values")
    if mount_quaternion.shape != (4,) or not np.all(np.isfinite(mount_quaternion)):
        errors.append("handbase_quat_eef_wxyz must have shape (4,) and finite values")
    elif not np.isclose(np.linalg.norm(mount_quaternion), 1.0, atol=1e-6, rtol=0.0):
        errors.append("handbase_quat_eef_wxyz must be unit length within 1e-6")

    lifecycle_notes: list[str] = []
    lifecycle_reference = entry.get("lifecycle_evidence")
    requested_valid = entry.get("episode_valid")
    episode_valid = False
    if lifecycle_concerns:
        lifecycle_notes.extend(lifecycle_concerns)
    elif isinstance(requested_valid, bool) and _nonempty_reference(lifecycle_reference):
        episode_valid = requested_valid
    elif requested_valid is True:
        lifecycle_notes.append("episode_valid=true ignored without lifecycle_evidence")
    else:
        lifecycle_notes.append("lifecycle evidence unavailable; episode_valid=false")

    if errors:
        return None, errors, lifecycle_notes
    assert isinstance(evidence, Mapping)
    return (
        {
            "references": {name: str(evidence[name]) for name in _SEMANTIC_EVIDENCE_NAMES},
            "tactile_scale_to_native": float(scale_value),
            "handbase_position_eef_m": mount_position,
            "handbase_quat_eef_wxyz": mount_quaternion,
            "episode_valid": episode_valid,
            "lifecycle_evidence": (
                str(lifecycle_reference) if _nonempty_reference(lifecycle_reference) else None
            ),
        },
        [],
        lifecycle_notes,
    )


def _row_nonfinite(values: np.ndarray) -> list[int]:
    if values.ndim == 1:
        valid = np.isfinite(values)
    else:
        valid = np.all(np.isfinite(values), axis=tuple(range(1, values.ndim)))
    return np.flatnonzero(~valid).astype(int).tolist()


def _tactile_invariant_errors(payload: np.ndarray, valid: np.ndarray) -> list[int]:
    axes = tuple(range(1, payload.ndim))
    fully_finite = np.all(np.isfinite(payload), axis=axes)
    contains_nan = np.any(np.isnan(payload), axis=axes)
    inconsistent = (valid & ~fully_finite) | (~valid & ~contains_nan)
    return np.flatnonzero(inconsistent).astype(int).tolist()


def _frame_valid(data: h5py.File) -> np.ndarray:
    """Map proven legacy controls without copying the legacy status enum."""
    return np.asarray(
        (data["flag_frame_status"][:] == 0)
        & data["flag_action_queued"][:]
        & data["observation_valid"][:]
        & data["flag_camera_fresh"][:]
        & data["arm_connected"][:]
        & data["hand_connected"][:]
        & ~data["hand_qpos_stale"][:],
        dtype=np.bool_,
    )


def _normalize_tactile(values: np.ndarray, scale: float) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    if scale == 1.0:
        return source.copy()
    return np.asarray(source * np.float32(scale), dtype=np.float32)


def _inspect_episode(
    episode: Path,
    evidence_entry: object,
) -> tuple[dict[str, object], EpisodePlan | None]:
    report: dict[str, object] = {
        "episode": episode.name,
        "source_hashes": {},
        "source_rows": None,
        "frame_valid_false_rows": [],
        "nonfinite_rows": {},
        "tactile_validity_mismatches": {},
        "migration_result": "blocked",
        "migration_reason": None,
        "zarr_eligible": False,
        "zarr_rejection_reason": None,
    }
    try:
        source_hashes = _source_hashes(episode)
    except (MigrationError, OSError) as exc:
        available_hashes, hash_errors = _available_source_hashes(episode)
        report["source_hashes"] = available_hashes
        report["migration_reason"] = "; ".join([str(exc), *hash_errors])
        report["zarr_rejection_reason"] = "migration source files are unavailable"
        return report, None
    report["source_hashes"] = source_hashes
    try:
        with h5py.File(episode / "data.h5", "r") as data:
            if "meta" not in data or not isinstance(data["meta"], h5py.Group):
                raise MigrationError("legacy data.h5 is missing meta")
            metadata = _legacy_metadata(data["meta"].attrs)
            rows = int(metadata["num_frames"])
            report["source_rows"] = rows
            _validate_legacy_datasets(data, rows=rows)
            frame_valid = _frame_valid(data)
            lifecycle_facts, lifecycle_concerns = _legacy_lifecycle_facts(
                data["meta"].attrs,
                data,
                rows=rows,
                control_hz=float(metadata["control_hz"]),
                frame_valid=frame_valid,
            )
            report["legacy_lifecycle_facts"] = lifecycle_facts
            _validate_legacy_depth(
                episode,
                rows=rows,
                height=int(metadata["camera_color_height"]),
                width=int(metadata["camera_color_width"]),
            )
            report["frame_valid_false_rows"] = np.flatnonzero(~frame_valid).astype(int).tolist()

            tactile_mismatches = {
                "hand_contact": _tactile_invariant_errors(
                    np.asarray(data["hand_contact"][:]),
                    np.asarray(data["hand_contact_valid"][:], dtype=bool),
                ),
                "hand_tactile_force": _tactile_invariant_errors(
                    np.asarray(data["hand_tactile_force"][:]),
                    np.asarray(data["hand_tactile_force_valid"][:], dtype=bool),
                ),
            }
            report["tactile_validity_mismatches"] = {
                name: rows for name, rows in tactile_mismatches.items() if rows
            }

            nonfinite_rows = {
                target: _row_nonfinite(np.asarray(data[source][:]))
                for target, source in _LEGACY_TO_V34.items()
            }
            report["nonfinite_rows"] = {name: rows for name, rows in nonfinite_rows.items() if rows}
    except (MigrationError, OSError, ValueError) as exc:
        report["migration_reason"] = str(exc)
        report["zarr_rejection_reason"] = "migration source is structurally invalid"
        return report, None

    evidence, evidence_errors, lifecycle_notes = _evidence_plan(
        evidence_entry,
        source_hashes,
        lifecycle_concerns,
    )
    report["lineage_evidence"] = evidence["references"] if evidence is not None else None
    report["lifecycle_evidence"] = evidence["lifecycle_evidence"] if evidence is not None else None
    report["lifecycle_decision"] = {
        "episode_valid": bool(evidence["episode_valid"]) if evidence is not None else False,
        "notes": lifecycle_notes,
    }
    mismatch_values = report["tactile_validity_mismatches"]
    assert isinstance(mismatch_values, Mapping)
    if evidence_errors or mismatch_values:
        reasons = evidence_errors + [
            f"{name} validity/payload mismatch at row {rows[0]}"
            for name, rows in mismatch_values.items()
            if isinstance(rows, list) and rows
        ]
        report["migration_reason"] = "; ".join(reasons)
        report["zarr_rejection_reason"] = "migration did not establish canonical Raw v34 semantics"
        return report, None

    assert evidence is not None
    scale = float(evidence["tactile_scale_to_native"])
    with h5py.File(episode / "data.h5", "r") as data:
        for name in ("hand_contact", "hand_tactile_force"):
            source = np.asarray(data[name][:])
            normalized = _normalize_tactile(source, scale)
            valid_name = f"{name}_valid"
            valid = np.asarray(data[valid_name][:], dtype=bool)
            axes = tuple(range(1, normalized.ndim))
            if np.any(valid & ~np.all(np.isfinite(normalized), axis=axes)):
                report["migration_reason"] = (
                    f"{name} normalization produces nonfinite valid payload"
                )
                report["zarr_rejection_reason"] = "migration normalization is not finite"
                return report, None

    metadata["handbase_position_eef_m"] = np.asarray(
        evidence["handbase_position_eef_m"], dtype=np.float64
    )
    metadata["handbase_quat_eef_wxyz"] = np.asarray(
        evidence["handbase_quat_eef_wxyz"], dtype=np.float64
    )
    metadata["episode_valid"] = bool(evidence["episode_valid"])
    plan = EpisodePlan(
        source=episode,
        source_hashes=source_hashes,
        metadata=metadata,
        frame_valid=frame_valid,
        tactile_scale_to_native=scale,
    )
    report["migration_result"] = "eligible"
    if not bool(metadata["episode_valid"]):
        report["zarr_rejection_reason"] = "episode_valid=false"
    elif not bool(np.all(frame_valid)):
        report["zarr_rejection_reason"] = "frame_valid=false"
    elif report["nonfinite_rows"]:
        report["zarr_rejection_reason"] = "nonfinite canonical payload"
    else:
        report["zarr_eligible"] = None
        report["zarr_rejection_reason"] = (
            "pending normal Raw-to-Zarr validation of RGB and derived modalities"
        )
    return report, plan


def _write_v34_episode(plan: EpisodePlan, destination: Path) -> None:
    """Write only v34 research data into an owned, unpublished directory."""
    destination.mkdir()
    with (
        h5py.File(plan.source / "data.h5", "r") as source,
        h5py.File(destination / "data.h5", "w") as target,
    ):
        meta = target.create_group("meta")
        for name, value in plan.metadata.items():
            meta.attrs[name] = value
        for target_name, spec in DATASET_SPECS.items():
            if target_name == "frame_valid":
                values = np.asarray(plan.frame_valid, dtype=spec.dtype)
            else:
                source_name = _LEGACY_TO_V34[target_name]
                values = np.asarray(source[source_name][:], dtype=spec.dtype)
                if target_name in {"hand_contact", "hand_tactile_force"}:
                    values = _normalize_tactile(values, plan.tactile_scale_to_native)
            if values.shape != (int(plan.metadata["num_frames"]),) + spec.tail_shape:
                raise MigrationError(f"{plan.source.name}: mapped {target_name} has wrong shape")
            target.create_dataset(target_name, data=values, compression="gzip")
    for name in ("depth.h5", "rgb.mp4"):
        shutil.copyfile(plan.source / name, destination / name)


def _equal_arrays(actual: np.ndarray, expected: np.ndarray) -> bool:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return False
    if np.issubdtype(actual.dtype, np.inexact):
        return bool(np.array_equal(actual, expected, equal_nan=True))
    return bool(np.array_equal(actual, expected))


def _verify_v34_episode(plan: EpisodePlan, destination: Path) -> None:
    """Reopen through the normal v34 reader and check every mapping exactly."""
    with EpisodeReader(destination) as reader, h5py.File(plan.source / "data.h5", "r") as source:
        if reader.num_frames != int(plan.metadata["num_frames"]):
            raise MigrationError(f"{plan.source.name}: v34 row count changed during migration")
        if reader.episode_valid != bool(plan.metadata["episode_valid"]):
            raise MigrationError(f"{plan.source.name}: v34 episode_valid changed during migration")
        for target_name, spec in DATASET_SPECS.items():
            actual = np.asarray(reader.h5f[target_name][:], dtype=spec.dtype)
            if target_name == "frame_valid":
                expected = np.asarray(plan.frame_valid, dtype=spec.dtype)
            else:
                expected = np.asarray(source[_LEGACY_TO_V34[target_name]][:], dtype=spec.dtype)
                if target_name in {"hand_contact", "hand_tactile_force"}:
                    expected = _normalize_tactile(expected, plan.tactile_scale_to_native)
            if not _equal_arrays(actual, expected):
                raise MigrationError(f"{plan.source.name}: v34 mapping mismatch for {target_name}")
    for name in ("depth.h5", "rgb.mp4"):
        if _sha256(destination / name) != plan.source_hashes[name]:
            raise MigrationError(f"{plan.source.name}: {name} was not copied byte-for-byte")


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError:
        return False


def _validate_paths(
    source: Path,
    output: Path,
    report: Path,
    evidence: Path | None,
) -> None:
    if not source.is_dir():
        raise MigrationError(f"SOURCE_ROOT is not a directory: {source}")
    source_resolved = source.resolve()
    output_resolved = output.resolve(strict=False)
    report_resolved = report.resolve(strict=False)
    if _is_within(output_resolved, source_resolved) or _is_within(source_resolved, output_resolved):
        raise MigrationError("OUTPUT_ROOT must not overlap SOURCE_ROOT")
    if _is_within(report_resolved, source_resolved) or _is_within(report_resolved, output_resolved):
        raise MigrationError("REPORT_JSON must be outside SOURCE_ROOT and OUTPUT_ROOT")
    if evidence is not None:
        evidence_resolved = evidence.resolve()
        if _is_within(evidence_resolved, output_resolved):
            raise MigrationError("EVIDENCE_JSON must be outside OUTPUT_ROOT")
        if report_resolved == evidence_resolved or _same_file(report, evidence):
            raise MigrationError("REPORT_JSON must not overwrite the evidence file")
    if target_is_occupied(output):
        raise FileExistsError(f"refusing to overwrite existing output root: {output}")
    if report.is_symlink():
        raise MigrationError("REPORT_JSON must not be a symlink")
    for source_file in source.rglob("*"):
        if source_file.is_file() and _same_file(report, source_file):
            raise MigrationError("REPORT_JSON must not overwrite a source file or media symlink")


def _load_evidence(path: Path | None) -> Mapping[str, object]:
    if path is None:
        return {"episodes": {}}
    if not path.is_file():
        raise MigrationError(f"evidence file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            loaded = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise MigrationError(f"could not read evidence JSON {path}: {exc}") from exc
    if not isinstance(loaded, Mapping) or not isinstance(loaded.get("episodes"), Mapping):
        raise MigrationError("evidence JSON must contain an episodes object")
    return loaded


def _episode_directories(source: Path) -> list[Path]:
    episodes = sorted(
        path for path in source.iterdir() if path.is_dir() and not path.name.startswith(".")
    )
    if not episodes:
        raise MigrationError(f"SOURCE_ROOT contains no episode directories: {source}")
    if any(path.is_symlink() for path in episodes):
        raise MigrationError("SOURCE_ROOT must not contain symlinked episode directories")
    return episodes


def _report_header(
    source: Path, output: Path, *, dry_run: bool, evidence_path: Path | None
) -> dict[str, object]:
    return {
        "source_root": str(source),
        "output_root": str(output),
        "source_schema": LEGACY_SCHEMA_VERSION,
        "target_schema": EPISODE_SCHEMA_VERSION,
        "dry_run": dry_run,
        "evidence_path": str(evidence_path) if evidence_path is not None else None,
        "tool_code_sha256": _sha256(Path(__file__).resolve()),
        "lineage_evidence_summary": {
            "required": list(_SEMANTIC_EVIDENCE_NAMES),
            "rule": "per-episode hashes and nonempty audited references are required",
        },
        "action_mapping": {
            "action_arm_joint_sent": "action_arm_joint_target",
            "action_hand_joint": "action_hand_joint_target",
            "legacy_action_arm_ee": "not migrated",
        },
        "tactile_normalization": {
            "representation": "XHand SDK-native bias-corrected payload",
            "allowed_scale_to_native": [1.0, 10.0],
            "invalid_payload_rule": "legacy false flag requires a payload containing NaN",
        },
        "camera_normalization": "copy audited aligned color-grid metadata only",
        "physical_calibration_decisions": "hand mount comes only from audited evidence JSON",
        "episodes": [],
    }


def _write_report(report: Mapping[str, object], path: Path) -> None:
    atomic_json_dump(report, path, indent=2, ensure_ascii=False)


def _record_source_hashes_after(
    source: Path,
    report_episodes: list[dict[str, object]],
) -> tuple[list[str], list[str]]:
    changed: list[str] = []
    unreadable: list[str] = []
    for entry in report_episodes:
        episode_path = source / str(entry["episode"])
        try:
            after = _source_hashes(episode_path)
        except (MigrationError, OSError) as exc:
            available_hashes, hash_errors = _available_source_hashes(episode_path)
            entry["source_hashes_after"] = available_hashes
            unreadable.append(f"{episode_path.name}: {exc}; {'; '.join(hash_errors)}")
            continue
        entry["source_hashes_after"] = after
        if after != entry["source_hashes"]:
            changed.append(episode_path.name)
    return changed, unreadable


def run_migration(
    source: Path,
    output: Path,
    report_path: Path,
    evidence_path: Path | None,
    *,
    dry_run: bool,
) -> int:
    _validate_paths(source, output, report_path, evidence_path)
    evidence_payload = _load_evidence(evidence_path)
    evidence_episodes = evidence_payload["episodes"]
    assert isinstance(evidence_episodes, Mapping)
    report = _report_header(source, output, dry_run=dry_run, evidence_path=evidence_path)
    report_episodes = report["episodes"]
    assert isinstance(report_episodes, list)

    plans: list[tuple[dict[str, object], EpisodePlan]] = []
    for episode in _episode_directories(source):
        entry, plan = _inspect_episode(episode, evidence_episodes.get(episode.name))
        report_episodes.append(entry)
        if plan is not None:
            plans.append((entry, plan))

    source_hash_changed, source_hash_unreadable = _record_source_hashes_after(
        source, report_episodes
    )
    report["source_hash_check"] = {
        "ok": not source_hash_changed and not source_hash_unreadable,
        "changed": source_hash_changed,
        "unreadable": source_hash_unreadable,
    }
    report["summary"] = {
        "source_episodes": len(report_episodes),
        "source_rows": sum(int(entry["source_rows"] or 0) for entry in report_episodes),
        "eligible_episodes": len(plans),
        "blocked_episodes": len(report_episodes) - len(plans),
        "output_published": False,
    }
    if dry_run:
        _write_report(report, report_path)
        return 1 if source_hash_changed else 0

    staging_root: Path | None = None
    try:
        if source_hash_changed:
            report["summary"]["reason"] = "source hashes changed during audit"  # type: ignore[index]
            _write_report(report, report_path)
            return 1
        if not plans:
            report["summary"]["reason"] = "no semantically eligible episodes; no output published"  # type: ignore[index]
            _write_report(report, report_path)
            return 1
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.parent.is_dir():
            raise MigrationError(f"OUTPUT_ROOT parent is not a directory: {output.parent}")
        staging_root = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.raw-v30-v34-", dir=output.parent)
        )
        for entry, plan in plans:
            episode_staging = staging_root / f".tmp_{plan.source.name}"
            episode_target = staging_root / plan.source.name
            try:
                _write_v34_episode(plan, episode_staging)
                _verify_v34_episode(plan, episode_staging)
                atomic_publish(episode_staging, episode_target)
                entry["migration_result"] = "migrated"
                entry["migration_reason"] = None
                entry["output_hashes"] = {
                    name: _sha256(episode_target / name) for name in _SOURCE_FILE_NAMES
                }
            except (MigrationError, OSError, ValueError) as exc:
                if episode_staging.exists():
                    shutil.rmtree(episode_staging)
                entry["migration_result"] = "failed"
                entry["migration_reason"] = str(exc)
                entry["zarr_eligible"] = False
                entry["zarr_rejection_reason"] = "Raw v34 migration failed"

        source_hash_changed, source_hash_unreadable = _record_source_hashes_after(
            source, report_episodes
        )
        if source_hash_changed or source_hash_unreadable:
            report["summary"]["reason"] = "source hashes changed during migration"  # type: ignore[index]
            report["source_hash_check"] = {
                "ok": False,
                "changed": source_hash_changed,
                "unreadable": source_hash_unreadable,
            }
            _write_report(report, report_path)
            return 1

        migrated = [entry for entry in report_episodes if entry["migration_result"] == "migrated"]
        if not migrated:
            report["summary"]["reason"] = "no episode completed migration; no output published"  # type: ignore[index]
            _write_report(report, report_path)
            return 1
        if target_is_occupied(output):
            raise FileExistsError(f"refusing to overwrite existing output root: {output}")
        atomic_publish(staging_root, output)
        staging_root = None
        report["summary"] = {
            "source_episodes": len(report_episodes),
            "source_rows": sum(int(entry["source_rows"] or 0) for entry in report_episodes),
            "eligible_episodes": len(plans),
            "migrated_episodes": len(migrated),
            "blocked_or_failed_episodes": len(report_episodes) - len(migrated),
            "output_published": True,
        }
        report["source_hash_check"] = {"ok": True, "changed": [], "unreadable": []}
        _write_report(report, report_path)
        return 0
    except BaseException:
        if staging_root is not None and staging_root.exists():
            shutil.rmtree(staging_root)
        raise
    finally:
        if staging_root is not None and staging_root.exists():
            shutil.rmtree(staging_root)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path, metavar="SOURCE_ROOT")
    parser.add_argument("--output", type=Path, required=True, metavar="OUTPUT_ROOT")
    parser.add_argument("--report", type=Path, required=True, metavar="REPORT_JSON")
    parser.add_argument(
        "--evidence",
        type=Path,
        metavar="EVIDENCE_JSON",
        help="externally audited evidence; absent means audit-only blockers",
    )
    parser.add_argument("--dry-run", action="store_true", help="audit only; never create Raw v34")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return run_migration(
            args.source_root.expanduser(),
            args.output.expanduser(),
            args.report.expanduser(),
            args.evidence.expanduser() if args.evidence is not None else None,
            dry_run=args.dry_run,
        )
    except (MigrationError, FileExistsError, OSError, ValueError) as exc:
        raise SystemExit(f"migration failed: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
