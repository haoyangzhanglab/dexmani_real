"""Offline replay preflight certificates bound to code/data safety inputs."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from dexmani_real.ipc.schema import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE


def _sha256_bytes(parts: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return digest.hexdigest()


def hash_arrays(*arrays: np.ndarray) -> str:
    parts: list[bytes] = []
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        parts.extend((str(contiguous.dtype).encode(), repr(contiguous.shape).encode(), contiguous.tobytes()))
    return _sha256_bytes(parts)


def hash_files(paths: Iterable[str | Path]) -> str:
    parts: list[bytes] = []
    for raw_path in sorted((Path(path) for path in paths), key=lambda path: str(path)):
        parts.extend((str(raw_path).encode(), raw_path.read_bytes()))
    return _sha256_bytes(parts)


def _canonical_scene_value(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return {field.name: _canonical_scene_value(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _canonical_scene_value(item) for key, item in sorted(value.items())}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_canonical_scene_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "to_dict"):
        return _canonical_scene_value(value.to_dict())  # type: ignore[union-attr]
    raise TypeError(f"unsupported collision scene value {type(value).__name__}")


def hash_collision_scene(paths: Iterable[str | Path], static_boxes: Iterable[object]) -> str:
    """Bind collision model contents and ordered, normalized static geometry."""
    model_contents_hash = _sha256_bytes(
        path.read_bytes() for path in sorted((Path(raw_path) for raw_path in paths), key=lambda item: str(item))
    )
    scene_json = json.dumps(
        _canonical_scene_value(tuple(static_boxes)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return _sha256_bytes((model_contents_hash.encode("ascii"), scene_json.encode("utf-8")))


@dataclass(frozen=True)
class PreflightCertificate:
    version: int
    trajectory_sha256: str
    collision_model_sha256: str
    workspace_sha256: str
    resolved_config_sha256: str
    source_episode: str
    frame_count: int
    hand_enabled: bool
    checks_run: tuple[str, ...]
    created_utc: str
    certificate_sha256: str
    collision_scene_sha256: str | None = None

    def _payload(self) -> dict[str, object]:
        data = asdict(self)
        data.pop("certificate_sha256")
        data["checks_run"] = list(self.checks_run)
        # Preserve the exact v1 payload so historical certificate checksums
        # remain readable and verifiable.
        if self.version == 1 and self.collision_scene_sha256 is None:
            data.pop("collision_scene_sha256")
        return data

    def verify_integrity(self) -> None:
        canonical = json.dumps(self._payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if actual != self.certificate_sha256:
            raise ValueError("preflight certificate checksum mismatch")

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite preflight certificate: {target}")
        payload = asdict(self)
        payload["checks_run"] = list(self.checks_run)
        fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True, indent=2, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.rename(temp_name, target)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise
        return target

    @classmethod
    def read(cls, path: str | Path) -> "PreflightCertificate":
        with Path(path).open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
        raw["checks_run"] = tuple(raw["checks_run"])
        certificate = cls(**raw)
        certificate.verify_integrity()
        return certificate


def create_preflight_certificate(
    *,
    source_episode: str,
    arm_actions: np.ndarray,
    hand_actions: np.ndarray | None,
    collision_model_paths: Iterable[str | Path],
    workspace_bounds_m: np.ndarray,
    resolved_config_sha256: str,
    transition_check: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], bool],
    workspace_check: Callable[[np.ndarray, np.ndarray], bool],
    table_check: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], bool],
    hand_enabled: bool | None = None,
    static_boxes: Iterable[object] = (),
) -> PreflightCertificate:
    """Densely validate every replay transition and checksum-bind the exact inputs."""
    arm_actions = np.asarray(arm_actions, dtype=np.float64)
    if arm_actions.ndim != 2 or arm_actions.shape[1:] != ARM_JOINT_SHAPE or not np.all(np.isfinite(arm_actions)):
        raise ValueError("preflight arm trajectory must be finite shape (T, 7)")
    if arm_actions.shape[0] < 1:
        raise ValueError("preflight trajectory is empty")
    if hand_actions is None:
        hand = np.zeros((arm_actions.shape[0], *HAND_JOINT_SHAPE), dtype=np.float64)
        resolved_hand_enabled = False if hand_enabled is None else bool(hand_enabled)
    else:
        hand = np.asarray(hand_actions, dtype=np.float64)
        if hand.shape != (arm_actions.shape[0], *HAND_JOINT_SHAPE) or not np.all(np.isfinite(hand)):
            raise ValueError("preflight hand trajectory must be finite shape (T, 12)")
        resolved_hand_enabled = True if hand_enabled is None else bool(hand_enabled)
    workspace = np.asarray(workspace_bounds_m, dtype=np.float64)
    if workspace.shape != (3, 2) or not np.all(np.isfinite(workspace)) or np.any(workspace[:, 0] > workspace[:, 1]):
        raise ValueError("preflight workspace must be finite shape (3, 2)")
    if len(resolved_config_sha256) != 64:
        raise ValueError("preflight requires a full resolved config SHA-256")

    # Endpoints and every transition are checked. Callbacks must themselves
    # sample densely enough for their geometry model; any exception fails closed.
    for index in range(max(1, arm_actions.shape[0] - 1)):
        start_index = min(index, arm_actions.shape[0] - 1)
        end_index = min(index + 1, arm_actions.shape[0] - 1)
        arm_start, arm_end = arm_actions[start_index], arm_actions[end_index]
        hand_start, hand_end = hand[start_index], hand[end_index]
        if not workspace_check(arm_start, arm_end):
            raise ValueError(f"preflight workspace rejection at transition {start_index}->{end_index}")
        if not transition_check(arm_start, arm_end, hand_start, hand_end):
            raise ValueError(f"preflight collision rejection at transition {start_index}->{end_index}")
        if not table_check(arm_start, arm_end, hand_start, hand_end):
            raise ValueError(f"preflight table rejection at transition {start_index}->{end_index}")

    trajectory_hash = hash_arrays(arm_actions, hand)
    collision_paths = tuple(collision_model_paths)
    boxes = tuple(static_boxes)
    collision_hash = hash_files(collision_paths)
    collision_scene_hash = hash_collision_scene(collision_paths, boxes)
    workspace_hash = hash_arrays(workspace)
    checks_run = (
        "shape_finite",
        "workspace_dense",
        "arm_hand_collision_dense",
        "environment_collision_dense",
        "table_clearance_dense",
    )
    payload: dict[str, object] = {
        "version": 2,
        "trajectory_sha256": trajectory_hash,
        "collision_model_sha256": collision_hash,
        "collision_scene_sha256": collision_scene_hash,
        "workspace_sha256": workspace_hash,
        "resolved_config_sha256": resolved_config_sha256,
        "source_episode": str(Path(source_episode).resolve()),
        "frame_count": int(arm_actions.shape[0]),
        "hand_enabled": resolved_hand_enabled,
        "checks_run": list(checks_run),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return PreflightCertificate(
        version=2,
        trajectory_sha256=trajectory_hash,
        collision_model_sha256=collision_hash,
        workspace_sha256=workspace_hash,
        resolved_config_sha256=resolved_config_sha256,
        source_episode=str(Path(source_episode).resolve()),
        frame_count=int(arm_actions.shape[0]),
        hand_enabled=resolved_hand_enabled,
        checks_run=checks_run,
        created_utc=str(payload["created_utc"]),
        certificate_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        collision_scene_sha256=collision_scene_hash,
    )


def verify_preflight_binding(
    certificate: PreflightCertificate,
    *,
    source_episode: str,
    arm_actions: np.ndarray,
    hand_actions: np.ndarray | None,
    collision_model_paths: Iterable[str | Path],
    workspace_bounds_m: np.ndarray,
    resolved_config_sha256: str,
    hand_enabled: bool | None = None,
    static_boxes: Iterable[object] = (),
) -> None:
    certificate.verify_integrity()
    boxes = tuple(static_boxes)
    if certificate.version not in (1, 2):
        raise ValueError(f"unsupported preflight certificate version {certificate.version}")
    if certificate.version == 1 and boxes:
        raise ValueError("preflight v1 certificate cannot validate a non-empty static collision scene")
    hand = (
        np.zeros((len(arm_actions), *HAND_JOINT_SHAPE), dtype=np.float64)
        if hand_actions is None
        else np.asarray(hand_actions, dtype=np.float64)
    )
    collision_paths = tuple(collision_model_paths)
    actual = {
        "source_episode": str(Path(source_episode).resolve()),
        "trajectory_sha256": hash_arrays(np.asarray(arm_actions, dtype=np.float64), hand),
        "collision_model_sha256": hash_files(collision_paths),
        "workspace_sha256": hash_arrays(np.asarray(workspace_bounds_m, dtype=np.float64)),
        "resolved_config_sha256": resolved_config_sha256,
        "frame_count": len(arm_actions),
        "hand_enabled": hand_actions is not None if hand_enabled is None else bool(hand_enabled),
    }
    if certificate.version == 2:
        actual["collision_scene_sha256"] = hash_collision_scene(collision_paths, boxes)
    mismatches = [key for key, value in actual.items() if getattr(certificate, key) != value]
    if mismatches:
        raise ValueError(f"preflight certificate binding mismatch: {mismatches}")
