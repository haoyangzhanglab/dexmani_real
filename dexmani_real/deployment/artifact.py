"""Resolve one hash-bound policy artifact without loading its checkpoint.

This module owns the untrusted experiment-directory boundary used before the
inference worker exists.  It validates the small adjacent index sidecar and
pins the selected checkpoint by path and lstat identity; checkpoint contents
are deliberately neither deserialized nor hashed here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dexmani_real.ipc.schema import (
    ARM_DOF,
    EE_POS_DIM,
    EE_ROT6D_DIM,
    HAND_DOF,
    MAX_POLICY_CHUNK_STEPS,
    POINT_CLOUD_FEATURE_DIM,
    SUPPORTED_POINT_CLOUD_COUNTS,
)

MAX_POLICY_ARTIFACT_INDEX_BYTES = 64 * 1024

_SIDECAR_KEYS = frozenset(
    {
        "schema_version",
        "checkpoint",
        "embedded_contract_sha256",
        "allocation",
        "producer",
    }
)
_CHECKPOINT_KEYS = frozenset({"filename", "size_bytes", "sha256"})
_ALLOCATION_KEYS = frozenset(
    {
        "task_name",
        "action_key",
        "action_dim",
        "n_obs_steps",
        "n_action_steps",
        "horizon",
        "required_action_steps",
        "control_dt_s",
        "sensor_modalities",
        "observation_fields",
        "requires_hand",
        "point_cloud_num_points",
        "point_cloud_feature_dim",
    }
)
_PRODUCER_KEYS = frozenset({"repository", "commit", "metadata_provenance"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
)
_READ_NOFOLLOW_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


@dataclass(frozen=True)
class FileLstatIdentity:
    """Stable file identity captured before the artifact is handed onward."""

    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    nlink: int

    @classmethod
    def from_stat(cls, info: os.stat_result) -> "FileLstatIdentity":
        return cls(
            device=info.st_dev,
            inode=info.st_ino,
            size_bytes=info.st_size,
            mtime_ns=info.st_mtime_ns,
            ctime_ns=info.st_ctime_ns,
            nlink=info.st_nlink,
        )

    def matches_stat(self, info: os.stat_result) -> bool:
        return self == self.from_stat(info)


@dataclass(frozen=True)
class DirectoryIdentity:
    """Stable identity for a held experiment-directory descriptor."""

    device: int
    inode: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, info: os.stat_result, label: str) -> "DirectoryIdentity":
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"{label} must be a directory")
        return cls(
            device=info.st_dev,
            inode=info.st_ino,
            ctime_ns=info.st_ctime_ns,
        )

    def matches_stat(self, info: os.stat_result) -> bool:
        return (
            info.st_dev,
            info.st_ino,
            info.st_ctime_ns,
        ) == (self.device, self.inode, self.ctime_ns)


@dataclass(frozen=True)
class PolicyArtifactContract:
    """Validated allocation values projected by a deployment sidecar."""

    task_name: str
    action_key: str
    action_dim: int
    n_obs_steps: int
    n_action_steps: int
    horizon: int
    required_action_steps: int
    control_dt_s: float
    sensor_modalities: tuple[str, ...]
    observation_fields: tuple[str, ...]
    requires_hand: bool
    point_cloud_num_points: int
    point_cloud_feature_dim: int


@dataclass(frozen=True)
class PolicyArtifactProducer:
    """Producer provenance retained from the exact sidecar index."""

    repository: str
    commit: str
    metadata_provenance: str


@dataclass(frozen=True)
class ResolvedPolicyArtifact:
    """One resolved checkpoint plus its validated, small allocation index."""

    experiment_dir: Path
    selector_path: Path
    checkpoint_path: Path
    sidecar_entry_path: Path
    sidecar_path: Path
    selector_name: str
    checkpoint_size_bytes: int
    checkpoint_sha256_from_index: str
    checkpoint_sha256_verified: bool
    embedded_contract_sha256: str
    index_sha256: str
    allocation_contract: PolicyArtifactContract
    producer: PolicyArtifactProducer
    selector_lstat_identity: FileLstatIdentity
    checkpoint_lstat_identity: FileLstatIdentity
    sidecar_entry_lstat_identity: FileLstatIdentity
    sidecar_lstat_identity: FileLstatIdentity
    experiment_directory_identity: DirectoryIdentity
    checkpoints_directory_identity: DirectoryIdentity


def resolve_policy_artifact(experiment_dir: str | Path) -> ResolvedPolicyArtifact:
    """Resolve and validate one experiment deployment artifact fail-closed.

    ``deployment_latest.pt`` has priority whenever its directory entry exists,
    including when it is dangling or malformed.  The resolver holds no-follow
    experiment/checkpoints directory descriptors for every artifact operation.
    It returns the fixed selected checkpoint basename and lstat identity after
    a final recheck; later preflight must use that path/identity directly and
    must not resolve the selector again around its full hash/load.  This
    resolver intentionally leaves the large checkpoint SHA-256 unverified.
    """
    experiment_fd: int | None = None
    checkpoints_fd: int | None = None
    try:
        experiment_path, experiment_fd, experiment_identity = (
            _open_experiment_directory(Path(experiment_dir))
        )
        _validate_config_at(experiment_fd)
        checkpoints_path, checkpoints_fd, checkpoints_identity = (
            _open_checkpoints_directory(experiment_fd, experiment_path)
        )

        selector_name, selector_lstat = _select_checkpoint_entry(checkpoints_fd)
        checkpoint_name, _, checkpoint_lstat = _resolve_one_hop_entry(
            checkpoints_fd, selector_name, "checkpoint selector"
        )
        checkpoint_identity = FileLstatIdentity.from_stat(checkpoint_lstat)
        _validate_checkpoint_at(checkpoints_fd, checkpoint_name, checkpoint_identity)

        sidecar_entry_name = f"{checkpoint_name}.deployment.json"
        sidecar_name, sidecar_entry_lstat, sidecar_lstat = _resolve_one_hop_entry(
            checkpoints_fd, sidecar_entry_name, "checkpoint sidecar"
        )
        sidecar_identity = FileLstatIdentity.from_stat(sidecar_lstat)
        sidecar_bytes = _read_bounded_regular_file_at(
            checkpoints_fd,
            sidecar_name,
            sidecar_identity,
            label="checkpoint sidecar",
            max_size_bytes=MAX_POLICY_ARTIFACT_INDEX_BYTES,
        )
        sidecar = _parse_canonical_sidecar(sidecar_bytes)
        checkpoint_sha256, embedded_contract_sha256, allocation, producer = (
            _validate_sidecar(
                sidecar,
                checkpoint_name=checkpoint_name,
                checkpoint_lstat=checkpoint_identity,
            )
        )

        selector_identity = FileLstatIdentity.from_stat(selector_lstat)
        sidecar_entry_identity = FileLstatIdentity.from_stat(sidecar_entry_lstat)
        _require_unchanged_lstat_at(
            checkpoints_fd, selector_name, selector_identity, "checkpoint selector"
        )
        _require_unchanged_lstat_at(
            checkpoints_fd, checkpoint_name, checkpoint_identity, "resolved checkpoint"
        )
        _require_unchanged_lstat_at(
            checkpoints_fd,
            sidecar_entry_name,
            sidecar_entry_identity,
            "checkpoint sidecar entry",
        )
        _require_unchanged_lstat_at(
            checkpoints_fd, sidecar_name, sidecar_identity, "checkpoint sidecar"
        )
        _require_display_file_matches(
            checkpoints_path / selector_name, selector_identity, "checkpoint selector"
        )
        _require_display_file_matches(
            checkpoints_path / checkpoint_name,
            checkpoint_identity,
            "resolved checkpoint",
        )
        _require_display_file_matches(
            checkpoints_path / sidecar_entry_name,
            sidecar_entry_identity,
            "checkpoint sidecar entry",
        )
        _require_display_file_matches(
            checkpoints_path / sidecar_name, sidecar_identity, "checkpoint sidecar"
        )
        _require_display_directory_matches(
            checkpoints_path, checkpoints_identity, "checkpoints directory"
        )
        _require_display_directory_matches(
            experiment_path, experiment_identity, "experiment root"
        )

        return ResolvedPolicyArtifact(
            experiment_dir=experiment_path,
            selector_path=checkpoints_path / selector_name,
            checkpoint_path=checkpoints_path / checkpoint_name,
            sidecar_entry_path=checkpoints_path / sidecar_entry_name,
            sidecar_path=checkpoints_path / sidecar_name,
            selector_name=selector_name,
            checkpoint_size_bytes=checkpoint_identity.size_bytes,
            checkpoint_sha256_from_index=checkpoint_sha256,
            checkpoint_sha256_verified=False,
            embedded_contract_sha256=embedded_contract_sha256,
            index_sha256=hashlib.sha256(sidecar_bytes).hexdigest(),
            allocation_contract=allocation,
            producer=producer,
            selector_lstat_identity=selector_identity,
            checkpoint_lstat_identity=checkpoint_identity,
            sidecar_entry_lstat_identity=sidecar_entry_identity,
            sidecar_lstat_identity=sidecar_identity,
            experiment_directory_identity=experiment_identity,
            checkpoints_directory_identity=checkpoints_identity,
        )
    except (OSError, RuntimeError) as exc:
        raise ValueError(
            "artifact filesystem changed or cannot be inspected safely"
        ) from exc
    finally:
        _close_fd(checkpoints_fd)
        _close_fd(experiment_fd)


def _open_experiment_directory(
    experiment_dir: Path,
) -> tuple[Path, int, DirectoryIdentity]:
    try:
        fd = os.open(experiment_dir, _DIRECTORY_OPEN_FLAGS)
    except OSError as exc:
        raise ValueError(
            f"experiment root is not a safe directory: {experiment_dir}"
        ) from exc
    try:
        identity = DirectoryIdentity.from_stat(os.fstat(fd), "experiment root")
        display_path = _display_directory_path(
            experiment_dir, identity, "experiment root"
        )
    except BaseException:
        _close_fd(fd)
        raise
    return display_path, fd, identity


def _open_checkpoints_directory(
    experiment_fd: int, experiment_path: Path
) -> tuple[Path, int, DirectoryIdentity]:
    try:
        fd = os.open("checkpoints", _DIRECTORY_OPEN_FLAGS, dir_fd=experiment_fd)
    except OSError as exc:
        raise ValueError("checkpoints directory is not a safe directory") from exc
    try:
        identity = DirectoryIdentity.from_stat(os.fstat(fd), "checkpoints directory")
        display_path = _display_directory_path(
            experiment_path / "checkpoints", identity, "checkpoints directory"
        )
    except BaseException:
        _close_fd(fd)
        raise
    return display_path, fd, identity


def _display_directory_path(
    source_path: Path, identity: DirectoryIdentity, label: str
) -> Path:
    try:
        display_path = source_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{label} cannot be resolved for display") from exc
    _require_display_directory_matches(display_path, identity, label)
    return display_path


def _validate_config_at(experiment_fd: int) -> None:
    config_lstat = _lstat_at(experiment_fd, "config.yaml", "experiment config.yaml")
    _require_regular_stat(config_lstat, "experiment config.yaml")
    config_identity = FileLstatIdentity.from_stat(config_lstat)
    config_fd = _open_regular_read_at(
        experiment_fd,
        "config.yaml",
        config_identity,
        "experiment config.yaml",
    )
    _close_fd(config_fd)
    _require_unchanged_lstat_at(
        experiment_fd, "config.yaml", config_identity, "experiment config.yaml"
    )


def _select_checkpoint_entry(directory_fd: int) -> tuple[str, os.stat_result]:
    for name in ("deployment_latest.pt", "latest.pt"):
        entry_lstat = _try_lstat_at(directory_fd, name, "checkpoint selector")
        if entry_lstat is not None:
            return name, entry_lstat
    raise ValueError(
        "no deployment selector found: expected deployment_latest.pt or latest.pt"
    )


def _resolve_one_hop_entry(
    directory_fd: int, entry_name: str, label: str
) -> tuple[str, os.stat_result, os.stat_result]:
    entry_lstat = _lstat_at(directory_fd, entry_name, label)
    if stat.S_ISLNK(entry_lstat.st_mode):
        target_name = _read_one_hop_basename(directory_fd, entry_name, label)
        try:
            target_lstat = _lstat_at(directory_fd, target_name, label)
        except ValueError as exc:
            raise ValueError(f"{label} is dangling or invalid") from exc
        _require_regular_stat(target_lstat, label)
        return target_name, entry_lstat, target_lstat
    _require_regular_stat(entry_lstat, label)
    return entry_name, entry_lstat, entry_lstat


def _read_one_hop_basename(directory_fd: int, entry_name: str, label: str) -> str:
    try:
        target = os.readlink(entry_name, dir_fd=directory_fd)
    except OSError as exc:
        raise ValueError(f"{label} symlink cannot be read") from exc
    if os.path.isabs(target):
        raise ValueError(f"{label} symlink target must be a relative basename")
    target_path = Path(target)
    if target_path.name != target or target in {"", ".", ".."}:
        raise ValueError(f"{label} symlink target must be a relative basename")
    return target


def _validate_checkpoint_at(
    directory_fd: int, checkpoint_name: str, identity: FileLstatIdentity
) -> None:
    if Path(checkpoint_name).suffix != ".pt":
        raise ValueError(
            f"resolved checkpoint must have a .pt suffix: {checkpoint_name!r}"
        )
    if identity.size_bytes <= 0:
        raise ValueError("resolved checkpoint must be non-empty")
    if identity.nlink != 1:
        raise ValueError("resolved checkpoint must have exactly one hard link")
    checkpoint_fd = _open_regular_read_at(
        directory_fd, checkpoint_name, identity, "resolved checkpoint"
    )
    _close_fd(checkpoint_fd)


def _read_bounded_regular_file_at(
    directory_fd: int,
    filename: str,
    identity: FileLstatIdentity,
    *,
    label: str,
    max_size_bytes: int,
) -> bytes:
    if identity.size_bytes > max_size_bytes:
        raise ValueError(f"{label} exceeds the {max_size_bytes}-byte index size bound")
    if identity.nlink != 1:
        raise ValueError(f"{label} must have exactly one hard link")
    fd = _open_regular_read_at(directory_fd, filename, identity, label)
    try:
        with os.fdopen(fd, "rb", closefd=False) as stream:
            payload = stream.read(max_size_bytes + 1)
        after_read = os.fstat(fd)
        if not identity.matches_stat(after_read):
            raise ValueError(f"{label} changed while it was read")
    except OSError as exc:
        raise ValueError(f"{label} cannot be read safely") from exc
    finally:
        _close_fd(fd)
    if len(payload) != identity.size_bytes or len(payload) > max_size_bytes:
        raise ValueError(f"{label} changed while it was read")
    return payload


def _open_regular_read_at(
    directory_fd: int,
    filename: str,
    identity: FileLstatIdentity,
    label: str,
) -> int:
    try:
        fd = os.open(filename, _READ_NOFOLLOW_FLAGS, dir_fd=directory_fd)
    except OSError as exc:
        raise ValueError(
            f"{label} cannot be opened without following symlinks"
        ) from exc
    try:
        opened = os.fstat(fd)
        _require_regular_stat(opened, label)
        if not identity.matches_stat(opened):
            raise ValueError(f"{label} changed before it could be opened")
    except BaseException:
        _close_fd(fd)
        raise
    return fd


def _parse_canonical_sidecar(payload: bytes) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("checkpoint sidecar must be valid UTF-8") from exc
    try:
        sidecar = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_non_finite_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("checkpoint sidecar must be a finite JSON object") from exc
    if type(sidecar) is not dict:
        raise ValueError("checkpoint sidecar must contain a JSON object")
    try:
        canonical = json.dumps(
            sidecar,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint sidecar must contain finite JSON values") from exc
    if payload != canonical:
        raise ValueError("checkpoint sidecar is not canonical JSON")
    return sidecar


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_non_finite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _validate_sidecar(
    sidecar: dict[str, Any],
    *,
    checkpoint_name: str,
    checkpoint_lstat: FileLstatIdentity,
) -> tuple[str, str, PolicyArtifactContract, PolicyArtifactProducer]:
    _require_exact_keys(sidecar, _SIDECAR_KEYS, "checkpoint sidecar")
    if _require_positive_int(sidecar["schema_version"], "schema_version") != 1:
        raise ValueError("checkpoint sidecar schema_version must be 1")

    checkpoint = _require_object(sidecar["checkpoint"], "checkpoint")
    _require_exact_keys(checkpoint, _CHECKPOINT_KEYS, "checkpoint")
    filename = _require_non_empty_string(checkpoint["filename"], "checkpoint.filename")
    if filename != checkpoint_name:
        raise ValueError(
            "checkpoint.filename does not match the selected resolved checkpoint"
        )
    if _require_positive_int(checkpoint["size_bytes"], "checkpoint.size_bytes") != (
        checkpoint_lstat.size_bytes
    ):
        raise ValueError("checkpoint.size_bytes does not match the selected checkpoint")
    checkpoint_sha256 = _require_sha256(checkpoint["sha256"], "checkpoint.sha256")
    embedded_contract_sha256 = _require_sha256(
        sidecar["embedded_contract_sha256"], "embedded_contract_sha256"
    )
    allocation = _validate_allocation(sidecar["allocation"])
    producer = _validate_producer(sidecar["producer"])
    return checkpoint_sha256, embedded_contract_sha256, allocation, producer


def _validate_allocation(value: Any) -> PolicyArtifactContract:
    allocation = _require_object(value, "allocation")
    _require_exact_keys(allocation, _ALLOCATION_KEYS, "allocation")
    task_name = _require_non_empty_string(
        allocation["task_name"], "allocation.task_name"
    )
    if task_name != task_name.strip():
        raise ValueError("allocation.task_name must not have surrounding whitespace")
    action_key = _require_non_empty_string(
        allocation["action_key"], "allocation.action_key"
    )
    action_dimensions = {
        "action": ARM_DOF + HAND_DOF,
        "action_ee": EE_POS_DIM + EE_ROT6D_DIM + HAND_DOF,
    }
    if action_key not in action_dimensions:
        raise ValueError(f"unsupported allocation.action_key={action_key!r}")
    action_dim = _require_positive_int(
        allocation["action_dim"], "allocation.action_dim"
    )
    if action_dim != action_dimensions[action_key]:
        raise ValueError(
            f"allocation.action_dim={action_dim} is inconsistent with "
            f"allocation.action_key={action_key!r}"
        )

    n_obs_steps = _require_positive_int(
        allocation["n_obs_steps"], "allocation.n_obs_steps"
    )
    n_action_steps = _require_positive_int(
        allocation["n_action_steps"], "allocation.n_action_steps"
    )
    horizon = _require_positive_int(allocation["horizon"], "allocation.horizon")
    required_action_steps = _require_positive_int(
        allocation["required_action_steps"], "allocation.required_action_steps"
    )
    expected_required_steps = horizon - (n_obs_steps - 1)
    if required_action_steps != expected_required_steps or required_action_steps <= 0:
        raise ValueError(
            "allocation.required_action_steps must equal horizon - (n_obs_steps - 1)"
        )
    if n_action_steps > required_action_steps:
        raise ValueError(
            "allocation.n_action_steps exceeds the actionable prediction window"
        )
    if required_action_steps > MAX_POLICY_CHUNK_STEPS:
        raise ValueError(
            f"allocation.required_action_steps={required_action_steps} exceeds "
            f"the IPC capacity {MAX_POLICY_CHUNK_STEPS}"
        )

    control_dt_s = _require_positive_finite_number(
        allocation["control_dt_s"], "allocation.control_dt_s"
    )
    sensor_modalities = _require_exact_string_list(
        allocation["sensor_modalities"],
        ("joint_state", "point_cloud"),
        "allocation.sensor_modalities",
    )
    observation_fields = _require_exact_string_list(
        allocation["observation_fields"],
        ("arm_qpos", "hand_qpos", "point_cloud"),
        "allocation.observation_fields",
    )
    requires_hand = allocation["requires_hand"]
    if not isinstance(requires_hand, bool) or not requires_hand:
        raise ValueError("allocation.requires_hand must be true for Real hand actions")
    point_cloud_num_points = _require_positive_int(
        allocation["point_cloud_num_points"], "allocation.point_cloud_num_points"
    )
    if point_cloud_num_points not in SUPPORTED_POINT_CLOUD_COUNTS:
        raise ValueError(
            "allocation.point_cloud_num_points must be one of "
            f"{sorted(SUPPORTED_POINT_CLOUD_COUNTS)}"
        )
    point_cloud_feature_dim = _require_positive_int(
        allocation["point_cloud_feature_dim"], "allocation.point_cloud_feature_dim"
    )
    if point_cloud_feature_dim != POINT_CLOUD_FEATURE_DIM:
        raise ValueError(
            f"allocation.point_cloud_feature_dim must be {POINT_CLOUD_FEATURE_DIM}"
        )
    return PolicyArtifactContract(
        task_name=task_name,
        action_key=action_key,
        action_dim=action_dim,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        horizon=horizon,
        required_action_steps=required_action_steps,
        control_dt_s=control_dt_s,
        sensor_modalities=sensor_modalities,
        observation_fields=observation_fields,
        requires_hand=requires_hand,
        point_cloud_num_points=point_cloud_num_points,
        point_cloud_feature_dim=point_cloud_feature_dim,
    )


def _validate_producer(value: Any) -> PolicyArtifactProducer:
    producer = _require_object(value, "producer")
    _require_exact_keys(producer, _PRODUCER_KEYS, "producer")
    repository = _require_non_empty_string(
        producer["repository"], "producer.repository"
    )
    if repository != "haoyangzhanglab/dexmani_policy":
        raise ValueError(
            "producer.repository is not the frozen DexMani Policy producer"
        )
    commit = _require_non_empty_string(producer["commit"], "producer.commit")
    if _GIT_COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("producer.commit must be a lowercase 40-hex git commit")
    metadata_provenance = _require_non_empty_string(
        producer["metadata_provenance"], "producer.metadata_provenance"
    )
    if metadata_provenance not in {"native", "retrofitted"}:
        raise ValueError(
            "producer.metadata_provenance must be 'native' or 'retrofitted'"
        )
    return PolicyArtifactProducer(
        repository=repository,
        commit=commit,
        metadata_provenance=metadata_provenance,
    )


def _require_exact_keys(
    value: dict[str, Any], expected: frozenset[str], label: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{label} keys mismatch: missing={missing}, extra={extra}")


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_positive_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be a finite positive number")
    return result


def _require_exact_string_list(
    value: Any, expected: tuple[str, ...], label: str
) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a string list")
    actual = tuple(value)
    if actual != expected:
        raise ValueError(f"{label} must equal {list(expected)!r}")
    return actual


def _require_sha256(value: Any, label: str) -> str:
    result = _require_non_empty_string(value, label)
    if _SHA256_RE.fullmatch(result) is None:
        raise ValueError(f"{label} must be a lowercase 64-hex SHA-256")
    return result


def _require_regular_stat(info: os.stat_result, label: str) -> None:
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} must be a regular file")


def _lstat_at(directory_fd: int, name: str, label: str) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"{label} cannot be inspected safely") from exc


def _try_lstat_at(directory_fd: int, name: str, label: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(f"{label} cannot be inspected safely") from exc


def _require_unchanged_lstat_at(
    directory_fd: int, name: str, identity: FileLstatIdentity, label: str
) -> None:
    current = FileLstatIdentity.from_stat(_lstat_at(directory_fd, name, label))
    if current != identity:
        raise ValueError(f"{label} changed while the artifact was resolved")


def _require_display_file_matches(
    display_path: Path, identity: FileLstatIdentity, label: str
) -> None:
    try:
        current = FileLstatIdentity.from_stat(display_path.lstat())
    except OSError as exc:
        raise ValueError(f"{label} display path changed while resolving") from exc
    if current != identity:
        raise ValueError(f"{label} display path changed while resolving")


def _require_display_directory_matches(
    display_path: Path, identity: DirectoryIdentity, label: str
) -> None:
    try:
        current = display_path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} display path changed while resolving") from exc
    if not stat.S_ISDIR(current.st_mode) or not identity.matches_stat(current):
        raise ValueError(f"{label} display path changed while resolving")


def _close_fd(fd: int | None) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass
