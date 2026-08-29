"""Isolated, hash-bound learned-policy preflight.

The parent sends only a pickle-safe Real receipt to a fresh ``spawn`` child.
The child reopens the fixed artifact entries through no-follow descriptors,
checks the indexed digest, then performs the one safe stream deserialize and
one fake-observation prediction. It never resolves a selector, creates
runtime channels, or imports a hardware owner.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from dexmani_real.deployment.artifact import DirectoryIdentity, FileLstatIdentity
from dexmani_real.deployment.config import (
    FIXED_POLICY_RUNTIME_TARGET,
    PolicyRuntimeConfig,
)
from dexmani_real.deployment.contracts import PolicyPrediction, PolicyRuntime
from dexmani_real.deployment.observation import (
    FrameWindow,
    ObservationBatch,
    PointCloudFrame,
)

_ARM_DOF = 7
_HAND_DOF = 12

_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
)
_READ_NOFOLLOW_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
_HASH_CHUNK_BYTES = 1024 * 1024
_MAX_CHILD_MESSAGE_BYTES = 16 * 1024
_MAX_CHILD_ERROR_CHARS = 2 * 1024
_CHILD_JOIN_TIMEOUT_S = 2.0
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")


@dataclass(frozen=True)
class PreflightResult:
    """Small JSON-safe receipt; it deliberately carries no model or tensor."""

    checkpoint_sha256: str
    checkpoint_sha256_verified: bool
    action_steps: int
    action_dim: int
    package_origin: str
    package_commit: str
    package_dirty: str
    package_source_tree_sha256: str
    package_version: str

    def __post_init__(self) -> None:
        if not _SHA256_RE.fullmatch(self.checkpoint_sha256):
            raise ValueError("preflight receipt checkpoint SHA-256 is invalid")
        if self.checkpoint_sha256_verified is not True:
            raise ValueError("preflight receipt must verify the checkpoint digest")
        for name in ("action_steps", "action_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"preflight receipt {name} is invalid")
        if (
            not isinstance(self.package_origin, str)
            or not self.package_origin
            or len(self.package_origin) > 4096
        ):
            raise ValueError("preflight receipt package origin is invalid")
        if not _COMMIT_RE.fullmatch(self.package_commit):
            raise ValueError("preflight receipt package commit is invalid")
        if self.package_dirty not in {"true", "false", "unknown"}:
            raise ValueError("preflight receipt package dirty marker is invalid")
        if not _SHA256_RE.fullmatch(self.package_source_tree_sha256):
            raise ValueError("preflight receipt package source tree SHA-256 is invalid")
        if not isinstance(self.package_version, str) or len(self.package_version) > 256:
            raise ValueError("preflight receipt package version is invalid")

    def to_wire(self) -> dict[str, Any]:
        return {
            "action_dim": self.action_dim,
            "action_steps": self.action_steps,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_sha256_verified": self.checkpoint_sha256_verified,
            "package_commit": self.package_commit,
            "package_dirty": self.package_dirty,
            "package_origin": self.package_origin,
            "package_source_tree_sha256": self.package_source_tree_sha256,
            "package_version": self.package_version,
        }

    @classmethod
    def from_wire(cls, value: Any) -> "PreflightResult":
        if not isinstance(value, dict) or set(value) != {
            "action_dim",
            "action_steps",
            "checkpoint_sha256",
            "checkpoint_sha256_verified",
            "package_commit",
            "package_dirty",
            "package_origin",
            "package_source_tree_sha256",
            "package_version",
        }:
            raise ValueError("policy preflight child returned an invalid receipt")
        return cls(**value)


def run_isolated_preflight(
    runtime_config: PolicyRuntimeConfig, *, timeout_s: float = 120.0
) -> PreflightResult:
    """Run one bounded child preflight and propagate every failure fail-closed."""
    if not isinstance(runtime_config, PolicyRuntimeConfig):
        raise TypeError("preflight requires PolicyRuntimeConfig")
    if runtime_config.artifact is None:
        raise ValueError("preflight requires an artifact-bound PolicyRuntimeConfig")
    if timeout_s <= 0.0:
        raise ValueError("preflight timeout_s must be positive")
    context = mp.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_preflight_child,
        args=(child_connection, runtime_config),
        name="dexmani_policy_preflight",
    )
    process.daemon = False
    started = False
    try:
        process.start()
        started = True
    except BaseException:
        child_connection.close()
        parent_connection.close()
        _close_process(process)
        raise
    else:
        child_connection.close()
    try:
        if not parent_connection.poll(timeout_s):
            raise TimeoutError("policy preflight child timed out")
        try:
            payload = parent_connection.recv_bytes(_MAX_CHILD_MESSAGE_BYTES)
        except EOFError as exc:
            raise RuntimeError(
                "policy preflight child exited without a receipt"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                "policy preflight child receipt exceeds its bound"
            ) from exc
        message = _decode_child_message(payload)
        process.join(timeout=_CHILD_JOIN_TIMEOUT_S)
        if process.is_alive():
            raise RuntimeError("policy preflight child did not exit after its receipt")
        if process.exitcode != 0:
            raise RuntimeError(
                f"policy preflight child exited with code {process.exitcode}"
            )
        if message["ok"] is not True:
            raise RuntimeError(f"policy preflight failed: {message['error']}")
        result = PreflightResult.from_wire(message["receipt"])
        allocation = runtime_config.artifact.allocation_contract
        if (
            result.checkpoint_sha256
            != runtime_config.artifact.checkpoint_sha256_from_index
            or result.action_steps != allocation.required_action_steps
            or result.action_dim != allocation.action_dim
        ):
            raise RuntimeError("policy preflight child receipt mismatches its request")
        return result
    finally:
        parent_connection.close()
        if started:
            _terminate_join_kill_close(process)
        else:
            _close_process(process)


def _decode_child_message(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("policy preflight child returned invalid JSON") from exc
    if not isinstance(value, dict) or set(value) != {"ok", "receipt", "error"}:
        raise RuntimeError("policy preflight child returned an invalid message schema")
    if (
        value["ok"] is True
        and value["error"] is None
        and isinstance(value["receipt"], dict)
    ):
        return value
    if (
        value["ok"] is False
        and value["receipt"] is None
        and isinstance(value["error"], str)
        and len(value["error"]) <= _MAX_CHILD_ERROR_CHARS
    ):
        return value
    raise RuntimeError("policy preflight child returned an invalid message")


def _encode_child_message(value: Mapping[str, Any]) -> bytes:
    payload = json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    if len(payload) > _MAX_CHILD_MESSAGE_BYTES:
        raise ValueError("policy preflight child message exceeds its bound")
    return payload


def _preflight_child(connection: Any, runtime_config: PolicyRuntimeConfig) -> None:
    try:
        result = _run_preflight_child(runtime_config)
        message: dict[str, Any] = {
            "ok": True,
            "receipt": result.to_wire(),
            "error": None,
        }
    except BaseException as exc:
        detail = f"{type(exc).__name__}: {exc}"[:_MAX_CHILD_ERROR_CHARS]
        message = {"ok": False, "receipt": None, "error": detail}
    try:
        connection.send_bytes(_encode_child_message(message))
    except (BrokenPipeError, EOFError, OSError, ValueError):
        pass
    finally:
        connection.close()


def _terminate_join_kill_close(process: Any) -> None:
    """Leave no child alive on timeout, EOF, receipt-hang, or parent failure."""
    try:
        if process.is_alive():
            process.terminate()
            process.join(timeout=_CHILD_JOIN_TIMEOUT_S)
        if process.is_alive():
            kill = getattr(process, "kill", None)
            if callable(kill):
                kill()
                process.join(timeout=_CHILD_JOIN_TIMEOUT_S)
        if process.is_alive():
            raise RuntimeError("policy preflight child could not be terminated")
    finally:
        if not process.is_alive():
            _close_process(process)


def _close_process(process: Any) -> None:
    close = getattr(process, "close", None)
    if callable(close):
        close()


def load_verified_policy_runtime(runtime_config: PolicyRuntimeConfig) -> PolicyRuntime:
    """Return a DexMani runtime from one verified checkpoint stream.

    This is the inference-child counterpart to ``--preflight-only``. It uses
    only the fixed artifact entries, verifies their no-follow identities and
    digest before policy import/deserialization, and never calls the disabled
    path-based ``PolicyRuntime.load`` API. The caller owns the returned runtime.
    """
    runtime, _actual_sha256, _provenance = _load_verified_policy_runtime(runtime_config)
    return runtime


def _load_verified_policy_runtime(
    runtime_config: PolicyRuntimeConfig,
) -> tuple[PolicyRuntime, str, Mapping[str, str]]:
    """Build one runtime after the fd-bound artifact verification sequence."""
    if not isinstance(runtime_config, PolicyRuntimeConfig):
        raise TypeError("verified policy load requires PolicyRuntimeConfig")
    artifact = runtime_config.artifact
    if artifact is None:
        raise ValueError("verified policy load requires a resolved artifact")
    if runtime_config.runtime_target != FIXED_POLICY_RUNTIME_TARGET:
        raise ValueError("artifact-bound policy runtime target is not fixed")
    experiment_fd: int | None = None
    checkpoints_fd: int | None = None
    checkpoint_fd: int | None = None
    runtime: PolicyRuntime | None = None
    try:
        experiment_fd = _open_directory(
            artifact.experiment_dir,
            artifact.experiment_directory_identity,
            "experiment root",
        )
        checkpoints_fd = _open_directory_at(
            experiment_fd,
            "checkpoints",
            artifact.checkpoints_directory_identity,
            "checkpoints directory",
        )
        selector_name = _strict_basename(artifact.selector_path, ".pt")
        checkpoint_name = _strict_basename(artifact.checkpoint_path, ".pt")
        sidecar_entry_name = _strict_basename(
            artifact.sidecar_entry_path, ".deployment.json"
        )
        sidecar_name = _strict_basename(artifact.sidecar_path, ".json")
        _require_entry_lstat_identity(
            checkpoints_fd,
            selector_name,
            artifact.selector_lstat_identity,
            "checkpoint selector",
        )
        _require_lstat_identity(
            checkpoints_fd,
            checkpoint_name,
            artifact.checkpoint_lstat_identity,
            "checkpoint",
        )
        _require_entry_lstat_identity(
            checkpoints_fd,
            sidecar_entry_name,
            artifact.sidecar_entry_lstat_identity,
            "checkpoint sidecar entry",
        )
        _require_lstat_identity(
            checkpoints_fd,
            sidecar_name,
            artifact.sidecar_lstat_identity,
            "checkpoint sidecar",
        )
        checkpoint_fd = os.open(
            checkpoint_name, _READ_NOFOLLOW_FLAGS, dir_fd=checkpoints_fd
        )
        _require_open_identity(
            checkpoint_fd, artifact.checkpoint_lstat_identity, "checkpoint"
        )
        with os.fdopen(checkpoint_fd, "rb", closefd=False) as stream:
            actual_sha256 = _sha256_stream(stream)
            _recheck_artifact_identities(
                artifact,
                experiment_fd,
                checkpoints_fd,
                selector_name,
                checkpoint_name,
                sidecar_entry_name,
                sidecar_name,
                checkpoint_fd,
            )
            if actual_sha256 != artifact.checkpoint_sha256_from_index:
                raise ValueError("checkpoint SHA-256 mismatches sidecar index")

            # No Policy code is imported before this standard-library gate.
            from dexmani_real.integrations.dexmani_policy import (
                precheck_policy_package_provenance,
            )

            provenance = precheck_policy_package_provenance(runtime_config)
            stream.seek(0)
            # The deployment loader is the only torch deserialize in this child.
            from dexmani_policy.common.checkpoint_io import (
                load_deployment_checkpoint_stream,
            )

            checkpoint = load_deployment_checkpoint_stream(stream)
            _recheck_artifact_identities(
                artifact,
                experiment_fd,
                checkpoints_fd,
                selector_name,
                checkpoint_name,
                sidecar_entry_name,
                sidecar_name,
                checkpoint_fd,
            )
        from dexmani_real.integrations.dexmani_policy import DexManiPolicyRuntime

        runtime = DexManiPolicyRuntime(runtime_config)
        try:
            runtime.load_loaded_checkpoint(checkpoint, package_provenance=provenance)
            _recheck_artifact_identities(
                artifact,
                experiment_fd,
                checkpoints_fd,
                selector_name,
                checkpoint_name,
                sidecar_entry_name,
                sidecar_name,
                checkpoint_fd,
            )
        except BaseException:
            try:
                runtime.close()
            except Exception:
                pass
            raise
        return runtime, actual_sha256, provenance
    finally:
        _close_fd(checkpoint_fd)
        _close_fd(checkpoints_fd)
        _close_fd(experiment_fd)


def _run_preflight_child(runtime_config: PolicyRuntimeConfig) -> PreflightResult:
    runtime, actual_sha256, provenance = _load_verified_policy_runtime(runtime_config)
    try:
        import torch

        np.random.seed(0)
        torch.manual_seed(0)
        observation = _fake_observation(runtime_config)
        with torch.inference_mode():
            prediction = runtime.predict(observation)
    finally:
        runtime.close()
    _validate_prediction(prediction, runtime_config)
    artifact = runtime_config.artifact
    assert artifact is not None
    return PreflightResult(
        checkpoint_sha256=actual_sha256,
        checkpoint_sha256_verified=True,
        action_steps=artifact.allocation_contract.required_action_steps,
        action_dim=artifact.allocation_contract.action_dim,
        package_origin=provenance["origin"],
        package_commit=provenance["commit"],
        package_dirty=provenance["dirty"],
        package_source_tree_sha256=provenance["source_tree_sha256"],
        package_version=provenance["version"],
    )


def _recheck_artifact_identities(
    artifact: Any,
    experiment_fd: int,
    checkpoints_fd: int,
    selector_name: str,
    checkpoint_name: str,
    sidecar_entry_name: str,
    sidecar_name: str,
    checkpoint_fd: int,
) -> None:
    _require_open_identity(
        checkpoint_fd, artifact.checkpoint_lstat_identity, "checkpoint"
    )
    _require_entry_lstat_identity(
        checkpoints_fd,
        selector_name,
        artifact.selector_lstat_identity,
        "checkpoint selector",
    )
    _require_lstat_identity(
        checkpoints_fd,
        checkpoint_name,
        artifact.checkpoint_lstat_identity,
        "checkpoint",
    )
    _require_entry_lstat_identity(
        checkpoints_fd,
        sidecar_entry_name,
        artifact.sidecar_entry_lstat_identity,
        "checkpoint sidecar entry",
    )
    _require_lstat_identity(
        checkpoints_fd,
        sidecar_name,
        artifact.sidecar_lstat_identity,
        "checkpoint sidecar",
    )
    _require_directory_identity(
        experiment_fd, artifact.experiment_directory_identity, "experiment root"
    )
    _require_directory_identity(
        checkpoints_fd, artifact.checkpoints_directory_identity, "checkpoints directory"
    )


def _strict_basename(path: Path, suffix: str) -> str:
    name = path.name
    if (
        path.parent.name != "checkpoints"
        or name != str(path.name)
        or not name.endswith(suffix)
    ):
        raise ValueError(f"artifact {suffix} path is not a fixed checkpoints basename")
    return name


def _open_directory(path: Path, identity: DirectoryIdentity, label: str) -> int:
    try:
        fd = os.open(path, _DIRECTORY_OPEN_FLAGS)
    except OSError as exc:
        raise ValueError(f"{label} cannot be opened safely") from exc
    try:
        _require_directory_identity(fd, identity, label)
    except BaseException:
        _close_fd(fd)
        raise
    return fd


def _open_directory_at(
    parent_fd: int, name: str, identity: DirectoryIdentity, label: str
) -> int:
    try:
        fd = os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError(f"{label} cannot be opened safely") from exc
    try:
        _require_directory_identity(fd, identity, label)
    except BaseException:
        _close_fd(fd)
        raise
    return fd


def _require_directory_identity(
    fd: int, identity: DirectoryIdentity, label: str
) -> None:
    info = os.fstat(fd)
    if not stat.S_ISDIR(info.st_mode) or not identity.matches_stat(info):
        raise ValueError(f"{label} identity changed")


def _require_entry_lstat_identity(
    directory_fd: int, name: str, identity: FileLstatIdentity, label: str
) -> None:
    try:
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"{label} cannot be inspected safely") from exc
    if not identity.matches_stat(info):
        raise ValueError(f"{label} identity changed")


def _require_lstat_identity(
    directory_fd: int, name: str, identity: FileLstatIdentity, label: str
) -> None:
    _require_entry_lstat_identity(directory_fd, name, identity, label)
    info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f"{label} identity changed")


def _require_open_identity(fd: int, identity: FileLstatIdentity, label: str) -> None:
    info = os.fstat(fd)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or not identity.matches_stat(info)
    ):
        raise ValueError(f"{label} identity changed")


def _sha256_stream(stream: Any) -> str:
    digest = hashlib.sha256()
    stream.seek(0)
    while True:
        block = stream.read(_HASH_CHUNK_BYTES)
        if not block:
            break
        digest.update(block)
    stream.seek(0)
    return digest.hexdigest()


def _fake_observation(runtime_config: PolicyRuntimeConfig) -> ObservationBatch:
    allocation = (
        runtime_config.artifact.allocation_contract if runtime_config.artifact else None
    )
    if allocation is None:
        raise ValueError("fake observation requires an artifact")
    count = allocation.n_obs_steps
    run_ns = 1_000_000_000
    source_ns = run_ns + np.arange(1, count + 1, dtype=np.uint64) * 1_000_000
    publish_ns = source_ns + 1
    sequence = np.arange(1, count + 1, dtype=np.uint64)
    valid = np.ones(count, dtype=np.uint8)
    arm = FrameWindow(
        values=np.zeros((count, _ARM_DOF), dtype=np.float32),
        source_sequence=sequence,
        source_monotonic_ns=source_ns,
        publish_monotonic_ns=publish_ns,
        valid_mask=valid,
    )
    hand = FrameWindow(
        values=np.zeros((count, _HAND_DOF), dtype=np.float32),
        source_sequence=sequence,
        source_monotonic_ns=source_ns,
        publish_monotonic_ns=publish_ns,
        valid_mask=valid,
    )
    points = np.zeros(
        (allocation.point_cloud_num_points, allocation.point_cloud_feature_dim),
        dtype=np.float32,
    )
    points[:, 0] = 0.4
    points[:, 3:] = 0.5
    history = tuple(
        PointCloudFrame(
            values=points,
            source_camera_sequence=int(index),
            source_monotonic_ns=int(source),
            publish_monotonic_ns=int(published),
            camera_generation=1,
        )
        for index, source, published in zip(
            sequence, source_ns, publish_ns, strict=True
        )
    )
    latest = history[-1]
    return ObservationBatch(
        observation_id=1,
        run_generation=1,
        run_started_monotonic_ns=run_ns,
        anchor_monotonic_ns=int(latest.publish_monotonic_ns + 1_000_000),
        latest_source_monotonic_ns=latest.source_monotonic_ns,
        logical_step_monotonic_ns=latest.source_monotonic_ns,
        arm_history=arm,
        hand_history=hand,
        pointcloud=latest,
        pointcloud_history=history,
    )


def _validate_prediction(prediction: Any, runtime_config: PolicyRuntimeConfig) -> None:
    artifact = runtime_config.artifact
    assert artifact is not None
    if not isinstance(prediction, PolicyPrediction):
        raise TypeError("fake preflight prediction must be a PolicyPrediction")
    action_key = artifact.allocation_contract.action_key
    if action_key == "action" and prediction.is_ee:
        raise ValueError(
            "fake preflight joint action artifact requires arm7 joint actions, not EE"
        )
    if action_key == "action_ee" and not prediction.is_ee:
        raise ValueError(
            "fake preflight EE action artifact requires ee_pos/ee_rot6d actions"
        )
    if action_key not in {"action", "action_ee"}:
        raise ValueError(
            f"fake preflight artifact has unsupported action_key={action_key!r}"
        )
    expected_steps = artifact.allocation_contract.required_action_steps
    if prediction.hand_qpos is None or prediction.hand_qpos.shape != (
        expected_steps,
        _HAND_DOF,
    ):
        raise ValueError("fake preflight prediction is missing hand12 actions")
    if prediction.arm_qpos is not None:
        if prediction.arm_qpos.shape != (expected_steps, _ARM_DOF):
            raise ValueError("fake preflight joint prediction has invalid arm7 actions")
        if prediction.ee_pos is not None or prediction.ee_rot6d is not None:
            raise ValueError(
                "fake preflight joint prediction mixes joint and EE actions"
            )
        arrays = [prediction.arm_qpos, prediction.hand_qpos]
    else:
        if (
            prediction.ee_pos is None
            or prediction.ee_rot6d is None
            or prediction.ee_pos.shape != (expected_steps, 3)
            or prediction.ee_rot6d.shape != (expected_steps, 6)
        ):
            raise ValueError("fake preflight EE prediction has invalid geometry fields")
        _validate_ee_rot6d(prediction.ee_rot6d)
        arrays = [prediction.ee_pos, prediction.ee_rot6d, prediction.hand_qpos]
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("fake preflight prediction contains NaN/Inf")


def _validate_ee_rot6d(rot6d: np.ndarray) -> None:
    """Reject degenerate 6D rotations without importing an IK/planning module."""
    first = np.asarray(rot6d[:, :3], dtype=np.float64)
    second = np.asarray(rot6d[:, 3:], dtype=np.float64)
    first_norm = np.linalg.norm(first, axis=1)
    if np.any(first_norm <= 1e-8):
        raise ValueError(
            "fake preflight EE prediction has degenerate rotation geometry"
        )
    second_orthogonal = (
        second
        - (
            np.sum(first * second, axis=1, keepdims=True)
            / np.square(first_norm)[:, None]
        )
        * first
    )
    if np.any(np.linalg.norm(second_orthogonal, axis=1) <= 1e-8):
        raise ValueError(
            "fake preflight EE prediction has degenerate rotation geometry"
        )


def _close_fd(fd: int | None) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass
