"""Isolated, hash-bound learned-policy preflight.

The parent sends only a pickle-safe Real receipt to a fresh ``spawn`` child.
The child reopens the fixed artifact entries through no-follow descriptors,
checks the indexed digest, then performs the one safe stream deserialize and
bounded synthetic-observation predictions. It never resolves a selector,
creates runtime channels, or imports a hardware owner.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from dexmani_real.deployment.artifact import DirectoryIdentity, FileLstatIdentity
from dexmani_real.deployment.config import PolicyRuntimeConfig
from dexmani_real.deployment.contracts import PolicyPrediction, PolicyRuntime
from dexmani_real.deployment.observation import (
    FrameWindow,
    ObservationBatch,
    PointCloudFrame,
    RgbFrame,
)
from dexmani_real.deployment.timing import build_target_grid, first_deliverable_index

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
_MIN_BENCHMARK_SAMPLES = 1
_MAX_BENCHMARK_SAMPLES = 1000
POLICY_WARMUP_SAMPLES = 5
POLICY_STABLE_WARMUP_SAMPLES = 3
MIN_STARTUP_DELIVERABLE_TARGETS = 2
_FORBIDDEN_CHECK_IMPORTS = (
    "dexmani_real.deployment.lifecycle",
    "dexmani_real.robot.arm_worker",
    "dexmani_real.robot.hand_worker",
    "dexmani_real.sensor.camera_worker",
    "dexmani_real.sensor.pointcloud_worker",
    "pyrealsense2",
)


@dataclass(frozen=True)
class PreflightResult:
    """Small JSON-safe receipt; ``action_steps`` is the executable chunk length."""

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


@dataclass(frozen=True)
class PolicyCheckResult:
    """JSON-safe hardware-free restore, qualification, and benchmark receipt."""

    checkpoint_sha256: str
    checkpoint_sha256_verified: bool
    action_steps: int
    action_dim: int
    package_origin: str
    package_commit: str
    package_dirty: str
    package_source_tree_sha256: str
    package_version: str
    device: str
    seed: int
    benchmark_samples: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_max_ms: float
    remaining_targets_min: int
    remaining_targets_p50: int
    remaining_targets_p95: int
    zero_deliverable_samples: int
    source_aware_schedulability: str
    gpu_peak_memory_bytes: int | None

    def __post_init__(self) -> None:
        PreflightResult(
            checkpoint_sha256=self.checkpoint_sha256,
            checkpoint_sha256_verified=self.checkpoint_sha256_verified,
            action_steps=self.action_steps,
            action_dim=self.action_dim,
            package_origin=self.package_origin,
            package_commit=self.package_commit,
            package_dirty=self.package_dirty,
            package_source_tree_sha256=self.package_source_tree_sha256,
            package_version=self.package_version,
        )
        _validate_benchmark_samples(self.benchmark_samples)
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("policy check device is invalid")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("policy check seed is invalid")
        for name in ("latency_p50_ms", "latency_p95_ms", "latency_max_ms"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"policy check {name} is invalid")
        for name in (
            "remaining_targets_min",
            "remaining_targets_p50",
            "remaining_targets_p95",
            "zero_deliverable_samples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"policy check {name} is invalid")
        if self.source_aware_schedulability != "NOT_MEASURED":
            raise ValueError(
                "synthetic policy check must not claim source-aware timing"
            )
        if self.gpu_peak_memory_bytes is not None and (
            isinstance(self.gpu_peak_memory_bytes, bool)
            or not isinstance(self.gpu_peak_memory_bytes, int)
            or self.gpu_peak_memory_bytes < 0
        ):
            raise ValueError("policy check GPU peak memory is invalid")

    def to_wire(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_wire(cls, value: Any) -> "PolicyCheckResult":
        expected = frozenset(cls.__dataclass_fields__)
        if not isinstance(value, dict) or frozenset(value) != expected:
            raise ValueError("policy check child returned an invalid receipt")
        return cls(**value)


def _validate_benchmark_samples(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not _MIN_BENCHMARK_SAMPLES <= value <= _MAX_BENCHMARK_SAMPLES
    ):
        raise ValueError(
            f"benchmark_samples must be an integer in "
            f"[{_MIN_BENCHMARK_SAMPLES}, {_MAX_BENCHMARK_SAMPLES}]"
        )
    return value


def theoretical_remaining_target_count(
    *,
    model_latency_s: float,
    steps: int,
    step_dt_ns: int,
    command_lead_s: float,
) -> int:
    """Return R2-grid targets strictly after model finish plus command lead."""
    for name, value in (
        ("model_latency_s", model_latency_s),
        ("command_lead_s", command_lead_s),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value))
            or value < 0.0
        ):
            raise ValueError(f"{name} must be finite and non-negative")
    origin_ns = 1
    targets = build_target_grid(origin_ns, steps, step_dt_ns)
    finished_ns = origin_ns + math.ceil(float(model_latency_s) * 1e9)
    command_lead_ns = math.ceil(float(command_lead_s) * 1e9)
    first_index = first_deliverable_index(targets, finished_ns, command_lead_ns)
    return len(targets) - first_index


def qualify_policy_warmup(
    timings_s: Any,
    *,
    steps: int,
    step_dt_ns: int,
    command_lead_s: float,
) -> tuple[int, ...]:
    """Apply the shared five-sample, last-three production latency gate."""
    values = tuple(timings_s)
    if len(values) != POLICY_WARMUP_SAMPLES:
        raise RuntimeError("policy runtime returned an incomplete warmup receipt")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise RuntimeError("policy runtime returned invalid warmup timing")
    remaining_targets = tuple(
        theoretical_remaining_target_count(
            model_latency_s=value,
            steps=steps,
            step_dt_ns=step_dt_ns,
            command_lead_s=command_lead_s,
        )
        for value in values[-POLICY_STABLE_WARMUP_SAMPLES:]
    )
    if any(
        remaining < MIN_STARTUP_DELIVERABLE_TARGETS for remaining in remaining_targets
    ):
        raise RuntimeError(
            "policy inference warmup exceeds the viable action window: "
            f"stable_max_ms={max(values[-POLICY_STABLE_WARMUP_SAMPLES:]) * 1e3:.3f} "
            f"stable_remaining_targets={remaining_targets} "
            f"minimum={MIN_STARTUP_DELIVERABLE_TARGETS}"
        )
    return remaining_targets


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
            or result.action_steps != allocation.n_action_steps
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


def run_isolated_policy_check(
    runtime_config: PolicyRuntimeConfig,
    *,
    benchmark_samples: int = 100,
    timeout_s: float = 300.0,
) -> PolicyCheckResult:
    """Load once in a spawn child and run hardware-free startup qualification."""
    if not isinstance(runtime_config, PolicyRuntimeConfig):
        raise TypeError("policy check requires PolicyRuntimeConfig")
    if runtime_config.artifact is None:
        raise ValueError("policy check requires an artifact-bound PolicyRuntimeConfig")
    samples = _validate_benchmark_samples(benchmark_samples)
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(float(timeout_s))
        or timeout_s <= 0.0
    ):
        raise ValueError("policy check timeout_s must be finite and positive")
    context = mp.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_policy_check_child,
        args=(child_connection, runtime_config, samples),
        name="dexmani_policy_check",
    )
    process.daemon = False
    started = False
    try:
        process.start()
        started = True
    except Exception:
        child_connection.close()
        parent_connection.close()
        _close_process(process)
        raise
    else:
        child_connection.close()
    try:
        if not parent_connection.poll(timeout_s):
            raise TimeoutError("policy check child timed out")
        try:
            payload = parent_connection.recv_bytes(_MAX_CHILD_MESSAGE_BYTES)
        except EOFError as exc:
            raise RuntimeError("policy check child exited without a receipt") from exc
        except OSError as exc:
            raise RuntimeError("policy check child receipt exceeds its bound") from exc
        message = _decode_child_message(payload)
        process.join(timeout=_CHILD_JOIN_TIMEOUT_S)
        if process.is_alive():
            raise RuntimeError("policy check child did not exit after its receipt")
        if process.exitcode != 0:
            raise RuntimeError(
                f"policy check child exited with code {process.exitcode}"
            )
        if message["ok"] is not True:
            raise RuntimeError(f"policy check failed: {message['error']}")
        result = PolicyCheckResult.from_wire(message["receipt"])
        allocation = runtime_config.artifact.allocation_contract
        if (
            result.checkpoint_sha256
            != runtime_config.artifact.checkpoint_sha256_from_index
            or result.action_steps != allocation.n_action_steps
            or result.action_dim != allocation.action_dim
            or result.device != runtime_config.deployment.device
            or result.seed != runtime_config.deployment.inference_seed
            or result.benchmark_samples != samples
        ):
            raise RuntimeError("policy check child receipt mismatches its request")
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


def _policy_check_child(
    connection: Any,
    runtime_config: PolicyRuntimeConfig,
    benchmark_samples: int,
) -> None:
    try:
        result = _run_policy_check_child(runtime_config, benchmark_samples)
        message: dict[str, Any] = {
            "ok": True,
            "receipt": result.to_wire(),
            "error": None,
        }
    except Exception as exc:
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

    This is the inference-child counterpart to offline ``check``. It uses
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
            from dexmani_real.deployment.policy_checkpoint import (
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

        # Runtime construction already initialized the checkpoint-bound RNG
        # streams. Do not replace them here: preflight must exercise the same
        # first prediction stream as the operational inference worker.
        contract_warmup = runtime.warmup(samples=1)
        if (
            len(contract_warmup) != 1
            or not np.isfinite(contract_warmup[0])
            or contract_warmup[0] < 0.0
        ):
            raise RuntimeError("policy preflight returned invalid warmup timing")
        observation = _synthetic_observation(runtime_config)
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
        action_steps=artifact.allocation_contract.n_action_steps,
        action_dim=artifact.allocation_contract.action_dim,
        package_origin=provenance["origin"],
        package_commit=provenance["commit"],
        package_dirty=provenance["dirty"],
        package_source_tree_sha256=provenance["source_tree_sha256"],
        package_version=provenance["version"],
    )


def _nearest_rank(values: list[float] | list[int], quantile: float) -> float | int:
    if not values:
        raise ValueError("nearest-rank quantile requires at least one value")
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _require_hardware_free_check_imports() -> None:
    loaded = sorted(
        name
        for name in sys.modules
        if name in _FORBIDDEN_CHECK_IMPORTS
        or name == "xarm"
        or name.startswith("xarm.")
    )
    if loaded:
        raise RuntimeError(f"policy check imported hardware owner modules: {loaded}")


def _theoretical_remaining_targets(
    runtime_config: PolicyRuntimeConfig, latency_s: float
) -> int:
    artifact = runtime_config.artifact
    assert artifact is not None
    allocation = artifact.allocation_contract
    step_dt_ns = int(round(allocation.control_dt_s * 1e9))
    return theoretical_remaining_target_count(
        model_latency_s=latency_s,
        steps=allocation.n_action_steps,
        step_dt_ns=step_dt_ns,
        command_lead_s=runtime_config.deployment.command_lead_s,
    )


def _run_policy_check_child(
    runtime_config: PolicyRuntimeConfig, benchmark_samples: int
) -> PolicyCheckResult:
    samples = _validate_benchmark_samples(benchmark_samples)
    runtime, actual_sha256, provenance = _load_verified_policy_runtime(runtime_config)
    try:
        import torch

        warmup_timings_s = runtime.warmup(samples=POLICY_WARMUP_SAMPLES)
        artifact = runtime_config.artifact
        assert artifact is not None
        qualify_policy_warmup(
            warmup_timings_s,
            steps=artifact.allocation_contract.n_action_steps,
            step_dt_ns=int(round(artifact.allocation_contract.control_dt_s * 1e9)),
            command_lead_s=runtime_config.deployment.command_lead_s,
        )

        observation = _synthetic_observation(runtime_config)
        device = torch.device(runtime_config.deployment.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        latencies_ms: list[float] = []
        remaining_targets: list[int] = []
        with torch.inference_mode():
            for _ in range(samples):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started_ns = time.perf_counter_ns()
                prediction = runtime.predict(observation)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed_s = (time.perf_counter_ns() - started_ns) / 1e9
                _validate_prediction(prediction, runtime_config)
                latencies_ms.append(elapsed_s * 1e3)
                remaining_targets.append(
                    _theoretical_remaining_targets(runtime_config, elapsed_s)
                )
        gpu_peak_memory_bytes = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        )
    finally:
        runtime.close()

    _require_hardware_free_check_imports()
    artifact = runtime_config.artifact
    assert artifact is not None
    allocation = artifact.allocation_contract
    return PolicyCheckResult(
        checkpoint_sha256=actual_sha256,
        checkpoint_sha256_verified=True,
        action_steps=allocation.n_action_steps,
        action_dim=allocation.action_dim,
        package_origin=provenance["origin"],
        package_commit=provenance["commit"],
        package_dirty=provenance["dirty"],
        package_source_tree_sha256=provenance["source_tree_sha256"],
        package_version=provenance["version"],
        device=runtime_config.deployment.device,
        seed=runtime_config.deployment.inference_seed,
        benchmark_samples=samples,
        latency_p50_ms=float(_nearest_rank(latencies_ms, 0.50)),
        latency_p95_ms=float(_nearest_rank(latencies_ms, 0.95)),
        latency_max_ms=max(latencies_ms),
        remaining_targets_min=min(remaining_targets),
        remaining_targets_p50=int(_nearest_rank(remaining_targets, 0.50)),
        remaining_targets_p95=int(_nearest_rank(remaining_targets, 0.95)),
        zero_deliverable_samples=sum(value == 0 for value in remaining_targets),
        source_aware_schedulability="NOT_MEASURED",
        gpu_peak_memory_bytes=gpu_peak_memory_bytes,
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


def _synthetic_observation(runtime_config: PolicyRuntimeConfig) -> ObservationBatch:
    allocation = (
        runtime_config.artifact.allocation_contract if runtime_config.artifact else None
    )
    if allocation is None:
        raise ValueError("synthetic observation requires an artifact")
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
    pointcloud_history: tuple[PointCloudFrame, ...] = ()
    if allocation.point_cloud_num_points is not None:
        assert allocation.point_cloud_feature_dim is not None
        points = np.zeros(
            (allocation.point_cloud_num_points, allocation.point_cloud_feature_dim),
            dtype=np.float32,
        )
        points[:, 0] = 0.4
        points[:, 3:] = 0.5
        pointcloud_history = tuple(
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
    rgb_history: tuple[RgbFrame, ...] = ()
    if allocation.rgb_shape is not None:
        rgb = np.zeros(allocation.rgb_shape, dtype=np.uint8)
        rgb_history = tuple(
            RgbFrame(
                values=rgb,
                source_camera_sequence=int(index),
                source_monotonic_ns=int(source),
                publish_monotonic_ns=int(published),
                camera_generation=1,
            )
            for index, source, published in zip(
                sequence, source_ns, publish_ns, strict=True
            )
        )
    latest_source_ns = int(source_ns[-1])
    latest_publish_ns = int(publish_ns[-1])
    latest_pointcloud = pointcloud_history[-1] if pointcloud_history else None
    return ObservationBatch(
        observation_id=1,
        run_generation=1,
        run_started_monotonic_ns=run_ns,
        anchor_monotonic_ns=latest_publish_ns + 1_000_000,
        latest_source_monotonic_ns=latest_source_ns,
        logical_step_monotonic_ns=latest_source_ns,
        arm_history=arm,
        hand_history=hand,
        pointcloud=latest_pointcloud,
        pointcloud_history=pointcloud_history,
        rgb_history=rgb_history,
    )


def _validate_prediction(prediction: Any, runtime_config: PolicyRuntimeConfig) -> None:
    artifact = runtime_config.artifact
    assert artifact is not None
    if not isinstance(prediction, PolicyPrediction):
        raise TypeError("synthetic preflight prediction must be a PolicyPrediction")
    action_key = artifact.allocation_contract.action_key
    if action_key == "action" and prediction.is_ee:
        raise ValueError(
            "synthetic preflight joint action artifact requires arm7 joint actions, not EE"
        )
    if action_key == "action_ee" and not prediction.is_ee:
        raise ValueError(
            "synthetic preflight EE action artifact requires ee_pos/ee_rot6d actions"
        )
    if action_key not in {"action", "action_ee"}:
        raise ValueError(
            f"synthetic preflight artifact has unsupported action_key={action_key!r}"
        )
    expected_steps = artifact.allocation_contract.n_action_steps
    if prediction.hand_qpos is None or prediction.hand_qpos.shape != (
        expected_steps,
        _HAND_DOF,
    ):
        raise ValueError("synthetic preflight prediction is missing hand12 actions")
    if prediction.arm_qpos is not None:
        if prediction.arm_qpos.shape != (expected_steps, _ARM_DOF):
            raise ValueError(
                "synthetic preflight joint prediction has invalid arm7 actions"
            )
        if prediction.ee_pos is not None or prediction.ee_rot6d is not None:
            raise ValueError(
                "synthetic preflight joint prediction mixes joint and EE actions"
            )
        arrays = [prediction.arm_qpos, prediction.hand_qpos]
    else:
        if (
            prediction.ee_pos is None
            or prediction.ee_rot6d is None
            or prediction.ee_pos.shape != (expected_steps, 3)
            or prediction.ee_rot6d.shape != (expected_steps, 6)
        ):
            raise ValueError(
                "synthetic preflight EE prediction has invalid geometry fields"
            )
        _validate_ee_rot6d(prediction.ee_rot6d)
        arrays = [prediction.ee_pos, prediction.ee_rot6d, prediction.hand_qpos]
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("synthetic preflight prediction contains NaN/Inf")


def _validate_ee_rot6d(rot6d: np.ndarray) -> None:
    """Reject degenerate 6D rotations without importing an IK/planning module."""
    first = np.asarray(rot6d[:, :3], dtype=np.float64)
    second = np.asarray(rot6d[:, 3:], dtype=np.float64)
    first_norm = np.linalg.norm(first, axis=1)
    if np.any(first_norm <= 1e-8):
        raise ValueError(
            "synthetic preflight EE prediction has degenerate rotation geometry"
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
            "synthetic preflight EE prediction has degenerate rotation geometry"
        )


def _close_fd(fd: int | None) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass
