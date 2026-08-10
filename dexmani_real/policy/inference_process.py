"""Optional learned-policy worker; it has no actuator or SharedStorage write authority."""

from __future__ import annotations

import hashlib
import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dexmani_real.ipc.schema import INFERENCE_CANDIDATE_DTYPE
from dexmani_real.policy.runtime import (
    ActionCandidate,
    ActionChunk,
    ActionSpec,
    FrozenArrayMap,
    ModalitySpec,
    ObservationSnapshot,
    ObservationSpec,
    PolicyBackend,
)
from dexmani_real.policy.tensor_block import ObservationTensorBlock
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager

logger = get_logger(__name__)


@dataclass(frozen=True)
class InferenceConfig:
    backend_entrypoint: str
    observation_spec: ObservationSpec
    action_spec: ActionSpec
    poll_hz: float = 256.0
    benchmark_deadline_s: float = 0.20
    manifest_sha256: str = ""
    resource_hashes: tuple[tuple[str, str], ...] = ()
    actuators: tuple[str, ...] = ("arm", "hand")

    def __post_init__(self) -> None:
        if (
            ":" not in self.backend_entrypoint
            or not np.isfinite(self.poll_hz)
            or not np.isfinite(self.benchmark_deadline_s)
            or self.poll_hz <= 0
            or self.benchmark_deadline_s <= 0
        ):
            raise ValueError("invalid inference entrypoint/rate/deadline")
        if not self.actuators or "arm" not in self.actuators or set(self.actuators) - {"arm", "hand"}:
            raise ValueError("inference actuators must contain arm and may contain hand")

    @classmethod
    def from_manifest(cls, path: str | Path) -> "InferenceConfig":
        """Load and verify a backend-neutral model resource manifest."""
        manifest_path = Path(path).resolve()
        raw = manifest_path.read_bytes()
        decoded = json.loads(raw.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise TypeError("inference manifest root must be an object")
        modalities_raw = decoded.get("modalities")
        if not isinstance(modalities_raw, list) or not modalities_raw:
            raise ValueError("inference manifest requires a non-empty modalities list")
        modalities: list[ModalitySpec] = []
        for item in modalities_raw:
            if not isinstance(item, dict):
                raise TypeError("each inference modality must be an object")
            modality_data = dict(item)
            if "shape" in modality_data:
                modality_data["shape"] = tuple(modality_data["shape"])
            modalities.append(ModalitySpec(**modality_data))
        observation_spec = ObservationSpec(
            tuple(modalities),
            control_hz=float(decoded.get("control_hz", 16.0)),
        )
        action_data = dict(decoded.get("action") or {})
        for shape_name in ("arm_shape", "hand_shape"):
            if shape_name in action_data:
                action_data[shape_name] = tuple(action_data[shape_name])
        action_spec = ActionSpec(**action_data)

        resources_raw = decoded.get("resources") or {}
        if not isinstance(resources_raw, dict):
            raise TypeError("inference manifest resources must be an object")
        resource_hashes: list[tuple[str, str]] = []
        for name, item in sorted(resources_raw.items()):
            if not isinstance(item, dict) or "path" not in item or "sha256" not in item:
                raise ValueError(f"resource {name!r} requires path and sha256")
            resource_path = (manifest_path.parent / str(item["path"])).resolve()
            expected = str(item["sha256"]).lower()
            actual = hashlib.sha256(resource_path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"resource {name!r} SHA-256 mismatch")
            resource_hashes.append((str(name), actual))
        return cls(
            backend_entrypoint=str(decoded["backend_entrypoint"]),
            observation_spec=observation_spec,
            action_spec=action_spec,
            poll_hz=float(decoded.get("poll_hz", 256.0)),
            benchmark_deadline_s=float(decoded.get("benchmark_deadline_s", action_spec.deadline_s)),
            manifest_sha256=hashlib.sha256(raw).hexdigest(),
            resource_hashes=tuple(resource_hashes),
            actuators=tuple(str(name) for name in decoded.get("actuators", ("arm", "hand"))),
        )


@dataclass(frozen=True)
class InferenceWorkerTransport:
    """Least-authority IPC view passed to the inference child."""

    is_running: Any
    heartbeat_s: Any
    ready: Any
    fault_latch: Any
    candidate_ring: Any
    component_status_ring: Any
    component_status_lock: Any
    session_generation: Any

    @classmethod
    def from_shared(cls, shared: Any) -> "InferenceWorkerTransport":
        return cls(
            is_running=shared.is_running,
            heartbeat_s=shared.inference_heartbeat_s,
            ready=shared.inference_ready,
            fault_latch=shared.error_state,
            candidate_ring=shared.inference_candidate_ring,
            component_status_ring=shared.component_status_ring,
            component_status_lock=shared.component_status_lock,
            session_generation=shared.session_generation,
        )


def _load_backend(entrypoint: str) -> PolicyBackend:
    module_name, class_name = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    backend = getattr(module, class_name)()
    if not isinstance(backend, PolicyBackend):
        raise TypeError(f"{entrypoint} does not implement PolicyBackend")
    return backend


def encode_candidate(candidate: ActionCandidate, *, chunk_length: int = 1) -> np.ndarray:
    frame = np.zeros(1, dtype=INFERENCE_CANDIDATE_DTYPE)
    for name in (
        "observation_id",
        "session_generation",
        "policy_epoch",
        "action_id",
        "chunk_id",
        "step_index",
        "created_monotonic_ns",
        "target_monotonic_ns",
        "valid_until_monotonic_ns",
        "is_hold",
    ):
        frame[name][0] = getattr(candidate, name)
    frame["chunk_length"][0] = int(chunk_length)
    if candidate.arm_qpos is not None:
        frame["has_arm"][0] = 1
        frame["arm_qpos"][0] = candidate.arm_qpos
    if candidate.hand_qpos is not None:
        frame["has_hand"][0] = 1
        frame["hand_qpos"][0] = candidate.hand_qpos
    return frame


def decode_candidate(frame: np.ndarray) -> tuple[ActionCandidate, int]:
    """Decode one fixed inference mailbox record with full validation."""
    if frame.shape != (1,) or frame.dtype != INFERENCE_CANDIDATE_DTYPE:
        raise ValueError("invalid inference candidate frame")
    record = frame[0]
    candidate = ActionCandidate(
        observation_id=int(record["observation_id"]),
        session_generation=int(record["session_generation"]),
        policy_epoch=int(record["policy_epoch"]),
        action_id=int(record["action_id"]),
        created_monotonic_ns=int(record["created_monotonic_ns"]),
        target_monotonic_ns=int(record["target_monotonic_ns"]),
        valid_until_monotonic_ns=int(record["valid_until_monotonic_ns"]),
        arm_qpos=np.array(record["arm_qpos"], copy=True) if bool(record["has_arm"]) else None,
        hand_qpos=np.array(record["hand_qpos"], copy=True) if bool(record["has_hand"]) else None,
        chunk_id=int(record["chunk_id"]),
        step_index=int(record["step_index"]),
        is_hold=bool(record["is_hold"]),
    )
    chunk_length = int(record["chunk_length"])
    if chunk_length <= 0 or candidate.step_index >= chunk_length:
        raise ValueError("invalid inference chunk metadata")
    return candidate, chunk_length


def _synthetic_snapshot(spec: ObservationSpec) -> ObservationSnapshot:
    now_ns = time.monotonic_ns()
    values = {
        modality.name: np.zeros((modality.history_length,) + modality.shape, dtype=np.dtype(modality.dtype))
        for modality in spec.modalities
    }
    times = FrozenArrayMap(
        tuple(
            (modality.name, np.full(modality.history_length, now_ns, dtype=np.uint64)) for modality in spec.modalities
        )
    )
    masks = FrozenArrayMap(
        tuple((modality.name, np.ones(modality.history_length, dtype=bool)) for modality in spec.modalities)
    )
    zero_timing = FrozenArrayMap(
        tuple((modality.name, np.zeros(modality.history_length, dtype=np.float64)) for modality in spec.modalities)
    )
    return ObservationSnapshot(
        observation_id=1,
        anchor_monotonic_ns=now_ns,
        values=FrozenArrayMap.validated(values, spec),
        source_monotonic_ns=times,
        publish_monotonic_ns=times,
        valid_history_mask=masks,
        session_generation=0,
        receive_monotonic_ns=times,
        source_age_s=zero_timing,
        source_skew_s=zero_timing,
    )


def _validate_output(
    output: ActionCandidate | ActionChunk,
    snapshot: ObservationSnapshot,
    action_spec: ActionSpec,
    actuators: tuple[str, ...] = ("arm", "hand"),
) -> tuple[ActionCandidate, ...]:
    candidates = output.steps if isinstance(output, ActionChunk) else (output,)
    if not candidates or len(candidates) > action_spec.chunk_length:
        raise ValueError("backend output chunk length violates ActionSpec")
    for candidate in candidates:
        if candidate.observation_id != snapshot.observation_id:
            raise ValueError("backend output does not match its observation")
        if (
            candidate.representation != action_spec.representation
            or candidate.units != action_spec.units
            or candidate.frame != action_spec.frame
        ):
            raise ValueError("backend output representation/units/frame violates ActionSpec")
        if candidate.arm_qpos is None:
            raise ValueError("DexMani learned backends must produce an arm action")
        if "hand" not in actuators and candidate.hand_qpos is not None:
            raise ValueError("backend produced hand action without the hand capability")
    return candidates


def inference_loop(
    transport: InferenceWorkerTransport,
    tensor_block: ObservationTensorBlock,
    config: InferenceConfig,
) -> None:
    """Load/warm selected backend lazily, then infer from immutable snapshots."""
    from dexmani_real.runtime.status import ComponentPhase, FaultCode
    from dexmani_real.shm.shared_storage import publish_component_status

    backend: PolicyBackend | None = None
    ready = False
    failed = False
    try:
        publish_component_status(transport, "inference", ComponentPhase.LOADING)
        backend = _load_backend(config.backend_entrypoint)
        backend.load()
        publish_component_status(transport, "inference", ComponentPhase.WARMING_UP)
        backend.warmup(config.observation_spec, config.action_spec)
        warmup_snapshot = _synthetic_snapshot(config.observation_spec)
        benchmark_started = time.monotonic()
        _validate_output(backend.infer(warmup_snapshot), warmup_snapshot, config.action_spec, config.actuators)
        benchmark_s = time.monotonic() - benchmark_started
        if benchmark_s > config.benchmark_deadline_s:
            raise TimeoutError(
                f"inference warmup benchmark {benchmark_s:.3f}s exceeds {config.benchmark_deadline_s:.3f}s"
            )
        transport.heartbeat_s.value = time.monotonic()
        transport.ready.set()
        publish_component_status(transport, "inference", ComponentPhase.READY)
        ready = True
        last_sequence = 0
        last_camera_generation: int | None = None
        running_published = False
        limiter = RateManager(config.poll_hz)
        while transport.is_running.value:
            transport.heartbeat_s.value = time.monotonic()
            result = tensor_block.read_latest()
            if result is not None:
                snapshot, sequence = result
                if sequence != last_sequence:
                    camera_generation_changed = (
                        last_camera_generation is not None and snapshot.camera_generation != last_camera_generation
                    )
                    if camera_generation_changed:
                        transport.ready.clear()
                        publish_component_status(transport, "inference", ComponentPhase.WARMING_UP)
                        backend.warmup(config.observation_spec, config.action_spec)
                        _validate_output(
                            backend.infer(snapshot),
                            snapshot,
                            config.action_spec,
                            config.actuators,
                        )
                        transport.heartbeat_s.value = time.monotonic()
                        transport.ready.set()
                        publish_component_status(transport, "inference", ComponentPhase.READY)
                        running_published = False
                        last_camera_generation = snapshot.camera_generation
                        last_sequence = sequence
                        limiter.wait()
                        continue
                    if not running_published:
                        publish_component_status(transport, "inference", ComponentPhase.RUNNING)
                        running_published = True
                    output = backend.infer(snapshot)
                    candidates = _validate_output(output, snapshot, config.action_spec, config.actuators)
                    for candidate in candidates:
                        transport.candidate_ring.write(encode_candidate(candidate, chunk_length=len(candidates)))
                    last_camera_generation = snapshot.camera_generation
                    last_sequence = sequence
            limiter.wait()
    except Exception:
        failed = True
        logger.error("Inference worker failed", exc_info=True)
        publish_component_status(
            transport,
            "inference",
            ComponentPhase.FAULT,
            fault_code=FaultCode.INFERENCE_FAILED,
            detail="backend load/warmup/infer failed; see process log",
        )
        # Startup failure keeps the system DISARMED; runtime failure is a
        # policy safety failure and therefore latches the global fault.
        if ready:
            transport.fault_latch.value = True
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                logger.warning("Inference backend close failed", exc_info=True)
        if not failed:
            publish_component_status(transport, "inference", ComponentPhase.STOPPED)
        logger.info("Inference worker exited")
