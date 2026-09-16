"""Legacy async trace reader for previously recorded rollout sidecars.

Synchronous rollout does not produce this format. Validation retains the v1
field semantics for the offline visualization tool.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

TRACE_VERSION = 1
_SCALARS = {
    "trace_version": "int32",
    "run_started_monotonic_ns": "uint64",
    "chunk_size": "int32",
    "action_dim": "int32",
}
_PREDICTIONS = {
    "prediction_sequence": "uint64",
    "prediction_publish_monotonic_ns": "uint64",
    "prediction_ingest_monotonic_ns": "uint64",
    "prediction_source_monotonic_ns": "uint64",
    "prediction_logical_step_monotonic_ns": "uint64",
    "prediction_first_index": "int32",
    "prediction_inference_latency_ms": "float64",
    "prediction_observation_age_ms": "float64",
    "prediction_observation_skew_ms": "float64",
}
_DECISIONS = {
    "decision_prediction_sequence": "uint64",
    "decision_chunk_index": "int32",
    "decision_due_monotonic_ns": "uint64",
    "decision_event_monotonic_ns": "uint64",
    "decision_frame_status": "uint8",
}


def _integer(value: object, dtype: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"expected {dtype} integer, got {value!r}")
    result = int(value)
    if not minimum <= result <= np.iinfo(dtype).max:
        raise ValueError(f"integer outside {dtype} range: {result}")
    return result


def trace_path_for_episode(episode: str | Path) -> Path:
    """Derive the sibling filename from a final published episode path."""
    path = Path(episode)
    if not path.name or path.name in {".", ".."}:
        raise ValueError("trace requires a named episode directory")
    return path.parent / f"{path.name}.policy_trace.npz"


class _TraceValidator:
    """Check legacy row ordering, values, and cross references."""

    def __init__(self, run_started_monotonic_ns: int, chunk_size: int, action_dim: int):
        _integer(run_started_monotonic_ns, "uint64", minimum=1)
        self.chunk_size = _integer(chunk_size, "int32", minimum=1)
        self.action_dim = _integer(action_dim, "int32", minimum=1)
        self._last_sequence: int | None = None
        self._sequences: set[int] = set()
        self._terminal_ids: set[tuple[int, int]] = set()

    def add_prediction(
        self,
        *,
        sequence: int,
        publish_ns: int,
        ingest_ns: int,
        source_ns: int,
        logical_step_ns: int,
        first_index: int,
        inference_latency_ms: float,
        observation_age_ms: float,
        observation_skew_ms: float,
        actions: np.ndarray,
    ) -> None:
        sequence = _integer(sequence, "uint64")
        # Legacy ingest time was sampled before reading the prediction ring,
        # so it may precede the ring commit; validate values, not their ordering.
        for value in (publish_ns, ingest_ns, source_ns, logical_step_ns):
            _integer(value, "uint64", minimum=1)
        if self._last_sequence is not None and sequence <= self._last_sequence:
            raise ValueError("prediction sequences must strictly increase")
        first_index = _integer(first_index, "int32", minimum=-1)
        if first_index >= self.chunk_size:
            raise ValueError("prediction first_index is outside the chunk")
        metrics = tuple(
            float(v)
            for v in (inference_latency_ms, observation_age_ms, observation_skew_ms)
        )
        if not all(np.isfinite(v) and v >= 0 for v in metrics):
            raise ValueError(
                "prediction timing metrics must be finite and non-negative"
            )
        action_values = np.asarray(actions, dtype=np.float64)
        if action_values.shape != (self.chunk_size, self.action_dim):
            raise ValueError("prediction actions shape does not match trace dimensions")
        if not np.all(np.isfinite(action_values)):
            raise ValueError("prediction actions must be finite")
        self._last_sequence = sequence
        self._sequences.add(sequence)

    def add_decision(
        self,
        *,
        sequence: int,
        chunk_index: int,
        due_ns: int,
        event_ns: int,
        frame_status: int,
    ) -> None:
        sequence = _integer(sequence, "uint64")
        chunk_index = _integer(chunk_index, "int32")
        due_ns = _integer(due_ns, "uint64", minimum=1)
        event_ns = _integer(event_ns, "uint64", minimum=1)
        frame_status = _integer(frame_status, "uint8")
        if sequence not in self._sequences:
            raise ValueError("decision references an unknown prediction")
        if chunk_index >= self.chunk_size:
            raise ValueError("decision chunk_index is outside the chunk")
        if frame_status not in {0, 2, 3}:
            raise ValueError("decision status must be 0, 2, or 3")
        if due_ns > event_ns:
            raise ValueError("decision due time exceeds terminal event time")
        identity = (sequence, chunk_index)
        if identity in self._terminal_ids:
            raise ValueError("duplicate terminal decision")
        self._terminal_ids.add(identity)


def validate_trace(payload: Mapping[str, np.ndarray]) -> None:
    """Validate exact v1 representation and row references before viewer admission."""
    fields = _SCALARS | _PREDICTIONS | _DECISIONS | {"actions": "float64"}
    if set(payload) != set(fields):
        raise ValueError("policy trace has missing or unexpected keys")
    for key, dtype in fields.items():
        if not isinstance(payload[key], np.ndarray) or payload[key].dtype != np.dtype(
            dtype
        ):
            raise ValueError(f"policy trace {key} must have dtype {dtype}")
    for key in _SCALARS:
        if payload[key].shape != ():
            raise ValueError(f"policy trace {key} must be scalar")
    if int(payload["trace_version"]) != TRACE_VERSION:
        raise ValueError("unsupported policy trace version")
    trace = _TraceValidator(
        int(payload["run_started_monotonic_ns"]),
        int(payload["chunk_size"]),
        int(payload["action_dim"]),
    )
    p = payload["prediction_sequence"].size
    e = payload["decision_prediction_sequence"].size
    for fields, count in ((_PREDICTIONS, p), (_DECISIONS, e)):
        if any(payload[key].shape != (count,) for key in fields):
            raise ValueError("policy trace row arrays have inconsistent shapes")
    if payload["actions"].shape != (p, trace.chunk_size, trace.action_dim):
        raise ValueError("policy trace actions have inconsistent shape")
    # Preserve the original v1 row and reference admission rules.
    for i in range(p):
        trace.add_prediction(
            sequence=payload["prediction_sequence"][i],
            publish_ns=payload["prediction_publish_monotonic_ns"][i],
            ingest_ns=payload["prediction_ingest_monotonic_ns"][i],
            source_ns=payload["prediction_source_monotonic_ns"][i],
            logical_step_ns=payload["prediction_logical_step_monotonic_ns"][i],
            first_index=payload["prediction_first_index"][i],
            inference_latency_ms=payload["prediction_inference_latency_ms"][i],
            observation_age_ms=payload["prediction_observation_age_ms"][i],
            observation_skew_ms=payload["prediction_observation_skew_ms"][i],
            actions=payload["actions"][i],
        )
    for i in range(e):
        trace.add_decision(
            sequence=payload["decision_prediction_sequence"][i],
            chunk_index=payload["decision_chunk_index"][i],
            due_ns=payload["decision_due_monotonic_ns"][i],
            event_ns=payload["decision_event_monotonic_ns"][i],
            frame_status=payload["decision_frame_status"][i],
        )


def load_trace(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("policy trace contains duplicate keys")
        payload = {key: archive[key] for key in archive.files}
    validate_trace(payload)
    return payload
