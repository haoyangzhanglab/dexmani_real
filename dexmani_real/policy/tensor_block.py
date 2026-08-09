"""Double-buffered seqlock tensor transport for learned-policy inference."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from dexmani_real.policy.runtime import FrozenArrayMap, ObservationSnapshot, ObservationSpec
from dexmani_real.shm.robot_ring import SeqlockRingBuffer


def _field_prefix(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", name):
        raise ValueError(f"modality name {name!r} is not safe for a structured tensor block")
    return name


def observation_tensor_dtype(spec: ObservationSpec) -> np.dtype:
    fields: list[tuple[Any, ...]] = [
        ("observation_id", "<u8"),
        ("anchor_monotonic_ns", "<u8"),
        ("session_generation", "<u8"),
        ("camera_generation", "<u8"),
    ]
    for modality in spec.modalities:
        prefix = _field_prefix(modality.name)
        history_shape = (modality.history_length,)
        fields.extend(
            [
                (f"{prefix}__value", np.dtype(modality.dtype), history_shape + modality.shape),
                (f"{prefix}__source_ns", "<u8", history_shape),
                (f"{prefix}__receive_ns", "<u8", history_shape),
                (f"{prefix}__publish_ns", "<u8", history_shape),
                (f"{prefix}__age_s", "<f8", history_shape),
                (f"{prefix}__skew_s", "<f8", history_shape),
                (f"{prefix}__valid", "<u1", history_shape),
            ]
        )
    return np.dtype(fields, align=True)


class ObservationTensorBlock:
    """Two fixed slots; one snapshot copy per policy/inference boundary."""

    def __init__(self, ring: SeqlockRingBuffer, spec: ObservationSpec) -> None:
        expected = observation_tensor_dtype(spec)
        if ring.dtype != expected or ring.maxlen != 2:
            raise ValueError("tensor block ring does not match ObservationSpec/double-buffer contract")
        self.ring = ring
        self.spec = spec

    @classmethod
    def create(cls, name: str, spec: ObservationSpec) -> "ObservationTensorBlock":
        ring = SeqlockRingBuffer.create_or_replace(name, observation_tensor_dtype(spec), maxlen=2)
        return cls(ring, spec)

    def write(self, snapshot: ObservationSnapshot) -> int:
        frame = np.zeros(1, dtype=self.ring.dtype)
        frame["observation_id"][0] = snapshot.observation_id
        frame["anchor_monotonic_ns"][0] = snapshot.anchor_monotonic_ns
        frame["session_generation"][0] = snapshot.session_generation
        frame["camera_generation"][0] = snapshot.camera_generation
        for modality in self.spec.modalities:
            prefix = _field_prefix(modality.name)
            frame[f"{prefix}__value"][0] = snapshot.values[modality.name]
            frame[f"{prefix}__source_ns"][0] = snapshot.source_monotonic_ns[modality.name]
            try:
                receive_ns = snapshot.receive_monotonic_ns[modality.name]
            except KeyError:
                receive_ns = snapshot.publish_monotonic_ns[modality.name]
            frame[f"{prefix}__receive_ns"][0] = receive_ns
            frame[f"{prefix}__publish_ns"][0] = snapshot.publish_monotonic_ns[modality.name]
            try:
                age_s = snapshot.source_age_s[modality.name]
            except KeyError:
                source_ns = np.asarray(snapshot.source_monotonic_ns[modality.name], dtype=np.uint64)
                valid = np.asarray(snapshot.valid_history_mask[modality.name], dtype=bool)
                age_s = np.where(valid, (snapshot.anchor_monotonic_ns - source_ns) / 1e9, np.nan)
            try:
                skew_s = snapshot.source_skew_s[modality.name]
            except KeyError:
                skew_s = np.where(snapshot.valid_history_mask[modality.name], 0.0, np.nan)
            frame[f"{prefix}__age_s"][0] = age_s
            frame[f"{prefix}__skew_s"][0] = skew_s
            frame[f"{prefix}__valid"][0] = snapshot.valid_history_mask[modality.name]
        return self.ring.write(frame)

    def read_latest(self) -> tuple[ObservationSnapshot, int] | None:
        result = self.ring.read_latest()
        if result is None:
            return None
        data, _publish_ns, sequence = result
        record = data[0]
        value_items: list[tuple[str, np.ndarray]] = []
        source_items: list[tuple[str, np.ndarray]] = []
        receive_items: list[tuple[str, np.ndarray]] = []
        publish_items: list[tuple[str, np.ndarray]] = []
        age_items: list[tuple[str, np.ndarray]] = []
        skew_items: list[tuple[str, np.ndarray]] = []
        valid_items: list[tuple[str, np.ndarray]] = []
        for modality in self.spec.modalities:
            prefix = _field_prefix(modality.name)
            value_items.append((modality.name, np.array(record[f"{prefix}__value"], copy=True)))
            source_items.append((modality.name, np.array(record[f"{prefix}__source_ns"], copy=True)))
            receive_items.append((modality.name, np.array(record[f"{prefix}__receive_ns"], copy=True)))
            publish_items.append((modality.name, np.array(record[f"{prefix}__publish_ns"], copy=True)))
            age_items.append((modality.name, np.array(record[f"{prefix}__age_s"], copy=True)))
            skew_items.append((modality.name, np.array(record[f"{prefix}__skew_s"], copy=True)))
            valid_items.append((modality.name, np.asarray(record[f"{prefix}__valid"], dtype=bool)))
        snapshot = ObservationSnapshot(
            observation_id=int(record["observation_id"]),
            anchor_monotonic_ns=int(record["anchor_monotonic_ns"]),
            values=FrozenArrayMap(tuple(value_items)),
            source_monotonic_ns=FrozenArrayMap(tuple(source_items)),
            publish_monotonic_ns=FrozenArrayMap(tuple(publish_items)),
            valid_history_mask=FrozenArrayMap(tuple(valid_items)),
            session_generation=int(record["session_generation"]),
            camera_generation=int(record["camera_generation"]),
            receive_monotonic_ns=FrozenArrayMap(tuple(receive_items)),
            source_age_s=FrozenArrayMap(tuple(age_items)),
            source_skew_s=FrozenArrayMap(tuple(skew_items)),
        )
        return snapshot, sequence

    def close(self) -> None:
        self.ring.close()

    def unlink(self) -> None:
        self.ring.unlink()
