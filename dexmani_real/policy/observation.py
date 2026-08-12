"""Causal, grid-anchored multi-rate observation construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from dexmani_real.policy.runtime import FrozenArrayMap, ObservationSnapshot, ObservationSpec


@dataclass(frozen=True)
class CausalFrame:
    value: np.ndarray
    sequence: int
    source_monotonic_ns: int
    publish_monotonic_ns: int
    receive_monotonic_ns: int = 0
    generation: int = 0

    def __post_init__(self) -> None:
        if self.sequence <= 0 or self.source_monotonic_ns <= 0 or self.publish_monotonic_ns <= 0:
            raise ValueError("causal frame identity/timestamps must be positive")
        if self.publish_monotonic_ns < self.source_monotonic_ns:
            raise ValueError("frame publish time precedes source time")
        if self.receive_monotonic_ns and not (
            self.source_monotonic_ns <= self.receive_monotonic_ns <= self.publish_monotonic_ns
        ):
            raise ValueError("frame receive time is outside source/publish interval")


def _readonly(value: Any, dtype: Any) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.flags.writeable = False
    return result


class SnapshotBuilder:
    """Select only verified ``source_time <= anchor`` frames for each history slot."""

    def __init__(self, spec: ObservationSpec, *, run_generation: int, camera_generation: int = 0) -> None:
        self.spec = spec
        self.run_generation = int(run_generation)
        self.camera_generation = int(camera_generation)
        self._next_observation_id = 1

    def build(
        self,
        *,
        anchor_monotonic_ns: int,
        frames: Mapping[str, list[CausalFrame] | tuple[CausalFrame, ...]],
    ) -> ObservationSnapshot:
        if anchor_monotonic_ns <= 0:
            raise ValueError("anchor_monotonic_ns must be positive")
        dt_ns = int(round(1e9 / self.spec.control_hz))
        values: dict[str, np.ndarray] = {}
        source_times: dict[str, np.ndarray] = {}
        publish_times: dict[str, np.ndarray] = {}
        receive_times: dict[str, np.ndarray] = {}
        valid_masks: dict[str, np.ndarray] = {}
        source_ages_s: dict[str, np.ndarray] = {}

        for modality in self.spec.modalities:
            candidates = sorted(
                frames.get(modality.name, ()), key=lambda frame: (frame.source_monotonic_ns, frame.sequence)
            )
            used_sequences: set[tuple[int, int]] = set()
            fill = np.nan if modality.padding == "invalid_nan" else 0
            history_values = [
                np.full(modality.shape, fill, dtype=np.dtype(modality.dtype)) for _ in range(modality.history_length)
            ]
            history_source = np.zeros(modality.history_length, dtype=np.uint64)
            history_publish = np.zeros(modality.history_length, dtype=np.uint64)
            history_receive = np.zeros(modality.history_length, dtype=np.uint64)
            history_valid = np.zeros(modality.history_length, dtype=bool)
            history_age_s = np.full(modality.history_length, np.nan, dtype=np.float64)
            # Fill newest-to-oldest so a short history puts the freshest real
            # frame in the newest slot.  A source sequence may be selected at
            # most once; missing history remains an explicit invalid pad.
            for history_index in reversed(range(modality.history_length)):
                target_ns = anchor_monotonic_ns - (modality.history_length - 1 - history_index) * dt_ns
                causal = [
                    frame
                    for frame in candidates
                    if frame.source_monotonic_ns <= target_ns
                    and (frame.receive_monotonic_ns or frame.publish_monotonic_ns) <= target_ns
                    and frame.publish_monotonic_ns <= target_ns
                    and (frame.generation, frame.sequence) not in used_sequences
                ]
                selected = causal[-1] if causal else None
                valid = selected is not None
                if selected is not None:
                    age_ns = target_ns - selected.source_monotonic_ns
                    valid = age_ns <= int(modality.max_age_s * 1e9)
                if valid and selected is not None:
                    value = np.asarray(selected.value, dtype=np.dtype(modality.dtype))
                    if value.shape != modality.shape or not np.all(np.isfinite(value)):
                        valid = False
                if valid and selected is not None:
                    history_values[history_index] = np.array(value, copy=True)
                    history_source[history_index] = selected.source_monotonic_ns
                    history_publish[history_index] = selected.publish_monotonic_ns
                    history_receive[history_index] = selected.receive_monotonic_ns or selected.publish_monotonic_ns
                    history_valid[history_index] = True
                    history_age_s[history_index] = age_ns / 1e9
                    used_sequences.add((selected.generation, selected.sequence))
            values[modality.name] = np.stack(history_values, axis=0)
            source_times[modality.name] = history_source
            publish_times[modality.name] = history_publish
            receive_times[modality.name] = history_receive
            valid_masks[modality.name] = history_valid
            source_ages_s[modality.name] = history_age_s

        # Compare modalities at the same history offset from the anchor.  A
        # stale source invalidates only that slot; other causal history remains
        # usable and the measured skew stays visible to the backend.
        newest_by_offset: dict[int, int] = {}
        for modality in self.spec.modalities:
            source = source_times[modality.name]
            valid_array = valid_masks[modality.name]
            for history_index in np.flatnonzero(valid_array):
                offset = modality.history_length - 1 - int(history_index)
                newest_by_offset[offset] = max(newest_by_offset.get(offset, 0), int(source[history_index]))

        source_skews_s: dict[str, np.ndarray] = {}
        for modality in self.spec.modalities:
            source = source_times[modality.name]
            valid_array = valid_masks[modality.name]
            skew_s = np.full(modality.history_length, np.nan, dtype=np.float64)
            fill = np.nan if modality.padding == "invalid_nan" else 0
            for history_index in np.flatnonzero(valid_array):
                offset = modality.history_length - 1 - int(history_index)
                skew_s[history_index] = (newest_by_offset[offset] - int(source[history_index])) / 1e9
                if skew_s[history_index] > modality.max_skew_s:
                    valid_array[history_index] = False
                    values[modality.name][history_index] = fill
            source_skews_s[modality.name] = skew_s

        def frozen_map(items: Mapping[str, np.ndarray], dtype: Any) -> FrozenArrayMap:
            return FrozenArrayMap(
                tuple((modality.name, _readonly(items[modality.name], dtype)) for modality in self.spec.modalities)
            )

        snapshot = ObservationSnapshot(
            observation_id=self._next_observation_id,
            anchor_monotonic_ns=anchor_monotonic_ns,
            values=FrozenArrayMap.validated(values, self.spec),
            source_monotonic_ns=frozen_map(source_times, np.uint64),
            publish_monotonic_ns=frozen_map(publish_times, np.uint64),
            valid_history_mask=frozen_map(valid_masks, bool),
            run_generation=self.run_generation,
            camera_generation=self.camera_generation,
            receive_monotonic_ns=frozen_map(receive_times, np.uint64),
            source_age_s=frozen_map(source_ages_s, np.float64),
            source_skew_s=frozen_map(source_skews_s, np.float64),
        )
        self._next_observation_id += 1
        return snapshot

    @staticmethod
    def validate_ring_capacities(spec: ObservationSpec, capacities: Mapping[str, int]) -> None:
        for modality in spec.modalities:
            actual = int(capacities.get(modality.name, 0))
            if actual < modality.required_ring_capacity:
                raise ValueError(
                    f"ring {modality.name!r} capacity {actual} < required {modality.required_ring_capacity}"
                )
