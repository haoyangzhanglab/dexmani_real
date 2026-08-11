"""Validated SharedStorage adapters for causal learned-policy observations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from dexmani_real.policy.observation import CausalFrame, SnapshotBuilder
from dexmani_real.policy.runtime import ModalitySpec, ObservationSnapshot, ObservationSpec
from dexmani_real.policy.tensor_block import observation_tensor_dtype


@dataclass(frozen=True)
class _RingField:
    ring_name: str
    field_name: str
    source_field: str
    publish_field: str | None
    sequence_field: str | None = None


_ROBOT_FIELDS: dict[str, _RingField] = {
    "arm_qpos": _RingField("arm_state_ring", "qpos", "source_monotonic_ns", "publish_monotonic_ns"),
    "arm_qvel": _RingField("arm_state_ring", "qvel", "source_monotonic_ns", "publish_monotonic_ns"),
    "arm_tau": _RingField("arm_state_ring", "tau", "source_monotonic_ns", "publish_monotonic_ns"),
    "arm_eef_pos": _RingField("arm_state_ring", "eef_pos", "source_monotonic_ns", "publish_monotonic_ns"),
    "arm_eef_rot6d": _RingField("arm_state_ring", "eef_rot6d", "source_monotonic_ns", "publish_monotonic_ns"),
    "hand_qpos": _RingField("hand_state_ring", "qpos", "source_monotonic_ns", "publish_monotonic_ns"),
    "hand_current": _RingField("hand_state_ring", "current", "source_monotonic_ns", "publish_monotonic_ns"),
    "hand_tactile_sum": _RingField("hand_state_ring", "tactile_sum", "source_monotonic_ns", "publish_monotonic_ns"),
    "hand_tactile_force": _RingField("hand_tactile_ring", "tactile_force", "source_monotonic_ns", None),
    "vr_wrist_pos": _RingField("vr_ring", "wrist_pos", "local_recv_ns", None),
    "vr_wrist_quat_wxyz": _RingField("vr_ring", "wrist_quat_wxyz", "local_recv_ns", None),
    "vr_landmarks": _RingField("vr_ring", "landmarks", "local_recv_ns", None),
    "vr_head_pos": _RingField("vr_ring", "head_pos", "head_recv_ts_ns", None, "head_sequence_id"),
    "vr_head_quat_wxyz": _RingField("vr_ring", "head_quat_wxyz", "head_recv_ts_ns", None, "head_sequence_id"),
}

_CAMERA_MODALITIES = {
    "camera_rgb": "rgb",
    "camera_depth": "depth",
    "camera_pointcloud": "pointcloud",
}


def with_observation_capacities(
    storage_config: Any,
    spec: ObservationSpec,
    *,
    max_observation_transport_bytes: int = 512 * 1024 * 1024,
) -> Any:
    """Return a new SharedStorageConfig large enough for one ObservationSpec."""
    field_by_ring = {
        "arm_state_ring": "arm_state_ring_maxlen",
        "hand_state_ring": "hand_state_ring_maxlen",
        "hand_tactile_ring": "hand_tactile_ring_maxlen",
        "vr_ring": "vr_ring_maxlen",
    }
    updates: dict[str, int] = {}
    for modality in spec.modalities:
        if modality.name in _ROBOT_FIELDS:
            config_field = field_by_ring[_ROBOT_FIELDS[modality.name].ring_name]
        elif modality.name in _CAMERA_MODALITIES:
            config_field = "camera_ring_maxlen"
        else:
            raise ValueError(f"unsupported observation modality {modality.name!r}")
        updates[config_field] = max(
            int(getattr(storage_config, config_field)),
            updates.get(config_field, 0),
            modality.required_ring_capacity,
        )

    resolved = replace(storage_config, **updates)
    camera_slot_bytes = (
        int(np.prod(resolved.camera_rgb_shape))
        + int(np.prod(resolved.camera_depth_shape)) * np.dtype(np.uint16).itemsize
        + int(np.prod(resolved.camera_pc_shape)) * np.dtype(np.float32).itemsize
    )
    observation_bytes = (
        camera_slot_bytes * int(resolved.camera_ring_maxlen) + 2 * observation_tensor_dtype(spec).itemsize
    )
    if observation_bytes > max_observation_transport_bytes:
        raise MemoryError(
            f"observation transports need about {observation_bytes} bytes, budget is {max_observation_transport_bytes}"
        )
    return resolved


def _field_shape_dtype(dtype: np.dtype, field_name: str) -> tuple[tuple[int, ...], np.dtype]:
    fields = dtype.fields
    if fields is None:
        raise ValueError("source dtype is not structured")
    if field_name not in fields:
        raise ValueError(f"source dtype has no field {field_name!r}")
    field_dtype = fields[field_name][0]
    if field_dtype.subdtype is None:
        return (), field_dtype
    base, shape = field_dtype.subdtype
    return tuple(int(dim) for dim in shape), np.dtype(base)


def _source_frame_valid(ring_name: str, record: np.void) -> bool:
    names = record.dtype.names or ()
    if ring_name in {"arm_state_ring", "hand_state_ring"}:
        return "state_valid" not in names or bool(record["state_valid"])
    if ring_name == "hand_tactile_ring":
        return "fresh" not in names or bool(record["fresh"])
    return True


class SharedObservationSource:
    """Build snapshots without allowing the inference worker to access SharedStorage."""

    def __init__(self, shared: Any, spec: ObservationSpec, *, max_tensor_bytes: int = 512 * 1024 * 1024) -> None:
        self.shared = shared
        self.spec = spec
        self._validate_contract(max_tensor_bytes=max_tensor_bytes)
        self._builder = SnapshotBuilder(
            spec,
            session_generation=int(shared.session_generation.value),
        )

    @property
    def requires_camera(self) -> bool:
        return any(modality.name in _CAMERA_MODALITIES for modality in self.spec.modalities)

    @property
    def requires_vr(self) -> bool:
        return any(modality.name.startswith("vr_") for modality in self.spec.modalities)

    @property
    def requires_hand(self) -> bool:
        return any(modality.name.startswith("hand_") for modality in self.spec.modalities)

    def _validate_contract(self, *, max_tensor_bytes: int) -> None:
        if max_tensor_bytes <= 0:
            raise ValueError("max_tensor_bytes must be positive")
        tensor_bytes = observation_tensor_dtype(self.spec).itemsize * 2
        if tensor_bytes > max_tensor_bytes:
            raise MemoryError(f"observation tensor block needs {tensor_bytes} bytes, budget is {max_tensor_bytes}")

        capacities: dict[str, int] = {}
        for modality in self.spec.modalities:
            if modality.name in _ROBOT_FIELDS:
                source = _ROBOT_FIELDS[modality.name]
                ring = getattr(self.shared, source.ring_name)
                source_shape, source_dtype = _field_shape_dtype(ring.dtype, source.field_name)
                if source_shape != modality.shape or source_dtype != np.dtype(modality.dtype):
                    raise ValueError(
                        f"modality {modality.name!r} expects {modality.shape}/{np.dtype(modality.dtype)}, "
                        f"source provides {source_shape}/{source_dtype}"
                    )
                capacities[modality.name] = int(ring.maxlen)
            elif modality.name in _CAMERA_MODALITIES:
                ring = self.shared.camera_ring
                shape = {
                    "camera_rgb": ring._rgb_shape,
                    "camera_depth": ring._depth_shape,
                    "camera_pointcloud": ring._pc_shape,
                }[modality.name]
                dtype = {
                    "camera_rgb": np.dtype(np.uint8),
                    "camera_depth": np.dtype(np.uint16),
                    "camera_pointcloud": np.dtype(np.float32),
                }[modality.name]
                if shape is None or tuple(shape) != modality.shape or dtype != np.dtype(modality.dtype):
                    raise ValueError(
                        f"modality {modality.name!r} expects {modality.shape}/{np.dtype(modality.dtype)}, "
                        f"source provides {shape}/{dtype}"
                    )
                capacities[modality.name] = int(ring.maxlen)
            else:
                raise ValueError(f"unsupported observation modality {modality.name!r}")
        SnapshotBuilder.validate_ring_capacities(self.spec, capacities)

    @staticmethod
    def _scan_count(modalities: list[ModalitySpec], maxlen: int) -> int:
        """Bound a history scan when producer rates make the horizon explicit."""
        if not modalities or any(modality.producer_hz is None for modality in modalities):
            return int(maxlen)
        return min(int(maxlen), max(modality.required_ring_capacity for modality in modalities))

    def _read_structured(
        self,
        modality: ModalitySpec,
        history: list[tuple[np.ndarray, int, int]],
    ) -> list[CausalFrame]:
        source = _ROBOT_FIELDS[modality.name]
        frames: list[CausalFrame] = []
        for data, ring_publish_ns, sequence in history:
            record = data[0]
            if not _source_frame_valid(source.ring_name, record):
                continue
            source_ns = int(record[source.source_field])
            publish_ns = int(record[source.publish_field]) if source.publish_field is not None else int(ring_publish_ns)
            if source_ns <= 0 or publish_ns < source_ns:
                continue
            # SDK HeadFrame sequence IDs start at zero, whereas CausalFrame
            # reserves zero for an unavailable source identity.
            source_sequence = int(record[source.sequence_field]) + 1 if source.sequence_field is not None else sequence
            frames.append(
                CausalFrame(
                    value=np.array(record[source.field_name], copy=True),
                    sequence=int(source_sequence),
                    source_monotonic_ns=source_ns,
                    publish_monotonic_ns=publish_ns,
                    receive_monotonic_ns=source_ns if source.ring_name == "vr_ring" else publish_ns,
                )
            )
        return frames

    def _select_camera_metadata(
        self,
        modality: ModalitySpec,
        metadata: list[tuple[np.ndarray, int, int]],
        *,
        anchor_monotonic_ns: int,
        generation: int,
    ) -> list[tuple[np.ndarray, int, int]]:
        """Select causal header identities before copying any large payload."""
        payload_name = _CAMERA_MODALITIES[modality.name]
        candidates: list[tuple[np.ndarray, int, int]] = []
        for header, ring_publish_ns, sequence in metadata:
            record = header[0]
            source_ns = int(record["source_monotonic_ns"])
            receive_ns = int(record["receive_monotonic_ns"])
            publish_ns = int(record["publish_monotonic_ns"]) or int(ring_publish_ns)
            if (
                source_ns <= 0
                or not source_ns <= receive_ns <= publish_ns
                or bool(record["duplicate"])
                or int(record["camera_generation"]) != generation
            ):
                continue
            if payload_name == "pointcloud" and not bool(record["pointcloud_valid"]):
                continue
            candidates.append((header, ring_publish_ns, int(sequence)))

        candidates.sort(key=lambda item: (int(item[0]["source_monotonic_ns"][0]), item[2]))
        selected: list[tuple[np.ndarray, int, int]] = []
        used_sequences: set[int] = set()
        dt_ns = int(round(1e9 / self.spec.control_hz))
        for history_index in reversed(range(modality.history_length)):
            target_ns = anchor_monotonic_ns - (modality.history_length - 1 - history_index) * dt_ns
            causal = [
                item
                for item in candidates
                if int(item[0]["source_monotonic_ns"][0]) <= target_ns
                and int(item[0]["receive_monotonic_ns"][0]) <= target_ns
                and (int(item[0]["publish_monotonic_ns"][0]) or int(item[1])) <= target_ns
                and item[2] not in used_sequences
            ]
            if not causal:
                continue
            chosen = causal[-1]
            if target_ns - int(chosen[0]["source_monotonic_ns"][0]) > int(modality.max_age_s * 1e9):
                continue
            selected.append(chosen)
            used_sequences.add(chosen[2])
        return selected

    def _camera_generation_at(
        self,
        anchor_monotonic_ns: int,
        metadata: list[tuple[np.ndarray, int, int]],
    ) -> int:
        """Return the newest generation with metadata causal to this anchor."""
        generation = self._builder.camera_generation
        for header, ring_publish_ns, _sequence in metadata:
            record = header[0]
            source_ns = int(record["source_monotonic_ns"])
            receive_ns = int(record["receive_monotonic_ns"])
            publish_ns = int(record["publish_monotonic_ns"]) or int(ring_publish_ns)
            if (
                source_ns > 0
                and source_ns <= receive_ns <= publish_ns <= anchor_monotonic_ns
                and not bool(record["duplicate"])
            ):
                generation = max(generation, int(record["camera_generation"]))
        return generation

    def _read_selected_camera(
        self,
        modalities: list[ModalitySpec],
        metadata: list[tuple[np.ndarray, int, int]],
        *,
        anchor_monotonic_ns: int,
        generation: int,
    ) -> dict[str, list[CausalFrame]]:
        selected = {
            modality.name: self._select_camera_metadata(
                modality,
                metadata,
                anchor_monotonic_ns=anchor_monotonic_ns,
                generation=generation,
            )
            for modality in modalities
        }
        modalities_by_sequence: dict[int, set[str]] = {}
        for modality in modalities:
            payload_name = _CAMERA_MODALITIES[modality.name]
            for _header, _publish_ns, sequence in selected[modality.name]:
                modalities_by_sequence.setdefault(sequence, set()).add(payload_name)

        payloads = {
            sequence: self.shared.camera_ring.read_sequence(
                sequence,
                modalities=tuple(sorted(payload_names)),
            )
            for sequence, payload_names in modalities_by_sequence.items()
        }
        frames: dict[str, list[CausalFrame]] = {modality.name: [] for modality in modalities}
        for modality in modalities:
            payload_name = _CAMERA_MODALITIES[modality.name]
            for header, ring_publish_ns, sequence in selected[modality.name]:
                payload = payloads.get(sequence)
                if payload is None or payload_name not in payload:
                    continue
                record = header[0]
                frames[modality.name].append(
                    CausalFrame(
                        value=payload[payload_name],
                        sequence=sequence,
                        source_monotonic_ns=int(record["source_monotonic_ns"]),
                        publish_monotonic_ns=int(record["publish_monotonic_ns"]) or int(ring_publish_ns),
                        receive_monotonic_ns=int(record["receive_monotonic_ns"]),
                        generation=generation,
                    )
                )
        return frames

    def build(self, *, anchor_monotonic_ns: int) -> ObservationSnapshot:
        frames: dict[str, list[CausalFrame]] = {}
        camera_modalities = [modality for modality in self.spec.modalities if modality.name in _CAMERA_MODALITIES]
        camera_metadata: list[tuple[np.ndarray, int, int]] = []
        camera_generation = self._builder.camera_generation
        if camera_modalities:
            camera_scan_count = self._scan_count(camera_modalities, self.shared.camera_ring.maxlen)
            camera_metadata = self.shared.camera_ring.get_last_metadata(camera_scan_count)
            camera_generation = self._camera_generation_at(anchor_monotonic_ns, camera_metadata)
            frames.update(
                self._read_selected_camera(
                    camera_modalities,
                    camera_metadata,
                    anchor_monotonic_ns=anchor_monotonic_ns,
                    generation=camera_generation,
                )
            )

        structured_by_ring: dict[str, list[ModalitySpec]] = {}
        for modality in self.spec.modalities:
            if modality.name in _ROBOT_FIELDS:
                structured_by_ring.setdefault(_ROBOT_FIELDS[modality.name].ring_name, []).append(modality)
        for ring_name, modalities in structured_by_ring.items():
            ring = getattr(self.shared, ring_name)
            history = ring.get_last_k(self._scan_count(modalities, ring.maxlen))
            for modality in modalities:
                frames[modality.name] = self._read_structured(modality, history)
        self._builder.camera_generation = camera_generation
        return self._builder.build(anchor_monotonic_ns=anchor_monotonic_ns, frames=frames)
